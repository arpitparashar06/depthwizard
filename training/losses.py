"""What HeightNet is trained to minimise."""

import torch
import torch.nn.functional as F

import heightnet as HN

DEFAULT_WEIGHTS = dict(l1=1.0, ce=0.3, grad=0.5, chamfer=0.1, seg=0.3, ssi=0.5, ssi_grad=0.25)


def downsample_target(h, valid, size):
    """Area-average a (B,H,W) height map with a (B,H,W) bool mask to `size`."""
    v = valid.float()[:, None]
    hs = F.adaptive_avg_pool2d((torch.nan_to_num(h) * valid.float())[:, None], size)
    vs = F.adaptive_avg_pool2d(v, size)
    out = hs / vs.clamp_min(1e-6)
    return out[:, 0], (vs[:, 0] > 0.5)


def downsample_labels(cls, size):
    """Nearest-neighbour subsample of (B,H,W) int labels."""
    return F.interpolate(cls[:, None].float(), size=size, mode="nearest")[:, 0].long()


def longtail_weights(h, valid, n_bands=16, power=0.5, clip=(0.2, 10.0)):
    u = HN.u_of(h)
    umax = float(torch.log1p(torch.tensor(HN.H_MAX / HN.H0)))
    band = (u / umax * n_bands).long().clamp(0, n_bands - 1)
    band_v = band[valid]
    if band_v.numel() == 0:
        return torch.ones_like(h)
    freq = torch.bincount(band_v, minlength=n_bands).float()
    freq = freq / freq.sum()
    wb = (freq.clamp_min(1e-4)) ** (-power)
    w = wb[band]
    w = w / w[valid].mean().clamp_min(1e-6)
    return w.clamp(*clip) * valid.float()


def gradient_loss(pred, target, valid, scales=4):
    """MiDaS-style: penalise |grad(pred - target)| where both pixels are valid."""
    total = pred.new_zeros(())
    for s in range(scales):
        step = 2 ** s
        r = (pred - target)[:, ::step, ::step]
        m = valid[:, ::step, ::step].float()
        gx = (r[:, :, 1:] - r[:, :, :-1]).abs() * m[:, :, 1:] * m[:, :, :-1]
        gy = (r[:, 1:, :] - r[:, :-1, :]).abs() * m[:, 1:, :] * m[:, :-1, :]
        n = m.sum().clamp_min(1.0)
        total = total + (gx.sum() + gy.sum()) / n
    return total / scales


def chamfer_loss(centers, h, valid, n_samples=2048):
    uc = HN.u_of(centers)                        # (B,K)
    losses = []
    for b in range(h.shape[0]):
        hv = h[b][valid[b]]
        if hv.numel() < 16:
            continue
        if hv.numel() > n_samples:
            idx = torch.randint(0, hv.numel(), (n_samples,), device=hv.device)
            hv = hv[idx]
        ut = HN.u_of(hv)
        d = (ut[:, None] - uc[b][None]) ** 2      # (N,K)
        losses.append(d.min(1).values.mean() + d.min(0).values.mean())
    if not losses:
        return centers.new_zeros(())
    return torch.stack(losses).mean()


def ssi_loss(rel, h, valid, min_mad_m=0.3):
    """Scale-and-shift-invariant L1 + gradient matching, per image."""
    l_abs, l_grad = [], []
    for b in range(rel.shape[0]):
        v = valid[b]
        if int(v.sum()) < 256:
            continue
        t = h[b][v]
        tm = t.median()
        ts = (t - tm).abs().mean()
        if float(ts) < min_mad_m:
            continue
        p = rel[b].float()
        pv = p[v]
        pm = pv.median()
        ps = (pv - pm).abs().mean().clamp_min(1e-4)
        pn = (p - pm) / ps
        tn = (torch.nan_to_num(h[b]) - tm) / ts
        l_abs.append((pn - tn).abs()[v].mean())
        l_grad.append(gradient_loss(pn[None], torch.where(v, tn, torch.zeros_like(tn))[None],
                                    v[None]))
    if not l_abs:
        z = rel.float().sum() * 0.0
        return z, z
    return torch.stack(l_abs).mean(), torch.stack(l_grad).mean()


def compute_losses(out, batch, weights=None, longtail_power=0.5):
    """Out: HeightNet.forward() dict."""
    w = dict(DEFAULT_WEIGHTS, **(weights or {}))
    pred = out["height"].float()
    size = pred.shape[-2:]
    h_full = batch["height"]
    valid_full = torch.isfinite(h_full)
    th, tv = downsample_target(h_full, valid_full, size)

    parts = {}
    pw = longtail_weights(th, tv, power=longtail_power)
    denom = pw.sum().clamp_min(1.0)
    l1 = ((pred - th).abs() * pw).sum() / denom
    parts["l1"] = l1

    # Soft labels over bins: a Gaussian in log-height space, one base bin wide
    logits = out["logits"].float()
    uc = HN.u_of(out["centers"].float())[:, :, None, None]         # (B,K,1,1)
    ut = HN.u_of(th)[:, None]                                        # (B,1,h,w)
    du = float(torch.log1p(torch.tensor(HN.H_MAX / HN.H0))) / logits.shape[1]
    q = torch.softmax(-((ut - uc) ** 2) / (2 * du * du), dim=1)
    ce_map = -(q * torch.log_softmax(logits, dim=1)).sum(1)
    parts["ce"] = (ce_map * pw).sum() / denom

    parts["grad"] = gradient_loss(pred, torch.where(tv, th, torch.zeros_like(th)), tv)
    parts["chamfer"] = chamfer_loss(out["centers"].float(), th, tv)

    if "rel" in out and (w.get("ssi", 0) > 0 or w.get("ssi_grad", 0) > 0):
        parts["ssi"], parts["ssi_grad"] = ssi_loss(out["rel"], th, tv)

    cls = batch.get("cls")
    if cls is not None and w.get("seg", 0) > 0:
        tc = downsample_labels(cls, size)
        tc = torch.where(tc == 0, torch.full_like(tc, HN.IGNORE_INDEX), tc)   # "others"
        if (tc != HN.IGNORE_INDEX).any():
            parts["seg"] = F.cross_entropy(out["seg"].float(), tc,
                                           ignore_index=HN.IGNORE_INDEX)
        else:
            parts["seg"] = out["seg"].float().sum() * 0.0
    total = sum(w[k] * v for k, v in parts.items() if k in w)
    return total, {k: float(v.detach()) for k, v in parts.items()}
