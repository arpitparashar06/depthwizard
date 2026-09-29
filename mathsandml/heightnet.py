"""DepthWizard's trained height model."""

from __future__ import annotations

import hashlib
import math
import os

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None

FORMAT = "depthwizard-heightnet-v1"
VARIANTS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",   # Apache-2.0
    "base": "depth-anything/Depth-Anything-V2-Base-hf",     # CC-BY-NC-4.0
    "large": "depth-anything/Depth-Anything-V2-Large-hf",   # CC-BY-NC-4.0
}
# GAMUS label order.
CLASSES = ("others", "ground", "low_vegetation", "building", "water", "road", "tree")
IGNORE_INDEX = 255

H0 = 2.0
H_MAX = 350.0
N_BINS = 64
GSD_MIN, GSD_MAX = 0.30, 1.20
GSD_DEFAULT = 0.5
PATCH = 14
TILE = 518
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_checkpoint():
    """Path the pipeline looks at: $DEPTHWIZARD_HEIGHTNET, else models/heightnet.pt."""
    p = os.environ.get("DEPTHWIZARD_HEIGHTNET", "").strip()
    return p or os.path.join(_REPO_ROOT, "models", "heightnet.pt")


def _is_unzipped_checkpoint(p):
    return os.path.isdir(p) and os.path.isfile(os.path.join(p, "data.pkl"))


def available(path=None):
    """The checkpoint path if a trained model can be used here, else None."""
    p = path or default_checkpoint()
    ok = os.path.isfile(p) or _is_unzipped_checkpoint(p)
    return p if (torch is not None and ok) else None


def file_digest(path, n=1 << 20):
    """Short content hash - part of the prediction cache key."""
    h = hashlib.sha1()
    if os.path.isdir(path):
        with open(os.path.join(path, "data.pkl"), "rb") as f:
            h.update(f.read(n))
        for root, _, names in sorted(os.walk(path)):
            for nm in sorted(names):
                h.update(f"{nm}:{os.path.getsize(os.path.join(root, nm))}".encode())
        return h.hexdigest()[:16]
    with open(path, "rb") as f:
        h.update(f.read(n))
        f.seek(0, 2)
        h.update(str(f.tell()).encode())
    return h.hexdigest()[:16]


def _rezip(folder):
    import io
    import zipfile
    buf = io.BytesIO()
    names = []
    for root, _, files in os.walk(folder):
        for nm in files:
            names.append(os.path.relpath(os.path.join(root, nm), folder))
    names.sort(key=lambda f: (f != "data.pkl", not f.startswith("data" + os.sep), f))
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as z:
        for f in names:
            zi = zipfile.ZipInfo("archive/" + f.replace(os.sep, "/"),
                                 date_time=(1980, 1, 1, 0, 0, 0))
            with open(os.path.join(folder, f), "rb") as fh:
                z.writestr(zi, fh.read())
    buf.seek(0)
    return buf


def u_of(h):
    """Height -> the log-like space the bins are uniform in."""
    return torch.log1p(h.clamp(min=0) / H0)


def robust_affine(src, ref, iters=4, max_n=200_000):
    """Robust ref ~ a*src + b (Cauchy weights), on a subsample for speed."""
    x = np.asarray(src, np.float64).ravel()
    y = np.asarray(ref, np.float64).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size > max_n:
        idx = np.random.default_rng(0).choice(x.size, max_n, replace=False)
        x, y = x[idx], y[idx]
    if x.size < 32 or x.std() < 1e-9:
        return 1.0, 0.0
    A = np.c_[x, np.ones_like(x)]
    w = np.ones_like(x)
    coef = np.array([1.0, 0.0])
    for _ in range(iters):
        coef, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
        if not np.all(np.isfinite(coef)):
            return 1.0, 0.0
        r = y - A @ coef
        scale = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        w = 1.0 / np.sqrt(1 + (r / (2 * scale)) ** 2)
    a, b = float(coef[0]), float(coef[1])
    return (a, b) if a > 0 else (1.0, 0.0)


def ground_envelope(p, px_size_m, window_m=80.0, pct=2.0, step_m=2.0, max_cells=400):
    """The ground under a relative surface: an 80 m robust opening."""
    from scipy.ndimage import gaussian_filter, maximum_filter, percentile_filter, zoom
    a = np.asarray(p, np.float64)
    H, W = a.shape
    px = max(float(px_size_m or 1.0), 1e-6)
    step = max(step_m, px, max(H, W) * px / max_cells)
    f = max(1, int(round(step / px)))
    Hc, Wc = max(1, H // f), max(1, W // f)
    ds = a[:Hc * f, :Wc * f].reshape(Hc, f, Wc, f).mean(axis=(1, 3))
    k = max(3, int(round(window_m / (px * f))) | 1)
    envelope = percentile_filter(ds, pct, size=k, mode="nearest")
    envelope = maximum_filter(envelope, size=k, mode="nearest")
    envelope = gaussian_filter(envelope, k / 4.0)
    up = zoom(envelope, (H / envelope.shape[0], W / envelope.shape[1]), order=1)
    if up.shape != (H, W):
        up = np.pad(up, ((0, max(0, H - up.shape[0])), (0, max(0, W - up.shape[1]))),
                    mode="edge")[:H, :W]
    return up


def relative_informative(rel, min_std=1e-4, min_cv=1e-3):
    r = np.asarray(rel, np.float64)
    r = r[np.isfinite(r)]
    if r.size < 100:
        return False
    sd = float(r.std())
    return sd > min_std and sd > min_cv * abs(float(r.mean()))


LEARNED_PRIOR_PCTS = (90.0, 95.0, 98.0, 99.0, 99.5)


def learned_scale(detail, metric, valid=None, min_height_m=1.0):
    d = np.asarray(detail, np.float64)
    m = np.asarray(metric, np.float64)
    ok = np.isfinite(d) & np.isfinite(m)
    if valid is not None:
        ok &= valid
    if ok.sum() < 100:
        return None
    d, m = d[ok], m[ok]
    ratios = []
    for q in LEARNED_PRIOR_PCTS:
        dq, mq = np.percentile(d, q), np.percentile(m, q)
        if dq > 1e-9 and mq > min_height_m:
            ratios.append(mq / dq)
    return float(np.median(ratios)) if ratios else None


STACK_BANDS = (0.0, 2.0, 10.0, 30.0)     # metres, on B's own prediction


def stack_weights_for(b_h, fusion):
    edges = list(fusion.get("bands", STACK_BANDS))
    w = np.asarray(fusion["w_a"], np.float64)
    idx = np.clip(np.searchsorted(edges, np.nan_to_num(b_h), side="right") - 1,
                  0, len(w) - 1)
    return w[idx]


def fuse(a_h, a_sig, b_h, b_sig, fusion=None, floor_m=0.3):
    """Blend A (calibrated) and B per pixel."""
    f = fusion or {}
    ca, cb = float(f.get("c_a", 1.0)), float(f.get("c_b", 1.0))
    va = (ca * np.maximum(np.nan_to_num(a_sig, nan=1e3), floor_m)) ** 2
    vb = (cb * np.maximum(np.nan_to_num(b_sig, nan=1e3), floor_m)) ** 2
    if f.get("mode") == "stacked" and f.get("w_a") is not None:
        wa = stack_weights_for(b_h, f)
        wb = 1.0 - wa
        h = wa * a_h + wb * b_h
        return h, np.sqrt(wa ** 2 * va + wb ** 2 * vb), wb
    wb = va / (va + vb)
    h = (1 - wb) * a_h + wb * b_h
    return h, np.sqrt(va * vb / (va + vb)), wb


def combine(pred, px_size_m, fusion=None, alpha=None, envelope=None):
    """A, B and A+B from one HeightPredictor.predict() result (heights above ground, metres)."""
    relative = np.asarray(pred["rel"], np.float64)
    if not relative_informative(relative):
        return dict(alpha=None, alpha_source="none: the relative output is flat",
                    a=None, a_sigma=None, fused=pred["height"], fused_sigma=pred["sigma"],
                    w_b=np.ones_like(pred["height"]), detail=np.zeros_like(relative))
    env = envelope(relative) if envelope else ground_envelope(relative, px_size_m)
    detail = relative - env
    src = "given"
    if alpha is None:
        alpha = learned_scale(detail, pred["height"])
        src = "learned prior (HeightNet metric head)"
    if alpha is None:
        return dict(alpha=None, alpha_source="none: nothing above ground to match",
                    a=None, a_sigma=None, fused=pred["height"], fused_sigma=pred["sigma"],
                    w_b=np.ones_like(pred["height"]), detail=detail)
    a_h = np.maximum(alpha * detail, 0.0)
    a_sig = alpha * np.asarray(pred.get("rel_sigma", np.zeros_like(relative)), np.float64)
    h, sigma, wb = fuse(a_h, a_sig, pred["height"], pred["sigma"], fusion)
    return dict(alpha=float(alpha), alpha_source=src, a=a_h, a_sigma=a_sig,
                fused=np.maximum(h, 0.0), fused_sigma=sigma, w_b=wb, detail=detail)


if torch is not None:

    class _ConvGN(nn.Sequential):
        def __init__(self, cin, cout, k=3):
            super().__init__(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
                             nn.GroupNorm(min(8, cout), cout), nn.GELU())

    class GSDEmbed(nn.Module):

        def __init__(self, dim=128, n_freq=6):
            super().__init__()
            self.register_buffer("freqs", (2.0 ** torch.arange(n_freq)) * math.pi / 4,
                                 persistent=False)
            self.mlp = nn.Sequential(nn.Linear(1 + 2 * n_freq, dim), nn.GELU(),
                                     nn.Linear(dim, dim))

        def forward(self, gsd):
            x = torch.log2(gsd.float().clamp(0.05, 20.0) / GSD_DEFAULT)[:, None]
            z = x * self.freqs[None]
            return self.mlp(torch.cat([x, z.sin(), z.cos()], -1))

    class HeightNet(nn.Module):

        def __init__(self, da, n_bins=N_BINS, h_max=H_MAX, n_classes=len(CLASSES),
                     emb_dim=128):
            super().__init__()
            self.da = da
            config = da.config
            fch = int(config.fusion_hidden_size)
            hid = int(config.backbone_config.hidden_size)
            hh = int(config.head_hidden_size)
            self.n_bins, self.h_max = int(n_bins), float(h_max)
            self.n_classes, self.emb_dim = int(n_classes), int(emb_dim)

            self.gsd_embed = GSDEmbed(emb_dim)
            # FiLM on the finest fused feature.
            self.film = nn.Linear(emb_dim, 2 * fch)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

            cin = fch // 2 + hh + 1
            self.seg_head = nn.Sequential(_ConvGN(cin, 64), _ConvGN(64, 64),
                                          nn.Conv2d(64, n_classes, 1))
            self.height_trunk = nn.Sequential(_ConvGN(cin + n_classes, 96),
                                              _ConvGN(96, 96))
            self.height_logits = nn.Conv2d(96, n_bins, 1)

            edges_u = torch.linspace(0.0, math.log1p(h_max / H0), n_bins + 1)
            edges_h = H0 * torch.expm1(edges_u)
            w = edges_h[1:] - edges_h[:-1]
            self.register_buffer("base_w", w / w.sum())
            cu = 0.5 * (edges_u[1:] + edges_u[:-1])
            with torch.no_grad():
                self.height_logits.bias.copy_(-1.5 * cu)

            self.bin_mlp = nn.Sequential(nn.Linear(hid + fch + emb_dim, 256), nn.GELU(),
                                         nn.Linear(256, n_bins))
            nn.init.zeros_(self.bin_mlp[-1].weight)
            nn.init.zeros_(self.bin_mlp[-1].bias)

        # bins
        def bin_centers(self, glob):
            """Per-image bin centres (B, K) in metres."""
            t = 1.5 * torch.tanh(self.bin_mlp(glob.float()))
            w = self.base_w[None] * t.exp()
            w = w / w.sum(-1, keepdim=True)
            edges = self.h_max * torch.cumsum(w, -1)
            edges = torch.cat([torch.zeros_like(edges[:, :1]), edges], -1)
            return 0.5 * (edges[:, :-1] + edges[:, 1:])

        # forward
        def forward(self, pixel_values, gsd):
            """pixel_values (B,3,H,W) ImageNet-normalised, H and W multiples of 14."""
            B, _, H, W = pixel_values.shape
            ph, pw = H // PATCH, W // PATCH
            bbox = self.da.backbone(pixel_values)
            fmaps = list(bbox.feature_maps)
            cls = fmaps[-1][:, 0]
            fused = self.da.neck(fmaps, ph, pw)
            head = self.da.head
            f = fused[getattr(head, "head_in_index", -1)]

            e = self.gsd_embed(gsd)
            g, b = self.film(e).to(f.dtype).chunk(2, -1)
            f = f * (1 + g[..., None, None]) + b[..., None, None]

            oh, ow = ph * PATCH // 2, pw * PATCH // 2
            x1 = F.interpolate(head.conv1(f), (oh, ow), mode="bilinear", align_corners=True)
            x2 = F.relu(head.conv2(x1))
            relative = F.relu(head.conv3(x2))[:, 0]                 # engine A output
            d = torch.log1p(relative.float())[:, None]
            d = (d - d.mean((2, 3), keepdim=True)).to(x2.dtype)
            feat = torch.cat([x1, x2, d], 1)

            seg = self.seg_head(feat)
            t = self.height_trunk(torch.cat([feat, seg.softmax(1)], 1))
            logits = self.height_logits(t)

            glob = torch.cat([cls.float(), f.float().mean((2, 3)), e.float()], -1)
            centers = self.bin_centers(glob)
            prob = logits.float().softmax(1)
            c = centers[:, :, None, None]
            h = (prob * c).sum(1)
            var = (prob * (c - h[:, None]) ** 2).sum(1)
            return dict(height=h, sigma=var.clamp_min(1e-6).sqrt(), logits=logits,
                        centers=centers, seg=seg, rel=relative)

    # build / save / load
    def build(variant="small", pretrained=True, da_config=None, **kw):
        from transformers import DepthAnythingConfig, DepthAnythingForDepthEstimation
        if da_config is not None:
            config = (DepthAnythingConfig.from_dict(dict(da_config))
                   if isinstance(da_config, dict) else da_config)
            da = DepthAnythingForDepthEstimation(config)
        else:
            repo = VARIANTS.get(variant, variant)
            if pretrained:
                da = DepthAnythingForDepthEstimation.from_pretrained(repo)
            else:
                da = DepthAnythingForDepthEstimation(DepthAnythingConfig.from_pretrained(repo))
        return HeightNet(da, **kw)

    def tiny_da_config():
        from transformers import DepthAnythingConfig, Dinov2Config
        bbox = Dinov2Config(hidden_size=48, num_hidden_layers=4, num_attention_heads=2,
                          intermediate_size=96, image_size=518, patch_size=14,
                          out_indices=[1, 2, 3, 4], reshape_hidden_states=False)
        return DepthAnythingConfig(backbone_config=bbox, neck_hidden_sizes=[12, 24, 48, 48],
                                   fusion_hidden_size=32, head_hidden_size=16,
                                   reassemble_hidden_size=48, patch_size=14)

    def _torch_load(path):
        src = _rezip(path) if _is_unzipped_checkpoint(path) else path
        try:
            return torch.load(src, map_location="cpu", weights_only=True)
        except Exception:
            if hasattr(src, "seek"):
                src.seek(0)
            return torch.load(src, map_location="cpu", weights_only=False)

    def set_fusion(path, fusion):
        ck = _torch_load(path)
        ck["fusion"] = {k: (v if isinstance(v, (str, list)) else float(v))
                        for k, v in fusion.items()}
        tmp = path + ".tmp"
        torch.save(ck, tmp)
        if os.path.isdir(path):
            import shutil
            shutil.rmtree(path)
        os.replace(tmp, path)

    def save_checkpoint(path, model, variant="small", extra=None, half=True):
        m = model.module if hasattr(model, "module") else model
        sd = {k: (v.detach().half() if half and v.is_floating_point() else v.detach()).cpu()
              for k, v in m.state_dict().items()}
        ck = dict(format=FORMAT, variant=variant, da_config=m.da.config.to_dict(),
                  heightnet=dict(n_bins=m.n_bins, h_max=m.h_max, n_classes=m.n_classes,
                                 emb_dim=m.emb_dim),
                  classes=list(CLASSES), gsd_range=[GSD_MIN, GSD_MAX],
                  state_dict=sd, train=extra or {})
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        torch.save(ck, tmp)
        os.replace(tmp, path)
        return path

    def load_checkpoint(path, device="cpu"):
        ck = _torch_load(path)
        if not isinstance(ck, dict) or ck.get("format") != FORMAT:
            raise ValueError(f"{path} is not a DepthWizard HeightNet checkpoint")
        model = build(da_config=ck["da_config"], **ck["heightnet"])
        sd = {k: (v.float() if v.is_floating_point() else v) for k, v in ck["state_dict"].items()}
        model.load_state_dict(sd, strict=True)
        return model.eval().to(device), ck

    def pick_device(want="auto"):
        want = (want or "auto").lower()
        if want == "cuda" or (want == "auto" and torch.cuda.is_available()):
            return torch.device("cuda")
        if want == "mps" or (want == "auto" and getattr(torch.backends, "mps", None)
                             and torch.backends.mps.is_available()):
            return torch.device("mps")
        return torch.device("cpu")

    def normalise(rgb_u8):
        a = rgb_u8.astype(np.float32) / 255.0
        a = (a - MEAN) / STD
        if a.ndim == 3:
            a = a[None]
        return torch.from_numpy(np.ascontiguousarray(a.transpose(0, 3, 1, 2)))

    # inference on a whole scene
    def _positions(n, tile, stride):
        if n <= tile:
            return [0]
        p = list(range(0, n - tile + 1, stride))
        if p[-1] != n - tile:
            p.append(n - tile)
        return p

    def _window(tile, ov):
        r = np.ones(tile, np.float32)
        if ov > 0:
            ramp = np.sin(np.linspace(0, np.pi / 2, ov + 2)[1:-1]) ** 2
            r[:ov] = ramp
            r[-ov:] = ramp[::-1]
        return np.maximum(np.outer(r, r), 1e-3)

    class HeightPredictor:

        def __init__(self, path=None, device="auto", batch=None):
            path = path or default_checkpoint()
            self.path = path
            self.device = pick_device(device)
            self.model, self.ckpt = load_checkpoint(path, "cpu")
            self.model.to(self.device)
            self.digest = file_digest(path)
            self.batch = batch or (4 if self.device.type == "cuda" else 1)
            self.variant = self.ckpt.get("variant", "?")
            self.train_info = self.ckpt.get("train", {})
            self.fusion = dict(self.ckpt.get("fusion") or {})

        def describe(self):
            t = self.train_info or {}
            data = t.get("data") or "LiDAR-labelled aerial scenes"
            return f"HeightNet-{self.variant} ({data})"

        @torch.no_grad()
        def _run(self, tiles, gsd):
            x = normalise(tiles).to(self.device)
            g = torch.full((x.shape[0],), float(gsd), device=self.device)
            if self.device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    out = self.model(x, g)
            else:
                out = self.model(x, g)
            size = tiles.shape[1:3]

            def up(t):
                return F.interpolate(t[:, None].float(), size=size, mode="bilinear",
                                     align_corners=False)[:, 0].cpu().numpy()
            p = F.interpolate(out["seg"].float().softmax(1), size=size, mode="bilinear",
                              align_corners=False)
            return up(out["height"]), up(out["sigma"]), p.cpu().numpy(), up(out["rel"])

        def _tiled(self, img, gsd, overlap=128):
            H, W = img.shape[:2]
            T = TILE
            ph, pw = max(0, T - H), max(0, T - W)
            if ph or pw:
                img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
            Hp, Wp = img.shape[:2]
            stride = T - overlap
            win = _window(T, overlap)
            K = self.model.n_classes
            acc_h = np.zeros((Hp, Wp), np.float32)
            acc_s = np.zeros((Hp, Wp), np.float32)
            acc_r = np.zeros((Hp, Wp), np.float32)
            acc_p = np.zeros((K, Hp, Wp), np.float32)
            acc_w = np.zeros((Hp, Wp), np.float32)
            boxes = [(y, x) for y in _positions(Hp, T, stride) for x in _positions(Wp, T, stride)]
            for i in range(0, len(boxes), self.batch):
                chunk = boxes[i:i + self.batch]
                tiles = np.stack([img[y:y + T, x:x + T] for y, x in chunk])
                h, s, p, relative = self._run(tiles, gsd)
                for j, (y, x) in enumerate(chunk):
                    seen = acc_w[y:y + T, x:x + T] > 1e-6
                    rj = relative[j]
                    if seen.sum() > 256:
                        ref = acc_r[y:y + T, x:x + T][seen] / acc_w[y:y + T, x:x + T][seen]
                        a, b = robust_affine(rj[seen], ref)
                        rj = a * rj + b
                    acc_h[y:y + T, x:x + T] += h[j] * win
                    acc_s[y:y + T, x:x + T] += (s[j] ** 2) * win
                    acc_r[y:y + T, x:x + T] += rj * win
                    acc_p[:, y:y + T, x:x + T] += p[j] * win
                    acc_w[y:y + T, x:x + T] += win
            acc_w = np.maximum(acc_w, 1e-6)
            return ((acc_h / acc_w)[:H, :W], np.sqrt(acc_s / acc_w)[:H, :W],
                    (acc_p / acc_w)[:, :H, :W], (acc_r / acc_w)[:H, :W], len(boxes))

        def predict(self, rgb, gsd_m=None, rotations=4, overlap=128, progress=None):
            """Rgb uint8 (H,W,3)."""
            import cv2
            H, W = rgb.shape[:2]
            gsd_in = float(gsd_m) if gsd_m else GSD_DEFAULT
            work = min(max(gsd_in, GSD_MIN), GSD_MAX)
            s = gsd_in / work
            img = rgb
            if abs(s - 1.0) > 0.02:
                nw, nh = max(8, int(round(W * s))), max(8, int(round(H * s)))
                img = cv2.resize(rgb, (nw, nh),
                                 interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
            n_rot = int(max(1, min(4, rotations)))
            mean = m2 = s2 = pr = None
            rmean = rm2 = None
            tiles = 0
            for i, k in enumerate(range(n_rot)):
                r = np.ascontiguousarray(np.rot90(img, k))
                h, sg, p, relative, nt = self._tiled(r, work, overlap)
                tiles += nt
                h = np.rot90(h, -k)
                sg = np.rot90(sg, -k)
                p = np.rot90(p, -k, axes=(1, 2))
                relative = np.rot90(relative, -k)
                if mean is None:
                    mean, m2 = h.copy(), np.zeros_like(h)
                    s2, pr = sg ** 2, p.copy()
                    rmean, rm2 = relative.copy(), np.zeros_like(relative)
                else:                                   # Welford across rotations
                    d = h - mean
                    mean += d / (i + 1)
                    m2 += d * (h - mean)
                    s2 += sg ** 2
                    pr += p
                    a, b = robust_affine(relative, rmean)    # onto the first pass's scale
                    relative = a * relative + b
                    d = relative - rmean
                    rmean += d / (i + 1)
                    rm2 += d * (relative - rmean)
                if progress:
                    progress((i + 1) / n_rot)
            s2 /= n_rot
            pr /= n_rot
            sigma = np.sqrt(s2 + m2 / n_rot)
            rsig = np.sqrt(rm2 / n_rot)
            if img is not rgb:
                def back(a):
                    return cv2.resize(np.ascontiguousarray(a), (W, H),
                                      interpolation=cv2.INTER_LINEAR)
                mean, sigma, rmean, rsig = back(mean), back(sigma), back(rmean), back(rsig)
                pr = np.stack([back(c) for c in pr])
            seg = pr.argmax(0).astype(np.uint8)
            return dict(height=np.maximum(mean, 0).astype(np.float32),
                        sigma=sigma.astype(np.float32), seg=seg,
                        seg_conf=pr.max(0).astype(np.float32),
                        rel=rmean.astype(np.float32), rel_sigma=rsig.astype(np.float32),
                        gsd_input=gsd_in, gsd_model=work, rotations=n_rot, tiles=tiles,
                        georeferenced=bool(gsd_m))


def _main():
    import argparse
    parser = argparse.ArgumentParser(description="Run the trained height model on one image.")
    parser.add_argument("image")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--rotations", type=int, default=4)
    parser.add_argument("--out", default=None, help="write height (m, float32) as .npy")
    args = parser.parse_args()
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import inference as I
    rgb, meta = I.load_image(args.image)
    predictor = HeightPredictor(args.ckpt)
    r = predictor.predict(rgb, meta.get("px_size_m"), rotations=args.rotations)
    h = r["height"]
    print(f"{predictor.describe()}  gsd {r['gsd_input']:.2f} m -> model {r['gsd_model']:.2f} m  "
          f"{r['tiles']} tile passes")
    print(f"height above ground: p50 {np.percentile(h, 50):.1f} m  p99 "
          f"{np.percentile(h, 99):.1f} m  max {h.max():.1f} m  "
          f"mean sigma {r['sigma'].mean():.2f} m")
    if args.out:
        np.save(args.out, h)


if __name__ == "__main__":
    _main()
