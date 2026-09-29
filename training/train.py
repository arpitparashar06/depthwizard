#!/usr/bin/env python3
"""Fine-tune Depth-Anything-V2 into HeightNet (metres above ground)."""

import argparse
import csv
import datetime
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "mathsandml"))
import heightnet as HN  # noqa: E402
import data as D  # noqa: E402
import losses as L  # noqa: E402

VARIANT_DEFAULTS = {
    # Batch/gpu lr_enc iters
    "small": dict(batch=8, lr_enc=2e-5, iters=30000),
    "base": dict(batch=4, lr_enc=1e-5, iters=24000),
    "large": dict(batch=2, lr_enc=5e-6, iters=20000),
    "tiny": dict(batch=2, lr_enc=1e-4, iters=20),
}


# distributed helpers
def setup_dist():
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo",
                                timeout=datetime.timedelta(hours=1))
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
        return rank, world, local
    return 0, 1, 0


def is_main(rank):
    return rank == 0


def log0(rank, *a):
    if rank == 0:
        print(*a, flush=True)


def param_groups(model, lr_enc, lr_dec, lr_head, layer_decay, wd):
    m = model.module if hasattr(model, "module") else model
    n_layers = len(m.da.backbone.encoder.layer)
    groups = {}

    def add(name, p, lr):
        no_wd = p.ndim <= 1 or name.endswith(".bias") or "norm" in name or "token" in name \
            or "position" in name
        key = (lr, no_wd)
        g = groups.setdefault(key, dict(params=[], lr=lr, weight_decay=0.0 if no_wd else wd))
        g["params"].append(p)

    for name, p in m.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("da.backbone."):
            if ".embeddings." in name:
                lid = 0
            elif ".encoder.layer." in name:
                lid = int(name.split(".encoder.layer.")[1].split(".")[0]) + 1
            else:
                lid = n_layers + 1
            add(name, p, lr_enc * layer_decay ** (n_layers + 1 - lid))
        elif name.startswith("da."):
            add(name, p, lr_dec)
        else:
            add(name, p, lr_head)
    out = list(groups.values())
    for g in out:
        g["base_lr"] = g["lr"]
    return out


def lr_factor(step, warmup, total, floor=0.01):
    if step < warmup:
        return (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


class EMA:
    def __init__(self, model, decay):
        m = model.module if hasattr(model, "module") else model
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in m.state_dict().items()}

    @torch.no_grad()
    def update(self, model, decay=None):
        d = self.decay if decay is None else decay
        m = model.module if hasattr(model, "module") else model
        for k, v in m.state_dict().items():
            s = self.shadow[k]
            if v.is_floating_point():
                s.mul_(d).add_(v.detach().float(), alpha=1 - d)
            else:
                s.copy_(v)

    def copy_to(self, model):
        m = model.module if hasattr(model, "module") else model
        m.load_state_dict({k: v.to(m.state_dict()[k].dtype) for k, v in self.shadow.items()})


# validation
@torch.no_grad()
def validate(model, loader, device, amp, max_batches=None):
    model.eval()
    se = ae = n = 0.0
    tall_se = tall_n = tall_bias = 0.0
    sx = sy = sxx = syy = sxy = 0.0
    inter = np.zeros(len(HN.CLASSES))
    union = np.zeros(len(HN.CLASSES))
    a_se = a_n = 0.0
    rel_r = []
    shown = None
    for bi, b in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        x = b["image"].to(device, non_blocking=True)
        g = b["gsd"].to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            out = model(x, g)
        size = b["height"].shape[-2:]
        pred = F.interpolate(out["height"][:, None].float(), size=size, mode="bilinear",
                             align_corners=False)[:, 0].cpu()
        t = b["height"]
        v = torch.isfinite(t)
        d = (pred - torch.nan_to_num(t))[v]
        se += float((d ** 2).sum())
        ae += float(d.abs().sum())
        n += int(v.sum())
        tv, pv = t[v].double(), pred[v].double()
        sx += float(tv.sum()); sy += float(pv.sum())
        sxx += float((tv * tv).sum()); syy += float((pv * pv).sum()); sxy += float((tv * pv).sum())
        tall = v & (torch.nan_to_num(t) >= 10)
        if tall.any():
            dt = (pred - torch.nan_to_num(t))[tall]
            tall_se += float((dt ** 2).sum())
            tall_bias += float(dt.sum())
            tall_n += int(tall.sum())
        cls = b["cls"]
        if (cls < len(HN.CLASSES)).any():
            pc = F.interpolate(out["seg"].float(), size=size, mode="bilinear",
                               align_corners=False).argmax(1).cpu()
            labelled = (cls >= 1) & (cls < len(HN.CLASSES))
            for c in range(1, len(HN.CLASSES)):
                pi, ti = (pc == c) & labelled, (cls == c)
                inter[c] += float((pi & ti).sum())
                union[c] += float((pi | ti).sum())
        th, tvh = L.downsample_target(t, v, out["rel"].shape[-2:])
        relh = out["rel"].float().cpu().numpy()
        bh = out["height"].float().cpu().numpy()
        for k in range(relh.shape[0]):
            vk = tvh[k].numpy()
            if vk.sum() < 256:
                continue
            gk = th[k].numpy()
            if gk[vk].std() > 0.3 and relh[k][vk].std() > 1e-9:
                rel_r.append(float(np.corrcoef(relh[k][vk], gk[vk])[0, 1]))
            if not HN.relative_informative(relh[k]):
                continue
            det = relh[k] - HN.ground_envelope(relh[k], float(b["gsd"][k]) * 2)
            alpha = HN.learned_scale(det, bh[k], vk)
            if alpha is not None:
                ea = (np.maximum(alpha * det, 0) - gk)[vk]
                a_se += float((ea ** 2).sum())
                a_n += int(vk.sum())
        if shown is None:
            shown = (b["image"][:4], t[:4], pred[:4],
                     F.interpolate(out["sigma"][:4, None].float(), size=size,
                                   mode="bilinear", align_corners=False)[:, 0].cpu())
    model.train()
    if n == 0:
        return {}, shown
    cov = sxy / n - (sx / n) * (sy / n)
    vx, vy = sxx / n - (sx / n) ** 2, syy / n - (sy / n) ** 2
    ious = [inter[c] / union[c] for c in range(1, len(HN.CLASSES)) if union[c] > 0]
    m = dict(rmse=math.sqrt(se / n), mae=ae / n,
             r=cov / math.sqrt(max(vx * vy, 1e-12)),
             rmse_tall=math.sqrt(tall_se / tall_n) if tall_n else float("nan"),
             bias_tall=tall_bias / tall_n if tall_n else float("nan"),
             miou=float(np.mean(ious)) if ious else float("nan"), pixels=n,
             rmse_a=math.sqrt(a_se / a_n) if a_n else float("nan"),
             rel_r=float(np.mean(rel_r)) if rel_r else float("nan"))
    return m, shown


def save_samples(path, shown, vmax=40.0):
    import cv2
    img, t, p, s = shown
    rows = []
    for i in range(img.shape[0]):
        rgb = img[i].numpy().transpose(1, 2, 0) * HN.STD + HN.MEAN
        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

        def cm(a, top):
            a = np.nan_to_num(a.numpy(), nan=0.0)
            return cv2.cvtColor(cv2.applyColorMap((np.clip(a / top, 0, 1) * 255).astype(np.uint8),
                                                  cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
        top = max(10.0, float(np.nanpercentile(t[i].numpy(), 99.5)) if torch.isfinite(t[i]).any() else vmax)
        rows.append(np.concatenate([rgb, cm(t[i], top), cm(p[i], top), cm(s[i], top / 4)], 1))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser(description="Fine-tune HeightNet")
    parser.add_argument("--data", nargs="+", required=True, help="tile roots (index.json)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--variant", default="small", choices=list(VARIANT_DEFAULTS))
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None, help="per GPU")
    parser.add_argument("--accum", type=int, default=1)
    parser.add_argument("--lr-enc", type=float, default=None)
    parser.add_argument("--lr-dec", type=float, default=1e-4)
    parser.add_argument("--lr-head", type=float, default=3e-4)
    parser.add_argument("--layer-decay", type=float, default=0.85)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--crop", type=int, default=HN.TILE)
    parser.add_argument("--mix", default="gamus=0.6,ahn=0.18,swisstopo=0.22")
    parser.add_argument("--longtail-power", type=float, default=0.5)
    parser.add_argument("--loss-weights", default="", help='JSON, e.g. {"seg":0.5}')
    parser.add_argument("--ema", type=float, default=0.999)
    parser.add_argument("--val-every", type=int, default=2000)
    parser.add_argument("--val-n", type=int, default=300, help="tiles per validation set")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--grad-ckpt", action="store_true", help="less memory, ~25%% slower")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--init", default=None, help="start from a HeightNet checkpoint's weights")
    parser.add_argument("--max-hours", type=float, default=11.3,
                    help="save and stop before a Kaggle session is killed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    dflt = VARIANT_DEFAULTS[args.variant]
    args.iters = args.iters or dflt["iters"]
    args.batch = args.batch or dflt["batch"]
    args.lr_enc = args.lr_enc or dflt["lr_enc"]
    rank, world, local = setup_dist()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    os.makedirs(args.out, exist_ok=True)
    t_start = time.time()

    # data
    train_tiles = D.gather(args.data, {"train"})
    val_sets = {}
    for root in args.data:
        vt = [t for t in D.load_index(root) if t["split"] == "val"]
        if vt:
            name = os.path.basename(os.path.normpath(root))
            val_sets[name] = vt
    by_src = {}
    for t in train_tiles:
        by_src[t["source"]] = by_src.get(t["source"], 0) + 1
    log0(rank, f"[data] train tiles {by_src}; val sets "
               f"{ {k: len(v) for k, v in val_sets.items()} }; mix {args.mix}")
    mix = D.parse_mix(args.mix)
    mix = {k: v for k, v in mix.items() if k in by_src} or None
    steps_per_rank = args.iters * args.batch * args.accum
    ds = D.TrainTiles(train_tiles, length=steps_per_rank + 64, crop=args.crop, mix=mix,
                      seed=args.seed * 1000 + rank)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True,
        drop_last=True, persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None)
    val_loaders = {}
    if is_main(rank):
        for name, vt in val_sets.items():
            vds = D.EvalTiles(vt, crop=args.crop, limit=args.val_n, seed=args.seed)
            val_loaders[name] = torch.utils.data.DataLoader(
                vds, batch_size=max(1, args.batch), shuffle=False,
                num_workers=min(2, args.workers))

    # model
    if args.variant == "tiny":
        model = HN.build(da_config=HN.tiny_da_config())
    else:
        model = HN.build(args.variant, pretrained=True)
    if args.init:
        m0, _ = HN.load_checkpoint(args.init)
        model.load_state_dict(m0.state_dict())
        log0(rank, f"[model] initialised from {args.init}")
    for n_, p in model.named_parameters():
        if "mask_token" in n_ or "fusion_stage.layers.0.residual_layer1" in n_:
            p.requires_grad_(False)
    if args.grad_ckpt:
        try:
            model.da.gradient_checkpointing_enable()
        except Exception as e:
            log0(rank, f"[model] gradient checkpointing unavailable: {e}")
    model.to(device)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    log0(rank, f"[model] HeightNet-{args.variant}: {n_par:.1f} M parameters, device {device}, "
               f"world {world}, amp {amp}")
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local] if device.type == "cuda" else None)

    opt = torch.optim.AdamW(param_groups(model, args.lr_enc, args.lr_dec, args.lr_head,
                                         args.layer_decay, args.wd), betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ema = EMA(model, args.ema)
    step, best = 0, float("inf")
    history = []
    if args.resume and os.path.exists(args.resume):
        st = torch.load(args.resume, map_location="cpu", weights_only=False)
        (model.module if world > 1 else model).load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        if st.get("scaler"):
            scaler.load_state_dict(st["scaler"])
        ema.shadow = {k: v.float().to(device) for k, v in st["ema"].items()}
        step, best, history = st["step"], st.get("best", best), st.get("history", [])
        log0(rank, f"[resume] from step {step}, best score {best:.3f}")
        prev_best = os.path.join(os.path.dirname(os.path.abspath(args.resume)), "heightnet_best.pt")
        here_best = os.path.join(args.out, "heightnet_best.pt")
        if is_main(rank) and os.path.exists(prev_best) and not os.path.exists(here_best):
            import shutil
            shutil.copy(prev_best, here_best)
        # And do not replay the first session's samples
        ds.seed += step
    weights = json.loads(args.loss_weights) if args.loss_weights else None

    def save_state(tag="state.pt"):
        if not is_main(rank):
            return
        m = model.module if world > 1 else model
        torch.save(dict(model=m.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(),
                        ema={k: v.half() for k, v in ema.shadow.items()}, step=step, best=best,
                        history=history, args=vars(args)),
                   os.path.join(args.out, tag + ".tmp"))
        os.replace(os.path.join(args.out, tag + ".tmp"), os.path.join(args.out, tag))

    def export(name, metrics=None):
        m = model.module if world > 1 else model
        live = {k: v.clone() for k, v in m.state_dict().items()}
        ema.copy_to(model)
        info = dict(step=step, variant=args.variant, iters=args.iters,
                    data=" + ".join(sorted(by_src)), val=metrics or {},
                    trained_utc=time.strftime("%Y-%m-%d %H:%M", time.gmtime()))
        HN.save_checkpoint(os.path.join(args.out, name), m, args.variant, extra=info)
        m.load_state_dict(live)

    def run_validation():
        nonlocal best
        m = model.module if world > 1 else model
        live = {k: v.clone() for k, v in m.state_dict().items()}
        ema.copy_to(model)
        row = dict(step=step)
        scores = []
        for name, vl in val_loaders.items():
            met, shown = validate(m, vl, device, amp)
            for k, v in met.items():
                row[f"{name}_{k}"] = v
            if met:
                scale = met["rmse"] if not np.isfinite(met["rmse_a"]) else \
                    0.5 * (met["rmse"] + met["rmse_a"])
                scores.append(scale)
                log0(rank, f"[val] {name:8s} B: RMSE {met['rmse']:.2f} m  MAE {met['mae']:.2f}  "
                           f"r {met['r']:.3f}  tall(>=10 m) RMSE {met['rmse_tall']:.2f} "
                           f"bias {met['bias_tall']:+.2f} | A: RMSE {met['rmse_a']:.2f} m  "
                           f"rank r {met['rel_r']:.3f} | mIoU {met['miou']:.3f}")
            if shown is not None:
                save_samples(os.path.join(args.out, "samples", f"{step:06d}_{name}.jpg"), shown)
        m.load_state_dict(live)
        score = float(np.mean(scores)) if scores else float("nan")
        row["score"] = score
        history.append(row)
        newfile = not os.path.exists(os.path.join(args.out, "val.csv"))
        with open(os.path.join(args.out, "val.csv"), "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if newfile:
                w.writeheader()
            w.writerow(row)
        if np.isfinite(score) and score < best:
            best = score
            export("heightnet_best.pt", row)
            log0(rank, f"[val] new best {score:.3f} -> heightnet_best.pt")
        export("heightnet_last.pt", row)

    # loop
    model.train()
    it = iter(loader)
    t0, seen, start_step = time.time(), 0, step
    logf = open(os.path.join(args.out, "log.csv"), "a", newline="") if is_main(rank) else None
    stop = False
    while step < args.iters and not stop:
        f = lr_factor(step, args.warmup, args.iters)
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * f
        opt.zero_grad(set_to_none=True)
        parts_acc = {}
        for micro in range(args.accum):
            b = next(it)
            x = b["image"].to(device, non_blocking=True)
            tgt = dict(height=b["height"].to(device, non_blocking=True),
                       cls=b["cls"].to(device, non_blocking=True))
            g_ = b["gsd"].to(device)
            sync = (micro == args.accum - 1) or world == 1
            ctx = model.no_sync() if (world > 1 and not sync) else _Null()
            with ctx:
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    out = model(x, g_)
                loss, parts = L.compute_losses(out, tgt, weights, args.longtail_power)
                scaler.scale(loss / args.accum).backward()
            for k, v in parts.items():
                parts_acc[k] = parts_acc.get(k, 0.0) + v / args.accum
            seen += x.shape[0] * world
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        ema.update(model, decay=min(args.ema, (1 + step) / (10 + step)))
        step += 1

        if is_main(rank) and (step % args.log_every == 0 or step == 1):
            elapsed = time.time() - t0
            ips = seen / max(elapsed, 1e-6)
            eta = (args.iters - step) * elapsed / max(step - start_step, 1)
            row = dict(step=step, lr=opt.param_groups[0]["lr"], grad_norm=float(gn),
                       img_per_s=ips, **{k: round(v, 4) for k, v in parts_acc.items()})
            print(f"[{step:6d}/{args.iters}] " + " ".join(
                f"{k} {v:.3f}" for k, v in parts_acc.items()) +
                f"  |g| {float(gn):.2f}  {ips:.1f} img/s  eta {eta / 3600:.1f} h", flush=True)
            if logf:
                w = csv.DictWriter(logf, fieldnames=list(row))
                if logf.tell() == 0:
                    w.writeheader()
                w.writerow(row)
                logf.flush()
        if step % args.val_every == 0 or step == args.iters:
            if is_main(rank):
                run_validation()
                save_state()
            if world > 1:
                dist.barrier()
        hours = (time.time() - t_start) / 3600
        flag = torch.tensor(float(hours > args.max_hours), device=device)
        if world > 1:
            dist.broadcast(flag, 0)
        if flag.item() > 0:
            log0(rank, f"[time] {hours:.2f} h used - saving and stopping at step {step}. "
                       f"Resume with --resume {os.path.join(args.out, 'state.pt')}")
            stop = True
    if is_main(rank):
        if step % args.val_every != 0 and step != args.iters:
            run_validation()
        save_state()
        with open(os.path.join(args.out, "summary.json"), "w") as f:
            json.dump(dict(step=step, iters=args.iters, best_score=best,
                           finished=step >= args.iters, hours=(time.time() - t_start) / 3600,
                           args=vars(args), last_val=history[-1] if history else None), f, indent=1)
        print(f"[done] step {step}/{args.iters}, best val score {best:.3f} m -> "
              f"{os.path.join(args.out, 'heightnet_best.pt')}", flush=True)
    if world > 1:
        dist.destroy_process_group()


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    main()
