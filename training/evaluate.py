#!/usr/bin/env python3
"""Score a HeightNet checkpoint on whole held-out tiles, through the same"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "mathsandml"))
import heightnet as HN  # noqa: E402
import data as D  # noqa: E402

BANDS = [(0, 2), (2, 10), (10, 30), (30, 1e9)]
ENGINES = ("A", "B", "A+B")


class Acc:
    def __init__(self):
        self.n = 0
        self.se = self.ae = self.sx = self.sy = self.sxx = self.syy = self.sxy = self.bias = 0.0

    def add(self, p, t):
        p, t = p.astype(np.float64), t.astype(np.float64)
        d = p - t
        self.n += d.size
        self.se += float((d * d).sum())
        self.ae += float(np.abs(d).sum())
        self.bias += float(d.sum())
        self.sx += float(t.sum()); self.sy += float(p.sum())
        self.sxx += float((t * t).sum()); self.syy += float((p * p).sum())
        self.sxy += float((t * p).sum())

    def result(self):
        if not self.n:
            return None
        n = self.n
        cov = self.sxy / n - self.sx / n * self.sy / n
        vx = self.sxx / n - (self.sx / n) ** 2
        vy = self.syy / n - (self.sy / n) ** 2
        return dict(rmse=math.sqrt(self.se / n), mae=self.ae / n, bias=self.bias / n,
                    r=cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else float("nan"),
                    pixels=n)


def tiles_for(roots, splits, limit, seed=0, exclude=()):
    import random
    out = []
    ex = set(exclude)
    for root in roots:
        tt = [t for t in D.load_index(root) if t["split"] in splits and t["id"] not in ex]
        if limit and len(tt) > limit:
            tt = random.Random(seed).sample(tt, limit)
        out += tt
    return out


def calibrate(hp, tiles, rotations, floor_m=0.3, per_tile=20000):
    """Decide the blend (see the module docstring)."""
    rng = np.random.default_rng(0)
    cols = {k: [] for k in ("a", "b", "sa", "sb", "y")}
    for t in tiles:
        rgb = D.read_rgb(os.path.join(t["root"], t["rgb"]))
        h = D.read_height(os.path.join(t["root"], t["height"]))
        r = hp.predict(rgb, t["gsd"], rotations=rotations)
        c = HN.combine(r, t["gsd"])
        if c["a"] is None:
            continue
        idx = np.flatnonzero(np.isfinite(h))
        if idx.size > per_tile:
            idx = rng.choice(idx, per_tile, replace=False)
        for k, arr in (("a", c["a"]), ("b", r["height"]), ("sa", c["a_sigma"]),
                       ("sb", r["sigma"]), ("y", h)):
            cols[k].append(np.asarray(arr, np.float64).ravel()[idx])
    if not cols["y"]:
        return None
    a, b, sa, sb, y = (np.concatenate(cols[k]) for k in ("a", "b", "sa", "sb", "y"))
    ca = max(float(np.median(np.abs(a - y) / np.maximum(sa, floor_m)) / 0.6745), 1e-3)
    cb = max(float(np.median(np.abs(b - y) / np.maximum(sb, floor_m)) / 0.6745), 1e-3)

    # Stacked: least-squares weight on A per band of B's predicted height
    edges = list(HN.STACK_BANDS)
    band = np.clip(np.searchsorted(edges, b, side="right") - 1, 0, len(edges) - 1)
    w_a = []
    for k in range(len(edges)):
        m = band == k
        d = b[m] - a[m]
        den = float((d * d).sum())
        w = float(((b[m] - y[m]) * d).sum() / den) if (m.sum() > 500 and den > 0) else 0.0
        w_a.append(min(max(w, 0.0), 1.0))

    def rmse(p):
        return float(np.sqrt(np.mean((p - y) ** 2)))
    base = dict(c_a=ca, c_b=cb, bands=edges, w_a=w_a, calib_tiles=len(cols["y"]))
    scores = {"A": rmse(a), "B": rmse(b),
              "ivw": rmse(HN.fuse(a, sa, b, sb, dict(base, mode="ivw"))[0]),
              "stacked": rmse(HN.fuse(a, sa, b, sb, dict(base, mode="stacked"))[0])}
    mode = "stacked" if scores["stacked"] <= scores["ivw"] else "ivw"
    print("[calib] RMSE on the calibration tiles: " +
          "  ".join(f"{k} {v:.2f}" for k, v in scores.items()) + f"  -> {mode}", flush=True)
    return dict(base, mode=mode, **{f"calib_rmse_{k}": v for k, v in scores.items()})


def main():
    parser = argparse.ArgumentParser(description="Score HeightNet (A, B, A+B) on held-out tiles")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--split", default="test,val")
    parser.add_argument("--calib-split", default="val")
    parser.add_argument("--calib-tiles", type=int, default=60, help="per data root")
    parser.add_argument("--no-calibrate", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--rotations", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="tiles per data root, 0 = all")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    predictor = HN.HeightPredictor(args.ckpt, device=args.device)
    if not args.no_calibrate:
        csplits = set(s.strip() for s in args.calib_split.split(","))
        ct = tiles_for(args.data, csplits, args.calib_tiles, seed=1)
        print(f"[calib] measuring each engine's honesty on {len(ct)} {sorted(csplits)} tiles",
              flush=True)
        fus = calibrate(predictor, ct, args.rotations)
        if fus:
            HN.set_fusion(args.ckpt, fus)
            predictor.fusion = fus
            print(f"[calib] mode {fus['mode']}; weight on A per B-height band "
                  f"{[round(w, 2) for w in fus['w_a']]} (bands {fus['bands']} m); "
                  f"c_a {fus['c_a']:.2f} c_b {fus['c_b']:.2f} -> written into {args.ckpt}",
                  flush=True)
        else:
            print("[calib] no tile had anything above ground to calibrate on - kept defaults")
        used = [t["id"] for t in ct]
    else:
        used = []

    splits = set(s.strip() for s in args.split.split(","))
    tiles = tiles_for(args.data, splits, args.limit, seed=2, exclude=used)
    if not tiles:
        sys.exit(f"no tiles with split in {splits}")
    print(f"[eval] {predictor.describe()} on {len(tiles)} tiles, {args.rotations} rotations, "
          f"device {predictor.device}, fusion {predictor.fusion or 'uncalibrated'}", flush=True)

    groups = {}

    def acc(engine, key):
        return groups.setdefault(key, {}).setdefault(engine, Acc())

    t0 = time.time()
    alphas, w_bs = [], []
    for i, t in enumerate(tiles):
        rgb = D.read_rgb(os.path.join(t["root"], t["rgb"]))
        h = D.read_height(os.path.join(t["root"], t["height"]))
        r = predictor.predict(rgb, t["gsd"], rotations=args.rotations)
        c = HN.combine(r, t["gsd"], fusion=predictor.fusion)
        v = np.isfinite(h)
        tv = h[v]
        preds = {"B": r["height"][v], "A+B": c["fused"][v]}
        if c["a"] is not None:
            preds["A"] = c["a"][v]
            alphas.append(c["alpha"])
            w_bs.append(float(np.mean(c["w_b"][v])))
        keys = ["all", f"source/{t['source']}", f"landscape/{t.get('landscape')}",
                f"region/{t['source']}/{t.get('region')}"]
        for k in keys:
            for e, p in preds.items():
                acc(e, k).add(p, tv)
            acc("flat-zero", k).add(np.zeros_like(tv), tv)
        for lo, hi in BANDS:
            m = (tv >= lo) & (tv < hi)
            if m.any():
                k = f"band/{lo:g}-{hi:g} m" if hi < 1e8 else f"band/{lo:g}+ m"
                for e, p in preds.items():
                    acc(e, k).add(p[m], tv[m])
        if (i + 1) % 25 == 0 or i + 1 == len(tiles):
            elapsed = time.time() - t0
            g = groups["all"]
            now = "  ".join(f"{e} {g[e].result()['rmse']:.2f}" for e in ENGINES if e in g)
            print(f"[eval] {i + 1}/{len(tiles)}  RMSE so far: {now} m  ({elapsed / (i + 1):.1f} s/tile)",
                  flush=True)

    result = {k: {e: acc_.result() for e, acc_ in v.items()} for k, v in sorted(groups.items())}
    summary = dict(checkpoint=args.ckpt, model=predictor.describe(), splits=sorted(splits),
                   tiles=len(tiles), rotations=args.rotations, fusion=predictor.fusion,
                   learned_alpha_median=float(np.median(alphas)) if alphas else None,
                   mean_weight_of_B=float(np.mean(w_bs)) if w_bs else None, results=result)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "eval.json"), "w") as f:
        json.dump(summary, f, indent=1)

    def cell(d, e, key="rmse"):
        x = (d.get(e) or {}).get(key) if d.get(e) else None
        return f"{x:.2f}" if x is not None else "-"
    lines = ["# HeightNet evaluation", "",
             f"{predictor.describe()} - {len(tiles)} held-out tiles ({', '.join(sorted(splits))}), "
             f"{args.rotations} rotations.", "",
             "A = relative depth calibrated by the learned prior (the problem statement's "
             "workflow); B = metres directly; A+B = fused per pixel (what the app ships).", "",
             "| group | A RMSE | B RMSE | **A+B RMSE** | A+B MAE | A+B r | flat-zero RMSE |",
             "|---|---|---|---|---|---|---|"]
    for k, d in result.items():
        lines.append(f"| {k} | {cell(d, 'A')} | {cell(d, 'B')} | **{cell(d, 'A+B')}** | "
                     f"{cell(d, 'A+B', 'mae')} | {cell(d, 'A+B', 'r')} | {cell(d, 'flat-zero')} |")
    lines += ["", f"Fusion calibration: {predictor.fusion or 'none'}. Mean weight given to B: "
                  f"{summary['mean_weight_of_B']}. Calibration tiles are excluded from the "
                  f"scores above."]
    with open(os.path.join(args.out, "eval.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
