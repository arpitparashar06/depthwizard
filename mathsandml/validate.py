"""How wrong is it?"""

import os
import json
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

NODATA_SENTINELS = (-9999.0, -32767.0, -32768.0, -3.4028234663852886e+38)


# 1. LOAD + PUT BOTH RASTERS ON ONE GRID
def _lstsq(A, y):
    with np.errstate(all="ignore"):
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef if np.all(np.isfinite(coef)) else None


def _sanitise(a):
    a = np.asarray(a, np.float64)
    a[~np.isfinite(a)] = np.nan
    for s in NODATA_SENTINELS:
        a[np.isclose(a, s, rtol=0, atol=1e-3)] = np.nan
    return a


def load_raster(path):
    """Returns (array float64, meta)."""
    meta = dict(path=path, crs=None, transform=None, px_size_m=None)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        try:
            import rasterio
            import math
            with rasterio.open(path) as ds:
                a = ds.read(1).astype(np.float64)
                if ds.nodata is not None:
                    a[a == ds.nodata] = np.nan
                if ds.crs is not None:
                    px = abs(ds.transform.a)
                    if ds.crs.is_geographic:
                        lat = (ds.bounds.bottom + ds.bounds.top) / 2.0
                        px = px * 111320.0 * math.cos(math.radians(lat))
                    meta.update(crs=ds.crs, transform=ds.transform, px_size_m=float(px))
                return _sanitise(a), meta
        except ImportError:
            pass
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    return _sanitise(np.array(Image.open(path))), meta


def reference_on_grid(ref, ref_meta, pred_meta, shape):
    """Put the reference onto the prediction's grid."""
    if ref.shape == shape:
        return ref, "same grid"

    if ref_meta.get("crs") is not None and pred_meta.get("crs") is not None:
        from rasterio.warp import reproject, Resampling
        dst = np.full(shape, np.nan, np.float64)
        reproject(ref, dst,
                  src_transform=ref_meta["transform"], src_crs=ref_meta["crs"],
                  dst_transform=pred_meta["transform"], dst_crs=pred_meta["crs"],
                  resampling=Resampling.bilinear,
                  src_nodata=np.nan, dst_nodata=np.nan)
        return dst, "reprojected"

    from PIL import Image
    filled = np.where(np.isfinite(ref), ref, np.nanmedian(ref))
    out = np.array(Image.fromarray(filled.astype(np.float32))
                   .resize((shape[1], shape[0]), Image.BILINEAR), np.float64)
    return out, "resized (footprints assumed identical - not verified)"


# 2. ALIGNMENT
def _robust_affine(x, y, iters=5):
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 10 or x.std() < 1e-9:
        return 1.0, 0.0
    A = np.c_[x, np.ones_like(x)]
    w = np.ones_like(x)
    coef = np.array([1.0, 0.0])
    for _ in range(iters):
        next_idx = _lstsq(A * w[:, None], y * w)
        if next_idx is None:
            break
        coef = next_idx
        with np.errstate(all="ignore"):
            r = y - A @ coef
            s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
            w = 1.0 / np.sqrt(1 + (r / (2 * s)) ** 2)
        if not np.all(np.isfinite(w)):
            break
    return float(coef[0]), float(coef[1])


def align(pred, ref, how):
    """How: 'raw' | 'shift' | 'affine'."""
    if how == "raw":
        return pred, dict(scale=1.0, offset=0.0)
    d = ref - pred
    if how == "shift":
        off = float(np.nanmedian(d))
        return pred + off, dict(scale=1.0, offset=off)
    a, b = _robust_affine(pred.ravel(), ref.ravel())
    return a * pred + b, dict(scale=a, offset=b)


# 3. METRICS
def metrics(pred, ref, mask=None):
    """Standard DSM accuracy set."""
    valid = np.isfinite(pred) & np.isfinite(ref)
    if mask is not None:
        valid &= mask
    n = int(valid.sum())
    if n < 10:
        return dict(n=n, note="too few valid pixels")

    p, r = pred[valid], ref[valid]
    e = p - r
    med = float(np.median(e))
    out = dict(
        n=n,
        coverage_pct=100.0 * n / pred.size,
        rmse=float(np.sqrt(np.mean(e ** 2))),
        mae=float(np.mean(np.abs(e))),
        medae=float(np.median(np.abs(e))),
        bias=float(np.mean(e)),
        std=float(np.std(e)),
        nmad=float(1.4826 * np.median(np.abs(e - med))),
        le90=float(np.percentile(np.abs(e), 90)),
        ref_range=float(np.percentile(r, 99) - np.percentile(r, 1)),
    )
    if p.std() > 1e-9 and r.std() > 1e-9:
        out["r"] = float(np.corrcoef(p, r)[0, 1])
        out["r2"] = out["r"] ** 2
        # 1 - SSE/SST: unlike r, this punishes bias and wrong scale
        out["nash_sutcliffe"] = float(1 - np.sum(e ** 2) / np.sum((r - r.mean()) ** 2))
    else:
        out["r"] = out["r2"] = out["nash_sutcliffe"] = float("nan")
    out["rmse_pct_of_range"] = 100.0 * out["rmse"] / max(out["ref_range"], 1e-9)
    return out


# 4. TERRAIN / OBJECT SPLIT
def object_height(surface, sigma_px):
    """Surface minus its own low-pass = height above local ground."""
    a = np.asarray(surface, np.float64)
    valid = np.isfinite(a)
    if not valid.all():
        a = np.where(valid, a, np.nanmedian(a[valid]) if valid.any() else 0.0)
    return np.where(valid, a - gaussian_filter(a, sigma_px), np.nan)


def attenuation(pred_obj, ref_obj, min_h=2.0, materiality=0.01):
    """Are our buildings the right height, and what one number would fix them?"""
    p_all = np.asarray(pred_obj, np.float64)
    r_all = np.asarray(ref_obj, np.float64)
    fin = np.isfinite(p_all) & np.isfinite(r_all)
    if fin.sum() < 50:
        return dict(n=int(fin.sum()), note="not enough finite pixels")

    p, r = p_all[fin], r_all[fin]
    denom = float(np.dot(p, p))
    gain = float(np.dot(p, r) / denom) if denom > 1e-12 else 1.0

    def _rmse(g):
        return float(np.sqrt(np.mean((g * p - r) ** 2)))

    rmse_1, rmse_g = _rmse(1.0), _rmse(gain)
    out = dict(n=int(fin.sum()),
               sxy=float(np.dot(p, r)), sxx=denom,
               suggested_alpha_gain=gain,
               gain_estimator="through-origin least squares over the full object band",
               rmse_at_gain_1=rmse_1,
               rmse_at_suggested=rmse_g,
               improves=bool(rmse_g < rmse_1 * (1.0 - materiality)))

    # Descriptive only: how tall are the structures both sides agree exist
    m = fin & (p_all > min_h) & (r_all > min_h)
    if m.sum() >= 50:
        pm, rm = p_all[m], r_all[m]
        pp, rp = float(np.percentile(pm, 99)), float(np.percentile(rm, 99))
        a, b = _robust_affine(pm, rm)
        out.update(n_structure=int(m.sum()),
                   height_ratio=pp / max(rp, 1e-9),   # predicted / true
                   pred_p99=pp, ref_p99=rp,
                   affine_slope=a, affine_intercept=b)
    return out


# 5. LANDSCAPE STRATIFICATION
def landscape_classes(surface, rgb=None, px_size_m=1.0, object_sigma_m=15.0):
    """Split the scene into urban / forest / hilly / sparse."""
    sigma = max(2.0, object_sigma_m / max(px_size_m, 1e-6))
    obj = object_height(surface, sigma)
    obj_f = np.where(np.isfinite(obj), obj, 0.0)

    win = max(3, int(round(sigma)))
    mean = uniform_filter(obj_f, win)
    rough = np.sqrt(np.maximum(uniform_filter(obj_f ** 2, win) - mean ** 2, 0))

    smooth = gaussian_filter(np.where(np.isfinite(surface), surface,
                                      np.nanmedian(surface)), sigma * 2)
    gy, gx = np.gradient(smooth, max(px_size_m, 1e-6))
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))

    if rgb is not None:
        c = np.asarray(rgb, np.float64)[..., :3]
        s = c.sum(2) + 1e-6
        veg = 2 * c[..., 1] / s - c[..., 0] / s - c[..., 2] / s   # excess green
    else:
        veg = np.zeros_like(rough)

    r_hi = np.nanpercentile(rough, 60)
    v_hi = np.nanpercentile(veg, 70) if rgb is not None else np.inf
    s_hi = max(np.nanpercentile(slope, 75), 5.0)

    cls = np.full(surface.shape, "sparse", dtype=object)
    hilly = slope > s_hi
    cls[hilly] = "hilly"
    urban = (rough > r_hi) & (veg <= v_hi)
    cls[urban] = "urban"
    forest = (rough > r_hi) & (veg > v_hi)
    cls[forest] = "forest"
    return cls, dict(roughness=rough, slope=slope, vegetation=veg,
                     thresholds=dict(rough=float(r_hi), veg=float(v_hi) if rgb is not None else None,
                                     slope=float(s_hi)),
                     object_sigma_px=float(sigma))


# 6. THE REPORT
def evaluate(pred, ref, rgb=None, px_size_m=1.0, object_sigma_m=15.0,
             match_resolution_m=None, class_mask=None):
    """Full comparison."""
    pred = _sanitise(pred)
    ref = _sanitise(ref)
    if pred.shape != ref.shape:
        raise ValueError(f"grids differ: pred {pred.shape} vs ref {ref.shape}")

    report = dict(px_size_m=float(px_size_m), shape=list(pred.shape))

    # resolution matching
    scored = pred
    if match_resolution_m and match_resolution_m > px_size_m * 1.5:
        s = match_resolution_m / px_size_m / 2.0
        ok = np.isfinite(pred)
        filled = np.where(ok, pred, np.nanmedian(pred[ok]))
        scored = np.where(ok, gaussian_filter(filled, s), np.nan)
        report["resolution_matched_to_m"] = float(match_resolution_m)

    # the three alignments, side by side
    report["alignment"] = {}
    for how in ("raw", "shift", "affine"):
        a, params = align(scored, ref, how)
        report["alignment"][how] = dict(metrics(a, ref), **params)

    best = "shift" if report["alignment"]["shift"]["rmse"] <= report["alignment"]["raw"]["rmse"] else "raw"
    report["headline_alignment"] = best
    report["headline"] = report["alignment"][best]
    aligned, _ = align(scored, ref, best)

    # object band
    sigma = max(2.0, object_sigma_m / max(px_size_m, 1e-6))
    p_obj, r_obj = object_height(aligned, sigma), object_height(ref, sigma)
    report["object_band"] = dict(metrics(p_obj, r_obj), sigma_px=float(sigma))
    report["attenuation"] = attenuation(p_obj, r_obj)

    # terrain band
    report["terrain_band"] = metrics(aligned - p_obj, ref - r_obj)

    # stratified
    if class_mask is None:
        cls, aux = landscape_classes(ref, rgb, px_size_m, object_sigma_m)
        report["class_source"] = "derived from reference surface + RGB (proxy, not land cover)"
        report["class_thresholds"] = aux["thresholds"]
    else:
        cls = np.asarray(class_mask, dtype=object)
        report["class_source"] = "user-supplied mask"

    report["by_landscape"] = {}
    for name in ("urban", "sparse", "hilly", "forest"):
        valid = (cls == name)
        if valid.sum() >= 50:
            report["by_landscape"][name] = dict(metrics(aligned, ref, valid),
                                             share_pct=100.0 * valid.mean())

    rmses = [v["rmse"] for v in report["by_landscape"].values() if "rmse" in v]
    if len(rmses) >= 2:
        report["stability"] = dict(
            worst_class=max(report["by_landscape"],
                            key=lambda k: report["by_landscape"][k].get("rmse", -1)),
            rmse_spread=float(max(rmses) - min(rmses)),
            rmse_ratio=float(max(rmses) / max(min(rmses), 1e-9)))

    # by height band
    report["by_height_band"] = {}
    bands = [("ground (<2 m)", -1e9, 2), ("low (2-10 m)", 2, 10),
             ("mid (10-30 m)", 10, 30), ("high (>30 m)", 30, 1e9)]
    for name, lo, hi in bands:
        valid = np.isfinite(r_obj) & (r_obj >= lo) & (r_obj < hi)
        if valid.sum() >= 50:
            report["by_height_band"][name] = dict(metrics(p_obj, r_obj, valid),
                                               share_pct=100.0 * valid.mean())

    report["_arrays"] = dict(aligned=aligned, error=aligned - ref, classes=cls)
    return report


# 7. PRESENTATION
def _row(m):
    return (f"{m.get('rmse', float('nan')):.2f} | {m.get('mae', float('nan')):.2f} | "
            f"{m.get('bias', float('nan')):+.2f} | {m.get('nmad', float('nan')):.2f} | "
            f"{m.get('r', float('nan')):.3f}")


def to_markdown(rep, units="m"):
    h, a = rep["headline"], rep["alignment"]
    lines = [f"### Accuracy vs reference  ({rep['shape'][1]} x {rep['shape'][0]} px, "
         f"{rep['px_size_m']:.2f} {units}/px)", ""]
    if "resolution_matched_to_m" in rep:
        lines.append(f"*Prediction low-passed to {rep['resolution_matched_to_m']:.0f} m "
                 f"to match the reference before scoring.*\n")

    lines += [f"| Alignment | RMSE | MAE | Bias | NMAD | r |",
          "|---|---|---|---|---|---|",
          f"| none | {_row(a['raw'])} |",
          f"| median shift ({a['shift']['offset']:+.1f} {units}) | {_row(a['shift'])} |",
          f"| robust affine (x{a['affine']['scale']:.2f}) | {_row(a['affine'])} |", ""]

    lines += [f"**Headline: RMSE {h['rmse']:.2f} {units}, MAE {h['mae']:.2f} {units}, "
          f"r = {h['r']:.3f}** "
          f"({rep['headline_alignment']} alignment, {h['n']:,} pixels, "
          f"{h['rmse_pct_of_range']:.1f}% of the reference's {h['ref_range']:.0f} {units} range)", ""]

    o, t = rep["object_band"], rep["terrain_band"]
    lines += ["| Band | RMSE | MAE | Bias | NMAD | r |", "|---|---|---|---|---|---|",
          f"| terrain (low-freq) | {_row(t)} |",
          f"| objects (above local ground) | {_row(o)} |", ""]

    at = rep["attenuation"]
    if "height_ratio" in at:
        d = 100 * (at["height_ratio"] - 1)
        word = "short" if d < 0 else "tall"
        lines.append(f"Where both surfaces agree a structure exists "
                 f"({at['n_structure']:,} px), predicted heights run "
                 f"**{abs(d):.0f}% too {word}** "
                 f"(p99 {at['pred_p99']:.1f} vs {at['ref_p99']:.1f} {units}).")
    if at.get("suggested_alpha_gain") is not None:
        g = at["suggested_alpha_gain"]
        if at.get("improves"):
            lines.append(f"Best single multiplier on alpha: **{g:.2f}** - it takes "
                     f"object-band RMSE from {at['rmse_at_gain_1']:.2f} to "
                     f"{at['rmse_at_suggested']:.2f} {units}. This is least "
                     f"squares through the origin, which is the argmin of "
                     f"squared error, so no other single gain does better.\n")
        else:
            lines.append(f"**Do not rescale.** The best possible multiplier is "
                     f"{g:.2f} and it does not meaningfully beat leaving alpha "
                     f"alone ({at['rmse_at_suggested']:.2f} vs "
                     f"{at['rmse_at_gain_1']:.2f} {units}). The error here is in "
                     f"WHERE the height sits, not how much of it there is, and "
                     f"a multiplier cannot move it.\n")

    if rep["by_landscape"]:
        lines += ["| Landscape | share | RMSE | MAE | Bias | NMAD | r |",
              "|---|---|---|---|---|---|---|"]
        for k, v in sorted(rep["by_landscape"].items(),
                           key=lambda kv: -kv[1].get("rmse", 0)):
            lines.append(f"| {k} | {v['share_pct']:.1f}% | {_row(v)} |")
        s = rep.get("stability")
        if s:
            lines.append(f"\nWeakest landscape: **{s['worst_class']}** "
                     f"(RMSE spread {s['rmse_spread']:.2f} {units}, "
                     f"ratio {s['rmse_ratio']:.1f}x across classes).")
        lines.append(f"\n<sub>{rep['class_source']}</sub>\n")

    if rep["by_height_band"]:
        lines += ["", "| Reference height | share | RMSE | MAE | Bias | NMAD | r |",
              "|---|---|---|---|---|---|---|"]
        for k, v in rep["by_height_band"].items():
            lines.append(f"| {k} | {v['share_pct']:.1f}% | {_row(v)} |")
    return "\n".join(lines)


def error_figures(rep, ref, rgb=None, outdir="outputs", units="m"):
    """Signed-error map + density scatter + per-class bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    err = rep["_arrays"]["error"]
    aligned = rep["_arrays"]["aligned"]
    paths = {}

    v = np.nanpercentile(np.abs(err), 95) or 1.0
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(err, cmap="RdBu_r", vmin=-v, vmax=v)
    ax.set_title(f"Signed error (estimate - reference), {units}")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.85)
    p = os.path.join(outdir, "error_map.png")
    fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
    paths["error_map"] = p

    valid = np.isfinite(aligned) & np.isfinite(ref)
    if valid.sum() > 100:
        x, y = ref[valid], aligned[valid]
        lo = float(min(np.percentile(x, 0.5), np.percentile(y, 0.5)))
        hi = float(max(np.percentile(x, 99.5), np.percentile(y, 99.5)))
        fig, ax = plt.subplots(figsize=(5.2, 5))
        ax.hexbin(x, y, gridsize=90, bins="log", extent=(lo, hi, lo, hi), cmap="viridis")
        ax.plot([lo, hi], [lo, hi], "--", color="#d62728", lw=1.3, label="1:1")
        ax.set_xlabel(f"reference ({units})")
        ax.set_ylabel(f"estimate ({units})")
        ax.set_title(f"RMSE {rep['headline']['rmse']:.2f} {units} · r {rep['headline']['r']:.3f}")
        ax.legend(loc="upper left", framealpha=.3)
        p = os.path.join(outdir, "scatter.png")
        fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
        paths["scatter"] = p

    if rep["by_landscape"]:
        ks = list(rep["by_landscape"])
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        ax.bar(ks, [rep["by_landscape"][k]["rmse"] for k in ks], color="#3f7fb5")
        ax.set_ylabel(f"RMSE ({units})")
        ax.set_title("Stability across landscapes")
        for i, k in enumerate(ks):
            ax.text(i, rep["by_landscape"][k]["rmse"],
                    f"{rep['by_landscape'][k]['share_pct']:.0f}%",
                    ha="center", va="bottom", fontsize=9)
        p = os.path.join(outdir, "stability.png")
        fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
        paths["stability"] = p
    return paths


def save_report(rep, outdir="outputs", units="m"):
    """Report.json + report.md, with the arrays stripped out of the JSON."""
    os.makedirs(outdir, exist_ok=True)
    clean = {k: v for k, v in rep.items() if k != "_arrays"}
    jp = os.path.join(outdir, "validation.json")
    with open(jp, "w") as f:
        json.dump(clean, f, indent=2, default=float)
    mp = os.path.join(outdir, "validation.md")
    with open(mp, "w", encoding="utf-8") as f:
        f.write(to_markdown(rep, units))
    return jp, mp


def validate_files(pred_path, ref_path, rgb_path=None, outdir="outputs",
                   object_sigma_m=15.0, match_resolution_m=None):
    """End-to-end from paths."""
    pred, pmeta = load_raster(pred_path)
    ref_raw, rmeta = load_raster(ref_path)
    ref, how = reference_on_grid(ref_raw, rmeta, pmeta, pred.shape)

    rgb = None
    if rgb_path:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        im = Image.open(rgb_path).convert("RGB")
        if im.size != (pred.shape[1], pred.shape[0]):
            im = im.resize((pred.shape[1], pred.shape[0]), Image.BILINEAR)
        rgb = np.array(im)

    px = pmeta.get("px_size_m") or 1.0
    if match_resolution_m is None and rmeta.get("px_size_m"):
        match_resolution_m = rmeta["px_size_m"]

    report = evaluate(pred, ref, rgb=rgb, px_size_m=px,
                   object_sigma_m=object_sigma_m,
                   match_resolution_m=match_resolution_m)
    report["reference"] = dict(path=ref_path, regrid=how,
                            px_size_m=rmeta.get("px_size_m"))
    return report, ref, rgb


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Score a DSM against reference elevation data.")
    ap.add_argument("pred")
    ap.add_argument("ref")
    ap.add_argument("--rgb", default=None, help="optical image, improves class split")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--sigma-m", type=float, default=15.0,
                    help="building scale; sets the terrain/object split")
    ap.add_argument("--match-res", type=float, default=None,
                    help="low-pass the prediction to this GSD before scoring")
    args = ap.parse_args()

    rep, ref, rgb = validate_files(args.pred, args.ref, args.rgb, args.outdir,
                                   args.sigma_m, args.match_res)
    print(to_markdown(rep))
    error_figures(rep, ref, rgb, args.outdir)
    print("\n" + " ".join(save_report(rep, args.outdir)))
