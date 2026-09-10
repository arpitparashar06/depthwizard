"""
validate.py - how wrong is it? Score our DSM against real LiDAR.

===========================================================================
READ THIS FIRST
===========================================================================
    validate_files(pred.tif, reference.tif)
        load_raster()          both files
        reference_on_grid()    put the reference onto OUR pixel grid, by true
                               reprojection when both are georeferenced
        evaluate()             everything below
            align()            three ways - see THE HONESTY PROBLEM
            metrics()          RMSE, MAE, bias, NMAD, LE90, r, Nash-Sutcliffe
            object_height()    split the surface into TERRAIN and OBJECTS and
                               score each separately
            attenuation()      are our buildings the right height, and what
                               single multiplier would fix them
            landscape_classes()  urban / sparse / hilly / forest, so the
                               report can show where it is weak
        to_markdown()          the report a human reads
        error_figures()        error map, scatter against 1:1, per-class bars

THE HONESTY PROBLEM, and why every number is quoted three times. Our surface
and a LiDAR DSM often sit on different vertical datums - if no coarse DEM was
available, ours is height above LOCAL GROUND while theirs is metres above sea
level. Scoring those raw measures the datum, not the model, and quietly
subtracting the difference flatters us. So the report shows all three side by
side and lets the reader pick:

    raw       nothing removed. The honest number
    shift     one constant offset removed. Normal practice between datums
    affine    offset AND scale removed. Diagnostic only

TWO MORE TRAPS THIS AVOIDS:

  * RESOLUTION. If the reference is 30 m and we are 0.5 m, we get punished for
    detail the reference cannot represent. match_resolution low-passes us to
    their GSD first, and keeps the unmatched score alongside.

  * THE TERRAIN BAND DOMINATES. On a hilly scene r can be 0.98 with every
    single building wrong, because the landform carries the correlation. That
    is why object heights are scored on their own.

===========================================================================

The evaluation criteria ask for RMSE, MAE and correlation against LiDAR or
reference data, and for "performance stability across urban, sparse, hilly and
forested landscapes". This module supplies both halves.

Three things in here matter more than the headline number:

  1. DATUM ALIGNMENT. Without a coarse DEM the pipeline outputs height above
     LOCAL GROUND; a LiDAR DSM is height above the ellipsoid or geoid. Scoring
     those raw is a constant offset of tens or hundreds of metres and tells you
     nothing about the shape. So every metric is reported three ways - raw,
     after a median shift, and after a robust affine fit - and the three are
     shown side by side rather than the flattering one being picked.

  2. RESOLUTION MATCHING. If the reference is 30 m SRTM and the prediction is
     0.5 m, the prediction is penalised for detail the reference cannot
     represent. `match_resolution` low-passes the prediction to the reference
     GSD before scoring, and the un-matched score is kept alongside.

  3. THE TERRAIN BAND DOMINATES. On a hilly scene, r=0.98 can be entirely the
     landform, with every building wrong. So object heights (surface minus its
     own low-pass) are scored separately, and the regression slope through them
     is the attenuation factor - the thing the README predicts is ~0.7.

CLI:
    python validate.py outputs/dsm.tif reference_lidar.tif --rgb scene.tif
"""

import os
import json
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

NODATA_SENTINELS = (-9999.0, -32767.0, -32768.0, -3.4028234663852886e+38)


# ----------------------------------------------------------------------------
# 1. LOAD + PUT BOTH RASTERS ON ONE GRID
# ----------------------------------------------------------------------------
def _lstsq(A, y):
    """Least squares that neither prints spurious warnings nor returns inf.

    macOS/Accelerate raises floating-point flags for the unused lanes of its
    vectorised matmul, so a healthy solve prints "divide by zero", "overflow"
    and "invalid" at once - noise that trains you to ignore the real thing. A
    genuinely singular design matrix returns inf/nan coefficients, which would
    propagate silently into the result; None lets the caller fall back.
    """
    with np.errstate(all="ignore"):
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef if np.all(np.isfinite(coef)) else None


def _sanitise(a):
    """float64 with every flavour of nodata turned into NaN."""
    a = np.asarray(a, np.float64)
    a[~np.isfinite(a)] = np.nan
    for s in NODATA_SENTINELS:
        a[np.isclose(a, s, rtol=0, atol=1e-3)] = np.nan
    return a


def load_raster(path):
    """Returns (array float64, meta). Works with or without georeferencing."""
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
    """Put the reference onto the prediction's grid.

    Georeferenced both sides -> true reproject, which is the only correct path.
    Otherwise -> resize, which assumes the two rasters cover the same footprint.
    That assumption is stated in the returned report, never made silently.
    """
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


# ----------------------------------------------------------------------------
# 2. ALIGNMENT
# ----------------------------------------------------------------------------
def _robust_affine(x, y, iters=5):
    """y ~= a*x + b, Cauchy-weighted so buildings and blunders cannot drag it."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 10 or x.std() < 1e-9:
        return 1.0, 0.0
    A = np.c_[x, np.ones_like(x)]
    w = np.ones_like(x)
    coef = np.array([1.0, 0.0])
    for _ in range(iters):
        nxt = _lstsq(A * w[:, None], y * w)
        if nxt is None:                 # degenerate reweighting - keep the last
            break
        coef = nxt
        with np.errstate(all="ignore"):
            r = y - A @ coef
            s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
            w = 1.0 / np.sqrt(1 + (r / (2 * s)) ** 2)
        if not np.all(np.isfinite(w)):
            break
    return float(coef[0]), float(coef[1])


def align(pred, ref, how):
    """how: 'raw' | 'shift' | 'affine'. Returns (aligned_pred, params)."""
    if how == "raw":
        return pred, dict(scale=1.0, offset=0.0)
    d = ref - pred
    if how == "shift":
        off = float(np.nanmedian(d))
        return pred + off, dict(scale=1.0, offset=off)
    a, b = _robust_affine(pred.ravel(), ref.ravel())
    return a * pred + b, dict(scale=a, offset=b)


# ----------------------------------------------------------------------------
# 3. METRICS
# ----------------------------------------------------------------------------
def metrics(pred, ref, mask=None):
    """Standard DSM accuracy set. Everything is computed on the same pixels."""
    m = np.isfinite(pred) & np.isfinite(ref)
    if mask is not None:
        m &= mask
    n = int(m.sum())
    if n < 10:
        return dict(n=n, note="too few valid pixels")

    p, r = pred[m], ref[m]
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


# ----------------------------------------------------------------------------
# 4. TERRAIN / OBJECT SPLIT
# ----------------------------------------------------------------------------
def object_height(surface, sigma_px):
    """Surface minus its own low-pass = height above local ground.

    NaNs are filled before the blur, otherwise one hole smears across a whole
    neighbourhood and the residual is wrong far from the actual gap.
    """
    a = np.asarray(surface, np.float64)
    ok = np.isfinite(a)
    if not ok.all():
        a = np.where(ok, a, np.nanmedian(a[ok]) if ok.any() else 0.0)
    return np.where(ok, a - gaussian_filter(a, sigma_px), np.nan)


def attenuation(pred_obj, ref_obj, min_h=2.0, materiality=0.01):
    """Are our buildings the right height, and what one number would fix them?

    IN PLAIN ENGLISH: suppose every building we produce is 30% too short. That
    is not a mistake in WHERE the height is, only in HOW MUCH, and one
    multiplier on alpha fixes the whole scene. This function finds the best
    possible such multiplier, and - just as important - says whether it is
    worth using at all. If the best possible multiplier barely helps, the error
    is in the placement, and rescaling cannot fix placement.

    "Best possible" is not a search: least squares through the origin IS the
    argmin of squared error, so sum(p*r) / sum(p*p) is provably the answer.

    ---------------------------------------------------------------------
    alpha is a pure multiplier, so the correction is a THROUGH-ORIGIN gain, and
    the only gain worth recommending is the one that minimises object-band
    error. That is exactly sum(p*r) / sum(p*p) over every finite pixel: least
    squares through the origin IS the argmin of squared error, so there is
    nothing to search and no other single multiplier can beat it.

    WHAT THIS USED TO DO, AND WHY IT WAS WRONG. It reported median(r / p) over
    a mask that required the REFERENCE to clear min_h but the PREDICTION only
    0.25 * min_h. That mask keeps every pixel the model MISSED - where r/p
    explodes - and drops every false positive, where the ratio would collapse.
    So the statistic measured miss rate, not scale, and was biased upward by
    construction. Measured on the Rotterdam tile:

        estimator                      gain    object-band RMSE after applying
        median(r/p)      (old)         2.825   10.83 m
        median(r)/median(p)            2.366    9.23 m
        LSQ on the masked pixels       1.102    6.16 m
        LSQ on the full band  (new)    0.736    5.95 m
        ---------------------------------------------------------------
        leaving alpha alone            1.000    6.06 m

    The old answer was not merely imprecise: taking its advice nearly doubled
    the error, and it pointed the wrong way - these structures are slightly too
    TALL, not 65% too short. The pooled report inherited the same bias and told
    every scene to rescale by 1.85.

    The height RATIO is still reported, because "are the buildings the right
    height" is a fair question - but it is measured only where BOTH surfaces
    agree a structure exists, and it is no longer used as the multiplier.

    Returns `improves` = False when the best possible gain does not beat 1.0 by
    `materiality`, which is the signal that the error is in WHERE the height
    sits rather than how much of it there is. Rescaling cannot fix that.
    """
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
               suggested_alpha_gain=gain,
               gain_estimator="through-origin least squares over the full object band",
               rmse_at_gain_1=rmse_1,
               rmse_at_suggested=rmse_g,
               improves=bool(rmse_g < rmse_1 * (1.0 - materiality)))

    # descriptive only: how tall are the structures both sides agree exist
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


# ----------------------------------------------------------------------------
# 5. LANDSCAPE STRATIFICATION
# ----------------------------------------------------------------------------
def landscape_classes(surface, rgb=None, px_size_m=1.0, object_sigma_m=15.0):
    """Split the scene into urban / forest / hilly / sparse.

    These are proxies computed from the imagery and the reference surface, not
    a land-cover product. Pass your own mask to `class_mask` if you have one.

      roughness   - local std of object height. Structures and canopy are high,
                    bare and flat ground is low.
      vegetation  - excess green from RGB. Separates canopy from concrete, which
                    roughness alone cannot do.
      terrain slope - slope of the SMOOTHED surface, so buildings do not
                    register as hills.

    Priority: forest > urban > hilly > sparse, because a wooded hillside should
    be scored as forest (canopy is the hard part) not as terrain.
    """
    sig = max(2.0, object_sigma_m / max(px_size_m, 1e-6))
    obj = object_height(surface, sig)
    obj_f = np.where(np.isfinite(obj), obj, 0.0)

    win = max(3, int(round(sig)))
    mean = uniform_filter(obj_f, win)
    rough = np.sqrt(np.maximum(uniform_filter(obj_f ** 2, win) - mean ** 2, 0))

    smooth = gaussian_filter(np.where(np.isfinite(surface), surface,
                                      np.nanmedian(surface)), sig * 2)
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
                     object_sigma_px=float(sig))


# ----------------------------------------------------------------------------
# 6. THE REPORT
# ----------------------------------------------------------------------------
def evaluate(pred, ref, rgb=None, px_size_m=1.0, object_sigma_m=15.0,
             match_resolution_m=None, class_mask=None):
    """Full comparison. Returns a dict; nothing is printed."""
    pred = _sanitise(pred)
    ref = _sanitise(ref)
    if pred.shape != ref.shape:
        raise ValueError(f"grids differ: pred {pred.shape} vs ref {ref.shape}")

    rep = dict(px_size_m=float(px_size_m), shape=list(pred.shape))

    # -- resolution matching --------------------------------------------------
    scored = pred
    if match_resolution_m and match_resolution_m > px_size_m * 1.5:
        s = match_resolution_m / px_size_m / 2.0
        ok = np.isfinite(pred)
        filled = np.where(ok, pred, np.nanmedian(pred[ok]))
        scored = np.where(ok, gaussian_filter(filled, s), np.nan)
        rep["resolution_matched_to_m"] = float(match_resolution_m)

    # -- the three alignments, side by side -----------------------------------
    rep["alignment"] = {}
    for how in ("raw", "shift", "affine"):
        a, params = align(scored, ref, how)
        rep["alignment"][how] = dict(metrics(a, ref), **params)

    best = "shift" if rep["alignment"]["shift"]["rmse"] <= rep["alignment"]["raw"]["rmse"] else "raw"
    rep["headline_alignment"] = best
    rep["headline"] = rep["alignment"][best]
    aligned, _ = align(scored, ref, best)

    # -- object band ----------------------------------------------------------
    sig = max(2.0, object_sigma_m / max(px_size_m, 1e-6))
    p_obj, r_obj = object_height(aligned, sig), object_height(ref, sig)
    rep["object_band"] = dict(metrics(p_obj, r_obj), sigma_px=float(sig))
    rep["attenuation"] = attenuation(p_obj, r_obj)

    # -- terrain band ---------------------------------------------------------
    rep["terrain_band"] = metrics(aligned - p_obj, ref - r_obj)

    # -- stratified -----------------------------------------------------------
    if class_mask is None:
        cls, aux = landscape_classes(ref, rgb, px_size_m, object_sigma_m)
        rep["class_source"] = "derived from reference surface + RGB (proxy, not land cover)"
        rep["class_thresholds"] = aux["thresholds"]
    else:
        cls = np.asarray(class_mask, dtype=object)
        rep["class_source"] = "user-supplied mask"

    rep["by_landscape"] = {}
    for name in ("urban", "sparse", "hilly", "forest"):
        m = (cls == name)
        if m.sum() >= 50:
            rep["by_landscape"][name] = dict(metrics(aligned, ref, m),
                                             share_pct=100.0 * m.mean())

    rmses = [v["rmse"] for v in rep["by_landscape"].values() if "rmse" in v]
    if len(rmses) >= 2:
        rep["stability"] = dict(
            worst_class=max(rep["by_landscape"],
                            key=lambda k: rep["by_landscape"][k].get("rmse", -1)),
            rmse_spread=float(max(rmses) - min(rmses)),
            rmse_ratio=float(max(rmses) / max(min(rmses), 1e-9)))

    # -- by height band -------------------------------------------------------
    rep["by_height_band"] = {}
    bands = [("ground (<2 m)", -1e9, 2), ("low (2-10 m)", 2, 10),
             ("mid (10-30 m)", 10, 30), ("high (>30 m)", 30, 1e9)]
    for name, lo, hi in bands:
        m = np.isfinite(r_obj) & (r_obj >= lo) & (r_obj < hi)
        if m.sum() >= 50:
            rep["by_height_band"][name] = dict(metrics(p_obj, r_obj, m),
                                               share_pct=100.0 * m.mean())

    rep["_arrays"] = dict(aligned=aligned, error=aligned - ref, classes=cls)
    return rep


# ----------------------------------------------------------------------------
# 7. PRESENTATION
# ----------------------------------------------------------------------------
def _row(m):
    return (f"{m.get('rmse', float('nan')):.2f} | {m.get('mae', float('nan')):.2f} | "
            f"{m.get('bias', float('nan')):+.2f} | {m.get('nmad', float('nan')):.2f} | "
            f"{m.get('r', float('nan')):.3f}")


def to_markdown(rep, units="m"):
    h, a = rep["headline"], rep["alignment"]
    L = [f"### Accuracy vs reference  ({rep['shape'][1]} x {rep['shape'][0]} px, "
         f"{rep['px_size_m']:.2f} {units}/px)", ""]
    if "resolution_matched_to_m" in rep:
        L.append(f"*Prediction low-passed to {rep['resolution_matched_to_m']:.0f} m "
                 f"to match the reference before scoring.*\n")

    L += [f"| Alignment | RMSE | MAE | Bias | NMAD | r |",
          "|---|---|---|---|---|---|",
          f"| none | {_row(a['raw'])} |",
          f"| median shift ({a['shift']['offset']:+.1f} {units}) | {_row(a['shift'])} |",
          f"| robust affine (x{a['affine']['scale']:.2f}) | {_row(a['affine'])} |", ""]

    L += [f"**Headline: RMSE {h['rmse']:.2f} {units}, MAE {h['mae']:.2f} {units}, "
          f"r = {h['r']:.3f}** "
          f"({rep['headline_alignment']} alignment, {h['n']:,} pixels, "
          f"{h['rmse_pct_of_range']:.1f}% of the reference's {h['ref_range']:.0f} {units} range)", ""]

    o, t = rep["object_band"], rep["terrain_band"]
    L += ["| Band | RMSE | MAE | Bias | NMAD | r |", "|---|---|---|---|---|---|",
          f"| terrain (low-freq) | {_row(t)} |",
          f"| objects (above local ground) | {_row(o)} |", ""]

    at = rep["attenuation"]
    if "height_ratio" in at:
        d = 100 * (at["height_ratio"] - 1)
        word = "short" if d < 0 else "tall"
        L.append(f"Where both surfaces agree a structure exists "
                 f"({at['n_structure']:,} px), predicted heights run "
                 f"**{abs(d):.0f}% too {word}** "
                 f"(p99 {at['pred_p99']:.1f} vs {at['ref_p99']:.1f} {units}).")
    if at.get("suggested_alpha_gain") is not None:
        g = at["suggested_alpha_gain"]
        if at.get("improves"):
            L.append(f"Best single multiplier on alpha: **{g:.2f}** - it takes "
                     f"object-band RMSE from {at['rmse_at_gain_1']:.2f} to "
                     f"{at['rmse_at_suggested']:.2f} {units}. This is least "
                     f"squares through the origin, which is the argmin of "
                     f"squared error, so no other single gain does better.\n")
        else:
            L.append(f"**Do not rescale.** The best possible multiplier is "
                     f"{g:.2f} and it does not meaningfully beat leaving alpha "
                     f"alone ({at['rmse_at_suggested']:.2f} vs "
                     f"{at['rmse_at_gain_1']:.2f} {units}). The error here is in "
                     f"WHERE the height sits, not how much of it there is, and "
                     f"a multiplier cannot move it.\n")

    if rep["by_landscape"]:
        L += ["| Landscape | share | RMSE | MAE | Bias | NMAD | r |",
              "|---|---|---|---|---|---|---|"]
        for k, v in sorted(rep["by_landscape"].items(),
                           key=lambda kv: -kv[1].get("rmse", 0)):
            L.append(f"| {k} | {v['share_pct']:.1f}% | {_row(v)} |")
        s = rep.get("stability")
        if s:
            L.append(f"\nWeakest landscape: **{s['worst_class']}** "
                     f"(RMSE spread {s['rmse_spread']:.2f} {units}, "
                     f"ratio {s['rmse_ratio']:.1f}x across classes).")
        L.append(f"\n<sub>{rep['class_source']}</sub>\n")

    if rep["by_height_band"]:
        L += ["", "| Reference height | share | RMSE | MAE | Bias | NMAD | r |",
              "|---|---|---|---|---|---|---|"]
        for k, v in rep["by_height_band"].items():
            L.append(f"| {k} | {v['share_pct']:.1f}% | {_row(v)} |")
    return "\n".join(L)


def error_figures(rep, ref, rgb=None, outdir="outputs", units="m"):
    """Signed-error map + density scatter + per-class bars. Returns paths."""
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

    m = np.isfinite(aligned) & np.isfinite(ref)
    if m.sum() > 100:
        x, y = ref[m], aligned[m]
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
    """report.json + report.md, with the arrays stripped out of the JSON."""
    os.makedirs(outdir, exist_ok=True)
    clean = {k: v for k, v in rep.items() if k != "_arrays"}
    jp = os.path.join(outdir, "validation.json")
    with open(jp, "w") as f:
        json.dump(clean, f, indent=2, default=float)
    mp = os.path.join(outdir, "validation.md")
    with open(mp, "w", encoding="utf-8") as f:
        f.write(to_markdown(rep, units))
    return jp, mp


# ----------------------------------------------------------------------------
def validate_files(pred_path, ref_path, rgb_path=None, outdir="outputs",
                   object_sigma_m=15.0, match_resolution_m=None):
    """End-to-end from paths. Used by the CLI and by the Gradio validation tab."""
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

    rep = evaluate(pred, ref, rgb=rgb, px_size_m=px,
                   object_sigma_m=object_sigma_m,
                   match_resolution_m=match_resolution_m)
    rep["reference"] = dict(path=ref_path, regrid=how,
                            px_size_m=rmeta.get("px_size_m"))
    return rep, ref, rgb


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
