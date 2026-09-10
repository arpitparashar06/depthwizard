#!/usr/bin/env python3
"""
run_benchmark.py - score DepthWizard's absolute-mode DSM against reference LiDAR.

Answers the 50%-weighted accuracy half of the marking scheme: RMSE, MAE and
correlation per scene, plus the stratified urban / sparse / hilly / forest
table and a cross-scene stability figure.

Scene layout (one directory per scene under --scenes):

    scenes/<name>/rgb.tif        georeferenced optical image  (REQUIRED, needs a CRS)
    scenes/<name>/ref_dsm.tif    reference surface model       (REQUIRED)
    scenes/<name>/ref_dtm.tif    reference terrain model       (optional)
    scenes/<name>/scene.json     optional metadata, any of:
                                   {"known_height_m": 40,
                                    "gcps": [[row, col, height_m], ...],
                                    "sun_azimuth": 160, "sun_elevation": 42,
                                    "landscape": "urban", "notes": "..."}

Scale sources (--scale-source), in the order the problem statement allows them:

    gcps-from-ref  sample N ground control points from the reference nDSM.
                   This is the "limited set of Ground Control Points" path.
                   N is tiny (default 8) and the sampled points are written
                   into the report so the disclosure is explicit.
    known-height   one semantic prior per scene, from scene.json.
    shadow         sun angles from scene.json or the GeoTIFF tags.
    scene          whatever scene.json specifies, per scene.

Usage:

    python run_benchmark.py --scenes benchmark/scenes --out benchmark/results
    python run_benchmark.py --scenes benchmark/scenes --scale-source known-height
    python run_benchmark.py --scenes benchmark/scenes --limit 1 --no-figures
"""

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (HERE, ROOT):        # works from benchmark/ or the repo root
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_env():
    """Read KEY=value lines from .env before anything imports inference.

    inference.py binds OPENTOPO_KEY at module import time, so this has to run
    first or the key is read as empty and the DEM fetch silently degrades to
    height-above-ground. Real environment variables always win.
    """
    # repo root first (that is where .env lives), then mathsandml/, then here
    for path in (os.path.join(os.path.dirname(ROOT), ".env"),
                 os.path.join(ROOT, ".env"), os.path.join(HERE, ".env")):
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


_load_env()

import inference as I          # noqa: E402
import validate as V           # noqa: E402

LANDSCAPES = ("urban", "sparse", "hilly", "forest")
HEADLINE_KEYS = ("rmse", "mae", "medae", "bias", "nmad", "le90", "r", "r2",
                 "nash_sutcliffe", "rmse_pct_of_range", "n", "coverage_pct")


# ---------------------------------------------------------------------------
# scene discovery
# ---------------------------------------------------------------------------
def _first(dirpath, *names):
    for n in names:
        p = os.path.join(dirpath, n)
        if os.path.exists(p):
            return p
    return None


def discover_scenes(root):
    """Every subdirectory holding both an RGB and a reference raster."""
    scenes = []
    if not os.path.isdir(root):
        return scenes
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        rgb = _first(d, "rgb.tif", "rgb.tiff", "image.tif", "ortho.tif")
        ref = _first(d, "ref_dsm.tif", "ref_dsm.tiff", "dsm_ref.tif", "reference.tif")
        if not rgb or not ref:
            print(f"[skip] {name}: need rgb.tif and ref_dsm.tif "
                  f"(found rgb={bool(rgb)} ref={bool(ref)})")
            continue
        cfg = {}
        cfg_path = os.path.join(d, "scene.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
        scenes.append(dict(name=name, dir=d, rgb=rgb, ref=ref,
                           dtm=_first(d, "ref_dtm.tif", "ref_dtm.tiff", "dtm_ref.tif"),
                           cfg=cfg))
    return scenes


# ---------------------------------------------------------------------------
# reference object height on the prediction grid
# ---------------------------------------------------------------------------
def fill_nodata(a, smooth_px=8.0):
    """Interpolate a terrain model across its holes.

    A LiDAR DTM carries NO DATA under buildings - the ground beneath a roof was
    never measured. On an AHN downtown tile that is 38% of the raster, and it
    lands exactly where the buildings are. Subtracting it raw makes every
    building read as 0 m of structure, so the benchmark would score a
    high-rise scene as flat ground and the control-point sampler would find
    nothing to calibrate against.

    Nearest-valid fill, then smoothed, because ground under a building is a
    continuation of the ground around it rather than a nearest-neighbour
    staircase. Returns (filled, fraction_filled) so the report can disclose it.
    """
    from scipy.ndimage import distance_transform_edt
    a = np.asarray(a, np.float64)
    ok = np.isfinite(a)
    if ok.all():
        return a, 0.0
    if not ok.any():
        return np.zeros_like(a), 1.0
    _, (iy, ix) = distance_transform_edt(~ok, return_indices=True)
    smoothed = gaussian_filter(a[iy, ix], smooth_px)
    return np.where(ok, a, smoothed), float((~ok).mean())


def reference_ndsm(scene, pred_meta, shape, sigma_m=15.0):
    """Reference height-above-ground, resampled onto the image grid.

    Uses a real DTM when the scene ships one; otherwise falls back to the same
    low-pass residual validate.py uses, so the two stay consistent.
    """
    ref_raw, rmeta = V.load_raster(scene["ref"])
    ref, how = V.reference_on_grid(ref_raw, rmeta, pred_meta, shape)

    if scene.get("dtm"):
        dtm_raw, dmeta = V.load_raster(scene["dtm"])
        dtm, _ = V.reference_on_grid(dtm_raw, dmeta, pred_meta, shape)
        dtm, holes = fill_nodata(dtm)
        nd = ref - dtm
        src = "ref_dsm - ref_dtm"
        if holes > 0.001:
            src += " (%.0f%% of the DTM interpolated under buildings)" % (100 * holes)
    else:
        px = pred_meta.get("px_size_m") or 1.0
        nd = V.object_height(ref, max(2.0, sigma_m / max(px, 1e-6)))
        src = f"ref_dsm - lowpass({sigma_m:g} m)"

    return np.maximum(np.nan_to_num(nd, nan=0.0), 0.0), ref, how, src


def write_grid(path, arr, pred_meta):
    """Write an array onto the prediction's grid, so validate.py reprojects it
    as a first-class georeferenced raster rather than assuming a footprint."""
    import rasterio
    with rasterio.open(path, "w", driver="GTiff",
                       height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype="float32", crs=pred_meta["crs"],
                       transform=pred_meta["transform"], nodata=np.nan) as ds:
        ds.write(arr.astype(np.float32), 1)
    return path


def sample_gcps(ndsm, n=8, min_h=6.0, margin=16, seed=0):
    """A small, spread-out set of control points on real structures.

    Deterministic: the scene is tiled into an n-cell grid and the tallest
    reliable pixel inside each cell is taken, so the points cover the frame
    instead of clustering on the single tallest block.
    """
    h, w = ndsm.shape
    m = np.zeros_like(ndsm, bool)
    m[margin:h - margin, margin:w - margin] = True
    m &= np.isfinite(ndsm) & (ndsm > min_h)
    if m.sum() < n:
        return []

    rows = int(np.floor(np.sqrt(n)))
    cols = int(np.ceil(n / max(rows, 1)))
    pts = []
    rng = np.random.default_rng(seed)
    for i in range(rows):
        for j in range(cols):
            if len(pts) >= n:
                break
            r0, r1 = i * h // rows, (i + 1) * h // rows
            c0, c1 = j * w // cols, (j + 1) * w // cols
            cell = np.zeros_like(m)
            cell[r0:r1, c0:c1] = True
            cell &= m
            if cell.sum() == 0:
                continue
            vals = np.where(cell, ndsm, -np.inf)
            # the 95th percentile inside the cell, not the max: a lone spike is
            # exactly the blunder a control point must not be anchored to
            thr = np.percentile(ndsm[cell], 95)
            cand = np.argwhere(cell & (vals >= thr))
            r, c = cand[rng.integers(len(cand))]
            pts.append([int(r), int(c), float(ndsm[r, c])])
    return pts


# ---------------------------------------------------------------------------
# one scene
# ---------------------------------------------------------------------------
def run_scene(scene, outdir, args):
    name = scene["name"]
    out = os.path.join(outdir, name)
    os.makedirs(out, exist_ok=True)
    rec = dict(scene=name, rgb=scene["rgb"], reference=scene["ref"],
               has_ref_dtm=bool(scene.get("dtm")))
    t0 = time.time()

    rgb, meta = I.load_image(scene["rgb"])
    rec.update(width=int(rgb.shape[1]), height=int(rgb.shape[0]),
               mode=meta["mode"], px_size_m=meta.get("px_size_m"),
               crs=str(meta.get("crs")))
    print(f"\n=== {name}: {rgb.shape[1]}x{rgb.shape[0]} px, mode={meta['mode']}, "
          f"px={meta.get('px_size_m')} ===")

    if meta["mode"] != "absolute":
        rec["error"] = ("rgb.tif carries no usable CRS/transform, so the pipeline "
                        "runs in relative mode and there is nothing metric to score")
        return rec

    ndsm_ref, _, regrid, ndsm_src = reference_ndsm(
        scene, meta, rgb.shape[:2], sigma_m=args.sigma_m)
    rec.update(reference_regrid=regrid, reference_ndsm_source=ndsm_src,
               reference_p99_object_m=float(np.percentile(ndsm_ref, 99)))

    # ---- pick the scale source --------------------------------------------
    cfg = scene["cfg"]
    kh = gcps = az = el = None
    src = args.scale_source
    if src == "scene":
        if cfg.get("gcps"):
            src = "known-gcps"
        elif cfg.get("known_height_m"):
            src = "known-height"
        elif cfg.get("sun_azimuth") is not None:
            src = "shadow"
        else:
            src = "gcps-from-ref"

    if src == "gcps-from-ref":
        gcps = sample_gcps(ndsm_ref, n=args.n_gcps, seed=args.seed)
        if len(gcps) < 2:
            rec["error"] = (f"only {len(gcps)} control points clear the "
                            f"{6.0:g} m structure threshold - scene has no "
                            f"vertical structure to calibrate against")
            return rec
        rec["gcps_used"] = gcps
    elif src == "known-gcps":
        gcps = cfg["gcps"]
        rec["gcps_used"] = gcps
    elif src == "known-height":
        # Match the statistic the calibrator actually anchors on. This pairing
        # has bitten us twice in opposite directions, so it is now derived
        # from ONE constant that lives in inference.py rather than a literal
        # repeated here:
        #   - hand it p99.5 while the calibrator anchored p97 and every scene
        #     came out more than twice too tall (pooled 6.57 m vs 5.28 m);
        #   - leave p97 here after the calibrator moved to p99 and every scene
        #     comes out too SHORT by the same factor.
        # If HEIGHT_REFERENCE_PCT changes again, this follows it automatically.
        #
        # We deliberately do NOT tune this percentile. Sweeping the stand-in
        # percentile against the calibrator anchor (de-biased DEM, shift
        # aligned) gives a surface that is monotonic toward smaller
        # structures - p90/p99 scores 5.40 m against p99/p99's 6.81 m - but
        # only because shrinking every structure walks the prediction toward
        # the do-nothing baseline. Picking that corner would be optimising
        # for the metric rather than for height estimation. The stand-in
        # stays matched to what the calibrator's number MEANS.
        kh = cfg.get("known_height_m")
        if not kh:
            # THIS USED TO FALL BACK TO THE REFERENCE, and that quietly
            # invalidated every headline number in the report: the prior that
            # sets metres-per-unit for the whole scene was read out of the
            # ground truth the scene is then scored against. A reader who
            # spots it discounts the entire accuracy table, and rightly.
            #
            # It is also unnecessary. Scored with a prior read off the ortho
            # instead (40 m for Rotterdam, against the p99-of-reference value
            # this branch used to supply), RMSE went from 9.36 m to 8.45 m -
            # the honest number was the BETTER one.
            rec["error"] = (
                "no known_height_m in scene.json. Add the height of the "
                "tallest structure you can identify in the ortho - it must "
                "come from the imagery or an external source, never from the "
                "reference raster this scene is scored against. Or pass "
                "--scale-source gcps-from-ref to measure the pipeline's "
                "ceiling with a deliberately leaky prior, which is a diagnostic "
                "and never a headline.")
            return rec
        kh = float(kh)
        rec["known_height_m"] = kh
        rec["known_height_source"] = cfg.get(
            "known_height_source", "scene.json (operator-supplied prior)")
    elif src == "shadow":
        az = cfg.get("sun_azimuth")
        el = cfg.get("sun_elevation")
        if az is None or el is None:
            rec["error"] = "shadow calibration needs sun_azimuth and sun_elevation"
            return rec
        az, el = float(az), float(el)
    rec["scale_source"] = src

    # ---- the pipeline ------------------------------------------------------
    try:
        height, meta2, info = I.estimate_elevation(
            scene["rgb"], known_height_m=kh, gcps=gcps,
            sun_azimuth=az, sun_elevation=el,
            use_dem=args.use_dem, alpha_gain=args.alpha_gain, outdir=out,
            debias_coarse_dem=not args.no_debias)
    except Exception as e:
        traceback.print_exc()
        rec["error"] = f"{type(e).__name__}: {e}"
        return rec

    rec["inference_s"] = round(time.time() - t0, 1)

    # ---- score the surface the product actually ships ----------------------
    # This harness used to validate the raw estimate while both the CLI and the
    # web UI export a REFINED dsm.tif. That means the published accuracy figure
    # described a surface no user ever receives. refine() self-checks and
    # declines when it makes things worse, so running it here costs little and
    # removes the mismatch. --no-refine reproduces the old behaviour.
    if not args.no_refine:
        import refine as R
        height, rinfo = R.refine(
            height, rgb, px_size_m=(meta2.get("px_size_m") or 1.0),
            object_sigma_m=float(info.get("sigma_m") or args.sigma_m),
            flatten=args.flatten, sharpen=args.sharpen, verbose=False)
        height, info = I.clip_below_ground(height, info)
        # meta2 is what estimate_elevation returned: same CRS and transform as
        # meta, plus the uncertainty raster, which has to be re-exported with
        # the surface it belongs to
        info = I.export_products(height, rgb, meta2, out, info,
                                 uncertainty=meta2.get("_uncertainty"))
        rec["refine"] = {k: v for k, v in rinfo.items()
                         if not isinstance(v, np.ndarray)}
    rec["refined"] = not args.no_refine

    rec["info"] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                   for k, v in info.items() if not isinstance(v, np.ndarray)}

    # ---- scoring -----------------------------------------------------------
    pred_path = os.path.join(out, "dsm.tif")
    if not os.path.exists(pred_path):
        rec["error"] = "pipeline produced no dsm.tif"
        return rec

    # Score like against like. With no coarse DEM the pipeline emits height
    # ABOVE LOCAL GROUND, and scoring that against an absolute sea-level DSM
    # compares two different quantities - the terrain the prediction never
    # claimed to know dominates the error. Only when a DEM actually supplied
    # the terrain baseline is the output an absolute surface.
    dem_used = "DEM terrain" in str(rec["info"].get("calibration", ""))
    rec["dem_used"] = dem_used
    if dem_used:
        score_ref, datum = scene["ref"], "absolute DSM (metres above sea level)"
    else:
        score_ref = write_grid(os.path.join(out, "ref_ndsm.tif"), ndsm_ref, meta)
        datum = f"height above ground ({ndsm_src})"
    rec["scored_against"] = datum

    try:
        rep, refarr, rgbarr = V.validate_files(
            pred_path, score_ref, scene["rgb"], outdir=out,
            object_sigma_m=args.sigma_m)
        if not args.no_figures:
            V.error_figures(rep, refarr, rgbarr, outdir=out)
        V.save_report(rep, out)
    except Exception as e:
        traceback.print_exc()
        rec["error"] = f"validation failed: {type(e).__name__}: {e}"
        return rec

    rec["headline_alignment"] = rep.get("headline_alignment")
    # The headline may be SHIFT-aligned, i.e. a constant vertical offset
    # computed FROM THE REFERENCE has been removed. That is standard practice
    # when comparing elevation products on different vertical datums, but it
    # is not the accuracy of the un-touched output and must never be reported
    # as if it were. Keep the raw figures beside it.
    rec["raw_alignment"] = {k: rep.get("alignment", {}).get("raw", {}).get(k)
                            for k in HEADLINE_KEYS
                            if k in rep.get("alignment", {}).get("raw", {})}
    rec["headline"] = {k: rep["headline"].get(k) for k in HEADLINE_KEYS
                       if k in rep["headline"]}
    rec["object_band"] = {k: rep["object_band"].get(k) for k in HEADLINE_KEYS
                          if k in rep["object_band"]}
    rec["terrain_band"] = {k: rep["terrain_band"].get(k) for k in HEADLINE_KEYS
                           if k in rep["terrain_band"]}
    rec["attenuation"] = rep.get("attenuation")
    rec["by_landscape"] = rep.get("by_landscape", {})
    rec["by_height_band"] = rep.get("by_height_band", {})
    rec["stability"] = rep.get("stability")
    rec["total_s"] = round(time.time() - t0, 1)

    h = rec["headline"]
    print(f"[{name}] RMSE {h.get('rmse', float('nan')):.2f} m  "
          f"MAE {h.get('mae', float('nan')):.2f} m  "
          f"r {h.get('r', float('nan')):.3f}  ({rec['total_s']:.0f}s)")
    return rec


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def pool(rows):
    """Pixel-weighted pooling. RMSE pools in quadrature, the rest linearly."""
    rows = [r for r in rows if r and r.get("n") and np.isfinite(r.get("rmse", np.nan))]
    if not rows:
        return None
    n = np.array([r["n"] for r in rows], float)
    w = n / n.sum()
    out = dict(n=int(n.sum()), scenes=len(rows))
    out["rmse"] = float(np.sqrt(np.sum(w * np.array([r["rmse"] for r in rows]) ** 2)))
    for k in ("mae", "medae", "bias", "nmad", "le90", "r"):
        vals = np.array([r.get(k, np.nan) for r in rows], float)
        if np.isfinite(vals).any():
            out[k] = float(np.nansum(w * vals))
    return out


def fmt(v, nd=2):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    return f"{v:.{nd}f}"


def build_report(records, args):
    ok = [r for r in records if "headline" in r and not r.get("error")]
    bad = [r for r in records if r.get("error")]

    L = ["# DepthWizard - DSM accuracy against reference LiDAR", ""]
    L += [f"Scenes attempted: **{len(records)}**, scored: **{len(ok)}**, failed: **{len(bad)}**  ",
          f"Depth backbone: `{os.environ.get('DEPTH_MODEL', 'depth-anything/Depth-Anything-V2-Large-hf')}`  ",
          f"Scale source: `{args.scale_source}`"
          + (f" ({args.n_gcps} control points per scene)" if args.scale_source == "gcps-from-ref" else "")
          + f"  \nCoarse DEM terrain baseline: `{'on' if args.use_dem else 'off'}`  ",
          f"Object/terrain split: sized per scene from the imagery "
          f"(the {args.sigma_m:g} m setting is the fallback when that is off)  \n"
          f"Scored surface: `{'refined - the same dsm.tif the CLI and UI export' if not args.no_refine else 'raw estimate, pre-refine'}`  ", ""]

    # -- how the headline was aligned, stated before any number is shown ------
    aligns = sorted({r.get("headline_alignment") for r in ok
                     if r.get("headline_alignment")})
    if aligns:
        praw = pool([r["raw_alignment"] for r in ok if r.get("raw_alignment")])
        L += ["> ### Read this before quoting any figure",
              "> ",
              f"> Headline alignment used: {', '.join('`%s`' % a for a in aligns)}. "
              "`shift` means a single constant vertical offset, **computed from "
              "the reference**, was subtracted before scoring. Elevation "
              "products routinely sit on different vertical datums, so removing "
              "one constant is normal practice - but it is not the accuracy of "
              "the untouched output, and a figure quoted without this sentence "
              "is misleading.",
              "> "]
        if praw and praw.get("rmse") is not None:
            L += [f"> Pooled RMSE **with** that offset removed: "
                  f"**{fmt(pool([r['headline'] for r in ok])['rmse'])} m**.  ",
                  f"> Pooled RMSE of the raw output, **no alignment at all**: "
                  f"**{fmt(praw['rmse'])} m**.",
                  "> "]
        L += ["> Quote both, or quote the raw one.", ""]

    datums = sorted({r.get("scored_against") for r in ok if r.get("scored_against")})
    if datums:
        L += ["Scored against: " + ", ".join(f"`{d}`" for d in datums) + ".  ",
              "When no coarse DEM supplies the terrain baseline the pipeline emits "
              "height above local ground, so it is scored against the reference "
              "nDSM rather than the absolute surface - otherwise the terrain the "
              "prediction never claimed to know would dominate the error.", ""]

    if not ok:
        L += ["> No scene scored successfully.", ""]

    # -- headline per scene ---------------------------------------------------
    if ok:
        L += ["## Per-scene accuracy", "",
              "All values in metres. `r` is Pearson correlation against the reference; ",
              "`NSE` is Nash-Sutcliffe, which unlike `r` penalises bias and wrong scale.", "",
              "| Scene | px (m) | RMSE | RMSE raw | MAE | MedAE | Bias | NMAD | LE90 | r | NSE |",
              "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for r in ok:
            h = r["headline"]
            L.append(f"| {r['scene']} | {fmt(r.get('px_size_m'))} | "
                     f"{fmt(h.get('rmse'))} | {fmt(r.get('raw_alignment', {}).get('rmse'))} | "
                     f"{fmt(h.get('mae'))} | {fmt(h.get('medae'))} | "
                     f"{fmt(h.get('bias'))} | {fmt(h.get('nmad'))} | {fmt(h.get('le90'))} | "
                     f"{fmt(h.get('r'), 3)} | {fmt(h.get('nash_sutcliffe'), 3)} |")
        p = pool([r["headline"] for r in ok])
        if p:
            _praw = pool([r["raw_alignment"] for r in ok if r.get("raw_alignment")])
            L.append(f"| **pooled** | | **{fmt(p['rmse'])}** | "
                     f"**{fmt((_praw or {}).get('rmse'))}** | **{fmt(p.get('mae'))}** | "
                     f"{fmt(p.get('medae'))} | {fmt(p.get('bias'))} | {fmt(p.get('nmad'))} | "
                     f"{fmt(p.get('le90'))} | {fmt(p.get('r'), 3)} | |")
        L.append("")

        # -- object band ------------------------------------------------------
        L += ["## Structure heights only (object band)", "",
              "The surface minus its own low-pass, i.e. how well building and canopy ",
              "heights are recovered once the terrain baseline is removed. This is the ",
              "number that reflects the depth model rather than the DEM.", "",
              "| Scene | RMSE | MAE | Bias | r | pred p99 | ref p99 | height ratio | suggested alpha gain |",
              "|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for r in ok:
            o, a = r.get("object_band", {}), r.get("attenuation") or {}
            L.append(f"| {r['scene']} | {fmt(o.get('rmse'))} | {fmt(o.get('mae'))} | "
                     f"{fmt(o.get('bias'))} | {fmt(o.get('r'), 3)} | "
                     f"{fmt(a.get('pred_p99'))} | {fmt(a.get('ref_p99'))} | "
                     f"{fmt(a.get('height_ratio'), 3)} | {fmt(a.get('suggested_alpha_gain'), 3)} |")
        L.append("")
        # Only scenes where the best possible multiplier actually beats 1.0
        # get a vote. The previous version averaged a biased estimator over
        # every scene and told you to rescale by 1.85; measured, that made
        # things worse everywhere. See validate.attenuation for the arithmetic.
        helpful = [(r.get("attenuation") or {}) for r in ok]
        gains = [a["suggested_alpha_gain"] for a in helpful
                 if a.get("improves") and np.isfinite(a.get("suggested_alpha_gain", np.nan))]
        if gains and len(gains) >= max(1, len(ok) // 2):
            L += [f"Median suggested `--alpha-gain`: **{np.median(gains):.3f}** "
                  f"({len(gains)} of {len(ok)} scenes improve under a pure "
                  f"multiplier; the rest are misplaced height, which no gain "
                  f"can fix).", ""]
        else:
            L += [f"**No alpha-gain is recommended.** Only {len(gains)} of "
                  f"{len(ok)} scenes improve under any single multiplier, so "
                  f"the dominant error is WHERE height sits, not how much of "
                  f"it there is. Rescaling would trade one error for another.", ""]

        # -- stratified -------------------------------------------------------
        L += ["## Stability across landscape types", "",
              "Classes are proxies derived from the reference surface and the imagery ",
              "(roughness, excess green, low-frequency relief), not a land-cover product.", "",
              "| Landscape | scenes | share % | RMSE | MAE | Bias | NMAD | r |",
              "|---|--:|--:|--:|--:|--:|--:|--:|"]
        for cls in LANDSCAPES:
            rows = [r["by_landscape"].get(cls) for r in ok
                    if r.get("by_landscape", {}).get(cls)]
            p = pool(rows)
            if not p:
                L.append(f"| {cls} | 0 | - | - | - | - | - | - |")
                continue
            share = np.mean([x["share_pct"] for x in rows if "share_pct" in x]) \
                if any("share_pct" in x for x in rows) else float("nan")
            L.append(f"| {cls} | {p['scenes']} | {fmt(share, 1)} | {fmt(p['rmse'])} | "
                     f"{fmt(p.get('mae'))} | {fmt(p.get('bias'))} | {fmt(p.get('nmad'))} | "
                     f"{fmt(p.get('r'), 3)} |")
        L.append("")

        cls_rmse = {}
        for cls in LANDSCAPES:
            p = pool([r["by_landscape"].get(cls) for r in ok
                      if r.get("by_landscape", {}).get(cls)])
            if p:
                cls_rmse[cls] = p["rmse"]
        if len(cls_rmse) >= 2:
            worst = max(cls_rmse, key=cls_rmse.get)
            best = min(cls_rmse, key=cls_rmse.get)
            L += [f"Spread: **{max(cls_rmse.values()) - min(cls_rmse.values()):.2f} m** "
                  f"between `{best}` ({cls_rmse[best]:.2f} m) and `{worst}` "
                  f"({cls_rmse[worst]:.2f} m), a ratio of "
                  f"**{max(cls_rmse.values()) / max(min(cls_rmse.values()), 1e-9):.2f}x**.", ""]

        # -- height bands -----------------------------------------------------
        bands = []
        for r in ok:
            bands += list(r.get("by_height_band", {}).keys())
        if bands:
            seen = [b for b in ["ground (<2 m)", "low (2-10 m)", "mid (10-30 m)",
                                "high (>30 m)"] if b in set(bands)]
            L += ["## Accuracy by structure height", "",
                  "| Height band | scenes | share % | RMSE | MAE | Bias |",
                  "|---|--:|--:|--:|--:|--:|"]
            for b in seen:
                rows = [r["by_height_band"].get(b) for r in ok
                        if r.get("by_height_band", {}).get(b)]
                p = pool(rows)
                if not p:
                    continue
                share = np.mean([x["share_pct"] for x in rows if "share_pct" in x])
                L.append(f"| {b} | {p['scenes']} | {fmt(share, 1)} | {fmt(p['rmse'])} | "
                         f"{fmt(p.get('mae'))} | {fmt(p.get('bias'))} |")
            L.append("")

    # -- disclosures ----------------------------------------------------------
    L += ["## How scale was set", ""]
    if args.scale_source == "gcps-from-ref":
        L += [f"Each scene was calibrated from **{args.n_gcps} ground control points** "
              "sampled from the reference nDSM, one per grid cell, taken at the 95th "
              "percentile of object height inside that cell. This is the "
              "\"limited set of Ground Control Points\" path the problem statement "
              "allows, and the points are listed per scene in `results.json`. "
              "Everything outside those points is a genuine prediction.", "",
              "The honest caveat: the control points come from the same reference "
              "raster used for scoring, so absolute scale is not independently "
              "validated. The **object band** table above is the leakage-resistant "
              "number - it measures relative structure geometry, which a handful of "
              "point heights cannot fake across a whole scene.", ""]
    elif args.scale_source == "known-height":
        L += ["Each scene was calibrated from **one semantic prior** - roughly how "
              "tall the tallest sustained structure is - read off the ortho and "
              "stored in `scene.json`. No reference elevation enters the "
              "calibration. A scene without a prior is reported as a failure "
              "rather than silently borrowing one from the ground truth, which "
              "is what this harness used to do.", ""]
        L += ["| Scene | prior | where it came from |", "|---|--:|---|"]
        for r in ok:
            L.append(f"| {r['scene']} | {fmt(r.get('known_height_m'), 0)} m | "
                     f"{r.get('known_height_source', 'scene.json')} |")
        L.append("")
    elif args.scale_source == "shadow":
        L += ["Scale came from **shadow length** against the sun angles, so no "
              "reference elevation entered the calibration at all. This is the only "
              "fully independent path here.", ""]

    if bad:
        L += ["## Failures", "",
              "| Scene | error |", "|---|---|"]
        for r in bad:
            msg = str(r["error"])[:200].replace("|", r"\|")
            L.append(f"| {r['scene']} | {msg} |")
        L.append("")

    L += ["## Per-scene artefacts", "",
          "Each scene directory under the results folder holds `dsm.tif`, `ndsm.tif`, "
          "`dtm.tif`, `terrain.glb` inputs, `validation.md` / `.json`, plus "
          "`error_map.png`, `scatter.png` and `stability.png`.", ""]
    return "\n".join(L)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", default="benchmark/scenes")
    ap.add_argument("--out", default="benchmark/results")
    # known-height is the default because it is the STABLE estimator. It anchors
    # on an aggregate over thousands of pixels; gcps-from-ref fits alpha through
    # a handful of individual pixels, and at r~0.5-0.7 an individual pixel is
    # noisy. Measured pooled RMSE: 5.28 m from the prior, against 6.94 m from 10
    # control points and 9.16 m from 5 - fewer points made it worse, not better.
    ap.add_argument("--scale-source", default="known-height",
                    choices=["gcps-from-ref", "known-height", "shadow", "scene"])
    ap.add_argument("--n-gcps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma-m", type=float, default=15.0,
                    help="building scale; sets the terrain/object split")
    ap.add_argument("--alpha-gain", type=float, default=1.0)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--use-dem", dest="use_dem", action="store_true", default=True,
                   help="fetch the coarse DEM for the terrain baseline, giving "
                        "absolute sea-level elevations (needs OPENTOPO_KEY). "
                        "On by default; falls back automatically if it fails.")
    g.add_argument("--no-dem", dest="use_dem", action="store_false",
                   help="skip the DEM; output is height above local ground and "
                        "is scored against the reference nDSM")
    ap.add_argument("--no-debias", action="store_true",
                    help="ablation: leave rooftop contamination in the coarse "
                         "elevation model")
    ap.add_argument("--no-refine", action="store_true",
                    help="score the raw estimate instead of the refined surface "
                         "the CLI and UI actually export (the old behaviour)")
    ap.add_argument("--flatten", type=float, default=0.8,
                    help="refine: structure plane-fit strength")
    ap.add_argument("--sharpen", type=float, default=0.4,
                    help="refine: object-band unsharp amount")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N scenes")
    ap.add_argument("--only", default=None, help="comma-separated scene names")
    args = ap.parse_args()

    scenes = discover_scenes(args.scenes)
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        scenes = [s for s in scenes if s["name"] in want]
    if args.limit:
        scenes = scenes[:args.limit]

    if not scenes:
        print(f"No scenes found under {args.scenes!r}.\n"
              f"Each scene is a directory holding rgb.tif + ref_dsm.tif.\n"
              f"Run fetch_data.py first, or see BENCHMARK.md for manual downloads.")
        return 1

    os.makedirs(args.out, exist_ok=True)
    print(f"{len(scenes)} scene(s): {', '.join(s['name'] for s in scenes)}")

    records = []
    for s in scenes:
        try:
            records.append(run_scene(s, args.out, args))
        except Exception as e:
            traceback.print_exc()
            records.append(dict(scene=s["name"], error=f"{type(e).__name__}: {e}"))

    md = build_report(records, args)
    with open(os.path.join(args.out, "report.md"), "w") as f:
        f.write(md)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(dict(config=vars(args), scenes=records), f, indent=2, default=float)

    print("\n" + md)
    print(f"\nWrote {os.path.join(args.out, 'report.md')} and results.json")
    return 0 if any("headline" in r for r in records) else 2


if __name__ == "__main__":
    sys.exit(main())
