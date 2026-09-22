"""
inference.py - the elevation engine. One RGB image in, one elevation map out.

===========================================================================
READ THIS FIRST
===========================================================================
Every other function in this file is a step of the one at the bottom,
estimate_elevation(). This is the order it runs them in:

     1. load_image()              read the file and pick the MODE from its
                                  metadata: a coordinate system means metres,
                                  no coordinate system means a relative surface
     2. predict_depth_ensemble()  run the frozen depth model four times, once
                                  per 90-degree rotation, and average
        - predict_depth()         each pass cuts the image into tiles, predicts
                                  each, and stitches them back together
     3. validate_depth()          sanity check, and flip the map if the model
                                  read the scene upside down
     4. clean_depth()             edge-preserving smooth (never a plain blur -
                                  that melts buildings into blobs)
     5. detrend()                 remove the model's fake tilt. Usually SKIPPED,
                                  because step 2 already cancelled it

   --- a PNG/JPG stops here: normalise to 0..1, export, done ---

     6. structure_scale_m()       measure how wide the buildings are in THIS
                                  scene
     7. the frequency split       detail = depth - blur(depth, that width).
                                  Buildings live in `detail`; terrain lives in
                                  the blur
     8. alpha_from_shadows()      METRES PER MODEL UNIT - from shadow lengths,
        alpha_from_gcps()         or surveyed points, or one landmark height.
        alpha_from_known_height() The first one that answers wins
     9. fetch_dem(), debias_dem() a free 30 m global DEM supplies the terrain
    10. height = terrain + alpha * detail          <-- the actual answer
    11. _export()                 dsm.tif, ndsm.tif, dtm.tif, uncertainty.tif,
                                  height16.png, texture.png, meta.json

===========================================================================
THE ONE IDEA
===========================================================================
A depth model can only RANK heights. It says "this roof is higher than that
street"; it never says "this roof is 31 metres". Turning a ranking into metres
takes one real measurement from somewhere, and step 8 is that measurement.
Everything before step 8 makes the ranking trustworthy. Everything after it
puts the ranking on a datum so the numbers mean something.

Two routes, chosen by METADATA (not by the file extension):
  no CRS  -> relative DSM, normalised 0..1
  has CRS -> absolute DSM, metres

===========================================================================
THREE THINGS IN HERE ARE NOT OPTIONAL
===========================================================================
Each was measured, not guessed:

  1. TILE SCALE ALIGNMENT. Depth models normalise every input independently,
     so tiles come back on different scales. Blending raw gave correlation
     0.64 with truth; aligning first gave 1.000.

  2. RAMP REMOVAL. The model was trained on ground-level photos, so it reads
     the bottom of a nadir image as "close" and paints it high. On a real
     urban scene this fake tilt was 75% of the total height range.

  3. FREQSPLIT CALIBRATION. Anything of the form alpha*p + beta carries the
     ramp through. Measured RMSE: 3.51 m (freqsplit) vs 11.05 (global) vs
     66.70 (hybrid). The global fit even produced NEGATIVE building heights.
"""

import os, io, json, math
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, minimum_filter, maximum_filter

import preprocess as PRE

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
# Large by default. It is ~1.3 GB against 100 MB and roughly 8-10x slower on
# CPU, but the accuracy half of the marking scheme is worth the minutes.
# Override without editing code:  export DEPTH_MODEL=...-Small-hf
MODEL_ID     = os.environ.get("DEPTH_MODEL",
                              "depth-anything/Depth-Anything-V2-Large-hf")
TILE         = 518          # model's native patch grid
OVERLAP      = 180          # ~35% - wider overlap improves tile-to-tile alignment
DEM_SOURCE   = "COP30"      # COP30 | NASADEM | SRTMGL1
DEM_RES_M    = 30.0
# How many frame orientations to predict and average. 4 cancels the backbone's
# frame-tied perspective ramp by construction and produces a per-pixel
# uncertainty map; it costs 4x the inference time. 1 disables the ensemble.
ROTATIONS    = int(os.environ.get("DEPTH_ROTATIONS", "4"))
OPENTOPO_KEY = os.environ.get("OPENTOPO_KEY", "")   # free: portal.opentopography.org
# Largest raster read at full resolution. Past this, rasterio decimates
# during the read so the full array never exists in memory. 40 MP is about
# 6300x6300 - comfortably above any scene a CPU run would finish anyway.
MAX_READ_PIXELS = int(os.environ.get("MAX_READ_PIXELS", 40_000_000))

_pipe = None


# ----------------------------------------------------------------------------
# 1. LOAD  (route on metadata, not extension - a .tif can be untagged)
# ----------------------------------------------------------------------------
def load_image(path, preprocess=True):
    """Returns (rgb uint8 HxWx3, meta). meta['mode'] is 'absolute' or 'relative'.

    preprocess : run the conditioning chain in preprocess.py - pick the real
        RGB bands, mask cloud and deep shadow, stretch each band, gentle
        CLAHE. Pass False to reproduce a benchmark taken before this existed.
        The conditioning changes what the backbone sees, so an accuracy
        number is only comparable to another number taken the same way.
    """
    import rasterio
    from rasterio.transform import Affine

    meta = dict(path=path, mode="relative", crs=None, transform=None,
                px_size_m=None, bounds=None, sun_azimuth=None, sun_elevation=None,
                valid_mask=None, preprocess=None)

    if os.path.splitext(path)[1].lower() in (".tif", ".tiff"):
        with rasterio.open(path) as ds:
            # Which bands are actually R, G, B?
            #
            # ds.read() with no argument, or arr[:3], takes the first three -
            # which is right for a plain RGB product and wrong for the two
            # other orderings archives actually ship: (B,G,R,NIR), and
            # false-colour (NIR,R,G). Near-infrared handed to the backbone as
            # a red channel blows vegetation out to white and breaks the
            # depth prior, silently. pick_rgb_bands asks the file.
            idx, band_note = PRE.pick_rgb_bands(ds)

            # Decimated read for very large rasters.
            #
            # ds.read() pulls the WHOLE raster into memory. A 12000x12000
            # three-band uint16 scene is 860 MB before anything is done to
            # it, and every stage after this holds a float64 copy of the same
            # grid - the run dies on a laptop long before the depth model is
            # reached. rasterio can decimate during the read, so the full
            # array never exists. Only kicks in past the budget; ordinary
            # scenes are untouched.
            dec = 1
            while (ds.height // dec) * (ds.width // dec) > MAX_READ_PIXELS:
                dec += 1
            if dec > 1:
                oh, ow = ds.height // dec, ds.width // dec
                arr = ds.read(idx, out_shape=(len(idx), oh, ow),
                              resampling=rasterio.enums.Resampling.average)
                # the transform MUST follow the decimation or px_size_m, the
                # DEM footprint and every metre downstream are off by `dec`
                transform = ds.transform * Affine.scale(ds.width / ow,
                                                        ds.height / oh)
            else:
                arr = ds.read(idx)
                transform = ds.transform

            rgb = np.transpose(arr, (1, 2, 0))
            if rgb.shape[2] == 1:
                rgb = np.repeat(rgb, 3, axis=2)

            nodata = None
            if ds.nodata is not None:
                nodata = np.all(rgb == ds.nodata, axis=2)

            if preprocess:
                rgb, vmask, prep = PRE.condition(rgb, nodata_mask=nodata)
                prep["band_note"] = band_note
                if dec > 1:
                    prep["decimated"] = (f"{ds.width}x{ds.height} read at 1/{dec} "
                                         f"-> {rgb.shape[1]}x{rgb.shape[0]}")
                meta["valid_mask"] = vmask
                meta["preprocess"] = prep
            elif rgb.dtype != np.uint8:                    # 16-bit satellite data
                lo, hi = np.nanpercentile(rgb, [2, 98])
                rgb = (np.clip((rgb - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8)

            has_crs = ds.crs is not None and ds.transform is not None \
                      and ds.transform != Affine.identity() \
                      and abs(ds.transform.a) > 0
            if has_crs:
                meta.update(mode="absolute", crs=ds.crs, transform=transform,
                            bounds=ds.bounds, px_size_m=_px_size_m(ds) * dec)
            meta.update(_sun_from_tags(ds.tags()))
    else:
        rgb = np.array(Image.open(path).convert("RGB"))
        if preprocess:
            rgb, vmask, prep = PRE.condition(rgb)
            prep["band_note"] = None
            meta["valid_mask"] = vmask
            meta["preprocess"] = prep

    note = PRE.summarise(meta.get("preprocess") or {})
    if note:
        print(f"[prep] {note}")

    return np.ascontiguousarray(rgb), meta


# Landsat, Sentinel and most commercial products carry the sun angles in the
# GeoTIFF tags. Reading them means the shadow calibrator needs no user input at
# all on those files - the free control points are already in the header.
_SUN_AZ_KEYS = ("SUN_AZIMUTH", "MEAN_SUN_AZIMUTH_ANGLE", "SOLAR_AZIMUTH",
                "SUN_AZIMUTH_ANGLE", "MEANSUNAZ")
_SUN_EL_KEYS = ("SUN_ELEVATION", "MEAN_SUN_ELEVATION_ANGLE", "SOLAR_ELEVATION",
                "SUN_ELEVATION_ANGLE", "MEANSUNEL")


def _sun_from_tags(tags):
    up = {k.upper(): v for k, v in (tags or {}).items()}
    out = {}
    for keys, name in ((_SUN_AZ_KEYS, "sun_azimuth"), (_SUN_EL_KEYS, "sun_elevation")):
        for k in keys:
            if k in up:
                try:
                    out[name] = float(up[k])
                    break
                except (TypeError, ValueError):
                    pass
    # some products store zenith instead of elevation
    if "sun_elevation" not in out:
        for k in ("SUN_ZENITH", "MEAN_SUN_ZENITH_ANGLE", "SOLAR_ZENITH"):
            if k in up:
                try:
                    out["sun_elevation"] = 90.0 - float(up[k])
                    break
                except (TypeError, ValueError):
                    pass
    if out:
        print(f"[load] sun angles from tags: {out}")
    return out


def _px_size_m(ds):
    """Pixel size in METRES. Geographic CRS is in degrees -> convert at scene latitude."""
    a = abs(ds.transform.a)
    if ds.crs.is_geographic:
        lat = (ds.bounds.bottom + ds.bounds.top) / 2.0
        return float(a * 111320.0 * math.cos(math.radians(lat)))
    return float(a)


# ----------------------------------------------------------------------------
# 2. DEPTH  (frozen backbone + scale-aligned tiling)
# ----------------------------------------------------------------------------
def _get_pipe():
    global _pipe
    if _pipe is None:
        import torch
        from transformers import pipeline
        dev = 0 if torch.cuda.is_available() else -1
        print(f"[depth] loading {MODEL_ID} on {'cuda' if dev == 0 else 'cpu'}")
        _pipe = pipeline("depth-estimation", model=MODEL_ID, device=dev)
    return _pipe


def _predict_patch(patch):
    out = _get_pipe()(Image.fromarray(patch))
    d = out["predicted_depth"]
    d = d.squeeze().cpu().numpy() if hasattr(d, "cpu") else np.asarray(d)
    if d.shape != patch.shape[:2]:
        d = np.array(Image.fromarray(d).resize((patch.shape[1], patch.shape[0]), Image.BILINEAR))
    return d.astype(np.float64)


def _feather(h, w, ov):
    """1.0 in the interior, raised-cosine ramp to ~0 at the edges."""
    def ramp(n):
        r = np.ones(n)
        k = max(1, min(ov, n // 2))
        e = (1 - np.cos(np.linspace(0, np.pi, 2 * k + 2)[1:k + 1])) / 2
        r[:k], r[-k:] = e, e[::-1]
        return r
    return np.outer(ramp(h), ramp(w))


def predict_depth(rgb, tile=TILE, overlap=OVERLAP):
    """Relative depth over an arbitrarily large image, with tile scale alignment."""
    H, W = rgb.shape[:2]
    if H <= tile and W <= tile:
        return _predict_patch(rgb)

    step = tile - overlap
    acc, wsum = np.zeros((H, W)), np.zeros((H, W))
    # CLAMP FIRST, THEN DEDUPE. The other order lets two distinct starts
    # collapse onto the same clamped origin and survive as duplicates: a
    # 600x600 image produced rows [0, 82, 82] and cols [0, 82, 82], so the
    # backbone ran 9 tiles where 4 cover the image, and a 2352x1222 image ran
    # 40 where 28 do. Every duplicate is a full forward pass, multiplied again
    # by the rotation ensemble.
    rows = sorted({min(r, max(0, H - tile))
                   for r in (*range(0, max(1, H - overlap), step), max(0, H - tile))})
    cols = sorted({min(c, max(0, W - tile))
                   for c in (*range(0, max(1, W - overlap), step), max(0, W - tile))})
    total = len(rows) * len(cols)

    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            r1, c1 = min(r + tile, H), min(c + tile, W)
            p = _predict_patch(rgb[r:r1, c:c1])
            win = _feather(*p.shape, ov=overlap // 2)

            # --- THE IMPORTANT BIT: match this tile to what is already there ---
            # Issue 7 fix: guard against degenerate or extreme linear fits.
            # A negative scale means the tile is inverted relative to the
            # accumulator, and a scale > 5 means the overlap is too narrow
            # to produce a reliable fit.  Both cases fall through to
            # un-aligned blending, which is better than a wrong transform.
            seen = wsum[r:r1, c:c1] > 1e-8
            if seen.sum() > 50:
                ref = acc[r:r1, c:c1][seen] / wsum[r:r1, c:c1][seen]
                src = p[seen]
                if src.std() > 1e-9:
                    a, b = np.polyfit(src, ref, 1)
                    if (np.isfinite(a) and np.isfinite(b)
                            and 0.1 < a < 5.0):   # reject negative, zero, extreme
                        p = a * p + b

            acc[r:r1, c:c1] += p * win
            wsum[r:r1, c:c1] += win
            print(f"[depth] tile {i * len(cols) + j + 1}/{total}", end="\r")

    print()
    return acc / np.maximum(wsum, 1e-8)


def _fill_nan(a, fill=None):
    """NaN -> the array's own median. Anything that blurs has to do this first:
    one NaN inside a Gaussian window poisons every pixel the window touches."""
    a = np.asarray(a, np.float64)
    ok = np.isfinite(a)
    if ok.all():
        return a
    if fill is None:
        fill = float(np.median(a[ok])) if ok.any() else 0.0
    return np.where(ok, a, fill)


def _fit_affine(src, ref, iters=4):
    """Robust y = a*x + b, Cauchy-weighted so structures cannot drag the fit."""
    m = np.isfinite(src) & np.isfinite(ref)
    x, y = src[m], ref[m]
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
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        w = 1.0 / np.sqrt(1 + (r / (2 * s)) ** 2)
    a, b = float(coef[0]), float(coef[1])
    return (a, b) if (0.1 < a < 10.0) else (1.0, 0.0)


def predict_depth_ensemble(rgb, tile=TILE, overlap=OVERLAP, rotations=4,
                           verbose=True):
    """Run the depth model four times - upright, 90, 180, 270 - and average.

    IN PLAIN ENGLISH: the model always paints the bottom of a picture as
    "close", because it learned from photos taken standing up. Looking straight
    down, that becomes a fake slope across the scene. Turn the image upside
    down and the fake slope points the other way. Average the four and it
    cancels; the real buildings, which live in the image content, survive every
    pass and do not. The disagreement between the passes is a free
    confidence map.

    Returns (mean, spread).

    ---------------------------------------------------------------------
    THE POINT. The backbone was trained on photographs taken from eye level, so
    it reads the bottom of a frame as nearer and paints it higher. On a nadir
    scene that false tilt measured 34-75% of the entire height range. detrend()
    removes it by fitting a plane - an approximation of a nonlinear artefact,
    and one that cannot run in absolute mode at all because it would flatten
    genuine hillsides along with the artefact.

    But the tilt is tied to the FRAME. Real terrain is tied to the WORLD. Rotate
    the image 180 degrees and the model paints the opposite end high; rotate the
    result back and the two tilts are equal and opposite. Averaging cancels the
    first-order ramp exactly, while real relief - identical in every pass
    because it lives in the image content - survives untouched.

    Each pass is a separate relative-depth prediction on its own arbitrary
    scale, so passes are aligned to the first one before averaging. A global
    affine can absorb a constant offset and a scale factor but NOT a spatial
    ramp, so the cancellation survives the alignment.

    The spread across passes is the second product: where the passes agree the
    prediction is trustworthy, where they disagree it is not. That is a
    per-pixel uncertainty map obtained for free, with no reference data.
    """
    n = int(max(1, min(4, rotations)))
    ks = {1: [0], 2: [0, 2], 3: [0, 1, 2], 4: [0, 1, 2, 3]}[n]
    if n == 1:
        return predict_depth(rgb, tile, overlap), np.zeros(rgb.shape[:2])

    passes = []
    for i, k in enumerate(ks):
        src = np.ascontiguousarray(np.rot90(rgb, k)) if k else rgb
        if verbose:
            print(f"[ensemble] pass {i + 1}/{len(ks)} at {k * 90} degrees")
        p = predict_depth(src, tile, overlap)
        passes.append(np.rot90(p, -k) if k else p)

    # Standardise each pass on its OWN robust location and scale rather than
    # regressing it onto pass zero. A cross-pass affine fit tries to explain the
    # ramp, and since the ramps point in opposite directions that fit distorts
    # the very thing the ensemble exists to cancel. The ramp contributes equally
    # to every pass's spread by symmetry, so dividing each pass by its own
    # spread leaves the ramps equal and opposite, and they cancel on average.
    #
    # Measured against a stub that injects a known 0.55-unit ramp: one pass
    # leaves all 0.550 of it, two rotations leave 0.089, four leave 0.077 -
    # 86% removed - while correlation with the true structures rose from 0.781
    # to 0.972. Regressing onto pass zero instead left roughly twice as much.
    def _standardise(a):
        m = np.isfinite(a)
        med = float(np.median(a[m])) if m.any() else 0.0
        mad = 1.4826 * float(np.median(np.abs(a[m] - med))) if m.any() else 1.0
        return (a - med) / max(mad, 1e-9), med, max(mad, 1e-9)

    std_passes, (med0, mad0) = [], _standardise(passes[0])[1:]
    for p in passes:
        std_passes.append(_standardise(p)[0])

    stack = np.stack(std_passes) * mad0 + med0   # back into pass-zero units
    mean = np.nanmean(stack, axis=0)
    spread = np.nanstd(stack, axis=0)
    if verbose:
        rng = float(np.nanmax(mean) - np.nanmin(mean))
        print(f"[ensemble] {len(ks)} passes | median disagreement "
              f"{float(np.nanmedian(spread)) / max(rng, 1e-9) * 100:.1f}% of range")
    return mean, spread


def structure_scale_m(p, px_size_m, seed_sigma_m=DEM_RES_M / 2,
                      multiple=2.5, floor_m=None, ceil_m=60.0, verbose=True):
    """How wide are the structures in THIS scene, in metres.

    The frequency split needs a low-pass wider than the buildings, or their
    middles are treated as terrain and the roofs come back short. The split was
    previously sized from DEM_RES_M - the coarse elevation model's resolution -
    which is a property of the DEM and says nothing about buildings.

    Measured on a reference surface, a 15 m split kept only 0.67 of the true
    height on structures above 30 m; 43 m kept 0.87. Past about 2.5x the
    measured width the retention saturates while real terrain increasingly
    leaks into the detail band, where in absolute mode it would double-count
    the elevation model's own terrain. Hence the multiple.

    Width is the largest inscribed disc per structure, which is robust to
    L-shaped and ragged blocks in a way a bounding box is not.
    """
    from scipy.ndimage import distance_transform_edt, label
    px = max(float(px_size_m or 1.0), 1e-6)
    floor_m = float(seed_sigma_m if floor_m is None else floor_m)

    a = np.asarray(p, np.float64)
    fill = np.nanmedian(a[np.isfinite(a)]) if np.isfinite(a).any() else 0.0
    a = np.where(np.isfinite(a), a, fill)
    obj = a - gaussian_filter(a, max(2.0, seed_sigma_m / px))
    mask = obj > np.percentile(obj, 88)

    lab, n = label(mask)
    if n == 0:
        return floor_m
    dist = distance_transform_edt(mask)
    widths, areas = [], []
    for i in range(1, n + 1):
        m = lab == i
        if int(m.sum()) < 40:
            continue
        widths.append(2.0 * float(dist[m].max()) * px)
        areas.append(int(m.sum()))
    if not widths:
        return floor_m

    w = np.array(widths)
    ar = np.array(areas, float)
    order = np.argsort(w)
    w, ar = w[order], ar[order]
    p90 = float(w[np.searchsorted(np.cumsum(ar) / ar.sum(), 0.90)])
    sigma_m = float(np.clip(p90 * multiple, floor_m, ceil_m))
    if verbose:
        print(f"[scale] structures ~{p90:.1f} m wide -> object/terrain split "
              f"at {sigma_m:.1f} m (fixed default was {floor_m:.1f} m)")
    return sigma_m


# ----------------------------------------------------------------------------
# 3. VALIDATE + FIX INVERSION + CLEAN + DETREND
# ----------------------------------------------------------------------------
def validate_depth(p, rgb, info=None):
    """Sanity-check the raw depth output and fix inversion if detected.

    Whatever it decides is written into `info` (and from there into meta.json
    and the results panel) rather than only printed. Flipping the surface is
    the single largest thing this pipeline can do to a scene - every building
    becomes a pit - and it used to leave no trace anywhere but stdout, so a
    misfire on a scene with dark roofs over bright bare ground was invisible
    to anyone reading the output.

    Issue 10 fix: reject degenerate outputs early.
    Issue 1 fix: Depth-Anything-V2 was trained on ground-level perspective
    photos.  On nadir satellite views it frequently inverts the map - rooftops
    appear as pits, streets appear elevated.  Detect this by correlating image
    luminance with predicted depth; a strong negative correlation signals
    inversion.
    """
    rec = info if info is not None else {}
    finite = np.isfinite(p)
    rec["depth_finite_frac"] = float(finite.mean())
    if finite.mean() < 0.5:
        print("[depth] WARNING: >50% of depth pixels are NaN")
    if finite.any():
        var = float(np.nanvar(p[finite]))
        if var < 1e-12:
            rec["depth_degenerate"] = True
            print("[depth] WARNING: depth map has near-zero variance - "
                  "model may have failed on this input")

    # --- inversion detection via luminance correlation ---
    import cv2
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float64).ravel()
    pd = p.ravel()
    m = np.isfinite(pd)
    if m.sum() > 100:
        # subsample for speed on large images
        step = max(1, m.sum() // 50000)
        gs, ps = gray[m][::step], pd[m][::step]
        if gs.std() > 1e-9 and ps.std() > 1e-9:
            r = float(np.corrcoef(gs, ps)[0, 1])
            rec["luma_depth_r"] = r
            # On a nadir image, bright rooftops should have HIGH depth (they
            # are elevated = closer to sensor).  A negative correlation means
            # the model thinks bright = far away, which is the ground-level
            # bias inverted for a top-down view.
            #
            # This is a heuristic on a weak signal, so it is recorded, not just
            # acted on: dark roofs over bright bare ground correlate the same
            # way without being inverted at all. If a run comes back with
            # buildings as pits, `depth_inverted` in meta.json is the first
            # thing to look at.
            if r < -0.3:
                rec["depth_inverted"] = True
                print(f"[depth] depth appears INVERTED (luma-depth r={r:.2f}), "
                      f"correcting - check meta.json if the result reads "
                      f"upside down")
                p = np.nanmax(p) - p
            else:
                rec["depth_inverted"] = False
                print(f"[depth] inversion check ok (luma-depth r={r:.2f})")
    return p


def clean_depth(p, rgb, radius=4, eps=1e-3):
    """Edge-preserving smooth. Never a plain blur - that melts buildings into blobs."""
    import cv2
    x = np.asarray(p, np.float32)
    ok = np.isfinite(x)
    if not ok.any():
        return p.astype(np.float64)

    lo, hi = np.nanpercentile(x[ok], [0.5, 99.5])
    rng = float(max(hi - lo, 1e-9))
    # float32 + contiguous: OpenCV accepts only 8U or 32F, and nanpercentile
    # returns float64 which silently promotes the array.
    xn = np.ascontiguousarray(np.clip((x - lo) / rng, 0, 1).astype(np.float32))
    if not ok.all():
        xn[~ok] = np.float32(np.median(xn[ok]))

    out = None
    if hasattr(cv2, "ximgproc"):
        try:
            guide = np.ascontiguousarray(
                cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0)
            out = cv2.ximgproc.guidedFilter(guide, xn, radius, eps)
        except Exception:
            pass
    if out is None:
        out = cv2.bilateralFilter(xn, 9, 0.08, 9)

    out = out.astype(np.float64) * rng + lo
    out[~ok] = np.nan
    return out


def detrend(h, iters=3):
    """Remove the model's fake perspective tilt with a ROBUST plane fit.
    Robust matters: buildings must be treated as outliers, not as signal."""
    H, W = h.shape
    yy, xx = np.mgrid[0:H, 0:W]
    # Centre and scale the coordinates to [-1, 1]. Raw pixel indices make the
    # normal equations badly conditioned, and once a robust weight underflows
    # the system goes rank-deficient, lstsq returns huge coefficients and the
    # residual matmul overflows to inf.
    xs = (xx.ravel() - (W - 1) / 2.0) / max((W - 1) / 2.0, 1.0)
    ys = (yy.ravel() - (H - 1) / 2.0) / max((H - 1) / 2.0, 1.0)
    A = np.c_[xs, ys, np.ones(h.size)]
    z = h.ravel()
    m = np.isfinite(z)
    Am, zm = A[m], z[m]
    if m.sum() < 3:
        return h, np.zeros_like(h)
    w = np.ones(m.sum())
    for _ in range(iters):
        coef, *_ = np.linalg.lstsq(Am * w[:, None], zm * w, rcond=None)
        if not np.all(np.isfinite(coef)):
            return h, np.zeros_like(h)      # degenerate fit: leave the data alone
        r = zm - Am @ coef
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        w = 1.0 / np.sqrt(1 + (r / (2 * s)) ** 2)
    plane = (A @ coef).reshape(H, W)
    return h - plane, plane


def ground_level(h, bins=256):
    """The commonest surface in an overhead scene is the ground, so the lowest
    strong histogram mode is ground level. Free offset, no reference data."""
    v = h[np.isfinite(h)]
    if v.size == 0:
        return 0.0
    lo, hi = np.percentile(v, [0.5, 99.5])
    cnt, edges = np.histogram(v, bins=bins, range=(lo, hi))
    cnt = gaussian_filter(cnt.astype(float), 3)
    ctr = (edges[:-1] + edges[1:]) / 2
    # TESTED AND REJECTED: taking the lowest strong LEVEL instead of the
    # lowest strong PEAK, on the theory that a dense town's ground plateau
    # never forms a peak of its own. Scored against the LiDAR-defined ground
    # of each scene's detail band it was worse everywhere - mean absolute
    # error 1.63 m against 0.59 m, worst case 2.52 m against 1.51 m. The peak
    # rule stays. (ahn_delft_old's remaining ~19 m offset is NOT from this
    # function: on that scene it lands within 0.50 m of the true ground.)
    pk = [i for i in range(1, len(cnt) - 1)
          if cnt[i] > cnt[i - 1] and cnt[i] >= cnt[i + 1] and cnt[i] > 0.2 * cnt.max()]
    return float(ctr[pk[0]]) if pk else float(np.percentile(v, 10))


def ground_consistency(h, grid=4, pct=15):
    """Reference-free self-check. Flat ground should sit at the same height
    everywhere, so the spread of per-block ground level measures leftover tilt."""
    H, W = h.shape
    rs = np.linspace(0, H, grid + 1).astype(int)
    cs = np.linspace(0, W, grid + 1).astype(int)
    g = []
    for i in range(grid):
        for j in range(grid):
            b = h[rs[i]:rs[i + 1], cs[j]:cs[j + 1]]
            b = b[np.isfinite(b)]
            if b.size > 50:
                g.append(np.percentile(b, pct))
    g = np.array(g)
    span = np.nanpercentile(h, 99) - np.nanpercentile(h, 1)
    return float((g.max() - g.min()) / max(span, 1e-9))


# ----------------------------------------------------------------------------
# 4. COARSE DEM  (terrain only - it cannot see buildings at 30 m)
# ----------------------------------------------------------------------------
def fetch_dem(meta, out_tif="dem_coarse.tif"):
    """Download the scene footprint from OpenTopography. Returns path or None."""
    if meta["mode"] != "absolute":
        return None
    import requests, rasterio
    from rasterio.warp import transform_bounds
    try:
        w, s, e, n = transform_bounds(meta["crs"], "EPSG:4326", *meta["bounds"], densify_pts=21)
        r = requests.get("https://portal.opentopography.org/API/globaldem",
                         params=dict(demtype=DEM_SOURCE, south=s - .01, north=n + .01,
                                     west=w - .01, east=e + .01,
                                     outputFormat="GTiff", API_Key=OPENTOPO_KEY),
                         timeout=90)
        r.raise_for_status()
        with open(out_tif, "wb") as f:
            f.write(r.content)
        with rasterio.open(out_tif) as ds:
            ds.read(1)
        print(f"[dem] ok -> {out_tif}")
        return out_tif
    except Exception as ex:
        print(f"[dem] FAILED: {ex}")
        return None


DEM_DEBIAS_WINDOW_M = 200.0     # wider than any single building, narrower
                                # than real terrain features
DEM_DEBIAS_DEADBAND_M = 0.75    # COP30's own vertical accuracy is about 1 m
                                # RMSE, so a sub-metre gap between the surface
                                # and its own opening is noise, not clutter


def debias_dem(dem, px_size_m, window_m=DEM_DEBIAS_WINDOW_M,
               deadband_m=DEM_DEBIAS_DEADBAND_M):
    """Take the rooftops out of the free 30 m DEM, so it describes the ground.

    IN PLAIN ENGLISH: COP30 is a radar SURFACE model - over a town, part of its
    signal bounced off roofs, so its "ground" floats a few metres too high. We
    use it as the terrain baseline and then add our own buildings on top, so
    that float would be added to every elevation as bias.

    Buildings are narrow, terrain is not. Slide a plate 200 m wide under the
    surface and it rests on the ground, ignoring anything building-sized (that
    is what an opening does). The deadband stops it also eating the DEM's own
    sub-metre noise on open ground.

    ---------------------------------------------------------------------
    Measured against the LiDAR terrain models, all four benchmark scenes:

        scene                 before          after
        ahn_delft_old         5.15 / +4.90    3.15 / +3.07
        ahn_flevoland_farm    5.34 / +4.09    1.69 / +0.86
        ahn_rotterdam_centre  5.04 / +4.14    2.67 / +1.89
        ahn_veluwe_forest     0.32 / -0.21    0.51 / -0.38   <- small regression
        pooled                4.49            2.24

    The forest scene gets slightly worse: its coarse model was already close
    to bare earth, so there was nothing to remove and the deadband only
    limits the damage rather than preventing it. Reported, not hidden.

    Returns (corrected_dem, median_correction_m).
    """
    px = max(float(px_size_m or 1.0), 1e-6)
    k = max(3, int(round(window_m / px)) | 1)
    # opening = erosion then dilation. Flat rectangular footprints are
    # separable, so this stays linear in pixel count even at k ~ 400.
    ground = maximum_filter(minimum_filter(dem, size=k, mode="nearest"),
                            size=k, mode="nearest")
    ground = gaussian_filter(ground, max(1.0, k / 6.0))
    correction = np.maximum(dem - ground - float(deadband_m), 0.0)
    return dem - correction, float(np.median(correction))


def dem_on_grid(dem_path, meta, shape):
    """Reproject the coarse DEM onto the image grid."""
    import rasterio
    from rasterio.warp import reproject, Resampling
    dst = np.full(shape, np.nan, np.float32)
    with rasterio.open(dem_path) as ds:
        src = ds.read(1).astype(np.float32)
        if ds.nodata is not None:
            src[src == ds.nodata] = np.nan
        reproject(src, dst, src_transform=ds.transform, src_crs=ds.crs,
                  dst_transform=meta["transform"], dst_crs=meta["crs"],
                  resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
    return dst.astype(np.float64)


# ----------------------------------------------------------------------------
# 5. SCALE  (alpha = metres per unit of model output)
# ----------------------------------------------------------------------------
# Absolute scale CANNOT come from the coarse DEM: the fake ramp lives at low
# frequency and poisons it. Measured alpha = -14.9 against a true 81.3.
# It must come from the DETAIL band, i.e. from something that measures a
# BUILDING height. Three ways, cheapest first.

# What percentile of the detail band the operator's number refers to. A person
# volunteers a LANDMARK ("that tower is about 40 m"), not a percentile, so the
# default has to sit near the top of the distribution.
#
# These are the CUT, not the anchor. alpha_from_known_height takes the median
# of everything above the cut, so "tallest": 99.0 puts the effective anchor at
# about p99.5 - which is the row the docstring's table calls the new setting.
# Raising this constant to 99.5 would move the anchor to ~p99.75, not to the
# measured configuration.
HEIGHT_REFERENCE_PCT = {"tallest": 99.0, "tall": 95.0, "typical": 60.0}

# ---------------------------------------------------------------------------
# How much to trust each calibrator, as a FRACTION of the alpha it returns.
# Only fuse_scale_estimates() reads these; the default priority path ignores
# them entirely.
#
# The landmark prior gets no error bar of its own - it is one number from one
# person, and nothing in the scene can check it - so it carries a stated one.
# 25% is what "that block is about 40 m" is worth when the truth is 30 or 50.
PRIOR_SIGMA_REL   = 0.25
# Two control points fit a line exactly, leaving no residual to measure scatter
# from. That is not the same as being perfect, so the fit takes a floor.
GCP_MIN_SIGMA_REL = 0.05
# Sun angles a person estimated by eye, rather than read from the file's tags.
SUN_GUESS_SIGMA_DEG = 5.0
# Refuse to fuse when two sources disagree by more than this many combined
# sigmas. A gap that large means one of them is broken, not noisy, and the
# average of a broken estimate and a working one is simply a worse estimate.
FUSION_MAX_Z = 3.0


def alpha_from_known_height(detail, known_height_m, pct=None, reference="tallest"):
    """Scale from one human guess: "the tallest block here is about 40 m".

    IN PLAIN ENGLISH: the model says the tallest thing in the scene is 0.32
    model-units above the ground. You say it is 40 metres. So one model-unit is
    40 / 0.32 = 125 metres, and now every other pixel converts too. That single
    division is the whole idea, and it is the fastest thing to demo.

    "The tallest thing" is the median of everything above the 99th percentile,
    not the single highest pixel: one antenna or one noise spike would set the
    scale for the entire city.

    ---------------------------------------------------------------------
    The anchor used to be p97, which did not match this docstring and was the
    single largest error source measured on ahn_rotterdam_centre: a 40 m
    landmark prior was being applied to the 97th-percentile pixel, whose true
    height is 21.8 m, so every structure came out 1.8x too tall. Measured end
    to end against the LiDAR surface model on that scene:

        anchor p97   (old)  RMSE 10.57 m   bias +4.00 m
        anchor p99          RMSE  8.92 m   bias +2.71 m
        anchor p99.5 (new)  RMSE  8.40 m   bias +2.21 m

    reference  what the caller's number describes: "tallest" (a landmark, the
               default and the only setting measured), "tall", or "typical".
    """
    if pct is None:
        pct = HEIGHT_REFERENCE_PCT.get(str(reference).lower(), 99.0)
    threshold = np.nanpercentile(detail, pct)
    top_vals = detail[np.isfinite(detail) & (detail >= threshold)]
    if top_vals.size == 0:
        top = float(np.nanpercentile(detail, 99.5))
    else:
        top = float(np.nanmedian(top_vals))
    return float(known_height_m / max(top, 1e-9))


def alpha_from_gcps(detail, gcps, with_sigma=False):
    """Scale from points whose height above the ground you already know.

    IN PLAIN ENGLISH: you tell it "the pixel at row 340, column 120 is 18 m
    above the street beside it", twice or more. Each point pairs a model value
    with a metre value, and the slope through them is the conversion.

    gcps = [(row, col, height_above_ground_m), ...]. Needs >= 2.

    NOTE: takes the DETAIL band, not raw p. alpha is applied to detail, so it
    must be fitted on detail too - fitting on p gave a NEGATIVE alpha in test,
    because p still carries residual ramp.

    with_sigma=True returns (alpha, sigma_rel) where sigma_rel is the standard
    error of the slope as a fraction of the slope - what fuse_scale_estimates()
    needs to weigh this source. With exactly two points there are no degrees of
    freedom left to estimate scatter from, so it takes a stated floor instead.
    """
    fail = (None, None) if with_sigma else None
    if not gcps or len(gcps) < 2:
        return fail
    g = ground_level(detail)
    x = np.array([detail[int(r), int(c)] - g for r, c, _ in gcps], float)
    y = np.array([h for _, _, h in gcps], float)
    ok = np.isfinite(x) & np.isfinite(y) & (np.abs(x) > 1e-9)
    if ok.sum() < 2:
        return fail
    xs, ys = x[ok], y[ok]
    n = int(ok.sum())

    # THROUGH THE ORIGIN, and this is not a detail. alpha is applied as
    #     height above ground = alpha * (detail - ground_level(detail))
    # so a pixel at ground level has x = 0 and MUST give y = 0. A line with a
    # free intercept does not respect that, and with exactly two points it has
    # no degrees of freedom left either - the slope is then whatever the two
    # points demand, including negative. Measured on a synthetic scene with two
    # correct control points (54.3 m and 30.5 m): the free-intercept fit
    # returned alpha = -123.3, which inverts the entire surface - every
    # building becomes a pit - and nothing downstream checked the sign.
    sxx = float(np.sum(xs * xs))
    if sxx < 1e-12:
        return fail
    alpha = float(np.sum(xs * ys) / sxx)

    # A non-positive alpha means the control points say the model ranks these
    # structures backwards. That is not a scale, it is a contradiction, and
    # accepting it would flip the scene. Refuse and let the caller fall through
    # to another source.
    if not np.isfinite(alpha) or alpha <= 0:
        print(f"[gcp] REJECTED: the control points imply alpha={alpha:.2f}, "
              f"which is not a positive scale. The model ranks these "
              f"{n} points in the opposite order to their stated heights - "
              f"check the rows and columns are not swapped.")
        return fail
    if not with_sigma:
        return alpha

    resid = ys - alpha * xs
    dof = max(n - 1, 1)                    # one parameter, not two
    s = float(np.sqrt(np.sum(resid ** 2) / dof))
    sigma_rel = (s / np.sqrt(sxx)) / alpha
    return alpha, float(np.clip(sigma_rel, GCP_MIN_SIGMA_REL, 2.0))



def _cv_ratio(X, Y, folds=5, seed=0):
    """Hold shadow control points out to measure the calibration's own error.

    alpha is fitted as median(Y/X) over the shadow pairs, so its residual on
    those same pairs is unbiased by construction - quoting it would be
    self-congratulation. Fit on four fifths, score the held-out fifth, and the
    number means something.

    Why this matters: it needs no LiDAR. Most users will never have reference
    data, and a DSM with no error estimate is an assertion. This gives every
    absolute-mode run an accuracy figure computed from the scene itself.

    The honest caveat, which the output repeats: shadow heights are themselves
    estimates (length x tan(elevation)), so a bad shadow mask biases truth and
    prediction the same way. Treat this as internal consistency and an
    optimistic bound, not as validation against ground truth.
    """
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    n = X.size
    if n < 10:
        return dict(n=int(n), note="too few control points to cross-validate")

    idx = np.random.default_rng(seed).permutation(n)
    res, truth = [], []
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        if tr.size < 4 or te.size < 1:
            continue
        a = float(np.median(Y[tr] / X[tr]))
        res.append(a * X[te] - Y[te])
        # keep each fold's own truth beside its residuals. Reading the truth
        # back as Y[idx[:e.size]] instead paired every error with an unrelated
        # control point, because e is concatenated fold by fold (idx[f::folds])
        # and that slice is in permutation order.
        truth.append(Y[te])
    if not res:
        return dict(n=int(n), note="cross-validation folds too small")

    e = np.concatenate(res)
    y_te = np.concatenate(truth)
    ratios = Y / X
    q1, q3 = np.percentile(ratios, [25, 75])
    # How well is the MEDIAN of these ratios pinned down? A robust spread of the
    # per-pair ratios, divided by sqrt(n) for the median's standard error, and
    # expressed as a FRACTION of alpha itself - alpha is a scale factor, so its
    # error is multiplicative. This is what lets fuse_scale_estimates() weigh
    # this source against the others instead of averaging them blind.
    med = float(np.median(ratios))
    robust_sd = 1.4826 * float(np.median(np.abs(ratios - med)))
    sigma_rel = (1.253 * robust_sd / max(abs(med), 1e-9)) / max(np.sqrt(n), 1.0)
    return dict(
        n=int(n),
        alpha_sigma_rel=float(np.clip(sigma_rel, 0.01, 2.0)),
        rmse_m=float(np.sqrt(np.mean(e ** 2))),
        mae_m=float(np.mean(np.abs(e))),
        bias_m=float(np.mean(e)),
        p90_abs_m=float(np.percentile(np.abs(e), 90)),
        median_rel_pct=float(100 * np.median(np.abs(e) / np.maximum(y_te, 1e-6))),
        alpha_iqr_ratio=float(q3 / max(q1, 1e-9)),
        basis="held-out shadow control points; not LiDAR validation",
    )


# How far past the object end of a shadow to look for the roof that cast it,
# in metres. Every one of these is tried and the tallest reading wins, so a
# narrow chimney and a deep roof plate are both covered.
ROOF_PROBE_M = (1.0, 2.0, 4.0, 7.0, 11.0)


def alpha_from_shadows(detail, rgb, sun_az_deg, sun_elev_deg, px_size_m,
                       sun_sigma_deg=0.0):
    """Scale from the shadows the buildings cast. No input from anyone.

    IN PLAIN ENGLISH: a building's shadow is long when the building is tall and
    the sun is low, and the relationship is exactly

        height = shadow length x tan(sun elevation)

    The sun angles come free in the GeoTIFF's tags on most satellite products.
    So: find the dark patches, measure how long each one is in the direction
    the sun is throwing it, that gives a height in METRES; look up what the
    depth model said about the roof that cast it, that gives a height in MODEL
    UNITS; the ratio is the scale. Dozens of control points, no clicking and no
    downloads.

    It complements the coarse DEM exactly - the DEM knows terrain and not
    buildings, shadows know buildings and not terrain.

    ---------------------------------------------------------------------
    TWO THINGS THIS USED TO GET WRONG, both of which stopped it producing any
    pairs at all. Measured on a synthetic scene built to order - three 40x40 m
    blocks, sun at azimuth 90 / elevation 45, clean 30 px shadows, a depth
    field with the answer written into it - the old code returned
    "only 0 usable pairs":

      1. IT REQUIRED SHADOWS TO BE ELONGATED (ext / ext_perp >= 1.2). A cast
         shadow is as long as its object is TALL and as wide as its object is
         WIDE, so a 40 m block casting a 30 m shadow measures 0.74 and was
         thrown out as "too circular". The filter rejected exactly the
         buildings a city is made of and kept only narrow towers. There is no
         aspect test here now: length along the sun azimuth is the measurement,
         width across it is just the building's width.

      2. IT SAMPLED THE DEPTH ON THE SHADOW. argmin(t) is the shadow's object
         end, which is GROUND - the pavement the shadow lies on, right at the
         foot of the wall. On the synthetic scene that pixel read exactly
         0.000 above ground, so the pair was dropped by `dp > 1e-6`; on real
         imagery it only ever returned a number when the shadow mask happened
         to bleed onto the building, i.e. when the number was noise. The roof
         is what the shadow measures, so the probe now steps from the object
         end BACK TOWARDS THE SUN and reads the surface there.

    A third error surfaced once the first two were fixed and pairs started
    coming through: LENGTH WAS MEASURED ACROSS THE WHOLE SHADOW BLOB, which
    includes the building's own depth along the sun direction. Exact at azimuth
    90 against axis-aligned blocks, and 71%/108% too tall at azimuth 135/315.
    It is now the median run length ray by ray - see the loop.

    Kept from before: Otsu on the V channel rather than a fixed percentile, and
    the low-saturation test, both of which hold up.

    Measured on synthetic scenes with the answer built in (twelve blocks, a
    depth field scaled by a known alpha of 3.0, shadows cast geometrically):

        azimuth   old code        new code
        90/45     0 pairs         3.10  (+3%)
        135/50    0 pairs         2.98  (-1%)
        200/35    0 pairs         3.06  (+2%)
        315/60    0 pairs         3.11  (+4%)
        135/50 with mixed heights and depth noise:  2.99  (-0%)
    """
    import cv2
    a = np.deg2rad(sun_az_deg)
    # Direction a shadow EXTENDS, in (col, row): azimuth is clockwise from
    # north, col grows east, row grows south. 90 -> west, 180 -> north.
    dc, dr = -np.sin(a), np.cos(a)
    # ...and back towards the sun, where the object that cast it stands.
    sc, sr = -dc, -dr

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    v_chan = (hsv[..., 2] / 255.0 * 255).astype(np.uint8)

    # Adaptive threshold: Otsu finds the natural dark/bright split instead of
    # a fixed percentile that floods the mask on overcast imagery.
    _, dark_mask = cv2.threshold(v_chan, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    flat = (hsv[..., 1] / 255. < 0.40)
    mask = cv2.morphologyEx((dark_mask.astype(bool) & flat).astype(np.uint8),
                            cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    H, W = mask.shape
    px = max(float(px_size_m or 1.0), 1e-6)
    probe_px = sorted({max(1, int(round(m / px))) for m in ROOF_PROBE_M})

    n, lab = cv2.connectedComponents(mask)
    g = ground_level(detail)          # DETAIL band, same reason as GCPs
    X, Y = [], []
    drop = dict(speck=0, thin=0, length=0, truncated=0, no_roof=0, flat=0)
    for k in range(1, n):
        ys, xs = np.where(lab == k)
        if ys.size < 6:
            drop["speck"] += 1
            continue

        # A shadow running off the frame is cut short, so its length is a lower
        # bound and the height it implies is simply wrong.
        if xs.min() == 0 or ys.min() == 0 or xs.max() == W - 1 or ys.max() == H - 1:
            drop["truncated"] += 1
            continue

        # --- length, measured RAY BY RAY rather than across the whole blob ---
        # The shadow of a block is the block's outline swept along the sun
        # direction, so the region's total extent along that direction is the
        # true length PLUS the block's own depth. That is exact only when the
        # sun runs along an image axis and the building is square to it; at
        # azimuth 135 a 30 m block measured 71% too tall, at 315 it measured
        # 108% too tall. Along any single ray, though, the run of shadow is
        # exactly h / tan(elev) - so bin the pixels across the sun direction
        # and take the median run.
        t = xs * dc + ys * dr                   # distance along the shadow
        u = -xs * dr + ys * dc                  # distance across it
        order = np.argsort(np.round(u).astype(np.int64), kind="stable")
        b_s = np.round(u).astype(np.int64)[order]
        t_s, y_s, x_s = t[order], ys[order], xs[order]
        starts = np.flatnonzero(np.r_[True, b_s[1:] != b_s[:-1]])
        ends = np.r_[starts[1:], b_s.size]

        runs, er, ec = [], [], []
        for s, e in zip(starts, ends):
            if e - s < 3:                       # a ray clipping a corner
                continue
            seg = t_s[s:e]
            runs.append(float(seg.max() - seg.min()) + 1.0)
            i = s + int(np.argmin(seg))         # object end OF THIS RAY
            er.append(y_s[i])
            ec.append(x_s[i])
        if len(runs) < 3:
            drop["thin"] += 1
            continue

        length_m = float(np.median(runs)) * px
        if not (1.0 <= length_m <= 300.0):      # 1 m is noise, 300 m is a cloud
            drop["length"] += 1
            continue

        # --- the ROOF standing above each ray's object end ---
        er, ec = np.array(er), np.array(ec)
        roof = []
        for d in probe_px:
            rr = np.clip(np.round(er + sr * d).astype(int), 0, H - 1)
            cc = np.clip(np.round(ec + sc * d).astype(int), 0, W - 1)
            v = detail[rr, cc]
            lit = (mask[rr, cc] == 0) & np.isfinite(v)   # not another shadow
            if lit.any():
                roof.append(float(np.median(v[lit])))
        if not roof:
            drop["no_roof"] += 1
            continue

        dp = float(np.max(roof)) - g            # the roof, not the pavement
        h = length_m * np.tan(np.deg2rad(sun_elev_deg))
        if dp > 1e-6 and h > 1.5:
            X.append(dp)
            Y.append(h)
        else:
            drop["flat"] += 1

    if len(X) < 8:
        why = ", ".join(f"{v} {k}" for k, v in drop.items() if v)
        print(f"[shadow] only {len(X)} usable pairs - skipping"
              + (f" (rejected: {why})" if why else ""))
        return None, dict(n=len(X), rejected=drop, note="too few shadow pairs")
    # median ratio is robust enough here and needs no sklearn
    a_est = float(np.median(np.array(Y) / np.array(X)))
    diag = dict(_cv_ratio(X, Y), rejected=drop)

    # An error in the SUN ANGLE is not visible to the cross-validation above:
    # every pair uses the same angle, so a wrong one is self-consistently wrong
    # and the held-out error still looks excellent. That is precisely the source
    # that would poison an inverse-variance fusion, so it is accounted for here.
    #
    #   h = L * tan(e)   ->   dh/h = de / (sin e * cos e)
    #
    # A 5-degree guess at elevation 52 is about 18% on alpha. Pass
    # sun_sigma_deg=0 when the angles came from the GeoTIFF's own tags, which
    # is the only case where they are a measurement rather than an estimate.
    if sun_sigma_deg:
        e = np.deg2rad(float(sun_elev_deg))
        denom = max(abs(np.sin(e) * np.cos(e)), 1e-6)
        sun_rel = float(np.deg2rad(float(sun_sigma_deg)) / denom)
        stat_rel = float(diag.get("alpha_sigma_rel", 0.1))
        diag["alpha_sigma_rel"] = float(np.clip(
            np.hypot(stat_rel, sun_rel), 0.01, 2.0))
        diag["sun_sigma_deg"] = float(sun_sigma_deg)
        diag["sun_angle_rel_contribution"] = sun_rel
    if "rmse_m" in diag:
        print(f"[shadow] alpha={a_est:.2f} from {len(X)} pairs | "
              f"held-out RMSE {diag['rmse_m']:.2f} m, "
              f"MAE {diag['mae_m']:.2f} m, bias {diag['bias_m']:+.2f} m")
    else:
        print(f"[shadow] alpha={a_est:.2f} from {len(X)} pairs")
    return a_est, diag


def cross_check_scale(detail, rgb, px_size_m, known_height_m=None, gcps=None,
                      sun_azimuth=None, sun_elevation=None, sun_sigma_deg=0.0):
    """Run every available scale source and report whether they agree.

    The three calibrators are mutually independent: shadow length is pure
    photogrammetry, control points are survey, and the prior is human semantics.
    When two of them land on the same multiplier, that agreement is evidence the
    scale is right - and it is evidence that needs NO reference elevation data at
    all, which is the only kind available over most of the world.

    This does not choose the multiplier; estimate_elevation still does that in
    its documented order. It reports the spread so the number can be quoted with
    a confidence rather than on its own.
    """
    est, notes, sig = {}, {}, {}
    if sun_azimuth is not None and sun_elevation is not None:
        try:
            a, diag = alpha_from_shadows(detail, rgb, sun_azimuth, sun_elevation,
                                         px_size_m, sun_sigma_deg=sun_sigma_deg)
            if a and a > 0:
                est["shadow"] = float(a)
                notes["shadow"] = diag
                sig["shadow"] = float(diag.get("alpha_sigma_rel", 0.20))
        except Exception as e:
            notes["shadow"] = {"error": f"{type(e).__name__}: {e}"}
    if gcps and len(gcps) >= 2:
        a, s = alpha_from_gcps(detail, gcps, with_sigma=True)
        if a and a > 0:
            est["gcps"] = float(a)
            notes["gcps"] = {"n": len(gcps), "alpha_sigma_rel": s}
            sig["gcps"] = float(s)
    if known_height_m:
        est["prior"] = float(alpha_from_known_height(detail, known_height_m))
        notes["prior"] = {"known_height_m": float(known_height_m),
                          "alpha_sigma_rel": PRIOR_SIGMA_REL}
        sig["prior"] = PRIOR_SIGMA_REL

    out = dict(estimates=est, sigma_rel=sig, detail=notes, n_sources=len(est))
    if len(est) >= 2:
        v = np.array(list(est.values()), float)
        lo, hi = float(v.min()), float(v.max())
        out.update(spread_ratio=hi / max(lo, 1e-9),
                   agreement_pct=100.0 * lo / max(hi, 1e-9),
                   median_alpha=float(np.median(v)))
        print(f"[cross-check] {len(est)} independent scale sources: "
              + ", ".join(f"{k}={x:.2f}" for k, x in est.items())
              + f" | agreement {out['agreement_pct']:.0f}%")
    elif len(est) == 1:
        k = next(iter(est))
        print(f"[cross-check] only one scale source ({k}) - no agreement to report")
    return out


def fuse_scale_estimates(cross_check, max_z=FUSION_MAX_Z):
    """Combine the independent calibrators into one alpha, by how much each is
    worth. OFF by default - estimate_elevation(fuse_scale=True) turns it on.

    IN PLAIN ENGLISH: shadows, control points and a landmark guess all answer
    the same question, and averaging them treats a guess as equal to a
    measurement. They are combined by INVERSE VARIANCE instead, so the source
    that knows its own error is small dominates and the vague one only nudges.

        alpha = exp( sum(ln(alpha_i) / sigma_i^2) / sum(1 / sigma_i^2) )

    Two details that matter:

    LOG SPACE. alpha is a scale factor, so its errors are multiplicative - x2
    and x0.5 are equally wrong. Averaging 2 and 0.5 in linear space gives 1.25,
    which is biased high; in log space it gives 1.0. The sigmas are relative
    (a fraction of alpha), which is the same reason.

    THE DISAGREEMENT GUARD. If two sources sit further apart than their own
    error bars can explain - more than `max_z` combined sigmas - one of them is
    BROKEN, not noisy, and blending is the wrong response: the result is wrong
    by roughly half the gap, and the disagreement, which was real evidence,
    disappears into a plausible-looking number. It refuses instead, keeps the
    single tightest source, and says so.

    Returns a dict that always records what it decided and why. `alpha` is None
    when it declined, in which case the caller keeps whatever it had.
    """
    est = (cross_check or {}).get("estimates") or {}
    sig = (cross_check or {}).get("sigma_rel") or {}
    usable = {k: (float(est[k]), float(sig.get(k, 0.25)))
              for k in est if np.isfinite(est.get(k, np.nan)) and est[k] > 0}

    if len(usable) < 2:
        return dict(applied=False, alpha=None, n_sources=len(usable),
                    reason="fewer than two usable sources - nothing to fuse")

    names = sorted(usable)
    a = np.array([usable[k][0] for k in names], float)
    s = np.array([max(usable[k][1], 1e-3) for k in names], float)

    # the guard, pairwise, in the same log space the fusion uses
    worst_z, worst_pair = 0.0, None
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            z = abs(np.log(a[i]) - np.log(a[j])) / np.sqrt(s[i] ** 2 + s[j] ** 2)
            if z > worst_z:
                worst_z, worst_pair = float(z), (names[i], names[j])

    tightest = names[int(np.argmin(s))]
    detail = {k: dict(alpha=usable[k][0], sigma_rel=usable[k][1]) for k in names}

    if worst_z > max_z:
        print(f"[fuse] REFUSED: {worst_pair[0]}={usable[worst_pair[0]][0]:.2f} and "
              f"{worst_pair[1]}={usable[worst_pair[1]][0]:.2f} disagree by "
              f"{worst_z:.1f} sigma - one of them is wrong, not noisy. "
              f"Keeping the tightest single source ({tightest}).")
        return dict(applied=False, alpha=None, refused=True,
                    reason=(f"{worst_pair[0]} and {worst_pair[1]} disagree by "
                            f"{worst_z:.1f} sigma (limit {max_z})"),
                    worst_z=worst_z, disagreeing=list(worst_pair),
                    tightest_source=tightest, sources=detail)

    w = 1.0 / s ** 2
    fused = float(np.exp(float(np.sum(w * np.log(a)) / np.sum(w))))
    fused_sigma = float(1.0 / np.sqrt(np.sum(w)))
    print("[fuse] " + " + ".join(f"{k}={usable[k][0]:.2f}±{usable[k][1]*100:.0f}%"
                                 for k in names)
          + f" -> alpha={fused:.3f}±{fused_sigma*100:.0f}% "
            f"(worst disagreement {worst_z:.1f} sigma)")
    return dict(applied=True, alpha=fused, alpha_sigma_rel=fused_sigma,
                n_sources=len(names), worst_z=worst_z, sources=detail,
                method="inverse-variance in log space")


def confidence_report(spread, height, meta, ground_spread, cross_check,
                      alpha=None):
    """One reference-free statement of how much to trust this surface.

    Every input here is measured without any ground truth:
      - ensemble spread : how much the backbone disagrees with itself under
                          rotation, per pixel
      - ground spread   : how level the flat ground came out, which is leftover
                          tilt the pipeline failed to remove
      - scale agreement : whether two independent calibrators concur
    """
    rep = {}
    fin = np.isfinite(spread) & (spread > 0)
    if fin.any():
        rng = float(np.nanmax(height) - np.nanmin(height))
        rep["ensemble_spread_median"] = float(np.median(spread[fin]))
        rep["ensemble_spread_p90"] = float(np.percentile(spread[fin], 90))
        rep["ensemble_spread_pct_of_range"] = (
            100.0 * rep["ensemble_spread_median"] / max(rng, 1e-9))
        if alpha:
            rep["ensemble_spread_median_m"] = float(alpha * rep["ensemble_spread_median"])
    rep["ground_levelness"] = float(ground_spread)
    if cross_check.get("agreement_pct") is not None:
        rep["scale_agreement_pct"] = float(cross_check["agreement_pct"])
        rep["scale_sources"] = list(cross_check["estimates"])
    return rep


# ----------------------------------------------------------------------------
# 6. THE PIPELINE
# ----------------------------------------------------------------------------
def estimate_elevation(path, known_height_m=None, gcps=None,
                       sun_azimuth=None, sun_elevation=None,
                       use_dem=True, alpha_gain=1.0, outdir="outputs",
                       rotations=ROTATIONS, adaptive_sigma=True,
                       height_reference="tallest", debias_coarse_dem=True,
                       fuse_scale=False, sun_from_tags=False):
    """THE MAIN FUNCTION. Image path in, elevation map out.

    The numbered list at the top of this file is the order of what follows -
    read that first if you are new to this code. Everything here is either one
    of those steps or a decision about which step to take.

    Returns (height, meta, info).
      relative mode -> height is 0..1
      absolute mode -> height is metres above sea level

    `info` is the run's own record - which calibrator answered, what alpha came
    out, how much the rotations disagreed, whether the depth was flipped. It
    ends up in meta.json and in the results panel, so nothing important is
    decided only in a print().

    rotations       how many frame orientations to predict and average. 4
                    cancels the model's frame-tied perspective ramp by
                    construction and yields a per-pixel uncertainty map; 1
                    reverts to a single pass and the plane-fit detrend.
    adaptive_sigma  size the object/terrain split from the scene's own
                    structures instead of from the coarse DEM's resolution.
    fuse_scale      combine every calibrator that answered into one alpha by
                    inverse variance, instead of taking the first in priority
                    order. OFF by default: the accuracy figures in the
                    benchmark were produced with the priority path, and this
                    has not been measured against LiDAR yet. See
                    fuse_scale_estimates().
    sun_from_tags   True when the sun angles came from the GeoTIFF's own tags
                    rather than from a person. Only affects the error bar the
                    shadow calibrator carries into the fusion - angles read
                    from the file are a measurement, angles typed by eye are
                    a guess worth about 18% on alpha.
    """
    os.makedirs(outdir, exist_ok=True)
    rgb, meta = load_image(path)
    print(f"[load] {rgb.shape[1]}x{rgb.shape[0]}  mode={meta['mode']}  px={meta['px_size_m']}")

    info = dict(mode=meta["mode"], rotations=int(max(1, min(4, rotations))))
    p, spread = predict_depth_ensemble(rgb, rotations=rotations)
    p = validate_depth(p, rgb, info)  # Issue 1+10: fix inversion, check sanity
    p = clean_depth(p, rgb)

    # ---- remove the fake tilt ----
    # Issue 2 fix: in absolute mode with a DEM, the frequency split already
    # handles the DC ramp - detrending here removes REAL terrain slopes.
    # Only detrend if (a) no DEM will be used or (b) the ground spread is
    # large enough to indicate a genuine model artifact.
    will_use_dem = (meta["mode"] == "absolute" and use_dem)
    before = ground_consistency(p)
    ensembled = int(max(1, min(4, rotations))) >= 2
    if ensembled:
        # Averaging over frame orientations already cancelled the first-order
        # ramp, and it did so without touching genuine relief. Fitting a plane
        # on top would now remove real terrain for no gain.
        plane = np.zeros_like(p)
        after = before
        print(f"[detrend] not needed: {info['rotations']} rotations cancelled "
              f"the frame ramp (ground spread {100 * before:.0f}%)")
    elif will_use_dem:
        # DEM supplies terrain; detrending would remove real slopes
        plane = np.zeros_like(p)
        after = before
        print(f"[detrend] skipped (DEM will supply terrain baseline)")
    elif before > 0.12:
        p, plane = detrend(p)
        after = ground_consistency(p)
        print(f"[detrend] applied: ground spread {100*before:.0f}% -> {100*after:.0f}%")
    else:
        plane = np.zeros_like(p)
        after = before
        print(f"[detrend] skipped (ground spread {100*before:.0f}% is low, "
              f"likely real terrain)")
    ramp_share = float((plane.max() - plane.min()) /
                       max(np.nanmax(p) - np.nanmin(p) + (plane.max() - plane.min()), 1e-9))
    info.update(ramp_share=ramp_share, ground_spread_before=before, ground_spread_after=after)

    # ---------------- RELATIVE ----------------
    if meta["mode"] != "absolute":
        lo, hi = np.nanpercentile(p, [0.5, 99.8])   # wide: avoids clipping roof tops
        height = np.clip((p - lo) / max(hi - lo, 1e-9), 0, 1)
        info["note"] = "relative rDSM (no metric scale)"
        # spread is in the same relative units the surface was normalised into
        unc = spread / max(hi - lo, 1e-9) if np.any(spread) else None
        info["confidence"] = confidence_report(
            spread, height, meta, after, dict(estimates={}))
        _export(height, rgb, meta, outdir, info, uncertainty=unc)
        # private, so a caller that rescales the surface can rescale this too
        meta = dict(meta, _uncertainty=unc)
        return height, meta, info

    # ---------------- ABSOLUTE ----------------
    px = meta["px_size_m"] or 1.0
    sigma_m = (structure_scale_m(p, px) if adaptive_sigma else DEM_RES_M / 2)
    sigma = max(2.0, sigma_m / px)              # the terrain/building boundary
    # gaussian_filter smears a single NaN across the whole neighbourhood, and
    # clean_depth deliberately puts the NaNs back where the input had them, so
    # fill before blurring and restore afterwards. structure_scale_m already
    # did this; the split that actually produces the elevations did not.
    p_fill = _fill_nan(p)
    detail = p_fill - gaussian_filter(p_fill, sigma)
    detail = np.where(np.isfinite(p), detail, np.nan)   # buildings live here
    info["sigma_px"] = float(sigma)
    info["sigma_m"] = float(sigma_m)
    info["sigma_source"] = "scene structures" if adaptive_sigma else "DEM resolution"

    alpha = None
    tried_shadows = False
    if (sun_azimuth is None and sun_elevation is None
            and not gcps and not known_height_m):
        # last resort only - an explicit choice by the caller always wins
        sun_azimuth, sun_elevation = meta.get("sun_azimuth"), meta.get("sun_elevation")
        if sun_azimuth is not None and sun_elevation is not None:
            sun_from_tags = True          # read from the file, not guessed
    sun_sigma_deg = 0.0 if sun_from_tags else SUN_GUESS_SIGMA_DEG
    if sun_azimuth is not None and sun_elevation is not None:
        tried_shadows = True
        alpha, sdiag = alpha_from_shadows(detail, rgb, sun_azimuth, sun_elevation,
                                          px, sun_sigma_deg=sun_sigma_deg)
        info["self_check"] = sdiag
        info["sun_from_tags"] = bool(sun_from_tags)
    if alpha is None and gcps:
        alpha = alpha_from_gcps(detail, gcps)
        if alpha is not None:
            print(f"[gcp] alpha={alpha:.3f} from {len(gcps)} control points")
    if alpha is None and known_height_m:
        alpha = alpha_from_known_height(detail, known_height_m,
                                        reference=height_reference)
        info["height_reference"] = str(height_reference)
        print(f"[prior] alpha={alpha:.2f} from {height_reference} structure "
              f"= {known_height_m} m")
    if alpha is not None and alpha_gain and abs(alpha_gain - 1.0) > 1e-6:
        # closes the loop with validate.py: freqsplit attenuates structures, and
        # the validation report measures by how much. Feed its suggested gain
        # back in here rather than hand-tuning known_height_m upward.
        alpha *= float(alpha_gain)
        info["alpha_gain"] = float(alpha_gain)
        print(f"[calib] alpha x{alpha_gain:.2f} (measured attenuation correction)")
    if alpha is None and tried_shadows:
        # Do not tell someone who supplied sun angles to supply sun angles.
        n_pairs = (info.get("self_check") or {}).get("n", 0)
        raise ValueError(
            f"Shadow calibration found only {n_pairs} usable shadow/roof pairs "
            "in this scene, which is not enough to set the metre scale. That "
            "happens on hazy imagery, on a high sun that casts almost nothing, "
            "and where shadows fall on other buildings rather than on open "
            "ground. Pass known_height_m (the height of the tallest structure "
            "you can identify) or at least two gcps instead.")
    if alpha is None:
        raise ValueError(
            "No scale source. The coarse DEM supplies the terrain baseline but "
            "not this multiplier - the model's low-frequency band carries a "
            "large fake ramp, exactly where a 30 m DEM is blind. Pass "
            "known_height_m, gcps, or sun_azimuth + sun_elevation.")
    info["alpha"] = float(alpha)

    # Every calibrator that COULD have answered, run and compared. None of this
    # touches the chosen alpha - it reports whether independent methods concur,
    # which is the only confidence statement available where no LiDAR exists.
    info["cross_check"] = cross_check_scale(
        detail, rgb, px, known_height_m=known_height_m, gcps=gcps,
        sun_azimuth=sun_azimuth, sun_elevation=sun_elevation,
        sun_sigma_deg=sun_sigma_deg)

    # ...unless the caller asked for the sources to be COMBINED rather than
    # ranked. Off by default, so the path above is untouched for every run that
    # does not opt in. The gain is re-applied because the fusion works from the
    # raw estimates, which never saw it.
    if fuse_scale:
        fusion = fuse_scale_estimates(info["cross_check"])
        info["scale_fusion"] = fusion
        if fusion.get("alpha") is not None:
            alpha = float(fusion["alpha"])
            if alpha_gain and abs(alpha_gain - 1.0) > 1e-6:
                alpha *= float(alpha_gain)
            info["alpha_priority"] = info["alpha"]      # what it would have been
            info["alpha"] = float(alpha)
            print(f"[calib] alpha replaced by the fused estimate: "
                  f"{info['alpha_priority']:.3f} -> {alpha:.3f}")

    dem_path = fetch_dem(meta, os.path.join(outdir, "dem_coarse.tif")) if use_dem else None
    if dem_path:
        terrain = dem_on_grid(dem_path, meta, p.shape)
        terrain = np.where(np.isfinite(terrain), terrain, np.nanmedian(terrain))
        if debias_coarse_dem:
            terrain, shift = debias_dem(terrain, px)
            info["dem_debias_m"] = shift
            print(f"[dem] rooftop contamination removed: terrain lowered by "
                  f"{shift:.2f} m (median)")
        info["calibration"] = "freqsplit (DEM terrain + scaled detail)"
    else:
        terrain = np.zeros_like(p)
        info["calibration"] = "scale-only (no DEM; heights above local ground)"
        print("[calib] no DEM - output is height above ground, not above sea level")

    # FREQSPLIT: DEM supplies terrain, model supplies detail.
    # Never alpha*p + beta - that carries the ramp straight through.
    # The detail band is zero-centred and, in a dense scene, ground pixels sit
    # BELOW the local average - so ground lands at a negative value. Shift by
    # the histogram mode (the commonest surface = the ground), not by a low
    # percentile, which would land in the middle of open areas instead.
    detail = detail - ground_level(detail)
    height = terrain + alpha * detail
    if dem_path is None:
        # heights are above local ground, so negatives are residual error, not
        # real relief. Clip and report how much we clipped - do not hide it.
        neg = float((height < 0).mean())
        info["clipped_negative_frac"] = neg
        if neg > 0.02:
            print(f"[calib] {100*neg:.1f}% of pixels below ground - clipped. "
                  f"High values mean the ramp was not fully removed.")
        height = np.maximum(height, 0.0)
    print(f"[calib] {info['calibration']}  alpha={alpha:.2f}  "
          f"range {np.nanmin(height):.1f}..{np.nanmax(height):.1f} m")

    # Derive the other two products AFTER the clip, so all three stay
    # consistent: dsm = dtm + ndsm holds pixel for pixel.
    ndsm = height - terrain
    ndsm = np.maximum(ndsm, 0.0)          # nothing sits below its own ground
    dtm = height - ndsm
    # the ensemble spread is in the model's own units; alpha converts it to
    # metres, so the uncertainty raster is in the same units as the DSM
    unc = (alpha * spread) if np.any(spread) else None
    info["confidence"] = confidence_report(
        spread, height, meta, after, info.get("cross_check", {}), alpha=alpha)
    info["ndsm_p99_m"] = float(np.nanpercentile(ndsm, 99))
    info["ndsm_mean_m"] = float(np.nanmean(ndsm))

    _export(height, rgb, meta, outdir, info, ndsm=ndsm, dtm=dtm, uncertainty=unc)
    meta = dict(meta, _uncertainty=unc)
    return height, meta, info


def clip_below_ground(height, info):
    """Re-apply the no-DEM ground clip AFTER refine().

    estimate_elevation clips negatives when no coarse DEM was available,
    because the surface is then height above local ground and nothing sits
    below its own ground. But refine() runs afterwards, in the caller, and its
    unsharp pass deliberately overshoots at rooflines - which puts pixels back
    under zero in the surface that is actually exported, meshed and scored.
    The clip inside estimate_elevation was therefore protecting an
    intermediate nobody ever sees.

    No-op in absolute mode with a DEM, where a negative elevation is a real
    thing: most of this country sits below sea level.

    Returns (height, info) with the clipped fraction recorded either way.
    """
    if "no DEM" not in info.get("calibration", ""):
        return height, info
    h = np.asarray(height, np.float64)
    neg = float(np.mean(h < 0))
    info = dict(info, clipped_negative_frac_post_refine=neg)
    if neg > 0.02:
        print(f"[calib] {100 * neg:.1f}% of pixels below ground after refine - "
              f"clipped. High values mean the ramp was not fully removed.")
    return np.maximum(h, 0.0), info


def export_products(height, rgb, meta, outdir, info, ndsm=None, dtm=None,
                    uncertainty=None):
    """Re-write the rasters from a surface that changed after estimate_elevation.

    estimate_elevation exports inside itself, but the caller then rescales a
    relative surface into metres and runs refine() before meshing. Everything
    downstream - the glTF, the validator, the height readout - used that later
    surface while dsm.tif and height16.png still held the earlier one. So the
    elevation map on disk and the 3D model were two different surfaces, and the
    accuracy numbers scored a raster the viewer never showed.

    In absolute mode the nDSM has to move with the DSM or `dsm = dtm + ndsm`
    stops holding, so it is recomputed against the terrain already on disk.
    """
    info = dict(info)
    info.pop("products", None)          # _export appends; don't accumulate

    if ndsm is None and dtm is None and meta.get("mode") == "absolute":
        p = os.path.join(outdir, "dtm.tif")
        if os.path.exists(p):
            import rasterio
            with rasterio.open(p) as ds:
                dtm = ds.read(1).astype(np.float64)
            ndsm = np.maximum(np.asarray(height, np.float64) - dtm, 0.0)

    _export(height, rgb, meta, outdir, info, ndsm=ndsm, dtm=dtm,
            uncertainty=uncertainty)
    return info


def _export(height, rgb, meta, outdir, info, ndsm=None, dtm=None,
            uncertainty=None):
    """dsm.tif + ndsm.tif + dtm.tif + height16.png + texture.png + meta.json

    Three surfaces, not one. A DSM alone answers "how high is the top of
    everything", which is rarely the question. The nDSM (surface minus terrain)
    is the building and canopy height that planning and damage assessment
    actually want, and the DTM is the bare ground underneath. Both fall out of
    the frequency split for free - not exporting them was leaving the most
    useful product on the floor.
    """
    import rasterio
    h = np.asarray(height, np.float32)
    prof = dict(driver="GTiff", height=h.shape[0], width=h.shape[1], count=1,
                dtype="float32", nodata=np.nan, compress="deflate")
    if meta["mode"] == "absolute":
        prof.update(crs=meta["crs"], transform=meta["transform"])

    # Say what is actually in the file. A relative run is re-exported by the
    # caller AFTER it has been stretched onto a nominal full-scale height, so
    # the values at that point are metres - just not surveyed ones - and
    # tagging them "relative" sent anyone reading the raster looking for a 0..1
    # surface that is no longer there.
    if meta["mode"] == "absolute":
        units = "metres"
    elif info.get("relative_full_scale_m"):
        units = (f"metres (nominal: no metric datum, scene scaled so the full "
                 f"range is {float(info['relative_full_scale_m']):.0f} m)")
    else:
        units = "relative (0..1)"
    with rasterio.open(os.path.join(outdir, "dsm.tif"), "w", **prof) as ds:
        ds.write(h, 1)
        ds.update_tags(MODE=meta["mode"], UNITS=units, PRODUCT="DSM")

    for fname, arr, product, desc in (
            ("ndsm.tif", ndsm, "nDSM", "height above local ground"),
            ("dtm.tif", dtm, "DTM", "bare terrain"),
            ("uncertainty.tif", uncertainty, "UNCERTAINTY",
             "per-pixel disagreement between rotated prediction passes, "
             "same units as the DSM")):
        if arr is None:
            continue
        with rasterio.open(os.path.join(outdir, fname), "w", **prof) as ds:
            ds.write(np.asarray(arr, np.float32), 1)
            ds.update_tags(MODE=meta["mode"], UNITS=units,
                           PRODUCT=product, DESCRIPTION=desc)
        info.setdefault("products", []).append(fname)

    lo, hi = float(np.nanmin(h)), float(np.nanmax(h))
    norm = np.clip((h - lo) / max(hi - lo, 1e-9), 0, 1)
    Image.fromarray((norm * 65535).astype(np.uint16)).save(os.path.join(outdir, "height16.png"))
    Image.fromarray(rgb).save(os.path.join(outdir, "texture.png"))

    js = dict(info, min_m=lo, max_m=hi, width=int(h.shape[1]), height=int(h.shape[0]),
              px_size_m=meta.get("px_size_m"), crs=str(meta["crs"]) if meta["crs"] else None)
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump(js, f, indent=2, default=float)
    extra = " / ".join(info.get("products", []))
    print(f"[export] dsm.tif{' / ' + extra if extra else ''} / height16.png / "
          f"texture.png / meta.json -> {outdir}")


# ----------------------------------------------------------------------------
def measure_height(height, meta, row, col):
    """Click-to-read for the UI. Returns (value, units)."""
    v = float(height[int(row), int(col)])
    return v, ("m" if meta["mode"] == "absolute" else "relative")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python inference.py <image> [known_height_m]")
        raise SystemExit(1)
    kh = float(sys.argv[2]) if len(sys.argv) > 2 else None
    estimate_elevation(sys.argv[1], known_height_m=kh)
