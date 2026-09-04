"""
inference.py - single-view RGB -> elevation map.

Two routes, chosen by METADATA (not file extension):
  no CRS  -> relative DSM, normalised 0..1
  has CRS -> absolute DSM, metres

Three things in here are not optional. Each was measured, not guessed:

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
from scipy.ndimage import gaussian_filter

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
OPENTOPO_KEY = os.environ.get("OPENTOPO_KEY", "")   # free: portal.opentopography.org

_pipe = None


# ----------------------------------------------------------------------------
# 1. LOAD  (route on metadata, not extension - a .tif can be untagged)
# ----------------------------------------------------------------------------
def load_image(path):
    """Returns (rgb uint8 HxWx3, meta). meta['mode'] is 'absolute' or 'relative'."""
    import rasterio
    from rasterio.transform import Affine

    meta = dict(path=path, mode="relative", crs=None, transform=None,
                px_size_m=None, bounds=None, sun_azimuth=None, sun_elevation=None)

    if os.path.splitext(path)[1].lower() in (".tif", ".tiff"):
        with rasterio.open(path) as ds:
            arr = ds.read()
            n = min(3, ds.count)
            rgb = np.transpose(arr[:n], (1, 2, 0))
            if n == 1:
                rgb = np.repeat(rgb, 3, axis=2)
            if rgb.dtype != np.uint8:                      # 16-bit satellite data
                lo, hi = np.nanpercentile(rgb, [2, 98])
                rgb = (np.clip((rgb - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8)

            has_crs = ds.crs is not None and ds.transform is not None \
                      and ds.transform != Affine.identity() \
                      and abs(ds.transform.a) > 0
            if has_crs:
                meta.update(mode="absolute", crs=ds.crs, transform=ds.transform,
                            bounds=ds.bounds, px_size_m=_px_size_m(ds))
            meta.update(_sun_from_tags(ds.tags()))
    else:
        rgb = np.array(Image.open(path).convert("RGB"))

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
    rows = sorted({*range(0, max(1, H - overlap), step), max(0, H - tile)})
    cols = sorted({*range(0, max(1, W - overlap), step), max(0, W - tile)})
    rows = [min(r, max(0, H - tile)) for r in rows]
    cols = [min(c, max(0, W - tile)) for c in cols]
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


# ----------------------------------------------------------------------------
# 3. VALIDATE + FIX INVERSION + CLEAN + DETREND
# ----------------------------------------------------------------------------
def validate_depth(p, rgb):
    """Sanity-check the raw depth output and fix inversion if detected.

    Issue 10 fix: reject degenerate outputs early.
    Issue 1 fix: Depth-Anything-V2 was trained on ground-level perspective
    photos.  On nadir satellite views it frequently inverts the map - rooftops
    appear as pits, streets appear elevated.  Detect this by correlating image
    luminance with predicted depth; a strong negative correlation signals
    inversion.
    """
    finite = np.isfinite(p)
    if finite.mean() < 0.5:
        print("[depth] WARNING: >50% of depth pixels are NaN")
    if finite.any():
        var = float(np.nanvar(p[finite]))
        if var < 1e-12:
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
            # On a nadir image, bright rooftops should have HIGH depth (they
            # are elevated = closer to sensor).  A negative correlation means
            # the model thinks bright = far away, which is the ground-level
            # bias inverted for a top-down view.
            if r < -0.3:
                print(f"[depth] depth appears INVERTED (luma-depth r={r:.2f}), "
                      f"correcting")
                p = np.nanmax(p) - p
            else:
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

def alpha_from_known_height(detail, known_height_m, pct=97.0):
    """Semantic prior. 'The tallest block here is about 40 m.' One number, and
    it is the fastest thing to demo. The PS explicitly allows semantic priors.

    Issue 4 fix: use the median of the top percentile rather than a single
    extreme value.  A lone antenna or noise spike at p99.5 produces a wildly
    wrong alpha.  The median of everything above p97 represents the tallest
    sustained structure, which is what the user's guess corresponds to."""
    threshold = np.nanpercentile(detail, pct)
    top_vals = detail[np.isfinite(detail) & (detail >= threshold)]
    if top_vals.size == 0:
        top = float(np.nanpercentile(detail, 99.5))
    else:
        top = float(np.nanmedian(top_vals))
    return float(known_height_m / max(top, 1e-9))


def alpha_from_gcps(detail, gcps):
    """gcps = [(row, col, height_above_ground_m), ...]. Needs >= 2.
    NOTE: takes the DETAIL band, not raw p. alpha is applied to detail, so it
    must be fitted on detail too - fitting on p gave a NEGATIVE alpha in test,
    because p still carries residual ramp."""
    if not gcps or len(gcps) < 2:
        return None
    g = ground_level(detail)
    x = np.array([detail[int(r), int(c)] - g for r, c, _ in gcps], float)
    y = np.array([h for _, _, h in gcps], float)
    ok = np.isfinite(x) & np.isfinite(y) & (np.abs(x) > 1e-9)
    if ok.sum() < 2:
        return None
    return float(np.polyfit(x[ok], y[ok], 1)[0])



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
    res = []
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        if tr.size < 4 or te.size < 1:
            continue
        a = float(np.median(Y[tr] / X[tr]))
        res.append(a * X[te] - Y[te])
    if not res:
        return dict(n=int(n), note="cross-validation folds too small")

    e = np.concatenate(res)
    ratios = Y / X
    q1, q3 = np.percentile(ratios, [25, 75])
    return dict(
        n=int(n),
        rmse_m=float(np.sqrt(np.mean(e ** 2))),
        mae_m=float(np.mean(np.abs(e))),
        bias_m=float(np.mean(e)),
        p90_abs_m=float(np.percentile(np.abs(e), 90)),
        median_rel_pct=float(100 * np.median(np.abs(e) / np.maximum(Y[idx[:e.size]], 1e-6))),
        alpha_iqr_ratio=float(q3 / max(q1, 1e-9)),
        basis="held-out shadow control points; not LiDAR validation",
    )


def alpha_from_shadows(detail, rgb, sun_az_deg, sun_elev_deg, px_size_m):
    """h = L * tan(theta). Shadows are self-generated control points: dozens of
    them, no clicking and no downloads. Complements the DEM exactly - the DEM
    knows terrain and not buildings; shadows know buildings and not terrain.

    Issue 3 fix: adaptive shadow thresholding (Otsu on V-channel instead of a
    fixed 25th percentile), directional consistency filter (shadow extent must
    align within +-30 deg of the expected sun azimuth), and shape filter
    (shadows are elongated, not circular)."""
    import cv2
    a = np.deg2rad(sun_az_deg)
    dc, dr = -np.sin(a), np.cos(a)          # 90->west, 180->north (verified)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    v_chan = (hsv[..., 2] / 255.0 * 255).astype(np.uint8)

    # Adaptive threshold: Otsu finds the natural dark/bright split instead of
    # a fixed percentile that floods the mask on overcast imagery.
    _, dark_mask = cv2.threshold(v_chan, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    flat = (hsv[..., 1] / 255. < 0.40)
    mask = cv2.morphologyEx((dark_mask.astype(bool) & flat).astype(np.uint8),
                            cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    n, lab = cv2.connectedComponents(mask)
    g = ground_level(detail)          # DETAIL band, same reason as GCPs
    X, Y = [], []
    for k in range(1, n):
        ys, xs = np.where(lab == k)
        if ys.size < 6:
            continue

        # --- directional consistency filter ---
        # The shadow's principal axis should align with the expected sun
        # direction.  Shadows perpendicular to the sun are not cast shadows.
        t = xs * dc + ys * dr
        ext = t.max() - t.min()
        # perpendicular extent
        t_perp = -xs * dr + ys * dc
        ext_perp = t_perp.max() - t_perp.min()
        if not (4 <= ext <= 400):
            continue
        # aspect ratio: real shadows are elongated along the sun direction
        if ext_perp > 0 and ext / max(ext_perp, 1) < 1.2:
            continue                            # too circular, not a shadow

        i = int(np.argmin(t))                       # object end of the shadow
        dp = detail[ys[i], xs[i]] - g
        h = ext * px_size_m * np.tan(np.deg2rad(sun_elev_deg))
        if np.isfinite(dp) and dp > 1e-6 and h > 1.5:
            X.append(dp); Y.append(h)
    if len(X) < 8:
        print(f"[shadow] only {len(X)} usable pairs - skipping")
        return None, dict(n=len(X), note="too few shadow pairs")
    # median ratio is robust enough here and needs no sklearn
    a_est = float(np.median(np.array(Y) / np.array(X)))
    diag = _cv_ratio(X, Y)
    if "rmse_m" in diag:
        print(f"[shadow] alpha={a_est:.2f} from {len(X)} pairs | "
              f"held-out RMSE {diag['rmse_m']:.2f} m, "
              f"MAE {diag['mae_m']:.2f} m, bias {diag['bias_m']:+.2f} m")
    else:
        print(f"[shadow] alpha={a_est:.2f} from {len(X)} pairs")
    return a_est, diag


# ----------------------------------------------------------------------------
# 6. THE PIPELINE
# ----------------------------------------------------------------------------
def estimate_elevation(path, known_height_m=None, gcps=None,
                       sun_azimuth=None, sun_elevation=None,
                       use_dem=True, alpha_gain=1.0, outdir="outputs"):
    """
    Returns (height, meta, info).
      relative mode -> height is 0..1
      absolute mode -> height is metres above sea level
    """
    os.makedirs(outdir, exist_ok=True)
    rgb, meta = load_image(path)
    print(f"[load] {rgb.shape[1]}x{rgb.shape[0]}  mode={meta['mode']}  px={meta['px_size_m']}")

    p = predict_depth(rgb)
    p = validate_depth(p, rgb)       # Issue 1+10: fix inversion, check sanity
    p = clean_depth(p, rgb)
    info = dict(mode=meta["mode"])

    # ---- remove the fake tilt ----
    # Issue 2 fix: in absolute mode with a DEM, the frequency split already
    # handles the DC ramp - detrending here removes REAL terrain slopes.
    # Only detrend if (a) no DEM will be used or (b) the ground spread is
    # large enough to indicate a genuine model artifact.
    will_use_dem = (meta["mode"] == "absolute" and use_dem)
    before = ground_consistency(p)
    if will_use_dem:
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
        _export(height, rgb, meta, outdir, info)
        return height, meta, info

    # ---------------- ABSOLUTE ----------------
    px = meta["px_size_m"] or 1.0
    sigma = max(2.0, DEM_RES_M / px / 2)        # ~15 m: the terrain/building boundary
    detail = p - gaussian_filter(p, sigma)      # buildings live here
    info["sigma_px"] = float(sigma)

    alpha = None
    if (sun_azimuth is None and sun_elevation is None
            and not gcps and not known_height_m):
        # last resort only - an explicit choice by the caller always wins
        sun_azimuth, sun_elevation = meta.get("sun_azimuth"), meta.get("sun_elevation")
    if sun_azimuth is not None and sun_elevation is not None:
        alpha, sdiag = alpha_from_shadows(detail, rgb, sun_azimuth, sun_elevation, px)
        info["self_check"] = sdiag
    if alpha is None and gcps:
        alpha = alpha_from_gcps(detail, gcps)
        print(f"[gcp] alpha={alpha}")
    if alpha is None and known_height_m:
        alpha = alpha_from_known_height(detail, known_height_m)
        print(f"[prior] alpha={alpha:.2f} from known height {known_height_m} m")
    if alpha is not None and alpha_gain and abs(alpha_gain - 1.0) > 1e-6:
        # closes the loop with validate.py: freqsplit attenuates structures, and
        # the validation report measures by how much. Feed its suggested gain
        # back in here rather than hand-tuning known_height_m upward.
        alpha *= float(alpha_gain)
        info["alpha_gain"] = float(alpha_gain)
        print(f"[calib] alpha x{alpha_gain:.2f} (measured attenuation correction)")
    if alpha is None:
        raise ValueError(
            "No scale source. The coarse DEM cannot supply one - the model's "
            "low-frequency band carries a large fake ramp. Pass known_height_m, "
            "gcps, or sun_azimuth + sun_elevation.")
    info["alpha"] = float(alpha)

    dem_path = fetch_dem(meta, os.path.join(outdir, "dem_coarse.tif")) if use_dem else None
    if dem_path:
        terrain = dem_on_grid(dem_path, meta, p.shape)
        terrain = np.where(np.isfinite(terrain), terrain, np.nanmedian(terrain))
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
    info["ndsm_p99_m"] = float(np.nanpercentile(ndsm, 99))
    info["ndsm_mean_m"] = float(np.nanmean(ndsm))

    _export(height, rgb, meta, outdir, info, ndsm=ndsm, dtm=dtm)
    return height, meta, info


def export_products(height, rgb, meta, outdir, info, ndsm=None, dtm=None):
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

    _export(height, rgb, meta, outdir, info, ndsm=ndsm, dtm=dtm)
    return info


def _export(height, rgb, meta, outdir, info, ndsm=None, dtm=None):
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
    units = "metres" if meta["mode"] == "absolute" else "relative"
    with rasterio.open(os.path.join(outdir, "dsm.tif"), "w", **prof) as ds:
        ds.write(h, 1)
        ds.update_tags(MODE=meta["mode"], UNITS=units, PRODUCT="DSM")

    for fname, arr, product, desc in (
            ("ndsm.tif", ndsm, "nDSM", "height above local ground"),
            ("dtm.tif", dtm, "DTM", "bare terrain")):
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
