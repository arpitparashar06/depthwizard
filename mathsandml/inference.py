"""The elevation engine."""

import os, io, json, math, hashlib, shutil, time
import numpy as np
from PIL import Image
from scipy.ndimage import (gaussian_filter, minimum_filter, maximum_filter,
                           percentile_filter, zoom)

import preprocess as PRE

MODEL_ID     = os.environ.get("DEPTH_MODEL",
                              "depth-anything/Depth-Anything-V2-Large-hf")
TILE         = 518          # model's native patch grid
OVERLAP      = 180
DEM_SOURCE   = "COP30"      # COP30 | NASADEM | SRTMGL1
DEM_RES_M    = 30.0
# How many frame orientations to predict and average.
ROTATIONS    = int(os.environ.get("DEPTH_ROTATIONS", "4"))
OPENTOPO_KEY = os.environ.get("OPENTOPO_KEY", "")
# Largest raster read at full resolution.
MAX_READ_PIXELS = int(os.environ.get("MAX_READ_PIXELS", 40_000_000))

# Where the model runs.
DEPTH_DEVICE = os.environ.get("DEPTH_DEVICE", "auto").strip().lower()

ENGINE_ENV = "DEPTHWIZARD_ENGINE"
ENGINES = ("zeroshot", "finetuned", "metric", "hybrid")
ENGINE_LABEL = {
    "zeroshot": "zero-shot depth + scale anchor",
    "finetuned": "fine-tuned relative depth + scale anchor (engine A)",
    "metric": "HeightNet metres (engine B)",
    "hybrid": "hybrid A+B: fine-tuned relative depth, calibrated and fused with HeightNet metres",
}
_ENGINE_ALIASES = {"zero-shot": "zeroshot", "frozen": "zeroshot", "depth": "zeroshot",
                   "a": "finetuned", "relative": "finetuned", "fine-tuned": "finetuned",
                   "b": "metric", "heightnet": "hybrid", "a+b": "hybrid", "combined": "hybrid"}


def refine_by_default(engine):
    """Whether refine.py's clean-up should run on this engine's surface."""
    return engine == "zeroshot"


def engine_needs_anchor(engine):
    """Engines whose metres come from a scale anchor (shadows/GCPs/prior)."""
    return engine in ("zeroshot", "finetuned")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_ROOT = os.environ.get("DEPTHWIZARD_CACHE", os.path.join(_REPO_ROOT, "cache"))
DEPTH_CACHE = os.environ.get("DEPTH_CACHE", "1") not in ("0", "false", "no", "")
DEPTH_CACHE_MB = float(os.environ.get("DEPTH_CACHE_MB", "2048"))
DEM_DIRS = [d for d in (os.environ.get("DEM_DIR", ""),
                        os.path.join(CACHE_ROOT, "dem")) if d]

_pipe = None
_pipe_device = None


def load_image(path, preprocess=True):
    """Returns (rgb uint8 HxWx3, meta)."""
    import rasterio
    from rasterio.transform import Affine

    meta = dict(path=path, mode="relative", crs=None, transform=None,
                px_size_m=None, bounds=None, sun_azimuth=None, sun_elevation=None,
                valid_mask=None, preprocess=None)

    if os.path.splitext(path)[1].lower() in (".tif", ".tiff"):
        with rasterio.open(path) as ds:
            idx, band_note = PRE.pick_rgb_bands(ds)

            dec = 1
            while (ds.height // dec) * (ds.width // dec) > MAX_READ_PIXELS:
                dec += 1
            if dec > 1:
                oh, ow = ds.height // dec, ds.width // dec
                arr = ds.read(idx, out_shape=(len(idx), oh, ow),
                              resampling=rasterio.enums.Resampling.average)
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
    # Some products store zenith instead of elevation
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
    a = abs(ds.transform.a)
    if ds.crs.is_geographic:
        lat = (ds.bounds.bottom + ds.bounds.top) / 2.0
        return float(a * 111320.0 * math.cos(math.radians(lat)))
    return float(a)


def _pick_device(want=None):
    import torch
    want = (want or DEPTH_DEVICE or "auto").lower()
    has_mps = bool(getattr(torch.backends, "mps", None)
                   and torch.backends.mps.is_available())
    if want == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if want == "mps":
        return "mps" if has_mps else "cpu"
    if want == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_pipe(device=None):
    global _pipe, _pipe_device
    if _pipe is None or (device and device != _pipe_device):
        from transformers import pipeline
        dev = device or _pick_device()
        print(f"[depth] loading {MODEL_ID} on {dev}")
        _pipe = pipeline("depth-estimation", model=MODEL_ID, device=dev)
        _pipe_device = dev
    return _pipe


def _predict_patch(patch):
    try:
        out = _get_pipe()(Image.fromarray(patch))
    except (RuntimeError, NotImplementedError) as ex:
        if _pipe_device != "mps":
            raise
        # An operator the Metal backend does not implement - finish on the CPU
        print(f"[depth] MPS failed ({type(ex).__name__}: {str(ex)[:80]}) - "
              f"falling back to the CPU")
        out = _get_pipe("cpu")(Image.fromarray(patch))
    d = out["predicted_depth"]
    d = d.squeeze().cpu().numpy() if hasattr(d, "cpu") else np.asarray(d)
    if d.shape != patch.shape[:2]:
        d = np.array(Image.fromarray(d).resize((patch.shape[1], patch.shape[0]), Image.BILINEAR))
    return d.astype(np.float64)


def _feather(h, w, ov):
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


_REAL_PREDICT_DEPTH = predict_depth


def _fill_nan(a, fill=None):
    a = np.asarray(a, np.float64)
    valid = np.isfinite(a)
    if valid.all():
        return a
    if fill is None:
        fill = float(np.median(a[valid])) if valid.any() else 0.0
    return np.where(valid, a, fill)


def _fit_affine(src, ref, iters=4):
    valid = np.isfinite(src) & np.isfinite(ref)
    x, y = src[valid], ref[valid]
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


def source_key(path):
    """Cache identity of an input file: its bytes plus the conditioning code."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    try:
        with open(PRE.__file__, "rb") as f:
            h.update(f.read())
    except Exception:
        pass
    return h.hexdigest()


def _depth_cache_path(rgb, tile, overlap, n, key=None):
    h = hashlib.sha1()
    h.update(f"v1|{MODEL_ID}|{tile}|{overlap}|{n}|{rgb.shape}".encode())
    if key:
        h.update(f"file:{key}".encode())
    else:
        h.update(np.ascontiguousarray(rgb).tobytes())
    return os.path.join(CACHE_ROOT, "depth", h.hexdigest()[:32] + ".npz")


def _depth_cache_get(path, shape=None):
    try:
        with np.load(path) as z:
            mean, spread = z["mean"].astype(np.float64), z["spread"].astype(np.float64)
        if shape is not None and mean.shape != tuple(shape[:2]):
            return None
        os.utime(path)
        return mean, spread
    except Exception:
        return None


def _depth_cache_put(path, mean, spread):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, mean=mean.astype(np.float32),
                            spread=spread.astype(np.float32))
        os.replace(tmp, path)
        # Oldest-first eviction past the size budget
        d = os.path.dirname(path)
        files = sorted((os.path.join(d, f) for f in os.listdir(d) if f.endswith(".npz")),
                       key=os.path.getmtime)
        total = sum(os.path.getsize(f) for f in files)
        while files and total > DEPTH_CACHE_MB * 1e6:
            f = files.pop(0)
            if f == path:
                continue
            total -= os.path.getsize(f)
            os.remove(f)
    except Exception as ex:
        print(f"[depth] cache write skipped: {ex}")


def predict_depth_ensemble(rgb, tile=TILE, overlap=OVERLAP, rotations=4,
                           verbose=True, cache=None, cache_key=None):
    """The rotation ensemble below, behind a disk cache."""
    n = int(max(1, min(4, rotations)))
    use = (DEPTH_CACHE if cache is None else bool(cache)) \
        and predict_depth is _REAL_PREDICT_DEPTH
    path = _depth_cache_path(rgb, tile, overlap, n, cache_key) if use else None
    if path and os.path.exists(path):
        hit = _depth_cache_get(path, rgb.shape)
        if hit is not None:
            if verbose:
                print(f"[depth] cache hit - backbone skipped ({os.path.basename(path)[:12]})")
            return hit
    mean, spread = _predict_depth_ensemble(rgb, tile, overlap, n, verbose)
    mean = np.asarray(mean, np.float32).astype(np.float64)
    spread = np.asarray(spread, np.float32).astype(np.float64)
    if path:
        _depth_cache_put(path, mean, spread)
    return mean, spread


def _predict_depth_ensemble(rgb, tile=TILE, overlap=OVERLAP, rotations=4,
                            verbose=True):
    # Predict at 0/90/180/270 and average; the fake tilt flips and cancels out.
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

    def _standardise(a):
        valid = np.isfinite(a)
        med = float(np.median(a[valid])) if valid.any() else 0.0
        mad = 1.4826 * float(np.median(np.abs(a[valid] - med))) if valid.any() else 1.0
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
    """How wide are the structures in this scene, in metres."""
    from scipy.ndimage import distance_transform_edt, label
    px = max(float(px_size_m or 1.0), 1e-6)
    floor_m = float(seed_sigma_m if floor_m is None else floor_m)

    a = np.asarray(p, np.float64)
    fill = np.nanmedian(a[np.isfinite(a)]) if np.isfinite(a).any() else 0.0
    a = np.where(np.isfinite(a), a, fill)
    obj = a - gaussian_filter(a, max(2.0, seed_sigma_m / px))
    mask = obj > np.percentile(obj, 88)

    labels, n = label(mask)
    if n == 0:
        return floor_m
    dist = distance_transform_edt(mask)
    widths, areas = [], []
    for i in range(1, n + 1):
        m = labels == i
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


def validate_depth(p, rgb, info=None):
    """Sanity-check the raw depth output and fix inversion if detected."""
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

    import cv2
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float64).ravel()
    pd = p.ravel()
    valid = np.isfinite(pd)
    if valid.sum() > 100:
        # Subsample for speed on large images
        step = max(1, valid.sum() // 50000)
        gs, ps = gray[valid][::step], pd[valid][::step]
        if gs.std() > 1e-9 and ps.std() > 1e-9:
            r = float(np.corrcoef(gs, ps)[0, 1])
            rec["luma_depth_r"] = r
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
    """Edge-preserving smooth."""
    import cv2
    x = np.asarray(p, np.float32)
    valid = np.isfinite(x)
    if not valid.any():
        return p.astype(np.float64)

    lo, hi = np.nanpercentile(x[valid], [0.5, 99.5])
    rng = float(max(hi - lo, 1e-9))
    xn = np.ascontiguousarray(np.clip((x - lo) / rng, 0, 1).astype(np.float32))
    if not valid.all():
        xn[~valid] = np.float32(np.median(xn[valid]))

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
    out[~valid] = np.nan
    return out


def detrend(h, iters=3):
    """Remove the model's fake perspective tilt with a robust plane fit."""
    H, W = h.shape
    yy, xx = np.mgrid[0:H, 0:W]
    # Centre and scale the coordinates to [-1, 1].
    xs = (xx.ravel() - (W - 1) / 2.0) / max((W - 1) / 2.0, 1.0)
    ys = (yy.ravel() - (H - 1) / 2.0) / max((H - 1) / 2.0, 1.0)
    A = np.c_[xs, ys, np.ones(h.size)]
    z = h.ravel()
    valid = np.isfinite(z)
    Am, zm = A[valid], z[valid]
    if valid.sum() < 3:
        return h, np.zeros_like(h)
    w = np.ones(valid.sum())
    for _ in range(iters):
        coef, *_ = np.linalg.lstsq(Am * w[:, None], zm * w, rcond=None)
        if not np.all(np.isfinite(coef)):
            return h, np.zeros_like(h)
        r = zm - Am @ coef
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        w = 1.0 / np.sqrt(1 + (r / (2 * s)) ** 2)
    plane = (A @ coef).reshape(H, W)
    return h - plane, plane


SPLIT = os.environ.get("DEPTHWIZARD_SPLIT", "envelope").strip().lower()
GROUND_WINDOW_M = 80.0
GROUND_PCT = 2.0


def ground_envelope(p, px_size_m, window_m=GROUND_WINDOW_M, pct=GROUND_PCT,
                    step_m=2.0, max_cells=400):
    """The ground under the depth map: a robust opening, in the map's own units."""
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
    if up.shape != (H, W):                       # zoom rounding
        up = np.pad(up, ((0, max(0, H - up.shape[0])), (0, max(0, W - up.shape[1]))),
                    mode="edge")[:H, :W]
    return up


def ground_level(h, bins=256):
    v = h[np.isfinite(h)]
    if v.size == 0:
        return 0.0
    lo, hi = np.percentile(v, [0.5, 99.5])
    count, edges = np.histogram(v, bins=bins, range=(lo, hi))
    count = gaussian_filter(count.astype(float), 3)
    ctr = (edges[:-1] + edges[1:]) / 2
    pk = [i for i in range(1, len(count) - 1)
          if count[i] > count[i - 1] and count[i] >= count[i + 1] and count[i] > 0.2 * count.max()]
    return float(ctr[pk[0]]) if pk else float(np.percentile(v, 10))


def ground_consistency(h, grid=4, pct=15):
    """Reference-free self-check."""
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


def _dem_request_bounds(meta, pad_deg=0.01):
    from rasterio.warp import transform_bounds
    w, s_, e, n = transform_bounds(meta["crs"], "EPSG:4326", *meta["bounds"],
                                   densify_pts=21)
    return w - pad_deg, s_ - pad_deg, e + pad_deg, n + pad_deg


def _local_dem_covering(meta):
    import glob
    import rasterio
    from rasterio.warp import transform_bounds
    for d in DEM_DIRS:
        if not d or not os.path.isdir(d):
            continue
        for path in sorted(glob.glob(os.path.join(d, "*.tif")) +
                           glob.glob(os.path.join(d, "*.tiff"))):
            try:
                with rasterio.open(path) as ds:
                    if ds.crs is None:
                        continue
                    l, b, r, t = transform_bounds(meta["crs"], ds.crs,
                                                  *meta["bounds"], densify_pts=21)
                    B = ds.bounds
                    if l >= B.left and r <= B.right and b >= B.bottom and t <= B.top:
                        cached = os.path.abspath(d) == os.path.abspath(
                            os.path.join(CACHE_ROOT, "dem"))
                        return path, ("cache" if cached else "local file") + \
                            f": {os.path.basename(path)}"
            except Exception:
                continue
    return None, None


def fetch_dem(meta, out_tif="dem_coarse.tif", info=None):
    """The coarse terrain model for the scene footprint."""
    if meta["mode"] != "absolute":
        return None
    import rasterio
    rec = info if info is not None else {}

    local, label = _local_dem_covering(meta)
    if local:
        try:
            if os.path.abspath(local) != os.path.abspath(out_tif):
                shutil.copyfile(local, out_tif)
            rec["dem_source"] = label
            print(f"[dem] using {label} (no download)")
            return out_tif
        except Exception as ex:
            print(f"[dem] could not use {local}: {ex}")

    if not OPENTOPO_KEY:
        rec["dem_source"] = "none (no cached DEM covers the scene and OPENTOPO_KEY is empty)"
        print("[dem] FAILED: no cached DEM covers this scene and OPENTOPO_KEY is "
              "empty - see .env.example, or put a DEM GeoTIFF in DEM_DIR")
        return None
    import requests
    try:
        w, s_, e, n = _dem_request_bounds(meta)
        r = requests.get("https://portal.opentopography.org/API/globaldem",
                         params=dict(demtype=DEM_SOURCE, south=s_, north=n,
                                     west=w, east=e,
                                     outputFormat="GTiff", API_Key=OPENTOPO_KEY),
                         timeout=90)
        r.raise_for_status()
        with open(out_tif, "wb") as f:
            f.write(r.content)
        with rasterio.open(out_tif) as ds:
            ds.read(1)
        rec["dem_source"] = f"OpenTopography {DEM_SOURCE}"
        print(f"[dem] ok -> {out_tif}")
        try:
            cdir = os.path.join(CACHE_ROOT, "dem")
            os.makedirs(cdir, exist_ok=True)
            shutil.copyfile(out_tif, os.path.join(
                cdir, f"{DEM_SOURCE}_{w:.4f}_{s_:.4f}_{e:.4f}_{n:.4f}.tif"))
        except Exception as ex:
            print(f"[dem] cache write skipped: {ex}")
        return out_tif
    except Exception as ex:
        rec["dem_source"] = f"none (download failed: {type(ex).__name__})"
        print(f"[dem] FAILED: {ex}")
        return None


DEM_DEBIAS_WINDOW_M = 200.0
                                # Than real terrain features
DEM_DEBIAS_DEADBAND_M = 0.75
_dem_max = os.environ.get("DEPTHWIZARD_DEM_MAX_M", "5").strip().lower()
DEM_DEBIAS_MAX_M = None if _dem_max in ("none", "off", "") else float(_dem_max)


def debias_dem(dem, px_size_m, window_m=DEM_DEBIAS_WINDOW_M,
               deadband_m=DEM_DEBIAS_DEADBAND_M, max_m=DEM_DEBIAS_MAX_M):
    """Take the rooftops out of the free 30 m DEM, so it describes the ground."""
    px = max(float(px_size_m or 1.0), 1e-6)
    k = max(3, int(round(window_m / px)) | 1)
    # Opening = erosion then dilation.
    ground = maximum_filter(minimum_filter(dem, size=k, mode="nearest"),
                            size=k, mode="nearest")
    ground = gaussian_filter(ground, max(1.0, k / 6.0))
    correction = np.maximum(dem - ground - float(deadband_m), 0.0)
    if max_m is not None:
        # Past a few metres it is relief, not rooftops - see DEM_DEBIAS_MAX_M
        correction = np.minimum(correction, float(max_m))
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


HEIGHT_REFERENCE_PCT = {"tallest": 99.0, "tall": 95.0, "typical": 60.0}

PRIOR_GAIN = float(os.environ.get("DEPTHWIZARD_PRIOR_GAIN",
                                   "0.80" if SPLIT == "gauss" else "1.0"))
GHSL_PRIOR_GAIN = float(os.environ.get("DEPTHWIZARD_GHSL_GAIN", "1.0"))


def prior_gain(reference="tallest", source="person"):
    """The measured correction for a landmark prior, 1.0 where none is measured."""
    if str(source).lower() == "ghsl":
        return GHSL_PRIOR_GAIN
    return PRIOR_GAIN if str(reference).lower() == "tallest" else 1.0


def ghsl_height_prior(meta, allow_download=True):
    """The automatic landmark height for a georeferenced scene - see ghsl.py."""
    try:
        import ghsl as GH
        return GH.height_prior(meta, CACHE_ROOT, allow_download=allow_download)
    except Exception as ex:                       # never break a run over a prior
        return None, f"GHSL lookup failed ({type(ex).__name__}: {str(ex)[:120]})"

PRIOR_SIGMA_REL   = 0.25
GCP_MIN_SIGMA_REL = 0.05
# Sun angles a person estimated by eye, rather than read from the file's tags.
SUN_GUESS_SIGMA_DEG = 5.0
FUSION_MAX_Z = 3.0


def alpha_from_known_height(detail, known_height_m, pct=None, reference="tallest"):
    """Scale from one human guess: "the tallest block here is about 40 m"."""
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
    """Scale from points whose height above the ground you already know."""
    fail = (None, None) if with_sigma else None
    if not gcps or len(gcps) < 2:
        return fail
    g = ground_level(detail)
    x = np.array([detail[int(r), int(c)] - g for r, c, _ in gcps], float)
    y = np.array([h for _, _, h in gcps], float)
    valid = np.isfinite(x) & np.isfinite(y) & (np.abs(x) > 1e-9)
    if valid.sum() < 2:
        return fail
    xs, ys = x[valid], y[valid]
    n = int(valid.sum())

    sxx = float(np.sum(xs * xs))
    if sxx < 1e-12:
        return fail
    alpha = float(np.sum(xs * ys) / sxx)

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
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    n = X.size
    if n < 10:
        return dict(n=int(n), note="too few control points to cross-validate")

    idx = np.random.default_rng(seed).permutation(n)
    result, truth = [], []
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        if tr.size < 4 or te.size < 1:
            continue
        a = float(np.median(Y[tr] / X[tr]))
        result.append(a * X[te] - Y[te])
        # Keep each fold's own truth beside its residuals.
        truth.append(Y[te])
    if not result:
        return dict(n=int(n), note="cross-validation folds too small")

    e = np.concatenate(result)
    y_te = np.concatenate(truth)
    ratios = Y / X
    q1, q3 = np.percentile(ratios, [25, 75])
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


ROOF_PROBE_M = (1.0, 2.0, 4.0, 7.0, 11.0)


def alpha_from_shadows(detail, rgb, sun_az_deg, sun_elev_deg, px_size_m,
                       sun_sigma_deg=0.0):
    """Scale from the shadows the buildings cast."""
    import cv2
    a = np.deg2rad(sun_az_deg)
    dc, dr = -np.sin(a), np.cos(a)
    # ...and back towards the sun, where the object that cast it stands.
    scale, sr = -dc, -dr

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    v_chan = (hsv[..., 2] / 255.0 * 255).astype(np.uint8)

    _, dark_mask = cv2.threshold(v_chan, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    flat = (hsv[..., 1] / 255. < 0.40)
    mask = cv2.morphologyEx((dark_mask.astype(bool) & flat).astype(np.uint8),
                            cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    H, W = mask.shape
    px = max(float(px_size_m or 1.0), 1e-6)
    probe_px = sorted({max(1, int(round(m / px))) for m in ROOF_PROBE_M})

    n, labelled = cv2.connectedComponents(mask)
    g = ground_level(detail)
    X, Y = [], []
    drop = dict(speck=0, thin=0, length=0, truncated=0, no_roof=0, flat=0)
    for k in range(1, n):
        ys, xs = np.where(labelled == k)
        if ys.size < 6:
            drop["speck"] += 1
            continue

        if xs.min() == 0 or ys.min() == 0 or xs.max() == W - 1 or ys.max() == H - 1:
            drop["truncated"] += 1
            continue

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

        er, ec = np.array(er), np.array(ec)
        roof = []
        for d in probe_px:
            rr = np.clip(np.round(er + sr * d).astype(int), 0, H - 1)
            cc = np.clip(np.round(ec + scale * d).astype(int), 0, W - 1)
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
    # Median ratio is robust enough here and needs no sklearn
    a_est = float(np.median(np.array(Y) / np.array(X)))
    diag = dict(_cv_ratio(X, Y), rejected=drop)

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
                      sun_azimuth=None, sun_elevation=None, sun_sigma_deg=0.0,
                      prior_source="person"):
    """Run every available scale source and report whether they agree."""
    est, notes, sigma = {}, {}, {}
    if sun_azimuth is not None and sun_elevation is not None:
        try:
            a, diag = alpha_from_shadows(detail, rgb, sun_azimuth, sun_elevation,
                                         px_size_m, sun_sigma_deg=sun_sigma_deg)
            if a and a > 0:
                est["shadow"] = float(a)
                notes["shadow"] = diag
                sigma["shadow"] = float(diag.get("alpha_sigma_rel", 0.20))
        except Exception as e:
            notes["shadow"] = {"error": f"{type(e).__name__}: {e}"}
    if gcps and len(gcps) >= 2:
        a, s = alpha_from_gcps(detail, gcps, with_sigma=True)
        if a and a > 0:
            est["gcps"] = float(a)
            notes["gcps"] = {"n": len(gcps), "alpha_sigma_rel": s}
            sigma["gcps"] = float(s)
    if known_height_m:
        est["prior"] = float(alpha_from_known_height(detail, known_height_m)
                             * prior_gain("tallest", prior_source))
        notes["prior"] = {"known_height_m": float(known_height_m),
                          "alpha_sigma_rel": PRIOR_SIGMA_REL}
        sigma["prior"] = PRIOR_SIGMA_REL

    out = dict(estimates=est, sigma_rel=sigma, detail=notes, n_sources=len(est))
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
    """Combine the independent calibrators into one alpha, by how much each is worth."""
    est = (cross_check or {}).get("estimates") or {}
    sigma = (cross_check or {}).get("sigma_rel") or {}
    usable = {k: (float(est[k]), float(sigma.get(k, 0.25)))
              for k in est if np.isfinite(est.get(k, np.nan)) and est[k] > 0}

    if len(usable) < 2:
        return dict(applied=False, alpha=None, n_sources=len(usable),
                    reason="fewer than two usable sources - nothing to fuse")

    names = sorted(usable)
    a = np.array([usable[k][0] for k in names], float)
    s = np.array([max(usable[k][1], 1e-3) for k in names], float)

    # The guard, pairwise, in the same log space the fusion uses
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
    """One reference-free statement of how much to trust this surface."""
    report = {}
    fin = np.isfinite(spread) & (spread > 0)
    if fin.any():
        rng = float(np.nanmax(height) - np.nanmin(height))
        report["ensemble_spread_median"] = float(np.median(spread[fin]))
        report["ensemble_spread_p90"] = float(np.percentile(spread[fin], 90))
        report["ensemble_spread_pct_of_range"] = (
            100.0 * report["ensemble_spread_median"] / max(rng, 1e-9))
        if alpha:
            report["ensemble_spread_median_m"] = float(alpha * report["ensemble_spread_median"])
    report["ground_levelness"] = float(ground_spread)
    if cross_check.get("agreement_pct") is not None:
        report["scale_agreement_pct"] = float(cross_check["agreement_pct"])
        report["scale_sources"] = list(cross_check["estimates"])
    return report


# 6. THE PIPELINE


def resolve_engine(engine=None):
    """One of engines for this call - see ENGINE_ENV above."""
    e = (engine or os.environ.get(ENGINE_ENV, "auto")).strip().lower()
    e = _ENGINE_ALIASES.get(e, e)
    if e == "zeroshot":
        return "zeroshot"
    if e not in ENGINES + ("auto",):
        raise ValueError(f"unknown engine {e!r} - expected one of {ENGINES} or auto")
    try:
        import heightnet as HN
        ok = HN.available()
    except Exception:
        HN, ok = None, None
    if e != "auto":
        if not ok:
            where = HN.default_checkpoint() if HN else "models/heightnet.pt"
            raise ValueError(f"engine {e!r} needs the trained model, but there is no "
                             f"usable checkpoint at {where} (it needs torch and the "
                             f".pt file from training/)")
        return e
    return "hybrid" if ok else "zeroshot"


_hn = None


def get_heightnet():
    """The loaded predictor, reused across runs; reloaded if the file changes."""
    global _hn
    import heightnet as HN
    path = HN.default_checkpoint()
    stamp = (path, os.path.getmtime(path))
    if _hn is None or getattr(_hn, "_stamp", None) != stamp:
        dev = DEPTH_DEVICE if DEPTH_DEVICE in ("cpu", "cuda", "mps") else "auto"
        _hn = HN.HeightPredictor(path, device=dev)
        _hn._stamp = stamp
        print(f"[heightnet] loaded {_hn.describe()} on {_hn.device}")
    return _hn


def _heightnet_predict(rgb, gsd, rotations, cache_key=None):
    predictor = get_heightnet()
    path = None
    if DEPTH_CACHE:
        h = hashlib.sha1()
        h.update((cache_key or hashlib.sha1(np.ascontiguousarray(rgb).tobytes())
                  .hexdigest()).encode())
        h.update(f"v2|{predictor.digest}|{gsd}|{rotations}|{rgb.shape}".encode())
        d = os.path.join(CACHE_ROOT, "heightnet")
        path = os.path.join(d, h.hexdigest()[:24] + ".npz")
        if os.path.exists(path):
            try:
                z = np.load(path)
                print(f"[heightnet] cached prediction {os.path.basename(path)}")
                return dict(height=z["height"], sigma=z["sigma"].astype(np.float32),
                            seg=z["seg"], rel=z["rel"],
                            rel_sigma=z["rel_sigma"].astype(np.float32),
                            gsd_model=float(z["gsd_model"]), tiles=0,
                            rotations=int(rotations)), predictor
            except Exception:
                pass
    t0 = time.time()
    try:
        r = predictor.predict(rgb, gsd, rotations=rotations)
    except RuntimeError as e:
        if predictor.device.type == "mps":                  # an op MPS lacks: redo on CPU
            print(f"[heightnet] MPS failed ({str(e)[:80]}), retrying on CPU")
            import torch
            predictor.device = torch.device("cpu")
            predictor.model.to(predictor.device)
            r = predictor.predict(rgb, gsd, rotations=rotations)
        else:
            raise
    print(f"[heightnet] {r['tiles']} tile passes, {r['rotations']} rotations, "
          f"model GSD {r['gsd_model']:.2f} m, {time.time() - t0:.1f}s on {predictor.device}")
    if path:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez_compressed(path, height=r["height"].astype(np.float32),
                                sigma=r["sigma"].astype(np.float16), seg=r["seg"],
                                rel=r["rel"].astype(np.float32),
                                rel_sigma=r["rel_sigma"].astype(np.float16),
                                gsd_model=r["gsd_model"])
        except Exception as e:
            print(f"[heightnet] cache write skipped: {e}")
    return r, predictor


def _export_landcover(seg, meta, outdir, info):
    import heightnet as HN
    Image.fromarray(seg.astype(np.uint8)).save(os.path.join(outdir, "landcover.png"))
    if meta["mode"] == "absolute":
        import rasterio
        prof = dict(driver="GTiff", height=seg.shape[0], width=seg.shape[1], count=1,
                    dtype="uint8", nodata=255, compress="deflate",
                    crs=meta["crs"], transform=meta["transform"])
        with rasterio.open(os.path.join(outdir, "landcover.tif"), "w", **prof) as ds:
            ds.write(seg.astype(np.uint8), 1)
            ds.update_tags(PRODUCT="LANDCOVER",
                           CLASSES=",".join(f"{i}={c}" for i, c in enumerate(HN.CLASSES)))
    n = max(seg.size, 1)
    info["landcover"] = {c: round(float((seg == i).sum()) / n, 4)
                         for i, c in enumerate(HN.CLASSES) if (seg == i).any()}


def _heightnet_gcp_gain(agl, gcps, min_h=0.5):
    H, W = agl.shape
    x, y = [], []
    for r, c, h in gcps:
        r, c = int(min(max(r, 0), H - 1)), int(min(max(c, 0), W - 1))
        x.append(agl[r, c])
        y.append(h)
    x, y = np.array(x, float), np.array(y, float)
    valid = np.isfinite(x) & np.isfinite(y) & (x > min_h)
    if valid.sum() < 2:
        print(f"[gcp] only {int(valid.sum())} control points sit on something the "
              f"model sees above {min_h} m - kept the model's own metres")
        return None
    a = float((x[valid] * y[valid]).sum() / (x[valid] ** 2).sum())
    return a if a > 0 else None


def _estimate_heightnet(path, rgb, meta, info, key, known_height_m=None, gcps=None,
                        sun_azimuth=None, sun_elevation=None, use_dem=True,
                        alpha_gain=1.0, outdir="outputs", rotations=ROTATIONS,
                        debias_coarse_dem=True, sun_from_tags=False,
                        prior_source="person", auto_prior=True, prior_info=None,
                        gsd_hint=None, height_reference="tallest"):
    absolute = meta["mode"] == "absolute"
    px = meta["px_size_m"] if absolute else (float(gsd_hint) if gsd_hint else None)
    r, hp = _heightnet_predict(rgb, px, rotations, cache_key=key)
    agl = r["height"].astype(np.float64)
    sigma = r["sigma"].astype(np.float64)
    vm = meta.get("valid_mask")
    if vm is not None and vm.shape == agl.shape:
        agl = np.where(vm, agl, np.nan)        # cloud / nodata: no claim made
    info.update(engine="metric", model=hp.describe(),
                model_train=hp.train_info.get("val") or {},
                model_gsd_m=float(r["gsd_model"]),
                split="none - the model predicts height above ground directly",
                sigma_m=15.0, ramp_share=0.0)
    ground = ground_consistency(np.nan_to_num(agl))
    info.update(ground_spread_before=ground, ground_spread_after=ground)
    _export_landcover(r["seg"], meta, outdir, info)

    if not absolute:
        full = float(np.nanpercentile(agl, 99.8)) if np.isfinite(agl).any() else 1.0
        full = max(full, 1.0)
        height = np.clip(agl / full, 0, 1)
        uncertainty = sigma / full
        info["note"] = ("relative rDSM from HeightNet (no georeference: metres "
                        f"assumed at {r['gsd_model']:.2f} m/px)")
        info["suggested_full_scale_m"] = full
        info["scale_source"] = None
        info["confidence"] = confidence_report(sigma, agl, meta, ground, dict(estimates={}),
                                               alpha=1.0)
        _export(height, rgb, meta, outdir, info, uncertainty=uncertainty)
        return height, dict(meta, _uncertainty=uncertainty), info

    alpha = 1.0
    info["scale_source"] = "HeightNet - metres learned from LiDAR, no anchor needed"
    if gcps:
        a = _heightnet_gcp_gain(agl, gcps)
        if a and a > 0:
            alpha = float(a)
            info["scale_source"] = (f"HeightNet corrected by {len(gcps)} ground control "
                                    f"points (x{alpha:.2f})")
            print(f"[gcp] HeightNet heights x{alpha:.3f} from {len(gcps)} control points")
    if alpha_gain and abs(alpha_gain - 1.0) > 1e-6:
        alpha *= float(alpha_gain)
        info["alpha_gain"] = float(alpha_gain)
    info["alpha"] = float(alpha)

    # Every other anchor becomes an independent check of the model's metres
    if sun_azimuth is None and sun_elevation is None:
        sun_azimuth, sun_elevation = meta.get("sun_azimuth"), meta.get("sun_elevation")
        sun_from_tags = sun_azimuth is not None and sun_elevation is not None
    if not known_height_m and auto_prior:
        try:
            prior, why = ghsl_height_prior(meta)
        except Exception as e:
            prior, why = None, f"{type(e).__name__}: {e}"
        if prior is not None:
            known_height_m, prior_source, prior_info = prior["known_height_m"], "ghsl", prior
        else:
            info["auto_prior_unavailable"] = why
    if prior_info:
        info["auto_prior"] = prior_info
    cc = cross_check_scale(np.nan_to_num(agl), rgb, px, known_height_m=known_height_m,
                           gcps=gcps, sun_azimuth=sun_azimuth, sun_elevation=sun_elevation,
                           sun_sigma_deg=0.0 if sun_from_tags else SUN_GUESS_SIGMA_DEG,
                           prior_source=prior_source)
    info["cross_check"] = cc
    if cc.get("estimates"):
        info["model_vs_anchors"] = {k: round(float(v), 3) for k, v in cc["estimates"].items()}
        print("[heightnet] independent anchors vs the model's metres: " + ", ".join(
            f"{k} x{v:.2f}" for k, v in cc["estimates"].items()) + "  (x1.00 = agree)")

    dem_path = (fetch_dem(meta, os.path.join(outdir, "dem_coarse.tif"), info)
                if use_dem else None)
    if dem_path:
        terrain = dem_on_grid(dem_path, meta, agl.shape)
        terrain = np.where(np.isfinite(terrain), terrain, np.nanmedian(terrain))
        if debias_coarse_dem:
            terrain, shift = debias_dem(terrain, px)
            info["dem_debias_m"] = shift
            print(f"[dem] rooftop contamination removed: terrain lowered by "
                  f"{shift:.2f} m (median)")
        info["calibration"] = "HeightNet (DEM terrain + predicted height above ground)"
    else:
        terrain = np.zeros_like(agl)
        info["calibration"] = "HeightNet, no DEM (heights above local ground)"
        print("[calib] no DEM - output is height above ground, not above sea level")

    ndsm = np.maximum(alpha * agl, 0.0)
    height = terrain + ndsm
    dtm = height - ndsm
    uncertainty = alpha * sigma
    print(f"[calib] {info['calibration']}  range {np.nanmin(height):.1f}.."
          f"{np.nanmax(height):.1f} m, objects up to {np.nanpercentile(ndsm, 99.5):.1f} m")
    info["confidence"] = confidence_report(uncertainty, height, meta, ground, cc, alpha=1.0)
    info["ndsm_p99_m"] = float(np.nanpercentile(ndsm, 99))
    info["ndsm_mean_m"] = float(np.nanmean(ndsm))
    _export(height, rgb, meta, outdir, info, ndsm=ndsm, dtm=dtm, uncertainty=uncertainty)
    return height, dict(meta, _uncertainty=uncertainty), info


def _hybrid_relative(rgb, meta, info, outdir, r, hp, gsd):
    import heightnet as HN
    c = HN.combine(r, gsd, fusion=hp.fusion)
    h = np.asarray(c["fused"], np.float64)
    full = max(float(np.nanpercentile(h, 99.8)) if np.isfinite(h).any() else 1.0, 1.0)
    height = np.clip(h / full, 0, 1)
    uncertainty = np.asarray(c["fused_sigma"], np.float64) / full
    info.update(note=("relative rDSM from the hybrid engine (no georeference: metres "
                      f"assumed at {r['gsd_model']:.2f} m/px)"),
                suggested_full_scale_m=full, scale_source=None,
                learned_alpha=c["alpha"], fusion_weight_b_mean=float(np.mean(c["w_b"])))
    info["confidence"] = confidence_report(np.asarray(c["fused_sigma"]), h, meta,
                                           info.get("ground_spread_before", 0.0),
                                           dict(estimates={}), alpha=1.0)
    _export(height, rgb, meta, outdir, info, uncertainty=uncertainty)
    return height, dict(meta, _uncertainty=uncertainty), info


def estimate_elevation(path, known_height_m=None, gcps=None,
                       sun_azimuth=None, sun_elevation=None,
                       use_dem=True, alpha_gain=1.0, outdir="outputs",
                       rotations=ROTATIONS, adaptive_sigma=True,
                       height_reference="tallest", debias_coarse_dem=True,
                       fuse_scale=False, sun_from_tags=False,
                       prior_source="person", auto_prior=True, prior_info=None,
                       engine=None, gsd_hint=None):
    """The MAIN function."""
    os.makedirs(outdir, exist_ok=True)
    rgb, meta = load_image(path)
    print(f"[load] {rgb.shape[1]}x{rgb.shape[0]}  mode={meta['mode']}  px={meta['px_size_m']}")

    info = dict(mode=meta["mode"], rotations=int(max(1, min(4, rotations))))
    try:
        key = source_key(path)
    except Exception:
        key = None
    eng = resolve_engine(engine)
    if eng == "metric":
        return _estimate_heightnet(
            path, rgb, meta, info, key, known_height_m=known_height_m, gcps=gcps,
            sun_azimuth=sun_azimuth, sun_elevation=sun_elevation, use_dem=use_dem,
            alpha_gain=alpha_gain, outdir=outdir, rotations=rotations,
            debias_coarse_dem=debias_coarse_dem, sun_from_tags=sun_from_tags,
            prior_source=prior_source, auto_prior=auto_prior, prior_info=prior_info,
            gsd_hint=gsd_hint, height_reference=height_reference)
    info["engine"] = eng
    hn_pred = hn = None
    if eng in ("finetuned", "hybrid"):
        gsd_run = meta["px_size_m"] if meta["mode"] == "absolute" else (
            float(gsd_hint) if gsd_hint else None)
        hn_pred, hn = _heightnet_predict(rgb, gsd_run, rotations, cache_key=key)
        p = hn_pred["rel"].astype(np.float64)
        spread = hn_pred["rel_sigma"].astype(np.float64)
        import heightnet as HN
        if not HN.relative_informative(p):
            if eng == "finetuned":
                raise ValueError("the fine-tuned model's relative depth is flat on this "
                                 "image - the checkpoint looks broken or untrained. Use "
                                 "engine zeroshot or metric.")
            print("[engine] relative output is flat - the hybrid falls back to engine B")
            info["hybrid_fallback"] = "relative output flat: engine B only"
            return _estimate_heightnet(
                path, rgb, meta, info, key, known_height_m=known_height_m, gcps=gcps,
                sun_azimuth=sun_azimuth, sun_elevation=sun_elevation, use_dem=use_dem,
                alpha_gain=alpha_gain, outdir=outdir, rotations=rotations,
                debias_coarse_dem=debias_coarse_dem, sun_from_tags=sun_from_tags,
                prior_source=prior_source, auto_prior=auto_prior, prior_info=prior_info,
                gsd_hint=gsd_hint, height_reference=height_reference)
        info.update(model=hn.describe(), depth_inverted=False,
                    depth_finite_frac=float(np.isfinite(p).mean()))
        print(f"[engine] {ENGINE_LABEL[eng]}")
        _export_landcover(hn_pred["seg"], meta, outdir, info)
    else:
        p, spread = predict_depth_ensemble(rgb, rotations=rotations, cache_key=key)
        p = validate_depth(p, rgb, info)
    p = clean_depth(p, rgb)

    will_use_dem = (meta["mode"] == "absolute" and use_dem)
    before = ground_consistency(p)
    ensembled = int(max(1, min(4, rotations))) >= 2
    if ensembled:
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

    # RELATIVE
    if meta["mode"] != "absolute" and eng == "hybrid":
        return _hybrid_relative(rgb, meta, info, outdir, hn_pred, hn,
                                float(gsd_hint) if gsd_hint else None)
    if meta["mode"] != "absolute":
        lo, hi = np.nanpercentile(p, [0.5, 99.8])
        height = np.clip((p - lo) / max(hi - lo, 1e-9), 0, 1)
        info["note"] = "relative rDSM (no metric scale)"
        # Spread is in the same relative units the surface was normalised into
        uncertainty = spread / max(hi - lo, 1e-9) if np.any(spread) else None
        info["confidence"] = confidence_report(
            spread, height, meta, after, dict(estimates={}))
        _export(height, rgb, meta, outdir, info, uncertainty=uncertainty)
        # Private, so a caller that rescales the surface can rescale this too
        meta = dict(meta, _uncertainty=uncertainty)
        return height, meta, info

    # ABSOLUTE
    px = meta["px_size_m"] or 1.0
    sigma_m = (structure_scale_m(p, px) if adaptive_sigma else DEM_RES_M / 2)
    sigma = max(2.0, sigma_m / px)
    p_fill = _fill_nan(p)
    if SPLIT == "gauss":
        detail = p_fill - gaussian_filter(p_fill, sigma)
    else:
        # Height above a ground-hugging envelope: ground ~0, no water trap
        detail = p_fill - ground_envelope(p_fill, px)
    detail = np.where(np.isfinite(p), detail, np.nan)   # buildings live here
    info["split"] = "gauss" if SPLIT == "gauss" else (
        f"ground envelope ({GROUND_WINDOW_M:g} m, p{GROUND_PCT:g})")
    info["sigma_px"] = float(sigma)
    info["sigma_m"] = float(sigma_m)
    info["sigma_source"] = "scene structures" if adaptive_sigma else "DEM resolution"

    alpha = None
    tried_shadows = False
    ghsl_reason = None
    info["scale_source"] = None
    if (sun_azimuth is None and sun_elevation is None
            and not gcps and not known_height_m):
        # Last resort only - an explicit choice by the caller always wins
        sun_azimuth, sun_elevation = meta.get("sun_azimuth"), meta.get("sun_elevation")
        if sun_azimuth is not None and sun_elevation is not None:
            sun_from_tags = True
    sun_sigma_deg = 0.0 if sun_from_tags else SUN_GUESS_SIGMA_DEG
    if sun_azimuth is not None and sun_elevation is not None:
        tried_shadows = True
        alpha, sdiag = alpha_from_shadows(detail, rgb, sun_azimuth, sun_elevation,
                                          px, sun_sigma_deg=sun_sigma_deg)
        info["self_check"] = sdiag
        info["sun_from_tags"] = bool(sun_from_tags)
        if alpha is not None:
            info["scale_source"] = "shadows (sun angles from the file)" \
                if sun_from_tags else "shadows (sun angles given)"
    if alpha is None and gcps:
        alpha = alpha_from_gcps(detail, gcps)
        if alpha is not None:
            print(f"[gcp] alpha={alpha:.3f} from {len(gcps)} control points")
            info["scale_source"] = f"{len(gcps)} ground control points"
    if (alpha is None and not known_height_m and not gcps and eng == "hybrid"):
        import heightnet as HN
        a_l = HN.learned_scale(detail, hn_pred["height"], np.isfinite(detail))
        if a_l is not None:
            alpha = a_l
            info["scale_source"] = "learned scene prior (HeightNet metric head)"
            print(f"[prior] alpha={alpha:.3f} from the learned scene prior")
    if (alpha is None and not known_height_m and not gcps and auto_prior):
        prior, ghsl_reason = ghsl_height_prior(meta)
        if prior is not None:
            known_height_m = prior["known_height_m"]
            prior_source = "ghsl"
            height_reference = "tallest"
            info["auto_prior"] = prior
            print(f"[ghsl] no scale given - GHSL building height under this "
                  f"scene: {known_height_m:g} m ({prior['n_built']} of "
                  f"{prior['n_cells']} 100 m cells built)")
        else:
            print(f"[ghsl] no automatic prior: {ghsl_reason}")
            info["auto_prior_unavailable"] = ghsl_reason
    if alpha is None and known_height_m:
        alpha = alpha_from_known_height(detail, known_height_m,
                                        reference=height_reference)
        info["height_reference"] = str(height_reference)
        info["prior_source"] = str(prior_source)
        if prior_info and "auto_prior" not in info:
            info["auto_prior"] = prior_info
        pg = prior_gain(height_reference, prior_source)
        print(f"[prior] alpha={alpha:.2f} from {height_reference} structure "
              f"= {known_height_m} m ({prior_source})")
        if abs(pg - 1.0) > 1e-6:
            alpha *= pg
            print(f"[prior] alpha x{pg:.2f} -> {alpha:.2f} (measured correction "
                  f"for a tallest-structure prior; DEPTHWIZARD_PRIOR_GAIN=1 "
                  f"turns it off)")
        info["prior_gain"] = float(pg)
        info["scale_source"] = (
            f"GHSL building height ({known_height_m:g} m) - automatic prior"
            if str(prior_source) == "ghsl" else
            f"{height_reference} structure = {known_height_m:g} m (given)")
    if alpha is not None and alpha_gain and abs(alpha_gain - 1.0) > 1e-6:
        alpha *= float(alpha_gain)
        info["alpha_gain"] = float(alpha_gain)
        print(f"[calib] alpha x{alpha_gain:.2f} (measured attenuation correction)")
    if alpha is None and eng == "hybrid":
        print("[prior] nothing above ground to calibrate the relative depth on - "
              "using the HeightNet metres directly (engine B)")
        info["hybrid_fallback"] = "no scale for engine A in this scene: engine B only"
        return _estimate_heightnet(
            path, rgb, meta, info, key, known_height_m=known_height_m, gcps=gcps,
            sun_azimuth=sun_azimuth, sun_elevation=sun_elevation, use_dem=use_dem,
            alpha_gain=alpha_gain, outdir=outdir, rotations=rotations,
            debias_coarse_dem=debias_coarse_dem, sun_from_tags=sun_from_tags,
            prior_source=prior_source, auto_prior=False, prior_info=prior_info,
            gsd_hint=gsd_hint, height_reference=height_reference)
    if alpha is None and tried_shadows:
        # Do not tell someone who supplied sun angles to supply sun angles.
        n_pairs = (info.get("self_check") or {}).get("n", 0)
        raise ValueError(
            f"Shadow calibration found only {n_pairs} usable shadow/roof pairs "
            "in this scene, which is not enough to set the metre scale. That "
            "happens on hazy imagery, on a high sun that casts almost nothing, "
            "and where shadows fall on other buildings rather than on open "
            "ground. Pass known_height_m (the height of the tallest structure "
            "you can identify) or at least two gcps instead."
            + (f" (No automatic GHSL prior either: {ghsl_reason}.)"
               if ghsl_reason else ""))
    if alpha is None:
        raise ValueError(
            "No scale source. The coarse DEM supplies the terrain baseline but "
            "not this multiplier - the model's low-frequency band carries a "
            "large fake ramp, exactly where a 30 m DEM is blind. Pass "
            "known_height_m, gcps, or sun_azimuth + sun_elevation."
            + (f" (No automatic GHSL prior either: {ghsl_reason}.)"
               if ghsl_reason else ""))
    info["alpha"] = float(alpha)

    # Every calibrator that could have answered, run and compared.
    info["cross_check"] = cross_check_scale(
        detail, rgb, px, known_height_m=known_height_m, gcps=gcps,
        sun_azimuth=sun_azimuth, sun_elevation=sun_elevation,
        sun_sigma_deg=sun_sigma_deg, prior_source=prior_source)

    # ...unless the caller asked for the sources to be combined rather than ranked.
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

    dem_path = (fetch_dem(meta, os.path.join(outdir, "dem_coarse.tif"), info)
                if use_dem else None)
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

    if SPLIT == "gauss":
        detail = detail - ground_level(detail)
    height = terrain + alpha * detail
    if dem_path is None:
        neg = float((height < 0).mean())
        info["clipped_negative_frac"] = neg
        if neg > 0.02:
            print(f"[calib] {100*neg:.1f}% of pixels below ground - clipped. "
                  f"High values mean the ramp was not fully removed.")
        height = np.maximum(height, 0.0)
    print(f"[calib] {info['calibration']}  alpha={alpha:.2f}  "
          f"range {np.nanmin(height):.1f}..{np.nanmax(height):.1f} m")

    if eng == "hybrid":
        import heightnet as HN
        b_h = np.asarray(hn_pred["height"], np.float64)
        b_s = np.asarray(hn_pred["sigma"], np.float64)
        if gcps:
            gb = _heightnet_gcp_gain(b_h, gcps)
            if gb:
                b_h, b_s = b_h * gb, b_s * gb
                info["metric_gcp_gain"] = float(gb)
        a_h = np.maximum(alpha * detail, 0.0)
        a_s = alpha * spread if np.any(spread) else np.full_like(a_h, 1.0)
        fused, fsig, wb = HN.fuse(a_h, a_s, b_h, b_s, fusion=hn.fusion)
        fused = np.where(np.isfinite(fused), fused, b_h)
        height = terrain + fused
        spread = fsig / max(alpha, 1e-9)
        b_alpha = HN.learned_scale(detail, b_h, np.isfinite(detail))
        info.update(fusion_weight_b_mean=float(np.nanmean(wb)),
                    fusion_calibration=dict(hn.fusion) or None)
        if b_alpha is not None and not str(info.get("scale_source", "")).startswith("learned"):
            info["model_vs_anchors"] = {"metric head": round(float(b_alpha / alpha), 3)}
        print(f"[fusion] A and B blended per pixel, mean weight on B "
              f"{info['fusion_weight_b_mean']:.2f}")
        info["calibration"] = (info.get("calibration", "") +
                               " + fused with HeightNet metres")

    ndsm = height - terrain
    ndsm = np.maximum(ndsm, 0.0)
    dtm = height - ndsm
    uncertainty = (alpha * spread) if np.any(spread) else None
    info["confidence"] = confidence_report(
        spread, height, meta, after, info.get("cross_check", {}), alpha=alpha)
    info["ndsm_p99_m"] = float(np.nanpercentile(ndsm, 99))
    info["ndsm_mean_m"] = float(np.nanmean(ndsm))

    _export(height, rgb, meta, outdir, info, ndsm=ndsm, dtm=dtm, uncertainty=uncertainty)
    meta = dict(meta, _uncertainty=uncertainty)
    return height, meta, info


def clip_below_ground(height, info):
    """Re-apply the no-DEM ground clip after refine()."""
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
    """Re-write the rasters from a surface that changed after estimate_elevation."""
    info = dict(info)
    info.pop("products", None)

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
    import rasterio
    h = np.asarray(height, np.float32)
    prof = dict(driver="GTiff", height=h.shape[0], width=h.shape[1], count=1,
                dtype="float32", nodata=np.nan, compress="deflate")
    if meta["mode"] == "absolute":
        prof.update(crs=meta["crs"], transform=meta["transform"])

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


def measure_height(height, meta, row, col):
    """Click-to-read for the UI."""
    v = float(height[int(row), int(col)])
    return v, ("m" if meta["mode"] == "absolute" else "relative")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python inference.py <image> [known_height_m]")
        raise SystemExit(1)
    kh = float(sys.argv[2]) if len(sys.argv) > 2 else None
    estimate_elevation(sys.argv[1], known_height_m=kh)
