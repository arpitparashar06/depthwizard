"""Colour layers for the 3D viewer, and exact readouts off the rasters."""
import json
import os
from functools import lru_cache

import numpy as np
from PIL import Image

RAMPS = {
    "blue":   "#cde2fb #b7d3f6 #9ec5f4 #86b6ef #6da7ec #5598e7 #3987e5 #2a78d6 "
              "#256abf #1c5cab #184f95 #104281 #0d366b",
    "orange": "#fbd7ca #f5c4b2 #f2b098 #eb9c7f #e68764 #df7249 #d95923 #c94908 "
              "#b44005 #9f3600 #8a2e00 #752600 #611e00",
    "violet": "#dcddf8 #cbcdf2 #bbbcef #ababe9 #9b9be4 #8c8ade #7d78da #706acb "
              "#635db5 #5650a2 #4a458d #3e397a #332f65",
    "red":    "#fed4d0 #fac0ba #f8aaa3 #f2958e #ee7e77 #e76762 #e24a48 #d2383a "
              "#bb3032 #a72528 #911e22 #7e1519 #681014",
}
RAMPS = {k: v.split() for k, v in RAMPS.items()}
NEUTRAL = "#f0efec"
# Estimate too low -> blue arm, too high -> red arm.
_ARM = (2, 3, 5, 7, 9, 10)
DIVERGING = ([RAMPS["blue"][i] for i in reversed(_ARM)] + [NEUTRAL]
             + [RAMPS["red"][i] for i in _ARM])

MAX_TEXTURE = 4096


def _hex(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.float64)


def colorize(a, stops, vmin, vmax):
    """Float array -> uint8 RGB through a piecewise-linear ramp."""
    a = np.asarray(a, np.float64)
    cols = np.stack([_hex(h) for h in stops])
    t = (a - vmin) / max(vmax - vmin, 1e-9)
    t = np.clip(np.where(np.isfinite(t), t, 0.5), 0, 1) * (len(stops) - 1)
    i = np.minimum(t.astype(int), len(stops) - 2)
    f = (t - i)[..., None]
    out = cols[i] * (1 - f) + cols[i + 1] * f
    out[~np.isfinite(a)] = _hex("#9a9aa3")
    return np.clip(out, 0, 255).astype(np.uint8)


def _save(rgb, path):
    im = Image.fromarray(rgb)
    if max(im.size) > MAX_TEXTURE:
        im.thumbnail((MAX_TEXTURE, MAX_TEXTURE), Image.LANCZOS)
    im.save(path)


def _read(path):
    if not os.path.exists(path):
        return None, None
    return _read_cached(path, os.path.getmtime(path))


@lru_cache(maxsize=24)
def _read_cached(path, _mtime):
    import rasterio
    with rasterio.open(path) as ds:
        a = ds.read(1).astype(np.float64)
        if ds.nodata is not None and np.isfinite(ds.nodata):
            a[a == ds.nodata] = np.nan
    return a, None


def _load_json(p, default):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return default


def slope_deg(h, px):
    gy, gx = np.gradient(np.asarray(h, np.float64), px, px)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def write_layers(outdir, dsm, px_size_m, uncertainty=None, units="m"):
    """Height / slope / uncertainty PNGs + layers.json."""
    dsm = np.asarray(dsm, np.float64)
    px = float(px_size_m or 1.0)
    layers = [dict(id="photo", label="Photo", file="texture.png")]

    fin = dsm[np.isfinite(dsm)]
    if fin.size:
        lo, hi = (float(v) for v in np.percentile(fin, [1, 99]))
        _save(colorize(dsm, RAMPS["blue"], lo, hi), os.path.join(outdir, "layer_height.png"))
        layers.append(dict(id="height", label="Height", file="layer_height.png",
                           kind="sequential", stops=RAMPS["blue"], vmin=lo, vmax=hi,
                           unit=units, note="surface elevation, 1st-99th percentile"))

        s = slope_deg(dsm, px)
        smax = float(min(60.0, max(10.0, np.nanpercentile(s, 99))))
        _save(colorize(s, RAMPS["orange"], 0.0, smax), os.path.join(outdir, "layer_slope.png"))
        layers.append(dict(id="slope", label="Slope", file="layer_slope.png",
                           kind="sequential", stops=RAMPS["orange"], vmin=0.0, vmax=smax,
                           unit="°", note="steepness of the surface, true (not exaggerated)"))

    if uncertainty is not None and np.isfinite(uncertainty).any():
        u = np.asarray(uncertainty, np.float64)
        umax = float(max(np.nanpercentile(u, 98), 1e-6))
        _save(colorize(u, RAMPS["violet"], 0.0, umax),
              os.path.join(outdir, "layer_uncertainty.png"))
        layers.append(dict(id="uncertainty", label="Uncertainty", file="layer_uncertainty.png",
                           kind="sequential", stops=RAMPS["violet"], vmin=0.0, vmax=umax,
                           unit=units, note="disagreement between the four rotated passes"))

    # Keep an error layer from an earlier validation of the same run
    old = _load_json(os.path.join(outdir, "layers.json"), {}).get("layers", [])
    layers += [l for l in old if l.get("id") == "error"]
    with open(os.path.join(outdir, "layers.json"), "w") as f:
        json.dump(dict(layers=layers), f, indent=1)
    return layers


def add_error_layer(outdir, error, offset_m=0.0, units="m"):
    """Signed error (estimate - reference) as a diverging layer."""
    e = np.asarray(error, np.float64)
    if not np.isfinite(e).any():
        return None
    v = float(max(np.nanpercentile(np.abs(e), 95), 0.5))
    _save(colorize(e, DIVERGING, -v, v), os.path.join(outdir, "layer_error.png"))
    spec = dict(id="error", label="Error", file="layer_error.png", kind="diverging",
                stops=DIVERGING, vmin=-v, vmax=v, unit=units,
                note=("estimate minus reference; blue = too low, red = too high"
                      + (f"; median datum offset of {offset_m:+.2f} m removed"
                         if abs(offset_m) > 1e-6 else "")))
    p = os.path.join(outdir, "layers.json")
    doc = _load_json(p, dict(layers=[dict(id="photo", label="Photo", file="texture.png")]))
    doc["layers"] = [l for l in doc.get("layers", []) if l.get("id") != "error"] + [spec]
    with open(p, "w") as f:
        json.dump(doc, f, indent=1)
    return spec


def write_reference_grid(outdir, ref, like="dsm.tif"):
    """The reference, already on the estimate's grid, for profile() and probe()."""
    import rasterio
    with rasterio.open(os.path.join(outdir, like)) as ds:
        prof = ds.profile
    prof.update(dtype="float32", count=1, nodata=np.nan)
    with rasterio.open(os.path.join(outdir, "reference_grid.tif"), "w", **prof) as ds:
        ds.write(np.asarray(ref, np.float32), 1)


_SERIES = (("dsm", "dsm.tif"), ("ndsm", "ndsm.tif"), ("dtm", "dtm.tif"),
           ("uncertainty", "uncertainty.tif"), ("reference", "reference_grid.tif"))


def _bilinear(a, rows, cols):
    from scipy.ndimage import map_coordinates
    valid = np.isfinite(a)
    filled = np.where(valid, a, np.nanmedian(a[valid]) if valid.any() else 0.0)
    v = map_coordinates(filled, [rows, cols], order=1, mode="nearest")
    hole = map_coordinates((~valid).astype(np.float64), [rows, cols], order=1,
                           mode="nearest") > 0.5
    v[hole] = np.nan
    return v


def profile(outdir, r0, c0, r1, c1, n=256, px_size_m=1.0):
    """Every raster of the run, sampled along the segment (r0,c0) -> (r1,c1)."""
    dsm, _ = _read(os.path.join(outdir, "dsm.tif"))
    if dsm is None:
        raise FileNotFoundError("dsm.tif")
    H, W = dsm.shape
    r0, r1 = (float(np.clip(v, 0, H - 1)) for v in (r0, r1))
    c0, c1 = (float(np.clip(v, 0, W - 1)) for v in (c0, c1))
    px = float(px_size_m or 1.0)
    length = float(np.hypot(r1 - r0, c1 - c0) * px)
    n = int(np.clip(n, 2, 2048))
    n = max(2, min(n, int(length / px) + 2)) if length > 0 else 2
    t = np.linspace(0.0, 1.0, n)
    rows, cols = r0 + t * (r1 - r0), c0 + t * (c1 - c0)
    out = dict(distance_m=(t * length).round(3).tolist(), length_m=length,
               start=[r0, c0], end=[r1, c1])
    for key, fname in _SERIES:
        a, _ = _read(os.path.join(outdir, fname))
        if a is None or a.shape != dsm.shape:
            continue
        v = _bilinear(a, rows, cols)
        out[key] = [None if not np.isfinite(x) else round(float(x), 3) for x in v]

    z = np.array([np.nan if v is None else v for v in out["dsm"]], float)
    if np.isfinite(z[[0, -1]]).all():
        dh = float(z[-1] - z[0])
        out.update(dh_m=dh, grade_deg=float(np.degrees(np.arctan2(dh, max(length, 1e-9)))))
    if n > 2 and length > 0:
        step = length / (n - 1)
        # Steepest sustained stretch, over ~2 m so a roof edge is a wall, not noise
        k = max(1, int(round(2.0 / max(step, 1e-9))))
        if n > k:
            s = np.degrees(np.arctan(np.abs(z[k:] - z[:-k]) / (k * step)))
            if np.isfinite(s).any():
                out["max_slope_deg"] = float(np.nanmax(s))
    if "reference" in out:
        r = np.array([np.nan if v is None else v for v in out["reference"]], float)
        valid = np.isfinite(r) & np.isfinite(z)
        if valid.sum() >= 3:
            d = z[valid] - r[valid]
            off = float(np.median(d))
            out["vs_reference"] = dict(
                rmse_m=float(np.sqrt(np.mean(d ** 2))),
                rmse_shift_m=float(np.sqrt(np.mean((d - off) ** 2))),
                median_offset_m=off, n=int(valid.sum()))
    return out


def probe(outdir, r, c):
    """Every raster of the run at one pixel (bilinear)."""
    result = {}
    dsm, _ = _read(os.path.join(outdir, "dsm.tif"))
    if dsm is None:
        raise FileNotFoundError("dsm.tif")
    H, W = dsm.shape
    r = float(np.clip(r, 0, H - 1)); c = float(np.clip(c, 0, W - 1))
    result.update(row=r, col=c)
    for key, fname in _SERIES:
        a, _ = _read(os.path.join(outdir, fname))
        if a is None or a.shape != dsm.shape:
            continue
        v = float(_bilinear(a, np.array([r]), np.array([c]))[0])
        result[key] = None if not np.isfinite(v) else round(v, 3)
    return result
