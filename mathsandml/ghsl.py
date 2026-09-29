"""An automatic height prior from the GHSL building-height grid."""
import glob
import math
import os
import shutil
import tempfile
import zipfile

import numpy as np

PRODUCT = "GHS_BUILT_H_ANBH_E2018_GLOBE_R2023A_54009_100"
BASE_URL = ("https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
            "GHS_BUILT_H_GLOBE_R2023A")
MOLLWEIDE = "ESRI:54009"
TILE_M = 1_000_000.0
X0 = -18_041_000.0          # left edge of column 1
Y0 = 9_000_000.0            # top edge of row 1
N_COLS, N_ROWS = 36, 18

MIN_BUILT_HEIGHT_M = 3.0
GHSL_FLOOR_M = 2.5
# Heights above this are no-data codes or corrupt reads, not buildings.
MAX_PLAUSIBLE_M = 500.0
# Which built cell sets the prior.
PRIOR_PCT = 95.0
DOWNLOAD_TIMEOUT_S = int(os.environ.get("GHSL_TIMEOUT_S", "180"))


# tiles
def tile_id(x, y):
    """GHSL tile holding Mollweide point (x, y), e.g. 'R4_C19'."""
    c = min(int(math.floor((x - X0) / TILE_M)) + 1, N_COLS)
    r = min(int(math.floor((Y0 - y) / TILE_M)) + 1, N_ROWS)
    return f"R{r}_C{c}"


def tiles_for_bounds(left, bottom, right, top):
    """Every tile a Mollweide bounding box touches, in row-major order."""
    eps = 1e-6
    c0 = max(1, min(int(math.floor((left - X0) / TILE_M)) + 1, N_COLS))
    c1 = max(1, min(int(math.floor((right - eps - X0) / TILE_M)) + 1, N_COLS))
    r0 = max(1, min(int(math.floor((Y0 - top) / TILE_M)) + 1, N_ROWS))
    r1 = max(1, min(int(math.floor((Y0 - (bottom + eps)) / TILE_M)) + 1, N_ROWS))
    return [f"R{r}_C{c}" for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def tile_url(tid):
    return (f"{BASE_URL}/{PRODUCT}/V1-0/tiles/"
            f"{PRODUCT}_V1_0_{tid}.zip")


def search_dirs(cache_root):
    dirs = []
    if os.environ.get("GHSL_DIR"):
        dirs.append(os.environ["GHSL_DIR"])
    dirs.append(os.path.join(cache_root, "ghsl"))
    return dirs


def _covers(path, bounds_moll):
    import rasterio
    from rasterio.warp import transform_bounds
    try:
        with rasterio.open(path) as ds:
            if ds.crs is None:
                return False
            l, b, r, t = transform_bounds(MOLLWEIDE, ds.crs, *bounds_moll,
                                          densify_pts=21)
            B = ds.bounds
            return l >= B.left and r <= B.right and b >= B.bottom and t <= B.top
    except Exception:
        return False


def _local_file(tid, tile_bounds, dirs):
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        named = sorted(glob.glob(os.path.join(d, f"*{tid}.tif")) +
                       glob.glob(os.path.join(d, "**", f"*{tid}.tif"),
                                 recursive=True))
        for p in named:
            if "BUILT_H" in os.path.basename(p).upper():
                return p
        for p in sorted(glob.glob(os.path.join(d, "*.tif")) +
                        glob.glob(os.path.join(d, "*.tiff"))):
            if "BUILT_H" in os.path.basename(p).upper() and _covers(p, tile_bounds):
                return p
    return None


def _download_tile(tid, cache_dir):
    import requests
    os.makedirs(cache_dir, exist_ok=True)
    url = tile_url(tid)
    print(f"[ghsl] downloading tile {tid} (one time per region) ...")
    with tempfile.TemporaryDirectory(dir=cache_dir) as tmp:
        zpath = os.path.join(tmp, f"{tid}.zip")
        with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT_S) as r:
            r.raise_for_status()
            with open(zpath, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        with zipfile.ZipFile(zpath) as z:
            tifs = [n for n in z.namelist() if n.lower().endswith((".tif", ".tiff"))]
            if not tifs:
                raise RuntimeError(f"no GeoTIFF inside {url}")
            z.extract(tifs[0], tmp)
            dst = os.path.join(cache_dir, os.path.basename(tifs[0]))
            shutil.move(os.path.join(tmp, tifs[0]), dst)
    print(f"[ghsl] cached -> {dst}")
    return dst


def _tile_bounds(tid):
    r, c = (int(v[1:]) for v in tid.split("_"))
    left = X0 + (c - 1) * TILE_M
    top = Y0 - (r - 1) * TILE_M
    right = -X0 if c == N_COLS else left + TILE_M
    return (left, top - TILE_M, right, top)


# values under the scene
def _read_cells(path, meta):
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window, from_bounds
    with rasterio.open(path) as ds:
        l, b, r, t = transform_bounds(meta["crs"], ds.crs, *meta["bounds"],
                                      densify_pts=21)
        w = from_bounds(l, b, r, t, ds.transform)
        c0 = max(0, int(math.floor(w.col_off + 1e-9)))
        r0 = max(0, int(math.floor(w.row_off + 1e-9)))
        c1 = min(ds.width, int(math.ceil(w.col_off + w.width - 1e-9)))
        r1 = min(ds.height, int(math.ceil(w.row_off + w.height - 1e-9)))
        if c1 <= c0 or r1 <= r0:          # the scene is not on this file
            return np.zeros(0)
        win = Window(c0, r0, c1 - c0, r1 - r0)
        a = ds.read(1, window=win, masked=True).astype(np.float64)
        scale = (ds.scales or (1.0,))[0] or 1.0
        offset = (ds.offsets or (0.0,))[0] or 0.0
    vals = np.ma.filled(a * scale + offset, np.nan).ravel()
    vals = vals[np.isfinite(vals)]
    vals = vals[(vals >= 0) & (vals <= MAX_PLAUSIBLE_M)]
    return vals


def cell_heights(meta, cache_root, allow_download=True, log=print):
    """(values, sources) for the scene, or (None, reason) when GHSL can't answer."""
    if meta.get("mode") != "absolute" or meta.get("crs") is None \
            or meta.get("bounds") is None:
        return None, "the image has no coordinates"
    from rasterio.warp import transform_bounds
    try:
        bm = transform_bounds(meta["crs"], MOLLWEIDE, *meta["bounds"],
                              densify_pts=21)
    except Exception as ex:
        return None, f"could not project the footprint ({type(ex).__name__})"

    dirs = search_dirs(cache_root)
    cache_dir = os.path.join(cache_root, "ghsl")
    allow = allow_download and os.environ.get("GHSL_DOWNLOAD", "1") not in (
        "0", "false", "no")
    vals, sources = [], []
    for tid in tiles_for_bounds(*bm):
        path = _local_file(tid, _tile_bounds(tid), dirs)
        label = None
        if path:
            label = f"{os.path.basename(path)}"
        elif allow:
            try:
                path = _download_tile(tid, cache_dir)
                label = f"{os.path.basename(path)} (downloaded)"
            except Exception as ex:
                return None, (f"tile {tid} not on disk and the download failed "
                              f"({type(ex).__name__}: {str(ex)[:120]})")
        else:
            return None, (f"tile {tid} not on disk and downloading is off - put "
                          f"it in GHSL_DIR or run warm_cache.py once online")
        v = _read_cells(path, meta)
        vals.append(v)
        sources.append(label)
    v = np.concatenate(vals) if vals else np.zeros(0)
    if v.size == 0:
        return None, "GHSL has no data under this footprint"
    return v, sources


def height_prior(meta, cache_root, pct=PRIOR_PCT, min_height_m=MIN_BUILT_HEIGHT_M,
                 allow_download=True, log=print):
    """The automatic landmark height for a georeferenced scene, or None."""
    v, src = cell_heights(meta, cache_root, allow_download=allow_download, log=log)
    if v is None:
        return None, src
    built = v[v >= min_height_m]
    if built.size == 0:
        floor = int(np.sum((v > 0) & (v < min_height_m)))
        if floor:
            return None, (f"GHSL marks {floor} of {v.size} cells under this "
                          f"scene as built, but all at its {GHSL_FLOOR_M:g} m "
                          f"floor - low-rise buildings too low for GHSL to "
                          f"give a height, so there is nothing to anchor on")
        return None, (f"GHSL shows no buildings under this scene "
                      f"({v.size} cells) - open country, forest or water has "
                      f"no building height to anchor on")
    h = float(np.percentile(built, pct))
    prior = dict(known_height_m=round(h, 1),
                 statistic=f"p{pct:g} of the average building height of the "
                           f"built 100 m cells under the scene",
                 n_cells=int(v.size), n_built=int(built.size),
                 built_mean_m=float(built.mean()),
                 built_max_m=float(built.max()),
                 product=PRODUCT, source="; ".join(src))
    return prior, "ok"


def _main(argv):
    import rasterio
    root = os.environ.get("DEPTHWIZARD_CACHE", os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache"))
    bad = 0
    for path in argv:
        try:
            with rasterio.open(path) as ds:
                meta = dict(mode="absolute" if ds.crs else "relative",
                            crs=ds.crs, bounds=ds.bounds)
            prior, why = height_prior(meta, root)
            name = (os.path.basename(os.path.dirname(path))
                    if os.path.basename(path) == "rgb.tif" else os.path.basename(path))
            if prior:
                print(f"[ghsl] {name}: {prior['known_height_m']:g} m "
                      f"({prior['n_built']}/{prior['n_cells']} cells built, "
                      f"mean {prior['built_mean_m']:.1f}, max {prior['built_max_m']:.1f}) "
                      f"<- {prior['source']}")
            else:
                print(f"[ghsl] {name}: no prior - {why}")
        except Exception as ex:
            bad += 1
            print(f"[ghsl] {path}: FAILED {type(ex).__name__}: {ex}")
    for p in sorted(glob.glob(os.path.join(root, "ghsl", "*.tif"))):
        with rasterio.open(p) as ds:
            print(f"[ghsl] file {os.path.basename(p)}: {ds.width}x{ds.height} "
                  f"{ds.dtypes[0]} nodata={ds.nodata} scale={ds.scales} "
                  f"offset={ds.offsets} crs={ds.crs.to_string()[:40]} "
                  f"res={ds.res} compress={ds.compression}")
    return 1 if bad else 0


if __name__ == "__main__":
    import sys
    sys.exit(_main(sys.argv[1:]))
