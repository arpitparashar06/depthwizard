#!/usr/bin/env python3
"""Dutch (AHN4) and Swiss (swissSURFACE3D) LiDAR tiles for the landscapes GAMUS"""

import argparse
import math
import os
import random
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "mathsandml"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "mathsandml", "benchmark"))
import data as D  # noqa: E402

REGIONS = {
    # Netherlands: AHN4
    "amsterdam_zuidas":     ("ahn", 120000, 484500, 2000, "urban", "train"),
    "den_haag_centre":      ("ahn", 81000, 455000, 1500, "urban", "train"),
    "utrecht_centre":       ("ahn", 136000, 456000, 2000, "urban", "train"),
    "eindhoven_centre":     ("ahn", 161500, 383000, 2000, "urban", "train"),
    "groningen_centre":     ("ahn", 233500, 582000, 1500, "urban", "val"),
    "hoge_veluwe":          ("ahn", 180000, 452000, 4000, "forest", "train"),
    "utrechtse_heuvelrug":  ("ahn", 157000, 446000, 4000, "forest", "train"),
    "sallandse_heuvelrug":  ("ahn", 232000, 484000, 3000, "forest", "train"),
    "drents_friese_wold":   ("ahn", 219000, 549000, 4000, "forest", "val"),
    "noordoostpolder":      ("ahn", 183000, 523000, 6000, "sparse", "train"),
    "zeeland_farmland":     ("ahn", 50000, 390000, 6000, "sparse", "train"),
    "groningen_farmland":   ("ahn", 242000, 594000, 6000, "sparse", "val"),
    "zuid_limburg_hills":   ("ahn", 189000, 311000, 5000, "hilly", "train"),
    # Switzerland: swisstopo
    "bern_centre":          ("swisstopo", 2600500, 1199700, 1500, "urban", "train"),
    "basel_centre":         ("swisstopo", 2611300, 1267300, 1500, "urban", "train"),
    "lausanne_centre":      ("swisstopo", 2538200, 1152600, 1500, "urban", "train"),
    "luzern_centre":        ("swisstopo", 2666000, 1211800, 1500, "urban", "train"),
    "zurich_oerlikon":      ("swisstopo", 2683500, 1252800, 1000, "urban", "train"),
    "geneve_centre":        ("swisstopo", 2500300, 1117800, 1500, "urban", "val"),
    "emmental":             ("swisstopo", 2625000, 1203000, 5000, "hilly", "train"),
    "napf_forest":          ("swisstopo", 2638000, 1210000, 4000, "forest", "train"),
    "sihlwald_forest":      ("swisstopo", 2684500, 1234000, 2500, "forest", "train"),
    "ticino_forest":        ("swisstopo", 2715000, 1110000, 5000, "forest", "train"),
    "jura_delemont":        ("swisstopo", 2593000, 1245000, 5000, "forest", "val"),
    "seeland_farmland":     ("swisstopo", 2580000, 1210000, 5000, "sparse", "train"),
    "broye_farmland":       ("swisstopo", 2562000, 1185000, 5000, "sparse", "train"),
    "grindelwald":          ("swisstopo", 2645500, 1164000, 3000, "hilly", "train"),
    "zermatt":              ("swisstopo", 2624500, 1096000, 3000, "hilly", "train"),
    "engelberg":            ("swisstopo", 2674000, 1186000, 3000, "hilly", "train"),
    "davos":                ("swisstopo", 2783500, 1186500, 4000, "hilly", "val"),
}
CRS = {"ahn": "EPSG:28992", "swisstopo": "EPSG:2056"}


def benchmark_centres():
    """Every benchmark scene centre, per source, from fetch_data.AOIS."""
    try:
        import fetch_data as FD
    except ImportError:
        return {}
    out = {}
    for src, aois in FD.AOIS.items():
        for name, (crs, cx, cy, *_rest) in aois.items():
            out.setdefault(src, []).append((name, float(cx), float(cy)))
    return out


def sample_centres(regions, per_region, tile_m, exclude_km, seed=0):
    bench = benchmark_centres()
    rng = random.Random(seed)
    out = []
    for name, (src, cx, cy, rad, land, split) in regions.items():
        got, tries = [], 0
        while len(got) < per_region and tries < per_region * 50:
            tries += 1
            r = rad * math.sqrt(rng.random())
            a = rng.uniform(0, 2 * math.pi)
            x = round((cx + r * math.cos(a)) / 50) * 50
            y = round((cy + r * math.sin(a)) / 50) * 50
            if any(math.hypot(x - bx, y - by) < exclude_km * 1000
                   for _, bx, by in bench.get(src, [])):
                continue
            if any(math.hypot(x - gx, y - gy) < tile_m * 0.8 for gx, gy in got):
                continue
            got.append((x, y))
        for k, (x, y) in enumerate(got):
            out.append(dict(id=f"{src}_{name}_{k:03d}", source=src, region=name,
                            landscape=land, split=split, cx=x, cy=y))
    return out


_AHN_NAMES = {}


def fetch_ahn_tile(cx, cy, size_m, px, outdir):
    import fetch_data as FD
    S = FD.SOURCES["ahn"]
    if not _AHN_NAMES:
        _AHN_NAMES["dsm"], _ = FD.discover_wcs(S["wcs"], S["dsm_pat"])
        _AHN_NAMES["dtm"], _ = FD.discover_wcs(S["wcs"], S["dtm_pat"])
        rgb_name, _, formats = FD.discover_wms(S["wms"], S["rgb_pat"])
        _AHN_NAMES["rgb"], _AHN_NAMES["fmt"] = rgb_name, FD.pick_format(formats)
    crs = CRS["ahn"]
    bbox = FD.bbox_from_centre(cx, cy, size_m)
    n = int(round(size_m / px))
    FD.wcs_coverage(S["wcs"], _AHN_NAMES["dsm"], bbox, crs, n, n,
                    os.path.join(outdir, "ref_dsm.tif"))
    FD.wcs_coverage(S["wcs"], _AHN_NAMES["dtm"], bbox, crs, n, n,
                    os.path.join(outdir, "ref_dtm.tif"))
    rgb = FD.wms_image(S["wms"], _AHN_NAMES["rgb"], bbox, crs, n, n, _AHN_NAMES["fmt"])
    FD.write_geotiff(os.path.join(outdir, "rgb.tif"), rgb, bbox, crs)


def _stac_hrefs(collection, bbox, crs):
    import re
    import fetch_data as FD
    from rasterio.warp import transform_bounds
    wgs = transform_bounds(crs, "EPSG:4326", *bbox, densify_pts=21)
    url = f"{FD.SOURCES['swisstopo']['stac']}/collections/{collection}/items"
    j = FD.get(url, dict(bbox=",".join(f"{v:.6f}" for v in wgs), limit=100)).json()
    best = {}
    for f in j.get("features") or []:
        candidates = [(a.get("eo:gsd", 9e9), a["href"]) for a in (f.get("assets") or {}).values()
                 if a.get("href", "").lower().endswith((".tif", ".tiff"))]
        if not candidates:
            continue
        candidates.sort()
        key = re.search(r"(\d{4}-\d{4})", f.get("id", ""))
        key = key.group(1) if key else f.get("id", "")
        when = (f.get("properties") or {}).get("datetime") or ""
        if key not in best or when > best[key][0]:
            best[key] = (when, candidates[0])
    if not best:
        raise RuntimeError(f"STAC: nothing for {collection}")
    return [c for _, c in best.values()]          # [(gsd, href)]


def _mosaic(hrefs, bbox, crs, n, bands, px):
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject, Resampling
    dst_tr = from_bounds(*bbox, n, n)
    dst = np.full((bands, n, n), np.nan, np.float32)
    for gsd, href in hrefs:
        level = -1
        g = float(gsd) if gsd and gsd < 9e8 else px
        while g * 2 ** (level + 2) <= px * 1.01:
            level += 1
        tmp = np.full((bands, n, n), np.nan, np.float32)
        opened = None
        for lv in ([level] if level >= 0 else []) + [None]:
            try:
                opened = rasterio.open(href, overview_level=lv) if lv is not None \
                    else rasterio.open(href)
                break
            except Exception:
                opened = None
        if opened is None:
            continue
        with opened as ds:
            reproject(rasterio.band(ds, list(range(1, bands + 1))), tmp,
                      dst_transform=dst_tr, dst_crs=crs, src_nodata=ds.nodata,
                      dst_nodata=np.nan, resampling=Resampling.bilinear)
        fill = np.isnan(dst) & np.isfinite(tmp)
        dst[fill] = tmp[fill]
    return dst


def fetch_swisstopo_tile(cx, cy, size_m, px, outdir):
    import fetch_data as FD
    S = FD.SOURCES["swisstopo"]
    crs = CRS["swisstopo"]
    bbox = FD.bbox_from_centre(cx, cy, size_m)
    n = int(round(size_m / px))
    dsm = _mosaic(_stac_hrefs(S["dsm_collection"], bbox, crs), bbox, crs, n, 1, px)
    FD.write_geotiff(os.path.join(outdir, "ref_dsm.tif"), dsm[0], bbox, crs)
    dtm = _mosaic(_stac_hrefs(S["dtm_collection"], bbox, crs), bbox, crs, n, 1, px)
    FD.write_geotiff(os.path.join(outdir, "ref_dtm.tif"), dtm[0], bbox, crs)
    rgb = _mosaic(_stac_hrefs(S["rgb_collection"], bbox, crs), bbox, crs, n, 3, px)
    rgb = np.clip(np.nan_to_num(rgb), 0, 255).astype(np.uint8).transpose(1, 2, 0)
    FD.write_geotiff(os.path.join(outdir, "rgb.tif"), rgb, bbox, crs)


def _read(path, band_count=None):
    import rasterio
    with rasterio.open(path) as ds:
        a = ds.read().astype(np.float32)
        nod = ds.nodata
    if nod is not None and np.isfinite(nod):
        a[a == nod] = np.nan
    a[np.abs(a) > 1e30] = np.nan
    return a


def fill_nearest(a):
    """Fill NaNs with the nearest valid value (DTM gaps under buildings)."""
    from scipy.ndimage import distance_transform_edt, gaussian_filter
    m = ~np.isfinite(a)
    if not m.any():
        return a
    if m.all():
        return a
    idx = distance_transform_edt(m, return_distances=False, return_indices=True)
    out = a[tuple(idx)]
    sm = gaussian_filter(out, 3)
    out[m] = sm[m]
    return out


def ndsm_from(dsm, dtm):
    """Height above ground."""
    water = ~np.isfinite(dsm) & ~np.isfinite(dtm)
    ground = fill_nearest(dtm)
    h = dsm - ground
    h[water] = 0.0
    h[h < -1.5] = np.nan
    return np.where(np.isfinite(h), np.clip(h, 0, 400), np.nan).astype(np.float32), water


def convert_folder(folder, out, entry, quality=95):
    rgb = _read(os.path.join(folder, "rgb.tif"))
    rgb = np.clip(np.nan_to_num(rgb[:3]), 0, 255).astype(np.uint8).transpose(1, 2, 0)
    dsm = _read(os.path.join(folder, "ref_dsm.tif"))[0]
    dtm = _read(os.path.join(folder, "ref_dtm.tif"))[0]
    h, water = ndsm_from(dsm, dtm)
    # Quality gates
    lum = rgb.mean(2)
    blank = float(((lum < 4) | (rgb.min(2) > 250)).mean())
    valid = float(np.isfinite(h).mean())
    water_fraction = float(water.mean())
    reason = None
    if blank > 0.15:
        reason = "blank image"
    elif valid < 0.8:
        reason = "missing height"
    elif water_fraction > 0.5:
        reason = "mostly water"
    if reason:
        return None, reason
    rp, hp, cp = D.write_tile(out, entry["split"], entry["id"], rgb, h, None, jpeg_q=quality)
    v = h[np.isfinite(h)]
    e = dict(entry, rgb=rp, height=hp, cls=None, valid=valid, water=water_fraction,
             h_p50=float(np.percentile(v, 50)), h_p99=float(np.percentile(v, 99)),
             h_max=float(v.max()))
    return e, None


def work(rec, out, tile_m, px, quality):
    fetchers = {"ahn": fetch_ahn_tile, "swisstopo": fetch_swisstopo_tile}
    height_path = os.path.join(out, rec["split"], f"{rec['id']}_h.png")
    if os.path.exists(height_path):
        return dict(rec, rgb=os.path.join(rec["split"], f"{rec['id']}_rgb.jpg"),
                    height=os.path.join(rec["split"], f"{rec['id']}_h.png"),
                    cls=None, gsd=px), None
    tmp = tempfile.mkdtemp(prefix="lidar_")
    try:
        for attempt in range(3):
            try:
                fetchers[rec["source"]](rec["cx"], rec["cy"], tile_m, px, tmp)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(4 * (attempt + 1))
        return convert_folder(tmp, out, dict(rec, gsd=px), quality)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=None)
    parser.add_argument("--per-region", type=int, default=35)
    parser.add_argument("--tile-m", type=float, default=512.0)
    parser.add_argument("--px", type=float, default=0.5)
    parser.add_argument("--exclude-km", type=float, default=3.0)
    parser.add_argument("--sources", default="ahn,swisstopo")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    srcs = {s.strip() for s in args.sources.split(",")}
    regions = {k: v for k, v in REGIONS.items() if v[0] in srcs}
    records = sample_centres(regions, args.per_region, args.tile_m, args.exclude_km, args.seed)
    if args.list or not args.out:
        for name, (src, cx, cy, rad, land, split) in regions.items():
            n = sum(r["region"] == name for r in records)
            print(f"  {name:22s} {src:9s} {land:7s} {split:5s} {n:3d} tiles  ({cx}, {cy}) r={rad} m")
        print(f"{len(records)} tile centres; benchmark scenes excluded within {args.exclude_km:g} km")
        return
    os.makedirs(args.out, exist_ok=True)
    print(f"[lidar] {len(records)} tiles of {args.tile_m:g} m at {args.px:g} m/px from "
          f"{len(regions)} regions", flush=True)
    entries, rejected, failed = [], {}, 0
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(work, r, args.out, args.tile_m, args.px, args.jpeg_quality): r for r in records}
        for n, f in enumerate(as_completed(futs), 1):
            r = futs[f]
            try:
                e, why = f.result()
                if e:
                    entries.append(e)
                else:
                    rejected[why] = rejected.get(why, 0) + 1
            except Exception as ex_:
                failed += 1
                print(f"[lidar] FAILED {r['id']}: {type(ex_).__name__}: {str(ex_)[:160]}",
                      flush=True)
            if n % 25 == 0 or n == len(records):
                rate = n / max(time.time() - t0, 1e-6)
                print(f"[lidar] {n}/{len(records)}  kept {len(entries)}  rejected "
                      f"{sum(rejected.values())}  failed {failed}  "
                      f"eta {(len(records) - n) / max(rate, 1e-6) / 60:.0f} min", flush=True)
                D.save_index(args.out, entries, dataset="AHN4 + swisstopo")
    D.save_index(args.out, sorted(entries, key=lambda e: e["id"]), dataset="AHN4 + swisstopo")
    by = {}
    for e in entries:
        k = f"{e['source']}/{e['landscape']}/{e['split']}"
        by[k] = by.get(k, 0) + 1
    print(f"[lidar] done: {len(entries)} tiles kept, rejected {rejected}, failed {failed}")
    for k in sorted(by):
        print(f"    {k:28s} {by[k]}")


if __name__ == "__main__":
    main()
