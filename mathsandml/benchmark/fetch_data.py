#!/usr/bin/env python3
"""Build benchmark scenes from open LiDAR + orthophoto services."""

import argparse
import io
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

try:
    import requests
except ImportError:
    sys.exit("pip install requests")

TIMEOUT = 120
UA = {"User-Agent": "depthwizard-benchmark/1.0"}


AOIS = {
    "ahn": {
        "rotterdam_centre": ("EPSG:28992", 92600, 436700, "urban",
                             "mid-rise perimeter blocks around a park, 6-10 storeys"),
        "delft_old":        ("EPSG:28992", 84400, 447300, "urban",
                             "dense low-rise, narrow streets"),
        "flevoland_farm":   ("EPSG:28992", 165000, 502000, "sparse",
                             "motorway and canal corridor with deciduous shelter belts"),
        "veluwe_forest":    ("EPSG:28992", 185500, 464000, "forest",
                             "a third conifer plantation, the rest open heath and a pond"),
    },
    "swisstopo": {
        "zurich_centre":    ("EPSG:2056", 2683400, 1247500, "urban",
                             "mid-rise European core on a slope"),
        "interlaken_hills": ("EPSG:2056", 2633000, 1170000, "sparse",
                             "detached chalets and meadow on a gentle slope"),
        # The genuinely steep ones - the hilly row of the marking scheme
        "wengen_slope":     ("EPSG:2056", 2637000, 1161750, "hilly",
                             "car-free village on a ~30% mountainside: hotels, "
                             "chalets and forest strips on real relief"),
        "lavaux_terraces":  ("EPSG:2056", 2546950, 1149250, "hilly",
                             "terraced vineyards and a wine village above Lake "
                             "Geneva - steep, stepped, built"),
        "jura_forest":      ("EPSG:2056", 2570000, 1220000, "forest",
                             "mixed forest on rolling terrain"),
    },
    "usgs": {
        "denver_metro":     ("EPSG:4326", -104.9903, 39.7392, "urban",
                             "3DEP is bare earth - terrain only, no buildings"),
        "appalachian_ridge":("EPSG:4326", -82.5515, 35.5951, "hilly",
                             "strong relief with forest cover"),
    },
}

SOURCES = {
    "ahn": dict(
        wcs="https://service.pdok.nl/rws/actueel-hoogtebestand-nederland/wcs/v1_0",
        wms="https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0",
        # Priority order: the sharpest product first, a generic catch last
        dsm_pat=[r"dsm[_\-]?0?5", r"\bdsm\b", r"dsm"],
        dtm_pat=[r"dtm[_\-]?0?5", r"\bdtm\b", r"dtm"],
        rgb_pat=[r"ortho.*(hr|8\s*cm)", r"actueel.*ortho", r"ortho"],
        max_px=2500,
        native_px=0.5,
    ),
    "swisstopo": dict(
        stac="https://data.geo.admin.ch/api/stac/v0.9",
        dsm_collection="ch.swisstopo.swisssurface3d-raster",
        dtm_collection="ch.swisstopo.swissalti3d",
        rgb_collection="ch.swisstopo.swissimage-dop10",
        native_px=0.5,
    ),
    "usgs": dict(
        dem="https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer",
        rgb="https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer",
        native_px=1.0,
    ),
}


def get(url, params=None, stream=False):
    r = requests.get(url, params=params, headers=UA, timeout=TIMEOUT, stream=stream)
    r.raise_for_status()
    return r


def strip_ns(tag):
    return tag.split("}")[-1].lower()


def bbox_from_centre(cx, cy, size_m):
    h = size_m / 2.0
    return (cx - h, cy - h, cx + h, cy + h)


def write_geotiff(path, arr, bbox, crs, count=None):
    import rasterio
    from rasterio.transform import from_bounds
    a = np.asarray(arr)
    if a.ndim == 2:
        a = a[None]
    elif a.shape[-1] in (3, 4):
        a = np.transpose(a[..., :3], (2, 0, 1))
    n = count or a.shape[0]
    w, s, e, nn = bbox
    tr = from_bounds(w, s, e, nn, a.shape[2], a.shape[1])
    with rasterio.open(path, "w", driver="GTiff", height=a.shape[1],
                       width=a.shape[2], count=n, dtype=a.dtype,
                       crs=crs, transform=tr) as ds:
        for i in range(n):
            ds.write(a[i], i + 1)
    return path


# OGC discovery
def _best(names, patterns):
    if isinstance(patterns, str):
        patterns = [patterns]
    for pat in patterns:
        hit = next((n for n in names if re.search(pat, n, re.I)), None)
        if hit:
            return hit
    return None


def discover_wcs(base, patterns):
    """Return the coverage name matching patterns, and the full list."""
    x = get(base, dict(SERVICE="WCS", VERSION="1.0.0",
                       REQUEST="GetCapabilities")).content
    root = ET.fromstring(x)
    names = [e.text.strip() for e in root.iter()
             if strip_ns(e.tag) == "name" and e.text and e.text.strip()]
    names = [n for n in names if not n.lower().startswith("wcs")]
    return _best(names, patterns), names


def discover_wms(base, patterns):
    """Return (layer, all layers, GetMap formats)."""
    x = get(base, dict(SERVICE="WMS", VERSION="1.3.0",
                       REQUEST="GetCapabilities")).content
    root = ET.fromstring(x)
    names = []
    for lay in root.iter():
        if strip_ns(lay.tag) != "layer":
            continue
        for ch in lay:
            if strip_ns(ch.tag) == "name" and ch.text:
                names.append(ch.text.strip())

    formats = []
    for req in root.iter():
        if strip_ns(req.tag) != "getmap":
            continue
        for ch in req.iter():
            if strip_ns(ch.tag) == "format" and ch.text:
                formats.append(ch.text.strip())
    return _best(names, patterns), names, formats


def pick_format(formats):
    """Prefer lossless, accept lossy, and never invent one the service lacks."""
    for want in ("image/png", "image/tiff", "image/jpeg"):
        for f in formats:
            if f.lower().startswith(want):
                return f
    return formats[0] if formats else "image/jpeg"


def wcs_coverage(base, coverage, bbox, crs, width, height, out):
    r = get(base, dict(SERVICE="WCS", VERSION="1.0.0", REQUEST="GetCoverage",
                       COVERAGE=coverage, CRS=crs, RESPONSE_CRS=crs,
                       BBOX=",".join(f"{v:.3f}" for v in bbox),
                       WIDTH=width, HEIGHT=height, FORMAT="GEOTIFF"))
    if b"ServiceException" in r.content[:2000] or b"<?xml" in r.content[:20]:
        raise RuntimeError(f"WCS refused the request: {r.content[:400]!r}")
    with open(out, "wb") as f:
        f.write(r.content)
    return out


def wms_image(base, layer, bbox, crs, width, height, fmt="image/jpeg"):
    from PIL import Image
    r = get(base, {"SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap",
                   "LAYERS": layer, "STYLES": "", "CRS": crs,
                   # WMS 1.3.0 honours the CRS axis order.
                   "BBOX": ",".join(f"{v:.3f}" for v in bbox),
                   "WIDTH": width, "HEIGHT": height, "FORMAT": fmt})
    head = r.content[:2000]
    if b"ServiceException" in head or head[:5] == b"<?xml":
        raise RuntimeError(f"WMS refused the request: {r.content[:400]!r}")
    return np.array(Image.open(io.BytesIO(r.content)).convert("RGB"))


# sources
def fetch_ahn(aoi, cfg, size_m, px, outdir):
    S = SOURCES["ahn"]
    crs, cx, cy, landscape, note = cfg
    bbox = bbox_from_centre(cx, cy, size_m)
    n = int(round(size_m / px))

    if n > S["max_px"]:
        raise RuntimeError(
            f"{n} px exceeds the service cap of {S['max_px']}. "
            f"Reduce --size-m (max {S['max_px'] * px:.0f} m at {px} m/px) "
            f"or raise --px.")

    dsm_name, all_cov = discover_wcs(S["wcs"], S["dsm_pat"])
    if not dsm_name:
        raise RuntimeError(f"no DSM coverage matched {S['dsm_pat']} in {all_cov}")
    dtm_name, _ = discover_wcs(S["wcs"], S["dtm_pat"])
    rgb_name, all_lay, formats = discover_wms(S["wms"], S["rgb_pat"])
    if not rgb_name:
        raise RuntimeError(f"no ortho layer matched {S['rgb_pat']} in {all_lay[:40]}")
    fmt = pick_format(formats)
    print(f"    coverage={dsm_name}  dtm={dtm_name}  ortho={rgb_name}  fmt={fmt}")

    wcs_coverage(S["wcs"], dsm_name, bbox, crs, n, n,
                 os.path.join(outdir, "ref_dsm.tif"))
    if dtm_name:
        try:
            wcs_coverage(S["wcs"], dtm_name, bbox, crs, n, n,
                         os.path.join(outdir, "ref_dtm.tif"))
        except Exception as e:
            print(f"    (no DTM: {e})")
    rgb = wms_image(S["wms"], rgb_name, bbox, crs, n, n, fmt)
    write_geotiff(os.path.join(outdir, "rgb.tif"), rgb, bbox, crs)
    return dict(landscape=landscape, note=note, source="AHN4 / PDOK",
                crs=crs, bbox=list(bbox), px_size_m=px,
                coverage=dsm_name, ortho=rgb_name, wms_format=fmt)


def fetch_swisstopo(aoi, cfg, size_m, px, outdir, parts=("dsm", "dtm", "rgb")):
    S = SOURCES["swisstopo"]
    crs, cx, cy, landscape, note = cfg
    bbox = bbox_from_centre(cx, cy, size_m)

    # STAC search wants WGS84
    from rasterio.warp import transform_bounds
    wgs = transform_bounds(crs, "EPSG:4326", *bbox, densify_pts=21)

    def tile_hrefs(collection, ext=(".tif", ".tiff")):
        url = f"{S['stac']}/collections/{collection}/items"
        j = get(url, dict(bbox=",".join(f"{v:.6f}" for v in wgs), limit=100)).json()
        feats = j.get("features") or []
        if not feats:
            raise RuntimeError(f"STAC returned no items for {collection} over {wgs}")

        best = {}
        for f in feats:
            candidates = [(a.get("eo:gsd", 9e9), a["href"])
                     for a in (f.get("assets") or {}).values()
                     if a.get("href", "").lower().endswith(ext)]
            if not candidates:
                continue
            candidates.sort()
            fid = f.get("id", "")
            key = re.search(r"(\d{4}-\d{4})", fid)          # the tile's km coords
            key = key.group(1) if key else fid
            when = (f.get("properties") or {}).get("datetime") or ""
            if key not in best or when > best[key][0]:
                best[key] = (when, candidates[0][1])
        if not best:
            raise RuntimeError(f"no GeoTIFF asset among {len(feats)} {collection} items")
        print(f"    {collection}: {len(best)} tile(s)")
        return [href for _, href in best.values()]

    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_bounds
    n = int(round(size_m / px))

    def crop_mosaic(hrefs, bands):
        dst_tr = from_bounds(*bbox, n, n)
        dst = np.full((bands, n, n), np.nan, np.float32)
        for href in hrefs:
            tmp = np.full((bands, n, n), np.nan, np.float32)
            with rasterio.open(href) as ds:
                reproject(rasterio.band(ds, list(range(1, bands + 1))), tmp,
                          dst_transform=dst_tr, dst_crs=crs,
                          src_nodata=ds.nodata, dst_nodata=np.nan,
                          resampling=Resampling.bilinear)
            fill = np.isnan(dst) & np.isfinite(tmp)
            dst[fill] = tmp[fill]
            if np.isfinite(dst).all():
                break
        gaps = float(np.isnan(dst).mean())
        if gaps > 0.001:
            print(f"    warning: {100 * gaps:.1f}% of the AOI had no tile coverage")
        return dst

    if "dsm" in parts:
        dsm = crop_mosaic(tile_hrefs(S["dsm_collection"]), 1)
        write_geotiff(os.path.join(outdir, "ref_dsm.tif"), dsm[0], bbox, crs)
    if "dtm" in parts:
        try:
            dtm = crop_mosaic(tile_hrefs(S["dtm_collection"]), 1)
            write_geotiff(os.path.join(outdir, "ref_dtm.tif"), dtm[0], bbox, crs)
        except Exception as e:
            print(f"    (no DTM: {type(e).__name__}: {str(e)[:120]})")
    if "rgb" in parts:
        rgb = crop_mosaic(tile_hrefs(S["rgb_collection"]), 3)
        rgb = np.nan_to_num(rgb, nan=0.0)
        write_geotiff(os.path.join(outdir, "rgb.tif"),
                      np.clip(rgb, 0, 255).astype(np.uint8).transpose(1, 2, 0),
                      bbox, crs)
    return dict(landscape=landscape, note=note, source="swissSURFACE3D + SWISSIMAGE",
                crs=crs, bbox=list(bbox), px_size_m=px)


def fetch_usgs(aoi, cfg, size_m, px, outdir):
    S = SOURCES["usgs"]
    _, lon, lat, landscape, note = cfg
    # Work in Web Mercator so size_m is metres
    from rasterio.warp import transform
    crs = "EPSG:3857"
    xs, ys = transform("EPSG:4326", crs, [lon], [lat])
    bbox = bbox_from_centre(xs[0], ys[0], size_m)
    n = int(round(size_m / px))
    common = dict(bbox=",".join(f"{v:.3f}" for v in bbox),
                  bboxSR=3857, imageSR=3857, size=f"{n},{n}", f="image")

    r = get(S["dem"] + "/exportImage", dict(common, format="tiff"))
    with open(os.path.join(outdir, "ref_dsm.tif"), "wb") as f:
        f.write(r.content)
    r = get(S["rgb"] + "/exportImage", dict(common, format="tiff"))
    with open(os.path.join(outdir, "rgb_raw.tif"), "wb") as f:
        f.write(r.content)

    import rasterio
    with rasterio.open(os.path.join(outdir, "rgb_raw.tif")) as ds:
        a = ds.read()[:3]
    write_geotiff(os.path.join(outdir, "rgb.tif"),
                  np.transpose(a, (1, 2, 0)).astype(np.uint8), bbox, crs)
    os.remove(os.path.join(outdir, "rgb_raw.tif"))
    return dict(landscape=landscape, note=note + " | 3DEP is BARE EARTH",
                source="USGS 3DEP + NAIP", crs=crs, bbox=list(bbox),
                px_size_m=px, bare_earth_reference=True)


FETCHERS = dict(ahn=fetch_ahn, swisstopo=fetch_swisstopo, usgs=fetch_usgs)


def check():
    """Probe every endpoint."""
    print("Probing services (no downloads)\n")
    ok = True
    for name, S in SOURCES.items():
        print(f"[{name}]")
        for key in ("wcs", "wms", "stac", "dem", "rgb"):
            if key not in S:
                continue
            url = S[key]
            try:
                if key == "wcs":
                    hit, all_ = discover_wcs(url, S["dsm_pat"])
                    print(f"  {key:5s} ok   DSM coverage -> {hit!r}  "
                          f"({len(all_)} coverages)")
                    if not hit:
                        ok = False
                elif key == "wms":
                    hit, all_, formats = discover_wms(url, S["rgb_pat"])
                    print(f"  {key:5s} ok   ortho layer  -> {hit!r}  "
                          f"({len(all_)} layers, formats={formats} "
                          f"-> using {pick_format(formats)})")
                    if not hit:
                        ok = False
                elif key == "stac":
                    j = get(f"{url}/collections/{S['dsm_collection']}").json()
                    print(f"  {key:5s} ok   collection   -> {j.get('id')!r}")
                else:
                    j = get(url, dict(f="json")).json()
                    print(f"  {key:5s} ok   service      -> "
                          f"{j.get('name') or j.get('serviceDescription', '')[:50]!r}")
            except Exception as e:
                ok = False
                print(f"  {key:5s} FAIL {type(e).__name__}: {str(e)[:130]}")
        print()
    print("All services reachable." if ok else
          "Some services failed. See BENCHMARK.md for the manual download route.")
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="ahn",
                    choices=["ahn", "swisstopo", "usgs", "all"])
    parser.add_argument("--aoi", default=None, help="one area of interest by name")
    parser.add_argument("--out", default="benchmark/scenes")
    parser.add_argument("--size-m", type=float, default=300.0,
                    help="square footprint in metres (300 m at 0.5 m = 600 px)")
    parser.add_argument("--px", type=float, default=0.5, help="output GSD in metres")
    parser.add_argument("--force", action="store_true",
                    help="re-download scenes that already exist (hand-entered "
                         "keys in scene.json are kept either way)")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.check:
        return check()
    if args.list:
        for src, aois in AOIS.items():
            print(f"\n[{src}]")
            for name, (crs, x, y, cls, note) in aois.items():
                print(f"  {name:20s} {cls:7s} {crs}  {x},{y}   {note}")
        return 0

    sources = list(FETCHERS) if args.source == "all" else [args.source]
    os.makedirs(args.out, exist_ok=True)
    made, failed = [], []

    for src in sources:
        for name, config in AOIS[src].items():
            if args.aoi and name != args.aoi:
                continue
            scene = f"{src}_{name}"
            d = os.path.join(args.out, scene)
            have = all(os.path.exists(os.path.join(d, f))
                       for f in ("rgb.tif", "ref_dsm.tif"))
            if have and not args.force:
                print(f"\n[{scene}] already present - skipped (--force to refetch)")
                if src == "swisstopo" and not os.path.exists(
                        os.path.join(d, "ref_dtm.tif")):
                    print("    adding the swissALTI3D terrain model")
                    try:
                        fetch_swisstopo(name, config, args.size_m, args.px, d,
                                        parts=("dtm",))
                    except Exception as e:
                        print(f"    (no DTM: {type(e).__name__}: {str(e)[:120]})")
                made.append(scene)
                continue
            os.makedirs(d, exist_ok=True)
            print(f"\n[{scene}] {config[3]} - {config[4]}")
            try:
                info = FETCHERS[src](name, config, args.size_m, args.px, d)
                info.update(aoi=name, size_m=args.size_m)
                # Keep what a human wrote into an existing scene.json
                old = {}
                sj = os.path.join(d, "scene.json")
                if os.path.exists(sj):
                    try:
                        with open(sj) as f:
                            old = json.load(f)
                    except Exception:
                        old = {}
                info = {**info, **{k: v for k, v in old.items()
                                   if k in ("known_height_m", "known_height_source",
                                            "gcps", "sun_azimuth", "sun_elevation",
                                            "landscape", "note")}}
                with open(sj, "w") as f:
                    json.dump(info, f, indent=2)
                print(f"    -> {d}")
                if info.get("known_height_m"):
                    made.append(scene)
                    continue
                print(f"       ACTION NEEDED: open {os.path.join(d, 'rgb.tif')} "
                      f"and add to scene.json:")
                print(f'         "known_height_m": <height of the tallest '
                      f'structure you can identify, in metres>,')
                print(f'         "known_height_source": "read off the ortho - '
                      f'NOT from the reference"')
                made.append(scene)
            except Exception as e:
                print(f"    FAILED  {type(e).__name__}: {str(e)[:200]}")
                failed.append((scene, f"{type(e).__name__}: {e}"))

    print(f"\n{len(made)} scene(s) built: {', '.join(made) if made else '-'}")
    if failed:
        print(f"{len(failed)} failed:")
        for s, e in failed:
            print(f"  {s}: {e[:160]}")
        print("\nIf a service name has drifted, run --check to see what it does "
              "publish, then adjust the pattern in SOURCES. Or use the manual "
              "download route in BENCHMARK.md.")
    return 0 if made else 1


if __name__ == "__main__":
    sys.exit(main())
