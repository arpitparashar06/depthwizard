#!/usr/bin/env python3
"""make_demo.py - cut a demo-sized GeoTIFF out of a large open image.

    python mathsandml/benchmark/make_demo.py            # the built-in Indian scene
    python mathsandml/benchmark/make_demo.py --url <cog.tif> --out demo/x.tif \\
           --center 83.55,18.254 --size-m 600 --px 0.5

Reads only the window it needs from a cloud-optimised GeoTIFF, resamples it to
the benchmark's 0.5 m, and writes an RGB GeoTIFF in the scene's own UTM zone
with the licence and attribution in its tags. A 10 cm drone mosaic is 150 MP;
the depth model at 0.5 m over 600 m is 16 tiles - minutes, not an hour.

There is no LiDAR under these scenes, so they are for the demo and the 3D
flythrough, never for the accuracy table.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

DEMOS = {
    # OpenAerialMap, CC-BY 4.0. The only openly licensed sub-metre image inside
    # India in the OAM catalogue when this was written (Sep 2026).
    "india_andhra": dict(
        url="https://oin-hotosm-temp.s3.us-east-1.amazonaws.com/"
            "685624e7e319ef725277456f/0/685624e7e319ef7252774570.tif",
        attribution="Imagery: 'sadanand' by hareesh via OpenAerialMap, CC-BY 4.0",
        license="CC-BY-4.0",
        note="Andhra Pradesh, near Vizianagaram - acquired 2025-06-20, 0.10 m GSD",
    ),
}


def utm_epsg(lon, lat):
    return (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="india_andhra", choices=sorted(DEMOS))
    ap.add_argument("--url", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--center", default=None, help="lon,lat - default: image centre")
    ap.add_argument("--size-m", type=float, default=600.0)
    ap.add_argument("--px", type=float, default=0.5)
    a = ap.parse_args()

    cfg = dict(DEMOS[a.name])
    url = a.url or cfg["url"]
    out = a.out or os.path.join("demo", a.name + ".tif")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.warp import reproject, transform, transform_bounds

    src_path = url if not url.startswith("http") else "/vsicurl/" + url
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", VSI_CACHE="TRUE"):
        with rasterio.open(src_path) as ds:
            w, s, e, n = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds, densify_pts=21)
            if a.center:
                lon, lat = (float(v) for v in a.center.split(","))
            else:
                lon, lat = (w + e) / 2, (s + n) / 2
            crs = f"EPSG:{utm_epsg(lon, lat)}"
            xs, ys = transform("EPSG:4326", crs, [lon], [lat])
            npx = int(round(a.size_m / a.px))
            half = npx * a.px / 2
            dst_tr = from_origin(xs[0] - half, ys[0] + half, a.px, a.px)
            bands = [b for b in range(1, min(ds.count, 3) + 1)]
            dst = np.zeros((len(bands), npx, npx), np.float32)
            print(f"[demo] {os.path.basename(url)}: {ds.width}x{ds.height} px, "
                  f"{ds.count} bands, {ds.crs} -> {npx}x{npx} at {a.px} m in {crs}")
            reproject(rasterio.band(ds, bands), dst, dst_transform=dst_tr, dst_crs=crs,
                      resampling=Resampling.average, src_nodata=ds.nodata, dst_nodata=0)
            valid = float((dst.max(axis=0) > 0).mean())
    if valid < 0.5:
        print(f"[demo] only {valid:.0%} of the window has imagery - pick --center "
              f"inside the footprint ({w:.4f},{s:.4f} .. {e:.4f},{n:.4f})")
        return 1
    rgb = np.clip(dst, 0, 255).astype(np.uint8)
    if rgb.shape[0] == 1:
        rgb = np.repeat(rgb, 3, axis=0)
    with rasterio.open(out, "w", driver="GTiff", height=npx, width=npx, count=3,
                       dtype="uint8", crs=crs, transform=dst_tr, compress="deflate",
                       photometric="RGB") as o:
        o.write(rgb)
        o.update_tags(ATTRIBUTION=cfg.get("attribution", url), LICENSE=cfg.get("license", ""),
                      SOURCE=url, NOTE=cfg.get("note", ""))
    with open(os.path.splitext(out)[0] + ".json", "w") as f:
        json.dump(dict(cfg, url=url, center=[lon, lat], size_m=a.size_m, px_size_m=a.px,
                       crs=crs, footprint_wgs84=[w, s, e, n], valid_fraction=valid), f, indent=2)
    print(f"[demo] wrote {out} ({valid:.0%} covered) - {cfg.get('attribution', '')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
