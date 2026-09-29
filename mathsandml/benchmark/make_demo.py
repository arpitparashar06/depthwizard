#!/usr/bin/env python3
"""Cut a demo-sized GeoTIFF out of a large open image."""
import argparse
import json
import math
import os
import sys

import numpy as np

DEMOS = {
    # OpenAerialMap, CC-BY 4.0.
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="india_andhra", choices=sorted(DEMOS))
    parser.add_argument("--url", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--center", default=None, help="lon,lat - default: image centre")
    parser.add_argument("--size-m", type=float, default=600.0)
    parser.add_argument("--px", type=float, default=0.5)
    args = parser.parse_args()

    config = dict(DEMOS[args.name])
    url = args.url or config["url"]
    out = args.out or os.path.join("demo", args.name + ".tif")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.warp import reproject, transform, transform_bounds

    src_path = url if not url.startswith("http") else "/vsicurl/" + url
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", VSI_CACHE="TRUE"):
        with rasterio.open(src_path) as ds:
            w, s, e, n = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds, densify_pts=21)
            if args.center:
                lon, lat = (float(v) for v in args.center.split(","))
            else:
                lon, lat = (w + e) / 2, (s + n) / 2
            crs = f"EPSG:{utm_epsg(lon, lat)}"
            xs, ys = transform("EPSG:4326", crs, [lon], [lat])
            npx = int(round(args.size_m / args.px))
            half = npx * args.px / 2
            dst_tr = from_origin(xs[0] - half, ys[0] + half, args.px, args.px)
            bands = [b for b in range(1, min(ds.count, 3) + 1)]
            dst = np.zeros((len(bands), npx, npx), np.float32)
            print(f"[demo] {os.path.basename(url)}: {ds.width}x{ds.height} px, "
                  f"{ds.count} bands, {ds.crs} -> {npx}x{npx} at {args.px} m in {crs}")
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
        o.update_tags(ATTRIBUTION=config.get("attribution", url), LICENSE=config.get("license", ""),
                      SOURCE=url, NOTE=config.get("note", ""))
    with open(os.path.splitext(out)[0] + ".json", "w") as f:
        json.dump(dict(config, url=url, center=[lon, lat], size_m=args.size_m, px_size_m=args.px,
                       crs=crs, footprint_wgs84=[w, s, e, n], valid_fraction=valid), f, indent=2)
    print(f"[demo] wrote {out} ({valid:.0%} covered) - {config.get('attribution', '')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
