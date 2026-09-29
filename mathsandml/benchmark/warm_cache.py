#!/usr/bin/env python3
"""Run the depth model once per image and fetch each DEM once."""
import argparse
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_benchmark import _load_env  # noqa: E402  (reads .env before inference)

_load_env()
import inference as I  # noqa: E402


def warm(path, dem=True, ghsl=True):
    t = time.time()
    rgb, meta = I.load_image(path)
    I.predict_depth_ensemble(rgb, rotations=I.ROTATIONS,
                             cache_key=I.source_key(path))
    src = "-"
    if dem and meta["mode"] == "absolute":
        info = {}
        with tempfile.TemporaryDirectory() as d:
            I.fetch_dem(meta, os.path.join(d, "dem.tif"), info)
        src = info.get("dem_source", "-")
    gh = "-"
    if ghsl and meta["mode"] == "absolute":
        prior, why = I.ghsl_height_prior(meta)
        gh = (f"{prior['known_height_m']:g} m" if prior else f"none ({why[:70]})")
    name = (os.path.basename(os.path.dirname(path))
            if os.path.basename(path) == "rgb.tif" else os.path.basename(path))
    print(f"[warm] {name}: "
          f"{rgb.shape[1]}x{rgb.shape[0]}  depth cached  DEM: {src}  "
          f"GHSL: {gh}  "
          f"({time.time() - t:.0f} s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="*")
    parser.add_argument("--scenes", default=None)
    parser.add_argument("--no-dem", action="store_true")
    parser.add_argument("--no-ghsl", action="store_true",
                    help="skip the GHSL building-height tile")
    args = parser.parse_args()
    paths = list(args.images)
    if args.scenes:
        for n in sorted(os.listdir(args.scenes)):
            p = os.path.join(args.scenes, n, "rgb.tif")
            if os.path.exists(p):
                paths.append(p)
    if not paths:
        print("nothing to warm")
        return 1
    bad = 0
    for p in paths:
        try:
            warm(p, dem=not args.no_dem, ghsl=not args.no_ghsl)
        except Exception as e:
            bad += 1
            print(f"[warm] {p}: FAILED {type(e).__name__}: {e}")
    print(f"\n{len(paths) - bad}/{len(paths)} warmed -> {I.CACHE_ROOT}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
