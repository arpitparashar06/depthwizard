#!/usr/bin/env python3
"""warm_cache.py - run the depth model once per image and fetch each DEM once.

    python mathsandml/benchmark/warm_cache.py --scenes mathsandml/benchmark/scenes
    python mathsandml/benchmark/warm_cache.py demo1.tif demo2.png

After this, every later run over the same images - the benchmark, a sweep of
calibration settings, a live demo with no internet - reads cache/depth,
cache/dem and cache/ghsl instead of calling the backbone, OpenTopography or
the GHSL server.
"""
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
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*")
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--no-dem", action="store_true")
    ap.add_argument("--no-ghsl", action="store_true",
                    help="skip the GHSL building-height tile")
    a = ap.parse_args()
    paths = list(a.images)
    if a.scenes:
        for n in sorted(os.listdir(a.scenes)):
            p = os.path.join(a.scenes, n, "rgb.tif")
            if os.path.exists(p):
                paths.append(p)
    if not paths:
        print("nothing to warm")
        return 1
    bad = 0
    for p in paths:
        try:
            warm(p, dem=not a.no_dem, ghsl=not a.no_ghsl)
        except Exception as e:
            bad += 1
            print(f"[warm] {p}: FAILED {type(e).__name__}: {e}")
    print(f"\n{len(paths) - bad}/{len(paths)} warmed -> {I.CACHE_ROOT}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
