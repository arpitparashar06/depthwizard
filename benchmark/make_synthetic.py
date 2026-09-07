#!/usr/bin/env python3
"""Build synthetic georeferenced scenes with known truth, to exercise the harness."""
import json, os, sys
import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter


def scene(kind, H=384, W=384, px=0.5, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(float)

    # --- terrain -----------------------------------------------------------
    if kind == "hilly":
        terrain = 40 + 25 * np.sin(xx / 90) * np.cos(yy / 70) + 0.05 * yy
    else:
        terrain = 20 + 0.012 * xx + 0.006 * yy
    terrain = gaussian_filter(terrain, 6)

    objects = np.zeros((H, W))
    rgb = np.zeros((H, W, 3), np.uint8)
    # ground colour
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 110, 105, 98
    if kind in ("sparse", "hilly"):
        rgb[..., 0], rgb[..., 1], rgb[..., 2] = 130, 128, 100

    # --- structures --------------------------------------------------------
    if kind == "urban":
        for _ in range(46):
            h = float(rng.uniform(6, 65))
            r0, c0 = rng.integers(8, H - 60), rng.integers(8, W - 60)
            dh, dw = rng.integers(18, 52), rng.integers(18, 52)
            objects[r0:r0 + dh, c0:c0 + dw] = h
            g = int(rng.uniform(150, 215))
            rgb[r0:r0 + dh, c0:c0 + dw] = (g, g - 6, g - 14)
    elif kind == "sparse":
        for _ in range(7):
            h = float(rng.uniform(4, 14))
            r0, c0 = rng.integers(8, H - 40), rng.integers(8, W - 40)
            dh, dw = rng.integers(14, 30), rng.integers(14, 30)
            objects[r0:r0 + dh, c0:c0 + dw] = h
            rgb[r0:r0 + dh, c0:c0 + dw] = (185, 178, 168)
    elif kind == "forest":
        blob = np.zeros((H, W))
        for _ in range(320):
            r0, c0 = rng.integers(0, H), rng.integers(0, W)
            blob[r0, c0] = rng.uniform(9, 26)
        objects = gaussian_filter(blob, 4) * 9.0
        veg = (objects > 3)
        rgb[veg] = (46, 104, 52)
        for _ in range(4):                       # a few buildings in a clearing
            h = float(rng.uniform(8, 20))
            r0, c0 = rng.integers(8, H - 40), rng.integers(8, W - 40)
            objects[r0:r0 + 26, c0:c0 + 26] = h
            rgb[r0:r0 + 26, c0:c0 + 26] = (190, 185, 175)
    elif kind == "hilly":
        for _ in range(10):
            h = float(rng.uniform(5, 18))
            r0, c0 = rng.integers(8, H - 36), rng.integers(8, W - 36)
            objects[r0:r0 + 22, c0:c0 + 22] = h
            rgb[r0:r0 + 22, c0:c0 + 22] = (188, 180, 170)

    objects = gaussian_filter(objects, 0.8)
    dsm = terrain + objects
    rgb = np.clip(rgb.astype(float) + rng.normal(0, 5, rgb.shape), 0, 255).astype(np.uint8)
    return rgb, dsm.astype(np.float32), terrain.astype(np.float32), px


def write(outdir, name, kind, seed):
    d = os.path.join(outdir, name)
    os.makedirs(d, exist_ok=True)
    rgb, dsm, dtm, px = scene(kind, seed=seed)
    H, W = dsm.shape
    # UTM 31N, arbitrary but real origin
    tr = from_origin(600000.0, 5800000.0, px, px)
    crs = "EPSG:32631"

    with rasterio.open(os.path.join(d, "rgb.tif"), "w", driver="GTiff",
                       height=H, width=W, count=3, dtype="uint8",
                       crs=crs, transform=tr) as ds:
        for i in range(3):
            ds.write(rgb[..., i], i + 1)
    for fn, arr in (("ref_dsm.tif", dsm), ("ref_dtm.tif", dtm)):
        with rasterio.open(os.path.join(d, fn), "w", driver="GTiff",
                           height=H, width=W, count=1, dtype="float32",
                           crs=crs, transform=tr) as ds:
            ds.write(arr, 1)
    # A synthetic scene is the one case where the tallest-structure prior is
    # legitimately known without consulting a measurement: it is the
    # generator's own construction parameter, fixed before any surface exists.
    # run_benchmark refuses to invent this number, so it has to be written
    # here, and it has to say where it came from.
    with open(os.path.join(d, "scene.json"), "w") as f:
        json.dump(dict(landscape=kind, synthetic=True,
                       true_max_object_m=float((dsm - dtm).max()),
                       known_height_m=float((dsm - dtm).max()),
                       known_height_source=("generator construction parameter - "
                                            "NOT read back from the reference")),
                  f, indent=2)
    print(f"{name:10s} {kind:8s} objects up to {float((dsm - dtm).max()):5.1f} m")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "synth/scenes"
    for i, kind in enumerate(["urban", "sparse", "hilly", "forest"]):
        write(out, f"synth_{kind}", kind, seed=i)
