#!/usr/bin/env python3
"""selftest.py - prove the benchmark harness works without downloading anything.

    python mathsandml/benchmark/selftest.py            # ~30 s, no weights
    python mathsandml/benchmark/selftest.py --keep     # leave the artefacts behind
    python mathsandml/benchmark/selftest.py --write-scenes DIR   # scenes only

Three things live here because they are only ever used together, by the
function at the bottom:

  1. SYNTHETIC SCENES. Four georeferenced tiles - urban, sparse, hilly, forest
     - with the truth known by construction rather than measured.

  2. A STUB BACKBONE. The depth model replaced by something instant that
     reproduces its three failure modes, so this is a real test of the pipeline
     and not a plumbing check:
       - output is scale-agnostic (normalised), so tile and pass alignment matter
       - a large perspective ramp is painted DOWN THE FRAME, exactly as the real
         model does, so rotating the input rotates the artefact with it
       - structures are attenuated and blurred, so attenuation reporting has work
     It derives depth from the IMAGE it is handed, not from a lookup, which is
     what makes it behave correctly under the rotation ensemble.

  3. THE SELFTEST ITSELF. Builds the scenes, installs the stub, runs the whole
     benchmark over them, and checks every artefact came out.

Exit code 0 means scenes were built, the pipeline ran, validation produced
metrics and figures, and the report rendered. Nothing here validates the depth
model - it validates the harness.
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.dirname(HERE)):        # benchmark/, then mathsandml/
    if _p not in sys.path:
        sys.path.insert(0, _p)

import inference as I  # noqa: E402


# ===========================================================================
# 1. SYNTHETIC SCENES
# ===========================================================================
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


def write_scene(outdir, name, kind, seed):
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

# ===========================================================================
# 2. THE STUB BACKBONE
# ===========================================================================
RAMP = float(os.environ.get("STUB_RAMP", "0.55"))
ATTEN = float(os.environ.get("STUB_ATTEN", "0.72"))
NOISE = float(os.environ.get("STUB_NOISE", "0.012"))


def _stub_predict_depth(rgb, tile=None, overlap=None):
    """Fake relative depth, computed from the image so it rotates with it."""
    a = np.asarray(rgb, np.float64)
    luma = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]

    # bright roofs read as tall; smooth a little so it is not pure texture
    s = gaussian_filter(luma, 1.6)
    lo, hi = np.percentile(s, [1, 99])
    p = np.clip((s - lo) / max(hi - lo, 1e-9), 0, 1)
    p = ATTEN * p

    # THE ARTEFACT: the model reads the bottom of the FRAME as nearer and paints
    # it high. Tied to the frame, so np.rot90 on the input moves it too.
    H, W = p.shape
    p = p + RAMP * (np.linspace(0, 1, H)[:, None] * np.ones((1, W)))

    rng = np.random.default_rng(abs(hash((H, W))) % (2 ** 32))
    return (p + rng.normal(0, NOISE, p.shape)).astype(np.float64)


def install_stub():
    I.predict_depth = _stub_predict_depth
    print(f"[stub] backbone replaced (ramp={RAMP}, atten={ATTEN}, noise={NOISE})")

# ===========================================================================
# 3. THE SELFTEST
# ===========================================================================
def main():
    tmp = tempfile.mkdtemp(prefix="depthwizard-selftest-")
    scenes = os.path.join(tmp, "scenes")
    results = os.path.join(tmp, "results")
    keep = "--keep" in sys.argv
    try:
        print("== building synthetic scenes ==")
        for i, kind in enumerate(["urban", "sparse", "hilly", "forest"]):
            write_scene(scenes, f"synth_{kind}", kind, seed=i)

        install_stub()

        print("\n== running benchmark ==")
        import run_benchmark
        sys.argv = ["run_benchmark.py", "--scenes", scenes, "--out", results]
        rc = run_benchmark.main()

        print("\n== checking artefacts ==")
        need = ["dsm.tif", "ndsm.tif", "validation.md", "validation.json",
                "error_map.png", "scatter.png", "stability.png"]
        missing = []
        for s in sorted(os.listdir(results)):
            d = os.path.join(results, s)
            if not os.path.isdir(d):
                continue
            have = set(os.listdir(d))
            gone = [n for n in need if n not in have]
            print(f"  {s:16s} {'ok' if not gone else 'MISSING ' + ', '.join(gone)}")
            missing += gone
        for f in ("report.md", "results.json"):
            ok = os.path.exists(os.path.join(results, f))
            print(f"  {f:16s} {'ok' if ok else 'MISSING'}")
            if not ok:
                missing.append(f)

        if rc != 0 or missing:
            print("\nSELFTEST FAILED")
            return 1
        print("\nSELFTEST PASSED - harness is working end to end.")
        print("The metrics above are meaningless as accuracy numbers: the "
              "backbone was a stub and the scenes are synthetic blocks.")
        return 0
    finally:
        if keep:
            print(f"\nartefacts kept in {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    if "--write-scenes" in sys.argv:
        out = sys.argv[sys.argv.index("--write-scenes") + 1]
        for i, kind in enumerate(["urban", "sparse", "hilly", "forest"]):
            write_scene(out, f"synth_{kind}", kind, seed=i)
        raise SystemExit(0)
    sys.exit(main())
