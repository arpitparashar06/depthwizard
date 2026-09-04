"""Replace the depth backbone with a truth-derived stub, so the harness can be
tested without downloading 1.3 GB of weights.

The stub mimics the three failure modes the real model has, which is the point:
  - output is scale-agnostic (normalised 0..1), so alpha calibration is exercised
  - a large fake ramp is painted across the frame (the nadir tilt)
  - structures are attenuated and blurred, so the attenuation report has work to do
"""
import os
import sys

import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import inference as I  # noqa: E402

RAMP = float(os.environ.get("STUB_RAMP", "0.55"))
ATTEN = float(os.environ.get("STUB_ATTEN", "0.72"))
NOISE = float(os.environ.get("STUB_NOISE", "0.012"))
_TRUTH_DIR = os.environ.get("STUB_TRUTH_DIR", "")


def _truth_for(shape):
    """Find the ref_dsm whose shape matches the image being predicted."""
    for root, _, files in os.walk(_TRUTH_DIR):
        if "ref_dsm.tif" in files:
            with rasterio.open(os.path.join(root, "ref_dsm.tif")) as ds:
                if (ds.height, ds.width) == shape:
                    a = ds.read(1).astype(np.float64)
            dtm = os.path.join(root, "ref_dtm.tif")
            if os.path.exists(dtm):
                with rasterio.open(dtm) as ds:
                    t = ds.read(1).astype(np.float64)
            else:
                t = gaussian_filter(a, 30)
            return a, t
    raise RuntimeError(f"no ref_dsm.tif of shape {shape} under {_TRUTH_DIR!r}")


def predict_depth(rgb, tile=None, overlap=None):
    H, W = rgb.shape[:2]
    dsm, dtm = _truth_for((H, W))
    obj = np.maximum(dsm - dtm, 0.0)

    # the model sees terrain smoothly and structures at reduced contrast
    sim = dtm + ATTEN * gaussian_filter(obj, 1.4)

    # normalise, because that is what a relative-depth model returns
    lo, hi = np.percentile(sim, [0.5, 99.5])
    p = (sim - lo) / max(hi - lo, 1e-9)

    # the nadir tilt: bottom of frame reads as "closer" and comes back high
    yy = np.linspace(0, 1, H)[:, None] * np.ones((1, W))
    p = p + RAMP * yy

    rng = np.random.default_rng(0)
    p = p + rng.normal(0, NOISE, p.shape)
    return p.astype(np.float64)


def install():
    I.predict_depth = predict_depth
    print(f"[stub] backbone replaced (ramp={RAMP}, atten={ATTEN}, noise={NOISE})")
