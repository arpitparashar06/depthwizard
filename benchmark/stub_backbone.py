"""Replace the depth backbone with a stub that reproduces its failure modes,
so the harness can be tested without downloading 1.3 GB of weights.

The stub derives depth from the IMAGE it is handed, not from a lookup, so it
behaves correctly under rotation - which is what makes it a real test of the
rotation ensemble rather than a plumbing check:

  - output is scale-agnostic (normalised), so tile and pass alignment matter
  - a large perspective ramp is painted DOWN THE FRAME, exactly as the real
    model does, so rotating the input rotates the artefact with it
  - structures are attenuated and blurred, so attenuation reporting has work
"""
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import inference as I  # noqa: E402

RAMP = float(os.environ.get("STUB_RAMP", "0.55"))
ATTEN = float(os.environ.get("STUB_ATTEN", "0.72"))
NOISE = float(os.environ.get("STUB_NOISE", "0.012"))


def predict_depth(rgb, tile=None, overlap=None):
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


def install():
    I.predict_depth = predict_depth
    print(f"[stub] backbone replaced (ramp={RAMP}, atten={ATTEN}, noise={NOISE})")
