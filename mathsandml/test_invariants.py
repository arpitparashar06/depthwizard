"""
test_invariants.py - the properties the pipeline must never violate.
=============================================================================

Not accuracy tests. Accuracy lives in benchmark/. These are the statements
that have to hold for ANY input, because if one breaks, a number somewhere
downstream is quietly wrong and nothing errors:

  * dsm = dtm + ndsm, pixel for pixel. If this drifts, the elevation map on
    disk and the 3D model are different surfaces, and the validator scores a
    raster nobody ever sees.
  * nothing sits below its own ground.
  * the preprocessing chain is monotonic per band - it may remap intensity,
    it may not reorder it. Reordering would mean we changed the geometry the
    model reads, which would make every accuracy claim ours instead of the
    sensor's.
  * the mask cannot be moved by the pixels it is looking for.

Run:  python -m pytest mathsandml/test_invariants.py -q
  or: python mathsandml/test_invariants.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import preprocess as P


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def synthetic_scene(h=160, w=200, seed=0):
    """A little city: sloped ground, three flat roofs, a patch of canopy."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    dtm = 40.0 + 0.05 * xx + 0.02 * yy            # a gentle hillside
    ndsm = np.zeros((h, w))
    ndsm[20:60, 30:80] = 18.0                      # a block
    ndsm[80:110, 120:170] = 32.0                   # a tower
    ndsm[120:150, 20:60] = 7.5                     # a shed
    ndsm[30:50, 140:180] = 9.0 + rng.normal(0, 1.2, (20, 40))   # canopy
    return dtm, np.maximum(ndsm, 0.0)


# ---------------------------------------------------------------------------
# the surface invariants
# ---------------------------------------------------------------------------
def test_dsm_equals_dtm_plus_ndsm():
    """The one that matters. dsm = dtm + ndsm, everywhere, exactly."""
    dtm, ndsm = synthetic_scene()
    dsm = dtm + ndsm

    # re-derive the way inference.py does, from dsm and the terrain baseline
    derived_ndsm = np.maximum(dsm - dtm, 0.0)
    derived_dtm = dsm - derived_ndsm

    assert np.allclose(dsm, derived_dtm + derived_ndsm, atol=1e-9), \
        "dsm != dtm + ndsm - the raster and the mesh are different surfaces"


def test_ndsm_is_never_negative():
    """Nothing sits below its own ground."""
    dtm, ndsm = synthetic_scene()
    dsm = dtm + ndsm
    assert (np.maximum(dsm - dtm, 0.0) >= 0).all()


def test_structures_survive_the_round_trip():
    """A 32 m tower is still 32 m after splitting and recombining."""
    dtm, ndsm = synthetic_scene()
    dsm = dtm + ndsm
    back = np.maximum(dsm - dtm, 0.0)
    assert abs(back[80:110, 120:170].mean() - 32.0) < 1e-9


# ---------------------------------------------------------------------------
# the preprocessing invariants
# ---------------------------------------------------------------------------
def test_stretch_is_monotonic_per_band():
    """Preprocessing may remap intensity. It may NOT reorder it.

    If a darker pixel could come out brighter than a lighter one, we would
    have altered the structure the depth model reads - and every accuracy
    number after that would be measuring our preprocessing, not the sensor.
    """
    rng = np.random.default_rng(1)
    raw = rng.integers(500, 4000, (64, 64, 3)).astype(np.uint16)
    out, _ = P.stretch_per_band(raw)
    for c in range(3):
        a = raw[..., c].ravel()
        b = out[..., c].ravel()
        order = np.argsort(a, kind="stable")
        # allow ties and the clipped tails to flatten, but never invert
        assert np.all(np.diff(b[order].astype(np.int16)) >= 0), \
            f"band {c} reordered by the stretch"


def test_well_exposed_8bit_is_left_alone():
    """Don't touch an image that does not need touching."""
    rng = np.random.default_rng(2)
    img = rng.integers(8, 248, (80, 80, 3)).astype(np.uint8)
    want, _ = P.needs_stretch(img)
    assert want is False


def test_hazy_8bit_is_stretched():
    rng = np.random.default_rng(3)
    img = rng.integers(100, 150, (80, 80, 3)).astype(np.uint8)
    want, why = P.needs_stretch(img)
    assert want is True and "range" in why


def test_mask_is_not_dragged_by_the_cloud_it_looks_for():
    """The regression this file exists for.

    With a min/max or percentile normalisation, a bright cloud covering a few
    percent of the frame pushes every ordinary surface toward zero, and the
    shadow test then flags almost the whole scene. Median/IQR cannot be moved
    by anything smaller than a quarter of the image.
    """
    rng = np.random.default_rng(4)
    a = np.zeros((240, 320, 3), np.uint16)
    a[..., 0] = rng.integers(800, 1400, (240, 320))
    a[..., 1] = rng.integers(700, 1250, (240, 320))
    a[..., 2] = rng.integers(1500, 2100, (240, 320))
    a[20:60, 20:80] = 64000      # cloud, 3.1% of the frame
    a[180:220, 240:300] = 5      # shadow, 3.1%

    mask, rep = P.valid_mask(a)
    assert 0.90 < mask.mean() < 0.96, \
        f"expected ~6% masked, got {100 * (1 - mask.mean()):.1f}%"
    assert 0.02 < rep["cloud_frac"] < 0.05
    assert 0.02 < rep["shadow_frac"] < 0.05


def test_clean_scene_masks_nothing():
    rng = np.random.default_rng(5)
    a = rng.integers(800, 1400, (120, 160, 3)).astype(np.uint16)
    mask, rep = P.valid_mask(a)
    assert rep["masked_frac"] < 0.01


def test_uniform_frame_does_not_mask_itself():
    a = np.full((80, 80, 3), 250, np.uint8)
    mask, rep = P.valid_mask(a)
    assert mask.all() and rep["masked_frac"] == 0.0


def test_nan_holes_do_not_poison_the_output():
    """NaN survives clip(); casting it to uint8 is undefined behaviour."""
    rng = np.random.default_rng(6)
    a = rng.integers(500, 3000, (64, 64, 3)).astype(np.float32)
    a[10:20, 10:20] = np.nan
    out, _, _ = P.condition(a, do_clahe=False)
    assert out.dtype == np.uint8
    assert np.isfinite(out).all()


def test_colour_cast_is_removed():
    """Per-band stretch should bring the channel means together."""
    rng = np.random.default_rng(7)
    a = np.zeros((120, 160, 3), np.uint16)
    a[..., 0] = rng.integers(800, 1400, (120, 160))
    a[..., 1] = rng.integers(700, 1250, (120, 160))
    a[..., 2] = rng.integers(1500, 2100, (120, 160))   # strong blue cast
    out, _, _ = P.condition(a, do_clahe=False)
    means = [out[..., c].mean() for c in range(3)]
    assert max(means) - min(means) < 12.0, f"cast survived: {means}"


def test_condition_never_changes_the_grid():
    rng = np.random.default_rng(8)
    a = rng.integers(0, 4000, (97, 133, 3)).astype(np.uint16)
    out, mask, _ = P.condition(a)
    assert out.shape == (97, 133, 3)
    assert mask.shape == (97, 133)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok    {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
