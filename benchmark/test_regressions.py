"""test_regressions.py - every bug found in review, locked down.

    python benchmark/test_regressions.py

No model weights, no network. Each test names the failure it prevents and,
where the bug produced a wrong NUMBER, asserts against the measured value
rather than against "it runs".
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

FAILED = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name +
          (("\n          -> " + str(detail)) if not cond else ""))
    if not cond:
        FAILED.append(name)


# ===========================================================================
print("\n1. attenuation() must recommend a gain that REDUCES error")
# The old median(r/p) estimator recommended x2.83 on the Rotterdam tile when
# the error-minimising gain was x0.74; applying its advice moved object-band
# RMSE from 6.06 m to 10.83 m. Reproduced here with a surface built to a known
# scale plus the two things that broke the estimator: misses and false
# positives.
import validate as V  # noqa: E402

# The pathology, built explicitly. Most structure pixels are PARTIAL misses -
# the reference has a building, the prediction has a stub. Those clear the old
# mask (pred > 0.25*min_h) and each contributes a ratio near 10, so the median
# of ratios lands on ~10 while the roofs that carry the actual height are only
# 20% short. This is the shape of the real Rotterdam distribution.
rng = np.random.default_rng(0)
partial_ref, partial_pred = np.full(8000, 10.0), np.full(8000, 1.0)
roof_ref, roof_pred = np.full(2000, 30.0), np.full(2000, 30.0 * 0.8)
flat = np.zeros(50000)
ref = np.concatenate([partial_ref, roof_ref, flat])
pred = np.concatenate([partial_pred, roof_pred, flat])
ref = ref + rng.normal(0, 0.05, ref.shape)
pred = pred + rng.normal(0, 0.05, pred.shape)

at = V.attenuation(pred, ref)
g = at["suggested_alpha_gain"]


def band_rmse(gain):
    return float(np.sqrt(np.mean((gain * pred - ref) ** 2)))


sweep = np.linspace(0.3, 4.0, 371)
best = float(sweep[int(np.argmin([band_rmse(x) for x in sweep]))])
check("suggested gain is the error-minimising one",
      abs(g - best) < 0.02, f"suggested {g:.3f} vs argmin {best:.3f}")
check("it does not make things worse",
      band_rmse(g) <= band_rmse(1.0) + 1e-9,
      f"RMSE {band_rmse(1.0):.3f} -> {band_rmse(g):.3f}")
# the old estimator, reproduced verbatim from the code that was replaced:
#     m = (ref_obj > min_h) & (pred_obj > 0.25 * min_h);  median(r / p)
old_mask = np.isfinite(pred) & np.isfinite(ref) & (ref > 2.0) & (pred > 0.5)
legacy = float(np.median(ref[old_mask] / pred[old_mask]))
check("the old estimator pointed somewhere much worse (bug reproduced)",
      band_rmse(legacy) > 2 * band_rmse(g),
      f"legacy gain {legacy:.2f} -> RMSE {band_rmse(legacy):.3f}, "
      f"new gain {g:.2f} -> {band_rmse(g):.3f}")
check("...and the old mask is what did it",
      legacy > 5 * g, f"legacy {legacy:.2f} vs new {g:.2f}")
check("report carries the evidence for its own recommendation",
      {"rmse_at_gain_1", "rmse_at_suggested", "improves"} <= set(at),
      sorted(at))

# a clean surface that is genuinely 0.7x should recover 1/0.7 = 1.43
clean_ref = np.concatenate([np.full(4000, 20.0), np.zeros(20000)])
at2 = V.attenuation(clean_ref * 0.7, clean_ref)
check("recovers a known pure scale error",
      abs(at2["suggested_alpha_gain"] - 1 / 0.7) < 0.02,
      f"got {at2['suggested_alpha_gain']:.3f}, want {1/0.7:.3f}")

# ===========================================================================
print("\n2. no spurious BLAS warnings, and singular fits never leak inf")
import warnings  # noqa: E402
import refine as R  # noqa: E402
import buildings as B  # noqa: E402

# The contract is: never hand back a coefficient vector that is not finite.
# A merely rank-deficient system is fine - lstsq returns the minimum-norm
# solution - so the guard exists for the cases that genuinely blow up.
x = np.arange(40.0)
cases = [("singular", np.c_[x, x, np.ones(40)], x),
         ("inf in y", np.c_[x, np.ones(40)], np.where(x == 3, np.inf, x)),
         ("nan in y", np.c_[x, np.ones(40)], np.where(x == 3, np.nan, x)),
         ("zero variance", np.c_[np.ones(40), np.ones(40)], np.ones(40))]
for name, A, y in cases:
    c = R._lstsq(A, y)
    check(f"_lstsq never returns non-finite ({name})",
          c is None or bool(np.all(np.isfinite(c))), c)
check("_lstsq still solves a well-posed system",
      R._lstsq(np.c_[x, np.ones(40)], 3 * x + 1) is not None)
check("...and solves it correctly",
      bool(np.allclose(R._lstsq(np.c_[x, np.ones(40)], 3 * x + 1), [3, 1])))
for mod, name in ((R, "refine"), (B, "buildings"), (V, "validate")):
    check(f"{name} has the guard", hasattr(mod, "_lstsq"))

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    surf = np.zeros((120, 120))
    surf[30:60, 30:34] = 18.0          # a one-cell-wide "building"
    surf[70:100, 70:100] = 12.0
    R.flatten_structures(surf, px_size_m=0.5, object_sigma_m=15.0)
    V._robust_affine(np.ones(200), np.ones(200))   # zero variance
    rw = [x for x in w if issubclass(x.category, RuntimeWarning)]
check("degenerate geometry raises no RuntimeWarning",
      not rw, [str(x.message) for x in rw])

out, _ = R.flatten_structures(surf, px_size_m=0.5, object_sigma_m=15.0)
check("flatten output stays finite", bool(np.all(np.isfinite(out))))

# ===========================================================================
print("\n3. the benchmark must refuse to calibrate from the reference")
import json  # noqa: E402

src = open(os.path.join(HERE, "run_benchmark.py")).read()
check("the reference-percentile fallback is gone",
      "median of reference nDSM above p" not in src)
check("it errors instead", "no known_height_m in scene.json" in src)

for d in sorted(os.listdir(os.path.join(HERE, "scenes"))):
    f = os.path.join(HERE, "scenes", d, "scene.json")
    if not os.path.exists(f):
        continue
    cfg = json.load(open(f))
    check(f"{d} ships an operator prior",
          bool(cfg.get("known_height_m")), cfg.get("known_height_m"))
    check(f"{d} says where the prior came from",
          "NOT" in str(cfg.get("known_height_source", "")).upper(),
          cfg.get("known_height_source"))

# ===========================================================================
print("\n4. the no-DEM clip survives refine()")
import inference as I  # noqa: E402

h = np.array([[-3.0, 1.0], [5.0, -0.5]])
out, info = I.clip_below_ground(h, {"calibration": "scale-only (no DEM; "
                                                   "heights above local ground)"})
check("negatives clipped when no DEM supplied terrain", float(out.min()) == 0.0)
check("the clipped fraction is recorded",
      info.get("clipped_negative_frac_post_refine") == 0.5, info)

out2, info2 = I.clip_below_ground(
    h, {"calibration": "freqsplit (DEM terrain + scaled detail)"})
check("below-sea-level elevations are left alone with a DEM",
      float(out2.min()) == -3.0,
      "the Netherlands is largely below NAP; clipping here would be wrong")

# ===========================================================================
print("\n5. refine gets the split the pipeline measured, not its own default")
for f in ("server.py", "run_geotiff.py"):
    src = open(os.path.join(ROOT, f)).read()
    check(f"{f} passes the scene's sigma to refine",
          'object_sigma_m=float(info.get("sigma_m")' in src)
    check(f"{f} re-clips after refine", "clip_below_ground(height, info)" in src)

print("\n" + ("ALL PASS" if not FAILED else "FAILED: " + ", ".join(FAILED)))
sys.exit(1 if FAILED else 0)
