"""tests.py - every bug found in review, locked down. No weights, no network.

    python mathsandml/benchmark/tests.py

Two parts, one process, one exit code:

  PART A - REGRESSIONS. Each check names the failure it prevents and, where the
           bug produced a wrong NUMBER, asserts against the measured value
           rather than against "it runs".

  PART B - THE DATUM GUARD. The two ways an absolute run used to lie, both of
           them silent, which is what made them expensive:

             1. server.py defaulted a missing height prior to 40 m. alpha is
                metres per model unit and that one number sets it for the
                entire scene, so a GeoTIFF uploaded with the field untouched
                came back in confident metres anchored to a guess.
                estimate_elevation was already written to refuse; the server
                never let it.

             2. fetch_dem catches its own failure and returns None, at which
                point the surface is height above LOCAL GROUND - but dsm.tif
                still says MODE=absolute, UNITS=metres, with a valid CRS. The
                only trace was a print() on the server's stdout, which the
                browser never sees. Scored against a sea-level reference that
                reads as a huge model error.

           Part B stubs the pipeline and drives the real server module, so it
           tests the decisions rather than the depth model. It runs last
           because it monkeypatches server's globals.
"""
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)              # mathsandml/
REPO = os.path.dirname(ROOT)
BACKEND = os.path.join(REPO, "backend")
# the science, then the server that calls into it
for _p in (ROOT, BACKEND):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FAILED = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name +
          (("\n          -> " + str(detail)) if not cond else ""))
    if not cond:
        FAILED.append(name)


print("=" * 62)
print("PART A - regressions")
print("=" * 62)
# ===========================================================================
print("\nA1. attenuation() must recommend a gain that REDUCES error")
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
print("\nA2. no spurious BLAS warnings, and singular fits never leak inf")
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
print("\nA3. the benchmark must refuse to calibrate from the reference")
import json  # noqa: E402

src = open(os.path.join(HERE, "run_benchmark.py")).read()
check("the reference-percentile fallback is gone",
      "median of reference nDSM above p" not in src)
check("it errors instead", "no known_height_m in scene.json" in src)

# scenes/ is gitignored - it holds hundreds of megabytes of LiDAR - so a fresh
# clone has no scenes at all. An unguarded listdir here crashed the whole suite
# with a traceback before the later sections ran, which is the opposite of what
# a test that advertises "no weights, no network, runs in seconds" should do.
SCENES = os.path.join(HERE, "scenes")
if not os.path.isdir(SCENES):
    print("  SKIP  no benchmark/scenes yet - run benchmark/fetch_data.py to "
          "download them, then re-run to check the per-scene priors")
for d in sorted(os.listdir(SCENES) if os.path.isdir(SCENES) else []):
    f = os.path.join(SCENES, d, "scene.json")
    if not os.path.exists(f):
        continue
    cfg = json.load(open(f))
    check(f"{d} ships an operator prior",
          bool(cfg.get("known_height_m")), cfg.get("known_height_m"))
    check(f"{d} says where the prior came from",
          "NOT" in str(cfg.get("known_height_source", "")).upper(),
          cfg.get("known_height_source"))

# ===========================================================================
print("\nA4. the no-DEM clip survives refine()")
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
print("\nA5. refine gets the split the pipeline measured, not its own default")
for f in ("server.py", "run_geotiff.py"):
    src = open(os.path.join(BACKEND, f)).read()
    check(f"{f} passes the scene's sigma to refine",
          'object_sigma_m=float(info.get("sigma_m")' in src)
    check(f"{f} re-clips after refine", "clip_below_ground(height, info)" in src)

# ===========================================================================
print("\nA6. the tile grid must not emit the same tile twice")
# Clamping the starts AFTER deduping let two distinct starts collapse onto the
# same clamped origin and both survive: 600x600 ran 9 tiles where 4 cover the
# image, 2352x1222 ran 40 where 28 do. Every duplicate is a full backbone
# forward pass, multiplied again by the rotation ensemble.
import server as _S  # noqa: E402


def _grid(H, W, tile=518, overlap=180):
    step = tile - overlap
    rows = sorted({min(r, max(0, H - tile))
                   for r in (*range(0, max(1, H - overlap), step), max(0, H - tile))})
    cols = sorted({min(c, max(0, W - tile))
                   for c in (*range(0, max(1, W - overlap), step), max(0, W - tile))})
    return rows, cols


for H, W, want in ((600, 600, 4), (700, 700, 4), (519, 519, 4), (2352, 1222, 28)):
    r, c = _grid(H, W)
    check(f"{H}x{W}: no duplicate tile origins",
          len(r) == len(set(r)) and len(c) == len(set(c)), (r, c))
    check(f"{H}x{W}: {want} tiles cover it", len(r) * len(c) == want,
          f"{len(r) * len(c)} tiles")
    check(f"{H}x{W}: the ETA counts the same tiles the pipeline runs",
          _S._tile_count(H, W) == len(r) * len(c),
          f"{_S._tile_count(H, W)} vs {len(r) * len(c)}")
    check(f"{H}x{W}: the grid still covers the last row and column",
          r[-1] == max(0, H - 518) and c[-1] == max(0, W - 518), (r[-1], c[-1]))

# ===========================================================================
print("\nA7. shadow calibration must find its control points, and get the scale right")
# Two bugs stopped this producing ANY pairs, and a third made the ones it did
# produce too tall:
#   * it required shadows to be elongated along the sun, so a wide block
#     casting a shorter shadow (ratio 0.74) was rejected as "too circular";
#   * it read the depth AT the shadow's object end, which is the pavement at
#     the foot of the wall - exactly 0 m above ground;
#   * it measured length across the whole shadow blob, which includes the
#     building's own depth along the sun direction (+71% at azimuth 135).
import inference as I2  # noqa: E402


def _shadow_scene(az, elev, alpha_true=3.0, h_m=30.0, size=30, px=1.0):
    """Twelve flat blocks, geometrically cast shadows, a depth field whose
    scale is known. Recovering alpha_true is the whole test."""
    H = W = 380
    a = np.deg2rad(az)
    dc, dr = -np.sin(a), np.cos(a)
    rgb = np.full((H, W, 3), 180, np.uint8)
    detail = np.zeros((H, W))
    blocks = np.zeros((H, W), bool)
    shadow = np.zeros((H, W), bool)
    for i in range(12):
        r0, c0 = 40 + (i // 3) * 80, 60 + (i % 3) * 90
        blocks[r0:r0 + size, c0:c0 + size] = True
        detail[r0:r0 + size, c0:c0 + size] = h_m / alpha_true
        yy, xx = np.mgrid[r0:r0 + size, c0:c0 + size]
        yy, xx = yy.ravel(), xx.ravel()
        for s in np.arange(1, h_m / np.tan(np.deg2rad(elev)) / px + 1, 0.5):
            rr = np.clip(np.round(yy + dr * s).astype(int), 0, H - 1)
            cc = np.clip(np.round(xx + dc * s).astype(int), 0, W - 1)
            shadow[rr, cc] = True
    shadow &= ~blocks
    rgb[blocks] = 230
    rgb[shadow] = 40
    return detail, rgb


for az, elev in ((90, 45), (135, 50), (315, 60)):
    d, rgb = _shadow_scene(az, elev)
    a_est, diag = I2.alpha_from_shadows(d, rgb, az, elev, 1.0)
    check(f"az {az}/{elev}: finds control points", a_est is not None,
          diag.get("rejected"))
    check(f"az {az}/{elev}: recovers the known scale within 10%",
          a_est is not None and abs(a_est / 3.0 - 1) < 0.10,
          f"alpha={a_est} against a true 3.0")

# ===========================================================================
print("\nA8. the held-out shadow error must be scored against its own truth")
# e is concatenated fold by fold (idx[f::folds]); reading the truth back as
# Y[idx[:e.size]] paired every residual with an unrelated control point.
_X = np.linspace(1, 10, 60)
_Y = 3.0 * _X
_d = I2._cv_ratio(_X, _Y)
check("a perfectly linear set has zero relative error",
      _d.get("median_rel_pct", 1) < 1e-6, _d.get("median_rel_pct"))

# ===========================================================================
print("\nA9. flipping the surface must leave a trace outside stdout")
_info = {}
_rgb = np.zeros((64, 64, 3), np.uint8)
_rgb[16:48, 16:48] = 240                      # bright roof
_p = np.zeros((64, 64))
_p[16:48, 16:48] = -5.0                       # ...reading LOW: inverted
I2.validate_depth(_p, _rgb, _info)
check("inversion is recorded in info", _info.get("depth_inverted") is True, _info)
check("the correlation it decided on is recorded too",
      isinstance(_info.get("luma_depth_r"), float), _info)
_info2 = {}
I2.validate_depth(-_p, _rgb, _info2)
check("a healthy scene records that it was checked and left alone",
      _info2.get("depth_inverted") is False, _info2)

print()
print("=" * 62)
print("PART B - the datum guard")
print("=" * 62)

# the same server module part A borrowed _tile_count from, now driven for real
import server as S  # noqa: E402

RGB = np.zeros((32, 32, 3), np.uint8)


def fake_load(mode="absolute", sun=False):
    def _l(path):
        return RGB, dict(
            path=path, mode=mode, transform=None, bounds=None,
            crs="EPSG:28992" if mode == "absolute" else None,
            px_size_m=0.5 if mode == "absolute" else None,
            sun_azimuth=145.0 if sun else None,
            sun_elevation=52.0 if sun else None)
    return _l


def stub_pipeline(calibration):
    """Everything downstream of load_image, replaced with something instant."""
    def est(path, **kw):
        return (np.ones((32, 32)) * 5.0,
                dict(mode="absolute", crs="EPSG:28992", px_size_m=0.5),
                dict(mode="absolute", calibration=calibration, alpha=12.345,
                     dem_debias_m=3.07))
    S.estimate_elevation = est
    S.refine = lambda h, rgb, **kw: (h, {})
    S.export_products = lambda h, rgb, meta, out, info, **kw: info
    mesh = types.SimpleNamespace(metadata={"z_exaggeration": 1.5, "base_m": 0.0},
                                 faces=[0, 1, 2])
    S.build_mesh = lambda *a, **k: mesh
    S.export_mesh = lambda m, p: p
    S.slope_map = lambda h, px: np.zeros_like(h)


def run(params, mode="absolute", sun=False,
        calibration="freqsplit (DEM terrain + scaled detail)", src="source.tif"):
    S.load_image = fake_load(mode, sun)
    stub_pipeline(calibration)
    jid = "test%08x" % (abs(hash(str(params) + calibration + str(sun) + mode)) & 0xFFFFFFFF)
    os.makedirs(os.path.join(S.JOBS_DIR, jid), exist_ok=True)
    S.JOBS[jid] = dict(id=jid, status="queued", progress=0.0, log=[],
                       result=None, error=None, source=src)
    S._run(jid, "/tmp/" + src, params)
    return S.JOBS[jid]


print("\nB1. a georeferenced image with no scale anchor must refuse, not guess")
j = run({"style": "smooth", "known_height_m": ""})
check("refuses", j["status"] == "error", j.get("error"))
check("names the fix in the message",
      "tallest structure" in (j.get("error") or "").lower(), j.get("error"))

print("\nB2. ...unless the GeoTIFF's own tags carry sun angles")
j = run({"style": "smooth", "known_height_m": ""}, sun=True)
check("proceeds", j["status"] == "done", j.get("error"))
check("says so in the job log", any("sun angles" in l for l in j["log"]), j["log"])

print("\nB3. with an anchor: sea-level datum, DEM and alpha both visible")
j = run({"style": "smooth", "known_height_m": 40})
check("proceeds", j["status"] == "done", j.get("error"))
check("datum reported as sea level", j["result"]["datum"] == "sea level",
      j["result"].get("datum"))
check("terrain source in the log", any("COP30" in l for l in j["log"]), j["log"])
check("alpha in the log", any("alpha=12.345" in l for l in j["log"]), j["log"])

print("\nB4. DEM fetch failed: the output means something else, and says so")
j = run({"style": "smooth", "known_height_m": 40},
        calibration="scale-only (no DEM; heights above local ground)")
check("datum reported as local ground", j["result"]["datum"] == "local ground",
      j["result"].get("datum"))
check("warning reaches the browser log",
      any("LOCAL GROUND" in l for l in j["log"]), j["log"])

print("\nB5. a plain PNG has no metric scale to get wrong - it still runs")
S.load_image = fake_load("relative")
stub_pipeline("")
S.estimate_elevation = lambda path, **kw: (
    np.linspace(0, 1, 32 * 32).reshape(32, 32),
    dict(mode="relative", crs=None, px_size_m=None), dict(mode="relative"))
jid = "testrel"
os.makedirs(os.path.join(S.JOBS_DIR, jid), exist_ok=True)
S.JOBS[jid] = dict(id=jid, status="queued", progress=0.0, log=[], result=None,
                   error=None, source="source.png")
S._run(jid, "/tmp/source.png", {"style": "smooth", "known_height_m": "", "gsd_m": 0.5})
j = S.JOBS[jid]
check("runs with no anchor", j["status"] == "done", j.get("error"))
check("datum reported as relative", j["result"]["datum"] == "relative",
      j["result"].get("datum"))
check("falls back to the 60 m full scale",
      j["result"]["info"].get("relative_full_scale_m") == 60.0,
      j["result"]["info"].get("relative_full_scale_m"))

print("\n" + ("ALL PASS" if not FAILED else "FAILED: " + ", ".join(FAILED)))
sys.exit(1 if FAILED else 0)
