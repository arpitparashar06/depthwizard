"""Every bug found in review, locked down."""
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)              # mathsandml/
REPO = os.path.dirname(ROOT)
BACKEND = os.path.join(REPO, "backend")
# The science, then the server that calls into it
for _p in (ROOT, BACKEND):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ["DEPTHWIZARD_ENGINE"] = "zeroshot"

FAILED = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name +
          (("\n          -> " + str(detail)) if not cond else ""))
    if not cond:
        FAILED.append(name)


print("=" * 62)
print("PART A - regressions")
print("=" * 62)
print("\nA1. attenuation() must recommend a gain that REDUCES error")
import validate as V  # noqa: E402

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

# A clean surface that is genuinely 0.7x should recover 1/0.7 = 1.43
clean_ref = np.concatenate([np.full(4000, 20.0), np.zeros(20000)])
at2 = V.attenuation(clean_ref * 0.7, clean_ref)
check("recovers a known pure scale error",
      abs(at2["suggested_alpha_gain"] - 1 / 0.7) < 0.02,
      f"got {at2['suggested_alpha_gain']:.3f}, want {1/0.7:.3f}")

print("\nA2. no spurious BLAS warnings, and singular fits never leak inf")
import warnings  # noqa: E402
import refine as R  # noqa: E402
import buildings as B  # noqa: E402

# The contract is: never hand back a coefficient vector that is not finite.
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

print("\nA3. the benchmark must refuse to calibrate from the reference")
import json  # noqa: E402

src = open(os.path.join(HERE, "run_benchmark.py")).read()
check("the reference-percentile fallback is gone",
      "median of reference nDSM above p" not in src)
check("it errors instead", "no known_height_m in scene.json" in src)

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

print("\nA5. refine gets the split the pipeline measured, not its own default")
for f in ("server.py", "run_geotiff.py"):
    src = open(os.path.join(BACKEND, f)).read()
    check(f"{f} passes the scene's sigma to refine",
          'object_sigma_m=float(info.get("sigma_m")' in src)
    check(f"{f} re-clips after refine", "clip_below_ground(height, info)" in src)

print("\nA6. the tile grid must not emit the same tile twice")
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

print("\nA7. shadow calibration must find its control points, and get the scale right")
import inference as I2  # noqa: E402


def _shadow_scene(az, elev, alpha_true=3.0, h_m=30.0, size=30, px=1.0):
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

print("\nA8. the held-out shadow error must be scored against its own truth")
_X = np.linspace(1, 10, 60)
_Y = 3.0 * _X
_d = I2._cv_ratio(_X, _Y)
check("a perfectly linear set has zero relative error",
      _d.get("median_rel_pct", 1) < 1e-6, _d.get("median_rel_pct"))

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

print("\nA10. fusing scale sources must respect how much each one is worth")
# The point of inverse-variance fusion is that a measured source beats a guess.


def _cc(**kw):
    return dict(estimates={k: v[0] for k, v in kw.items()},
                sigma_rel={k: v[1] for k, v in kw.items()})


f = I2.fuse_scale_estimates(_cc(shadow=(3.0, 0.05), prior=(4.5, 0.25)))
check("it fuses when the sources are compatible", f.get("applied") is True, f)
check("the tight source dominates the vague one",
      f["alpha"] is not None and abs(f["alpha"] - 3.0) < 0.25,
      f"fused {f.get('alpha')}, a plain average would be 3.75")
check("the plain average would have been materially worse",
      abs(f["alpha"] - 3.0) < abs(3.75 - 3.0) / 2,
      f"fused {f['alpha']:.3f} vs average 3.75, truth-ish 3.0")
check("the fused estimate is tighter than either input",
      f["alpha_sigma_rel"] < 0.05, f.get("alpha_sigma_rel"))

f2 = I2.fuse_scale_estimates(_cc(a=(2.0, 0.10), b=(0.5, 0.10)), max_z=1e9)
check("equal-weight fusion is multiplicative (x2 and x0.5 cancel to 1.0)",
      f2.get("alpha") is not None and abs(f2["alpha"] - 1.0) < 1e-9,
      f"got {f2.get('alpha')}, arithmetic mean would be 1.25")
f2b = I2.fuse_scale_estimates(_cc(a=(2.0, 0.10), b=(1.8, 0.10)))
check("...and it holds for a pair the guard actually allows",
      abs(f2b["alpha"] - np.sqrt(2.0 * 1.8)) < 1e-9,
      f"got {f2b.get('alpha')}, want {np.sqrt(2.0*1.8):.6f}")

# The guard: a gap too big for the error bars is a broken source, not noise
f3 = I2.fuse_scale_estimates(_cc(shadow=(3.0, 0.05), prior=(9.0, 0.10)))
check("it REFUSES when two sources disagree beyond their error bars",
      f3.get("applied") is False and f3.get("refused") is True, f3)
check("...and says which pair disagreed",
      set(f3.get("disagreeing", [])) == {"shadow", "prior"}, f3.get("disagreeing"))
check("...and names the tightest source to fall back on",
      f3.get("tightest_source") == "shadow", f3.get("tightest_source"))
check("...and returns no alpha, so the caller keeps what it had",
      f3.get("alpha") is None, f3.get("alpha"))

# One source is not a fusion
f4 = I2.fuse_scale_estimates(_cc(prior=(4.0, 0.25)))
check("a single source is left alone", f4.get("applied") is False and
      f4.get("alpha") is None, f4)
check("...and it says why", "fewer than two" in f4.get("reason", ""), f4.get("reason"))

# Guessed sun angles must carry a wider error bar than tags-derived ones
d, rgb = _shadow_scene(135, 50)
_, tag_diag = I2.alpha_from_shadows(d, rgb, 135, 50, 1.0, sun_sigma_deg=0.0)
_, guess_diag = I2.alpha_from_shadows(d, rgb, 135, 50, 1.0,
                                      sun_sigma_deg=I2.SUN_GUESS_SIGMA_DEG)
check("angles read from the file's tags are treated as a measurement",
      tag_diag["alpha_sigma_rel"] < guess_diag["alpha_sigma_rel"],
      (tag_diag["alpha_sigma_rel"], guess_diag["alpha_sigma_rel"]))
check("a 5-degree guess costs roughly the 18% the docstring claims",
      0.12 < guess_diag["alpha_sigma_rel"] < 0.30,
      guess_diag["alpha_sigma_rel"])

print("\nA11. control points must fit through the origin, and never flip the scene")
_det = np.zeros((64, 64))
_det[10:20, 10:20] = 0.40          # a tall roof
_det[40:50, 40:50] = 0.20
# True alpha 100: 0.40 -> 40 m, 0.20 -> 20 m
_a, _s = I2.alpha_from_gcps(_det, [(15, 15, 40.0), (45, 45, 20.0)], with_sigma=True)
check("two consistent points recover the known scale",
      _a is not None and abs(_a - 100.0) < 1.0, f"alpha={_a}")
check("...and it reports an uncertainty for the fusion to weigh",
      _s is not None and 0 < _s < 1.0, _s)
_a2, _s2 = I2.alpha_from_gcps(_det, [(15, 15, 20.0), (45, 45, 40.0)],
                              with_sigma=True)
check("points that rank backwards can no longer produce a negative alpha",
      _a2 is None or _a2 > 0, f"got {_a2}")
check("...and their disagreement shows up as a much wider error bar",
      _a2 is None or _s2 > 5 * _s, f"consistent {_s:.3f} vs backwards {_s2:.3f}")

_det2 = np.zeros((64, 64))
_det2[10:20, 10:20] = -0.50
_det2[40:50, 40:50] = 0.10
_flip = I2.alpha_from_gcps(_det2, [(15, 15, 40.0), (45, 45, 5.0)])
check("a control point that implies a negative scale is REFUSED",
      _flip is None, f"got {_flip}")
_cc_neg = dict(estimates={"gcps": -123.0, "prior": 113.0},
               sigma_rel={"gcps": 0.05, "prior": 0.25})
check("a negative estimate never reaches the fusion",
      I2.fuse_scale_estimates(_cc_neg).get("alpha") is None)

# Fusion must stay off unless asked for
import inspect  # noqa: E402
check("fuse_scale defaults to off",
      inspect.signature(I2.estimate_elevation).parameters["fuse_scale"].default is False)

print("\nA12. the depth cache: one backbone run per image, never a stale answer")
import tempfile as _tf  # noqa: E402
_cdir = _tf.mkdtemp(prefix="dw-cache-")
_old_root, _old_on = I2.CACHE_ROOT, I2.DEPTH_CACHE
I2.CACHE_ROOT, I2.DEPTH_CACHE = _cdir, True
_calls = []
_real_inner = I2._predict_depth_ensemble


def _fake_inner(rgb, tile, overlap, n, verbose):
    _calls.append(n)
    g = np.random.default_rng(len(_calls))
    return g.random(rgb.shape[:2]) * 3.14159, g.random(rgb.shape[:2])


I2._predict_depth_ensemble = _fake_inner
_img = (np.random.default_rng(0).random((24, 30, 3)) * 255).astype(np.uint8)
_a = I2.predict_depth_ensemble(_img, rotations=4, cache_key="k1", verbose=False)
_b = I2.predict_depth_ensemble(_img, rotations=4, cache_key="k1", verbose=False)
check("second call with the same key does not run the backbone", len(_calls) == 1, _calls)
check("a cached result is bit-identical to the fresh one",
      np.array_equal(_a[0], _b[0]) and np.array_equal(_a[1], _b[1]))
I2.predict_depth_ensemble(_img, rotations=2, cache_key="k1", verbose=False)
I2.predict_depth_ensemble(_img, rotations=4, cache_key="k2", verbose=False)
check("a different rotation count or a different file misses the cache", len(_calls) == 3, _calls)
_stub_saved = I2.predict_depth
I2.predict_depth = lambda *a, **k: None
I2.predict_depth_ensemble(_img, rotations=4, cache_key="k1", verbose=False)
check("a stubbed backbone bypasses the cache entirely (the tests must not "
      "leave entries a real run could find)", len(_calls) == 4, _calls)
I2.predict_depth = _stub_saved
I2._predict_depth_ensemble = _real_inner
I2.CACHE_ROOT, I2.DEPTH_CACHE = _old_root, _old_on

# The key follows the conditioning code, not only the file
_f = os.path.join(_cdir, "x.png")
from PIL import Image as _Im  # noqa: E402
_Im.fromarray(_img).save(_f)
check("source_key is stable for the same file", I2.source_key(_f) == I2.source_key(_f))

print("\nA13. a cached or local DEM that covers the scene is used without a key")
import rasterio as _rio  # noqa: E402
from rasterio.transform import from_bounds as _fb  # noqa: E402
_demdir = _tf.mkdtemp(prefix="dw-dem-")
with _rio.open(os.path.join(_demdir, "local.tif"), "w", driver="GTiff", height=20, width=20,
               count=1, dtype="float32", crs="EPSG:4326",
               transform=_fb(4.0, 51.0, 5.0, 52.0, 20, 20)) as _ds:
    _ds.write(np.full((1, 20, 20), 3.0, np.float32))
_old_dirs, _old_key = I2.DEM_DIRS, I2.OPENTOPO_KEY
I2.DEM_DIRS, I2.OPENTOPO_KEY = [_demdir], ""
from rasterio.crs import CRS as _CRS  # noqa: E402
_meta = dict(mode="absolute", crs=_CRS.from_epsg(4326),
             bounds=(4.40, 51.40, 4.45, 51.45))
_info = {}
_got = I2.fetch_dem(_meta, os.path.join(_demdir, "out.tif"), _info)
check("a DEM on disk covering the scene is used, no key needed",
      _got is not None and "local file" in _info.get("dem_source", ""), _info)
_meta2 = dict(_meta, bounds=(10.0, 10.0, 10.1, 10.1))
_info2 = {}
check("a DEM that does not cover the scene is not used",
      I2.fetch_dem(_meta2, os.path.join(_demdir, "out2.tif"), _info2) is None, _info2)
check("...and the reason is recorded", "no cached DEM" in _info2.get("dem_source", ""), _info2)
I2.DEM_DIRS, I2.OPENTOPO_KEY = _old_dirs, _old_key

print("\nA14. fetch_data never overwrites a prior someone typed into scene.json")
import fetch_data as _FD  # noqa: E402
_sd = _tf.mkdtemp(prefix="dw-scene-")
_scene = os.path.join(_sd, "ahn_rotterdam_centre")
os.makedirs(_scene)
for _n in ("rgb.tif", "ref_dsm.tif"):
    open(os.path.join(_scene, _n), "w").close()
import json as _json  # noqa: E402
with open(os.path.join(_scene, "scene.json"), "w") as _fh:
    _json.dump({"known_height_m": 40, "known_height_source": "ortho"}, _fh)
_argv = sys.argv
sys.argv = ["fetch_data.py", "--source", "ahn", "--aoi", "rotterdam_centre", "--out", _sd]
_FD.FETCHERS = dict(_FD.FETCHERS, ahn=lambda *a, **k: (_ for _ in ()).throw(
    AssertionError("an existing scene must not be re-downloaded")))
try:
    _rc = _FD.main()
finally:
    sys.argv = _argv
with open(os.path.join(_scene, "scene.json")) as _fh:
    _kept = _json.load(_fh)
check("an existing scene is skipped, not re-downloaded", _rc == 0)
check("its known_height_m survives", _kept.get("known_height_m") == 40, _kept)

print("\nA15. analysis layers and the profile read the rasters, in true units")
import layers as _LY  # noqa: E402
_jd = _tf.mkdtemp(prefix="dw-job-")
_H, _W = 40, 60
_dsm = np.fromfunction(lambda r, c: 100.0 + 0.5 * c, (_H, _W))   # 0.5 m per pixel east
_prof = dict(driver="GTiff", height=_H, width=_W, count=1, dtype="float32",
             crs="EPSG:28992", transform=_fb(0, 0, _W * 0.5, _H * 0.5, _W, _H), nodata=np.nan)
for _n, _arr in (("dsm.tif", _dsm), ("uncertainty.tif", np.full((_H, _W), 0.3))):
    with _rio.open(os.path.join(_jd, _n), "w", **_prof) as _ds:
        _ds.write(_arr.astype(np.float32), 1)
_ls = _LY.write_layers(_jd, _dsm, 0.5, uncertainty=np.full((_H, _W), 0.3))
check("height, slope and uncertainty layers are written",
      {"photo", "height", "slope", "uncertainty"} <= {l["id"] for l in _ls}, [l["id"] for l in _ls])
check("every layer image is the size of the texture",
      all(_Im.open(os.path.join(_jd, l["file"])).size == (_W, _H)
          for l in _ls if l["id"] != "photo"))
_p = _LY.profile(_jd, 10, 0, 10, 40, n=41, px_size_m=0.5)
check("profile length is in metres", abs(_p["length_m"] - 20.0) < 1e-6, _p["length_m"])
check("profile height change matches the surface (0.5 m per pixel x 40 px)",
      abs(_p["dh_m"] - 20.0) < 1e-3, _p.get("dh_m"))
check("profile grade is the true 45 degrees", abs(_p["grade_deg"] - 45.0) < 0.1, _p.get("grade_deg"))
_LY.write_reference_grid(_jd, _dsm - 1.0)
_p2 = _LY.profile(_jd, 10, 0, 10, 40, n=41, px_size_m=0.5)
check("with a reference, the profile reports the offset it removes",
      abs(_p2["vs_reference"]["median_offset_m"] - 1.0) < 1e-3
      and _p2["vs_reference"]["rmse_shift_m"] < 1e-3, _p2.get("vs_reference"))
_spec = _LY.add_error_layer(_jd, np.linspace(-2, 2, _H * _W).reshape(_H, _W), offset_m=0.4)
check("the error layer is diverging and symmetric about zero",
      _spec["kind"] == "diverging" and abs(_spec["vmin"] + _spec["vmax"]) < 1e-9, _spec)
check("re-writing the other layers keeps the error layer",
      "error" in {l["id"] for l in _LY.write_layers(_jd, _dsm, 0.5)})

print("\nA20. the landmark-prior correction is the measured amplitude, and scoped")
_env_gain = os.environ.get("DEPTHWIZARD_PRIOR_GAIN")
check("default correction matches the split it was measured with",
      _env_gain is not None
      or abs(I2.PRIOR_GAIN - (0.80 if I2.SPLIT == "gauss" else 1.0)) < 1e-9,
      (I2.SPLIT, I2.PRIOR_GAIN))
check("applies to a person's tallest-structure number",
      I2.prior_gain("tallest", "person") == I2.PRIOR_GAIN)
check("never to a reference nobody measured",
      I2.prior_gain("typical", "person") == 1.0 and I2.prior_gain("tall") == 1.0)
check("GHSL carries its own factor, not the person's 0.80",
      I2.prior_gain("tallest", "ghsl") == I2.GHSL_PRIOR_GAIN)
check("GHSL's default is no correction - none transferred between scenes",
      os.environ.get("DEPTHWIZARD_GHSL_GAIN") is not None
      or I2.GHSL_PRIOR_GAIN == 1.0, I2.GHSL_PRIOR_GAIN)

print("\nA21. GHSL: the right tile, the right cells, and an honest refusal")
import ghsl as _GH  # noqa: E402
from rasterio.warp import transform as _tr  # noqa: E402
for _name, _lon, _lat, _want in (("Rotterdam", 4.48, 51.915, "R3_C19"),
                                 ("Zurich", 8.54, 47.37, "R4_C19"),
                                 ("Vizianagaram", 83.55, 18.254, "R7_C27"),
                                 ("Delhi", 77.2, 28.6, "R6_C26")):
    _x, _y = _tr("EPSG:4326", _GH.MOLLWEIDE, [_lon], [_lat])
    check(f"{_name} lands in {_want}", _GH.tile_id(_x[0], _y[0]) == _want,
          _GH.tile_id(_x[0], _y[0]))
check("the last column is 1082 km wide",
      _GH.tile_id(18_000_000, 500_000) == "R9_C36"
      and _GH._tile_bounds("R9_C36")[2] == 18_041_000)
check("a box across a tile edge asks for both tiles",
      _GH.tiles_for_bounds(-5000, -5000, 5000, 5000) == ["R9_C19", "R10_C19"])

# A synthetic GHSL file: 10 km of Mollweide around Rotterdam, one 30 m block
_gd = _tf.mkdtemp()
_cx, _cy = _tr("EPSG:4326", _GH.MOLLWEIDE, [4.48], [51.915])
_cx, _cy = round(_cx[0], -2), round(_cy[0], -2)
_gh = np.zeros((100, 100), np.float32)
_gh[48:52, 48:52] = 30.0            # the block under the scene
_gh[48:52, 52:53] = 6.0             # a low neighbour
_gh[0, 0] = 65535
with _rio.open(os.path.join(_gd, f"{_GH.PRODUCT}_V1_0_R3_C19.tif"), "w",
               driver="GTiff", width=100, height=100, count=1, dtype="float32",
               crs=_CRS.from_string(_GH.MOLLWEIDE), nodata=65535,
               transform=_fb(_cx - 5000, _cy - 5000, _cx + 5000, _cy + 5000,
                             100, 100)) as _ds:
    _ds.write(_gh, 1)
_scene = dict(mode="absolute", crs=_CRS.from_string(_GH.MOLLWEIDE),
              bounds=(_cx - 200, _cy - 200, _cx + 500, _cy + 200))
os.environ["GHSL_DIR"] = _gd
_prior, _why = _GH.height_prior(_scene, _tf.mkdtemp(), allow_download=False)
check("reads the block under the scene", _prior is not None
      and 25.0 <= _prior["known_height_m"] <= 30.0, (_prior, _why))
check("...and says how built-up the scene is",
      _prior is not None and 0 < _prior["n_built"] < _prior["n_cells"], _prior)
_empty = dict(_scene, bounds=(_cx - 3000, _cy - 3000, _cx - 2600, _cy - 2600))
_p0, _w0 = _GH.height_prior(_empty, _tf.mkdtemp(), allow_download=False)
check("no buildings -> no prior, and the reason says so",
      _p0 is None and "no buildings" in _w0, _w0)
_flo = np.zeros((100, 100), np.float32)
_flo[48:52, 48:55] = 2.5
_fd = _tf.mkdtemp()
with _rio.open(os.path.join(_fd, f"{_GH.PRODUCT}_V1_0_R3_C19.tif"), "w",
               driver="GTiff", width=100, height=100, count=1, dtype="float32",
               crs=_CRS.from_string(_GH.MOLLWEIDE), nodata=255,
               transform=_fb(_cx - 5000, _cy - 5000, _cx + 5000, _cy + 5000,
                             100, 100)) as _ds:
    _ds.write(_flo, 1)
os.environ["GHSL_DIR"] = _fd
_pf, _wf = _GH.height_prior(_scene, _tf.mkdtemp(), allow_download=False)
check("cells at GHSL's 2.5 m floor are not a height to anchor on",
      _pf is None and "floor" in _wf, _wf)
os.environ["GHSL_DIR"] = _gd
del os.environ["GHSL_DIR"]
_p1, _w1 = _GH.height_prior(_scene, _tf.mkdtemp(), allow_download=False)
check("no tile and no download -> no prior, and it names the fix",
      _p1 is None and "GHSL_DIR" in _w1, _w1)
check("a relative image never asks GHSL",
      _GH.height_prior(dict(mode="relative"), _gd)[0] is None)

print("\nA22. ground comes from an envelope, not from the lowest histogram peak")
_g = np.zeros((200, 200))
_g[:, 90:130] = -8.0
_g[20:60, 20:60] = 10.0              # a 20 m block at 0.5 m/px
_g += 0.3 * np.random.default_rng(0).standard_normal(_g.shape)
_env = I2.ground_envelope(_g, 0.5)
_h = _g - _env
_street = np.zeros_like(_g, bool); _street[120:190, 20:80] = True
_old = (_g - I2.gaussian_filter(_g, 30)); _old = _old - I2.ground_level(_old)
_lift_new = float(np.median(_h[_street])); _lift_old = float(np.median(_old[_street]))
check("streets beside a canal are no longer lifted by the canal depth",
      abs(_lift_new) < 2.0 and _lift_old > 3 * abs(_lift_new),
      f"envelope {_lift_new:.2f} m, old rule {_lift_old:.2f} m")
check("a block narrower than the window keeps its height",
      9.0 < float(np.median(_h[25:55, 25:55])) < 12.0, float(np.median(_h[25:55, 25:55])))
check("the envelope is the default split", I2.SPLIT == "envelope"
      or os.environ.get("DEPTHWIZARD_SPLIT") is not None, I2.SPLIT)

print("\nA23. the DEM debias removes rooftops but never real relief")
_yy, _xx = np.mgrid[0:300, 0:300]
_hill = 60.0 * np.exp(-((_xx - 150) ** 2 + (_yy - 150) ** 2) / (2 * 40.0 ** 2))
_roofs = np.zeros_like(_hill); _roofs[40:60, 40:70] = 3.0
_deb, _ = I2.debias_dem(_hill + _roofs, 1.0)
check("a hill loses at most the cap",
      float(np.max((_hill + _roofs) - _deb)) <= I2.DEM_DEBIAS_MAX_M + 1e-6
      if I2.DEM_DEBIAS_MAX_M is not None else True,
      float(np.max((_hill + _roofs) - _deb)))
check("rooftop clutter on flat ground still comes off",
      float(np.median(_deb[40:60, 40:70])) < 1.5, float(np.median(_deb[40:60, 40:70])))


print("\nA24. the trained engines: A, B and A+B - metres honestly, fallbacks honestly")
try:
    import torch  # noqa: F401
    import heightnet as HN
    _have_torch = True
except Exception as _e:
    _have_torch = False
    print(f"  SKIP  torch not installed ({_e})")
if _have_torch:
    import tempfile
    import rasterio
    from rasterio.transform import from_origin
    from PIL import Image as _Im
    _tmp = tempfile.mkdtemp(prefix="hn_tests_")
    torch.manual_seed(0)
    _ck = os.path.join(_tmp, "tiny.pt")
    HN.save_checkpoint(_ck, HN.build(da_config=HN.tiny_da_config()), "tiny")
    _mr = HN.build(da_config=HN.tiny_da_config())
    with torch.no_grad():
        _mr.da.head.conv3.weight.mul_(1e5)
        _mr.da.head.conv3.bias.fill_(2.0)
    _ckr = os.path.join(_tmp, "tiny_rel.pt")
    HN.save_checkpoint(_ckr, _mr, "tiny")
    _old_env = dict(os.environ)
    os.environ["DEPTHWIZARD_HEIGHTNET"] = _ck
    os.environ["DEPTHWIZARD_CACHE"] = os.path.join(_tmp, "cache")

    check("explicit engines resolve",
          [I2.resolve_engine(e) for e in ("finetuned", "metric", "hybrid", "heightnet", "A")]
          == ["finetuned", "metric", "hybrid", "hybrid", "finetuned"])
    check("the test pin (zeroshot) is honoured", I2.resolve_engine() == "zeroshot")
    os.environ["DEPTHWIZARD_ENGINE"] = "auto"
    check("auto picks the hybrid when a checkpoint exists", I2.resolve_engine() == "hybrid")
    os.environ["DEPTHWIZARD_HEIGHTNET"] = os.path.join(_tmp, "missing.pt")
    check("auto falls back to zero-shot without one", I2.resolve_engine() == "zeroshot")
    try:
        I2.resolve_engine("metric")
        check("forcing a missing model is an error, not a silent fallback", False)
    except ValueError:
        check("forcing a missing model is an error, not a silent fallback", True)
    os.environ["DEPTHWIZARD_HEIGHTNET"] = _ck
    os.environ["DEPTHWIZARD_ENGINE"] = "zeroshot"
    check("A needs an anchor like zero-shot; B and A+B do not",
          I2.engine_needs_anchor("finetuned") and I2.engine_needs_anchor("zeroshot")
          and not I2.engine_needs_anchor("metric") and not I2.engine_needs_anchor("hybrid"))

    _hp = HN.HeightPredictor(_ck, device="cpu")
    _img = (np.random.default_rng(1).random((300, 451, 3)) * 255).astype(np.uint8)
    for _g in (None, 0.2, 0.5, 2.5):
        _r = _hp.predict(_img, _g, rotations=2)
        check(f"predict keeps the input grid at gsd={_g} (B, A, land cover)",
              _r["height"].shape == _img.shape[:2] and _r["seg"].shape == _img.shape[:2]
              and _r["rel"].shape == _img.shape[:2]
              and np.isfinite(_r["height"]).all() and np.isfinite(_r["rel"]).all()
              and (_r["sigma"] >= 0).all(), (_r["height"].shape, _r["rel"].shape))
    _sq = (np.random.default_rng(2).random((518, 518, 3)) * 255).astype(np.uint8)
    _a = _hp.predict(_sq, 0.5, rotations=4)["height"]
    _b = _hp.predict(np.ascontiguousarray(np.rot90(_sq)), 0.5, rotations=4)["height"]
    check("4-rotation ensemble is rotation-equivariant",
          float(np.abs(np.rot90(_a) - _b).max()) < 1e-3,
          float(np.abs(np.rot90(_a) - _b).max()))

    # The combining maths, on answers known exactly
    _det = np.zeros((200, 200)); _det[50:100, 50:100] = 3.0; _det[120:140, 20:180] = 1.0
    check("learned prior recovers a known scale exactly",
          abs(HN.learned_scale(_det, 7.5 * _det) - 7.5) < 1e-9)
    check("learned prior declines a flat scene (nothing to match)",
          HN.learned_scale(np.zeros((50, 50)), np.zeros((50, 50))) is None)
    _fh, _fs, _wb = HN.fuse(np.full(3, 10.0), np.full(3, 1.0), np.full(3, 20.0), np.full(3, 3.0))
    check("fusion is inverse-variance (sigma 1 vs 3 -> 90/10)",
          abs(_fh[0] - 11.0) < 1e-9 and abs(_wb[0] - 0.1) < 1e-9, (_fh, _wb))
    _fh2, _, _wb2 = HN.fuse(np.full(3, 10.0), np.full(3, 1.0), np.full(3, 20.0), np.full(3, 3.0),
                           fusion=dict(c_a=3.0, c_b=1.0))
    check("...and honours the measured honesty factors", abs(_wb2[0] - 0.5) < 1e-9, _wb2)
    _st = dict(mode="stacked", bands=[0, 2, 10, 30], w_a=[0.5, 0.25, 0.0, 0.0])
    _fh3, _, _wb3 = HN.fuse(np.full(4, 10.0), np.ones(4), np.array([1.0, 5.0, 20.0, 40.0]),
                            np.ones(4), _st)
    check("stacked fusion: one weight per band of B's height, B alone for towers",
          np.allclose(_fh3, [5.5, 6.25, 20.0, 40.0]) and np.allclose(_wb3, [0.5, 0.75, 1, 1]),
          (_fh3, _wb3))
    # A checkpoint that a browser unzipped into a folder still loads
    import zipfile as _zf
    _dirck = os.path.join(_tmp, "unzipped.pt")
    with _zf.ZipFile(_ck) as _z:
        _top = _z.namelist()[0].split("/")[0]
        for _n in _z.namelist():
            _dst = os.path.join(_dirck, _n[len(_top) + 1:])
            os.makedirs(os.path.dirname(_dst), exist_ok=True)
            with open(_dst, "wb") as _fh_:
                _fh_.write(_z.read(_n))
    _m1, _ = HN.load_checkpoint(_ck)
    _m2, _ = HN.load_checkpoint(_dirck)
    check("a Safari-unzipped checkpoint folder loads, weights identical",
          HN.available(_dirck) is not None and all(
              torch.equal(x, y) for x, y in zip(_m1.state_dict().values(),
                                                _m2.state_dict().values())))

    # A georeferenced scene with no scale source
    _tif = os.path.join(_tmp, "geo.tif")
    with rasterio.open(_tif, "w", driver="GTiff", width=200, height=160, count=3,
                       dtype="uint8", crs="EPSG:28992",
                       transform=from_origin(84000, 447000, 0.5, 0.5)) as _ds:
        _ds.write((np.random.default_rng(3).random((3, 160, 200)) * 255).astype(np.uint8))
    _out = os.path.join(_tmp, "out")
    _h, _m, _info = I2.estimate_elevation(_tif, use_dem=False, outdir=_out, rotations=2,
                                          engine="metric", auto_prior=False)
    check("B: absolute run needs no anchor",
          _info.get("engine") == "metric" and "HeightNet" in str(_info.get("scale_source")),
          _info.get("scale_source"))
    check("B: no DEM says so, heights are above ground",
          "no DEM" in _info.get("calibration", "") and float(np.nanmin(_h)) >= 0,
          _info.get("calibration"))
    check("land cover exported", os.path.exists(os.path.join(_out, "landcover.png"))
          and os.path.exists(os.path.join(_out, "landcover.tif")))
    _pts = [(40, 50, 2.0 * float(_h[40, 50])), (120, 150, 2.0 * float(_h[120, 150])),
            (80, 20, 2.0 * float(_h[80, 20]))]
    _h2, _, _i2 = I2.estimate_elevation(_tif, use_dem=False, outdir=_out, rotations=2,
                                        engine="metric", auto_prior=False, gcps=_pts)
    check("B: surveyed control points correct the model's metres",
          abs(_i2.get("alpha", 0) - 2.0) < 0.1 and "control points" in _i2["scale_source"],
          (_i2.get("alpha"), _i2.get("scale_source")))
    _hF, _, _iF = I2.estimate_elevation(_tif, use_dem=False, outdir=_out, rotations=2,
                                        engine="hybrid", auto_prior=False)
    check("A+B with a flat relative output falls back to B, and says so",
          _iF.get("engine") == "metric" and _iF.get("hybrid_fallback"),
          (_iF.get("engine"), _iF.get("hybrid_fallback")))
    try:
        I2.estimate_elevation(_tif, use_dem=False, outdir=_out, rotations=1,
                              engine="finetuned", auto_prior=False, known_height_m=12.0)
        check("A refuses to calibrate a flat relative output", False)
    except ValueError as _e:
        check("A refuses to calibrate a flat relative output", "flat" in str(_e), str(_e))
    os.environ["DEPTHWIZARD_HEIGHTNET"] = _ckr
    try:
        I2.estimate_elevation(_tif, use_dem=False, outdir=_out, rotations=1,
                              engine="finetuned", auto_prior=False)
        check("A: with no anchor it refuses, exactly like zero-shot", False)
    except ValueError:
        check("A: with no anchor it refuses, exactly like zero-shot", True)
    _hA, _, _iA = I2.estimate_elevation(_tif, use_dem=False, outdir=_out, rotations=2,
                                        engine="finetuned", auto_prior=False,
                                        known_height_m=12.0)
    check("A: with an anchor it runs the calibration module",
          _iA.get("engine") == "finetuned" and "12" in str(_iA.get("scale_source")),
          _iA.get("scale_source"))
    _hH, _mH, _iH = I2.estimate_elevation(_tif, use_dem=False, outdir=_out, rotations=2,
                                          engine="hybrid", auto_prior=False)
    check("A+B: scale from the learned scene prior, then fused",
          _iH.get("engine") == "hybrid"
          and _iH.get("scale_source") == "learned scene prior (HeightNet metric head)"
          and 0.0 <= _iH.get("fusion_weight_b_mean", -1) <= 1.0,
          (_iH.get("engine"), _iH.get("scale_source"), _iH.get("fusion_weight_b_mean")))
    check("A+B: heights above ground, uncertainty exported",
          float(np.nanmin(_hH)) >= 0 and _mH.get("_uncertainty") is not None
          and os.path.exists(os.path.join(_out, "uncertainty.tif")))
    _png = os.path.join(_tmp, "plain.png")
    _Im.fromarray(_img).save(_png)
    for _e in ("metric", "hybrid"):
        _hr, _mr_, _ir = I2.estimate_elevation(_png, outdir=_out, rotations=1,
                                               engine=_e, gsd_hint=0.4)
        check(f"PNG through {_e}: a 0..1 surface plus its full scale in metres",
              float(np.nanmax(_hr)) <= 1.0 and float(np.nanmin(_hr)) >= 0.0
              and _ir.get("suggested_full_scale_m", 0) > 0,
              (_ir.get("suggested_full_scale_m"), _ir.get("engine")))
    os.environ.clear()
    os.environ.update(_old_env)

print()
print("=" * 62)
print("PART B - the datum guard")
print("=" * 62)

# The same server module part A borrowed _tile_count from, now driven for real
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


GHSL_ANSWER = [(None, "stubbed: offline")]
S.ghsl_height_prior = lambda meta, **kw: GHSL_ANSWER[0]
SEEN = {}


def stub_pipeline(calibration):
    """Everything downstream of load_image, replaced with something instant."""
    def est(path, **kw):
        SEEN.clear()
        SEEN.update(kw)
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


print("\nB1. no scale anchor and no GHSL answer: refuse, never guess")
j = run({"style": "smooth", "known_height_m": ""})
check("refuses", j["status"] == "error", j.get("error"))
check("names the fix in the message",
      "tallest structure" in (j.get("error") or "").lower(), j.get("error"))
check("...and why the automatic prior could not help",
      "stubbed: offline" in (j.get("error") or ""), j.get("error"))

print("\nB1b. no scale anchor but GHSL answers: run, on a labelled prior")
GHSL_ANSWER[0] = (dict(known_height_m=27.5, n_built=6, n_cells=16,
                       statistic="p95", source="stub"), "ok")
j = run({"style": "smooth", "known_height_m": ""})
check("proceeds", j["status"] == "done", j.get("error"))
check("the prior reaches the pipeline as GHSL, not as a person's number",
      SEEN.get("known_height_m") == 27.5 and SEEN.get("prior_source") == "ghsl",
      {k: SEEN.get(k) for k in ("known_height_m", "prior_source")})
check("the job log says the scale is a GHSL prior",
      any("GHSL" in l and "prior" in l for l in j["log"]), j["log"])

print("\nB1c. ...unless the automatic prior is switched off")
j = run({"style": "smooth", "known_height_m": "", "auto_prior": False})
check("refuses", j["status"] == "error", j.get("error"))
GHSL_ANSWER[0] = (None, "stubbed: offline")

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

print("\nB6. the hybrid installed: no anchor needed, nothing refused")
S.resolve_engine = lambda e=None: "hybrid"
GHSL_ANSWER[0] = (None, "stubbed: offline")
j = run({"style": "smooth", "known_height_m": ""})
check("runs without a height prior", j["status"] == "done", j.get("error"))
check("the log says which engine answered",
      any("hybrid A+B" in l for l in j["log"]), j["log"][:6])
check("the engine is passed through", SEEN.get("engine") == "hybrid", SEEN.get("engine"))

print("\nB6b. engine A installed: it still needs an anchor, like zero-shot")
S.resolve_engine = lambda e=None: "finetuned"
j = run({"style": "smooth", "known_height_m": ""})
check("refuses without an anchor", j["status"] == "error", j.get("status"))

print("\nB7. PNG through HeightNet: the model's measured range sets the scale")
S.load_image = fake_load("relative")
stub_pipeline("")
S.estimate_elevation = lambda path, **kw: (
    np.linspace(0, 1, 32 * 32).reshape(32, 32),
    dict(mode="relative", crs=None, px_size_m=None),
    dict(mode="relative", engine="hybrid", suggested_full_scale_m=23.5))
jid = "testrel_hn"
os.makedirs(os.path.join(S.JOBS_DIR, jid), exist_ok=True)
S.JOBS[jid] = dict(id=jid, status="queued", progress=0.0, log=[], result=None,
                   error=None, source="source.png")
S._run(jid, "/tmp/source.png", {"style": "smooth", "known_height_m": "", "gsd_m": 0.5})
j = S.JOBS[jid]
check("uses the model's 23.5 m, not the fixed 60 m",
      j["status"] == "done" and j["result"]["info"].get("relative_full_scale_m") == 23.5,
      (j.get("error"), (j.get("result") or {}).get("info", {}).get("relative_full_scale_m")))

print("\nB8. the old edge clean-up runs for zero-shot only - it smears the fine-tuned edges")
_calls = []
for _eng in ("zeroshot", "hybrid", "finetuned", "metric"):
    S.resolve_engine = (lambda e=None, _x=_eng: _x)
    S.load_image = fake_load("absolute")
    stub_pipeline("freqsplit (DEM terrain + scaled detail)")
    _est = S.estimate_elevation
    S.estimate_elevation = (lambda path, _f=_est, _x=_eng, **kw:
                            (lambda r: (r[0], r[1], dict(r[2], engine=_x)))(_f(path, **kw)))
    S.refine = (lambda h, rgb, _x=_eng, **kw: (_calls.append(_x), (h, {}))[1])
    jid = "testrefine_" + _eng
    os.makedirs(os.path.join(S.JOBS_DIR, jid), exist_ok=True)
    S.JOBS[jid] = dict(id=jid, status="queued", progress=0.0, log=[], result=None,
                       error=None, source="source.tif")
    S._run(jid, "/tmp/source.tif", {"style": "smooth", "known_height_m": 20})
check("refine ran for zero-shot and skipped for A, B and A+B", _calls == ["zeroshot"], _calls)
check("forcing it still works",
      __import__("inference").refine_by_default("zeroshot") and
      not __import__("inference").refine_by_default("hybrid"))

print("\n" + ("ALL PASS" if not FAILED else "FAILED: " + ", ".join(FAILED)))
sys.exit(1 if FAILED else 0)
