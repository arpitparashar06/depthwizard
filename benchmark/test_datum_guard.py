"""test_datum_guard.py - the two ways an absolute run used to lie, locked down.

Run:  python benchmark/test_datum_guard.py     (needs the venv, not the model)

Both failures this covers were silent, which is what made them expensive:

  1. server.py defaulted a missing height prior to 40 m. alpha is metres per
     model unit and that one number sets it for the entire scene, so a GeoTIFF
     uploaded with the field untouched came back in confident metres that were
     anchored to a guess. estimate_elevation was already written to refuse;
     the server never let it.

  2. fetch_dem catches its own failure and returns None, at which point the
     surface is height above LOCAL GROUND - but dsm.tif still says
     MODE=absolute, UNITS=metres, with a valid CRS. The only trace was a
     print() on the server's stdout, which the browser never sees. Scored
     against a sea-level reference that reads as a huge model error.

The pipeline is stubbed. This tests the decisions, not the depth model.
"""
import os, sys, types
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import server as S

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


FAILED = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name +
          (("\n          -> " + str(detail)) if not cond else ""))
    if not cond:
        FAILED.append(name)


print(__doc__.splitlines()[0])
print("\n1. a georeferenced image with no scale anchor must refuse, not guess")
j = run({"style": "smooth", "known_height_m": ""})
check("refuses", j["status"] == "error", j.get("error"))
check("names the fix in the message",
      "tallest structure" in (j.get("error") or "").lower(), j.get("error"))

print("\n2. ...unless the GeoTIFF's own tags carry sun angles")
j = run({"style": "smooth", "known_height_m": ""}, sun=True)
check("proceeds", j["status"] == "done", j.get("error"))
check("says so in the job log", any("sun angles" in l for l in j["log"]), j["log"])

print("\n3. with an anchor: sea-level datum, DEM and alpha both visible")
j = run({"style": "smooth", "known_height_m": 40})
check("proceeds", j["status"] == "done", j.get("error"))
check("datum reported as sea level", j["result"]["datum"] == "sea level",
      j["result"].get("datum"))
check("terrain source in the log", any("COP30" in l for l in j["log"]), j["log"])
check("alpha in the log", any("alpha=12.345" in l for l in j["log"]), j["log"])

print("\n4. DEM fetch failed: the output means something else, and says so")
j = run({"style": "smooth", "known_height_m": 40},
        calibration="scale-only (no DEM; heights above local ground)")
check("datum reported as local ground", j["result"]["datum"] == "local ground",
      j["result"].get("datum"))
check("warning reaches the browser log",
      any("LOCAL GROUND" in l for l in j["log"]), j["log"])

print("\n5. a plain PNG has no metric scale to get wrong - it still runs")
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
