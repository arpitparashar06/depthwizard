"""
server.py - the web API. Takes an upload, runs the pipeline, serves the results.

===========================================================================
READ THIS FIRST
===========================================================================
Five endpoints, and one function that does all the work:

    POST /api/jobs                  upload an image + settings -> {job_id}
                                    starts _run() on a background thread and
                                    returns immediately
    GET  /api/jobs/{id}             status, progress 0..1, log lines, result
                                    the browser polls this about once a second
    GET  /api/jobs/{id}/files/{n}   download dsm.tif, terrain.glb, a figure...
    POST /api/jobs/{id}/validate    upload reference LiDAR -> accuracy report
    GET  /api/jobs/{id}/layers      analysis layers the viewer can drape
    GET  /api/jobs/{id}/profile     exact heights along a line (r0,c0 -> r1,c1)
    GET  /api/jobs/{id}/probe       exact heights at one pixel
    GET  /api/health                which model checkpoint is loaded

    _run(job_id, path, params)      THE WHOLE PIPELINE, in order:
        load_image()                 what did we just get, and does it have
                                     coordinates
        [decide the scale source]    landmark height / control points / sun
                                     angles - and REFUSE to run a georeferenced
                                     image without one
        estimate_elevation()         inference.py: the elevation map
        [rescale]                    a PNG has no metres, so stretch 0..1 onto
                                     a nominal full-scale height
        refine()                     refine.py: domes -> flat roofs
        clip_below_ground()          nothing sits below its own ground
        export_products()            rewrite the rasters from the FINAL surface
        build_city()                 mesh_builder.py: the .glb
        [write result.json]          so a server restart does not lose the job

WHY A BACKGROUND THREAD. A Large-backbone run on a laptop CPU takes minutes.
An HTTP request that waits for it times out in the browser, so the job runs on
a worker thread and the browser watches its progress log instead.

WHY THE JOB FOLDER IS THE DATABASE. Every product lands in jobs/<id>/, result
included. There is nothing to set up, and a finished run survives a restart.

    Run it:  python backend/server.py       (or: uvicorn server:app --reload)
"""

import os
import io
import logging
import json
import shutil
import uuid
import threading
import traceback
import numpy as np

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


# _bootstrap puts ../mathsandml on the import path and reads the repo-root
# .env. It has to come first: inference binds OPENTOPO_KEY at import time, so
# a late .env read leaves the key empty, fetch_dem is refused, and every
# georeferenced run quietly returns height above local ground while still
# tagging the GeoTIFF MODE=absolute.
from _bootstrap import ROOT, JOBS_DIR, FRONTEND_DIST, load_env  # noqa: E402

load_env()

from inference import (estimate_elevation, load_image, export_products,  # noqa: E402
                       clip_below_ground, ghsl_height_prior, resolve_engine,
                       engine_needs_anchor, ENGINE_LABEL, refine_by_default)
from mesh_builder import build_mesh, build_city, export_mesh, slope_map  # noqa: E402
from refine import refine                                               # noqa: E402
import validate as V                                                    # noqa: E402
import layers as LY                                                     # noqa: E402

os.makedirs(JOBS_DIR, exist_ok=True)

RELATIVE_FULL_SCALE_M = 60.0

app = FastAPI(title="DepthWizard API")
app.add_middleware(
    CORSMiddleware,
    # Vite dev server runs on 5173; in production the build is served from here
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"], allow_headers=["*"],
)

JOBS = {}
LOCK = threading.Lock()


class _QuietPolling(logging.Filter):
    """Drop the status-poll requests from uvicorn's access log.

    The browser polls GET /api/jobs/{id} roughly once a second for the whole
    run, so a five-minute job buries every line of real pipeline progress under
    three hundred identical 200 OKs. Everything else still logs normally, and
    DEPTHWIZARD_LOG_POLLS=1 puts them back for debugging.
    """
    def filter(self, record):
        if os.environ.get("DEPTHWIZARD_LOG_POLLS"):
            return True
        msg = record.getMessage()
        return not ('GET /api/jobs/' in msg and 'files/' not in msg)


logging.getLogger("uvicorn.access").addFilter(_QuietPolling())


def _set(job_id, **kw):
    with LOCK:
        JOBS[job_id].update(kw)


# Measured: a 600x600 tile is 9 tiles, 4 passes, ~64 s end to end on a laptop
# CPU with the Large backbone - so about 1.5 s per model run once the mesh and
# refine stages are accounted for separately.
SECONDS_PER_TILE = 1.5


def _tile_count(H, W, tile=518, overlap=180):
    """Same arithmetic predict_depth uses, so the estimate matches reality."""
    if H <= tile and W <= tile:
        return 1
    step = tile - overlap
    # clamp then dedupe, exactly as predict_depth does - see the note there
    rows = {min(r, max(0, H - tile))
            for r in (*range(0, max(1, H - overlap), step), max(0, H - tile))}
    cols = {min(c, max(0, W - tile))
            for c in (*range(0, max(1, W - overlap), step), max(0, W - tile))}
    return len(rows) * len(cols)


def _parse_gcps(raw):
    """[[row, col, height_m], ...] or [{row, col, height_m}, ...] -> tuples.

    The browser sends JSON, so a control point arrives as a list or an object
    depending on who wrote the form. alpha_from_gcps unpacks three values per
    entry, so normalise here rather than making the science module guess.
    """
    out = []
    for g in (raw or []):
        if isinstance(g, dict):
            r = g.get("row"); c = g.get("col")
            h = g.get("height_m", g.get("h", g.get("height")))
        else:
            try:
                r, c, h = g
            except (TypeError, ValueError):
                raise ValueError(f"a ground control point needs row, col and "
                                 f"height - got {g!r}")
        if r in (None, "") or c in (None, "") or h in (None, ""):
            continue
        out.append((int(float(r)), int(float(c)), float(h)))
    return out


def _log(job_id, msg):
    with LOCK:
        JOBS[job_id]["log"].append(msg)
        JOBS[job_id]["log"] = JOBS[job_id]["log"][-200:]
    print(f"[{job_id[:6]}] {msg}")


# ---------------------------------------------------------------------------
def _run(job_id, src_path, params):
    out = os.path.join(JOBS_DIR, job_id)
    try:
        _set(job_id, status="running", progress=0.05)
        _log(job_id, "reading image")
        rgb, meta = load_image(src_path)
        mode = meta["mode"]
        _log(job_id, f"{rgb.shape[1]}x{rgb.shape[0]} px, mode={mode}")

        # What the conditioning chain did to the image before the model saw
        # it. Band reordering and cloud masking change the result, so they
        # belong in the run's own log, not only in meta.json.
        try:
            import preprocess as PRE
            _prep = PRE.summarise(meta.get("preprocess") or {})
            if _prep:
                _log(job_id, f"preprocessing: {_prep}")
        except Exception:
            pass

        # Cost scales with tile count, not pixel count, and people quite
        # reasonably assume a stalled-looking terminal means a crash. A
        # 2352x1222 image is 40 tiles against a 600x600 tile's 9, so the same
        # settings take minutes rather than one. Say so before it starts.
        rots = int(params.get("rotations") or 4)
        engine = resolve_engine(params.get("engine"))
        if engine != "zeroshot":
            _log(job_id, f"engine: {ENGINE_LABEL[engine]}" + (
                "" if engine_needs_anchor(engine) else
                " - no scale anchor required; any given is used as a check"))
        if engine == "zeroshot":
            n_tiles = _tile_count(rgb.shape[0], rgb.shape[1])
            est = n_tiles * rots * SECONDS_PER_TILE
            _log(job_id, f"{n_tiles} tiles x {rots} passes = {n_tiles * rots} model "
                         f"runs, roughly {est / 60:.0f}-{est * 1.6 / 60:.0f} min on CPU")
            if est > 180:
                _log(job_id, "large image - drop rotations to 1, or use the Small "
                             "backbone (DEPTH_MODEL=...-Small-hf), to go faster")

        kh = gcps = az = el = None
        prior_source, prior_info = "person", None
        auto_after_sun = False
        if mode == "absolute":
            src = params.get("scale_source", "known_height")
            if src == "known_height":
                # No made-up default here. alpha is metres per model unit and
                # this one number sets it for the whole scene, so quietly
                # falling back to 40 m rescales every elevation in the output
                # while the UI still reports metres. In order: the person's
                # number, the sun angles in the GeoTIFF tags, then GHSL's
                # measured building height at these coordinates (ghsl.py).
                # Only when none of those exists does the run stop and ask.
                raw = params.get("known_height_m")
                kh = float(raw) if raw not in (None, "") else 0.0
                if kh <= 0:
                    kh = None
                    auto = params.get("auto_prior", True) not in (
                        False, "false", 0, "0")
                    if not engine_needs_anchor(engine):
                        # the trained model sets the scale itself (hybrid: the
                        # learned prior; metric: directly); GHSL and the sun
                        # tags are still consulted inside, as fallback / check
                        auto_after_sun = auto
                    elif (meta.get("sun_azimuth") is not None
                            and meta.get("sun_elevation") is not None):
                        # the shadows answer first; GHSL stays the fallback
                        # inside estimate_elevation if they find too few pairs
                        auto_after_sun = auto
                        _log(job_id, "no height prior given - calibrating from "
                                     "the sun angles in the GeoTIFF tags")
                    else:
                        if auto:
                            _log(job_id, "no height prior given - looking up GHSL "
                                         "building height for these coordinates")
                            prior, why = ghsl_height_prior(meta)
                        else:
                            prior, why = None, "the automatic prior is switched off"
                        if prior is None:
                            raise ValueError(
                                "This image is georeferenced, so the output is in "
                                "metres - and metres need a scale anchor. Enter the "
                                "height of the tallest structure you can identify "
                                "in the scene, or supply ground control points or "
                                "sun angles. The automatic prior could not help: "
                                f"{why}.")
                        kh = float(prior["known_height_m"])
                        prior_source, prior_info = "ghsl", prior
                        _log(job_id, f"scale from GHSL: buildings under this scene "
                                     f"reach about {kh:g} m ({prior['n_built']} of "
                                     f"{prior['n_cells']} 100 m cells built). A prior, "
                                     f"not a measurement - enter the tallest "
                                     f"structure to override it")
            elif src == "gcps":
                gcps = _parse_gcps(params.get("gcps"))
                if len(gcps) < 2:
                    raise ValueError(
                        "Ground control points need at least two entries, each "
                        "row / column / height-above-ground in metres. Two "
                        "points fix the multiplier; more only make it steadier.")
                _log(job_id, f"scale from {len(gcps)} ground control points")
            elif src == "sun":
                az, el = params.get("sun_azimuth"), params.get("sun_elevation")
                if az in (None, "") or el in (None, ""):
                    az, el = meta.get("sun_azimuth"), meta.get("sun_elevation")
                if az is None or el is None:
                    raise ValueError(
                        "Shadow calibration needs the sun azimuth and elevation. "
                        "Landsat, Sentinel and most commercial products carry "
                        "them in the GeoTIFF tags; this file does not, so enter "
                        "them or pick another scale source.")
                az, el = float(az), float(el)
                if not (0.0 <= az <= 360.0 and 1.0 <= el <= 89.0):
                    raise ValueError(f"sun angles out of range: azimuth {az}, "
                                     f"elevation {el}")
                _log(job_id, f"scale from shadows, sun at {az:g} / {el:g}")
            else:
                raise ValueError(f"unknown scale_source {src!r} - expected "
                                 f"'known_height', 'gcps' or 'sun'")

        _set(job_id, progress=0.15)
        _log(job_id, f"running {ENGINE_LABEL[engine]}" if engine != "zeroshot" else
                     f"running depth backbone ({os.environ.get('DEPTH_MODEL', 'Large')})")
        height, meta, info = estimate_elevation(
            src_path, known_height_m=kh, gcps=gcps,
            rotations=int(params.get("rotations") or 4),
            adaptive_sigma=bool(params.get("adaptive_sigma", True)),
            height_reference=params.get("height_reference") or "tallest",
            debias_coarse_dem=bool(params.get("debias_coarse_dem", True)),
            fuse_scale=bool(params.get("fuse_scale", False)),
            sun_azimuth=az, sun_elevation=el,
            prior_source=prior_source, prior_info=prior_info,
            # decided above; the only second chance is GHSL after tag shadows
            # that found too few pairs, and the run log says so if it happens
            auto_prior=auto_after_sun,
            use_dem=bool(params.get("use_dem", True)),
            alpha_gain=float(params.get("alpha_gain") or 1.0),
            outdir=out, engine=engine,
            gsd_hint=float(params.get("gsd_m") or 0) or None)

        # The coarse-DEM fetch happens deep inside estimate_elevation and its
        # failure is caught there, so without this the only trace is the
        # server's stdout. It matters: with no DEM the surface is height above
        # LOCAL GROUND while dsm.tif still carries MODE=absolute and
        # UNITS=metres. Anyone scoring that against a sea-level reference sees
        # a datum-sized error and blames the depth model.
        if mode == "absolute":
            calib = info.get("calibration", "")
            if "no DEM" in calib:
                _log(job_id, "WARNING: coarse DEM unavailable - the surface is "
                             "height above LOCAL GROUND, not above sea level. "
                             "Score it against an nDSM, not an absolute DSM.")
            else:
                dbg = info.get("dem_debias_m")
                _log(job_id, f"terrain baseline from {info.get('dem_source') or 'COP30'}" +
                     (f", rooftop bias removed ({dbg:.2f} m)"
                      if isinstance(dbg, (int, float)) else ""))
            if info.get("engine") in ("metric", "hybrid"):
                _log(job_id, f"scale: {info.get('scale_source')}")
                chk = info.get("model_vs_anchors") or {}
                if chk:
                    _log(job_id, "independent checks of the model's metres: " +
                         ", ".join(f"{k} x{v:.2f}" for k, v in chk.items()) +
                         " (x1.00 = agree)")
            elif isinstance(info.get("alpha"), (int, float)):
                _log(job_id, f"scale: alpha={info['alpha']:.3f} m per model unit"
                             + (f" - from {info['scale_source']}"
                                if info.get("scale_source") else ""))
            fus = info.get("scale_fusion") or {}
            if fus.get("applied"):
                _log(job_id, f"scale fused from {fus['n_sources']} sources "
                             f"(+/-{fus['alpha_sigma_rel']*100:.0f}%); the "
                             f"priority path would have used "
                             f"{info.get('alpha_priority', float('nan')):.3f}")
            elif fus.get("refused"):
                _log(job_id, f"WARNING: scale fusion refused - {fus['reason']}. "
                             f"Kept the tightest single source "
                             f"({fus.get('tightest_source')}).")

        px = meta.get("px_size_m") or float(params.get("gsd_m") or 0.5)
        if mode == "relative":
            # relative depth is 0..1; give it plausible metres before anything
            # downstream reasons in metres
            # a typed height wins; else what HeightNet measured (metres at the
            # assumed GSD); else the fixed nominal range
            scale = (float(params.get("known_height_m") or 0)
                     or float(info.get("suggested_full_scale_m") or 0)
                     or RELATIVE_FULL_SCALE_M)
            height = height * scale
            meta = dict(meta, px_size_m=px)
            info["relative_full_scale_m"] = scale
            if meta.get("_uncertainty") is not None:
                # the uncertainty is in the same units as the surface, so it
                # has to travel with it through the rescale
                meta["_uncertainty"] = meta["_uncertainty"] * scale
            _log(job_id, f"relative mode scaled to {scale:.0f} m full range")

        _set(job_id, progress=0.6)
        rinfo = {}
        want = params.get("refine")
        do_refine = (refine_by_default(info.get("engine") or "zeroshot") if want in (None, "", "auto")
                     else want not in (False, "false", 0, "0", "off"))
        if not do_refine:
            _log(job_id, "edge clean-up skipped: the fine-tuned model's edges are "
                         "already where the image's are, and the clean-up smears them")
        elif params.get("flatten", 0.8) or params.get("sharpen", 0.4):
            _log(job_id, "refining against image edges")
            # Use the split the pipeline measured for THIS scene. refine's own
            # 15 m default is narrower than the buildings in a dense tile, so
            # the middle of a large footprint reads as terrain and never gets
            # flattened - the exact failure inference.py's frequency split
            # exists to avoid. refine still self-checks and declines if the
            # result is less crisp than what it was handed.
            height, rinfo = refine(height, rgb, px_size_m=px,
                                   object_sigma_m=float(info.get("sigma_m") or 15.0),
                                   flatten=float(params.get("flatten", 0.8)),
                                   sharpen=float(params.get("sharpen", 0.4)))
            if rinfo.get("applied") is False:
                _log(job_id, "refinement declined - it reduced edge crispness")

        # refine's unsharp pass overshoots at rooflines, so the no-DEM clip has
        # to be re-applied to the surface that actually gets exported
        height, info = clip_below_ground(height, info)

        # The surface has moved since estimate_elevation exported it: a relative
        # scene was rescaled into metres, and refine may have altered it. Write
        # the rasters again so dsm.tif, ndsm.tif, height16.png and the mesh are
        # all the SAME surface - otherwise the elevation map on disk and the 3D
        # model disagree, and validation scores a raster nobody ever sees.
        info = export_products(height, rgb, meta, out, info,
                               uncertainty=meta.get("_uncertainty"))
        _log(job_id, "products re-exported from the final surface")
        try:
            LY.write_layers(out, height, px, uncertainty=meta.get("_uncertainty"),
                            units="m*" if mode == "relative" else "m")
            _log(job_id, "analysis layers written (height, slope, uncertainty)")
        except Exception as ex:           # a layer is a nicety, never a failure
            _log(job_id, f"analysis layers skipped: {ex}")

        _set(job_id, progress=0.75)
        style = params.get("style", "city")
        z_exag = float(params.get("z_exaggeration") or 1.5)
        grid = int(params.get("target_grid") or 256)

        if style == "city":
            import buildings as B
            # Issue 8 fix: always derive nDSM via morphological opening.
            # The freq-split ndsm.tif uses a Gaussian low-pass for the ground
            # surface, while buildings.py uses grey_opening. The two produce
            # different ground estimates, so a building's "height above ground"
            # differs between extraction and extrusion, causing double-counting
            # or half-height structures. Using a single consistent method
            # (morphological opening) everywhere avoids this.
            nd = B.normalised_height(height, px, max_building_m=90.0)
            _log(job_id, "deriving nDSM by morphological ground filter")
            mesh, cinfo = build_city(height, nd, rgb, px_size_m=px,
                                     target_grid=grid, z_exaggeration=z_exag,
                                     is_relative=False)
            rinfo.update(cinfo)
            glb = os.path.join(out, "terrain.glb")
            mesh.export(glb)
            md = dict(mesh.metadata)
            tris = cinfo["ground_tris"] + cinfo["roof_tris"] + cinfo["wall_tris"]
            _log(job_id, f"{cinfo['n']} buildings extruded")
        else:
            mesh = build_mesh(height, rgb, px_size_m=px, target_grid=grid,
                              z_exaggeration=z_exag, style=style, is_relative=False)
            glb = export_mesh(mesh, os.path.join(out, "terrain.glb"))
            md = dict(mesh.metadata)
            tris = len(mesh.faces)

        slope = slope_map(height, px)
        result = dict(
            mode=mode, width=int(rgb.shape[1]), height=int(rgb.shape[0]),
            px_size_m=float(px),
            # what the numbers are measured FROM - the one fact a reader needs
            # before quoting any elevation out of this run
            datum=("local ground" if "no DEM" in info.get("calibration", "")
                   else "sea level" if mode == "absolute" else "relative"),
            # where the metre scale came from - a person, the file, or GHSL
            scale_source=info.get("scale_source"),
            min_m=float(np.nanmin(height)), max_m=float(np.nanmax(height)),
            relief_m=float(np.nanmax(height) - np.nanmin(height)),
            median_slope_deg=float(np.nanmedian(slope)),
            p99_slope_deg=float(np.nanpercentile(slope, 99)),
            triangles=int(tris), style=style,
            z_exaggeration=float(md.get("z_exaggeration", 1.0)),
            base_m=float(md.get("base_m", 0.0)),
            info={k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                  for k, v in {**info, **rinfo}.items()
                  if not isinstance(v, np.ndarray)},
            files=[f for f in sorted(os.listdir(out)) if not f.startswith(".")],
        )
        # JOBS is an in-memory dict, so a server restart used to lose every
        # finished job while the browser kept polling its id forever. The
        # products are all on disk already; writing the result beside them
        # makes the job recoverable.
        with open(os.path.join(out, "result.json"), "w") as f:
            json.dump(result, f, default=float)

        _set(job_id, status="done", progress=1.0, result=result)
        _log(job_id, "done")

    except Exception as e:
        traceback.print_exc()
        _set(job_id, status="error", error=f"{type(e).__name__}: {e}")
        _log(job_id, f"failed: {e}")


# ---------------------------------------------------------------------------
@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...), params: str = Form("{}")):
    try:
        p = json.loads(params)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"params is not valid JSON: {e}")

    job_id = uuid.uuid4().hex
    out = os.path.join(JOBS_DIR, job_id)
    os.makedirs(out, exist_ok=True)
    # keep the original extension: load_image decides relative vs absolute from
    # it, and a GeoTIFF renamed to .bin loses its CRS handling
    ext = os.path.splitext(file.filename or "")[1].lower() or ".png"
    src = os.path.join(out, f"source{ext}")
    with open(src, "wb") as f:
        shutil.copyfileobj(file.file, f)

    with LOCK:
        JOBS[job_id] = dict(id=job_id, status="queued", progress=0.0,
                            log=[], result=None, error=None,
                            source=os.path.basename(src))
    threading.Thread(target=_run, args=(job_id, src, p), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    with LOCK:
        j = JOBS.get(job_id)
    if j is not None:
        return j

    # Not in memory. Either this process was restarted, or the id is wrong.
    # A finished run left result.json behind, so it can still be served.
    out = os.path.join(JOBS_DIR, job_id)
    done = os.path.join(out, "result.json")
    if os.path.exists(done):
        with open(done) as f:
            result = json.load(f)
        return dict(id=job_id, status="done", progress=1.0, result=result,
                    error=None, log=["recovered from disk after a restart"],
                    source=next((n for n in sorted(os.listdir(out))
                                 if n.startswith("source")), None))

    # The folder exists but there is no result: the run was interrupted before
    # it finished. Say so, rather than 404ing forever while the browser polls.
    if os.path.isdir(out):
        raise HTTPException(410, "This run was interrupted before it finished - "
                                 "the server stopped while it was working. "
                                 "Upload the image again to restart it.")

    raise HTTPException(404, "no such job")


@app.get("/api/jobs/{job_id}/files/{name}")
def job_file(job_id: str, name: str):
    # basename() so a crafted name cannot walk out of the job directory
    path = os.path.join(JOBS_DIR, job_id, os.path.basename(name))
    if not os.path.exists(path):
        raise HTTPException(404, "no such file")
    return FileResponse(path)


@app.post("/api/jobs/{job_id}/validate")
async def validate_job(job_id: str, reference: UploadFile = File(...),
                       sigma_m: float = Form(15.0)):
    out = os.path.join(JOBS_DIR, job_id)
    pred = os.path.join(out, "dsm.tif")
    if not os.path.exists(pred):
        raise HTTPException(404, "run a job first")

    ref = os.path.join(out, "reference" + (os.path.splitext(
        reference.filename or "")[1].lower() or ".tif"))
    with open(ref, "wb") as f:
        shutil.copyfileobj(reference.file, f)

    rgb = next((os.path.join(out, f) for f in os.listdir(out)
                if f.startswith("source")), None)
    try:
        rep, refarr, rgbarr = V.validate_files(pred, ref, rgb, outdir=out,
                                               object_sigma_m=float(sigma_m))
        V.error_figures(rep, refarr, rgbarr, outdir=out)
        V.save_report(rep, out)
    except Exception as e:
        raise HTTPException(400, f"{type(e).__name__}: {e}")
    # the reference on the estimate's own grid, so the viewer's profile and
    # probe can show LiDAR beside the estimate, and the error as a 3D layer
    try:
        LY.write_reference_grid(out, refarr)
        LY.add_error_layer(out, rep["_arrays"]["error"],
                           offset_m=float(rep["headline"].get("offset", 0.0) or 0.0))
    except Exception as ex:
        print(f"[validate] error layer skipped: {ex}")

    clean = {k: v for k, v in rep.items() if k != "_arrays"}
    return JSONResponse(json.loads(json.dumps(clean, default=float)))


def _job_px(out):
    """Pixel size of a finished job, from what the run itself recorded."""
    for name in ("result.json", "meta.json"):
        try:
            with open(os.path.join(out, name)) as f:
                v = json.load(f).get("px_size_m")
            if v:
                return float(v)
        except Exception:
            pass
    return 1.0


def _job_dir(job_id):
    out = os.path.join(JOBS_DIR, os.path.basename(job_id))
    if not os.path.exists(os.path.join(out, "dsm.tif")):
        raise HTTPException(404, "run a job first")
    return out


@app.get("/api/jobs/{job_id}/layers")
def job_layers(job_id: str):
    out = _job_dir(job_id)
    p = os.path.join(out, "layers.json")
    if not os.path.exists(p):
        # a run from before layers existed - build them now from its rasters
        try:
            dsm, _ = LY._read(os.path.join(out, "dsm.tif"))
            unc, _ = LY._read(os.path.join(out, "uncertainty.tif"))
            with open(os.path.join(out, "result.json")) as f:
                units = "m*" if json.load(f).get("mode") == "relative" else "m"
            LY.write_layers(out, dsm, _job_px(out), uncertainty=unc, units=units)
        except Exception as e:
            raise HTTPException(500, f"could not build layers: {e}")
    with open(p) as f:
        return json.load(f)


@app.get("/api/jobs/{job_id}/profile")
def job_profile(job_id: str, r0: float, c0: float, r1: float, c1: float, n: int = 256):
    out = _job_dir(job_id)
    return JSONResponse(json.loads(json.dumps(
        LY.profile(out, r0, c0, r1, c1, n=n, px_size_m=_job_px(out)), default=float)))


@app.get("/api/jobs/{job_id}/probe")
def job_probe(job_id: str, r: float, c: float):
    return LY.probe(_job_dir(job_id), r, c)


@app.get("/api/health")
def health():
    try:
        engine = resolve_engine()
    except Exception as e:
        engine = f"unavailable: {e}"
    out = {"ok": True, "engine": engine, "model": os.environ.get(
        "DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Large-hf")}
    if engine in ENGINE_LABEL and engine != "zeroshot":
        import heightnet as HN
        out["heightnet"] = HN.default_checkpoint()
        out["engine_label"] = ENGINE_LABEL[engine]
    return out


# Serve the built front end if it exists, so production is one process.
if os.path.isdir(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", 8000)))
