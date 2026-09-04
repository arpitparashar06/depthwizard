"""
server.py - FastAPI backend. Replaces the Gradio app entirely.

Gradio was doing three jobs badly at once: HTTP server, UI framework, and file
host. Splitting them means the React front end owns the interface, this file
owns the pipeline, and the 3D viewer stops being an iframe pointing at a second
localhost port.

    uvicorn server:app --reload --port 8000

Nothing in inference.py, buildings.py, mesh_builder.py or validate.py changed
to support this - they were already plain functions. Only the shell moved.

Jobs run on a worker thread with a progress log, because a Large-backbone run
on CPU takes minutes and a blocking request would time out in the browser.
"""

import os
import io
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

from inference import estimate_elevation, load_image, export_products
from mesh_builder import build_mesh, build_city, export_mesh, slope_map
from refine import refine
import validate as V

ROOT = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = os.path.join(ROOT, "jobs")
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


def _set(job_id, **kw):
    with LOCK:
        JOBS[job_id].update(kw)


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

        kh = gcps = az = el = None
        if mode == "absolute":
            src = params.get("scale_source", "known_height")
            if src == "known_height":
                kh = float(params.get("known_height_m") or 40)
            elif src == "gcps":
                gcps = params.get("gcps") or []
                if len(gcps) < 2:
                    raise ValueError("ground control points need at least two entries")
            else:
                az = float(params.get("sun_azimuth") or 0)
                el = float(params.get("sun_elevation") or 45)

        _set(job_id, progress=0.15)
        _log(job_id, f"running depth backbone ({os.environ.get('DEPTH_MODEL', 'Large')})")
        height, meta, info = estimate_elevation(
            src_path, known_height_m=kh, gcps=gcps,
            sun_azimuth=az, sun_elevation=el,
            use_dem=bool(params.get("use_dem", True)),
            alpha_gain=float(params.get("alpha_gain") or 1.0),
            outdir=out)

        px = meta.get("px_size_m") or float(params.get("gsd_m") or 0.5)
        if mode == "relative":
            # relative depth is 0..1; give it plausible metres before anything
            # downstream reasons in metres
            scale = float(params.get("known_height_m") or 0) or RELATIVE_FULL_SCALE_M
            height = height * scale
            meta = dict(meta, px_size_m=px)
            info["relative_full_scale_m"] = scale
            _log(job_id, f"relative mode scaled to {scale:.0f} m full range")

        _set(job_id, progress=0.6)
        rinfo = {}
        if params.get("flatten", 0.8) or params.get("sharpen", 0.4):
            _log(job_id, "refining against image edges")
            height, rinfo = refine(height, rgb, px_size_m=px,
                                   flatten=float(params.get("flatten", 0.8)),
                                   sharpen=float(params.get("sharpen", 0.4)))
            if rinfo.get("applied") is False:
                _log(job_id, "refinement declined - it reduced edge crispness")

        # The surface has moved since estimate_elevation exported it: a relative
        # scene was rescaled into metres, and refine may have altered it. Write
        # the rasters again so dsm.tif, ndsm.tif, height16.png and the mesh are
        # all the SAME surface - otherwise the elevation map on disk and the 3D
        # model disagree, and validation scores a raster nobody ever sees.
        info = export_products(height, rgb, meta, out, info)
        _log(job_id, "products re-exported from the final surface")

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
    if j is None:
        raise HTTPException(404, "no such job")
    return j


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

    clean = {k: v for k, v in rep.items() if k != "_arrays"}
    return JSONResponse(json.loads(json.dumps(clean, default=float)))


@app.get("/api/health")
def health():
    return {"ok": True, "model": os.environ.get(
        "DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Large-hf")}


# Serve the built front end if it exists, so production is one process.
_dist = os.path.join(ROOT, "web", "dist")
if os.path.isdir(_dist):
    app.mount("/", StaticFiles(directory=_dist, html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", 8000)))
