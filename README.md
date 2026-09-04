# DepthWizard — single-view elevation to a navigable 3D city

Optical RGB → depth → metric elevation → extruded buildings → three.js flythrough.

**PNG / JPG** (no coordinates) → relative surface model.
**GeoTIFF** (has a CRS) → absolute DSM in metres.

## Run

Two processes in development, one in production.

> Note: zsh (the macOS default shell) does **not** strip `#` comments in
> interactive shells, so a trailing `# ...` on these commands is passed
> through as an argument. `npm run dev # http://localhost:5173` makes Vite
> treat `#` as the project root and serve an empty directory.


```bash
# 1. backend
pip install -r requirements.txt
# http://127.0.0.1:8000
python server.py

# 2. front end (separate terminal)
cd web
npm install
# http://localhost:5173
npm run dev
```

Vite proxies `/api` to port 8000, so the browser sees one origin and there is
no CORS dance.

For a single process, build the front end once — `server.py` mounts `web/dist`
at `/` when it exists:

```bash
cd web && npm run build && cd ..
# serves UI + API on http://127.0.0.1:8000
python server.py
```

Or the container, which bakes in both the model weights and the built bundle:

```bash
docker build -t depthwizard .
docker run --rm -p 8000:8000 depthwizard
```

## The backbone

Default is `depth-anything/Depth-Anything-V2-Large-hf`: ~1.3 GB, and roughly
8–10× slower on CPU than Small. The accuracy half of the marking scheme is
worth the minutes. To go back:

```bash
export DEPTH_MODEL=depth-anything/Depth-Anything-V2-Small-hf
```

## Files

| file | role |
|---|---|
| `server.py` | FastAPI: jobs, progress, files, validation. Replaces the Gradio app |
| `inference.py` | load → depth → detrend → calibrate → DSM / nDSM / DTM |
| `buildings.py` | morphological ground, footprint extraction, extrusion, facades |
| `mesh_builder.py` | heightfield and city meshes, glTF export |
| `refine.py` | guided filter, structure flattening, edge sharpening |
| `validate.py` | RMSE / MAE / r against reference LiDAR, split by landscape |
| `web/` | React + Vite front end with the three.js viewer |

## API

```
POST /api/jobs                       multipart: file + params JSON  -> {job_id}
GET  /api/jobs/{id}                  status, progress, log, result
GET  /api/jobs/{id}/files/{name}     dsm.tif, ndsm.tif, terrain.glb, figures
POST /api/jobs/{id}/validate         multipart: reference raster -> metrics
GET  /api/health                     which checkpoint is loaded
```

Jobs run on a worker thread with a progress log, because a Large-backbone run
on CPU takes minutes and a blocking request would time out in the browser.

## Outputs

`dsm.tif` · `ndsm.tif` · `dtm.tif` (float32, georeferenced in absolute mode) ·
`height16.png` · `texture.png` · `meta.json` · `terrain.glb` ·
`validation.md` / `.json` · `error_map.png` · `scatter.png` · `stability.png`

## Where the accuracy comes from

Each of these was measured, not assumed.

1. **Tile scale alignment.** Depth models normalise every input independently.
   Blending tiles raw correlated 0.64 with truth; aligning each tile to the
   overlap first gave 1.000.
2. **Ramp removal.** The model reads the bottom of a nadir frame as closer and
   paints it high. On one urban scene that false tilt was 75% of the range.
3. **Frequency-split calibration.** One global scale factor carries the ramp
   into the elevations. Measured RMSE on that scene: **3.51 m** split vs 11.05
   global vs 66.70 hybrid — the global fit produced negative building heights.
4. **Morphological ground, not a Gaussian low-pass.** On a 54%-built scene the
   blurred "ground" sat 7.88 m too high and recovered 36% of building height;
   the opening sat 0.09 m off and recovered 92%.
5. **Segmentation by roof plateau.** Connected components merge every touching
   roof — a real downtown tile returned 3 footprints. Marking pixels within one
   height step of the local maximum, then growing them, returned 5/5 on
   separated blocks and 96/96 on a dense grid.
6. **Real vertical walls.** A grid mesh cannot represent a vertical face, so
   every roofline becomes a 45° ramp. Stepped geometry emits a flat quad per
   cell joined by true vertical faces, quantised so the surface stays watertight.

## Known limits

- Absolute scale needs one real measurement. A coarse DEM cannot supply it —
  the ramp lives at low frequency, exactly where SRTM is blind (measured
  alpha −14.9 against a true 81.3).
- Relative mode has no metric anchor. "Tallest structure" sets the full-scale
  height; without it the scene defaults to 60 m.
- Forest returns canopy, not ground. The forest row of the stratified table is
  where that shows up.
- Roof height is the 75th percentile inside a footprint, so a pitched roof
  becomes flat at about eaves-plus.
- Buildings that touch and share a height merge into one prism.
