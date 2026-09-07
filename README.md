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

Or skip the browser entirely — one image, one command:

```bash
python run_geotiff.py benchmark/scenes/ahn_rotterdam_centre/rgb.tif --tallest 40
```

It writes the same products the UI does, prints what the heights are measured
from, and takes `--reference <lidar.tif>` to score the result in the same run.
`--help` lists the calibration flags (`--tallest`, `--gcp`, `--sun`).

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
| `run_geotiff.py` | one-command CLI: image in, DSM + nDSM + DTM + glTF out, no browser |
| `server.py` | FastAPI: jobs, progress, files, validation. Replaces the Gradio app |
| `inference.py` | load → depth → detrend → calibrate → DSM / nDSM / DTM |
| `buildings.py` | morphological ground, footprint extraction, extrusion, facades |
| `mesh_builder.py` | heightfield and city meshes, glTF export |
| `refine.py` | guided filter, structure flattening, edge sharpening |
| `validate.py` | RMSE / MAE / r against reference LiDAR, split by landscape |
| `web/` | React + Vite front end with the three.js viewer |
| `benchmark/test_datum_guard.py` | regression test: a georeferenced run must refuse to guess its scale, and must say what its heights are measured from |
| `benchmark/test_regressions.py` | regression test: the alpha-gain estimator, the singular-fit guards, the leakage refusal, the post-refine clip |

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
5. **The suggested alpha-gain is the one that minimises error.** It used to be
   `median(reference / predicted)` over a mask that required the reference to
   have a structure but let the prediction through at a quarter of that. Every
   pixel the model half-missed contributed a huge ratio, so the statistic
   measured miss rate and pointed the wrong way: on Rotterdam it advised x2.83
   when the error-minimising gain was x0.74, and taking its advice moved
   object-band RMSE from 6.06 m to 10.83 m. It is now through-origin least
   squares, which *is* the argmin of squared error, and the report refuses to
   recommend any gain that does not measurably help.
6. **Segmentation by roof plateau.** Connected components merge every touching
   roof — a real downtown tile returned 3 footprints. Marking pixels within one
   height step of the local maximum, then growing them, returned 5/5 on
   separated blocks and 96/96 on a dense grid.
7. **Real vertical walls.** A grid mesh cannot represent a vertical face, so
   every roofline becomes a 45° ramp. Stepped geometry emits a flat quad per
   cell joined by true vertical faces, quantised so the surface stays watertight.

## Tests

Neither needs the model weights or the network; both run in seconds.

```bash
python benchmark/test_datum_guard.py    # 13 checks
python benchmark/test_regressions.py    # 34 checks
python benchmark/selftest.py            # the whole harness on synthetic scenes
```

## Known limits

- Absolute scale needs one real measurement. A coarse DEM cannot supply it —
  the ramp lives at low frequency, exactly where SRTM is blind (measured
  alpha −14.9 against a true 81.3). So a georeferenced image **stops and asks**
  rather than falling back on a default: that one number sets metres-per-unit
  for the whole scene, and a guessed 40 m returns confident metres anchored to
  nothing. A GeoTIFF whose tags carry sun angles calibrates itself from shadows
  and needs no input.
- When the coarse DEM cannot be fetched (no key, no network, rate limited) the
  pipeline still runs, but the surface is then height above **local ground**,
  not above sea level, while the GeoTIFF tags still read `MODE=absolute`. The
  job log warns and the results panel reports `Measured from: LOCAL GROUND`.
  Score that against an nDSM, never against an absolute DSM.
- Relative mode has no metric anchor. "Tallest structure" sets the full-scale
  height; without it the scene defaults to 60 m.
- Forest returns canopy, not ground. The forest row of the stratified table is
  where that shows up.
- Roof height is the 75th percentile inside a footprint, so a pitched roof
  becomes flat at about eaves-plus.
- Buildings that touch and share a height merge into one prism.
