# DepthWizard — single-view elevation to a navigable 3D city

Optical RGB → depth → metric elevation → extruded buildings → three.js flythrough.

**PNG / JPG** (no coordinates) → relative surface model.
**GeoTIFF** (has a CRS) → absolute DSM in metres.

## Where things live

Three folders, so you can find the part you are looking for without reading
the rest:

```
backend/       the HTTP API and the command line - FastAPI, jobs, files
mathsandml/    the science - depth, calibration, geometry, scoring, benchmark
frontend/      React + Vite + three.js: upload, controls, flythrough, report
training/      fine-tune the trained engine (HeightNet) on GAMUS + LiDAR - runs on Kaggle
models/        heightnet.pt goes here once trained (the app picks it up by itself)
```

The traffic is one-way. `mathsandml/` imports nothing from `backend/`, so every
piece of the science runs, tests and benchmarks on its own; `backend/` puts
`mathsandml/` on the import path (`backend/_bootstrap.py`) and calls into it;
`frontend/` only ever speaks HTTP.

| folder | file | role |
|---|---|---|
| `backend/` | `server.py` | FastAPI: jobs, progress, files, validation |
| | `run_geotiff.py` | one-command CLI: image in, DSM + nDSM + DTM + glTF out |
| | `_bootstrap.py` | finds `mathsandml/`, reads `.env` before anything imports it |
| `mathsandml/` | `inference.py` | load → depth → detrend → calibrate → DSM / nDSM / DTM |
| | `buildings.py` | morphological ground, footprints, extrusion, facades |
| | `mesh_builder.py` | heightfield and city meshes, glTF export |
| | `refine.py` | guided filter, structure flattening, edge sharpening |
| | `validate.py` | RMSE / MAE / r against reference LiDAR, split by landscape |
| | `layers.py` | height / slope / uncertainty / error layers, exact profiles and probes |
| | `ghsl.py` | automatic scale prior: GHSL building height under the scene |
| | `heightnet.py` | the fine-tuned model: relative depth (A) + metres (B) + land cover; learned prior and fusion |
| `training/` | `prepare_gamus.py` / `prepare_lidar.py` | GAMUS and Dutch/Swiss LiDAR to training tiles |
| | `train.py` / `evaluate.py` | fine-tune (1-2 GPUs, resumable) and score on held-out tiles |
| | `kaggle/*.ipynb`, `TRAINING.md` | the two notebooks to run, and the step-by-step guide |
| | `benchmark/` | scene fetcher, cache warmer, harness, report, and the regression tests |
| `frontend/` | `src/App.jsx` | controls: scale source, geometry, layers, the run itself |
| | `src/Viewer.jsx` | three.js flythrough, layers, click-to-measure, true-metre readouts |
| | `src/components/` | dropzone, progress, results, profile chart, validation panel |
| root | `start.command` / `start.sh` / `start.bat` | one double-click: set up, then serve on :8000 |
| | `run_on_mac.command` | fetch the benchmark scenes and warm the caches (needs the internet once) |
| | `src/styles.css` | the whole look, in the order the interface is built |

## Run

**One double-click.** `start.command` on macOS, `start.bat` on Windows,
`./start.sh` on Linux. The first run creates `.venv`, installs
`requirements.txt` and downloads the depth model once; every run after that
starts in seconds, opens the browser at http://127.0.0.1:8000, and works with
no internet (see *Caches and offline use* below). The built interface ships in
`frontend/dist`, so Node is not needed to run it.

A run's address carries its id (`?job=...`), so a reload, a crashed tab or a
restarted server comes straight back to the same result.

For development it is two processes, one in production.

> Note: zsh (the macOS default shell) does **not** strip `#` comments in
> interactive shells, so a trailing `# ...` on these commands is passed
> through as an argument. `npm run dev # http://localhost:5173` makes Vite
> treat `#` as the project root and serve an empty directory.


```bash
# 1. backend  (from the repo root)
pip install -r requirements.txt
# http://127.0.0.1:8000
python backend/server.py

# 2. front end (separate terminal)
cd frontend
npm install
# http://localhost:5173
npm run dev
```

Vite proxies `/api` to port 8000, so the browser sees one origin and there is
no CORS dance.

Or skip the browser entirely — one image, one command:

```bash
python backend/run_geotiff.py \
    mathsandml/benchmark/scenes/ahn_rotterdam_centre/rgb.tif --tallest 40
```

It writes the same products the UI does, prints what the heights are measured
from and where the metre scale came from, and takes `--reference <lidar.tif>`
to score the result in the same run. `--help` lists the calibration flags
(`--tallest`, `--gcp`, `--sun`). With none of them, a GeoTIFF uses the GHSL
building height at its coordinates (see *Where the scale comes from*).

For a single process, build the front end once — `server.py` mounts
`frontend/dist` at `/` when it exists:

```bash
cd frontend && npm run build && cd ..
# serves UI + API on http://127.0.0.1:8000
python backend/server.py
```

Or the container, which bakes in both the model weights and the built bundle:

```bash
docker build -t depthwizard .
docker run --rm -p 8000:8000 depthwizard
```

## Analysis in the viewer

- **Layers.** The photo on the ground and roofs can be swapped for *height*,
  *slope*, *uncertainty* (disagreement between the four rotated passes) and,
  after a validation, *error* (estimate minus reference). Same UVs, same mesh -
  only the texture changes - with the colour key in the top-right corner. The
  switcher sits beside the tabs; `L` cycles layers while flying. Magnitudes are
  one hue light-to-dark; the error is blue (too low) - grey - red (too high).
  Never a rainbow, which would paint terraces the data does not have.
- **Exact readouts.** Each probe pin is converted to source-image pixels and the
  server answers from the GeoTIFFs, not from the display mesh: DSM, height above
  ground, uncertainty, and the reference LiDAR once one has been uploaded.
- **Elevation profile.** Two pins draw a profile under the viewer: estimate,
  uncertainty band and reference on one metre axis, length, height change,
  grade, steepest 2 m, and RMSE against the reference along that line. The
  same line is draped on the terrain, and the chart's crosshair moves a marker
  along it. `Table (CSV)` downloads the samples.

## Caches and offline use

The depth model runs **once per image**. `cache/depth` holds every prediction,
keyed on the input file, the conditioning code, the checkpoint and the tiling,
so a changed input can never be served a stale answer. Re-running a scene with
a different scale source, gain or mesh style skips the backbone.

The coarse DEM downloads **once per place**. Every OpenTopography download is
kept in `cache/dem`, and any GeoTIFF DEM that covers a scene - in `cache/dem`
or in a folder named by `DEM_DIR` (CartoDEM, SRTM, COP30 tiles) - is used
before the network is tried, with no key needed. Where the terrain came from
is written into the run log and `meta.json` (`dem_source`).

To prepare a demo machine while it still has a connection:

```bash
python mathsandml/benchmark/warm_cache.py demo1.tif demo2.tif   # model + DEM, once each
```

After that the demo images run with the network unplugged, in seconds.

## Engines

The zero-shot pipeline below is one of four engines. The other three use the
fine-tuned model (`training/`, installed as `models/heightnet.pt`), which has
two outputs: **A**, relative depth fine-tuned for top-down images, and **B**,
metres above ground.

| engine | what runs | needs a scale anchor? |
|---|---|---|
| `zeroshot` | Depth-Anything-V2 as released + the calibration module below | yes |
| `finetuned` (A) | fine-tuned relative depth + the same calibration module | yes |
| `metric` (B) | fine-tuned metres directly | no |
| `hybrid` (A+B) | A through the calibration module, scaled by the **learned scene prior** (B) when no measured anchor exists, then fused with B per pixel by uncertainty | no |

The hybrid keeps the problem statement's workflow - relative depth, then a
conversion module to metres (DEM + GCPs / shadows / scene-level prior) - and
uses B as the best scene prior available. With a checkpoint installed the app,
CLI and `/api/health` default to `hybrid`; without one, to `zeroshot`.
`DEPTHWIZARD_ENGINE=zeroshot|finetuned|metric|hybrid` forces one, and
`python mathsandml/benchmark/compare_engines.py` scores all of them on the 9
LiDAR scenes. How to train: `training/TRAINING.md`.

## Where the scale comes from

A depth model only ranks heights; one number - alpha, metres per model unit -
turns the ranking into metres. For a georeferenced image the run takes it from
the first of these that exists, and writes which one into the log, `meta.json`
and the results panel (`Scale from`):

1. **Shadows** - sun angles typed in, or read from the GeoTIFF tags.
2. **Ground control points** - two or more pixels of known height.
3. **The tallest structure** a person can identify, in metres.
4. **GHSL, automatically.** With none of the above, `mathsandml/ghsl.py`
   looks up the EU JRC Global Human Settlement Layer's average building height
   (GHS-BUILT-H R2023A, 100 m cells, 2018) under the image's footprint and uses
   the 95th percentile of the built cells as the tallest-structure number. It
   is a prior, not a measurement of this image, and is labelled as one.

Only when GHSL cannot answer either does the run stop and ask: no tile and no
network, no buildings under the scene (open country, forest, water), or only
cells at GHSL's 2.5 m floor, which means "built, too low to measure". That last
case covers the Andhra demo image - a low-rise village - so it still needs a
typed height. Indian cities do have GHSL heights: central Visakhapatnam reads
about 18 m, Bhubaneswar about 14 m (95th percentile of the cells around them).

The GHSL tile (1000 km square, 10-40 MB) downloads once per region into
`cache/ghsl`; `fetch_ghsl.command` (double-click) fetches the tiles for the
benchmark scenes and the demo image, and `warm_cache.py` fetches them along
with the model and the DEM, so a demo machine needs no network afterwards. Tiles or the global mosaic you already have can go
in a folder named by `GHSL_DIR`. `GHSL_DOWNLOAD=0` never touches the network.

**A person's tallest-structure number is used as given**
(`DEPTHWIZARD_PRIOR_GAIN=1.0`), and that is measured. With the ground-envelope
split (see *Where the accuracy comes from*, item 9) the structure size comes
out right on average - reference/predicted amplitude 1.08 over nine scenes -
and a factor fitted on eight scenes and applied to the ninth made all nine
worse. The old Gaussian split (`DEPTHWIZARD_SPLIT=gauss`) needed x0.80, and
still gets it. The report's `suggested alpha gain` would lower RMSE further,
but only by shrinking every building toward zero, so it is never the default.

**How good the GHSL prior is, measured** against the LiDAR benchmark with the
real tiles (`mathsandml/benchmark/results/ghsl/report.md`). GHSL has buildings
under 5 of the 9 scenes; the other four are forest, farmland or a vineyard
village at the 2.5 m floor.

| Scene | GHSL prior | person's prior | RMSE, GHSL | RMSE, person |
|---|--:|--:|--:|--:|
| ahn_delft_old | 13.0 m | 15 m | 4.12 m | 4.43 m |
| ahn_rotterdam_centre | 22.4 m | 40 m | 7.26 m | 7.76 m |
| swisstopo_interlaken_hills | 7.2 m | 12 m | 1.69 m | 1.67 m |
| swisstopo_wengen_slope | 4.0 m | 20 m | 5.10 m | 3.81 m |
| swisstopo_zurich_centre | 26.9 m | 25 m | 7.92 m | 7.19 m |

So with nobody typing anything it lands close to a person's guess: better on
two scenes, level on one, worse on two. Pooled over these five: 5.70 m with
one offset removed, 6.05 m raw.

GHSL gets **no correction factor** (`DEPTHWIZARD_GHSL_GAIN=1.0`), and that is
measured, not a placeholder: the structure-size ratio against LiDAR ranged from
0.92 (Zurich) to 3.85 (Wengen), and a factor fitted on four scenes made the
held-out scenes worse. GHSL's error is scene-specific - 100 m cell
averages miss Rotterdam's few tall blocks and Wengen's forest - so no single
multiplier fixes it.

## The backbone

Default is `depth-anything/Depth-Anything-V2-Large-hf`: ~1.3 GB, and roughly
8–10× slower on CPU than Small. The accuracy half of the marking scheme is
worth the minutes. To go back:

```bash
export DEPTH_MODEL=depth-anything/Depth-Anything-V2-Small-hf
```

`DEPTH_DEVICE=auto|cpu|mps|cuda` picks where it runs (default: CUDA when
present, else CPU). On Apple silicon, `python backend/check_device.py` compares
the Metal GPU against the CPU on a real tile and prints both the speed-up and
the correlation between the two outputs - switch to `mps` only if it says the
outputs match. MPS falls back to the CPU on any operator it cannot run.

## API

```
POST /api/jobs                       multipart: file + params JSON  -> {job_id}
GET  /api/jobs/{id}                  status, progress, log, result
GET  /api/jobs/{id}/files/{name}     dsm.tif, ndsm.tif, terrain.glb, figures
POST /api/jobs/{id}/validate         multipart: reference raster -> metrics
GET  /api/jobs/{id}/layers           drapeable analysis layers + legends
GET  /api/jobs/{id}/profile          ?r0&c0&r1&c1 -> every raster along a line
GET  /api/jobs/{id}/probe            ?r&c -> every raster at one pixel
GET  /api/health                     which checkpoint is loaded
```

Jobs run on a worker thread with a progress log, because a Large-backbone run
on CPU takes minutes and a blocking request would time out in the browser.

## Outputs

`dsm.tif` · `ndsm.tif` · `dtm.tif` (float32, georeferenced in absolute mode) ·
`height16.png` · `texture.png` · `meta.json` · `terrain.glb` ·
`uncertainty.tif` · `layer_*.png` + `layers.json` ·
`validation.md` / `.json` · `error_map.png` · `scatter.png` · `stability.png` ·
`reference_grid.tif` (the uploaded reference on the estimate's grid)

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
8. **Shadows measured ray by ray, and read off the roof.** The shadow
   calibrator returned *zero* usable control points on a scene built to order.
   Three reasons, all now fixed and all covered by `benchmark/tests.py`: it
   required shadows to be elongated along the sun, so a wide block casting a
   shorter shadow (ratio 0.74) was rejected as "too circular"; it read the
   depth at the shadow's object end, which is the pavement at the foot of the
   wall and reads exactly 0 m above ground; and it measured length across the
   whole shadow blob, which includes the building's own depth along the sun
   direction (+71% at azimuth 135°, +108% at 315°). It now bins the shadow
   across the sun direction and takes the median run, then probes back towards
   the sun for the roof. On synthetic scenes with the answer built in it
   recovers a known scale of 3.0 to within 4% at every sun angle tested.
9. **Ground from an envelope, not from the lowest histogram peak.** Height
   above ground used to be the depth minus a Gaussian blur, shifted so the
   lowest strong histogram peak sat at zero. That peak is often *water*: the
   model reads canals and rivers as the lowest surface, so every street and
   field was lifted by the gap - +11 m across Delft (canals), +7.6 m across
   Zurich (the Limmat). Ground is now a robust envelope (2nd-percentile
   opening, 80 m) slid up under the depth map; ground lands at ~0 and a block
   narrower than 80 m keeps its full height. Correlation of height above
   ground with the LiDAR nDSM: 0.66 -> 0.73; worst ground lift 8.1 m -> 1.7 m.
10. **The DEM clean-up stops at 5 m.** The opening that takes rooftops out of
    COP30 also took hilltops and valley sides off, because it cannot tell a
    hill from a building: terrain shape error went 1.95 -> 5.82 m in the Jura
    and 5.49 -> 7.33 m in Zurich. Rooftops blur into a 30 m radar cell as a
    few metres, so the correction is capped at 5 m: pooled terrain RMSE
    4.14 -> 3.01 m, with the flat scenes keeping their correction.

Items 9 and 10 together, all nine scenes, typed priors: pooled RMSE
6.31 -> 5.08 m (one offset removed) and 8.76 -> 5.37 m raw, MAE 4.42 ->
3.45 m, r 0.73 -> 0.78. Six scenes improve; Flevoland (+0.6 m), Rotterdam
(+0.2 m) and Interlaken (+0.1 m) get slightly worse.

## Reproducing the accuracy numbers

`mathsandml/benchmark/results/report.md` was regenerated on 2026-09-26 from the
cached depth predictions: all 9 scenes scored, pooled RMSE **5.08 m** with a
constant datum offset removed and **5.37 m** raw, MAE 3.45 m, r 0.78. Lavaux
and Wengen got their `known_height_m` read off the orthophoto on that date
(`scene.json` says so and says it was not taken from the reference). Towers
over 30 m still come back ~30 m short: the depth model does not see them,
which only a better model will fix. Regenerate over every scene after any
change to the pipeline:

```bash
bash mathsandml/benchmark/run_all.sh            # fetch + score + write the report
SOURCE=all bash mathsandml/benchmark/run_all.sh # AHN + swisstopo + USGS
FAST=1 bash mathsandml/benchmark/run_all.sh     # Small backbone, first pass
```

It takes minutes per scene on a CPU with the Large backbone. The report quotes
both the raw RMSE and the datum-shifted one, and the raw number is the honest
one.

## Tests

Neither needs the model weights or the network; both run in seconds.

```bash
python mathsandml/benchmark/tests.py       # every check named after the bug it prevents
python mathsandml/benchmark/selftest.py    # the whole harness on synthetic scenes
python mathsandml/test_invariants.py       # dsm = dtm + ndsm, conditioning never lies
```

`tests.py` is part A, the regressions - every bug found in review, each check
named after the failure it prevents - and part B, the datum guard, which stubs
the pipeline and drives the real server to prove an absolute run refuses to
guess its scale and says what its heights are measured from. It skips the
per-scene prior checks when `mathsandml/benchmark/scenes/` is absent (that
folder is gitignored - hundreds of megabytes of LiDAR) and runs everything
else.

`training/selftest.py` runs the whole training chain - fake GAMUS files,
tile conversion, a few training steps with a tiny random backbone, resume,
export, evaluation, inference - in about a minute on a CPU. Run it before
spending GPU hours.

`selftest.py` carries its own synthetic scenes and a stub backbone that
reproduces the real model's failure modes, so it exercises the whole benchmark
without downloading 1.3 GB of weights.

## Known limits

- Absolute scale needs one real number. A coarse DEM cannot supply it —
  the ramp lives at low frequency, exactly where SRTM is blind (measured
  alpha −14.9 against a true 81.3). So a georeferenced image never falls back
  on a made-up default: a guessed 40 m returns confident metres anchored to
  nothing. It uses shadows, control points or a person's number, then the
  GHSL building height at its coordinates, and only **stops and asks** when
  none of those exists. The GHSL prior is 100 m and 2018-vintage, and has no
  height for low-rise villages (GHSL's 2.5 m floor).
- Very tall buildings come back short: the one >30 m block in the benchmark
  reads ~30 m low. That is the zero-shot depth model not ranking it; no
  post-processing fixes it. Fixing it is what HeightNet is for (long-tail
  weighted training on LiDAR) - measure it with
  `compare_engines.py` once a checkpoint exists.
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
