# Benchmarking DepthWizard against reference LiDAR

The accuracy half of the marking scheme (50%) asks for RMSE, MAE and correlation
against reference data, and for stability across urban, sparse, hilly and
forested landscapes. This directory produces exactly that, as
`benchmark/results/report.md`.

Everything here runs on your Mac. The pipeline needs the Depth-Anything weights
and the reference services need the open internet, and neither is reachable
from the cloud sandbox this was prepared in.

---

## 1. One-time setup

```bash
cd depthwizard
source .venv/bin/activate          # the venv already has torch + rasterio
pip install -r requirements.txt    # no-op if it is already current
```

The OpenTopography key is already saved in `depthwizard/.env`, and
`run_benchmark.py` loads it before importing `inference` — which matters,
because `inference.py` binds `OPENTOPO_KEY` at import time, so a key exported
later would be read as empty and the DEM fetch would silently degrade. A real
environment variable still wins over the file, and `.env` is in `.gitignore`.

The key was verified live: a request with a deliberately invalid bounding box
came back with a bbox complaint rather than an auth error, which is what a
working key looks like.

Without a key the pipeline still runs — it just outputs height above local
ground instead of height above sea level, and the benchmark scores it against
the reference nDSM instead of the reference DSM. Both are legitimate; see §5.

---

## 2. Get the reference data

```bash
python fetch_data.py --check       # probe every service, download nothing
python fetch_data.py --list        # show the built-in areas of interest
```

`--check` is the first thing to run. It prints which service answered and which
layer or coverage name it discovered. If a name has drifted since this was
written, `--check` shows you what the service *does* publish and you adjust one
regex in the `SOURCES` table at the top of `fetch_data.py`.

Then fetch:

```bash
python fetch_data.py --source ahn        --out benchmark/scenes
python fetch_data.py --source swisstopo  --out benchmark/scenes
python fetch_data.py --source usgs       --out benchmark/scenes
```

Default footprint is 300 m square at 0.5 m GSD, i.e. 600 x 600 px — about the
size of the images already in `jobs/`, and a sane first run on CPU. `--size-m`
and `--px` change that.

Default footprint is capped by the services: PDOK's WMS refuses anything over
2500 x 2500 px, so at 0.5 m GSD the largest AHN scene is 1250 m. The fetcher
checks this before requesting and says so rather than letting the service
return an exception page.

### What each source gives you

All four services below were probed live on 2026-09-03 and the names in the
table are what they actually publish today, not guesses.

| Source | Reference | Imagery | CRS | Covers |
|---|---|---|---|---|
| `ahn` | AHN4 `dsm_05m` **and** `dtm_05m`, 0.5 m, ≥10 pts/m², flown 2020–22 | `Actueel_orthoHR`, 8 cm | EPSG:28992 | urban, sparse |
| `swisstopo` | swissSURFACE3D Raster, 0.5 m COG tiles | SWISSIMAGE 10 cm | EPSG:2056 | urban, hilly, forest |
| `usgs` | 3DEP, 1 m — **bare earth**, self-described | NAIP | EPSG:3857 | hilly |

Three things that probing turned up, all now handled in code:

- PDOK's orthophoto WMS advertises **`image/jpeg` and nothing else**. A
  hard-coded PNG request comes back as a ServiceException, so the format is
  read from GetCapabilities and the best offered one is used.
- AHN's WCS accepts **EPSG:28992 only** and **GEOTIFF only**.
- swisstopo publishes **1 km² tiles**, so a 300 m AOI can straddle two or four
  of them, and each tile exists at several resolutions (SWISSIMAGE ships 0.1 m
  *and* 2.0 m). The fetcher mosaics every tile covering the AOI, keeps the
  newest revision of each, picks the finest `eo:gsd` asset, and warns if any
  part of the footprint had no coverage.

AHN is the strongest pairing: a true LiDAR surface model *and* a terrain model,
so the reference nDSM is a real measured difference rather than a low-pass
approximation. Start there.

USGS 3DEP is a **bare-earth** product. A USGS scene therefore scores terrain
reconstruction, not building heights, and `scene.json` records
`bare_earth_reference: true` so you don't read the urban row the wrong way.

### If a service is down or renamed

Any pair of overlapping rasters works. Drop them in by hand:

```
benchmark/scenes/<name>/rgb.tif       georeferenced optical image (must have a CRS)
benchmark/scenes/<name>/ref_dsm.tif   reference surface model
benchmark/scenes/<name>/ref_dtm.tif   optional terrain model
benchmark/scenes/<name>/scene.json    optional: {"landscape": "urban", "known_height_m": 40}
```

Manual sources, no scripting needed:

- **AHN** — <https://www.ahn.nl/ahn-viewer>, draw a box, download the 0.5 m DSM
  and DTM tiles. Orthophoto from <https://www.pdok.nl> ("Luchtfoto Actueel
  Ortho 8cm RGB").
- **swisstopo** — <https://www.swisstopo.admin.ch/en/geodata/height/surface3d.html>
  for the DSM, SWISSIMAGE from the same download portal.
- **ISPRS Vaihingen / Potsdam** — <https://www.isprs.org/education/benchmarks/UrbanSemLab/>.
  Needs a registration form, which is why it is not scripted. It ships an
  orthophoto plus a published nDSM, so save the nDSM as `ref_dsm.tif` and a
  zero raster as `ref_dtm.tif`, or just leave the DTM out and let the low-pass
  fallback handle it. This is the one source with numbers in the literature to
  compare against.

The only hard requirement is that `rgb.tif` carries a CRS and a transform. If
it doesn't, the pipeline drops to relative mode and there is nothing metric to
score — the runner reports that per scene rather than producing a meaningless
number.

---

## 3. Run the benchmark

```bash
python run_benchmark.py --scenes benchmark/scenes --out benchmark/results
```

On CPU with the Large backbone, budget a few minutes per 600 x 600 scene. To
sanity-check the plumbing first:

```bash
DEPTH_MODEL=depth-anything/Depth-Anything-V2-Small-hf \
  python run_benchmark.py --limit 1 --no-figures
```

Then the real run. `--use-dem` is on by default and the key is in place, so the
output is an absolute sea-level DSM scored against the reference DSM. Add
`--no-dem` to score height-above-ground against the reference nDSM instead —
useful for isolating structure accuracy from terrain. If the DEM fetch fails
for any reason the run does not break: it falls back to height-above-ground and
the report says which datum each scene was scored on.

Outputs:

```
benchmark/results/report.md      the tables below, ready to paste
benchmark/results/results.json   every metric, plus the control points used
benchmark/results/<scene>/       dsm.tif, ndsm.tif, dtm.tif, validation.md,
                                 error_map.png, scatter.png, stability.png
```

---

## 4. Choosing how scale is set

`estimate_elevation` cannot invent absolute scale — the README's "known limits"
explains why the coarse DEM can't supply it. So the benchmark has to declare
where scale came from, and `--scale-source` controls that.

| Flag | What it does | Independence |
|---|---|---|
| `gcps-from-ref` *(default)* | samples 8 control points from the reference nDSM, one per grid cell | partial — see below |
| `known-height` | one semantic prior per scene from `scene.json` | good, if the prior is genuinely external |
| `shadow` | sun azimuth + elevation, from `scene.json` or the GeoTIFF tags | **full** — no reference data enters calibration |
| `scene` | whatever each `scene.json` specifies | mixed |

The default is the "limited set of Ground Control Points" route the problem
statement explicitly allows, and 8 points across a 600 x 600 scene is a
defensible reading of "limited". The points used are written into
`results.json` so the disclosure is on the record.

But be straight about it when you present: those points come from the same
raster used for scoring, so **absolute scale is not independently validated**
in that mode. Two things to do about it:

1. Quote the **object band** table as the headline. It measures relative
   structure geometry, which a handful of point heights cannot fake across a
   whole scene.
2. Run `--scale-source shadow` on at least one scene with known sun angles.
   Nothing from the reference enters the calibration there, so it is the number
   that answers "does this work without ground truth". Most satellite products
   carry sun angles in the GeoTIFF tags and `load_image` reads them
   automatically.

---

## 5. Reading the report

**Per-scene accuracy** is the headline RMSE / MAE / correlation. `NSE`
(Nash–Sutcliffe) is there because `r` alone is generous: a prediction with the
wrong scale and a large bias can still correlate at 0.95. NSE punishes both.

**Structure heights only (object band)** removes the terrain baseline from both
sides. This isolates what the depth model contributes, and its
`suggested alpha gain` closes the loop back into the pipeline — re-run with
`--alpha-gain <median>` to correct the systematic attenuation the frequency
split introduces.

**Stability across landscape types** is the stratified table the marking scheme
asks for. Classes are proxies derived from the reference surface and the
imagery — roughness, excess green, low-frequency relief — not a land-cover
product, and the report says so. Forest will be the worst row: a DSM returns
canopy, and the pipeline has no way to see through it. That is a known limit,
not a bug, and it is better to name it than to have it found.

**What is scored against what** matters and the report states it per run:

- `--use-dem` on, and the DEM fetch succeeded → the output is an absolute
  surface, scored against `ref_dsm.tif`.
- otherwise → the output is height above local ground, scored against the
  reference nDSM (`ref_dsm - ref_dtm`, or a low-pass residual if no DTM).

Scoring height-above-ground against an absolute sea-level DSM would be
comparing two different quantities, and the terrain the prediction never
claimed to know would dominate the error. The runner will not do it.

---

## 6. Known sharp edges

- **Object/terrain split vs building size.** The split is set from the DEM
  resolution (`DEM_RES_M / px / 2`), which at 0.5 m GSD is a 15 m low-pass.
  Buildings wider than roughly 30 m are partly absorbed into "terrain" and come
  back attenuated. The `height ratio` column measures it; `--sigma-m` and
  `--alpha-gain` are the levers. On synthetic scenes built from large blocks
  this alone accounted for most of the error.
- **GCP fitting is sensitive.** `alpha_from_gcps` fits a slope through
  control points on the detail band, and when structures are attenuated the
  fitted alpha gets large and amplifies noise everywhere. If a scene reports a
  wild `alpha` in `results.json`, or a high `clipped_negative_frac`, try
  `--scale-source known-height` for that scene.
- **Forest is canopy.** Expect the forest row to be the worst. Say so first.
