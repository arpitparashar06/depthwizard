> # SUPERSEDED - DO NOT QUOTE THESE NUMBERS
>
> This report was generated before two defects were fixed, and both of them
> affect the figures below.
>
> 1. **The scale prior leaked from the ground truth.** No `scene.json` carried
>    a `known_height_m`, so the harness fell back to a percentile of the
>    reference nDSM - the very raster each scene is then scored against. Every
>    headline number here was calibrated with the answer. The scenes now ship
>    priors read off the ortho, and the harness refuses to invent one.
> 2. **The suggested `--alpha-gain` of 1.853 is wrong and would make things
>    worse.** The estimator behind it measured miss rate rather than scale.
>    Measured on Rotterdam: it advised x2.83 where the error-minimising gain
>    was x0.74, taking object-band RMSE from 6.06 m to 10.83 m.
>
> Regenerate before quoting anything:
>
> ```bash
> python benchmark/run_benchmark.py --scenes benchmark/scenes --out benchmark/results
> ```
>
> For reference, Rotterdam re-run by hand with an honest 40 m prior scored
> **RMSE 8.45 m** against the 9.36 m below - the leak was not even helping.

# DepthWizard - DSM accuracy against reference LiDAR

Scenes attempted: **4**, scored: **4**, failed: **0**  
Depth backbone: `depth-anything/Depth-Anything-V2-Large-hf`  
Scale source: `known-height`  
Coarse DEM terrain baseline: `on`  
Object/terrain split: 15 m  

> ### Read this before quoting any figure
> 
> Headline alignment used: `shift`. `shift` means a single constant vertical offset, **computed from the reference**, was subtracted before scoring. Elevation products routinely sit on different vertical datums, so removing one constant is normal practice - but it is not the accuracy of the untouched output, and a figure quoted without this sentence is misleading.
> 
> Pooled RMSE **with** that offset removed: **6.79 m**.  
> Pooled RMSE of the raw output, **no alignment at all**: **11.96 m**.
> 
> Quote both, or quote the raw one.

Scored against: `absolute DSM (metres above sea level)`.  
When no coarse DEM supplies the terrain baseline the pipeline emits height above local ground, so it is scored against the reference nDSM rather than the absolute surface - otherwise the terrain the prediction never claimed to know would dominate the error.

## Per-scene accuracy

All values in metres. `r` is Pearson correlation against the reference; 
`NSE` is Nash-Sutcliffe, which unlike `r` penalises bias and wrong scale.

| Scene | px (m) | RMSE | RMSE raw | MAE | MedAE | Bias | NMAD | LE90 | r | NSE |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ahn_delft_old | 0.50 | 7.40 | 21.04 | 5.61 | 4.25 | -1.43 | 6.30 | 13.12 | 0.706 | -1.177 |
| ahn_flevoland_farm | 0.50 | 4.96 | 6.05 | 3.55 | 2.36 | 0.28 | 3.51 | 8.32 | 0.557 | -0.551 |
| ahn_rotterdam_centre | 0.50 | 9.36 | 9.43 | 5.42 | 3.13 | 0.81 | 4.64 | 10.75 | 0.409 | -0.349 |
| ahn_veluwe_forest | 0.50 | 4.08 | 4.09 | 2.55 | 1.19 | -0.67 | 1.76 | 7.07 | -0.183 | -3.257 |
| **pooled** | | **6.79** | **11.96** | **4.27** | 2.72 | -0.24 | 4.03 | 9.78 | 0.366 | |

## Structure heights only (object band)

The surface minus its own low-pass, i.e. how well building and canopy 
heights are recovered once the terrain baseline is removed. This is the 
number that reflects the depth model rather than the DEM.

| Scene | RMSE | MAE | Bias | r | pred p99 | ref p99 | height ratio | suggested alpha gain |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| ahn_delft_old | 5.47 | 4.34 | 0.84 | 0.718 | 16.32 | 9.08 | 1.817 | 0.550 |
| ahn_flevoland_farm | 3.04 | 1.86 | 0.16 | 0.405 | 6.83 | 17.96 | 0.453 | 2.209 |
| ahn_rotterdam_centre | 6.18 | 3.96 | -0.07 | 0.520 | 26.79 | 35.73 | 0.496 | 2.018 |
| ahn_veluwe_forest | 2.30 | 1.30 | -0.02 | -0.499 | 4.08 | 5.55 | 0.592 | 1.688 |

Median suggested `--alpha-gain`: **1.853** (re-run with this to close the attenuation loop).

## Stability across landscape types

Classes are proxies derived from the reference surface and the imagery 
(roughness, excess green, low-frequency relief), not a land-cover product.

| Landscape | scenes | share % | RMSE | MAE | Bias | NMAD | r |
|---|--:|--:|--:|--:|--:|--:|--:|
| urban | 4 | 27.8 | 7.61 | 5.40 | 0.21 | 6.06 | 0.456 |
| sparse | 4 | 58.8 | 5.87 | 3.36 | 0.06 | 3.41 | 0.608 |
| hilly | 1 | 4.9 | 3.60 | 2.23 | -1.79 | 1.70 | 0.935 |
| forest | 4 | 12.2 | 8.75 | 6.16 | -2.48 | 6.20 | 0.070 |

Spread: **5.15 m** between `hilly` (3.60 m) and `forest` (8.75 m), a ratio of **2.43x**.

## Accuracy by structure height

| Height band | scenes | share % | RMSE | MAE | Bias |
|---|--:|--:|--:|--:|--:|
| ground (<2 m) | 4 | 75.1 | 3.61 | 2.14 | 0.66 |
| low (2-10 m) | 4 | 18.5 | 5.97 | 4.80 | -0.21 |
| mid (10-30 m) | 3 | 3.3 | 9.29 | 8.41 | -8.32 |
| high (>30 m) | 1 | 0.5 | 31.93 | 30.11 | -30.11 |

## How scale was set

Each scene was calibrated from **one semantic prior** - roughly how tall the tallest sustained structure is. Where `scene.json` did not supply one, the p99.5 of the reference nDSM stood in, which is noted per scene in `results.json`.

## Per-scene artefacts

Each scene directory under the results folder holds `dsm.tif`, `ndsm.tif`, `dtm.tif`, `terrain.glb` inputs, `validation.md` / `.json`, plus `error_map.png`, `scatter.png` and `stability.png`.
