# DepthWizard - DSM accuracy against reference LiDAR

Scenes attempted: **4**, scored: **4**, failed: **0**  
Depth backbone: `depth-anything/Depth-Anything-V2-Large-hf`  
Scale source: `known-height`  
Coarse DEM terrain baseline: `on`  
Object/terrain split: 15 m  

Scored against: `absolute DSM (metres above sea level)`.  
When no coarse DEM supplies the terrain baseline the pipeline emits height above local ground, so it is scored against the reference nDSM rather than the absolute surface - otherwise the terrain the prediction never claimed to know would dominate the error.

## Per-scene accuracy

All values in metres. `r` is Pearson correlation against the reference; 
`NSE` is Nash-Sutcliffe, which unlike `r` penalises bias and wrong scale.

| Scene | px (m) | RMSE | MAE | MedAE | Bias | NMAD | LE90 | r | NSE |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ahn_delft_old | 0.50 | 6.02 | 4.73 | 3.94 | -0.25 | 5.85 | 9.65 | 0.744 | -0.440 |
| ahn_flevoland_farm | 0.50 | 6.37 | 4.71 | 3.39 | 0.26 | 5.03 | 10.97 | 0.551 | -1.567 |
| ahn_rotterdam_centre | 0.50 | 7.42 | 4.76 | 3.17 | -0.31 | 4.70 | 10.55 | 0.533 | 0.153 |
| ahn_veluwe_forest | 0.50 | 1.70 | 1.05 | 0.48 | -0.34 | 0.71 | 3.02 | 0.678 | 0.260 |
| **pooled** | | **5.80** | **3.80** | 2.72 | -0.17 | 4.04 | 8.52 | 0.625 | |

## Structure heights only (object band)

The surface minus its own low-pass, i.e. how well building and canopy 
heights are recovered once the terrain baseline is removed. This is the 
number that reflects the depth model rather than the DEM.

| Scene | RMSE | MAE | Bias | r | pred p99 | ref p99 | height ratio | suggested alpha gain |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| ahn_delft_old | 5.48 | 4.29 | 0.84 | 0.715 | 17.06 | 9.16 | 1.769 | 0.565 |
| ahn_flevoland_farm | 4.31 | 2.89 | 0.19 | 0.430 | 15.42 | 17.96 | 0.910 | 1.098 |
| ahn_rotterdam_centre | 5.97 | 3.87 | -0.07 | 0.560 | 24.73 | 35.60 | 0.561 | 1.783 |
| ahn_veluwe_forest | 1.35 | 0.79 | 0.04 | 0.525 | 6.38 | 6.49 | 0.936 | 1.068 |

Median suggested `--alpha-gain`: **1.083** (re-run with this to close the attenuation loop).

## Stability across landscape types

Classes are proxies derived from the reference surface and the imagery 
(roughness, excess green, low-frequency relief), not a land-cover product.

| Landscape | scenes | share % | RMSE | MAE | Bias | NMAD | r |
|---|--:|--:|--:|--:|--:|--:|--:|
| urban | 4 | 27.8 | 7.50 | 5.58 | 0.17 | 6.45 | 0.507 |
| sparse | 4 | 58.8 | 4.66 | 2.88 | -0.21 | 3.07 | 0.638 |
| hilly | 1 | 4.9 | 3.39 | 2.11 | -0.70 | 1.88 | 0.834 |
| forest | 4 | 12.2 | 6.29 | 4.11 | -0.67 | 4.45 | 0.553 |

Spread: **4.11 m** between `hilly` (3.39 m) and `urban` (7.50 m), a ratio of **2.21x**.

## Accuracy by structure height

| Height band | scenes | share % | RMSE | MAE | Bias |
|---|--:|--:|--:|--:|--:|
| ground (<2 m) | 4 | 75.1 | 3.86 | 2.37 | 0.48 |
| low (2-10 m) | 4 | 18.5 | 5.81 | 4.51 | 0.50 |
| mid (10-30 m) | 3 | 3.3 | 8.45 | 7.20 | -7.11 |
| high (>30 m) | 1 | 0.5 | 32.67 | 30.46 | -30.46 |

## How scale was set

Each scene was calibrated from **one semantic prior** - roughly how tall the tallest sustained structure is. Where `scene.json` did not supply one, the p99.5 of the reference nDSM stood in, which is noted per scene in `results.json`.

## Per-scene artefacts

Each scene directory under the results folder holds `dsm.tif`, `ndsm.tif`, `dtm.tif`, `terrain.glb` inputs, `validation.md` / `.json`, plus `error_map.png`, `scatter.png` and `stability.png`.
