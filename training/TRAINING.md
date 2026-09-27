# Training HeightNet

HeightNet is Depth-Anything-V2 fine-tuned on GAMUS and LiDAR. One network,
three outputs:

* **A - relative depth**, fine-tuned for top-down images (scale-and-shift-
  invariant loss, the way Depth-Anything itself was trained).
* **B - metres above ground**, directly (adaptive height bins).
* **land cover** - ground, low vegetation, building, water, road, tree.

The app ships them **combined** (engine `hybrid`), in the problem statement's
own workflow:

```
Satellite RGB ─► fine-tuned DA-V2 ─┬─► A relative depth ─► Scale calibration module ─► DSM
                                    │                        • SRTM/COP30 DEM: terrain + datum
                                    │                        • GCPs / shadows when available
                                    └─► B metres ───────────►• learned scene prior (fits A to B)
                                             │                         │
                                             └──── fused per pixel by uncertainty ◄┘
```

B enters twice: as the scene-level prior that sets A's scale when no measured
anchor exists (the problem statement lists "scene-level statistics ...
semantic priors" among the allowed calibrators; it replaces GHSL's 100 m, 2018
prior), and as a second opinion blended in wherever it is more certain than A.

| engine | what runs | needs a scale anchor? |
|---|---|---|
| `zeroshot` | Depth-Anything-V2 as released + calibration module (today) | yes |
| `finetuned` (A) | fine-tuned relative depth + the same calibration module | yes |
| `metric` (B) | fine-tuned metres directly | no |
| `hybrid` (A+B) | A calibrated by the learned prior (or a measured anchor), fused with B | no |

`benchmark/compare_engines.py` scores all of them on the 9 LiDAR scenes and
decides the default.

Everything runs on **free Kaggle GPUs**. You upload two notebooks and press
"Save & Run All" twice. Nothing has to be installed on your Mac.

---

## 1. The model

```
RGB tile (518 x 518) ──► DINOv2 ViT encoder ──► DPT neck ──► fused features
      │                  (Depth-Anything-V2,                     │
      │                   pretrained, fine-tuned                 │ FiLM  ◄── GSD embedding
      │                   with layer-wise LR decay)              ▼        (metres per pixel)
      │                                       pretrained head conv1/conv2 ──► A: relative depth
      │                                                          │                (conv3, fine-tuned)
      │                                     ┌────────────────────┴──────────────┐
      │                                     ▼                                   ▼
      │                          semantic head (7 classes)  ──softmax──►  height trunk
      │                                                                        │
      └─ CLS token + pooled features + GSD ──► bin MLP ──► 64 adaptive bins    ▼
                                                     (log-spaced)       per-pixel bin logits
                                                                              │
                                        height = Σ p·centre,   uncertainty = spread of p
```

Each part comes from a paper, and each one addresses a specific failure:

| part | why it is there | source |
|---|---|---|
| Depth-Anything-V2 backbone, fine-tuned (not frozen) | it already knows geometry from 62 M images; fine-tuning closes the gap between ground-level photos and a top-down view | Yang et al. 2024; *Depth Any Canopy* 2024 (same backbone fine-tuned for canopy height) |
| transformer, not CNN | global context matters: CMX/TIMF beat every CNN on GAMUS | GAMUS paper, Xiong et al. 2023 (Tables 3-5) |
| semantic head that **feeds** the height head | height and land cover are strongly tied; buildings and trees gain the most | GAMUS paper, Sec. 5.2 |
| 64 log-spaced **adaptive bins** + classification | height maps are long-tailed, and plain regression under-shoots tall objects | HTC-DC Net (Chen, Xiong, Zhu 2023), AdaBins (Bhat et al. 2021) |
| long-tail loss weighting | a 40 m roof pixel counts for more than a lawn pixel | HTC-DC Net's head-tail balance, as sample reweighting |
| GSD conditioning (FiLM) | the same 10 m block is 30 px wide at 0.33 m and 10 px wide at 1 m; the model is *told* which | ours |
| gradient-matching loss | keeps building edges sharp | MiDaS (Ranftl et al. 2020) |
| satellite-style augmentation | GAMUS is aerial; the app gets satellite images: haze, blur, noise, JPEG, random GSD 0.3-1.2 m | ours |
| relative-depth output trained scale-and-shift-invariant | keeps the "relative depth → calibration" workflow the problem statement asks for, with a depth model that finally knows the top-down view | MiDaS / Depth-Anything training loss |
| rotation ensemble at inference | the same 4-rotation averaging the zero-shot engine uses; also gives uncertainty | DepthWizard |
| uncertainty-weighted fusion, with measured honesty factors | A is sharper, B knows metres; each is trusted where its own error is smaller | ours (calibrated on held-out tiles by `evaluate.py`) |

**Losses** (`losses.py`): for B, long-tail-weighted L1 + 0.3 × soft bin
cross-entropy + 0.5 × gradient matching + 0.1 × bin chamfer; for A, 0.5 ×
scale-and-shift-invariant L1 + 0.25 × its gradient matching; for land cover,
0.3 × cross-entropy.

**Training recipe** (`train.py`): AdamW. Backbone learning rate is 2e-5 for
Small and 1e-5 for Base, with 0.85 layer decay; neck 1e-4; new heads 3e-4.
1,000 warm-up steps, then cosine decay. fp16 mixed precision, an EMA of the
weights (that is what ships), gradient clipping at 1.0, and DDP across both
Kaggle T4s.

## 2. The data

| source | what | licence | role |
|---|---|---|---|
| **GAMUS** (`earthflow/GAMUS`, Hugging Face) | US cities, 0.33 m aerial RGB + LiDAR nDSM + 6-class land cover, 1024 px tiles | CC-BY-4.0 | the recommended dataset: urban, tall buildings, labels |
| **AHN4** (PDOK, NL) | 0.5 m LiDAR DSM + DTM + national orthophoto | CC0 | Dutch towns, pine forest, polders, the Limburg hills |
| **swisstopo** (STAC API, CH) | swissSURFACE3D + swissALTI3D + SWISSIMAGE | Swiss open government data (free use; credit swisstopo) | Alps, Jura forest, farmland, Swiss towns |

31 regions, about 1,000 LiDAR tiles. Whole regions are held out for
validation. **No tile within 3 km of any benchmark scene** is used (checked
against `fetch_data.AOIS` in code), so `run_benchmark.py` stays a fair,
never-seen test.

Sampling mix per batch: GAMUS 60 %, AHN 18 %, swisstopo 22 %, balanced across
landscapes inside each source (`--mix`).

Backbone licences: Depth-Anything-V2-**Small** is Apache-2.0. **Base** and
**Large** are CC-BY-NC-4.0, which is fine for SIH but not for commercial use.

## 3. Step by step (Kaggle)

You need a Kaggle account with a **verified phone number**. Kaggle requires
it for GPUs and Internet access.

### Step 0 (optional, 1 minute, on any computer with PyTorch)

```bash
python training/selftest.py
```

This runs data prep, training, resume, export, evaluation and inference on
synthetic data. If it passes, a Kaggle run can only fail for Kaggle reasons.

### Step 1: build the training tiles (CPU, about 2-4 h, unattended)

1. kaggle.com → **Create → New Notebook → File → Import Notebook** →
   `training/kaggle/01_prepare_data.ipynb`.
2. Right panel: **Accelerator: None**, **Internet: On**.
3. *(Recommended)* Create a free Hugging Face account → Settings → Access
   Tokens → a *read* token. In Kaggle: **Add-ons → Secrets → Add** with the
   name `HF_TOKEN`. GAMUS is about 20,000 files, and anonymous downloads get
   rate-limited.
4. **Save Version → Save & Run All (Commit)**. You can close the tab.
5. When it has finished, open the version → **Output** → **New Dataset** →
   name it `depthwizard-tiles`.

Want a 5-minute trial first? Set `GAMUS_LIMIT = 30` and
`LIDAR_PER_REGION = 3` in the first code cell.

### Step 2: train (GPU T4 x2, about 3-4 h for Small)

1. Import `training/kaggle/02_train_heightnet.ipynb`.
2. **Add Input** → `depthwizard-tiles`. **Accelerator: GPU T4 x2**,
   **Internet: On**.
3. **Save Version → Save & Run All (Commit)**.
4. At the end the notebook decides **how A and B are blended** on held-out
   validation tiles (inverse-variance, or a stacked weight per height band -
   whichever scores better; never worse than B alone), writes that into the
   checkpoint, and scores A, B and A+B on other held-out tiles across all
   cities (`eval/eval.md`).
5. Output → download **`heightnet.pt`** → copy it to
   `depthwizard/models/heightnet.pt`.

   **Safari users:** Safari's "Open safe files after downloading" unzips a
   `.pt` into a folder of the same name. The app accepts that folder as it
   is, but to get the plain file untick that option (Safari → Settings →
   General) or download with Chrome.

### Step 2b (optional): re-calibrate an existing model

`training/kaggle/03_recalibrate.ipynb` (one T4, 30-60 min) takes a model you
already trained, re-decides the blend and re-scores it - no training. Use it
after changing the blending code, or for a model trained before the stacked
blend existed.

The run stops by itself at 10.5 h. If it had not finished, run the notebook
again with the previous version's output added as a second input. It finds
`state.pt` and carries on from there.

### Step 3: measure it on the benchmark (on the Mac)

```bash
python mathsandml/benchmark/compare_engines.py
```

One table, six rows: today's pipeline (with the scene prior and with GHSL),
A with the same two anchors, B, and A+B - pooled RMSE, raw RMSE, MAE, r and
RMSE per landscape. Only make the hybrid the default if it wins **here**: this
is the only test on scenes the model has never seen, in the landscapes the
marking scheme names. Nothing is tuned on these scenes.

### Step 4: use it

Start the app as usual. The results panel shows **Engine: A+B** with the
model name and how much weight B got, "Scale from" says *learned scene prior*,
the height field says it is optional, and `/api/health` reports
`"engine": "hybrid"`. Pick another engine with `DEPTHWIZARD_ENGINE=zeroshot`,
`finetuned`, `metric` or `hybrid`; delete the file to go back entirely.

## 4. Reading the training log

```
[ 4000/30000] l1 1.842 ce 2.113 grad 0.402 chamfer 0.061 ssi 0.412 ssi_grad 0.233 seg 0.512  |g| 0.84  41.2 img/s  eta 2.9 h
[val] gamus    B: RMSE 3.10 m  MAE 1.42  r 0.912  tall(>=10 m) RMSE 6.80 bias -2.10 | A: RMSE 3.40 m  rank r 0.890 | mIoU 0.612
[val] lidar    B: RMSE 1.95 m  MAE 0.88  r 0.874  tall(>=10 m) RMSE 5.10 bias -1.30 | A: RMSE 2.10 m  rank r 0.850 | mIoU nan
```

*(The numbers above show the layout only; they are not results.)*

* **B's `bias` on tall pixels** is the number to watch. It starts strongly
  negative (towers too short) and should move towards 0.
* **A's `rank r`** is how well the relative depth orders heights (1.0 =
  perfectly). It should climb from the pretrained model's value; A's RMSE is
  A after B has set its scale, i.e. the hybrid's A half.
* **`lidar`** is the held-out Dutch/Swiss regions: forest, hills and
  farmland. If GAMUS improves while this one gets worse, the model is
  over-fitting to US cities. Lower the GAMUS share (`--mix`).
* `samples/*.jpg`: image | LiDAR | prediction | uncertainty, saved at every
  validation.
* `heightnet_best.pt` is picked by the mean of A's and B's RMSE over the two
  validation sets - the hybrid needs both halves good.

## 5. With two weeks: a plan

| days | run | why |
|---|---|---|
| 1 | Step 1 (data) | once; every later run reuses the dataset |
| 1-2 | Small, default settings, then the benchmark | the safe win. If it beats the zero-shot engine on the 9 scenes, it is the new default |
| 3-5 | **Base** (`VARIANT = "base"`), about 2 sessions with resume | usually a clear step up for dense cities; check it still runs acceptably on a laptop CPU |
| 6-8 | one or two ablations (`--longtail-power 0.8`, `--mix gamus=0.5,ahn=0.2,swisstopo=0.3`) | choose by **benchmark**, not by GAMUS validation |
| 9+ | freeze the model, update the PPT numbers | GAMUS test RMSE, held-out LiDAR RMSE, benchmark vs zero-shot |

Kaggle's weekly GPU quota (about 30 h at the time of writing) covers this
plan with room to spare.

## 6. If something goes wrong

| symptom | fix |
|---|---|
| `429` / very slow GAMUS download | add the `HF_TOKEN` secret (step 1.3). The script backs off and retries on its own |
| `UNIT CHECK ... look like centimetres` | rerun prepare with `--height-scale 0.01` |
| many `[lidar] FAILED` lines | a service was busy. Rerun the notebook: finished tiles are skipped |
| output over Kaggle's 20 GB | `--jpeg-quality 90` in the GAMUS cell, or `GAMUS_LIMIT = 2500` |
| CUDA out of memory | Base: it already uses `--grad-ckpt`; lower `--batch` to 2 and add `--accum 2` |
| `img/s` very low | the CPU is the bottleneck; the notebook already sets 2 workers per GPU on Kaggle's 4 cores |
| the app still says zero-shot | the file must be named `models/heightnet.pt`; `python mathsandml/heightnet.py image.tif` shows whether it loads |

## 7. Honest limits

* **GAMUS and AHN/swisstopo are aerial orthophotos, not satellite images.**
  The augmentations imitate satellite conditions, but only a satellite image
  with LiDAR under it proves it. Test on one before claiming it.
* **Dates differ.** Orthophoto and LiDAR can be years apart. New buildings and
  felled trees are label noise the model has to live with.
* The model predicts **height above ground**. Terrain still comes from the 30 m
  DEM, debiased as before, so hilly-scene accuracy is still bounded by that DEM.
* Everything in this folder was tested end to end on synthetic data and a tiny
  network (`selftest.py`, and `tests.py` A24 / B6-B7). Accuracy exists only
  after a real Kaggle run. Report numbers from `evaluate.py` and
  `run_benchmark.py`, never from this document.
