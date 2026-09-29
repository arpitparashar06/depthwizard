#!/usr/bin/env python3
"""Build the two Kaggle notebooks from the source files in this repo, so the"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(HERE, "kaggle")

CODE = {
    "prepare": ["mathsandml/heightnet.py", "mathsandml/preprocess.py",
                "mathsandml/benchmark/fetch_data.py",
                "training/data.py", "training/losses.py", "training/train.py",
                "training/evaluate.py", "training/prepare_gamus.py",
                "training/prepare_lidar.py", "training/selftest.py"],
    "recal": ["mathsandml/heightnet.py", "mathsandml/preprocess.py",
              "training/data.py", "training/evaluate.py"],
    "train": ["mathsandml/heightnet.py", "mathsandml/preprocess.py",
              "training/data.py", "training/losses.py", "training/train.py",
              "training/evaluate.py", "training/prepare_gamus.py", "training/selftest.py"],
}


_ids = iter(range(10 ** 6))


def md(text):
    return dict(cell_type="markdown", id=f"c{next(_ids):04d}", metadata={},
                source=text.strip("\n").splitlines(True))


def code(text):
    return dict(cell_type="code", id=f"c{next(_ids):04d}", metadata={},
                execution_count=None, outputs=[], source=text.strip("\n").splitlines(True))


def writefile_cells(files):
    cells = [code("import os\nfor d in ('dw/mathsandml/benchmark', 'dw/training'):\n"
                  "    os.makedirs(d, exist_ok=True)\nprint('folders ready')")]
    for relative in files:
        with open(os.path.join(REPO, relative)) as f:
            src = f.read()
        cells.append(code(f"%%writefile dw/{relative}\n" + src))
    return cells


def notebook(cells):
    return dict(cells=cells, metadata=dict(
        kernelspec=dict(display_name="Python 3", language="python", name="python3"),
        language_info=dict(name="python"),
        kaggle=dict(accelerator="none", isInternetEnabled=True)), nbformat=4, nbformat_minor=5)


SH = '''
# Run a command, stream its output, and STOP the notebook if it fails
# (a plain "!command" carries on after an error).
import subprocess
def sh(cmd):
    print("$", cmd, flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        if "Warning" not in line and "warnings.warn" not in line:
            print(line, end="", flush=True)
    if p.wait() != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {cmd}")
'''


HF_SECRET = '''
# Optional but recommended: a free Hugging Face token lifts the download rate
# limit (GAMUS is ~20,000 files). Add it under Add-ons -> Secrets as HF_TOKEN.
import os
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN loaded from Kaggle secrets")
except Exception:
    print("no HF_TOKEN secret - downloading anonymously (slower if rate-limited)")
'''


def prepare_notebook():
    cells = [md('''
# DepthWizard - 1. prepare the training data

Builds the training tiles for **HeightNet** and saves them as this notebook's
output, so the training notebook can attach them as a dataset.

* **GAMUS** (Hugging Face `earthflow/GAMUS`): US cities, 0.33 m, LiDAR height
  above ground + land-cover labels. Downloaded one tile at a time, converted,
  and deleted, so the disk never holds the 80 GB original.
* **AHN4 (Netherlands) + swisstopo (Switzerland)**: forest, farmland, hills,
  mountains and European towns - the landscapes GAMUS does not have. Tiles
  near any DepthWizard benchmark scene are **excluded**, so the benchmark
  stays a fair test.

**Settings (right-hand panel):** Accelerator **None** (CPU is enough),
Internet **On**. Then **Save Version -> Save & Run All (Commit)** and close
the tab - it runs in the background (roughly 2-4 hours).

**When it has finished:** open the version -> **Output** -> **New Dataset**,
name it `depthwizard-tiles`. That dataset is the input of notebook 2.
'''),
             code('''
GAMUS_LIMIT = 0          # tiles per split; 0 = all of GAMUS. 30 = a 5-minute trial run
LIDAR_PER_REGION = 35    # tiles per region (31 regions); 5 = a quick trial
OUT = "/kaggle/working/data"
'''),
             code("!pip install -q h5py rasterio huggingface_hub"),
             code(SH),
             code(HF_SECRET)]
    cells += writefile_cells(CODE["prepare"])
    cells += [md("### Sanity check: the whole chain on synthetic data (about a minute)"),
              code('sh("cd dw/training && python selftest.py")'),
              md("### GAMUS -> tiles"),
              code('sh(f"python dw/training/prepare_gamus.py --out {OUT}/gamus --workers 16 "\n'
                   '   f"--limit {GAMUS_LIMIT}")'),
              md("### Dutch and Swiss LiDAR -> tiles"),
              code('sh(f"python dw/training/prepare_lidar.py --out {OUT}/lidar --workers 8 "\n'
                   '   f"--per-region {LIDAR_PER_REGION}")'),
              md("### What came out"),
              code('''
import json, os, collections
for name in ("gamus", "lidar"):
    p = os.path.join(OUT, name, "index.json")
    if not os.path.exists(p):
        print(name, "- missing"); continue
    t = json.load(open(p))["tiles"]
    c = collections.Counter((x["source"], x.get("landscape"), x["split"]) for x in t)
    print(f"{name}: {len(t)} tiles")
    for k, v in sorted(c.items()):
        print("   ", k, v)
!du -sh {OUT}/*
!rm -rf dw     # keep the output to the data only
''')]
    return notebook(cells)


def train_notebook():
    cells = [md('''
# DepthWizard - 2. train HeightNet

Fine-tunes Depth-Anything-V2 on GAMUS + LiDAR into **HeightNet**, one network
with three outputs:

* **A - relative depth** for top-down images, which the app calibrates to metres
  (DEM + GCPs / shadows / scene prior) - the problem statement's workflow;
* **B - metres above ground**, directly - the scene prior that calibrates A,
  and a second opinion fused in per pixel;
* **land cover** (6 classes).

At the end it scores A, B and A+B on held-out tiles and stores the fusion
calibration inside the checkpoint.

**Settings:** Accelerator **GPU T4 x2**, Internet **On** (it downloads the
pretrained Depth-Anything-V2 weights). **Add Input** -> your
`depthwizard-tiles` dataset (from notebook 1).

Then **Save Version -> Save & Run All (Commit)**. It trains in the background,
validates every 2,000 steps, keeps the best weights, and stops by itself
before Kaggle's 12-hour limit.

**Resume** (if a run stopped at the time limit): add the previous version's
output as a second input. The cell below finds `state.pt` and carries on.

**Afterwards:** Output -> download `heightnet.pt` -> put it in
`depthwizard/models/heightnet.pt`. The app uses it from then on.
'''),
             code('''
VARIANT = "small"      # "small" (25 M params, ~3-4 h on T4 x2) or "base" (98 M, ~9-11 h)
ITERS = None           # None = the variant's default (small 30k, base 24k)
EVAL_TILES = 400       # held-out tiles per dataset for the final report
MAX_HOURS = 10.5       # leave time for the evaluation before Kaggle's 12 h limit

import glob, os, torch
roots = sorted({os.path.dirname(p) for p in glob.glob("/kaggle/input/**/index.json", recursive=True)})
states = sorted(glob.glob("/kaggle/input/**/state.pt", recursive=True))
RESUME = states[-1] if states else None
N_GPU = max(1, torch.cuda.device_count())
WORKERS = max(2, (os.cpu_count() or 4) // N_GPU)
OUT = f"/kaggle/working/runs/{VARIANT}"
ROOTS_ARG = " ".join(roots)
print("data roots:", roots)
print("resume from:", RESUME)
print("GPUs:", N_GPU, [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
assert roots, "attach the depthwizard-tiles dataset (Add Input)"
'''),
             code("!pip install -q h5py"),
             code(SH)]
    cells += writefile_cells(CODE["train"])
    cells += [md("### Sanity checks (a few minutes)\n"
                 "1. the whole chain on synthetic data;  2. twenty real steps on both GPUs "
                 "with the real backbone - catches multi-GPU and memory problems in "
                 "minutes instead of hours in."),
              code('''
sh("cd dw/training && python selftest.py")
launcher = f"torchrun --standalone --nproc_per_node={N_GPU}" if N_GPU > 1 else "python"
extra = " --grad-ckpt" if VARIANT == "base" else ""
sh(f"{launcher} dw/training/train.py --data {ROOTS_ARG} --variant {VARIANT} --out /tmp/smoke "
   f"--iters 20 --val-every 20 --val-n 8 --warmup 5 --log-every 5 --workers {WORKERS}{extra}")
'''),
              md("### Train"),
              code('''
args = f"--data {ROOTS_ARG} --variant {VARIANT} --out {OUT} --workers {WORKERS} --max-hours {MAX_HOURS}"
if ITERS: args += f" --iters {ITERS}"
if VARIANT == "base": args += " --grad-ckpt"
if RESUME: args += f" --resume {RESUME}"
sh(f"{launcher} dw/training/train.py {args}")
'''),
              md("### Validation curve and sample predictions\n"
                 "Each sample row: image | LiDAR | HeightNet | uncertainty."),
              code('''
import pandas as pd, glob
from IPython.display import Image, display
v = pd.read_csv(f"{OUT}/val.csv"); display(v.tail(10))
for p in sorted(glob.glob(f"{OUT}/samples/*.jpg"))[-2:]:
    print(p); display(Image(p, width=900))
'''),
              md("### Score A, B and A+B on held-out tiles (GAMUS test + held-out LiDAR regions)\n"
                 "First measures how honest each output's uncertainty is on validation "
                 "tiles and writes that into `heightnet_best.pt` - the app's fusion uses it."),
              code('sh(f"python dw/training/evaluate.py --ckpt {OUT}/heightnet_best.pt "\n'
                   '   f"--data {ROOTS_ARG} --split test,val --limit {EVAL_TILES} "\n'
                   '   f"--out {OUT}/eval")'),
              code('''
import shutil
shutil.copy(f"{OUT}/heightnet_best.pt", "/kaggle/working/heightnet.pt")
!ls -la /kaggle/working/heightnet.pt {OUT}
print("Download heightnet.pt from the Output tab -> depthwizard/models/heightnet.pt")
!rm -rf dw
''')]
    nb = notebook(cells)
    nb["metadata"]["kaggle"]["accelerator"] = "nvidiaTeslaT4"
    return nb


def recal_notebook():
    cells = [md('''
# DepthWizard - 3. re-calibrate and re-score a trained model

No training: takes a `heightnet.pt` you already have, decides how its two
outputs are blended (inverse-variance vs a stacked weight per height band,
whichever is better on held-out tiles), writes that into the file, and scores
A, B and A+B again on tiles across **all** cities. About 30-60 minutes.

**Settings:** Accelerator **GPU T4** (one is enough), Internet **Off** is fine.
**Add Input:** your `depthwizard-tiles` data **and** the output of the
training notebook (it contains `runs/<variant>/heightnet_best.pt`).
Then **Save Version -> Save & Run All** and download the new `heightnet.pt`.
'''),
             code('''
EVAL_TILES = 400       # held-out tiles per dataset for the report
CALIB_TILES = 80       # validation tiles per dataset used to decide the blend
import glob, os, shutil
roots = sorted({os.path.dirname(p) for p in glob.glob("/kaggle/input/**/index.json", recursive=True)})
cks = sorted(glob.glob("/kaggle/input/**/heightnet_best.pt", recursive=True)) or \\
      sorted(glob.glob("/kaggle/input/**/heightnet*.pt", recursive=True))
assert roots, "attach the depthwizard-tiles data"
assert cks, "attach the training notebook's output (heightnet_best.pt)"
shutil.copy(cks[-1], "/kaggle/working/heightnet.pt")     # inputs are read-only
ROOTS_ARG = " ".join(roots)
print("data:", roots, "\\nmodel:", cks[-1])
'''),
             code(SH)]
    cells += writefile_cells(CODE["recal"])
    cells += [code('sh(f"python dw/training/evaluate.py --ckpt /kaggle/working/heightnet.pt "\n'
                   '   f"--data {ROOTS_ARG} --split test,val --limit {EVAL_TILES} "\n'
                   '   f"--calib-tiles {CALIB_TILES} --out /kaggle/working/eval")'),
              code('!rm -rf dw && ls -la /kaggle/working && cat /kaggle/working/eval/eval.md')]
    nb = notebook(cells)
    nb["metadata"]["kaggle"]["accelerator"] = "nvidiaTeslaT4"
    return nb


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, nb in (("01_prepare_data.ipynb", prepare_notebook()),
                     ("02_train_heightnet.ipynb", train_notebook()),
                     ("03_recalibrate.ipynb", recal_notebook())):
        with open(os.path.join(OUT, name), "w") as f:
            json.dump(nb, f, indent=1)
        print("wrote", os.path.join(OUT, name))


if __name__ == "__main__":
    main()
