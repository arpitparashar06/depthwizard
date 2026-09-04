#!/usr/bin/env bash
# One command to produce benchmark/results/report.md.
#
#   cd ~/Downloads/newest/depthwizard && bash benchmark/run_all.sh
#
# Options:
#   FAST=1     use the Small backbone (~100 MB, 8-10x quicker) for a first pass
#   SIZE=200   smaller footprint in metres if the first run feels slow
#   SOURCE=all fetch swisstopo and USGS too, not just AHN
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
ROOT="$PWD"

for f in benchmark/fetch_data.py benchmark/run_benchmark.py inference.py validate.py; do
  [ -f "$f" ] && continue
  echo "Not in the DepthWizard project: $ROOT is missing $f" >&2
  echo "Run it as:  cd <the depthwizard folder> && bash benchmark/run_all.sh" >&2
  exit 1
done
SIZE="${SIZE:-300}"
SOURCE="${SOURCE:-ahn}"

echo "== DepthWizard benchmark =="
echo "   project : $ROOT"

# --- venv -------------------------------------------------------------------
if [ -x ".venv/bin/python" ]; then
  PY="$ROOT/.venv/bin/python"
elif [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate && PY="$(command -v python3)"
else
  echo "   no .venv found - using system python3"
  PY="$(command -v python3)"
fi
echo "   python  : $PY  ($("$PY" --version 2>&1))"

if [ "${FAST:-0}" = "1" ]; then
  export DEPTH_MODEL="depth-anything/Depth-Anything-V2-Small-hf"
  echo "   backbone: Small (FAST=1)"
else
  echo "   backbone: Large (default) - minutes per scene on CPU; FAST=1 to speed up"
fi

# --- preflight --------------------------------------------------------------
echo
echo "== 1/4 checking dependencies =="
"$PY" - <<'EOF' || { echo "   -> pip install -r requirements.txt"; exit 1; }
import importlib.util, sys
missing = []
for m in ("numpy", "scipy", "PIL", "cv2", "torch", "transformers",
          "rasterio", "trimesh", "matplotlib", "requests"):
    try:
        if importlib.util.find_spec(m) is None:
            missing.append(m)
    except (ImportError, ValueError):
        missing.append(m)
print("   missing:", ", ".join(missing) if missing else "nothing")
sys.exit(1 if missing else 0)
EOF

echo
echo "== 2/4 probing the reference services =="
"$PY" benchmark/fetch_data.py --check || {
  echo
  echo "   A service did not answer. The run can still go ahead with whatever"
  echo "   scenes already exist under benchmark/scenes, or drop rasters in by"
  echo "   hand - see benchmark/BENCHMARK.md."
}

echo
echo "== 3/4 fetching scenes (${SOURCE}, ${SIZE} m) =="
"$PY" benchmark/fetch_data.py --source "$SOURCE" --size-m "$SIZE" \
      --out benchmark/scenes

if ! ls benchmark/scenes/*/ref_dsm.tif >/dev/null 2>&1; then
  echo
  echo "   No scenes were built, so there is nothing to score."
  echo "   See benchmark/BENCHMARK.md for the manual download route."
  exit 1
fi
echo "   scenes: $(ls -d benchmark/scenes/*/ 2>/dev/null | wc -l | tr -d ' ')"

echo
echo "== 4/4 running the benchmark =="
"$PY" benchmark/run_benchmark.py --scenes benchmark/scenes --out benchmark/results
rc=$?

echo
if [ -f benchmark/results/report.md ]; then
  echo "== done =="
  echo "   report : $ROOT/benchmark/results/report.md"
  echo "   raw    : $ROOT/benchmark/results/results.json"
  echo
  echo "   Nothing to copy back - Claude can read these from the connected folder."
else
  echo "== no report was written (exit $rc) =="
  echo "   The console output above says which stage failed."
fi
exit $rc
