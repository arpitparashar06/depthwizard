#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1   # repo root
ROOT="$PWD"
BENCH="mathsandml/benchmark"

for f in "$BENCH/fetch_data.py" "$BENCH/run_benchmark.py" \
         mathsandml/inference.py mathsandml/validate.py; do
  [ -f "$f" ] && continue
  echo "Not in the DepthWizard project: $ROOT is missing $f" >&2
  echo "Run it as:  cd <the depthwizard folder> && bash mathsandml/benchmark/run_all.sh" >&2
  exit 1
done
SIZE="${SIZE:-300}"
SOURCES="${SOURCES:-${SOURCE:-ahn swisstopo}}"

echo "== DepthWizard benchmark =="
echo "   project : $ROOT"

# venv
if [ -x ".venv/bin/python" ]; then
  PY="$ROOT/.venv/bin/python"
elif [ -f ".venv/bin/activate" ]; then
  # Shellcheck disable=SC1091
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

# preflight
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
"$PY" "$BENCH/fetch_data.py" --check || {
  echo
  echo "   A service did not answer. The run can still go ahead with whatever"
  echo "   scenes already exist under mathsandml/benchmark/scenes, or drop rasters in by"
  echo "   hand - see mathsandml/benchmark/BENCHMARK.md."
}

echo
echo "== 3/4 fetching scenes (${SOURCES}, ${SIZE} m) =="
for src in $SOURCES; do
  "$PY" "$BENCH/fetch_data.py" --source "$src" --size-m "$SIZE" \
        --out "$BENCH/scenes" ${REFETCH:+--force}
done

# A scene without a prior is scored as a failure, never silently
"$PY" - "$BENCH/scenes" <<'EOF2'
import json, os, sys
root = sys.argv[1]
for n in sorted(os.listdir(root)):
    p = os.path.join(root, n, "scene.json")
    if os.path.isdir(os.path.join(root, n)) and (
            not os.path.exists(p) or not json.load(open(p)).get("known_height_m")):
        print(f"   NOTE: {n} has no known_height_m in scene.json - it will be "
              f"reported as a failure until one is read off the ortho")
EOF2

if ! ls "$BENCH"/scenes/*/ref_dsm.tif >/dev/null 2>&1; then
  echo
  echo "   No scenes were built, so there is nothing to score."
  echo "   See mathsandml/benchmark/BENCHMARK.md for the manual download route."
  exit 1
fi
echo "   scenes: $(ls -d "$BENCH"/scenes/*/ 2>/dev/null | wc -l | tr -d ' ')"

echo
echo "== 4/4 running the benchmark =="
"$PY" "$BENCH/run_benchmark.py" --scenes "$BENCH/scenes" --out "$BENCH/results" \
      --scale-source known-height --compare gcps-from-ref --loso-gain
rc=$?

echo
if [ -f "$BENCH/results/report.md" ]; then
  echo "== done =="
  echo "   report : $ROOT/$BENCH/results/report.md"
  echo "   raw    : $ROOT/$BENCH/results/results.json"
  echo
  echo "   Both files are plain text - open them, or paste the report into the README."
else
  echo "== no report was written (exit $rc) =="
  echo "   The console output above says which stage failed."
fi
exit $rc
