#!/bin/bash
cd "$(dirname "$0")" || exit 1
mkdir -p cache/logs
LOG="cache/logs/mac_run.log"
rm -f cache/logs/DONE cache/logs/FAILED
exec > >(tee "$LOG") 2>&1

PY=".venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"
echo "== DepthWizard - fetch scenes and warm the caches =="
date
echo "python: $PY ($("$PY" --version 2>&1))"

echo; echo "== 1/3 scenes =="
"$PY" mathsandml/benchmark/fetch_data.py --source ahn --out mathsandml/benchmark/scenes
"$PY" mathsandml/benchmark/fetch_data.py --source swisstopo --out mathsandml/benchmark/scenes

echo; echo "== demo scene from India (OpenAerialMap, CC-BY 4.0) =="
[ -f demo/india_andhra.tif ] || "$PY" mathsandml/benchmark/make_demo.py || echo "(demo scene skipped)"

echo; echo "== 2/3 CPU vs Apple GPU =="
"$PY" backend/check_device.py || echo "(device check skipped)"

echo; echo "== 3/3 depth model + DEMs, once per scene =="
if "$PY" mathsandml/benchmark/warm_cache.py --scenes mathsandml/benchmark/scenes $(ls demo/*.tif 2>/dev/null); then
  touch cache/logs/DONE; echo; echo "ALL DONE - you can close this window."
else
  touch cache/logs/FAILED; echo; echo "Something failed - the log above says what."
fi
date
