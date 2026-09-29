#!/bin/bash
cd "$(dirname "$0")" || exit 1
mkdir -p cache/logs
LOG=cache/logs/ghsl_fetch.log
PY=.venv/bin/python
[ -x "$PY" ] || PY="$(command -v python3)"
{
  date
  echo "python: $PY"
  "$PY" mathsandml/ghsl.py mathsandml/benchmark/scenes/*/rgb.tif demo/*.tif
  echo "exit=$?"
} 2>&1 | tee "$LOG"
echo "DONE" >> "$LOG"
echo
echo "Done - you can close this window."
