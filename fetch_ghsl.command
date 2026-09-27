#!/bin/bash
# fetch_ghsl.command - double-click on macOS.
#
# Downloads, once, the GHSL building-height tiles under the benchmark scenes
# and the demo image into cache/ghsl, and prints the automatic height prior
# each one gets. After this the GHSL prior works with no network. Output is
# also written to cache/logs/ghsl_fetch.log.
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
