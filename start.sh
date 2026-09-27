#!/usr/bin/env bash
# DepthWizard - one command, or one double-click (start.command on macOS).
#
#   ./start.sh            set up on first run, then serve UI + API on :8000
#   PORT=9000 ./start.sh  another port
#
# First run: creates .venv, installs requirements.txt, downloads the depth model
# once (~1.3 GB). Every later run starts in seconds and works with NO internet:
# the model is cached by Hugging Face, the built interface ships in
# frontend/dist, and DEMs come from cache/dem (warm it with
# mathsandml/benchmark/warm_cache.py while you still have a connection).
set -u
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
PORT="${PORT:-8000}"
URL="http://127.0.0.1:${PORT}"

say() { printf '\n== %s ==\n' "$*"; }

# ---- python -----------------------------------------------------------------
if [ ! -x .venv/bin/python ]; then
  say "first run: creating .venv"
  PYBOOT="$(command -v python3 || command -v python)"
  [ -n "$PYBOOT" ] || { echo "Python 3.9+ is required - install it and run again."; exit 1; }
  "$PYBOOT" -m venv .venv || exit 1
  .venv/bin/python -m pip install --upgrade pip >/dev/null
  say "installing requirements (one time, a few minutes)"
  .venv/bin/python -m pip install -r requirements.txt || exit 1
fi
PY=.venv/bin/python

# ---- network or not ---------------------------------------------------------
# Offline, transformers retries the Hub before falling back to its cache,
# which looks like a hang on stage. If the Hub does not answer in 3 s, go
# straight to the cache.
if ! "$PY" -c "import urllib.request; urllib.request.urlopen('https://huggingface.co', timeout=3)" >/dev/null 2>&1; then
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  echo "   no internet - using cached model weights and cached DEMs only"
fi

# ---- the model, once ----------------------------------------------------------
[ "${SKIP_MODEL_CHECK:-0}" = "1" ] || "$PY" - <<'PY_EOF' || { echo "   could not load the depth model - connect once so it can download"; exit 1; }
import os, sys
sys.path.insert(0, "backend")
from _bootstrap import load_env
load_env()
from transformers import pipeline
m = os.environ.get("DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Large-hf")
print(f"   depth model: {m}")
pipeline("depth-estimation", model=m)
PY_EOF

# ---- the interface ------------------------------------------------------------
if [ ! -f frontend/dist/index.html ]; then
  if command -v npm >/dev/null; then
    say "building the interface"
    (cd frontend && npm install --no-audit --no-fund && npm run build) || exit 1
  else
    echo "frontend/dist is missing and npm is not installed - install Node 18+ or restore frontend/dist"
    exit 1
  fi
fi

# ---- serve ----------------------------------------------------------------------
say "DepthWizard on $URL  (Ctrl-C to stop)"
PORT="$PORT" "$PY" backend/server.py &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT INT TERM
for _ in $(seq 1 60); do
  "$PY" -c "import urllib.request; urllib.request.urlopen('$URL/api/health', timeout=1)" 2>/dev/null && break
  sleep 1
done
if command -v open >/dev/null; then open "$URL"
elif command -v xdg-open >/dev/null; then xdg-open "$URL" >/dev/null 2>&1
fi
wait $SERVER
