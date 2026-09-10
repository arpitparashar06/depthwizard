"""_bootstrap.py - one place that knows where the other two folders are.

The repo is split three ways so a reader can find things:

    backend/      this folder: the HTTP API and the command line
    mathsandml/   the science - depth, calibration, geometry, scoring
    frontend/     React + Vite + three.js

Nothing in mathsandml/ imports anything from backend/, so the science can be
run, tested and benchmarked on its own. The traffic is one-way: the entry
points in here put mathsandml/ on the import path and call into it.

Import this FIRST in any backend entry point, before importing inference:
inference binds OPENTOPO_KEY at module import time, so the .env has to be read
before that happens or the coarse DEM fetch is refused and every georeferenced
run silently returns height above local ground.
"""

import os
import sys

BACKEND = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BACKEND)
MATHSANDML = os.path.join(ROOT, "mathsandml")
FRONTEND_DIST = os.path.join(ROOT, "frontend", "dist")
JOBS_DIR = os.path.join(ROOT, "jobs")

if MATHSANDML not in sys.path:
    sys.path.insert(0, MATHSANDML)


def load_env(path=None):
    """Read KEY=value lines from the repo-root .env. Real environment wins."""
    path = path or os.path.join(ROOT, ".env")
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))
