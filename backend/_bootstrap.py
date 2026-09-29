"""One place that knows where the other two folders are."""

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
    """Read KEY=value lines from the repo-root .env."""
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
