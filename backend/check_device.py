#!/usr/bin/env python3
"""Is the Apple-silicon GPU (MPS) safe and faster here?"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
from _bootstrap import load_env, MATHSANDML  # noqa: E402

load_env()
import numpy as np  # noqa: E402
import inference as I  # noqa: E402


def main():
    img = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        MATHSANDML, "benchmark", "scenes", "ahn_rotterdam_centre", "rgb.tif")
    rgb, _ = I.load_image(img)
    tile = np.ascontiguousarray(rgb[:I.TILE, :I.TILE])
    out = {}
    for dev in ("cpu", "mps"):
        if I._pick_device(dev) != dev:
            print(f"{dev}: not available")
            continue
        I._get_pipe(dev)
        I._predict_patch(tile)                       # warm-up
        t = time.time()
        for _ in range(3):
            d = I._predict_patch(tile)
        out[dev] = d
        print(f"{dev}: {(time.time() - t) / 3:.2f} s per tile")
    if len(out) == 2:
        r = float(np.corrcoef(out["cpu"].ravel(), out["mps"].ravel())[0, 1])
        relative = float(np.abs(out["cpu"] - out["mps"]).max()
                    / max(np.ptp(out["cpu"]), 1e-9))
        print(f"cpu vs mps: r = {r:.6f}, worst pixel differs by {100 * relative:.3f}% of range")
        print("MPS is safe to use (export DEPTH_DEVICE=mps)" if r > 0.9999 and relative < 0.01
              else "MPS output differs - keep the CPU")


if __name__ == "__main__":
    main()
