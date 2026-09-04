#!/usr/bin/env python3
"""
selftest.py - prove the benchmark harness works without downloading anything.

Builds four synthetic georeferenced scenes with known truth, replaces the depth
backbone with a stub that mimics its three real failure modes (scale-agnostic
output, a large nadir ramp, attenuated structures), and runs the full
benchmark. Nothing here validates the depth model - it validates the harness.

    python benchmark/selftest.py

Exit code 0 means scenes were built, the pipeline ran, validation produced
metrics and figures, and the report rendered.
"""
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.dirname(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main():
    tmp = tempfile.mkdtemp(prefix="depthwizard-selftest-")
    scenes = os.path.join(tmp, "scenes")
    results = os.path.join(tmp, "results")
    keep = "--keep" in sys.argv
    try:
        import make_synthetic
        print("== building synthetic scenes ==")
        for i, kind in enumerate(["urban", "sparse", "hilly", "forest"]):
            make_synthetic.write(scenes, f"synth_{kind}", kind, seed=i)

        os.environ["STUB_TRUTH_DIR"] = scenes
        import stub_backbone
        stub_backbone.install()

        print("\n== running benchmark ==")
        import run_benchmark
        sys.argv = ["run_benchmark.py", "--scenes", scenes, "--out", results]
        rc = run_benchmark.main()

        print("\n== checking artefacts ==")
        need = ["dsm.tif", "ndsm.tif", "validation.md", "validation.json",
                "error_map.png", "scatter.png", "stability.png"]
        missing = []
        for s in sorted(os.listdir(results)):
            d = os.path.join(results, s)
            if not os.path.isdir(d):
                continue
            have = set(os.listdir(d))
            gone = [n for n in need if n not in have]
            print(f"  {s:16s} {'ok' if not gone else 'MISSING ' + ', '.join(gone)}")
            missing += gone
        for f in ("report.md", "results.json"):
            ok = os.path.exists(os.path.join(results, f))
            print(f"  {f:16s} {'ok' if ok else 'MISSING'}")
            if not ok:
                missing.append(f)

        if rc != 0 or missing:
            print("\nSELFTEST FAILED")
            return 1
        print("\nSELFTEST PASSED - harness is working end to end.")
        print("The metrics above are meaningless as accuracy numbers: the "
              "backbone was a stub and the scenes are synthetic blocks.")
        return 0
    finally:
        if keep:
            print(f"\nartefacts kept in {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
