#!/usr/bin/env python3
"""
compare_engines.py - the one table that decides the default engine.

Runs the 9-scene LiDAR benchmark once per configuration and puts the results
side by side:

    current   zero-shot depth + the scale prior in scene.json       (today)
    current   zero-shot depth + GHSL, automatic
    A         fine-tuned relative depth + the same scene.json prior
    A         fine-tuned relative depth + GHSL, automatic
    B         fine-tuned metric head, metres directly, automatic
    A+B       hybrid: A calibrated by the learned prior, fused with B, automatic

Rows that need the trained model are skipped (and say so) until
models/heightnet.pt exists. Nothing here tunes anything on these scenes: the
fusion weights come from held-out training regions (training/evaluate.py), so
the table is a fair test.

    python mathsandml/benchmark/compare_engines.py
    python mathsandml/benchmark/compare_engines.py --scenes mathsandml/benchmark/scenes \
        --out mathsandml/benchmark/results/engines
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

CONFIGS = [
    # key, label, engine, scale source, needs model
    ("zeroshot_prior", "current: zero-shot + scene prior", "zeroshot", "known-height", False),
    ("zeroshot_ghsl", "current: zero-shot + GHSL (automatic)", "zeroshot", "ghsl", False),
    ("a_prior", "A: fine-tuned relative + scene prior", "finetuned", "known-height", True),
    ("a_ghsl", "A: fine-tuned relative + GHSL (automatic)", "finetuned", "ghsl", True),
    ("b", "B: metres directly (automatic)", "metric", "model", True),
    ("hybrid", "A+B: hybrid (automatic)", "hybrid", "model", True),
    ("hybrid_prior", "A+B: hybrid + scene prior", "hybrid", "known-height", True),
]
LANDSCAPES = ("urban", "sparse", "hilly", "forest")


def fmt(v, nd=2):
    return "-" if v is None or v != v else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scenes", default=os.path.join(HERE, "scenes"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "engines"))
    ap.add_argument("--only", default=None, help="comma-separated config keys")
    ap.add_argument("--reuse", action="store_true",
                    help="use results.json already in --out instead of re-running")
    a = ap.parse_args()

    import heightnet as HN
    have_model = HN.available() is not None
    keys = set(a.only.split(",")) if a.only else None
    rows = []
    for key, label, engine, src, needs in CONFIGS:
        if keys and key not in keys:
            continue
        out = os.path.join(a.out, key)
        if needs and not have_model:
            rows.append(dict(key=key, label=label, skipped=f"no trained model at "
                                                           f"{HN.default_checkpoint()}"))
            continue
        res = os.path.join(out, "results.json")
        if not (a.reuse and os.path.exists(res)):
            cmd = [sys.executable, os.path.join(HERE, "run_benchmark.py"), "--scenes", a.scenes,
                   "--out", out, "--scale-source", src, "--engine", engine, "--no-figures"]
            print(f"\n##### {label}\n$ {' '.join(cmd)}", flush=True)
            subprocess.run(cmd, cwd=os.path.dirname(ROOT))
        if not os.path.exists(res):
            rows.append(dict(key=key, label=label, skipped="benchmark run failed"))
            continue
        results = json.load(open(res))
        if not any(not r.get("error") for r in results["scenes"]):
            err = next((str(r.get("error")) for r in results["scenes"]), "no scenes")
            rows.append(dict(key=key, label=label, skipped="every scene failed - "
                                                           + err[:90].replace("|", "/")))
            continue
        rows.append(dict(key=key, label=label, engine=engine, source=src, results=results))

    import run_benchmark as RB
    lines = ["# Engine comparison - 9-scene LiDAR benchmark", "",
             "All metres. RMSE is shift-aligned (a constant vertical datum offset "
             "removed, as in report.md); **RMSE raw** is the untouched output. "
             "Landscape columns are pooled RMSE per landscape.", "",
             "| configuration | scenes | RMSE | **RMSE raw** | MAE | r | "
             + " | ".join(LANDSCAPES) + " |",
             "|---|--:|--:|--:|--:|--:|" + "--:|" * len(LANDSCAPES)]
    summary = []
    for row in rows:
        if row.get("skipped"):
            lines.append(f"| {row['label']} | - | *skipped: {row['skipped']}* | | | | "
                         + " | ".join("" for _ in LANDSCAPES) + " |")
            continue
        recs = [r for r in row["results"]["scenes"] if not r.get("error")]
        p = RB.pool([r["headline"] for r in recs]) or {}
        praw = RB.pool([r["raw_alignment"] for r in recs if r.get("raw_alignment")]) or {}
        land = {}
        for cls in LANDSCAPES:
            q = RB.pool([r["by_landscape"].get(cls) for r in recs
                         if r.get("by_landscape", {}).get(cls)])
            land[cls] = q["rmse"] if q else None
        n_all = len(row["results"]["scenes"])
        lines.append(f"| {row['label']} | {len(recs)}/{n_all} | {fmt(p.get('rmse'))} | "
                     f"**{fmt(praw.get('rmse'))}** | {fmt(p.get('mae'))} | "
                     f"{fmt(p.get('r'), 3)} | "
                     + " | ".join(fmt(land[c]) for c in LANDSCAPES) + " |")
        summary.append(dict(key=row["key"], label=row["label"], scenes=len(recs),
                            rmse=p.get("rmse"), rmse_raw=praw.get("rmse"),
                            mae=p.get("mae"), r=p.get("r"), by_landscape=land))
    lines += ["", "How to read it: the default engine should be the best **RMSE raw** "
                  "among the automatic rows (no human input), without losing any "
                  "landscape badly. 'scene prior' rows use a height a person read off "
                  "each ortho - a fair comparison between engines, but not what an "
                  "unattended run gets."]
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "engines.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(a.out, "engines.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print("\n" + "\n".join(lines))
    print(f"\nWrote {os.path.join(a.out, 'engines.md')}")


if __name__ == "__main__":
    main()
