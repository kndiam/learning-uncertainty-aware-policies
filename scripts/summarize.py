"""
Aggregate saved seeds into tables (mean and 95% t-interval across seeds).

    python -m scripts.summarize                       # everything in results/
    python -m scripts.summarize --pattern "clf_PENDIGITS_g1.0_seed*.npz"

Writes results/tables/<task>_<dataset>_g<gamma>_{adaptivity,gap}.csv and prints
markdown to the console.
"""

import argparse
import glob
import os
import re
from collections import defaultdict

import pandas as pd

from ablation_utils.experiment import load
from ablation_utils.stats import seed_summary, gap_summary, aggregate

MAIN_STATS = ["size_high", "size_low", "did_size", "did_miss", "did_alpha",
              "auroc_size_eu", "spearman_u_size", "miss_high", "miss_low",
              "aisl", "aisl_minus_control"]


def _fmt(t):
    m, lo, hi, n = t
    return f"{m:.3f}" if n < 2 else f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--pattern", default="*_seed*.npz")
    a = p.parse_args()

    groups = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(a.results, a.pattern))):
        groups[re.sub(r"_seed\d+\.npz$", "", os.path.basename(f))].append(load(f))
    os.makedirs(os.path.join(a.results, "tables"), exist_ok=True)

    for name, runs in groups.items():
        methods = sorted({k.split("/")[0] for k in runs[0] if "/" in k})
        policies = sorted({k.split("/")[1] for k in runs[0] if k.count("/") == 2})
        rows = []
        for m in methods:
            for pol in policies:
                agg = aggregate([seed_summary(r, m, pol) for r in runs])
                rows.append({"method": m, "policy": pol, "seeds": agg["size_high"][3],
                             **{k: _fmt(agg[k]) for k in MAIN_STATS if k in agg}})
        df = pd.DataFrame(rows).set_index(["method", "policy"])
        df.to_csv(os.path.join(a.results, "tables", f"{name}_adaptivity.csv"))
        print(f"\n## {name}: EU adaptivity  ({len(runs)} seeds)\n")
        print(df.to_markdown())

        if name.startswith("clf") and any("/learned/" in k for k in runs[0]):
            g = pd.DataFrame({m: {k: _fmt(v) for k, v in aggregate([gap_summary(r, m) for r in runs]).items()}
                              for m in methods}).T
            g.to_csv(os.path.join(a.results, "tables", f"{name}_gap.csv"))
            print(f"\n## {name}: learned vs exact decomposition\n")
            print(g.to_markdown())


if __name__ == "__main__":
    main()
