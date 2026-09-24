"""
Run the EU-adaptivity experiment for several seeds and save one .npz per seed.

    python -m scripts.run_seeds --task clf --dataset PENDIGITS --seeds 0 1 2 3 4
    python -m scripts.run_seeds --task reg --dataset CALIFORNIA --seeds 0 1 2 3 4 --learned

Without --learned only the exact, exact_bounded and smoothed policies are computed
(no network training; seconds to minutes per seed).
"""

import argparse
import time

from ablation_utils import config as C
from ablation_utils.experiment import run_classification_seed, run_regression_seed, save


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["clf", "reg"], required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--gamma", type=float, default=C.GAMMA)
    p.add_argument("--learned", action="store_true", help="also train the policy nets")
    p.add_argument("--num-runs", type=int, default=4)
    p.add_argument("--num-epochs", type=int, default=200)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="results")
    a = p.parse_args()

    run = run_classification_seed if a.task == "clf" else run_regression_seed
    for seed in a.seeds:
        t = time.perf_counter()
        res = run(a.dataset, seed, gamma=a.gamma, learned=a.learned, num_runs=a.num_runs,
                  num_epochs=a.num_epochs, device=a.device)
        path = f"{a.out}/{a.task}_{a.dataset}_g{a.gamma}_seed{seed}.npz"
        save(res, path)
        print(f"seed {seed}: {time.perf_counter() - t:.1f}s -> {path}")


if __name__ == "__main__":
    main()
