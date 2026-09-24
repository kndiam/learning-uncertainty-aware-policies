"""
Wall-clock cost of the exact vs learned coverage policy as the test set grows.

    python -m scripts.time_policies --dataset PENDIGITS --sizes 1000 10000 100000 1000000

Test points are resampled (with replacement) from the real test split so the
score distribution is realistic at every size. Reports the median of --reps runs:
  exact_vectorized  ablation_utils.exact.exact_alpha_clf
  exact_loop        the per-point notebook loop (only up to --loop-max points)
  learned_inference one forward pass of a trained policy net
  learned_training  one-off cost of training that net (independent of n_test)
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch

from ablation_utils import config as C
from ablation_utils import neural_net as anet
from ablation_utils.exact import exact_alpha_clf
from ablation_utils.uncertainty import UncertaintyData, MMIUncertainty, conformal_pvalues, conformal_pvalues_loo


def _loop_exact(Sigma, S_all, n, lam, w, eps=1e-6):
    """The per-point implementation from the notebooks, for reference."""
    out = np.empty(len(S_all))
    for i, s_cand in enumerate(S_all):
        with np.errstate(divide="ignore"):
            bp = (Sigma / np.sort(s_cand) + 1.0) / (n + 1)
        c = np.concatenate([bp + eps, [1.0 / (n + 1) + eps, 1.0]])
        c = np.unique(c[(c > 1.0 / (n + 1)) & (c <= 1.0)])
        tau = Sigma / ((n + 1) * c - 1.0)
        L = (s_cand[None, :] <= tau[:, None]).sum(1) + lam * w[i] * c
        out[i] = c[L.argmin()]
    return out


def _median_time(fn, reps, device):
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="PENDIGITS")
    p.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 100000, 1000000])
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--loop-max", type=int, default=10000)
    p.add_argument("--num-epochs", type=int, default=200)
    p.add_argument("--gamma", type=float, default=C.GAMMA)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    cfg = C.CLASSIFICATION[a.dataset]
    data = cfg["data"](0)
    model = cfg["model"](0).fit(data.X_train, data.y_train)
    n = data.n_calib
    S_calib_all = 1.0 - model.predict_proba(data.X_calib)
    S_calib = S_calib_all[np.arange(n), data.y_calib]
    S_test_all = 1.0 - model.predict_proba(data.X_test)
    Sigma_loo, Sigma = S_calib.sum() - S_calib, float(S_calib.sum())
    E_loo = n * S_calib_all / (Sigma_loo[:, None] + S_calib_all)
    lam = C.lambda_clf(a.dataset, E_loo, data.num_classes)

    mmi = MMIUncertainty(form="pi").fit(None)
    mmi.calibrate(UncertaintyData(X=data.X_calib, P=conformal_pvalues_loo(S_calib, S_calib_all)))
    u_loo = mmi.transform_loo(UncertaintyData(X=data.X_calib, P=conformal_pvalues_loo(S_calib, S_calib_all)))
    u_test = mmi.transform_normalized(UncertaintyData(X=data.X_test, P=conformal_pvalues(S_calib, S_test_all)))

    loader = anet.make_loader(Sigma_loo, S_calib_all, E_loo, u_loo,
                              w=C.coverage_weight(u_loo, a.gamma, "mmi_pi"))
    torch.manual_seed(0)
    t = time.perf_counter()
    net, _ = anet.train_alpha_net(loader, lam, num_epochs=a.num_epochs, patience=20, device=a.device)
    t_train = time.perf_counter() - t
    net.eval()

    rng, rows = np.random.default_rng(0), []
    for m in a.sizes:
        idx = rng.integers(0, len(S_test_all), m)
        S, u = S_test_all[idx], u_test[idx]
        w = C.coverage_weight(u, a.gamma, "mmi_pi")

        def infer():              # feature construction (sorting the scores) counts as inference
            feats = torch.tensor(anet.make_features(np.full(m, Sigma), S, u),
                                 dtype=torch.float32).to(a.device)
            with torch.no_grad():
                net(feats)

        if m <= a.loop_max:       # the two exact implementations must agree
            assert np.array_equal(exact_alpha_clf(Sigma, S, n, lam, w), _loop_exact(Sigma, S, n, lam, w))

        row = {"n_test": m,
               "exact_vectorized": _median_time(lambda: exact_alpha_clf(Sigma, S, n, lam, w), a.reps, a.device),
               "learned_inference": _median_time(infer, a.reps, a.device),
               "learned_training": t_train}
        row["exact_loop"] = (_median_time(lambda: _loop_exact(Sigma, S, n, lam, w), 1, a.device)
                             if m <= a.loop_max else np.nan)
        rows.append(row)
        print(row)

    df = pd.DataFrame(rows).set_index("n_test")
    print("\nseconds (median over reps)\n")
    print(df.to_markdown(floatfmt=".4g"))
    import os
    os.makedirs("results", exist_ok=True)
    df.to_csv(f"results/timing_{a.dataset}.csv")


if __name__ == "__main__":
    main()
