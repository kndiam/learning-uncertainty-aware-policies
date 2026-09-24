"""
Per-seed statistics and their aggregation across seeds.

The replication unit is the seed: each statistic is computed once per seed
(on n_test points), and inference is a t-interval across seeds. Per-point
p-values with n_test ~ 10^4 are always tiny and carry no information.

    rows = [seed_summary(res, "mmi_pi", "exact") for res in results_for_one_dataset]
    table = aggregate(rows)
"""

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def _get(res, method, policy, key):
    v = np.asarray(res[f"{method}/{policy}/{key}"], dtype=float)
    return v.mean(axis=0) if v.ndim == 2 else v          # learned: average the runs


def seed_summary(res, method, policy, control="constant"):
    """
    One seed, one (method, policy). DiD compares against the control under the
    same policy, point by point: ((m - c) on high EU) - ((m - c) on low EU).
    """
    eu = np.asarray(res["eu"], bool)
    out = {}
    for key in ("alpha", "size", "miss"):
        v, c = _get(res, method, policy, key), _get(res, control, policy, key)
        out[f"{key}_high"], out[f"{key}_low"] = v[eu].mean(), v[~eu].mean()
        out[f"{key}_gap"] = v[eu].mean() - v[~eu].mean()
        d = v - c
        out[f"did_{key}"] = d[eu].mean() - d[~eu].mean()

    size = _get(res, method, policy, "size")
    out["auroc_size_eu"] = roc_auc_score(eu, size) if np.ptp(size) > 0 else 0.5
    u = np.asarray(res[f"{method}/u"], float)
    out["spearman_u_size"] = (stats.spearmanr(u, size).statistic
                              if np.ptp(u) > 0 and np.ptp(size) > 0 else np.nan)

    if f"{method}/{policy}/aisl" in res:                  # regression
        a, c = _get(res, method, policy, "aisl"), _get(res, control, policy, "aisl")
        out["aisl"], out["aisl_high"], out["aisl_low"] = a.mean(), a[eu].mean(), a[~eu].mean()
        out["aisl_minus_control"] = (a - c).mean()
    return out


def gap_summary(res, method):
    """
    Learned-vs-exact decomposition for one seed (classification). Objective values
    are the exact loss Size + lam * w * alpha, averaged over test points.
    """
    lam, w = float(res["lam"]), np.asarray(res[f"{method}/w"], float)
    out = {}
    for pol in ("exact", "exact_bounded", "smoothed", "learned"):
        if f"{method}/{pol}/alpha" not in res:
            continue
        a, s = (np.asarray(res[f"{method}/{pol}/{k}"], float) for k in ("alpha", "size"))
        out[f"loss_{pol}"] = float(np.mean(s + lam * w * a))       # (runs, m) broadcasts fine
    if "loss_learned" in out:
        a_nn = np.asarray(res[f"{method}/learned/alpha"], float)
        for pol in ("exact", "exact_bounded", "smoothed"):
            out[f"mean_abs_alpha_learned_minus_{pol}"] = float(
                np.mean(np.abs(a_nn - np.asarray(res[f"{method}/{pol}/alpha"], float))))
    return out


def aggregate(rows, conf=0.95):
    """list of per-seed dicts -> {stat: (mean, ci_low, ci_high, n)} with a t-interval over seeds."""
    out = {}
    for k in rows[0]:
        v = np.array([r[k] for r in rows], float)
        v = v[np.isfinite(v)]
        n = len(v)
        if n == 0:
            out[k] = (np.nan, np.nan, np.nan, 0)
            continue
        m = v.mean()
        h = stats.t.ppf(0.5 + conf / 2, n - 1) * v.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        out[k] = (m, m - h, m + h, n)
    return out
