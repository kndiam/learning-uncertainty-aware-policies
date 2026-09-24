"""Fast conformal p-values agree with the original broadcast implementation."""
import numpy as np
from ablation_utils.uncertainty import conformal_pvalues, conformal_pvalues_loo


def _pvals_broadcast(S_calib_true, S_test_all):
    n = len(S_calib_true)
    counts = (S_calib_true[None, :, None] >= S_test_all[:, None, :]).sum(axis=1)
    return (1 + counts) / (n + 1)


def _pvals_loo_broadcast(S_calib_true, S_calib_all):
    n = len(S_calib_true)
    counts = (S_calib_true[None, :, None] >= S_calib_all[:, None, :]).sum(axis=1)
    self_hit = S_calib_true[:, None] >= S_calib_all
    return (1 + counts - self_hit) / n


def test_pvalues_match_broadcast():
    rng = np.random.default_rng(0)
    for _ in range(5):
        n, m, K = 300, 400, 7
        S_all = np.round(rng.random((n, K)), 2)          # rounding forces ties
        S_true = S_all[np.arange(n), rng.integers(0, K, n)]
        S_test = np.round(rng.random((m, K)), 2)
        assert np.array_equal(conformal_pvalues(S_true, S_test), _pvals_broadcast(S_true, S_test))
        assert np.array_equal(conformal_pvalues_loo(S_true, S_all), _pvals_loo_broadcast(S_true, S_all))
