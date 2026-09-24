"""Vectorized exact policies agree with the per-point notebook implementation."""
import numpy as np
from ablation_utils.exact import (exact_alpha_clf, set_sizes, covered, smoothed_alpha_clf,
                                  smoothed_loss, exact_alpha_reg, interval_radius)


# copied from classification_closed_form_evaluation_eu_aware.ipynb, cell 12
def exact_size(Sigma, s_cand, n, alpha):
    if alpha <= 1.0 / (n + 1):
        return len(s_cand)
    tau = Sigma / ((n + 1) * alpha - 1.0)
    return int(np.sum(s_cand <= tau))


def exact_min(Sigma, s_cand, n, lam, w=1.0, eps=1e-6):
    s_sorted = np.sort(s_cand)
    with np.errstate(divide="ignore"):
        breakpoints = ((Sigma / s_sorted) + 1.0) / (n + 1)
    cands = np.concatenate([breakpoints + eps, [1.0 / (n + 1) + eps, 1.0]])
    cands = np.unique(cands[(cands > 1.0 / (n + 1)) & (cands <= 1.0)])
    losses = np.array([exact_size(Sigma, s_cand, n, a) + lam * w * a for a in cands])
    return float(cands[int(np.argmin(losses))])


def _random_problem(rng, m=1500, K=10, n=500):
    S = 1.0 - rng.dirichlet(np.full(K, 0.3), size=m)
    S[:40, 0] = 0.0                 # p_hat = 1 -> score 0 -> breakpoint at infinity
    S[40:80, 1] = S[40:80, 2]       # ties
    return S, rng.uniform(50, 300), rng.uniform(5, 40), rng.uniform(0.2, 3.0, m), n


def test_exact_clf_matches_loop():
    rng = np.random.default_rng(0)
    for _ in range(3):
        S, Sigma, lam, w, n = _random_problem(rng)
        a_vec = exact_alpha_clf(Sigma, S, n, lam, w)
        a_loop = np.array([exact_min(Sigma, S[i], n, lam, w[i]) for i in range(len(S))])
        assert np.array_equal(a_vec, a_loop)
        sz = np.array([exact_size(Sigma, S[i], n, a_vec[i]) for i in range(len(S))])
        assert np.array_equal(set_sizes(Sigma, S, n, a_vec), sz)


def test_smoothed_is_a_minimum_and_approaches_exact():
    rng = np.random.default_rng(1)
    S, Sigma, lam, w, n = _random_problem(rng, m=300)
    E = (n + 1) * S / (Sigma + S)
    a_s = smoothed_alpha_clf(E, lam, w)
    dense = np.linspace(1.0 / E.max(), 1.0, 20000)
    L_dense = smoothed_loss(E, np.broadcast_to(dense, (len(E), len(dense))), lam, w).min(1)
    assert np.all(smoothed_loss(E, a_s, lam, w) <= L_dense + 1e-3)
    # sharper surrogate -> smaller excess exact loss, on the same alpha domain
    lo = 1.0 / E.max()
    a_ex = exact_alpha_clf(Sigma, S, n, lam, w, min_alpha=lo)
    L = lambda a: set_sizes(Sigma, S, n, a) + lam * w * a
    gaps = [np.mean(L(smoothed_alpha_clf(E, lam, w, min_alpha=lo, k_sharp=k, n_grid=4000)) - L(a_ex))
            for k in (5.0, 50.0, 500.0)]
    assert gaps[0] > gaps[1] > gaps[2] >= 0


def test_min_alpha_bound():
    rng = np.random.default_rng(2)
    S, Sigma, lam, w, n = _random_problem(rng, m=500)
    lo = 0.3
    a = exact_alpha_clf(Sigma, S, n, lam, w, min_alpha=lo)
    assert np.all(a >= lo)
    grid = np.linspace(lo, 1.0, 5000)
    L = lambda a: set_sizes(Sigma, S, n, a) + lam * w * a
    L_grid = np.min([L(np.full(len(S), g)) for g in grid], axis=0)
    assert np.all(L(a) <= L_grid + 1e-3)   # eps offset past each breakpoint


def test_regression_closed_form_is_stationary():
    Sigma, n, lam = 120.0, 999, 30.0
    w = np.array([0.5, 1.0, 2.0])
    a = exact_alpha_reg(Sigma, n, lam, w)
    L = lambda a: 2 * interval_radius(Sigma, n, a) + lam * w * a
    for d in (1e-4, -1e-4):
        assert np.all(L(a) <= L(a + d))
    assert np.all(np.diff(a) < 0)       # larger weight -> smaller alpha
