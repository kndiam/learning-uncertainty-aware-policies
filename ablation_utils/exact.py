"""
Exact (and smoothed-exact) coverage policies, vectorized over test points.

Classification, soft-rank e-values:
    y in C_alpha(x)  <=>  E(x, y) = (n+1) S(x, y) / (Sigma + S(x, y)) <= 1/alpha
    <=>  S(x, y) <= tau(alpha) = Sigma / ((n+1) alpha - 1)            (alpha > 1/(n+1))

Objective per test point, with weight w(x) = c(u(x)):
    L(alpha; x) = Size(C_alpha(x)) + lam * w(x) * alpha

Size is a non-increasing step function of alpha that drops at the breakpoints
    b_y = (Sigma / S(x, y) + 1) / (n + 1),
so L is minimised at the left edge of a step: alpha in {b_y + eps} U {1/(n+1) + eps, 1}.

Regression, MAE scores:
    Size(alpha) = 2 Sigma / ((n+1) alpha - 1)
    alpha*(x)   = min{1, (1 + sqrt(2 Sigma (n+1) / (lam w(x)))) / (n+1)}
"""

import numpy as np
from scipy.special import expit


# ── classification ────────────────────────────────────────────────────────────

def set_sizes(Sigma, S_all, n, alphas):
    """(m,) exact set sizes. S_all: (m, K) candidate scores; alphas: (m,)."""
    alphas = np.asarray(alphas, dtype=float)
    denom  = (n + 1) * alphas - 1.0
    tau    = np.where(denom > 0, Sigma / np.clip(denom, 1e-300, None), np.inf)
    return (S_all <= tau[:, None]).sum(axis=1)


def covered(Sigma, S_true, n, alphas):
    """(m,) bool, true label inside the set."""
    E_true = (n + 1) * S_true / (Sigma + S_true)
    return E_true <= 1.0 / np.asarray(alphas, dtype=float)


def exact_alpha_clf(Sigma, S_all, n, lam, w=1.0, min_alpha=None, eps=1e-6):
    """
    (m,) exact minimiser of Size + lam * w * alpha over (1/(n+1), 1].

    Vectorized version of the per-point `exact_min` in the notebooks; returns the
    same alphas (tests/test_exact.py). Sorted ascending, dropping the j-th
    smallest score (0-indexed) leaves j labels in the set.

    min_alpha : optional lower bound on alpha. Pass the trained net's bound
                (1 / E_loo.max()) to get the exact optimum over the same domain
                the net can reach.
    """
    S_all = np.asarray(S_all, dtype=float)
    m, K  = S_all.shape
    w     = np.broadcast_to(np.asarray(w, dtype=float), (m,))
    lo    = 1.0 / (n + 1) + eps if min_alpha is None else max(1.0 / (n + 1) + eps, float(min_alpha))
    s     = np.sort(S_all, axis=1)
    with np.errstate(divide="ignore"):
        bp = (Sigma / s + 1.0) / (n + 1) + eps
    a  = np.concatenate([bp, np.full((m, 1), lo), np.ones((m, 1))], axis=1)
    sz = np.concatenate([np.broadcast_to(np.arange(K), (m, K)),
                         set_sizes(Sigma, S_all, n, np.full(m, lo))[:, None],
                         (S_all <= Sigma / n).sum(axis=1, keepdims=True)], axis=1)
    ok = (a >= lo) & (a <= 1.0)
    L  = np.where(ok, sz + lam * w[:, None] * a, np.inf)
    return a[np.arange(m), L.argmin(axis=1)]


def smoothed_loss(E, alphas, lam, w=1.0, k_sharp=5.0):
    """
    The surrogate the policy network is trained on (neural_net.smooth_size):
        sum_y sigmoid(k (1/alpha - E(x, y))) + lam * w * alpha
    E: (m, K); alphas: (m,) or (m, G). Returns (m,) or (m, G).
    """
    a = np.asarray(alphas, dtype=float)
    w = np.asarray(w, dtype=float)
    if a.ndim == 1:
        return expit(k_sharp * (1.0 / a[:, None] - E)).sum(1) + lam * w * a
    size = expit(k_sharp * (1.0 / a[:, :, None] - E[:, None, :])).sum(-1)
    return size + lam * np.broadcast_to(w, (len(E),))[:, None] * a


def smoothed_alpha_clf(E, lam, w=1.0, min_alpha=None, max_alpha=1.0, k_sharp=5.0,
                       n_grid=1000, n_refine=200, chunk=2000):
    """
    (m,) per-point minimiser of the smoothed surrogate over [min_alpha, max_alpha].

    Use the same bounds as the trained net: train_alpha_net sets
    min_alpha = 1 / E_loo.max() for classification.
    Coarse grid, then a fine grid around the coarse argmin.
    """
    E = np.asarray(E, dtype=float)
    m = len(E)
    w = np.broadcast_to(np.asarray(w, dtype=float), (m,))
    lo = 1.0 / E.max() if min_alpha is None else float(min_alpha)
    grid = np.linspace(lo, max_alpha, n_grid)
    h = grid[1] - grid[0]
    out = np.empty(m)
    for i in range(0, m, chunk):
        e, ww = E[i:i + chunk], w[i:i + chunk]
        a0 = grid[smoothed_loss(e, np.broadcast_to(grid, (len(e), n_grid)), lam, ww, k_sharp).argmin(1)]
        fine = np.clip(a0[:, None] + np.linspace(-h, h, n_refine)[None, :], lo, max_alpha)
        out[i:i + chunk] = fine[np.arange(len(e)), smoothed_loss(e, fine, lam, ww, k_sharp).argmin(1)]
    return out


# ── regression (MAE scores) ───────────────────────────────────────────────────

def interval_radius(Sigma, n, alphas):
    denom = (n + 1) * np.asarray(alphas, dtype=float) - 1.0
    return np.where(denom > 0, Sigma / np.clip(denom, 1e-12, None), np.inf)


def exact_alpha_reg(Sigma, n, lam, w=1.0, max_alpha=1.0):
    w = np.clip(np.asarray(w, dtype=float), 1e-12, None)
    return np.minimum((1.0 + np.sqrt(2.0 * Sigma * (n + 1) / (lam * w))) / (n + 1), max_alpha)
