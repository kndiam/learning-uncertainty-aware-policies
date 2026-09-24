"""
One place for everything that must match across notebooks and seeds:
dataset arguments, base models, the lambda rule, and the coverage weight c(u).

    from ablation_utils.config import CLASSIFICATION, REGRESSION, coverage_weight
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ablation_utils import data as D
from ablation_utils.uncertainty import ProbabilityEnsemble


# ── coverage weight c(u) ──────────────────────────────────────────────────────
# Theory chapter: c(0) = 1 and c non-decreasing, so u -> 0 recovers the
# EU-unaware objective and alpha*(x) is non-increasing in u(x).
# u is the calibration-normalised uncertainty (mean 1 on the calibration split).

GAMMA = 1.0


def coverage_weight(u, gamma=GAMMA, method=None):
    """c(u) = 1 + gamma * u.  The constant control carries no uncertainty: c = 1."""
    u = np.asarray(u, dtype=float)
    if method == "constant":
        return np.ones_like(u)
    return 1.0 + gamma * u


# ── classification ────────────────────────────────────────────────────────────
# Base models: one ensemble per dataset, seeded from the experiment seed.

def _tree_ensemble(max_depth=None, k=25):
    def make(seed):
        return ProbabilityEnsemble(DecisionTreeClassifier(max_depth=max_depth), k=k,
                                   seeds=range(1000 * seed, 1000 * seed + k))
    return make


CLASSIFICATION = {
    "BLOBS": dict(
        data=lambda seed: D.blobs(eu_classes=(5, 6, 7, 8), eu_keep=0.1, seed=seed),
        model=_tree_ensemble(max_depth=6)),
    "PENDIGITS": dict(
        data=lambda seed: D.pendigits(eu_classes=(5, 6, 7, 8, 9), eu_keep=0.25, seed=seed),
        model=_tree_ensemble(max_depth=6)),
    "FASHION_MNIST": dict(
        data=lambda seed: D.fashion_mnist(eu_keep=0.3, n_train=10000, n_calib=5000,
                                          n_test=20000, seed=seed),
        model=_tree_ensemble(max_depth=None)),
}


def lambda_upper_clf(E_loo, num_classes):
    """The notebooks' "highest recommended lambda": (K - 1) * max E_loo, rounded down to a multiple of 5.
    Above this, alpha collapses to its floor."""
    return int(((num_classes - 1) * E_loo.max()) // 5) * 5


# The notebooks disagree here: compare_uncertainties uses lam = 10, while
# _closed_form, _eu_aware and _mmi use the upper bound above. Set one value per
# dataset; None means "use the upper bound".
LAMBDA_CLF = {"BLOBS": 10, "PENDIGITS": 10, "FASHION_MNIST": 10}


def lambda_clf(dataname, E_loo, num_classes):
    lam = LAMBDA_CLF.get(dataname)
    return lambda_upper_clf(E_loo, num_classes) if lam is None else lam


# ── regression ────────────────────────────────────────────────────────────────

def _tree_reg_ensemble(k=25):
    def make(seed):
        return ProbabilityEnsemble(DecisionTreeRegressor(min_samples_leaf=5), k=k,
                                   seeds=range(1000 * seed, 1000 * seed + k))
    return make


REGRESSION = {
    "HETERO_1D":  dict(data=lambda seed: D.hetero_1d(seed=seed),  model=_tree_reg_ensemble()),
    "CALIFORNIA": dict(data=lambda seed: D.california(seed=seed), model=_tree_reg_ensemble()),
    "BIKE":       dict(data=lambda seed: D.bike(seed=seed),       model=_tree_reg_ensemble()),
}

ALPHA_TARGET_REG = 0.10


def lambda_reg(Sigma, n_calib, alpha_target=ALPHA_TARGET_REG):
    """Rule used in regression/compare_uncertainties: alpha*(c = 1) = alpha_target."""
    m = n_calib + 1
    return 2.0 * Sigma * m / ((m * alpha_target - 1.0) ** 2)
