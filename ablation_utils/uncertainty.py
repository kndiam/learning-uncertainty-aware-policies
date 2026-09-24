"""Uncertainty estimators for the coverage-policy ablation."""

import numpy as np
from dataclasses import dataclass
from sklearn.base import clone
from sklearn.neighbors import NearestNeighbors
from sklearn.base import is_regressor  
    

@dataclass
class UncertaintyData:
    """
    X : (n, d)  inputs
    E : (n, K)  e-values, optional  
    S : (n, K)  candidate-label scores, optional
    P : (n, K)  p-values, optional 
    """
    X: np.ndarray
    E: np.ndarray | None = None
    S: np.ndarray | None = None
    P: np.ndarray | None = None

    def require(self, *fields):
        for f in fields:
            if getattr(self, f) is None:
                raise ValueError(
                    f"{type(self).__name__} needs `{f}` for this method; got None."
                )


class UncertaintyMethod:

    def fit(self, data):
        return self

    def transform(self, data):
        raise NotImplementedError

    def calibrate(self, data):
        """Record the reference scale from the CALIBRATION split. Call only once."""
        u = np.asarray(self.transform(data), dtype=float)
        self.n_calib_ = len(u)
        self.u_sum_   = float(u.sum())
        self.u_scale_ = float(u.mean())
        if not np.isfinite(self.u_scale_) or self.u_scale_ <= 0:
            raise ValueError(f"{type(self).__name__}: degenerate u_scale_.")
        return self

    def _check_calibrated(self):
        if not hasattr(self, "u_scale_"):
            raise RuntimeError(
                f"{type(self).__name__}: call calibrate(calib_data) first."
            )

    def transform_loo(self, data):
        """u_i / mean_{j != i} u_j. Used for the calibration split."""
        self._check_calibrated()
        u = np.asarray(self.transform(data), dtype=float)
        if len(u) != self.n_calib_:
            raise ValueError(
                f"transform_loo expects the calibration split "
                f"(n={self.n_calib_}), got n={len(u)}."
            )
        return u * (self.n_calib_ - 1) / np.clip(self.u_sum_ - u, 1e-12, None)

    def transform_normalized(self, data):
        """u / u_scale_. Use at test time."""
        self._check_calibrated()
        return np.asarray(self.transform(data), dtype=float) / self.u_scale_


# ── Probability ensemble ───────────────────────────────────────────────────────

class ProbabilityEnsemble:
    """
    k independently seeded classifiers -- the source of second-order probabilities.
    """

    def __init__(self, base_estimator, k=5, seeds=None):
        self.k       = k
        self.seeds   = list(range(k)) if seeds is None else list(seeds)
        assert len(self.seeds) == k
        self.members = [clone(base_estimator) for _ in self.seeds]
        for m, s in zip(self.members, self.seeds):
            if "random_state" in m.get_params():
                m.set_params(random_state=s)
        self._fitted = False

    def fit(self, X, y, bootstrap=True):
        X, y = np.asarray(X), np.asarray(y)
        self.classes_ = np.unique(y)
        n = len(X)
        for m, s in zip(self.members, self.seeds):
            idx = np.random.default_rng(s).integers(0, n, n) if bootstrap else np.arange(n)
            m.fit(X[idx], y[idx])
        self._fitted = True
        return self

    def predict_all(self, X):
            """(k, n) member predictions."""
            return np.stack([m.predict(X) for m in self.members], axis=0)

    
    def predict_proba_all(self, X):
        """(k, n, K)"""
        if not self._fitted:
            raise RuntimeError("ProbabilityEnsemble: call fit(X, y) first.")
        return np.stack([m.predict_proba(X) for m in self.members], axis=0)

    def predict_proba(self, X):
        """scalarized mean over ensemble members: (n, K)"""
        return self.predict_proba_all(X).mean(axis=0)

    def predict(self, X):
        """Label with the highest averaged probability, (n,)."""
        if is_regressor(self.members[0]):
            return self.predict_all(X).mean(axis=0)
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def score(self, X, y):
        y = np.asarray(y)
        if is_regressor(self.members[0]):                 
            return float(np.abs(y - self.predict(X)).mean())
        return float((self.predict(X) == y).mean())


def conformal_pvalues(S_calib_true, S_test_all):
    n = len(S_calib_true)
    counts = (S_calib_true[None, :, None] >= S_test_all[:, None, :]).sum(axis=1)
    return (1 + counts) / (n + 1)

def conformal_pvalues_loo(S_calib_true, S_calib_all):
    """
    Conformal p-values for calibration points used as pseudo-test points.

    Point j is excluded from its own rank count, so the LOO calibration set has
    size n-1 and the denominator is n. Use conformal_pvalues() at test time.
    """
    n        = len(S_calib_true)
    counts   = (S_calib_true[None, :, None] >= S_calib_all[:, None, :]).sum(axis=1)
    self_hit = (S_calib_true[:, None] >= S_calib_all)          # (n, K)
    return (1 + counts - self_hit) / n

def soft_rank_evalues(S_calib_true, S_test_all):
    """E(x, y) = (n+1) S(x, y) / (Sigma + S(x, y))"""
    n = len(S_calib_true)
    return (n + 1) * S_test_all / (S_calib_true.sum() + S_test_all)


def evals_to_pvals(evals):
    return np.minimum(1.0, 1.0 / np.clip(evals, 1e-15, None))

def mmi_tv(pvals):
    """Second-largest p-value per point."""
    return np.sort(pvals, axis=1)[:, ::-1][:, 1]

def mmi_pi(pvals):
    """MMI = sum_{k=2}^{K+1} (pi_(k-1) - pi_(k)) pi_(k),  pi_(K+1) = 0."""
    desc   = np.sort(pvals, axis=1)[:, ::-1]
    padded = np.concatenate([desc, np.zeros((len(desc), 1))], axis=1)
    gaps   = padded[:, :-1] - padded[:, 1:]
    vals   = padded[:, 1:]
    return (gaps[:, 1:] * vals[:, 1:]).sum(axis=1)

def mmi_pi_e(evals):
    """"
    MMI_pi, but using e-values instead of p-values. 
    """
    pvals = evals_to_pvals(evals)
    return mmi_pi(pvals)

def mmi_tv_e(evals):
    """"
    MMI_tv, but using e-values instead of p-values. 
    """
    pvals = evals_to_pvals(evals)
    return mmi_tv(pvals)


def shannon_entropy(probs, axis=-1):
    return -np.sum(np.clip(probs, 1e-15, 1.0) * np.log2(np.clip(probs, 1e-15, 1.0)),
                   axis=axis)

def _members(ensemble):
    """Fitted members of a ProbabilityEnsemble (.members) or an sklearn forest (.estimators_)."""
    for attr in ("members", "estimators_"):
        if hasattr(ensemble, attr):
            return list(getattr(ensemble, attr))
    raise TypeError(f"{type(ensemble).__name__} has neither `.members` nor `.estimators_`.")

class ConstantEU(UncertaintyMethod):
    """Baseline placeholder for uncertainty."""

    def transform(self, data):
        return np.ones(len(data.X), dtype=float)


class EntropyEU(UncertaintyMethod):
    """
    Mutual-information epistemic uncertainty:  EU = TU - AU.

        TU = H[ mean_m p_m ]      entropy of the ensemble-averaged distribution
        AU = mean_m H[ p_m ]      average entropy of the individual members
        EU = TU - AU              how much the label distribution moves when
                                  you learn which member you are using

    Both terms are divided by log2(K) so u is in [0, 1] for any number of
    classes.
    """

    def __init__(self, ensemble):
        self.ensemble = ensemble

    def transform(self, data):
        if is_regressor(_members(self.ensemble)[0]):
            raise TypeError("EntropyEU is classification-only for now; use VarianceEU for regression.")
        probs = self.ensemble.predict_proba_all(data.X)   
        K = probs.shape[-1]

        mean_probs = probs.mean(axis=0)                     
        total_uncertainty = shannon_entropy(mean_probs, axis=-1) / np.log2(K)   

        member_entropies = shannon_entropy(probs, axis=-1) / np.log2(K)         
        aleatoric_uncertainty = member_entropies.mean(axis=0)                   #
        # EU = TU-AU
        return np.clip(total_uncertainty - aleatoric_uncertainty, 0.0, None)


class VarianceEU(UncertaintyMethod):
    """Sum over classes of the probability variance across members."""

    def __init__(self, ensemble):
        self.ensemble = ensemble

    def transform(self, data):
        members = _members(self.ensemble)
        X = np.asarray(data.X)
        if is_regressor(members[0]):
            return np.stack([m.predict(X) for m in members]).var(axis=0)
        return np.stack([m.predict_proba(X) for m in members]).var(axis=0).sum(axis=1)


class KNNRadius(UncertaintyMethod):
    def __init__(self, n_neighbors=25, standardize=True):
        self.n_neighbors = n_neighbors
        self.standardize = standardize

    def fit(self, data):
        X = np.asarray(data.X, dtype=float)
        if self.standardize:
            self.mu_ = X.mean(axis=0)
            self.sd_ = np.clip(X.std(axis=0), 1e-12, None)
            X = (X - self.mu_) / self.sd_
        self.nn_ = NearestNeighbors(n_neighbors=self.n_neighbors).fit(X)
        return self

    def transform(self, data):
        if not hasattr(self, "nn_"):
            raise RuntimeError("KNNRadius: call fit(train_data) first.")
        X = np.asarray(data.X, dtype=float)
        if self.standardize:
            X = (X - self.mu_) / self.sd_
        return self.nn_.kneighbors(X)[0][:, -1]


class MMIUncertainty(UncertaintyMethod):
    """
    Maximal Mean Imprecision over the credal set induced by the conformal p-values.

    Reads data.P. The caller supplies LOO p-values for the calibration split and
    full-calibration p-values at test time.

    form : "tv" (second-largest p-value) or "pi" (full form)
    """

    def __init__(self, form="pi", evals=False):
        if form not in ("tv", "pi"):
            raise ValueError("form must be 'tv' or 'pi'.")
        self.form = form
        self.evals = evals

    def transform(self, data):
        if self.evals:
            data.require("E")
            E = np.asarray(data.E, dtype=float)
            return mmi_tv_e(E) if self.form == "tv" else mmi_pi_e(E)
        else:
            data.require("P")
            P = np.asarray(data.P, dtype=float)
        
            return mmi_tv(P) if self.form == "tv" else mmi_pi(P)