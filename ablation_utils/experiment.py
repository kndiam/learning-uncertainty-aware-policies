"""
One seed of the EU-adaptivity experiment, end to end, returning per-test-point arrays.

    res = run_classification_seed("PENDIGITS", seed=0, learned=True)
    save(res, "results/clf_PENDIGITS_g1.0_seed0.npz")

Everything that varies with `seed`: the data split and EU thinning, the base-model
ensemble, and the policy nets. Everything else comes from ablation_utils/config.py.

Result keys (m = n_test):
    eu (m,) bool, y_test (m,), lam, gamma, n_calib, seed
    <method>/u, <method>/w                        normalised u(x) and weight c(u(x))
    <method>/<policy>/alpha|size|miss            policy in {exact, exact_bounded, smoothed, learned}
    regression also stores <method>/<policy>/aisl
    learned arrays are (num_runs, m).
"""

import numpy as np

from ablation_utils import config as C
from ablation_utils import exact as X
from ablation_utils.uncertainty import (UncertaintyData, ConstantEU, EntropyEU, VarianceEU,
                                        KNNRadius, MMIUncertainty, conformal_pvalues,
                                        conformal_pvalues_loo)


def _methods_clf(model):
    return {"constant": ConstantEU(), "entropy": EntropyEU(model), "variance": VarianceEU(model),
            "knn": KNNRadius(n_neighbors=25, standardize=False), "mmi_pi": MMIUncertainty(form="pi")}


def _methods_reg(model):
    return {"constant": ConstantEU(), "variance": VarianceEU(model),
            "knn": KNNRadius(n_neighbors=25, standardize=True)}


def _learned(train_loaders, lam, seed, regression, num_runs, num_epochs, lr, device):
    from ablation_utils import neural_net as anet      # torch only needed here
    return anet.run_uncertainty_ablation(train_loaders, lam, num_epochs=num_epochs, lr=lr,
                                         num_runs=num_runs, patience=20, device=device,
                                         regression=regression, seed=seed)


# ── classification ────────────────────────────────────────────────────────────

def run_classification_seed(dataname, seed, gamma=C.GAMMA, learned=False, num_runs=4,
                            num_epochs=200, lr=1e-3, device="cpu"):
    cfg  = C.CLASSIFICATION[dataname]
    data = cfg["data"](seed)
    model = cfg["model"](seed).fit(data.X_train, data.y_train)
    K, n = data.num_classes, data.n_calib

    S_calib_all = 1.0 - model.predict_proba(data.X_calib)
    S_calib     = S_calib_all[np.arange(n), data.y_calib]
    S_test_all  = 1.0 - model.predict_proba(data.X_test)
    S_test      = S_test_all[np.arange(len(data.y_test)), data.y_test]

    Sigma_loo = S_calib.sum() - S_calib
    E_loo     = n * S_calib_all / (Sigma_loo[:, None] + S_calib_all)          # n_loo + 1 = n
    Sigma     = float(S_calib.sum())
    E_test    = (n + 1) * S_test_all / (Sigma + S_test_all)
    lam       = C.lambda_clf(dataname, E_loo, K)
    min_alpha = 1.0 / E_loo.max()                                              # train_alpha_net's bound

    calib_data = UncertaintyData(X=data.X_calib, E=E_loo, S=S_calib_all,
                                 P=conformal_pvalues_loo(S_calib, S_calib_all))
    test_data  = UncertaintyData(X=data.X_test, E=E_test, S=S_test_all,
                                 P=conformal_pvalues(S_calib, S_test_all))
    methods = _methods_clf(model)

    res = dict(eu=np.isin(data.y_test, data.eu_classes), y_test=data.y_test, lam=lam,
               gamma=gamma, n_calib=n, seed=seed, min_alpha=min_alpha,
               base_acc=float((model.predict(data.X_test) == data.y_test).mean()))
    loaders = {}
    for name, m in methods.items():
        m.fit(UncertaintyData(X=data.X_train)).calibrate(calib_data)
        u_loo, u_test = m.transform_loo(calib_data), m.transform_normalized(test_data)
        w_loo, w_test = (C.coverage_weight(u, gamma, name) for u in (u_loo, u_test))
        res[f"{name}/u"], res[f"{name}/w"] = u_test, w_test

        policies = {
            "exact":         X.exact_alpha_clf(Sigma, S_test_all, n, lam, w_test),
            "exact_bounded": X.exact_alpha_clf(Sigma, S_test_all, n, lam, w_test, min_alpha=min_alpha),
            "smoothed":      X.smoothed_alpha_clf(E_test, lam, w_test, min_alpha=min_alpha),
        }
        for pol, a in policies.items():
            res[f"{name}/{pol}/alpha"] = a
            res[f"{name}/{pol}/size"]  = X.set_sizes(Sigma, S_test_all, n, a)
            res[f"{name}/{pol}/miss"]  = ~X.covered(Sigma, S_test, n, a)

        if learned:
            from ablation_utils import neural_net as anet
            loaders[name] = anet.make_loader(Sigma_loo, S_calib_all, E_loo, u_loo, w=w_loo)
            res[f"{name}/_u_test_raw"] = u_test

    if learned:
        from ablation_utils import neural_net as anet
        nets = _learned(loaders, lam, seed, False, num_runs, num_epochs, lr, device)
        for name in methods:
            u_test = res.pop(f"{name}/_u_test_raw")
            runs = [anet.evaluate_adaptive(net, S_test_all, data.y_test, Sigma, u_test, n, device=device)
                    for net in nets[name]["nets"]]
            res[f"{name}/learned/alpha"] = np.stack([a for _, _, a in runs])
            res[f"{name}/learned/size"]  = np.stack([s for _, s, _ in runs])
            res[f"{name}/learned/miss"]  = np.stack([1 - c for c, _, _ in runs]).astype(bool)
    return res


# ── regression ────────────────────────────────────────────────────────────────

def run_regression_seed(dataname, seed, gamma=C.GAMMA, learned=False, num_runs=4,
                        num_epochs=200, lr=1e-2, device="cpu"):
    cfg  = C.REGRESSION[dataname]
    data = cfg["data"](seed)
    model = cfg["model"](seed).fit(data.X_train, data.y_train)
    n = data.n_calib

    f_test  = model.predict(data.X_test)
    S_calib = np.abs(model.predict(data.X_calib) - data.y_calib)
    S_test  = np.abs(f_test - data.y_test)
    Sigma_loo, Sigma = S_calib.sum() - S_calib, float(S_calib.sum())
    lam = C.lambda_reg(Sigma, n)

    calib_data, test_data = UncertaintyData(X=data.X_calib), UncertaintyData(X=data.X_test)
    methods = _methods_reg(model)

    res = dict(eu=np.asarray(data.region_test, bool), lam=lam, gamma=gamma, n_calib=n,
               seed=seed, base_mae=float(S_test.mean()))

    def _store(name, pol, a):
        r = X.interval_radius(Sigma, n, a)
        res[f"{name}/{pol}/alpha"] = a
        res[f"{name}/{pol}/size"]  = 2 * r
        res[f"{name}/{pol}/miss"]  = S_test > r
        res[f"{name}/{pol}/aisl"]  = 2 * r + (2.0 / a) * np.maximum(S_test - r, 0.0)

    loaders, U_test = {}, {}
    for name, m in methods.items():
        m.fit(UncertaintyData(X=data.X_train)).calibrate(calib_data)
        u_loo, u_test = m.transform_loo(calib_data), m.transform_normalized(test_data)
        w_loo, w_test = (C.coverage_weight(u, gamma, name) for u in (u_loo, u_test))
        res[f"{name}/u"], res[f"{name}/w"] = u_test, w_test
        _store(name, "exact", X.exact_alpha_reg(Sigma, n, lam, w_test))
        if learned:
            from ablation_utils import neural_net as anet
            loaders[name] = anet.make_loader_reg(Sigma_loo, n - 1, u_loo, w=w_loo)
            U_test[name] = u_test

    if learned:
        import torch
        from ablation_utils import neural_net as anet
        nets = _learned(loaders, lam, seed, True, num_runs, num_epochs, lr, device)
        for name in methods:
            feats = torch.tensor(anet.make_features_reg(Sigma, n, U_test[name]), dtype=torch.float32)
            A = []
            for net in nets[name]["nets"]:
                net.eval()
                with torch.no_grad():
                    A.append(net(feats.to(device)).cpu().numpy())
            keys = ("alpha", "size", "miss", "aisl")
            per_run = []
            for a in A:
                _store(name, "learned", a)
                per_run.append([res[f"{name}/learned/{k}"] for k in keys])
            for j, k in enumerate(keys):
                res[f"{name}/learned/{k}"] = np.stack([r[j] for r in per_run])
    return res


# ── io ────────────────────────────────────────────────────────────────────────

def save(res, path):
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in res.items()})


def load(path):
    with np.load(path) as z:
        return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}
