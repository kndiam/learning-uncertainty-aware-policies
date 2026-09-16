import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import AutoMinorLocator
from tueplots import bundles
from torch.utils.data import TensorDataset, DataLoader

"""
This file contains the neural network and training code for the adaptive coverage policy learner and associated utility functions.
It also includes utility functions for plotting results, smoothing data, and comparison to other methods of building prediction sets.
"""


def make_features(Sigma, S, u):
    """[Sigma, sorted scores, u] -> (n, K+2).  Training AND evaluation call this."""
    return np.concatenate([
        np.asarray(Sigma).reshape(-1, 1),
        np.sort(S, axis=1),
        np.asarray(u).reshape(-1, 1),
    ], axis=1)


def make_loader(Sigma, S, E, u, batch_size=32, shuffle=True):
    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
    ds = TensorDataset(t(make_features(Sigma, S, u)), t(E), t(u))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

class AlphaNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, max_alpha=1.0, min_alpha=0.0):
        super().__init__()
        self.fc1     = nn.Linear(input_dim, hidden_dim)
        self.fc2     = nn.Linear(hidden_dim, 1)
        self.relu    = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.max_alpha = max_alpha
        self.min_alpha = min_alpha
        self._initialize_to_one()


    def _initialize_to_one(self):
        with torch.no_grad():
            self.fc2.weight.normal_(0, 0.01)
            self.fc2.bias.fill_(5.0)

    def forward(self, x):
        h = self.relu(self.fc1(x))
        a = self.sigmoid(self.fc2(h)).squeeze(-1)
        return self.min_alpha + (self.max_alpha - self.min_alpha) * a


def smooth_size(E, alpha, k_sharp=5.0):
    return torch.sigmoid(k_sharp * (1.0 / alpha.unsqueeze(1) - E)).sum(dim=1)


def soft_rank_evalues(S_calib_true, S_test_all):
    n = len(S_calib_true)
    Sigma = S_calib_true.sum()
    return (n + 1) * S_test_all / (Sigma + S_test_all)


def conformal_set(scores, T, alpha, calibration_size, num_classes):
    eps = 1e-15
    frac = (calibration_size + 1) * scores / (T + scores + eps)
    return [y for y in range(num_classes) if frac[y] <= 1 / alpha]
 
def expected_size_clf(loo_scores, T, alpha, n):
    """
    A class c is in the set when
        (n+1) * s_c / (T + s_c) <= 1/alpha.
    Expected size = mean over calibration points of that count.
 
    loo_scores : (N, C) per-class scores  (first C cols of your X_train)
    T          : (N,) or scalar -- sum of LOO calibration scores (col C of X_train)
    alpha      : miscoverage level
    n          : calibration size in the (n+1) factor
    Returns the mean set size (float).
    """
    S = np.asarray(loo_scores, dtype=float)
    N = S.shape[0]
    T = np.broadcast_to(np.asarray(T, dtype=float), (N,)).reshape(N, 1)
    frac = (n + 1) * S / (T + S)
    sizes = (frac <= 1.0 / alpha).sum(axis=1)
    return float(sizes.mean())

import copy


class EarlyStopping:
    """
    Stop when the loss has not improved by `min_delta` for `patience` epochs,
    and restore the best weights seen.

    Watches TRAINING loss -- there is no held-out split here -- so this is a
    plateau detector, not an overfitting guard. Same as the reference; just
    call it that in the write-up.
    """

    def __init__(self, patience=10, min_delta=1e-4, restore_best_weights=True):
        self.patience             = patience
        self.min_delta            = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss            = float('inf')
        self.counter              = 0
        self.best_weights         = None

    def __call__(self, loss, model):
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter   = 0
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())
            return False

        self.counter += 1
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def _pad_to(arr, length):
        """Forward-fill a truncated run so curves from different runs stack."""
        arr = np.asarray(arr, dtype=float)
        if len(arr) >= length:
            return arr[:length]
        return np.concatenate([arr, np.full(length - len(arr), arr[-1])])

def train_alpha_net(train_loader, lam, run_id=0, num_epochs=200, lr=1e-3,
                    max_alpha=1.0, patience=None, device='cpu'):
    """
    Changes:
      - input_dim inferred from the data instead of hardcoded `1 + K`
      - `lr` is actually used (Adam was pinned at 1e-3)
      - `batch_size` and `K` removed -- both were dead parameters
      - optional early stopping
    """
    input_dim = train_loader.dataset.tensors[0].shape[1]
    E_all = train_loader.dataset.tensors[1]     
    min_alpha = float(1.0 / E_all.max())
    alpha_net = AlphaNet(input_dim=input_dim, max_alpha=max_alpha, min_alpha=min_alpha).to(device)
    optimizer = torch.optim.Adam(alpha_net.parameters(), lr=lr)
    stopper   = EarlyStopping(patience=patience) if patience else None

    all_losses, all_sizes, all_alphas = [], [], []

    for epoch in range(num_epochs):
        total_loss = total_size = total_alpha = 0.0

        for x_batch, E_batch, u_batch in train_loader:
            x_batch = x_batch.to(device)
            E_batch = E_batch.to(device)
            u_batch = u_batch.to(device)

            alpha_pred  = alpha_net(x_batch)
            batch_sizes = smooth_size(E_batch, alpha_pred)
            loss = (batch_sizes + lam * u_batch * alpha_pred).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss  += loss.item()
            total_size  += batch_sizes.mean().item()
            total_alpha += alpha_pred.mean().item()

        nb = len(train_loader)
        all_losses.append(total_loss  / nb)
        all_sizes.append(total_size  / nb)
        all_alphas.append(total_alpha / nb)

        if stopper and stopper(all_losses[-1], alpha_net):
            print(f"  early stop at epoch {epoch}")
            break

        if epoch % 25 == 0 or epoch == num_epochs - 1:
            print(f"  lam {lam} | run {run_id+1} | epoch {epoch}/{num_epochs} | "
                  f"loss {all_losses[-1]:.4f} | size {all_sizes[-1]:.4f} | "
                  f"alpha {all_alphas[-1]:.4f}")

    return alpha_net, {"losses": np.array(all_losses),
                       "sizes":  np.array(all_sizes),
                       "alphas": np.array(all_alphas)}


def multi_lambda_run_train_alpha_net(train_loader, lambdas, num_epochs=200, lr=1e-3,
                                     num_runs=5, max_alpha=1.0, patience=None,
                                     device='cpu'):
    """
    One result per lambda, keyed by lambda. Same shape as
    run_uncertainty_ablation, so plot_performance takes either.
    """
    all_results = {}

    for lam in lambdas:
        losses, sizes, alphas, nets = [], [], [], []

        for run in range(num_runs):
            np.random.seed(run)
            torch.manual_seed(run)
            print(f"\n[lambda={lam}] run {run + 1}/{num_runs}")

            net, meta = train_alpha_net(
                train_loader, lam, run_id=run, num_epochs=num_epochs, lr=lr,
                max_alpha=max_alpha, patience=patience, device=device,
            )
            losses.append(meta["losses"])
            sizes.append(meta["sizes"])
            alphas.append(meta["alphas"])
            nets.append(net)

        all_results[lam] = {
            "losses": np.array(losses),
            "sizes":  np.array(sizes),
            "alphas": np.array(alphas),
            "nets":   nets,
        }

    return all_results


def run_uncertainty_ablation(train_loaders, lam, num_epochs=200, lr=1e-3,
                             num_runs=5, max_alpha=1.0, patience=None, device='cpu'):
    """
    One result per uncertainty method, with key as uncertainty method name.
    """
    all_results = {}

    for name, loader in train_loaders.items():
        losses, sizes, alphas, nets = [], [], [], []

        for run in range(num_runs):
            np.random.seed(run)
            torch.manual_seed(run)
            print(f"\n[{name}] run {run + 1}/{num_runs}")

            net, meta = train_alpha_net(
                loader, lam, run_id=run, num_epochs=num_epochs, lr=lr,
                max_alpha=max_alpha, patience=patience, device=device,
            )
            losses.append(meta["losses"])
            sizes.append(meta["sizes"])
            alphas.append(meta["alphas"])
            nets.append(net)

        all_results[name] = {
            "losses": np.array(losses),
            "sizes":  np.array(sizes),
            "alphas": np.array(alphas),
            "nets":   nets,
        }

    return all_results


def compute_calibration_scores(calibration_subset, net, criterion, device):
    cal_scores = []
    with torch.no_grad():
        for x_c, y_c in calibration_subset:
            x_c = x_c.unsqueeze(0).to(device)
            y_c = torch.tensor([y_c], dtype=torch.long).to(device)
            logit = net(x_c)
            cal_scores.append(criterion(logit, y_c).item())
    T = torch.tensor(sum(cal_scores)).to(device)
    return cal_scores, T


def score_point(x, net, criterion, num_classes, device):
    x = x.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = net(x).squeeze(0)
    scores = [
        criterion(logits.unsqueeze(0), torch.tensor([y], dtype=torch.long).to(device)).item()
        for y in range(num_classes)
    ]
    return torch.tensor(scores).to(device)


def evaluate_adaptive(alpha_net, S_test_all, y_test, Sigma, u_test,
                      n_calib, device='cpu'):
    n_test = len(S_test_all)
    feats  = make_features(np.full(n_test, Sigma), S_test_all, u_test)

    alpha_net.eval()
    with torch.no_grad():
        alphas = alpha_net(
            torch.tensor(feats, dtype=torch.float32).to(device)
        ).cpu().numpy()

    # y in C(x)  <=>  E(x, y) <= 1 / alpha
    E_test = (n_calib + 1) * S_test_all / (Sigma + S_test_all)
    in_set = E_test <= (1.0 / alphas)[:, None]
    sizes     = in_set.sum(axis=1)
    coverages = in_set[np.arange(n_test), y_test].astype(int)

    return coverages, sizes, alphas


def evaluate_fixed(all_scores, all_y_test, T, fixed_alpha, calibration_size, num_classes):
    coverages, sizes = [], []
    for scores_tensor, y_test in zip(all_scores, all_y_test):
        C_fixed = conformal_set(scores_tensor, T, fixed_alpha, calibration_size, num_classes)
        coverages.append(int(y_test in C_fixed))
        sizes.append(len(C_fixed))
    return coverages, sizes


def evaluate_standard(all_scores, all_y_test, cal_scores, alpha, calibration_size, num_classes):
    quantile_index = int(np.ceil((1 - alpha) * (calibration_size + 1))) - 1
    threshold = sorted(cal_scores)[quantile_index]

    coverages, sizes = [], []
    for scores_tensor, y_test in zip(all_scores, all_y_test):
        C_standard = [y for y in range(num_classes) if scores_tensor[y].item() <= threshold]
        coverages.append(int(y_test in C_standard))
        sizes.append(len(C_standard))
    return coverages, sizes


def reliability(coverages, alphas, n_bins=10, min_count=20):
    """
    Empirical coverage against promised coverage 1 - alpha.
    Returns (nominal, empirical, counts).
    """
    coverages = np.asarray(coverages, dtype=float)
    nominal   = 1.0 - np.asarray(alphas, dtype=float)

    edges = np.unique(np.quantile(nominal, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return (np.array([nominal.mean()]),
                np.array([coverages.mean()]),
                np.array([len(coverages)]))

    idx = np.clip(np.digitize(nominal, edges[1:-1]), 0, len(edges) - 2)
    xs, ys, ns = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() >= min_count:
            xs.append(nominal[m].mean())
            ys.append(coverages[m].mean())
            ns.append(m.sum())
    return np.array(xs), np.array(ys), np.array(ns)


##### PLOTTING #######

_PALETTE = ["#0077bb", "#cc3311", "#44aa99", "#ee7733",
            "#332288", "#aa3377", "#999933", "#882255"]


def moving_average(arr, window_size):
    return np.convolve(arr, np.ones(window_size) / window_size, mode="valid")


def smooth_all(arr_2d, window):
    """
    (num_runs, num_epochs) -> (mean, std, window_used) over runs of the
    smoothed curves. Returns the window it actually used, which the caller
    needs to place the x-axis correctly.
    """
    arr_2d = np.asarray(arr_2d, dtype=float)
    window = int(min(window, arr_2d.shape[1]))
    smoothed = np.array([moving_average(run, window) for run in arr_2d])
    return np.nanmean(smoothed, axis=0), np.nanstd(smoothed, axis=0), window


def plot_performance(all_results, path, labels=None, window_size=10,
                     keyword=None, mark_early_stop=True):
    """
    Three panels: loss, mean smooth set size, mean alpha.

    all_results : {name: {"losses", "sizes", "alphas", "stopped_at", "nets"}}
                  each curve array shaped (num_runs, num_epochs)
    labels      : {name: color}. None -> all methods, palette colors.
                  Pass a subset to plot only those.
    keyword     : legend prefix, e.g. "lambda". None -> bare method name.
    """
    if labels is None:
        labels = {name: _PALETTE[i % len(_PALETTE)]
                  for i, name in enumerate(all_results)}

    plt.rcParams.update(bundles.icml2024(usetex=False))
    plt.rcParams.update({
        "axes.labelsize": 18, "axes.titlesize": 18,
        "xtick.labelsize": 16, "ytick.labelsize": 16,
        "legend.fontsize": 16, "lines.linewidth": 2, "axes.linewidth": 2,
    })

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    panels = [("losses", "Loss",                 "Training Loss"),
              ("sizes",  "Mean Size (classes)",  "Mean Smooth Size per Epoch"),
              ("alphas", r"Mean $\tilde\alpha$", r"Mean $\tilde\alpha$ per Epoch")]

    for ax, (key, ylab, title) in zip(axs, panels):
        for name, color in labels.items():
            res = all_results[name]
            mean, std, w = smooth_all(res[key], window_size)

            # mode="valid": smoothed point i covers epochs i .. i+w-1
            ep = np.arange(len(mean)) + (w - 1) / 2 + 1

            legend = name if keyword is None else f"{keyword}={name}"
            ax.plot(ep, mean, label=legend, color=color)
            ax.fill_between(ep, mean - std, mean + std, color=color, alpha=0.3)

            # dot the tail wherever the curve is forward-filled padding
            if mark_early_stop and "stopped_at" in res:
                stop = int(np.median(res["stopped_at"]))
                if stop < res[key].shape[1]:
                    cut = max(int(stop - (w - 1) / 2 - 1), 1)
                    ax.plot(ep[cut - 1:], mean[cut - 1:],
                            color=color, linestyle=":", linewidth=2)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(True)

    axs[2].legend(frameon=True)

    for ax in axs:
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(which="both",  length=4)
        ax.tick_params(which="minor", length=2, width=1.5)
        ax.tick_params(which="major", width=2)

    plt.tight_layout()
    plt.savefig(path, format="pdf", bbox_inches="tight")
    plt.show()