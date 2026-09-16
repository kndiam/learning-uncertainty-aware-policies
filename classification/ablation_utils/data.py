"""Datasets and the EU flag for the coverage-policy ablation."""

import numpy as np
from dataclasses import dataclass


@dataclass
class SplitData:
    """
    X_train, y_train : (n_tr, d), (n_tr,)  undersampled for every EU class
    X_calib, y_calib : (n_ca, d), (n_ca,)  untouched
    X_test,  y_test  : (n_te, d), (n_te,)  untouched
    eu_classes : tuple  the classes whose TRAINING data was thinned
    num_classes : int
    name : str

    Only the training split is modified, so calib and test keep the original
    class balance and stay exchangeable.
    """
    X_train: np.ndarray
    y_train: np.ndarray
    X_calib: np.ndarray
    y_calib: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    eu_classes: tuple
    num_classes: int
    name: str = ""

    @property
    def n_calib(self):
        return len(self.y_calib)

    @property
    def group_test(self):
        """(n_te,) of "eu" / "rest", by test label."""
        return np.where(np.isin(self.y_test, self.eu_classes), "eu", "rest")

    def summary(self):
        counts = np.bincount(self.y_train, minlength=self.num_classes)
        lines = [
            f"{self.name}: K = {self.num_classes}, n_train = {len(self.y_train)}, "
            f"n_calib = {self.n_calib}, n_test = {len(self.y_test)}",
            f"  eu classes     {list(self.eu_classes)}",
            f"  train / class  {counts.tolist()}",
        ]
        g = self.group_test
        for name in ("rest", "eu"):
            m = g == name
            if not m.any():
                continue
            cls = np.unique(self.y_test[m])
            lines.append(f"  {name:<5}  train {int(counts[cls].sum()):>6}"
                         f"   test {int(m.sum()):>6}")
        return "\n".join(lines)


def undersample(y, classes, keep, rng):
    """
    Indices to keep. Every class in `classes` is cut to `keep` training points:
    a fraction of its own count if keep < 1, an absolute count if keep >= 1.
    Nothing else is touched.
    """
    mask = np.ones(len(y), dtype=bool)
    for c in classes:
        idx = np.flatnonzero(y == c)
        n = int(keep) if keep >= 1 else round(keep * len(idx))
        n = int(np.clip(n, 1, len(idx)))
        mask[rng.choice(idx, len(idx) - n, replace=False)] = False
    return mask


def blobs(eu_classes=(5, 6, 7, 8, 9), eu_keep=0.03, num_classes=10,
          n_train=2000, n_calib=200, n_test=2000,
          radius=6.0, sigma=1.0, seed=0):
    """
    Isotropic Gaussian blobs on a circle. Centre spacing is
    2 radius sin(pi / K); raise `radius` for an easier baseline, `sigma` for a
    harder one.
    """
    angles  = 2 * np.pi * np.arange(num_classes) / num_classes
    centres = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)

    def gen(n, s):
        r = np.random.default_rng(s)
        y = r.integers(0, num_classes, n)
        X = centres[y] + r.normal(0, sigma, size=(n, 2))
        return X.astype(np.float32), y.astype(np.int64)

    X_train, y_train = gen(n_train, seed)
    X_calib, y_calib = gen(n_calib, seed + 1)
    X_test,  y_test  = gen(n_test,  seed + 2)

    keep = undersample(y_train, eu_classes, eu_keep,
                       np.random.default_rng(seed + 3))

    return SplitData(
        X_train=X_train[keep], y_train=y_train[keep],
        X_calib=X_calib, y_calib=y_calib,
        X_test=X_test, y_test=y_test,
        eu_classes=tuple(eu_classes), num_classes=num_classes,
        name=f"blobs-{num_classes}",
    )


def pendigits(eu_classes=(5, 6, 7, 8, 9), eu_keep=0.03,
              n_train=6000, n_calib=1000, n_test=3000, seed=0):
    """
    UCI Pen-Based Recognition of Handwritten Digits. 10 near-balanced classes,
    10 992 rows, 16 numeric features 
    """
    from sklearn.datasets import fetch_openml

    raw = fetch_openml("pendigits", version=1, as_frame=False)
    X = raw.data.astype(np.float32)
    y = raw.target.astype(np.int64)

    idx = np.random.default_rng(seed).permutation(len(y))
    a, b, c = n_train, n_train + n_calib, n_train + n_calib + n_test
    i_tr, i_ca, i_te = idx[:a], idx[a:b], idx[b:c]

    keep = undersample(y[i_tr], eu_classes, eu_keep,
                       np.random.default_rng(seed + 3))

    return SplitData(
        X_train=X[i_tr][keep], y_train=y[i_tr][keep],
        X_calib=X[i_ca], y_calib=y[i_ca],
        X_test=X[i_te], y_test=y[i_te],
        eu_classes=tuple(eu_classes), num_classes=10,
        name="pendigits",
    )

def fashion_mnist(eu_classes=(4, 5, 6, 7, 8), eu_keep=0.03,
                  n_train=6000, n_calib=1000, n_test=3000, seed=0):
    """
    Fashion-MNIST. 10 balanced classes, 70 000 rows, 784 raw pixels scaled to
    [0, 1]. Harder than pendigits before thinning (~88% for a forest), and the
    classes are unevenly separable: 0/2/4/6 (t-shirt, pullover, coat, shirt)
    overlap, 1/8/9 (trouser, bag, ankle boot) are nearly clean.
    """
    from sklearn.datasets import fetch_openml

    raw = fetch_openml("Fashion-MNIST", version=1, as_frame=False)
    X = raw.data.astype(np.float32) / 255.0
    y = raw.target.astype(np.int64)

    idx = np.random.default_rng(seed).permutation(len(y))
    a, b, c = n_train, n_train + n_calib, n_train + n_calib + n_test
    i_tr, i_ca, i_te = idx[:a], idx[a:b], idx[b:c]

    keep = undersample(y[i_tr], eu_classes, eu_keep,
                       np.random.default_rng(seed + 3))

    return SplitData(
        X_train=X[i_tr][keep], y_train=y[i_tr][keep],
        X_calib=X[i_ca], y_calib=y[i_ca],
        X_test=X[i_te], y_test=y[i_te],
        eu_classes=tuple(eu_classes), num_classes=10,
        name="fashion-mnist",
    )