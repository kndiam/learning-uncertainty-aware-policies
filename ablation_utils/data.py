"""Datasets and the EU flag for the coverage-policy ablation."""

import numpy as np
from dataclasses import dataclass
import pandas as pd


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
    [0, 1].
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



@dataclass
class RegSplitData:
    """
    SplitData class for regression
    X_*, y_*  :X  as in SplitData, y is float
    eu_train, eu_calib, eu_test : bool masks, True inside the thinned region
    eu_region : str  human-readable description of the region
    sigma_test : (n_te,) true noise sd, synthetic data only
    """
    X_train: np.ndarray
    y_train: np.ndarray
    X_calib: np.ndarray
    y_calib: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    eu_train: np.ndarray
    eu_calib: np.ndarray
    eu_test: np.ndarray
    eu_region: str
    name: str = ""
    sigma_test: np.ndarray | None = None
 
    @property
    def n_calib(self):
        return len(self.y_calib)
 
    @property
    def group_test(self):
        """(n_te,) of "eu" / "rest", by test input."""
        return np.where(self.eu_test, "eu", "rest")
 
    def summary(self):
        lines = [
            f"{self.name}: n_train = {len(self.y_train)}, "
            f"n_calib = {self.n_calib}, n_test = {len(self.y_test)}, "
            f"d = {self.X_train.shape[1]}",
            f"  eu region      {self.eu_region}",
            f"  y range        [{self.y_test.min():.3g}, {self.y_test.max():.3g}]"
            f"   sd {self.y_test.std():.3g}",
        ]
        for name, tr, te in (("rest", ~self.eu_train, ~self.eu_test),
                             ("eu",    self.eu_train,  self.eu_test)):
            lines.append(f"  {name:<5}  train {int(tr.sum()):>6}"
                         f"   test {int(te.sum()):>6}")
        return "\n".join(lines)
 
 
def undersample_region(in_region, keep, rng):
    """
    Indices to keep. Points with in_region True are cut to `keep`:
    a fraction of their count if keep < 1, an absolute count if keep >= 1.
    Points outside the region are untouched.
    """
    mask = np.ones(len(in_region), dtype=bool)
    idx = np.flatnonzero(in_region)
    if len(idx) == 0:
        raise ValueError("EU region is empty in the training split.")
    n = int(keep) if keep >= 1 else round(keep * len(idx))
    n = int(np.clip(n, 1, len(idx)))
    mask[rng.choice(idx, len(idx) - n, replace=False)] = False
    return mask
 
 
def _split_idx(n, n_train, n_calib, n_test, seed):
    if n_train + n_calib + n_test > n:
        raise ValueError(f"requested {n_train + n_calib + n_test} rows, "
                         f"dataset has {n}.")
    idx = np.random.default_rng(seed).permutation(n)
    a, b, c = n_train, n_train + n_calib, n_train + n_calib + n_test
    return idx[:a], idx[a:b], idx[b:c]
 
 
def _reg_split(X, y, region, eu_keep, n_train, n_calib, n_test, seed,
               eu_region, name):
    i_tr, i_ca, i_te = _split_idx(len(y), n_train, n_calib, n_test, seed)
    keep = undersample_region(region[i_tr], eu_keep,
                              np.random.default_rng(seed + 3))
    return RegSplitData(
        X_train=X[i_tr][keep], y_train=y[i_tr][keep],
        X_calib=X[i_ca], y_calib=y[i_ca],
        X_test=X[i_te], y_test=y[i_te],
        eu_train=region[i_tr][keep], eu_calib=region[i_ca],
        eu_test=region[i_te],
        eu_region=eu_region, name=name,
    )
 
 
def hetero_sine(eu_interval=(0.5, 1.5), eu_keep=0.03, 
                n_train=2000, n_calib=1000, n_test=2000, amplitude=10,
                x_range=(-3.0, 3.0), noise_base=0.1, noise_slope=0.3, seed=0):
    """
    y = sin(2x) + 0.3x + sigma(x) eps,  sigma(x) = noise_base + noise_slope |x|,
    x ~ U(x_range), eps ~ N(0, 1).
    Training points with x in `eu_interval` are thinned. 
    The true noise sd on the test split is returned as `sigma_test`.
    """
    lo, hi = eu_interval
 
    def gen(n, s):
        r = np.random.default_rng(s)
        x = r.uniform(*x_range, size=n)
        sig = noise_base + noise_slope * np.abs(x)
        y = amplitude * np.sin(2 * x) + 0.3 * x + sig * r.normal(size=n)
        return x[:, None].astype(np.float32), y.astype(np.float32), sig
 
    X_train, y_train, _      = gen(n_train, seed)
    X_calib, y_calib, _      = gen(n_calib, seed + 1)
    X_test,  y_test,  s_test = gen(n_test,  seed + 2)
 
    in_eu = lambda X: (X[:, 0] >= lo) & (X[:, 0] <= hi)
    keep = undersample_region(in_eu(X_train), eu_keep,
                              np.random.default_rng(seed + 3))
 
    return RegSplitData(
        X_train=X_train[keep], y_train=y_train[keep],
        X_calib=X_calib, y_calib=y_calib,
        X_test=X_test, y_test=y_test,
        eu_train=in_eu(X_train)[keep], eu_calib=in_eu(X_calib),
        eu_test=in_eu(X_test),
        eu_region=f"x in [{lo}, {hi}]", name="hetero-sine",
        sigma_test=s_test.astype(np.float32),
    )
 
 
def bike_sharing(eu_interval=(7, 8, 9), eu_keep=0.03, log_target=False,
                 n_train=8000, n_calib=2000, n_test=5000, seed=0):
    """
    UCI Bike Sharing, hourly (OpenML "Bike_Sharing_Demand" v2), 17 379 rows.
    Training rows whose hour is in eu_interval are thinned;

    """
    from sklearn.datasets import fetch_openml
 
    df = fetch_openml("Bike_Sharing_Demand", version=2, as_frame=True).frame
    y = df.pop("count").to_numpy(dtype=np.float32)
    if log_target:
        y = np.log1p(y)
    hour = df["hour"].to_numpy(dtype=int)
 
    cat = [c for c in df.columns if df[c].dtype.name in ("category", "object")]
    df = pd.get_dummies(df, columns=cat, dtype=np.float32)
    X = df.to_numpy(dtype=np.float32)
 
    return _reg_split(X, y, np.isin(hour, eu_interval), eu_keep,
                      n_train, n_calib, n_test, seed,
                      eu_region=f"hour in {list(eu_interval)}",
                      name="bike-sharing" + ("-log" if log_target else ""))
 
 
def california_housing(eu_box=((37.2, 38.2), (-122.6, -121.7)), eu_keep=0.03,
                       n_train=8000, n_calib=2000, n_test=5000, seed=0):
    """
    California Housing (sklearn), 20 640 rows, 8 features.
    
    Training rows inside the lat/long box are thinned; 
    the default box is the SF Bay Area
    eu_box = ((lat_min, lat_max), (lon_min, lon_max)).
    """
    from sklearn.datasets import fetch_california_housing
 
    raw = fetch_california_housing()
    X = raw.data.astype(np.float32)
    y = raw.target.astype(np.float32)
 
    (la0, la1), (lo0, lo1) = eu_box
    lat, lon = X[:, raw.feature_names.index("Latitude")], \
               X[:, raw.feature_names.index("Longitude")]
    region = (lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1)
 
    return _reg_split(X, y, region, eu_keep,
                      n_train, n_calib, n_test, seed,
                      eu_region=f"lat {la0}-{la1}, lon {lo0}-{lo1}",
                      name="california-housing")
 






