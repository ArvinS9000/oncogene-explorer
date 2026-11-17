"""Expression transforms.

Anything that *learns* from data (variance ranks, scaling parameters, supervised selection)
is implemented as a scikit-learn transformer so it is fitted only on training folds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


def is_log_scale(values: np.ndarray | pd.DataFrame, max_threshold: float = 30.0) -> bool:
    """Heuristic: log2 microarray intensities rarely exceed ~16; raw intensities reach 1e4+."""
    arr = np.asarray(values, dtype=float)
    return bool(np.nanmax(arr) < max_threshold and np.nanmin(arr) >= 0)


def log2_if_needed(X: pd.DataFrame, max_threshold: float = 30.0,
                   pseudocount: float = 1.0) -> tuple[pd.DataFrame, bool]:
    """Return ``(X', transformed)``. Applies log2(x + pseudocount) only to linear-scale data.

    This is a fixed, sample-independent transform and therefore safe to apply globally.
    """
    if is_log_scale(X.values, max_threshold):
        return X, False
    if (X.values < 0).any():
        raise ValueError("Negative intensities on a linear scale; cannot log-transform safely")
    return np.log2(X + pseudocount), True


class TopVarianceSelector(BaseEstimator, TransformerMixin):
    """Keep the ``n_features`` highest-variance columns *of the training data*.

    Unsupervised, but its ranking is learned, so it lives inside the CV pipeline.
    ``n_features=None`` (or >= n columns) keeps everything.
    """

    def __init__(self, n_features: int | None = 5000):
        self.n_features = n_features

    def fit(self, X, y=None):
        X = check_array(X, dtype=float, ensure_all_finite=True)
        var = X.var(axis=0, ddof=1)
        self.n_features_in_ = X.shape[1]
        k = X.shape[1] if self.n_features is None else min(int(self.n_features), X.shape[1])
        # Stable sort -> deterministic ties.
        idx = np.argsort(-var, kind="stable")[:k]
        self.support_ = np.zeros(X.shape[1], dtype=bool)
        self.support_[idx] = True
        self.variances_ = var
        return self

    def transform(self, X):
        check_is_fitted(self, "support_")
        X = check_array(X, dtype=float, ensure_all_finite=True)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        return X[:, self.support_]

    def get_support(self, indices: bool = False):
        check_is_fitted(self, "support_")
        return np.flatnonzero(self.support_) if indices else self.support_
