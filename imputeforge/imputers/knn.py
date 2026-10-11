"""kNN imputation: sklearn Tier-0 wrapper + pure-numpy offline fallback.

Author: 晨星
"""

from __future__ import annotations

import numpy as np

from imputeforge.core.errors import BackendUnavailable
from imputeforge.imputers.base import BaseImputer, available_sklearn, check_matrix


class KNNImputerSK(BaseImputer):
    """sklearn KNNImputer (nan-Euclidean, uniform weights)."""

    name = "knn"

    def __init__(self, k: int = 5) -> None:
        if not available_sklearn():
            raise BackendUnavailable("sklearn not importable")
        super().__init__()
        self.k = k

    def fit(self, X: np.ndarray) -> KNNImputerSK:
        from sklearn.impute import KNNImputer

        X = check_matrix(X)
        self._model = KNNImputer(n_neighbors=self.k, weights="uniform")
        self._model.fit(X)
        return super().fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._require_fitted()
        X = check_matrix(X)
        return np.asarray(self._model.transform(X), dtype=np.float64)


class KNNImputerNP(BaseImputer):
    """Pure-numpy kNN imputer (offline fallback when sklearn is absent)."""

    name = "knn_np"

    def __init__(self, k: int = 5) -> None:
        super().__init__()
        self.k = k

    def fit(self, X: np.ndarray) -> KNNImputerNP:
        X = check_matrix(X)
        self._train = X.copy()
        self._means, self._stds = self._col_fill_stats(X)
        return super().fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._require_fitted()
        X = check_matrix(X)
        out = X.copy()
        miss = np.isnan(out)
        if not miss.any():
            return out
        Ztr = np.where(np.isnan(self._train), 0.0, self._train - self._means)
        Zx = np.where(np.isnan(X), 0.0, X - self._means)
        rows = np.where(miss.any(axis=1))[0]
        for i in rows:
            cols = np.where(miss[i])[0]
            # Distance is measured only on the dimensions *observed* in the query
            # row, and only against training rows that observe the target column.
            # This is the nan-aware semantics of sklearn's KNNImputer and it is
            # what makes self-matching impossible: a row never observes its own
            # missing cell, so it can never be its own neighbour.
            obsi = ~np.isnan(X[i])
            zi = Zx[i][obsi]
            for j in cols:
                valid = np.where(~np.isnan(self._train[:, j]))[0]
                if valid.size == 0:
                    out[i, j] = self._means[j]
                    continue
                diff = Ztr[np.ix_(valid, np.where(obsi)[0])] - zi
                d = np.einsum("ij,ij->i", diff, diff)
                order = np.argsort(d, kind="stable")[: min(self.k, valid.size)]
                out[i, j] = float(np.mean(self._train[valid[order], j]))
        return out
