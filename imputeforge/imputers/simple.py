"""Simple univariate imputers: column mean / median.

Author: 晨星
"""

from __future__ import annotations

import numpy as np

from imputeforge.imputers.base import BaseImputer, check_matrix


class MeanImputer(BaseImputer):
    name = "mean"

    def fit(self, X: np.ndarray) -> MeanImputer:
        X = check_matrix(X)
        self.fill_, _ = self._col_fill_stats(X)
        return super().fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._require_fitted()
        X = check_matrix(X)
        out = X.copy()
        miss = np.isnan(out)
        if miss.any():
            idx = np.where(miss)
            out[idx] = self.fill_[idx[1]]
        return out


class MedianImputer(BaseImputer):
    name = "median"

    def fit(self, X: np.ndarray) -> MedianImputer:
        X = check_matrix(X)
        fills = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            col = X[:, j]
            obs = col[~np.isnan(col)]
            fills[j] = float(np.median(obs)) if obs.size else 0.0
        self.fill_ = fills
        return super().fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._require_fitted()
        X = check_matrix(X)
        out = X.copy()
        miss = np.isnan(out)
        if miss.any():
            idx = np.where(miss)
            out[idx] = self.fill_[idx[1]]
        return out
