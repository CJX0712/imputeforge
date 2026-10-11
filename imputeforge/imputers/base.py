"""Base imputer machinery + backend detection.

Author: 晨星
"""

from __future__ import annotations

import numpy as np

from imputeforge.core.errors import ShapeError

_SKLEARN_OK: bool | None = None


def available_sklearn() -> bool:
    """Probe whether sklearn is importable (cached)."""
    global _SKLEARN_OK
    if _SKLEARN_OK is None:
        try:
            import sklearn  # noqa: F401

            _SKLEARN_OK = True
        except ImportError:
            _SKLEARN_OK = False
    return _SKLEARN_OK


def check_matrix(X: np.ndarray) -> np.ndarray:
    """Validate the input contract; return as float64 ndarray."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ShapeError(f"X must be 2-D, got ndim={X.ndim}")
    if X.shape[0] < 2 or X.shape[1] < 1:
        raise ShapeError(f"X too small: {X.shape}")
    if np.isinf(X).any():
        raise ShapeError("X contains inf")
    return X


class BaseImputer:
    """Template base: fit stores nothing by default; subclasses override."""

    name = "base"

    def __init__(self) -> None:
        self._fitted = False

    def fit(self, X: np.ndarray) -> BaseImputer:
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise ShapeError(f"{self.name}: transform before fit")

    @staticmethod
    def _col_fill_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Observed-column means and stds (std floor 1e-8), NaN-aware."""
        obs = ~np.isnan(X)
        counts = obs.sum(axis=0).astype(float)
        safe = np.where(obs, X, 0.0)
        means = safe.sum(axis=0) / np.maximum(counts, 1.0)
        var = (np.where(obs, (X - means) ** 2, 0.0)).sum(axis=0) / np.maximum(counts, 1.0)
        stds = np.sqrt(np.maximum(var, 0.0))
        stds = np.where(stds < 1e-8, 1.0, stds)
        means = np.where(counts < 1, 0.0, means)
        return means, stds
