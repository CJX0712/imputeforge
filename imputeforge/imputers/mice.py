"""MICE (chained equations) imputers via sklearn IterativeImputer.

- MiceRidgeImputer: BayesianRidge chained regressions (classic MICE linear)
- MiceForestImputer: ExtraTrees chained regressions -> the missForest family
  (Stekhoven & Buehlmann 2012), the acknowledged strong baseline in
  Jager/Allhorn/Biessmann 2021 "A benchmark for data imputation methods".

Author: 晨星
"""

from __future__ import annotations

import contextlib
import warnings

import numpy as np

from imputeforge.core.errors import BackendUnavailable
from imputeforge.imputers.base import BaseImputer, available_sklearn, check_matrix


@contextlib.contextmanager
def _quiet():
    """Silence IterativeImputer's 'early stopping not reached' notice locally.

    ``max_iter`` is a deliberate budget in this library, so the notice is
    expected bookkeeping rather than an anomaly; it is suppressed here only
    (never process-wide) to keep benchmark output readable.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yield


class _MiceBase(BaseImputer):
    """Shared wrapper around sklearn.experimental IterativeImputer."""

    def _build(self) -> object:  # pragma: no cover - overridden
        raise NotImplementedError

    def __init__(self, max_iter: int = 4, seed: int = 0) -> None:
        if not available_sklearn():
            raise BackendUnavailable("sklearn not importable")
        super().__init__()
        self.max_iter = max_iter
        self.seed = seed

    def fit(self, X: np.ndarray) -> _MiceBase:
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer

        X = check_matrix(X)
        self._model = IterativeImputer(
            estimator=self._build(),
            max_iter=self.max_iter,
            initial_strategy="mean",
            sample_posterior=False,
            random_state=self.seed,
            skip_complete=True,
        )
        with _quiet():
            self._model.fit(X)
        return super().fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._require_fitted()
        X = check_matrix(X)
        with _quiet():
            return np.asarray(self._model.transform(X), dtype=np.float64)


class MiceRidgeImputer(_MiceBase):
    """MICE with Bayesian Ridge conditional models."""

    name = "mice_ridge"

    def _build(self):
        from sklearn.linear_model import BayesianRidge

        return BayesianRidge()


class MiceForestImputer(_MiceBase):
    """MICE with ExtraTrees conditional models (missForest family)."""

    name = "mice_forest"

    def __init__(self, max_iter: int = 4, n_estimators: int = 12, seed: int = 0) -> None:
        super().__init__(max_iter=max_iter, seed=seed)
        self.n_estimators = n_estimators

    def _build(self):
        from sklearn.ensemble import ExtraTreesRegressor

        return ExtraTreesRegressor(
            n_estimators=self.n_estimators,
            random_state=self.seed,
            n_jobs=1,
            min_samples_leaf=5,
        )
