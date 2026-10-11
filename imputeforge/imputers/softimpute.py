"""SoftImpute: nuclear-norm regularized imputation via SVT (pure numpy).

Mazumder, Hastie & Tibshirani 2010 objective:
    min_Z  1/2 || P_Omega(X - Z) ||_F^2 + lambda ||Z||_*

Solved by iterative singular-value soft-thresholding; lambda is selected on an
augmented-mask cross-validation grid computed *inside* fit, so no holdout truth
leaks in (the truth of the augmented entries is never used at deploy time).
Hard invariant: the objective is monotone non-increasing across iterations.

Author: 晨星
"""

from __future__ import annotations

import numpy as np

from imputeforge.core.errors import ConvergenceError
from imputeforge.imputers.base import BaseImputer, check_matrix


class SoftImputer(BaseImputer):
    """SVT-based low-rank imputer with CV-selected lambda."""

    name = "softimpute"

    def __init__(
        self,
        lambda_grid: tuple[float, ...] = (0.05, 0.1, 0.2),
        max_iter: int = 60,
        tol: float = 1e-6,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.lambda_grid = lambda_grid
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

    # ---- SVT ------------------------------------------------------------
    def _svt(self, X: np.ndarray, lam: float) -> tuple[np.ndarray, list[float]]:
        means, _ = self._col_fill_stats(X)
        obs = ~np.isnan(X)
        if obs.all():
            # Nothing to complete: the minimiser of 1/2||P_Omega(X-Z)||^2 + lam||Z||_*
            # is not X itself once lam > 0, so short-circuit to keep the imputer a
            # strict no-op on fully observed input.
            self.objectives_ = [0.0]
            return X.copy(), [0.0]
        Z = np.where(obs, X, means)
        sigma_max = float(np.linalg.svd(Z, compute_uv=False)[0])
        lam = lam * max(sigma_max, 1e-8)
        objs: list[float] = []
        # Mazumder, Hastie & Tibshirani (2010) SoftImpute in its proximal-gradient
        # form:  Z_{k+1} = S_lambda( P_Omega(X) + P_Omega^perp(Z_k) ).
        # The clamp happens on the way IN (it is exactly Z_k - grad g(Z_k) with
        # g(Z) = 1/2||P_Omega(Z-X)||_F^2, whose Lipschitz constant is 1, so step
        # size 1 is optimal).  The iterate itself must NOT be re-clamped on the
        # way out: doing so breaks the descent property and the objective rises.
        prev = Z
        for _ in range(self.max_iter):
            A = np.where(obs, X, Z)
            U, s, Vt = np.linalg.svd(A, full_matrices=False)
            thr = np.maximum(s - lam, 0.0)
            Z_new = (U * thr) @ Vt
            # Singular values of Z_new are exactly `thr`, so the nuclear norm is
            # thr.sum(); no second SVD is needed and this is the true objective.
            obj = 0.5 * float(np.sum((Z_new[obs] - X[obs]) ** 2)) + lam * float(thr.sum())
            objs.append(obj)
            if not np.isfinite(obj):  # pragma: no cover - defensive
                raise ConvergenceError("SoftImpute objective non-finite")
            if len(objs) > 1 and objs[-1] > objs[-2] + 1e-9 * max(1.0, abs(objs[-2])):
                raise ConvergenceError("SoftImpute objective increased")
            delta = float(np.linalg.norm(Z_new - prev))
            prev = Z_new
            Z = Z_new
            if delta < self.tol:
                break
        self.objectives_ = objs
        return Z, objs

    # ---- lambda selection (no holdout truth used) -----------------------
    def _pick_lambda(self, X: np.ndarray) -> float:
        rng = np.random.default_rng(self.seed + 9176)
        obs = ~np.isnan(X)
        if obs.sum() < 10:
            return float(self.lambda_grid[0])
        probe = (rng.random(X.shape) < 0.1) & obs
        if probe.sum() < 5:
            return float(self.lambda_grid[0])
        Xp = X.copy()
        Xp[probe] = np.nan
        best, best_err = float(self.lambda_grid[0]), np.inf
        for lam in self.lambda_grid:
            Z, _ = self._svt(Xp, lam)
            err = float(np.sqrt(np.mean((Z[probe] - X[probe]) ** 2)))
            if err < best_err:
                best, best_err = lam, err
        return best

    def fit(self, X: np.ndarray) -> SoftImputer:
        X = check_matrix(X)
        if not np.isnan(X).any():
            self.lambda_ = 0.0
            self.objectives_ = [0.0]
            self.completed_ = X.copy()
            return super().fit(X)
        self.lambda_ = self._pick_lambda(X)
        Z, objs = self._svt(X, self.lambda_)
        self.objectives_ = objs
        self.completed_ = Z
        return super().fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._require_fitted()
        X = check_matrix(X)
        out = X.copy()
        miss = np.isnan(out)
        if not miss.any():
            return out
        Z, _ = self._svt(X, self.lambda_)
        out[miss] = Z[miss]
        return out
