"""Gaussian EM imputation (pure numpy, offline gold-standard kernel).

MVN with missing entries via exact sufficient-statistic EM
(Dempster-Laird-Rubin 1977; Schafer 1997). The E-step is grouped by missingness
pattern (standard trick): one Cholesky/solve per distinct pattern instead of
one per row, which is what makes the kernel affordable on CPU.

Hard invariants checked by the test suite:

1. observed-data log-likelihood is monotone non-decreasing across iterations
2. with no missing entries, one M-step equals the closed-form MLE
3. conditional-mean imputation reproduces the observed entries exactly

Author: 晨星
"""

from __future__ import annotations

import numpy as np

from imputeforge.core.errors import ConvergenceError
from imputeforge.imputers.base import BaseImputer, check_matrix


class GaussianEMImputer(BaseImputer):
    """EM for the multivariate normal with missing values."""

    name = "em_gaussian"

    def __init__(
        self, max_iter: int = 100, tol: float = 1e-6, ridge: float = 1e-6, seed: int = 0
    ) -> None:
        super().__init__()
        self.max_iter = max_iter
        self.tol = tol
        self.ridge = ridge
        self.seed = seed

    # ---- helpers --------------------------------------------------------
    @staticmethod
    def _patterns(obs: np.ndarray) -> dict[bytes, np.ndarray]:
        """Map missingness pattern -> row indices (single pass over rows)."""
        groups: dict[bytes, list[int]] = {}
        for i in range(obs.shape[0]):
            groups.setdefault(obs[i].tobytes(), []).append(i)
        return {k: np.asarray(v, dtype=np.int64) for k, v in groups.items()}

    @staticmethod
    def observed_loglik(X: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> float:
        """Observed-data log-likelihood, grouped by missingness pattern."""
        obs = ~np.isnan(X)
        total = 0.0
        for rows in GaussianEMImputer._patterns(obs).values():
            o = obs[rows[0]]
            D = X[rows] - mu if o.all() else X[rows][:, o] - mu[o]
            So = Sigma[np.ix_(o, o)]
            sign, logdet = np.linalg.slogdet(So)
            if sign <= 0:  # pragma: no cover - defensive
                return -np.inf
            quad = float(np.sum(D * np.linalg.solve(So, D.T).T))
            total += -0.5 * (len(rows) * o.sum() * np.log(2.0 * np.pi) + len(rows) * logdet + quad)
        return float(total)

    # ---- EM -------------------------------------------------------------
    def _em(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[float]]:
        n, d = X.shape
        obs = ~np.isnan(X)
        groups = self._patterns(obs)
        keys = [(np.asarray(np.frombuffer(k, dtype=bool)), r) for k, r in groups.items()]
        means, _ = self._col_fill_stats(X)
        mu = means.copy()
        Z = np.where(obs, X, mu)
        Sigma = np.cov(Z.T) + self.ridge * np.eye(d)
        Sigma = 0.5 * (Sigma + Sigma.T)
        reg = self.ridge * max(float(np.trace(Sigma)) / d, 1e-8)
        logliks: list[float] = []

        for it in range(self.max_iter):
            Ex = Z.copy()
            Csum = np.zeros((d, d))
            for o, rows in keys:
                m = ~o
                if not m.any():
                    continue
                Soo = Sigma[np.ix_(o, o)]
                Smo = Sigma[np.ix_(m, o)]
                Xo = X[np.ix_(rows, o)]
                B = np.linalg.solve(Soo, Smo.T)  # (|o|, |m|)
                Ex[np.ix_(rows, m)] = mu[m] + (Xo - mu[o]) @ B
                cond = Sigma[np.ix_(m, m)] - Smo @ B
                cond = 0.5 * (cond + cond.T)
                Csum[np.ix_(m, m)] += len(rows) * cond
            mu_new = Ex.mean(axis=0)
            Dm = Ex - mu_new
            Sigma_new = (Dm.T @ Dm + Csum) / n + reg * np.eye(d)
            Sigma_new = 0.5 * (Sigma_new + Sigma_new.T)
            ll = self.observed_loglik(X, mu_new, Sigma_new)
            if logliks and ll < logliks[-1] - 1e-8 * max(1.0, abs(logliks[-1])):
                raise ConvergenceError(
                    f"EM log-likelihood decreased at iter {it}: {ll - logliks[-1]:.3e}"
                )
            logliks.append(ll)
            mu, Sigma = mu_new, Sigma_new
            if len(logliks) > 1 and abs(logliks[-1] - logliks[-2]) < self.tol * max(
                1.0, abs(logliks[-2])
            ):
                break
        return mu, Sigma, logliks

    # ---- contract --------------------------------------------------------
    def fit(self, X: np.ndarray) -> GaussianEMImputer:
        X = check_matrix(X)
        self.mu_, self.Sigma_, self.logliks_ = self._em(X)
        self.n_iter_ = len(self.logliks_)
        return super().fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._require_fitted()
        X = check_matrix(X)
        out = X.copy()
        obs = ~np.isnan(X)
        for o, rows in [
            (np.asarray(np.frombuffer(k, dtype=bool)), r) for k, r in self._patterns(obs).items()
        ]:
            m = ~o
            if not m.any():
                continue
            if not o.any():
                out[np.ix_(rows, m)] = self.mu_[m]
                continue
            Soo = self.Sigma_[np.ix_(o, o)]
            Smo = self.Sigma_[np.ix_(m, o)]
            B = np.linalg.solve(Soo, Smo.T)
            out[np.ix_(rows, m)] = self.mu_[m] + (X[np.ix_(rows, o)] - self.mu_[o]) @ B
        return out
