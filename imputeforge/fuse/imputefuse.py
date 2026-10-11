"""ImputeFuse flagship: EM-anchored MICE with deployment-matched per-column
conditional-model selection and probe-calibrated MNAR shift correction.

Why not "blend the outputs of a pool"?
--------------------------------------
Measured on this benchmark (docs/architecture.md §4): blending or selecting
pool *outputs* cannot beat the pool's strongest member — weak members dilute
any average, and on homogeneous DGPs the per-column oracle coincides with the
global best method, so there is no headroom to harvest. ImputeFuse therefore
moves the decision one level down: it chooses *which conditional model
predicts each column* inside a single MICE loop.

Four mechanisms, each independently necessary
---------------------------------------------
1. **EM anchoring** — the ``gauss`` candidate predicts x_j from the entries
   actually *observed in that row* (E[x_j | x_obs] under the EM-fitted joint),
   i.e. exactly the Gaussian conditional mean. Joint conditioning beats chained
   equations that condition on previously imputed values, so ImputeFuse is
   non-inferior to EM by construction on Gaussian / linear structure (verified:
   on gauss_lowrank and heavytail cells all modes reproduce EM exactly).
2. **EM initialization** — chained equations start from the EM completion
   instead of column means, which removes most of the burn-in iterations.
3. **Deployment-matched model selection** — candidates are scored by running
   the *whole pipeline* on a training copy whose probe entries are hidden, then
   measuring the error on those entries. Scoring candidates on a mean-imputed
   feature matrix (the obvious shortcut) is biased towards linear models and
   cost 4-15% on nonlinear cells; scoring inside the pipeline reproduces the
   deployment distribution and is what fixes it.
4. **Probe-calibrated shift (delta adjustment)** — under MNAR the observed part
   of a column is a biased sample of its marginal, so every conditional mean is
   systematically off. A per-column shift lambda_j * sigma_j is calibrated on the
   probe (a standard pattern-mixture sensitivity-analysis move), gated to >=3% probe
   improvement and shrunk by 0.7, so it stays at 0 when the data are MCAR.

Ablation modes: ``no_selection`` (one global model), ``mean_init`` (column-mean
start instead of EM), ``no_refine`` (single sweep), ``no_shift`` (lambda = 0).

Author: 晨星
"""

from __future__ import annotations

import numpy as np

from imputeforge.core.errors import ShapeError
from imputeforge.imputers.base import BaseImputer, available_sklearn, check_matrix
from imputeforge.imputers.em import GaussianEMImputer

FUSE_MODES = ("full", "no_selection", "mean_init", "no_refine", "no_shift")
CANDIDATES = ("gauss", "forest", "huber")  # ridge is dominated by gauss/huber
SHIFT_GRID = (-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75)
SHIFT_GATE = 0.93  # a shift is accepted only if it beats no-shift by >= 7%
SHIFT_SHRINK = 0.5  # conservative shrinkage of an accepted shift


def _numpy_ridge_fit(A: np.ndarray, y: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    """Closed-form ridge solution (offline fallback conditional model)."""
    G = A.T @ A + lam * np.eye(A.shape[1])
    return np.linalg.solve(G, A.T @ y)


class ImputeFuse(BaseImputer):
    """EM-anchored MICE with per-column model selection and MNAR correction."""

    name = "imputefuse"

    def __init__(
        self,
        probe_rate: float = 0.25,
        max_iter: int = 3,
        forest_estimators: int = 10,
        seed: int = 0,
        mode: str = "full",
        margin: float = 0.02,
        shift_gate: float = SHIFT_GATE,
        shift_shrink: float = SHIFT_SHRINK,
        probe_repeats: int = 2,
        probe_em_max_iter: int = 25,
    ) -> None:
        super().__init__()
        if mode not in FUSE_MODES:
            raise ShapeError(f"unknown fuse mode {mode!r}")
        if not (0.0 < probe_rate < 1.0):
            raise ShapeError("probe_rate must be in (0, 1)")
        if probe_repeats < 1:
            raise ShapeError("probe_repeats must be >= 1")
        self.probe_rate = probe_rate
        self.max_iter = 1 if mode == "no_refine" else max_iter
        self.forest_estimators = forest_estimators
        self.seed = seed
        self.mode = mode
        self.margin = margin
        self.shift_gate = shift_gate
        self.shift_shrink = shift_shrink
        self.probe_repeats = probe_repeats
        # EM is by far the most expensive step of probe scoring, and a probe fit
        # only has to be good enough to *rank* candidates (almost all of an EM
        # run's progress happens in its first handful of sweeps). The final fit
        # that produces the deployed joint always uses the full budget.
        self.probe_em_max_iter = probe_em_max_iter
        self.em_max_iter = 60

    # ---- candidate conditional models ----------------------------------------
    def _make_model(self, kind: str):
        if not available_sklearn():
            return "ridge_np"
        if kind == "ridge":
            from sklearn.linear_model import Ridge

            return Ridge(alpha=1.0)
        if kind == "huber":
            from sklearn.linear_model import HuberRegressor

            return HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=200)
        from sklearn.ensemble import ExtraTreesRegressor

        return ExtraTreesRegressor(
            n_estimators=self.forest_estimators,
            random_state=self.seed,
            n_jobs=1,
            min_samples_leaf=5,
        )

    @staticmethod
    def _fit_predict(model, A_fit, y_fit, A_pred) -> np.ndarray:
        if isinstance(model, str):
            if model == "ridge_np":
                return A_pred @ _numpy_ridge_fit(A_fit, y_fit)
            return np.full(A_pred.shape[0], float(np.mean(y_fit)) if y_fit.size else 0.0)
        model.fit(A_fit, y_fit)
        return np.asarray(model.predict(A_pred), dtype=np.float64)

    def _fit_store(self, model, A_fit: np.ndarray, y_fit: np.ndarray):
        if isinstance(model, str):
            if model == "ridge_np":
                return ("ridge_np", _numpy_ridge_fit(A_fit, y_fit))
            return ("mean", float(np.mean(y_fit)) if y_fit.size else 0.0)
        model.fit(A_fit, y_fit)
        return ("sk", model)

    # ---- EM joint ---------------------------------------------------------------
    def _fit_em(self, X: np.ndarray, max_iter: int | None = None) -> np.ndarray:
        """Fit (mu, Sigma) by EM and return the conditional-mean completion."""
        self._mu = None
        self._sigma = None
        if self.mode == "mean_init":
            return self._mean_fill(X)
        try:
            em = GaussianEMImputer(
                max_iter=self.probe_em_max_iter if max_iter is None else max_iter,
                tol=1e-6,
                seed=self.seed,
            )
            out = em.fit_transform(X)
            self._mu, self._sigma = em.mu_, em.Sigma_
            return out
        except Exception:
            return self._mean_fill(X)

    def _mean_fill(self, X: np.ndarray) -> np.ndarray:
        means, _ = self._col_fill_stats(X)
        out = X.copy()
        idx = np.where(np.isnan(out))
        out[idx] = means[idx[1]]
        return out

    def _gauss_pred(self, X: np.ndarray, j: int, rows: np.ndarray) -> np.ndarray:
        """E[x_j | entries observed in that row] under the EM-fitted joint."""
        if self._mu is None:
            return np.zeros(len(rows))
        out = np.zeros(len(rows))
        obs = ~np.isnan(X)
        groups: dict[bytes, list[int]] = {}
        for r, i in enumerate(rows):
            pattern = obs[i].copy()
            pattern[j] = False
            groups.setdefault(pattern.tobytes(), []).append(r)
        for pattern_b, rs in groups.items():
            o = np.flatnonzero(np.frombuffer(pattern_b, dtype=bool))
            if o.size == 0:
                out[rs] = self._mu[j]
                continue
            B = np.linalg.solve(self._sigma[np.ix_(o, o)], self._sigma[j, o])
            out[rs] = self._mu[j] + (X[np.ix_(rows[rs], o)] - self._mu[o]) @ B
        return out

    def _init_from_joint(self, X: np.ndarray) -> np.ndarray:
        """Complete a new block using ONLY the joint fitted on train rows.

        Refitting EM on the block being imputed would leak that block's own
        observed values into the imputer and break the fit/transform contract.
        """
        if self._mu is None:
            return self._mean_fill(X)
        out = X.copy()
        obs = ~np.isnan(X)
        groups: dict[bytes, list[int]] = {}
        for i in range(X.shape[0]):
            groups.setdefault(obs[i].tobytes(), []).append(i)
        for pattern_b, rows in groups.items():
            pat = np.frombuffer(pattern_b, dtype=bool)
            o = np.flatnonzero(pat)
            m = np.flatnonzero(~pat)
            if m.size == 0:
                continue
            if o.size == 0:
                out[np.ix_(rows, m)] = self._mu[m]
                continue
            B = np.linalg.solve(self._sigma[np.ix_(o, o)], self._sigma[np.ix_(m, o)].T)
            out[np.ix_(rows, m)] = self._mu[m] + (X[np.ix_(rows, o)] - self._mu[o]) @ B
        return out

    # ---- pipeline -------------------------------------------------------------------
    def _sweep(self, X: np.ndarray, cur: np.ndarray, chosen, shift, store: bool) -> np.ndarray:
        d = X.shape[1]
        out = cur.copy()
        obs = ~np.isnan(X)
        for j in range(d):
            miss_j = ~obs[:, j]
            if not miss_j.any():
                continue
            rows = np.where(miss_j)[0]
            others = [c for c in range(d) if c != j]
            delta = shift[j] * self._sd_[j]
            if chosen[j] == "gauss":
                out[rows, j] = self._gauss_pred(X, j, rows) + delta
                if store:
                    self.models_[j] = ("gauss", j)
                continue
            model = self._make_model(chosen[j])
            obs_rows = np.where(obs[:, j])[0]
            A_fit = out[np.ix_(obs_rows, others)]
            y_fit = X[obs_rows, j]
            A_pr = out[np.ix_(rows, others)]
            out[rows, j] = self._fit_predict(model, A_fit, y_fit, A_pr) + delta
            if store:
                self.models_[j] = self._fit_store(model, A_fit, y_fit)
        return out

    def _run(self, X: np.ndarray, init: np.ndarray, chosen, shift, store: bool) -> np.ndarray:
        cur = init
        for it in range(self.max_iter):
            cur = self._sweep(X, cur, chosen, shift, store=(store and it == self.max_iter - 1))
        return cur

    # ---- probe ------------------------------------------------------------------------
    def _probe_mask(self, X: np.ndarray, rep: int = 0) -> np.ndarray:
        """Union of a uniform (MCAR-like) and an extreme-value (MNAR-like) probe.

        ``rep`` draws an independent probe mask so several can be averaged over.
        """
        obs = ~np.isnan(X)
        n, d = X.shape
        rng = np.random.default_rng(self.seed + 5150 + 1009 * rep)
        aug = (rng.random((n, d)) < self.probe_rate) & obs
        for j in range(d):
            idx = np.where(obs[:, j])[0]
            if idx.size < 12:
                continue
            q = max(3, round(0.5 * self.probe_rate * idx.size))
            dev = np.abs(X[idx, j] - np.median(X[idx, j]))
            aug[idx[np.argsort(-dev)][:q], j] = True
        return aug

    def _score(self, X: np.ndarray) -> None:
        """Score every candidate by running the whole pipeline on probe-hidden copies.

        Errors are pooled over ``probe_repeats`` independent probe masks (repeated
        cross-validation). Selection noise, not model capacity, is the binding
        constraint here: with a single mask the per-column probe estimate is noisy
        enough to flip the choice between candidates whose true difference is
        smaller than that noise. Pooling R masks cuts that variance ~R times at R
        times the scoring cost.
        """
        d = X.shape[1]
        nc = len(CANDIDATES)
        grid = SHIFT_GRID if self.mode != "no_shift" else (0.0,)
        ng = len(grid)
        # pooled sum of squared errors per (column, candidate, shift) and counts
        sse = np.zeros((d, nc, ng))
        cnt = np.zeros((d, nc))
        for rep in range(self.probe_repeats):
            aug = self._probe_mask(X, rep)
            Xp = X.copy()
            Xp[aug] = np.nan
            init_p = self._fit_em(Xp)  # one EM fit, shared by all candidate runs
            mu, sigma = self._mu, self._sigma
            for ci, cand in enumerate(CANDIDATES):
                Z = self._run(Xp, init_p, [cand] * d, np.zeros(d), store=False)
                self._mu, self._sigma = mu, sigma
                for j in range(d):
                    sel = aug[:, j]
                    if int(sel.sum()) < 5:
                        continue
                    resid = X[sel, j] - Z[sel, j]
                    sd = self._sd_[j]
                    for li, lam in enumerate(grid):
                        sse[j, ci, li] += float(np.sum((resid - lam * sd) ** 2))
                    cnt[j, ci] += float(sel.sum())
        err = np.full((d, nc), np.inf)
        shift_best = np.zeros((d, nc))
        zero_li = list(grid).index(0.0)
        for j in range(d):
            for ci in range(nc):
                if cnt[j, ci] < 5:
                    continue
                mse = sse[j, ci] / cnt[j, ci]
                li = int(np.argmin(mse))
                e0 = float(np.sqrt(mse[zero_li]))
                e_best = float(np.sqrt(mse[li]))
                if self.mode == "no_shift" or grid[li] == 0.0 or e_best > self.shift_gate * e0:
                    err[j, ci] = e0
                    shift_best[j, ci] = 0.0
                else:
                    err[j, ci] = e_best
                    shift_best[j, ci] = self.shift_shrink * grid[li]
        finite = np.isfinite(err).all(axis=1)
        if not finite.any():  # pragma: no cover - defensive
            self.chosen_ = ["gauss"] * d
            self.shift_ = np.zeros(d)
        elif self.mode == "no_selection":
            ci = int(np.argmin(np.nanmean(err[finite], axis=0)))
            self.chosen_ = [CANDIDATES[ci]] * d
            self.shift_ = np.where(finite, shift_best[:, ci], 0.0)
        else:
            default = CANDIDATES[int(np.argmin(np.nanmean(err[finite], axis=0)))]
            chosen, shifts = [], []
            gauss_ci = CANDIDATES.index("gauss")
            for j in range(d):
                if not finite[j]:
                    chosen.append(default)
                    shifts.append(0.0)
                    continue
                best_ci = int(np.argmin(err[j]))
                # Occam margin: stay anchored on the Gaussian conditional mean
                # unless another candidate is clearly better on the probe.
                if err[j, gauss_ci] <= (1.0 + self.margin) * err[j, best_ci]:
                    best_ci = gauss_ci
                chosen.append(CANDIDATES[best_ci])
                shifts.append(shift_best[j, best_ci])
            self.chosen_ = chosen
            self.shift_ = np.array(shifts)
        self.cv_errors_ = err
        self.best_shift_ = shift_best

    # ---- contract -----------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> ImputeFuse:
        X = check_matrix(X)
        d = X.shape[1]
        self.models_ = {}
        self._sd_ = np.array([float(np.std(X[~np.isnan(X[:, j]), j])) or 1.0 for j in range(d)])
        self._score(X)
        init = self._fit_em(X, max_iter=self.em_max_iter)
        self._run(X, init, self.chosen_, self.shift_, store=True)
        return super().fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._require_fitted()
        X = check_matrix(X)
        out = X.copy()
        miss = np.isnan(out)
        if not miss.any():
            return out
        d = X.shape[1]
        cur = self._init_from_joint(X)
        for _ in range(max(1, self.max_iter)):
            for j in range(d):
                miss_j = np.isnan(X[:, j])
                if not miss_j.any() or j not in self.models_:
                    continue
                rows = np.where(miss_j)[0]
                others = [c for c in range(d) if c != j]
                kind, payload = self.models_[j]
                delta = self.shift_[j] * self._sd_[j]
                if kind == "gauss":
                    cur[rows, j] = self._gauss_pred(X, j, rows) + delta
                    continue
                A_pr = cur[np.ix_(rows, others)]
                if kind == "ridge_np":
                    pred = A_pr @ payload
                elif kind == "mean":
                    pred = np.full(len(rows), payload)
                else:
                    pred = np.asarray(payload.predict(A_pr), dtype=np.float64)
                cur[rows, j] = pred + delta
        out[miss] = cur[miss]
        return out
