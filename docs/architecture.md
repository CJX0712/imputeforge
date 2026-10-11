# ImputeForge — Architecture

Author: 晨星 · v0.1.0

Tabular missing-value imputation under MCAR / MAR / MNAR. The flagship
`ImputeFuse` is an EM-anchored MICE loop with **deployment-matched per-column
conditional-model selection** and **probe-calibrated MNAR shift correction**.

---

## 1. Call graph (acyclic, single direction)

```
cli.py ──► pipeline ──► { data (dgp, masks), imputers, fuse, eval } ──► core
                                    │
                                    └─► imputeforge.core.{types,errors,config,seed,interfaces}
```

`core` imports nothing from the upper layers; `fuse` depends on `imputers` but
never the reverse; `eval` is a leaf. Every layer is independently testable.

| Package | Responsibility |
|---------|----------------|
| `core/` | `Config` (env-overridable + schema-validated), typed `Dataset`/`MaskedSplit`/`CellResult`, error taxonomy `E100–E500`, single seed entry `set_all()`, `Imputer` Protocol |
| `data/` | Four DGPs + three missingness mechanisms with calibrated rates |
| `imputers/` | Method pool: simple, kNN (sklearn + pure-numpy), MICE (ridge/forest), Gaussian EM, SoftImpute |
| `fuse/` | `ImputeFuse` flagship |
| `eval/` | Column-wise NRMSE |
| `pipeline/` | Cell construction, evaluation, aggregation, headline, ablation, determinism |

---

## 2. Why not "blend the outputs of a pool" — measured, negative, kept

The obvious design — run several imputers and average / weight / stack their
outputs — was implemented and measured six different ways before being rejected.
All numbers below are reductions in aggregate column-NRMSE relative to the
strongest single method; **negative means the fusion rule lost**.

| Fusion rule | Effect vs strongest single method |
|-------------|----------------------------------|
| per-column CV selection over pool outputs | −1.5% |
| inverse-error softmax weighting (T=1) | negative |
| inverse-error softmax weighting (T=4) | negative |
| inverse-error softmax weighting (T=0.5) | negative |
| equal-weight average | −23.2% |
| stacking (non-negative least squares) | negative |
| stacking with shrinkage | negative |

Root cause (diagnosed, not guessed):

1. **Dilution.** Averaging admits weak members (mean/median imputation sit at
   NRMSE ≈ 1.21 while the strong methods sit at 0.83–0.86); no weighting scheme
   recovers that.
2. **Zero headroom on homogeneous DGPs.** On `gauss_lowrank` the per-column
   oracle *coincides* with the globally best method, so selecting per column
   cannot beat it and can only lose through estimation noise.
3. **Selection noise > between-method difference.** On heavy-tailed cells the
   Spearman correlation between probe (CV) error and test error across methods
   was only **0.52** — the selection statistic is too noisy to rank methods.

Conclusion: the decision had to move one level down — not *which imputer's
output to trust*, but *which conditional model predicts each column inside one
MICE loop*. That is what `ImputeFuse` does.

---

## 3. The flagship: four mechanisms, each independently necessary

```
fit(X_train)
  │
  ├─ (1) EM joint  ──────────► (mu, Sigma);  gauss candidate = E[x_j | x_obs(row)]
  │
  ├─ (2) probe scoring ──────► for r in probe_repeats:
  │                              hide probe_rate of observed entries (uniform ∪ extreme-value)
  │                              run the WHOLE pipeline on the hidden copy
  │                              pool squared error per (column, candidate, shift)
  │                            → per-column winner in {gauss, forest, huber} + lambda_j
  │
  ├─ (3) EM init ────────────► chained equations start from the EM completion
  │
  └─ (4) MICE sweeps ────────► max_iter sweeps; store the final per-column model
```

| # | Mechanism | Ablation (removing it) |
|---|-----------|------------------------|
| 1 | EM anchoring (`gauss` candidate = exact Gaussian conditional mean) | structural — see §6 |
| 2 | Deployment-matched per-column selection | `no_selection` (one global model) is worse |
| 3 | EM initialization | `mean_init` clearly worse |
| 4 | Probe-calibrated MNAR shift | `no_shift` worse in aggregate, much worse on MNAR cells |

Exact ablation deltas are printed by `python examples/run_demo.py` and stored in
`examples/ablation.json`; nothing here is hand-typed.

### 3.1 Deployment-matched scoring (the fix that mattered most)

Scoring candidates on a **mean-imputed feature matrix** — the obvious shortcut —
is biased toward linear models: the features a candidate sees at scoring time are
not the ones it sees at deployment. Measured cost: **4–15% on nonlinear cells**.
`_score()` therefore hides probe entries and runs the *entire pipeline* on the
hidden copy, so the candidate is measured on the distribution it will actually
face.

### 3.2 Repeated-probe pooling

With a single probe mask, the per-column estimate is noisy enough to flip the
choice between candidates whose true difference is smaller than the noise (the
root cause in §2.3). `_score()` pools squared errors over `probe_repeats`
independent masks (repeated cross-validation) and selects the shift `lambda_j`
on the pooled objective as well. Measured sweep (2 seeds, baseline
`em_gaussian = 0.8433`):

| probe_repeats | margin | fuse | reduction |
|---------------|--------|------|-----------|
| 1 | 0.00 | 0.8231 | +2.40% |
| 1 | 0.02 | 0.8246 | +2.23% |
| 2 | 0.00 | 0.8245 | +2.23% |
| **2** | **0.02** | **0.8212** | **+2.63%** ← shipped default |
| 2 | 0.05 | 0.8306 | +1.51% |
| 3 | 0.02 | 0.8205 | +2.71% |
| 3 | 0.05 | 0.8332 | +1.20% |

The gain is real but small, which is itself the finding: **the binding
constraint is the benchmark's headroom, not the selection rule.**

### 3.3 MNAR shift calibration

Under MNAR the observed part of a column is a biased sample of its marginal, so
every conditional mean is displaced. A per-column shift `lambda_j * sigma_j` is
calibrated on the probe over the grid `±{0.25, 0.5, 0.75} σ` — the standard
pattern-mixture sensitivity-analysis move. Two guards keep it honest:

- **Gate** (`shift_gate = 0.93`): accept only if probe error improves by ≥ 7%.
- **Shrink** (`shift_shrink = 0.5`): apply half the calibrated displacement.

Verified consequence: on MCAR cells the gate rejects the shift and `ImputeFuse`
reproduces Gaussian EM **bit-for-bit** (`maxdiff = 0.0000`). The mechanism fires
only where the data actually demand it.

---

## 4. Method pool

| Name | Implementation | Family |
|------|----------------|--------|
| `mean` / `median` | numpy | trivial floor |
| `knn` | sklearn `KNNImputer` (nan-Euclidean) | neighbourhood |
| `knn_np` | pure numpy, nan-aware, self-excluded | offline fallback |
| `mice_ridge` | sklearn `IterativeImputer(BayesianRidge)` | classic MICE (van Buuren) |
| `mice_forest` | sklearn `IterativeImputer(ExtraTreesRegressor)` | **missForest family** (Stekhoven & Bühlmann 2012) |
| `em_gaussian` | pure numpy, pattern-grouped EM | joint Gaussian MLE |
| `softimpute` | pure numpy SVT | Mazumder/Hastie/Tibshirani 2010 |
| `imputefuse` | this work | flagship |

Tier-0 reuse: scikit-learn provides MICE, kNN, trees and linear models; the
kernels that sklearn does not ship (Gaussian EM, SoftImpute) are implemented
directly on numpy with textbook-faithful algorithms. Nothing is downloaded at
runtime; when sklearn is absent the pool degrades to `knn_np` and the bench marks
the missing entries `skipped` rather than inventing numbers.

---

## 5. Invariants (each guarded by a test)

| Invariant | Test |
|-----------|------|
| EM observed-data log-likelihood is monotone non-increasing | `tests/test_em.py` |
| With no missing values, one EM M-step = closed-form MLE | `tests/test_em.py` |
| Conditional-mean completion leaves observed entries bit-identical | `tests/test_em.py` |
| EM beats column-mean on correlated data | `tests/test_em.py` |
| SoftImpute objective is monotone non-increasing | `tests/test_softimpute.py` |
| SoftImpute is a no-op on fully observed input | `tests/test_softimpute.py` |
| kNN never picks a row as its own neighbour; distance is nan-aware | `tests/test_simple_knn.py` |
| `gauss`-only `ImputeFuse` with `no_shift` ≡ Gaussian EM | `tests/test_fuse.py` |
| No leakage: changing masked test truth cannot change a fit | `tests/test_leakage.py` |
| Mask rates hit their target after calibration | `tests/test_masks.py` |

---

## 6. Non-inferiority to EM (structural, not empirical)

The `gauss` candidate predicts `x_j` from the entries **observed in that row**,
which under the EM-fitted joint is exactly the Gaussian conditional mean —
i.e. exactly what `em_gaussian` does, but computed inside the MICE loop. Hence
`ImputeFuse` with all columns forced to `gauss` and `no_shift` reproduces
Gaussian EM bit-for-bit. Gains over EM therefore come only from columns where
the probe says a nonlinear or robust conditional model is genuinely better; the
downside is bounded by selection noise, which repeated-probe pooling reduces.

## 7. No-leakage protocol

- Every imputer is `fit` on training rows only.
- `transform` never refits the joint: `ImputeFuse._init_from_joint()` reuses the
  `(mu, Sigma)` fitted on train. (An earlier version refit EM on the block being
  imputed — this both violated the fit/transform contract and *lost* accuracy.)
- The per-column NRMSE normalizer is the standard deviation of the generator's
  full column, a property of the DGP rather than of the test block.
- Probe masks hide only entries that are observed in the training block; the
  probe error never sees test rows.

## 8. Determinism

`core.seed.set_all(seed)` is the single seed entry; it also pins
`OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS = 1`, because multi-threaded BLAS makes
`svd`/`solve` results order-dependent. `python cli.py detcheck` runs one cell
twice and asserts the metric vector is bit-identical.

## 9. Complexity

Per MICE sweep over `d` columns with `m` candidate models: `O(d · m · T(n, d))`
where `T` is the cost of fitting the conditional model (ExtraTrees:
`O(n log n · d · trees)`). Probe scoring multiplies this by
`probe_repeats · |CANDIDATES|`. EM is `O(k · d³)` for `k` missingness patterns
(usually far fewer than `2^d`).

## 10. Known limitations

- Point estimates only; no multiple imputation and no predictive intervals, so
  imputed values must not be treated as observations in downstream inference.
- Numeric columns only in this release.
- The per-column selection statistic remains noisy when a column has very few
  observed entries (`< 5` probe cells ⇒ column falls back to the global winner).
- See §3.2: the benchmark's headroom, not the fusion rule, is the binding
  constraint on measured gains.
