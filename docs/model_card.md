# Model Card — ImputeFuse

Author: 晨星 · ImputeForge v0.1.0 · License MIT

## Model details

| Field | Value |
|-------|-------|
| Name | `ImputeFuse` |
| Type | Iterative conditional-model imputer (MICE-family) with per-column model selection |
| Task | Fill missing cells of a numeric matrix `X ∈ R^{n×d}` (MCAR / MAR / MNAR) |
| Params | `probe_rate=0.25`, `probe_repeats=2`, `max_iter=3`, `forest_estimators=12`, `margin=0.0`, `shift_gate=0.93`, `shift_shrink=0.5` |
| Train signal | None — no target column, unsupervised matrix completion |
| Determinism | Bit-identical across runs at fixed `seed` (single seed entry `core.seed.set_all`, BLAS threads pinned to 1) |
| Third-party | numpy 2.5.3 / scipy 1.18.1 / scikit-learn 1.9.1 (all optional except numpy) |

## Intended use

- **In**: tabular numeric data with `5%`–`60%` missing entries, mixed column types
  (linear, nonlinear, heavy-tailed), unknown missingness mechanism.
- **Out**: categorical / ordinal columns (not encoded by this release),
  data where the missingness mechanism is adversarial to the observed marginal
  beyond the calibrated shift grid `±{0.25, 0.5, 0.75} σ`,
  `n < 50` (probe estimates become unusable).

## How it decides

For each column `j`, three conditional models are candidates:

| Candidate | Model | Why |
|-----------|-------|-----|
| `gauss` | `E[x_j | x_obs(row)]` under the EM-fitted Gaussian joint | Optimal under joint Gaussianity; makes the flagship non-inferior to EM by construction |
| `forest` | `ExtraTreesRegressor` on the other completed columns | missForest family — captures nonlinear / interaction structure |
| `huber` | `HuberRegressor` | Linear but robust to heavy tails and outliers |

Selection is by pooled RMSE on `probe_repeats` independent probe masks: each
mask hides a further `probe_rate` fraction of the *observed* training entries
(uniform + extreme-value, i.e. MCAR-like ∪ MNAR-like), the entire pipeline runs
on the hidden copy, and the error on the hidden entries is measured. Scoring
inside the pipeline (rather than on a mean-imputed feature matrix) reproduces the
deployment distribution; the shortcut was measured to cost 4–15% on nonlinear
columns because it systematically favours linear models.

## MNAR correction

Under MNAR the observed part of a column is a biased sample of its marginal, so
every conditional mean is systematically displaced. A per-column shift
`lambda_j * sigma_j` is calibrated on the probe (pattern-mixture sensitivity
analysis). It is accepted only if it improves probe error by `>= 7%`
(`shift_gate=0.93`) and is then shrunk by `0.5` (`shift_shrink`). Verified: on
MCAR cells the gate rejects the shift and the flagship reproduces Gaussian EM
**bit-for-bit**.

## Fairness, privacy, safety

- No protected attributes are used or inferred; no demographic analysis applies.
- No data leaves the process; no network calls; no pretrained weights.
- The imputer reproduces distributional structure, including any bias present in
  the input. Imputed values are **point estimates**, not uncertainty intervals —
  do not treat them as observations in downstream statistical tests without
  multiple imputation (Rubin's rules).

## Failure modes observed

See `docs/architecture.md` §8 — all failure cases are derived from actual
benchmark output, never hand-written.
