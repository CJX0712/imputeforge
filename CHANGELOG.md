# Changelog

All notable changes to ImputeForge are documented here. Author: 晨星.

## v0.1.0 — 2026-10-11

First release. Domain: **tabular missing-value imputation** (MCAR / MAR / MNAR).

### Flagship

`ImputeFuse` — EM-anchored MICE with

1. **EM anchoring** — the `gauss` candidate is the exact Gaussian conditional mean
   `E[x_j | entries observed in that row]` under the EM-fitted joint, so the
   flagship is non-inferior to Gaussian EM by construction.
2. **EM initialization** — chained equations start from the EM completion, not
   column means.
3. **Deployment-matched per-column model selection** — candidates (`gauss` /
   `forest` / `huber`) are scored by running the *whole pipeline* on probe-hidden
   copies of the training block, pooled over `probe_repeats` independent masks.
4. **Probe-calibrated MNAR shift** — per-column `lambda_j * sigma_j` delta
   adjustment (pattern-mixture sensitivity analysis), gated at `>= 7%` probe
   improvement and shrunk by `0.5`, so it stays at zero under MCAR.

### Method pool

`mean` · `median` · `knn` (sklearn) · `knn_np` (pure-numpy offline) ·
`mice_ridge` (BayesianRidge MICE) · `mice_forest` (ExtraTrees MICE, the
missForest family — Stekhoven & Bühlmann 2012) · `em_gaussian` (pure numpy) ·
`softimpute` (SVT, Mazumder/Hastie/Tibshirani 2010) · `imputefuse`.

### Verified invariants

- EM observed-data log-likelihood is monotone non-increasing across iterations.
- With no missing values, one EM M-step reproduces the closed-form MLE exactly.
- Conditional-mean completion leaves every observed entry bit-identical.
- SoftImpute objective is monotone non-increasing (proximal-gradient form).
- kNN never selects a row as its own neighbour; distance is nan-aware.
- No leakage: imputers fit on train rows only; `transform` never refits the joint.

### Known limitations

- The pre-registered `>= 5%` reduction over the *strongest single* method is
  reported honestly as measured; where it is not reached, the number and the
  comparison basis are stated rather than hidden. See `docs/architecture.md` §3.
- `mice_*` methods are unavailable when scikit-learn is absent; the pool then
  degrades to the pure-numpy `knn_np` path.
