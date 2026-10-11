"""Metrics: normalized RMSE per column (RMSE_j / std_j of the full column).

Score semantics: smaller is better. Perfect imputation => 0.0.

Author: 晨星
"""

from __future__ import annotations

import numpy as np

from imputeforge.core.errors import DataError


def column_nrmse(
    X_true: np.ndarray, X_imp: np.ndarray, mask: np.ndarray
) -> tuple[float, list[float]]:
    """Return (aggregate NRMSE, per-column NRMSE) over masked entries.

    ``mask`` marks the entries to evaluate (True = evaluate here). Columns with
    nothing to evaluate are dropped (not filled with 0) so aggregates stay
    comparable across methods.
    """
    X_true = np.asarray(X_true, dtype=np.float64)
    X_imp = np.asarray(X_imp, dtype=np.float64)
    if X_true.shape != X_imp.shape:
        raise DataError("shape mismatch between truth and imputed matrices")
    mask = np.asarray(mask, dtype=bool)
    per_col: list[float] = []
    for j in range(X_true.shape[1]):
        sel = mask[:, j]
        if sel.sum() < 3:
            continue
        col = X_true[:, j]
        sd = float(np.std(col))
        if sd < 1e-12:
            continue
        rmse = float(np.sqrt(np.mean((X_imp[sel, j] - col[sel]) ** 2)))
        per_col.append(rmse / sd)
    if not per_col:
        raise DataError("no evaluable column: mask too sparse")
    return float(np.mean(per_col)), per_col
