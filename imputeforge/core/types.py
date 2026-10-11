"""Shared dataclasses.

Author: 晨星
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Dataset:
    """A complete (unmasked) synthetic matrix plus its provenance."""

    X: np.ndarray  # (n, d) full data, no NaN
    name: str
    dgp: str


@dataclass
class MaskedSplit:
    """Train/test rows with missing entries (NaN) and the ground truth."""

    X_train: np.ndarray  # with NaN at missing entries
    X_test: np.ndarray  # with NaN at missing entries
    X_train_full: np.ndarray
    X_test_full: np.ndarray
    mask_train: np.ndarray  # bool, True = missing
    mask_test: np.ndarray  # bool, True = missing
    cell: str  # "<dgp>/<mechanism>"
    seed: int


@dataclass
class CellResult:
    """Per (cell, seed, method) evaluation record."""

    cell: str
    seed: int
    method: str
    nrmse: float  # aggregate over columns: RMSE_j / std_j, averaged
    per_column: list[float] = field(default_factory=list)
    elapsed_sec: float = 0.0
    skipped: bool = False
    error: str = ""
