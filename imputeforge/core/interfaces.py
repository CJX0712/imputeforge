"""Protocol contracts.

Convention: ``X`` carries ``np.nan`` at missing entries. ``fit`` sees only
training rows; ``transform`` fills missing entries of any row block and must
leave observed entries untouched (except documented smoothing imputers).

Author: 晨星
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Imputer(Protocol):
    """Uniform imputer contract."""

    name: str

    def fit(self, X: np.ndarray) -> Imputer:
        """Fit on a matrix with NaN at missing entries (train rows only)."""
        ...

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return a completed copy of X (same shape, no NaN)."""
        ...

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Convenience: fit then transform on the same block."""
        ...
