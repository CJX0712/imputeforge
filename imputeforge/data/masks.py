"""Missingness mechanisms: MCAR / MAR / MNAR (Little & Rubin taxonomy).

All generators return a boolean mask (True = missing) and are deterministic
given the injected rng. Target rates are matched by bisection on the logit
intercept so every cell has a comparable missing rate.

Author: 晨星
"""

from __future__ import annotations

import numpy as np

from imputeforge.core.errors import DataError

MECHANISMS = ("mcar", "mar", "mnar")


def mcar(X: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Missing Completely At Random: iid Bernoulli(rate)."""
    return rng.random(X.shape) < rate


def _calibrate_intercept(logit_lin: np.ndarray, rate: float) -> float:
    """Bisection on intercept so mean(sigmoid(lin + b)) ~= rate."""

    def mean_prob(b: float) -> float:
        z = np.clip(logit_lin + b, -30.0, 30.0)
        return float(np.mean(1.0 / (1.0 + np.exp(-z))))

    lo, hi = -30.0, 30.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if mean_prob(mid) < rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def mar(X: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Missing At Random: missingness of column j depends on *other* columns.

    Every column gets a driver set disjoint from itself, so the mask is a
    deterministic function of observed values only (by construction MAR).
    """
    n, d = X.shape
    mask = np.zeros((n, d), dtype=bool)
    perm = rng.permutation(d)
    for j in range(d):
        # driver = next column in a fixed permutation (never j itself)
        driver = perm[(int(np.where(perm == j)[0][0]) + 1) % d]
        lin = 1.5 * X[:, driver]
        b = _calibrate_intercept(lin, rate)
        prob = 1.0 / (1.0 + np.exp(-(lin + b)))
        mask[:, j] = rng.random(n) < prob
    return mask


def mnar(X: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Missing Not At Random: high values of the column itself go missing."""
    n, d = X.shape
    mask = np.zeros((n, d), dtype=bool)
    for j in range(d):
        lin = 2.0 * X[:, j]  # own (unobserved-after-masking) value drives it
        b = _calibrate_intercept(lin, rate)
        prob = 1.0 / (1.0 + np.exp(-(lin + b)))
        mask[:, j] = rng.random(n) < prob
    return mask


_GENERATORS = {"mcar": mcar, "mar": mar, "mnar": mnar}


def apply_mechanism(
    X: np.ndarray, mechanism: str, rate: float, rng: np.random.Generator
) -> np.ndarray:
    """Return a boolean missing mask (True = missing) for ``mechanism``."""
    if mechanism not in _GENERATORS:
        raise DataError(f"unknown mechanism {mechanism!r}; choose from {sorted(_GENERATORS)}")
    mask = _GENERATORS[mechanism](X, rate, rng)
    # Guard: no fully-missing column or row (degenerate for fitting).
    n, d = X.shape
    for j in range(d):
        if mask[:, j].all():
            mask[rng.integers(0, n), j] = False
    for i in range(n):
        if mask[i, :].all():
            mask[i, rng.integers(0, d)] = False
    return mask


def mask_values(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return a copy of X with np.nan at masked entries."""
    out = X.astype(float).copy()
    out[mask] = np.nan
    return out
