"""Synthetic data-generating processes (deterministic given seed).

Three families chosen so that *different* imputation inductive biases win on
different families — the prerequisite for a fusion flagship to add value:

- ``gauss_lowrank``: low-rank correlated Gaussian -> EM-Gaussian / SoftImpute win
- ``nonlinear``:     nonlinear interactions        -> forest-MICE / kNN win
- ``heavytail``:     multivariate Student-t        -> median / robust wins

Author: 晨星
"""

from __future__ import annotations

import numpy as np

from imputeforge.core.errors import DataError


def mixed_hetero(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Heterogeneous columns: linear, nonlinear and heavy-tailed blocks.

    Real tabular data mixes column families; this is the regime where a
    *column-adaptive* imputer has genuine headroom over any single method.
    All blocks share the same latent factors, so every column is predictable
    from the others -- but with a different functional form per block.
    """
    k = 3
    Z = rng.standard_normal((n, k))
    block = max(1, d // 3)
    cols: list[np.ndarray] = []
    while len(cols) < d:
        i = len(cols)
        if i < block:  # linear block (ridge / EM territory)
            cols.append(0.9 * Z[:, i % k] + 0.15 * rng.standard_normal(n))
        elif i < 2 * block:  # nonlinear block (forest territory)
            z0, z1 = Z[:, i % k], Z[:, (i + 1) % k]
            cols.append(np.sin(z0) + 0.6 * z0 * z1 + 0.15 * rng.standard_normal(n))
        else:  # heavy-tailed block (robust territory)
            t = rng.standard_t(4.0, size=n)
            cols.append(0.6 * t + 0.6 * Z[:, i % k] + 0.1 * rng.standard_normal(n))
    X = np.column_stack(cols[:d])
    return _standardize(X)


DGP_NAMES = ("gauss_lowrank", "nonlinear", "heavytail", "mixed_hetero")


def _standardize(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd


def gauss_lowrank(n: int, d: int, rng: np.random.Generator, rank: int = 3) -> np.ndarray:
    """Low-rank factor model with small isotropic noise."""
    Z = rng.standard_normal((n, rank))
    W = rng.standard_normal((rank, d)) / np.sqrt(rank)
    X = Z @ W + 0.15 * rng.standard_normal((n, d))
    return _standardize(X)


def nonlinear(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Nonlinear interactions: squares, sine, products on shared latents.

    Difficulty knob: noise is kept small enough that the conditional structure
    is learnable (best imputer NRMSE ~0.5, clearly better than the mean).
    """
    k = max(3, d // 2)
    Z = rng.standard_normal((n, k))
    cols: list[np.ndarray] = []
    cols.append(Z[:, 0] + 0.1 * rng.standard_normal(n))
    cols.append(Z[:, 0] ** 2 - 1.0 + 0.1 * rng.standard_normal(n))
    cols.append(np.sin(Z[:, 1 % k]) + 0.1 * rng.standard_normal(n))
    cols.append(Z[:, 0] * Z[:, 2 % k] + 0.1 * rng.standard_normal(n))
    j = 4
    while j < d:
        z0, z1 = Z[:, j % k], Z[:, (j + 1) % k]
        cols.append(np.tanh(z0 + z1) + 0.15 * rng.standard_normal(n))
        j += 1
    X = np.column_stack(cols[:d])
    return _standardize(X)


def heavytail(n: int, d: int, rng: np.random.Generator, df: float = 4.0) -> np.ndarray:
    """Multivariate Student-t with AR(1) correlation (heavy tails)."""
    rho = 0.6
    idx = np.arange(d)
    cov = rho ** np.abs(idx[:, None] - idx[None, :])
    L = np.linalg.cholesky(cov + 1e-10 * np.eye(d))
    G = rng.standard_normal((n, d)) @ L.T
    chi2 = rng.chisquare(df, size=(n, 1)) / df
    X = G / np.sqrt(chi2)
    return _standardize(X)


_GENERATORS = {
    "gauss_lowrank": gauss_lowrank,
    "nonlinear": nonlinear,
    "heavytail": heavytail,
    "mixed_hetero": mixed_hetero,
}


def generate(name: str, n: int, d: int, seed: int) -> np.ndarray:
    """Generate a standardized (n, d) matrix for DGP ``name``."""
    if name not in _GENERATORS:
        raise DataError(f"unknown dgp {name!r}; choose from {sorted(_GENERATORS)}")
    rng = np.random.default_rng(seed)
    X = _GENERATORS[name](n, d, rng)
    if X.shape != (n, d):  # pragma: no cover - defensive
        raise DataError(f"dgp {name} returned shape {X.shape}, expected {(n, d)}")
    return X
