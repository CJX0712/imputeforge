"""Tests for the SoftImpute (SVT) kernel."""

import numpy as np

from imputeforge.imputers.softimpute import SoftImputer


def _data(n=80, d=5, rate=0.3, seed=2):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, 2))
    X = A @ rng.standard_normal((2, d)) + 0.1 * rng.standard_normal((n, d))
    X[rng.random(X.shape) < rate] = np.nan
    return X


def test_objective_is_monotone():
    X = _data()
    si = SoftImputer(lambda_grid=(0.1,))
    si.fit(X)
    objs = si.objectives_
    assert len(objs) >= 2
    assert np.diff(objs).max() < 1e-9 * max(1.0, abs(objs[0]))


def test_no_missing_is_identity():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((40, 4))
    out = SoftImputer().fit_transform(X.copy())
    assert np.array_equal(out, X)


def test_output_has_no_nan():
    X = _data()
    out = SoftImputer().fit_transform(X)
    assert not np.isnan(out).any()
