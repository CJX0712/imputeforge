"""Tests for the simple and kNN imputers."""

import numpy as np

from imputeforge.imputers.knn import KNNImputerNP, KNNImputerSK
from imputeforge.imputers.simple import MeanImputer, MedianImputer


def _data(n=60, d=4, rate=0.3, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    X[rng.random(X.shape) < rate] = np.nan
    return X


def test_mean_imputer_matches_closed_form():
    X = _data()
    out = MeanImputer().fit_transform(X)
    miss = np.isnan(X)
    expected = np.nanmean(X, axis=0)[np.where(miss)[1]]
    assert np.allclose(out[miss], expected)


def test_median_imputer_matches_closed_form():
    X = _data()
    out = MedianImputer().fit_transform(X)
    miss = np.isnan(X)
    expected = np.nanmedian(X, axis=0)[np.where(miss)[1]]
    assert np.allclose(out[miss], expected)


def test_no_missing_is_identity_for_all_simple_methods():
    rng = np.random.default_rng(9)
    X = rng.standard_normal((30, 3))
    for imp in (MeanImputer(), MedianImputer(), KNNImputerNP(k=3)):
        assert np.array_equal(imp.fit_transform(X.copy()), X)


def test_knn_numpy_recovers_exact_twin_row():
    """With k=1 and an exact twin row, the missing entry is recovered exactly."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((20, 3))
    X[1] = X[0]
    Xq = X.copy()
    Xq[0, 2] = np.nan
    out = KNNImputerNP(k=1).fit(Xq).transform(Xq)
    assert abs(out[0, 2] - X[1, 2]) < 1e-12


def test_knn_sklearn_wrapper_runs():
    X = _data(n=80)
    out = KNNImputerSK(k=4).fit_transform(X)
    assert not np.isnan(out).any()
