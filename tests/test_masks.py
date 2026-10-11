"""Tests for the missingness mechanisms (MCAR / MAR / MNAR)."""

import numpy as np
import pytest

from imputeforge.data.masks import MECHANISMS, apply_mechanism, mask_values


def _X(n=400, d=6, seed=8):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d))


@pytest.mark.parametrize("mech", MECHANISMS)
def test_missing_rate_close_to_target(mech):
    X = _X()
    m = apply_mechanism(X, mech, 0.25, np.random.default_rng(0))
    assert abs(m.mean() - 0.25) < 0.05, f"{mech} rate {m.mean():.3f}"


def test_no_fully_missing_column_or_row():
    X = _X(n=60)
    for mech in MECHANISMS:
        m = apply_mechanism(X, mech, 0.6, np.random.default_rng(1))
        assert not m.all(axis=0).any(), mech
        assert not m.all(axis=1).any(), mech


def test_mnar_removes_high_values():
    X = _X()
    m = apply_mechanism(X, "mnar", 0.25, np.random.default_rng(2))
    masked = np.array([X[m[:, j], j].mean() for j in range(X.shape[1])])
    observed = np.array([X[~m[:, j], j].mean() for j in range(X.shape[1])])
    assert (masked > observed).all()


def test_unknown_mechanism_raises():
    from imputeforge.core.errors import DataError

    with pytest.raises(DataError):
        apply_mechanism(_X(), "nope", 0.2, np.random.default_rng(0))


def test_mask_values_puts_nan_at_mask():
    X = _X(n=20)
    m = np.zeros(X.shape, dtype=bool)
    m[0, 0] = True
    out = mask_values(X, m)
    assert np.isnan(out[0, 0])
    assert not np.isnan(out[1:, :]).any()
