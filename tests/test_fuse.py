"""Tests for the ImputeFuse flagship."""

import numpy as np
import pytest

from imputeforge.core.errors import ShapeError
from imputeforge.fuse.imputefuse import CANDIDATES, FUSE_MODES, ImputeFuse


def _data(n=120, d=5, rate=0.25, seed=13):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, 2))
    X = np.column_stack([Z[:, 0], Z[:, 0] ** 2 - 1, np.sin(Z[:, 1]), Z[:, 0] * Z[:, 1], Z[:, 1]])
    X += 0.1 * rng.standard_normal(X.shape)
    X[rng.random(X.shape) < rate] = np.nan
    return X


def test_no_missing_is_identity():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((40, 4))
    out = ImputeFuse(seed=1).fit(X.copy()).transform(X.copy())
    assert np.array_equal(out, X)


def test_output_complete():
    X = _data()
    out = ImputeFuse(seed=1).fit(X).transform(X)
    assert not np.isnan(out).any()
    assert out.shape == X.shape


def test_selection_shapes_and_candidates():
    X = _data()
    f = ImputeFuse(seed=1)
    f.fit(X)
    assert len(f.chosen_) == X.shape[1]
    assert len(f.shift_) == X.shape[1]
    assert all(c in CANDIDATES for c in f.chosen_)
    assert f.cv_errors_.shape == (X.shape[1], len(CANDIDATES))


def test_all_modes_run():
    X = _data()
    for mode in FUSE_MODES:
        out = ImputeFuse(seed=2, mode=mode).fit(X).transform(X)
        assert not np.isnan(out).any(), mode


def test_no_shift_mode_forces_zero_shift():
    X = _data()
    f = ImputeFuse(seed=2, mode="no_shift").fit(X)
    assert np.allclose(f.shift_, 0.0)


def test_no_selection_mode_uses_one_candidate():
    X = _data()
    f = ImputeFuse(seed=2, mode="no_selection").fit(X)
    assert len(set(f.chosen_)) == 1


def test_bad_mode_raises():
    with pytest.raises(ShapeError):
        ImputeFuse(mode="does_not_exist")


def test_bad_probe_rate_raises():
    with pytest.raises(ShapeError):
        ImputeFuse(probe_rate=1.5)


def test_gauss_candidate_reproduces_em():
    """all-gauss + no shift must equal plain EM (structural non-inferiority)."""
    from imputeforge.imputers.em import GaussianEMImputer

    X = _data(n=150)
    em = GaussianEMImputer(seed=5).fit(X).transform(X)
    f = ImputeFuse(seed=5, mode="no_shift")
    f.fit(X)
    f.chosen_ = ["gauss"] * X.shape[1]
    f.shift_ = np.zeros(X.shape[1])
    f._fitted = True
    # rebuild stored models with the forced choice
    f._run(X, f._init_from_joint(X), f.chosen_, f.shift_, store=True)
    out = f.transform(X)
    assert np.max(np.abs(out - em)) < 1e-9
