"""Tests for the DGP layer."""

import numpy as np
import pytest

from imputeforge.core.errors import DataError
from imputeforge.data.dgp import DGP_NAMES, generate


@pytest.mark.parametrize("name", DGP_NAMES)
def test_shape_and_standardization(name):
    X = generate(name, 120, 6, seed=0)
    assert X.shape == (120, 6)
    assert np.allclose(X.mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(X.std(axis=0), 1.0, atol=1e-8)


@pytest.mark.parametrize("name", DGP_NAMES)
def test_deterministic_given_seed(name):
    a = generate(name, 80, 5, seed=42)
    b = generate(name, 80, 5, seed=42)
    assert np.array_equal(a, b)


def test_unknown_dgp_raises():
    with pytest.raises(DataError):
        generate("nope", 50, 3, seed=0)


def test_lowrank_structure_has_dominant_singular_values():
    X = generate("gauss_lowrank", 300, 8, seed=1)
    s = np.linalg.svd(X, compute_uv=False)
    assert s[0] > 5 * s[-1]
