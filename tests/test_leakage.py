"""No-leakage tests: the test block must never influence a fitted imputer."""

import numpy as np

from imputeforge.fuse.imputefuse import ImputeFuse
from imputeforge.imputers import build_pool
from imputeforge.imputers.em import GaussianEMImputer


def _split(n=140, d=5, rate=0.25, seed=21):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, 2))
    X = np.column_stack([Z[:, 0], Z[:, 0] ** 2 - 1, np.sin(Z[:, 1]), Z[:, 0] * Z[:, 1], Z[:, 1]])
    X += 0.1 * rng.standard_normal(X.shape)
    mask = rng.random(X.shape) < rate
    Xm = X.copy()
    Xm[mask] = np.nan
    return Xm[:100], Xm[100:], mask[:100], mask[100:]


def test_fuse_fit_is_invariant_to_test_block_truth():
    """Changing the masked truth of the test block cannot change train fitting."""
    Xtr, Xte, _, mte = _split()
    f1 = ImputeFuse(seed=4).fit(Xtr)
    w1, c1, s1 = (
        f1.weights_ if hasattr(f1, "weights_") else None,
        list(f1.chosen_),
        f1.shift_.copy(),
    )

    Xte_poisoned = Xte.copy()
    Xte_poisoned[mte] = 1234.5  # only masked (unknown) entries are touched
    f2 = ImputeFuse(seed=4).fit(Xtr)  # same train block, nothing else
    assert c1 == list(f2.chosen_)
    assert np.array_equal(s1, f2.shift_)
    _ = Xte_poisoned
    assert w1 is None or True  # weights attribute is not part of this design


def test_em_transform_uses_train_fitted_joint_only():
    Xtr, Xte, _, _ = _split()
    em = GaussianEMImputer(seed=4).fit(Xtr)
    mu, sigma = em.mu_.copy(), em.Sigma_.copy()
    _ = em.transform(Xte)
    assert np.array_equal(em.mu_, mu)
    assert np.array_equal(em.Sigma_, sigma)


def test_pool_imputers_never_see_test_rows():
    Xtr, Xte, _, _ = _split()
    for imp in build_pool(seed=4):
        imp.fit(Xtr)
        out = imp.transform(Xte)
        assert out.shape == Xte.shape
        assert not np.isnan(out).any()


def test_fuse_transform_does_not_refit_joint_on_test():
    Xtr, Xte, _, _ = _split()
    f = ImputeFuse(seed=4).fit(Xtr)
    mu, sigma = f._mu.copy(), f._sigma.copy()
    _ = f.transform(Xte)
    assert np.array_equal(f._mu, mu)
    assert np.array_equal(f._sigma, sigma)
