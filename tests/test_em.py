"""Tests for the EM kernel: monotonicity and closed-form agreement."""

import numpy as np

from imputeforge.imputers.em import GaussianEMImputer


def _data(n=120, d=4, rate=0.3, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    X[rng.random(X.shape) < rate] = np.nan
    return X


def test_loglik_monotone():
    X = _data()
    em = GaussianEMImputer(max_iter=40)
    em.fit(X)
    lls = em.logliks_
    assert len(lls) >= 2
    diffs = np.diff(lls)
    assert diffs.min() > -1e-9 * max(1.0, abs(lls[0])), f"log-likelihood decreased: {diffs.min()}"


def test_no_missing_matches_closed_form_mle():
    rng = np.random.default_rng(11)
    X = rng.standard_normal((200, 4)) @ np.diag([1.0, 2.0, 0.5, 1.5]) + 0.5
    em = GaussianEMImputer(max_iter=50, ridge=0.0)
    em.fit(X)
    assert np.allclose(em.mu_, X.mean(axis=0), atol=1e-8)
    assert np.allclose(em.Sigma_, np.cov(X.T, bias=True), atol=1e-6)


def test_conditional_fill_preserves_observed_entries():
    X = _data()
    out = GaussianEMImputer().fit_transform(X)
    obs = ~np.isnan(X)
    assert np.array_equal(out[obs], X[obs])
    assert not np.isnan(out).any()


def test_correlated_data_improves_over_mean():
    """EM must beat column-mean whenever the columns are correlated.

    (On independent columns the conditional mean collapses to the marginal mean,
    so a correlated / low-rank DGP is the honest setting for this claim.)
    """
    rng = np.random.default_rng(5)
    A = rng.standard_normal((200, 3))
    full = A @ rng.standard_normal((3, 5)) + 0.25 * rng.standard_normal((200, 5))
    X = full.copy()
    miss = rng.random(X.shape) < 0.25
    X[miss] = np.nan
    em = GaussianEMImputer().fit_transform(X)
    col_mean = np.nanmean(X, axis=0)
    mean_fill = np.where(miss, np.broadcast_to(col_mean, X.shape), X)
    err_em = float(np.sqrt(np.mean((em[miss] - full[miss]) ** 2)))
    err_mean = float(np.sqrt(np.mean((mean_fill[miss] - full[miss]) ** 2)))
    assert err_em < 0.9 * err_mean, f"em={err_em:.4f} mean={err_mean:.4f}"
