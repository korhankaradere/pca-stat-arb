import numpy as np
import pytest

from pca_stat_arb.pca import fit_eigenportfolios
from pca_stat_arb.residuals import fit_residuals, s_scores
from pca_stat_arb.signals import FLAT, LONG, SHORT, update_positions


def test_eigenportfolios_capture_the_factor_space(factor_market):
    returns, loadings = factor_market
    R = returns.to_numpy()[-252:]
    eig = fit_eigenportfolios(R, n_factors=3)

    # Marchenko-Pastur: pure-noise eigenvalues stay below (1 + sqrt(N/T))^2. The 3 true
    # factors should sit above that edge and the 4th should not.
    T, N = R.shape
    mp_edge = (1 + np.sqrt(N / T)) ** 2
    assert eig.eigenvalues[2] > mp_edge > eig.eigenvalues[3]
    # The first eigenportfolio should be close to "the market": all weights the same sign.
    assert (eig.eigenvectors[:, 0] > 0).all()


def test_regression_residuals_are_orthogonal_to_factors(factor_market):
    returns, _ = factor_market
    R = returns.to_numpy()[-252:]
    eig = fit_eigenportfolios(R, n_factors=3)
    recent = R[-60:]
    F = recent @ eig.weights
    fit = fit_residuals(recent, F)
    eps = recent - fit.alpha - F @ fit.betas.T
    assert np.allclose(F.T @ eps, 0, atol=1e-10)
    # With an intercept, residuals sum to zero, so the cumulative residual ends at 0.
    assert np.allclose(fit.x_last, 0, atol=1e-12)


def test_ou_fit_recovers_mean_reversion_speed():
    rng = np.random.default_rng(1)
    kappa_true, n_paths, L = 30.0, 400, 60
    b = np.exp(-kappa_true / 252)
    X = np.zeros((L + 1, n_paths))
    for t in range(L):
        X[t + 1] = b * X[t] + rng.normal(0, 0.01, n_paths)
    eps = np.diff(X, axis=0)
    # Use a single irrelevant "factor" so the regression is a no-op on a pure residual.
    fit = fit_residuals(eps, rng.normal(0, 1e-6, (L, 1)))
    # Short windows bias kappa upwards (known small-sample AR(1) bias), so check the median is in range.
    assert 0.5 * kappa_true < np.nanmedian(fit.kappa) < 3 * kappa_true


def test_s_scores_are_nan_for_slow_reverters():
    from pca_stat_arb.residuals import ResidualFit

    fit = ResidualFit(
        alpha=np.zeros(3), betas=np.zeros((3, 1)),
        kappa=np.array([50.0, 2.0, np.nan]), m=np.array([0.01, 0.0, 0.0]),
        sigma_eq=np.array([0.02, 0.02, 0.02]), x_last=np.zeros(3),
    )
    s = s_scores(fit, kappa_min=8.4, center_mean=False)
    assert s[0] == pytest.approx(-0.5)
    assert np.isnan(s[1]) and np.isnan(s[2])


def test_signal_state_machine():
    kw = dict(s_open=1.25, s_close_long=0.5, s_close_short=0.75)
    state = np.array([FLAT, FLAT, LONG, LONG, SHORT, SHORT, LONG, LONG])
    s = np.array([-1.5, 1.5, -0.8, -0.4, 0.9, 0.7, 1.4, np.nan])
    out = update_positions(state, s, **kw)
    expected = [LONG, SHORT, LONG, FLAT, SHORT, FLAT, SHORT, FLAT]
    assert out.tolist() == expected
