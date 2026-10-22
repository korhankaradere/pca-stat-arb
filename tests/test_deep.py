import numpy as np
import pandas as pd
import pytest
import torch

from conftest import simulate_factor_market
from pca_stat_arb.accounting import pnl_from_holdings
from pca_stat_arb.deep.config import DeepConfig
from pca_stat_arb.deep.model import CNNTransformer, allocate
from pca_stat_arb.deep.residual_panel import build_residual_panel, cumulative_residual_features
from pca_stat_arb.deep.walkforward import run_walkforward

# Small and fast; synthetic dates run from 2015 to mid-2018.
TINY = DeepConfig(n_factors=3, lookback=10, block_days=40, max_epochs=6, patience=3,
                  val_days=60, seeds=(0,), first_trade_year=2017, learning_rate=3e-3)


@pytest.fixture(scope="module")
def market():
    returns, _ = simulate_factor_market(T=900, N=40, kappa=40.0, seed=4)
    return returns


@pytest.fixture(scope="module")
def panel(market):
    return build_residual_panel(market, TINY.n_factors, TINY.pca_window, TINY.beta_window)


def test_residual_positions_earn_exactly_the_stock_pnl(market, panel):
    """w . eps_{t+1} == P&L of holding the implied stocks w - Q B^T w from close t."""
    rng = np.random.default_rng(1)
    T, N = panel.eps.shape
    W_res = np.zeros((T, N))
    days = np.flatnonzero(panel.tradable.any(axis=1))[:-1]
    W_res[days] = rng.normal(size=(len(days), N)) * panel.tradable[days]
    W_stock = np.array([panel.stock_weights(t, W_res[t]) for t in range(T)])
    pnl = pnl_from_holdings(market.index, market.to_numpy(), W_stock, cost_bps=0.0)
    resid_pnl = (W_res[:-1] * np.nan_to_num(panel.eps[1:])).sum(axis=1)
    np.testing.assert_allclose(pnl["gross"].to_numpy()[1:], resid_pnl, rtol=1e-4, atol=1e-6)


def test_panel_and_features_have_no_lookahead(market, panel):
    cut = 600
    noisy = market.copy()
    noisy.iloc[cut + 1 :] = np.random.default_rng(2).normal(0, 0.05, noisy.iloc[cut + 1 :].shape)
    other = build_residual_panel(noisy, TINY.n_factors, TINY.pca_window, TINY.beta_window)
    for a, b in [(panel.eps, other.eps), (panel.Q, other.Q), (panel.B, other.B)]:
        np.testing.assert_array_equal(a[: cut + 1], b[: cut + 1])
    X1, v1 = cumulative_residual_features(panel, TINY.lookback)
    X2, v2 = cumulative_residual_features(other, TINY.lookback)
    np.testing.assert_array_equal(X1[: cut + 1], X2[: cut + 1])
    np.testing.assert_array_equal(v1[: cut + 1], v2[: cut + 1])


def test_allocation_is_unit_gross_and_ignores_invalid_stocks():
    scores = torch.tensor([[1.0, -3.0, 5.0], [0.5, 0.5, 9.0]])
    valid = torch.tensor([[True, True, False], [True, True, False]])
    w = allocate(scores, valid)
    np.testing.assert_allclose(w.abs().sum(dim=1).numpy(), [1.0, 1.0], rtol=1e-6)
    assert (w[:, 2] == 0).all()


def test_model_maps_paths_to_one_score_each():
    model = CNNTransformer(lookback=10)
    assert model(torch.randn(7, 10)).shape == (7,)


def test_walkforward_learns_mean_reversion_and_has_no_lookahead(market, panel):
    res = run_walkforward(market, panel, TINY, last_year=2018, log=lambda *_: None)
    traded = res.residual_weights.abs().sum(axis=1) > 0
    assert traded.loc["2017"].all() and not traded.loc[:"2016"].any()
    # The synthetic residuals are strongly mean-reverting, so a working model makes money.
    assert res.pnl["gross"].mean() > 0

    # Scramble everything after a mid-2017 date: holdings up to that date must not change.
    cut = market.index.get_loc(market.loc["2017-06"].index[0])
    noisy = market.copy()
    noisy.iloc[cut + 1 :] = np.random.default_rng(3).normal(0, 0.05, noisy.iloc[cut + 1 :].shape)
    other = run_walkforward(noisy, build_residual_panel(noisy, TINY.n_factors, TINY.pca_window, TINY.beta_window),
                            TINY, last_year=2018, log=lambda *_: None)
    pd.testing.assert_frame_equal(res.residual_weights.iloc[: cut + 1], other.residual_weights.iloc[: cut + 1])


def _sharpe(x):
    return x.mean() / x.std() * np.sqrt(252)


def test_training_improves_on_the_untrained_network(market, panel):
    from dataclasses import replace

    untrained = run_walkforward(market, panel, replace(TINY, max_epochs=0), last_year=2018, log=lambda *_: None)
    trained = run_walkforward(market, panel, TINY, last_year=2018, log=lambda *_: None)
    assert _sharpe(trained.pnl["gross"]) > _sharpe(untrained.pnl["gross"])


def test_no_profit_on_pure_noise(market):
    """With no mean reversion in the data, the model must not find 'alpha'."""
    noise = pd.DataFrame(np.random.default_rng(0).normal(0, 0.02, market.shape), index=market.index, columns=market.columns)
    res = run_walkforward(noise, build_residual_panel(noise, TINY.n_factors, TINY.pca_window, TINY.beta_window),
                          TINY, last_year=2018, log=lambda *_: None)
    assert abs(_sharpe(res.pnl["gross"])) < 2.0  # ~1.4 years of data: the standard error is ~0.85


def test_interrupted_walkforward_resumes_from_checkpoints(market, panel, tmp_path):
    first = run_walkforward(market, panel, TINY, last_year=2018, log=lambda *_: None, checkpoint_dir=tmp_path)
    msgs = []
    again = run_walkforward(market, panel, TINY, last_year=2018, log=msgs.append, checkpoint_dir=tmp_path)
    assert sum("loaded from checkpoint" in m for m in msgs) == 2  # 2017 and 2018: nothing retrained
    pd.testing.assert_frame_equal(first.residual_weights, again.residual_weights)
