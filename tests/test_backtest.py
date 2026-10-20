import numpy as np
import pandas as pd

from pca_stat_arb.backtest import run_backtest
from pca_stat_arb.config import StrategyConfig

CFG = StrategyConfig(pca_window=252, n_factors=3, residual_window=60, cost_bps=0.0)


def test_strategy_profits_on_synthetic_mean_reversion(factor_market):
    returns, _ = factor_market
    res = run_backtest(returns, CFG)
    assert (res.states != 0).to_numpy().any()
    assert res.net_returns.mean() > 0


def test_positions_are_approximately_factor_neutral(factor_market):
    returns, _ = factor_market
    res = run_backtest(returns, CFG)
    # Hedged book should have far less market exposure than the unhedged stock leg.
    w = res.weights.iloc[-50:]
    gross = w.abs().sum(axis=1)
    net = w.sum(axis=1).abs()
    active = gross > 0
    assert (net[active] < gross[active]).all()


def test_no_lookahead(factor_market):
    """Changing returns after day t must not change any weight set on or before t."""
    returns, _ = factor_market
    cut = 450
    base = run_backtest(returns, CFG)
    perturbed = returns.copy()
    perturbed.iloc[cut + 1 :] = np.random.default_rng(9).normal(0, 0.05, perturbed.iloc[cut + 1 :].shape)
    res = run_backtest(perturbed, CFG)
    pd.testing.assert_frame_equal(base.weights.iloc[: cut + 1], res.weights.iloc[: cut + 1])


def test_costs_reduce_pnl(factor_market):
    returns, _ = factor_market
    free = run_backtest(returns, CFG)
    costly = run_backtest(returns, StrategyConfig(**{**CFG.__dict__, "cost_bps": 10.0}))
    assert costly.net_returns.sum() < free.net_returns.sum()
    assert np.allclose(free.pnl["gross"], costly.pnl["gross"])


def test_stocks_with_missing_history_are_not_traded(factor_market):
    returns, _ = factor_market
    returns = returns.copy()
    returns.iloc[300:320, 0] = np.nan  # S0 has a gap; it's untradable for pca_window days after
    res = run_backtest(returns, CFG)
    assert (res.weights.iloc[300 : 319 + 252, 0] == 0).all()


def _volume_like(returns, seed=3):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.lognormal(13, 0.4, returns.shape), index=returns.index, columns=returns.columns)


def test_frozen_hedges_cut_turnover(factor_market):
    returns, _ = factor_market
    daily = run_backtest(returns, CFG)
    frozen = run_backtest(returns, StrategyConfig(**{**CFG.__dict__, "hedge_mode": "frozen"}))
    to = lambda r: r.weights.diff().abs().sum(axis=1).mean()
    assert to(frozen) < to(daily)
    # Same signals: hedging style must not change which positions are open.
    pd.testing.assert_frame_equal(daily.states, frozen.states)


def test_frozen_hedge_is_released_on_exit(factor_market):
    returns, _ = factor_market
    res = run_backtest(returns, StrategyConfig(**{**CFG.__dict__, "hedge_mode": "frozen"}))
    flat_days = (res.states == 0).all(axis=1) & (res.weights.index >= res.pnl.index[0])
    assert np.allclose(res.weights[flat_days], 0)


def test_no_lookahead_with_frozen_hedges_and_volume(factor_market):
    returns, _ = factor_market
    volume = _volume_like(returns)
    cfg = StrategyConfig(**{**CFG.__dict__, "hedge_mode": "frozen", "volume_adjust": True})
    cut = 450
    base = run_backtest(returns, cfg, volume=volume)
    rng = np.random.default_rng(9)
    r2, v2 = returns.copy(), volume.copy()
    r2.iloc[cut + 1 :] = rng.normal(0, 0.05, r2.iloc[cut + 1 :].shape)
    v2.iloc[cut + 1 :] = rng.lognormal(15, 1.0, v2.iloc[cut + 1 :].shape)
    res = run_backtest(r2, cfg, volume=v2)
    pd.testing.assert_frame_equal(base.weights.iloc[: cut + 1], res.weights.iloc[: cut + 1])


def test_volume_adjust_requires_volume(factor_market):
    import pytest

    returns, _ = factor_market
    with pytest.raises(ValueError):
        run_backtest(returns, StrategyConfig(**{**CFG.__dict__, "volume_adjust": True}))


def test_rebalance_band_cuts_turnover_and_keeps_lookahead_safety(factor_market):
    returns, _ = factor_market
    cfg = StrategyConfig(**{**CFG.__dict__, "rebalance_band": 0.002})
    plain, banded = run_backtest(returns, CFG), run_backtest(returns, cfg)
    to = lambda r: r.weights.diff().abs().sum(axis=1).mean()
    assert to(banded) < to(plain)
    # Every executed change is bigger than the band (excluding the diff's first row and the
    # final-day unwind, which happens outside the trading loop).
    changes = banded.weights.diff().abs().to_numpy()[1:-1]
    assert ((changes == 0) | (changes > 0.002)).all()

    cut = 450
    r2 = returns.copy()
    r2.iloc[cut + 1 :] = np.random.default_rng(9).normal(0, 0.05, r2.iloc[cut + 1 :].shape)
    pd.testing.assert_frame_equal(
        banded.weights.iloc[: cut + 1], run_backtest(r2, cfg).weights.iloc[: cut + 1]
    )


ETF_CFG = StrategyConfig(**{**CFG.__dict__, "factor_model": "etf"})


def test_etf_model_profits_and_hedges_factor_exposure(factor_market_with_factors):
    from pca_stat_arb.residuals import fit_residuals

    returns, loadings, etfs = factor_market_with_factors
    res = run_backtest(returns, ETF_CFG, etf_returns=etfs)
    assert res.net_returns.mean() > 0

    # Exact: on any day, the ETF hedge is -B^T w for the betas fitted on that day's window.
    t = 500
    L = ETF_CFG.residual_window
    fit = fit_residuals(returns.to_numpy()[t - L + 1 : t + 1], etfs.to_numpy()[t - L + 1 : t + 1])
    w = res.weights.to_numpy()[t]
    np.testing.assert_allclose(res.etf_weights.to_numpy()[t], -(fit.betas.T @ w), atol=1e-12)

    # Statistical: the dominant market factor's exposure is mostly hedged away. (Weak factors
    # are hedged less well: 60 days cannot pin their betas down. That is estimation noise.)
    W, H = res.weights.to_numpy(), res.etf_weights.to_numpy()
    active = np.abs(W).sum(axis=1) > 0
    hedged = np.median(np.abs((W @ loadings + H)[active, 0]))
    unhedged = np.median(np.abs((W @ loadings)[active, 0]))
    assert hedged < 0.5 * unhedged


def test_etf_model_charges_each_leg_its_own_cost(factor_market_with_factors):
    returns, _, etfs = factor_market_with_factors
    cfg = StrategyConfig(**{**ETF_CFG.__dict__, "cost_bps": 5.0, "etf_cost_bps": 1.0})
    pnl = run_backtest(returns, cfg, etf_returns=etfs).pnl
    assert (pnl["etf_turnover"] > 0).any()
    expected = pnl["stock_turnover"] * 5e-4 + pnl["etf_turnover"] * 1e-4
    np.testing.assert_allclose(pnl["cost"], expected)


def test_no_lookahead_etf_model(factor_market_with_factors):
    returns, _, etfs = factor_market_with_factors
    cut = 450
    base = run_backtest(returns, ETF_CFG, etf_returns=etfs)
    rng = np.random.default_rng(9)
    r2, e2 = returns.copy(), etfs.copy()
    r2.iloc[cut + 1 :] = rng.normal(0, 0.05, r2.iloc[cut + 1 :].shape)
    e2.iloc[cut + 1 :] = rng.normal(0, 0.05, e2.iloc[cut + 1 :].shape)
    res = run_backtest(r2, ETF_CFG, etf_returns=e2)
    pd.testing.assert_frame_equal(base.weights.iloc[: cut + 1], res.weights.iloc[: cut + 1])
    pd.testing.assert_frame_equal(base.etf_weights.iloc[: cut + 1], res.etf_weights.iloc[: cut + 1])


def test_execution_lag_delays_holdings_by_one_day(factor_market):
    returns, _ = factor_market
    now = run_backtest(returns, CFG)
    late = run_backtest(returns, StrategyConfig(**{**CFG.__dict__, "execution_lag": 1}))
    pd.testing.assert_frame_equal(now.states, late.states)  # same signals...
    np.testing.assert_allclose(late.weights.to_numpy()[1:], now.weights.to_numpy()[:-1])  # ...traded a day later
    assert (late.weights.iloc[0] == 0).all()
