"""Settings for the deep-learning strategy. Architecture defaults follow the paper; they are
fixed in advance rather than tuned, so the model counts as one trial, not a search."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeepConfig:
    # Residual factor model
    n_factors: int = 8            # PCA factors removed (the Marchenko-Pastur count found in-sample)
    pca_window: int = 252
    beta_window: int = 60
    # Signal extractor
    lookback: int = 30            # days of cumulative residuals the network sees
    filters: int = 8
    heads: int = 4
    dropout: float = 0.25
    # Training
    cost_bps: float = 5.0         # charged in the training objective, same as in evaluation
    execution_lag: int = 0        # the network is trained for the lag it will be deployed with
    block_days: int = 126         # consecutive days per Sharpe estimate in one training step
    max_epochs: int = 60
    patience: int = 10            # early stopping on validation net Sharpe
    learning_rate: float = 1e-3
    val_days: int = 252           # last part of each training window held out for early stopping
    seeds: tuple[int, ...] = (0, 1, 2)  # ensemble members (weights are averaged)
    # Walk-forward: retrain at the start of each year, trade that year with the frozen model
    first_trade_year: int = 2014
