"""Typed configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DataConfig:
    source: str = "yfinance"
    universe: str = "sp500_current"
    start: str = "2010-01-01"
    end: str = "2025-12-31"


@dataclass(frozen=True)
class StrategyConfig:
    pca_window: int = 252
    n_factors: int = 15
    residual_window: int = 60
    kappa_min: float = 252 / 30
    s_open: float = 1.25
    s_close_long: float = 0.50
    s_close_short: float = 0.75
    center_mean: bool = True
    position_size: float = 0.01
    cost_bps: float = 5.0
    # "daily": re-hedge every open position to today's factor loadings (AL baseline).
    # "frozen": fix each position's hedge at entry and hold it until exit (much less turnover).
    hedge_mode: str = "daily"
    # AL's "trading time" signal: scale returns by avg_volume / volume before the residual fit.
    volume_adjust: bool = False
    volume_window: int = 10
    # No-trade band: leave a stock's weight unchanged if the target is within this distance
    # (in units of capital) of the current holding. 0 disables it.
    rebalance_band: float = 0.0

    def __post_init__(self) -> None:
        if self.hedge_mode not in ("daily", "frozen"):
            raise ValueError(f"unknown hedge_mode: {self.hedge_mode!r}")
        if self.residual_window > self.pca_window:
            raise ValueError("residual_window must not exceed pca_window")
        if self.n_factors >= self.residual_window - 2:
            raise ValueError("too many factors for the residual regression window")


@dataclass(frozen=True)
class EvaluationConfig:
    # Everything before split_date is in-sample (use it for design and tuning); everything
    # from split_date on is the held-out test. Only look at the test once a variant is final.
    split_date: str = "2019-01-01"


@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls(
            data=DataConfig(**raw.get("data", {})),
            strategy=StrategyConfig(**raw.get("strategy", {})),
            evaluation=EvaluationConfig(**raw.get("evaluation", {})),
        )
