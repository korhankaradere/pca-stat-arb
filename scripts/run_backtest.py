"""Run the strategy end to end from a config file.

    python scripts/run_backtest.py configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pca_stat_arb import Config
from pca_stat_arb.backtest import run_backtest
from pca_stat_arb.experiment import load_market_data
from pca_stat_arb.metrics import split_summary, turnover


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="configs/default.yaml")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    returns, volume = load_market_data(cfg)
    print(f"returns: {returns.shape[0]} days x {returns.shape[1]} stocks, "
          f"{returns.index[0].date()} -> {returns.index[-1].date()}")

    res = run_backtest(returns, cfg.strategy, volume=volume, progress=True)

    # In-sample only: the held-out test is revealed via scripts/experiments.py --reveal-test.
    stats = split_summary(res.net_returns, cfg.evaluation.split_date, include_test=False).iloc[0]
    print("\nin-sample (before", cfg.evaluation.split_date + ")")
    print(stats.to_string(float_format=lambda x: f"{x:,.4f}"))

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    res.pnl.to_csv(out / "pnl.csv")
    stats.to_csv(out / "summary.csv")

    fig, ax = plt.subplots(figsize=(10, 4))
    res.pnl[["gross", "net"]].cumsum().plot(ax=ax)
    ax.set_ylabel("cumulative P&L (fraction of capital)")
    ax.set_title("PCA stat-arb (Avellaneda-Lee baseline)")
    fig.tight_layout()
    fig.savefig(out / "cumulative_pnl.png", dpi=150)
    print(f"\nsaved results to {out}/")


if __name__ == "__main__":
    main()
