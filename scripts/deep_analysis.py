"""Figures comparing the deep-learning strategy with the PCA strategies (in-sample walk-forward).

    python scripts/deep_analysis.py

Reads results/deep/*_pnl.csv, results/etf_and_lag/*_pnl.csv and the saved deep models, and
writes results/deep/figures/*.png plus results/deep/comparison.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from pca_stat_arb.deep.config import DeepConfig
from pca_stat_arb.deep.model import CNNTransformer

OUT = ROOT / "results" / "deep" / "figures"
START, END = "2014-01-01", "2019-01-01"   # the years the deep model trades in-sample

# Reference palette (dataviz skill), light mode. Fixed slot per strategy, never cycled.
COLORS = {"Deep (CNN+Transformer)": "#2a78d6", "PCA original": "#eb6834", "PCA 8 factors + band": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8, "legend.frameon": False,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 1.6,
    "savefig.dpi": 160, "savefig.bbox": "tight",
})

STRATEGIES = {  # label -> {lag: pnl csv}
    "Deep (CNN+Transformer)": {0: "deep/deep_lag0_pnl.csv", 1: "deep/deep_lag1_pnl.csv"},
    "PCA original": {0: "etf_and_lag/baseline_pnl.csv", 1: "etf_and_lag/baseline_lag1_pnl.csv"},
    "PCA 8 factors + band": {0: "etf_and_lag/f8_band20_pnl.csv", 1: "etf_and_lag/f8_band20_lag1_pnl.csv"},
}


def load(rel: str) -> pd.DataFrame | None:
    path = ROOT / "results" / rel
    if not path.exists():
        return None
    p = pd.read_csv(path, index_col=0, parse_dates=True)
    return p[(p.index >= START) & (p.index < END)]


def sharpe(x: pd.Series) -> float:
    return float(x.mean() / x.std() * np.sqrt(252))


def net_at(p: pd.DataFrame, bps: float) -> pd.Series:
    return p["gross"] - p["stock_turnover"] * bps / 1e4 - p["etf_turnover"] * 1e-4


def comparison(pnls: dict) -> pd.DataFrame:
    rows = []
    for (label, lag), p in pnls.items():
        to = p["stock_turnover"].mean()
        rows.append({
            "strategy": label, "lag": lag, "gross_sharpe": sharpe(p["gross"]), "net_sharpe": sharpe(p["net"]),
            "net_return_pct": p["net"].mean() * 252 * 100, "turnover": to,
            "breakeven_bps": (p["gross"].mean() - p["etf_turnover"].mean() * 1e-4) / to * 1e4,
            "max_dd_pct": float((p["net"].cumsum() - p["net"].cumsum().cummax()).min() * 100),
        })
    return pd.DataFrame(rows)


def fig_cumulative(pnls: dict):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, lag in zip(axes, [0, 1]):
        for label, color in COLORS.items():
            p = pnls.get((label, lag))
            if p is not None:
                ax.plot(p.index, p["net"].cumsum() * 100, color=color, label=label)
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_title("Trade at the signal's close" if lag == 0 else "Trade one day later", loc="left")
    axes[0].set_ylabel("cumulative net P&L, % of capital (5 bps)")
    axes[0].legend(loc="lower left")
    fig.savefig(OUT / "cumulative_net_pnl.png")
    plt.close(fig)


def fig_cost_sensitivity(pnls: dict):
    bps = np.linspace(0, 10, 41)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    for ax, lag in zip(axes, [0, 1]):
        for label, color in COLORS.items():
            p = pnls.get((label, lag))
            if p is not None:
                ax.plot(bps, [sharpe(net_at(p, c)) for c in bps], color=color, label=label)
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.axvline(5, color=INK2, linestyle="--", linewidth=0.8)
        ax.text(5.1, ax.get_ylim()[1] * 0.92, "assumed 5 bps", color=INK2, fontsize=7)
        ax.set_xlabel("one-way stock trading cost (bps)")
        ax.set_title("Same close" if lag == 0 else "One day later", loc="left")
    axes[0].set_ylabel("net Sharpe ratio")
    axes[0].legend(loc="lower left")
    fig.savefig(OUT / "cost_sensitivity.png")
    plt.close(fig)


def fig_yearly(pnls: dict, lag: int):
    years = sorted({y for p in pnls.values() for y in p.index.year})
    labels = [l for l in COLORS if (l, lag) in pnls]
    width = 0.8 / len(labels)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for k, label in enumerate(labels):
        p = pnls[(label, lag)]
        vals = [sharpe(p["net"][p.index.year == y]) for y in years]
        ax.bar(np.arange(len(years)) + (k - (len(labels) - 1) / 2) * width, vals, width * 0.92,
               color=COLORS[label], label=label)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xticks(np.arange(len(years)), years)
    ax.set_ylabel("net Sharpe ratio")
    ax.set_title(f"Net Sharpe by year ({'same close' if lag == 0 else 'one day later'})", loc="left")
    ax.legend(loc="upper left", ncol=3)
    fig.savefig(OUT / f"yearly_sharpe_lag{lag}.png")
    plt.close(fig)


@torch.no_grad()
def fig_learned_response(lag: int):
    """Feed the final-year ensemble stylised residual paths and plot its score."""
    path = ROOT / "results" / "deep" / f"deep_lag{lag}_models.pt"
    if not path.exists():
        return None
    saved = torch.load(path)
    year = max(saved)
    cfg = DeepConfig()
    models = []
    for state in saved[year]:
        m = CNNTransformer(cfg.lookback, cfg.filters, cfg.heads, cfg.dropout)
        m.load_state_dict(state)
        models.append(m.eval())
    z = np.linspace(-3, 3, 61)
    L = cfg.lookback
    shapes = {
        "sudden move today": lambda v: np.r_[np.zeros(L - 1), v],
        "steady drift over 30 days": lambda v: np.linspace(0, v, L),
        "jump 15 days ago, then flat": lambda v: np.r_[np.zeros(L - 15), np.full(15, v)],
    }
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for (name, make), color in zip(shapes.items(), colors):
        paths = torch.tensor(np.stack([make(v) for v in z]), dtype=torch.float32)
        score = torch.stack([m(paths) for m in models]).mean(0).numpy()
        ax.plot(z, score, color=color, label=name)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_xlabel("cumulative residual at the end of the window (standardised)")
    ax.set_ylabel("model score (> 0 = buy)")
    ax.set_title(f"What the {year} model learned (lag {lag})", loc="left")
    ax.legend(loc="best")
    fig.savefig(OUT / f"learned_response_lag{lag}.png")
    plt.close(fig)
    return year


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pnls = {(label, lag): p for label, by_lag in STRATEGIES.items() for lag, rel in by_lag.items()
            if (p := load(rel)) is not None}
    table = comparison(pnls)
    table.to_csv(ROOT / "results" / "deep" / "comparison.csv", index=False)
    print(table.round(2).to_string(index=False))
    fig_cumulative(pnls)
    fig_cost_sensitivity(pnls)
    for lag in (0, 1):
        fig_yearly(pnls, lag)
        fig_learned_response(lag)
    print(f"figures in {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
