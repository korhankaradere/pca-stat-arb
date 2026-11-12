"""Figures, tables and numbers for the mid-project report (report/midproject/).

    python scripts/make_midproject_assets.py

Reads only saved outputs: results/*/*_pnl.csv, the deep-model checkpoints and the cached price
data. Nothing is re-run and every statistic is in-sample (before the evaluation split date).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib
import matplotlib.dates

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from make_report_assets import (  # shared styling and variant loading
    AQUA, BLUE, INK, INK2, ORANGE, WIDTH, load_all_variants, load_variant_overrides, neg,
)
from pca_stat_arb import Config
from pca_stat_arb.deep.config import DeepConfig
from pca_stat_arb.experiment import load_market_data
from pca_stat_arb.metrics import summary

OUT = ROOT / "report" / "midproject"
FIG, GEN = OUT / "figures", OUT / "generated"
LAG = ROOT / "results" / "etf_and_lag"


def read_pnl(name: str, split) -> pd.DataFrame:
    p = pd.read_csv(LAG / f"{name}_pnl.csv", index_col=0, parse_dates=True)
    return p[p.index < split]


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print(f"  figures/{name}.pdf")


def fig_lag(split):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.4), sharey=True)
    for ax, name, title in [(axes[0], "baseline", "AL baseline ($m=15$)"),
                            (axes[1], "f8_band20", "$m=8$, band 0.002")]:
        for lag, color, label in [(0, BLUE, "same close (lag 0)"), (1, ORANGE, "next close (lag 1)")]:
            p = read_pnl(name if lag == 0 else f"{name}_lag1", split)
            ax.plot(p.index, p["gross"].cumsum() * 100, color=color, label=label)
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(2))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
        ax.set_title(title + ", gross", loc="left", color=INK)
    axes[0].set_ylabel("cumulative gross P&L (% of capital)")
    axes[0].legend(loc="upper left")
    fig.tight_layout(w_pad=1.0)
    save(fig, "lag_gross")


def cost_curve(p: pd.DataFrame, c_stock_bps: np.ndarray, c_etf_bps: float) -> np.ndarray:
    out = []
    for c in c_stock_bps:
        net = p["gross"] - p["stock_turnover"] * c / 1e4 - p["etf_turnover"] * c_etf_bps / 1e4
        out.append(net.mean() / net.std() * np.sqrt(252))
    return np.array(out)


def fig_cost_sensitivity(split, etf_cost_bps):
    c = np.linspace(0, 8, 33)
    series = [("baseline", "AL baseline", BLUE), ("f8_band20", "$m=8$ + band", ORANGE), ("etf", "sector ETFs", AQUA)]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.4), sharey=True)
    for ax, lag in zip(axes, [0, 1]):
        for name, label, color in series:
            p = read_pnl(name if lag == 0 else f"{name}_lag1", split)
            ax.plot(c, cost_curve(p, c, etf_cost_bps), color=color, label=label)
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.axvline(5, color=INK2, linestyle="--", linewidth=0.8)
        ax.set_xlabel("one-way stock cost (bps)")
        ax.set_title(f"execution lag {lag}", loc="left", color=INK)
    axes[0].set_ylabel("net Sharpe ratio")
    axes[0].legend(loc="lower left")
    fig.tight_layout(w_pad=1.0)
    save(fig, "cost_sensitivity")


def table(rows: list[tuple[str, str]], t: pd.DataFrame, betas: dict, path: Path):
    lines = [r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
             r"Variant & Gross SR & Net SR & Net ret. & Max DD & TO$_{\text{stock}}$ & TO$_{\text{ETF}}$ & BE (bps)\\",
             r"\midrule"]
    for key, label in rows:
        if key is None:
            lines.append(r"\midrule")
            continue
        r = t.loc[key]
        lines.append(f"{label} & {r.gross_sharpe:.2f} & {r.net_sharpe:.2f} & {r.net_ann_ret*100:.1f}\\% & "
                     f"{r.max_dd*100:.1f}\\% & {r.stock_turnover:.2f} & {r.etf_turnover:.2f} & {r.breakeven_bps:.1f}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text(neg("\n".join(lines)) + "\n")
    print(f"  generated/{path.name}")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    GEN.mkdir(parents=True, exist_ok=True)
    cfg = Config.from_yaml(ROOT / "configs" / "default.yaml")
    split = pd.Timestamp(cfg.evaluation.split_date)
    dcfg = DeepConfig()

    fig_lag(split)
    fig_cost_sensitivity(split, cfg.strategy.etf_cost_bps)

    # Execution-lag / ETF table straight from the experiment summary.
    s = pd.read_csv(LAG / "summary_in_sample.csv").set_index("variant")
    table([("baseline", "AL baseline ($m=15$)"), ("baseline_lag1", "\\quad lag 1"), (None, ""),
           ("f8_band20", "$m=8$, $\\delta=0.002$"), ("f8_band20_lag1", "\\quad lag 1"), (None, ""),
           ("f3_band05", "$m=3$, $\\delta=0.0005$"), ("f3_band05_lag1", "\\quad lag 1"), (None, ""),
           ("etf", "Sector ETFs (1 bp hedges)"), ("etf_lag1", "\\quad lag 1"),
           ("etf_cost2", "\\quad ETF cost 2 bps")], s, {}, GEN / "lag_table.tex")

    # Market beta of daily P&L against the equal-weighted universe (proxy for the market).
    returns, _ = load_market_data(cfg)
    mkt = returns.mean(axis=1)
    betas = {}
    for name in ["baseline", "f8_band20", "f3_band05", "etf"]:
        p = read_pnl(name, split)
        m = mkt.reindex(p.index)
        betas[name] = (np.cov(p["net"], m)[0, 1] / m.var(), np.corrcoef(p["net"], m)[0, 1])

    # All variants (for counts) and per-variant stats.
    overrides = load_variant_overrides()
    allv = load_all_variants(split, cfg.strategy, overrides)
    key = {n: tuple(sorted(overrides.get(n, {}).items())) for n in allv.index}
    allv = allv.loc[~pd.Series(key).duplicated().values]
    n_variants = len(allv)

    # Deep-model tables and numbers are shared with the technical report (make_report_assets.py).

    from pca_stat_arb.deep.model import CNNTransformer

    n_params = sum(p.numel() for p in CNNTransformer(dcfg.lookback, dcfg.filters, dcfg.heads, dcfg.dropout).parameters())
    n_tests = len(json.loads((ROOT / ".pytest_cache" / "v" / "cache" / "nodeids").read_text()))

    b0 = read_pnl("baseline", split)
    vals = {
        "MPNVariants": n_variants,
        "MPNTests": n_tests,
        "MPDeepParams": f"{n_params:,}".replace(",", "{,}"),
        "MPBetaBase": f"{betas['baseline'][0]:.3f}", "MPCorrBase": f"{betas['baseline'][1]:.2f}",
        "MPBetaBest": f"{betas['f8_band20'][0]:.3f}", "MPCorrBest": f"{betas['f8_band20'][1]:.2f}",
        "MPBetaETF": f"{betas['etf'][0]:.3f}",
        "MPBaseCostAnn": f"{b0['cost'].mean() * 252 * 100:.1f}",
        "MPBaseGrossAnn": f"{b0['gross'].mean() * 252 * 100:.1f}",
    }
    for name, tag in [("baseline", "Base"), ("baseline_lag1", "BaseLag"), ("f8_band20", "Best"),
                      ("f8_band20_lag1", "BestLag"), ("etf", "ETF"), ("etf_lag1", "ETFLag")]:
        vals[f"MP{tag}GrossSR"] = f"{s.loc[name, 'gross_sharpe']:.2f}"
        vals[f"MP{tag}NetSR"] = f"{s.loc[name, 'net_sharpe']:.2f}"
        vals[f"MP{tag}BE"] = f"{s.loc[name, 'breakeven_bps']:.1f}"
    vals["MPETFStockTO"] = f"{s.loc['etf', 'stock_turnover']:.2f}"
    vals["MPETFEtfTO"] = f"{s.loc['etf', 'etf_turnover']:.2f}"
    vals["MPLagLossBase"] = f"{(1 - s.loc['baseline_lag1', 'gross_sharpe'] / s.loc['baseline', 'gross_sharpe']) * 100:.0f}"
    vals["MPLagLossBest"] = f"{(1 - s.loc['f8_band20_lag1', 'gross_sharpe'] / s.loc['f8_band20', 'gross_sharpe']) * 100:.0f}"
    vals["MPLagLossETF"] = f"{(1 - s.loc['etf_lag1', 'gross_sharpe'] / s.loc['etf', 'gross_sharpe']) * 100:.0f}"
    lines = [f"\\newcommand{{\\{k}}}{{{neg(str(v))}}}" for k, v in vals.items()]
    (GEN / "numbers.tex").write_text("% Generated by scripts/make_midproject_assets.py; do not edit.\n" + "\n".join(lines) + "\n")
    print("  generated/numbers.tex")


if __name__ == "__main__":
    main()
