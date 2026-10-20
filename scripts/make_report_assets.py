"""Generate the technical report's figures and tables from the code's own output.

    python scripts/make_report_assets.py

Writes report/figures/*.pdf and report/generated/*.tex. Data is truncated at the
evaluation split date first, so nothing from the held-out test period can reach the report.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pca_stat_arb import Config
from pca_stat_arb.backtest import run_backtest
from pca_stat_arb.data.universe import UNIVERSE_DIR
from pca_stat_arb.experiment import load_market_data
from pca_stat_arb.metrics import summary
from pca_stat_arb.pca import fit_eigenportfolios

FIG = ROOT / "report" / "figures"
GEN = ROOT / "report" / "generated"

# Reference palette (dataviz skill), light mode. Text never wears series colours.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 1.4, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
WIDTH = 6.0  # inches: full text width of an A4 article with ~2.5cm margins


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print(f"  figures/{name}.pdf")


# ---------------------------------------------------------------- figures

def fig_spectrum(window: pd.DataFrame, n_factors: int):
    R = window.dropna(axis=1).to_numpy()
    T, N = R.shape
    eig = fit_eigenportfolios(R, n_factors)
    lam = eig.eigenvalues
    edge = (1 + np.sqrt(N / T)) ** 2
    k = 40
    fig, ax = plt.subplots(figsize=(WIDTH, 2.4))
    colors = [BLUE if i < n_factors else "#b9b8b2" for i in range(k)]
    ax.bar(np.arange(1, k + 1), lam[:k], color=colors, width=0.75)
    ax.axhline(edge, color=INK2, linestyle="--", linewidth=1)
    ax.text(k + 0.4, edge, f"Marchenko–Pastur edge {edge:.2f}", va="bottom", ha="right", color=INK2, fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("eigenvalue rank")
    ax.set_ylabel("eigenvalue (log scale)")
    ax.set_xlim(0.3, k + 0.7)
    above = int((lam > edge).sum())
    ax.set_title(f"Correlation spectrum, {N} stocks, {T} days ending {window.index[-1].date()}",
                 loc="left", color=INK)
    save(fig, "spectrum")
    return {"N": N, "T": T, "edge": edge, "above": above, "lam1_share": lam[0] / lam.sum(),
            "kept_share": lam[:n_factors].sum() / lam.sum()}


def fig_sector_loadings(window: pd.DataFrame, sectors: pd.Series):
    R = window.dropna(axis=1)
    eig = fit_eigenportfolios(R.to_numpy(), 3)
    v = pd.DataFrame(eig.eigenvectors, index=R.columns, columns=["PC1", "PC2", "PC3"])
    by_sector = v.join(sectors).groupby("sector").mean() * np.sqrt(len(v))  # scale: 1 = equal weight
    order = by_sector["PC2"].sort_values().index
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.8), sharey=True)
    for ax, pc, color in zip(axes, ["PC1", "PC2", "PC3"], [BLUE, ORANGE, AQUA]):
        vals = by_sector.loc[order, pc]
        ax.barh(np.arange(len(vals)), vals, color=color, height=0.7)
        ax.axvline(0, color=INK2, linewidth=0.8)
        ax.set_title(pc, loc="left", color=INK)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(np.arange(len(order)), order)
    axes[1].set_xlabel("mean eigenvector loading by GICS sector (1 = equal weight)")
    save(fig, "sector_loadings")


def fig_explained(explained: pd.Series):
    s = explained.dropna()
    fig, ax = plt.subplots(figsize=(WIDTH, 2.0))
    ax.plot(s.index, s.values * 100, color=BLUE)
    ax.set_ylabel("% variance, 15 factors")
    ax.set_title("Share of total variance explained by the kept factors (trailing 252-day window)", loc="left", color=INK)
    save(fig, "explained_variance")
    return {"ev_mean": s.mean(), "ev_min": s.min(), "ev_max": s.max(), "ev_argmax": s.idxmax()}


def fig_sscore(res, ticker: str, year: int, cfg):
    s = res.s_scores[ticker]
    st = res.states[ticker]
    sel = (s.index.year == year)
    s, st = s[sel], st[sel]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.3))
    for state, color in [(1, BLUE), (-1, ORANGE)]:
        ax.fill_between(st.index, -3.2, 3.2, where=(st == state).values, color=color, alpha=0.22,
                        step="post", linewidth=0, label="long" if state == 1 else "short")
    ax.plot(s.index, s.values, color=INK, linewidth=1.1, label="s-score")
    for y, lab in [(cfg.s_open, "open short"), (-cfg.s_open, "open long"),
                   (cfg.s_close_short, "close short"), (-cfg.s_close_long, "close long")]:
        ax.axhline(y, color=INK2, linestyle=":" if "close" in lab else "--", linewidth=0.8)
        ax.text(1.01, y, lab, va="center", fontsize=7, color=INK2, transform=ax.get_yaxis_transform())
    ax.set_ylim(-3.2, 3.2)
    ax.set_ylabel("s-score")
    ax.set_title(f"{ticker}, {year}: s-score and position (shaded)", loc="left", color=INK)
    ax.legend(loc="lower left", ncol=3)
    save(fig, "sscore_example")


def fig_cum_pnl(pnl: pd.DataFrame):
    cum = pnl[["gross", "net"]].cumsum() * 100
    fig, ax = plt.subplots(figsize=(WIDTH, 2.4))
    ax.plot(cum.index, cum["gross"], color=BLUE)
    ax.plot(cum.index, cum["net"], color=ORANGE)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.text(cum.index[-1], cum["gross"].iloc[-1], "  gross", va="center", color=INK)
    ax.text(cum.index[-1], cum["net"].iloc[-1], "  net of 5 bps", va="center", color=INK)
    ax.set_ylabel("cumulative P&L (% of capital)")
    ax.set_title("AL baseline, in-sample", loc="left", color=INK)
    ax.legend(["gross", "net of 5 bps"], loc="upper left")
    save(fig, "cum_pnl_baseline")


def fig_factor_tradeoff(table: pd.DataFrame):
    rows = {1: "f1", 2: "f2", 3: "f3", 5: "f5", 8: "f8", 15: "baseline", 30: "factors30"}
    t = table.loc[list(rows.values())]
    m = np.array(list(rows.keys()))
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.1))
    for ax, col, label, color in [
        (axes[0], "gross_sharpe", "gross Sharpe", BLUE),
        (axes[1], "turnover", "daily turnover (× capital)", ORANGE),
        (axes[2], "breakeven_bps", "break-even cost (bps)", AQUA),
    ]:
        ax.plot(m, t[col].values, color=color, marker="o", markersize=4)
        ax.set_xscale("log")
        ax.set_xticks(m, [str(x) for x in m])
        ax.minorticks_off()
        ax.set_title(label, loc="left", color=INK)
        ax.set_xlabel("number of factors m")
    axes[2].axhline(5, color=INK2, linestyle="--", linewidth=0.8)
    axes[2].text(m[-1], 5, "assumed cost", va="bottom", ha="right", fontsize=7, color=INK2)
    fig.tight_layout(w_pad=1.5)
    save(fig, "factor_tradeoff")


# ---------------------------------------------------------------- tables

def load_all_variants(split: pd.Timestamp, cost_bps: float) -> pd.DataFrame:
    """Recompute every variant's in-sample stats from its saved daily P&L, so all rows are
    computed the same way. Turnover is recovered from the cost column (cost = turnover * bps)."""
    rows = {}
    for path in sorted((ROOT / "results").glob("*/*_pnl.csv")):
        name = path.stem.removesuffix("_pnl")
        if name in rows:
            continue
        pnl = pd.read_csv(path, index_col=0, parse_dates=True)
        pnl = pnl[pnl.index < split]
        net, gross = summary(pnl["net"]), summary(pnl["gross"])
        turnover = (pnl["cost"] / (cost_bps / 1e4)).mean()
        rows[name] = {
            "gross_sharpe": gross["sharpe"], "net_sharpe": net["sharpe"],
            "net_ann_ret": net["ann_return"], "ann_vol": net["ann_vol"], "max_dd": net["max_drawdown"],
            "turnover": turnover, "breakeven_bps": gross["ann_return"] / (turnover * 252) * 1e4,
        }
    return pd.DataFrame(rows).T


def tex_escape(s: str) -> str:
    return s.replace("_", r"\_")


def neg(text: str) -> str:
    """Typeset a leading hyphen as a proper minus sign."""
    return re.sub(r"(?<![\w.])-(?=\d)", r"\\ensuremath{-}", text)


def write_variant_table(t: pd.DataFrame, overrides: dict):
    lines = [r"\begin{longtable}{@{}l>{\raggedright\arraybackslash\footnotesize}p{4.4cm}rrrrr@{}}", r"\toprule",
             r"Variant & Overrides & Gross SR & Net SR & Vol & Turnover & BE (bps)\\", r"\midrule", r"\endhead"]
    for name, row in t.sort_values("net_sharpe", ascending=False).iterrows():
        ov = overrides.get(name, {})
        ov_s = ", ".join(f"{k}={v}" for k, v in ov.items()) or "--"
        lines.append(
            f"\\texttt{{{tex_escape(name)}}} & \\texttt{{{tex_escape(ov_s)}}} & "
            f"{row.gross_sharpe:.2f} & {row.net_sharpe:.2f} & {row.ann_vol*100:.1f}\\% & "
            f"{row.turnover:.2f} & {row.breakeven_bps:.1f}\\\\"
        )
    lines += [r"\bottomrule", r"\end{longtable}"]
    (GEN / "variants_table.tex").write_text(neg("\n".join(lines)) + "\n")
    print("  generated/variants_table.tex")


def write_main_table(t: pd.DataFrame):
    pick = [("baseline", "AL baseline ($m=15$, daily hedge)"),
            ("frozen_hedge", "Frozen hedge"),
            ("volume_signal", "Volume-adjusted signal"),
            ("open200", "Wider entry ($\\bar s_{\\text{open}}=2.0$)"),
            ("f3", "$m=3$"),
            ("f3_band05", "$m=3$, band $0.0005$"),
            ("f8_band20", "$m=8$, band $0.002$")]
    lines = [r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r"Variant & Gross SR & Net SR & Net return & Max DD & Turnover & BE (bps)\\", r"\midrule"]
    for key, label in pick:
        r = t.loc[key]
        lines.append(f"{label} & {r.gross_sharpe:.2f} & {r.net_sharpe:.2f} & {r.net_ann_ret*100:.1f}\\% & "
                     f"{r.max_dd*100:.1f}\\% & {r.turnover:.2f} & {r.breakeven_bps:.1f}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "main_table.tex").write_text(neg("\n".join(lines)) + "\n")
    print("  generated/main_table.tex")


def write_macros(values: dict):
    lines = [f"\\newcommand{{\\{k}}}{{{neg(str(v))}}}" for k, v in values.items()]
    (GEN / "numbers.tex").write_text("% Generated by scripts/make_report_assets.py; do not edit.\n" + "\n".join(lines) + "\n")
    print("  generated/numbers.tex")


# ---------------------------------------------------------------- main

def main():
    import yaml

    FIG.mkdir(parents=True, exist_ok=True)
    GEN.mkdir(parents=True, exist_ok=True)
    cfg = Config.from_yaml(ROOT / "configs" / "default.yaml")
    split = pd.Timestamp(cfg.evaluation.split_date)

    returns, volume = load_market_data(cfg)
    n_total = returns.shape[1]
    full_start, full_end = returns.index[0], returns.index[-1]
    returns = returns[returns.index < split]  # hard wall: the report only ever sees in-sample data
    volume = volume[volume.index < split] if volume is not None else None
    sectors = pd.read_csv(UNIVERSE_DIR / "sp500_current.csv").set_index("ticker")["sector"]

    print("running in-sample baseline ...")
    res = run_backtest(returns, cfg.strategy, volume=volume)
    base = summary(res.net_returns)
    gross = summary(res.pnl["gross"])

    window = returns.iloc[-cfg.strategy.pca_window:]
    spec = fig_spectrum(window, cfg.strategy.n_factors)
    fig_sector_loadings(window, sectors)
    ev = fig_explained(res.explained_variance)
    fig_sscore(res, "JPM", 2017, cfg.strategy)
    fig_cum_pnl(res.pnl)

    table = load_all_variants(split, cfg.strategy.cost_bps)
    overrides = {}
    for f in ["experiments", "sweep_thresholds", "sweep_factors", "sweep_band"]:
        overrides.update(yaml.safe_load((ROOT / "configs" / f"{f}.yaml").read_text())["variants"])
    # The same configuration run under two names counts as one trial.
    key = {name: tuple(sorted(overrides.get(name, {}).items())) for name in table.index}
    table = table.loc[~pd.Series(key).duplicated().values]
    fig_factor_tradeoff(table)
    write_variant_table(table, overrides)
    write_main_table(table)

    stock_leg = (res.states.diff().abs() * cfg.strategy.position_size).sum(axis=1).loc[res.pnl.index].mean()
    total_to = res.weights.diff().abs().sum(axis=1).loc[res.pnl.index].mean()
    open_names = (res.states != 0).sum(axis=1).loc[res.pnl.index]
    write_macros({
        "NStocksTotal": n_total,
        "DataStart": full_start.strftime("%-d %B %Y"),
        "DataEnd": full_end.strftime("%-d %B %Y"),
        "SplitDate": split.strftime("%-d %B %Y"),
        "InSampleStart": res.pnl.index[0].strftime("%-d %B %Y"),
        "InSampleDays": len(res.pnl),
        "SpecN": spec["N"], "SpecT": spec["T"], "MPEdge": f"{spec['edge']:.2f}",
        "NAboveEdge": spec["above"], "LamOneShare": f"{spec['lam1_share']*100:.0f}",
        "KeptShare": f"{spec['kept_share']*100:.0f}",
        "EVMean": f"{ev['ev_mean']*100:.0f}", "EVMin": f"{ev['ev_min']*100:.0f}",
        "EVMax": f"{ev['ev_max']*100:.0f}", "EVMaxDate": ev["ev_argmax"].strftime("%B %Y"),
        "BaseGrossSR": f"{gross['sharpe']:.2f}", "BaseNetSR": f"{base['sharpe']:.2f}",
        "BaseGrossRet": f"{gross['ann_return']*100:.1f}", "BaseNetRet": f"{base['ann_return']*100:.1f}",
        "BaseVol": f"{base['ann_vol']*100:.1f}", "BaseMaxDD": f"{base['max_drawdown']*100:.1f}",
        "BaseTurnover": f"{total_to:.2f}", "BaseStockLegTurnover": f"{stock_leg:.2f}",
        "BaseHedgeTurnover": f"{total_to - stock_leg:.2f}",
        "BaseGrossLev": f"{res.weights.abs().sum(axis=1).loc[res.pnl.index].mean():.2f}",
        "BaseOpenMean": f"{open_names.mean():.0f}",
        "BaseHolding": f"{2 * open_names.mean() * cfg.strategy.position_size / stock_leg:.1f}",
        "BaseBreakeven": f"{gross['ann_return'] / (total_to * 252) * 1e4:.1f}",
        "NVariants": len(table),
    })


if __name__ == "__main__":
    main()
