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
import matplotlib.dates

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pca_stat_arb import Config
from pca_stat_arb.backtest import run_backtest
from pca_stat_arb.data.base import CACHE_DIR
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
    "font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
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

def load_variant_overrides() -> dict:
    """Every variant defined in any experiment config, by name."""
    import yaml

    overrides = {}
    for path in sorted((ROOT / "configs").glob("*.yaml")):
        overrides.update((yaml.safe_load(path.read_text()) or {}).get("variants", {}))
    return overrides


def load_all_variants(split: pd.Timestamp, strategy, overrides: dict) -> pd.DataFrame:
    """Recompute every variant's in-sample stats from its saved daily P&L, so all rows are
    computed the same way. Break-even is the one-way *stock* cost at which net P&L would be
    zero, with any ETF hedge leg charged at its own cost."""
    rows = {}
    for path in sorted((ROOT / "results").glob("*/*_pnl.csv")):
        name = path.stem.removesuffix("_pnl")
        if path.parent.name == "deep":  # the deep model is reported separately, not as a rule-based variant
            continue
        if name in rows:
            continue
        if name not in overrides:
            raise KeyError(f"{path}: variant {name!r} is not defined in any configs/*.yaml")
        pnl = pd.read_csv(path, index_col=0, parse_dates=True)
        pnl = pnl[pnl.index < split]
        net, gross = summary(pnl["net"]), summary(pnl["gross"])
        if "stock_turnover" in pnl:
            stock_to, etf_to = pnl["stock_turnover"].mean(), pnl["etf_turnover"].mean()
        else:  # runs saved before turnover was recorded: recover it from cost = turnover * bps
            stock_to, etf_to = (pnl["cost"] / (strategy.cost_bps / 1e4)).mean(), 0.0
        etf_cost = overrides[name].get("etf_cost_bps", strategy.etf_cost_bps) / 1e4
        rows[name] = {
            "gross_sharpe": gross["sharpe"], "net_sharpe": net["sharpe"],
            "net_ann_ret": net["ann_return"], "ann_vol": net["ann_vol"], "max_dd": net["max_drawdown"],
            "turnover": stock_to + etf_to,
            "breakeven_bps": (pnl["gross"].mean() - etf_to * etf_cost) / stock_to * 1e4,
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


# ---------------------------------------------------------------- execution lag, ETFs, deep model

DEEP_START, DEEP_END = "2014-01-01", "2019-01-01"  # the years the deep model trades in-sample
LAGDIR = ROOT / "results" / "etf_and_lag"
DEEPDIR = ROOT / "results" / "deep"


def _pnl(path: Path, start=None, end=None) -> pd.DataFrame:
    p = pd.read_csv(path, index_col=0, parse_dates=True)
    if start is not None:
        p = p[p.index >= start]
    return p[p.index < end] if end is not None else p


def _stats(p: pd.DataFrame, etf_cost_bps: float = 1.0) -> dict:
    net, gross = summary(p["net"]), summary(p["gross"])
    to_s, to_e = p["stock_turnover"].mean(), p["etf_turnover"].mean()
    return {"gross_sharpe": gross["sharpe"], "net_sharpe": net["sharpe"], "net_ann_ret": net["ann_return"],
            "ann_vol": net["ann_vol"], "max_dd": net["max_drawdown"], "stock_turnover": to_s, "etf_turnover": to_e,
            "breakeven_bps": (p["gross"].mean() - to_e * etf_cost_bps / 1e4) / to_s * 1e4}


def _row(label: str, r: dict) -> str:
    return (f"{label} & {r['gross_sharpe']:.2f} & {r['net_sharpe']:.2f} & {r['net_ann_ret']*100:.1f}\\% & "
            f"{r['ann_vol']*100:.1f}\\% & {r['max_dd']*100:.1f}\\% & {r['stock_turnover']:.2f} & {r['breakeven_bps']:.1f}\\\\")


def _table(rows, path: Path):
    lines = [r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
             r"Variant & Gross SR & Net SR & Net ret. & Vol & Max DD & TO$_{\text{stock}}$ & BE (bps)\\", r"\midrule"]
    for item in rows:
        lines.append(r"\midrule" if item is None else _row(*item))
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text(neg("\n".join(lines)) + "\n")
    print(f"  generated/{path.name}")


def fig_lag_gross(split):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.4), sharey=True)
    for ax, name, title in [(axes[0], "baseline", "AL baseline ($m=15$)"), (axes[1], "f8_band20", "$m=8$, band 0.002")]:
        for lag, color, label in [(0, BLUE, "same close (lag 0)"), (1, ORANGE, "next close (lag 1)")]:
            p = _pnl(LAGDIR / f"{name if lag == 0 else name + '_lag1'}_pnl.csv", end=split)
            ax.plot(p.index, p["gross"].cumsum() * 100, color=color, label=label)
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(2))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
        ax.set_title(title + ", gross", loc="left", color=INK)
    axes[0].set_ylabel("cumulative gross P&L (% of capital)")
    axes[0].legend(loc="upper left")
    fig.tight_layout(w_pad=1.0)
    save(fig, "lag_gross")


DEEP_SERIES = [("deep", "Deep (CNN+Transformer)", BLUE), ("baseline", "AL baseline", ORANGE), ("f8_band20", "$m=8$ + band", AQUA)]


def _deep_runs():
    """(series key, lag) -> P&L over 2014-2018."""
    runs = {}
    for lag in (0, 1):
        runs[("deep", lag)] = _pnl(DEEPDIR / f"deep_lag{lag}_pnl.csv", DEEP_START, DEEP_END)
        for name in ("baseline", "f8_band20"):
            runs[(name, lag)] = _pnl(LAGDIR / f"{name if lag == 0 else name + '_lag1'}_pnl.csv", DEEP_START, DEEP_END)
    return runs


def fig_deep_cum(runs):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.6), sharey=True)
    for ax, lag in zip(axes, (0, 1)):
        for key, label, color in DEEP_SERIES:
            p = runs[(key, lag)]
            ax.plot(p.index, p["net"].cumsum() * 100, color=color, label=label)
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_title("same close (lag 0)" if lag == 0 else "next close (lag 1)", loc="left", color=INK)
    axes[0].set_ylabel("cumulative net P&L (% of capital)")
    axes[0].legend(loc="lower left")
    fig.tight_layout(w_pad=1.0)
    save(fig, "deep_cum_net")


def fig_deep_costs(runs):
    c = np.linspace(0, 8, 33)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.4), sharey=True)
    for ax, lag in zip(axes, (0, 1)):
        for key, label, color in DEEP_SERIES:
            p = runs[(key, lag)]
            sr = [((p["gross"] - p["stock_turnover"] * x / 1e4) / 1).pipe(lambda n: n.mean() / n.std() * np.sqrt(252)) for x in c]
            ax.plot(c, sr, color=color, label=label)
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.axvline(5, color=INK2, linestyle="--", linewidth=0.8)
        ax.set_xlabel("one-way stock cost (bps)")
        ax.set_title(f"execution lag {lag}", loc="left", color=INK)
    axes[0].set_ylabel("net Sharpe ratio, 2014--2018".replace("--", "–"))
    axes[0].legend(loc="lower left")
    fig.tight_layout(w_pad=1.0)
    save(fig, "deep_costs")


def fig_deep_response(lag: int = 1):
    """Score of the final-year ensemble on stylised residual paths."""
    import torch
    from pca_stat_arb.deep.config import DeepConfig
    from pca_stat_arb.deep.model import CNNTransformer

    saved = torch.load(DEEPDIR / f"deep_lag{lag}_models.pt", weights_only=False)
    year = max(saved)
    cfg = DeepConfig()
    models = []
    for state in saved[year]:
        m = CNNTransformer(cfg.lookback, cfg.filters, cfg.heads, cfg.dropout)
        m.load_state_dict(state)
        models.append(m.eval())
    z = np.linspace(-3, 3, 61)
    L = cfg.lookback
    shapes = [("move on the last day", lambda v: np.r_[np.zeros(L - 1), v], BLUE),
              ("steady drift over 30 days", lambda v: np.linspace(0, v, L), ORANGE),
              ("jump 15 days ago, then flat", lambda v: np.r_[np.zeros(L - 15), np.full(15, v)], AQUA)]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.4))
    with torch.no_grad():
        for name, make, color in shapes:
            paths = torch.tensor(np.stack([make(v) for v in z]), dtype=torch.float32)
            ax.plot(z, torch.stack([m(paths) for m in models]).mean(0).numpy(), color=color, label=name)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_xlabel("cumulative residual at the end of the 30-day window (standardised)")
    ax.set_ylabel("ensemble score")
    ax.set_title(f"Scores of the {year} ensemble (lag {lag}) on stylised residual paths", loc="left", color=INK)
    ax.legend(loc="upper right")
    save(fig, "deep_response")
    return year


def deep_decomposition(returns: pd.DataFrame) -> dict:
    """Split the deep model's gross residual P&L into (a) an equal-weight long position in all
    residuals with the same net exposure and (b) the remaining loser-versus-winner tilt."""
    import glob
    panel = np.load(sorted(glob.glob(str(CACHE_DIR / "residual_panel_*.npz")))[0])
    eps = panel["eps"]
    if eps.shape != returns.shape:
        raise ValueError("residual panel does not match the in-sample returns")
    out = {}
    for lag in (0, 1):
        W = pd.read_parquet(DEEPDIR / f"deep_lag{lag}_residual_weights.parquet").reindex(returns.index).fillna(0).to_numpy()
        h, T = 1 + lag, len(returns)
        tgt = np.zeros_like(eps)
        tgt[: T - h] = np.nan_to_num(eps[h:])
        held = W != 0
        n = held.sum(axis=1).clip(min=1)
        Wbar = np.where(held, (W.sum(axis=1) / n)[:, None], 0.0)
        keep = (np.abs(W).sum(axis=1) > 0) & (returns.index >= DEEP_START)
        a, b = (Wbar * tgt).sum(axis=1)[keep], ((W - Wbar) * tgt).sum(axis=1)[keep]
        tot = a + b
        sr = lambda x: x.mean() / x.std() * np.sqrt(252)
        out[lag] = {"net_exposure": W[keep].sum(axis=1).mean(), "long_share": (W[keep] > 0).sum() / held[keep].sum(),
                    "sr_total": sr(tot), "sr_a": sr(a), "sr_b": sr(b),
                    "share_a": a.mean() / tot.mean(), "share_b": b.mean() / tot.mean(), "corr": np.corrcoef(a, b)[0, 1]}
    return out


def write_deep_folds():
    lines = [r"\begin{tabular}{@{}lllrrrrrr@{}}", r"\toprule",
             r" & & & \multicolumn{3}{c}{Lag 0} & \multicolumn{3}{c}{Lag 1}\\",
             r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
             r"Trade year & Train from & Validate on & Seed 0 & Seed 1 & Seed 2 & Seed 0 & Seed 1 & Seed 2\\", r"\midrule"]
    f0 = pd.read_csv(DEEPDIR / "deep_lag0_folds.csv").set_index("year")
    f1 = pd.read_csv(DEEPDIR / "deep_lag1_folds.csv").set_index("year")
    for y in f0.index:
        vals = [f"{f[f'val_sharpe_seed{s}']:.2f}" for f in (f0.loc[y], f1.loc[y]) for s in (0, 1, 2)]
        lines.append(f"{y} & {f0.loc[y, 'train_start']} & {y - 1} & " + " & ".join(vals) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "deep_folds.tex").write_text(neg("\n".join(lines)) + "\n")
    print("  generated/deep_folds.tex")


def deep_and_lag_assets(split: pd.Timestamp, returns: pd.DataFrame) -> dict:
    """Figures, tables and macros for the execution-lag, ETF and deep-model results."""
    fig_lag_gross(split)
    lag = {n: _stats(_pnl(LAGDIR / f"{n}_pnl.csv", end=split)) for n in
           ["baseline", "baseline_lag1", "f8_band20", "f8_band20_lag1", "f3_band05", "f3_band05_lag1", "etf", "etf_lag1"]}
    lag["etf_cost2"] = _stats(_pnl(LAGDIR / "etf_cost2_pnl.csv", end=split), etf_cost_bps=2.0)
    _table([("AL baseline ($m=15$)", lag["baseline"]), ("\\quad lag 1", lag["baseline_lag1"]), None,
            ("$m=8$, $\\delta=0.002$", lag["f8_band20"]), ("\\quad lag 1", lag["f8_band20_lag1"]), None,
            ("$m=3$, $\\delta=0.0005$", lag["f3_band05"]), ("\\quad lag 1", lag["f3_band05_lag1"]), None,
            ("Sector ETFs", lag["etf"]), ("\\quad lag 1", lag["etf_lag1"]), ("\\quad ETF cost 2~bps", lag["etf_cost2"])],
           GEN / "lag_table.tex")

    runs = _deep_runs()
    d = {k: _stats(v) for k, v in runs.items()}
    _table([("Deep model", d[("deep", 0)]), ("\\quad lag 1", d[("deep", 1)]), None,
            ("AL baseline", d[("baseline", 0)]), ("\\quad lag 1", d[("baseline", 1)]), None,
            ("$m=8$, $\\delta=0.002$\\textsuperscript{*}", d[("f8_band20", 0)]),
            ("\\quad lag 1\\textsuperscript{*}", d[("f8_band20", 1)])], GEN / "deep_table.tex")
    fig_deep_cum(runs)
    fig_deep_costs(runs)
    resp_year = fig_deep_response(1)
    write_deep_folds()
    dec = deep_decomposition(returns)

    mkt = returns.mean(axis=1)
    beta = {}
    for lag_ in (0, 1):
        p = runs[("deep", lag_)]
        m = mkt.reindex(p.index)
        beta[lag_] = np.cov(p["net"], m)[0, 1] / m.var()
    net_by_year = {lag_: runs[("deep", lag_)]["net"].groupby(runs[("deep", lag_)].index.year).sum() * 100 for lag_ in (0, 1)}
    lagloss = lambda a, b: f"{(1 - b['gross_sharpe'] / a['gross_sharpe']) * 100:.0f}"
    m = {
        "LagBaseGrossSR": f"{lag['baseline_lag1']['gross_sharpe']:.2f}", "LagBaseNetSR": f"{lag['baseline_lag1']['net_sharpe']:.2f}",
        "LagBaseBE": f"{lag['baseline_lag1']['breakeven_bps']:.1f}",
        "LagBestGrossSR": f"{lag['f8_band20_lag1']['gross_sharpe']:.2f}", "LagBestNetSR": f"{lag['f8_band20_lag1']['net_sharpe']:.2f}",
        "LagBestBE": f"{lag['f8_band20_lag1']['breakeven_bps']:.1f}",
        "LagLossBase": lagloss(lag["baseline"], lag["baseline_lag1"]), "LagLossBest": lagloss(lag["f8_band20"], lag["f8_band20_lag1"]),
        "LagLossETF": lagloss(lag["etf"], lag["etf_lag1"]),
        "ETFGrossSR": f"{lag['etf']['gross_sharpe']:.2f}", "ETFNetSR": f"{lag['etf']['net_sharpe']:.2f}",
        "ETFLagNetSR": f"{lag['etf_lag1']['net_sharpe']:.2f}", "ETFBE": f"{lag['etf']['breakeven_bps']:.1f}",
        "ETFStockTO": f"{lag['etf']['stock_turnover']:.2f}", "ETFEtfTO": f"{lag['etf']['etf_turnover']:.2f}",
        "DeepGrossSR": f"{d[('deep', 0)]['gross_sharpe']:.2f}", "DeepNetSR": f"{d[('deep', 0)]['net_sharpe']:.2f}",
        "DeepBE": f"{d[('deep', 0)]['breakeven_bps']:.1f}", "DeepTO": f"{d[('deep', 0)]['stock_turnover']:.2f}",
        "DeepLagGrossSR": f"{d[('deep', 1)]['gross_sharpe']:.2f}", "DeepLagNetSR": f"{d[('deep', 1)]['net_sharpe']:.2f}",
        "DeepLagBE": f"{d[('deep', 1)]['breakeven_bps']:.1f}", "DeepLagTO": f"{d[('deep', 1)]['stock_turnover']:.2f}",
        "DeepLagVol": f"{d[('deep', 1)]['ann_vol']*100:.1f}", "DeepLagMaxDD": f"{d[('deep', 1)]['max_dd']*100:.1f}",
        "DeepLagLoss": lagloss(d[("deep", 0)], d[("deep", 1)]),
        "ALDeepYearsGrossSR": f"{d[('baseline', 0)]['gross_sharpe']:.2f}", "ALDeepYearsNetSR": f"{d[('baseline', 0)]['net_sharpe']:.2f}",
        "ALDeepYearsLagGrossSR": f"{d[('baseline', 1)]['gross_sharpe']:.2f}", "ALDeepYearsLagNetSR": f"{d[('baseline', 1)]['net_sharpe']:.2f}",
        "ALDeepYearsLagBE": f"{d[('baseline', 1)]['breakeven_bps']:.1f}",
        "ALDeepYearsLagLoss": lagloss(d[("baseline", 0)], d[("baseline", 1)]),
        "DeepBeta": f"{beta[1]:.3f}", "DeepRespYear": resp_year,
        "DeepNetExposure": f"{dec[1]['net_exposure']:.2f}", "DeepLongShare": f"{dec[1]['long_share']*100:.0f}",
        "DeepTiltShareLagZero": f"{dec[0]['share_b']*100:.0f}", "DeepTiltShareLagOne": f"{dec[1]['share_b']*100:.0f}",
        "DeepLongShareLagZero": f"{dec[0]['share_a']*100:.0f}", "DeepLongShareLagOne": f"{dec[1]['share_a']*100:.0f}",
        "DeepTiltSRLagZero": f"{dec[0]['sr_b']:.2f}", "DeepTiltSRLagOne": f"{dec[1]['sr_b']:.2f}",
        "DeepLongSRLagZero": f"{dec[0]['sr_a']:.2f}", "DeepLongSRLagOne": f"{dec[1]['sr_a']:.2f}",
        "DeepDecompCorr": f"{max(abs(dec[0]['corr']), abs(dec[1]['corr'])):.2f}",
        "DeepResidSRLagOne": f"{dec[1]['sr_total']:.2f}",
        "DeepLagWorstYear": int(net_by_year[1].idxmin()), "DeepLagWorstYearRet": f"{net_by_year[1].min():.1f}",
        "DeepLagPosYears": int((net_by_year[1] > 0).sum()),
    }
    return m


# ---------------------------------------------------------------- main

def main():
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

    overrides = load_variant_overrides()
    table = load_all_variants(split, cfg.strategy, overrides)
    # The same configuration run under two names counts as one trial.
    key = {name: tuple(sorted(overrides.get(name, {}).items())) for name in table.index}
    table = table.loc[~pd.Series(key).duplicated().values]
    fig_factor_tradeoff(table)
    write_variant_table(table, overrides)
    write_main_table(table)
    deep_macros = deep_and_lag_assets(split, returns)

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
        **deep_macros,
    })


if __name__ == "__main__":
    main()
