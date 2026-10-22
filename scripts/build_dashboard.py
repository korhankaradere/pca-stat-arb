"""Build the interactive results dashboard (dashboard/index.html) from saved daily P&L.

    python scripts/build_dashboard.py

Only in-sample data (before the evaluation split date) is embedded. The page recomputes net
P&L in the browser from gross P&L and turnover, so the viewer can change the trading cost.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from pca_stat_arb import Config

# key -> (label, {lag: results-relative csv}, note). Order fixes each strategy's colour slot.
STRATEGIES = {
    "deep": ("Deep learning (CNN + Transformer)",
             {0: "deep/deep_lag0_pnl.csv", 1: "deep/deep_lag1_pnl.csv"},
             "Walk-forward: each year traded by a model trained only on earlier years."),
    "pca": ("PCA original (Avellaneda & Lee)",
            {0: "etf_and_lag/baseline_pnl.csv", 1: "etf_and_lag/baseline_lag1_pnl.csv"},
            "15 factors, the paper's thresholds, daily re-hedging."),
    "pca_best": ("PCA, 8 factors + no-trade band",
                 {0: "etf_and_lag/f8_band20_pnl.csv", 1: "etf_and_lag/f8_band20_lag1_pnl.csv"},
                 "Best in-sample PCA variant. Chosen using 2011–2018, so its numbers are flattered."),
}


def main() -> None:
    cfg = Config.from_yaml(ROOT / "configs" / "default.yaml")
    split = pd.Timestamp(cfg.evaluation.split_date)
    frames, meta = {}, {}
    for key, (label, by_lag, note) in STRATEGIES.items():
        for lag, rel in by_lag.items():
            path = ROOT / "results" / rel
            if not path.exists():
                continue
            p = pd.read_csv(path, index_col=0, parse_dates=True)
            frames[(key, lag)] = p[p.index < split]  # hard wall: nothing from the test period
        meta[key] = {"label": label, "note": note, "available": any((key, l) in frames for l in (0, 1))}

    dates = sorted(set().union(*[f.index for f in frames.values()]))
    idx = pd.DatetimeIndex(dates)
    series = {}
    for (key, lag), p in frames.items():
        p = p.reindex(idx)
        r = lambda s, d: [None if pd.isna(v) else round(float(v), d) for v in s]
        series[f"{key}_{lag}"] = {
            "gross": r(p["gross"], 7), "sto": r(p["stock_turnover"], 4), "eto": r(p["etf_turnover"], 4),
        }
    data = {
        "dates": [d.strftime("%Y-%m-%d") for d in idx],
        "strategies": meta,
        "series": series,
        "split": split.strftime("%Y-%m-%d"),
        "etf_cost_bps": cfg.strategy.etf_cost_bps,
        "generated": pd.Timestamp.now().strftime("%d %B %Y"),
    }
    template = (ROOT / "dashboard" / "template.html").read_text()
    out = ROOT / "dashboard" / "index.html"
    out.write_text(template.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"))))
    print(f"{out.relative_to(ROOT)}: {len(idx)} days, series {sorted(series)} ({out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
