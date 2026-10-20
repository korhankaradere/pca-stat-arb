"""Compare strategy variants. In-sample results only, unless --reveal-test is passed.

    python scripts/experiments.py                 # design/tuning: in-sample only
    python scripts/experiments.py --reveal-test   # once a variant is final: show the test period

Variants live in configs/experiments.yaml as overrides of the base config's strategy section.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import yaml

from pca_stat_arb import Config
from pca_stat_arb.experiment import load_etf_returns, load_market_data, run_variant
from pca_stat_arb.metrics import split_summary

_DATA = {}


def _init(cfg_path):
    cfg = Config.from_yaml(cfg_path)
    _DATA["cfg"] = cfg
    _DATA["returns"], _DATA["volume"] = load_market_data(cfg)
    _DATA["etf"] = load_etf_returns(cfg, _DATA["returns"].index)


def _run(item):
    name, overrides = item
    res = run_variant(_DATA["cfg"], overrides, _DATA["returns"], _DATA["volume"], _DATA["etf"])
    return name, res.pnl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--experiments", default="configs/experiments.yaml")
    parser.add_argument("--only", nargs="*", help="run just these variant names")
    parser.add_argument("--reveal-test", action="store_true")
    parser.add_argument("--out", default="results/experiments")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    variants = yaml.safe_load(Path(args.experiments).read_text())["variants"]
    if args.only:
        variants = {k: v for k, v in variants.items() if k in args.only}

    # Fill the data cache once here, so the parallel workers only ever read it.
    load_etf_returns(cfg, load_market_data(cfg)[0].index)

    with ProcessPoolExecutor(max_workers=min(len(variants), 4), initializer=_init, initargs=(args.config,)) as ex:
        results = list(ex.map(_run, (variants.items())))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    split = pd.Timestamp(cfg.evaluation.split_date)
    rows = []
    for name, pnl in results:
        pnl.to_csv(out / f"{name}_pnl.csv")
        tbl = split_summary(pnl["net"], cfg.evaluation.split_date, include_test=args.reveal_test)
        gross = split_summary(pnl["gross"], cfg.evaluation.split_date, include_test=args.reveal_test)
        etf_cost = (variants[name].get("etf_cost_bps", cfg.strategy.etf_cost_bps)) / 1e4
        for period in tbl.index:
            p = pnl[pnl.index < split] if period == "in_sample" else (
                pnl[pnl.index >= split] if period == "test" else pnl)
            stock_to, etf_to = p["stock_turnover"].mean(), p["etf_turnover"].mean()
            rows.append({
                "variant": name, "period": period,
                "net_sharpe": tbl.loc[period, "sharpe"], "gross_sharpe": gross.loc[period, "sharpe"],
                "net_ann_ret": tbl.loc[period, "ann_return"], "ann_vol": tbl.loc[period, "ann_vol"],
                "max_dd": tbl.loc[period, "max_drawdown"],
                "stock_turnover": stock_to, "etf_turnover": etf_to,
                # One-way *stock* cost (bps) at which net P&L would be zero, with the ETF leg
                # charged at its own cost: (gross - ETF costs) per unit of stock traded.
                "breakeven_bps": (p["gross"].mean() - etf_to * etf_cost) / stock_to * 1e4,
            })
    table = pd.DataFrame(rows).set_index(["period", "variant"]).sort_index()
    table.to_csv(out / ("summary_with_test.csv" if args.reveal_test else "summary_in_sample.csv"))
    print(f"split date {cfg.evaluation.split_date}; " + ("TEST PERIOD REVEALED" if args.reveal_test else "in-sample only"))
    with pd.option_context("display.width", 140):
        print(table.round(3).to_string())


if __name__ == "__main__":
    main()
