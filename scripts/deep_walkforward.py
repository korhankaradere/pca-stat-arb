"""Walk-forward training and evaluation of the deep-learning stat-arb strategy (in-sample only).

    python scripts/deep_walkforward.py --lag 0
    python scripts/deep_walkforward.py --lag 1

Each year from DeepConfig.first_trade_year to the last in-sample year is traded by an
ensemble trained only on earlier data. Data is cut at the evaluation split date first, so the
held-out test period is never touched.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch

from pca_stat_arb import Config
from pca_stat_arb.data.base import CACHE_DIR
from pca_stat_arb.deep.config import DeepConfig
from pca_stat_arb.deep.residual_panel import ResidualPanel, build_residual_panel
from pca_stat_arb.deep.walkforward import run_walkforward
from pca_stat_arb.experiment import load_market_data
from pca_stat_arb.metrics import summary


def cached_panel(returns: pd.DataFrame, cfg: DeepConfig) -> ResidualPanel:
    key = hashlib.sha1(
        f"{list(returns.columns)}|{returns.index[0]}|{returns.index[-1]}|{cfg.n_factors}|{cfg.pca_window}|{cfg.beta_window}".encode()
    ).hexdigest()[:10]
    path = CACHE_DIR / f"residual_panel_{key}.npz"
    if path.exists():
        z = np.load(path)
        return ResidualPanel(returns.index, returns.columns, z["eps"], z["Q"], z["B"], z["tradable"])
    print("building residual panel (one-off, ~1 min) ...")
    panel = build_residual_panel(returns, cfg.n_factors, cfg.pca_window, cfg.beta_window)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, eps=panel.eps, Q=panel.Q, B=panel.B, tradable=panel.tradable)
    tmp.replace(path)
    return panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lag", type=int, default=0, help="execution lag the model is trained and traded with")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="results/deep")
    parser.add_argument("--benchmark", action="store_true", help="time one training step and exit")
    args = parser.parse_args()

    cfg = Config.from_yaml(ROOT / "configs" / "default.yaml")
    dcfg = replace(DeepConfig(), execution_lag=args.lag)
    split = pd.Timestamp(cfg.evaluation.split_date)
    returns, _ = load_market_data(cfg)
    returns = returns[returns.index < split]  # hard wall: in-sample only
    panel = cached_panel(returns, dcfg)

    if args.benchmark:
        from pca_stat_arb.deep.model import CNNTransformer
        from pca_stat_arb.deep.residual_panel import cumulative_residual_features
        from pca_stat_arb.deep.train import make_tensors, portfolio_path, sharpe
        X, valid = cumulative_residual_features(panel, dcfg.lookback)
        for dev in ["cpu", "mps"] if torch.backends.mps.is_available() else ["cpu"]:
            d = make_tensors(X, valid, panel.eps, panel.Q, panel.B, 0, dev)
            model = CNNTransformer(dcfg.lookback).to(dev)
            opt = torch.optim.Adam(model.parameters())
            for i in range(4):
                if i == 1:
                    t0 = time.time()
                net, _, _ = portfolio_path(model, d, slice(1000, 1000 + dcfg.block_days), dcfg.cost_bps)
                loss = -sharpe(net[1:]); opt.zero_grad(); loss.backward(); opt.step()
            if dev == "mps":
                torch.mps.synchronize()
            print(f"{dev}: {(time.time() - t0) / 3:.3f} s per training step")
        return

    if args.device == "mps":
        # Hard cap (~6 GB on a 16 GB Mac): fail with an error rather than push the system into swap.
        torch.mps.set_per_process_memory_fraction(0.375)
    t0 = time.time()
    res = run_walkforward(returns, panel, dcfg, last_year=split.year - 1, device=args.device,
                          checkpoint_dir=ROOT / "results" / "deep" / "checkpoints")
    name = f"deep_lag{args.lag}"
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    res.pnl.to_csv(out / f"{name}_pnl.csv")
    res.folds.to_csv(out / f"{name}_folds.csv", index=False)
    res.residual_weights.astype("float32").to_parquet(out / f"{name}_residual_weights.parquet")
    torch.save(res.models, out / f"{name}_models.pt")

    net, gross = summary(res.pnl["net"]), summary(res.pnl["gross"])
    to = res.pnl["stock_turnover"].mean()
    print(f"\n{name}: walk-forward {res.pnl.index[0].date()}..{res.pnl.index[-1].date()} "
          f"({(time.time() - t0) / 60:.1f} min)")
    print(f"  gross Sharpe {gross['sharpe']:+.2f}   net Sharpe {net['sharpe']:+.2f}   "
          f"net return {net['ann_return']*100:+.1f}%/yr   turnover {to:.2f}/day   "
          f"break-even {res.pnl['gross'].mean() / to * 1e4:.1f} bps")
    print(res.folds.to_string(index=False))


if __name__ == "__main__":
    main()
