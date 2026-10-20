# PCA-Based Statistical Arbitrage on US Equities

A replication and extension of Avellaneda & Lee (2010), *Statistical arbitrage in the US equities market*, Quantitative Finance 10(7).

## Method

Every day *t*, using only data up to the close of *t*:

1. **Eigenportfolios.** Take the correlation matrix of the last 252 days of standardised returns and keep the top 15 eigenvectors. Eigenportfolio weights are `Q_i^(j) = v_i^(j) / σ_i`.
2. **Factor regression.** Regress each stock's last 60 daily returns on the 15 eigenportfolio returns. Call the residuals ε.
3. **OU fit.** Fit an AR(1) to the cumulative residual `X = Σε`. This gives κ (speed of mean reversion), m and σ_eq. Stocks with κ < 252/30 (mean-reversion time over about 30 days) are dropped.
4. **s-score.** `s = (X_last − m̄) / σ_eq`, where m̄ is centred cross-sectionally.
5. **Trade.** Open a long when s < −1.25 and a short when s > 1.25. Close a long when s > −0.5 and a short when s < 0.75. Each position is hedged with the eigenportfolios, so the book is factor-neutral.
6. Weights set at close *t* earn day *t+1*'s return. Costs are charged on turnover, at 5 bps one-way.

## Layout

```
src/pca_stat_arb/
  config.py        typed config, loaded from configs/*.yaml
  pca.py           eigenportfolios
  residuals.py     factor regression, OU fit, s-scores
  signals.py       trading rule
  backtest.py      daily walk-forward engine
  metrics.py       Sharpe, drawdown, turnover
  data/            ReturnsSource interface; yfinance (working), CRSP (planned)
tests/             synthetic-market tests (including a no-lookahead check)
scripts/run_backtest.py
```

## Usage

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
.venv/bin/python scripts/run_backtest.py configs/default.yaml          # one config, writes results/
.venv/bin/python scripts/experiments.py --experiments configs/sweep_band.yaml   # compare variants
```

## Technical report

LaTeX draft in `report/`. Every figure, table and quoted number is generated from the code's output by `scripts/make_report_assets.py`, using in-sample data only.

```bash
sh report/build.sh      # regenerates assets, then compiles report/main.pdf with Tectonic
```

## Evaluation protocol

- **In-sample** is 2011–2018 and is used for all design and tuning. **Test** is 2019–2025, held out.
- `scripts/experiments.py` shows in-sample results only. `--reveal-test` is used once, for the final shortlisted variants.
- Every variant tried is saved under `results/`. The trial count feeds the deflated-Sharpe adjustment (phase 2).
- `breakeven_bps` is the one-way cost at which net P&L would be zero, i.e. gross return per unit of turnover. Compare it with the assumed 5 bps.

## Findings so far (in-sample 2011–2018, 5 bps one-way)

| Variant | Gross Sharpe | Net Sharpe | Turnover/day | Break-even (bps) |
|---|---|---|---|---|
| AL baseline (15 factors) | 1.33 | −1.18 | 0.67 | 2.6 |
| Frozen hedges | 1.16 | −1.10 | 0.61 | — |
| Volume-adjusted signal | 1.29 | −1.15 | 0.63 | — |
| Wider entry (s > 2.0) | 0.83 | −1.14 | 0.28 | 2.1 |
| 3 factors | 1.29 | 0.09 | 0.41 | 5.4 |
| 3 factors + 0.0005 band | 1.24 | 0.17 | 0.36 | 5.8 |
| 8 factors + 0.002 band | 1.44 | 0.16 | 0.37 | 5.6 |

1. The mean-reversion alpha is real but thin: gross Sharpe is about 1.3–1.5, yet only about 2.5 bps per unit traded at the AL settings.
2. Costs come mainly from **hedge baskets traded at entry and exit**, not from daily re-hedging. Each factor adds a basket across every stock in the universe.
3. Fewer factors and a no-trade band roughly double the edge per trade. More extreme entry thresholds do not help; extreme s-scores aren't more informative.
4. The volume ("trading time") adjustment doesn't help in 2011–18, unlike AL's 2003–07 result.
5. **Execution lag matters.** Trading one day after the signal (`execution_lag: 1`) cuts gross Sharpe by 20–45% (baseline 1.33 → 0.74; 8 factors + band 1.44 → 1.18). With realistic timing, every variant is negative net of 5 bps.
6. **Sector-ETF factors** (`factor_model: etf`, hedges traded at 1 bp) halve stock turnover. But the residual signal is weaker (gross Sharpe 1.22), so the net result is about break-even (Sharpe +0.02 at lag 0, −0.39 at lag 1).

## Roadmap

1. ~~Evaluation protocol, cost diagnostics, frozen hedges, volume signal, no-trade band~~
2. Research tooling: deflated Sharpe, bootstrap CIs, cost-sensitivity curves, regime and decay analysis, notebooks
3. Extensions: sector-ETF hedging (cheap execution), Marchenko–Pastur factor selection, covariance shrinkage, ML signal layer
4. CRSP/WRDS data (removes survivorship bias)
5. Final held-out test, report and dashboard

## Known limitations

- **Survivorship bias.** The universe is *today's* S&P 500 applied back to 2010. Stocks that were delisted or removed are missing, so absolute numbers are biased. The planned fix is CRSP via WRDS (`data/crsp_source.py`).
- A missing next-day return (for example a delisting) is treated as 0.
- Days with zero Yahoo volume are treated as non-trading. Yahoo backfills prices before some US listings (SW, FERG, AMCR).
- Costs are a flat 5 bps on every name. Real large-cap costs vary, and hedges could be done more cheaply through ETFs.
- Capital is fixed at 1 with no compounding; there is no borrow cost and no market impact.
