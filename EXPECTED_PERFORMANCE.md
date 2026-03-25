# Expected Performance (Backtest)

## Deployed Configuration

| Parameter          | Value                                |
|--------------------|--------------------------------------|
| Pair               | BTC/USDT futures                     |
| Timeframe          | 15m                                  |
| Direction           | Long only                            |
| Training window    | 12 calendar months (rolling)         |
| Model              | LightGBM (L=16, ff=0.5, mdil=100, lr=0.01) |
| Entry signal       | Expanding quantile >= 198 / 200 (top 1%) |
| Exit signal        | Expanding quantile < 180 / 200 (below top 10%) |
| Fee                | 0.05% per side (0.10% round-trip)    |
| Stoploss           | -20% cumulative                      |

## Headline Numbers

| Metric                    | Value      |
|---------------------------|------------|
| **Net return**            | +283.8%    |
| **Sharpe ratio**          | 1.41       |
| **Total trades**          | 1,380      |
| **Backtest period**       | Jan 2021 - Mar 2026 (~5.25 years) |
| **CAGR**                  | ~29%       |
| **Avg monthly return**    | ~2.2%      |
| **Trades per day**        | ~0.7       |
| **Trades per month**      | ~22        |

Note: the first 12 months of data (Dec 2019 - Dec 2020) are used as initial
training window, so out-of-sample predictions start from Jan 2021.

## Annualized Breakdown

| Year   | Approximate contribution |
|--------|--------------------------|
| 2021   | First out-of-sample year (model trained on 2020 data) |
| 2022   | Bear market — tests model robustness |
| 2023   | Recovery year |
| 2024   | Bull market |
| 2025-Q1| Most recent quarter |

(Per-year P&L breakdown requires running `python main.py backtest`.)

## Fee Impact

- 1,380 trades x 2 transitions (entry + exit) = ~2,760 fee events
- At 0.05% per event: ~1.38% total fee drag over the full period
- Gross return before fees is higher (~+300%+ estimated)

## Neighbor Robustness

The deployed config was chosen not as the absolute best, but for smooth
degradation across nearby parameter values:

| Variation from deployed      | Net return | Sharpe |
|------------------------------|------------|--------|
| **Deployed** (ff=0.5, mdil=100, exit=90%) | +283.8%    | 1.41   |
| exit=85% (wider exit)       | +275.7%    | 1.37   |
| exit=92% (tighter exit)     | +253.2%    | 1.35   |
| bins=150 (fewer bins)       | +250.2%    | 1.49   |
| bins=100 (fewer bins)       | +241.2%    | 1.37   |
| bins=300 (more bins)        | +235.0%    | 1.29   |

No cliff edges — all neighbors remain profitable with Sharpe > 1.2.

## Grid Search Context

- 300 model configs tested, 75,600 signal combinations evaluated
- All top 20 results use training_months=12 and feature_fraction=0.5
- The absolute best config (#1: L=31, mdil=50) returned +299.6% but was
  not deployed due to less stable neighborhood

## Caveats

1. **No holdout set** — the grid search evaluates all candidates on the same
   historical sample. Top results are likely optimistic for live trading.

2. **No slippage modeled** — backtest assumes fill at Open price. Real fills
   on Binance Futures may be 1-5 bps worse, especially during volatility.

3. **Regime dependence** — the 5.25-year backtest includes both bull and bear
   markets, but future regimes may differ.

4. **Parameter selection bias** — choosing the 2nd-best out of 75,600 combos
   carries inherent overfitting risk despite neighborhood checks.

5. **Retraining assumption** — performance assumes successful monthly retraining.
   Model degradation between retrains is not modeled.

## Realistic Expectations for Live Trading

Given the caveats above, a conservative estimate for live performance would
discount the backtest by 30-50%:

| Scenario      | Expected annual return | Sharpe |
|---------------|----------------------|--------|
| Optimistic    | ~25-29% (matches backtest CAGR) | ~1.4  |
| Moderate      | ~15-20%              | ~0.8-1.0 |
| Conservative  | ~8-12%               | ~0.5-0.7 |

The moderate scenario accounts for slippage, execution differences, and
mild regime shift. The conservative scenario adds parameter overfitting risk.
