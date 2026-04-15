# LightGBM Boosting Strategy

## Overview

Long-only BTC/USDT futures strategy using a LightGBM regression model on 15-minute candles.
Training uses BTC + ETH for additional samples, while inference and trading use BTC only.

The repo now treats the deployed live setup as an artifact-owned contract. The published
`model_info.json` contains both model settings and signal settings, and the live strategy,
artifact replay, and CLI backtest all consume that same contract by default.

## Current Deployed Contract

| Parameter | Value |
|-----------|-------|
| Interval | `15m` |
| Inference symbol | `BTCUSDT` |
| Train months | `12` |
| Num leaves | `31` |
| Min data in leaf | `50` |
| Feature fraction | `0.5` |
| Learning rate | `0.01` |
| Bins | `100` |
| Entry quantile | `100` |
| Exit quantile | `90` |
| Direction | `high` |
| Stoploss | `-20%` |
| Quantile method | `rolling` |
| Fee assumption | `0.05%` per transition |

## Model Objective

The target is **next-bar simple return**, not log return.

- `ret1bar = close / open - 1`
- `fwd1bar = ret1bar.shift(-1)`

That means the model predicts the next bar's simple open-to-close return.

## Features

The current feature set is driven by `feature_flags` and includes:

- Returns: `ret1bar` through `ret10bar` using simple returns
- Indicators: `bop`, `cci`, `mfi`, `rsi`, `stochrsi`, `slowk`, `slowd`, `natr`
- Alpha factors: `alpha054`, `alpha001`

Feature generation is shared between offline training, live inference, and replay.

## Training

Training uses calendar-month rolling cross-validation:

- Train window: 12 calendar months
- Test window: 1 calendar month
- Embargo: first 20 bars per symbol in each test month
- Validation split: last 10% of each training fold
- Training symbols: BTC + ETH
- Live inference symbol: BTC only

The main training path is `src/modeling.train_and_predict()`. Grid search and retraining
both call that same function instead of reimplementing training logic.

## Signal Generation

Predictions are converted to **rolling quantile bins**, not expanding quantiles. The
rolling window length is derived from `train_months`, so the signal distribution reflects
the currently deployed model regime instead of the entire historical archive.

Entry and exit use a long-only hysteresis state machine:

- Enter long when quantile is `>= 100`
- Exit long when quantile falls below `90`
- Force close on the last bar of each month
- Block new entries during the first hour of the month

The shared signal engine is used by:

- offline hysteresis backtest
- live strategy state derivation
- live-artifact replay

## Parity Boundaries

Two different parity claims matter:

- **Signal parity**: given candles + artifact, offline and live paths should produce the same features, predictions, quantiles, and desired position state.
- **Execution parity**: actual fills, slippage, order-book pricing, timeouts, and funding in Freqtrade/exchange behavior.

This repository is designed to make **signal parity** tight. It does **not** claim that the
vectorized backtest is an execution-faithful simulation of live Binance futures trading.

## Execution Differences

The vectorized backtest remains an approximation:

- Backtest stoploss is based on cumulative bar returns inside the signal engine.
- Freqtrade stoploss is based on live unrealized PnL from the entry price.
- Live trading uses Binance futures order placement, order-book pricing, and unfilled timeouts.
- Funding is a live futures effect and is not fully modeled by the generic vectorized backtest.

For artifact-aware validation of the live stack, use:

```bash
python freqtrade_live/backtest_live_window.py --start 2026-04-04
```

## Artifact Contract

The published `model_info.json` now carries the live contract fields used by all consumers:

- `interval`
- `inference_symbol`
- `feature_flags`
- `best_iteration`
- `train_months`
- `bins`
- `entry_quantile`
- `exit_quantile`
- `direction`
- `stoploss`
- `fee_assumption`
- `quantile_method`

If required contract keys are missing, live/replay now reject the artifact instead of
quietly falling back to stale hardcoded settings.
