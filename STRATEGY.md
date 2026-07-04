# LightGBM Boosting Strategy

## Overview

Long-only BTC/USDT futures strategy using a LightGBM regression model on 15-minute candles.
Training uses BTC + ETH for additional samples, while inference and trading use BTC only.

The repo now treats the deployed live setup as an artifact-owned contract. The published
`model_info.json` contains both model settings and signal settings, and the live strategy,
artifact replay, and CLI backtest all consume that same contract by default.

## Current Deployed Contract

This table is the single documented snapshot of the deployed contract; the
authoritative source is the published `model_info.json`
(`freqtrade_live/shared/models/`), with defaults defined in
`src/strategy_contract.py`. Other docs link here instead of restating values.

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
| Fee assumption | `0.03%` per transition |

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

Feature generation is shared between offline training, live inference, and replay,
and every feature is window-invariant at the decision bar: its value does not
depend on where the dataframe starts, given enough warmup (pinned by
`tests/test_feature_window_parity.py`). This matters because the live bot only
sees the freqtrade kline window, not the full history. In particular `alpha001`
uses a fixed rolling time-series rank (`ALPHA001_RANK_WINDOW = 480` bars in
`src/features.py`, per symbol); before 2026-06-10 it was an expanding rank from
dataframe start, which made live values diverge from research and flipped ~4% of
entries — models trained before that date are not comparable with current code.

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

### Live execution invariants

Two properties keep the live bot equivalent to the offline state machine:

- **Window coverage.** The hysteresis machine restarts flat at the start of
  whatever dataframe it is given, and its only reset point is the month-end
  force-flat. The live kline window (`ohlcv_candle_limit +
  startup_candle_count` bars) must therefore always reach past the last month
  boundary; `LightGBMStrategy.startup_candle_count = 3600` budgets a 31-day
  month (2976 bars) + the alpha001 rank window (480) + feature warmup.
- **Level-based signals.** The bot publishes `enter_long` while
  `desired_position == 1` and `exit_long` while it is `0` (NaN bars publish
  nothing). Freqtrade's trade state arbitrates, so a missed limit-order fill is
  retried on the next candle instead of being lost, and held bars satisfy
  `position[T+1] = desired[T]` — the same single bar of execution delay the
  offline target (`fwd1bar`) encodes.

## Parity Boundaries

Two different parity claims matter:

- **Signal parity**: given candles + artifact, offline and live paths should produce the same features, predictions, quantiles, and desired position state.
- **Execution parity**: actual fills, slippage, order-book pricing, timeouts, and funding in Freqtrade/exchange behavior.

This repository is designed to make **signal parity** tight. It does **not** claim that the
vectorized backtest is an execution-faithful simulation of live Binance futures trading.

Signal parity is measured, not assumed: a trade-by-trade diff of the dry-run
ledger against the artifact replay (June 2026) matched 18 of 20 trades
bar-exactly with a mean per-trade |PnL difference| of 0.12 percentage points.
See `REVIEW_FINDINGS.md` for the full review and `EXPECTED_PERFORMANCE.md` for
the numbers and caveats.

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
