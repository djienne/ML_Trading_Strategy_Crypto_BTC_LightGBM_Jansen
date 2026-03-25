# LightGBM Boosting Strategy

## Overview

Long-only BTC/USDT strategy using a LightGBM gradient boosting model to predict
1-bar forward returns on 15-minute candles. The model is trained on BTC+ETH data
for more training samples and less overfitting, but trading signals are generated
and executed only on BTC.

## Model

| Parameter           | Value   |
|---------------------|---------|
| Objective           | Regression (predict `fwd1bar` = next-bar log return) |
| Learning rate       | 0.01 (fixed, per Ch.12 reference) |
| Num leaves          | 16 |
| Min data in leaf    | 100 |
| Feature fraction    | 0.5 |
| Boosting rounds     | 5000 (with early stopping) |
| Early stopping      | 50 rounds patience |
| Seed                | 42 |
| Device              | CPU (faster than GPU for this data size) |

## Features (20 total)

- **Returns:** ret1bar through ret10bar (log returns over 1-10 bars)
- **Technical indicators:** BOP, CCI, MFI, RSI, StochRSI, SlowK, SlowD, NATR
- **Alpha factors:** alpha054 (close vs open rank correlation), alpha001 (expanding pct rank of returns)

## Training

### Cross-validation: Calendar-month rolling window

- **Training window:** 12 calendar months (rolling forward)
- **Test window:** 1 calendar month
- **Embargo:** First 20 bars of each test month are skipped to prevent
  feature rolling-window overlap with training data
- **Train/val split:** 90/10 within each training fold for early stopping
- **Training symbols:** BTC + ETH (multi-symbol for more data)
- **Inference symbol:** BTC only

### Schedule (live)

- Model is retrained on the 1st of every month
- Retrainer checks hourly; fires when `day >= 1` and hasn't trained this month
- Uses `src/modeling.train_and_predict()` -- same code as backtest

## Signal Generation

### Quantile assignment

Predictions are converted to quantile bins using an **expanding percentile rank**
(Fenwick tree, O(n log n), numba-accelerated). This avoids forward-looking bias
since each bar's quantile only uses predictions seen so far.

| Parameter     | Value |
|---------------|-------|
| Bins          | 200   |
| Min periods   | 1000 (= bins * 5) |
| Method        | Expanding pct rank via Fenwick tree |

### Entry / exit rules (hysteresis)

| Rule          | Condition |
|---------------|-----------|
| Enter long    | quantile >= 198 (top 1%) |
| Exit long     | quantile < 180 (drops below top 10%) |
| Direction     | Long only (`can_short = False`) |

The hysteresis prevents whipsawing: once in a position, stay until the signal
drops significantly (from top 1% all the way down to below top 10%).

### Monthly boundaries

| Boundary           | Rule | Why |
|--------------------|------|-----|
| **Grace period**   | No new entries when `day == 1 AND hour == 0` (first hour of month) | Model is being retrained during this window |
| **Month-end close**| Force close any open position on the last bar of the month | Clean slate before model retraining; prevents holding stale-model positions |
| **Detection**      | `(timestamp + interval).month != timestamp.month` | Identifies the last bar before month rolls over |

Order of operations in the hysteresis loop (per bar):
1. Check `is_month_end` -- force close if position is open
2. Check `is_grace` -- skip entry/exit logic, hold current state (0 after month-end)
3. Normal entry/exit quantile checks

### Fee model

| Parameter     | Value |
|---------------|-------|
| Fee per trade | 0.05% (0.0005) |
| Applied on    | Each position change (entry and exit counted separately) |

### Known backtest vs live approximations

- **Stoploss mechanism:** The backtest tracks cumulative bar returns since entry and
  caps at -20%. Freqtrade's native stoploss is price-based (triggers on unrealized
  drawdown from entry). For 15-minute bars the difference is small (compounding
  over a few bars ≈ linear), but in a flash crash spanning many bars they can diverge.
- **Exit timing:** Backtest exits at bar close; Freqtrade uses limit/market orders
  with potential slippage and timeouts.

## Code Architecture

### Single source of truth: `compute_signal_returns()`

Signal computation is implemented **once** in `src/backtest.py:compute_signal_returns()`.
This function is called by:
- The CLI backtest (`python main.py backtest`)
- The grid search (`grid_search_full.py`)

This ensures identical results across all evaluation paths.

### Key files

| File | Purpose |
|------|---------|
| `src/modeling.py` | Model training with calendar-month rolling CV |
| `src/backtest.py` | `compute_signal_returns()`, hysteresis signal loops, backtesting |
| `src/features.py` | Feature engineering (20 features) |
| `src/utils.py` | Expanding quantile via Fenwick tree |
| `src/pipeline.py` | Orchestrates train/evaluate/backtest pipeline |
| `src/strategy.py` | CLI argument parsing |
| `grid_search_full.py` | Hyperparameter grid search |
| `freqtrade_live/user_data/strategies/LightGBMStrategy.py` | Live Freqtrade strategy |
| `freqtrade_live/retrainer/retrain.py` | Monthly model retrainer (Docker) |
| `freqtrade_live/docker-compose.yml` | Docker Compose (trader + retrainer) |

### Live deployment (Docker)

Two containers via Docker Compose:
1. **freqtrade** -- runs the `LightGBMStrategy`, loads model from shared volume
2. **retrainer** -- monthly retraining, downloads latest data, saves model atomically

Shared volume (`./shared/`) contains models, predictions, and model metadata.
Atomic writes (`.tmp` + `os.replace()`) ensure crash safety.

## Backtest Results

Grid search winner (300 configs, ~6 years of 15m BTC data, Dec 2020 — Mar 2026):
- **Net return:** +283.81%
- **Trades:** 1,380
- **Sharpe ratio:** 1.41
- **Config:** tm=12, L=16, ff=0.5, mdil=100, bins=200, entry>=198, exit<180

Chosen for neighborhood robustness: adjacent parameter values (ff=0.25→+202%,
mdil=50→+197%, mdil=200→+147%) degrade gradually, not as a cliff.

**Selection bias caveat:** The grid search evaluates ~75K candidates on the same
historical sample with no final holdout or nested validation. Top results are
likely optimistic for live deployment. Consider reserving the last 6-12 months
as a true out-of-sample test before deploying any winner.

## CLI Usage

```bash
# Train with deployed config (all defaults match the winner)
python main.py train --retrain --boost-rounds 5000

# Backtest (all defaults match the winner)
python main.py backtest
    --fee 0.0005 --side long

# Grid search (all combinations)
python grid_search_full.py
```
