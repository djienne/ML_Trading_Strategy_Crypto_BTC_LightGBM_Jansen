# Expected Performance Notes

This repository has two useful evaluation layers, and they answer different questions:

- `python main.py backtest`: fast vectorized signal/state PnL approximation
- `python freqtrade_live/backtest_live_window.py`: replay of published live artifacts over real candles

Use the first for fast research. Use the second when you want the closest artifact-aware check
of what the live stack would have tried to do.

## Current Deployed Contract

| Parameter | Value |
|-----------|-------|
| Pair | BTC/USDT futures |
| Timeframe | `15m` |
| Direction | Long only (`high`) |
| Train months | `12` |
| Model | LightGBM (`num_leaves=31`, `min_data_in_leaf=50`, `feature_fraction=0.5`, `lr=0.01`) |
| Entry | Rolling quantile `>= 100 / 100` |
| Exit | Rolling quantile `< 90 / 100` |
| Stoploss | `-20%` |
| Fee assumption | `0.05%` per transition |

## What To Trust

- Trust the published artifact contract for the currently deployed configuration.
- Trust replay more than the generic vectorized backtest for live parity questions.
- Treat old hand-written summary metrics as stale unless they were regenerated from the current artifact.

## Fee Math Correction

The old fee example in this repo was wrong by 100x.

- `1,380` closed trades implies about `2,760` fee events if you count entry and exit separately.
- `2,760 * 0.05% = 138%` as a simple additive fee sum, not `1.38%`.

That simple sum is still only a rough intuition, because realized fee drag depends on when the
transitions occur and how returns compound over time. But the old arithmetic was definitely wrong.

## Live vs Backtest Expectations

The vectorized backtest is still useful, but it is not a fill-faithful Binance futures simulator.
Expect divergence from live results because of:

- order-book pricing and unfilled timeouts in Freqtrade
- exchange slippage and latency
- funding on futures positions
- different stoploss mechanics between vectorized backtest and live trading: the
  vectorized stoploss triggers on the bar-by-bar open→close cumulative return,
  not the intrabar low Freqtrade uses. The backtest does mirror live's
  "re-entry requires a fresh entry signal after a stoploss close" behaviour, but
  the exact trigger price/bar can still differ.

## Recommended Validation Flow

1. Use `python main.py backtest` to sanity-check the strategy contract quickly.
2. Use `python freqtrade_live/backtest_live_window.py` for artifact-aware replay over the live window you care about.
3. Compare replay against Freqtrade dry-run/live logs for execution-quality drift.

## Reporting Guidance

If you want fresh headline numbers, regenerate them from the current artifact and include:

- artifact training date
- contract fields from `model_info.json`
- vectorized backtest result
- artifact replay result
- explicit note on whether funding/slippage were modeled
