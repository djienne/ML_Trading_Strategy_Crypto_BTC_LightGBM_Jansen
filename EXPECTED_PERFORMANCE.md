# Expected Performance Notes

This repository has two useful evaluation layers, and they answer different questions:

- `python main.py backtest`: fast vectorized signal/state PnL approximation
- `python freqtrade_live/backtest_live_window.py`: replay of published live artifacts over real candles

Use the first for fast research. Use the second when you want the closest artifact-aware check
of what the live stack would have tried to do.

The deployed contract values (pair, timeframe, model hyperparameters,
entry/exit quantiles, stoploss, fee assumption) are listed once in
`STRATEGY.md` ("Current Deployed Contract") and sourced from the published
`model_info.json`.

## Current Backtest Results (data through 2026-07-04)

Contract-driven vectorized backtest (`python main.py backtest --symbol BTCUSDT
--interval 15m`), current code (post alpha001 fix), models retrained
2026-07-04 on BTC+ETH data through 2026-07-04:

```
Period: 2020-12-01 → 2026-07-03 (~5.6 years)
Total Trades: 2682
Total Gross Return: 743.6%
Total Net Return:   277.4%  (~27% CAGR)
Annualized Sharpe:  1.33
Fee: 0.03% per transition, Stoploss: -20%
```

> **These numbers predate the deployed model.** The model actually running is
> the one published in `freqtrade_live/shared/models/model_info.json`, whose
> `training_date` is **2026-08-13T21:19:40Z** — about six weeks after the
> 2026-07-04 data cut-off above. The block is therefore a record of the
> 2026-07-04 artifact, not of what is trading today, and the "What To Trust"
> rule below (treat old hand-written summary metrics as stale unless
> regenerated from the current artifact) applies to it. Regenerate before
> quoting these figures as the deployed model's performance.

The fee assumption is now 0.03% per transition (the repo-wide backtest
default in `src/strategy_contract.py` and the published contract). At the
older, more conservative 0.05% assumption the same run yields net +120.7%
(~15% CAGR), Sharpe 0.83 — fee drag dominates at this trade frequency, so
treat realized maker/taker mix as a first-order performance driver.

These current-code numbers are the honest baseline; the April-2026 grid-search
numbers carry selection bias (see below) and predate the 2026-06-10 alpha001
redefinition.

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
3. Compare replay against the Freqtrade trade ledger (`tradesv3.sqlite`) for execution-quality drift, and against the bot's per-candle `user_data/logs/signal_state.csv` to root-cause any signal-level disagreement (prediction vs quantile vs desired-position).

## Selection Bias In Grid-Search Numbers

The grid search (`grid_search_full.py`) scores hundreds of hyperparameter/signal
configurations on the same out-of-fold predictions that the headline metrics are
reported from. The winner's reported Sharpe/net return is therefore an upper
estimate (winner's curse): there is no untouched holdout behind it. Treat
`GRID_SEARCH_RESULTS.md` numbers as relative rankings between configs, not as
unbiased forecasts of live performance. The most honest forward-looking numbers
are the dry-run ledger and the artifact replay.

## Expected Live Performance

Conservative live estimates derived from the current backtest baseline
(0.03% fee, through 2026-07-04):

| Metric | Backtest | Expected Live (conservative) |
|--------|----------|------------------------------|
| CAGR | ~27% | 10-18% |
| Sharpe | 1.33 | 0.6-0.9 |
| Trades/year | ~480 | ~480 |
| Avg trade duration | ~1-2 days | ~1-2 days |
| Max drawdown | ~15-25% | 20-35% |
| Win rate | ~50-55% | ~45-50% |

Why live will likely underperform the backtest:

1. **Slippage**: the backtest assumes fills at the next bar's open; live has
   order-book pricing and unfilled timeouts.
2. **Latency**: the backtest evaluates on closed candles; live has ~seconds delay.
3. **Fee mix**: the backtest uses a flat per-transition fee; realized live fees
   depend on the maker/taker mix and funding.
4. **Model staleness**: live retrains monthly; between retrains the regime may shift.
5. **Stoploss mechanics**: see "Live vs Backtest Expectations" above.

## Measured Live/Replay Parity (June 2026)

Signal parity is measured, not assumed: after the 1-bar execution-parity fix,
18 of 20 dry-run trades matched the artifact replay bar-exactly, with a mean
per-trade |PnL difference| of 0.12 percentage points; the residual mismatches
are knife-edge top-1%-quantile entries. The full trade-by-trade evidence and
methodology live in `REVIEW_FINDINGS.md` ("Measured Evidence").

## Reporting Guidance

If you want fresh headline numbers, regenerate them from the current artifact and include:

- artifact training date
- contract fields from `model_info.json`
- vectorized backtest result
- artifact replay result
- explicit note on whether funding/slippage were modeled
