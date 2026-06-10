# Machine Learning LightGBM Crypto Trading Strategy

This repo implements a modular ML trading strategy workflow inspired by the Chapter 12 of "Machine Learning for Algorithmic Trading" by Stefan Jansen. The pipeline is split into independent stages so you can run only what you need: download data, build features, train models (with persistence), evaluate signals, and backtest a chosen quantile.

The pipeline supports any candle interval shorter than 1 month (e.g., 1m, 5m, 1h, 1d). Some plots in this README come from earlier 1-minute experiments; the currently deployed live setup (`config.json`, `freqtrade_live/`) trades BTC/USDT futures on 15-minute candles. For short timeframes (e.g., 1m/5m), profitability is only realistic with very low fees (below 0.5 bps); this is generally not achievable for taker trading, but the short-term signal could be used, for example, as alpha for a high frequency market-making model that relies on limit maker orders (very low fees, sometimes rebates).

<figure>
  <img src="plot/ALL_1m_equity_q5000_longshort_5000_date.png" alt="ALL 1m equity curve (q5000 longshort, date scope)" width="700">
  <figcaption>Equity curve for the q5000 longshort setup.</figcaption>
</figure>

## Requirements

- Python 3.10+
- Packages from `requirements.txt`

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Configuration

Edit `config.json` to set data and download behavior. The current repo config
(matching the deployed live setup):

```json
{
  "candle_interval": "15m",
  "request_delay": 0.3,
  "train_symbols": ["BTCUSDT", "ETHUSDT"],
  "inference_symbol": "BTCUSDT",
  "start_date": "2020-01-01",
  "feather_dir": "data/feather",
  "binance_base_url": "https://fapi.binance.com",
  "max_klines_per_request": 1500
}
```

Optional path overrides (defaults shown):

- `data_dir`: `data`
- `processed_dir`: `data/processed`
- `predictions_dir`: `data/predictions`
- `eval_dir`: `data/eval`
- `models_dir`: `models`
- `feature_flags`: toggles feature groups (see below)

Bar configuration:

- `bar_type`: `time` (default) or `volume`.
- `volume_bar_size`: required when `bar_type` is `volume`; volume threshold per bar using the base OHLCV `volume` units.
- `candle_interval`: for volume bars, this is the base time-bar granularity used to build volume bars (e.g., `1m`).

Example for volume bars:

```json
{
  "bar_type": "volume",
  "volume_bar_size": 1000000,
  "candle_interval": "1m"
}
```

Symbol controls:

- `train_symbols`: list used for download, features, and training.
- `inference_symbol`: default symbol for evaluation/backtest (CLI `--symbol` overrides; should exist in `train_symbols`).
- `symbols`: legacy key accepted as a fallback for `train_symbols`.

Feature flags example:

```json
{
  "feature_flags": {
    "returns": true,
    "bop": true,
    "cci": true,
    "mfi": true,
    "rsi": true,
    "stochrsi": true,
    "stoch": true,
    "natr": true,
    "alpha001": true,
    "alpha054": true
  }
}
```

## Pipeline Commands

Run all commands via `main.py`:

### 1) Download data

```powershell
python main.py download
```

By default this uses `train_symbols`. Override symbols:

```powershell
python main.py download --symbols BTCUSDT ETHUSDT
```

### 2) Build features + target

```powershell
python main.py features
```

By default this processes all symbols from `train_symbols` and writes an `ALL` feature file.
To run a single symbol:

```powershell
python main.py features --symbol BTCUSDT --single
```

Rebuild even if cached:

```powershell
python main.py features --recompute
```

### 3) Train model + save predictions (persistent)

```powershell
python main.py train
```

By default this trains on all symbols from `train_symbols` and saves combined predictions.
When training on all symbols, predictions and models are stored under `ALL`.
To train only on one symbol:

```powershell
python main.py train --symbol BTCUSDT --single
```

If models already exist, rerunning `train` continues training by default and adds more boosting rounds (see `--continue-rounds`).

Retrain from scratch (clears saved models and predictions):

```powershell
python main.py train --retrain
```

Continue training with a custom number of extra rounds:

```powershell
python main.py train --continue-rounds 100
```

Control the number of boosting rounds used for fresh training (including `--retrain`):

```powershell
python main.py train --retrain --boost-rounds 1000
```

### 4) Evaluate prediction performance (quantiles)

```powershell
python main.py evaluate --bins 10
```

Evaluation uses the combined predictions (if available) and filters to the target
symbol (default: `inference_symbol`, falling back to the first `train_symbols` entry).
To evaluate a different symbol:

```powershell
python main.py evaluate --bins 10 --symbol XRPUSDT
```

Full flag list:

- `--symbol SYMBOL` — override the evaluation symbol (defaults to `inference_symbol`).
- `--interval INTERVAL` — override the bar interval.
- `--bins BINS` — number of quantiles.
- `--quantile-scope {auto,timestamp,date,global,expanding}` — how quantiles are assigned. Default `auto` uses `expanding` for single-symbol and `timestamp` for multi-symbol data. `global` ranks against the full sample including future bars (lookahead bias — a warning is printed); use it only for diagnostic plots, never for performance claims. Override example:

```powershell
python main.py evaluate --bins 10 --quantile-scope expanding
```

### 5) Backtest a quantile threshold

```powershell
python main.py backtest --bins 10 --quantile 8 --side long --fee 0.001
```

Backtest uses the combined predictions (if available) and trades only the target
symbol (default: `inference_symbol`, falling back to the first `train_symbols` entry).
Equity curve and standardized-signal (alpha) plots are saved under `plot/` for each
backtest run.

**Contract-driven defaults.** When `freqtrade_live/shared/models/model_info.json` is
present and matches the requested symbol/interval, the CLI backtest defaults to that
published live contract for `bins`, entry/exit quantiles, fee, stoploss, direction,
and training-window assumptions. CLI overrides still work, but the backtest prints a
warning and saves the resolved configuration to `plot/*_backtest_*.json` for
traceability. See the [Strategy contract](#strategy-contract) subsection below.

Full flag list:

- `--symbol SYMBOL` — override the symbol traded.
- `--interval INTERVAL` — override the bar interval.
- `--bins BINS` — number of quantiles. Defaults to the contract value when available.
- `--quantile QUANTILE` — entry quantile threshold (long uses `>=` the chosen bin,
  short uses `<=`). Defaults to the contract value.
- `--exit-quantile EXIT_QUANTILE` — exit quantile threshold (exit long when the
  quantile drops below this bin). Defaults to the contract value.
- `--side {auto,long,short,longshort}` — trading side. Default `long`. `longshort`
  uses upper/lower tails and skips the short leg if the tails overlap.
- `--fee FEE` — one-way fee per entry and per exit. Defaults to the contract value.
- `--stoploss STOPLOSS` — per-trade stoploss, e.g. `-0.20` to close at a 20% loss.
  Defaults to the contract value.
- `--ic-thresh IC_THRESH` — skip bars whose validation IC is below this threshold.
- `--direction {high,low}` — hysteresis direction. Defaults to the contract value.
- `--quantile-scope {auto,timestamp,date,global,expanding}` — how quantiles are
  assigned. Default `auto`. Avoid `global` (full-sample lookahead; see the
  evaluate flags above); the hysteresis path ignores scope and uses the
  contract's quantile method.

## Artifacts

The pipeline persists intermediate outputs so you can resume after a restart:

- Features: `data/processed/{symbol}_{bar_id}_model_data.feather`
- Predictions: `data/predictions/{symbol}_{bar_id}_predictions.feather`
- Models: `models/{symbol}_{bar_id}/fold_XX.txt`
- Evaluation summary: `data/eval/{symbol}_{bar_id}_quantiles_{bins}_{scope}.csv`
- Evaluation plot: `data/eval/{symbol}_{bar_id}_quantiles_{bins}_{scope}.png`
- Equity curve plot: `plot/{symbol}_{bar_id}_equity_{rule}_{bins}_{scope}.png`
- Alpha factor plot: `plot/{symbol}_{bar_id}_alpha_{rule}_{bins}_{scope}.png`

When training on multiple symbols, `{symbol}` is `ALL` for the features, predictions, and model directory.
`bar_id` is `{interval}` for time bars, or `vol{volume_bar_size}_{interval}` for volume bars.

## Live Deployment

The `freqtrade_live/` directory contains a complete live (or dry-run) deployment
based on Freqtrade. It runs two Docker containers side by side (see
`freqtrade_live/docker-compose.yml`):

- **`lgbm_trader`** (built from `Dockerfile.freqtrade`) — runs the Freqtrade bot on
  Binance Futures with the `LightGBMStrategy`. Exposes the Freqtrade REST API on
  `127.0.0.1:8080`. Mounts `user_data/` for configs/logs/trades, `shared/` for
  model artifacts, and the repo's `src/` (read-only) so the strategy imports the
  exact same feature/signal code as research and retraining.
- **`lgbm_retrainer`** (built from `Dockerfile.retrainer`) — runs the monthly
  retraining loop. Shares `shared/` with the trader so new artifacts appear
  atomically for the live strategy to pick up.

Bring the stack up with the usual `docker compose up -d --build` from inside
`freqtrade_live/`. The default configuration is **dry-run** (`"dry_run": true`
in `user_data/config.json`, simulated 1000 USDT wallet), which needs no
exchange credentials.

To trade real money you must additionally:

1. Copy `user_data/config-private.json.template` to
   `user_data/config-private.json` and fill in your Binance API key/secret.
2. Add `--config /freqtrade/user_data/config-private.json` to the freqtrade
   service `command` in `docker-compose.yml` (the private config is **not**
   loaded by default).
3. Set `"dry_run": false` and replace the placeholder `api_server`
   password/JWT secret in `user_data/config.json`.

### Shared artifact directory

`freqtrade_live/shared/models/` is the contract surface between training and
trading:

- `current` — atomic pointer file naming the `archive/<stamp>/` snapshot the
  live strategy should load. Flipping it (a single `os.replace`) is the publish
  commit, so the trader always sees a consistent `{model, info, predictions}`
  set.
- `archive/<stamp>/` — immutable snapshots. The strategy loads from the one
  named by `current`; the replay tool stitches them to reconstruct exactly
  which artifacts were live at any past moment.
- `latest_model.txt` — flat copy of the deployed LightGBM booster (legacy
  fallback for pre-pointer deployments).
- `model_info.json` — the strategy contract (see below).
- `latest_predictions.feather` — historical out-of-fold predictions used to
  seed the rolling quantile window in both the live strategy and the replay
  tool.

### Retrainer

`freqtrade_live/retrainer/retrain.py` reuses the same `src/modeling.py` code as the
offline pipeline. On startup it retrains only if the published model is missing,
stale, its config differs, or the training source code (`src/features.py`,
`src/modeling.py`, `src/utils.py`) has changed; afterwards it retrains on the 1st
of each month. Note this means any edit to those files on the host triggers a
full retrain within the hourly poll — restart the trader afterwards so its
in-memory code matches the new model. Training uses BTCUSDT + ETHUSDT for more
data and less overfit, but inference is BTCUSDT-only. Model hyperparameters are
fixed to the deployed grid-search winner (`NUM_LEAVES=31`, `FEATURE_FRACTION=0.5`,
`MIN_DATA_IN_LEAF=50`, `LR=0.01`, `BOOST_ROUNDS=5000`); the signal-contract
values (bins, entry/exit quantiles, direction, stoploss, fee, quantile method,
train months) are single-sourced from `src/strategy_contract.py`. Publishing is
atomic: predictions are saved first, each file is written via a `.tmp` sidecar +
`os.replace`, a snapshot of the new artifacts is copied into `archive/<stamp>/`,
and the `current` pointer is flipped last.

### Live strategy

`freqtrade_live/user_data/strategies/LightGBMStrategy.py` reloads the booster
whenever the `current` pointer names a new snapshot (falling back to
`latest_model.txt` mtime for pre-pointer deployments), reads the contract from
`model_info.json` for timeframe / symbol / bins / entry / exit / direction /
stoploss, and seeds rolling quantiles from `latest_predictions.feather`.

Signals are **level-based** (`compute_live_level_signals`): `enter_long` is set
while `desired_position == 1` and `exit_long` while it is `0` (NaN warmup bars
publish nothing). Freqtrade's own trade state arbitrates, so a limit order that
fails to fill during the signal candle is retried on the next one, and the held
bars still satisfy `position[T+1] = desired[T]` — a single bar of execution
delay that lands the position on exactly the candle the model predicts
(`fwd1bar` = the next candle's return).

`startup_candle_count = 3600` is a correctness invariant, not a warmup
nicety: the hysteresis state machine restarts flat at the start of the kline
window freqtrade provides (`ohlcv_candle_limit + startup_candle_count` bars)
and only resets at month ends, so the window must always reach past the last
month boundary (31-day month = 2976 bars, plus the alpha001 rank window and
feature warmup). Each decision bar's `(prediction, quantile,
desired_position)` is appended to `user_data/logs/signal_state.csv` for exact
live-vs-replay diffing.

### Replay tool

`freqtrade_live/backtest_live_window.py` replays the exact published artifacts
over any past window, automatically stitching snapshots from `shared/models/` and
`shared/models/archive/`. Typical use to sanity-check a live dry-run:

```powershell
python freqtrade_live/backtest_live_window.py --start 2026-04-15 --end 2026-04-17 --print-trades
```

Flags:

- `--start`, `--end` — UTC window bounds (date-only values round to 00:00:00 /
  end-of-day). `--end` defaults to the latest closed candle.
- `--initial-balance` — starting balance for the PnL summary. Default `1000`.
- `--fee` — one-way fee per entry/exit. Defaults to the artifact's fee assumption.
- `--print-trades` — print the full trade table.
- `--trades-csv PATH` — optionally dump the trade table as CSV.

Use this whenever you need bar-for-bar parity with what the live bot actually
did; the offline `main.py backtest` is scoped to quick signal sanity checks and
is not artifact-aware. One caveat: the replay runs *today's* signal/feature code
against the archived models — snapshots capture models and contracts, not code —
so windows that predate a signal-code change will legitimately disagree with
what the bot did at the time (measured parity numbers are in
`EXPECTED_PERFORMANCE.md` and `REVIEW_FINDINGS.md`). Its feature warmup
(`FEATURE_WARMUP_BARS`) mirrors the live bot's `startup_candle_count` budget.

### PnL monitor

`freqtrade_live/show_PnL.py` discovers every running container whose name
contains `lgbm`, queries its Freqtrade API (`/profit`, `/status`, `/trades`),
and prints a per-container table of trade count, time since last trade, win
rate, profit factor, annualized Sharpe, max drawdown, days since first trade,
and annualized CAGR. The Sharpe uses the same convention as the vectorized
backtest (per-bar returns with flat bars counted as 0, annualized by
sqrt(bars/year)), so it is directly comparable to grid-search Sharpe numbers.
Credentials are read from `freqtrade_live/user_data/config.json`.

### Strategy contract

`freqtrade_live/shared/models/model_info.json` is the single source of truth that
binds training → offline backtest → live strategy. It records the interval,
inference symbol, bins, entry/exit quantiles, direction, stoploss, fee assumption,
train_months, best_iteration, feature_flags, and quantile method used when the
model was trained. The schema and validation live in `src/strategy_contract.py`
(`build_strategy_contract`, `read_strategy_contract`, `REQUIRED_KEYS`). Both
`python main.py backtest` (via `src/pipeline.py`) and the live strategy read this
file; changes to the contract propagate to every consumer without manual wiring.

## Alpha Plot

<figure>
  <img src="plot/ALL_1m_alpha_q5000_longshort_5000_date.png" alt="ALL 1m alpha factor (q5000 longshort, date scope)" width="700">
  <figcaption>Alpha factor derived from the standardized signal.</figcaption>
</figure>

## Notes

- The pipeline supports multi-symbol training and single-symbol evaluation/backtest by design; use `train_symbols` for the training set and `inference_symbol` (or `--symbol`) for evaluation/backtest.
- Candle intervals shorter than 1 month are supported; the deployed setup uses `15m`.
- Quantile assignment defaults to `--quantile-scope auto`, which uses `expanding` for single-symbol data and `timestamp` for multi-symbol data. Override with `--quantile-scope {timestamp,date,global,expanding}` if needed (`global` has full-sample lookahead and is for diagnostics only).
- The backtest is a vectorized approximation meant for quick signal sanity checks, not a full execution-quality simulation.
- The offline backtest, `backtest_live_window.py`, and the live bot all apply a single bar of execution delay between the `desired_position` value and the held candle: `desired_position == 1` on candle T means holding candle T+1, which is the candle the model's target (`fwd1bar`) actually predicts. The offline backtest bakes this into the target (`fwd1bar` = next candle's return); the replay tool shifts positions by one bar; the live bot publishes level signals and lets Freqtrade's fill-at-next-bar provide the delay.
- All features are window-invariant at the decision bar (pinned by `tests/test_feature_window_parity.py`): alpha001 uses a fixed rolling rank window (`ALPHA001_RANK_WINDOW` in `src/features.py`) rather than an expanding rank, so live, replay, and research compute identical values regardless of where their dataframes start.
- `freqtrade_live/backtest_live_window.py` is the artifact-aware replay tool to use when checking live parity instead of only offline signal PnL.
- The backtest alpha factor is a 1-day rolling z-score of the trading signal per symbol (min 60 minutes, both scaled to bars), scaled by 0.01 and averaged by timestamp for plotting.
- A full code/parity review with measured live-vs-replay numbers (June 2026) lives in `REVIEW_FINDINGS.md`.
