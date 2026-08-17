# freqtrade_live — deployed LightGBM stack

The parent `boosting_strategy_LightGBM/` is the **research/backtest** side (data download, grid search, `src/`, `main.py`). This subfolder is the **live deployment**: two containers, one trading and one retraining the model it trades on.

## Services vs containers

- service `freqtrade` → container `lgbm_trader`, built from `Dockerfile.freqtrade` (`freqtradeorg/freqtrade:2025.10` + lightgbm, scipy, pyarrow)
- service `retrainer` → container `lgbm_retrainer`, built from `Dockerfile.retrainer` (`python:3.10-slim` + lightgbm, pandas, numpy, scipy, aiohttp, pyarrow)

`docker compose` takes the **service** name — `docker compose restart retrainer`, *not* `docker compose restart lgbm_retrainer`. Container names are for `docker logs` / `docker exec` only.

## Identity

| | |
|---|---|
| Strategy class / file | `LightGBMStrategy` — `user_data/strategies/LightGBMStrategy.py` |
| Timeframe | `15m` (from the strategy; `startup_candle_count = 3600`, `can_short = False`) |
| Mode / pair | `futures`, `isolated`, `BTC/USDT:USDT`, `StaticPairList`, `max_open_trades: 1` |
| API port | `127.0.0.1:8080:8080` |
| Limits | trader `mem_limit: 2g`, retrainer `mem_limit: 4g`; `cpus: 0.1` both |
| Image tag | none — compose builds unnamed, no `ft-freqtrade_live:2025.10` |
| Extra mounts | `../src:ro` in both; `../download_data.py:ro` and `../data/feather` in the retrainer |

## Model handoff

`retrainer/retrain.py` writes into `shared/models/`: `latest_model.txt` (last-fold booster), `model_info.json` (published contract + `training_date`), `latest_predictions.feather` (history for the rolling quantiles), an immutable `archive/<stamp>/` snapshot of all three, and `current` — a pointer naming the snapshot to load. Publishing commits with a single atomic flip of `current`, so the trader never sees a half-written set. The strategy's `bot_loop_start` calls `_maybe_reload_model()`, which resolves `current` → `archive/<stamp>/` and falls back to the flat files keyed by mtime; a model whose `interval` disagrees with the strategy `timeframe` is **rejected**, not loaded.

The retrainer polls hourly (`CHECK_INTERVAL_SECONDS = 3600`) and retrains on or after the 1st of any month it has not yet trained in; it also retrains at startup if artifacts are missing/stale or the training config or `src/` training code changed.

**Staleness check:** `cat shared/models/current` and the `training_date` in `shared/models/model_info.json` — currently `2026-08-13T21:19:40Z`, with 14 archived snapshots. If that month is older than last month the retrainer is not publishing; check `docker logs lgbm_retrainer`.

## Strategy contract

`../STRATEGY.md` ("Current Deployed Contract") is authoritative for parameters — not restated here. `../EXPECTED_PERFORMANCE.md` has the headline numbers, but they come from **data through 2026-07-04** with models retrained on that date, while the deployed model was retrained **2026-08-13**; treat them as indicative, not as this model's record. `backtest_live_window.py` replays the actually published artifacts.

## Run

```powershell
docker compose up -d --build
docker compose logs -f freqtrade
docker compose restart retrainer
```

## Careful

- `user_data/config.json` still ships placeholder `api_server` credentials (`GENERATE_A_RANDOM_SECRET_BEFORE_DEPLOYING`, `SET_A_REAL_PASSWORD_BEFORE_DEPLOYING`) and listens on `0.0.0.0` inside the container; only the loopback port publish keeps it off the LAN.
- **No `commands.txt`** here (unlike ~50 other bots in the fleet) and **no `Dockerfile.technical`** — the trader image is `Dockerfile.freqtrade`. `Dockerfile.freqtrade.bak` and `user_data/config.json.bak` are leftovers, not live.
- `retrainer/retrain.py` is baked into the image *and* bind-mounted read-only: code edits need a service restart, not a rebuild. The trader appends one row per 15m candle to `user_data/logs/signal_state.csv`; a stalled file means the strategy is not producing decisions.

Fleet defaults apply (binance, dry-run, 0.1 CPU / 512M, `ft-<folder>:2025.10`); only deviations are listed above.
<!-- ft-facts: container=lgbm_trader strategy=LightGBMStrategy file=user_data/strategies/LightGBMStrategy.py port=8080 tf=15m mode=futures -->
