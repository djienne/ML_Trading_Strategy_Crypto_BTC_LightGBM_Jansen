# Machine Learning LightBGMCrypto Trading Strategy

This repo implements a modular ML trading strategy workflow inspired by the Chapter 12 of "Machine Learning for Algorithmic Trading" by Stefan Jansen. The pipeline is split into independent stages so you can run only what you need: download data, build features, train models (with persistence), evaluate signals, and backtest a chosen quantile.

The pipeline supports any candle interval shorter than 1 month (e.g., 1m, 5m, 1h, 1d). The examples and defaults in this README focus on 1-minute candles. For short timeframes (e.g., 1m/5m), profitability is only realistic with very low fees (below 0.5 bps); this is generally not achievable for taker trading, but the short-term signal could be used, for example, as alpha for a high frequency market-making model that relies on limit maker orders (very low fees, sometimes rebates).

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

Edit `config.json` to set data and download behavior:

```json
{
  "candle_interval": "1m",
  "bar_type": "time",
  "request_delay": 0.3,
  "train_symbols": ["BTCUSDT", "XRPUSDT"],
  "inference_symbol": "BTCUSDT",
  "start_date": "2018-01-01",
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

Optional quantile scope for single-symbol data (intraday defaults to `date`, daily+ defaults to `global`):

```powershell
python main.py evaluate --bins 10 --quantile-scope date
```

### 5) Backtest a quantile threshold

```powershell
python main.py backtest --bins 10 --quantile 8 --side long --fee 0.001
```

Backtest uses the combined predictions (if available) and trades only the target
symbol (default: `inference_symbol`, falling back to the first `train_symbols` entry).
If `--quantile` is omitted, the backtest uses long top / short bottom quantiles.
Equity curve and standardized-signal (alpha) plots are saved under `plot/` for each backtest run.
`--quantile` is a threshold: long uses `>=` and short uses `<=` the chosen bin. `--side longshort`
uses upper/lower tails (and skips the short leg if the tails overlap).

## Artifacts

The pipeline persists intermediate outputs so you can resume after a restart:

- Features: `data/processed/{symbol}_{bar_id}_model_data.feather`
- Predictions: `data/predictions/{symbol}_{bar_id}_predictions.feather`
- Models: `models/{symbol}_{bar_id}/fold_XX.txt`
- Evaluation summary: `data/eval/{symbol}_{bar_id}_quantiles_{bins}.csv`
- Evaluation plot: `data/eval/{symbol}_{bar_id}_quantiles_{bins}_{scope}.png`
- Equity curve plot: `plot/{symbol}_{bar_id}_equity_{rule}_{bins}_{scope}.png`
- Alpha factor plot: `plot/{symbol}_{bar_id}_alpha_{rule}_{bins}_{scope}.png`

When training on multiple symbols, `{symbol}` is `ALL` for the features, predictions, and model directory.
`bar_id` is `{interval}` for time bars, or `vol{volume_bar_size}_{interval}` for volume bars.

## Alpha Plot

<figure>
  <img src="plot/ALL_1m_alpha_q5000_longshort_5000_date.png" alt="ALL 1m alpha factor (q5000 longshort, date scope)" width="700">
  <figcaption>Alpha factor derived from the standardized signal.</figcaption>
</figure>

## Notes

- The pipeline supports multi-symbol training and single-symbol evaluation/backtest by design; use `train_symbols` for the training set and `inference_symbol` (or `--symbol`) for evaluation/backtest.
- Candle intervals shorter than 1 month are supported; the README examples assume `1m`.
- Quantile assignment defaults to `timestamp` for multi-symbol data, `date` for intraday single-symbol data, and `global` for daily+ or volume-bar single-symbol data; override with `--quantile-scope` if needed.
- The backtest is a vectorized approximation meant for quick signal sanity checks, not a full execution-quality simulation.
- The backtest alpha factor is a 1-day rolling z-score of the trading signal per symbol (min 60 minutes, both scaled to bars), scaled by 0.01 and averaged by timestamp for plotting.
