"""
LightGBM model retrainer for the freqtrade_live deployment.

Uses the EXACT SAME training code as the backtest (src/modeling.py)
to guarantee identical models and signals.  The last fold's model
from the rolling cross-validation is deployed as the live model.

Lifecycle
---------
1. On startup: clean up orphaned .tmp files, then train immediately
   if no model exists.
2. Monthly (1st of each month): download latest BTC data, retrain,
   and save the last fold's model to the shared volume.
"""

import glob
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Path setup — src/ and download_data.py are mounted into /app by Docker
# ---------------------------------------------------------------------------
sys.path.insert(0, "/app")

from src.features import engineer_features, prepare_target
from src.data_io import load_data, load_data_multi, save_frame
from src.modeling import train_and_predict

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("retrainer")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Training uses multiple symbols for more data and less overfitting.
# Live trading and backtesting only trade BTCUSDT.
TRAIN_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INFERENCE_SYMBOL = "BTCUSDT"
INTERVAL = "15m"
DATA_DIR = "/app/shared/data/feather"
MODEL_DIR = "/app/shared/models"
FOLD_DIR = os.path.join(MODEL_DIR, "folds")
MODEL_PATH = os.path.join(MODEL_DIR, "latest_model.txt")
MODEL_INFO_PATH = os.path.join(MODEL_DIR, "model_info.json")
PREDICTIONS_PATH = os.path.join(MODEL_DIR, "latest_predictions.feather")
DOWNLOAD_CONFIG_PATH = "/app/_download_config.json"

# Must match src/modeling.py defaults exactly for reproducibility.
BOOST_ROUNDS = 5000
MIN_TRAINING_ROWS = 10_000

# All features enabled (must match config.json feature_flags).
FEATURE_FLAGS = {
    "returns": True,
    "bop": True,
    "cci": True,
    "mfi": True,
    "rsi": True,
    "stochrsi": True,
    "stoch": True,
    "natr": True,
    "alpha054": True,
    "alpha001": True,
}

# Schedule
CHECK_INTERVAL_SECONDS = 3600  # poll once per hour
TRAIN_DAY_OF_MONTH = 1


# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

def cleanup_tmp_files():
    """Remove any orphaned .tmp files from a previous crash."""
    if not os.path.isdir(MODEL_DIR):
        return
    for tmp in glob.glob(os.path.join(MODEL_DIR, "*.tmp")):
        try:
            os.remove(tmp)
            logger.info("Cleaned up stale tmp: %s", tmp)
        except OSError as exc:
            logger.warning("Could not remove %s: %s", tmp, exc)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def download_latest_data():
    """Download candles for all training symbols from Binance."""
    logger.info("Downloading latest data for %s %s ...", TRAIN_SYMBOLS, INTERVAL)

    config = {
        "candle_interval": INTERVAL,
        "request_delay": 0.3,
        "train_symbols": TRAIN_SYMBOLS,
        "start_date": "2020-01-01",
        "feather_dir": DATA_DIR,
        "binance_base_url": "https://fapi.binance.com",
        "max_klines_per_request": 1500,
    }
    os.makedirs(os.path.dirname(DOWNLOAD_CONFIG_PATH) or ".", exist_ok=True)
    with open(DOWNLOAD_CONFIG_PATH, "w") as fh:
        json.dump(config, fh)

    import download_data  # noqa: delay import (mounted at runtime)

    download_data.main(DOWNLOAD_CONFIG_PATH, TRAIN_SYMBOLS)
    logger.info("Download complete.")


def load_and_prepare():
    """Load raw OHLCV for all training symbols, compute features, prepare target."""
    logger.info("Loading data for %s %s ...", TRAIN_SYMBOLS, INTERVAL)
    df = load_data_multi(DATA_DIR, TRAIN_SYMBOLS, INTERVAL)
    if df is None or df.empty:
        logger.error("No data loaded for %s %s", TRAIN_SYMBOLS, INTERVAL)
        return None

    logger.info("Loaded %d rows. Computing features ...", len(df))
    features_df = engineer_features(
        df,
        interval=INTERVAL,
        bar_type="time",
        feature_flags=FEATURE_FLAGS,
    )
    model_data = prepare_target(
        df,
        features_df,
        interval=INTERVAL,
        bar_type="time",
        feature_flags=FEATURE_FLAGS,
    )
    logger.info(
        "Feature matrix ready: %d rows, %d columns.",
        len(model_data),
        len(model_data.columns),
    )
    return model_data


def validate_training_data(model_data):
    """Check data quality before training. Returns True if acceptable."""
    if model_data is None or model_data.empty:
        logger.error("Validation failed: no data.")
        return False

    if len(model_data) < MIN_TRAINING_ROWS:
        logger.error(
            "Validation failed: only %d rows (minimum %d).",
            len(model_data),
            MIN_TRAINING_ROWS,
        )
        return False

    target_col = "fwd1bar"
    if target_col not in model_data.columns:
        logger.error("Validation failed: missing target column '%s'.", target_col)
        return False

    target_finite = np.isfinite(model_data[target_col]).sum()
    if target_finite < MIN_TRAINING_ROWS:
        logger.error(
            "Validation failed: only %d finite target values (minimum %d).",
            target_finite,
            MIN_TRAINING_ROWS,
        )
        return False

    feature_cols = [c for c in model_data.columns if c != target_col]
    for col in feature_cols:
        nan_frac = model_data[col].isna().mean()
        if nan_frac > 0.5:
            logger.error(
                "Validation failed: feature '%s' has %.0f%% NaN values.",
                col,
                nan_frac * 100,
            )
            return False

    logger.info("Data validation passed (%d rows, %d features).", len(model_data), len(feature_cols))
    return True


def train_model(model_data):
    """Train using the SAME rolling-CV code as the backtest.

    Calls src/modeling.train_and_predict() which uses:
      - 12-month train / 1-month test rolling window
      - 90/10 train/val split within each fold
      - early stopping at 50 rounds, max 250 rounds
      - seed=42 for reproducibility
      - lookahead=20

    The last fold's model is used as the live model.
    """
    os.makedirs(FOLD_DIR, exist_ok=True)

    # Use the exact same function as `python main.py train`
    predictions = train_and_predict(
        model_data,
        interval=INTERVAL,
        bar_type="time",
        model_dir=FOLD_DIR,
        resume=False,
        boost_rounds=BOOST_ROUNDS,
    )

    # Find the last fold model — this is the one trained on the most
    # recent 12-month window, which is what we want for live trading.
    fold_files = sorted(glob.glob(os.path.join(FOLD_DIR, "fold_*.txt")))
    if not fold_files:
        raise RuntimeError("No fold models produced by train_and_predict.")

    last_fold_path = fold_files[-1]
    last_fold_num = len(fold_files)

    # Load the last fold model to get metadata
    model = lgb.Booster(model_file=last_fold_path)
    feature_names = model.feature_name()
    best_iter = model.best_iteration if model.best_iteration else model.current_iteration()

    # Compute validation IC for the last fold's predictions
    if not predictions.empty and "target" in predictions.columns and "prediction" in predictions.columns:
        val_ic = spearmanr(predictions["target"], predictions["prediction"])[0]
        if np.isnan(val_ic):
            val_ic = 0.0
    else:
        val_ic = 0.0

    logger.info(
        "Rolling CV complete: %d folds. Using last fold (%s) as live model. "
        "best_iter=%d, overall_IC=%.4f",
        last_fold_num,
        os.path.basename(last_fold_path),
        best_iter,
        val_ic,
    )

    return last_fold_path, feature_names, val_ic, best_iter, last_fold_num, predictions


def save_model(last_fold_path, feature_names, val_ic, best_iter, num_folds, predictions):
    """Copy last fold model as latest_model.txt and save metadata."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Atomic copy: last fold -> latest_model.txt
    tmp = MODEL_PATH + ".tmp"
    try:
        shutil.copy2(last_fold_path, tmp)
        os.replace(tmp, MODEL_PATH)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise

    # Verify
    try:
        check = lgb.Booster(model_file=MODEL_PATH)
        if not check.feature_name():
            logger.critical("Post-save check: model has no features!")
    except Exception:
        logger.exception("Post-save check: saved model file is unreadable!")

    # Save predictions for later comparison with backtest
    if predictions is not None and not predictions.empty:
        try:
            save_frame(predictions, PREDICTIONS_PATH)
            logger.info("Saved predictions -> %s", PREDICTIONS_PATH)
        except Exception:
            logger.exception("Failed to save predictions.")

    # Metadata (atomic)
    info = {
        "training_date": datetime.now(timezone.utc).isoformat(),
        "train_symbols": TRAIN_SYMBOLS,
        "inference_symbol": INFERENCE_SYMBOL,
        "interval": INTERVAL,
        "feature_names": feature_names,
        "best_iteration": best_iter,
        "overall_ic": float(val_ic),
        "num_folds": num_folds,
        "last_fold_file": os.path.basename(last_fold_path),
        "boost_rounds": BOOST_ROUNDS,
        "training_method": "rolling_cv (src/modeling.train_and_predict)",
    }
    tmp_info = MODEL_INFO_PATH + ".tmp"
    try:
        with open(tmp_info, "w") as fh:
            json.dump(info, fh, indent=2)
        os.replace(tmp_info, MODEL_INFO_PATH)
    except Exception:
        if os.path.exists(tmp_info):
            os.remove(tmp_info)
        raise

    logger.info("Saved model -> %s  info -> %s", MODEL_PATH, MODEL_INFO_PATH)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline():
    """Download -> features -> validate -> train (rolling CV) -> save last fold."""
    try:
        download_latest_data()
        data = load_and_prepare()

        if not validate_training_data(data):
            logger.error("Data validation failed. Keeping existing model.")
            return False

        last_fold, feats, ic, best, n_folds, preds = train_model(data)
        save_model(last_fold, feats, ic, best, n_folds, preds)
        return True
    except Exception:
        logger.exception("Training pipeline failed.")
        return False


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

def should_train_now(last_train_month):
    """True if we haven't trained yet this month and it's >= the 1st."""
    now = datetime.now(timezone.utc)
    key = now.strftime("%Y-%m")
    if last_train_month == key:
        return False
    return now.day >= TRAIN_DAY_OF_MONTH


def main():
    logger.info("=== LightGBM Retrainer started ===")
    logger.info(
        "TrainSymbols=%s  Inference=%s  Interval=%s  BoostRounds=%d  Method=rolling_cv",
        TRAIN_SYMBOLS,
        INFERENCE_SYMBOL,
        INTERVAL,
        BOOST_ROUNDS,
    )

    cleanup_tmp_files()

    last_train_month = None
    if os.path.exists(MODEL_INFO_PATH):
        try:
            with open(MODEL_INFO_PATH) as fh:
                info = json.load(fh)
            dt = datetime.fromisoformat(info["training_date"])
            last_train_month = dt.strftime("%Y-%m")
            logger.info(
                "Existing model trained on %s (month %s, %d folds).",
                info["training_date"],
                last_train_month,
                info.get("num_folds", "?"),
            )
        except Exception:
            logger.warning("Could not read model_info.json; will retrain.")

    if not os.path.exists(MODEL_PATH):
        logger.info("No model found. Training immediately ...")
        if run_pipeline():
            last_train_month = datetime.now(timezone.utc).strftime("%Y-%m")

    while True:
        try:
            if should_train_now(last_train_month):
                logger.info("Monthly retraining triggered.")
                if run_pipeline():
                    last_train_month = datetime.now(timezone.utc).strftime("%Y-%m")
                else:
                    logger.error("Training failed; will retry next hour.")
        except Exception:
            logger.exception("Error in main loop.")

        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
