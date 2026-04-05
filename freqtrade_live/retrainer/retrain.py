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

import gc
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
from src.data_io import load_data, load_data_multi, save_frame, select_symbol
from src.modeling import train_and_predict
from src.utils import get_time_index

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

# Model hyperparams — must match the deployed grid search winner.
BOOST_ROUNDS = 5000
TRAIN_MONTHS = 12
NUM_LEAVES = 31
MIN_DATA_IN_LEAF = 50
FEATURE_FRACTION = 0.5
LEARNING_RATE = 0.01
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

    # Check per-symbol data freshness
    max_staleness_days = 3
    ts = pd.to_datetime(get_time_index(model_data.index))
    now = pd.Timestamp.now(tz="UTC")
    for sym in TRAIN_SYMBOLS:
        sym_data = select_symbol(model_data, sym)
        if sym_data.empty:
            logger.error("Validation failed: no data for %s.", sym)
            return False
        sym_ts = pd.to_datetime(get_time_index(sym_data.index))
        latest = sym_ts.max()
        age_days = (now - pd.Timestamp(latest, tz="UTC")).days
        if age_days > max_staleness_days:
            logger.error(
                "Validation failed: %s data is %d days old (max %d).",
                sym, age_days, max_staleness_days,
            )
            return False

    logger.info("Data validation passed (%d rows, %d features).", len(model_data), len(feature_cols))
    return True


def train_model(model_data, last_fold_only=False):
    """Train using the SAME rolling-CV code as the backtest.

    Calls src/modeling.train_and_predict() with params from module constants:
      - TRAIN_MONTHS-month train / 1-month test rolling window
      - 90/10 train/val split within each fold (sorted by time)
      - early stopping at 50 rounds, max BOOST_ROUNDS rounds
      - Per-symbol embargo of 20 bars
      - seed=42 for reproducibility

    The last fold's model is deployed as the live model.
    When last_fold_only=True, only the last fold is trained (fast startup).
    """
    # Clear old folds to prevent stale models from being promoted
    if os.path.isdir(FOLD_DIR):
        shutil.rmtree(FOLD_DIR)
    os.makedirs(FOLD_DIR, exist_ok=True)

    # Use the exact same function as `python main.py train`
    predictions, train_meta = train_and_predict(
        model_data,
        interval=INTERVAL,
        bar_type="time",
        model_dir=FOLD_DIR,
        resume=False,
        boost_rounds=BOOST_ROUNDS,
        train_months=TRAIN_MONTHS,
        num_leaves=NUM_LEAVES,
        min_data_in_leaf=MIN_DATA_IN_LEAF,
        feature_fraction=FEATURE_FRACTION,
        learning_rate=LEARNING_RATE,
        last_fold_only=last_fold_only,
    )

    # Find the last fold model — this is the one trained on the most
    # recent 12-month window, which is what we want for live trading.
    fold_files = sorted(glob.glob(os.path.join(FOLD_DIR, "fold_*.txt")))
    if not fold_files:
        raise RuntimeError("No fold models produced by train_and_predict.")

    last_fold_path = fold_files[-1]
    last_fold_num = len(fold_files)

    # Use best_iteration from training (not from reloaded model, which
    # loses early-stopping metadata and returns -1).
    model = lgb.Booster(model_file=last_fold_path)
    feature_names = model.feature_name()
    best_iter = train_meta.get("last_best_iteration")
    if best_iter is None:
        best_iter = model.current_iteration()

    # Compute overall IC and last fold's validation IC
    if not predictions.empty and "target" in predictions.columns and "prediction" in predictions.columns:
        val_ic = spearmanr(predictions["target"], predictions["prediction"])[0]
        if np.isnan(val_ic):
            val_ic = 0.0
    else:
        val_ic = 0.0

    # Last fold's val_ic — used by live strategy for IC filtering
    last_fold_val_ic = 0.0
    if not predictions.empty and "val_ic" in predictions.columns:
        last_fold_val_ic = float(predictions["val_ic"].iloc[-1])
        if np.isnan(last_fold_val_ic):
            last_fold_val_ic = 0.0

    logger.info(
        "Rolling CV complete: %d folds. Using last fold (%s) as live model. "
        "best_iter=%d, overall_IC=%.4f",
        last_fold_num,
        os.path.basename(last_fold_path),
        best_iter,
        val_ic,
    )

    return last_fold_path, feature_names, val_ic, last_fold_val_ic, best_iter, last_fold_num, predictions


def save_model(last_fold_path, feature_names, val_ic, last_fold_val_ic, best_iter, num_folds, predictions,
               skip_predictions=False):
    """Copy last fold model as latest_model.txt and save metadata.

    Order matters: predictions and metadata are saved BEFORE the model file,
    because the live strategy triggers a reload when latest_model.txt changes.
    Predictions must already be on disk when that reload happens.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Save predictions FIRST (before model triggers strategy reload)
    #    Skip on startup retrain — keep existing prediction history.
    if not skip_predictions and predictions is not None and not predictions.empty:
        try:
            save_frame(predictions, PREDICTIONS_PATH)
            logger.info("Saved predictions -> %s", PREDICTIONS_PATH)
        except Exception:
            logger.exception("Failed to save predictions — aborting model publish.")
            raise

    # 2. Metadata (atomic)
    info = {
        "training_date": datetime.now(timezone.utc).isoformat(),
        "train_symbols": TRAIN_SYMBOLS,
        "inference_symbol": INFERENCE_SYMBOL,
        "interval": INTERVAL,
        "feature_names": feature_names,
        "best_iteration": best_iter,
        "overall_ic": float(val_ic),
        "last_fold_val_ic": float(last_fold_val_ic),
        "num_folds": num_folds,
        "last_fold_file": os.path.basename(last_fold_path),
        "boost_rounds": BOOST_ROUNDS,
        "train_months": TRAIN_MONTHS,
        "feature_flags": FEATURE_FLAGS,
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

    # 3. Model file LAST — this triggers strategy reload via mtime check
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

    logger.info("Saved predictions -> model -> info: %s", MODEL_PATH)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline():
    """Download -> features -> validate -> train ALL folds -> save model + predictions."""
    try:
        download_latest_data()
        data = load_and_prepare()

        if not validate_training_data(data):
            logger.error("Data validation failed. Keeping existing model.")
            return False

        last_fold, feats, ic, last_ic, best, n_folds, preds = train_model(data)
        save_model(last_fold, feats, ic, last_ic, best, n_folds, preds)
        gc.collect()
        return True
    except Exception:
        logger.exception("Training pipeline failed.")
        return False


def run_startup_retrain():
    """Fast startup retrain: last fold only, keep existing prediction history."""
    try:
        download_latest_data()
        data = load_and_prepare()

        if not validate_training_data(data):
            logger.error("Data validation failed. Keeping existing model.")
            return False

        last_fold, feats, ic, last_ic, best, n_folds, preds = train_model(
            data, last_fold_only=True,
        )
        save_model(last_fold, feats, ic, last_ic, best, n_folds, preds,
                   skip_predictions=True)
        gc.collect()
        return True
    except Exception:
        logger.exception("Startup retrain failed.")
        return False


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

def _predictions_sufficient():
    """Check if existing prediction history has enough data for rolling quantile."""
    if not os.path.exists(PREDICTIONS_PATH):
        return False
    try:
        preds = load_frame(PREDICTIONS_PATH)
        if "prediction" not in preds.columns:
            return False
        # Filter to inference symbol
        if isinstance(preds.index, pd.MultiIndex):
            sym_level = preds.index.get_level_values("symbol")
            preds = preds[sym_level == INFERENCE_SYMBOL]
        # Need at least rolling-window-size rows for the quantile to work
        from src.utils import interval_to_minutes
        bars_per_month = int(30.4375 * 24 * 60 / interval_to_minutes(INTERVAL))
        min_rows = bars_per_month * TRAIN_MONTHS
        if len(preds) < min_rows:
            logger.warning(
                "Prediction history too sparse: %d rows < %d minimum.",
                len(preds), min_rows,
            )
            return False
        logger.info("Prediction history OK: %d rows (need >= %d).", len(preds), min_rows)
        return True
    except Exception:
        logger.exception("Could not validate prediction history.")
        return False


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

    # Remove the old model so the trader disables signals until retrain
    # completes. This prevents trading with a stale model on startup.
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
        logger.info("Removed stale model — trader will wait for fresh retrain.")

    # Always retrain on startup to pick up code/feature changes.
    # Use fast last-fold-only if prediction history has enough data;
    # otherwise do a full retrain to generate/rebuild the prediction history.
    if _predictions_sufficient():
        logger.info("Startup retrain (last fold only) ...")
        ok = run_startup_retrain()
    else:
        logger.info("Insufficient prediction history — full retrain required ...")
        ok = run_pipeline()
    if ok:
        last_train_month = datetime.now(timezone.utc).strftime("%Y-%m")
    else:
        logger.error("Startup retrain failed; using existing model if available.")
        last_train_month = None
        if os.path.exists(MODEL_INFO_PATH):
            try:
                with open(MODEL_INFO_PATH) as fh:
                    info = json.load(fh)
                last_train_month = datetime.fromisoformat(info["training_date"]).strftime("%Y-%m")
            except Exception:
                pass

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
