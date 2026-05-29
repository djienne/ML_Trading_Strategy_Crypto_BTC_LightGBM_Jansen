"""
LightGBM model retrainer for the freqtrade_live deployment.

Uses the EXACT SAME training code as the backtest (src/modeling.py)
to guarantee identical models and signals.  The last fold's model
from the rolling cross-validation is deployed as the live model.

Lifecycle
---------
1. On startup: clean up orphaned .tmp files, then full retrain only if
   the published model/history are missing, stale, or the training code changed.
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
from src.data_io import load_data, load_data_multi, load_frame, save_frame, select_symbol
from src.modeling import train_and_predict
from src.strategy_contract import build_strategy_contract
from src.utils import get_time_index, interval_to_minutes

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
ARCHIVE_DIR = os.path.join(MODEL_DIR, "archive")
# Atomic pointer naming the archive snapshot the trader should load. Flipping
# this file (single os.replace) is the publish commit; the snapshot dir it names
# is immutable, so the trader always sees a consistent {model,info,preds} set.
CURRENT_POINTER_PATH = os.path.join(MODEL_DIR, "current")
DOWNLOAD_CONFIG_PATH = "/app/_download_config.json"
TRAINING_SOURCE_PATHS = (
    "/app/src/features.py",
    "/app/src/modeling.py",
    "/app/src/utils.py",
)

# Model hyperparams — must match the deployed grid search winner.
BOOST_ROUNDS = 5000
TRAIN_MONTHS = 12
NUM_LEAVES = 31
MIN_DATA_IN_LEAF = 50
FEATURE_FRACTION = 0.5
LEARNING_RATE = 0.01
MIN_TRAINING_ROWS = 10_000
BINS = 100
ENTRY_QUANTILE = 100
EXIT_QUANTILE = 90
DIRECTION = "high"
STOPLOSS = -0.20
FEE_ASSUMPTION = 0.0005
QUANTILE_METHOD = "rolling"

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
TRAIN_HEARTBEAT_SECONDS = 30


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


def _training_progress(message):
    logger.info("[train] %s", message)


def _archive_stamp(training_date: str) -> str:
    ts = datetime.fromisoformat(training_date)
    ts_utc = ts.astimezone(timezone.utc).replace(tzinfo=timezone.utc)
    return ts_utc.strftime("%Y-%m-%dT%H-%M-%SZ")


def _archive_snapshot(training_date: str, source: str, model_info: dict | None = None) -> bool:
    snapshot_dir = os.path.join(ARCHIVE_DIR, _archive_stamp(training_date))
    if os.path.isdir(snapshot_dir):
        return False

    required = [MODEL_PATH, MODEL_INFO_PATH, PREDICTIONS_PATH]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        logger.warning("Skipping archive snapshot; missing artifacts: %s", missing)
        return False

    model_info = model_info or {}
    tmp_dir = snapshot_dir + ".tmp"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        shutil.copy2(MODEL_PATH, os.path.join(tmp_dir, "latest_model.txt"))
        shutil.copy2(MODEL_INFO_PATH, os.path.join(tmp_dir, "model_info.json"))
        shutil.copy2(PREDICTIONS_PATH, os.path.join(tmp_dir, "latest_predictions.feather"))
        archive_meta = {
            "archived_at": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "training_date": training_date,
            "interval": model_info.get("interval", INTERVAL),
            "inference_symbol": model_info.get("inference_symbol", INFERENCE_SYMBOL),
        }
        with open(os.path.join(tmp_dir, "archive_meta.json"), "w") as fh:
            json.dump(archive_meta, fh, indent=2)
        os.makedirs(ARCHIVE_DIR, exist_ok=True)
        shutil.move(tmp_dir, snapshot_dir)
        logger.info("Archived live artifacts -> %s", snapshot_dir)
        return True
    except Exception:
        logger.exception("Failed to archive live artifacts to %s", snapshot_dir)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return False


def _write_current_pointer(stamp: str) -> None:
    """Atomically flip the `current` pointer to an archive snapshot name."""
    tmp = CURRENT_POINTER_PATH + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(stamp)
        os.replace(tmp, CURRENT_POINTER_PATH)
        logger.info("Flipped current pointer -> %s", stamp)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _archive_current_if_needed():
    if not os.path.exists(MODEL_INFO_PATH):
        return False
    try:
        with open(MODEL_INFO_PATH) as fh:
            info = json.load(fh)
        training_date = info.get("training_date")
        if not training_date:
            logger.warning("Current model_info.json has no training_date; skipping archive seed.")
            return False
        return _archive_snapshot(training_date, source="startup_existing", model_info=info)
    except Exception:
        logger.exception("Could not archive current live artifacts.")
        return False


def _expected_model_config() -> dict:
    return {
        "train_symbols": TRAIN_SYMBOLS,
        "inference_symbol": INFERENCE_SYMBOL,
        "interval": INTERVAL,
        "boost_rounds": BOOST_ROUNDS,
        "train_months": TRAIN_MONTHS,
        "num_leaves": NUM_LEAVES,
        "min_data_in_leaf": MIN_DATA_IN_LEAF,
        "feature_fraction": FEATURE_FRACTION,
        "learning_rate": LEARNING_RATE,
        "feature_flags": FEATURE_FLAGS,
        "bins": BINS,
        "entry_quantile": ENTRY_QUANTILE,
        "exit_quantile": EXIT_QUANTILE,
        "direction": DIRECTION,
        "stoploss": STOPLOSS,
        "fee_assumption": FEE_ASSUMPTION,
        "quantile_method": QUANTILE_METHOD,
    }


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
    When last_fold_only=True, only the last fold is trained; the retrainer
    currently always trains all folds (both startup and monthly run the full
    pipeline), so this stays False in production.
    """
    # Clear old folds to prevent stale models from being promoted
    if os.path.isdir(FOLD_DIR):
        shutil.rmtree(FOLD_DIR)
    os.makedirs(FOLD_DIR, exist_ok=True)
    logger.info(
        "Starting rolling CV training via src.modeling.train_and_predict "
        "(last_fold_only=%s, heartbeat=%ss).",
        last_fold_only,
        TRAIN_HEARTBEAT_SECONDS,
    )

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
        progress=_training_progress,
        heartbeat_seconds=TRAIN_HEARTBEAT_SECONDS,
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


def save_model(last_fold_path, feature_names, val_ic, last_fold_val_ic, best_iter, num_folds, predictions):
    """Copy last fold model as latest_model.txt and save metadata.

    Order matters: predictions and metadata are saved BEFORE the model file,
    because the live strategy triggers a reload when latest_model.txt changes.
    Predictions must already be on disk when that reload happens.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Save predictions FIRST (before model triggers strategy reload)
    if predictions is not None and not predictions.empty:
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
        "num_leaves": NUM_LEAVES,
        "min_data_in_leaf": MIN_DATA_IN_LEAF,
        "feature_fraction": FEATURE_FRACTION,
        "learning_rate": LEARNING_RATE,
        "feature_flags": FEATURE_FLAGS,
        "training_method": "rolling_cv (src/modeling.train_and_predict)",
    }
    info.update(
        build_strategy_contract(
            interval=INTERVAL,
            inference_symbol=INFERENCE_SYMBOL,
            feature_flags=FEATURE_FLAGS,
            best_iteration=best_iter,
            train_months=TRAIN_MONTHS,
            bins=BINS,
            entry_quantile=ENTRY_QUANTILE,
            exit_quantile=EXIT_QUANTILE,
            direction=DIRECTION,
            stoploss=STOPLOSS,
            fee_assumption=FEE_ASSUMPTION,
            quantile_method=QUANTILE_METHOD,
        )
    )
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

    # Verify — remove corrupted model and abort if check fails
    try:
        check = lgb.Booster(model_file=MODEL_PATH)
        if not check.feature_name():
            os.remove(MODEL_PATH)
            raise RuntimeError("Post-save check: model has no features!")
    except lgb.basic.LightGBMError:
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        raise RuntimeError("Post-save check: saved model file is unreadable!")

    _archive_snapshot(info["training_date"], source="publish", model_info=info)

    # Commit: flip the `current` pointer to the immutable snapshot. The trader
    # reloads when this stamp changes; the flat files written above remain as a
    # fallback for readers that predate the pointer.
    stamp = _archive_stamp(info["training_date"])
    if os.path.isdir(os.path.join(ARCHIVE_DIR, stamp)):
        _write_current_pointer(stamp)
    else:
        logger.warning(
            "Archive snapshot %s missing; current pointer left unchanged "
            "(trader falls back to flat latest_model.txt).",
            stamp,
        )
    logger.info("Saved predictions -> model -> info -> current: %s", stamp)


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
        bars_per_month = int(30.4375 * 24 * 60 / interval_to_minutes(INTERVAL))
        min_rows = bars_per_month * TRAIN_MONTHS
        if len(preds) < min_rows:
            logger.warning(
                "Prediction history too sparse: %d rows < %d minimum.",
                len(preds), min_rows,
            )
            return False
        # Check data freshness — stale predictions trigger full retrain
        newest = pd.to_datetime(get_time_index(preds.index)).max()
        if getattr(newest, "tzinfo", None) is not None:
            newest = newest.tz_convert("UTC").tz_localize(None)
        now_utc = pd.Timestamp.now(tz="UTC").tz_localize(None)
        age_days = (now_utc - pd.Timestamp(newest)).days
        if age_days > 7:
            logger.warning("Prediction history too stale: newest is %d days old.", age_days)
            return False
        logger.info("Prediction history OK: %d rows (need >= %d), %d days old.", len(preds), min_rows, age_days)
        return True
    except Exception:
        logger.exception("Could not validate prediction history.")
        return False


def _load_last_train_month():
    if not os.path.exists(MODEL_INFO_PATH):
        return None
    try:
        with open(MODEL_INFO_PATH) as fh:
            info = json.load(fh)
        return datetime.fromisoformat(info["training_date"]).strftime("%Y-%m")
    except Exception:
        logger.exception("Could not read last training month from model_info.json")
        return None


def _load_current_model_info():
    if not os.path.exists(MODEL_INFO_PATH):
        return None
    try:
        with open(MODEL_INFO_PATH) as fh:
            return json.load(fh)
    except Exception:
        logger.exception("Could not read model_info.json")
        return None


def _training_config_changed(model_info: dict | None):
    if not model_info:
        return True

    expected = _expected_model_config()
    for key, value in expected.items():
        if model_info.get(key) != value:
            logger.info(
                "Published model config mismatch for %s: have=%r expected=%r; full retrain required.",
                key,
                model_info.get(key),
                value,
            )
            return True
    return False


def _training_sources_changed():
    if not os.path.exists(MODEL_PATH):
        return True
    try:
        model_mtime = os.path.getmtime(MODEL_PATH)
    except OSError:
        return True

    for path in TRAINING_SOURCE_PATHS:
        try:
            source_mtime = os.path.getmtime(path)
        except OSError:
            logger.warning("Could not stat training dependency %s; forcing full retrain.", path)
            return True
        if source_mtime > model_mtime:
            logger.info(
                "Training source %s is newer than the published model; full retrain required.",
                path,
            )
            return True
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
    _archive_current_if_needed()

    model_exists = os.path.exists(MODEL_PATH)
    current_model_info = _load_current_model_info() if model_exists else None
    predictions_ok = _predictions_sufficient()
    config_changed = _training_config_changed(current_model_info) if model_exists else True
    sources_changed = _training_sources_changed() if model_exists else True

    if model_exists and predictions_ok and not config_changed and not sources_changed:
        logger.info("Existing model and prediction history are healthy; skipping startup retrain.")
        last_train_month = _load_last_train_month()
    else:
        if not model_exists:
            reason = "no published model"
        elif not predictions_ok:
            reason = "prediction history missing, sparse, or stale"
        elif config_changed:
            reason = "published model config no longer matches retrainer settings"
        else:
            reason = "training source changed since the last model publish"

        logger.info("Startup full retrain required: %s.", reason)
        if run_pipeline():
            last_train_month = datetime.now(timezone.utc).strftime("%Y-%m")
        else:
            logger.error("Startup retrain failed; using existing model if available.")
            last_train_month = _load_last_train_month()

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
