import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
from freqtrade.strategy import IStrategy
from pandas import DataFrame

# Add parent path so we can import the project's src/ modules.
# In the Docker container, ../src is mounted at /freqtrade/src.
sys.path.insert(0, "/freqtrade")

from src.features import engineer_features
from src.utils import assign_decile_rolling, interval_to_minutes
from src.data_io import load_frame

logger = logging.getLogger(__name__)

_TMP_STALE_SECONDS = 300


class LightGBMStrategy(IStrategy):
    """
    LightGBM long-only strategy for BTC/USDT:USDT on Binance Futures.

    Designed to produce signals identical to the backtest by:
    - Using the same model with the same best_iteration for predictions
    - Using the full prediction history from rolling CV for expanding quantile
    - Using the same entry/exit thresholds with Freqtrade's position manager
      providing natural hysteresis
    """

    INTERFACE_VERSION = 3

    timeframe = "15m"
    startup_candle_count = 50  # feature warmup margin (RSI/CCI need 14, alpha001 needs 20)

    can_short = False  # long-only
    # Note: Freqtrade stoploss uses real-time unrealized P&L from entry price.
    # The backtest stoploss tracks cumulative geometric return since entry.
    # At 1x leverage these are very similar but not identical.
    stoploss = -0.20
    minimal_roi = {"0": 100}  # exits via signals only

    # Paths inside the container (mapped from shared/ volume)
    MODEL_PATH = "/freqtrade/shared/models/latest_model.txt"
    MODEL_INFO_PATH = "/freqtrade/shared/models/model_info.json"
    PRED_HISTORY_PATH = "/freqtrade/shared/models/latest_predictions.feather"

    # Quantile signal parameters (bins=200).
    # Entry: quantile >= 198 (top 1%)
    # Exit:  quantile < 180 (drops below top 10%)
    # Must match backtest: --bins 200 --quantile 198 --exit-quantile 180
    BINS = 200
    ENTRY_QUANTILE = 198   # top 1% (99th percentile)
    EXIT_QUANTILE = 180    # exit when drops below top 10% (200 * 0.90)
    TRAIN_MONTHS = 12      # must match retrainer; used for rolling quantile window
    IC_THRESH = 0.0        # disable signals if last fold's val_ic < this

    # ---- internal state ----
    _model = None
    _model_mtime = 0.0
    _feature_names = None
    _best_iteration = None  # from model_info.json — matches backtest
    _feature_flags = None   # from model_info.json — ensures feature parity
    _no_model_warned = False
    _pred_history = None  # DatetimeIndex Series of historical predictions

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def bot_loop_start(self, current_time=None, **kwargs):
        self._cleanup_stale_tmp()
        self._maybe_reload_model()

    def _cleanup_stale_tmp(self):
        tmp_path = self.MODEL_PATH + ".tmp"
        try:
            if os.path.exists(tmp_path):
                age = time.time() - os.path.getmtime(tmp_path)
                if age > _TMP_STALE_SECONDS:
                    os.remove(tmp_path)
                    logger.info("Cleaned up stale tmp: %s", tmp_path)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Model + history loading
    # ------------------------------------------------------------------

    def _maybe_reload_model(self):
        if not os.path.exists(self.MODEL_PATH):
            if self._model is None and not self._no_model_warned:
                logger.warning(
                    "LightGBMStrategy: no model at %s — signals disabled.",
                    self.MODEL_PATH,
                )
                self._no_model_warned = True
            return

        try:
            file_size = os.path.getsize(self.MODEL_PATH)
        except OSError:
            return
        if file_size == 0:
            return

        mtime = os.path.getmtime(self.MODEL_PATH)
        if mtime == self._model_mtime:
            return

        try:
            model = lgb.Booster(model_file=self.MODEL_PATH)
            names = model.feature_name()
            if not names:
                logger.error("Model has no feature names; skipping.")
                return

            # Validate model metadata (interval, symbol) before accepting.
            model_info = self._load_model_info()
            if model_info:
                model_interval = model_info.get("interval")
                if model_interval and model_interval != self.timeframe:
                    logger.error(
                        "Model interval %s != strategy timeframe %s — rejecting.",
                        model_interval, self.timeframe,
                    )
                    return
                model_symbol = model_info.get("inference_symbol")
                if model_symbol and model_symbol != "BTCUSDT":
                    logger.error(
                        "Model inference_symbol %s != BTCUSDT — rejecting.",
                        model_symbol,
                    )
                    return
                # IC filter: reject model if last fold's validation IC is too low
                last_ic = model_info.get("last_fold_val_ic")
                if last_ic is not None and self.IC_THRESH is not None:
                    if float(last_ic) < self.IC_THRESH:
                        logger.warning(
                            "Model last_fold_val_ic=%.4f < IC_THRESH=%.4f — signals disabled.",
                            float(last_ic), self.IC_THRESH,
                        )
                        return

            best_iter = model_info.get("best_iteration") if model_info else None
            if best_iter is not None:
                best_iter = int(best_iter)

            self._model = model
            self._feature_names = names
            self._best_iteration = best_iter
            self._feature_flags = model_info.get("feature_flags") if model_info else None
            self._no_model_warned = False
            logger.info(
                "Loaded model (mtime %s, %d features, best_iter=%s)",
                datetime.fromtimestamp(mtime).isoformat(),
                len(names),
                best_iter,
            )
            self._load_prediction_history()
            # Only latch mtime after EVERYTHING succeeds (model + history).
            # If history load disabled the model, don't latch — allow retry.
            if self._model is not None:
                self._model_mtime = mtime
            else:
                logger.warning("Model disabled by history check — will retry on next loop.")
        except Exception:
            logger.exception("Failed to load model — keeping previous")

    def _load_model_info(self):
        """Load full model_info.json dict for validation and best_iteration."""
        if not os.path.exists(self.MODEL_INFO_PATH):
            return None
        try:
            with open(self.MODEL_INFO_PATH) as fh:
                return json.load(fh)
        except Exception:
            logger.exception("Could not read model_info.json")
        return None

    def _load_prediction_history(self):
        """Load historical predictions from rolling CV folds.

        These are the SAME predictions the backtest uses for expanding
        quantile computation. The index is flattened to DatetimeIndex
        (dropping the symbol level) for consistent alignment with
        live predictions.
        """
        if not os.path.exists(self.PRED_HISTORY_PATH):
            logger.error(
                "No prediction history at %s — signals DISABLED until model reload.",
                self.PRED_HISTORY_PATH,
            )
            self._pred_history = None
            self._model = None  # disable signals — live-only quantiles are unreliable
            return

        try:
            hist = load_frame(self.PRED_HISTORY_PATH)
            if "prediction" not in hist.columns:
                logger.error("Prediction history has no 'prediction' column — signals DISABLED.")
                self._pred_history = None
                self._model = None
                return

            pred = hist["prediction"]

            # Filter to inference symbol (BTCUSDT) and flatten to DatetimeIndex.
            # The backtest produces MultiIndex(symbol, timestamp) with BTC+ETH.
            # We must filter to BTC only — mixing symbols would contaminate
            # the expanding quantile distribution.
            if isinstance(pred.index, pd.MultiIndex):
                symbol_level = pred.index.get_level_values("symbol")
                pred = pred[symbol_level == "BTCUSDT"]
                pred = pred.droplevel("symbol").sort_index()

            self._pred_history = pred
            logger.info(
                "Loaded %d historical predictions for expanding quantile.",
                len(pred),
            )
        except Exception:
            logger.exception("Failed to load prediction history.")
            self._pred_history = None

    # ------------------------------------------------------------------
    # Indicators / prediction
    # ------------------------------------------------------------------

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        self._maybe_reload_model()

        if self._model is None:
            dataframe["prediction"] = np.nan
            dataframe["quantile"] = np.nan
            return dataframe

        try:
            dataframe = self._compute_predictions(dataframe)
        except Exception:
            logger.exception(
                "populate_indicators failed for %s — NaN signals",
                metadata.get("pair", "?"),
            )
            dataframe["prediction"] = np.nan
            dataframe["quantile"] = np.nan

        return dataframe

    def _compute_predictions(self, dataframe: DataFrame) -> DataFrame:
        df_indexed = dataframe.set_index("date")

        features_df = engineer_features(
            df_indexed,
            interval="15m",
            bar_type="time",
            feature_flags=self._feature_flags,
        )

        # Build prediction matrix in model's expected column order.
        model_features = self._feature_names
        missing = [c for c in model_features if c not in features_df.columns]
        if missing:
            raise ValueError(
                f"Feature schema mismatch: model expects {missing} "
                f"but engineer_features() did not produce them."
            )
        X = features_df[model_features]

        # Use best_iteration to match backtest predictions exactly.
        # The backtest does: model.predict(X_test, num_iteration=best_iter)
        if self._best_iteration is not None:
            predictions = self._model.predict(
                X.values, num_iteration=self._best_iteration
            )
        else:
            predictions = self._model.predict(X.values)

        pred_series = pd.Series(predictions, index=features_df.index)

        # Rolling quantile: rank each prediction against the last
        # TRAIN_MONTHS worth of predictions, matching the backtest.
        bars_per_month = int(30.4375 * 24 * 60 / interval_to_minutes(self.timeframe))
        q_window = bars_per_month * self.TRAIN_MONTHS

        if self._pred_history is not None:
            earliest = pred_series.index.min()
            # Normalize both sides to naive UTC for reliable comparison.
            if earliest.tzinfo is not None:
                earliest = earliest.tz_convert("UTC").tz_localize(None)
            hist_idx = self._pred_history.index
            if hist_idx.tzinfo is not None:
                hist_idx = hist_idx.tz_convert("UTC").tz_localize(None)
            # Only prepend enough history for the rolling window
            hist = self._pred_history[hist_idx < earliest].tail(q_window)
            combined = pd.concat([hist, pred_series])
            quantiles_full = assign_decile_rolling(combined, bins=self.BINS, window=q_window)
            quantiles = quantiles_full.reindex(pred_series.index)
        else:
            quantiles = assign_decile_rolling(pred_series, bins=self.BINS, window=q_window)

        dataframe["prediction"] = pred_series.reindex(df_indexed.index).values
        dataframe["quantile"] = quantiles.reindex(df_indexed.index).values

        return dataframe

    # ------------------------------------------------------------------
    # Entry / exit signals (long-only)
    # ------------------------------------------------------------------

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Enter long when quantile >= ENTRY_QUANTILE.
        # Matches backtest's _hysteresis_signal: val >= entry_q.
        # Blocked during: grace period (first hour of month) and month-end bars.
        # Month-end block matches backtest's `continue` after forced close.
        is_grace = (dataframe["date"].dt.day == 1) & (dataframe["date"].dt.hour == 0)
        is_month_end = (
            (dataframe["date"] + pd.Timedelta(self.timeframe)).dt.month
            != dataframe["date"].dt.month
        )
        dataframe.loc[
            (dataframe["quantile"] >= self.ENTRY_QUANTILE) & ~is_grace & ~is_month_end,
            "enter_long",
        ] = 1
        # Suppress shorts: Freqtrade 2025.7 futures mode converts exit_long
        # to enter_short when flat — this prevents that.
        dataframe["enter_short"] = 0
        # Clear signals on the last row (current unclosed candle) so the bot
        # only acts on fully-closed candle data, matching the backtest.
        dataframe.loc[dataframe.index[-1:], "enter_long"] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit long when quantile < 180 (drops below top 10%).
        dataframe.loc[
            dataframe["quantile"] < self.EXIT_QUANTILE,
            "exit_long",
        ] = 1

        # Force close on last bar of month (before model retraining).
        is_month_end = (
            (dataframe["date"] + pd.Timedelta(self.timeframe)).dt.month
            != dataframe["date"].dt.month
        )
        dataframe.loc[is_month_end, "exit_long"] = 1

        dataframe["exit_short"] = 0
        # Clear signals on the last row (current unclosed candle).
        dataframe.loc[dataframe.index[-1:], "exit_long"] = 0
        return dataframe
