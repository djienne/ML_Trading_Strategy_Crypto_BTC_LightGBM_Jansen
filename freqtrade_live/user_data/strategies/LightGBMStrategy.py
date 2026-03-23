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
from src.utils import assign_decile_expanding
from src.data_io import load_frame

logger = logging.getLogger(__name__)

_TMP_STALE_SECONDS = 300


class LightGBMStrategy(IStrategy):
    """
    LightGBM long-only strategy for BTC/USDC:USDC on Hyperliquid.

    Designed to produce signals identical to the backtest by:
    - Using the same model with the same best_iteration for predictions
    - Using the full prediction history from rolling CV for expanding quantile
    - Using the same entry/exit thresholds with Freqtrade's position manager
      providing natural hysteresis
    """

    INTERFACE_VERSION = 3

    timeframe = "15m"
    startup_candle_count = 100  # feature warmup only; quantile history from file

    can_short = False  # long-only
    stoploss = -0.10
    minimal_roi = {"0": 100}  # exits via signals only

    # Paths inside the container (mapped from shared/ volume)
    MODEL_PATH = "/freqtrade/shared/models/latest_model.txt"
    MODEL_INFO_PATH = "/freqtrade/shared/models/model_info.json"
    PRED_HISTORY_PATH = "/freqtrade/shared/models/latest_predictions.feather"

    # Quantile signal parameters (bins=200).
    # Entry: top bin (quantile >= 200, top 0.5%)
    # Exit:  quantile < 170 (drops below top 15%)
    # Backtest: +207% net, 596 trades, Sharpe 1.65 over 5 years
    BINS = 200
    ENTRY_QUANTILE = 200   # top bin
    EXIT_QUANTILE = 170    # exit when drops below top 15% (200 * 0.85)

    # ---- internal state ----
    _model = None
    _model_mtime = 0.0
    _feature_names = None
    _best_iteration = None  # from model_info.json — matches backtest
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

            # Load best_iteration from model_info.json to match backtest.
            best_iter = self._load_best_iteration()

            self._model = model
            self._feature_names = names
            self._best_iteration = best_iter
            self._model_mtime = mtime
            self._no_model_warned = False
            logger.info(
                "Loaded model (mtime %s, %d features, best_iter=%s)",
                datetime.fromtimestamp(mtime).isoformat(),
                len(names),
                best_iter,
            )
            self._load_prediction_history()
        except Exception:
            logger.exception("Failed to load model — keeping previous")

    def _load_best_iteration(self):
        """Read best_iteration from model_info.json.

        The backtest uses model.predict(X, num_iteration=best_iter).
        We must do the same to get identical predictions.
        """
        if not os.path.exists(self.MODEL_INFO_PATH):
            return None
        try:
            with open(self.MODEL_INFO_PATH) as fh:
                info = json.load(fh)
            best = info.get("best_iteration")
            if best is not None:
                return int(best)
        except Exception:
            logger.exception("Could not read best_iteration from model_info.json")
        return None

    def _load_prediction_history(self):
        """Load historical predictions from rolling CV folds.

        These are the SAME predictions the backtest uses for expanding
        quantile computation. The index is flattened to DatetimeIndex
        (dropping the symbol level) for consistent alignment with
        live predictions.
        """
        if not os.path.exists(self.PRED_HISTORY_PATH):
            logger.warning(
                "No prediction history at %s — quantile will use live data only.",
                self.PRED_HISTORY_PATH,
            )
            self._pred_history = None
            return

        try:
            hist = load_frame(self.PRED_HISTORY_PATH)
            if "prediction" not in hist.columns:
                logger.warning("Prediction history has no 'prediction' column.")
                self._pred_history = None
                return

            pred = hist["prediction"]

            # Flatten MultiIndex to DatetimeIndex if needed.
            # The backtest produces MultiIndex(symbol, timestamp).
            # The live strategy predicts with a plain DatetimeIndex.
            # Both must use the same index type for pd.concat to work.
            if isinstance(pred.index, pd.MultiIndex):
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
        )

        # Build prediction matrix in model's expected column order.
        model_features = self._feature_names
        X = pd.DataFrame(index=features_df.index)
        for col in model_features:
            X[col] = features_df[col] if col in features_df.columns else np.nan

        # Use best_iteration to match backtest predictions exactly.
        # The backtest does: model.predict(X_test, num_iteration=best_iter)
        if self._best_iteration is not None:
            predictions = self._model.predict(
                X.values, num_iteration=self._best_iteration
            )
        else:
            predictions = self._model.predict(X.values)

        pred_series = pd.Series(predictions, index=features_df.index)

        # Expanding quantile with full historical context.
        # Prepend prediction history from rolling CV so quantile values
        # match the backtest exactly.
        if self._pred_history is not None:
            earliest = pred_series.index.min()
            hist = self._pred_history[self._pred_history.index < earliest]
            combined = pd.concat([hist, pred_series])
            quantiles_full = assign_decile_expanding(combined, bins=self.BINS)
            quantiles = quantiles_full.reindex(pred_series.index)
        else:
            quantiles = assign_decile_expanding(pred_series, bins=self.BINS)

        dataframe["prediction"] = pred_series.reindex(df_indexed.index).values
        dataframe["quantile"] = quantiles.reindex(df_indexed.index).values

        return dataframe

    # ------------------------------------------------------------------
    # Entry / exit signals (long-only)
    # ------------------------------------------------------------------

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Enter long when quantile >= ENTRY_QUANTILE (top bin).
        # Matches backtest's _hysteresis_signal: val >= entry_q.
        # Grace period: no entries during first hour of month (model retraining).
        is_grace = (dataframe["date"].dt.day == 1) & (dataframe["date"].dt.hour == 0)
        dataframe.loc[
            (dataframe["quantile"] >= self.ENTRY_QUANTILE) & ~is_grace,
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit long when quantile < 170 (drops below top 15%).
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

        return dataframe
