import json
import logging
import os
import sys
import time
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from freqtrade.strategy import IStrategy
from pandas import DataFrame

# Add parent path so we can import the project's src/ modules.
# In the Docker container, ../src is mounted at /freqtrade/src.
sys.path.insert(0, "/freqtrade")

from src.data_io import load_frame
from src.features import engineer_features
from src.signal_engine import (
    compute_desired_position,
    compute_live_transition_signals,
    compute_prediction_quantiles,
)
from src.strategy_contract import StrategyContractError, read_strategy_contract

logger = logging.getLogger(__name__)

_TMP_STALE_SECONDS = 300


class LightGBMStrategy(IStrategy):
    """
    LightGBM long-only strategy for BTC/USDT:USDT on Binance Futures.

    Designed to keep signal parity with offline backtests by:
    - using the same trained model with the same best_iteration
    - loading the published strategy contract from model_info.json
    - computing rolling quantiles from the shared prediction history
    - deriving hysteresis state from the same shared signal engine
    """

    INTERFACE_VERSION = 3

    timeframe = "15m"
    startup_candle_count = 50  # feature warmup margin (RSI/CCI need 14, alpha001 needs 20)

    can_short = False  # long-only
    # Default until a model contract is loaded. The published contract owns the live value.
    stoploss = -0.20
    minimal_roi = {"0": 100}  # exits via signals only

    # Paths inside the container (mapped from shared/ volume)
    MODEL_PATH = "/freqtrade/shared/models/latest_model.txt"
    MODEL_INFO_PATH = "/freqtrade/shared/models/model_info.json"
    PRED_HISTORY_PATH = "/freqtrade/shared/models/latest_predictions.feather"
    IC_THRESH = None  # no IC filtering (grid search winner uses no_filt)

    # ---- internal state ----
    _model = None
    _model_mtime = 0.0
    _feature_names = None
    _best_iteration = None  # from model_info.json; matches backtest
    _feature_flags = None   # from model_info.json; ensures feature parity
    _contract = None
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
                    "LightGBMStrategy: no model at %s; signals disabled.",
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

            model_info = self._load_model_info()
            if not model_info:
                logger.error("Missing model_info.json; rejecting model reload.")
                return

            try:
                contract = read_strategy_contract(model_info)
            except StrategyContractError as exc:
                logger.error("Invalid strategy contract in model_info.json: %s", exc)
                return

            if contract["interval"] != self.timeframe:
                logger.error(
                    "Model interval %s != strategy timeframe %s; rejecting.",
                    contract["interval"],
                    self.timeframe,
                )
                return
            if contract["inference_symbol"] != "BTCUSDT":
                logger.error(
                    "Model inference_symbol %s != BTCUSDT; rejecting.",
                    contract["inference_symbol"],
                )
                return

            last_ic = model_info.get("last_fold_val_ic")
            if last_ic is not None and self.IC_THRESH is not None:
                if float(last_ic) < self.IC_THRESH:
                    logger.warning(
                        "Model last_fold_val_ic=%.4f < IC_THRESH=%.4f; signals disabled.",
                        float(last_ic),
                        self.IC_THRESH,
                    )
                    return

            self._model = model
            self._feature_names = names
            self._best_iteration = contract["best_iteration"]
            self._feature_flags = contract["feature_flags"]
            self._contract = contract
            self.stoploss = contract["stoploss"]
            self._no_model_warned = False
            logger.info(
                "Loaded model (mtime %s, %d features, best_iter=%s)",
                datetime.fromtimestamp(mtime).isoformat(),
                len(names),
                self._best_iteration,
            )
            self._load_prediction_history()
            # Only latch mtime after everything succeeds (model + history).
            if self._model is not None:
                self._model_mtime = mtime
            else:
                logger.warning("Model disabled by history check; will retry on next loop.")
        except Exception:
            logger.exception("Failed to load model; keeping previous")

    def _load_model_info(self):
        """Load full model_info.json dict for contract validation."""
        if not os.path.exists(self.MODEL_INFO_PATH):
            return None
        try:
            with open(self.MODEL_INFO_PATH, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            logger.exception("Could not read model_info.json")
        return None

    def _load_prediction_history(self):
        """Load historical predictions from rolling CV folds.

        These are the same predictions the offline backtest uses to anchor
        rolling quantiles before fresh live bars are appended.
        """
        if not os.path.exists(self.PRED_HISTORY_PATH):
            logger.error(
                "No prediction history at %s; signals disabled until model reload.",
                self.PRED_HISTORY_PATH,
            )
            self._pred_history = None
            self._model = None
            return

        try:
            hist = load_frame(self.PRED_HISTORY_PATH)
            if "prediction" not in hist.columns:
                logger.error("Prediction history has no 'prediction' column; signals disabled.")
                self._pred_history = None
                self._model = None
                return

            pred = hist["prediction"]
            if isinstance(pred.index, pd.MultiIndex):
                symbol_level = pred.index.get_level_values("symbol")
                pred = pred[symbol_level == "BTCUSDT"]
                pred = pred.droplevel("symbol").sort_index()

            self._pred_history = pred
            logger.info(
                "Loaded %d historical predictions for rolling quantiles.",
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
            dataframe["desired_position"] = np.nan
            return dataframe

        try:
            dataframe = self._compute_predictions(dataframe)
        except Exception:
            logger.exception(
                "populate_indicators failed for %s; NaN signals",
                metadata.get("pair", "?"),
            )
            dataframe["prediction"] = np.nan
            dataframe["quantile"] = np.nan
            dataframe["desired_position"] = np.nan

        return dataframe

    def _compute_predictions(self, dataframe: DataFrame) -> DataFrame:
        if self._contract is None:
            raise ValueError("Missing published strategy contract.")

        df_indexed = dataframe.set_index("date")

        features_df = engineer_features(
            df_indexed,
            interval=self.timeframe,
            bar_type="time",
            feature_flags=self._feature_flags,
        )

        model_features = self._feature_names
        missing = [c for c in model_features if c not in features_df.columns]
        if missing:
            raise ValueError(
                f"Feature schema mismatch: model expects {missing} "
                f"but engineer_features() did not produce them."
            )
        X = features_df[model_features]

        if self._best_iteration is not None:
            predictions = self._model.predict(
                X.values,
                num_iteration=self._best_iteration,
            )
        else:
            predictions = self._model.predict(X.values)

        pred_series = pd.Series(predictions, index=features_df.index)
        quantiles, q_window = compute_prediction_quantiles(
            pred_series,
            self._contract["bins"],
            interval=self.timeframe,
            train_months=self._contract["train_months"],
            quantile_method=self._contract["quantile_method"],
            history=self._pred_history,
        )
        if self._pred_history is not None and q_window is not None and len(self._pred_history) < q_window:
            logger.warning(
                "Sparse prediction history: %d bars < %d window; quantiles may be less stable.",
                len(self._pred_history),
                q_window,
            )
        desired_position, _valid = compute_desired_position(
            quantiles,
            pred_series.index,
            self.timeframe,
            self._contract["entry_quantile"],
            self._contract["exit_quantile"],
            direction=self._contract["direction"],
        )

        dataframe["prediction"] = pred_series.reindex(df_indexed.index).values
        dataframe["quantile"] = quantiles.reindex(df_indexed.index).values
        dataframe["desired_position"] = desired_position.reindex(df_indexed.index).values

        return dataframe

    # ------------------------------------------------------------------
    # Entry / exit signals (long-only)
    # ------------------------------------------------------------------

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        desired_position = dataframe.get("desired_position", pd.Series(0, index=dataframe.index))
        entry_signal, _exit_signal = compute_live_transition_signals(desired_position)
        dataframe["enter_long"] = 0
        dataframe.loc[entry_signal, "enter_long"] = 1
        # Suppress shorts: Freqtrade 2025.7 futures mode converts exit_long
        # to enter_short when flat; this prevents that.
        dataframe["enter_short"] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        desired_position = dataframe.get("desired_position", pd.Series(0, index=dataframe.index))
        _entry_signal, exit_signal = compute_live_transition_signals(desired_position)
        dataframe["exit_long"] = 0
        dataframe.loc[exit_signal, "exit_long"] = 1
        dataframe["exit_short"] = 0
        return dataframe
