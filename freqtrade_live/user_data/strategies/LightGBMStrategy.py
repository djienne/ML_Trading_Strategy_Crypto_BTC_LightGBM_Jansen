import json
import logging
import os
import sys
import time

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
    compute_live_level_signals,
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
    - publishing level-based entry/exit flags (enter while desired==1, exit
      while desired==0) so missed fills retry on the next candle and the held
      bars satisfy position[T+1] = desired[T]
    - keeping startup_candle_count large enough that the kline window always
      covers the last month boundary, the state machine's reset point
    """

    INTERFACE_VERSION = 3

    timeframe = "15m"
    # The hysteresis state machine resets at every month end, so its state at
    # any candle only depends on quantiles since the last month boundary. The
    # kline window freqtrade provides in dry-run/live is
    # ohlcv_candle_limit (1000 on Binance) + startup_candle_count, and the
    # state machine restarts flat at the start of that window - so the window
    # MUST always reach back past the last month boundary or live positions can
    # silently desync from the backtest (missed exits, missed month-end
    # force-flat). Budget: 2976 bars (31-day month) + 480 (alpha001 rank
    # window) + ~44 (alpha001 warmup chain) + margin = 3600. This needs 4
    # OHLCV startup calls, below freqtrade's hard cap of 5.
    startup_candle_count = 3600

    can_short = False  # long-only
    # Default until a model contract is loaded. The published contract owns the live value.
    stoploss = -0.20
    minimal_roi = {"0": 100}  # exits via signals only

    # Paths inside the container (mapped from shared/ volume)
    MODELS_DIR = "/freqtrade/shared/models"
    ARCHIVE_DIR = "/freqtrade/shared/models/archive"
    # Atomic pointer naming the archive snapshot to load. Preferred over the flat
    # files below, which remain as a fallback for pre-pointer deployments.
    CURRENT_POINTER_PATH = "/freqtrade/shared/models/current"
    MODEL_PATH = "/freqtrade/shared/models/latest_model.txt"
    MODEL_INFO_PATH = "/freqtrade/shared/models/model_info.json"
    PRED_HISTORY_PATH = "/freqtrade/shared/models/latest_predictions.feather"
    IC_THRESH = None  # no IC filtering (grid search winner uses no_filt)
    # Per-candle record of what the bot actually computed on each decision bar,
    # so live behavior can be diffed exactly against the replay tool later.
    SIGNAL_LOG_PATH = "/freqtrade/user_data/logs/signal_state.csv"

    # ---- internal state ----
    _model = None
    _model_version = None  # pointer stamp or flat-mtime key; reload when it changes
    _feature_names = None
    _best_iteration = None  # from model_info.json; matches backtest
    _feature_flags = None   # from model_info.json; ensures feature parity
    _contract = None
    _no_model_warned = False
    _pred_history = None  # DatetimeIndex Series of historical predictions
    _last_logged_candle = None  # last decision bar written to SIGNAL_LOG_PATH

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

    def _resolve_artifacts(self):
        """Return (model_path, info_path, pred_path, version) for the snapshot to
        load, or None if no model is available.

        Prefers the atomic ``current`` pointer -> ``archive/<stamp>/`` snapshot
        (a fully-consistent {model, info, preds} set); falls back to the legacy
        flat files keyed by model mtime so a pre-pointer deployment keeps working.
        """
        try:
            if os.path.exists(self.CURRENT_POINTER_PATH):
                with open(self.CURRENT_POINTER_PATH, encoding="utf-8") as fh:
                    stamp = fh.read().strip()
                if stamp:
                    snap = os.path.join(self.ARCHIVE_DIR, stamp)
                    model_p = os.path.join(snap, "latest_model.txt")
                    info_p = os.path.join(snap, "model_info.json")
                    pred_p = os.path.join(snap, "latest_predictions.feather")
                    if (
                        os.path.exists(model_p)
                        and os.path.exists(info_p)
                        and os.path.exists(pred_p)
                    ):
                        return model_p, info_p, pred_p, "ptr:" + stamp
                    logger.warning(
                        "current pointer names %s but its snapshot is incomplete; "
                        "falling back to flat files.",
                        stamp,
                    )
        except OSError:
            logger.exception("Could not read current pointer; using flat files.")

        if os.path.exists(self.MODEL_PATH):
            try:
                mtime = os.path.getmtime(self.MODEL_PATH)
            except OSError:
                return None
            return (
                self.MODEL_PATH,
                self.MODEL_INFO_PATH,
                self.PRED_HISTORY_PATH,
                "mtime:%s" % mtime,
            )
        return None

    def _maybe_reload_model(self):
        resolved = self._resolve_artifacts()
        if resolved is None:
            if self._model is None and not self._no_model_warned:
                logger.warning(
                    "LightGBMStrategy: no model artifacts under %s; signals disabled.",
                    self.MODELS_DIR,
                )
                self._no_model_warned = True
            return

        model_path, info_path, pred_path, version = resolved
        if version == self._model_version:
            return

        try:
            file_size = os.path.getsize(model_path)
        except OSError:
            return
        if file_size == 0:
            return

        try:
            model = lgb.Booster(model_file=model_path)
            names = model.feature_name()
            if not names:
                logger.error("Model has no feature names; skipping.")
                return

            model_info = self._load_model_info(info_path)
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
                "Loaded model (version %s, %d features, best_iter=%s)",
                version,
                len(names),
                self._best_iteration,
            )
            self._load_prediction_history(pred_path)
            # Only latch the version after everything succeeds (model + history).
            if self._model is not None:
                self._model_version = version
            else:
                logger.warning("Model disabled by history check; will retry on next loop.")
        except Exception:
            logger.exception("Failed to load model; keeping previous")

    def _load_model_info(self, path):
        """Load full model_info.json dict for contract validation."""
        if not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            logger.exception("Could not read model_info.json")
        return None

    def _load_prediction_history(self, path):
        """Load historical predictions from rolling CV folds.

        These are the same predictions the offline backtest uses to anchor
        rolling quantiles before fresh live bars are appended.
        """
        if not os.path.exists(path):
            logger.error(
                "No prediction history at %s; signals disabled until model reload.",
                path,
            )
            self._pred_history = None
            self._model = None
            return

        try:
            hist = load_frame(path)
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

        if len(pred_series):
            last_ts = pred_series.index[-1]
            self._log_signal_state(
                last_ts,
                pred_series.iloc[-1],
                quantiles.iloc[-1],
                desired_position.iloc[-1],
            )

        return dataframe

    def _log_signal_state(self, ts, prediction, quantile, desired):
        """Append the decision bar's computed state to SIGNAL_LOG_PATH."""
        if ts == self._last_logged_candle:
            return
        try:
            os.makedirs(os.path.dirname(self.SIGNAL_LOG_PATH), exist_ok=True)
            new_file = not os.path.exists(self.SIGNAL_LOG_PATH)
            with open(self.SIGNAL_LOG_PATH, "a", encoding="utf-8") as fh:
                if new_file:
                    fh.write("date,model_version,prediction,quantile,desired_position\n")
                fh.write(
                    "%s,%s,%.12g,%s,%s\n"
                    % (ts, self._model_version, prediction, quantile, desired)
                )
            self._last_logged_candle = ts
        except OSError:
            logger.exception("Could not append signal state log.")

    # ------------------------------------------------------------------
    # Entry / exit signals (long-only)
    # ------------------------------------------------------------------

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Level-based flags: enter while desired==1, exit while desired==0.
        # Freqtrade's own trade state decides which flag matters, so a missed
        # fill is retried on the next candle instead of being lost the way an
        # edge-triggered (transition-only) signal would be.
        desired_position = dataframe.get("desired_position", pd.Series(np.nan, index=dataframe.index))
        entry_signal, _exit_signal = compute_live_level_signals(desired_position)
        dataframe["enter_long"] = 0
        dataframe.loc[entry_signal, "enter_long"] = 1
        # Suppress shorts: Freqtrade 2025.7 futures mode converts exit_long
        # to enter_short when flat; this prevents that.
        dataframe["enter_short"] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        desired_position = dataframe.get("desired_position", pd.Series(np.nan, index=dataframe.index))
        _entry_signal, exit_signal = compute_live_level_signals(desired_position)
        dataframe["exit_long"] = 0
        dataframe.loc[exit_signal, "exit_long"] = 1
        dataframe["exit_short"] = 0
        return dataframe
