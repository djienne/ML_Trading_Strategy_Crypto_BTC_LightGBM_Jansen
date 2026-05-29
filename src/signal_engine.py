import numpy as np
import pandas as pd

from src.utils import assign_decile_expanding, assign_decile_rolling, interval_to_minutes

try:
    from numba import njit as _njit
except ImportError:
    def _njit(fn=None, **kwargs):
        if fn is not None:
            return fn
        return lambda f: f


@_njit(cache=True)
def _hysteresis_signal(q_values, entry_q, exit_q, is_grace=None, is_month_end=None):
    n = len(q_values)
    out = np.zeros(n, dtype=np.int64)
    pos = 0
    for i in range(n):
        if is_month_end is not None and is_month_end[i] and pos == 1:
            pos = 0
            out[i] = pos
            continue

        if is_grace is not None and is_grace[i]:
            out[i] = pos
            continue

        val = q_values[i]
        if pos == 0 and val >= entry_q:
            pos = 1
        elif pos == 1 and val < exit_q:
            pos = 0
        out[i] = pos
    return out


@_njit(cache=True)
def _hysteresis_signal_low(q_values, entry_q, exit_q, is_grace=None, is_month_end=None):
    n = len(q_values)
    out = np.zeros(n, dtype=np.int64)
    pos = 0
    for i in range(n):
        if is_month_end is not None and is_month_end[i] and pos == 1:
            pos = 0
            out[i] = pos
            continue
        if is_grace is not None and is_grace[i]:
            out[i] = pos
            continue
        val = q_values[i]
        if pos == 0 and val <= entry_q:
            pos = 1
        elif pos == 1 and val > exit_q:
            pos = 0
        out[i] = pos
    return out


def quantile_window_bars(interval, train_months):
    bars_per_month = int(30.4375 * 24 * 60 / interval_to_minutes(interval))
    return bars_per_month * train_months


def _normalize_datetime_index(index):
    idx = pd.DatetimeIndex(pd.to_datetime(index))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx


def compute_prediction_quantiles(
    pred_series,
    bins,
    *,
    interval=None,
    train_months=None,
    window=None,
    quantile_method="rolling",
    history=None,
):
    if not isinstance(pred_series, pd.Series):
        pred_series = pd.Series(pred_series)

    quantile_method = quantile_method.lower()
    if quantile_method == "expanding":
        return assign_decile_expanding(pred_series, bins=bins), None

    if quantile_method != "rolling":
        raise ValueError(f"Unsupported quantile_method: {quantile_method}")

    if window is None:
        if interval is None or train_months is None:
            raise ValueError("rolling quantiles require interval+train_months or window")
        window = quantile_window_bars(interval, train_months)

    if history is not None:
        hist = pd.Series(history).copy()
        if not hist.empty:
            hist.index = _normalize_datetime_index(hist.index)
            pred = pred_series.copy()
            original_index = pred.index
            normalized_pred_index = _normalize_datetime_index(pred.index)
            pred.index = normalized_pred_index
            earliest = normalized_pred_index.min()
            seeded = pd.concat([hist[hist.index < earliest].tail(window), pred])
            quantiles = assign_decile_rolling(seeded, bins=bins, window=window)
            quantiles = quantiles.reindex(normalized_pred_index)
            quantiles.index = original_index
            return quantiles, window

    return assign_decile_rolling(pred_series, bins=bins, window=window), window


def compute_desired_position(
    quantiles,
    timestamps,
    interval,
    entry_quantile,
    exit_quantile,
    *,
    direction="high",
):
    if not isinstance(quantiles, pd.Series):
        quantiles = pd.Series(quantiles, index=timestamps)

    ts_all = _normalize_datetime_index(timestamps)
    valid = quantiles.notna()
    desired = pd.Series(0, index=quantiles.index, dtype="int64")
    if not valid.any():
        return desired, valid

    ts_valid = ts_all[valid.to_numpy()]
    q_arr = quantiles[valid].astype(int).to_numpy(dtype=np.int64)
    is_grace = np.array((ts_valid.day == 1) & (ts_valid.hour == 0))
    is_month_end = np.array((ts_valid + pd.Timedelta(interval)).month != ts_valid.month)

    if direction == "high":
        desired_valid = _hysteresis_signal(
            q_arr,
            entry_quantile,
            exit_quantile,
            is_grace,
            is_month_end,
        )
    elif direction == "low":
        desired_valid = _hysteresis_signal_low(
            q_arr,
            entry_quantile,
            exit_quantile,
            is_grace,
            is_month_end,
        )
    else:
        raise ValueError(f"Unsupported direction: {direction}")

    desired.loc[valid] = desired_valid
    return desired, valid


def compute_live_transition_signals(desired_position):
    # desired_position[T] is derived from prediction[T], whose target is
    # fwd1bar[T] = the open->close return of candle T+1. So a 0->1 transition at
    # candle T means "hold candle T+1". The entry/exit signal is published on the
    # transition candle T itself; Freqtrade reads it on the just-closed candle and
    # fills at the open of candle T+1 -- a single bar of execution delay, which is
    # exactly the bar the model predicts. No extra shift here (see
    # shift_position_for_execution for the offline mirror of this convention).
    desired = pd.Series(desired_position).fillna(0).astype("int64")
    prev = desired.shift(1, fill_value=0)
    enter_base = (desired == 1) & (prev == 0)
    exit_base = (desired == 0) & (prev == 1)
    return enter_base, exit_base


def shift_position_for_execution(desired_position, live_from=None):
    desired = pd.Series(desired_position).fillna(0).astype("int64")
    if live_from is not None:
        desired = desired.copy()
        desired.loc[desired.index < live_from] = 0
    # 1-bar delay matches the live pipeline: compute_live_transition_signals
    # publishes the signal on the transition candle, Freqtrade then fills at the
    # next bar's open. So desired[T] -> position held on candle T+1, capturing
    # the return the model predicted (fwd1bar[T]).
    return desired.shift(1, fill_value=0).astype("int64")
