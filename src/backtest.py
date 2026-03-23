import math
import os

import numpy as np
import pandas as pd

from src.evaluation import add_quantile_labels
from src.utils import estimate_bar_minutes, get_symbol_key, interval_to_minutes, resolve_quantile_scope

try:
    from numba import njit as _njit
except ImportError:
    def _njit(fn=None, **kwargs):
        if fn is not None:
            return fn
        return lambda f: f


@_njit(cache=True)
def _hysteresis_signal(q_values, entry_q, exit_q, is_grace=None, is_month_end=None):
    """Numba-accelerated hysteresis signal loop with monthly boundaries.

    If is_grace/is_month_end boolean arrays are provided, enforces:
    - is_grace: no new entries (hold existing position at 0)
    - is_month_end: force close any open position
    """
    n = len(q_values)
    out = np.zeros(n, dtype=np.int64)
    pos = 0
    for i in range(n):
        # Force close on last bar of month
        if is_month_end is not None and is_month_end[i] and pos == 1:
            pos = 0

        # Grace period: don't open new positions
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
    """Buy at bottom bin, exit when rises above exit_q. With monthly boundaries."""
    n = len(q_values)
    out = np.zeros(n, dtype=np.int64)
    pos = 0
    for i in range(n):
        if is_month_end is not None and is_month_end[i] and pos == 1:
            pos = 0
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


def compute_signal_returns(pred_series, target_series, timestamps,
                           bins, entry_q, exit_q, interval, fee,
                           direction="high"):
    """Compute long-only hysteresis returns with monthly boundaries.

    This is the single shared function used by both the backtest CLI
    and grid search scripts to ensure identical signal computation.

    Parameters
    ----------
    pred_series : array-like of predictions (float)
    target_series : array-like of targets (float)
    timestamps : DatetimeIndex or array of timestamps
    bins : int — number of quantile bins
    entry_q : int — entry quantile threshold (>= for high, <= for low)
    exit_q : int — exit quantile threshold (< for high, > for low)
    interval : str — candle interval (e.g. "15m") for month-end detection
    fee : float — fee per trade
    direction : "high" or "low"

    Returns
    -------
    dict with: trades, gross, net, sharpe
    """
    from src.utils import assign_decile_expanding

    quantiles = assign_decile_expanding(
        pd.Series(pred_series, index=timestamps) if not isinstance(pred_series, pd.Series) else pred_series,
        bins=bins,
    )
    valid = quantiles.notna()
    valid_mask = valid.values if hasattr(valid, 'values') else np.asarray(valid)
    q_arr = quantiles[valid].values.astype(np.int64)
    t_arr = np.asarray(target_series)[valid_mask]

    ts_all = pd.DatetimeIndex(timestamps)
    ts_valid = ts_all[valid_mask]
    is_grace = np.array((ts_valid.day == 1) & (ts_valid.hour == 0))
    is_month_end = np.array((ts_valid + pd.Timedelta(interval)).month != ts_valid.month)

    if direction == "high":
        sig = _hysteresis_signal(q_arr, entry_q, exit_q, is_grace, is_month_end)
    else:
        sig = _hysteresis_signal_low(q_arr, entry_q, exit_q, is_grace, is_month_end)

    gross = sig * t_arr
    prev = np.empty_like(sig)
    prev[0] = 0
    prev[1:] = sig[:-1]
    tc = np.abs(sig - prev)
    net = gross - tc * fee

    total_trades = int(tc.sum())
    total_gross = float(np.prod(1 + gross) - 1)
    total_net = float(np.prod(1 + net) - 1)
    ns = float(net.std())
    bars_per_year = 525600 / interval_to_minutes(interval)
    sharpe = float(net.mean() / ns * bars_per_year ** 0.5) if ns > 0 else 0.0

    return dict(trades=total_trades, gross=total_gross, net=total_net, sharpe=sharpe)


ALPHA_WINDOW_DAYS = 1
ALPHA_MIN_PERIODS_MINUTES = 60
ALPHA_SCALE = 0.01


def plot_equity_curve(minute_perf, output_path, title=None):
    import matplotlib.pyplot as plt

    if minute_perf is None or minute_perf.empty:
        print("No backtest results to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    minute_perf["cum_net"].plot(ax=ax, label="Net")
    if "cum_gross" in minute_perf.columns:
        minute_perf["cum_gross"].plot(ax=ax, label="Gross", alpha=0.7)

    ax.set_title(title or "Equity Curve")
    ax.set_ylabel("Equity (cumulative)")
    ax.set_xlabel("Time")
    ax.legend()

    fig.tight_layout()
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved equity curve plot: {output_path}")


def resolve_alpha_params(interval, bar_type, index):
    if bar_type == "volume":
        interval_minutes = estimate_bar_minutes(index) or interval_to_minutes(interval)
    else:
        interval_minutes = interval_to_minutes(interval)
    bars_per_day = 1440 / interval_minutes
    window = max(1, int(math.ceil(ALPHA_WINDOW_DAYS * bars_per_day)))
    min_periods = max(1, int(math.ceil(ALPHA_MIN_PERIODS_MINUTES / interval_minutes)))
    min_periods = min(min_periods, window)
    label = f"{window} bar" if window == 1 else f"{window} bars"
    return window, min_periods, label


def compute_alpha_factor(predictions, window, min_periods):
    if predictions is None or predictions.empty:
        return pd.Series(dtype=float)

    frame = predictions[["timestamp", "symbol", "signal"]].copy()
    frame = frame.sort_values(["symbol", "timestamp"]).set_index("timestamp")

    rolling = frame.groupby("symbol")["signal"].rolling(window=window, min_periods=min_periods)
    mean = rolling.mean().reset_index(level=0, drop=True)
    std = rolling.std(ddof=0).reset_index(level=0, drop=True).replace(0, np.nan)

    alpha = (frame["signal"] - mean) / std
    alpha = alpha.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    alpha = alpha * ALPHA_SCALE
    alpha.name = "alpha_factor"
    alpha = alpha.reset_index()
    return alpha.groupby("timestamp")["alpha_factor"].mean()


def plot_alpha_factor(alpha_series, output_path, title=None):
    import matplotlib.pyplot as plt

    if alpha_series is None or alpha_series.empty:
        print("No alpha factor to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    alpha_series.plot(ax=ax, color="tab:blue", linewidth=1)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_title(title or "Alpha Factor (Standardized Signal)")
    ax.set_ylabel("Alpha (z-score)")
    ax.set_xlabel("Time")

    fig.tight_layout()
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved alpha factor plot: {output_path}")


def backtest(
    predictions,
    fee=0.001,
    bins=10,
    target_quantile=None,
    exit_quantile=None,
    side="auto",
    quantile_scope="auto",
    interval="1m",
    bar_type="time",
    plot_path=None,
    plot_label=None,
    alpha_plot_path=None,
):
    print("\nStarting Backtest...")
    if predictions.empty:
        print("No predictions to backtest.")
        return

    symbol_count = pd.Index(get_symbol_key(predictions.index)).nunique()
    scope_used = resolve_quantile_scope(
        quantile_scope,
        symbol_count,
        interval=interval,
        bar_type=bar_type,
    )

    resolved_side = side
    opposite_quantile = None
    if target_quantile is None:
        rule = "long top / short bottom"
    else:
        if resolved_side == "auto":
            resolved_side = "long" if target_quantile > bins / 2 else "short"
        if resolved_side == "longshort":
            opposite_quantile = bins - target_quantile + 1
            if opposite_quantile >= target_quantile:
                rule = f"quantile >= {target_quantile} long (short leg skipped; overlap)"
            else:
                rule = f"quantile >= {target_quantile} long / <= {opposite_quantile} short"
        else:
            comparator = ">=" if resolved_side == "long" else "<="
            rule = f"quantile {comparator} {target_quantile} ({resolved_side})"
    print(f"Signal setup: scope={scope_used}, bins={bins}, rule={rule}.")

    predictions = add_quantile_labels(
        predictions,
        bins=bins,
        scope=scope_used,
        interval=interval,
        bar_type=bar_type,
    )
    if predictions.empty:
        print("No valid predictions to backtest after dropping NaNs.")
        return

    print("Calculating signal quantiles...")
    predictions["signal"] = 0
    if target_quantile is None:
        predictions.loc[predictions["quantile"] == bins, "signal"] = 1
        predictions.loc[predictions["quantile"] == 1, "signal"] = -1
    elif exit_quantile is not None and resolved_side == "long":
        # Hysteresis: enter long when quantile >= target_quantile,
        # stay long until quantile < exit_quantile.
        # Monthly boundaries: no entries first hour of month, force close last bar.
        q = predictions["quantile"].values.astype(np.int64)
        ts = pd.to_datetime(predictions["timestamp"])
        is_grace = ((ts.dt.day == 1) & (ts.dt.hour == 0)).values
        is_month_end = (ts + pd.Timedelta(interval)).dt.month.values != ts.dt.month.values
        predictions["signal"] = _hysteresis_signal(
            q, target_quantile, exit_quantile, is_grace, is_month_end,
        )
    else:
        if resolved_side == "long":
            predictions.loc[predictions["quantile"] >= target_quantile, "signal"] = 1
        elif resolved_side == "short":
            predictions.loc[predictions["quantile"] <= target_quantile, "signal"] = -1
        elif resolved_side == "longshort":
            predictions.loc[predictions["quantile"] >= target_quantile, "signal"] = 1
            if opposite_quantile is None or opposite_quantile >= target_quantile:
                print("Warning: opposite quantile overlaps target; short leg skipped.")
            else:
                predictions.loc[predictions["quantile"] <= opposite_quantile, "signal"] = -1
        else:
            raise ValueError(f"Unsupported side: {resolved_side}")

    predictions["strategy_gross"] = predictions["signal"] * predictions["target"]

    predictions["prev_signal"] = predictions.groupby("symbol")["signal"].shift(1).fillna(0)
    predictions["trades"] = (predictions["signal"] - predictions["prev_signal"]).abs()
    predictions["costs"] = predictions["trades"] * fee

    predictions["strategy_net"] = predictions["strategy_gross"] - predictions["costs"]

    alpha_series = None
    alpha_window, alpha_min_periods, alpha_window_label = resolve_alpha_params(
        interval,
        bar_type,
        predictions.index,
    )
    if alpha_plot_path:
        alpha_series = compute_alpha_factor(
            predictions,
            window=alpha_window,
            min_periods=alpha_min_periods,
        )

    minute_perf = predictions.groupby("timestamp")[["strategy_gross", "strategy_net"]].mean()

    minute_perf["cum_gross"] = (1 + minute_perf["strategy_gross"]).cumprod()
    minute_perf["cum_net"] = (1 + minute_perf["strategy_net"]).cumprod()

    total_trades = predictions["trades"].sum()
    total_return_gross = minute_perf["cum_gross"].iloc[-1] - 1
    total_return_net = minute_perf["cum_net"].iloc[-1] - 1

    net_std = minute_perf["strategy_net"].std()
    if net_std == 0 or net_std != net_std:
        bar_sharpe = 0.0
    else:
        bar_sharpe = minute_perf["strategy_net"].mean() / net_std
    if bar_type == "volume":
        interval_minutes = estimate_bar_minutes(predictions.index) or interval_to_minutes(interval)
    else:
        interval_minutes = interval_to_minutes(interval)
    bars_per_year = 525600 / interval_minutes
    annual_sharpe = bar_sharpe * (bars_per_year**0.5)

    print("-" * 30)
    print(f"Backtest Results ({minute_perf.index.min()} to {minute_perf.index.max()})")
    print("-" * 30)
    print(f"Transaction Fee: {fee * 100}%")
    print(f"Total Trades: {int(total_trades)}")
    print(f"Total Gross Return: {total_return_gross:.2%}")
    print(f"Total Net Return:   {total_return_net:.2%}")
    print(f"Annualized Sharpe:  {annual_sharpe:.4f}")
    print("-" * 30)

    if total_return_net < -0.9:
        print("\nNote: The strategy lost most of its capital due to fees.")
        print("This is expected for a 1-bar strategy with 0.1% fees per trade.")
        print(
            "To make this profitable, fees must be negligible (e.g. maker rebates) "
            "or signal alpha much higher."
        )

    if plot_path:
        prefix = f"{plot_label} " if plot_label else ""
        plot_title = (
            f"{prefix}Equity Curve ({rule}, bins={bins}, scope={scope_used}, fee={fee:.3%})"
        )
        plot_equity_curve(minute_perf, plot_path, title=plot_title)

    if alpha_plot_path:
        prefix = f"{plot_label} " if plot_label else ""
        alpha_title = (
            f"{prefix}Alpha Factor ({rule}, window={alpha_window_label}, scope={scope_used})"
        )
        plot_alpha_factor(alpha_series, alpha_plot_path, title=alpha_title)
