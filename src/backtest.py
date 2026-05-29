import math
import os

import numpy as np
import pandas as pd

from src.evaluation import add_quantile_labels
from src.signal_engine import compute_desired_position, compute_prediction_quantiles
from src.utils import estimate_bar_minutes, get_symbol_key, get_time_index, interval_to_minutes, resolve_quantile_scope

try:
    from numba import njit as _njit
except ImportError:
    def _njit(fn=None, **kwargs):
        if fn is not None:
            return fn
        return lambda f: f


@_njit(cache=True)
def _apply_stoploss(signal, returns, stoploss):
    """Apply stoploss: cap the breach bar's return, flatten the position.

    Returns (modified_signal, capped_returns, sl_exit_flags).
    - modified_signal[i] = 0 on breach bar (position closed, exit fee charged)
    - capped_returns[i] = capped loss on breach bar
    - sl_exit_flags[i] = 1 on breach bars (so the capped return is still applied
      even though signal=0)

    After a stoploss exit the position stays flat until the underlying signal
    returns to 0 and then forms a fresh entry. This matches live Freqtrade,
    which closes the trade on a stoploss and will not re-enter until a new
    enter_long signal fires (a 0->1 desired transition); without this lock the
    vectorized backtest would re-enter on the very next bar while the signal is
    still high and could immediately re-trigger the stoploss.

    Note: the trigger here is the bar-by-bar open->close (cumulative `returns`)
    breach, not the intrabar low Freqtrade actually uses, so stoploss timing is
    an approximation, not a fill-faithful simulation.
    """
    n = len(signal)
    out = signal.copy()
    capped = returns.copy()
    sl_exit = np.zeros(n, dtype=np.int64)
    cum_ret = 0.0
    in_trade = False
    locked = False  # set after a stoploss exit; cleared when signal goes flat
    for i in range(n):
        if out[i] == 1:
            if locked:
                # Stoploss already closed this run; require a fresh entry.
                out[i] = 0
                continue
            if not in_trade:
                in_trade = True
                cum_ret = returns[i]
            else:
                cum_ret = (1 + cum_ret) * (1 + returns[i]) - 1
            if cum_ret <= stoploss:
                # Cap this bar so cumulative hits exactly stoploss
                cum_prev = (1 + cum_ret) / (1 + returns[i]) - 1
                if 1 + cum_prev > 1e-12:
                    capped[i] = (1 + stoploss) / (1 + cum_prev) - 1
                else:
                    capped[i] = stoploss
                out[i] = 0       # FLATTEN — triggers exit fee via tc
                sl_exit[i] = 1   # mark: capped return still applies
                in_trade = False
                cum_ret = 0.0
                locked = True
        else:
            in_trade = False
            cum_ret = 0.0
            locked = False
    return out, capped, sl_exit


def compute_signal_returns(pred_series, target_series, timestamps,
                           bins, entry_q, exit_q, interval, fee,
                           direction="high", stoploss=None,
                           quantile_window=None,
                           quantile_method=None):
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
    quantile_window : int or None — rolling window in bars for quantile
        computation.  Matches the training period so quantiles reflect the
        current model's prediction distribution.  None = expanding (legacy).

    Returns
    -------
    dict with: trades, gross, net, sharpe, valid_mask, signal, quantiles
    """
    pred_s = pd.Series(pred_series, index=timestamps) if not isinstance(pred_series, pd.Series) else pred_series
    if quantile_method is None:
        quantile_method = "rolling" if quantile_window is not None else "expanding"
    quantiles, _window = compute_prediction_quantiles(
        pred_s,
        bins,
        window=quantile_window,
        quantile_method=quantile_method,
    )
    valid = quantiles.notna()
    valid_mask = valid.values if hasattr(valid, 'values') else np.asarray(valid)
    t_arr = np.asarray(target_series)[valid_mask]
    desired_position, _valid = compute_desired_position(
        quantiles,
        timestamps,
        interval,
        entry_q,
        exit_q,
        direction=direction,
    )
    sig = desired_position[valid].to_numpy(dtype=np.int64)

    # Empty signal guard (e.g. all quantiles NaN on tiny dataset)
    if len(sig) == 0:
        return dict(trades=0, gross=0.0, net=0.0, sharpe=0.0,
                    valid_mask=valid_mask, signal=sig, quantiles=quantiles)

    # Apply stoploss if set (e.g. -0.20 = close if trade loses 20%)
    sl_exit = np.zeros_like(sig)
    if stoploss is not None and stoploss < 0:
        sig, t_arr, sl_exit = _apply_stoploss(sig, t_arr, stoploss)

    # Gross returns: normal bars use sig*return, stoploss-exit bars use capped return
    gross = sig * t_arr + sl_exit * t_arr
    prev = np.empty_like(sig)
    prev[0] = 0
    prev[1:] = sig[:-1]
    # Transaction costs: stoploss exits show sig=0 (from 1), so tc detects them.
    # Also count sl_exit bars as having an exit transition if prev was 1.
    tc = np.abs(sig - prev)
    net = gross - tc * fee

    total_trades = int(tc.sum())
    total_gross = float(np.prod(1 + gross) - 1)
    total_net = float(np.prod(1 + net) - 1)
    ns = float(net.std())
    bars_per_year = 525600 / interval_to_minutes(interval)
    sharpe = float(net.mean() / ns * bars_per_year ** 0.5) if ns > 0 else 0.0

    return dict(trades=total_trades, gross=total_gross, net=total_net, sharpe=sharpe,
                valid_mask=valid_mask, signal=sig, quantiles=quantiles,
                gross_arr=gross, net_arr=net)


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


def score_hysteresis_predictions(
    predictions,
    bins,
    target_quantile,
    exit_quantile,
    interval,
    fee,
    direction="high",
    stoploss=None,
    ic_thresh=None,
    train_months=None,
    quantile_method="rolling",
):
    """Pure helper to score predictions using the hysteresis engine.

    Returns (scored_dataframe, metrics_dict).
    """
    predictions = predictions.copy()
    predictions = predictions.dropna(subset=["target", "prediction"])
    time_index = pd.to_datetime(get_time_index(predictions.index))
    symbol_index = get_symbol_key(predictions.index)
    predictions = predictions.reset_index(drop=True)
    predictions["timestamp"] = time_index
    predictions["symbol"] = symbol_index

    if predictions.empty:
        return None, None

    # Compute rolling quantile window from training period
    q_window = None
    if train_months is not None:
        bars_per_month = int(30.4375 * 24 * 60 / interval_to_minutes(interval))
        q_window = bars_per_month * train_months

    # IC filter: skip bars where validation IC < threshold (same as grid search)
    pred_col = predictions["prediction"]
    target_col = predictions["target"]
    ts = predictions["timestamp"]
    ic_mask = None
    if ic_thresh is not None and "val_ic" in predictions.columns:
        ic_mask = (predictions["val_ic"] >= ic_thresh).values
        pred_col = pred_col[ic_mask]
        target_col = target_col[ic_mask]
        ts = ts[ic_mask]

    r = compute_signal_returns(
        pred_col, target_col, ts,
        bins, target_quantile, exit_quantile, interval, fee,
        direction=direction, stoploss=stoploss,
        quantile_window=q_window,
        quantile_method=quantile_method,
    )

    # Map results back — when IC filtering is active, results are on the
    # filtered subset. Use the filtered index to map back to full predictions.
    if ic_mask is not None:
        filtered_idx = predictions.index[ic_mask]
        valid_in_filtered = r["valid_mask"]
        valid_full_idx = filtered_idx[valid_in_filtered]

        predictions["signal"] = 0
        predictions.loc[valid_full_idx, "signal"] = r["signal"]
        predictions["quantile"] = np.nan
        predictions.loc[filtered_idx, "quantile"] = r["quantiles"].values
        predictions["strategy_gross"] = 0.0
        predictions["strategy_net"] = 0.0
        predictions.loc[valid_full_idx, "strategy_gross"] = r["gross_arr"]
        predictions.loc[valid_full_idx, "strategy_net"] = r["net_arr"]
    else:
        predictions["signal"] = 0
        predictions.loc[r["valid_mask"], "signal"] = r["signal"]
        predictions["quantile"] = r["quantiles"].values
        predictions["strategy_gross"] = 0.0
        predictions["strategy_net"] = 0.0
        predictions.loc[r["valid_mask"], "strategy_gross"] = r["gross_arr"]
        predictions.loc[r["valid_mask"], "strategy_net"] = r["net_arr"]

    predictions["prev_signal"] = predictions.groupby("symbol")["signal"].shift(1).fillna(0)
    predictions["trades"] = (predictions["signal"] - predictions["prev_signal"]).abs()
    predictions["costs"] = predictions["trades"] * fee

    metrics = {
        "trades": r["trades"],
        "gross": r["gross"],
        "net": r["net"],
        "sharpe": r["sharpe"],
        "ic_kept": ic_mask.sum() if ic_mask is not None else len(predictions),
        "ic_total": len(predictions),
        "q_window": q_window,
    }
    return predictions, metrics


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
    stoploss=None,
    ic_thresh=None,
    train_months=None,
    direction="high",
    quantile_method="rolling",
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
            # high direction: top bin is long. low direction: bottom bin is long.
            if direction == "high":
                resolved_side = "long" if target_quantile > bins / 2 else "short"
            else:
                resolved_side = "long" if target_quantile <= bins / 2 else "short"

        if resolved_side == "longshort":
            opposite_quantile = bins - target_quantile + 1
            if opposite_quantile >= target_quantile:
                rule = f"quantile >= {target_quantile} long (short leg skipped; overlap)"
            else:
                rule = f"quantile >= {target_quantile} long / <= {opposite_quantile} short"
        else:
            comparator = ">=" if resolved_side == "long" else "<="
            if direction == "low" and resolved_side == "long": comparator = "<="
            if direction == "low" and resolved_side == "short": comparator = ">="
            rule = f"quantile {comparator} {target_quantile} ({resolved_side}, {direction})"
    if exit_quantile is not None and resolved_side == "long":
        # Hysteresis path: compute_signal_returns() handles quantiles internally.
        # This ensures the CLI backtest matches the grid search exactly.
        print(f"Signal setup: quantiles={quantile_method}, bins={bins}, rule={rule}.")
        if scope_used != quantile_method:
            print(f"Note: hysteresis mode ignores scope={scope_used} and uses {quantile_method} quantiles.")

        predictions, m = score_hysteresis_predictions(
            predictions, bins, target_quantile, exit_quantile, interval, fee,
            direction=direction, stoploss=stoploss, ic_thresh=ic_thresh,
            train_months=train_months, quantile_method=quantile_method
        )

        if predictions is None:
            print("No valid predictions to backtest after dropping NaNs.")
            return

        if m["q_window"] is not None:
            print(f"Quantile window: {m['q_window']} bars ({train_months} months of {interval} candles)")
        if ic_thresh is not None:
            print(f"IC filter (>={ic_thresh}): kept {m['ic_kept']}/{m['ic_total']} bars.")

        summary_start = predictions["timestamp"].min()
        summary_end = predictions["timestamp"].max()

        print("------------------------------")
        print(f"Backtest Results ({summary_start} to {summary_end})")
        print("------------------------------")
        print(f"Transaction Fee: {fee:.2%}")
        if stoploss:
            print(f"Stoploss: {stoploss:.0%}")
        print(f"Total Trades: {m['trades']}")
        print(f"Total Gross Return: {m['gross']:.2%}")
        print(f"Total Net Return:   {m['net']:.2%}")
        print(f"Annualized Sharpe:  {m['sharpe']:.4f}")
        print("------------------------------")

        if plot_path:
            # Re-index to ensure continuous timeline for plotting
            perf = predictions.set_index("timestamp").sort_index()
            # cumprod return needs to handle symbols correctly if multi-symbol
            # but for BTC-only trading we can just use the flat net return.
            perf["cum_gross"] = (1 + perf["strategy_gross"]).cumprod() - 1
            perf["cum_net"] = (1 + perf["strategy_net"]).cumprod() - 1
            plot_equity_curve(perf, plot_path, title=f"Hysteresis {plot_label} (e={target_quantile}, x={exit_quantile})")

        if alpha_plot_path:
            alpha_window, alpha_min_periods, alpha_window_label = resolve_alpha_params(
                interval,
                bar_type,
                predictions["timestamp"],
            )
            alpha_series = compute_alpha_factor(
                predictions,
                window=alpha_window,
                min_periods=alpha_min_periods,
            )
            prefix = f"{plot_label} " if plot_label else ""
            alpha_title = (
                f"{prefix}Alpha Factor ({rule}, window={alpha_window_label}, scope={scope_used})"
            )
            plot_alpha_factor(alpha_series, alpha_plot_path, title=alpha_title)

        return # Prevent fallthrough to non-hysteresis reporting
    else:
        print(f"Signal setup: scope={scope_used}, bins={bins}, rule={rule}.")
        # Non-hysteresis paths need add_quantile_labels() for quantile assignment
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
        predictions["signal"] = 0
        if target_quantile is None:
            predictions.loc[predictions["quantile"] == bins, "signal"] = 1
            predictions.loc[predictions["quantile"] == 1, "signal"] = -1
        elif resolved_side == "long":
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
    if stoploss is not None:
        print(f"Stoploss: {stoploss:.0%}")
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
