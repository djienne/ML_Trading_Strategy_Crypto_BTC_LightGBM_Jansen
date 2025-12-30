import os

import numpy as np
import pandas as pd

from src.evaluation import add_quantile_labels
from src.utils import get_symbol_key, resolve_quantile_scope

ALPHA_WINDOW = "1D"
ALPHA_MIN_PERIODS = 60
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


def compute_alpha_factor(predictions, window=ALPHA_WINDOW, min_periods=ALPHA_MIN_PERIODS):
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
    side="auto",
    quantile_scope="auto",
    plot_path=None,
    plot_label=None,
    alpha_plot_path=None,
):
    print("\nStarting Backtest...")
    if predictions.empty:
        print("No predictions to backtest.")
        return

    symbol_count = pd.Index(get_symbol_key(predictions.index)).nunique()
    scope_used = resolve_quantile_scope(quantile_scope, symbol_count)

    resolved_side = side
    opposite_quantile = None
    if target_quantile is None:
        rule = "long top / short bottom"
    else:
        if resolved_side == "auto":
            resolved_side = "long" if target_quantile > bins / 2 else "short"
        if resolved_side == "longshort":
            opposite_quantile = bins - target_quantile + 1
            if opposite_quantile == target_quantile:
                rule = f"quantile {target_quantile} (long; opposite quantile is same)"
            else:
                rule = f"quantile {target_quantile} long / {opposite_quantile} short"
        else:
            rule = f"quantile {target_quantile} ({resolved_side})"
    print(f"Signal setup: scope={scope_used}, bins={bins}, rule={rule}.")

    predictions = add_quantile_labels(predictions, bins=bins, scope=scope_used)
    if predictions.empty:
        print("No valid predictions to backtest after dropping NaNs.")
        return

    print("Calculating signal quantiles...")
    predictions["signal"] = 0
    if target_quantile is None:
        predictions.loc[predictions["quantile"] == bins, "signal"] = 1
        predictions.loc[predictions["quantile"] == 1, "signal"] = -1
    else:
        if resolved_side == "long":
            predictions.loc[predictions["quantile"] == target_quantile, "signal"] = 1
        elif resolved_side == "short":
            predictions.loc[predictions["quantile"] == target_quantile, "signal"] = -1
        elif resolved_side == "longshort":
            predictions.loc[predictions["quantile"] == target_quantile, "signal"] = 1
            if opposite_quantile == target_quantile:
                print("Warning: opposite quantile equals target; short leg skipped.")
            else:
                predictions.loc[predictions["quantile"] == opposite_quantile, "signal"] = -1
        else:
            raise ValueError(f"Unsupported side: {resolved_side}")

    predictions["strategy_gross"] = predictions["signal"] * predictions["target"]

    predictions["prev_signal"] = predictions.groupby("symbol")["signal"].shift(1).fillna(0)
    predictions["trades"] = (predictions["signal"] - predictions["prev_signal"]).abs()
    predictions["costs"] = predictions["trades"] * fee

    predictions["strategy_net"] = predictions["strategy_gross"] - predictions["costs"]

    alpha_series = None
    alpha_window = ALPHA_WINDOW
    alpha_min_periods = ALPHA_MIN_PERIODS
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
        minute_sharpe = 0.0
    else:
        minute_sharpe = minute_perf["strategy_net"].mean() / net_std
    annual_sharpe = minute_sharpe * (525600**0.5)

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
        print("This is expected for a 1-minute strategy with 0.1% fees per trade.")
        print(
            "To make this profitable, fees must be negligible (e.g. maker rebates) "
            "or signal alpha much higher."
        )

    if plot_path:
        prefix = f"{plot_label} " if plot_label else ""
        plot_title = f"{prefix}Equity Curve ({rule}, bins={bins}, scope={scope_used})"
        plot_equity_curve(minute_perf, plot_path, title=plot_title)

    if alpha_plot_path:
        prefix = f"{plot_label} " if plot_label else ""
        alpha_title = (
            f"{prefix}Alpha Factor ({rule}, window={alpha_window}, scope={scope_used})"
        )
        plot_alpha_factor(alpha_series, alpha_plot_path, title=alpha_title)
