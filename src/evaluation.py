import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.utils import assign_decile, assign_decile_expanding, get_symbol_key, get_time_index, resolve_quantile_scope


def add_quantile_labels(predictions, bins=10, scope="auto", interval=None, bar_type=None):
    predictions = predictions.copy()
    predictions = predictions.dropna(subset=["target", "prediction"])
    if predictions.empty:
        return predictions

    time_index = pd.to_datetime(get_time_index(predictions.index))
    symbol_index = get_symbol_key(predictions.index)
    symbol_count = pd.Index(symbol_index).nunique()
    if "timestamp" in predictions.columns:
        predictions = predictions.drop(columns=["timestamp"])
    if "symbol" in predictions.columns:
        predictions = predictions.drop(columns=["symbol"])
    predictions = predictions.reset_index(drop=True)
    predictions["timestamp"] = time_index
    predictions["symbol"] = symbol_index

    scope = resolve_quantile_scope(scope, symbol_count, interval=interval, bar_type=bar_type)
    if scope == "global":
        predictions["quantile"] = assign_decile(predictions["prediction"], bins=bins)
        return predictions.sort_values(["symbol", "timestamp"])

    if scope == "expanding":
        predictions = predictions.sort_values(["symbol", "timestamp"])
        predictions["quantile"] = predictions.groupby("symbol", sort=False)["prediction"].transform(
            lambda x: assign_decile_expanding(x, bins=bins)
        )
        predictions = predictions.dropna(subset=["quantile"])
        predictions["quantile"] = predictions["quantile"].astype(int)
        return predictions.sort_values(["symbol", "timestamp"])

    if scope == "date":
        group_key = predictions["timestamp"].dt.date
    else:
        group_key = predictions["timestamp"]

    predictions = predictions.sort_values(["timestamp", "symbol"])
    predictions["quantile"] = predictions.groupby(group_key, sort=False)["prediction"].transform(
        lambda x: assign_decile(x, bins=bins)
    )
    predictions = predictions.sort_values(["symbol", "timestamp"])
    return predictions


def compute_quantile_returns(predictions):
    return (
        predictions.groupby(["timestamp", "quantile"])["target"]
        .mean()
        .unstack("quantile")
        .sort_index()
    )


def plot_quantile_performance(returns_by_quantile, returns_filled, bins, output_path, scope_used):
    import matplotlib.pyplot as plt

    avg_returns = returns_by_quantile.mean().reindex(range(1, bins + 1))
    cum_returns = (1 + returns_filled).cumprod().sub(1)

    fig, axes = plt.subplots(ncols=2, figsize=(12, 4))
    avg_returns.mul(10000).plot(kind="bar", ax=axes[0])
    axes[0].set_title("Avg 1-bar Return by Quantile")
    axes[0].set_ylabel("Return (bps)")
    axes[0].set_xlabel("Quantile")

    cum_returns.plot(ax=axes[1], legend=False)
    axes[1].set_title("Cumulative Return by Quantile")
    axes[1].set_ylabel("Return")
    axes[1].set_xlabel("")

    fig.suptitle(f"Quantile Performance (scope: {scope_used})")
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def evaluate_predictions(
    predictions,
    bins=10,
    quantile_scope="auto",
    plot_path=None,
    interval=None,
    bar_type=None,
):
    print("\nEvaluating prediction performance...")
    symbol_count = pd.Index(get_symbol_key(predictions.index)).nunique()
    scope_used = resolve_quantile_scope(
        quantile_scope,
        symbol_count,
        interval=interval,
        bar_type=bar_type,
    )
    print(f"Predictions input: {len(predictions)} rows across {symbol_count} symbol(s).")
    predictions = add_quantile_labels(
        predictions,
        bins=bins,
        scope=scope_used,
        interval=interval,
        bar_type=bar_type,
    )
    if predictions.empty:
        print("No valid predictions to evaluate.")
        return None

    symbol_count_effective = predictions["symbol"].nunique()
    ic = spearmanr(predictions["target"], predictions["prediction"])[0]
    if np.isnan(ic):
        ic = 0.0

    if symbol_count_effective < 2 or scope_used != "timestamp":
        ic_mean = 0.0
        ic_median = 0.0
        print("IC by bar skipped (requires >=2 symbols with timestamp scope).")
    else:
        by_minute = predictions.groupby("timestamp")
        ic_by_minute = by_minute.apply(lambda x: spearmanr(x["target"], x["prediction"])[0])
        ic_by_minute = ic_by_minute.replace([np.inf, -np.inf], np.nan).dropna()
        ic_mean = ic_by_minute.mean() if not ic_by_minute.empty else 0.0
        ic_median = ic_by_minute.median() if not ic_by_minute.empty else 0.0

    returns_by_quantile = compute_quantile_returns(predictions)
    returns_filled = returns_by_quantile.fillna(0)
    quantile_index = pd.Index(range(1, bins + 1), name="quantile")
    avg_returns = returns_by_quantile.mean().reindex(quantile_index)
    cum_returns = (1 + returns_filled).cumprod().iloc[-1].sub(1).reindex(quantile_index)

    print(f"Quantile scope: {scope_used} (bins={bins})")
    print(f"IC (overall): {ic:.4%}")
    print(f"IC by bar: mean={ic_mean:.4%} median={ic_median:.4%}")
    print("\nAverage return by quantile:")
    print(avg_returns.apply(lambda x: f"{x:.4%}").to_string())
    print("\nCumulative return by quantile:")
    print(cum_returns.apply(lambda x: f"{x:.4%}").to_string())

    summary = pd.DataFrame({"avg_return": avg_returns, "cum_return": cum_returns})
    if plot_path:
        plot_quantile_performance(returns_by_quantile, returns_filled, bins, plot_path, scope_used)
    return summary
