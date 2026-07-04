import numpy as np
import pandas as pd

try:
    from numba import njit as _njit
except ImportError:
    def _njit(fn=None, **kwargs):
        """No-op fallback when numba is not installed."""
        if fn is not None:
            return fn
        return lambda f: f


def get_time_index(index):
    if isinstance(index, pd.MultiIndex):
        return index.get_level_values(-1)
    return index


def get_date_key(index):
    time_index = pd.to_datetime(get_time_index(index))
    return time_index.date


def get_symbol_key(index):
    if isinstance(index, pd.MultiIndex):
        return index.get_level_values(0)
    return pd.Index(["SINGLE"] * len(index), name="symbol")


def assign_decile(x, bins=10):
    ranks = x.rank(method="first")
    n = len(ranks)
    if n == 0:
        return ranks
    if n == 1:
        return pd.Series([bins], index=x.index, dtype=int)
    if n < bins:
        scaled = (ranks - 1).div(n - 1).mul(bins - 1).add(1)
        return pd.Series(np.floor(scaled).astype(int), index=x.index)
    try:
        return pd.qcut(ranks, bins, labels=False) + 1
    except ValueError:
        scaled = (ranks - 1).div(n - 1).mul(bins - 1).add(1)
        return pd.Series(np.floor(scaled).astype(int), index=x.index)


def resolve_quantile_scope(scope, symbol_count, interval=None, bar_type=None):
    if scope != "auto":
        return scope
    if symbol_count > 1:
        return "timestamp"
    return "expanding"


@_njit(cache=True)
def _fenwick_loop(rank_in_sorted, n):
    """Fenwick tree loop — numba-accelerated inner kernel."""
    bit = np.zeros(n + 1, dtype=np.int64)
    exp_rank = np.empty(n, dtype=np.float64)

    for i in range(n):
        r = rank_in_sorted[i] + 1  # 1-indexed
        # query: count of inserted elements with sorted-rank <= r
        s = 0
        j = r
        while j > 0:
            s += bit[j]
            j -= j & (-j)
        exp_rank[i] = float(s + 1)  # +1 counts self
        # update: mark this rank as inserted
        j = r
        while j <= n:
            bit[j] += 1
            j += j & (-j)

    count = np.arange(1, n + 1, dtype=np.float64)
    return exp_rank / count


def _expanding_pct_rank(values):
    """Return expanding percentile rank for a 1-D numpy array.

    Uses a Fenwick tree (Binary Indexed Tree) for O(n log n) total time.
    At position *i* the rank equals the fraction of values in ``values[0:i+1]``
    that are <= ``values[i]`` (using stable tie-breaking by first occurrence).
    """
    n = len(values)
    if n == 0:
        return np.empty(0, dtype=np.float64)

    order = np.argsort(values, kind="mergesort")  # stable sort
    rank_in_sorted = np.empty(n, dtype=np.int64)
    rank_in_sorted[order] = np.arange(n)

    return _fenwick_loop(rank_in_sorted, n)


def expanding_pct_rank(x, min_periods=1):
    """Expanding percentile rank for a pandas Series.

    Values before *min_periods* are set to NaN.  NaN values in the
    input are preserved as NaN in the output (they are excluded from
    the ranking).
    """
    vals = x.values
    notna = ~np.isnan(vals) if vals.dtype.kind == "f" else np.ones(len(vals), dtype=bool)
    n_valid = int(notna.sum())
    if n_valid == 0:
        return pd.Series(np.nan, index=x.index)

    clean = vals[notna]
    ranked = _expanding_pct_rank(clean)
    if min_periods > 1 and min_periods <= n_valid:
        ranked[: min_periods - 1] = np.nan
    elif min_periods > n_valid:
        ranked[:] = np.nan

    result = np.full(len(vals), np.nan, dtype=np.float64)
    result[notna] = ranked
    return pd.Series(result, index=x.index)


def assign_decile_expanding(x, bins=10, min_periods=None):
    """Assign quantile bins using an expanding window (no forward-looking).

    At each position the current value is ranked against all previous values
    only.  Positions before *min_periods* are returned as NaN.
    """
    if min_periods is None:
        min_periods = bins * 5
    pct = expanding_pct_rank(x, min_periods=min_periods)
    raw_bin = np.ceil(pct * bins).clip(1, bins)
    return raw_bin.where(pct.notna()).astype("Int64")


def rolling_pct_rank(x, window, min_periods=None):
    """Rolling percentile rank over a fixed window.

    At position *i* the rank equals the fraction of values in the window
    ``[max(0, i-window+1) : i+1]`` that are <= ``x[i]``.
    Positions before *min_periods* valid values are NaN.
    """
    if min_periods is None:
        min_periods = 1

    def _rank_in_window(w):
        if np.isnan(w[-1]):
            return np.nan
        return np.sum(w <= w[-1]) / len(w)

    result = x.rolling(window, min_periods=min_periods).apply(
        _rank_in_window, raw=True,
    )
    return result


def assign_decile_rolling(x, bins=10, window=None, min_periods=None):
    """Assign quantile bins using a rolling window (no forward-looking).

    Like assign_decile_expanding but ranks against only the last *window*
    values instead of all previous values.
    """
    if window is None:
        raise ValueError("window is required for assign_decile_rolling")
    if min_periods is None:
        min_periods = min(bins * 5, window)
    pct = rolling_pct_rank(x, window=window, min_periods=min_periods)
    raw_bin = np.ceil(pct * bins).clip(1, bins)
    return raw_bin.where(pct.notna()).astype("Int64")


def resolve_feature_flags(config):
    defaults = {
        "returns": True,
        "bop": True,
        "cci": True,
        "mfi": True,
        "rsi": True,
        "stochrsi": True,
        "stoch": True,
        "natr": True,
        "alpha001": True,
        "alpha054": True,
    }
    flags = dict(defaults)
    flags.update(config.get("feature_flags", {}))
    return flags


def resolve_bar_type(config):
    bar_type = (config.get("bar_type") or "time").strip().lower()
    if bar_type not in ("time", "volume"):
        raise ValueError(f"Unsupported bar_type: {bar_type}")
    return bar_type


def format_volume_size(value):
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        cleaned = "".join(ch for ch in cleaned if ch.isalnum() or ch in (".", "-"))
        return cleaned.replace(".", "p")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number.is_integer():
        return str(int(number))
    text = f"{number}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


MINUTES_PER_YEAR = 525600  # 365-day year


def bars_per_year(interval_minutes):
    """Annualization basis shared by the vectorized backtest and the live
    PnL monitor, so their Sharpe numbers stay directly comparable."""
    return MINUTES_PER_YEAR / interval_minutes


def interval_to_minutes(interval):
    if not interval:
        raise ValueError("Interval is required.")
    unit = interval[-1]
    try:
        value = int(interval[:-1])
    except ValueError as exc:
        raise ValueError(f"Unsupported interval: {interval}") from exc
    if unit == "m":
        return value
    if unit == "h":
        return value * 60
    if unit == "d":
        return value * 1440
    if unit == "w":
        return value * 10080
    if unit == "M":
        return value * 43200
    raise ValueError(f"Unsupported interval: {interval}")


def estimate_bar_minutes(index):
    time_index = pd.to_datetime(get_time_index(index))
    if len(time_index) < 2:
        return None
    if isinstance(index, pd.MultiIndex):
        symbol_index = get_symbol_key(index)
        frame = pd.DataFrame({"symbol": symbol_index, "timestamp": time_index})
        frame = frame.sort_values(["symbol", "timestamp"])
        diffs = frame.groupby("symbol")["timestamp"].diff().dropna()
    else:
        diffs = pd.Series(time_index).sort_values().diff().dropna()
    if diffs.empty:
        return None
    minutes = diffs.median().total_seconds() / 60.0
    return minutes if minutes > 0 else None


def get_train_symbols(config):
    symbols = config.get("train_symbols") or config.get("symbols") or ["BTCUSDT"]
    return [s for s in symbols if s]


def get_inference_symbol(config, train_symbols=None):
    configured = config.get("inference_symbol")
    if configured:
        return configured
    train_symbols = train_symbols or get_train_symbols(config)
    return train_symbols[0] if train_symbols else "BTCUSDT"


def get_config_symbols(config):
    return get_train_symbols(config)
