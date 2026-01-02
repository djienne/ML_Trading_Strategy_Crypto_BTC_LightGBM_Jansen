import numpy as np
import pandas as pd


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
    if bar_type == "volume":
        return "global"
    if interval:
        try:
            interval_minutes = interval_to_minutes(interval)
        except ValueError:
            interval_minutes = None
        if interval_minutes is not None and interval_minutes >= 1440:
            return "global"
    return "date"


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
