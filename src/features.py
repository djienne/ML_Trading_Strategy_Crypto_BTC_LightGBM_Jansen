import numpy as np
import pandas as pd

from src.utils import get_date_key, get_symbol_key, interval_to_minutes


def _resolve_group_key(index, interval, bar_type):
    if bar_type == "volume":
        return get_symbol_key(index)
    interval_minutes = interval_to_minutes(interval)
    if interval_minutes < 1440:
        date_key = get_date_key(index)
        return [get_symbol_key(index), date_key]
    return get_symbol_key(index)


def engineer_features(df, interval="1m", bar_type="time", feature_flags=None):
    symbol_key = get_symbol_key(df.index)
    symbol_count = pd.Index(symbol_key).nunique()
    print(f"Engineering features: {len(df)} rows across {symbol_count} symbol(s).")
    data = pd.DataFrame(index=df.index)

    # Use 'close' and 'open' from the downloaded data
    # In download_data.py, columns are lowercase: 'open', 'high', 'low', 'close', 'volume'
    # Ret1bar = Close(t) / Open(t) - 1
    # RetKbar = Close(t) / Open(t-k+1) - 1

    feature_flags = feature_flags or {}
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataframe missing columns: {sorted(missing)}")

    group_key = _resolve_group_key(df.index, interval, bar_type)

    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    close = df["close"]
    volume = df["volume"]

    # Feature 1: 1-bar return (intraday)
    data["ret1bar"] = close.div(open_).sub(1)

    if feature_flags.get("returns", True):
        # Features 2-10: Multi-bar returns ending at current bar
        for t in range(2, 11):
            # Align shifts within each day to avoid crossing day boundaries.
            shifted_open = open_.groupby(group_key).shift(t - 1)
            data[f"ret{t}bar"] = close.div(shifted_open).sub(1)

    # Technical/volume-derived features (TA-Lib-like, OHLCV only)
    tp = (high + low + close).div(3)
    denom = (high - low).replace(0, np.nan)
    if feature_flags.get("bop", True):
        data["bop"] = close.sub(open_).div(denom)

    period = 14
    if feature_flags.get("cci", True):
        tp_ma = (
            tp.groupby(symbol_key)
            .rolling(window=period, min_periods=period)
            .mean()
            .reset_index(level=0, drop=True)
        )
        tp_md = (
            (tp - tp_ma).abs()
            .groupby(symbol_key)
            .rolling(window=period, min_periods=period)
            .mean()
            .reset_index(level=0, drop=True)
        )
        data["cci"] = (tp - tp_ma).div(0.015 * tp_md)

    if feature_flags.get("mfi", True):
        tp_diff = tp.groupby(symbol_key).diff()
        raw_mf = tp.mul(volume)
        pos_mf = raw_mf.where(tp_diff > 0, 0.0)
        neg_mf = raw_mf.where(tp_diff < 0, 0.0).abs()
        pos_sum = (
            pos_mf.groupby(symbol_key)
            .rolling(window=period, min_periods=period)
            .sum()
            .reset_index(level=0, drop=True)
        )
        neg_sum = (
            neg_mf.groupby(symbol_key)
            .rolling(window=period, min_periods=period)
            .sum()
            .reset_index(level=0, drop=True)
        )
        mfr = pos_sum.div(neg_sum.replace(0, np.nan))
        mfi = 100 - (100 / (1 + mfr))
        mfi = mfi.mask((neg_sum == 0) & (pos_sum > 0), 100)
        mfi = mfi.mask((pos_sum == 0) & (neg_sum > 0), 0)
        data["mfi"] = mfi

    rsi = None
    if feature_flags.get("rsi", True) or feature_flags.get("stochrsi", True):
        delta = close.groupby(symbol_key).diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = (
            gain.groupby(symbol_key)
            .apply(lambda x: x.ewm(alpha=1 / period, adjust=False, min_periods=period).mean())
            .reset_index(level=0, drop=True)
        )
        avg_loss = (
            loss.groupby(symbol_key)
            .apply(lambda x: x.ewm(alpha=1 / period, adjust=False, min_periods=period).mean())
            .reset_index(level=0, drop=True)
        )
        rs = avg_gain.div(avg_loss.replace(0, np.nan))
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
        rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0)
        if feature_flags.get("rsi", True):
            data["rsi"] = rsi

    if feature_flags.get("stochrsi", True) and rsi is not None:
        rsi_min = (
            rsi.groupby(symbol_key)
            .rolling(window=period, min_periods=period)
            .min()
            .reset_index(level=0, drop=True)
        )
        rsi_max = (
            rsi.groupby(symbol_key)
            .rolling(window=period, min_periods=period)
            .max()
            .reset_index(level=0, drop=True)
        )
        stochrsi = rsi.sub(rsi_min).div((rsi_max - rsi_min).replace(0, np.nan))
        data["stochrsi"] = stochrsi.mul(100)

    if feature_flags.get("stoch", True):
        low_min = (
            low.groupby(symbol_key)
            .rolling(window=period, min_periods=period)
            .min()
            .reset_index(level=0, drop=True)
        )
        high_max = (
            high.groupby(symbol_key)
            .rolling(window=period, min_periods=period)
            .max()
            .reset_index(level=0, drop=True)
        )
        fastk = close.sub(low_min).div((high_max - low_min).replace(0, np.nan)).mul(100)
        slowk = (
            fastk.groupby(symbol_key)
            .rolling(window=3, min_periods=3)
            .mean()
            .reset_index(level=0, drop=True)
        )
        slowd = (
            slowk.groupby(symbol_key)
            .rolling(window=3, min_periods=3)
            .mean()
            .reset_index(level=0, drop=True)
        )
        data["slowk"] = slowk
        data["slowd"] = slowd

    if feature_flags.get("natr", True):
        prev_close = close.groupby(symbol_key).shift(1)
        tr = pd.concat(
            [
                high.sub(low),
                high.sub(prev_close).abs(),
                low.sub(prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = (
            tr.groupby(symbol_key)
            .apply(lambda x: x.ewm(alpha=1 / period, adjust=False, min_periods=period).mean())
            .reset_index(level=0, drop=True)
        )
        data["natr"] = atr.div(close).mul(100)

    if feature_flags.get("alpha054", True):
        denom_alpha054 = low.sub(high).replace(0, -0.0001).mul(close.pow(5))
        data["alpha054"] = (
            low.sub(close)
            .mul(open_.pow(5))
            .mul(-1)
            .div(denom_alpha054)
        )

    if feature_flags.get("alpha001", True):
        r = data["ret1bar"]
        std_r = (
            r.groupby(symbol_key)
            .rolling(window=20, min_periods=20)
            .std()
            .reset_index(level=0, drop=True)
        )
        c_adj = close.where(r >= 0, std_r)
        powered = c_adj.pow(2)
        argmax = (
            powered.groupby(symbol_key)
            .rolling(window=5, min_periods=5)
            .apply(lambda x: float(np.argmax(x) + 1), raw=True)
            .reset_index(level=0, drop=True)
        )
        if isinstance(data.index, pd.MultiIndex):
            ranked = argmax.groupby(level=-1).rank(pct=True)
        else:
            ranked = argmax.rank(pct=True)
        data["alpha001"] = ranked.sub(0.5)

    return data


def prepare_target(df, data, interval="1m", bar_type="time", feature_flags=None):
    # Target: 1-bar forward return
    # According to the book/notebook logic:
    # "aim predict the 1-bar forward return"
    # "assume throughout that we can always buy (sell) at the first (last) trade price for a given bar"
    # This usually means entering at Open(t+1) and exiting at Close(t+1).
    # The return for that trade is (Close(t+1) / Open(t+1)) - 1.
    # This corresponds to 'ret1bar' shifted by -1.

    group_key = _resolve_group_key(df.index, interval, bar_type)
    data["fwd1bar"] = data["ret1bar"].groupby(group_key).shift(-1)

    feature_flags = feature_flags or {}
    if not feature_flags.get("returns", True):
        ret_cols = [c for c in data.columns if c.startswith("ret")]
        data = data.drop(columns=ret_cols, errors="ignore")

    # Drop NaN values created by lags and shifts
    data = data.dropna()
    return data
