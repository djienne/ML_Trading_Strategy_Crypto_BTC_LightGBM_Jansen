import os

import pandas as pd


def normalize_index_names(df):
    df = df.copy()
    if isinstance(df.index, pd.MultiIndex):
        names = list(df.index.names)
        if not names[0]:
            names[0] = "symbol"
        if len(names) > 1 and (not names[-1] or names[-1] != "timestamp"):
            names[-1] = "timestamp"
        df.index.set_names(names, inplace=True)
    else:
        if not df.index.name or df.index.name != "timestamp":
            df.index.name = "timestamp"
    return df


def save_frame(df, path):
    df = normalize_index_names(df)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    try:
        df.reset_index().to_feather(tmp)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def load_frame(path):
    df = pd.read_feather(path)
    if "timestamp" not in df.columns and "open_time" in df.columns:
        df = df.rename(columns={"open_time": "timestamp"})
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if "symbol" in df.columns:
            df = df.set_index(["symbol", "timestamp"]).sort_index()
        else:
            df = df.set_index("timestamp").sort_index()
    return df


def is_month_key(value: str) -> bool:
    return (
        len(value) == 7
        and value[4] == "-"
        and value[:4].isdigit()
        and value[5:7].isdigit()
    )


def list_partition_files(data_dir, symbol, interval):
    partition_dir = os.path.join(data_dir, symbol, interval)
    if not os.path.isdir(partition_dir):
        return []
    files = []
    for name in os.listdir(partition_dir):
        if not name.endswith(".feather"):
            continue
        stem = name[:-8]
        if not is_month_key(stem):
            continue
        files.append(os.path.join(partition_dir, name))
    return sorted(files)


def load_data(data_dir, symbol, interval):
    # Construct path based on partitioned config structure.
    # Expected: data/feather/BTCUSDT/1m/2024-01.feather
    partition_files = list_partition_files(data_dir, symbol, interval)
    if partition_files:
        print(f"{symbol} {interval}: loading {len(partition_files)} monthly partitions...")
        frames = []
        for path in partition_files:
            try:
                frames.append(pd.read_feather(path))
            except Exception as exc:
                print(f"Warning: skipping corrupted partition {path}: {exc}")
        if not frames:
            print(f"{symbol} {interval}: all partitions corrupted or empty.")
            return None
        df = pd.concat(frames, ignore_index=True)
    else:
        file_path = os.path.join(data_dir, f"{symbol}_{interval}.feather")
        if not os.path.exists(file_path):
            print(f"{symbol} {interval}: data file not found: {file_path}")
            return None

        print(f"{symbol} {interval}: loading {file_path}...")
        df = pd.read_feather(file_path)

    if "open_time" in df.columns:
        df = df.drop_duplicates(subset=["open_time"])

    # Ensure datetime index
    if "open_time" in df.columns:
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df.set_index("open_time", inplace=True)
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

    return df.sort_index()


def load_data_multi(data_dir, symbols, interval):
    frames = []
    for symbol in symbols:
        df = load_data(data_dir, symbol, interval)
        if df is None or df.empty:
            print(f"No data loaded for {symbol}.")
            continue
        df = df.copy()
        df["symbol"] = symbol
        df = df.set_index("symbol", append=True).swaplevel(0, 1).sort_index()
        frames.append(df)
    if not frames:
        return None
    return pd.concat(frames).sort_index()


def select_symbol(predictions, symbol):
    if not symbol:
        return predictions
    if isinstance(predictions.index, pd.MultiIndex):
        level = predictions.index.get_level_values(0)
        if symbol not in level:
            available = list(level.unique())
            print(f"Warning: symbol {symbol} not found in predictions (available: {available})")
            return predictions.iloc[0:0]  # empty with same schema
        return predictions.xs(symbol, level=0, drop_level=False)
    return predictions
