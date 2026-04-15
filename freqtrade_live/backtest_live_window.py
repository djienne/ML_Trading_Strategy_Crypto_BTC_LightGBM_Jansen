#!/usr/bin/env python3
"""
Replay live-model artifacts over Binance futures candles.

This script is designed to compare the live Freqtrade dry-run bot against
the exact artifacts that were published for live trading:
  - current artifacts in shared/models/
  - archived snapshots in shared/models/archive/

For a requested window, the script automatically picks the snapshot that was
live at that time. If the window spans multiple monthly retrains, it stitches
those snapshots together.

Usage examples
--------------
python freqtrade_live/backtest_live_window.py --start 2026-04-04
python freqtrade_live/backtest_live_window.py --start 2026-04-04 --end 2026-05-15
python freqtrade_live/backtest_live_window.py --start 2026-04-04 --print-trades
python freqtrade_live/backtest_live_window.py --start 2026-04-04 --trades-csv trades.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.backtest import _apply_stoploss
from src.data_io import load_frame
from src.features import engineer_features
from src.signal_engine import (
    compute_desired_position,
    compute_prediction_quantiles,
    shift_position_for_execution,
)
from src.strategy_contract import read_strategy_contract
from src.utils import get_time_index, interval_to_minutes


MODEL_DIR = SCRIPT_DIR / "shared" / "models"
ARCHIVE_DIR = MODEL_DIR / "archive"
CURRENT_MODEL_PATH = MODEL_DIR / "latest_model.txt"
CURRENT_INFO_PATH = MODEL_DIR / "model_info.json"
CURRENT_PRED_PATH = MODEL_DIR / "latest_predictions.feather"
BINANCE_FUTURES_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"

DEFAULT_INITIAL_BALANCE = 1000.0
FEATURE_WARMUP_BARS = 200


@dataclass
class ArtifactSnapshot:
    name: str
    training_date: pd.Timestamp
    live_from: pd.Timestamp
    timeframe: str
    inference_symbol: str
    train_months: int
    feature_flags: dict[str, Any] | None
    best_iteration: int | None
    bins: int
    entry_quantile: int
    exit_quantile: int
    direction: str
    stoploss: float
    fee_assumption: float
    quantile_method: str
    model_path: Path
    info_path: Path
    predictions_path: Path
    is_current: bool


@dataclass
class ReplaySegment:
    snapshot: ArtifactSnapshot
    segment_start: pd.Timestamp
    segment_end: pd.Timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay the live dry-run strategy from archived/current model artifacts.",
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Replay start in UTC. Date-only values mean 00:00:00 UTC.",
    )
    parser.add_argument(
        "--end",
        help="Replay end in UTC. Date-only values mean end-of-day UTC. Defaults to the latest closed candle.",
    )
    parser.add_argument(
        "--initial-balance",
        type=float,
        default=DEFAULT_INITIAL_BALANCE,
        help="Starting balance for the PnL summary. Default: 1000.",
    )
    parser.add_argument(
        "--fee",
        type=float,
        default=None,
        help="One-way fee per entry/exit transition. Default: use the artifact fee assumption.",
    )
    parser.add_argument(
        "--print-trades",
        action="store_true",
        help="Print the full trade table.",
    )
    parser.add_argument(
        "--trades-csv",
        help="Optional path to save the trade table as CSV.",
    )
    return parser.parse_args()


def parse_timestamp(text: str, *, end_of_day: bool = False) -> pd.Timestamp:
    value = text.strip()
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    if len(value) <= 10:
        if end_of_day:
            ts = ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        else:
            ts = ts.normalize()
    return ts


def floor_to_interval(ts: pd.Timestamp, timeframe: str) -> pd.Timestamp:
    return ts.floor(f"{interval_to_minutes(timeframe)}min")


def ceil_to_interval(ts: pd.Timestamp, timeframe: str) -> pd.Timestamp:
    floored = floor_to_interval(ts, timeframe)
    if floored == ts:
        return ts
    return floored + pd.Timedelta(minutes=interval_to_minutes(timeframe))


def load_snapshot(info_path: Path, model_path: Path, predictions_path: Path, name: str, is_current: bool) -> ArtifactSnapshot:
    with open(info_path, encoding="utf-8") as fh:
        info = json.load(fh)
    contract = read_strategy_contract(info)

    training_date = pd.Timestamp(info["training_date"])
    if training_date.tzinfo is not None:
        training_date = training_date.tz_convert("UTC").tz_localize(None)

    timeframe = contract["interval"]
    return ArtifactSnapshot(
        name=name,
        training_date=training_date,
        live_from=ceil_to_interval(training_date, timeframe),
        timeframe=timeframe,
        inference_symbol=contract["inference_symbol"],
        train_months=contract["train_months"],
        feature_flags=contract["feature_flags"],
        best_iteration=contract["best_iteration"],
        bins=contract["bins"],
        entry_quantile=contract["entry_quantile"],
        exit_quantile=contract["exit_quantile"],
        direction=contract["direction"],
        stoploss=contract["stoploss"],
        fee_assumption=contract["fee_assumption"],
        quantile_method=contract["quantile_method"],
        model_path=model_path,
        info_path=info_path,
        predictions_path=predictions_path,
        is_current=is_current,
    )


def discover_snapshots() -> list[ArtifactSnapshot]:
    snapshots: dict[str, ArtifactSnapshot] = {}

    if CURRENT_INFO_PATH.exists() and CURRENT_MODEL_PATH.exists() and CURRENT_PRED_PATH.exists():
        current = load_snapshot(
            CURRENT_INFO_PATH,
            CURRENT_MODEL_PATH,
            CURRENT_PRED_PATH,
            name="current",
            is_current=True,
        )
        snapshots[current.training_date.isoformat()] = current

    if ARCHIVE_DIR.exists():
        for path in sorted(ARCHIVE_DIR.iterdir()):
            if not path.is_dir():
                continue
            info_path = path / "model_info.json"
            model_path = path / "latest_model.txt"
            predictions_path = path / "latest_predictions.feather"
            if not (info_path.exists() and model_path.exists() and predictions_path.exists()):
                continue
            snapshot = load_snapshot(
                info_path,
                model_path,
                predictions_path,
                name=path.name,
                is_current=False,
            )
            key = snapshot.training_date.isoformat()
            if key not in snapshots:
                snapshots[key] = snapshot

    ordered = sorted(snapshots.values(), key=lambda snap: snap.live_from)
    if not ordered:
        raise SystemExit("No live-model artifacts found in shared/models or shared/models/archive.")
    return ordered


def build_segments(
    snapshots: list[ArtifactSnapshot],
    start_requested: pd.Timestamp,
    end_requested: pd.Timestamp,
) -> tuple[list[ReplaySegment], pd.Timestamp]:
    eligible = [snap for snap in snapshots if snap.live_from <= end_requested]
    if not eligible:
        raise SystemExit("No snapshot is available before the requested end date.")

    first_snapshot = eligible[0]
    replay_start = max(start_requested, first_snapshot.live_from)

    active_idx = 0
    for i, snapshot in enumerate(eligible):
        if snapshot.live_from <= replay_start:
            active_idx = i

    replay_start = max(replay_start, eligible[active_idx].live_from)
    segments: list[ReplaySegment] = []

    for i in range(active_idx, len(eligible)):
        snapshot = eligible[i]
        if snapshot.live_from > end_requested:
            break

        seg_start = max(replay_start if i == active_idx else snapshot.live_from, snapshot.live_from)
        next_snapshot = eligible[i + 1] if i + 1 < len(eligible) else None
        if next_snapshot is not None:
            seg_end = min(
                end_requested,
                next_snapshot.live_from - pd.Timedelta(minutes=interval_to_minutes(snapshot.timeframe)),
            )
        else:
            seg_end = end_requested

        if seg_start <= seg_end:
            segments.append(ReplaySegment(snapshot=snapshot, segment_start=seg_start, segment_end=seg_end))

    if not segments:
        raise SystemExit("No replayable segment found for the requested window.")
    return segments, replay_start


def fetch_klines(symbol: str, timeframe: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    step = pd.Timedelta(minutes=interval_to_minutes(timeframe))
    rows: list[list[Any]] = []
    cursor = start_ts

    while cursor <= end_ts:
        params = {
            "symbol": symbol,
            "interval": timeframe,
            "startTime": int(cursor.timestamp() * 1000),
            "endTime": int(end_ts.timestamp() * 1000),
            "limit": 1500,
        }
        url = f"{BINANCE_FUTURES_KLINES_URL}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=30) as resp:
            batch = json.load(resp)
        if not batch:
            break

        rows.extend(batch)
        last_open = pd.to_datetime(batch[-1][0], unit="ms", utc=True).tz_convert(None)
        next_cursor = last_open + step
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(0.1)

    if not rows:
        raise RuntimeError("No Binance candles returned for the requested window.")

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trade_count",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]
    frame = pd.DataFrame(rows, columns=columns).drop_duplicates(subset=["open_time"])
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["date"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True).dt.tz_convert(None)
    frame = frame[["date", "open", "high", "low", "close", "volume"]].sort_values("date")
    return frame.reset_index(drop=True)


def load_prediction_history(predictions_path: Path, symbol: str) -> pd.Series:
    hist = load_frame(str(predictions_path))["prediction"]
    if isinstance(hist.index, pd.MultiIndex):
        hist = hist[hist.index.get_level_values("symbol") == symbol]
        hist = hist.droplevel("symbol").sort_index()
    hist_index = pd.to_datetime(get_time_index(hist.index))
    if getattr(hist_index, "tz", None) is not None:
        hist.index = hist_index.tz_convert("UTC").tz_localize(None)
    return hist.sort_index()


def build_trade_table(
    timestamps: pd.DatetimeIndex,
    gross_arr: np.ndarray,
    net_arr: np.ndarray,
    position_arr: np.ndarray,
    snapshot_labels: list[str],
) -> pd.DataFrame:
    prev = np.empty_like(position_arr)
    prev[0] = 0
    prev[1:] = position_arr[:-1]
    trades: list[dict[str, Any]] = []
    entry_idx: int | None = None

    for i, ts in enumerate(timestamps):
        if position_arr[i] == 1 and prev[i] == 0:
            entry_idx = i
        elif position_arr[i] == 0 and prev[i] == 1 and entry_idx is not None:
            gross_ret = float(np.prod(1 + gross_arr[entry_idx : i + 1]) - 1)
            net_ret = float(np.prod(1 + net_arr[entry_idx : i + 1]) - 1)
            trades.append(
                {
                    "snapshot": snapshot_labels[entry_idx],
                    "entry_time": timestamps[entry_idx],
                    "exit_time": ts,
                    "bars": i - entry_idx + 1,
                    "gross_return_pct": gross_ret * 100,
                    "net_return_pct": net_ret * 100,
                }
            )
            entry_idx = None

    return pd.DataFrame(trades)


def replay_segment(
    segment: ReplaySegment,
    candles: pd.DataFrame,
) -> pd.DataFrame:
    warmup = pd.Timedelta(minutes=interval_to_minutes(segment.snapshot.timeframe) * FEATURE_WARMUP_BARS)
    local_start = segment.snapshot.live_from - warmup
    local = candles[(candles["date"] >= local_start) & (candles["date"] <= segment.segment_end)].copy()
    if local.empty:
        raise RuntimeError(f"No candle data for segment {segment.segment_start} -> {segment.segment_end}")

    model = lgb.Booster(model_file=str(segment.snapshot.model_path))
    features_df = engineer_features(
        local.set_index("date")[["open", "high", "low", "close", "volume"]],
        interval=segment.snapshot.timeframe,
        bar_type="time",
        feature_flags=segment.snapshot.feature_flags,
    )
    feature_names = model.feature_name()
    missing = [name for name in feature_names if name not in features_df.columns]
    if missing:
        raise RuntimeError(f"Feature mismatch for snapshot {segment.snapshot.name}: {missing}")

    pred_series = pd.Series(
        model.predict(
            features_df[feature_names].values,
            num_iteration=segment.snapshot.best_iteration,
        ),
        index=features_df.index,
        name="prediction",
    )
    returns = local.set_index("date")["ret1bar"].reindex(pred_series.index)
    history = load_prediction_history(segment.snapshot.predictions_path, segment.snapshot.inference_symbol)
    quantiles, _window = compute_prediction_quantiles(
        pred_series,
        segment.snapshot.bins,
        interval=segment.snapshot.timeframe,
        train_months=segment.snapshot.train_months,
        quantile_method=segment.snapshot.quantile_method,
        history=history,
    )
    desired_position, _valid = compute_desired_position(
        quantiles,
        pred_series.index,
        segment.snapshot.timeframe,
        segment.snapshot.entry_quantile,
        segment.snapshot.exit_quantile,
        direction=segment.snapshot.direction,
    )
    position_series = shift_position_for_execution(
        desired_position,
        live_from=segment.snapshot.live_from,
    )

    idx = pred_series.index
    mask = (idx >= segment.segment_start) & (idx <= segment.segment_end)
    return pd.DataFrame(
        {
            "timestamp": idx[mask],
            "return": returns[mask].to_numpy(dtype=np.float64),
            "position": position_series[mask].to_numpy(dtype=np.int64),
            "snapshot": segment.snapshot.name,
        }
    )


def main() -> None:
    args = parse_args()
    snapshots = discover_snapshots()

    timeframes = {snapshot.timeframe for snapshot in snapshots}
    symbols = {snapshot.inference_symbol for snapshot in snapshots}
    if len(timeframes) != 1:
        raise SystemExit(f"Mixed timeframes across snapshots are not supported: {sorted(timeframes)}")
    if len(symbols) != 1:
        raise SystemExit(f"Mixed inference symbols across snapshots are not supported: {sorted(symbols)}")

    timeframe = next(iter(timeframes))
    inference_symbol = next(iter(symbols))
    start_requested = parse_timestamp(args.start)
    end_requested = (
        parse_timestamp(args.end, end_of_day=True)
        if args.end
        else floor_to_interval(pd.Timestamp.now(tz="UTC").tz_localize(None), timeframe)
    )
    replay_end = floor_to_interval(end_requested, timeframe)
    segments, replay_start = build_segments(snapshots, start_requested, replay_end)
    stoploss_values = {segment.snapshot.stoploss for segment in segments}
    if len(stoploss_values) != 1:
        raise SystemExit(
            f"Mixed stoploss values across snapshots are not supported: {sorted(stoploss_values)}"
        )
    stoploss = stoploss_values.pop()

    if args.fee is None:
        fee_values = {segment.snapshot.fee_assumption for segment in segments}
        if len(fee_values) != 1:
            raise SystemExit(
                "Mixed fee assumptions across snapshots. Pass --fee explicitly for replay."
            )
        fee = fee_values.pop()
    else:
        fee = args.fee

    if replay_start > replay_end:
        raise SystemExit(f"Replay start {replay_start} is after replay end {replay_end}.")

    if replay_start > start_requested:
        print(
            f"Requested start {start_requested} is before the earliest matching live snapshot; "
            f"clamping to {replay_start}.",
        )

    warmup = pd.Timedelta(minutes=interval_to_minutes(timeframe) * FEATURE_WARMUP_BARS)
    fetch_start = min(segment.snapshot.live_from for segment in segments) - warmup
    candles = fetch_klines(inference_symbol, timeframe, fetch_start, replay_end)
    candles["ret1bar"] = candles["close"] / candles["open"] - 1.0

    segment_frames = [replay_segment(segment, candles) for segment in segments]
    replay_frame = pd.concat(segment_frames, ignore_index=True).sort_values("timestamp")

    timestamps = pd.DatetimeIndex(replay_frame["timestamp"])
    position_arr = replay_frame["position"].to_numpy(dtype=np.int64)
    returns_arr = replay_frame["return"].to_numpy(dtype=np.float64)
    snapshot_labels = replay_frame["snapshot"].tolist()

    position_after_sl, returns_after_sl, sl_exit = _apply_stoploss(
        position_arr,
        returns_arr,
        stoploss,
    )
    gross_arr = position_after_sl * returns_after_sl + sl_exit * returns_after_sl
    prev = np.empty_like(position_after_sl)
    prev[0] = 0
    prev[1:] = position_after_sl[:-1]
    trades_arr = np.abs(position_after_sl - prev)
    net_arr = gross_arr - trades_arr * fee

    gross_return = float(np.prod(1 + gross_arr) - 1)
    net_return = float(np.prod(1 + net_arr) - 1)
    end_balance = args.initial_balance * (1 + net_return)
    trade_table = build_trade_table(timestamps, gross_arr, net_arr, position_after_sl, snapshot_labels)

    strategy_triplets = {
        (
            segment.snapshot.bins,
            segment.snapshot.entry_quantile,
            segment.snapshot.exit_quantile,
        )
        for segment in segments
    }
    direction_values = {segment.snapshot.direction for segment in segments}

    print()
    print("Live-Artifact Replay")
    print("-------------------")
    print(f"Symbol:            {inference_symbol}")
    print(f"Timeframe:         {timeframe}")
    print(f"Requested start:   {start_requested}")
    print(f"Replay start:      {replay_start}")
    print(f"Replay end:        {timestamps.max() if len(timestamps) else replay_end}")
    print(f"Fee:               {fee:.4%}")
    print(f"Stoploss:          {stoploss:.2%}")
    if len(strategy_triplets) == 1:
        bins, entry_quantile, exit_quantile = next(iter(strategy_triplets))
        print(f"Bins / Entry / Exit: {bins} / {entry_quantile} / {exit_quantile}")
    else:
        print("Bins / Entry / Exit: mixed by snapshot")
    if len(direction_values) == 1:
        print(f"Direction:         {next(iter(direction_values))}")
    else:
        print("Direction:         mixed by snapshot")
    print(f"Bars:              {len(timestamps)}")
    print(f"Closed trades:     {len(trade_table)}")
    print(f"Gross return:      {gross_return * 100:.4f}%")
    print(f"Net return:        {net_return * 100:.4f}%")
    print(f"PnL on {args.initial_balance:.2f}:   {end_balance - args.initial_balance:.4f}")
    print(f"Ending balance:    {end_balance:.4f}")
    print("Snapshots used:")
    for segment in segments:
        live_from = segment.snapshot.live_from
        training_date = segment.snapshot.training_date
        current_flag = " (current)" if segment.snapshot.is_current else ""
        print(
            f"  - {segment.snapshot.name}{current_flag}: "
            f"trained {training_date}, live_from {live_from}, "
            f"segment {segment.segment_start} -> {segment.segment_end}, "
            f"bins={segment.snapshot.bins} entry={segment.snapshot.entry_quantile} "
            f"exit={segment.snapshot.exit_quantile}",
        )
    if not trade_table.empty:
        print(f"First entry:       {trade_table.iloc[0]['entry_time']}")
        print(f"Last exit:         {trade_table.iloc[-1]['exit_time']}")

    if args.trades_csv:
        trade_table.to_csv(args.trades_csv, index=False)
        print(f"Saved trades CSV:  {args.trades_csv}")

    if args.print_trades:
        if trade_table.empty:
            print("\nNo closed trades in this window.")
        else:
            print()
            with pd.option_context("display.max_rows", None, "display.width", 160):
                print(trade_table.to_string(index=False))


if __name__ == "__main__":
    main()
