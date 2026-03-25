"""
Full grid search: BTC+ETH training, BTC-only trading, 15m candles.
Fixed: lr=0.01, patience=50, boost_rounds=5000

Uses train_and_predict() from src/modeling.py and compute_signal_returns()
from src/backtest.py — the exact same code paths as the CLI — to ensure
identical model training and signal computation.
"""

import csv
import gc
import pandas as pd
import numpy as np
from src.data_io import load_data_multi, select_symbol
from src.features import engineer_features, prepare_target
from src.modeling import train_and_predict
from src.utils import get_time_index, get_symbol_key
from src.backtest import compute_signal_returns

# Fixed params
LR = 0.01
BOOST_ROUNDS = 5000
FEE = 0.0005
STOPLOSS = -0.20  # must match live strategy stoploss
INTERVAL = "15m"
RESULTS_CSV = "grid_search_results.csv"

# Load data once
print("Loading BTC+ETH 15m data...", flush=True)
df = load_data_multi("data/feather", ["BTCUSDT", "ETHUSDT"], INTERVAL)
features_df = engineer_features(df, interval=INTERVAL, bar_type="time")
data = prepare_target(df, features_df, interval=INTERVAL, bar_type="time")
data = data.sort_index()
del df, features_df
gc.collect()

print(f"Data: {len(data)} rows", flush=True)

# Grid
TRAIN_MONTHS_LIST = [2, 4, 6, 12]
MODEL_CONFIGS = [
    (l, f, m)
    for l in [16, 31, 63]
    for f in [0.1, 0.25, 0.5, 0.8, 1.0]
    for m in [50, 100, 200, 300, 500]
]
BINS_LIST = [100, 150, 200, 300, 400, 500]

SIGNAL_HIGH = [(100, 92), (100, 90), (100, 85), (100, 80), (99, 92), (99, 90), (99, 85)]
SIGNAL_LOW = [(1, 8), (1, 10), (1, 15), (1, 20), (2, 10), (2, 15), (2, 20)]
IC_THRESHOLDS = [None, 0.0, 0.01]


# CSV header
csv_fields = ["tm", "leaves", "ff", "mdil", "direction", "bins",
              "entry_pct", "exit_pct", "ic_thresh", "trades", "gross", "net", "sharpe"]
with open(RESULTS_CSV, "w", newline="") as f:
    csv.DictWriter(f, csv_fields).writeheader()

total_configs = len(TRAIN_MONTHS_LIST) * len(MODEL_CONFIGS)
config_num = 0
best_overall = {"net": -999}

for train_months in TRAIN_MONTHS_LIST:
    for leaves, ff, mdil in MODEL_CONFIGS:
        config_num += 1

        # Use the SAME training function as the CLI
        predictions = train_and_predict(
            data,
            interval=INTERVAL,
            bar_type="time",
            boost_rounds=BOOST_ROUNDS,
            train_months=train_months,
            num_leaves=leaves,
            min_data_in_leaf=mdil,
            feature_fraction=ff,
            learning_rate=LR,
        )

        if predictions.empty:
            continue

        # Filter to BTC only (same as CLI backtest)
        btc_preds = select_symbol(predictions, "BTCUSDT")
        del predictions
        if btc_preds.empty:
            gc.collect()
            continue

        btc_ts = pd.to_datetime(get_time_index(btc_preds.index))
        pred_vals = btc_preds["prediction"]
        target_vals = btc_preds["target"]
        ic_vals = btc_preds["val_ic"]

        best_this_config = {"net": -999}
        batch_results = []

        for ic_thresh in IC_THRESHOLDS:
            # IC filter: skip bad-IC bars entirely (don't zero — that distorts
            # expanding quantiles for future bars)
            if ic_thresh is not None:
                ic_mask = (ic_vals >= ic_thresh).values
                filtered_preds = pred_vals[ic_mask]
                filtered_targets = target_vals[ic_mask]
                filtered_ts = btc_ts[ic_mask]
            else:
                filtered_preds = pred_vals
                filtered_targets = target_vals
                filtered_ts = btc_ts

            for bins in BINS_LIST:
                # Normal: buy high
                for entry_pct, exit_pct in SIGNAL_HIGH:
                    entry_q = int(bins * entry_pct / 100)
                    exit_q = int(bins * exit_pct / 100)
                    if exit_q >= entry_q:
                        continue
                    r = compute_signal_returns(
                        filtered_preds, filtered_targets, filtered_ts,
                        bins, entry_q, exit_q, INTERVAL, FEE, direction="high", stoploss=STOPLOSS,
                    )
                    row = dict(tm=train_months, leaves=leaves, ff=ff, mdil=mdil,
                               direction="high", bins=bins, entry_pct=entry_pct,
                               exit_pct=exit_pct, ic_thresh=ic_thresh,
                               trades=r["trades"], gross=r["gross"],
                               net=r["net"], sharpe=r["sharpe"])
                    batch_results.append(row)
                    if r["net"] > best_this_config.get("net", -999):
                        best_this_config = row

                # Flipped: buy low
                for entry_low, exit_low_pct in SIGNAL_LOW:
                    exit_q = int(bins * exit_low_pct / 100)
                    if exit_q <= entry_low:
                        continue
                    r = compute_signal_returns(
                        filtered_preds, filtered_targets, filtered_ts,
                        bins, entry_low, exit_q, INTERVAL, FEE, direction="low", stoploss=STOPLOSS,
                    )
                    row = dict(tm=train_months, leaves=leaves, ff=ff, mdil=mdil,
                               direction="low", bins=bins, entry_pct=entry_low,
                               exit_pct=exit_low_pct, ic_thresh=ic_thresh,
                               trades=r["trades"], gross=r["gross"],
                               net=r["net"], sharpe=r["sharpe"])
                    batch_results.append(row)
                    if r["net"] > best_this_config.get("net", -999):
                        best_this_config = row

        # Write batch to CSV
        with open(RESULTS_CSV, "a", newline="") as f:
            w = csv.DictWriter(f, csv_fields)
            w.writerows(batch_results)

        if best_this_config.get("net", -999) > best_overall.get("net", -999):
            best_overall = best_this_config

        b = best_this_config
        ic_str = f"ic>{b.get('ic_thresh')}" if b.get('ic_thresh') is not None else "no_filt"
        print(
            f"[{config_num:3d}/{total_configs}] "
            f"tm={train_months:2d} L={leaves:2d} ff={ff} mdil={mdil:3d} | "
            f"best: {b.get('direction','?')} bins={b.get('bins','?')} "
            f"e={b.get('entry_pct','?')} x={b.get('exit_pct','?')}% {ic_str} "
            f"net={b.get('net',0):+.2%} trades={b.get('trades',0)} "
            f"sharpe={b.get('sharpe',0):.2f}",
            flush=True,
        )

        del btc_preds, batch_results
        gc.collect()

# Final summary from CSV
print("\n\n========== TOP 30 ==========", flush=True)
with open(RESULTS_CSV) as f:
    all_results = list(csv.DictReader(f))
for r in all_results:
    r["net"] = float(r["net"])
    r["sharpe"] = float(r["sharpe"])
    r["trades"] = int(r["trades"])
all_results.sort(key=lambda x: -x["net"])
for i, r in enumerate(all_results[:30]):
    ic_str = f"ic>{r['ic_thresh']}" if r['ic_thresh'] != "" else "no_filt"
    print(
        f"{i+1:2d}. tm={r['tm']:>2s} L={r['leaves']:>2s} ff={r['ff']:>4s} mdil={r['mdil']:>3s} "
        f"{r['direction']:4s} bins={r['bins']:>3s} e={r['entry_pct']} x={r['exit_pct']}% "
        f"{ic_str:8s} | trades={r['trades']:5d} net={r['net']:+8.2%} "
        f"sharpe={r['sharpe']:+.2f}",
        flush=True,
    )

print("\n========== BEST BY TRAINING PERIOD ==========", flush=True)
for tm in TRAIN_MONTHS_LIST:
    these = [r for r in all_results if r["tm"] == str(tm)]
    if not these:
        continue
    best = max(these, key=lambda x: x["net"])
    ic_str = f"ic>{best['ic_thresh']}" if best['ic_thresh'] != "" else "no_filt"
    print(
        f"  {tm:2d}m: {best['direction']:4s} net={best['net']:+8.2%} trades={best['trades']:5d} "
        f"sharpe={best['sharpe']:+.2f} L={best['leaves']} ff={best['ff']} "
        f"mdil={best['mdil']} bins={best['bins']} e={best['entry_pct']} "
        f"x={best['exit_pct']}% {ic_str}",
        flush=True,
    )

print("\n========== BEST HIGH vs LOW ==========", flush=True)
for d in ["high", "low"]:
    these = [r for r in all_results if r["direction"] == d]
    if not these:
        continue
    best = max(these, key=lambda x: x["net"])
    print(f"  {d:4s}: net={best['net']:+8.2%} sharpe={best['sharpe']:+.2f}", flush=True)

print("\n========== IC FILTER IMPACT ==========", flush=True)
for ic_t in ["", "0.0", "0.01"]:
    these = [r for r in all_results if r["ic_thresh"] == ic_t]
    if not these:
        continue
    best = max(these, key=lambda x: x["net"])
    label = f"ic>{ic_t}" if ic_t else "no_filter"
    print(f"  {label:12s}: best net={best['net']:+8.2%} sharpe={best['sharpe']:+.2f}", flush=True)
