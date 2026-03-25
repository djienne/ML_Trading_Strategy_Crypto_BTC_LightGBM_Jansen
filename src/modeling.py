import gc
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.utils import get_symbol_key, get_time_index


def ic_lgbm(preds, train_data):
    ic = spearmanr(preds, train_data.get_label())[0]
    if np.isnan(ic):
        ic = 0.0
    return "ic", ic, True


def _calendar_month_splits(timestamps, train_months=12, embargo=20, symbols=None):
    """Generate rolling splits aligned to calendar month boundaries.

    Each fold:
      - train: N calendar months ending on the last day of month M-1
      - test:  all bars in calendar month M, skipping the first *embargo*
               bars **per symbol** to prevent feature rolling-window overlap

    Returns list of (train_mask, test_mask) boolean arrays.
    """
    months = timestamps.to_period("M")
    unique_months = months.unique().sort_values()

    if len(unique_months) < train_months + 1:
        return []

    splits = []
    for i in range(train_months, len(unique_months)):
        test_month = unique_months[i]
        train_start_month = unique_months[i - train_months]
        train_end_month = unique_months[i - 1]

        train_mask = (months >= train_start_month) & (months <= train_end_month)
        test_mask_raw = months == test_month

        if embargo > 0:
            test_indices = np.where(test_mask_raw)[0]
            if symbols is not None:
                # Per-symbol embargo: drop first `embargo` rows for EACH symbol
                test_syms = symbols[test_indices]
                keep = np.ones(len(test_indices), dtype=bool)
                for sym in np.unique(test_syms):
                    sym_pos = np.where(test_syms == sym)[0]
                    keep[sym_pos[:embargo]] = False
                valid_test = test_indices[keep]
            else:
                # Single symbol: simple slice
                valid_test = test_indices[embargo:]
            if len(valid_test) == 0:
                continue
            test_mask = np.zeros(len(timestamps), dtype=bool)
            test_mask[valid_test] = True
        else:
            test_mask = test_mask_raw

        if train_mask.sum() > 0 and test_mask.sum() > 0:
            splits.append((train_mask, test_mask))

    return splits


def train_and_predict(
    data,
    interval="1m",
    bar_type="time",
    model_dir=None,
    resume=False,
    boost_rounds=5000,
    continue_rounds=50,
    train_months=12,
    num_leaves=16,
    min_data_in_leaf=100,
    feature_fraction=0.5,
    learning_rate=0.01,
):
    print("Starting model training (calendar-month rolling window)...")
    data = data.sort_index()

    target_col = "fwd1bar"
    if target_col not in data.columns:
        raise ValueError(f"Missing target column: {target_col}")

    feature_cols = [c for c in data.columns if c != target_col]
    symbol_count = pd.Index(get_symbol_key(data.index)).nunique()
    print(
        f"Training data: {len(data)} rows, {symbol_count} symbol(s), "
        f"{len(feature_cols)} features."
    )
    if resume:
        print(f"Training mode: resume (+{continue_rounds} rounds per fold when available).")
    else:
        print("Training mode: fresh models per fold.")

    params = dict(
        objective="regression",
        metric=["rmse"],
        device="cpu",
        num_leaves=num_leaves,
        min_data_in_leaf=min_data_in_leaf,
        feature_fraction=feature_fraction,
        learning_rate=learning_rate,
        verbose=-1,
        seed=42,
        num_threads=6,
    )
    num_boost_round = boost_rounds

    # Build calendar-month splits with per-symbol embargo
    timestamps = pd.to_datetime(get_time_index(data.index))
    symbols = np.asarray(get_symbol_key(data.index))
    splits = _calendar_month_splits(timestamps, train_months=train_months, embargo=20, symbols=symbols)

    if not splits:
        print(f"Warning: Insufficient data for {train_months}-month train + 1-month test.")
        print(f"Need at least {train_months + 1} calendar months of data.")
        return pd.DataFrame()

    print(f"Generated {len(splits)} calendar-month folds.")

    all_predictions = []

    for fold, (train_mask, test_mask) in enumerate(splits):
        fold_num = fold + 1
        train_ts = timestamps[train_mask]
        test_ts = timestamps[test_mask]
        print(
            f"Fold {fold_num}/{len(splits)}: "
            f"train {train_ts.min().strftime('%Y-%m-%d')} -> "
            f"{train_ts.max().strftime('%Y-%m-%d')} ({train_mask.sum()} rows), "
            f"test {test_ts.min().strftime('%Y-%m')} ({test_mask.sum()} rows)"
        )

        train_data_fold = data.iloc[np.where(train_mask)[0]]
        test_data_fold = data.iloc[np.where(test_mask)[0]]

        # Sort training fold by timestamp so last 10% is the most recent
        # time slice across all symbols (not biased by symbol order).
        train_ts_fold = pd.to_datetime(get_time_index(train_data_fold.index))
        train_data_fold = train_data_fold.iloc[np.argsort(train_ts_fold)]

        # Reserve last 10% of training data as validation for early stopping
        n_train = len(train_data_fold)
        n_val = max(1, int(n_train * 0.1))
        actual_train = train_data_fold.iloc[:-n_val]
        val_data = train_data_fold.iloc[-n_val:]

        X_train = actual_train[feature_cols]
        y_train = actual_train[target_col]
        X_val = val_data[feature_cols]
        y_val = val_data[target_col]
        X_test = test_data_fold[feature_cols]
        y_test = test_data_fold[target_col]

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)

        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        init_model = None
        if resume and model_dir:
            init_path = os.path.join(model_dir, f"fold_{fold_num:02d}.txt")
            if os.path.exists(init_path):
                init_model = init_path
                print(f"  Fold {fold_num}: resuming from {init_path} (+{continue_rounds} rounds)")

        rounds = continue_rounds if init_model else num_boost_round
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=rounds,
            valid_sets=[lgb_train, lgb_eval],
            feval=ic_lgbm,
            callbacks=callbacks,
            init_model=init_model,
        )

        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"fold_{fold_num:02d}.txt")
            model.save_model(model_path)

        best_iter = model.best_iteration or model.current_iteration()

        # Validation IC (used by grid search IC filtering)
        val_preds_arr = model.predict(X_val, num_iteration=best_iter)
        val_ic, _ = spearmanr(y_val, val_preds_arr)
        if np.isnan(val_ic):
            val_ic = 0.0

        preds = model.predict(X_test, num_iteration=best_iter)

        fold_preds = pd.DataFrame(
            {"target": y_test, "prediction": preds, "val_ic": val_ic},
            index=y_test.index,
        )
        all_predictions.append(fold_preds)

        ic, _ = spearmanr(y_test, preds)
        if np.isnan(ic):
            ic = 0.0
        print(f"  Fold {fold_num} IC: {ic:.4f} val_IC: {val_ic:.4f} (best_iter={best_iter})")
        del model, lgb_train, lgb_eval
        gc.collect()

    if not all_predictions:
        return pd.DataFrame()

    return pd.concat(all_predictions).sort_index()
