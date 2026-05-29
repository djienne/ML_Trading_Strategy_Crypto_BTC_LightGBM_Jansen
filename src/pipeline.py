import json
import os
import shutil

import pandas as pd

from src.backtest import backtest
from src.bars import build_volume_bars
from src.config_io import resolve_prediction_paths
from src.data_io import load_data, load_data_multi, load_frame, save_frame, select_symbol
from src.evaluation import evaluate_predictions
from src.features import engineer_features, prepare_target
from src.modeling import train_and_predict
from src.strategy_contract import (
    DEFAULT_BINS,
    DEFAULT_DIRECTION,
    DEFAULT_ENTRY_QUANTILE,
    DEFAULT_EXIT_QUANTILE,
    DEFAULT_FEE_ASSUMPTION,
    DEFAULT_QUANTILE_METHOD,
    DEFAULT_STOPLOSS,
    DEFAULT_TRAIN_MONTHS,
    load_local_live_strategy_contract,
)
from src.utils import (
    get_symbol_key,
    get_train_symbols,
    resolve_bar_type,
    resolve_feature_flags,
    resolve_quantile_scope,
)


def run_download(config_path, symbols_override=None):
    import download_data

    download_data.main(config_path, symbols_override)


def run_features(config, symbol, interval, paths, recompute=False, all_symbols=False):
    features_path = paths["features_path"]
    print(f"Features output: {features_path}")
    if os.path.exists(features_path) and not recompute:
        print(f"Features already exist: {features_path}")
        print("Use --recompute to rebuild.")
        return features_path

    if all_symbols:
        symbols = get_train_symbols(config)
        print(f"Feature scope: ALL ({len(symbols)} symbol(s)).")
        df = load_data_multi(paths["feather_dir"], symbols, interval)
    else:
        print(f"Feature scope: SINGLE ({symbol}).")
        df = load_data(paths["feather_dir"], symbol, interval)
    if df is None:
        return None

    bar_type = resolve_bar_type(config)
    if bar_type == "volume":
        volume_size = config.get("volume_bar_size")
        print(f"Building volume bars (size={volume_size})...")
        df = build_volume_bars(df, volume_size)
        if df is None or df.empty:
            print("No volume bars generated.")
            return None

    feature_flags = resolve_feature_flags(config)
    features_df = engineer_features(
        df,
        interval=interval,
        bar_type=bar_type,
        feature_flags=feature_flags,
    )
    model_data = prepare_target(
        df,
        features_df,
        interval=interval,
        bar_type=bar_type,
        feature_flags=feature_flags,
    )
    save_frame(model_data, features_path)
    print(f"Saved features: {features_path} ({len(model_data)} rows)")
    return features_path


def load_predictions_for_symbol(config, target_symbol, interval):
    paths, used_all = resolve_prediction_paths(config, interval, target_symbol)
    predictions_path = paths["predictions_path"]
    if not os.path.exists(predictions_path):
        print(f"Missing predictions: {predictions_path}")
        print("Run the train stage first.")
        return None, None

    print(f"Loading predictions from {predictions_path}...")
    predictions = load_frame(predictions_path)
    if used_all:
        symbol_count = pd.Index(get_symbol_key(predictions.index)).nunique()
        print(f"Predictions scope: ALL ({symbol_count} symbol(s)); filtering to {target_symbol}.")
        predictions = select_symbol(predictions, target_symbol)
    else:
        print(f"Predictions scope: SINGLE ({target_symbol}).")
    return predictions, paths


def run_train(config, symbol, interval, paths, retrain=False, boost_rounds=250, continue_rounds=50,
              train_months=12, num_leaves=16, min_data_in_leaf=100, feature_fraction=0.5, learning_rate=0.01):
    features_path = paths["features_path"]
    predictions_path = paths["predictions_path"]
    model_dir = paths["model_dir"]
    print(f"Model output: {model_dir}")
    print(f"Predictions output: {predictions_path}")

    # Auto-detect training-config changes — force retrain if params differ from
    # last run. Includes feature_flags + bar_type so toggling a feature (which
    # changes the model's feature set) forces a rebuild instead of silently
    # reusing a model trained on a different feature set.
    current_params = dict(train_months=train_months, num_leaves=num_leaves,
                          min_data_in_leaf=min_data_in_leaf, feature_fraction=feature_fraction,
                          learning_rate=learning_rate, boost_rounds=boost_rounds,
                          feature_flags=resolve_feature_flags(config),
                          bar_type=resolve_bar_type(config))
    params_path = os.path.join(model_dir, "train_params.json")
    if not retrain and os.path.exists(params_path):
        try:
            with open(params_path) as f:
                old_params = json.load(f)
            if old_params != current_params:
                print(f"Hyperparams changed — forcing retrain.")
                retrain = True
        except Exception:
            pass

    if retrain:
        if os.path.exists(predictions_path):
            os.remove(predictions_path)
        if os.path.isdir(model_dir):
            shutil.rmtree(model_dir)

    existing_models = []
    if os.path.isdir(model_dir):
        existing_models = [f for f in os.listdir(model_dir) if f.endswith(".txt")]

    if os.path.exists(predictions_path) and not retrain and not existing_models:
        print(f"Skipping training; predictions already exist: {predictions_path}")
        print("Use --retrain to rebuild.")
        return load_frame(predictions_path)

    if existing_models and continue_rounds <= 0:
        print(f"Skipping training; models already exist in: {model_dir}")
        print("Use --continue-rounds > 0 to keep training or --retrain to rebuild.")
        return load_frame(predictions_path) if os.path.exists(predictions_path) else None

    if not os.path.exists(features_path):
        print(f"Missing features: {features_path}")
        print("Run the features stage first.")
        return None

    print(f"Loading features from {features_path}...")
    model_data = load_frame(features_path)
    resume = bool(existing_models) and not retrain
    bar_type = resolve_bar_type(config)
    predictions, _meta = train_and_predict(
        model_data,
        interval=interval,
        bar_type=bar_type,
        boost_rounds=boost_rounds,
        model_dir=model_dir,
        resume=resume,
        continue_rounds=continue_rounds,
        train_months=train_months,
        num_leaves=num_leaves,
        min_data_in_leaf=min_data_in_leaf,
        feature_fraction=feature_fraction,
        learning_rate=learning_rate,
    )
    if predictions.empty:
        print("No predictions generated.")
        return predictions

    save_frame(predictions, predictions_path)
    # Save params so we can detect changes on next run
    os.makedirs(model_dir, exist_ok=True)
    with open(params_path, "w") as f:
        json.dump(current_params, f, indent=2)
    print(f"Saved predictions: {predictions_path} ({len(predictions)} rows)")
    return predictions


def run_evaluate(config, target_symbol, interval, bins=10, quantile_scope="auto"):
    print(f"Evaluation target: {target_symbol}")
    predictions, paths = load_predictions_for_symbol(config, target_symbol, interval)
    if predictions is None:
        return

    symbol_count = pd.Index(get_symbol_key(predictions.index)).nunique()
    bar_type = resolve_bar_type(config)
    scope_used = resolve_quantile_scope(
        quantile_scope,
        symbol_count,
        interval=interval,
        bar_type=bar_type,
    )
    plot_path = os.path.join(
        paths["eval_dir"],
        f"{os.path.basename(paths['predictions_path']).replace('_predictions.feather', '')}"
        f"_quantiles_{bins}_{scope_used}.png",
    )
    summary = evaluate_predictions(
        predictions,
        bins=bins,
        quantile_scope=scope_used,
        plot_path=plot_path,
        interval=interval,
        bar_type=bar_type,
    )
    if summary is None:
        return

    os.makedirs(paths["eval_dir"], exist_ok=True)
    eval_path = os.path.join(
        paths["eval_dir"],
        f"{os.path.basename(paths['predictions_path']).replace('_predictions.feather', '')}"
        f"_quantiles_{bins}_{scope_used}.csv",
    )
    summary.to_csv(eval_path)
    print(f"\nSaved evaluation summary: {eval_path}")
    print(f"Saved evaluation plot: {plot_path}")


def run_backtest(
    config,
    target_symbol,
    interval,
    bins=None,
    quantile=None,
    exit_quantile=None,
    side="auto",
    fee=None,
    quantile_scope="auto",
    stoploss=None,
    ic_thresh=None,
    train_months=None,
    direction=None,
):
    print(f"Backtest target: {target_symbol}")
    predictions, paths = load_predictions_for_symbol(config, target_symbol, interval)
    if predictions is None:
        return

    symbol_count = pd.Index(get_symbol_key(predictions.index)).nunique()
    bar_type = resolve_bar_type(config)
    scope_used = resolve_quantile_scope(
        quantile_scope,
        symbol_count,
        interval=interval,
        bar_type=bar_type,
    )
    artifact_contract, artifact_path, artifact_match = load_local_live_strategy_contract(
        interval=interval,
        symbol=target_symbol,
    )
    if artifact_contract is not None:
        print(f"Using deployed strategy contract from {artifact_path}.")
    elif artifact_path and artifact_match is not None:
        print(
            "Live artifact contract exists but does not match this backtest "
            f"(have {artifact_match['inference_symbol']} {artifact_match['interval']})."
        )

    def resolve_contract_value(name, cli_value, fallback):
        if artifact_contract is not None:
            artifact_value = artifact_contract[name]
            if cli_value is None:
                return artifact_value
            if cli_value != artifact_value:
                print(
                    f"Warning: overriding artifact {name}={artifact_value!r} "
                    f"with CLI value {cli_value!r}."
                )
            return cli_value
        return fallback if cli_value is None else cli_value

    resolved_bins = resolve_contract_value("bins", bins, DEFAULT_BINS)
    resolved_quantile = resolve_contract_value("entry_quantile", quantile, DEFAULT_ENTRY_QUANTILE)
    resolved_exit_quantile = resolve_contract_value(
        "exit_quantile",
        exit_quantile,
        DEFAULT_EXIT_QUANTILE,
    )
    resolved_fee = resolve_contract_value("fee_assumption", fee, DEFAULT_FEE_ASSUMPTION)
    resolved_stoploss = resolve_contract_value("stoploss", stoploss, DEFAULT_STOPLOSS)
    resolved_train_months = resolve_contract_value("train_months", train_months, config.get("train_months", DEFAULT_TRAIN_MONTHS))
    resolved_direction = resolve_contract_value("direction", direction, DEFAULT_DIRECTION)
    resolved_quantile_method = (
        artifact_contract["quantile_method"]
        if artifact_contract is not None
        else DEFAULT_QUANTILE_METHOD
    )
    base_name = os.path.basename(paths["predictions_path"]).replace("_predictions.feather", "")
    if resolved_quantile is None:
        rule_tag = "top_bottom"
        filename_scope = scope_used
    else:
        resolved_side = side
        if resolved_side == "auto":
            resolved_side = "long" if resolved_quantile > resolved_bins / 2 else "short"
        rule_tag = f"q{resolved_quantile}_{resolved_side}_{resolved_direction}"
        filename_scope = (
            resolved_quantile_method
            if resolved_exit_quantile is not None and resolved_side == "long"
            else scope_used
        )

    plot_path = os.path.join("plot", f"{base_name}_equity_{rule_tag}_{resolved_bins}_{filename_scope}.png")
    alpha_plot_path = os.path.join("plot", f"{base_name}_alpha_{rule_tag}_{resolved_bins}_{filename_scope}.png")
    config_path = os.path.join("plot", f"{base_name}_backtest_{rule_tag}_{resolved_bins}_{filename_scope}.json")
    plot_label = f"{target_symbol} {paths['bar_id']}"
    os.makedirs("plot", exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "artifact_contract_path": artifact_path if artifact_contract is not None else None,
                "interval": interval,
                "target_symbol": target_symbol,
                "bins": resolved_bins,
                "entry_quantile": resolved_quantile,
                "exit_quantile": resolved_exit_quantile,
                "fee_assumption": resolved_fee,
                "stoploss": resolved_stoploss,
                "train_months": resolved_train_months,
                "direction": resolved_direction,
                "quantile_method": resolved_quantile_method,
                "side": side,
                "quantile_scope": quantile_scope,
                "ic_thresh": ic_thresh,
            },
            fh,
            indent=2,
        )
    print(f"Saved backtest config: {config_path}")

    backtest(
        predictions,
        fee=resolved_fee,
        bins=resolved_bins,
        target_quantile=resolved_quantile,
        exit_quantile=resolved_exit_quantile,
        side=side,
        quantile_scope=quantile_scope,
        interval=interval,
        bar_type=bar_type,
        plot_path=plot_path,
        plot_label=plot_label,
        stoploss=resolved_stoploss,
        ic_thresh=ic_thresh,
        alpha_plot_path=alpha_plot_path,
        train_months=resolved_train_months,
        direction=resolved_direction,
        quantile_method=resolved_quantile_method,
    )
