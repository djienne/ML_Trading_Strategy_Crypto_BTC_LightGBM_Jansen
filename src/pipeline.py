import os
import shutil

import pandas as pd

from src.backtest import backtest
from src.config_io import resolve_prediction_paths
from src.data_io import load_data, load_data_multi, load_frame, save_frame, select_symbol
from src.evaluation import evaluate_predictions
from src.features import engineer_features, prepare_target
from src.modeling import train_and_predict
from src.utils import get_symbol_key, get_train_symbols, resolve_feature_flags, resolve_quantile_scope


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

    feature_flags = resolve_feature_flags(config)
    features_df = engineer_features(df, feature_flags=feature_flags)
    model_data = prepare_target(df, features_df, feature_flags=feature_flags)
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


def run_train(config, symbol, interval, paths, retrain=False, continue_rounds=50):
    features_path = paths["features_path"]
    predictions_path = paths["predictions_path"]
    model_dir = paths["model_dir"]
    print(f"Model output: {model_dir}")
    print(f"Predictions output: {predictions_path}")

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
    predictions = train_and_predict(
        model_data,
        model_dir=model_dir,
        resume=resume,
        continue_rounds=continue_rounds,
    )
    if predictions.empty:
        print("No predictions generated.")
        return predictions

    save_frame(predictions, predictions_path)
    print(f"Saved predictions: {predictions_path} ({len(predictions)} rows)")
    return predictions


def run_evaluate(config, target_symbol, interval, bins=10, quantile_scope="auto"):
    print(f"Evaluation target: {target_symbol}")
    predictions, paths = load_predictions_for_symbol(config, target_symbol, interval)
    if predictions is None:
        return

    symbol_count = pd.Index(get_symbol_key(predictions.index)).nunique()
    scope_used = resolve_quantile_scope(quantile_scope, symbol_count)
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
    bins=10,
    quantile=None,
    side="auto",
    fee=0.001,
    quantile_scope="auto",
):
    print(f"Backtest target: {target_symbol}")
    predictions, paths = load_predictions_for_symbol(config, target_symbol, interval)
    if predictions is None:
        return

    symbol_count = pd.Index(get_symbol_key(predictions.index)).nunique()
    scope_used = resolve_quantile_scope(quantile_scope, symbol_count)
    base_name = os.path.basename(paths["predictions_path"]).replace("_predictions.feather", "")
    if quantile is None:
        rule_tag = "top_bottom"
    else:
        resolved_side = side
        if resolved_side == "auto":
            resolved_side = "long" if quantile > bins / 2 else "short"
        rule_tag = f"q{quantile}_{resolved_side}"

    plot_path = os.path.join("plot", f"{base_name}_equity_{rule_tag}_{bins}_{scope_used}.png")
    alpha_plot_path = os.path.join("plot", f"{base_name}_alpha_{rule_tag}_{bins}_{scope_used}.png")
    plot_label = f"{target_symbol} {interval}"

    backtest(
        predictions,
        fee=fee,
        bins=bins,
        target_quantile=quantile,
        side=side,
        quantile_scope=quantile_scope,
        plot_path=plot_path,
        plot_label=plot_label,
        alpha_plot_path=alpha_plot_path,
    )
