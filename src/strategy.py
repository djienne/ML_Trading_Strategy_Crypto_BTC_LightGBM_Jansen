import argparse
import warnings

from src.config_io import load_config, resolve_bar_id, resolve_paths
from src.pipeline import run_backtest, run_download, run_evaluate, run_features, run_train
from src.utils import get_inference_symbol, get_train_symbols, resolve_bar_type


warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description="Boosting bar-based strategy pipeline")
    parser.add_argument("--config", default="config.json", help="Path to config.json")

    subparsers = parser.add_subparsers(dest="command")

    download_parser = subparsers.add_parser("download", help="Download raw data")
    download_parser.add_argument("--symbols", nargs="*", help="Override symbols list")

    features_parser = subparsers.add_parser("features", help="Create features and targets")
    features_parser.add_argument("--symbol", help="Override symbol")
    features_parser.add_argument("--interval", help="Override interval")
    features_parser.add_argument("--recompute", action="store_true", help="Rebuild features")
    features_scope = features_parser.add_mutually_exclusive_group()
    features_scope.add_argument(
        "--all", action="store_true", default=True, help="Process all symbols (default)"
    )
    features_scope.add_argument("--single", action="store_true", help="Process only the target symbol")

    train_parser = subparsers.add_parser("train", help="Train model and save predictions")
    train_parser.add_argument("--symbol", help="Override symbol")
    train_parser.add_argument("--interval", help="Override interval")
    train_parser.add_argument("--retrain", action="store_true", help="Retrain from scratch")
    train_parser.add_argument(
        "--boost-rounds",
        type=int,
        default=250,
        help="Boosting rounds per fold for fresh training",
    )
    train_scope = train_parser.add_mutually_exclusive_group()
    train_scope.add_argument("--all", action="store_true", default=True, help="Train on all symbols (default)")
    train_scope.add_argument("--single", action="store_true", help="Train on the target symbol only")
    train_parser.add_argument(
        "--continue-rounds",
        type=int,
        default=50,
        help="Additional boosting rounds when resuming existing models",
    )
    train_parser.add_argument("--train-months", type=int, default=12, help="Rolling training window in months")
    train_parser.add_argument("--num-leaves", type=int, default=16)
    train_parser.add_argument("--min-data-in-leaf", type=int, default=100)
    train_parser.add_argument("--feature-fraction", type=float, default=0.5)
    train_parser.add_argument("--learning-rate", type=float, default=0.01)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate predictions by quantile")
    eval_parser.add_argument("--symbol", help="Override symbol")
    eval_parser.add_argument("--interval", help="Override interval")
    eval_parser.add_argument("--bins", type=int, default=200, help="Number of quantiles")
    eval_parser.add_argument(
        "--quantile-scope",
        choices=["auto", "timestamp", "date", "global", "expanding"],
        default="auto",
        help="How to assign quantiles (auto uses expanding for single-symbol, timestamp for multi-symbol)",
    )

    backtest_parser = subparsers.add_parser("backtest", help="Backtest signals by quantile")
    backtest_parser.add_argument("--symbol", help="Override symbol")
    backtest_parser.add_argument("--interval", help="Override interval")
    backtest_parser.add_argument("--bins", type=int, default=200, help="Number of quantiles")
    backtest_parser.add_argument(
        "--quantile",
        type=int,
        default=198,
        help="Entry quantile threshold (long uses >=). Default 198 = top 1%% with 200 bins.",
    )
    backtest_parser.add_argument(
        "--exit-quantile",
        type=int,
        default=180,
        help="Exit quantile threshold (exit long when <). Default 180 = drops below top 10%% with 200 bins.",
    )
    backtest_parser.add_argument(
        "--side",
        choices=["auto", "long", "short", "longshort"],
        default="long",
    )
    backtest_parser.add_argument("--fee", type=float, default=0.0005,
                                     help="One-way fee per trade (entry and exit each, default 0.05%%)")
    backtest_parser.add_argument("--stoploss", type=float, default=-0.20,
                                     help="Stoploss per trade (e.g. -0.20 = close if trade loses 20%%)")
    backtest_parser.add_argument("--ic-thresh", type=float, default=None,
                                     help="Skip bars where validation IC < threshold (e.g. 0.0)")
    backtest_parser.add_argument(
        "--quantile-scope",
        choices=["auto", "timestamp", "date", "global", "expanding"],
        default="auto",
        help="How to assign quantiles (auto uses expanding for single-symbol, timestamp for multi-symbol)",
    )

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    if args.command == "download":
        run_download(args.config, args.symbols)
        return

    config = load_config(args.config)
    train_symbols = get_train_symbols(config)
    default_inference = get_inference_symbol(config, train_symbols)
    target_symbol = getattr(args, "symbol", None) or default_inference
    if default_inference not in train_symbols and train_symbols:
        print(
            f"Warning: inference_symbol {default_inference} not in train_symbols; "
            "evaluation/backtest will only use that symbol."
        )
    interval = getattr(args, "interval", None) or config.get("candle_interval", "1m")

    all_symbols = True
    if getattr(args, "single", False) or getattr(args, "symbol", None):
        all_symbols = False

    scope_symbol = "ALL" if all_symbols else target_symbol
    bar_type = resolve_bar_type(config)
    paths = resolve_paths(config, scope_symbol, interval)
    bar_id = paths["bar_id"]

    print(f"Interval: {interval}")
    if bar_type == "volume":
        volume_size = config.get("volume_bar_size")
        print(f"Bar type: volume (size={volume_size})")
        print(f"Bar label: {bar_id}")
    else:
        print("Bar type: time")
    if args.command in ("features", "train"):
        if all_symbols:
            print(f"Train Symbols: ALL ({', '.join(train_symbols)})")
            if len(train_symbols) <= 1:
                only_symbol = train_symbols[0] if train_symbols else target_symbol
                print(f"Note: only one symbol configured, so ALL uses {only_symbol} only.")
        else:
            print(f"Train Symbol: {target_symbol}")
    elif args.command in ("evaluate", "backtest"):
        print(f"Inference Symbol: {target_symbol}")

    if args.command == "features":
        run_features(
            config,
            target_symbol,
            interval,
            paths,
            recompute=args.recompute,
            all_symbols=all_symbols,
        )
    elif args.command == "train":
        run_train(
            config,
            scope_symbol,
            interval,
            paths,
            retrain=args.retrain,
            boost_rounds=args.boost_rounds,
            continue_rounds=args.continue_rounds,
            train_months=args.train_months,
            num_leaves=args.num_leaves,
            min_data_in_leaf=args.min_data_in_leaf,
            feature_fraction=args.feature_fraction,
            learning_rate=args.learning_rate,
        )
    elif args.command == "evaluate":
        run_evaluate(
            config,
            target_symbol,
            interval,
            bins=args.bins,
            quantile_scope=args.quantile_scope,
        )
    elif args.command == "backtest":
        run_backtest(
            config,
            target_symbol,
            interval,
            bins=args.bins,
            quantile=args.quantile,
            exit_quantile=args.exit_quantile,
            side=args.side,
            fee=args.fee,
            quantile_scope=args.quantile_scope,
            stoploss=args.stoploss,
            ic_thresh=args.ic_thresh,
            train_months=config.get("train_months", 12),
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
