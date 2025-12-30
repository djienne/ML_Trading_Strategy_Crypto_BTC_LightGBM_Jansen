import argparse
import warnings

from src.config_io import load_config, resolve_paths
from src.pipeline import run_backtest, run_download, run_evaluate, run_features, run_train
from src.utils import get_inference_symbol, get_train_symbols


warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description="Boosting 1-min strategy pipeline")
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
    train_scope = train_parser.add_mutually_exclusive_group()
    train_scope.add_argument("--all", action="store_true", default=True, help="Train on all symbols (default)")
    train_scope.add_argument("--single", action="store_true", help="Train on the target symbol only")
    train_parser.add_argument(
        "--continue-rounds",
        type=int,
        default=50,
        help="Additional boosting rounds when resuming existing models",
    )

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate predictions by quantile")
    eval_parser.add_argument("--symbol", help="Override symbol")
    eval_parser.add_argument("--interval", help="Override interval")
    eval_parser.add_argument("--bins", type=int, default=10, help="Number of quantiles")
    eval_parser.add_argument(
        "--quantile-scope",
        choices=["auto", "timestamp", "date", "global"],
        default="auto",
        help="How to assign quantiles (auto uses timestamp for multi-symbol, date for single-symbol)",
    )

    backtest_parser = subparsers.add_parser("backtest", help="Backtest signals by quantile")
    backtest_parser.add_argument("--symbol", help="Override symbol")
    backtest_parser.add_argument("--interval", help="Override interval")
    backtest_parser.add_argument("--bins", type=int, default=10, help="Number of quantiles")
    backtest_parser.add_argument("--quantile", type=int, help="Target quantile to trade")
    backtest_parser.add_argument(
        "--side",
        choices=["auto", "long", "short", "longshort"],
        default="auto",
    )
    backtest_parser.add_argument("--fee", type=float, default=0.001)
    backtest_parser.add_argument(
        "--quantile-scope",
        choices=["auto", "timestamp", "date", "global"],
        default="auto",
        help="How to assign quantiles (auto uses timestamp for multi-symbol, date for single-symbol)",
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
    paths = resolve_paths(config, scope_symbol, interval)

    print(f"Interval: {interval}")
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
            continue_rounds=args.continue_rounds,
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
            side=args.side,
            fee=args.fee,
            quantile_scope=args.quantile_scope,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
