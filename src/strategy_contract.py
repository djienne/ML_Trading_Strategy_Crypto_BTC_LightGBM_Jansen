import json
import os


DEFAULT_DIRECTION = "high"
DEFAULT_FEE_ASSUMPTION = 0.0005
DEFAULT_QUANTILE_METHOD = "rolling"
DEFAULT_STOPLOSS = -0.20
DEFAULT_BINS = 100
DEFAULT_ENTRY_QUANTILE = 100
DEFAULT_EXIT_QUANTILE = 90
DEFAULT_TRAIN_MONTHS = 12
DEFAULT_INTERVAL = "15m"
DEFAULT_INFERENCE_SYMBOL = "BTCUSDT"

REQUIRED_KEYS = (
    "interval",
    "inference_symbol",
    "feature_flags",
    "best_iteration",
    "train_months",
    "bins",
    "entry_quantile",
    "exit_quantile",
    "direction",
    "stoploss",
    "fee_assumption",
    "quantile_method",
)


class StrategyContractError(ValueError):
    """Raised when model_info.json is missing required strategy fields."""


def build_strategy_contract(
    *,
    interval,
    inference_symbol,
    feature_flags,
    best_iteration,
    train_months,
    bins,
    entry_quantile,
    exit_quantile,
    direction=DEFAULT_DIRECTION,
    stoploss=DEFAULT_STOPLOSS,
    fee_assumption=DEFAULT_FEE_ASSUMPTION,
    quantile_method=DEFAULT_QUANTILE_METHOD,
):
    return {
        "interval": interval,
        "inference_symbol": inference_symbol,
        "feature_flags": feature_flags,
        "best_iteration": best_iteration,
        "train_months": train_months,
        "bins": bins,
        "entry_quantile": entry_quantile,
        "exit_quantile": exit_quantile,
        "direction": direction,
        "stoploss": stoploss,
        "fee_assumption": fee_assumption,
        "quantile_method": quantile_method,
    }


def read_strategy_contract(model_info):
    missing = [key for key in REQUIRED_KEYS if key not in model_info]
    if missing:
        raise StrategyContractError(
            f"model_info.json is missing required strategy contract keys: {missing}"
        )

    contract = build_strategy_contract(
        interval=str(model_info["interval"]),
        inference_symbol=str(model_info["inference_symbol"]),
        feature_flags=dict(model_info["feature_flags"] or {}),
        best_iteration=(
            None
            if model_info["best_iteration"] is None
            else int(model_info["best_iteration"])
        ),
        train_months=int(model_info["train_months"]),
        bins=int(model_info["bins"]),
        entry_quantile=int(model_info["entry_quantile"]),
        exit_quantile=int(model_info["exit_quantile"]),
        direction=str(model_info["direction"]).lower(),
        stoploss=float(model_info["stoploss"]),
        fee_assumption=float(model_info["fee_assumption"]),
        quantile_method=str(model_info["quantile_method"]).lower(),
    )

    if contract["direction"] not in {"high", "low"}:
        raise StrategyContractError(
            f"Unsupported direction in strategy contract: {contract['direction']}"
        )
    if contract["quantile_method"] not in {"rolling", "expanding"}:
        raise StrategyContractError(
            "Unsupported quantile_method in strategy contract: "
            f"{contract['quantile_method']}"
        )
    if contract["bins"] < 2:
        raise StrategyContractError("bins must be >= 2")
    if contract["train_months"] < 1:
        raise StrategyContractError("train_months must be >= 1")
    if not 1 <= contract["entry_quantile"] <= contract["bins"]:
        raise StrategyContractError("entry_quantile must fall within [1, bins]")
    if not 1 <= contract["exit_quantile"] <= contract["bins"]:
        raise StrategyContractError("exit_quantile must fall within [1, bins]")

    return contract


def default_live_model_info_path():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    return os.path.join(root_dir, "freqtrade_live", "shared", "models", "model_info.json")


def load_strategy_contract_from_path(path):
    with open(path, encoding="utf-8") as fh:
        model_info = json.load(fh)
    return read_strategy_contract(model_info), model_info


def load_local_live_strategy_contract(interval=None, symbol=None):
    path = default_live_model_info_path()
    if not os.path.exists(path):
        return None, None, None

    contract, model_info = load_strategy_contract_from_path(path)
    if interval and contract["interval"] != interval:
        return None, path, contract
    if symbol and contract["inference_symbol"] != symbol:
        return None, path, contract
    return contract, path, model_info
