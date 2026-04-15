import unittest

from src.strategy_contract import StrategyContractError, read_strategy_contract


class StrategyContractTests(unittest.TestCase):
    def test_read_strategy_contract_requires_signal_fields(self):
        with self.assertRaises(StrategyContractError):
            read_strategy_contract(
                {
                    "interval": "15m",
                    "inference_symbol": "BTCUSDT",
                    "feature_flags": {},
                    "best_iteration": 42,
                    "train_months": 12,
                }
            )

    def test_read_strategy_contract_normalizes_values(self):
        contract = read_strategy_contract(
            {
                "interval": "15m",
                "inference_symbol": "BTCUSDT",
                "feature_flags": {"returns": True},
                "best_iteration": "38",
                "train_months": "12",
                "bins": "100",
                "entry_quantile": "100",
                "exit_quantile": "90",
                "direction": "HIGH",
                "stoploss": "-0.2",
                "fee_assumption": "0.0005",
                "quantile_method": "ROLLING",
            }
        )

        self.assertEqual(contract["best_iteration"], 38)
        self.assertEqual(contract["train_months"], 12)
        self.assertEqual(contract["bins"], 100)
        self.assertEqual(contract["entry_quantile"], 100)
        self.assertEqual(contract["exit_quantile"], 90)
        self.assertEqual(contract["direction"], "high")
        self.assertEqual(contract["quantile_method"], "rolling")
