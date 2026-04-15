import unittest

import pandas as pd

from src.modeling import train_and_predict


class TrainAndPredictTests(unittest.TestCase):
    def test_insufficient_data_returns_tuple(self):
        index = pd.date_range("2024-01-01", periods=32, freq="15min", name="timestamp")
        data = pd.DataFrame(
            {
                "feature_a": range(len(index)),
                "fwd1bar": [0.0] * len(index),
            },
            index=index,
        )

        predictions, meta = train_and_predict(
            data,
            interval="15m",
            train_months=12,
            boost_rounds=1,
        )

        self.assertTrue(predictions.empty)
        self.assertEqual(meta, {"last_best_iteration": None})
