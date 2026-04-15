import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.modeling import train_and_predict


class _FakeBooster:
    best_iteration = 3

    def predict(self, X, num_iteration=None):
        return np.linspace(0.0, 1.0, len(X), dtype=float)

    def current_iteration(self):
        return 3


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
        events = []

        predictions, meta = train_and_predict(
            data,
            interval="15m",
            train_months=12,
            boost_rounds=1,
            progress=events.append,
            heartbeat_seconds=0,
        )

        self.assertTrue(predictions.empty)
        self.assertEqual(meta, {"last_best_iteration": None})
        joined = "\n".join(events)
        self.assertIn("Starting model training", joined)
        self.assertIn("Insufficient data", joined)

    def test_train_and_predict_reports_progress_messages(self):
        index = pd.date_range("2024-01-01", periods=30, freq="D", name="timestamp")
        data = pd.DataFrame(
            {
                "feature_a": np.linspace(0.0, 1.0, len(index)),
                "fwd1bar": np.linspace(-0.5, 0.5, len(index)),
            },
            index=index,
        )
        train_mask = np.array([True] * 24 + [False] * 6)
        test_mask = np.array([False] * 24 + [True] * 6)
        events = []

        with patch("src.modeling._calendar_month_splits", return_value=[(train_mask, test_mask)]):
            with patch("src.modeling.lgb.train", return_value=_FakeBooster()):
                predictions, meta = train_and_predict(
                    data,
                    interval="1d",
                    boost_rounds=5,
                    progress=events.append,
                    heartbeat_seconds=0,
                )

        self.assertFalse(predictions.empty)
        self.assertEqual(meta, {"last_best_iteration": 3})
        joined = "\n".join(events)
        self.assertIn("Starting model training", joined)
        self.assertIn("Generated 1 calendar-month folds.", joined)
        self.assertIn("Fold 1/1:", joined)
        self.assertIn("starting LightGBM training", joined)
        self.assertIn("Fold 1 IC:", joined)
