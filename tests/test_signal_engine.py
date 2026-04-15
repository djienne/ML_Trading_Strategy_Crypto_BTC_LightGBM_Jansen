import unittest

import pandas as pd

from src.signal_engine import (
    compute_desired_position,
    compute_live_transition_signals,
    shift_position_for_execution,
)


class SignalEngineTests(unittest.TestCase):
    def test_month_end_and_grace_period_behavior(self):
        timestamps = pd.DatetimeIndex(
            [
                "2026-01-31 23:15:00",
                "2026-01-31 23:30:00",
                "2026-01-31 23:45:00",
                "2026-02-01 00:00:00",
                "2026-02-01 00:15:00",
                "2026-02-01 01:00:00",
                "2026-02-01 01:15:00",
            ]
        )
        quantiles = pd.Series([95, 100, 100, 100, 100, 100, 85], index=timestamps)

        desired_position, valid = compute_desired_position(
            quantiles,
            timestamps,
            "15m",
            100,
            90,
            direction="high",
        )
        executed_position = shift_position_for_execution(
            desired_position,
            live_from=timestamps[0],
        )
        entry_signal, exit_signal = compute_live_transition_signals(desired_position)

        self.assertTrue(valid.all())
        self.assertEqual(desired_position.tolist(), [0, 1, 0, 0, 0, 1, 0])
        self.assertEqual(executed_position.tolist(), [0, 0, 1, 0, 0, 0, 1])
        self.assertEqual(entry_signal.astype(int).tolist(), [0, 0, 1, 0, 0, 0, 1])
        self.assertEqual(exit_signal.astype(int).tolist(), [0, 0, 0, 1, 0, 0, 0])
