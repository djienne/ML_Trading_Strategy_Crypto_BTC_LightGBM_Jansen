"""Feature window-invariance: live, replay and research must agree.

The live bot computes features on the freqtrade kline window
(ohlcv_candle_limit + startup_candle_count bars), the replay tool on
FEATURE_WARMUP_BARS of history, and research on everything since 2020. Every
feature must therefore converge to the same value at the decision bar no
matter where the dataframe starts, given enough warmup. alpha001 used to be
an expanding rank from dataframe start, which broke this; it is now a fixed
rolling-window rank (ALPHA001_RANK_WINDOW).
"""

import unittest

import numpy as np
import pandas as pd

from src.features import ALPHA001_RANK_WINDOW, engineer_features


def synthetic_ohlcv(n, seed=7):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.002, n)
    close = 30000.0 * np.cumprod(1 + rets)
    open_ = np.concatenate([[30000.0], close[:-1]])
    spread = np.abs(rng.normal(0, 0.001, n)) + 1e-4
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = rng.uniform(10, 1000, n)
    index = pd.date_range("2025-01-01", periods=n, freq="15min")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


class FeatureWindowParityTests(unittest.TestCase):
    def test_all_features_window_invariant_at_decision_bar(self):
        """The last row of every feature must not depend on dataframe start.

        Tail length must exceed the slowest warmup: the alpha001 chain
        (1 diff + 19 std + 4 argmax + ALPHA001_RANK_WINDOW rank bars) and the
        EMA-based features (rsi, natr), which converge geometrically.
        """
        df = synthetic_ohlcv(6000)
        full = engineer_features(df, interval="15m", bar_type="time")

        for tail in (ALPHA001_RANK_WINDOW + 600, 3600):
            truncated = engineer_features(df.tail(tail), interval="15m", bar_type="time")
            for column in full.columns:
                np.testing.assert_allclose(
                    truncated[column].iloc[-1],
                    full[column].iloc[-1],
                    rtol=1e-6,
                    atol=1e-9,
                    err_msg=(
                        f"{column} at the decision bar depends on where the "
                        f"dataframe starts (tail={tail}); live and research "
                        f"would disagree"
                    ),
                )

    def test_alpha001_rank_is_exactly_window_bounded(self):
        """alpha001 must be bit-identical once the rank window is covered."""
        df = synthetic_ohlcv(6000)
        full = engineer_features(df, interval="15m", bar_type="time")
        tail = ALPHA001_RANK_WINDOW + 30  # rank window + argmax/std chain (24) + slack
        truncated = engineer_features(df.tail(tail), interval="15m", bar_type="time")
        self.assertEqual(
            truncated["alpha001"].iloc[-1],
            full["alpha001"].iloc[-1],
            "alpha001 should be an exact fixed-window statistic",
        )


if __name__ == "__main__":
    unittest.main()
