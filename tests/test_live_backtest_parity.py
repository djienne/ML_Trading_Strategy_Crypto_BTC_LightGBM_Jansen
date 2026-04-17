"""Parity test: offline backtest shift vs. simulated live pipeline.

The offline backtest obtains the held-bar position series by calling
``shift_position_for_execution`` on ``desired_position``. The live bot obtains
it by calling ``compute_live_transition_signals`` and letting Freqtrade fill
at the next bar's open. If these two paths disagree on which 15-min candle
is actually held, offline backtests will not match live dry-run results.

This test wires both paths through the same ``desired_position`` series and
asserts they produce the same held-bar series.
"""

import unittest

import pandas as pd

from src.signal_engine import (
    compute_live_transition_signals,
    shift_position_for_execution,
)


def simulate_freqtrade_fills(enter_signal, exit_signal):
    """Convert (enter_signal, exit_signal) to a held-bar 0/1 series.

    Matches Freqtrade's convention that ``enter_long=1`` / ``exit_long=1`` at
    row ``T`` causes the order to fill at the open of bar ``T+1``. So at bar
    ``t`` we first record the current held state, then apply any signal at
    ``t`` to update state for bar ``t+1``.
    """
    index = enter_signal.index
    enter = enter_signal.astype(bool).to_numpy()
    exit_ = exit_signal.astype(bool).to_numpy()
    pos = 0
    out = []
    for t in range(len(index)):
        out.append(pos)
        if pos == 1 and exit_[t]:
            pos = 0
        elif pos == 0 and enter[t]:
            pos = 1
    return pd.Series(out, index=index, dtype="int64")


class LiveBacktestParityTests(unittest.TestCase):
    def test_shift_matches_simulated_live_fills(self):
        timestamps = pd.date_range("2026-04-15 00:00", periods=9, freq="15min")
        desired = pd.Series([0, 0, 1, 1, 1, 0, 0, 1, 0], index=timestamps, dtype="int64")

        backtest_position = shift_position_for_execution(desired)

        enter_signal, exit_signal = compute_live_transition_signals(desired)
        live_position = simulate_freqtrade_fills(enter_signal, exit_signal)

        self.assertEqual(
            backtest_position.tolist(),
            live_position.tolist(),
            msg=(
                "shift_position_for_execution and the live pipeline "
                "(compute_live_transition_signals + Freqtrade fill-at-next-bar) "
                "disagree on which bar is held.\n"
                f"  backtest: {backtest_position.tolist()}\n"
                f"  live:     {live_position.tolist()}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
