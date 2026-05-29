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

import numpy as np
import pandas as pd

from src.backtest import compute_signal_returns
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

    def test_offline_backtest_uses_no_execution_shift(self):
        """`compute_signal_returns` must earn desired[t] * target[t] with no shift.

        The offline path bakes the single bar of execution delay into the target
        (target = fwd1bar = next candle's return), so its per-bar gross return is
        exactly desired[t] * target[t]. If a shift were (re)introduced here, the
        offline backtest would diverge from the live/replay path again.
        """
        n = 400
        timestamps = pd.date_range("2026-01-01", periods=n, freq="15min")
        # Deterministic, varied predictions so expanding quantiles produce a mix
        # of bins and several enter/exit transitions.
        t = np.arange(n)
        preds = np.sin(t / 7.0) + 0.3 * np.sin(t / 3.0)
        targets = np.cos(t / 5.0) * 0.001

        r = compute_signal_returns(
            pd.Series(preds, index=timestamps),
            targets,
            timestamps,
            bins=10,
            entry_q=10,
            exit_q=8,
            interval="15m",
            fee=0.0,
        )

        self.assertGreater(len(r["signal"]), 0, "test setup produced no valid bars")
        target_valid = np.asarray(targets)[r["valid_mask"]]
        np.testing.assert_allclose(
            r["gross_arr"],
            r["signal"] * target_valid,
            err_msg="offline gross != desired[t] * target[t]; an execution shift leaked in",
        )

    def test_offline_target_convention_matches_replay_shift(self):
        """The offline target convention and the replay shift earn the same bar.

        Offline earns desired[t] * fwd1bar[t] where fwd1bar[t] = ret1bar[t+1].
        Replay earns shift_position_for_execution(desired)[t] * ret1bar[t].
        These must capture the *same* ret1bar for the same desired[t]:
            offline_contrib[t] == replay_contrib[t+1]
        which is what keeps main.py backtest, the replay tool, and the live bot
        all holding candle T+1 for a transition at candle T.
        """
        timestamps = pd.date_range("2026-04-15 00:00", periods=9, freq="15min")
        desired = pd.Series([0, 0, 1, 1, 1, 0, 0, 1, 0], index=timestamps, dtype="int64")
        ret1bar = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

        fwd1bar = pd.Series(ret1bar, index=timestamps).shift(-1).to_numpy()
        offline_contrib = desired.to_numpy() * fwd1bar  # earns ret1bar[t+1]

        replay_position = shift_position_for_execution(desired).to_numpy()
        replay_contrib = replay_position * ret1bar  # earns ret1bar[t]

        # offline_contrib[t] should equal replay_contrib[t+1] (same economic bar).
        np.testing.assert_allclose(
            offline_contrib[:-1],
            replay_contrib[1:],
            err_msg="offline target convention and replay shift hold different bars",
        )


if __name__ == "__main__":
    unittest.main()
