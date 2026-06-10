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
    compute_desired_position,
    compute_live_level_signals,
    compute_live_transition_signals,
    shift_position_for_execution,
)


def simulate_freqtrade_fills(enter_signal, exit_signal, missed_fills=()):
    """Convert (enter, exit) flags to a held-bar 0/1 series, freqtrade-style.

    A flag at row T fills at the open of bar T+1. Freqtrade only acts on
    enter when flat and on exit when holding, which is what makes level-based
    flags safe. ``missed_fills`` is a set of bar indices whose fill attempt
    fails (entry order never executed during that bar) - used to show that
    level-based signals retry while edge-triggered signals lose the trade.
    """
    index = enter_signal.index
    enter = enter_signal.astype(bool).to_numpy()
    exit_ = exit_signal.astype(bool).to_numpy()
    pos = 0
    pending = None  # "enter"/"exit" decided on the previous bar
    out = []
    for t in range(len(index)):
        if pending == "enter" and t not in missed_fills:
            pos = 1
        elif pending == "exit" and t not in missed_fills:
            pos = 0
        out.append(pos)
        if pos == 1 and exit_[t]:
            pending = "exit"
        elif pos == 0 and enter[t]:
            pending = "enter"
        else:
            pending = None
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

    def test_level_signals_match_shift_contract(self):
        """Level flags + freqtrade fill mechanics == shift_position_for_execution.

        The live bot publishes enter_long while desired==1 and exit_long while
        desired==0 (NaN bars publish nothing). With freqtrade acting on enter
        only when flat and exit only when holding, the held bars must be the
        offline convention desired.shift(1).
        """
        timestamps = pd.date_range("2026-04-15 00:00", periods=12, freq="15min")
        desired = pd.Series(
            [np.nan, np.nan, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0],
            index=timestamps,
            dtype="float64",
        )

        enter, exit_ = compute_live_level_signals(desired)
        self.assertFalse(enter.iloc[0] or exit_.iloc[0], "NaN bars must publish no flags")

        live_position = simulate_freqtrade_fills(enter, exit_)
        backtest_position = shift_position_for_execution(desired)
        self.assertEqual(backtest_position.tolist(), live_position.tolist())

    def test_level_signals_recover_a_missed_fill_where_edge_signals_lose_it(self):
        """A fill that fails during the signal candle must be retried.

        Edge-triggered flags only exist on the transition candle: if the limit
        order does not fill while that candle is current, the trade is lost
        for good (and an unfilled exit would orphan an open position). Level
        flags persist while desired holds, so the next candle retries.
        """
        timestamps = pd.date_range("2026-04-15 00:00", periods=8, freq="15min")
        desired = pd.Series([0, 1, 1, 1, 1, 0, 0, 0], index=timestamps, dtype="int64")
        missed = {2}  # the fill following the 0->1 transition at index 1 fails

        enter_edge, exit_edge = compute_live_transition_signals(desired)
        held_edge = simulate_freqtrade_fills(enter_edge, exit_edge, missed_fills=missed)
        self.assertEqual(held_edge.tolist(), [0] * 8, "edge signals should lose the trade")

        enter_lvl, exit_lvl = compute_live_level_signals(desired)
        held_lvl = simulate_freqtrade_fills(enter_lvl, exit_lvl, missed_fills=missed)
        self.assertEqual(
            held_lvl.tolist(),
            [0, 0, 0, 1, 1, 1, 0, 0],
            "level signals should enter one bar late and still exit on schedule",
        )

    def test_truncated_window_state_needs_the_month_boundary(self):
        """The hysteresis machine restarts flat at the window start.

        Because the machine force-flats at every month end, its state only
        depends on quantiles since the last month boundary. A live kline
        window that reaches past that boundary therefore reproduces the
        full-history state; a window that starts after an in-month entry
        print silently goes false-flat. This pins the invariant behind
        LightGBMStrategy.startup_candle_count.
        """
        timestamps = pd.date_range("2026-01-15 00:00", "2026-02-28 23:45", freq="15min")
        quantiles = pd.Series(95, index=timestamps, dtype="float64")
        quantiles.loc["2026-02-03 10:00"] = 100  # single in-month entry print

        def desired_from(start):
            window = quantiles.loc[start:]
            desired, _ = compute_desired_position(
                window, window.index, "15m", 100, 90, direction="high"
            )
            return desired

        full = desired_from("2026-01-15")
        hold_window = full.loc["2026-02-03 10:00":"2026-02-28 23:30"]
        self.assertTrue(
            (hold_window == 1).all(),
            "full history should hold from the Feb 3 print until month end",
        )
        self.assertEqual(int(full.iloc[-1]), 0, "the month-end bar must force-flat")

        covers_boundary = desired_from("2026-01-25")
        pd.testing.assert_series_equal(
            full.loc["2026-02-01":],
            covers_boundary.loc["2026-02-01":],
            check_names=False,
        )

        misses_print = desired_from("2026-02-05")
        self.assertEqual(
            int(misses_print.sum()),
            0,
            "a window starting after the entry print goes false-flat: this is "
            "the desync startup_candle_count must prevent",
        )

    def test_fee_parity_offline_vs_replay_accounting(self):
        """Offline (fwd1bar) and replay (shifted position) agree with fees on.

        Same trades, same fee charges, same compounded net return - just
        indexed one bar apart. Guards the fee/transition accounting in
        compute_signal_returns against the replay-tool convention.
        """
        n = 400
        fee = 0.0005
        timestamps = pd.date_range("2026-01-01", periods=n, freq="15min")
        t = np.arange(n)
        preds = np.sin(t / 7.0) + 0.3 * np.sin(t / 3.0)
        preds[-5:] = -5.0  # force flat at the end so both paths close out
        ret1bar = np.cos(t / 5.0) * 0.001
        fwd1bar = pd.Series(ret1bar, index=timestamps).shift(-1).fillna(0.0)

        r = compute_signal_returns(
            pd.Series(preds, index=timestamps),
            fwd1bar.to_numpy(),
            timestamps,
            bins=10,
            entry_q=10,
            exit_q=8,
            interval="15m",
            fee=fee,
        )
        offline_net = float(np.prod(1 + r["net_arr"]) - 1)
        trades_offline = r["trades"]

        valid_index = timestamps[r["valid_mask"]]
        desired = pd.Series(r["signal"], index=valid_index)
        position = shift_position_for_execution(desired)
        ret_valid = pd.Series(ret1bar, index=timestamps).reindex(valid_index)
        gross_replay = position.to_numpy() * ret_valid.to_numpy()
        prev = np.concatenate([[0], position.to_numpy()[:-1]])
        tc = np.abs(position.to_numpy() - prev)
        net_replay = gross_replay - tc * fee
        replay_net = float(np.prod(1 + net_replay) - 1)

        self.assertEqual(trades_offline, int(tc.sum()))
        self.assertGreater(trades_offline, 0, "test setup produced no trades")
        # The replay path misses the very last offline contribution (it lands
        # one bar past the series end); the setup forces desired flat at the
        # end so both compound the same bars.
        self.assertAlmostEqual(offline_net, replay_net, places=12)


if __name__ == "__main__":
    unittest.main()
