# Code & Parity Review — LightGBM 15m BTC Strategy

Date: 2026-06-10. Scope: code quality, freqtrade implementation, and equivalence
between the live/dry-run bot and the custom backtest/optimization pipeline.
All findings were verified empirically where possible; fixes applied in the same
change set are marked **[FIXED]**.

## Verdict

The architecture is sound and the parity story is genuinely good: all four
execution paths (research backtest, live strategy, retrainer, replay tool)
share `src/features.py`, `src/signal_engine.py` and `src/strategy_contract.py`
— no duplicated signal logic anywhere. The measured result backs this up:
**after the 1-bar parity fix (`aef0d1a`), 18 of 20 dry-run trades matched the
artifact replay bar-exactly, with mean |PnL diff| of 0.12 pp per trade.**

The review found one latent live-correctness hazard, one structural feature
parity break, and one execution-robustness gap — all fixed — plus several
documented approximations that are acceptable but should stay visible.

## Measured Evidence

1. **Dry-run ledger (48 trades, 2026-04-15 → 06-10):** all trades 1–3 bars
   long, all closed by `exit_signal`, none crossing month boundaries, no
   stoploss hits. All 48 entry orders filled; 9/48 exits needed at least one
   30s-timeout cancel/re-place (passive `price_side: same` limit orders).
2. **Replay vs ledger:** pre-fix era (Apr 15–May 31): all 28 trades offset by
   exactly +1 bar — the old live code had a 2-bar effective delay; the replay
   runs today's 1-bar code. Post-fix era (Jun 1–10): 18/20 bar-exact matches,
   1 trade offset, 1 trade missing from replay (both on Jun 10 — knife-edge
   flips, see F2).
3. **Hysteresis window scan (5.3 years of research OOF predictions, 1,137
   trades):** median trade 2 bars, p99 31 bars; the worst gap between a held
   bar and its latest q≥100 print was 418 bars (4.4 days). Zero desyncs would
   have occurred at the old ~1050-bar live window — F1 was latent, not active.
4. **alpha001 window drift through the deployed model (Jan–Jun 2026, 15,360
   bars):** with the old expanding rank, live (~1050-bar window) vs research
   (full history) flipped ~25 bars at the q≥100 entry boundary and changed
   ~4% of entries (145 → 140), despite a mean feature delta of only 0.009 and
   alpha001 having ≈0 gain importance in the deployed 13-tree model. Lesson:
   the top-1% entry threshold is a knife-edge — feature parity must be exact,
   not approximate.

## Findings

### F1 — Hysteresis state restarts flat at the kline-window start **[FIXED]**

`_hysteresis_signal` (src/signal_engine.py) starts at `pos=0` wherever the
dataframe starts. Freqtrade live/dry-run only supplies
`ohlcv_candle_limit + startup_candle_count` candles (verified in freqtrade
2025.7 `exchange.py`: `ohlcv_df.tail(candle_limit + self._startup_candle_count)`);
with the old `startup_candle_count = 50` that was ~1050 bars ≈ 10.9 days. The
machine's only reset point is the month-end force-flat, so whenever the window
failed to reach the last month boundary, live state could silently desync
(false-flat only): missed exits, including the month-end force-flat, leaving an
open trade protected only by the −20% stoploss.

Measurement (3) shows zero historical occurrences with current contract
parameters — but the margin was ~2.5×, regime-dependent, and the failure mode
is silent. **Fix:** `startup_candle_count = 3600` (31-day month coverage 2976 +
alpha001 window 480 + warmup ~44 + margin), giving a ~4600-bar live window that
provably always covers the reset point. 4 OHLCV startup calls (freqtrade cap
is 5). Pinned by `test_truncated_window_state_needs_the_month_boundary`.

### F2 — alpha001 was window-length dependent **[FIXED — requires retrain]**

`alpha001` used an *expanding* percentile rank from dataframe start, so the
same candle had different values in research (history since 2020), live
(~1050 bars) and replay (200 warmup bars). Measurement (4) quantified the
impact: ~4% of entries flipped. **Fix:** fixed rolling rank window
(`ALPHA001_RANK_WINDOW = 480` in src/features.py) — the value at the decision
bar is now identical everywhere once ~524 warmup bars exist, pinned by
`tests/test_feature_window_parity.py`. The replay warmup was raised to match
(`FEATURE_WARMUP_BARS = 3600`).

Consequences to be aware of:
- The deployed model was trained on the old expanding alpha001. The retrainer
  auto-detects the `features.py` change (mtime) and will fully retrain within
  its hourly poll. **Restart the retrainer first, wait for the publish
  (~15 min), then restart the bot** — a bot restart before the new model lands
  would briefly pair new-feature code with the old model (feature names are
  unchanged, so nothing would reject the mismatch).
- Research artifacts (`data/processed`, `data/predictions`, `models/`,
  grid-search results) predate the new definition; regenerate before comparing
  numbers against new runs.
- Replays of windows before this change now carry a (measured, tiny) alpha001
  definition drift — same caveat as any code change: archives capture models,
  not code.

### F3 — Edge-triggered signals could permanently lose a missed fill **[FIXED]**

`enter_long`/`exit_long` fired only on the desired-position transition candle.
With passive limit orders (`price_side: "same"`, top-of-book, 30s timeout), a
fill that doesn't happen while the signal candle is current is lost: a missed
entry skips the trade (backtest assumes a guaranteed fill at next open); a
missed exit orphans an open position. The ledger shows 9/48 exits already
needed timeout retries within their candle. **Fix:** level-based flags
(`compute_live_level_signals`): `enter_long` while `desired == 1`, `exit_long`
while `desired == 0` (NaN bars publish nothing, so warmup/data hiccups can
never force an exit). Freqtrade's own trade state arbitrates, fills self-heal
on the next candle, and held bars still satisfy `position[T+1] = desired[T]`
(pinned by `test_level_signals_match_shift_contract` and
`test_level_signals_recover_a_missed_fill_where_edge_signals_lose_it`).

### F4 — Replay tool cannot validate windows that predate code changes

Archived snapshots capture models/contracts, not code. The pre-/post-`aef0d1a`
offset in measurement (2) is the concrete example. A docstring caveat was added
to `backtest_live_window.py`. If exact auditability matters later, archive a
git SHA inside `model_info.json` at publish time and warn on mismatch during
replay.

### F5 — Live per-candle state was not persisted **[FIXED]**

Live-vs-replay diffs previously had to be reconstructed from the trade ledger.
The strategy now appends each decision bar's `(date, model_version, prediction,
quantile, desired_position)` to `user_data/logs/signal_state.csv`, enabling
exact future diffs (including quantile knife-edge analysis).

### F6 — Known, accepted approximations (unchanged, documented)

- **Stoploss:** vectorized backtest triggers on cumulative open→close returns,
  not intrabar lows (documented in `_apply_stoploss`). No dry-run stoploss has
  fired to date.
- **Fees/fills:** backtest charges a flat fee per side at next-bar open
  (0.05% at the time of this review; 0.03% default since 2026-07-04); live uses
  maker-side limit orders (cheaper when filled, ~0.12 pp/trade observed fill
  noise). Funding fees are not modeled (see EXPECTED_PERFORMANCE.md).
- **Month-start handoff:** research OOF predictions skip the first 20 bars of
  each month (embargo) and switch fold models exactly at month start; live
  trades the previous model until the retrainer publishes (minutes to hours
  into day 1). The replay tool models this correctly via `live_from`.
- **Quantile-window content:** live seeds rolling quantiles with artifact OOF
  history plus current-model predictions for the kline window; research uses
  pure OOF predictions. Contributes to the knife-edge flip class; bounded and
  now observable via the signal-state log.

### F7 — Research-side notes

- **Grid-search selection bias:** the winner is selected on the same OOF
  predictions its headline metrics are reported from (no untouched holdout).
  Caveat added to EXPECTED_PERFORMANCE.md.
- **`--quantile-scope global` lookahead:** ranks against the full sample
  including future bars; a loud warning now prints (src/evaluation.py). The
  default `auto` resolves to safe scopes; the hysteresis backtest path never
  uses it.
- **Retrainer constant drift [FIXED]:** retrain.py now imports the signal
  contract values (bins, quantiles, direction, stoploss, fee, method,
  train months, interval, symbol) from `src/strategy_contract.py` instead of
  duplicating literals.
- Mid-month retrains are triggered by source-file mtime changes (9 archives in
  2 months vs 3 monthly ones). This is by design, but be aware every src edit
  on the host swaps the live model within the hour while the bot keeps its
  in-memory code until restarted.

### Positives worth keeping

- Single shared signal kernel + contract validation across all four consumers.
- Atomic model handoff (archive snapshot + `current` pointer + `os.replace`),
  predictions-first publish order, post-save model verification.
- The 1-bar execution convention is documented in code and pinned by tests;
  the commit history shows it was empirically debugged and the fix measurably
  worked.
- Honest in-code documentation of approximations (stoploss, fees).
- Calendar-month rolling CV with per-symbol embargo; per-symbol feature
  grouping prevents cross-symbol contamination.

## Deployment checklist for this change set

> **Historical record (June 2026) — already executed; do not re-run.** This was
> the one-time rollout runbook for the F1/F2/F3 fixes above. It was carried out
> in June 2026 and the stack has retrained many times since (the deployed
> `model_info.json` training date is 2026-08-13). It is kept only as the record
> of how that change set was rolled out. Note the ordering rule it encodes —
> retrainer first, wait for the publish, then the trader — still applies to any
> future `src/features.py` change. `docker compose` takes the **service** names
> (`retrainer`, `freqtrade`); `lgbm_retrainer` / `lgbm_trader` are the container
> names and only work with `docker logs` / `docker exec`.

1. `docker compose restart retrainer` — it detects the `features.py`
   change and retrains (~15 min based on the 06-10 run).
2. Wait for the new `current` pointer / `model_info.json` training date.
3. `docker compose restart freqtrade` — picks up the new strategy code
   (level signals, `startup_candle_count = 3600`; expect a one-time
   "Using 4 calls to get OHLCV" warning at startup).
4. Confirm `user_data/logs/signal_state.csv` starts appending one row per
   15m candle.
5. After a few days: re-run
   `python freqtrade_live/backtest_live_window.py --start <restart date>` and
   diff against the ledger; entries/exits should be bar-exact except documented
   knife-edge flips, which can now be root-caused from the signal-state log.
