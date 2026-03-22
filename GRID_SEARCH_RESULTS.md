# Grid Search Results — LightGBM BTC Trading Strategy

## Setup
- **Symbol**: BTCUSDT (Binance Futures)
- **Data**: Dec 2019 → Mar 2026
- **Training**: Rolling CV — 12 calendar months train, 1 month test
- **Validation**: Last 10% of training set for early stopping
- **Features (20)**: ret1bar-ret10bar, bop, cci, mfi, rsi, stochrsi, slowk, slowd, natr, alpha054, alpha001
- **Target**: fwd1bar (next bar's return: Close(t+1)/Open(t+1) - 1)
- **Signal**: Long when quantile == top bin, Short when quantile == bottom bin (expanding scope)
- **Seed**: 42 (deterministic)

---

## Round 1: Cross-timeframe sweep (fee=0.05%, long+short, 10000 max rounds)

| TF | LR | Pat | Bins | Trades | Gross | Net | Sharpe |
|----|------|------|------|--------|-------|------|--------|
| 15m | 0.10 | 50 | 200 | 686 | +109% | **+49%** | 0.68 |
| 15m | 0.10 | 50 | 100 | 1,428 | +157% | +26% | 0.38 |
| 15m | 0.10 | 50 | 500 | 308 | +42% | +22% | 0.50 |
| 15m | 0.01 | 200 | 200 | 1,124 | +97% | +13% | 0.24 |
| 15m | 0.05 | 100 | 200 | 730 | +51% | +5% | 0.14 |
| 5m | 0.10 | 50 | 500 | 598 | +38% | +2% | 0.09 |
| 1m | all configs | all | all | many | high gross | **-100%** | negative |
| 3m | all configs | all | all | many | high gross | **negative** | negative |

**Key finding**: 15m candles are the only profitable timeframe at 0.05% fee. 1m and 3m candles generate too many trades.

---

## Round 2: Extended 15m grid (fee=0.05%, long+short, 10000 max rounds)

### num_leaves=16

| LR | Pat | Best Bins | Net | Trades | Sharpe | Avg Iter |
|------|------|-----------|------|--------|--------|----------|
| 0.005 | 300 | 200 | +32% | 724 | 0.49 | 59 |
| 0.005 | 50 | 300 | +20% | 758 | 0.34 | 19 |
| 0.030 | 50 | 300 | +32% | 910 | 0.47 | 12 |
| 0.100 | 20 | 200 | +29% | 776 | 0.44 | 5 |
| **0.150** | **50** | **150** | **+54%** | **1,350** | **0.62** | **5** |
| 0.150 | 30 | 150 | +43% | 1,268 | 0.53 | 4 |

### num_leaves=31

| LR | Pat | Best Bins | Net | Trades | Sharpe | Avg Iter |
|------|------|-----------|------|--------|--------|----------|
| 0.020 | 200 | 100 | +48% | 1,470 | 0.61 | 17 |
| 0.020 | 100 | 100 | +46% | 1,472 | 0.61 | 16 |
| **0.100** | **100** | **150** | **+57%** | **946** | **0.73** | **5** |
| 0.100 | 50 | 150 | +50% | 902 | 0.67 | 5 |
| 0.100 | 30 | 200 | +49% | 670 | 0.68 | 5 |
| 0.200 | 20 | 200 | +57% | 936 | 0.68 | 3 |

### num_leaves=63 (partial — still running)

| LR | Pat | Best Bins | Net | Trades | Sharpe | Avg Iter |
|------|------|-----------|------|--------|--------|----------|
| 0.005 | 100 | 100 | +37% | 1,326 | 0.54 | 43 |
| 0.005 | 300 | 200 | +33% | 682 | 0.53 | 56 |
| (more results pending...) | | | | | | |

---

## Best Results So Far

| Rank | Leaves | LR | Pat | Bins | Trades | Net | Sharpe | Iter |
|------|--------|------|------|------|--------|------|--------|------|
| 1 | 31 | 0.10 | 100 | 150 | 946 | **+57%** | **0.73** | 5 |
| 2 | 31 | 0.20 | 20 | 200 | 936 | +57% | 0.68 | 3 |
| 3 | 16 | 0.15 | 50 | 150 | 1,350 | +54% | 0.62 | 5 |
| 4 | 31 | 0.10 | 50 | 150 | 902 | +50% | 0.67 | 5 |
| 5 | 31 | 0.10 | 30 | 200 | 670 | +49% | 0.68 | 5 |
| 6 | 31 | 0.20 | 50 | 100 | 1,604 | +50% | 0.57 | 3 |
| 7 | 31 | 0.02 | 200 | 100 | 1,470 | +48% | 0.61 | 17 |

---

## Round 3: Long-Only with Hysteresis Entry/Exit (15m, lr=0.10, leaves=31, fee=0.05%)

Best config from Round 2 (leaves=31, lr=0.10, patience=100), tested long-only with various entry/exit percentile thresholds. Hysteresis: once long, stay long until quantile drops below exit threshold.

### Top Results

| Bins | Entry (>=) | Exit (<) | Trades | Gross | **Net** | **Sharpe** |
|------|------------|----------|--------|-------|---------|-----------|
| **200** | **198 (top 1%)** | **180 (top 10%)** | **1,096** | **+250%** | **+102%** | **0.98** |
| 200 | 198 (top 1%) | 100 (top 50%) | 1,066 | +235% | +97% | 0.91 |
| 200 | 198 (top 1%) | 160 (top 20%) | 1,076 | +236% | +96% | 0.93 |
| 200 | 198 (top 1%) | 196 (top 2%) | 1,190 | +250% | +93% | 0.96 |
| 100 | 100 (top 1%) | 90 (top 10%) | 624 | +158% | +89% | 0.95 |
| 100 | 100 (top 1%) | 50 (top 50%) | 604 | +153% | +87% | 0.91 |
| 500 | 495 (top 1%) | 475 (top 5%) | 812 | +175% | +84% | 0.91 |
| 100 | 99 (top 2%) | 90 (top 10%) | 1,562 | +260% | +65% | 0.69 |

### Key Findings
- **Long-only is dramatically better** than long+short (+102% vs +57%)
- **Hysteresis reduces whipsaws**: stay long until prediction drops significantly
- **Top 1% entry, exit at top 10%** is the sweet spot (bins=200, entry>=198, exit<180)
- More selective entry (top 0.5%) reduces trades but keeps high returns
- Less selective entry (top 2%+) increases trades but fee drag hurts

## Round 4: Broader Long-Only Sweep (15m + 5m, varied LR/leaves/bins/thresholds)

8 LR/patience/leaves combos x 2 timeframes x 4 bin sizes x 7 entry/exit thresholds.
All 5m configs were net negative. All top 20 results are 15m.

### Top 10

| Leaves | LR | Pat | Bins | Entry | Exit | Trades | Gross | **Net** | **Sharpe** |
|--------|------|------|------|-------|------|--------|-------|---------|-----------|
| 16 | 0.15 | 50 | 150 | top bin | <90% | 552 | +230% | **+151%** | **1.28** |
| 16 | 0.15 | 50 | 150 | top bin | <80% | 546 | +223% | +146% | 1.24 |
| 31 | 0.20 | 20 | 200 | >=99% | <90% | 868 | +258% | +132% | 1.13 |
| 16 | 0.15 | 50 | 200 | >=99% | <80% | 1,058 | +293% | +131% | 1.07 |
| 63 | 0.10 | 100 | 200 | >=99% | <90% | 1,024 | +285% | +130% | 1.19 |
| 16 | 0.15 | 50 | 200 | >=99% | <90% | 1,072 | +290% | +128% | 1.07 |
| 31 | 0.20 | 20 | 100 | top bin | <90% | 608 | +209% | +128% | 1.14 |
| 16 | 0.15 | 50 | 100 | top bin | <90% | 786 | +234% | +125% | 1.10 |
| 63 | 0.10 | 100 | 200 | >=99% | <80% | 992 | +267% | +123% | 1.12 |
| 16 | 0.15 | 50 | 200 | top bin | <90% | 446 | +178% | +123% | 1.15 |

## Round 5: Expanded regularization grid (15m only, fee=0.05%, long-only)

Tested subsample, feature_fraction, min_data_in_leaf, training window. 112 model configs x 33 signal configs.

### Key Findings
- **feature_fraction=0.5** beats 0.8 (more regularization helps)
- **min_data_in_leaf=100** beats 200
- **No bagging needed** (subsample=1.0 dominates)
- **12-month training window** optimal (6m/18m both worse)
- **Exit at 92%** slightly better than 90%

### Top 5
| L | LR | ff | mdil | Bins | Entry | Exit | Trades | Net | Sharpe |
|---|----|----|------|------|-------|------|--------|-----|--------|
| 16 | 0.15 | 0.5 | 100 | 150 | top bin | <92% | 642 | +166% | 1.41 |
| 16 | 0.15 | 0.5 | 100 | 100 | top bin | <92% | 948 | +164% | 1.32 |
| 16 | 0.15 | 0.5 | 100 | 200 | top bin | <92% | 472 | +157% | 1.42 |
| 16 | 0.15 | 0.8 | 200 | 150 | top bin | <88% | 552 | +156% | 1.30 |
| 16 | 0.15 | 0.5 | 100 | 100 | top bin | <90% | 940 | +154% | 1.27 |

---

## Best Overall Configuration (deployed)

| Parameter | Value |
|-----------|-------|
| Timeframe | **15m** |
| Training symbols | **BTCUSDT + ETHUSDT** (trade only BTC) |
| Learning Rate | **0.15** |
| Patience | **50** |
| num_leaves | **16** |
| min_data_in_leaf | **100** |
| feature_fraction | **0.5** |
| Training window | **12 calendar months** |
| Embargo | **20 bars** between train/test months |
| Bins | **150** |
| Entry | **quantile == 150 (top bin)** |
| Exit | **quantile < 138 (drops below top 8%)** |
| Direction | **Long-only** |
| Monthly boundaries | **1-hour grace at month start, force close at month end** |
| Fee | 0.05% |

BTC-only training backtest: +170% net, Sharpe 1.44 (higher but more overfitting risk).
BTC+ETH training backtest: +138% net, Sharpe ~1.25 (chosen for robustness).
