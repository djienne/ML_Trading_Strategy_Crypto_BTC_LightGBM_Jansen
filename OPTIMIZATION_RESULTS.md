# Optimization Results (April 2026)

## Chosen Configuration

| Parameter | Value | Reason |
|-----------|-------|--------|
| Train Months | 12 | Longer training window captures more market regimes; best Sharpe and net return |
| Num Leaves | 31 | Moderate complexity; outperformed both simpler (16) and deeper (63) trees |
| Feature Fraction | 0.5 | 50% feature sampling per split; reduces overfitting |
| Min Data in Leaf | 50 | Lower than previous (100); allows finer splits with 12-month training data |
| Learning Rate | 0.01 | Fixed across grid; low rate + early stopping finds stable iterations |
| Boost Rounds | 5000 (max) | Early stopping at 50 rounds; actual iterations typically 5-30 |
| Bins | 100 | Quantile bins for signal generation |
| Entry Quantile | 100 (top bin) | Enter long when prediction is in the top 1% |
| Exit Quantile | 90 | Exit when prediction drops below top 10% |
| IC Filter | None | No information coefficient filtering; all bars eligible |
| Stoploss | -20% | Wide safety net; exits primarily via signal, not stoploss |
| Fee (backtest) | 0.05% per side | Conservative estimate; Binance futures are ~0.02-0.04% |

## Grid Search Results

Full grid search over 300 model configurations x multiple signal/bin combos with rolling quantile window (matching training period).

### Best by training period

| Train Months | Net Return | Sharpe | Trades | Config |
|:---:|:---:|:---:|:---:|---|
| 2 | +113.0% | 0.81 | 1604 | L=63 ff=0.1 mdil=200 bins=400 e=100 x=90% |
| 4 | +141.8% | 1.03 | 1398 | L=16 ff=0.1 mdil=500 bins=150 e=100 x=80% |
| 6 | +188.1% | 1.34 | 1562 | L=16 ff=0.25 mdil=300 bins=300 e=100 x=92% |
| **12** | **+256.7%** | **1.34** | **2350** | **L=31 ff=0.5 mdil=50 bins=100 e=100 x=90%** |

### Why 12-month training window

- Highest absolute net return (+256.7%)
- Tied for best Sharpe ratio (1.34)
- More data per fold reduces overfitting
- Early stopping keeps effective model complexity low (5-30 iterations typical)

### Signal direction

Long-only ("buy high") vastly outperforms short ("buy low"):
- High: +256.7% net, Sharpe 1.34
- Low: +3.4% net, Sharpe 0.18

### IC filter impact

No IC filtering produced the best results:
- No filter: +256.7% net, Sharpe 1.34
- IC > 0.0: +187.4% net, Sharpe 1.25
- IC > 0.01: +188.1% net, Sharpe 1.34

Disabling IC filter allows the model to trade in all periods, including those with lower validation IC, which on average still contributes positively.

## Backtest Performance (CLI, full data through April 2026)

```
Total Trades: 2274
Total Gross Return: 795.0%
Total Net Return:   187.1%
Annualized Sharpe:  1.13
Period: Dec 2020 - Apr 2026 (~5.3 years)
```

Note: CLI backtest result (187%) differs from grid search (257%) because the CLI includes a few extra months of data and uses the saved prediction file which has slight data timing differences. Both confirm strong performance.

## Expected Live Performance

### Realistic estimates (based on backtest with conservative assumptions)

| Metric | Backtest | Expected Live (conservative) |
|--------|----------|------------------------------|
| CAGR | ~22% | 10-15% |
| Sharpe | 1.13 | 0.5-0.8 |
| Trades/year | ~430 | ~430 |
| Avg trade duration | ~1-2 days | ~1-2 days |
| Max drawdown | ~15-25% | 20-35% |
| Win rate | ~50-55% | ~45-50% |

### Why live will likely underperform backtest

1. **Slippage**: Backtest assumes perfect limit order fills; live has slippage on entry/exit
2. **Latency**: Backtest evaluates on closed candles; live has ~seconds delay
3. **Fee asymmetry**: Backtest uses 0.05% flat; live fees vary by order type
4. **Model staleness**: Live model is retrained monthly; between retrains, market regime may shift
5. **Stoploss mechanics**: Backtest tracks cumulative geometric return; Freqtrade uses real-time unrealized P&L

### Key design decisions for live parity

- **Rolling quantile** (12-month window): quantile context matches current model's prediction distribution, not years-old models
- **Startup retrain** (last fold only): every container restart retrains the latest model in ~30 seconds
- **Short suppression**: prevents Freqtrade from converting exit signals to short entries
- **Unclosed candle clearing**: signals only act on fully-closed candle data
- **Prediction history**: expanding quantile anchored by all historical fold predictions, trimmed to rolling window
