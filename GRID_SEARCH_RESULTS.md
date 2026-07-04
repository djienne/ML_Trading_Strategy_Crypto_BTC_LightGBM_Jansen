# Grid Search Results

> **Historical record (April 2026).** These runs predate the 2026-06-10
> alpha001 redefinition (expanding → rolling-480 rank, see `STRATEGY.md`), so
> they are not directly comparable with current-code runs — regenerate before
> comparing. The winner was also selected on the same out-of-fold predictions
> these metrics are reported from, so treat the numbers as relative rankings,
> not unbiased forecasts (see `EXPECTED_PERFORMANCE.md`, "Selection Bias In
> Grid-Search Numbers"). For the currently deployed contract values and
> up-to-date backtest results, see `STRATEGY.md` and `EXPECTED_PERFORMANCE.md`.

## Setup
- **Data:** BTC+ETH 15m, Dec 2019 - Mar 2026 (~436K rows)
- **Training:** BTC+ETH, trading BTC only
- **Fixed:** LR=0.01, patience=50, boost_rounds=5000, fee=0.05%, stoploss=-20%
- **Grid:** train_months=[2,4,6,12], leaves=[16,31,63], ff=[0.1,0.25,0.5,0.8,1.0], mdil=[50,100,200,300,500]
- **Signal:** bins=[100-500], entry=[99-100%], exit=[80-92%], IC=[none,>0.0,>0.01], direction=[high,low]
- **Total:** 300 model configs, 75600 signal combos

## Top 20 by Net Return
| # | tm | L | ff | mdil | bins | entry | exit | IC | trades | net | sharpe |
|---|----|----|------|------|------|-------|------|-----|--------|-----|--------|
| 1 | 12 | 31 | 0.5 | 50 | 200 | 99% | 85% | none | 1862 | +299.6% | 1.43 |
| 2 | 12 | 16 | 0.5 | 100 | 200 | 99% | 90% | none | 1380 | +283.8% | 1.41 |
| 3 | 12 | 16 | 0.5 | 100 | 200 | 99% | 85% | none | 1364 | +275.7% | 1.37 |
| 4 | 12 | 16 | 0.5 | 100 | 200 | 99% | 92% | none | 1396 | +253.2% | 1.35 |
| 5 | 12 | 16 | 0.5 | 100 | 150 | 100% | 80% | none | 642 | +250.2% | 1.49 |
| 6 | 12 | 31 | 0.5 | 50 | 200 | 99% | 90% | none | 1880 | +247.4% | 1.31 |
| 7 | 12 | 16 | 0.5 | 100 | 200 | 100% | 80% | none | 522 | +244.2% | 1.53 |
| 8 | 12 | 16 | 0.5 | 100 | 100 | 100% | 85% | none | 904 | +241.2% | 1.37 |
| 9 | 12 | 16 | 0.5 | 100 | 150 | 100% | 85% | none | 644 | +237.2% | 1.46 |
| 10 | 12 | 31 | 0.5 | 50 | 200 | 99% | 92% | none | 1892 | +236.4% | 1.29 |
| 11 | 4 | 16 | 0.5 | 50 | 300 | 99% | 85% | >0.0 | 1938 | +235.9% | 1.23 |
| 12 | 12 | 16 | 0.5 | 100 | 300 | 99% | 85% | none | 1224 | +235.0% | 1.29 |
| 13 | 12 | 16 | 0.5 | 100 | 200 | 100% | 85% | none | 524 | +234.8% | 1.51 |
| 14 | 12 | 16 | 0.5 | 100 | 150 | 100% | 90% | none | 646 | +234.7% | 1.46 |
| 15 | 12 | 16 | 0.5 | 100 | 300 | 99% | 90% | none | 1238 | +234.3% | 1.30 |
| 16 | 12 | 16 | 0.5 | 100 | 200 | 100% | 90% | none | 524 | +232.3% | 1.51 |
| 17 | 12 | 16 | 0.5 | 100 | 100 | 100% | 90% | none | 912 | +228.3% | 1.35 |
| 18 | 4 | 16 | 0.5 | 50 | 300 | 99% | 90% | >0.0 | 1978 | +227.2% | 1.25 |
| 19 | 4 | 16 | 0.5 | 50 | 300 | 99% | 85% | >0.01 | 1932 | +227.2% | 1.21 |
| 20 | 12 | 16 | 0.5 | 100 | 100 | 100% | 80% | none | 896 | +225.8% | 1.30 |

## Best by Training Period
- **2m:** net=+155.6% sharpe=1.11 trades=656 (L=16 ff=0.5 mdil=200 bins=500 e=100% x=80% >0.01)
- **4m:** net=+235.9% sharpe=1.23 trades=1938 (L=16 ff=0.5 mdil=50 bins=300 e=99% x=85% >0.0)
- **6m:** net=+204.4% sharpe=1.29 trades=1716 (L=63 ff=0.25 mdil=300 bins=200 e=99% x=85% >0.01)
- **12m:** net=+299.6% sharpe=1.43 trades=1862 (L=31 ff=0.5 mdil=50 bins=200 e=99% x=85% none)

## Deployed Configuration

The configuration actually deployed (see `STRATEGY.md` for the authoritative
contract table, sourced from `model_info.json`) is
`tm=12, L=31, ff=0.5, mdil=50, bins=100, entry>=100 (top 1%), exit<90, no IC filter`.
Its April-2026 grid metrics were **net +256.7%, Sharpe 1.34, 2350 trades**.

Why this one rather than the raw top-ranked row:

- **Neighborhood robustness:** performance degrades smoothly in every parameter
  direction around it, whereas the top rows sit on sharper local optima.
- **12-month training window:** highest net return and tied-best Sharpe in the
  train-months sweep; more data per fold reduces overfitting, and early
  stopping keeps effective complexity low (typical best iteration 5–30).
- **Long-only, direction `high`:** "buy high" (+256.7%, Sharpe 1.34) vastly
  outperformed "buy low" (+3.4%, Sharpe 0.18).
- **No IC filter:** no filter (+256.7%, 1.34) beat IC>0.0 (+187.4%, 1.25) and
  IC>0.01 (+188.1%, 1.34) — trading through low-validation-IC periods still
  contributed positively on average.
