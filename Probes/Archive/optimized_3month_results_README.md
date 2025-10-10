# 3-Month Backtesting Results

## Performance Summary

| Ticker | Total Return | Sharpe Ratio | Max Drawdown | Win Rate | Trades |
|--------|-------------|--------------|--------------|----------|--------|
| **TSLA** | 17.74% | 2.246 | -9.00% | 100.0% | 2 |
| **BA** | -4.09% | -0.908 | -9.38% | 50.0% | 6 |
| **BTC-USD** | -1.80% | -0.168 | -15.55% | 54.5% | 11 |

## Overall Performance
- **Average Return**: 3.95%
- **Average Sharpe**: 0.390
- **Total Trades**: 19

## Generated Files
- `optimized_3month_backtest_results.json` - Complete results
- `optimized_quantstats_[TICKER].png` - Combined QuantStats charts (3 subplots each)
- `optimized_trading_chart_[TICKER].png` - Price + Buy/Sell signals

## Strategy Details
- **Time Period**: 3 months (July 9, 2025 - October 7, 2025)
- **Confidence Threshold**: 0.3
- **Models**: ProbeTrain + SAE (synthetic fallback)
- **Signal Logic**: Buy when both models agree on positive sentiment, Sell when they disagree

## Trading Log

### TSLA Trading Decisions

| Date | Action | Price | ProbeTrain Prob | SAE Prob | Reason | News Headline |
|------|--------|-------|----------------|----------|--------|---------------|
| 2025-07-10 | BUY | $309.87 | 0.346 | 0.479 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.479) | Dell leads S&P 500 gains, Nvidia CEO confirms xAI investment... |
| 2025-07-24 | BUY | $305.30 | 0.346 | 0.405 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.405) | Dell leads S&P 500 gains, Nvidia CEO confirms xAI investment... |
| 2025-08-01 | BUY | $302.63 | 0.340 | 0.746 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.746) | Elon Musk’s cheaper new Teslas aren’t much of a bargain... |
| 2025-08-07 | BUY | $322.27 | 0.346 | 0.794 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.794) | Dell leads S&P 500 gains, Nvidia CEO confirms xAI investment... |
| 2025-08-15 | BUY | $330.56 | 0.340 | 0.673 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.673) | Elon Musk’s cheaper new Teslas aren’t much of a bargain... |
| 2025-08-21 | BUY | $320.11 | 0.346 | 0.661 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.661) | Dell leads S&P 500 gains, Nvidia CEO confirms xAI investment... |
| 2025-08-29 | BUY | $333.87 | 0.340 | 0.869 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.869) | Elon Musk’s cheaper new Teslas aren’t much of a bargain... |
| 2025-09-05 | SELL | $350.84 | 0.346 | 0.461 | SELL: Models disagree - ProbeTrain: 1 (0.346), SAE: -1 (0.461) | Dell leads S&P 500 gains, Nvidia CEO confirms xAI investment... |
| 2025-09-15 | BUY | $410.04 | 0.340 | 0.722 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.722) | Elon Musk’s cheaper new Teslas aren’t much of a bargain... |
| 2025-09-19 | BUY | $426.07 | 0.346 | 0.482 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.482) | Dell leads S&P 500 gains, Nvidia CEO confirms xAI investment... |
| 2025-10-03 | SELL | $429.83 | 0.346 | 0.589 | SELL: Models disagree - ProbeTrain: 1 (0.346), SAE: -1 (0.589) | Dell leads S&P 500 gains, Nvidia CEO confirms xAI investment... |

### BA Trading Decisions

| Date | Action | Price | ProbeTrain Prob | SAE Prob | Reason | News Headline |
|------|--------|-------|----------------|----------|--------|---------------|
| 2025-07-11 | BUY | $226.84 | 0.346 | 0.751 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.751) | Air Lease Announces Encouraging Activity Update for Q3 2025... |
| 2025-07-15 | BUY | $230.00 | 0.336 | 0.641 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.336, SAE: 0.641) | Boeing 737 loses status as world’s most popular jet... |
| 2025-07-16 | BUY | $229.90 | 0.352 | 0.611 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.352, SAE: 0.611) | Company News for Oct 7, 2025... |
| 2025-07-17 | SELL | $231.00 | 0.339 | 0.787 | SELL: Models disagree - ProbeTrain: 1 (0.339), SAE: -1 (0.787) | How Investors May Respond To Boeing (BA) Ramping Up 737 MAX Production and Secur... |
| 2025-07-21 | SELL | $229.32 | 0.340 | 0.464 | SELL: Models disagree - ProbeTrain: 1 (0.340), SAE: -1 (0.464) | Airbus A320 flies past Boeing 737 as most-delivered jet in history... |
| 2025-07-25 | BUY | $233.06 | 0.346 | 0.796 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.796) | Air Lease Announces Encouraging Activity Update for Q3 2025... |
| 2025-07-31 | BUY | $221.84 | 0.339 | 0.651 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.651) | How Investors May Respond To Boeing (BA) Ramping Up 737 MAX Production and Secur... |
| 2025-08-04 | BUY | $222.34 | 0.340 | 0.687 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.687) | Airbus A320 flies past Boeing 737 as most-delivered jet in history... |
| 2025-08-08 | BUY | $229.12 | 0.346 | 0.521 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.521) | Air Lease Announces Encouraging Activity Update for Q3 2025... |
| 2025-08-12 | SELL | $232.61 | 0.336 | 0.696 | SELL: Models disagree - ProbeTrain: 1 (0.336), SAE: -1 (0.696) | Boeing 737 loses status as world’s most popular jet... |
| 2025-08-13 | BUY | $233.37 | 0.352 | 0.763 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.352, SAE: 0.763) | Company News for Oct 7, 2025... |
| 2025-08-18 | BUY | $232.41 | 0.340 | 0.599 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.599) | Airbus A320 flies past Boeing 737 as most-delivered jet in history... |
| 2025-08-22 | BUY | $230.12 | 0.346 | 0.527 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.527) | Air Lease Announces Encouraging Activity Update for Q3 2025... |
| 2025-08-26 | SELL | $234.83 | 0.336 | 0.424 | SELL: Models disagree - ProbeTrain: 1 (0.336), SAE: -1 (0.424) | Boeing 737 loses status as world’s most popular jet... |
| 2025-08-27 | SELL | $235.62 | 0.352 | 0.649 | SELL: Models disagree - ProbeTrain: 1 (0.352), SAE: -1 (0.649) | Company News for Oct 7, 2025... |
| 2025-09-08 | BUY | $230.95 | 0.346 | 0.470 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.470) | Air Lease Announces Encouraging Activity Update for Q3 2025... |
| 2025-09-10 | SELL | $227.52 | 0.336 | 0.533 | SELL: Models disagree - ProbeTrain: 1 (0.336), SAE: -1 (0.533) | Boeing 737 loses status as world’s most popular jet... |
| 2025-09-11 | BUY | $219.99 | 0.352 | 0.450 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.352, SAE: 0.450) | Company News for Oct 7, 2025... |
| 2025-09-12 | BUY | $215.94 | 0.339 | 0.451 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.451) | How Investors May Respond To Boeing (BA) Ramping Up 737 MAX Production and Secur... |
| 2025-09-24 | BUY | $215.10 | 0.336 | 0.690 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.336, SAE: 0.690) | Boeing 737 loses status as world’s most popular jet... |
| 2025-09-25 | SELL | $213.53 | 0.352 | 0.701 | SELL: Models disagree - ProbeTrain: 1 (0.352), SAE: -1 (0.701) | Company News for Oct 7, 2025... |
| 2025-09-26 | BUY | $221.26 | 0.339 | 0.885 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.885) | How Investors May Respond To Boeing (BA) Ramping Up 737 MAX Production and Secur... |
| 2025-09-30 | BUY | $215.83 | 0.340 | 0.563 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.563) | Airbus A320 flies past Boeing 737 as most-delivered jet in history... |
| 2025-10-06 | BUY | $219.73 | 0.346 | 0.722 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.722) | Air Lease Announces Encouraging Activity Update for Q3 2025... |

### BTC-USD Trading Decisions

| Date | Action | Price | ProbeTrain Prob | SAE Prob | Reason | News Headline |
|------|--------|-------|----------------|----------|--------|---------------|
| 2025-07-11 | SELL | $117516.99 | 0.346 | 0.616 | SELL: Models disagree - ProbeTrain: 1 (0.346), SAE: -1 (0.616) | Why the bitcoin trade 'is too large to ignore'... |
| 2025-07-12 | BUY | $117435.23 | 0.342 | 0.611 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.611) | Bitcoin as a reserve currency: What is the 'debasement' trade?... |
| 2025-07-13 | SELL | $119116.12 | 0.346 | 0.753 | SELL: Models disagree - ProbeTrain: 1 (0.346), SAE: -1 (0.753) | Market takeaways: Volatility creeps up, bitcoin check-in... |
| 2025-07-15 | SELL | $117777.19 | 0.344 | 0.770 | SELL: Models disagree - ProbeTrain: 1 (0.344), SAE: -1 (0.770) | If You'd Invested $10,000 in Bitcoin 5 Years Ago, Here's How Much You'd Have Tod... |
| 2025-07-17 | BUY | $119289.84 | 0.349 | 0.712 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.712) | Tech, Media & Telecom Roundup: Market Talk... |
| 2025-07-19 | SELL | $117939.98 | 0.345 | 0.559 | SELL: Models disagree - ProbeTrain: 1 (0.345), SAE: -1 (0.559) | BlackRock dominates all ETFs with $3.5bn haul: ‘that’s how hungry the fish are’... |
| 2025-07-21 | BUY | $117439.54 | 0.346 | 0.425 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.425) | Why the bitcoin trade 'is too large to ignore'... |
| 2025-07-23 | BUY | $118754.96 | 0.346 | 0.533 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.533) | Market takeaways: Volatility creeps up, bitcoin check-in... |
| 2025-07-25 | BUY | $117635.88 | 0.344 | 0.854 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.854) | If You'd Invested $10,000 in Bitcoin 5 Years Ago, Here's How Much You'd Have Tod... |
| 2025-07-27 | SELL | $119448.49 | 0.349 | 0.690 | SELL: Models disagree - ProbeTrain: 1 (0.349), SAE: -1 (0.690) | Tech, Media & Telecom Roundup: Market Talk... |
| 2025-07-31 | BUY | $115758.20 | 0.346 | 0.668 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.668) | Why the bitcoin trade 'is too large to ignore'... |
| 2025-08-01 | BUY | $113320.09 | 0.342 | 0.801 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.801) | Bitcoin as a reserve currency: What is the 'debasement' trade?... |
| 2025-08-02 | BUY | $112526.91 | 0.346 | 0.683 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.683) | Market takeaways: Volatility creeps up, bitcoin check-in... |
| 2025-08-06 | BUY | $115028.00 | 0.349 | 0.653 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.653) | Tech, Media & Telecom Roundup: Market Talk... |
| 2025-08-08 | SELL | $116688.73 | 0.345 | 0.411 | SELL: Models disagree - ProbeTrain: 1 (0.345), SAE: -1 (0.411) | BlackRock dominates all ETFs with $3.5bn haul: ‘that’s how hungry the fish are’... |
| 2025-08-10 | BUY | $119306.76 | 0.346 | 0.525 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.525) | Why the bitcoin trade 'is too large to ignore'... |
| 2025-08-12 | BUY | $120172.91 | 0.346 | 0.654 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.654) | Market takeaways: Volatility creeps up, bitcoin check-in... |
| 2025-08-14 | BUY | $118359.58 | 0.344 | 0.762 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.762) | If You'd Invested $10,000 in Bitcoin 5 Years Ago, Here's How Much You'd Have Tod... |
| 2025-08-15 | SELL | $117398.35 | 0.351 | 0.628 | SELL: Models disagree - ProbeTrain: 1 (0.351), SAE: -1 (0.628) | Bit Digital Ethereum Treasury Soars to $675M, Cementing Spot as Top 6 Holder... |
| 2025-08-16 | BUY | $117491.35 | 0.349 | 0.766 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.766) | Tech, Media & Telecom Roundup: Market Talk... |
| 2025-08-18 | SELL | $116252.31 | 0.345 | 0.646 | SELL: Models disagree - ProbeTrain: 1 (0.345), SAE: -1 (0.646) | BlackRock dominates all ETFs with $3.5bn haul: ‘that’s how hungry the fish are’... |
| 2025-08-21 | SELL | $112419.03 | 0.342 | 0.763 | SELL: Models disagree - ProbeTrain: 1 (0.342), SAE: -1 (0.763) | Bitcoin as a reserve currency: What is the 'debasement' trade?... |
| 2025-08-22 | BUY | $116874.09 | 0.346 | 0.745 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.745) | Market takeaways: Volatility creeps up, bitcoin check-in... |
| 2025-08-24 | BUY | $113458.43 | 0.344 | 0.616 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.616) | If You'd Invested $10,000 in Bitcoin 5 Years Ago, Here's How Much You'd Have Tod... |
| 2025-08-25 | BUY | $110124.35 | 0.351 | 0.626 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.351, SAE: 0.626) | Bit Digital Ethereum Treasury Soars to $675M, Cementing Spot as Top 6 Holder... |
| 2025-08-26 | BUY | $111802.66 | 0.349 | 0.664 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.664) | Tech, Media & Telecom Roundup: Market Talk... |
| 2025-08-28 | BUY | $112544.80 | 0.345 | 0.647 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.647) | BlackRock dominates all ETFs with $3.5bn haul: ‘that’s how hungry the fish are’... |
| 2025-08-30 | SELL | $108808.07 | 0.346 | 0.546 | SELL: Models disagree - ProbeTrain: 1 (0.346), SAE: -1 (0.546) | Why the bitcoin trade 'is too large to ignore'... |
| 2025-09-01 | BUY | $109250.59 | 0.346 | 0.451 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.451) | Market takeaways: Volatility creeps up, bitcoin check-in... |
| 2025-09-03 | SELL | $111723.21 | 0.344 | 0.518 | SELL: Models disagree - ProbeTrain: 1 (0.344), SAE: -1 (0.518) | If You'd Invested $10,000 in Bitcoin 5 Years Ago, Here's How Much You'd Have Tod... |
| 2025-09-04 | BUY | $110723.60 | 0.351 | 0.630 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.351, SAE: 0.630) | Bit Digital Ethereum Treasury Soars to $675M, Cementing Spot as Top 6 Holder... |
| 2025-09-05 | SELL | $110650.98 | 0.349 | 0.660 | SELL: Models disagree - ProbeTrain: 1 (0.349), SAE: -1 (0.660) | Tech, Media & Telecom Roundup: Market Talk... |
| 2025-09-07 | SELL | $111167.62 | 0.345 | 0.675 | SELL: Models disagree - ProbeTrain: 1 (0.345), SAE: -1 (0.675) | BlackRock dominates all ETFs with $3.5bn haul: ‘that’s how hungry the fish are’... |
| 2025-09-09 | BUY | $111530.55 | 0.346 | 0.536 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.536) | Why the bitcoin trade 'is too large to ignore'... |
| 2025-09-10 | SELL | $113955.36 | 0.342 | 0.791 | SELL: Models disagree - ProbeTrain: 1 (0.342), SAE: -1 (0.791) | Bitcoin as a reserve currency: What is the 'debasement' trade?... |
| 2025-09-11 | SELL | $115507.54 | 0.346 | 0.544 | SELL: Models disagree - ProbeTrain: 1 (0.346), SAE: -1 (0.544) | Market takeaways: Volatility creeps up, bitcoin check-in... |
| 2025-09-13 | SELL | $115950.51 | 0.344 | 0.561 | SELL: Models disagree - ProbeTrain: 1 (0.344), SAE: -1 (0.561) | If You'd Invested $10,000 in Bitcoin 5 Years Ago, Here's How Much You'd Have Tod... |
| 2025-09-14 | BUY | $115407.66 | 0.351 | 0.627 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.351, SAE: 0.627) | Bit Digital Ethereum Treasury Soars to $675M, Cementing Spot as Top 6 Holder... |
| 2025-09-15 | BUY | $115444.88 | 0.349 | 0.441 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.441) | Tech, Media & Telecom Roundup: Market Talk... |
| 2025-09-23 | BUY | $112014.50 | 0.344 | 0.593 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.593) | If You'd Invested $10,000 in Bitcoin 5 Years Ago, Here's How Much You'd Have Tod... |
| 2025-09-24 | BUY | $113328.63 | 0.351 | 0.722 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.351, SAE: 0.722) | Bit Digital Ethereum Treasury Soars to $675M, Cementing Spot as Top 6 Holder... |
| 2025-09-25 | BUY | $109049.29 | 0.349 | 0.744 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.744) | Tech, Media & Telecom Roundup: Market Talk... |
| 2025-09-27 | BUY | $109681.95 | 0.345 | 0.683 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.683) | BlackRock dominates all ETFs with $3.5bn haul: ‘that’s how hungry the fish are’... |
| 2025-09-29 | BUY | $114400.38 | 0.346 | 0.861 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.861) | Why the bitcoin trade 'is too large to ignore'... |
| 2025-09-30 | BUY | $114056.09 | 0.342 | 0.899 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.899) | Bitcoin as a reserve currency: What is the 'debasement' trade?... |
| 2025-10-01 | BUY | $118648.93 | 0.346 | 0.752 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.752) | Market takeaways: Volatility creeps up, bitcoin check-in... |
| 2025-10-03 | BUY | $122266.53 | 0.344 | 0.867 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.867) | If You'd Invested $10,000 in Bitcoin 5 Years Ago, Here's How Much You'd Have Tod... |
| 2025-10-05 | BUY | $123513.48 | 0.349 | 0.609 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.609) | Tech, Media & Telecom Roundup: Market Talk... |
| 2025-10-07 | BUY | $121451.38 | 0.345 | 0.892 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.892) | BlackRock dominates all ETFs with $3.5bn haul: ‘that’s how hungry the fish are’... |
