# 5-Year Backtesting Results

## Performance Summary

| Ticker | Total Return | Sharpe Ratio | Max Drawdown | Win Rate | Trades |
|--------|-------------|--------------|--------------|----------|--------|
| **BA** | 34.06% | 0.209 | -51.26% | 52.8% | 36 |

## Overall Performance
- **Average Return**: 34.06%
- **Average Sharpe**: 0.209
- **Total Trades**: 36

## Generated Files
- `optimized_5year_backtest_results.json` - Complete results
- `optimized_quantstats_[TICKER].png` - Combined QuantStats charts (3 subplots each)
- `optimized_trading_chart_[TICKER].png` - Price + Buy/Sell signals

## Strategy Details
- **Time Period**: 5 years of historical data
- **Confidence Threshold**: 0.3
- **Models**: ProbeTrain + SAE (synthetic fallback)
- **Signal Logic**: Buy when both models agree on positive sentiment, Sell when both models agree on negative sentiment, No trade when models disagree
- **Backtesting Platform**: **VectorBT** for portfolio simulation and signal processing
- **Performance Analytics**: **QuantStats** for comprehensive financial metrics and risk analysis

## Performance Charts

![Boeing QuantStats Analysis](optimized_quantstats_BA.png)
*Boeing QuantStats Performance Analysis - Returns, Cumulative Returns, and Drawdown*

![Boeing Trading Chart](optimized_trading_chart_BA.png)
*Boeing Price Chart with Buy/Sell Signals*

## Trading Log

### BA Trading Decisions

| Date | Action | Price | ProbeTrain Prob | SAE Prob | Reason | News Headline |
|------|--------|-------|----------------|----------|--------|---------------|
| 2015-06-11 | BUY | $126.62 | 0.339 | 0.789 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.789) | U S  stocks extend gains  as robust data outweighs Greek concerns... |
| 2015-06-11 | BUY | $126.62 | 0.343 | 0.658 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.658) | Boeing eyes potential for over 1 000 jets in mid market... |
| 2015-06-12 | BUY | $126.47 | 0.345 | 0.650 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.650) | Exclusive  Two Zodiac aero plants in U S  show why delays persist... |
| 2015-06-12 | BUY | $126.47 | 0.348 | 0.685 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.348, SAE: 0.685) | Boeing near deal to sell 100 737 MAX to AerCap  sources... |
| 2015-06-15 | SELL | $126.02 | 0.346 | 0.720 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.346, SAE: 0.720) | Exclusive  Russia  U S  competing for space partnership with Brazil... |
| 2015-06-15 | BUY | $126.02 | 0.350 | 0.494 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.350, SAE: 0.494) | Garuda commits to buying 30 Boeing 787  30 737 MAX aircraft... |
| 2015-06-17 | SELL | $127.03 | 0.340 | 0.651 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.340, SAE: 0.651) | GE s Immelt  U S  retreat from ExIm  trade will cost jobs  influence... |
| 2015-06-18 | SELL | $128.76 | 0.339 | 0.413 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.339, SAE: 0.413) | U S  cyber hack unsettles  frustrates U S  defense industry... |
| 2015-06-18 | BUY | $128.76 | 0.337 | 0.607 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.607) | Airbus steals show with last minute  14 billion Wizz deal... |
| 2015-06-23 | BUY | $127.92 | 0.339 | 0.541 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.541) | U S  export lender heads for partial shutdown... |
| 2015-06-23 | BUY | $127.92 | 0.350 | 0.793 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.350, SAE: 0.793) | China may buy 50 70 Airbus A330 jets  sources... |
| 2015-06-30 | BUY | $122.86 | 0.343 | 0.635 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.635) | Virgin Atlantic to cut 500 jobs  WSJ report... |
| 2015-06-30 | SELL | $122.86 | 0.338 | 0.769 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.338, SAE: 0.769) | Obama  Ex Im lapse means  lost sales  lost customers  lost jobs ... |
| 2015-07-08 | BUY | $125.69 | 0.340 | 0.796 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.796) | IForex Daily   July 08  2014... |
| 2015-07-08 | BUY | $125.69 | 0.339 | 0.649 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.649) | Bomber decision due in August or September  U S  Air Force... |
| 2015-07-14 | BUY | $130.86 | 0.344 | 0.688 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.688) | FedEx in talks to buy 25 Boeing 767 freighters  Bloomberg... |
| 2015-07-14 | BUY | $130.86 | 0.344 | 0.460 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.460) | From remote outpost  India looks to check China s Indian Ocean thrust... |
| 2015-07-16 | BUY | $131.51 | 0.350 | 0.731 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.350, SAE: 0.731) | A year after MH17 downed  families want justice... |
| 2015-07-17 | BUY | $130.05 | 0.337 | 0.789 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.789) | Google propels Nasdaq to another record high close... |
| 2015-07-20 | BUY | $129.95 | 0.351 | 0.470 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.351, SAE: 0.470) | U S  Admiral says his South China Sea surveillance flight  routine ... |
| 2015-07-22 | BUY | $129.72 | 0.341 | 0.898 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.898) | Boeing profit beats estimates as plane deliveries surge ... |
| 2015-07-23 | BUY | $129.41 | 0.338 | 0.767 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.767) | U S  Senate backs Ex Im Bank renewal in procedural vote... |
| 2015-07-24 | BUY | $127.59 | 0.338 | 0.638 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.638) | Boeing looks at pricey titanium in bid to stem 787 losses... |
| 2015-07-27 | BUY | $124.91 | 0.340 | 0.887 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.887) | U S  stocks fall for fifth straight session  as China sell off weighs... |
| 2015-07-29 | BUY | $127.66 | 0.344 | 0.462 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.462) | Boeing may move work abroad with Ex Im future uncertain  chairman... |
| 2015-07-30 | BUY | $126.66 | 0.341 | 0.505 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.505) | U S  to deliver eight F 16 aircraft to Egypt... |
| 2015-07-30 | BUY | $126.66 | 0.337 | 0.865 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.865) | Malaysia says Indian Ocean airplane debris is part of a Boeing 777... |
| 2015-08-13 | BUY | $129.07 | 0.354 | 0.617 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.354, SAE: 0.617) | Plane with 54 on board crashes in remote Indonesian region... |
| 2015-08-14 | BUY | $129.32 | 0.360 | 0.500 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.360, SAE: 0.500) | Dow Jones Sector Watch  August 14  2015... |
| 2015-08-17 | BUY | $128.74 | 0.345 | 0.414 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.414) | Reunion police and army halt MH370 debris search... |
| 2015-08-21 | BUY | $117.40 | 0.342 | 0.666 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.666) | Boeing on track to meet 737 MAX production targets  spokesman... |
| 2015-08-25 | SELL | $111.85 | 0.336 | 0.615 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.336, SAE: 0.615) | Boeing raises China 20 year aircraft demand  says outlook rosy... |
| 2015-08-31 | BUY | $116.48 | 0.346 | 0.847 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.847) | U S  stocks lower at close of trade  Dow Jones Industrial Average down 0 69 ... |
| 2015-09-04 | BUY | $115.66 | 0.343 | 0.429 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.429) | Boeing sticks with 747 production plans despite sales drought... |
| 2015-09-09 | SELL | $118.23 | 0.343 | 0.678 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.343, SAE: 0.678) | Pentagon chief to protect funds for cyber  space  electronic warfare... |
| 2015-09-09 | BUY | $118.23 | 0.340 | 0.811 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.811) | Boeing plans to further speed up 767 aircraft production... |
| 2015-09-14 | BUY | $119.82 | 0.341 | 0.790 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.790) | European stocks climb with Fed meeting in focus  Dax up 0 91 ... |
| 2015-09-15 | BUY | $121.49 | 0.342 | 0.762 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.762) | GE to move turbine jobs to Europe  China due to EXIM bank closure... |
| 2015-09-23 | BUY | $117.36 | 0.349 | 0.658 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.658) | India To Buy Boeing Helicopters In  2 5B Deal... |
| 2015-09-23 | SELL | $117.36 | 0.337 | 0.673 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.337, SAE: 0.673) | China agrees to purchase 300 Boeing airplanes for  38 billion price tag... |
| 2015-09-24 | BUY | $115.65 | 0.337 | 0.458 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.458) | D j  Vu All Over Again... |
| 2015-09-28 | BUY | $114.21 | 0.342 | 0.414 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.414) | Airbus supersalesman Leahy pushes back talk of retirement... |
| 2015-09-29 | BUY | $114.76 | 0.350 | 0.848 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.350, SAE: 0.848) | Boeing studies plan to offer 737 freighter conversions... |
| 2015-10-13 | BUY | $125.04 | 0.340 | 0.852 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.852) | Flight MH17 shot down by Russian built Buk missile  Dutch report says... |
| 2015-10-15 | BUY | $122.46 | 0.347 | 0.637 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.637) | Markets Retreat On Weak Economic Reports... |
| 2015-10-16 | BUY | $122.65 | 0.357 | 0.707 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.357, SAE: 0.707) | Honeywell profit beats as costs fall... |
| 2015-10-16 | BUY | $122.65 | 0.341 | 0.503 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.503) | Delta CEO warning prompts questions about Boeing 777 production levels... |
| 2015-10-21 | BUY | $125.85 | 0.338 | 0.692 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.692) | Wall St  falls  Valeant  healthcare drop ... |
| 2015-10-21 | BUY | $125.85 | 0.340 | 0.617 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.617) | Pentagon says F 35 jet cost to rise if Canada  others skip orders... |
| 2015-10-21 | BUY | $125.85 | 0.344 | 0.787 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.787) | Exclusive  China made regional jet set for delivery  but no U S  certification... |
| 2015-10-21 | BUY | $125.85 | 0.347 | 0.841 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.841) | Boeing lifts 2015 outlook as profit jumps 25 percent... |
| 2015-10-26 | BUY | $130.76 | 0.344 | 0.591 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.591) | Boeing wins  898 million contract for 15 EA 18G fighter jets... |
| 2015-10-27 | SELL | $132.33 | 0.341 | 0.842 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.341, SAE: 0.842) | U S  picks Northrop Grumman to build next long range bomber... |
| 2015-10-30 | BUY | $131.98 | 0.346 | 0.403 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.403) | Must Have Stocks To Survive Earnings Slow Down... |
| 2015-11-06 | BUY | $132.68 | 0.346 | 0.801 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.801) | Russia declines to act on recommendation over Boeing 737 certificates... |
| 2015-11-09 | BUY | $130.92 | 0.347 | 0.764 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.764) | Emirates signs  16 billion engine deal with GE for 777x fleet... |
| 2015-11-09 | BUY | $130.92 | 0.349 | 0.480 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.480) | India s Jet Airways Orders 75 Boeing 737 Planes... |
| 2015-11-10 | BUY | $130.18 | 0.343 | 0.696 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.696) | Japan s first commercial jet in 50 years makes maiden flight... |
| 2015-11-12 | SELL | $127.83 | 0.344 | 0.760 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.344, SAE: 0.760) | Northrop s long range U S  bomber work paused after protest... |
| 2015-11-13 | BUY | $127.88 | 0.348 | 0.758 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.348, SAE: 0.758) | Northrop upbeat on E 2D early warning sales despite UAE loss... |
| 2015-11-18 | BUY | $132.98 | 0.341 | 0.636 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.636) | U S  expects F 35 to be part of Canada s next jet competition... |
| 2015-11-18 | SELL | $132.98 | 0.337 | 0.794 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.337, SAE: 0.794) | Two Air France flights from the United States to Paris diverted  FAA... |
| 2015-11-23 | BUY | $133.04 | 0.350 | 0.626 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.350, SAE: 0.626) | Around The World In 4 Hours ... |
| 2015-11-23 | BUY | $133.04 | 0.340 | 0.513 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.513) | U S  says China to take tougher stance against trade secret theft ... |
| 2015-11-25 | BUY | $132.22 | 0.336 | 0.585 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.336, SAE: 0.585) | Boeing says to roll out first 737 MAX in early December... |
| 2015-12-04 | BUY | $133.18 | 0.341 | 0.657 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.657) | U S  arms makers strain to meet demand as Mideast conflicts rage... |
| 2015-12-10 | BUY | $131.31 | 0.342 | 0.737 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.737) | Aircraft orders slow as cycle peaks  potential for production impact seen... |
| 2015-12-16 | BUY | $132.83 | 0.347 | 0.817 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.817) | Boeing says China Postal Airlines orders ten 737 800 converted freighters... |
| 2015-12-21 | SELL | $126.36 | 0.338 | 0.792 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.338, SAE: 0.792) | Southwest pays  2 8 million fine to settle safety violations with FAA... |
| 2015-12-22 | BUY | $127.69 | 0.343 | 0.883 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.883) | Pentagon eyes proposal for M A changes in  weeks ... |
| 2016-01-04 | BUY | $126.01 | 0.345 | 0.687 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.687) | Boeing wins  855 million contract for T 38C logistics support... |
| 2016-01-15 | BUY | $112.67 | 0.347 | 0.639 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.639) | IAG says its A380 options too expensive  looking to lease second hand... |
| 2016-01-21 | BUY | $110.67 | 0.334 | 0.657 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.334, SAE: 0.657) | United Continental CEO to return from leave  results miss... |
| 2016-01-21 | BUY | $110.67 | 0.343 | 0.776 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.776) | Sanctions lifted  Iran s Rouhani heads to Europe to drum up business... |
| 2016-01-27 | BUY | $104.55 | 0.344 | 0.475 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.475) | McCain vows to undo U S  legislation that eased Russian rocket engine ban... |
| 2016-01-28 | BUY | $105.84 | 0.342 | 0.622 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.622) | Is Alcoa Inc A Value Buy ... |
| 2016-01-28 | BUY | $105.84 | 0.344 | 0.842 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.842) | Campaigning in style  How Jeb Bush blew through his war chest ... |
| 2016-02-03 | SELL | $109.30 | 0.337 | 0.860 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.337, SAE: 0.860) | Flight control  China banks pilot jet leasing firms in chase for  228 billion ma... |
| 2016-02-04 | BUY | $110.86 | 0.337 | 0.765 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.765) | U S  Navy examining rise in health issues among F A 18 pilots  lawmaker... |
| 2016-02-05 | BUY | $109.92 | 0.345 | 0.576 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.576) | Exclusive  Pentagon s budget plan funds 404 Lockheed F 35 jets   sources ... |
| 2016-02-12 | BUY | $98.32 | 0.349 | 0.879 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.879) | Dow Theory Signal  What Does It Mean ... |
| 2016-02-18 | SELL | $106.42 | 0.339 | 0.739 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.339, SAE: 0.739) | Afghan army to fly first unmanned aircraft in March  U S  official says... |
| 2016-02-19 | BUY | $104.24 | 0.347 | 0.663 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.663) | Canada likely to aid struggling Bombardier  government sources... |
| 2016-02-19 | BUY | $104.24 | 0.337 | 0.674 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.674) | Shift in U S  sanctions could ground Russian rocket engines  general... |
| 2016-02-19 | SELL | $104.24 | 0.343 | 0.409 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.343, SAE: 0.409) | Branson s Virgin Galactic unveils new passenger spaceship... |
| 2016-02-22 | BUY | $106.25 | 0.349 | 0.838 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.838) | United orders 25 more Boeing 737 worth  2 billion  sources... |
| 2016-02-26 | SELL | $106.95 | 0.337 | 0.469 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.337, SAE: 0.469) | United Tech rejects Honeywell s  90 7 billion offer... |
| 2016-03-01 | BUY | $108.58 | 0.350 | 0.798 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.350, SAE: 0.798) | Aerojet on track to finish AR1 rocket engine work by 2019  CEO... |
| 2016-03-02 | BUY | $108.30 | 0.343 | 0.749 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.749) | Possible debris from Malaysia flight MH370 found near Mozambique  NBC... |
| 2016-03-08 | BUY | $110.74 | 0.348 | 0.705 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.348, SAE: 0.705) | Lockheed aims to build satellites 40 percent quicker  lower costs ... |
| 2016-03-08 | BUY | $110.74 | 0.343 | 0.481 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.481) | Bezos  space company aims for passenger flights in 2018... |
| 2016-03-09 | BUY | $111.20 | 0.340 | 0.866 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.866) | Amazon to start air delivery network with leasing deal... |
| 2016-03-11 | BUY | $112.81 | 0.344 | 0.635 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.635) | South African teen finds suspected piece of missing MH370 plane... |
| 2016-03-11 | BUY | $112.81 | 0.342 | 0.606 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.606) | Boeing CTO John Tracy set to retire in July after 35 years with company... |
| 2016-03-16 | BUY | $115.47 | 0.334 | 0.625 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.334, SAE: 0.625) | Plane crashes in Russia  all 62 people on board killed... |
| 2016-03-31 | BUY | $114.90 | 0.343 | 0.661 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.661) | U S  futures slip ahead of more Fed appearances and jobless claims... |
| 2016-03-31 | BUY | $114.90 | 0.344 | 0.831 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.831) | The Yellen Rally... |
| 2016-03-31 | BUY | $114.90 | 0.347 | 0.423 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.423) | Debris found in Mauritius to be examined by MH370 investigators... |
| 2016-04-11 | BUY | $115.78 | 0.340 | 0.697 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.697) | U S  planemaker Boeing discusses sales in Iran... |
| 2016-04-13 | BUY | $118.70 | 0.340 | 0.643 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.643) | United Launch Alliance suspends Atlas 5 flights... |
| 2016-04-14 | BUY | $118.44 | 0.336 | 0.629 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.336, SAE: 0.629) | Boeing wins U S  Army helicopter deal worth  1 5 billion... |
| 2016-04-14 | BUY | $118.44 | 0.349 | 0.520 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.520) | United Launch Alliance to lay off up to 875 by end of 2017  CEO... |
| 2016-04-22 | BUY | $118.62 | 0.344 | 0.781 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.781) | Boeing  BA  Poised for Q1 Earnings Beat  Will Stock Gain ... |
| 2016-04-27 | BUY | $124.08 | 0.356 | 0.498 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.356, SAE: 0.498) | Boeing short sales rise 1 4 percent to 27 4 million shares... |
| 2016-04-28 | BUY | $122.01 | 0.337 | 0.717 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.717) | Kenya Airways pilots call off strike saying some demands met... |
| 2016-05-03 | BUY | $119.92 | 0.339 | 0.724 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.724) | Singapore set to pick military helicopters as arms spending rises... |
| 2016-05-03 | BUY | $119.92 | 0.343 | 0.448 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.448) | Solar powered plane lands in Arizona on round the world flight... |
| 2016-05-06 | BUY | $120.62 | 0.344 | 0.882 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.882) | Upbeat Aerospace   Defense Results Lift ETFs ... |
| 2016-05-09 | BUY | $119.57 | 0.339 | 0.435 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.435) | UPS backed Rwandan blood deliveries show drones  promise  hurdles... |
| 2016-05-16 | BUY | $122.40 | 0.339 | 0.757 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.757) | Boeing s 787 Dreamliner faces new challenge  slow sales... |
| 2016-05-19 | BUY | $116.87 | 0.338 | 0.572 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.572) | Solar plane lands in Dayton  Ohio on latest leg of round the world flight... |
| 2016-05-19 | BUY | $116.87 | 0.340 | 0.614 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.614) | Boeing Wins  11 3B Order From VietJet For 737 MAX Airplanes... |
| 2016-05-23 | BUY | $116.42 | 0.350 | 0.752 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.350, SAE: 0.752) | Values To Watch  Boeing  Pfizer  AMAT And USD JPY... |
| 2016-05-25 | BUY | $117.59 | 0.337 | 0.766 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.766) | TransDigm To Acquire Data Device Corporation For  1B... |
| 2016-06-01 | BUY | $115.35 | 0.355 | 0.568 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.355, SAE: 0.568) | iFOREX Daily Analysis   June 01  2016... |
| 2016-06-06 | BUY | $120.36 | 0.337 | 0.865 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.865) | U S  stocks rise sharply as Yellen gives little hint on rate hike timing... |
| 2016-06-06 | BUY | $120.36 | 0.353 | 0.693 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.353, SAE: 0.693) | Stock Market News For June 07  2016... |
| 2016-06-06 | BUY | $120.36 | 0.344 | 0.713 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.713) | U S  stocks higher at close of trade  Dow Jones Industrial Average up 0 64 ... |
| 2016-06-06 | BUY | $120.36 | 0.347 | 0.513 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.513) | 5 Swing Trades To Watch  INTU  QCOM  LVS  BA  TLRD... |
| 2016-06-07 | SELL | $120.39 | 0.341 | 0.749 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.341, SAE: 0.749) | Boeing  Major Commercial Jet Order From Iran In The Cards ... |
| 2016-06-07 | SELL | $120.39 | 0.334 | 0.552 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.334, SAE: 0.552) | Northrop Secures  600M In Missile Contract Modification... |
| 2016-06-08 | BUY | $121.36 | 0.349 | 0.421 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.421) | Boeing  BA  Wins  668M Army Deal For Apache Helicopters... |
| 2016-06-08 | BUY | $121.36 | 0.351 | 0.767 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.351, SAE: 0.767) | Stock Market News For June 09  2016... |
| 2016-06-08 | BUY | $121.36 | 0.339 | 0.751 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.751) | Solar plane lands in New York City during bid to circle the globe... |
| 2016-06-09 | BUY | $121.46 | 0.352 | 0.750 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.352, SAE: 0.750) | Boeing  Escaping  125 135 Will Lead To Trendline Target... |
| 2016-06-10 | BUY | $119.67 | 0.356 | 0.497 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.356, SAE: 0.497) | Lockheed may shift F 35 fighter work away from Canada... |
| 2016-06-10 | BUY | $119.67 | 0.347 | 0.753 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.753) |  Gogo Inc  GOGO  Prices 12 50  Senior Notes Worth  525M... |
| 2016-06-20 | BUY | $121.14 | 0.353 | 0.862 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.353, SAE: 0.862) | Boeing  BA  Set To Win  4 Billion Order For 747 Freighters... |
| 2016-06-21 | BUY | $120.01 | 0.342 | 0.586 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.586) | Receding Brexit fears lift markets... |
| 2016-06-21 | BUY | $120.01 | 0.342 | 0.845 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.845) | Iran And Boeing  BA  Make Jet Sale Deal Official... |
| 2016-06-22 | BUY | $120.24 | 0.344 | 0.756 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.756) | Debris in Tanzania to be examined for link to missing MH370   report... |
| 2016-06-23 | BUY | $121.87 | 0.342 | 0.740 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.740) | Solar plane lands in Spain after three day Atlantic crossing... |
| 2016-06-23 | BUY | $121.87 | 0.342 | 0.733 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.733) | Singapore Airlines flight returns to Changi  catches fire  no casualties... |
| 2016-06-27 | BUY | $111.97 | 0.345 | 0.692 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.692) | Doubts grow over Airbus A380 sale to Iran  sources... |
| 2016-06-30 | BUY | $118.51 | 0.337 | 0.452 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.452) | Japan plans July fighter jet tender seen worth  40 billion as China tensions sim... |
| 2016-07-05 | BUY | $115.86 | 0.346 | 0.606 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.606) | GE s Affiliate To Lease 45 Aircraft To Bohai s Subsidiaries... |
| 2016-07-07 | BUY | $116.04 | 0.340 | 0.696 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.696) | Lockheed welcomes Canada shift toward  more choices  for fighters... |
| 2016-07-07 | BUY | $116.04 | 0.335 | 0.404 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.335, SAE: 0.404) | Boeing Reports Q2 Deliveries  Commercial Up  Defense Drags... |
| 2016-07-11 | BUY | $120.49 | 0.348 | 0.865 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.348, SAE: 0.865) | Company News For July 12  2016... |
| 2016-07-12 | BUY | $119.37 | 0.342 | 0.818 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.818) | Chinese lessor orders ARJ 21 jets in  2 3 billion deal... |
| 2016-07-12 | BUY | $119.37 | 0.339 | 0.621 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.621) | TUI finalizes  1 4 billion order for Boeing planes... |
| 2016-07-13 | BUY | $118.73 | 0.355 | 0.764 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.355, SAE: 0.764) | Boeing  Ruili Airlines Sign Deal For 787 9 Dreamliners... |
| 2016-07-13 | BUY | $118.73 | 0.341 | 0.433 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.433) | Will Boeing Get An Order For 737 MAX From Qatar Airways ... |
| 2016-07-14 | BUY | $120.04 | 0.340 | 0.740 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.740) | Boeing wins orders commitments worth  26 8 bn ... |
| 2016-07-14 | BUY | $120.04 | 0.338 | 0.607 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.607) | U S  aims for start this year on buying more military satellites... |
| 2016-07-18 | BUY | $121.76 | 0.337 | 0.614 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.614) | Is This The Airlines Liftoff Investors Have Been Waiting For ... |
| 2016-07-21 | BUY | $121.85 | 0.348 | 0.729 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.348, SAE: 0.729) | Boeing  BA  To Report Q2 Earnings  Can It Pull A Surprise ... |
| 2016-07-21 | BUY | $121.85 | 0.346 | 0.594 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.594) | MH370 search team raises prospect plane could lie elsewhere... |
| 2016-07-26 | BUY | $123.05 | 0.352 | 0.430 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.352, SAE: 0.430) | Defense Stocks To Report Earnings On Jul 27  BA  CW  GD  NOC... |
| 2016-07-26 | BUY | $123.05 | 0.355 | 0.855 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.355, SAE: 0.855) |  Stock Market News For July 27  2016... |
| 2016-07-27 | BUY | $124.07 | 0.339 | 0.843 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.843) | L 3 Communications  LLL  Tops Q2 Earnings  Ups  16 View... |
| 2016-07-28 | BUY | $121.37 | 0.368 | 0.658 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.368, SAE: 0.658) | Market Update   28 07 2016... |
| 2016-07-28 | BUY | $121.37 | 0.338 | 0.677 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.677) | Boeing says it could end 747 production... |
| 2016-07-28 | BUY | $121.37 | 0.345 | 0.717 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.717) | Spirit AeroSystems  SPR  Q2 Earnings  Stock To Surprise ... |
| 2016-07-29 | BUY | $121.97 | 0.345 | 0.678 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.678) | Wing part found in Tanzania is  highly likely  from MH370  Australia minister... |
| 2016-08-03 | BUY | $120.33 | 0.353 | 0.510 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.353, SAE: 0.510) | 3 Stocks To Watch  BA  NVX  CMP... |
| 2016-08-04 | BUY | $119.73 | 0.339 | 0.796 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.796) | Zacks Industry Outlook Highlights  Lockheed Martin  Northrop Grumman  General Dy... |
| 2016-08-10 | BUY | $121.71 | 0.343 | 0.821 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.821) | Boeing won t raise 787 production unless market demands it... |
| 2016-08-15 | BUY | $123.90 | 0.347 | 0.416 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.416) | Why America Will Win The New  Space Race ... |
| 2016-08-22 | BUY | $124.20 | 0.347 | 0.741 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.741) | Boeing internal forecast sees 535 jetliner sales in 2016  below target  AvWeek... |
| 2016-08-29 | BUY | $122.28 | 0.337 | 0.853 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.853) | Chinese owned Zhongwang USA enters U S  aluminum market with Aleris buy... |
| 2016-08-29 | BUY | $122.28 | 0.348 | 0.621 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.348, SAE: 0.621) | Why Is Copa Holdings  CPA  A Must Add To Your Portfolio ... |
| 2016-08-30 | BUY | $120.36 | 0.357 | 0.581 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.357, SAE: 0.581) | Joe Sutter  father of the 747  passes away at 95... |
| 2016-08-30 | BUY | $120.36 | 0.339 | 0.896 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.896) | SpaceX signs first customer for used Falcon rocket... |
| 2016-08-31 | BUY | $119.10 | 0.351 | 0.682 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.351, SAE: 0.682) | Japanese airline ANA to replace 100 Rolls engines on 787s... |
| 2016-09-01 | BUY | $119.52 | 0.341 | 0.538 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.538) | Exclusive  U S  set to approve sales of Boeing fighters to Qatar  Kuwait  source... |
| 2016-09-01 | BUY | $119.52 | 0.340 | 0.470 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.470) | Embraer  ERJ  Wins  249M Colorful Guizhou Airlines Deal... |
| 2016-09-02 | BUY | $120.68 | 0.339 | 0.577 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.577) | Defense Stock Roundup  Deals Galore For Lockheed Martin  Northrop s  376M Contra... |
| 2016-09-02 | BUY | $120.68 | 0.345 | 0.625 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.625) | Japan asks European aviation agency to ensure safety of Rolls Royce 787 engines... |
| 2016-09-06 | BUY | $122.36 | 0.347 | 0.694 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.694) | All Nippon Airways To Replace Engines In Boeing 787 Fleet... |
| 2016-09-08 | BUY | $122.28 | 0.343 | 0.818 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.818) | Boeing To Redesign 737 Max 9 To Vie With Airbus A321neo... |
| 2016-09-09 | BUY | $118.26 | 0.341 | 0.708 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.708) | NASA asteroid probe may find clues to origins of life on Earth... |
| 2016-09-09 | BUY | $118.26 | 0.345 | 0.637 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.637) | U S  stocks lower at close of trade  Dow Jones Industrial Average down 2 13 ... |
| 2016-09-14 | BUY | $117.47 | 0.350 | 0.556 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.350, SAE: 0.556) | Boeing  China Aircraft Demand To Rise  Market To Cross  1T... |
| 2016-09-14 | BUY | $117.47 | 0.353 | 0.687 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.353, SAE: 0.687) | Boeing CEO says 777 output may be cut further if sales lag endures... |
| 2016-09-15 | SELL | $117.56 | 0.343 | 0.682 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.343, SAE: 0.682) | United Continental  UAL  To Expand Operations In San Juan ... |
| 2016-09-19 | BUY | $117.29 | 0.349 | 0.886 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.886) | L 3 Communications Secures  163M Navy Contract For T 45... |
| 2016-09-21 | BUY | $120.13 | 0.337 | 0.734 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.734) | U S  begins unblocking jetliner sales to Iran... |
| 2016-09-22 | BUY | $121.33 | 0.347 | 0.614 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.614) | Boeing still studying larger 737 Max 10 to enter service after 2019... |
| 2016-09-27 | SELL | $120.82 | 0.341 | 0.739 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.341, SAE: 0.739) | 3 Better Picks Than Southwest  LUV  In The Airline Sector... |
| 2016-09-27 | BUY | $120.82 | 0.348 | 0.544 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.348, SAE: 0.544) | 5 Top Stocks With Powerful Net Profit Margin... |
| 2016-09-28 | BUY | $121.66 | 0.339 | 0.788 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.788) | Ship shake  Hanjin woes may help float tech  data start ups... |
| 2016-09-28 | BUY | $121.66 | 0.338 | 0.629 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.629) | Malaysian flight MH17 downed by Russian made missile  prosecutors... |
| 2016-09-28 | BUY | $121.66 | 0.340 | 0.612 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.612) | The Zacks Analyst Blog Highlights  Amazon  MasterCard  Boeing  Deutsche Bank And... |
| 2016-09-28 | BUY | $121.66 | 0.350 | 0.685 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.350, SAE: 0.685) | After promising start  Rolls Royce boss East must deliver... |
| 2016-09-29 | BUY | $120.56 | 0.349 | 0.633 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.633) | Will Lockheed Boeing s ULA Gain From The  861M Deal ... |
| 2016-09-29 | BUY | $120.56 | 0.345 | 0.634 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.634) | Lockheed Martin Unit Wins 2 Air Force Deals Worth  217M... |
| 2016-09-30 | BUY | $121.21 | 0.348 | 0.669 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.348, SAE: 0.669) | Indonesia s Garuda s U S  return to generate 5 percent of 2017 revenue  CEO... |
| 2016-10-04 | BUY | $121.68 | 0.349 | 0.737 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.737) | Spirit AeroSystems  SPR  Upgraded On Robust Performance... |
| 2016-10-05 | BUY | $123.90 | 0.345 | 0.717 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.717) | Should You Buy Aerospace   Defense ETFs Now ... |
| 2016-10-06 | BUY | $123.68 | 0.344 | 0.889 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.889) | B E Aerospace At 52 Week High On Positive Demand Outlook... |
| 2016-10-06 | BUY | $123.68 | 0.352 | 0.797 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.352, SAE: 0.797) | Stock Market News For October 10  2016... |
| 2016-10-06 | BUY | $123.68 | 0.341 | 0.506 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.506) | Boeing Reports Q3 Deliveries  Commercial Down  Defense Up... |
| 2016-10-07 | BUY | $123.15 | 0.341 | 0.624 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.624) | Malaysia says debris found in Mauritius is from missing Flight MH370... |
| 2016-10-07 | BUY | $123.15 | 0.337 | 0.421 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.421) | Northrop Grumman Secures Two Air Force Deals Worth  96M... |
| 2016-10-07 | BUY | $123.15 | 0.339 | 0.710 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.710) | Qatar Airways places  18 billion order for 100 Boeing jets source... |
| 2016-10-10 | BUY | $124.98 | 0.353 | 0.633 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.353, SAE: 0.633) | Lockheed Martin  LMT  Wins Navy Deal For SEWIP Block 2... |
| 2016-10-10 | BUY | $124.98 | 0.347 | 0.686 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.686) | AAR Corp  Signs A  125M PBH Deal To Boost African Business... |
| 2016-10-10 | BUY | $124.98 | 0.343 | 0.430 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.430) | Boeing Secures  18 6 Billion Deal With Qatar Airways... |
| 2016-10-12 | BUY | $122.52 | 0.338 | 0.418 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.418) | Facebook  FB  Wants To Fly Drones Over Menlo Park Campus... |
| 2016-10-12 | SELL | $122.52 | 0.338 | 0.775 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.338, SAE: 0.775) | Samsung s woes highlight explosive limits of lithium batteries... |
| 2016-10-20 | BUY | $124.98 | 0.344 | 0.770 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.770) | Can Northrop  NOC  Pull A Surprise This Earnings Season ... |
| 2016-10-21 | BUY | $124.79 | 0.338 | 0.689 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.689) | Allegheny  ATI  Q3 Earnings  Disappointment In The Cards ... |
| 2016-10-24 | BUY | $126.46 | 0.344 | 0.825 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.825) | Lockheed Martin  LMT  Tops Q3 Earnings  Sales   16 View Up... |
| 2016-10-26 | BUY | $133.91 | 0.347 | 0.513 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.513) | Boeing raises FY guidance as Q3 core EPS rises 39  to  3 51 ... |
| 2016-10-26 | BUY | $133.91 | 0.338 | 0.850 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.850) | India offers to buy 200 foreign combat jets   if  they re Made in India... |
| 2016-10-26 | BUY | $133.91 | 0.344 | 0.601 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.601) | Auto industry cyber security group hires Boeing veteran... |
| 2016-10-26 | BUY | $133.91 | 0.353 | 0.533 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.353, SAE: 0.533) | Boeing CEO says will not cut 777 output rate by more than 2 a month... |
| 2016-10-26 | BUY | $133.91 | 0.361 | 0.762 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.361, SAE: 0.762) | 5 Things To Watch When Amazon Reports Earnings On Thursday... |
| 2016-10-27 | BUY | $131.86 | 0.354 | 0.838 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.354, SAE: 0.838) | UPS places  5 3 billion order for 14 Boeing 747 cargo jets... |
| 2016-10-28 | BUY | $131.58 | 0.342 | 0.828 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.828) | 3 ETFs To Buy On Encouraging Aerospace   Defense Earnings... |
| 2016-11-02 | BUY | $129.50 | 0.344 | 0.705 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.705) | Huntington Ingalls  HII  Misses Q3 Earnings On Lower Sales... |
| 2016-11-14 | BUY | $139.06 | 0.334 | 0.759 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.334, SAE: 0.759) | The Zacks Analyst Blog Highlights  Arotech  Engility Holdings  Northrop Grumman ... |
| 2016-11-15 | BUY | $137.32 | 0.344 | 0.597 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.597) | Boeing shares fall as United Airlines defers 737 order... |
| 2016-11-15 | BUY | $137.32 | 0.339 | 0.545 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.545) | Boeing says still expects to raise 737 output after United fleet changes... |
| 2016-11-18 | BUY | $135.69 | 0.341 | 0.778 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.778) | Defense Stock Roundup  Lockheed Martin  Raytheon  Leidos Win DoD Contracts... |
| 2016-11-21 | BUY | $136.31 | 0.339 | 0.491 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.491) | Boeing  BA  Hits 52 Week High On Steady Flow Of Contracts... |
| 2016-11-21 | BUY | $136.31 | 0.337 | 0.599 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.599) | Thales  The Ugly Duckling Has Become A Swan ... |
| 2016-11-22 | BUY | $138.63 | 0.346 | 0.768 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.768) | Bell Boeing JV Wins  267M Navy Deal For CV 22  MV 22 Jets... |
| 2016-11-22 | BUY | $138.63 | 0.342 | 0.892 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.892) | Alaska Airlines Launches San Diego To Newark Flight... |
| 2016-11-22 | BUY | $138.63 | 0.345 | 0.674 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.674) | U S  stocks higher at close of trade  Dow Jones Industrial Average up 0 35 ... |
| 2016-11-25 | BUY | $139.11 | 0.342 | 0.668 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.668) | Enthusiasts lose one vintage plane but press on south through Africa... |
| 2016-11-28 | BUY | $138.86 | 0.344 | 0.789 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.789) | WTO rules against tax break for Boeing 777X jet... |
| 2016-11-28 | BUY | $138.86 | 0.344 | 0.609 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.609) | Kuwait plans to buy 28 Boeing F 18 jets  official says... |
| 2016-11-28 | BUY | $138.86 | 0.341 | 0.485 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.485) | Spirit Aero Systems Scales 52 Week High On Upbeat Outlook... |
| 2016-12-07 | BUY | $142.91 | 0.346 | 0.616 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.616) | Will Trump s Victory Spell Doom For Lockheed Martin s F 35  ... |
| 2016-12-08 | BUY | $144.07 | 0.346 | 0.755 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.755) | Iran signs  16 6 billion deal for 80 Boeing planes  IRNA... |
| 2016-12-08 | BUY | $144.07 | 0.342 | 0.635 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.635) | Time To Take Trump Seriously On Infrastructure Spending ... |
| 2016-12-08 | BUY | $144.07 | 0.357 | 0.892 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.357, SAE: 0.892) | U S  futures point to steady open as markets pause... |
| 2016-12-12 | BUY | $145.71 | 0.354 | 0.857 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.354, SAE: 0.857) | Boeing to cut 777 production rate by 40 percent in August 2017... |
| 2016-12-12 | BUY | $145.71 | 0.344 | 0.758 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.758) | Trump tweets that the F 35 s cost is  out of control    and now the stock of the... |
| 2016-12-13 | BUY | $145.25 | 0.340 | 0.727 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.727) | Opening Bell  Fed Meets Today   Check Your Twitter Feed Now... |
| 2016-12-13 | BUY | $145.25 | 0.368 | 0.658 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.368, SAE: 0.658) | Market Update   13 12 2016... |
| 2016-12-14 | BUY | $143.22 | 0.344 | 0.690 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.690) | Investigators begin final sweep of MH370 search area... |
| 2016-12-15 | BUY | $142.57 | 0.342 | 0.585 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.585) | IranAir confirms cutting Airbus order  dropping A380s... |
| 2016-12-15 | BUY | $142.57 | 0.356 | 0.626 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.356, SAE: 0.626) | 5 Reasons The Fed Won t Even Hike Twice In 2017... |
| 2016-12-15 | BUY | $142.57 | 0.345 | 0.844 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.844) | Regional jet makers eye China market boost but obstacles loom... |
| 2016-12-21 | BUY | $146.01 | 0.352 | 0.713 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.352, SAE: 0.713) | Too Late To Join Boeing s Bull Party... |
| 2016-12-21 | BUY | $146.01 | 0.337 | 0.470 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.470) | Israel s Spacecom buys Boeing satellite for  161 million  to launch in 2019... |
| 2016-12-21 | BUY | $146.01 | 0.346 | 0.500 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.500) | U S  stocks lower at close of trade  Dow Jones Industrial Average down 0 16 ... |
| 2016-12-27 | BUY | $146.01 | 0.349 | 0.434 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.434) | Delta to cancel order for 18 Boeing 787 Dreamliner aircraft... |
| 2017-01-04 | BUY | $147.06 | 0.342 | 0.454 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.454) | Boeing inches toward goal with  8 25 billion order from GE... |
| 2017-01-04 | BUY | $147.06 | 0.359 | 0.635 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.359, SAE: 0.635) | S P 500 Futures  Day 3 Of 251 Trading Days In 2017 ... |
| 2017-01-09 | BUY | $146.79 | 0.342 | 0.695 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.695) | S P 500 Futures  Day 4 And New All Time Contract Highs ... |
| 2017-01-10 | SELL | $147.48 | 0.336 | 0.797 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.336, SAE: 0.797) | Boeing  new lobby group aim to keep  8 7 billion in state tax breaks... |
| 2017-01-12 | BUY | $146.76 | 0.336 | 0.586 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.336, SAE: 0.586) | Why Are WMT And BA Laying Off Workers If The Economy Is In Good Shape ... |
| 2017-01-13 | SELL | $147.26 | 0.338 | 0.467 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.338, SAE: 0.467) | Airbus versus Boeing  Iran deals the difference in plane battle... |
| 2017-01-13 | BUY | $147.26 | 0.346 | 0.805 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.805) | Lockheed Martin CEO meets Trump  says deal to lower F 35 costs is close... |
| 2017-01-17 | BUY | $146.18 | 0.347 | 0.839 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.839) | Underwater search for missing Malaysian flight ends without a trace... |
| 2017-01-20 | BUY | $147.91 | 0.354 | 0.802 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.354, SAE: 0.802) | Rockwell Collins sees defense market uptick  small 777 impact... |
| 2017-01-23 | BUY | $146.34 | 0.350 | 0.828 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.350, SAE: 0.828) | Why You Shouldn t Bet Against America... |
| 2017-01-24 | SELL | $148.85 | 0.341 | 0.735 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.341, SAE: 0.735) | High Uncertainty For Industrials Under Trump Administration ... |
| 2017-01-25 | BUY | $155.17 | 0.355 | 0.612 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.355, SAE: 0.612) | Top 5 Things to Know In the Market on Wednesday... |
| 2017-01-25 | BUY | $155.17 | 0.351 | 0.726 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.351, SAE: 0.726) | Boeing core Q4 EPS up 54  at  2 47 as revenues mostly flat ... |
| 2017-01-27 | BUY | $155.48 | 0.338 | 0.784 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.784) | Boeing wins  2 1 billion Pentagon contract for 15 KC 46 refueling aircraft... |
| 2017-02-01 | BUY | $152.02 | 0.344 | 0.628 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.628) | Exclusive  Japan secures extra cost cuts on U S  F 35 fighter jet package   sour... |
| 2017-02-01 | BUY | $152.02 | 0.354 | 0.767 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.354, SAE: 0.767) | Amazon plans more than 200 daily flights from new cargo hub... |
| 2017-02-03 | BUY | $150.57 | 0.356 | 0.739 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.356, SAE: 0.739) | iFOREX Daily Analysis   February 03 2017... |
| 2017-02-07 | BUY | $154.37 | 0.344 | 0.788 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.788) | Exclusive  Boeing s space taxis to use more than 600 3D printed parts... |
| 2017-02-08 | BUY | $153.18 | 0.340 | 0.601 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.601) | Trump  aviation executives to discuss infrastructure Thursday  sources... |
| 2017-02-15 | BUY | $158.32 | 0.344 | 0.880 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.880) | Lies  Damn Lies  And Taxes... |
| 2017-02-15 | BUY | $158.32 | 0.353 | 0.835 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.353, SAE: 0.835) | Global aerospace boom is over  but likely no bust ahead  analyst... |
| 2017-02-15 | BUY | $158.32 | 0.345 | 0.649 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.649) | Maiden flight of China built amphibious aircraft expected in first half of 2017... |
| 2017-02-28 | BUY | $168.54 | 0.348 | 0.862 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.348, SAE: 0.862) | US Stocks To Surge On Upcoming Jobs Data... |
| 2017-02-28 | BUY | $168.54 | 0.352 | 0.615 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.352, SAE: 0.615) | Boeing Wins  679M Navy Deal To Procure 12 Combat Aircraft... |
| 2017-03-01 | BUY | $171.98 | 0.347 | 0.630 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.630) | Dow 21k And A March Rate Hike... |
| 2017-03-03 | BUY | $170.36 | 0.344 | 0.605 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.605) | Boeing sounds out Indian carriers on 737 MAX 10 aircraft... |
| 2017-03-07 | SELL | $170.21 | 0.348 | 0.649 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.348, SAE: 0.649) | United looking at second hand aircraft  rules out A380... |
| 2017-03-08 | BUY | $169.95 | 0.339 | 0.654 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.654) | Exclusive  Airbus may ditch A380 s grand staircase as sales tumble... |
| 2017-03-08 | BUY | $169.95 | 0.348 | 0.840 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.348, SAE: 0.840) | Aircraft lessors lukewarm on Boeing s planned 737 10 jet... |
| 2017-03-10 | BUY | $167.11 | 0.344 | 0.640 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.640) | U S  stocks higher at close of trade  Dow Jones Industrial Average up 0 21 ... |
| 2017-03-16 | SELL | $166.63 | 0.340 | 0.653 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.340, SAE: 0.653) | Trump eyes  flexible  Islamic State war fund  Guantanamo upgrade... |
| 2017-03-16 | BUY | $166.63 | 0.339 | 0.615 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.615) | Boeing  U S  government sign  3 4 billion deal for AH 64E Apache helicopters... |
| 2017-03-16 | BUY | $166.63 | 0.335 | 0.780 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.335, SAE: 0.780) | India s Steel Companies Are Entering The Defense Business Supply Chain... |
| 2017-03-20 | BUY | $167.75 | 0.344 | 0.647 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.647) | 4 Trades To Soar With Boeing... |
| 2017-04-06 | BUY | $165.86 | 0.350 | 0.686 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.350, SAE: 0.686) | Japan s ANA to lease four Boeing 737 800s as MRJ delays continue... |
| 2017-04-07 | BUY | $167.25 | 0.345 | 0.702 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.702) | Boeing Is Heading Lower  And Fast ... |
| 2017-04-07 | BUY | $167.25 | 0.347 | 0.768 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.768) | U S  South  not just Mexico  stands in way of Rust Belt jobs revival... |
| 2017-04-11 | BUY | $166.98 | 0.342 | 0.625 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.625) | Zacks Industry Outlook Highlights  Arconic  Boeing  Lockheed Martin  Amerigo Res... |
| 2017-04-18 | BUY | $166.31 | 0.344 | 0.498 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.498) | Stratasys  SSYS  Jumped 12  On Pepper Jaffray Rating Upgrade... |
| 2017-04-21 | BUY | $168.68 | 0.357 | 0.459 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.357, SAE: 0.459) | GE shares fall on cash  business worries though profit beats... |
| 2017-04-24 | BUY | $170.25 | 0.340 | 0.546 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.546) | Ready for take off  China s answer to Boeing now just needs to sell... |
| 2017-04-25 | BUY | $171.60 | 0.354 | 0.726 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.354, SAE: 0.726) | Frank Holmes  Gold Could Hit  1500 ... |
| 2017-04-25 | BUY | $171.60 | 0.345 | 0.600 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.600) | Exclusive  Boeing near decision to launch 737 10 jet   sources... |
| 2017-04-25 | BUY | $171.60 | 0.342 | 0.797 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.797) | Is Raytheon Company  RTN  Set To Beat Again In Q1 Earnings ... |
| 2017-04-25 | BUY | $171.60 | 0.349 | 0.560 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.560) | Boeing  BA  Beats On Q1 Earnings  Updates 2017 Guidance... |
| 2017-04-26 | BUY | $169.92 | 0.354 | 0.787 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.354, SAE: 0.787) | Economic Calendar And Watch List  4 26 2017... |
| 2017-04-26 | BUY | $169.92 | 0.346 | 0.699 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.699) | Defense Stock Roundup  TXT  HON  LMT And COL... |
| 2017-04-26 | BUY | $169.92 | 0.344 | 0.753 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.753) | Boeing Q1 core EPS  2 01 vs estimate  1 94 ... |
| 2017-04-27 | BUY | $171.33 | 0.341 | 0.644 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.644) | Trump says was  psyched to terminate NAFTA  but reconsidered... |
| 2017-04-28 | BUY | $172.84 | 0.341 | 0.764 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.764) | Durable Goods Grow Indicating Lack Of Growth... |
| 2017-05-01 | BUY | $170.56 | 0.339 | 0.869 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.869) | At least 27 hurt in turbulence on Aeroflot Moscow Bangkok flight... |
| 2017-05-03 | SELL | $171.49 | 0.341 | 0.866 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.341, SAE: 0.866) | El Al Israel Airlines to start nonstop flights to Miami... |
| 2017-05-04 | BUY | $171.19 | 0.338 | 0.509 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.509) | C919 jet set for maiden flight  in test of China s aviation ambitions... |
| 2017-05-05 | BUY | $173.01 | 0.346 | 0.704 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.704) | China s C919 jet lands successfully after maiden flight... |
| 2017-05-08 | BUY | $173.95 | 0.340 | 0.847 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.847) | More concerns over the A400M... |
| 2017-05-08 | BUY | $173.95 | 0.338 | 0.891 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.891) | Canada political pressures force PM s hand on U S  trade disputes... |
| 2017-05-10 | BUY | $172.61 | 0.346 | 0.882 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.882) | Boeing suspends 737 MAX flights due to engine issue... |
| 2017-05-10 | BUY | $172.61 | 0.335 | 0.638 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.335, SAE: 0.638) | Boeing 737 MAX issue a short term blip   Cowen... |
| 2017-05-11 | BUY | $173.32 | 0.341 | 0.419 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.419) | Boeing must clear 737 MAX engines with FAA to fly again... |
| 2017-05-11 | BUY | $173.32 | 0.346 | 0.885 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.885) | Dow Jones Industrial Average Remains Weak Versus Nasdaq  S P 500... |
| 2017-05-12 | SELL | $172.67 | 0.341 | 0.591 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.341, SAE: 0.591) | Boeing resumes test flights of 737 MAX... |
| 2017-05-17 | BUY | $168.46 | 0.335 | 0.638 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.335, SAE: 0.638) | Orbital ATK  OA  Wins  53 Million Contracts From U S  Army... |
| 2017-05-18 | BUY | $167.16 | 0.347 | 0.668 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.668) | Can ViaSat  VSAT  Keep Earnings Streak Alive In Q4 Earnings ... |
| 2017-05-18 | BUY | $167.16 | 0.357 | 0.829 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.357, SAE: 0.829) | Lockheed Wins  112M Deal To Offer System Support For THAAD... |
| 2017-05-19 | BUY | $170.33 | 0.350 | 0.595 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.350, SAE: 0.595) | Dow notches more than 100 points but ends week in negative... |
| 2017-05-22 | BUY | $173.07 | 0.345 | 0.719 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.719) | Aerospace  Defense ETFs Rise On U S  Saudi Deal... |
| 2017-05-22 | BUY | $173.07 | 0.347 | 0.464 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.464) | Defense Stocks Hit All Time Highs On Arms Deal  5 Best Buys... |
| 2017-05-22 | BUY | $173.07 | 0.344 | 0.621 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.621) | Wall St up as defense stocks gain on Saudi deal ... |
| 2017-05-22 | BUY | $173.07 | 0.347 | 0.868 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.868) | Aerospace And Defense ETFs Soar On U S Saudi Deal... |
| 2017-05-23 | BUY | $172.90 | 0.337 | 0.814 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.814) | The Zacks Analyst Blog Highlights  Northrop Grumman  Rockwell Collins  Curtiss W... |
| 2017-05-23 | BUY | $172.90 | 0.339 | 0.678 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.678) | U S  defense budget proposal sees modest increase despite hawkish rhetoric... |
| 2017-05-23 | BUY | $172.90 | 0.350 | 0.714 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.350, SAE: 0.714) | Boeing  BA  Wins  1 1B Deal For Kill Vehicle Development... |
| 2017-05-26 | BUY | $175.82 | 0.349 | 0.644 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.644) | Lockheed Wins  44M Navy Deal For Trident II Support Services... |
| 2017-05-26 | BUY | $175.82 | 0.347 | 0.643 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.643) | Boeing  BA  Wins  89M Navy Contract For F A 18 Aircraft... |
| 2017-05-31 | SELL | $176.80 | 0.335 | 0.789 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.335, SAE: 0.789) | United fined for flying plane  not in airworthy condition  23 times... |
| 2017-06-01 | BUY | $176.51 | 0.349 | 0.759 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.759) | Lockheed Martin  LMT  Secures  42M Contract For TB 37 MFTA... |
| 2017-06-01 | BUY | $176.51 | 0.347 | 0.628 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.628) | U S  stocks higher at close of trade  Dow Jones Industrial Average up 0 65 ... |
| 2017-06-01 | BUY | $176.51 | 0.349 | 0.789 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.789) | Northrop Grumman  NOC  Wins  244M Deal To Offer AESA Radars... |
| 2017-06-02 | BUY | $179.25 | 0.345 | 0.724 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.724) | Lockheed Martin  LMT  Secures  414M Contract For JASSM ER... |
| 2017-06-06 | BUY | $175.97 | 0.351 | 0.583 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.351, SAE: 0.583) | 3 Top Defense Stocks To Buy Now... |
| 2017-06-07 | BUY | $177.24 | 0.347 | 0.727 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.727) | General Dynamics Wins  105M Deal To Serve USS Makin Island... |
| 2017-06-07 | BUY | $177.24 | 0.347 | 0.827 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.827) | Iran s Aseman signs final deal for 30 Boeing 737s  IRNA... |
| 2017-06-12 | BUY | $179.03 | 0.347 | 0.709 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.709) | Lockheed Martin Unit Wins  108M Air Force Deal For ARTS V2... |
| 2017-06-13 | BUY | $180.06 | 0.341 | 0.655 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.655) | Ryanair in talks with Boeing over new version of 737   sources... |
| 2017-06-14 | BUY | $181.28 | 0.344 | 0.611 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.611) | China s CALC joins line up for Boeing 737 MAX 10  sources... |
| 2017-06-15 | BUY | $184.17 | 0.347 | 0.606 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.606) | Stratasys  SSYS  Seals 3D Printing Deal With Boom Supersonic... |
| 2017-06-15 | BUY | $184.17 | 0.344 | 0.792 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.792) | Boeing sees strong interest in potential new 737 model... |
| 2017-06-15 | BUY | $184.17 | 0.359 | 0.794 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.359, SAE: 0.794) | Textron  TXT  Wins  146 6M Aircraft Repair Order From Navy... |
| 2017-06-16 | BUY | $185.10 | 0.350 | 0.896 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.350, SAE: 0.896) | Boeing  Rolls Royce not yet worried about Middle East orders... |
| 2017-06-16 | BUY | $185.10 | 0.351 | 0.544 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.351, SAE: 0.544) | Lockheed Martin  LMT  Secures  472M FMS Contract For GMLRS... |
| 2017-06-16 | BUY | $185.10 | 0.362 | 0.661 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.362, SAE: 0.661) | iFOREX Daily Analysis   June 16 2017... |
| 2017-06-16 | BUY | $185.10 | 0.336 | 0.760 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.336, SAE: 0.760) | Defense Stock Roundup  James Mattis Favors Budget Hike  UTX Ups Dividend  BA   H... |
| 2017-06-19 | BUY | $187.59 | 0.344 | 0.698 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.698) | Investigating The Durable Goods Ripple Effects Of A Slowdown In Aircraft... |
| 2017-06-19 | BUY | $187.59 | 0.336 | 0.895 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.336, SAE: 0.895) | Here s Why Boeing Is A  Buy ... |
| 2017-06-20 | BUY | $186.88 | 0.347 | 0.665 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.665) | ViaSat  VSAT  To Provide STT For Apache Guardian Helicopters... |
| 2017-06-20 | BUY | $186.88 | 0.345 | 0.846 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.846) | ACG orders 20 Boeing 737 MAX 10 jets... |
| 2017-06-21 | BUY | $187.67 | 0.341 | 0.624 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.624) | Boeing Wins 100 737 MAX 10 Jets Order From United Airlines... |
| 2017-06-21 | BUY | $187.67 | 0.344 | 0.745 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.745) | This Week In The Markets  Milk And Cookies... |
| 2017-06-22 | BUY | $187.93 | 0.348 | 0.588 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.348, SAE: 0.588) | Boeing wins hot Paris order race... |
| 2017-06-22 | BUY | $187.93 | 0.339 | 0.577 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.577) | Lockheed Wins Modification Deal For AEGIS Ballistic Missile... |
| 2017-06-26 | BUY | $188.44 | 0.354 | 0.777 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.354, SAE: 0.777) | Boeing Wins Order For 15 737 MAX 10 Jets From Copa Holdings... |
| 2017-06-26 | BUY | $188.44 | 0.351 | 0.673 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.351, SAE: 0.673) | Lockheed s Unit Wins  131M Navy Deal For A RCI Services... |
| 2017-06-29 | BUY | $186.06 | 0.349 | 0.601 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.601) | Northrop  NOC  Wins  179M Deal To Supply LAIRCM s Hardware... |
| 2017-07-03 | SELL | $187.13 | 0.335 | 0.740 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.335, SAE: 0.740) | Airbus unveils leaner structure and sales shake up... |
| 2017-07-03 | BUY | $187.13 | 0.347 | 0.734 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.734) | Northrop  NOC  Wins Navy Deal For AN ALQ 240 Spare Parts... |
| 2017-07-03 | BUY | $187.13 | 0.337 | 0.825 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.825) | The Zacks Analyst Blog Highlights  Lockheed Martin  General Dynamics  Boeing  No... |
| 2017-07-05 | BUY | $190.16 | 0.344 | 0.609 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.609) | New Dow Theory Signal  What s It Mean For Stocks  ... |
| 2017-07-11 | BUY | $194.36 | 0.342 | 0.630 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.630) | Investing  The Incredible Power Of Staying In The Now ... |
| 2017-07-13 | BUY | $194.33 | 0.349 | 0.860 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.860) | Malaysia says MH17 suspects to face charges could be known by year end... |
| 2017-07-13 | BUY | $194.33 | 0.342 | 0.883 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.883) | General Dynamics  Unit Wins  7 7M Deal From The U S  Navy... |
| 2017-07-17 | BUY | $196.94 | 0.350 | 0.558 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.350, SAE: 0.558) | Lockheed Martin Unit Wins  130M FMS Deal For PAC 3 Missile... |
| 2017-07-19 | BUY | $198.71 | 0.342 | 0.884 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.884) | MH370 search data unveils fishing hot spots  ancient geological movements... |
| 2017-07-20 | BUY | $198.14 | 0.342 | 0.668 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.668) | Stratasys To Produce Polymer Parts For Airbus  A350 Aircraft... |
| 2017-07-20 | BUY | $198.14 | 0.349 | 0.558 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.558) | The week ahead  5 things to watch on the economic calendar... |
| 2017-07-20 | BUY | $198.14 | 0.350 | 0.681 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.350, SAE: 0.681) | Lockheed  LMT  Wins  77M Deal To Support AEGIS Modernization... |
| 2017-07-21 | BUY | $199.89 | 0.342 | 0.767 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.767) | US stocks retreat from highs  nasdaq snaps 10 day win streak... |
| 2017-07-25 | BUY | $200.20 | 0.341 | 0.823 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.823) | Northrop Grumman  NOC  Tops On Q2 Earnings  Ups  17 EPS View... |
| 2017-07-26 | BUY | $219.97 | 0.342 | 0.872 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.872) | Boeing shares make biggest gain since 2009 on second quarter profit... |
| 2017-07-26 | BUY | $219.97 | 0.343 | 0.611 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.611) | Jobless Claims  Durable Goods  Q2 Earnings  All Favorable  Except Twitter ... |
| 2017-07-26 | BUY | $219.97 | 0.346 | 0.641 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.641) | The Fed Knows When To Hold  Em... |
| 2017-07-27 | BUY | $227.09 | 0.360 | 0.474 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.360, SAE: 0.474) | iFOREX Daily Analysis   July 27 2017... |
| 2017-07-31 | BUY | $228.46 | 0.351 | 0.746 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.351, SAE: 0.746) | Boeing expects India to order up to 2 100 aircraft over 20 years... |
| 2017-07-31 | SELL | $228.46 | 0.342 | 0.720 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.342, SAE: 0.720) | Kremlin orders Washington to slash embassy staff in Russia... |
| 2017-08-01 | BUY | $225.62 | 0.358 | 0.845 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.358, SAE: 0.845) | iFOREX Daily Analysis   August 01 2017... |
| 2017-08-07 | BUY | $226.36 | 0.353 | 0.672 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.353, SAE: 0.672) | Stock Market News For August 08  2017... |
| 2017-08-09 | BUY | $222.10 | 0.338 | 0.744 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.744) | Defense Stock Roundup  HII  LDOS  OA Top Q2 Earnings  UTX May Buy COL... |
| 2017-08-10 | BUY | $220.75 | 0.337 | 0.443 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.443) | The Zacks Analyst Blog Highlights  The Boeing  Raytheon Company  Northrop Grumma... |
| 2017-08-14 | BUY | $224.79 | 0.339 | 0.792 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.792) | Is This The Start Of A Hot New Metals Bull Market ... |
| 2017-08-16 | BUY | $225.21 | 0.337 | 0.400 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.400) | Top Research Reports For Home Depot  U S  Bancorp   Alibaba ... |
| 2017-08-16 | BUY | $225.21 | 0.352 | 0.588 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.352, SAE: 0.588) | Lockheed Martin  LMT  Wins  22M Deal For Trident II Support... |
| 2017-08-16 | BUY | $225.21 | 0.355 | 0.867 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.355, SAE: 0.867) | Raytheon Lockheed Martin JV Wins  134M Army Deal For Javelin... |
| 2017-08-17 | BUY | $223.32 | 0.339 | 0.800 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.800) | The Zacks Analyst Blog Highlights  Lockheed Martin  Boeing  General Dynamics And... |
| 2017-08-17 | BUY | $223.32 | 0.344 | 0.535 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.535) | India clears purchase of six Boeing helicopters in  650 million deal... |
| 2017-08-18 | SELL | $223.49 | 0.336 | 0.683 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.336, SAE: 0.683) | US Industrial Production Misses Expectations  ETFs In Focus... |
| 2017-08-22 | BUY | $227.26 | 0.341 | 0.432 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.432) | Top Analyst Reports For Allergan  Ecolab   Exelon ... |
| 2017-08-22 | BUY | $227.26 | 0.337 | 0.725 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.725) | 5 Top Defense Stocks To Buy On Trump s Afghanistan Strategy... |
| 2017-08-22 | BUY | $227.26 | 0.344 | 0.855 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.855) | Is It The Right Time To Invest In Huntington Ingalls  HII  ... |
| 2017-08-22 | BUY | $227.26 | 0.354 | 0.846 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.354, SAE: 0.846) | Northrop  NOC  Wins  329M Deal To Upgrade Minuteman Missile... |
| 2017-08-23 | BUY | $225.68 | 0.339 | 0.578 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.578) | U S  Air Force picks Raytheon  Lockheed for next gen cruise missile... |
| 2017-08-24 | BUY | $226.30 | 0.343 | 0.467 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.467) | Dow 30 Stock Roundup  Cisco To Buy Springpath  Caterpillar July Sales Surge... |
| 2017-08-24 | BUY | $226.30 | 0.336 | 0.710 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.336, SAE: 0.710) | First Airbus A380 parked amid search for new operator... |
| 2017-08-25 | SELL | $223.60 | 0.341 | 0.706 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.341, SAE: 0.706) | U S  core capital goods orders rise  shipments surge... |
| 2017-08-28 | BUY | $224.82 | 0.349 | 0.615 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.615) | Why Is Boeing  BA  Up 1 6  Since The Last Earnings Report ... |
| 2017-08-28 | BUY | $224.82 | 0.350 | 0.795 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.350, SAE: 0.795) | Lockheed Martin Wins  47 8M Deal From US Air Force For F 16... |
| 2017-08-29 | BUY | $227.96 | 0.344 | 0.891 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.891) | U S  stocks higher at close of trade  Dow Jones Industrial Average up 0 26 ... |
| 2017-08-30 | BUY | $227.93 | 0.337 | 0.445 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.445) | United Technologies Rockwell Merger  Behemoth In The Making ... |
| 2017-08-30 | BUY | $227.93 | 0.337 | 0.590 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.590) | Hurricane Harvey could cost United Airlines more than  265 million... |
| 2017-09-05 | SELL | $224.65 | 0.341 | 0.886 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.341, SAE: 0.886) | Japan Airlines plane makes emergency landing in Tokyo... |
| 2017-09-05 | SELL | $224.65 | 0.340 | 0.727 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.340, SAE: 0.727) | Another Buying Opportunity ... |
| 2017-09-05 | BUY | $224.65 | 0.340 | 0.632 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.632) | UTC touts  23 billion deal as stock drops  Boeing turns critic... |
| 2017-09-12 | BUY | $228.05 | 0.342 | 0.806 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.806) | State Department OKs possible sale to Canada of  5 23 billion in military equipm... |
| 2017-09-12 | BUY | $228.05 | 0.353 | 0.572 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.353, SAE: 0.572) | Boeing says signs deal to sell 8 787s  8 737s to Malaysia... |
| 2017-09-13 | BUY | $229.32 | 0.348 | 0.642 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.348, SAE: 0.642) | S P 500 Nears 2 500  5 Stock Picks For Big Profits... |
| 2017-09-14 | BUY | $232.45 | 0.357 | 0.884 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.357, SAE: 0.884) | Boeing  BA  Wins  677M Navy Deal For F A 18E F Aircraft... |
| 2017-09-14 | BUY | $232.45 | 0.357 | 0.810 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.357, SAE: 0.810) | Stock Market News For Sep 15  2017... |
| 2017-09-14 | BUY | $232.45 | 0.352 | 0.795 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.352, SAE: 0.795) | Dow 30 000 Or Bitcoin  30 000  Who Will Win The Race ... |
| 2017-09-18 | BUY | $239.89 | 0.342 | 0.767 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.767) | Northrop Grumman To Buy Orbital ATK In Major  7 8 Billion Defense Deal... |
| 2017-09-19 | BUY | $239.31 | 0.341 | 0.790 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.790) | Zhongwang Marches On With Global Ambitions... |
| 2017-09-19 | BUY | $239.31 | 0.337 | 0.836 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.836) | Exclusive  U S  defense firms want control over tech in Make in India plan... |
| 2017-09-20 | BUY | $242.15 | 0.340 | 0.663 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.663) | Beyond North Korea  4 More Reasons To Buy Defense ETFs... |
| 2017-09-21 | SELL | $242.70 | 0.335 | 0.788 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.335, SAE: 0.788) | Trump Widens North Korea Sanctions  ETFs In Focus... |
| 2017-09-21 | BUY | $242.70 | 0.349 | 0.681 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.681) | Textron Wins  30M Beechcraft King Air 350 Maintenance Deal... |
| 2017-09-22 | BUY | $243.09 | 0.338 | 0.569 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.569) | Ryanair crisis exposes low cost scramble for senior pilots... |
| 2017-09-26 | BUY | $240.48 | 0.348 | 0.601 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.348, SAE: 0.601) | Lockheed  LMT  Wins Deal To Support Japan s AEGIS System ... |
| 2017-09-26 | BUY | $240.48 | 0.340 | 0.874 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.874) | Bombardier shares leap 14 percent ahead of U S  trade court ruling... |
| 2017-09-26 | BUY | $240.48 | 0.342 | 0.666 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.666) | Bombardier skies as much as 14  ahead of U S  trade ruling... |
| 2017-09-28 | BUY | $241.02 | 0.349 | 0.530 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.530) | Northrop s  NOC  Unit Wins Navy Deal To Support LAIRCM... |
| 2017-10-02 | BUY | $242.65 | 0.356 | 0.636 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.356, SAE: 0.636) | Australian searchers say fruitless end is  unacceptable  in final report on MH37... |
| 2017-10-03 | BUY | $242.15 | 0.339 | 0.428 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.428) | Monarch boss  absolutely devastated  after airline s sudden collapse... |
| 2017-10-04 | BUY | $242.43 | 0.361 | 0.872 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.361, SAE: 0.872) | Timeshare stocks on watch amid M A talk... |
| 2017-10-05 | BUY | $245.40 | 0.338 | 0.636 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.636) | Northrop Grumman s Unit Wins  130M Deal For Global Hawk... |
| 2017-10-05 | BUY | $245.40 | 0.350 | 0.700 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.350, SAE: 0.700) | Pence promises the moon and beyond... |
| 2017-10-05 | BUY | $245.40 | 0.336 | 0.573 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.336, SAE: 0.573) | Boeing s Q3 Commercial Deliveries Up Y Y  Defense Order Lags... |
| 2017-10-05 | BUY | $245.40 | 0.353 | 0.771 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.353, SAE: 0.771) | Stock Market News For Oct 9  2017... |
| 2017-10-05 | BUY | $245.40 | 0.339 | 0.495 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.495) | 3 ETFs To Play Upbeat Global Manufacturing ... |
| 2017-10-10 | BUY | $247.33 | 0.362 | 0.636 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.362, SAE: 0.636) | Raytheon  RTN  Wins  54M Deal For F A 18 Aircraft Support... |
| 2017-10-10 | BUY | $247.33 | 0.344 | 0.577 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.577) | Embraer s Q3 Deliveries Down Y Y  Backlog Jumps To  18 8B... |
| 2017-10-11 | BUY | $247.82 | 0.353 | 0.845 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.353, SAE: 0.845) | Boeing  BA  Secures 75 737 MAX Jets Order From Jet Airways... |
| 2017-10-11 | BUY | $247.82 | 0.336 | 0.618 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.336, SAE: 0.618) | Spirit AeroSystems  SPR  Unveils  2B Growth Plan  Backlog Up... |
| 2017-10-11 | BUY | $247.82 | 0.340 | 0.707 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.707) | Emirates willing to cooperate with rival UAE airline Etihad... |
| 2017-10-12 | BUY | $248.26 | 0.342 | 0.652 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.652) | Northrop Grumman Wins  13M Deal For Spider Increment IA... |
| 2017-10-12 | BUY | $248.26 | 0.355 | 0.504 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.355, SAE: 0.504) | Huntington Ingalls  HII  Wins  65M Deal To Modify USS Helena... |
| 2017-10-16 | BUY | $246.22 | 0.345 | 0.779 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.779) | Bombardier pursues options for aerospace division  no deal imminent  sources... |
| 2017-10-16 | BUY | $246.22 | 0.342 | 0.776 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.776) | Boeing  BA  Secures  240M Air Force Deal For AWACS Program... |
| 2017-10-16 | BUY | $246.22 | 0.355 | 0.800 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.355, SAE: 0.800) | Northrop  NOC  Wins Navy Deal To Upgrade EA 18G Aircraft... |
| 2017-10-16 | BUY | $246.22 | 0.347 | 0.775 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.775) | Will Bell Segment Impair Textron s  TXT  Q3 Earnings ... |
| 2017-10-17 | BUY | $245.15 | 0.340 | 0.525 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.525) | Habitual cheat  Kobe Steel faked product data for more than 10 years   source... |
| 2017-10-17 | BUY | $245.15 | 0.344 | 0.749 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.749) | Malaysia says no decision yet on new offers to search for missing MH370... |
| 2017-10-18 | BUY | $246.49 | 0.345 | 0.539 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.539) | Qantas flight to San Francisco turns back after  technical issue ... |
| 2017-10-18 | BUY | $246.49 | 0.356 | 0.755 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.356, SAE: 0.755) |  Stock Market News For Oct 19  2017... |
| 2017-10-19 | BUY | $245.54 | 0.337 | 0.829 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.829) | Textron s  TXT  Unit Secures  333M Contract From U S  Army... |
| 2017-10-19 | BUY | $245.54 | 0.355 | 0.707 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.355, SAE: 0.707) | Northrop  NOC  Arm Wins Deal For Radar Installation In MQ 8C... |
| 2017-10-19 | BUY | $245.54 | 0.345 | 0.636 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.636) | Textron  TXT  Tops Q2 Earnings  Lowers  17 EPS Outlook... |
| 2017-10-20 | BUY | $250.96 | 0.341 | 0.646 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.646) | Big deal for Boeing order book ... |
| 2017-10-20 | BUY | $250.96 | 0.340 | 0.708 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.708) | Singapore Airlines to finalize  13 8 billion Boeing order next week... |
| 2017-10-24 | BUY | $252.14 | 0.344 | 0.605 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.605) | Boeing  BA  Beats Q3 Earnings And Revenues Estimates... |
| 2017-10-25 | BUY | $244.96 | 0.343 | 0.824 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.824) | Exclusive  Canada pushed for Airbus deal as Bombardier courted China... |
| 2017-10-25 | BUY | $244.96 | 0.344 | 0.579 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.579) | Are Options Traders Betting On A Big Move In Boeing  BA  Stock  ... |
| 2017-10-31 | BUY | $244.54 | 0.340 | 0.742 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.742) | U S  nuclear arsenal to cost  1 2 trillion over next 30 years  CBO... |
| 2017-11-01 | BUY | $245.03 | 0.354 | 0.852 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.354, SAE: 0.852) | AAR  6 5  as federal court upholds award... |
| 2017-11-02 | BUY | $248.95 | 0.344 | 0.784 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.784) | Goldman s Blankfein is only major financial firm CEO to join Trump on China trip... |
| 2017-11-08 | SELL | $251.73 | 0.338 | 0.879 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.338, SAE: 0.879) | Boeing signs China deals worth  37 billion  state TV... |
| 2017-11-09 | BUY | $250.35 | 0.340 | 0.898 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.898) | Boeing signs deal to sell 300 planes worth  37 billion to China... |
| 2017-11-09 | BUY | $250.35 | 0.343 | 0.721 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.721) | Trump s  250 billion China  miracle  adds gloss to  off kilter  trade... |
| 2017-11-09 | BUY | $250.35 | 0.339 | 0.674 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.674) | General Electric signs  3 5 billion deals in China... |
| 2017-11-09 | BUY | $250.35 | 0.343 | 0.743 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.743) | Exclusive  Japan to delay multi billion dollar fighter jet development   sources... |
| 2017-11-09 | BUY | $250.35 | 0.343 | 0.767 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.767) | Membership revoked  Veteran GE s spot in exclusive Dow may be shaky... |
| 2017-11-09 | BUY | $250.35 | 0.355 | 0.749 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.355, SAE: 0.749) | Emirates places provisional order for 40 Boeing 787 10... |
| 2017-11-09 | BUY | $250.35 | 0.351 | 0.479 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.351, SAE: 0.479) | Boeing nears deal with Emirates for 787 10 jets  sources... |
| 2017-11-13 | BUY | $250.08 | 0.349 | 0.772 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.772) | Dubai pressures Airbus as A380 order hopes fizzle... |
| 2017-11-13 | BUY | $250.08 | 0.350 | 0.440 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.350, SAE: 0.440) | Stock Market News For Nov 14  2017... |
| 2017-11-14 | BUY | $249.46 | 0.344 | 0.688 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.688) | Boeing To Deliver 20 737 MAX Jets Worth  2 2B To ALAFCO... |
| 2017-11-14 | BUY | $249.46 | 0.341 | 0.489 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.489) | Ethiopian could buy 10 20 of Boeing s proposed mid sized jet... |
| 2017-11-15 | BUY | $250.50 | 0.356 | 0.615 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.356, SAE: 0.615) | Top 5 Things to Know in the Market on Wednesday... |
| 2017-11-15 | BUY | $250.50 | 0.342 | 0.816 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.816) | Boeing in 175 plane deal with budget carrier flydubai... |
| 2017-11-15 | BUY | $250.50 | 0.349 | 0.422 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.422) | Company News For Nov 16  2017... |
| 2017-11-16 | BUY | $251.30 | 0.338 | 0.452 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.452) | Chatham Lodging Buys Greater Charleston Hotel For  20 2M... |
| 2017-11-16 | BUY | $251.30 | 0.356 | 0.770 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.356, SAE: 0.770) | Boeing  BA  Secures  11B Order For 75 737 MAX Airplanes... |
| 2017-11-16 | BUY | $251.30 | 0.341 | 0.849 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.849) | Airbus faces tricky hurdles over stalled A380 Emirates deal... |
| 2017-11-17 | SELL | $249.93 | 0.335 | 0.755 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.335, SAE: 0.755) | Is U S  North Korea Tiff Driving The U S  Defense Industry ... |
| 2017-11-20 | BUY | $252.19 | 0.343 | 0.608 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.608) | China s Okay Airways orders five Boeing 787 9 Dreamliners for  1 4 billion... |
| 2017-11-20 | BUY | $252.19 | 0.357 | 0.780 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.357, SAE: 0.780) | Okay Airways orders five 787s... |
| 2017-11-21 | BUY | $254.44 | 0.343 | 0.402 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.402) | Why You Should Bank On Industrial Metals For The Long Haul... |
| 2017-11-28 | BUY | $255.39 | 0.344 | 0.741 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.741) | Airbus poaches Rolls Royce executive to head aircraft sales... |
| 2017-11-29 | BUY | $256.64 | 0.356 | 0.881 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.356, SAE: 0.881) | Lockheed Martin Wins  72M Deal To Support Trident II Missile... |
| 2017-12-04 | BUY | $264.90 | 0.342 | 0.832 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.832) | Astrotech up 22  premarket after TRACER 1000 demos... |
| 2017-12-04 | SELL | $264.90 | 0.346 | 0.429 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.346, SAE: 0.429) | Exclusive  Major U S  trucking firm Daseke buys three firms... |
| 2017-12-05 | BUY | $262.59 | 0.349 | 0.695 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.695) | Premarket Losers as of 9 05 am... |
| 2017-12-05 | BUY | $262.59 | 0.351 | 0.661 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.351, SAE: 0.661) | Reuters  Canada scraps plan to buy 18 Boeing fighter jets... |
| 2017-12-05 | BUY | $262.59 | 0.352 | 0.491 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.352, SAE: 0.491) | Boeing seeks 777 8 tweak for Qantas... |
| 2017-12-05 | BUY | $262.59 | 0.352 | 0.702 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.352, SAE: 0.702) | Canada scraps plan to buy Boeing fighters amid trade dispute  sources... |
| 2017-12-06 | BUY | $265.19 | 0.344 | 0.563 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.563) | Spirit AeroSystems Up On  1B Expansion Plan  1K Job Additions... |
| 2017-12-06 | BUY | $265.19 | 0.340 | 0.443 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.443) | Air New Zealand cancels flights after  events  involving Rolls Royce engines... |
| 2017-12-07 | BUY | $268.72 | 0.343 | 0.734 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.734) | General Dynamics Wins  46M Deal For Submarine Maintenance... |
| 2017-12-07 | BUY | $268.72 | 0.363 | 0.453 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.363, SAE: 0.453) | L 3 Technologies Secures Logistics Deal For T 1A Aircraft... |
| 2017-12-08 | BUY | $272.46 | 0.343 | 0.882 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.882) | Boeing unmoved by reported Canadian plan to buy used Australian jets... |
| 2017-12-12 | BUY | $276.31 | 0.353 | 0.445 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.353, SAE: 0.445) | Stock Market News For Dec 13  2017... |
| 2017-12-12 | BUY | $276.31 | 0.351 | 0.612 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.351, SAE: 0.612) | Stocks Mixed As Tech Dips ... |
| 2017-12-12 | BUY | $276.31 | 0.346 | 0.505 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.505) | Dow Jones Notches 118 Points Amid Tax Reform Optimism   ... |
| 2017-12-14 | BUY | $280.07 | 0.344 | 0.699 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.699) | Boeing  BA  Hits A 52 Week High On Consistent Performance... |
| 2017-12-15 | BUY | $280.12 | 0.342 | 0.858 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.858) | Boeing price target hiked to  310 as Morgan Stanley sees solid year ahead... |
| 2017-12-20 | BUY | $283.90 | 0.337 | 0.744 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.744) | Will Tax Cuts Keep Us At 3  GDP ... |
| 2017-12-20 | BUY | $283.90 | 0.339 | 0.642 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.642) | Boeing seen eyeing broad Embraer deal  but no firm proposal made... |
| 2017-12-21 | BUY | $281.16 | 0.349 | 0.823 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.823) | Boeing Secures  1 2B Deal For Manufacturing P 8A Aircraft... |
| 2017-12-21 | BUY | $281.16 | 0.344 | 0.671 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.671) | U S  tax cuts mix pleasure and pain for Europe s dollar earners... |
| 2017-12-21 | SELL | $281.16 | 0.335 | 0.896 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.335, SAE: 0.896) | U S  Equity ETF  DDM  Hits A New 52 Week High... |
| 2017-12-21 | BUY | $281.16 | 0.334 | 0.474 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.334, SAE: 0.474) | World s largest amphibious aircraft makes maiden flight in China... |
| 2017-12-21 | BUY | $281.16 | 0.345 | 0.685 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.685) | Boeing secures  27 billion order for 175 aircraft from flydubai... |
| 2017-12-22 | BUY | $281.23 | 0.335 | 0.793 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.335, SAE: 0.793) | Temer says Brazil will not cede control of Embraer but open to cash infusion... |
| 2017-12-22 | SELL | $281.23 | 0.339 | 0.496 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.339, SAE: 0.496) | U S  core capital goods orders dip  shipments increase... |
| 2017-12-27 | BUY | $281.72 | 0.343 | 0.822 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.822) | Airbus ready to phase out A380 if fails to win Emirates deal  sources... |
| 2018-01-02 | BUY | $282.89 | 0.352 | 0.768 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.352, SAE: 0.768) | Airbus delivered over 700 jets in 2017  met target  sources... |
| 2018-01-03 | BUY | $283.80 | 0.341 | 0.716 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.716) | AAR Corp   AIR  Progresses In MRO Business  Should You Hold ... |
| 2018-01-04 | BUY | $282.72 | 0.345 | 0.604 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.604) | Boeing Wins  193M FMS Deal To Build Small Diameter Bomb... |
| 2018-01-04 | BUY | $282.72 | 0.346 | 0.768 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.768) | Best Week Of The Year ... |
| 2018-01-05 | BUY | $294.32 | 0.342 | 0.523 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.523) | Dow Jones Bags 200 Points  Notches Best Week Since November... |
| 2018-01-05 | BUY | $294.32 | 0.344 | 0.828 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.828) | Embraer down 5  on profit taking  talks with Boeing advancing... |
| 2018-01-09 | BUY | $303.46 | 0.340 | 0.680 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.680) | Boeing shares have more room to fly in 2018  analysts... |
| 2018-01-10 | BUY | $305.21 | 0.341 | 0.677 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.677) | Canada takes U S  to WTO  U S  says case helps China... |
| 2018-01-10 | SELL | $305.21 | 0.338 | 0.407 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.338, SAE: 0.407) | Boeing unveils prototype for unmanned electric cargo air vehicle... |
| 2018-01-11 | BUY | $312.70 | 0.340 | 0.670 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.670) | Kenya Airways CEO sees 10 percent revenue bump from direct U S  flights... |
| 2018-01-11 | BUY | $312.70 | 0.341 | 0.657 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.657) | Walmart hikes minimum wage  announces layoffs on same day... |
| 2018-01-11 | BUY | $312.70 | 0.347 | 0.874 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.874) | Sidoti positive on Brink s Company... |
| 2018-01-12 | BUY | $320.41 | 0.344 | 0.646 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.646) | U S  stocks higher at close of trade  Dow Jones Industrial Average up 0 89 ... |
| 2018-01-12 | BUY | $320.41 | 0.341 | 0.770 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.770) | Airbus May Stall A380 Production  Is Boeing 777X In Focus ... |
| 2018-01-16 | BUY | $319.41 | 0.355 | 0.811 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.355, SAE: 0.811) | Company News For Jan 17  2018... |
| 2018-01-18 | SELL | $324.17 | 0.342 | 0.632 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.342, SAE: 0.632) | General Electric hits fresh multi year low... |
| 2018-01-18 | BUY | $324.17 | 0.340 | 0.641 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.641) | Exclusive  Boeing willing to preserve Brazil s  golden share  in Embraer deal... |
| 2018-01-22 | BUY | $322.11 | 0.344 | 0.695 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.695) | GD Vs  LMT  Which Is A Better Buy Ahead Of Q4 Earnings ... |
| 2018-01-23 | BUY | $319.81 | 0.361 | 0.658 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.361, SAE: 0.658) | S P 500 Beats 54 Year Old Record  Top 5 Gainers... |
| 2018-01-23 | BUY | $319.81 | 0.348 | 0.623 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.348, SAE: 0.623) | Boeing completing 787 10 flight tests with GE engines... |
| 2018-01-24 | BUY | $318.96 | 0.341 | 0.634 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.634) | Northrop Grumman  NOC  Beats Q4 Earnings   Revenue Estimates... |
| 2018-01-24 | BUY | $318.96 | 0.340 | 0.668 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.668) | Will Solid Orders Boost Lockheed Martin s  LMT  Q4 Earnings ... |
| 2018-01-24 | BUY | $318.96 | 0.346 | 0.522 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.522) | Qatar Airways  first Airbus A350 1000 to be delivered Feb 15 20  CEO... |
| 2018-01-25 | BUY | $326.98 | 0.338 | 0.720 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.720) | This Fantastic 4 Makes FANG Look Tame... |
| 2018-01-25 | BUY | $326.98 | 0.351 | 0.735 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.351, SAE: 0.735) | Boeing  BA  Clinches  150M Deal To Support CDA Platform... |
| 2018-01-26 | BUY | $327.09 | 0.357 | 0.683 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.357, SAE: 0.683) | Rockwell Collins beats by  0 04  beats on revenue... |
| 2018-01-26 | BUY | $327.09 | 0.338 | 0.421 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.421) | At NAFTA talks  Canada hails jet case as victory for free trade... |
| 2018-01-29 | BUY | $324.80 | 0.336 | 0.898 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.336, SAE: 0.898) | Market Confusion On U S  Dollar  Euro Strength And New Inputs... |
| 2018-01-29 | BUY | $324.80 | 0.340 | 0.647 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.647) | Meredith  MDP  Hikes Dividend 4 8  Just Ahead Of Q2 Earnings... |
| 2018-01-29 | SELL | $324.80 | 0.337 | 0.835 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.337, SAE: 0.835) | Progress In The NAFTA Negotiations ... |
| 2018-01-30 | BUY | $321.84 | 0.345 | 0.610 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.610) | Boeing KC 46A tanker could be certified in 60 days  military brass says... |
| 2018-01-30 | BUY | $321.84 | 0.344 | 0.620 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.620) | Boeing  BA  Beats On Q4 Earnings And Revenue Estimates... |
| 2018-01-31 | BUY | $337.71 | 0.353 | 0.758 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.353, SAE: 0.758) | Markets Rebound Bigly On ADP  Q4  SOTU... |
| 2018-01-31 | BUY | $337.71 | 0.346 | 0.636 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.636) | ADP  Q4  SOTU Boost Market... |
| 2018-01-31 | BUY | $337.71 | 0.341 | 0.891 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.891) | Globalization  Keynote Speech In Luxembourg... |
| 2018-02-01 | BUY | $340.16 | 0.341 | 0.724 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.724) | Boeing s Strong Results Buoy The Dow Jones Industrial Average  For Now... |
| 2018-02-01 | SELL | $340.16 | 0.337 | 0.809 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.337, SAE: 0.809) | Lowe s to pay U S  staff  1 000 bonus following tax reform... |
| 2018-02-02 | SELL | $332.51 | 0.338 | 0.829 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.338, SAE: 0.829) | No Samsung phones  Nike uniforms for North Koreans  Sanctions cloud Olympic perk... |
| 2018-02-05 | BUY | $313.42 | 0.347 | 0.499 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.499) | Boeing debuts first 737 MAX 7... |
| 2018-02-05 | BUY | $313.42 | 0.346 | 0.716 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.716) | A decade after recession  a jump in U S  states with wage gains... |
| 2018-02-05 | BUY | $313.42 | 0.342 | 0.880 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.880) | Model planes and chocolate hearts  Air Berlin surprised by auction success... |
| 2018-02-05 | BUY | $313.42 | 0.344 | 0.663 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.663) | Boeing  India in talks over F A 18 fighter deal... |
| 2018-02-08 | BUY | $315.71 | 0.342 | 0.761 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.761) | Before You  Buy The Dip   Look At This Chart ... |
| 2018-02-08 | BUY | $315.71 | 0.337 | 0.862 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.862) | The Zacks Analyst Blog Highlights  Lockheed Martin  Boeing  Textron  Triumph And... |
| 2018-02-15 | BUY | $341.38 | 0.342 | 0.842 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.842) | All 65 passengers  crew feared dead in Iranian plane crash... |
| 2018-02-15 | BUY | $341.38 | 0.344 | 0.622 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.622) | Boeing stays in race to supply Canada with fighter jets  sources... |
| 2018-02-16 | BUY | $340.02 | 0.335 | 0.859 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.335, SAE: 0.859) | Iran cannot confirm missing plane found  freeze hampers search... |
| 2018-02-16 | BUY | $340.02 | 0.338 | 0.707 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.707) | Trump Chalks Out Fiscal 2019 Budget Plan  5 Defense Picks... |
| 2018-02-16 | BUY | $340.02 | 0.349 | 0.744 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.744) | 5 ETFs That Deserve Honor On Presidents  Day... |
| 2018-02-21 | BUY | $337.46 | 0.359 | 0.701 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.359, SAE: 0.701) | Curtiss Wright beats by  0 06  beats on revenue... |
| 2018-02-22 | BUY | $340.86 | 0.343 | 0.672 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.672) | Boeing to have 51 percent stake in venture with Embraer  paper... |
| 2018-02-26 | BUY | $348.10 | 0.351 | 0.656 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.351, SAE: 0.656) | Company News For Feb 27  2018... |
| 2018-03-01 | BUY | $334.90 | 0.343 | 0.611 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.611) | 5 Top Dividend Aristocrats To Buy In March... |
| 2018-03-01 | BUY | $334.90 | 0.343 | 0.715 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.715) | 5 Stocks To  Steel  The Show On Proposed Trump Tariffs... |
| 2018-03-06 | BUY | $334.16 | 0.349 | 0.650 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.650) | Boeing  BA  To Deliver 787 Dreamliners To Hawaiian Airlines... |
| 2018-03-06 | BUY | $334.16 | 0.355 | 0.414 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.355, SAE: 0.414) | Boeing Secures  282M Deal To Supply Parts For 19 P 8A Jets... |
| 2018-03-07 | BUY | $332.36 | 0.341 | 0.634 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.634) | Malaysia says MH370 report to be released after latest search ends... |
| 2018-03-07 | BUY | $332.36 | 0.342 | 0.406 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.406) | Cohn s Exit Raises Odds Of A Trade War  4 Top Stocks To Buy... |
| 2018-03-07 | BUY | $332.36 | 0.343 | 0.744 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.744) | France signs deals worth  16 billion in India  to deepen defense  security ties... |
| 2018-03-07 | BUY | $332.36 | 0.347 | 0.783 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.783) | The 3 Best  Rising Rate Plays  For Dividends And Upside ... |
| 2018-03-08 | BUY | $333.98 | 0.348 | 0.424 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.348, SAE: 0.424) | Putin order for bomb threat plane to be downed in 2014 canceled after false alar... |
| 2018-03-08 | BUY | $333.98 | 0.352 | 0.852 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.352, SAE: 0.852) | Stock Market News For March 12  2018... |
| 2018-03-09 | BUY | $339.52 | 0.353 | 0.712 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.353, SAE: 0.712) | Huntington Ingalls Wins USS Fitzgerald Repair Deal Worth  77M... |
| 2018-03-12 | BUY | $329.63 | 0.348 | 0.762 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.348, SAE: 0.762) | New High For NASDAQ As Streak Hits 7 Days... |
| 2018-03-12 | BUY | $329.63 | 0.350 | 0.644 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.350, SAE: 0.644) | General Dynamics Wins  696M Deal For Nuclear Submarines... |
| 2018-03-12 | BUY | $329.63 | 0.349 | 0.680 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.680) | Raytheon  RTN  Wins  40M Deal For B 2 Antenna Modification... |
| 2018-03-13 | BUY | $324.34 | 0.343 | 0.745 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.745) | Boeing studying options for further boost to 737 production... |
| 2018-03-14 | BUY | $316.29 | 0.347 | 0.789 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.789) | Dow sheds 250 points as trade war worries weigh on industrials... |
| 2018-03-14 | BUY | $316.29 | 0.354 | 0.540 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.354, SAE: 0.540) | Day Ahead  Top 3 Things To Watch... |
| 2018-03-14 | BUY | $316.29 | 0.341 | 0.417 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.417) | Financials fare worst today... |
| 2018-03-16 | BUY | $316.49 | 0.352 | 0.746 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.352, SAE: 0.746) | Top 5 Things That Moved Markets This Past Week... |
| 2018-03-20 | BUY | $323.35 | 0.351 | 0.717 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.351, SAE: 0.717) | Stock Market News For Mar 21  2018... |
| 2018-03-20 | BUY | $323.35 | 0.341 | 0.583 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.583) | Why Value Investors Should Consider Boeing  BA  Right Now... |
| 2018-03-20 | BUY | $323.35 | 0.337 | 0.733 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.733) | Exclusive  Trump to boost exports of lethal drones to more U S  allies... |
| 2018-03-21 | BUY | $322.84 | 0.349 | 0.654 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.654) | China Southern places  3 6 billion Boeing 737 MAX order for Xiamen Airlines subs... |
| 2018-03-21 | BUY | $322.84 | 0.339 | 0.609 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.609) | Southwest Airlines  4  after lowering Q1 RASM outlook... |
| 2018-03-22 | BUY | $306.09 | 0.342 | 0.603 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.603) | Aerospace   Defense Industry Outlook   March 2018... |
| 2018-03-22 | BUY | $306.09 | 0.352 | 0.639 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.352, SAE: 0.639) | Day Ahead  Top 3 Things to Watch ... |
| 2018-03-23 | BUY | $307.42 | 0.340 | 0.414 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.414) | Canada calls end to trade case positive for  longstanding  Boeing ties... |
| 2018-03-23 | BUY | $307.42 | 0.344 | 0.716 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.716) | Boeing set to win American wide body jet order  sources... |
| 2018-03-26 | BUY | $315.05 | 0.342 | 0.756 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.756) | Boeing Remains A Buy For ValuEngine... |
| 2018-03-26 | BUY | $315.05 | 0.350 | 0.761 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.350, SAE: 0.761) | Lockheed  LMT  Wins  54M Deal To Aid THAAD Missile System... |
| 2018-03-26 | BUY | $315.05 | 0.342 | 0.490 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.490) | Boeing completes Dreamliner family with first 787 10 delivery... |
| 2018-03-27 | BUY | $307.54 | 0.356 | 0.777 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.356, SAE: 0.777) | Raytheon  RTN  Wins  85M Deal To Supply H 60 Spare Parts... |
| 2018-03-29 | BUY | $314.01 | 0.351 | 0.773 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.351, SAE: 0.773) | Boeing  BA  Wins  1 2B Deal To Support Kuwait s F A 18 Jets... |
| 2018-04-02 | BUY | $308.80 | 0.338 | 0.689 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.689) | General Dynamics New Offer For CSRA Outbids CACI ... |
| 2018-04-03 | BUY | $316.83 | 0.343 | 0.692 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.692) | Lockheed  LMT  Wins  279M FMS Contract For PAC 3 Missiles... |
| 2018-04-03 | BUY | $316.83 | 0.353 | 0.727 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.353, SAE: 0.727) | Boeing To Deliver 75 737 Max Jets Worth  8 8B To Jet Airways ... |
| 2018-04-04 | BUY | $313.59 | 0.350 | 0.559 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.350, SAE: 0.559) | Stock Markets And A Civilized Trade Solution... |
| 2018-04-04 | SELL | $313.59 | 0.346 | 0.422 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.346, SAE: 0.422) | China Slaps Tariffs  Dollar Dives ... |
| 2018-04-04 | SELL | $313.59 | 0.334 | 0.790 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.334, SAE: 0.790) | U S  Leaves Door Open to Talks With China Amid Trade War Fears... |
| 2018-04-04 | BUY | $313.59 | 0.335 | 0.770 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.335, SAE: 0.770) | U S  Leaves Door Open to China Talks Amid Trade War Fears... |
| 2018-04-04 | BUY | $313.59 | 0.343 | 0.688 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.688) | India s Jet Airways agrees to buy 75 Boeing 737 MAX jets worth  8 8 billion... |
| 2018-04-04 | BUY | $313.59 | 0.352 | 0.824 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.352, SAE: 0.824) | Company News For April 5  2018... |
| 2018-04-05 | BUY | $322.17 | 0.345 | 0.604 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.604) | Lockheed Martin  LMT  Wins  117M Contract For LRFD Delivery... |
| 2018-04-05 | BUY | $322.17 | 0.355 | 0.573 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.355, SAE: 0.573) | L3 Technologies Wins Deal To Upgrade F A 18   EA 18 Jets... |
| 2018-04-05 | BUY | $322.17 | 0.337 | 0.715 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.715) | Kudlow  China Push Time to Talk Message Amid Trade Tension... |
| 2018-04-06 | BUY | $312.32 | 0.336 | 0.504 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.336, SAE: 0.504) | Trump threatens more China tariffs  Beijing ready to hit back... |
| 2018-04-06 | BUY | $312.32 | 0.341 | 0.778 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.778) | India opens contest for more than 100 fighter jets... |
| 2018-04-06 | BUY | $312.32 | 0.342 | 0.838 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.838) | Defense Stock Roundup  BA  RTN Win Deals  GD Closes CSRA Buyout... |
| 2018-04-09 | BUY | $308.84 | 0.344 | 0.767 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.767) | Boeing Wins F 15SA Deal For Cumulative Face Value Of  305M... |
| 2018-04-09 | BUY | $308.84 | 0.354 | 0.793 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.354, SAE: 0.793) | An Antidote To Global Trade Jitters  Check Out These Buying Options... |
| 2018-04-10 | BUY | $320.67 | 0.341 | 0.705 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.705) | Malaysia Airlines launches widebody tender process  could oust Boeing 787 deal  ... |
| 2018-04-10 | BUY | $320.67 | 0.339 | 0.626 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.626) | Boeing  BA  Q1 Commercial Deliveries Up Y Y  Defense Slips... |
| 2018-04-10 | BUY | $320.67 | 0.355 | 0.657 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.355, SAE: 0.657) | Stock Market News For Apr 11  2018... |
| 2018-04-11 | BUY | $313.51 | 0.345 | 0.480 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.480) | Airbus sees backloaded deliveries in 2018  reaffirms target... |
| 2018-04-11 | BUY | $313.51 | 0.349 | 0.737 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.737) | Boeing  Lion Air Group Ink  6 2B Deal For 50 737 MAX 10 Jets... |
| 2018-04-13 | BUY | $315.35 | 0.339 | 0.680 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.680) | Analyst Who Called  98 Russia Crash Says Stock Outlook Is  Grim ... |
| 2018-04-16 | SELL | $317.74 | 0.343 | 0.638 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.343, SAE: 0.638) | 3 Defense Picks If Russia Intervenes In U S  Syria Crisis... |
| 2018-04-17 | BUY | $322.48 | 0.355 | 0.739 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.355, SAE: 0.739) | Lockheed Martin Wins  200M Deal To Upgrade THAAD And PATRIOT... |
| 2018-04-17 | BUY | $322.48 | 0.346 | 0.834 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.834) | Textron  TXT  Tops Q1 Earnings Estimate  Keeps  18 EPS View... |
| 2018-04-19 | BUY | $326.23 | 0.341 | 0.504 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.504) | Trump launches effort to boost U S  weapons sales abroad... |
| 2018-04-19 | BUY | $326.23 | 0.345 | 0.867 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.867) | GE Rated SELL In The Aftermath Of Southwest Incident... |
| 2018-04-20 | BUY | $324.34 | 0.343 | 0.815 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.815) | Can Order Growth Drive Lockheed Martin s  LMT  Q1 Earnings ... |
| 2018-04-20 | BUY | $324.34 | 0.339 | 0.602 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.602) | Will Higher 737 Deliveries Drive Boeing s  BA  Q1 Earnings ... |
| 2018-04-23 | BUY | $324.51 | 0.337 | 0.523 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.523) | An Airline ETF For Those Seeking Profit Margins... |
| 2018-04-23 | BUY | $324.51 | 0.343 | 0.717 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.717) | Will Missile Systems Unit Drive Raytheon  RTN  Q1 Earnings ... |
| 2018-04-23 | BUY | $324.51 | 0.349 | 0.618 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.618) | Lockheed  LMT  Beats On Q1 Earnings  Raises  18 Guidance... |
| 2018-04-24 | BUY | $315.14 | 0.343 | 0.730 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.730) | Should You Buy Boeing  BA  Ahead Of Earnings ... |
| 2018-04-24 | BUY | $315.14 | 0.342 | 0.728 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.728) | A Funny Thing Happened On The Way To Q1 Earnings   ... |
| 2018-04-25 | BUY | $328.36 | 0.349 | 0.783 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.783) | Financial  tech stocks lead Wall Street lower as yields rise... |
| 2018-04-25 | BUY | $328.36 | 0.352 | 0.536 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.352, SAE: 0.536) | Stocks  U S  Futures Fall as Bond Yields Rise... |
| 2018-04-25 | BUY | $328.36 | 0.356 | 0.784 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.356, SAE: 0.784) | Top 5 Things to Know in the Market on Wednesday... |
| 2018-04-25 | BUY | $328.36 | 0.340 | 0.684 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.684) | Boeing cruises past forecasts as margins  sales grow... |
| 2018-04-25 | BUY | $328.36 | 0.350 | 0.793 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.350, SAE: 0.793) | U S  Treasury yield curve flattening to intensify with U S  pension plan bond bu... |
| 2018-04-26 | SELL | $328.29 | 0.340 | 0.761 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.340, SAE: 0.761) | U S  Corporate Earnings Buoy Global Equities Despite 3  U S  Yields... |
| 2018-04-26 | SELL | $328.29 | 0.342 | 0.851 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.342, SAE: 0.851) | U S  core capital goods orders  shipments fall in March... |
| 2018-04-26 | BUY | $328.29 | 0.346 | 0.764 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.764) | Rockwell Collins  COL  Beats On Q2 Earnings  Sales Up Y Y... |
| 2018-04-26 | BUY | $328.29 | 0.359 | 0.707 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.359, SAE: 0.707) | iFOREX Daily Analysis   April 26 2018... |
| 2018-04-27 | BUY | $326.46 | 0.341 | 0.508 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.508) | Southwest Airlines orders 40 Boeing 737 MAX jets worth  4 68 billion... |
| 2018-04-30 | BUY | $319.45 | 0.341 | 0.681 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.681) | Embraer  ERJ  Misses Q1 Earnings Estimates  Keeps  18 View... |
| 2018-05-01 | BUY | $315.60 | 0.346 | 0.623 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.623) | Spirit AeroSystems  SPR  Q1 Earnings Miss  Revenues Up Y Y... |
| 2018-05-01 | BUY | $315.60 | 0.343 | 0.740 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.740) | Gogo  GOGO  To Report Q1 Earnings  What s In The Cards ... |
| 2018-05-02 | SELL | $310.48 | 0.345 | 0.440 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.345, SAE: 0.440) | Southwest jet makes emergency stop in Cleveland with cracked window... |
| 2018-05-02 | BUY | $310.48 | 0.339 | 0.763 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.763) | First NASA lander to study Mars  interior due for California launch... |
| 2018-05-03 | BUY | $316.70 | 0.347 | 0.895 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.895) | Rolls Royce says working quickly to fix Trent 1000 problems... |
| 2018-05-04 | SELL | $320.28 | 0.338 | 0.495 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.338, SAE: 0.495) | Spacecraft for detecting  Marsquakes  set for rare California launch... |
| 2018-05-10 | BUY | $331.16 | 0.339 | 0.745 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.745) | Defense Stocks Rally On Iran Deal  4 Hot Picks... |
| 2018-05-14 | BUY | $331.66 | 0.348 | 0.620 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.348, SAE: 0.620) | Why You Should Hold Genpact  G  Stock Right Now... |
| 2018-05-15 | BUY | $329.28 | 0.354 | 0.703 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.354, SAE: 0.703) | Wolfe cashes in some chips on TransDigm... |
| 2018-05-15 | BUY | $329.28 | 0.343 | 0.832 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.832) | Northrop Grumman  NOC  Approves 9  Increase In Dividend... |
| 2018-05-16 | BUY | $328.17 | 0.351 | 0.742 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.351, SAE: 0.742) | How Rusal escaped the noose of U S  sanctions... |
| 2018-05-18 | BUY | $338.05 | 0.346 | 0.711 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.711) | Passenger plane crashes in Cuba  killing more than 100 people... |
| 2018-05-21 | BUY | $350.26 | 0.348 | 0.774 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.348, SAE: 0.774) | Cuba plane crash death toll rises to 111  Mexico suspends lease company... |
| 2018-05-21 | BUY | $350.26 | 0.341 | 0.823 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.823) | Sector ETFs   Stocks To Surge As US China Trade Fear Ebbs... |
| 2018-05-21 | BUY | $350.26 | 0.351 | 0.789 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.351, SAE: 0.789) | Stock Market News For May 22  2018... |
| 2018-05-21 | BUY | $350.26 | 0.338 | 0.585 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.585) | Delta To Expand In New York With Flights To Orange County... |
| 2018-05-22 | BUY | $341.70 | 0.349 | 0.665 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.665) | Pound Softer Ahead Of Mark Carney In The Hot Seat... |
| 2018-05-22 | BUY | $341.70 | 0.337 | 0.844 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.844) | U S  China Trade Truce Optimism Fades In Asia... |
| 2018-05-22 | BUY | $341.70 | 0.339 | 0.584 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.584) | 3 Stocks Back In Play After U S  China Trade War Stalls... |
| 2018-05-23 | BUY | $345.73 | 0.341 | 0.603 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.603) | Malaysia says search for Flight MH370 to end next week... |
| 2018-05-25 | BUY | $346.58 | 0.344 | 0.776 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.776) | U S  ambassador skips encounter with sanctioned Russian tycoon... |
| 2018-05-25 | BUY | $346.58 | 0.349 | 0.659 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.659) | U S  business spending on equipment improving... |
| 2018-05-29 | BUY | $339.25 | 0.339 | 0.895 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.895) | SEC Obtains Court Order Against ICO Defrauding Investors of Millions... |
| 2018-06-01 | BUY | $343.33 | 0.344 | 0.761 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.761) | Trade War Tensions Flare Up  Must Watch ETFs   Stocks... |
| 2018-06-05 | BUY | $346.59 | 0.341 | 0.683 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.683) | Emirates sees oil  dollar  double whammy   but demand strong... |
| 2018-06-06 | BUY | $357.62 | 0.340 | 0.473 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.473) | Northrop Grumman Wins FTC Nod To Buy Orbital ATK  Ups View... |
| 2018-06-08 | BUY | $355.63 | 0.340 | 0.841 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.841) | Looking For Earnings Beat  5 Stocks Likely To Fit The Bill... |
| 2018-06-08 | BUY | $355.63 | 0.338 | 0.601 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.601) | China s export growth steady in May  import growth faster but not from U S ... |
| 2018-06-11 | BUY | $357.02 | 0.342 | 0.867 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.867) | U S  bound flight diverted to Ireland after security scare... |
| 2018-06-11 | BUY | $357.02 | 0.347 | 0.815 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.815) | Why Is Air Lease  AL  Up 1 2  Since Its Last Earnings Report ... |
| 2018-06-11 | BUY | $357.02 | 0.336 | 0.802 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.336, SAE: 0.802) | Rolls Royce  preparing to cut thousands of jobs  says engine problem has spread... |
| 2018-06-12 | BUY | $356.71 | 0.344 | 0.411 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.411) | Defense Stock Roundup  Denuclearization  NOC OA Merger  Deals Wins For BA   LMT ... |
| 2018-06-13 | BUY | $350.20 | 0.345 | 0.698 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.698) | Northrop Grumman Secures  62M Navy Deal For BAMS D Program... |
| 2018-06-14 | BUY | $348.82 | 0.352 | 0.426 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.352, SAE: 0.426) | Boeing Wins  179M Deal To Aid Kuwait s Fleet Of F A 18 Jets... |
| 2018-06-15 | SELL | $344.45 | 0.345 | 0.899 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.345, SAE: 0.899) | U S  trade tariffs push industrials  materials stocks lower... |
| 2018-06-15 | BUY | $344.45 | 0.356 | 0.674 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.356, SAE: 0.674) | Boeing  BA  Wins  1 5B Navy Deal For F A And E A18 Aircraft ... |
| 2018-06-18 | BUY | $341.43 | 0.345 | 0.802 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.802) | General Dynamics Wins  225M Navy Deal For Nuclear Submarines... |
| 2018-06-19 | SELL | $328.32 | 0.339 | 0.651 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.339, SAE: 0.651) | US China Trade War Will Cripple U S  Economy... |
| 2018-06-19 | BUY | $328.32 | 0.336 | 0.686 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.336, SAE: 0.686) | China Takes A Hit  What s Next ... |
| 2018-06-19 | BUY | $328.32 | 0.342 | 0.449 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.449) | Milestone for 1st Detect s TRACER 1000... |
| 2018-06-19 | BUY | $328.32 | 0.351 | 0.762 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.351, SAE: 0.762) | Stock Market News For Jun 20  2018... |
| 2018-06-19 | SELL | $328.32 | 0.335 | 0.723 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.335, SAE: 0.723) | New Tariffs  Same Old Reaction... |
| 2018-06-19 | BUY | $328.32 | 0.337 | 0.559 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.559) | As Boeing Dips on Trade Concerns  Market Cycles Suggest More Downside... |
| 2018-06-20 | BUY | $329.83 | 0.345 | 0.680 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.680) | Boeing mid market jet in service by 2025 ... |
| 2018-06-20 | BUY | $329.83 | 0.338 | 0.573 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.573) | Boeing says Export Import Bank vital to U S  growth... |
| 2018-06-21 | BUY | $324.99 | 0.346 | 0.770 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.770) | Singapore Airlines to shift planes from SilkAir to budget arm Scoot... |
| 2018-06-21 | BUY | $324.99 | 0.347 | 0.468 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.468) | 3  Screaming Buy  Dividends Up To 10 1  With Huge Gains On Tap... |
| 2018-06-25 | BUY | $318.77 | 0.342 | 0.730 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.730) | South Korea picks Boeing P 8 for  1 7 billion maritime patrol aircraft contract... |
| 2018-06-25 | BUY | $318.77 | 0.345 | 0.705 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.705) | India s Jet Airways says to buy additional 75 Boeing 737 Max jets... |
| 2018-06-25 | BUY | $318.77 | 0.355 | 0.751 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.355, SAE: 0.751) | Sector ETF Week In Review For June 18 22... |
| 2018-06-26 | BUY | $318.87 | 0.365 | 0.593 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.365, SAE: 0.593) | Daily Market Report   26 06 2018... |
| 2018-07-02 | BUY | $323.47 | 0.355 | 0.845 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.355, SAE: 0.845) | Japan s ANA to cancel 113 domestic flights to inspect Rolls Royce engines... |
| 2018-07-06 | BUY | $322.08 | 0.338 | 0.826 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.826) | Exclusive  Northrop Grumman angles for role in Japanese stealth fighter program ... |
| 2018-07-06 | SELL | $322.08 | 0.336 | 0.608 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.336, SAE: 0.608) | Cost of one of those  expensive  U S  South Korea military exercises   14 millio... |
| 2018-07-09 | BUY | $329.09 | 0.344 | 0.693 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.693) | Saudi Arabian Airlines in Boeing 777X order talks  sources... |
| 2018-07-10 | BUY | $334.13 | 0.340 | 0.816 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.816) | What Trade War  Positive Tone Seems To Prevail As Earnings Season Looms... |
| 2018-07-10 | BUY | $334.13 | 0.340 | 0.790 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.790) | Airbus renames Canadian jet as A220  seen near U S  deal... |
| 2018-07-12 | BUY | $333.05 | 0.347 | 0.465 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.465) | Exclusive  U S  and Europe clash over global supersonic jet noise standards... |
| 2018-07-12 | BUY | $333.05 | 0.338 | 0.701 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.701) | Brazil s Embraer sees demand for 10 550 smaller jets in next 20 years... |
| 2018-07-13 | BUY | $337.63 | 0.342 | 0.859 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.859) | JetBlue got discount of up to 72  on its Airbus order  Moody s says... |
| 2018-07-16 | BUY | $342.74 | 0.341 | 0.790 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.790) | 5 Trades For Monday  Adobe Systems  Boeing  CME  Ulta Beauty And United Tech... |
| 2018-07-17 | BUY | $343.49 | 0.334 | 0.886 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.334, SAE: 0.886) | Boeing awarded  3 9 billion contract for two 747 8 presidential aircraft  Pentag... |
| 2018-07-18 | BUY | $346.71 | 0.345 | 0.888 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.888) |  Heartfelt joy  as first Ethiopia Eritrea flight in 20 years seals peace deal... |
| 2018-07-18 | BUY | $346.71 | 0.345 | 0.684 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.684) | Stocks   Dow Notches Five Day Winning Streak as Morgan Stanley Fuels Rally... |
| 2018-07-20 | BUY | $341.58 | 0.346 | 0.621 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.621) | Malaysia to release report on missing flight MH370 on July 30... |
| 2018-07-20 | BUY | $341.58 | 0.347 | 0.802 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.802) | Honeywell profit beats on higher demand for aircraft parts... |
| 2018-07-23 | BUY | $340.01 | 0.356 | 0.812 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.356, SAE: 0.812) | iFOREX Daily Analysis   July 23 2018... |
| 2018-07-23 | BUY | $340.01 | 0.339 | 0.816 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.816) | Opening Bell  U S  Futures Slip  USD Stabilizes  Global Bonds Hit ... |
| 2018-07-24 | BUY | $344.83 | 0.341 | 0.828 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.828) | Brazil Workers Party would use currency reserves for infrastructure... |
| 2018-07-24 | BUY | $344.83 | 0.366 | 0.625 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.366, SAE: 0.625) | Your Passport To Underappreciated 7  Yields... |
| 2018-07-30 | BUY | $337.89 | 0.352 | 0.565 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.352, SAE: 0.565) | Sector ETF Week In Review For The Week Of July 23 27... |
| 2018-07-30 | BUY | $337.89 | 0.348 | 0.755 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.348, SAE: 0.755) | U S  stocks lower at close of trade  Dow Jones Industrial Average down 0 57 ... |
| 2018-08-15 | BUY | $320.89 | 0.342 | 0.640 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.640) | Boeing Falls 3 14  as Earnings Season Continues... |
| 2018-08-16 | BUY | $334.64 | 0.351 | 0.738 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.351, SAE: 0.738) | Week Ahead  Bullish Stocks Likely To Slow  Oil To Rise  Euro To Fall... |
| 2018-08-21 | BUY | $342.18 | 0.350 | 0.477 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.350, SAE: 0.477) | Day Ahead  Top 3 Things to Watch... |
| 2018-08-22 | BUY | $338.56 | 0.338 | 0.767 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.767) | Stocks   S P 500 Snaps Win Streak as Political Turmoil Grips Wall Street... |
| 2018-08-24 | BUY | $337.93 | 0.344 | 0.750 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.750) | Greenback Marks Time Ahead Of Powell... |
| 2018-08-28 | BUY | $339.70 | 0.338 | 0.570 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.570) | Stocks   S P Notches Another Record as Worries Over Trade Rumble Recede... |
| 2018-08-28 | BUY | $339.70 | 0.342 | 0.883 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.883) | Equities Gain As U S  Mexico Trade Talks Bear Fruit... |
| 2018-08-29 | BUY | $338.71 | 0.338 | 0.703 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.703) | Russian passenger jet goes off runway in Sochi  18 injured... |
| 2018-08-29 | BUY | $338.71 | 0.337 | 0.678 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.678) | Election Anxiety Puts  33 Billion of Deals on Hold in Brazil... |
| 2018-09-04 | BUY | $334.90 | 0.349 | 0.689 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.689) | Lockheed to make wings for F 16 jet in India with partner Tata... |
| 2018-09-04 | BUY | $334.90 | 0.345 | 0.431 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.431) | U S  stocks lower at close of trade  Dow Jones Industrial Average down 0 05 ... |
| 2018-09-06 | BUY | $339.76 | 0.348 | 0.630 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.348, SAE: 0.630) | Week Ahead  U S  Stocks Slip But Recovery Expected On Macro  Technicals... |
| 2018-09-10 | BUY | $330.66 | 0.342 | 0.788 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.788) | Boeing calling back retirees to fix 737 production snags... |
| 2018-09-12 | BUY | $341.83 | 0.352 | 0.865 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.352, SAE: 0.865) | Boeing 737 recovery plan  taking hold  amid factory snarl  CEO... |
| 2018-09-18 | BUY | $351.67 | 0.337 | 0.668 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.668) | Trump Escalates Trade War  Expect A  Rare Earth  Response From China... |
| 2018-09-21 | BUY | $360.03 | 0.343 | 0.767 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.767) | Ahead Of The U S  Election Lockheed s Downside Outweighs Its Upside... |
| 2018-09-24 | BUY | $355.93 | 0.344 | 0.784 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.784) | 5 Trade Ideas For Monday  Boeing  BlackRock  Discover Financial  Gilead   NetApp... |
| 2018-09-27 | BUY | $355.35 | 0.346 | 0.480 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.480) | Rolls Royce still grappling with Trent 1000 engine issues... |
| 2018-10-01 | BUY | $369.76 | 0.342 | 0.590 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.590) | General Electric replaces CEO with outsider  shares soar... |
| 2018-10-01 | BUY | $369.76 | 0.339 | 0.738 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.738) | Qatar Airways upgrades part of A350 order to biggest model... |
| 2018-10-03 | BUY | $379.44 | 0.340 | 0.634 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.634) | Opening Bell  Stocks Rebound On Italian  Budget   Euro Recovers... |
| 2018-10-03 | BUY | $379.44 | 0.336 | 0.686 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.336, SAE: 0.686) | India risks U S  sanctions with  5 billion purchase of Russian missiles... |
| 2018-10-04 | BUY | $377.21 | 0.356 | 0.438 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.356, SAE: 0.438) | iFOREX Daily Analysis   October 04 2018... |
| 2018-10-09 | BUY | $372.81 | 0.357 | 0.501 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.357, SAE: 0.501) | General Electric  How Low Can It Go ... |
| 2018-10-09 | BUY | $372.81 | 0.356 | 0.749 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.356, SAE: 0.749) | U S  foreign military sales total  55 6 billion  up 33 percent  U S  official... |
| 2018-10-10 | BUY | $355.43 | 0.338 | 0.799 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.799) | U S  says China spy charged with trying to steal aviation secrets... |
| 2018-10-12 | BUY | $348.31 | 0.338 | 0.719 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.719) | Rocket failure astronauts will go back into space   Russian official... |
| 2018-10-15 | BUY | $347.12 | 0.337 | 0.788 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.788) | Corrections Are A Part Of Market Cycles  Have Cash At The Ready ... |
| 2018-10-15 | BUY | $347.12 | 0.341 | 0.733 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.733) | Oil  Debt and Iran  Weapons in Any U S  Saudi Fight Over Khashoggi... |
| 2018-10-18 | SELL | $347.57 | 0.343 | 0.681 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.343, SAE: 0.681) | U S  to exit nuclear treaty with Russia... |
| 2018-10-24 | BUY | $343.03 | 0.344 | 0.680 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.680) | Boeing Rises 3 ... |
| 2018-10-25 | BUY | $351.85 | 0.338 | 0.645 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.645) | Exclusive  2 DEER  China s HNA Group seeks buyer for  300 million  Dream Jet    ... |
| 2018-10-29 | BUY | $324.59 | 0.341 | 0.771 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.771) | EasyJet expects to be flying electric planes by 2030... |
| 2018-10-29 | BUY | $324.59 | 0.348 | 0.755 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.348, SAE: 0.755) | Doomed Indonesian plane with 189 on board had asked to return to base... |
| 2018-10-30 | BUY | $338.44 | 0.344 | 0.732 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.732) | Boeing Rises 3 ... |
| 2018-11-01 | BUY | $351.17 | 0.339 | 0.757 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.757) | Vietjet to finalize  6 5 billion Airbus order  sources... |
| 2018-11-05 | BUY | $350.12 | 0.341 | 0.591 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.591) | The Zacks Analyst Blog Highlights  Aerojet Rocketdyne Holdings  Lockheed Martin ... |
| 2018-11-05 | BUY | $350.12 | 0.351 | 0.696 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.351, SAE: 0.696) | In China  female pilots strain to hold up half the sky... |
| 2018-11-05 | BUY | $350.12 | 0.336 | 0.654 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.336, SAE: 0.654) | Hope Fades in Iran as U S  Sanctions Bite... |
| 2018-11-06 | BUY | $354.46 | 0.342 | 0.774 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.774) | Airbus stages maiden flight of upgraded A330 800... |
| 2018-11-07 | BUY | $359.83 | 0.339 | 0.539 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.539) | Boeing to report lower 737 deliveries amid supplier delays  executive... |
| 2018-11-08 | BUY | $360.28 | 0.346 | 0.727 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.727) | U S  arms exports up 13 percent over 2017 as Trump champions deals... |
| 2018-11-14 | SELL | $334.96 | 0.339 | 0.618 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.339, SAE: 0.618) | FAA  Boeing study need for 737 MAX software changes after crash... |
| 2018-11-14 | BUY | $334.96 | 0.353 | 0.871 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.353, SAE: 0.871) | Boeing Wins  71M Deal To Aid Minuteman III Missile Program... |
| 2018-11-14 | BUY | $334.96 | 0.340 | 0.754 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.754) | U S  safety board holds hearing on fatal Southwest engine explosion... |
| 2018-11-15 | BUY | $331.90 | 0.349 | 0.424 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.424) | American was  unaware  of some 737 MAX functions... |
| 2018-11-19 | BUY | $311.86 | 0.355 | 0.576 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.355, SAE: 0.576) | Boeing  BA  To Supply 12 737 MAX 8 To Caribbean Airlines ... |
| 2018-11-20 | BUY | $308.71 | 0.336 | 0.808 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.336, SAE: 0.808) | Canadian air force running short of pilots  jets  top watchdog... |
| 2018-11-20 | BUY | $308.71 | 0.344 | 0.631 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.631) | Initial Jobless Claims Up 3000 From Last Thursday s 221K... |
| 2018-11-23 | BUY | $303.48 | 0.340 | 0.679 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.679) | HNA Seeks Advice with Cinda for Asset Disposals... |
| 2018-11-23 | BUY | $303.48 | 0.349 | 0.690 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.690) | 4 Defense Stocks Up More Than 8  On A Year to Date Basis... |
| 2018-11-26 | BUY | $307.41 | 0.351 | 0.657 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.351, SAE: 0.657) | Aerospace ETFs Rise On China s Approval To UTX COL Deal... |
| 2018-11-26 | BUY | $307.41 | 0.345 | 0.606 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.606) | Qatar expects to receive six F 15 fighter jets from U S  by March 2021... |
| 2018-11-26 | BUY | $307.41 | 0.341 | 0.730 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.730) | Airbus delivers first A330neo in Boeing 787 dogfight... |
| 2018-11-27 | BUY | $309.03 | 0.334 | 0.746 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.334, SAE: 0.746) | The Zacks Analyst Blog Highlights  Boeing  PepsiCo  Citigroup  United Technologi... |
| 2018-11-27 | BUY | $309.03 | 0.359 | 0.750 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.359, SAE: 0.750) | Lockheed Martin Wins  79M Deal To Support Apache Helicopter... |
| 2018-11-28 | BUY | $324.06 | 0.348 | 0.758 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.348, SAE: 0.758) | Bell Boeing Wins  21M Deal To Support V 22 Family Of Jets... |
| 2018-11-28 | BUY | $324.06 | 0.339 | 0.730 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.730) | Saudi Arabia  U S  Ink  15B LOA For Lockheed s THAAD Missile... |
| 2018-11-28 | BUY | $324.06 | 0.348 | 0.753 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.348, SAE: 0.753) | Stock Market News For Nov 29  2018... |
| 2018-11-29 | BUY | $332.86 | 0.354 | 0.408 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.354, SAE: 0.408) | Boeing Wins Deal To Support F A 18E F And EA 18G Programs... |
| 2018-11-29 | BUY | $332.86 | 0.356 | 0.849 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.356, SAE: 0.849) | Spirit Aerosystems  SPR  Down 6 8  Since Last Earnings Report  Can It Rebound ... |
| 2018-11-29 | BUY | $332.86 | 0.350 | 0.853 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.350, SAE: 0.853) | Stock Market News For Dec 3  2018... |
| 2018-11-29 | BUY | $332.86 | 0.342 | 0.683 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.683) | Exclusive  Boeing eyes Lion Air crash software upgrade in 6 8 weeks... |
| 2018-11-29 | BUY | $332.86 | 0.341 | 0.515 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.515) | Europe Slips As Trump Tweets About 25  Auto Tariffs  Dow Starts  Santa Rally  As... |
| 2018-12-03 | BUY | $349.77 | 0.352 | 0.824 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.352, SAE: 0.824) | Boeing Rises 5 ... |
| 2018-12-04 | SELL | $332.81 | 0.340 | 0.439 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.340, SAE: 0.439) | Northrop  NOC  Increases Share Buyback Authorization By  3B... |
| 2018-12-06 | BUY | $322.51 | 0.344 | 0.710 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.710) | Lion Air says November passenger numbers fell less than 5 percent after deadly c... |
| 2018-12-06 | BUY | $322.51 | 0.336 | 0.684 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.336, SAE: 0.684) | India aviation watchdog advises training for 737 MAX pilots after Lion Air crash... |
| 2018-12-06 | BUY | $322.51 | 0.341 | 0.597 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.597) | Northrop Grumman Wins  450M Joint Threat Emitter Support Deal... |
| 2018-12-10 | BUY | $317.11 | 0.342 | 0.734 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.734) | Triumph Group Gains On Product Expansion   Rising Jet Demand... |
| 2018-12-10 | BUY | $317.11 | 0.354 | 0.679 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.354, SAE: 0.679) | Raytheon Wins  21M Navy Deal To Support SSDS MK 2 Program... |
| 2018-12-10 | BUY | $317.11 | 0.355 | 0.879 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.355, SAE: 0.879) | Right Back Where The Stock Market Started  But Worse... |
| 2018-12-11 | BUY | $312.92 | 0.343 | 0.793 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.793) | Boeing s 737 deliveries rise in November... |
| 2018-12-11 | BUY | $312.92 | 0.347 | 0.857 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.857) | Gol Linhas  Expedited Fleet Renewal Plan To Boost Efficiency... |
| 2018-12-11 | BUY | $312.92 | 0.341 | 0.748 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.748) | U S  stocks mixed at close of trade  Dow Jones Industrial Average down 0 22 ... |
| 2018-12-12 | BUY | $317.44 | 0.344 | 0.791 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.791) | Boeing Rises 3 ... |
| 2018-12-12 | BUY | $317.44 | 0.339 | 0.702 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.702) | Curtiss Wright  CW  Gains On Growth Prospects   Rising Demand... |
| 2018-12-12 | BUY | $317.44 | 0.339 | 0.864 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.864) | Iran airlines need 500 planes  official mulls Sukhoi  reports... |
| 2018-12-12 | BUY | $317.44 | 0.355 | 0.596 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.355, SAE: 0.596) | Rolls Royce expects 2018 profit  free cash flow in upper half of forecast range... |
| 2018-12-13 | BUY | $316.26 | 0.355 | 0.700 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.355, SAE: 0.700) | 5 Defense Stocks Likely To Stay Ahead Of Sector In 2019... |
| 2018-12-14 | SELL | $309.73 | 0.334 | 0.723 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.334, SAE: 0.723) | Europe Edged Down On Growth Concern And Lingering US China Trade Tensions... |
| 2018-12-17 | BUY | $307.18 | 0.350 | 0.787 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.350, SAE: 0.787) | Raytheon Wins  149M Navy Deal For Developing SM 2 Block IIIC... |
| 2018-12-17 | BUY | $307.18 | 0.341 | 0.795 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.795) | Boeing  BA  Stock Moves  0 82   What You Should Know... |
| 2018-12-17 | BUY | $307.18 | 0.352 | 0.823 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.352, SAE: 0.823) | AAR Q2 2018 Earnings Preview... |
| 2018-12-18 | BUY | $318.77 | 0.351 | 0.851 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.351, SAE: 0.851) |  Company News For Dec 19  2018... |
| 2018-12-18 | BUY | $318.77 | 0.345 | 0.840 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.840) | Time To Buy BA Stock On The Cheap After Boeing Dividend Hike ... |
| 2018-12-18 | BUY | $318.77 | 0.344 | 0.539 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.539) | Boeing Rises 3 ... |
| 2018-12-20 | BUY | $304.19 | 0.340 | 0.734 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.734) | Astronauts return safely to Earth from Space Station... |
| 2018-12-20 | BUY | $304.19 | 0.337 | 0.654 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.654) | Trump  annoyed by resignation letter  pushes out Mattis early... |
| 2018-12-20 | BUY | $304.19 | 0.344 | 0.511 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.511) | Airbus tests market for A321XLR jet launch by mid 2019  sources... |
| 2018-12-20 | BUY | $304.19 | 0.340 | 0.805 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.805) | Boeing Views Demand For 2 3K Jets In India For Next 20 Years... |
| 2018-12-24 | BUY | $285.83 | 0.355 | 0.645 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.355, SAE: 0.645) | Boeing Inks Deal To Deliver 50 737 Jets To Flyadeal For  5 9B... |
| 2018-12-26 | BUY | $305.04 | 0.344 | 0.741 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.741) | Frank Holmes Predicts Gold Explosion To The Upside... |
| 2018-12-26 | BUY | $305.04 | 0.356 | 0.489 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.356, SAE: 0.489) | Raytheon Wins  72M Deal For Aegis FCS Equipment Production... |
| 2018-12-26 | BUY | $305.04 | 0.340 | 0.727 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.727) | Boeing  BA  Inks  240M Deal To Aid Missile Defense System... |
| 2018-12-26 | BUY | $305.04 | 0.346 | 0.887 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.887) | Northrop Grumman s  NOC  Unit Inks  1 3B Deal For LITENING... |
| 2018-12-27 | BUY | $308.16 | 0.352 | 0.877 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.352, SAE: 0.877) | Esterline Technologies  ESL  Soars To 52 Week High  Time To Cash Out ... |
| 2018-12-27 | BUY | $308.16 | 0.348 | 0.735 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.348, SAE: 0.735) | Boeing  BA  Outpaces Stock Market Gains  What You Should Know... |
| 2018-12-27 | BUY | $308.16 | 0.347 | 0.697 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.697) | Boeing Wins  49M FMS Deal To Support Qatar s AH 64E Aircraft... |
| 2018-12-31 | BUY | $313.37 | 0.348 | 0.636 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.348, SAE: 0.636) |  Company News For Jan 2  2019... |
| 2018-12-31 | BUY | $313.37 | 0.340 | 0.652 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.652) | Fatalities on commercial passenger aircraft rise in 2018... |
| 2019-01-04 | BUY | $317.82 | 0.346 | 0.859 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.859) | Pioneering Southwest Airlines co founder Herb Kelleher dies at 87... |
| 2019-01-07 | BUY | $318.82 | 0.348 | 0.797 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.348, SAE: 0.797) | Boeing  BA  Gains But Lags Market  What You Should Know... |
| 2019-01-08 | BUY | $330.89 | 0.344 | 0.849 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.849) | Boeing Rises 3 ... |
| 2019-01-08 | BUY | $330.89 | 0.344 | 0.601 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.601) | U S  stocks higher at close of trade  Dow Jones Industrial Average up 1 09 ... |
| 2019-01-08 | BUY | $330.89 | 0.342 | 0.688 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.688) | Search resumes for Flight 610 cockpit voice recorder... |
| 2019-01-08 | BUY | $330.89 | 0.354 | 0.621 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.354, SAE: 0.621) | Boeing delivers record 806 aircraft in 2018  shares jump 4 percent... |
| 2019-01-09 | BUY | $334.10 | 0.351 | 0.808 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.351, SAE: 0.808) | Opening Bell  Stocks Shine Ahead Of Fed Minutes  Oil Retakes  50 ... |
| 2019-01-09 | BUY | $334.10 | 0.352 | 0.678 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.352, SAE: 0.678) | 4 Reasons To Add Engility Holdings  EGL  To Your Portfolio... |
| 2019-01-11 | BUY | $342.91 | 0.340 | 0.659 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.659) | Embraer  2 2  as board ratifies Boeing deal after Brazil s OK... |
| 2019-01-11 | BUY | $342.91 | 0.340 | 0.695 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.695) | Exclusive  Lessors to India s troubled Jet Airways consider taking back planes  ... |
| 2019-01-16 | BUY | $342.10 | 0.340 | 0.736 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.736) | Boeing  BA  Is An Incredible Growth Stock  3 Reasons Why... |
| 2019-01-17 | BUY | $348.93 | 0.354 | 0.574 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.354, SAE: 0.574) | Stock Market News For Jan 21  2019... |
| 2019-01-17 | BUY | $348.93 | 0.346 | 0.732 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.732) | U S  stocks higher at close of trade  Dow Jones Industrial Average up 0 67 ... |
| 2019-01-22 | BUY | $347.77 | 0.343 | 0.533 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.533) | Earnings Preview  Boeing  BA  Q4 Earnings Expected To Decline... |
| 2019-01-24 | BUY | $348.13 | 0.353 | 0.759 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.353, SAE: 0.759) | Textron  TXT  Catches Eye  Stock Jumps 5 6 ... |
| 2019-01-28 | BUY | $352.70 | 0.345 | 0.672 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.672) | Industrials SPDR holds up despite rough sessions for CAT and GE... |
| 2019-01-28 | SELL | $352.70 | 0.343 | 0.609 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.343, SAE: 0.609) | Explainer  Key Issues  implications of U S  China trade talks... |
| 2019-01-28 | BUY | $352.70 | 0.340 | 0.490 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.490) | BA  GD Q4 Earnings Due On Jan 30  Here Are The Key Predictions... |
| 2019-01-28 | BUY | $352.70 | 0.345 | 0.844 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.844) | Stocks   Caterpillar  Nvidia Warnings Weigh on Opening... |
| 2019-01-28 | BUY | $352.70 | 0.341 | 0.870 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.870) | Will Cost Growth For KC 46 Hurt Boeing s  BA  Q4 Earnings ... |
| 2019-01-28 | BUY | $352.70 | 0.342 | 0.834 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.834) | Asian Markets Decline as U S  Files Charges Against Huawei Ahead of Trade Talks... |
| 2019-01-28 | BUY | $352.70 | 0.341 | 0.741 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.741) | The Zacks Analyst Blog Highlights  Johnson   Johnson  Boeing  Starbucks  Norfolk... |
| 2019-01-29 | BUY | $354.58 | 0.350 | 0.508 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.350, SAE: 0.508) | ADP Jobs Stay Robust  Q4 Numbers Look Good Too... |
| 2019-01-29 | BUY | $354.58 | 0.345 | 0.828 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.828) | Exclusive  Boeing nears  3 5 billion 737 MAX jet deal with Japan s ANA   sources... |
| 2019-01-30 | BUY | $376.75 | 0.355 | 0.868 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.355, SAE: 0.868) | Boeing beats by  0 91  beats on revenue... |
| 2019-01-30 | BUY | $376.75 | 0.342 | 0.721 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.721) | Raytheon  RTN  Q4 Earnings Surpass Estimates  Sales Up Y Y... |
| 2019-01-30 | BUY | $376.75 | 0.347 | 0.791 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.791) | Boeing  BA  Looks Good  Stock Adds 6 3  In Session... |
| 2019-01-30 | BUY | $376.75 | 0.350 | 0.768 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.350, SAE: 0.768) | Boeing Rises 7 ... |
| 2019-01-30 | BUY | $376.75 | 0.350 | 0.882 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.350, SAE: 0.882) | Boeing Shares Soar on Q4 Results  Upbeat Guidance... |
| 2019-01-31 | BUY | $374.71 | 0.347 | 0.665 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.665) | Airbus A380 under threat as Emirates weighs rejigged order  sources... |
| 2019-01-31 | BUY | $374.71 | 0.349 | 0.827 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.827) | Germany drops F 35 from fighter tender  Boeing F A 18 and Eurofighter to battle ... |
| 2019-02-01 | BUY | $376.46 | 0.344 | 0.604 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.604) | IAG s Walsh to Airbus  drop A380 price to boost sales... |
| 2019-02-04 | BUY | $385.76 | 0.339 | 0.605 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.605) | Boeing  BA  Hits 52 Week High  Can The Run Continue ... |
| 2019-02-04 | BUY | $385.76 | 0.352 | 0.734 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.352, SAE: 0.734) | General Dynamics To Provide Support Services To DDG 51 Ships... |
| 2019-02-05 | SELL | $398.57 | 0.335 | 0.777 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.335, SAE: 0.777) | 2 Mutual Funds To Benefit From America s Defense Resolve... |
| 2019-02-05 | BUY | $398.57 | 0.342 | 0.837 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.837) | Boeing makes  significant investment  in supersonic jet developer Aerion... |
| 2019-02-06 | BUY | $399.47 | 0.337 | 0.543 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.543) | 5 Winners From Trump s State Of The Union Address... |
| 2019-02-06 | BUY | $399.47 | 0.341 | 0.548 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.548) | Etihad approaches banks for over  500 million for Boeing deliveries  sources... |
| 2019-02-06 | BUY | $399.47 | 0.350 | 0.622 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.350, SAE: 0.622) | General Dynamics   GD  NASSCO Wins  67M Deal To Repair DDG 51... |
| 2019-02-06 | BUY | $399.47 | 0.337 | 0.642 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.642) | Boeing says aircraft demand supports even faster 737 production... |
| 2019-02-07 | BUY | $395.68 | 0.338 | 0.788 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.788) | 2 Funds To Gain From Trump s State Of The Union Address... |
| 2019-02-07 | BUY | $395.68 | 0.354 | 0.757 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.354, SAE: 0.757) | Stock Market News For Feb 8  2019... |
| 2019-02-08 | BUY | $395.43 | 0.349 | 0.692 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.692) | Fifty years on  Boeing s 747 clings to life as cargo carrier... |
| 2019-02-11 | BUY | $394.49 | 0.352 | 0.662 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.352, SAE: 0.662) | Stock Market News For Feb 12  2019... |
| 2019-02-11 | BUY | $394.49 | 0.352 | 0.814 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.352, SAE: 0.814) | Lockheed Martin Wins Deal To Support Aegis Weapon System... |
| 2019-02-12 | BUY | $401.10 | 0.361 | 0.799 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.361, SAE: 0.799) | Raytheon Wins  88M Deal To Support F A 18   EA 18G Aircraft... |
| 2019-02-13 | BUY | $400.96 | 0.344 | 0.822 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.822) | Huntington Ingalls  HII  Q4 Earnings Top  Revenues Up Y Y... |
| 2019-02-13 | BUY | $400.96 | 0.341 | 0.414 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.414) | Copa Holdings  CPA  Misses Q4 Earnings Estimates  Down Y Y... |
| 2019-02-14 | SELL | $400.22 | 0.335 | 0.703 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.335, SAE: 0.703) | Explainer  What s at stake in U S  China trade talks... |
| 2019-02-15 | BUY | $408.18 | 0.340 | 0.546 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.546) | Why Aerospace   Defense ETFs Are Soaring In 2019... |
| 2019-02-19 | BUY | $406.51 | 0.338 | 0.853 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.853) | Moving Average Crossover Alert  Boeing... |
| 2019-02-20 | SELL | $411.68 | 0.339 | 0.892 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.339, SAE: 0.892) | Norwegian Air aims to fly stranded plane out of Iran in next few days... |
| 2019-02-21 | SELL | $407.80 | 0.340 | 0.635 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.340, SAE: 0.635) | U S  carriers compete for new slots at Tokyo s Haneda airport... |
| 2019-02-21 | BUY | $407.80 | 0.338 | 0.550 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.550) | Two bodies recovered after Amazon cargo plane crashes into Texas bay... |
| 2019-02-22 | BUY | $414.12 | 0.351 | 0.410 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.351, SAE: 0.410) | The Stock Market Drops  And There May Still Be More To Come... |
| 2019-02-25 | BUY | $416.87 | 0.344 | 0.629 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.629) | Tariff Deadline Vanishes  Providing Strong Tailwind To Start Week... |
| 2019-02-25 | BUY | $416.87 | 0.340 | 0.541 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.541) | Would be Bangladeshi plane hijacker had toy gun  police... |
| 2019-02-27 | SELL | $425.24 | 0.342 | 0.756 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.342, SAE: 0.756) | Trump hails North Korea s  awesome  potential ahead of talks with Kim... |
| 2019-03-04 | BUY | $422.56 | 0.341 | 0.429 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.429) | S P 500 Closes Above 2 800 After Nearly 4 Months  5 Picks ... |
| 2019-03-05 | BUY | $420.05 | 0.337 | 0.413 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.413) | Southwest says recent mechanics  disruption costing millions weekly... |
| 2019-03-05 | BUY | $420.05 | 0.340 | 0.656 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.656) | Here s Why Boeing  BA  Stock Looks Like A Strong Buy Right Now... |
| 2019-03-05 | BUY | $420.05 | 0.352 | 0.868 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.352, SAE: 0.868) | Lockheed Martin Wins  946M Support Deal For Saudi s THAAD... |
| 2019-03-07 | BUY | $412.66 | 0.340 | 0.512 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.512) | Boeing s 737 MAX back in spotlight after second fatal crash... |
| 2019-03-07 | BUY | $412.66 | 0.345 | 0.763 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.763) | Retail Sales Improve In January... |
| 2019-03-07 | BUY | $412.66 | 0.337 | 0.866 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.866) | American Outdoor Brands   AOBC  Q3 Earnings Beat  Sales Miss... |
| 2019-03-07 | BUY | $412.66 | 0.344 | 0.719 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.719) | Semtech  SMTC  To Report Q4 Earnings  What s In The Offing ... |
| 2019-03-07 | BUY | $412.66 | 0.348 | 0.774 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.348, SAE: 0.774) | American Airlines grounds 14 planes due to overhead bin issue... |
| 2019-03-11 | BUY | $390.64 | 0.346 | 0.674 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.674) | FED Chair Jerome Powell In No Rush To Hike Rates... |
| 2019-03-11 | BUY | $390.64 | 0.357 | 0.785 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.357, SAE: 0.785) | Top 5 Things to Know in The Market on Monday... |
| 2019-03-11 | BUY | $390.64 | 0.340 | 0.706 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.706) | Shares bounce after worst week of year  Brexit stresses sterling... |
| 2019-03-11 | SELL | $390.64 | 0.338 | 0.644 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.338, SAE: 0.644) | In Line CPI Data For February... |
| 2019-03-11 | BUY | $390.64 | 0.339 | 0.828 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.828) | Ethiopian plane smoked and shuddered before deadly plunge... |
| 2019-03-12 | BUY | $366.62 | 0.341 | 0.614 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.614) | Strong Day For Markets With The QQQ Performing Best... |
| 2019-03-12 | BUY | $366.62 | 0.342 | 0.714 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.714) | Perficient  Colgate Palmolive  Boeing  Columbia Sportswear And Foot Locker Highl... |
| 2019-03-12 | BUY | $366.62 | 0.350 | 0.516 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.350, SAE: 0.516) | Bell Boeing JV Secures Navy Deal To Upgrade MV 22 Aircraft... |
| 2019-03-12 | BUY | $366.62 | 0.350 | 0.665 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.350, SAE: 0.665) | Stock Market News For Mar 13  2019... |
| 2019-03-12 | SELL | $366.62 | 0.341 | 0.636 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.341, SAE: 0.636) | Wall Street Pulled Higher By Tech Sector... |
| 2019-03-12 | BUY | $366.62 | 0.344 | 0.529 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.529) | Stock Market Pop Could Fizzle Fast... |
| 2019-03-12 | BUY | $366.62 | 0.343 | 0.439 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.439) | U S  airlines stand by 737 MAX as some customers  nations reject it... |
| 2019-03-12 | BUY | $366.62 | 0.357 | 0.517 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.357, SAE: 0.517) | Top 5 Things to Know in The Market on Tuesday... |
| 2019-03-12 | BUY | $366.62 | 0.344 | 0.817 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.817) | U S  to mandate design changes on Boeing 737 MAX 8 after crashes... |
| 2019-03-12 | BUY | $366.62 | 0.343 | 0.531 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.531) | Boeing delivers 95 jets in first two months of 2019... |
| 2019-03-12 | BUY | $366.62 | 0.346 | 0.732 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.732) | Durable Goods For Jan   0 4   Feb  PPI In Line At  0 2 ... |
| 2019-03-12 | BUY | $366.62 | 0.347 | 0.778 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.778) | Boeing shares dip again as more countries ground 737 MAX 8 planes... |
| 2019-03-13 | BUY | $368.31 | 0.336 | 0.561 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.336, SAE: 0.561) | Wall Street rises  Boeing up despite U S  grounding of 737 MAX jets... |
| 2019-03-13 | BUY | $368.31 | 0.344 | 0.606 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.606) | U S  Business Equipment Orders Advance by Most in Six Months... |
| 2019-03-13 | BUY | $368.31 | 0.356 | 0.744 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.356, SAE: 0.744) | Lockheed Martin Wins  84M Deal To Design Missile Technologies... |
| 2019-03-13 | BUY | $368.31 | 0.341 | 0.529 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.529) | S P Retakes 2800 On Third Straight Day Of Gains... |
| 2019-03-13 | BUY | $368.31 | 0.345 | 0.803 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.803) | Why Boeing Shares Will Rebound Once Again... |
| 2019-03-14 | SELL | $364.56 | 0.335 | 0.635 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.335, SAE: 0.635) | Brexit Votes Go As Planned  Key China Data In Focus... |
| 2019-03-14 | BUY | $364.56 | 0.363 | 0.703 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.363, SAE: 0.703) | Markets Turn To The Eastern Front... |
| 2019-03-14 | BUY | $364.56 | 0.348 | 0.705 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.348, SAE: 0.705) | Lockheed Martin Clinches  507M PAC 3 Missile Production Deal... |
| 2019-03-14 | BUY | $364.56 | 0.344 | 0.504 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.504) | General Electric CEO sets low 2019 profit targets  vows better from 2020... |
| 2019-03-14 | BUY | $364.56 | 0.351 | 0.673 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.351, SAE: 0.673) | Company News For Mar 15  2019... |
| 2019-03-14 | BUY | $364.56 | 0.344 | 0.710 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.710) | U S  lawmakers say Boeing 737 MAX 8 grounded for at least  weeks ... |
| 2019-03-14 | BUY | $364.56 | 0.342 | 0.711 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.711) | Boeing says finalizing software upgrade  revising pilot training for 737 Max... |
| 2019-03-14 | BUY | $364.56 | 0.347 | 0.538 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.538) | Northrop Grumman Wins  45M Navy Deal To Support LCS Program... |
| 2019-03-14 | BUY | $364.56 | 0.351 | 0.778 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.351, SAE: 0.778) | Stock Market News  WTI Crude Rose 4 14  Last Week... |
| 2019-03-14 | BUY | $364.56 | 0.344 | 0.797 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.797) | Boeing s safety analysis of 737 MAX flight control had crucial flaws  Seattle Ti... |
| 2019-03-15 | BUY | $370.11 | 0.341 | 0.833 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.833) | Take Five  Shall we try again  World markets themes for the week ahead... |
| 2019-03-15 | BUY | $370.11 | 0.335 | 0.640 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.335, SAE: 0.640) | NASDAQ  S P Recover Last Week s Losses    And Then Some ... |
| 2019-03-18 | BUY | $363.56 | 0.340 | 0.653 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.653) | Anomalous Price Action On NQ Emini... |
| 2019-03-18 | BUY | $363.56 | 0.344 | 0.435 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.435) | Top Stock Picks For The Week Of Mar 18  2019... |
| 2019-03-18 | BUY | $363.56 | 0.357 | 0.624 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.357, SAE: 0.624) | Top 5 Things to Know in The Market on Monday... |
| 2019-03-18 | BUY | $363.56 | 0.337 | 0.788 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.788) | Is There A New Bullish Cycle Coming For The Rest Of March ... |
| 2019-03-18 | BUY | $363.56 | 0.343 | 0.423 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.423) | Boeing shares fall again after probe report into FAA approval of 737 MAX... |
| 2019-03-18 | BUY | $363.56 | 0.335 | 0.712 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.335, SAE: 0.712) | Wall Street Rises But US30 Index Lags... |
| 2019-03-18 | BUY | $363.56 | 0.357 | 0.837 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.357, SAE: 0.837) | 8 Stocks To Watch As Global Markets Race Higher On March 18... |
| 2019-03-19 | BUY | $364.68 | 0.340 | 0.657 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.657) | Boeing reshuffles top engineers amid 737 MAX crisis... |
| 2019-03-19 | BUY | $364.68 | 0.343 | 0.650 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.650) | Lenders and travelers stick with Ethiopian Airlines  for now... |
| 2019-03-19 | BUY | $364.68 | 0.338 | 0.845 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.845) | China Considers Excluding Boeing 737 Max From Trade Deal... |
| 2019-03-19 | BUY | $364.68 | 0.345 | 0.814 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.814) | Stakes rise for Boeing as EU  Canada step up scrutiny of 737 MAX after crashes... |
| 2019-03-19 | BUY | $364.68 | 0.358 | 0.867 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.358, SAE: 0.867) | Raytheon To Provide Support Services To ESSM Block 2 Program... |
| 2019-03-19 | SELL | $364.68 | 0.339 | 0.858 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.339, SAE: 0.858) | Democrats seek probe of key FAA decisions on 737 MAX approval... |
| 2019-03-19 | BUY | $364.68 | 0.349 | 0.740 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.740) | Boeing  BA  Gains As Market Dips  What You Should Know... |
| 2019-03-19 | BUY | $364.68 | 0.344 | 0.409 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.409) | US30 And China50 Continued To Rally Yesterday... |
| 2019-03-20 | BUY | $367.35 | 0.339 | 0.892 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.892) | Boeing faces growing pressure in Washington over 737 MAX crashes... |
| 2019-03-20 | BUY | $367.35 | 0.342 | 0.839 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.839) | Boeing delays by months test flights for U S  human space program  sources... |
| 2019-03-21 | BUY | $363.97 | 0.343 | 0.487 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.487) | Defense Stock Roundup  GD  BA  LMT Win Big Deals  AIR Beats On Q3 Earnings... |
| 2019-03-21 | BUY | $363.97 | 0.344 | 0.814 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.814) | Boeing Wins  4B Deal For Delivery Of F A 18 Jets To U S  Navy... |
| 2019-03-21 | BUY | $363.97 | 0.348 | 0.767 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.348, SAE: 0.767) | Company News For Mar 25  2019... |
| 2019-03-22 | BUY | $353.69 | 0.346 | 0.640 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.640) | General Dynamics   GD  NASSCO Wins  465M Deal For CVN Carrier... |
| 2019-03-22 | BUY | $353.69 | 0.336 | 0.844 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.336, SAE: 0.844) | Curtis Wright Gains On High Growth Prospects Amid Debt Woes... |
| 2019-03-25 | BUY | $361.78 | 0.335 | 0.651 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.335, SAE: 0.651) | Boeing sets briefing on 737 MAX as Ethiopian carrier expresses confidence in pla... |
| 2019-03-25 | BUY | $361.78 | 0.338 | 0.707 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.707) | BA Stock Remains A Strong Buy As Ethiopian Airlines Backs Boeing... |
| 2019-03-26 | BUY | $361.71 | 0.347 | 0.679 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.679) | Southwest s 737 Max Ferry Flight Makes Emergency Landing... |
| 2019-03-26 | BUY | $361.71 | 0.344 | 0.615 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.615) | Nikkei Led Asian Markets Higher With 2 1  Gain... |
| 2019-03-26 | BUY | $361.71 | 0.341 | 0.697 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.697) | Airbus Secures  35B Deal From China  Boeing s Loss Rises... |
| 2019-03-26 | BUY | $361.71 | 0.344 | 0.665 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.665) | Southwest 737 MAX 8 passenger less flight lands safely after declaring emergency... |
| 2019-03-26 | SELL | $361.71 | 0.344 | 0.829 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.344, SAE: 0.829) | Southwest 737 MAX makes emergency landing  says computer system not to blame... |
| 2019-03-26 | BUY | $361.71 | 0.339 | 0.484 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.484) | Lufthansa plans to buy either Boeing 737 MAX or Airbus A320neo... |
| 2019-03-27 | BUY | $365.45 | 0.356 | 0.685 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.356, SAE: 0.685) | Top 5 Things to Know in The Market on Wednesday... |
| 2019-03-27 | SELL | $365.45 | 0.339 | 0.753 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.339, SAE: 0.753) | Southwest Cuts Q1 View On 737 MAX Groundings   Other Concerns... |
| 2019-03-27 | SELL | $365.45 | 0.341 | 0.743 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.341, SAE: 0.743) | Southwest trims first quarter outlook after 737 MAX groundings... |
| 2019-03-27 | BUY | $365.45 | 0.358 | 0.628 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.358, SAE: 0.628) | Raytheon To Provide Support Services To Radar Program Systems... |
| 2019-03-27 | BUY | $365.45 | 0.337 | 0.711 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.711) | Boeing unveils 737 MAX software fix after fatal crashes... |
| 2019-03-28 | SELL | $365.67 | 0.337 | 0.781 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.337, SAE: 0.781) | No call for simulators in new Boeing 737 MAX training proposals... |
| 2019-03-28 | SELL | $365.67 | 0.344 | 0.478 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.344, SAE: 0.478) | Southwest  LUV  Calls Off 737 MAX Operations Through May End... |
| 2019-03-28 | BUY | $365.67 | 0.352 | 0.817 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.352, SAE: 0.817) | Stock Market News For Apr 1  2019... |
| 2019-03-29 | BUY | $372.49 | 0.346 | 0.619 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.619) | Swing Trading Strategy  03 29 19... |
| 2019-03-29 | BUY | $372.49 | 0.340 | 0.796 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.796) | Anti stall system active before Ethiopian 737 MAX crash  sources... |
| 2019-04-01 | BUY | $382.37 | 0.341 | 0.696 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.696) | Inverted US Yield Curve  Should FX Traders Be Worried  ... |
| 2019-04-02 | BUY | $381.60 | 0.347 | 0.777 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.777) | Boeing to submit 737 MAX software upgrade  in the coming weeks ... |
| 2019-04-02 | BUY | $381.60 | 0.339 | 0.638 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.638) | Brazil s Gol will not cancel Boeing 737 MAX orders  newspaper... |
| 2019-04-02 | BUY | $381.60 | 0.353 | 0.767 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.353, SAE: 0.767) | Singapore Airlines grounds two 787 10s citing Rolls Royce engine problem... |
| 2019-04-02 | SELL | $381.60 | 0.337 | 0.447 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.337, SAE: 0.447) | Daimler CEO  Boeing safety debate highlights challenge for autonomous tech... |
| 2019-04-02 | BUY | $381.60 | 0.339 | 0.707 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.707) | Air Canada removes Boeing s 737 MAX from schedule until July... |
| 2019-04-02 | BUY | $381.60 | 0.352 | 0.823 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.352, SAE: 0.823) | Durable Goods Orders Fall in February as Aircraft Demand Weakens... |
| 2019-04-03 | BUY | $375.73 | 0.343 | 0.569 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.569) | What To Expect From Boeing s  BA  Q1 Earnings Amid 737 MAX Headwinds ... |
| 2019-04-03 | BUY | $375.73 | 0.347 | 0.812 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.812) | 4 Lucrative Defense Stocks To Add To Your Portfolio Now... |
| 2019-04-04 | BUY | $386.59 | 0.348 | 0.661 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.348, SAE: 0.661) | Pilots of doomed Ethiopian Airlines flight struggled for control as plane s nose... |
| 2019-04-04 | BUY | $386.59 | 0.362 | 0.777 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.362, SAE: 0.777) | 4 Reasons To Bet On Dow Jones ETFs In Q2... |
| 2019-04-04 | BUY | $386.59 | 0.339 | 0.752 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.752) | Explainer  Ethiopia crash raises questions over handling of faults on Boeing 737... |
| 2019-04-04 | BUY | $386.59 | 0.335 | 0.799 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.335, SAE: 0.799) | UAE s aviation body to join FAA panel on Boeing 737 MAX... |
| 2019-04-04 | BUY | $386.59 | 0.343 | 0.797 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.797) | Ethiopian crash report highlights sensors  software  leaves questions... |
| 2019-04-04 | BUY | $386.59 | 0.344 | 0.893 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.893) | How flawed software and excess speed doomed an Ethiopian Airlines 737 MAX... |
| 2019-04-04 | BUY | $386.59 | 0.348 | 0.609 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.348, SAE: 0.609) | Company News For Apr 8  2019... |
| 2019-04-04 | BUY | $386.59 | 0.338 | 0.783 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.783) | Will Boeing s  BA  737 Max Production Cut Benefit Airbus ... |
| 2019-04-04 | SELL | $386.59 | 0.337 | 0.894 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.337, SAE: 0.894) | American Airlines extends 737 MAX cancellations through June 5... |
| 2019-04-05 | BUY | $382.75 | 0.344 | 0.803 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.803) | Stocks Are About To Break Out Toward All Time Highs... |
| 2019-04-05 | BUY | $382.75 | 0.346 | 0.648 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.648) | U S  stocks higher at close of trade  Dow Jones Industrial Average up 0 15 ... |
| 2019-04-05 | BUY | $382.75 | 0.335 | 0.718 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.335, SAE: 0.718) | Here s What To Expect From Delta Air Lines  DAL  Q1 2019 Earnings ... |
| 2019-04-08 | BUY | $365.75 | 0.345 | 0.696 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.696) | Comeback Gives S P An 8 Session Winning Streak... |
| 2019-04-08 | BUY | $365.75 | 0.350 | 0.664 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.350, SAE: 0.664) | Commercial Border Delays Hit Auto Industry... |
| 2019-04-08 | SELL | $365.75 | 0.337 | 0.632 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.337, SAE: 0.632) | U S  Dollar Firms But Beware Of Chasing It Higher... |
| 2019-04-08 | BUY | $365.75 | 0.340 | 0.765 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.765) | Wall Street analysts cut 737 MAX delivery forecast... |
| 2019-04-09 | BUY | $360.40 | 0.342 | 0.867 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.867) | Share rally cools as Trump turns trade heat on Europe... |
| 2019-04-09 | BUY | $360.40 | 0.339 | 0.791 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.791) | Airbus says U S  sanctions on its aircraft would have no legal basis... |
| 2019-04-09 | BUY | $360.40 | 0.335 | 0.768 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.335, SAE: 0.768) | United Airlines says using larger jets on 737 MAX routes is  costing money ... |
| 2019-04-09 | BUY | $360.40 | 0.348 | 0.749 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.348, SAE: 0.749) | China Aircraft Leasing says no change to Boeing 737 MAX order... |
| 2019-04-09 | BUY | $360.40 | 0.351 | 0.547 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.351, SAE: 0.547) | Stock Market News For Apr 10  2019... |
| 2019-04-09 | BUY | $360.40 | 0.346 | 0.635 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.635) | Swing Trading Strategy Report  413 ... |
| 2019-04-09 | BUY | $360.40 | 0.344 | 0.705 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.705) | Boeing shareholders sue over 737 MAX crashes  disclosures... |
| 2019-04-10 | BUY | $356.39 | 0.336 | 0.740 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.336, SAE: 0.740) | Airbus Shares Looking Up Vs  Boeing  But Trade Headwinds Could Threaten Both... |
| 2019-04-10 | BUY | $356.39 | 0.346 | 0.742 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.742) | Industrials  gains put to test as earnings ramp up... |
| 2019-04-10 | BUY | $356.39 | 0.341 | 0.872 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.872) | Exclusive  Germany sees 8 86 billion euro cost to operate Tornado jets to 2030... |
| 2019-04-11 | SELL | $361.49 | 0.338 | 0.407 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.338, SAE: 0.407) | EU clears way for start of formal trade talks with U S... |
| 2019-04-11 | BUY | $361.49 | 0.338 | 0.763 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.763) | Opening Bell  Stocks Climb On Dovish Fed  ECB  Oil Rally Slips... |
| 2019-04-11 | BUY | $361.49 | 0.337 | 0.522 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.522) | FAA to meet with U S  airlines  pilot unions on Boeing 737 MAX... |
| 2019-04-11 | SELL | $361.49 | 0.339 | 0.482 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.339, SAE: 0.482) | American Airlines extends Boeing 737 MAX cancellations through August 19... |
| 2019-04-11 | BUY | $361.49 | 0.343 | 0.802 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.802) | U S  stocks mixed at close of trade  Dow Jones Industrial Average down 0 05 ... |
| 2019-04-12 | BUY | $370.75 | 0.343 | 0.697 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.697) | Europe Back In The Crosshairs Of U S  Trade Policy... |
| 2019-04-12 | BUY | $370.75 | 0.336 | 0.683 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.336, SAE: 0.683) | Exclusive  EU tariffs to target 20 billion euros of U S  imports   diplomats... |
| 2019-04-15 | BUY | $366.67 | 0.338 | 0.806 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.806) | U S  airlines face too many travelers  too few planes in 737 MAX summer dilemma... |
| 2019-04-15 | SELL | $366.67 | 0.335 | 0.794 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.335, SAE: 0.794) | EU to Begin Trade Talks With U S  as Tariff Threats Escalate... |
| 2019-04-16 | BUY | $372.78 | 0.344 | 0.682 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.682) | Earnings Preview  Boeing  BA  Q1 Earnings Expected To Decline... |
| 2019-04-16 | BUY | $372.78 | 0.341 | 0.656 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.656) | United Airlines beats first quarter profit estimate  holds 2019 target... |
| 2019-04-16 | BUY | $372.78 | 0.343 | 0.674 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.674) | Boeing 737 MAX joint governmental review will begin April 29  FAA... |
| 2019-04-16 | SELL | $372.78 | 0.335 | 0.774 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.335, SAE: 0.774) | China forms task force to review design changes to Boeing 737 MAX... |
| 2019-04-17 | SELL | $368.68 | 0.337 | 0.714 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.337, SAE: 0.714) | United sees Boeing s 737 MAX flying this summer  deliveries before year end... |
| 2019-04-18 | BUY | $371.17 | 0.340 | 0.401 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.401) | Boeing Reports Next Week  What To Expect... |
| 2019-04-23 | BUY | $365.26 | 0.350 | 0.878 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.350, SAE: 0.878) | United Technologies profit beats on Rockwell Collins boost... |
| 2019-04-23 | BUY | $365.26 | 0.348 | 0.887 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.348, SAE: 0.887) | Boeing  BA  Q1 Earnings Top  Down Y Y On Lower 737 Deliveries... |
| 2019-04-24 | BUY | $366.67 | 0.352 | 0.706 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.352, SAE: 0.706) | The Corn And Ethanol Report 04 24 19... |
| 2019-04-24 | BUY | $366.67 | 0.336 | 0.872 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.336, SAE: 0.872) | European shares hit highest since August on Credit Suisse  SAP... |
| 2019-04-24 | BUY | $366.67 | 0.347 | 0.669 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.669) | Stocks   S P  Nasdaq Slip as Falling Energy Stocks Weigh... |
| 2019-04-24 | BUY | $366.67 | 0.339 | 0.622 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.622) | Norwegian Air reschedules aircraft delivery  to cut 2019 20 capex by  2 1 billio... |
| 2019-04-25 | BUY | $373.83 | 0.347 | 0.804 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.804) | Southwest Airlines says bookings strong  may not always be 737 only carrier... |
| 2019-04-25 | BUY | $373.83 | 0.342 | 0.716 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.716) | flydubai s financial outlook for 2018 unchanged despite Boeing groundings  spoke... |
| 2019-04-25 | BUY | $373.83 | 0.343 | 0.486 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.486) | Pilots demand better training if Boeing wants to rebuild trust in 737 MAX... |
| 2019-04-25 | BUY | $373.83 | 0.351 | 0.733 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.351, SAE: 0.733) | Will 787 Production Aid Spirit AeroSystems  SPR  Q1 Earnings ... |
| 2019-04-30 | BUY | $368.84 | 0.336 | 0.792 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.336, SAE: 0.792) | Southwest Airlines loses top spot in passenger satisfaction  even as others gain... |
| 2019-05-01 | BUY | $367.97 | 0.350 | 0.752 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.350, SAE: 0.752) | Boeing says optional 737 MAX alert was  not activated as intended ... |
| 2019-05-01 | BUY | $367.97 | 0.344 | 0.417 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.417) | U S  House panel will hold May 15 hearing on grounded Boeing 737 MAX... |
| 2019-05-01 | BUY | $367.97 | 0.336 | 0.835 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.336, SAE: 0.835) | Boeing names special adviser to CEO amid 737 MAX crisis... |
| 2019-05-01 | BUY | $367.97 | 0.340 | 0.867 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.867) | Probe begins after Boeing 737 slides off runway into Florida river... |
| 2019-05-02 | BUY | $367.00 | 0.341 | 0.681 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.681) | Boeing did not disclose 737 MAX alert issue to FAA for 13 months... |
| 2019-05-09 | BUY | $347.82 | 0.336 | 0.554 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.336, SAE: 0.554) | U S  goods trade deficit with China tumbles to five year low... |
| 2019-05-09 | BUY | $347.82 | 0.340 | 0.761 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.761) | After U S  complaint  Canada to soften rules for jet competition to allow Lockhe... |
| 2019-05-09 | BUY | $347.82 | 0.337 | 0.493 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.493) | UAE s aviation authority says timing of lifting of Boeing 737 MAX ban still unkn... |
| 2019-05-09 | BUY | $347.82 | 0.342 | 0.614 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.614) | U S  goods trade deficit with China drops to five year low... |
| 2019-05-13 | BUY | $331.36 | 0.345 | 0.796 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.796) | Buy Deere  DE  Stock Before Q2 2019 Earnings Amid Renewed Trade War Fears ... |
| 2019-05-13 | BUY | $331.36 | 0.350 | 0.830 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.350, SAE: 0.830) | Canada s WestJet to be taken private in C 3 5 billion cash deal... |
| 2019-05-14 | SELL | $336.93 | 0.340 | 0.537 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.340, SAE: 0.537) | Boeing deliveries hammered by 737 MAX groundings... |
| 2019-05-14 | BUY | $336.93 | 0.343 | 0.632 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.632) | Northrop Grumman  NOC  Approves 10  Increase In Dividend... |
| 2019-05-14 | BUY | $336.93 | 0.337 | 0.611 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.611) | The Gloves Are Off... |
| 2019-05-15 | BUY | $339.48 | 0.340 | 0.693 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.693) | Boeing made mistakes on 737 MAX says Southwest CEO  hopeful planes return in U S... |
| 2019-05-15 | BUY | $339.48 | 0.337 | 0.781 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.781) | No easy options for China as trade war  U S  pressure bite... |
| 2019-05-15 | BUY | $339.48 | 0.359 | 0.681 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.359, SAE: 0.681) | Supersonic jets must be no noisier than existing planes  airports group... |
| 2019-05-16 | BUY | $347.51 | 0.346 | 0.419 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.419) | Cuba says Boeing 737 plane crash last year likely due to crew errors... |
| 2019-05-16 | BUY | $347.51 | 0.346 | 0.825 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.825) | Singapore Airlines says its order for 31 Boeing 737 MAX jets  intact ... |
| 2019-05-20 | BUY | $346.51 | 0.339 | 0.751 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.751) | Air New Zealand picks Boeing for wide body jet order  sources... |
| 2019-05-20 | BUY | $346.51 | 0.342 | 0.403 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.403) | European shares under pressure from chipmakers  Ryanair  trade worries... |
| 2019-05-21 | BUY | $352.36 | 0.338 | 0.623 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.623) | U S  airline group sees record summer travel  despite 737 MAX grounding... |
| 2019-05-21 | BUY | $352.36 | 0.345 | 0.657 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.657) | Turkish Airlines chairman expects Boeing 737 compensation  to meet Friday... |
| 2019-05-21 | SELL | $352.36 | 0.346 | 0.408 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.346, SAE: 0.408) | Equities Find Traction While Dollar Firms... |
| 2019-05-22 | BUY | $346.50 | 0.346 | 0.546 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.546) | 5 Airline Stocks To Gain On Record High Summer Travel... |
| 2019-05-22 | BUY | $346.50 | 0.349 | 0.746 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.746) | Why Is Hexcel  HXL  Up 8 2  Since Last Earnings Report ... |
| 2019-05-22 | BUY | $346.50 | 0.340 | 0.840 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.840) | Hexcel  HXL  Shares Up On Solid Long Term Financial Outlook... |
| 2019-05-22 | BUY | $346.50 | 0.360 | 0.793 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.360, SAE: 0.793) | Boeing Wins Deal To Offer Training Services For F 15 Program... |
| 2019-05-22 | BUY | $346.50 | 0.354 | 0.716 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.354, SAE: 0.716) | American Airlines  AAL  Down 12 7  Since Last Earnings Report  Can It Rebound ... |
| 2019-05-23 | BUY | $344.31 | 0.349 | 0.486 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.486) | Why Is Boeing  BA  Down 8 4  Since Last Earnings Report ... |
| 2019-05-23 | BUY | $344.31 | 0.345 | 0.816 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.816) | Exclusive  U S  airlines expect Boeing 737 MAX jets need up to 150 hours of work... |
| 2019-05-23 | BUY | $344.31 | 0.338 | 0.855 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.855) | Durable Goods Orders Tumble In April... |
| 2019-05-24 | BUY | $348.58 | 0.334 | 0.607 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.334, SAE: 0.607) | British Airways to resume Pakistan flights next week after a decade... |
| 2019-05-24 | BUY | $348.58 | 0.340 | 0.757 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.757) | Boeing faces SEC probe into disclosures about 737 MAX problems  Bloomberg... |
| 2019-05-24 | BUY | $348.58 | 0.338 | 0.724 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.724) | Durable Goods Orders Growth To Slow In April... |
| 2019-05-24 | SELL | $348.58 | 0.337 | 0.734 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.337, SAE: 0.734) | China airline association estimates losses from 737 MAX grounding at  579 millio... |
| 2019-05-24 | BUY | $348.58 | 0.352 | 0.887 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.352, SAE: 0.887) | Stock Market News For May 28  2019... |
| 2019-05-24 | BUY | $348.58 | 0.353 | 0.838 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.353, SAE: 0.838) | Gol Linhas  GOL  To Start Flights To Aracatuba In November... |
| 2019-05-28 | BUY | $348.56 | 0.340 | 0.686 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.686) | Trump seeks extra  1 6 billion in NASA spending to return to moon by 2024... |
| 2019-05-29 | BUY | $342.59 | 0.356 | 0.762 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.356, SAE: 0.762) | Top 5 Things to Know in the Market on Wednesday... |
| 2019-05-29 | BUY | $342.59 | 0.343 | 0.748 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.748) | Stocks    Dick s Sporting Goods Rises Premarket  Canada Goose  Boeing Slump... |
| 2019-05-29 | SELL | $342.59 | 0.339 | 0.877 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.339, SAE: 0.877) | A Brazil Firm Sweats Over Getting Caught in U S  China Trade War... |
| 2019-05-29 | BUY | $342.59 | 0.350 | 0.647 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.350, SAE: 0.647) | Boeing s 737 MAX Woes Aren t Over Yet  Wait Before Buying Stock... |
| 2019-05-30 | SELL | $343.64 | 0.336 | 0.622 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.336, SAE: 0.622) | U S  regulators say some Boeing 737 MAX planes may have faulty parts... |
| 2019-05-30 | BUY | $343.64 | 0.342 | 0.528 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.528) | Trade gloom  rising oil  Boeing 737 MAX woes to cloud aviation summit... |
| 2019-05-30 | BUY | $343.64 | 0.355 | 0.747 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.355, SAE: 0.747) | Why Is Spirit Aerosystems  SPR  Down 6 5  Since Last Earnings Report ... |
| 2019-06-03 | SELL | $332.85 | 0.334 | 0.602 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.334, SAE: 0.602) | Factbox  Trump administration departures  firings  reassignments... |
| 2019-06-03 | BUY | $332.85 | 0.339 | 0.674 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.674) | Stocks   U S  Futures Slip Amid Global Trade Tensions... |
| 2019-06-03 | BUY | $332.85 | 0.337 | 0.892 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.892) | U S  to sell 34 surveillance drones to allies in South China Sea region... |
| 2019-06-04 | BUY | $338.48 | 0.349 | 0.633 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.633) | More Problems For Boeing s 737  When It Will Fly Again ... |
| 2019-06-05 | BUY | $342.54 | 0.350 | 0.432 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.350, SAE: 0.432) | Boeing s 777X faces engine snags  questions rise over delivery goal... |
| 2019-06-05 | BUY | $342.54 | 0.337 | 0.606 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.606) | NASA s first SpaceX astronauts ready for  messy camping trip  to space... |
| 2019-06-06 | BUY | $344.39 | 0.339 | 0.516 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.516) | United Technologies  Raytheon to create  120 billion aerospace and defense giant... |
| 2019-06-06 | BUY | $344.39 | 0.347 | 0.787 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.787) | Exclusive  U S  pursues sale of over  2 billion in weapons to Taiwan  sources sa... |
| 2019-06-06 | BUY | $344.39 | 0.344 | 0.496 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.496) | Bear Of The Day  Boeing  BA ... |
| 2019-06-07 | BUY | $347.40 | 0.349 | 0.607 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.607) | Stocks     Beyond Meat  Zoom  Barnes   Noble Rise Premarket  DocuSign Falls... |
| 2019-06-07 | BUY | $347.40 | 0.341 | 0.434 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.434) | Spirit cuts workweek in wake of 737 MAX groundings  union... |
| 2019-06-10 | SELL | $347.50 | 0.335 | 0.718 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.335, SAE: 0.718) | Trump says United Tech  Raytheon deal may hurt competition... |
| 2019-06-10 | BUY | $347.50 | 0.340 | 0.771 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.771) | United Technologies   Raytheon  Becoming The  2 Player In Aerospace   Defense... |
| 2019-06-11 | BUY | $343.11 | 0.353 | 0.709 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.353, SAE: 0.709) | The Call Of The Void... |
| 2019-06-11 | BUY | $343.11 | 0.349 | 0.472 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.472) | Boeing May deliveries fall 56  as 737 MAX grounding continues to weigh... |
| 2019-06-12 | BUY | $340.85 | 0.341 | 0.636 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.636) | FAA says has no timetable for Boeing 737 MAX s return to service... |
| 2019-06-12 | BUY | $340.85 | 0.341 | 0.531 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.531) | American Airlines CEO says  highly likely  Boeing 737 MAX will fly by mid August... |
| 2019-06-12 | BUY | $340.85 | 0.353 | 0.439 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.353, SAE: 0.439) | Boeing  BA  Wins  194M Deal For MH 47G Rotary Wing Aircraft ... |
| 2019-06-13 | BUY | $342.65 | 0.346 | 0.508 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.508) | 3 Stocks To Watch In The Coming Week  Canopy Growth  GE  Adobe... |
| 2019-06-13 | BUY | $342.65 | 0.338 | 0.677 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.677) | USD Resilience In The Face Of Aggressive Fed Easing Expectations... |
| 2019-06-13 | BUY | $342.65 | 0.342 | 0.636 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.636) | Exclusive  Boeing seeking to reduce scope  duration of some physical tests for n... |
| 2019-06-13 | BUY | $342.65 | 0.353 | 0.684 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.353, SAE: 0.684) | Boeing Wins  41M Navy Deal For F A 18E F Aircraft Support... |
| 2019-06-13 | BUY | $342.65 | 0.344 | 0.636 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.636) | Airbus delivers first A350 to Japan after landmark deal... |
| 2019-06-14 | BUY | $340.98 | 0.341 | 0.631 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.631) | Exclusive  Business and pleasure   how Russian oil giant Rosneft uses its corpor... |
| 2019-06-14 | SELL | $340.98 | 0.342 | 0.779 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.342, SAE: 0.779) | Norwegian Air expects Boeing 737 MAX fleet to remain grounded until end of Augus... |
| 2019-06-17 | BUY | $348.58 | 0.342 | 0.608 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.608) | Boeing Hikes 20 Year Jetliner Demand Forecast By 6 7  To  16T... |
| 2019-06-17 | BUY | $348.58 | 0.358 | 0.866 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.358, SAE: 0.866) | Top 5 Things to Know in the Market on Monday... |
| 2019-06-17 | BUY | $348.58 | 0.339 | 0.794 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.794) | Boeing lifts 20 year industry demand forecast to  6 8 trillion... |
| 2019-06-17 | BUY | $348.58 | 0.344 | 0.410 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.410) | Stocks   Stocks Flat as Financials Fall on Growing Rate Cut Hopes... |
| 2019-06-18 | SELL | $367.30 | 0.336 | 0.862 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.336, SAE: 0.862) | With Esper  Pentagon inherits an army veteran long focused on China... |
| 2019-06-19 | BUY | $362.00 | 0.338 | 0.600 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.600) | Eurofighter  NATO launch studies on long term evolution of fighter... |
| 2019-06-19 | BUY | $362.00 | 0.344 | 0.473 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.473) | StockBeat  IAG s Rescue Act for the 737 MAX Goes Down Badly... |
| 2019-06-19 | BUY | $362.00 | 0.344 | 0.713 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.713) | Stocks   U S  Futures Flat Ahead of Fed Rate Decision... |
| 2019-06-19 | BUY | $362.00 | 0.337 | 0.522 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.522) | Southwest Falls Midday  Pilots Seek Compensation for Grounded Boeings... |
| 2019-06-19 | BUY | $362.00 | 0.343 | 0.426 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.426) | Trump s big show on July 4  patriotic speech or campaign rally ... |
| 2019-06-20 | BUY | $368.20 | 0.349 | 0.750 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.750) | Company News For Jun 21  2019... |
| 2019-06-24 | BUY | $367.33 | 0.346 | 0.690 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.690) | Headwinds Hound American Airlines  Better To Ditch The Stock ... |
| 2019-06-25 | BUY | $362.74 | 0.357 | 0.436 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.357, SAE: 0.436) | Durable Goods  Advance Trade Worse Than Expected... |
| 2019-06-25 | BUY | $362.74 | 0.343 | 0.609 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.609) | Boeing  Until The 737 Max Flies  This Dow Stock Is In A Holding Pattern... |
| 2019-06-25 | BUY | $362.74 | 0.337 | 0.670 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.670) | USDJPY Almost Flat On Hopes And Hypes Of U S  China Trade Truce And Trump s Iran... |
| 2019-06-26 | SELL | $368.26 | 0.346 | 0.660 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.346, SAE: 0.660) | United Airlines extends 737 MAX cancellations until Sept  3... |
| 2019-06-26 | BUY | $368.26 | 0.344 | 0.763 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.763) | Q1 GDP Stays 3 1   Jobless Claims Climb To 227K... |
| 2019-06-27 | BUY | $357.54 | 0.353 | 0.786 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.353, SAE: 0.786) | Stock Market News For Jun 28  2019... |
| 2019-06-27 | BUY | $357.54 | 0.339 | 0.719 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.719) | Boeing shares slip as grounded 737 MAX faces new hurdle... |
| 2019-06-27 | BUY | $357.54 | 0.342 | 0.473 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.473) | Boeing thinks it will complete it software update for 737 MAX by September  offi... |
| 2019-06-27 | BUY | $357.54 | 0.343 | 0.795 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.795) | Stocks   Boeing  ConAgra Fall Premarket  Ford  Walgreens Rise ... |
| 2019-06-28 | BUY | $357.53 | 0.340 | 0.701 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.701) | Boeing s 737 Max Software Outsourced to  9 an Hour Engineers... |
| 2019-06-28 | BUY | $357.53 | 0.336 | 0.793 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.336, SAE: 0.793) | U S  lags in export financing arms race fueled by China  EXIM report... |
| 2019-06-28 | BUY | $357.53 | 0.359 | 0.618 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.359, SAE: 0.618) | Lockheed Martin Wins  106M Deal To Support Apache Helicopter... |
| 2019-07-01 | BUY | $350.11 | 0.347 | 0.769 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.769) | U S  factory activity falls to more than two and a half year low... |
| 2019-07-01 | BUY | $350.11 | 0.337 | 0.718 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.718) | Southwest expects Boeing 737 MAX cancellations beyond Oct  1  CEO... |
| 2019-07-02 | BUY | $347.85 | 0.357 | 0.625 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.357, SAE: 0.625) | Top 5 Things to Know in the Market on Tuesday... |
| 2019-07-02 | BUY | $347.85 | 0.336 | 0.890 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.336, SAE: 0.890) | U N  aviation agency to review global pilot training in shadow of 737 MAX crashe... |
| 2019-07-03 | BUY | $348.16 | 0.337 | 0.730 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.730) | U S  factory orders fall for second straight month... |
| 2019-07-05 | BUY | $349.52 | 0.344 | 0.537 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.537) | U S  job growth seen accelerating  rate cut still expected... |
| 2019-07-08 | SELL | $344.87 | 0.336 | 0.526 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.336, SAE: 0.526) | Russia s Ural Airlines plans to get first Boeing 737 MAX in December  Interfax... |
| 2019-07-09 | BUY | $346.80 | 0.348 | 0.493 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.348, SAE: 0.493) | Airbus confirms deliveries rose 28  in first half of 2019... |
| 2019-07-12 | BUY | $358.82 | 0.353 | 0.636 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.353, SAE: 0.636) | Thomas Cook turns to China s Fosun to save oldest travel firm... |
| 2019-07-12 | BUY | $358.82 | 0.334 | 0.465 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.334, SAE: 0.465) | Boeing 737 MAX to remain off United Airlines schedule until November 3... |
| 2019-07-15 | BUY | $355.17 | 0.344 | 0.593 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.593) | Stocks   Citigroup  Amazon com  Rise Premarket  Boeing  CIRCOR Fall... |
| 2019-07-15 | BUY | $355.17 | 0.340 | 0.787 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.787) | United Airlines  UAL  Q2 2019 Earnings Preview... |
| 2019-07-16 | SELL | $356.29 | 0.337 | 0.401 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.337, SAE: 0.401) | Factbox  Airlines count the cost of Boeing 737 MAX grounding... |
| 2019-07-16 | BUY | $356.29 | 0.343 | 0.690 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.690) | Ryanair cuts summer 2020 growth rate on Boeing MAX doubts... |
| 2019-07-16 | BUY | $356.29 | 0.337 | 0.665 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.665) | United Airlines quarterly profit beats on strong travel demand  higher fares... |
| 2019-07-16 | BUY | $356.29 | 0.343 | 0.424 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.424) | Ryanair CEO says confident in  great  Boeing 737 MAX despite delays... |
| 2019-07-17 | BUY | $362.94 | 0.345 | 0.872 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.872) | United says may not receive full delivery of 737 MAX by next year... |
| 2019-07-17 | BUY | $362.94 | 0.335 | 0.778 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.335, SAE: 0.778) | United Airlines  UAL  Beats On Q2 Earnings  Tweaks  19 View... |
| 2019-07-18 | BUY | $354.68 | 0.340 | 0.867 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.867) | Will Escalated Costs Dampen Southwest s  LUV  Q2 Earnings ... |
| 2019-07-18 | BUY | $354.68 | 0.345 | 0.535 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.535) | Will 737 Max Issue Hamper Boeing s  BA  Earnings In Q2 ... |
| 2019-07-18 | BUY | $354.68 | 0.340 | 0.758 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.758) | Is A Beat Likely For American Airlines  AAL  In Q2 Earnings ... |
| 2019-07-19 | BUY | $370.64 | 0.345 | 0.539 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.539) | Buy Boeing  BA  Stock Ahead Of Q2 Earnings Report ... |
| 2019-07-19 | BUY | $370.64 | 0.337 | 0.731 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.731) | Boeing takes  4 9 billion charge related to 737 MAX... |
| 2019-07-19 | SELL | $370.64 | 0.338 | 0.781 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.338, SAE: 0.781) | Britain s SSP says hit by Boeing 737 MAX grounding... |
| 2019-07-19 | BUY | $370.64 | 0.346 | 0.669 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.669) | U S  stocks lower at close of trade  Dow Jones Industrial Average down 0 25 ... |
| 2019-07-19 | BUY | $370.64 | 0.342 | 0.798 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.798) | Explainer  NASA aims to build on moon as a way station for Mars... |
| 2019-07-22 | BUY | $366.77 | 0.344 | 0.889 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.889) | Factors Likely To Influence Raytheon s  RTN  Earnings In Q2... |
| 2019-07-23 | BUY | $366.43 | 0.347 | 0.790 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.790) | Boeing  BA  Q2 Earnings Down Y Y On Lower 737 Deliveries... |
| 2019-07-23 | BUY | $366.43 | 0.341 | 0.699 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.699) | Airline Stocks  Q2 Earnings Due On July 25  LUV  AAL   ALK... |
| 2019-07-23 | BUY | $366.43 | 0.336 | 0.695 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.336, SAE: 0.695) | Facebook Hit By FTC For  5B  Q2 Earnings From BA  CAT  T   More... |
| 2019-07-23 | BUY | $366.43 | 0.341 | 0.551 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.551) | Delta  Potbelly  Intel  Samsung And Advanced Micro Highlighted As Zacks Bull And... |
| 2019-07-23 | BUY | $366.43 | 0.345 | 0.585 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.585) | U S  OPEN   Dollar And Stocks Rally On Earnings And Debt Deal... |
| 2019-07-24 | BUY | $354.99 | 0.339 | 0.718 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.718) | Southwest  LUV  Q2 Earnings Beat  Q3   2019 Cost View Dull ... |
| 2019-07-25 | SELL | $341.89 | 0.337 | 0.872 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.337, SAE: 0.872) | U S  Economic Growth Seen Stumbling as Trade Hits Companies... |
| 2019-07-25 | BUY | $341.89 | 0.336 | 0.411 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.336, SAE: 0.411) | As Boeing targets October  FAA official says no timeline for 737 MAX... |
| 2019-07-25 | BUY | $341.89 | 0.342 | 0.653 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.653) | U S  Senate confirms Milley as chairman of Joint Chiefs... |
| 2019-07-25 | BUY | $341.89 | 0.337 | 0.631 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.631) | Southwest Airlines posts quarterly profit despite 737 MAX blow... |
| 2019-07-30 | BUY | $341.27 | 0.345 | 0.555 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.555) | Top Analyst Reports For Amazon  Boeing   NextEra ... |
| 2019-08-01 | BUY | $328.34 | 0.353 | 0.900 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.353, SAE: 0.900) | Stock Market News For Aug 5  2019... |
| 2019-08-02 | BUY | $333.51 | 0.348 | 0.879 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.348, SAE: 0.879) | U S  factory orders rise less than expected in June... |
| 2019-08-02 | BUY | $333.51 | 0.337 | 0.834 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.834) | U S  employment growth seen slowing in July  wage gains steady... |
| 2019-08-02 | SELL | $333.51 | 0.340 | 0.791 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.340, SAE: 0.791) | China Pledges Countermeasures as Trump Escalates Trade War Again... |
| 2019-08-05 | BUY | $325.16 | 0.338 | 0.641 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.641) | U S  Announces Another Round Of Tariffs On Chinese Goods... |
| 2019-08-05 | BUY | $325.16 | 0.342 | 0.718 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.718) | 4 Transportation Stocks Likely To Top Q2 Earnings Estimates ... |
| 2019-08-08 | SELL | $332.42 | 0.339 | 0.785 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.339, SAE: 0.785) | Why Trump Can t Afford To Intervene In The Dollar... |
| 2019-08-09 | BUY | $333.61 | 0.343 | 0.602 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.602) | Ryanair Spanish pilots threaten strike as unrest spreads... |
| 2019-08-09 | BUY | $333.61 | 0.346 | 0.615 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.615) | Air Lease  AL  Stock Down On Q2 Earnings And Revenue Miss... |
| 2019-08-12 | BUY | $329.05 | 0.356 | 0.838 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.356, SAE: 0.838) | Yes  You Can Time The Market  Find Out How   August 12  2019... |
| 2019-08-13 | BUY | $328.97 | 0.342 | 0.604 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.604) | Norwegian to stop flying from Ireland to U S  and Canada... |
| 2019-08-14 | BUY | $316.68 | 0.353 | 0.568 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.353, SAE: 0.568) | United  UAL  Down 13 5  Since Last Earnings Report  Can It Rebound ... |
| 2019-08-14 | BUY | $316.68 | 0.344 | 0.709 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.709) | Lockheed Wins  99M Contract To Support Production Of JASSM... |
| 2019-08-15 | BUY | $324.17 | 0.343 | 0.445 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.445) | Boeing delays delivery of ultra long range version of 777X... |
| 2019-08-19 | BUY | $329.88 | 0.342 | 0.577 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.577) | Time To Start Dollar Cost Averaging Into Boeing ... |
| 2019-08-20 | BUY | $327.87 | 0.350 | 0.658 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.350, SAE: 0.658) | Why Is Hexcel  HXL  Down 2 8  Since Last Earnings Report ... |
| 2019-08-21 | BUY | $336.02 | 0.343 | 0.740 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.740) | U S  stocks higher at close of trade  Dow Jones Industrial Average up 0 93 ... |
| 2019-08-22 | SELL | $350.27 | 0.343 | 0.830 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.343, SAE: 0.830) | United Airlines To Start Global Flights From Multiple Hubs... |
| 2019-08-22 | SELL | $350.27 | 0.337 | 0.517 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.337, SAE: 0.517) | Boeing 737 MAX And The FAA  A Clash Of Culture And Convexity... |
| 2019-08-22 | BUY | $350.27 | 0.343 | 0.452 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.452) | U S  stocks mixed at close of trade  Dow Jones Industrial Average up 0 19 ... |
| 2019-08-22 | BUY | $350.27 | 0.356 | 0.788 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.356, SAE: 0.788) | Boeing Secures  146M FMS Deal To Aid Apache Aircraft Program... |
| 2019-08-22 | BUY | $350.27 | 0.349 | 0.745 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.745) | Why Is Boeing  BA  Up 1 8  Since Last Earnings Report ... |
| 2019-08-22 | BUY | $350.27 | 0.345 | 0.694 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.694) | Bolivia president does about face and will now accept aid to put out wildfires... |
| 2019-08-22 | BUY | $350.27 | 0.334 | 0.799 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.334, SAE: 0.799) | FAA says invites global Boeing 737 MAX pilots to simulator tests... |
| 2019-08-22 | BUY | $350.27 | 0.347 | 0.521 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.521) | Stock Market News For Aug 23  2019... |
| 2019-08-23 | BUY | $351.85 | 0.352 | 0.536 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.352, SAE: 0.536) | Why Is American Airlines  AAL  Down 19 7  Since Last Earnings Report ... |
| 2019-08-26 | BUY | $354.85 | 0.336 | 0.766 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.336, SAE: 0.766) | Trump s  Call From China  Helps Markets Recover... |
| 2019-08-27 | BUY | $350.59 | 0.339 | 0.818 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.818) | Exclusive  Financial hit from 737 MAX will not slow appetite for services deals ... |
| 2019-08-27 | BUY | $350.59 | 0.357 | 0.820 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.357, SAE: 0.820) | Top 5 Things to Know in the Market on Tuesday... |
| 2019-08-27 | BUY | $350.59 | 0.345 | 0.605 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.605) | American Airlines At 52 Week Low  Does It Still Hold Promise ... |
| 2019-08-28 | BUY | $355.76 | 0.351 | 0.820 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.351, SAE: 0.820) | Boeing  BA  Wins  500M Deal To Support Qatar s F 15 Program... |
| 2019-08-28 | SELL | $355.76 | 0.343 | 0.676 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.343, SAE: 0.676) | United Airlines moving its Boeing 737 MAX jets to short term storage in Arizona... |
| 2019-08-29 | SELL | $358.50 | 0.347 | 0.673 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.347, SAE: 0.673) | Spanish cabin crew strike forces minimal Ryanair cancelations... |
| 2019-08-29 | BUY | $358.50 | 0.338 | 0.828 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.828) | Trade war dims outlook for U S  business operations in China  survey... |
| 2019-08-30 | BUY | $359.84 | 0.347 | 0.617 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.617) | Top Research Reports For Boeing  Starbucks   3M ... |
| 2019-08-30 | SELL | $359.84 | 0.344 | 0.799 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.344, SAE: 0.799) | United extends Boeing 737 MAX flight cancellations until December 19... |
| 2019-09-03 | BUY | $350.28 | 0.345 | 0.675 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.675) | Boeing Continues To Fall On Safety Woes  As Market Cycles Point Lower... |
| 2019-09-04 | SELL | $352.36 | 0.338 | 0.533 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.338, SAE: 0.533) | Bull Of The Day  Allegiant Travel Company  ALGT ... |
| 2019-09-05 | SELL | $356.09 | 0.335 | 0.686 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.335, SAE: 0.686) | Explainer  U S   China more divided than ever as new trade talks loom... |
| 2019-09-05 | BUY | $356.09 | 0.353 | 0.719 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.353, SAE: 0.719) | How To Time The Markets Like An Investing Pro   September 05  2019... |
| 2019-09-05 | BUY | $356.09 | 0.346 | 0.461 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.461) | Boeing suspends load test for new 777X aircraft... |
| 2019-09-06 | BUY | $358.76 | 0.343 | 0.894 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.894) | Air Lease  AL  Up 4 3  Since Last Earnings Report  Can It Continue ... |
| 2019-09-09 | BUY | $354.59 | 0.344 | 0.697 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.697) | Company News For Sep 10  2019... |
| 2019-09-10 | SELL | $365.17 | 0.339 | 0.767 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.339, SAE: 0.767) | Families meet with U S  transport chief after 737 MAX crashes... |
| 2019-09-10 | BUY | $365.17 | 0.358 | 0.632 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.358, SAE: 0.632) |  Lockheed Martin Wins  41M Deal To Support Apache Helicopter... |
| 2019-09-11 | BUY | $378.47 | 0.357 | 0.707 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.357, SAE: 0.707) | Boeing  BA  Secures Navy Deal To Support P 8A Jet Program... |
| 2019-09-12 | BUY | $371.24 | 0.334 | 0.818 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.334, SAE: 0.818) | Next Trump Tariffs May Soon Hit Europe s Luxury Goods Exporters... |
| 2019-09-13 | BUY | $375.32 | 0.347 | 0.827 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.827) | Crisis hit Boeing readies huge effort to return 737 MAX to the skies... |
| 2019-09-16 | BUY | $374.42 | 0.337 | 0.603 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.603) | U S  Air Fares Rise For 2 Months In A Row  What Lies Ahead ... |
| 2019-09-16 | BUY | $374.42 | 0.341 | 0.570 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.570) | China will need 8 090 new airplanes over next 20 years  Boeing... |
| 2019-09-17 | SELL | $379.71 | 0.335 | 0.751 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.335, SAE: 0.751) | U S  FAA to brief international regulators on status of Boeing 737 MAX... |
| 2019-09-18 | BUY | $381.90 | 0.345 | 0.643 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.643) | Lone 737 MAX criss crossed Canada for pilot checks during grounding... |
| 2019-09-19 | BUY | $379.95 | 0.348 | 0.756 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.348, SAE: 0.756) | Ryanair s O Leary wins bonus approval as pilots face axe... |
| 2019-09-19 | BUY | $379.95 | 0.343 | 0.681 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.681) | Ryanair expects to be flying Boeing 737 MAX by February March 2020... |
| 2019-09-20 | BUY | $374.96 | 0.349 | 0.625 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.625) | FAA waiting for more software details before 737 MAX can return to service... |
| 2019-09-23 | BUY | $372.63 | 0.340 | 0.503 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.503) | 5 Winners From U S  China s  Constructive  Trade Talks... |
| 2019-09-24 | BUY | $377.19 | 0.357 | 0.608 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.357, SAE: 0.608) | Boeing Wins  227M Deal To Support F A 18 And EA 18 Programs... |
| 2019-09-25 | BUY | $381.69 | 0.349 | 0.693 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.693) | TUI to Boost Winter Holiday Fleet After Thomas Cook Collapse... |
| 2019-09-25 | BUY | $381.69 | 0.341 | 0.432 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.432) | Russian jet hard lands in Siberia  49 people seek medical aid  RIA... |
| 2019-09-26 | BUY | $382.37 | 0.340 | 0.847 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.847) | Drake lends  185 million jet to Kings for long trip... |
| 2019-09-26 | BUY | $382.37 | 0.347 | 0.744 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.744) | The Gold Update  Gold At The 3 4 Post Is Putting In A Year Over Which To Boast... |
| 2019-09-26 | BUY | $382.37 | 0.343 | 0.712 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.712) | Boeing assumptions on 737 MAX emergency response faulted  NTSB... |
| 2019-09-26 | BUY | $382.37 | 0.341 | 0.849 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.849) | WTO to back around  7 5 billion U S  tariffs on EU in aircraft spat  sources... |
| 2019-09-26 | BUY | $382.37 | 0.343 | 0.726 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.726) | Boeing  BA  Wins  280M Deal To Support SDBI Weapon System... |
| 2019-09-27 | SELL | $378.39 | 0.342 | 0.494 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.342, SAE: 0.494) | U S  core capital goods orders unexpectedly fall... |
| 2019-09-27 | BUY | $378.39 | 0.341 | 0.631 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.631) | Explainer  The jet subsidy row that threatens transatlantic trade war... |
| 2019-10-01 | SELL | $370.56 | 0.340 | 0.631 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.340, SAE: 0.631) | American Airlines pilots demand compensation over Boeing 737 MAX grounding... |
| 2019-10-01 | BUY | $370.56 | 0.356 | 0.469 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.356, SAE: 0.469) | Lockheed Wins  495M Deal To Support Trident II Missile Program... |
| 2019-10-02 | BUY | $363.07 | 0.350 | 0.860 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.350, SAE: 0.860) | Vintage B 17 bomber makes fiery fatal landing in Connecticut  seven killed... |
| 2019-10-02 | BUY | $363.07 | 0.338 | 0.543 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.543) | Exclusive  Regulators weigh  startle factors  for Boeing 737 MAX pilot training ... |
| 2019-10-02 | BUY | $363.07 | 0.350 | 0.497 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.350, SAE: 0.497) | U S  orders speedy checks for cracks on 165 Boeing 737 NG planes... |
| 2019-10-03 | BUY | $367.72 | 0.342 | 0.703 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.703) | U S  committee seeks to interview Boeing engineer on safety of 737 MAX... |
| 2019-10-07 | BUY | $372.14 | 0.349 | 0.727 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.727) | US To Sell 150 Javelin Missiles To Ukraine  Stocks In Focus... |
| 2019-10-07 | BUY | $372.14 | 0.344 | 0.777 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.777) | Southwest pilots sue Boeing for misleading them on 737 MAX... |
| 2019-10-08 | BUY | $369.73 | 0.338 | 0.484 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.484) | Stocks   Wall Street Falls on Persistent Trade War Concerns ... |
| 2019-10-08 | BUY | $369.73 | 0.340 | 0.425 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.425) | Ethiopian Airlines flight makes emergency landing in Dakar  no casualties... |
| 2019-10-09 | BUY | $370.58 | 0.346 | 0.670 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.670) | American Airlines cancels 737 MAX flights until Jan  16... |
| 2019-10-09 | BUY | $370.58 | 0.341 | 0.626 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.626) | Boeing  BA  Q3 Defense Deliveries Soar Y Y On Solid Orders... |
| 2019-10-09 | BUY | $370.58 | 0.342 | 0.799 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.799) | Will Boeing  BA  Beat Estimates Again In Its Next Earnings Report ... |
| 2019-10-10 | SELL | $366.67 | 0.338 | 0.774 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.338, SAE: 0.774) | The Zacks Analyst Blog Highlights  Delta  American Airlines  United And Southwes... |
| 2019-10-10 | BUY | $366.67 | 0.348 | 0.498 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.348, SAE: 0.498) | Boeing s  BA  787 Fleet Hit By Aeroflot s Order Cancellation... |
| 2019-10-10 | BUY | $366.67 | 0.340 | 0.675 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.675) | Southwest  Gol ground 13 Boeing 737 NG airplanes after checks... |
| 2019-10-11 | BUY | $370.54 | 0.342 | 0.409 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.409) | FAA must ramp up staffing to oversee airplane certification after 737 MAX  panel... |
| 2019-10-11 | SELL | $370.54 | 0.343 | 0.652 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.343, SAE: 0.652) | United Airlines cancels Boeing 737 MAX flights until January 6... |
| 2019-10-11 | BUY | $370.54 | 0.337 | 0.794 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.794) | Boeing board strips CEO of chairman title amid 737 MAX crisis... |
| 2019-10-11 | BUY | $370.54 | 0.338 | 0.709 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.709) | Royal Air Maroc says has not canceled Boeing 737 MAX orders... |
| 2019-10-15 | BUY | $366.63 | 0.340 | 0.853 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.853) | United lifts 2019 profit target as strong travel demand outweighs MAX crisis... |
| 2019-10-15 | BUY | $366.63 | 0.346 | 0.466 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.466) | Boeing  BA  Expected To Beat Earnings Estimates  What To Know Ahead Of Q3 Releas... |
| 2019-10-15 | BUY | $366.63 | 0.342 | 0.714 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.714) | Southwest pilots say 737 MAX return may be delayed beyond Boeing s fourth quarte... |
| 2019-10-15 | BUY | $366.63 | 0.344 | 0.520 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.520) | U S  stocks higher at close of trade  Dow Jones Industrial Average up 0 89 ... |
| 2019-10-16 | BUY | $368.08 | 0.345 | 0.679 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.679) | Lockheed  LMT  To Report Q3 Earnings  What s In The Cards ... |
| 2019-10-16 | SELL | $368.08 | 0.339 | 0.645 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.339, SAE: 0.645) | Abu Dhabi s Etihad sets up low cost airline with Air Arabia... |
| 2019-10-17 | BUY | $364.75 | 0.352 | 0.871 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.352, SAE: 0.871) | Top Stock Reports For Johnson   Johnson  Boeing   Pfizer... |
| 2019-10-17 | BUY | $364.75 | 0.343 | 0.663 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.663) | 4 Airline Stocks Likely To Outshine Q3 Earnings Estimates... |
| 2019-10-17 | BUY | $364.75 | 0.342 | 0.566 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.566) | Southwest delays return of its Boeing 737 MAX jets to February... |
| 2019-10-18 | BUY | $339.98 | 0.338 | 0.624 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.624) | Norwegian Air to sell five aircraft  boosting cash... |
| 2019-10-18 | BUY | $339.98 | 0.345 | 0.877 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.877) | Boeing pilots  messages on 737 MAX safety raise new questions... |
| 2019-10-21 | BUY | $327.19 | 0.344 | 0.647 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.647) | Boeing Falls 5 ... |
| 2019-10-21 | BUY | $327.19 | 0.341 | 0.818 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.818) | Hexcel  HXL  Q3 Earnings Beat Estimates  Revenues Up Y Y... |
| 2019-10-21 | BUY | $327.19 | 0.344 | 0.772 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.772) | U S  stocks higher at close of trade  Dow Jones Industrial Average up 0 21 ... |
| 2019-10-21 | SELL | $327.19 | 0.336 | 0.621 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.336, SAE: 0.621) | Boeing expresses regret over ex pilot s 737 MAX messages  faults simulator... |
| 2019-10-21 | BUY | $327.19 | 0.342 | 0.513 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.513) | Boeing Preview   Worst Yet To Come In Wake Of Plane Maker s Max Crisis... |
| 2019-10-21 | BUY | $327.19 | 0.343 | 0.824 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.824) | Who Letwin The Dogs Out  Brexit Update And Monday Markets... |
| 2019-10-22 | BUY | $333.06 | 0.338 | 0.603 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.603) | Boeing ousts senior executive as 737 MAX crisis grows... |
| 2019-10-23 | BUY | $336.52 | 0.341 | 0.645 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.645) | Boeing Earnings Miss  Revenue Beats In Q3... |
| 2019-10-23 | BUY | $336.52 | 0.336 | 0.647 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.336, SAE: 0.647) | Southwest  LUV  Q3 Earnings Top Estimates Amid Rising Costs ... |
| 2019-10-24 | BUY | $340.52 | 0.341 | 0.746 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.746) | Brazil working to bring Boeing 737 MAX plane back into service this year... |
| 2019-10-24 | BUY | $340.52 | 0.334 | 0.869 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.334, SAE: 0.869) | U S  lawmakers will press Boeing CEO for answers on 737 MAX crashes... |
| 2019-10-24 | SELL | $340.52 | 0.345 | 0.825 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.345, SAE: 0.825) | U S  Senate Democrats introduce aviation safety bill after Boeing MAX crashes... |
| 2019-10-24 | BUY | $340.52 | 0.348 | 0.563 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.348, SAE: 0.563) | American Airlines profit beats estimates on lower fuel costs... |
| 2019-10-24 | SELL | $340.52 | 0.340 | 0.556 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.340, SAE: 0.556) | Southwest profit jumps 7 2  as demand  MAX cancellations push up fares... |
| 2019-10-24 | BUY | $340.52 | 0.340 | 0.707 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.707) | South Korea grounds nine Boeing 737 NG planes with cracks... |
| 2019-10-25 | BUY | $335.86 | 0.346 | 0.457 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.457) | The Keys To Successfully Timing The Markets   October 25  2019... |
| 2019-10-30 | BUY | $342.02 | 0.341 | 0.771 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.771) | GE still sees  1 4 billion cash cost from Boeing 737 MAX grounding in 2019... |
| 2019-10-30 | BUY | $342.02 | 0.343 | 0.695 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.695) | Bear Of The Day  Caterpillar Inc   CAT ... |
| 2019-10-30 | BUY | $342.02 | 0.349 | 0.848 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.848) | GE raises cash forecast  beats on adjusted EPS  lifting shares... |
| 2019-10-31 | BUY | $335.94 | 0.342 | 0.799 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.799) | Boeing says up to 50 planes grounded globally over cracks  AFP... |
| 2019-10-31 | BUY | $335.94 | 0.342 | 0.704 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.704) | Qantas says cracks found in three 737 NG jets  will minimize customer impact... |
| 2019-11-05 | BUY | $354.10 | 0.343 | 0.711 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.711) | Regulators should work together on certifying Boeing 737 MAX  IATA s de Juniac... |
| 2019-11-05 | BUY | $354.10 | 0.351 | 0.636 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.351, SAE: 0.636) | U S  trade deficit falls in September to  52 5 billion... |
| 2019-11-06 | BUY | $350.05 | 0.342 | 0.462 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.462) | Boeing to invest  1 billion in global safety drive  sources... |
| 2019-11-07 | BUY | $355.20 | 0.346 | 0.447 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.447) | Stocks   Qualcomm  Nielsen Rise Premarket  Expedia  Roku Fall... |
| 2019-11-08 | BUY | $348.92 | 0.342 | 0.462 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.462) | NewsBreak  Southwest Air Doesn t See 737 Max Flying Before March... |
| 2019-11-08 | BUY | $348.92 | 0.342 | 0.624 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.624) | General Electric Offers Some Hope  But Is Its Stock Spike Sustainable ... |
| 2019-11-11 | BUY | $364.79 | 0.342 | 0.513 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.513) | S P 500  Nasdaq slip on trade uncertainty  Boeing buoys Dow... |
| 2019-11-11 | BUY | $364.79 | 0.347 | 0.847 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.847) | Stock Market News For Nov 12  2019... |
| 2019-11-11 | BUY | $364.79 | 0.352 | 0.601 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.352, SAE: 0.601) | Boeing Rises 5 ... |
| 2019-11-11 | BUY | $364.79 | 0.341 | 0.709 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.709) | Southwest will speed up inspections of 38 used 737 airplanes after FAA concerns... |
| 2019-11-12 | BUY | $360.73 | 0.344 | 0.896 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.896) | Flying 14 hours or more  Boeing sees longer routes as  key  for growth... |
| 2019-11-12 | BUY | $360.73 | 0.347 | 0.593 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.593) | Markets Trade Nervously In Asia... |
| 2019-11-13 | BUY | $360.36 | 0.340 | 0.411 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.411) | Indonesia waiting on major global aviation regulators for return of 737 MAX  off... |
| 2019-11-15 | BUY | $369.48 | 0.335 | 0.720 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.335, SAE: 0.720) | U S  FAA regulator head tells team to  take whatever time needed  on 737 MAX... |
| 2019-11-18 | BUY | $367.27 | 0.341 | 0.645 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.645) | Russian military exports unaffected by sanctions  Rostec CEO... |
| 2019-11-18 | BUY | $367.27 | 0.342 | 0.792 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.792) | The Zacks Analyst Blog Highlights  Alphabet  Amazon  Johnson   Johnson  Boeing A... |
| 2019-11-18 | BUY | $367.27 | 0.345 | 0.544 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.544) | Lawsuit against Boeing seeks to hold board liable for 737 MAX problems... |
| 2019-11-18 | BUY | $367.27 | 0.345 | 0.896 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.896) | Boeing to give Southwest board 737 MAX update this week... |
| 2019-11-18 | BUY | $367.27 | 0.356 | 0.753 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.356, SAE: 0.753) | Top Analyst Reports For Alphabet  Amazon   Johnson   Johnson... |
| 2019-11-19 | BUY | $364.83 | 0.337 | 0.408 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.408) | Dismal U S  Industrial Output Puts Focus On ETFs ... |
| 2019-11-20 | BUY | $368.72 | 0.337 | 0.826 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.826) | U S  aviation regulator pledges rigorous certification for Boeing 777X... |
| 2019-11-20 | BUY | $368.72 | 0.348 | 0.561 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.348, SAE: 0.561) | Emirates jet swap opens door to Boeing 787 deal in Dubai... |
| 2019-11-21 | BUY | $364.27 | 0.348 | 0.624 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.348, SAE: 0.624) | Why Is Boeing  BA  Up 6 4  Since Last Earnings Report ... |
| 2019-11-21 | BUY | $364.27 | 0.344 | 0.585 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.585) | Major 737 MAX customer SMBC says it will take time to return plane to service... |
| 2019-11-21 | BUY | $364.27 | 0.352 | 0.771 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.352, SAE: 0.771) | Raytheon Wins  72M Deal To Support AMRAAM Weapon System... |
| 2019-11-22 | BUY | $369.14 | 0.340 | 0.512 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.512) | Jet industry s grand masters fight to a draw in Dubai... |
| 2019-11-22 | BUY | $369.14 | 0.340 | 0.408 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.408) | Boeing s top communications official to retire as 737 MAX crisis drags on... |
| 2019-11-25 | BUY | $370.92 | 0.351 | 0.641 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.351, SAE: 0.641) | Curtiss Wright Signs Agreement To Acquire 901D Holdings... |
| 2019-11-25 | BUY | $370.92 | 0.342 | 0.562 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.562) | Brazil s Guedes is not worried by deficit  currency fluctuations... |
| 2019-11-25 | BUY | $370.92 | 0.354 | 0.676 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.354, SAE: 0.676) | Boeing Wins  172M Navy Deal For F A 18E F Aircraft Support... |
| 2019-11-26 | BUY | $371.30 | 0.339 | 0.710 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.710) | Report cites pilot error in 2016 Russia Flydubai plane crash... |
| 2019-12-03 | BUY | $350.00 | 0.343 | 0.779 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.779) | U S  sends first Salvadoran back to Guatemala under asylum deal... |
| 2019-12-03 | BUY | $350.00 | 0.346 | 0.775 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.775) | India weighs tougher rules for Boeing 737 MAX on return to flying  source... |
| 2019-12-04 | BUY | $346.78 | 0.345 | 0.789 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.789) | U S  stocks higher at close of trade  Dow Jones Industrial Average up 0 53 ... |
| 2019-12-05 | BUY | $343.64 | 0.342 | 0.688 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.688) | Airbus faces delivery challenge  poised to win jet order race... |
| 2019-12-05 | BUY | $343.64 | 0.341 | 0.706 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.706) | U S  factory orders rebound in October  shipments unchanged... |
| 2019-12-06 | BUY | $352.00 | 0.347 | 0.702 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.702) | Why Is Air Lease  AL  Up 0 5  Since Last Earnings Report ... |
| 2019-12-09 | BUY | $349.13 | 0.344 | 0.548 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.548) | Former Boeing employee who warned about 737 problems will testify at hearing... |
| 2019-12-10 | BUY | $345.84 | 0.343 | 0.423 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.423) | Ryanair says Europe delays may mean no MAX jets next summer... |
| 2019-12-10 | BUY | $345.84 | 0.349 | 0.735 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.735) | Boeing deliveries halved in first 11 months of 2019... |
| 2019-12-10 | BUY | $345.84 | 0.340 | 0.780 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.780) | Boeing reports 30 new orders for the troubled 737 Max in November... |
| 2019-12-10 | BUY | $345.84 | 0.344 | 0.642 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.642) | Can Dow ETFs Be A Winner In 2020 ... |
| 2019-12-11 | BUY | $347.93 | 0.343 | 0.686 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.686) | FAA won t rule out fining Boeing over 737 Max safety disclosures... |
| 2019-12-11 | BUY | $347.93 | 0.347 | 0.510 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.510) | IAI warns on converted Boeing 737 freighters... |
| 2019-12-11 | BUY | $347.93 | 0.349 | 0.615 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.615) | The Market Timing Secrets No One Talks About   December 12  2019... |
| 2019-12-12 | SELL | $344.24 | 0.336 | 0.605 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.336, SAE: 0.605) | China has  important concerns  about Boeing 737 MAX design changes  regulator... |
| 2019-12-12 | BUY | $344.24 | 0.343 | 0.701 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.701) | Southwest reaches deal with Boeing over 737 Max... |
| 2019-12-12 | BUY | $344.24 | 0.350 | 0.614 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.350, SAE: 0.614) | Exclusive  Boeing delays plans for record 737 production until 2021   sources... |
| 2019-12-12 | BUY | $344.24 | 0.347 | 0.751 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.751) | Empire State Up 3 5  Boeing To Suspend MAX ... |
| 2019-12-13 | BUY | $339.65 | 0.344 | 0.704 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.704) | Regulator puts loading curbs on 737 jets converted to freighters by Israel Aeros... |
| 2019-12-16 | BUY | $325.07 | 0.346 | 0.817 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.817) | Boeing to temporarily halt 737 Max production in January... |
| 2019-12-16 | BUY | $325.07 | 0.349 | 0.500 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.500) | Will Boeing Halt or Just Slow Down 737 Max Production ... |
| 2019-12-16 | BUY | $325.07 | 0.348 | 0.742 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.348, SAE: 0.742) | Boeing suspends 737 production for the first time in 20 years  WSJ... |
| 2019-12-16 | BUY | $325.07 | 0.354 | 0.655 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.354, SAE: 0.655) | Top 5 Things to Know in the Market on Monday  16th December... |
| 2019-12-17 | BUY | $325.07 | 0.343 | 0.776 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.776) | Boeing will still burn  1 billion a month on 737 Max even with production halt  ... |
| 2019-12-17 | BUY | $325.07 | 0.335 | 0.664 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.335, SAE: 0.664) | Explainer  How the 737 MAX production freeze affects airlines across the globe... |
| 2019-12-17 | BUY | $325.07 | 0.349 | 0.775 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.775) | Boeing To Halt 737 Production  ETF Losers   One Likely Winner... |
| 2019-12-17 | BUY | $325.07 | 0.341 | 0.790 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.790) | Boeing Will Suspend 737 Max Production  Thousands Of Jobs At Risk... |
| 2019-12-17 | BUY | $325.07 | 0.361 | 0.745 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.361, SAE: 0.745) | Top 5 Things to Watch in Markets  on Dec  17... |
| 2019-12-17 | BUY | $325.07 | 0.335 | 0.628 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.335, SAE: 0.628) | Boeing 737 MAX freeze divides suppliers into haves and have nots... |
| 2019-12-17 | BUY | $325.07 | 0.349 | 0.468 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.468) | Daily Markets Broadcast December 17  2019... |
| 2019-12-17 | BUY | $325.07 | 0.343 | 0.445 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.445) | Boeing s production pause will not end 737 Max cash burn  analysts... |
| 2019-12-18 | BUY | $328.72 | 0.342 | 0.552 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.552) | U S  stocks mixed at close of trade  Dow Jones Industrial Average down 0 10 ... |
| 2019-12-18 | BUY | $328.72 | 0.344 | 0.740 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.740) | 2019 Hits A Purple Patch So Far  Check The Movers   Shakers... |
| 2019-12-19 | BUY | $331.53 | 0.344 | 0.568 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.568) | Company News For Dec 23  2019... |
| 2019-12-19 | BUY | $331.53 | 0.340 | 0.767 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.767) | Boeing s 737 MAX Production Halt Means More Pain Ahead For Shareholders... |
| 2019-12-20 | BUY | $326.06 | 0.339 | 0.612 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.612) | U S  third quarter growth unrevised at 2 1 ... |
| 2019-12-23 | BUY | $335.55 | 0.341 | 0.898 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.898) | Boeing fires CEO Muilenburg to restore confidence amid 737 crisis... |
| 2019-12-23 | BUY | $335.55 | 0.345 | 0.715 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.715) | Rallying Into The Holiday... |
| 2019-12-23 | BUY | $335.55 | 0.350 | 0.594 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.350, SAE: 0.594) | Boeing contributing 70 points to the Dow s 85 point rise early Monday... |
| 2019-12-23 | BUY | $335.55 | 0.341 | 0.797 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.797) | Boeing CEO had to be fired in order for the 737 Max to get FAA certified  Jim Cr... |
| 2019-12-23 | BUY | $335.55 | 0.346 | 0.634 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.634) | Sterling Continues To Unwind Election Gains... |
| 2019-12-23 | BUY | $335.55 | 0.340 | 0.510 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.510) | Boeing CEO Dennis Muilenburg is out  as the company struggles with 737 Max crisi... |
| 2019-12-23 | BUY | $335.55 | 0.339 | 0.508 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.508) | Airline Stock Roundup  DAL To Rejoin A4A  AAL To Debut In Africa  UAL In Focus... |
| 2019-12-24 | BUY | $331.03 | 0.343 | 0.755 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.755) | Emirates  long serving boss to hand over the controls next year... |
| 2019-12-26 | BUY | $327.97 | 0.347 | 0.883 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.883) | Defense Stock Roundup  LMT Wins Big Deal  AIR Posts Better Than Expected Q2 Resu... |
| 2019-12-26 | BUY | $327.97 | 0.346 | 0.708 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.708) | Boeing secures  560M  in Apache orders... |
| 2019-12-31 | BUY | $323.83 | 0.335 | 0.724 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.335, SAE: 0.724) | Market Risks And Opportunities In The New Year... |
| 2020-01-02 | BUY | $331.35 | 0.341 | 0.700 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.341, SAE: 0.700) | Boeing  FAA reviewing wiring issue that could cause short circuit on 737 Max... |
| 2020-01-02 | BUY | $331.35 | 0.338 | 0.792 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.792) | Airbus bumps Boeing from top spot in 2019 with 863 jet deliveries... |
| 2020-01-02 | BUY | $331.35 | 0.335 | 0.767 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.335, SAE: 0.767) | Boeing  FAA reviewing wiring issue on grounded 737 MAX... |
| 2020-01-03 | BUY | $330.79 | 0.346 | 0.618 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.618) | U S  factory sector in deepest slump in more than 10 years... |
| 2020-01-06 | BUY | $331.77 | 0.344 | 0.879 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.879) | ETFs To Suffer As ISM Index Drops To Lowest Level In A Decade... |
| 2020-01-06 | BUY | $331.77 | 0.343 | 0.784 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.784) | Boeing may tap debt market to help pay 737 MAX costs  WSJ... |
| 2020-01-06 | BUY | $331.77 | 0.335 | 0.672 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.335, SAE: 0.672) | As Problems Mount  Boeing Should Spin Off Defense Operations... |
| 2020-01-06 | SELL | $331.77 | 0.337 | 0.854 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.337, SAE: 0.854) | Boeing reassigns thousands of 737 Max workers while supplier Spirit mulls layoff... |
| 2020-01-07 | BUY | $335.29 | 0.344 | 0.602 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.602) | Boeing changes stance  recommends 737 MAX simulator training for pilots... |
| 2020-01-07 | BUY | $335.29 | 0.342 | 0.442 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.442) | Boeing to recommend simulator training for 737 MAX pilots... |
| 2020-01-07 | BUY | $335.29 | 0.353 | 0.628 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.353, SAE: 0.628) | Boeing reverses 21 point drag on Dow to become 59 point contributor... |
| 2020-01-08 | BUY | $329.41 | 0.337 | 0.860 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.860) | S P 500  Nasdaq hit record highs as U S  Iran escalation fears fade... |
| 2020-01-08 | SELL | $329.41 | 0.336 | 0.489 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.336, SAE: 0.489) | European shares rebound from early losses as U S  Iran tensions ebb... |
| 2020-01-08 | BUY | $329.41 | 0.362 | 0.583 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.362, SAE: 0.583) | Top 5 Things to Watch on Wednesday  Jan 8... |
| 2020-01-08 | BUY | $329.41 | 0.344 | 0.790 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.790) | Dow Jones News  Walgreens Misses Estimates  Boeing Price Targets Cut... |
| 2020-01-08 | BUY | $329.41 | 0.345 | 0.785 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.785) | Another Boeing  BA  737 Crash  2020 Starts On A Torrid Note... |
| 2020-01-08 | BUY | $329.41 | 0.340 | 0.853 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.853) | Canada wants big role in Iran crash probe despite lack of diplomatic ties  Trude... |
| 2020-01-08 | BUY | $329.41 | 0.338 | 0.677 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.677) | Better Buy  Boeing vs  Brookfield Asset Management... |
| 2020-01-09 | SELL | $334.35 | 0.338 | 0.699 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.338, SAE: 0.699) | Iran Admits Downing Jetliner  Sparking Global Anger and Protests... |
| 2020-01-09 | BUY | $334.35 | 0.348 | 0.558 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.348, SAE: 0.558) | Boeing  BA  Wins  42M Navy Deal To Modify P 8A Jet Program... |
| 2020-01-10 | BUY | $327.97 | 0.339 | 0.873 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.873) | Stocks   Dow Reverses Course  Loses 130 After Hitting 29 000... |
| 2020-01-10 | BUY | $327.97 | 0.344 | 0.650 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.650) | Explainer  Reading  black boxes  of Ukrainian jet that crashed in Iran... |
| 2020-01-10 | BUY | $327.97 | 0.344 | 0.713 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.713) | European Open   Iran  U S  Jobs  Gold  Oil  Bitcoin... |
| 2020-01-10 | BUY | $327.97 | 0.340 | 0.747 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.747) | Iranians turn to social media to grieve and rage over doomed plane... |
| 2020-01-10 | BUY | $327.97 | 0.338 | 0.543 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.543) | Boeing hit with  5 4M FAA fine for defective 737 MAX wing parts... |
| 2020-01-10 | BUY | $327.97 | 0.356 | 0.803 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.356, SAE: 0.803) | Top 5 Things to Watch on Friday Jan  10... |
| 2020-01-10 | BUY | $327.97 | 0.338 | 0.520 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.520) | Boeing supplier Spirit Aero to layoff 2 800 workers in wake of 737 Max debacle... |
| 2020-01-13 | BUY | $328.27 | 0.338 | 0.444 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.444) | Boeing s new CEO takes the reins with little margin for error as 737 Max crisis ... |
| 2020-01-13 | BUY | $328.27 | 0.343 | 0.654 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.654) | Stocks  Lululemon up in Premarket on Strong Q4  Alphabet Eyes  1 Trln Mark... |
| 2020-01-13 | BUY | $328.27 | 0.342 | 0.830 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.830) | American Airlines Stock Down On Bearish Q4 TRASM Outlook... |
| 2020-01-14 | BUY | $330.38 | 0.344 | 0.775 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.775) | China to ramp up U S  car  aircraft  energy purchases in trade deal  source... |
| 2020-01-14 | BUY | $330.38 | 0.337 | 0.601 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.601) | Will Boeing Skip Its Dividend Hike in 2020 ... |
| 2020-01-14 | BUY | $330.38 | 0.343 | 0.607 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.607) | American Airlines pushes back Boeing 737 Max return to April... |
| 2020-01-15 | BUY | $327.85 | 0.352 | 0.681 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.352, SAE: 0.681) | Boeing Wins Deal To Supply Spare Parts For F A 18 Aircraft... |
| 2020-01-15 | BUY | $327.85 | 0.337 | 0.862 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.862) | Trump urges Boeing to move fast on resolving 737 MAX issues... |
| 2020-01-16 | BUY | $330.04 | 0.339 | 0.621 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.339, SAE: 0.621) | Southwest pulls Boeing 737 Max until June as airlines dig in for longer delays... |
| 2020-01-16 | BUY | $330.04 | 0.337 | 0.786 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.786) | DOT panel calls for more oversight after review of Boeing 737 Max approval... |
| 2020-01-16 | BUY | $330.04 | 0.346 | 0.879 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.879) | Iran aims to examine downed plane s black boxes  no plan yet to send them abroad... |
| 2020-01-16 | SELL | $330.04 | 0.340 | 0.682 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.340, SAE: 0.682) | Southwest extends 737 MAX cancellations through June 6... |
| 2020-01-16 | BUY | $330.04 | 0.344 | 0.621 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.621) | Better Buy  General Electric vs  Boeing... |
| 2020-01-17 | BUY | $322.23 | 0.342 | 0.475 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.475) | A  50 Billion Hole Adds Intrigue to China s U S  Export Binge... |
| 2020-01-17 | BUY | $322.23 | 0.342 | 0.742 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.742) | Boeing Falls as Analyst Sounds Alarm on  Significant  737 Max Costs... |
| 2020-01-17 | BUY | $322.23 | 0.338 | 0.606 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.606) | Can t sell your presidential plane  Mexico mulls raffle instead... |
| 2020-01-17 | BUY | $322.23 | 0.347 | 0.738 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.347, SAE: 0.738) | Boeing seeks to borrow  10 billion or more as 737 Max crisis wears on... |
| 2020-01-21 | BUY | $311.52 | 0.351 | 0.795 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.351, SAE: 0.795) | U S  may grow more quickly this year than projections  Mnuchin... |
| 2020-01-21 | BUY | $311.52 | 0.348 | 0.628 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.348, SAE: 0.628) | Boeing plans to stage first flight of 777X plane this week  sources... |
| 2020-01-21 | BUY | $311.52 | 0.345 | 0.601 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.345, SAE: 0.601) | Boeing stock halted for news pending amid reports of further delays for 737 Max ... |
| 2020-01-21 | BUY | $311.52 | 0.349 | 0.702 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.349, SAE: 0.702) | Boeing expects 737 Max back to service in mid 2020... |
| 2020-01-21 | SELL | $311.52 | 0.334 | 0.541 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.334, SAE: 0.541) | Brazil s GOL sees 737 MAX flying by April  compensation talks ongoing... |
| 2020-01-21 | BUY | $311.52 | 0.338 | 0.852 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.852) | Boeing doesn t expect regulators to sign off on 737 Max until June or July... |
| 2020-01-21 | BUY | $311.52 | 0.344 | 0.792 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.792) | Breaking  Boeing Reportedly Sees 737 Return in June July  Shares Tumble... |
| 2020-01-21 | BUY | $311.52 | 0.343 | 0.662 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.662) | Coronavirus Infects Wall Street  Stocks To Gain   Lose... |
| 2020-01-21 | BUY | $311.52 | 0.340 | 0.871 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.871) | Eaton To Sell Hydraulics For  3 3B  Focus On Core Business ... |
| 2020-01-21 | BUY | $311.52 | 0.343 | 0.683 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.683) | United beats Wall Street expectations despite ongoing 737 MAX delays... |
| 2020-01-21 | BUY | $311.52 | 0.344 | 0.406 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.406) | Boeing reportedly sees no 737 MAX signoff until summer... |
| 2020-01-22 | BUY | $307.17 | 0.343 | 0.721 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.721) | Simple Market Timing Strategies That Work   January 23  2020... |
| 2020-01-22 | BUY | $307.17 | 0.350 | 0.620 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.350, SAE: 0.620) | StockBeat  Dieselgate Returns to Haunt Daimler  Again... |
| 2020-01-22 | BUY | $307.17 | 0.336 | 0.667 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.336, SAE: 0.667) | United Airlines doesn t expect to fly the Boeing 737 Max this summer... |
| 2020-01-22 | BUY | $307.17 | 0.343 | 0.604 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.604) | The Boeing 737 MAX Crisis Will Crush American Airlines  Profit Again in 2020... |
| 2020-01-22 | BUY | $307.17 | 0.353 | 0.656 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.353, SAE: 0.656) | Top 5 Things to Know in the Market on Wednesday  22nd January... |
| 2020-01-23 | BUY | $315.91 | 0.337 | 0.623 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.623) | U S  weekly jobless claims rise modestly  labor market tight... |
| 2020-01-23 | BUY | $315.91 | 0.338 | 0.436 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.338, SAE: 0.436) | This May Be The Most Important Week So Far In 2020  Here s Why... |
| 2020-01-24 | BUY | $321.14 | 0.346 | 0.668 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.668) | The Corn And Ethanol Report  01 24 2020... |
| 2020-01-27 | BUY | $314.73 | 0.334 | 0.702 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.334, SAE: 0.702) | A Larger Debt Offering From Boeing Might Actually Be Positive... |
| 2020-01-27 | BUY | $314.73 | 0.346 | 0.495 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.495) | Defense Stocks  Q4 Earnings Lineup For Jan 29  BA  GD  TXT... |
| 2020-01-28 | BUY | $314.69 | 0.350 | 0.755 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.350, SAE: 0.755) | Boeing  BA  Q4 Earnings  Revenues Miss On Lower 737 Deliveries... |
| 2020-01-28 | BUY | $314.69 | 0.352 | 0.765 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.352, SAE: 0.765) | UTC says Collins 2020 profit to be hurt mainly due to MAX grounding... |
| 2020-01-28 | BUY | $314.69 | 0.344 | 0.567 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.567) | What to expect from Boeing s earnings as 737 Max crisis continues... |
| 2020-01-29 | BUY | $320.12 | 0.343 | 0.414 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.414) |  My stomach turned    Boeing CEO outraged by internal 737 Max messages before de... |
| 2020-01-29 | BUY | $320.12 | 0.344 | 0.767 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.344, SAE: 0.767) | Boeing Co  BA  Q4 2019 Earnings Call Transcript... |
| 2020-01-29 | BUY | $320.12 | 0.340 | 0.806 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.806) | Stocks   Wall Street Mixed Early  Reflecting Inconsistent Earnings... |
| 2020-01-29 | BUY | $320.12 | 0.340 | 0.645 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.340, SAE: 0.645) | Explainer  What the U S  Federal Reserve is watching this year... |
| 2020-01-29 | BUY | $320.12 | 0.348 | 0.753 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.348, SAE: 0.753) | Buy Caterpillar  CAT  Stock Before Q4 Earnings On Possible 2020 Comeback ... |
| 2020-01-31 | SELL | $316.39 | 0.342 | 0.774 | SELL: Both models agree on negative sentiment (ProbeTrain: 0.342, SAE: 0.774) | Trade Helps U S  Economy Grow 2 1  While Consumption Slows... |
| 2020-01-31 | BUY | $316.39 | 0.346 | 0.855 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.346, SAE: 0.855) | Spirit Aero CFO resigns on accounting irregularity  shares fall 7 ... |
| 2020-01-31 | BUY | $316.39 | 0.343 | 0.479 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.343, SAE: 0.479) | Southwest disputes U S  government audit on safety lapses  shares slide... |
| 2020-06-08 | BUY | $230.50 | 0.351 | 0.528 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.351, SAE: 0.528) | Tech Stocks And FAANGS Strong Again To Start Day As Market Awaits Fed... |
| 2020-06-09 | BUY | $216.74 | 0.337 | 0.730 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.337, SAE: 0.730) | Travel Stocks Lose Momentum In Pre-Market After Leading Yesterday's Sharp Rally... |
| 2020-06-09 | BUY | $216.74 | 0.342 | 0.871 | BUY: Both models agree on positive sentiment (ProbeTrain: 0.342, SAE: 0.871) | Tuesday's Market Minute: Stocks Pause After S&P Turns Positive... |
