
# Financial Trading

### Trading
- We can categorize traders into a few types by how long they hold their trading positions, that is their holding periods.
- `Day traders` hold their positions throughout the day, but usually not overnight. They tend to trade frequently.
- `Swing traders` hold their positions from a few days to several weeks.
-  `Position traders` holds positions from a few months to several years.

#### Resample the data
- Depending on the trading style, we may want to look at the time series data from different intervals such as hourly, daily , weekly etc.
- For e.g a swing trader would prefer to look at daily price snapshot instead of every hour.

```python
# resample hourly data to daily or weekly data
eurusd_daily = eurusd_h.resample('D').mean()
eurusd_daily = eurusd_h.resample('W').mean()
```

- Typically we downsample from a narrower time frame to a wider time frame, such as hourly to daily. This will result in fewer number of rows, and the sampled data of the wider time frame is the aggregated result of the lower time frame. It can also be the min, max or the sum instead of mean.

#### Calculate daily returns
- It is also helpful to get familiar with our trading data by checking past returns and volatility.
- We can use `pct_change()` method to calculate percentage change in the price, also known as price return. It computes the percentage change from the preceding row by default, so if we use daily price data, we will get daily returns.
- By plotting the results, we can obtain a quick understanding of typical return ranges and the volatility profile of a financial asset.

```python
# calculate daily returns
stock_data['daily_return'] = stock_data['Close'].pct_change() * 100
```

#### Plotting a histogram of daily returns
- It is helpful to plot a histogram of daily price returns. A histogram is a visual representation of the distribution of the underlying data. We can use bins to decide how granular we want it to be.

```python
stock_data['daily_return'].hist(bins=100)
plt.show()
```

#### Data transformation
- The financial market reflects fear, greed and human behavioral biases, hence the market data is inherently noisy and messy.
- To make sense of the data, traders perform various data transformations and create so-called technical indicators. A very common indicator is the simple moving average or SMA.
- It is simply the arithmetic mean of the price over a specified period.

```python
stock_data['sma_50'] = stock_data['Close'].rolling(window=50).mean()
```

#### Ploting the rolling average

```python
import matplotlib.pyplot as plt

plt.plot(stock_data['Close'], label='Close')
plt.plot(stock_data['sma_50'], label='SMA_50')
plt.legend()
plt.show()
```

### Financial trading with bt
- `bt` provides a flexible framework for defining and backtesting trading strategies in Python. A trading strategy is a method of buying and selling financial assests based on predefined rules.
- For technical trading, rules are usually based on technical indicators and signals. Backtesting is a way to assess the effectiveness of a strategy by testing it on historical data. The test result is evaluated to determine how it would have performed if used in the past, and whether it will be viable for future trading.
- There are 4 steps to define and backtest a strategy with bt.
- `1.` First we obtain historical price data of the assets we are going to trade
- `2.` Second we define the strategy
- `3.` Next we backtest the strategy with the historical data, and finally we evalaute the result.

#### Get the data
- **A ticker is an abbreviated identifier for a public traded stock**, and the "Adjusted Close" price is adjusted for events like corporate actions such as stock splits, dividends, etc.
- Prices of multiple securities can be downloaded at once by specifying multiple tickers within a single string separated by commas. Use start and end to specify the start date and end date.

```python
import bt

# download historical prices, use get to fetch data online directly
# by default, it downloads the "Adjusted Close" prices from Yahoo Finance by tickers

bt_data = bt.get('goog, amzn, tsla', start='2020-6-1', end='2020-12-1')
```

#### Define the strategy
- Next we define our strategy with `bt.Strategy`. The "Strategy" contains trading logics by combining various "algos".
- This unique feature of `bt` allows us to easily create strategies by mixing and matching different algos, each of which acts like a small task force that performs a specific operation.
- within strategy we first assign a name.Then we define a list of algos in the square brackets. The first algo specifies when to execute trades.Here we specify a simple rule to execute trades every week using "RunWeekly".
- The second algo specifies what data the strategy will apply to, for simplicity we apply to all the data using "SelectAll".
- The third algo specifies, in the case of multiple assets, what weights apply to each asset. Here "weightEqually" means for example, if we have two stocks, we will always allocate equal amounts of capital to each stock.
- The last algo specifies that it will re-balance the asset weights according to what we have specified in the previous step.
- **We now have a strategy that will execute trades weekly on a portfolio that holds several stocks.** It will sell a stock that has risen in price and redistribute the profit to buy a stock that has fallen in price, maintaining an equal amount of holdings in each stock.

```python
# define the strategy

bt_strategy = bt.Strategy('Trade_Weekly', 
                           [bt.algos.RunWeekly(), #Run weekly
                            bt.algos.SelectAll(), #use all data
                            bt.algos.WeighEqually(), # Maintain equal weights
                            bt.algos.Rebalance()])    # Rebalance
```

#### Backtest
- Now we can perform backtesting using `bt.Backtest` to combine the data and previously defined strategy, and create a "backtest".
- Call `bt.run()` to run the backtest and save the result.

```python
# create a backtest

bt_test = bt.Backtest(bt_strategy, bt_data)
```

#### Evaluate the result
- We can use `.plot` to plot and review the result. The line chart shows if we apply the strategy to trade Google, Amazon, Tesla stocks weekly, buy and sell them to maintain an equal weighted stock portfolio, in the six months during 2020 our portfolio will increase from 100 to 180.
- We can also use "get_underscore_transcations" to print out the transcation details.

```python
# plot the result
bt_res.plot(title='Backtest result')

# get trade results
bt_res.get_transcations()
```

### Technical Indicators

#### Trend Indicator MA
- A technical indicator is a calculation based on historical market data such as price, volumes etc. They are essential to technical analysis, which assumes that the market is efficient and prices have incorporated all public informtion such as financial news or public policies.
- Traders use technical indicators to gain insight into past price patterns and to anticipate possible future price movements.

#### Types of indicators
- **Trend indicators** : such as **Moving Average, Average Directional Movement Index** , measure the direction or strength of a trend.
- **Momentum indicators** : such as **Relative Strength Index (RSI)** , measure the velocity of price movement, that is the rate of change in an upward or downward direction. 
- **Volatilty indicators** : such as **Bollinger Bands** , measure the magnitude of price deviations.
                        
## The TA-Lib package 

### TA-Lib : Technical Analysis Library
- We can use the TA Lib package to implement technical indicators in python.
- TA Lib includes over 150 indicators and is very popular among technical traders.

#### Moving average indicators
- Simple moving average(SMA) and Exponential Moving average (EMA). They are called moving averages because every average value is calculated using data points of the most recent n periods, and hence moves along with the price.
- Calculating the averages creates a smoothing effect which helps to give a clearer indication of which direction the price is moving - upward, downward or sideways.
- Moving averages calculated based on a longer lookback period have more smooting effects than a shorter one.

#### Simple Moving Average (SMA)
- SMA is the arithmetic mean of the past n prices. N is the choosen number of periods for calculating the mean.
- With `talib` , we can simply call `talib.SMA` and pass the DataFrame column.

```python
import talib

# calculate two SMAs

stock_data['SMA_short'] = talib.SMA(stock_data['Close'], timeperiod=10)
stock_data['SMA_long'] = talib.SMA(stock_data['Close'], timeperiod=50)
```

#### Plotting the SMA

```python
import matplotlib.pyplot as plt

# plot SMA with the price
plt.plot(stock_data['SMA_short'], label='SMA_short')
plt.plot(stock_data['SMA_long'], label='SMA_long')
plt.plot(stock_data['Close'], label='Close')

# customize and show the plot
plt.legend()
plt.title('SMAs')
plt.show()
```

#### Exponential Moving Averge (EMA)
- EMA is the exponential moving average of the last n prices, where the weight decreases exponentially with each previous price.

```
EMAn = Pn * multiplier + previousEMA * (1 - multiplier)
multiplier = 2/(n+1)
```

```python
# caclculate two EMAs

stock_data['EMA_short'] = talib.EMA(stock_data['Close'], timeperiod=10)
stock_data['EMA_long'] = talib.EMA(stock_data['Close'], timeperiod=50)
```

#### SMA vs EMA
- The main difference between SMAs and EMAs is that EMAs give higher weight to the more recent data, while SMAs assign equal weight to all data points.

### Strength indicator: Average Directional Movement Index or ADX
- Its a Popular trend strength indicator. Measures the stength of a trend.
- ADX can indicate **whether an asset price is trending or merely moving sideways**.
- However, it does not tell the direction of a trend, that is bullish(rising prices) or bearish(falling prices)
- ADX oscillates between 0 and 100. In general,an ADX less than 25 indicates the market is going sideways and has no clear trend **(ADX <= 25: no trend)** 
- An ADX above 25 indicates the market is trending, and an ADX above 50 suggests a strong trending market.

#### How ADX is calculated?
- ADX is obtained using lengthly and complicated calculations. Simply put, ADX is derived from two other indicators: +DI and -DI.
- **+DI (Plus Directional Index)** : quantifies the presence of an uptrend 
- **-DI (minus directional index)** :  quantifies the presence of a downtrend.
- **ADX is the smoothed averages of the differnce between +DI and -DI**. The calculation input for ADX includes the high, low and close prices of each period.

#### Implementing ADX in Python
- Originally Welles Wilder used a 14-period lookback window for ADX calculations, which became the industry standard.
- The longer the lookback timeperiod window, the less sensitive the ADX is to price fluctuations. In other words, a 14-day ADX is more sensitive to daily price changes than a 21-day ADX.
- ADX starts to rise when the price is steadily trending up. The ADX starts to decline when the uptrend in price is stalling and price is moving sideways.

```python
# calculate ADX
stock_data['ADX'] = talib.ADX(stock_data['High'], stock_data['Low'], stock_data['Close'], timeperiod=14)
```

#### Plotting ADX
- Usually an ADX plot is placed horizontally under a price plot, so we can observe the price and indicator changes together along the same timeline.

```python
import matplotlib.pyplot as plt

# create subplots
fig, (ax1, ax2) = plt.subplots(2)

# plot the ADX with the price
ax1.set_ylabel('Price')
ax1.plot(stock_data['Close'])
ax2.set_ylabel('ADX')
ax2.plot(stock_data['ADX'])

ax1.set_title('Price and ADX'])
plt.show()
```

### Momentum indicator : RSI
- Momentum indicator : **Relative Strength Index or RSI**. RSI has been the most popular indicator used to measure momentum, which is the speed of rising or falling in prices. The RSI oscillates between 0 and 100.
- Traditionally an **RSI over 70 indicates an overbought market condition** , which means the **asset is overvalued and the price may reverse**.
- An **RSI below 30** suggests an **oversold market condition** , which means the **asset is undervalued** and the price may rally.

#### RSI calculation
- RS or Relative Strength, is the average of the upward price changes over a chosen n periods, divided by the average of downward price changes over those n periods. `RSI = 100 - 100/(1 + RS)
- The formula is constructed in such a way that an RSI is bounded between 0 and 100.

#### Implementing RSI in Python
- Similar to ADX, Welles Wilder used a 14-period lookback window for RSI calculations, which became the industry standard.
- Note the longer the lookback window, the less sensitive the RSI is to the price fluctuations. Traders may want to change the default time period to suit their specific trading time horizons or as a variable input for testing different trading strategies.
- When the RSI is falling near 30, the price bottoms out and gradually recovers, and when the RSI value is approaching 70, the price reaches new highs and is more likely to pull back. 

```python
# calculate RSI
stock_data['RSI'] = talib.RSI(stock_data['Close'], timeperiod=14)
```

### Volatility indicator: Bollinger Bands
- Developed by John Bollinger, a famous technical trader.
- Bollinger bands are designed to gauge price volatility, that is price deviations from the mean.
- **Bollinger Bands are composed of three lines** : 
- **a middle band** which is an **n-period simple moving average line**, where n=20 by default
- **upper and lower band that are drawn k standard deviations above and below the middle band**, where k equals 2 by default.
- Traders may choose the n and k to suit the trading time horizons and strategy needs. For example, a trader may choose 10-period moving average and 1.5 standard deviations for a shorter term strategy, or a 50-period moving average and 2.5 standard deviation for a longer-term strategy.

#### Bolinger Bands implications
- Since the upper and lower bands are calculated based on standard deviations from the mean price, they adjust to volatility swings in the underlying price. 
- **The wider the Bollinger Bands, the more volatile the asset prices**.
- In addition, Bollinger Bands intend to answer the question: **is the price too high or too low on a relative basis?**
- **Statistically speaking, if the upper and the lower bands are based on 1 std-dev, they contain about 68% of the recent price moves**. Similarly, if the bands are based on **2 standard devs, they contain about 95% of recent price moves**
- In other words, the price only moves out of the upper or lower bands in 5% of the cases. Hence we can say price is relatively extensive when it is close to the upper band, and relatively cheap when it is close to the lower band.

#### Implementing Bollinger Bands in Python
- `nbdevup` and `nbdevdb` specifies the number of std -devs away from the middle band for the upper and lower band respectively, which is 2 by default.

```python
# define the bollinger bands
upper, mid, lower = talib.BBANDS(stock_data['Close'],
                                 nbdevup=2,
                                 bndevdn=2,
                                 timeperiod=20)
```

#### Plotting Bollinger Bands
- The Bollinger bands becomes wider when the price has big upward or downward swings. When the green price data gets closer to the red upper or red lower band, it tends to reverse temporairly before continuing the original upward or downward movement.

```python
import matplotlib.pyplot as plt

# plot the bollinger bands
plt.plot(stock_data['Close'], label='Price')
plt.plot(upper, label='Upper band')
plt.plot(mid, label='Middle  band')
plt.plot(lower, label='Lower band')

# Customize and show the plot
plt.title('Bollinger Bands')
plt.legend()
plt.show()
```

### Trading Strategies
- Construct signals and use them to build trading strategies. Two main styles of trading strategies : trend following and mean reversion.Build and backtest more sophisticated trading strategies.

#### What are trading signals
- Trading signals are triggers to long(buy) or short(sell) financial assets based on predetermined criteria. They can be constructed using one technical indicator, multiple technical indicators or a combination of market data and indicators.
- Trading signals are commonly used in algorithmic trading, where trading strategies make decisions based on quantitative rules, and remove human discretion.

#### A signal example
- Signal : It is constructed by comparing the price with its n-period simple moving average or SMA. A long signal is triggered to buy the asset when its price rises above the SMA, and exit the long trade when its price drops the SMA.

#### How to implement signals in bt
- There is a 4 step process of defining and backtesting trading strategies.
- 1. Get the data and calculate indicators
- 2. Define the signal-based strategy
- 3. Create and run a backtest
- 4. Review the backtest result


- There are two main ways to implement signals in bt strategies. One way is to use algos `bt.algos.SelectWhere()`  to filter price levels for constructing the signal.
- Another way is to use algos WeighTarget `bt.algos.WeighTarget()`

#### Construct the signal
- Let's build a signal based strategy with bt step by step using the price and SMA based signal.
- First we need to obtain the price data and calculate the moving average indicator. Here we use `bt.get` to download the stock price data directly online.
- To calculate the SMA, we can apply `.rolling(20).mean()` to the price data. Alternatively we can use the `talib` library's SMA function.

```python
# get price data by the stock ticker
price_data = bt.get('aapl', start='2019-11-1', end='2020-12-1')

# calculate SMA
sma = price_data.rolling(20).mean()

# or
import talib
sma = talib.SMA(price_data['Close'], timeperiod=20)
```

#### Define a signal-based strategy
- This is handled by algos SelectWhere. It takes the argument price large than SMA, which essentially is a Boolean DataFrame containing selection logic.
- If the condition is true, that is the price rises above the SMA, a long signal is triggered to enter long positions of the asset.
- For simplifications to the strategy we assume: First we will use the strategy for trading one asset, or one stock at a time. When we are trading multiple stocks or assets, their price correlations are important to consider for proper position sizing and asset allocation.
- Another simplication we make is to assume there is no slippage or commission in the trade execution. **`Slippage` is the difference between the expected price of a trade and the price at which the trade is executed, and often occurs when there is a supply and demand imbalance**.
- Commissions are fees charged by brokers when executing a trade. These are practical considerations in real trading, here we are focusing on basics.

```python
# Define the signal-based strategy

bt_strategy = bt.Strategy('AboveEMA',
                          [bt.algos.SelectWhere(price_data > sma),
                          bt.algos.WeighEqually(),
                          bt.algos.Rebalance()])
```

#### Backtest the signal based strategy
- The line chart shows how much the beginning capital balance increases over time from a baseline of 100. Note that flat line areas indicate **periods when we don't have any positions**, so the trading account balance does not change.
- Overall, the strategy is profitable based on the backtest performed on the historical data.

```python
# create the backtest and run it
bt_backtest = bt.Backtest(bt_strategy, price_data)
bt_result = bt.run(bt_backtest)

# plot the backtest result
bt_result.plot(title='Backtest result')
```

#### Trend-following strategies
- The two most popular types of trading strategies are trend following and mean reversion.
- Trend following also known as a momentum strategy, bets that the price trend will continue in the same direction. When the price rises above its moving average, enter a long position to bet the price will continue to rise.
- Traders commonly use `trend indicators` such as moving averages, ADX etc to construct trading signals for `trend following strategies`. 
- `Mean reversion strategies`, conversely, bet that when the market reaches an **overbought or oversold condition, the price tends to reverse back towards the mean**.
- Traders commonly use indicators such as **RSI, Bollinger Bands etc, to construct trading signals for mean-reversion strategies**.
- Markets are constantly **moving in and out of phases of trending and mean reversion**. Therefore it's beneficial to develop strategies for both phases.

#### MA crossover strategy
- MA crossover strategy, which involves two moving average indicators, one longer and one shorter. Consider EMA for example
- When the short-term EMA crosses the long-term EMA, it is a long signal, as it suggests the price is picking up a momentum.
- When the short-term EMA crosses below the long-term EMA, it's a short signal, as it suggests that the price is losing momentum.

#### Calculate the indicators
- We set the signal value to 0 for the initial n periods that do not have enough data points for the EMA. Then we define a signal.
- When the short-term EMA value is larger than the long-term EMA value, the signal is one indicating a long position, when the short term EMA is smaller than the long-term EMA, the signal is -1 indicating a short position.
- Note that shorting a stock essentially means betting the price will go down, and entails selling borrowed shares, and later buying them back at market price.

```python
import talib

# calculate the indicators
EMA_short = talib.EMA(price_data['Close'], timeperiod=10).to_frame()
EMA_long = talib.EMA(price_data['Close'], timeperiod=40).to_frame()

# construct the signal
# create the signal dataframe

signal = EMA_long.copy()
signal[EMA_long.isnull()] = 0

# construct the signal
signal[EMA_short > EMA_long] = 1
signal[EMA_short < EMA_long] = -1
```

#### Plot the signal
- We can plot the signal with the price and EMA indicators together.
- Use `.columns` attribute of the DataFrame to rename the data columns and create a plot
- Since the signal has a different scale from the price and EMA data, define `secondary_y` to the signal column to plot it on secondary y axis on the right.
- The chart gives a clear indication of where to take long or short positions.

```python
# plot the signal, prices and MAs

combined_df = bt.merge(signal, price_data, EMA_short, EMA_long)
combined_df.columns = ['Signal', 'Price', 'EMA_short', 'EMA_long']
combined_df.plot(secondary_y=['Signal'])
```

#### Define the strategy with the signal
- The signal value 1, -1, 0 will dictate which period we will have long positions, short positions, or no positions.

```python
# define strategy

bt_strategy = bt.Strategy('EMA_crossover', [bt.algos.WeighTarget(signal),
                        bt.algos.Rebalance()])
```

#### Backtest the signal based strategy

```python
# create the backtest and run it
bt_backtest = bt.Backtest(bt_strategy, price_data)
bt_result = bt.run(bt_backtest)

# plot the backtest result
bt_result.plot(title='Backtest result')
```

### Strategy optimization and benchmarking

#### How to decide the values of input parameters?
- When we construct the signal based on the price and SMA comparison, what is the SMA lookback period we should use? Can a 10,20 or 50 period SMA result in a more profitable strategy?
- The solution is to conduct strategy optimization, which is the process of testing a range of input parameter values to find the ones that give better strategy performance based on historical data.

#### Strategy optimization example
- To find the SMA lookback period that can optimize the strategy performance,we want to run multiple backtests by varying the SMA timeperiod parameters on different runs.
- Pass period as parameter to change the SMA lookback period. In addition, we can pass the stock ticker, start date and end date to run backtests on different stocks and on different historical periods.

```python
def signal_strategy(ticker, period, name, start='2018-4-1', end='2020-11-1'):
    # get the data and calculate SMA
    price_data = bt.get(ticker, start=start, end=end)
    sma = price_data.rolling(period).mean()
    # define the signal-based strategy
    bt_strategy = bt.Strategy(name,
                              [bt.algos.SelectWhere(price_data>sma),
                              bt.algos.WeighEqually(),
                              bt.algos.Rebalance()])
    # return the backtest 
    return bt.Backtest(bt_strategy, price_data)
```

- Call the function several times to pass different SMA lookback parameters
- Each function will return a `bt.Backtest` . Call `bt.run` to run all the Backtests at once, and plot the results in one chart for easy comparison.

```python
ticker = 'aapl'
sma20 = signal_strategy(ticker, period=20, name='SMA20')
sma50 = signal_strategy(ticker, period=50, name='SMA50')
sma100 = signal_strategy(ticker, period=100, name='SMA100')

# run backtests and compare results
bt_results = bt.run(sma20, sma50, sma100)
bt_results.plot(title='Strategy optimization')
```

#### Benchmark
- A benchmark is a standard or point of reference against which a strategy can be compared or assessed.
- For example, a strategy that uses signals to actively trade stocks can use a passive buy and hold strategy as a benchmark. A benchmark can also be chosen based on the market segments and asset risk profiles.
- For example, the S&P 500 index is often used as a benchmark for US equity performance, and US Treasuries are used for measuring bond risks and returns.

#### Benchmarking example
- Instead of using a signal-based strategy to actively trade a stock, we will passively hold the stock and use its performance as the benchmark.

```python
def buy_and_hold(ticker, name, start='2018-11-1', end='2020-12-1'):
    # get the data
    price_data = bt.get(ticker, start=start_date, end=end_date)
    # define the benchmark strategy
    bt_strategy = bt.Strategy(name,
                            [bt.algos.RunOnce(),
                            bt.algos.SelectAll(),
                            bt.algos.WeighEqually(),
                            bt.algos.Rebalance()])
    # return the backtest
    return bt.Backtest(bt_strategy, price_data)
```

- Define a separate function to define a benchmark. Use `bt.algos.RunOnce()` to implement a passive strategy, that is buy a stock at the beginning of the period and just hold until the end of the period.

```
benchmark = buy_and_hold(ticker, name='benchmark')
# run all backtests and plot the results
bt_results = bt.run(sma20, sma50, sma100, benchmark)
bt_results.plot(title='Strategy benchmarking')
```


















