
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
                            bt.algos.WeightEqually(), # Maintain equal weights
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



                        











