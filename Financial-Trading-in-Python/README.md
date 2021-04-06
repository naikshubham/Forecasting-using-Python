
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





