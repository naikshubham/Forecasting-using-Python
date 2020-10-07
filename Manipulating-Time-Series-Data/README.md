
### Manipulating time series data

#### Basic building block : pd.Timestamp

```python
import pandas as pd
from datetime import datetime
time_stamp = pd.Timestamp(datetime(2017, 1 ,1))
pd.Timestamp('2017-01-01') == time_stamp

time_stamp.year
time_stamp.weekday_name
```

#### More building blocks : pd.Period & freq
- pandas also has a data type for time periods.The period object always has a frequency, with months as the default.
- It also has a method to convert between frequencies, for instance from monthly to daily frequencies. Period object has freq attribute to store frequency info
- We can a period to a timestamp object, and a timestamp back to a period object.

```python
period = pd.Period('2017-02')
period # default : month-end

period.asfreq('D') # convert to daily

period.to_timestamp().to_period('M')
```

#### Basic date arithmetic
- Starting with a period object for January 2017 at monthly frequency, just add the number 2 to get a monthly period for March 2017.
- Timestamps can also have frequency information. If we create the timestamp for Jan 31 2017 with monthly frequency and add 1, we get a timestamp for Feb 28th.

```python
period + 2

pd.Timestamp('2017-02-28 00:00:00', freq='M')
```

#### Sequences of  dates & times
- To create a time series we need a sequence of dates. To create a sequence of Timestamps, use the pandas function `pd.date_range` with `start, end, periods, freq`. The default is daily freq.
- The function returns the sequence of dates as a DateTimeindex with frequency information.
- We can convert the index to a PeriodIndex, just like we could Timestamps to Period objects.
- Now we can create a time series by setting the DateTimeIndex as the index of our Dataframe

```python
index = pd.date_range(start='2017-1-1', periods=12, freq='M')

index[0]

index.to_period()
```

#### Create a time series : pd.DateTimeIndex
- Now we can create a time series by setting the DateTimeIndex as the index of our DataFrame.
- DataFrame columns containing dates will be assigned the datetime64 data type, where 'ns' means nanoseconds.

```python
pd.DataFrame({'data':index}).info()

data = np.random.random((size=12, 2))
pd.DataFrame(data=data, index=index).info()
```

<p align="center">
  <img src="../images/frequency.JPG" width="350" title="frequency">
</p>


### Indexing & resampling time series
#### Time series transformation
- Parsing string dates and converting to datetime64
- Selecting subperiods of our time series and setting or changing the frequency of the DateTimeIndex. We can change the frequency to a higher or lower value: upsampling involves increasing the time frequency, which requires generating new data.Downsampling means decreasing the time frequency, which requires aggregating data.

#### Getting GOOG stock prices

```python
google = pd.read_csv('google.csv')
google.info()
```

#### Converting string dates to datetime64
- `pd.to_datetime()`
- `.set_index()`
- `inplace()` : Don't create copy
- The resulting DateTimeIndex lets us treat the entire dataframe as time series data.

```python
google.data = pd.to_datetime(google.data)
google.set_index('date', inplace=True)
```

#### Plotting time series

```python
google.price.plot(title='Google stock price')
plt.tight_layout()
plt.show()
```

#### Partial string indexing
- We can use strings that represent the complete date, or relevant parts of a date.
- If we just pass a string representing a year, pandas returns all dates within this year.
- If we pass a slice that starts with one month and ends at another, we get all dates within that range.

```python
google['2015'].info() # pass string for part of date
google['2015-3':'2016-2'].info() #slice includes last month
google.loc['2016-6-1', 'price'] # use full date with .loc
```

- We can also use `.loc` with a complete date and a column label to select a specific stock price.

#### .asfreq() : set frequency
- We can set the freq info using `.asfreq`. As a result, the DateTimeIndex now contains many dates where stock wasn't bought or sold.
- We can also convert the DateTimeIndex to business day freq. Pandas has a list of days commonly considered business days.

```python
google.asfreq('D').info() # set calendar day freq
google = google.asfreq('B') # change to calendar day freq
```














