# Forecasting-using-ARIMA-Models-in-Python
Used ARIMA class models to forecast the future.


Time series data are everywhere 
- Science
- Technolgy
- Business
- Finance 
- Policy

### Trend
A positive trend generally slopes up, the values increase with time.Similarly negative trend is when the values decrease.

### Seasonality
A seasonal time series has patterns that repeat at regular intervals. eg. High sales every weekend.

### Cyclicality
In contrast, cyclicality has repeating patterns but no fix period.

### White Noise
White Noise is an important concept in time series and ARIMA models. White Noise is a series of values where each value is **uncorrelated** with the previous value.
e.g Flipping a coin : The outcome of coin flipping does'nt depend on outcome of coin flipping that came before.

### Stationarity
To model a time-series it must be stationary. Stationary means distribution of data doesn't change with time. For time series to be stationary it must fulfill 3 criterias:

- Trend Stationary : Trend is zero (series has zero trend, it is'nt growing and shrinking)
- Variance is constant : Average distance of the data point from zero line is'nt changing.
- Autocorrelation is constant : each value in time series related to its neighbours stays the same.

### Train-test split
- Generally in ML we have a training set on which you train your model on and a test set on which you test the model.
- In Time series its the same. The train-test split is different however, we train on the data earlier in the time series and test on the data coming later.































- 