from statsmodels.tsa.stattools import adfuller
import pandas as pd

earthquake = pd.read_csv('earthquakes_per_year.csv')
results = adfuller(earthquake['earthquakes_per_year'])

# print test statistic
print(result[0])

# print p-value
print(result[1])

# print critical values
print(result[4])

# Run ADF test on the 'city_population' column of city

city = pd.read_csv('city.csv')
result = adfuller(city['city_population'])

# plot the time series
fig, ax = plt.subplots()
city.plot(ax=ax)
plt.show()

# Print the test statistics and p-value
print("ADF Statistic :', result[0])
print("p-value:', result[1])

# take the first difference of city dropping the NaN values. Assign this to city_stationary and run the test again
city_stationary = city.diff().dropna()

# Run ADF test on the differenced time series
result = adfuller(city_stationary['city_population'])

# plot the differenced time series
fig, ax = plt.subplots()
city_stationary.plot(ax=ax)
plt.show()

print("ADF Statistic :", result[0])
print("p-value :", result[1])

# calculate the second difference of the time series
city_stationary = city.diff().dropna().diff().dropna()

# Run ADF test on the differenced time series
result = adfuller(city_stationary['city_population'])

# plot the differenced time series
fig, ax = plt.subplots()
city_stationary.plot(ax=ax)
plt.show()

# print the test statistic and p-value
print("ADF statistic:', result[0])
print("p-value:', result[1])

# A p-value of 0.000000016 is very significant! This time series is now stationary and ready for modelling.

# Compare differencing and log return transform
amazon = pd.read_csv('amazon.csv')
amazon_diff = amazon.diff().dropna()

# Run the test and print
result_diff = adfuller(amazon_diff['close'])
print(result_diff)

# Calculate log-return and drop nans
amazon_log = np.log(amazon/amazon.shift(1)).dropna()

#Run test and print
result_log = adfuller(amazon_log['close'])
print(result_log)


## Notice that both the differenced and the log-return transformed time series have a small p-value, but the log transformed time series has a much more negative test statistic. This means the log-return tranformation is better.





