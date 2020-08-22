
# Time Series Analysis

### Some useful Pandas Tools
- Changing an index to datetime
`df.index = pd.to_datetime(df.index)`

- Plotting data
`df.plot()`

- Join two DataFrames
`df1.join(df2)`

- Often, we will want to convert prices to returns, which we can do with the `pct_change` method. Or if we want differences we can use the `diff` method.

```python
df['col'].pct_change()
df['col'].diff()
```

- We can compute the correlation of two series using the `corr` method and the autocorrelation using the `autocorr` method.
`df['ABC'].corr(df['XYZ'])`

### Correlation of Two Time Series
- A scatter plot helps to visualize the relationship between two time series. The **correlation coefficient** is the measure of how much two series vary together.
- A correlation of 1 means that the two series have a perfect linear relationship with no deviations. High correlations means that the two series strongly vary together. A low correlation means that they vary together but there is a weak association.
- And a high negative correlation means they vary in opposite directions, but still with a linear relationship.

#### Common mistake : Correlation of Two trending Series
- Consider two time series that are both trending. Even if the two series are totally unrelated, we could still get a very high correlation. That's why, when we look at the correlation of two stocks, we should look at the correlation of their **returns** not their levels.

#### Example : Correlation of Large Cap and Small Cap stocks
- Computing the correlation of two financial time series, the S&P500 index of large cap stocks and the Russell 2000 index of small cap stocks, using pandas correlation method.

1. First compute the percentage changes of both series. This gives the **returns of the series instead of prices**

```python
df['SPX_Ret'] = df['SPX_Prices'].pct_change()
df['R2000_Ret'] = df['R2000_Prices'].pct_changes()
```

2. Visualize the correlation using the scatter plot

```python
plt.scatter(df['SPX_Ret'], df['R2000_Ret'])
plt.show()
```

3. Use pandas correlation method for series

```python
correlation = df['SPX_Ret'].corr(df['R2000_Ret'])
print('Correlation is :", correlation)
```

### Simple Linear Regressions
- A simple linear regression finds the slope beta and intercept alpha of a line that's a best fit between a dependent variable y and an independent variable x.
- The x's and y's can be two time series. A LR is also known as Ordinary Least squares or OLS, because it minimizes the sum of the squared distances between the data points and the regression line.

#### Python packages to perform regression
- Regression techniques are very common and therefore, there are many packages in python that can be used.
- In statsmodels

```python
import statsmodels.api as sm
sm.OLS(y, x).fit()
```

- In numpy

```python
np.polyfit(x, y, deg=1) # if deg=1, it fits to data to a line which is LR
```

- In pandas

```python
pd.ols(y,x)
```

- In scipy

```python
from scipy import stats
stats.linregress(x, y)
```

#### Example : Regression of Small Cap Returns on Large Cap
- Regress the returns of the small cap stocks on the returns of large cap stocks.
- We need to add a column of ones as a dependent, right hand side variable. The reason we have to do this is bcz the regression function assumes that if there is no constant column, then we want to run the regression without an intercept.
- **By adding a column of ones, statsmodels will compute the regression coefficeint of that column as well, which can be interpreted as the intercept of the line.**
- The statsmodels method **add_constant** is a simple way to add a constant.

```python
import statsmodels.api as sm

# as before, compute percentage changes in both series
df['SPX_ret'] = df['SPX_Prices'].pct_change()
df['R2000_Ret'] = df['R2000_Prices'].pct_change()

# add a constant to the dataframe for the regression intercept
df = sm.add_constant(df)

# Delete the rows of NaN
df = df.dropna()

# run the regression
results = sm.OLS(df['R2000_Ret'], df[['const', 'SPX_Ret']]).fit()
print(results.summary())
```

- The first argument of the statsmodel regression is the series that represents the 
**dependent variable y** , and the next argument contains the **independent variable** or variables.
- In this case, **the dependent variable is the R2000 returns and the independent variables are the constant and SPX returns**. The **method "fit" runs the regression** and results are saved in a class instance called results.
- The summary method of results shows the entire regression output. The **coef 1.1412 is the slope of the regression**, which is also referred to as beta. The **coef above that is the intercept**, which is very close to zero.
- Intercept : results.params[0], slope : results.params[1]
- Another stats is the R squared of 0.753.

#### Relationship between R-squared and Correlation
- From the scatter diagrams, we saw that the correlation measures how closely the data are clustered along a line. The **R-Squared** also measures **how well the linear regression line fits the data**.
- So as we would expect, there is a relationship between correlation and R-squared.
- The magnitude of the correlation is the square root of the R-squared. **[corr(x,y)]^2 = R^2 (or R-squared)**.
- The **sign of the correlation is the sign of the slope of the regression line**. **sign(corr) = sign(regression slope)**
- If the regression line is positively sloped, the correlation is positive and if the regression line is negatively sloped, the correlation is negative.

### Autocorrelation
- So far,we have looked at the correlation of two time series. Autocorrelation is the correlation of a single time series with a lagged copy of itself. It's also called "serial correlation".
- Often, when we refer to a series's autocorrelation, we mean the "lag-one" autocorrelation. So when using daily data, for e.g, the autocorrelation would be the correlation of the series with the same series lagged by one day.

#### Interpretation of autocorrelation
- What does it mean when a series has a positive or negative autocorrelation? With financial time series, when returns have a negative autocorrelation, we say it is "mean reverting". Alternatively, if a series has a +ve autocorrelation, we say it is "trend-following".

#### Traders Use Autocorrelation to Make Money
- Many hedge fund strategies are only slightly more complex versions of mean reversion and momentum strategies.
- Since stocks have historically has negative autocorrelation over horizons of about a week, one popular strategy is to buy stocks that have dropped over the last week and sell stocks that have gone up.

#### example of positive autiocorrelation : exchange rates
- Example of how we would compute the monthly autocorrelation for the Japanese Yen-US Dollar exchange rate.

```python
# convert index to datetime
df.index = pd.to_datetime(df.index)

# downsample from daily to monthly data
df = df.resample(rule='M', how='last') # rule indicates desired freq :'M' for monthly, how indicates how to do resampling, we can use the first date of the period, the last date or even an average

# compute returns from prices
df['Return'] = df['Price'].pct_change()

# compute autocorrelation
autocorrelation = df['Return'].autocorr()
print("Autocorrelation:", autocorrelation)
```

### Autocorrelation Function, ACF
- The sample autocorrelation function, or ACF, shows not only the lag-one autocorrelation, but the entire autocorrelation function for different lags.
- **Any significant non-zero autocorrelations** implies that the series can be forecast from the past.

#### ACF Example 1: Simple Autocorrelation Function
- This autocorrelation function implies that we can forecast the next value of the series from the last two values, since the lag-one and lag-two autocorrelations differ from zero.

<img src="./data/ac_1.JPG" width="350" title="ACF">

#### ACF Example 2 : Seasonal Earnings

<img src="./data/acf_2.JPG" width="350" title="ACF">

- Consider the time series of quarterly earnings of the company H&R Block. A vast majority of earnings occurs in the quarter that taxes are due.In this case, we can clearly see a seasonal pattern in the quartely data on the left, and the autocorrelation function on the right shows strong autocorrelation at lags 4, 8, 12, 16 and 20.

#### ACF Example 3: Useful for Model Selection

<img src="./data/acf_3.JPG" width="350" title="ACF">

- **Model selection** : The ACF can also be useful for selecting a parsimonious model for fitting the data. In this example, the pattern of autocorrelation suggests a model for the series.

#### Plot ACF in Python
- `plot_acf` is the statsmodels function for plotting the autocorrelation function.
- The input x is a series or array. The argument **lags** indicates how many lags of the autocorrelation function will be plotted. The **alpha argument sets the width of the confidence interval**.

```python
from statsmodels.graphics.tsaplots import plot_acf

# plot the acf
plot_acf(x, lags=20, aplha=0.05)
```

#### Confidence Interval of ACF
- Argument `alpha` sets the width of confidence interval. For e.g if alpha=0.05, that means that if the true autocorrelation at that lag is zero, there is only a 5% chance the sample autocorrelation will fall outside that window.
- We will get a wider confidence interval if we set alpha lower, or if we have fewer observations. If we don't want to see confidence intervals in our plot, set alpha = 1.

#### ACF values instead of Plot
- Besides plotting the ACF, we can also extract its numerical values using a similar Python function, `acf`, instead of plot_acf.

```python
from statsmodels.tsa.stattools import acf
print(acf(x))
```

### White Noise
- White Noise is a series with : **constant mean, constant variance, zero autocorrelations at all lags**
- There are several special cases of white noise: If the data is white noise but also has a normal or Gaussian distribution, then it is called Gaussian White Noise.

#### Simulating white noise
- Its very easy to generate white noise. 

```python
import numpy as np
noise = np.random.normal(loc=0, scale=1, size=500) # loc=mean, scale=std-dev

plt.plot(noise)
```

#### Autocorrelation of White Noise
- All the autocorrelations of a white noise series are zero.

```python
plot_acf(noise, lags=50)
```

#### Stock Market Returns : Close to White Noise
- Autocorrelation Function for the S&P500. There are no lags where the autocorrelation is significantly different from zero.

### Random Walk
- In a random walk, todays price = yesterdays price + some noise. (pt = pt-1 + $t)
- THe change in price of a random walk is just white noise. Incidentally, if prices are in logs, then the difference in log prices is one way to measure returns.
- The bottom line is that if stock prices follow a random walk, then stock returns are White Noise.
- **We can't forecast a random walk**. The best guess for tmrw's price is simply tdy's price.
- In a random walk with drift, prices on average drift by `mu` every period.And the change in price for a random walk with drift is still white noise but with a mean of mu.

#### Statistical Test for Random Walk
- To test whether a series like stock prices follows a random walk, we can regress current prices on lagged prices. **If the slope coefficent beta, is not significantly different from one**, then we cannot reject the null hypothesis that the series is a random walk.
- However, if the slope coefficient is significantly less than one, then we can reject the null hypothesis that the series is a random walk. 
- An identical way to do that test is to regress the difference in prices on the lagged price, and instead of testing whether the slope coefficent is 1, now we test whether it is zero. This is called the **Dickey-Fuller** test.
- If we add more lagged prices on the right hand side, then it's called the **Augmented Dickey-Fuller test.**

#### ADF Test in Python
- statsmodels has a function, adfuller, for performing the Augmented Dickey-Fuller Test.

```python
from statsmodels.tsa.stattools import adfuller

# run augmented dickey test
adfuller(x)
```

#### Example : IS the S&P500 a Random Walk?
- The main output we're interested in is the **p-value of the test**.
- If the p-value is less than 5%, we can reject the null hypothesis that the series is a random walk with 95% confidence.
- In this case, the p-value is much higher than 0.5, its 0.78. Hence, we cannot reject the null hypothesis that S&P500 is a random walk.

```python
# run augmented dickey-fuller test on spx data
results = adfuller(df['SPX'])

# print p-value
print(results[1])

# print full results
print(results)
```

### Stationarity
- **Strong stationarity** : entire distribution of data is time-invariant. It means that the joint distribution of the observations do not depend on time.
- **Weak stationarity** : Mean, variance and autocorrelation are time-invariant (i.e for autocorrelation, corr(Xt, Xt-T) is only a function of T). A less restrictive version of stationarity, and one that is easier to test. Mean, variance and autocorrelations of the oberservations do not depend on time.
- For autocorrelation, the correlation between X-t and X-(t-tau) is only a function of the lag tau, and not a function of time.

#### Why do we care for stationary
- If a process is not stationary, then it becomes difficult to model. 
- Modeling involves estimating a set of parameters, and if a process is not stationary, and the parameters are different at each point in time, then there are too many parameters to estimate. We may end up having more parameters than actual data.If parameters vary with time, too many parameters to estimate.
- So stationarity is necessary for a parsimonious model, one with a smaller set of parameters to estimate. Can only estimate a parsimonious model with a few parameters.

#### Examples of Nonstationary series
- A random walk is a common type of non-stationary series. The variance grows with time. For e.g, if stock prices are a random walk, then the uncertainty about prices tomorrow is much less than the uncertainty 10 years from now.
- Seasonal series are also non-stationary.

#### Transforming Non stationary Series into Stationary Series
- Many non-stationary series can be made stationary through a simple transformation. A random-walk is a non-stationary series, but if we take the first differences, the new series is White Noise, which is stationary.

```python
# random walk
plot.plot(SPY)

# first difference
plot.plot(SPY.diff())
```

- Quarterly earnings for H&R Block, which has a large seasonal component and is therefore not stationary. If we take the `seasonal difference`, by taking the difference with lag of 4, the transformed series looks stationary.

```python
plot.plot(HRB.diff(4))
```

- Sometimes we many need to do 2 transformations. If we see amazon quarterly revenues, its growing exponentially as well as exhibiting a string seasonal pattern.































































