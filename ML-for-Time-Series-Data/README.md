## ML for Time Series

### A Machine Learning Pipeline
- Feature extraction : what kinds of special features leverage a signal that changes over time?
- Model fitting : what kinds of models are suitable for asking questions with timeseries data?
- Prediction and Validation : How can we validate a model that uses timeseries data? What considerations must we make because it changes in time?

### ML and time series data

#### The Heartbeat Acoustic Data
- Audio is a very common kind of timeseries data. Audio tends to have a very high sampling frequency(often above 20,000 samples per second!)
- Dataset is audio data recorded from the hearts of medical patients. A subset of these patients have heart abnormalities. Can we use only this heartbeat data to detect which subjects have abnormalities?

#### Loading Audio data
- Audio data is often stored in wav files. Each of the files in the dataset contains the audio data for one heartbeat session, as well as the sampling rate for that data.
- Librosa to read audio dataset. It has functions for extracting features, vizualizations and analysis for audio data.
- Data is sored in audio and sampling freq is stored in sfreq. If the sampling freq is 2205, it means there are 2205 samples recorded per second.

```python
from glob import glob
files = glob('data/*.wav')

import librosa as lr
audio, sfreq = lr.load('data/file.wav')
```

#### Inferring time from samples
- **Using only the sampling freq, we can infer the time point of each datapoint in the audio file, relative to the start of the file.**

#### Creating array of timestamps
- To do so, we have two options. The first is to generate a range of indices from zero to the number of datapoints in our audio file, divide each index by the sampling freq, and we have a timepoint for each data point. 
- The second option is to calculate the final timepoint of our audio data using a similar method. Then, use the linspace function to generate evenly-spaced numbers between 0 and the final timepoint.
- In either case, we should have an array of numbers of the same length as our audio data.

```python
# option 1
indices = np.arange(0, len(audio))
time = indices / sfreq

# option 2
final_time = (len(audio) - 1) / sfreq
time = np.linspace(0, final_time, sfreq)
```

#### The New York Stock Exchange dataset
- It runs over a much longer timespan than audio data, and has a sampling freq on the order of one sample per day(compared with 2205 samples per second with the audio data)
- Our goal is to predict the stock value of a company using historical data from the market. As we are predicting a continuos output value,this is a regression problem.

#### Converting a column to a time series
- To ensure that a column within a DataFrame is treated as time series, use the `to_datetime()` function.

```python
df['date'] = pd.to_datetime(df['date'])
```

### Classifying a time series

#### Classification and feature engineering
- Always visualize raw data before fitting models. There is lot of complexity in any machine learning step, and visualizing raw data is important to make sure we know where to begin.
- To plot raw audio, we need 2 things: the raw audio waveform, usually in a 1-d or 2d array. We also need a time point of each sample

```python
ixs = np.arange(audio.shape[-1])
time = ixs / sfreq
fig, ax = plt.subplots()
ax.plot(time, audio)
```

#### What features to use
- Using raw data as input to a classifier is usually too noisy to be useful. An easy first step is to calculate summary statistics of our data, which removes the "time" dimension and gives us a more traditional classification dataset.
- For each time series, we calculate several summary statistics. These then can be used as features for a model. This way we can expand a single feature (raw audio amplitude) to several features (here, the min, max, and average of each sample).

#### Calculating multiple features
- Calculate multiple features for several timeseries. By using the **axis=-1** , we collapse across the last dimension, which is time. The result is an array of numbers, one per timeseries.
- We collapsed a 2-d array into a 1-d array for each feature of interest. We can then combine these as inputs to a model. In the case of classification, we also need a label for each timeseries that allows us to build a classifier.

```python
means = np.mean(audio, axis=-1)
maxs = np.max(audio, axis=-1)
stds = np.std(audio, axis=-1)

print(means.shape)
```

#### Preparing features for scikit learn
- For scikit learn ensure that data has correct shape, which is samples by features.Here we can use the column_stack function, which lets us stack 1-d arrays by turning them into the columns of a 2-d array.
- In addition, the labels array is 1-d, so we need to reshape it so that it is 2-d.

```python
from sklearn.svm import LinearSVC

# reshape to 2-d to work with scikit-learn
X = np.column_stack([means, maxs, stds])
y = labels.reshape([-1, 1])
model = LinearSVC()
model.fit(X, y)
```

#### Scoring scikit-learn model

```python
from sklearn.metrics import accuracy_score

# different input data
predictions = model.predict(X_test)

# score our model with % correct manually
percent_score = sum(predictions == labels_test) / len(labels_test)

# using a sklearn scorer
percent_score = accuracy_score(labels_test, predictions)
```

### Improving features for classification
- Some features that are more unique to timeseries data. Calculate the **envelope** of each heartbeat sound.
- The envelope throws away information about the fine-grained changes in the signal, focusing on the general shape of the audio waveform. To do this we'll need to calculate the audio's amplitude, then smooth it over time.

#### Smoothing over time
- We can remove noise in timeseries data by smoothing it with a rolling window. Instead of averaging over all time, we can do a local average.
- This is called smooting timeseries. This means defining a window around each timepoint, calculating the mean of this window, and then repeating this for each timepoint.

```python
# Calculating a rolling window statistic
window_size=50 # larger the window smoother the result

windowed = audio.rolling(window=window_size)
audio_smooth = windowed.mean()
```

#### Calculating the auditory envelope
- First rectify audio, then smooth it. Calculate the "absolute value" of each timepoint. This is also called "rectification", because we ensure that all time points are positive.
- Next, we calculate a rolling mean to smooth the signal.

```python
audio_rectified = audio.apply(np.abs)
audio_envelope = audio_rectified.rolling(50).mean()
```

##### Feature engineering the envelope
- Once we've calculated the acoustic envelope, we can create better features for our classifier. Here we calculate several common statistics of each auditory envelope, and calculate them in a way that scikit-learn can use.
- Even though we're calculating the same statistics(avg, std-dev and max), they are on different features, and so have different information about the stimulus.

```python
# calculate several features of the envelope, one per sound
envelope_mean = np.mean(audio_envelope, axis=0)
envelope_std = np.std(audio_envelope, axis=0)
envelope_max = np.max(audio_envelope, axis=0)

# create training data for a classifier
X = np.column_stack([envelope_mean, envelope_std, envelope_max])
y = labels.reshape([-1, 1])
```

##### Using cross_val_score

```python
from sklearn.model_selection import cross_val_score

model = LinearSVC()
scores = cross_val_score(model, X, y, cv=3)
print(scores)
```

#### Auditory features : The Tempogram
- There are several more advanced features that can be calculated with timeseries data. Each attempts to detect particular patterns over time, and summarize them statistically.
- For example, a tempogram tells us the tempo of the sound at each moment.

#### Computing the tempogram
- It tells us the moment by moment tempo of the sound. We can then use this to calculate features for our classifier.

```python
# calculate the tempo of 1-d sound array
import librosa as lr
audio_tempo = lr.beat.tempo(audio, sr=sfreq, hop_length=2**6, aggregate=None)
```

### The spectrogram - spectral changes to sound over time

#### Fourier transforms
- Spectograms are common in timeseries analysis. Key part of the spectrograms- the fourier transform.
- This approach summarizes a time series as a collection of fast-and-slow moving waves.
- The Fourier Transform or FFT is a way to tell us how these waves can be combined in different amounts to create our time series. It describes for a window of time, the presence of fast-and-slow-oscillations that are present in a timeseries.
- **The slower oscillations are on the left (closer to 0) and the faster oscillations are on the right**. This is a more rich representation of the audio signal.

### Spectrograms : combination of windows Fourier transforms
- A spectrogram is a collection of windowed Fourier transforms over time.We can calculate multiple fourier transforms in a sliding window to see how it changes over time.
- For each timepoint, we take a window of time around it, calculate a fourier transform for the window, then slide to the next window(similar to calculating the rolling mean).

1. Choose a window size and shape
2. At a timepoint, calculate the FFT for that window
3. Slide the window over by one
4. Aggregate the results
5. Its called a Short-Time Fourier Transform(STFT)

- To calculate the spectogram, we squar each value of the STFT.Note how the spectral content of the sound changes over time.
- Because it is speech, we can see interesting patterns that corresponds to spoken words(e.g vowels or consonants)

#### Calculating the STFT
- Calculate the STFT of the audio file, then convert the output to decibels to visualize it more cleanly with specshow (which results in the visualized spectograms). 

```python
from librosa.core import stft, amplitude_to_db
from librosa.display import specshow

# calculate our STFT
HOP_LENGTH = 2**4
SIZE_WINDOW = 2**7
audio_spec = stft(audio, hop_length=HOP_LENGTH, n_fft=SIZE_WINDOW)

# Convert into decibels for visualization
spec_db = amplitude_to_db(audio_spec) # ensures all values are postive real nos

# visualize
specshow(spec_db, sr=sfreq, x_axis='time', y_axis='hz', hop_length=HOP_LENGTH)
```

#### Spectral feature engineering
- Each timeseries has a unique spectral pattern to it. This means we can use patterns in the spectrogram to distinguish classes from one another.
- For example, we can calculate the spectral centroid and bandwidth over time. These describe where most of the spectral energy lies over time.

#### Calculate spectral features

```python
# calculate the spectral centroid and bandwidth for the spectrogram
bandwidths = lr.feature.spectral_bandwidth(S=spec)[0]
centroids = lr.feature.spectral_centroid(S=spec)[0]

# display these features on top of the spectrogram
ax = specshow(spec, x_axis='time', y_axis='hz' hop_length=HOP_LENGTH)
ax.plot(times_spec, centroids)
ax.fill_between(times_spec, centroids - bandwidths / 2, centroids + bandwidths / 2, alpha=0.5)
```

#### Combining spectral and temporal features in a classifier
- As a final step, we can combine each of the features into a single input matrix for our classifier.

```python
centroids_all = []
bandwidths_all = []

for spec in spectrograms:
  bandwidths = lr.feature.spectral_bandwidth(S=lr.db_to_amplitude(spec))
  centroids = lr.feature.spectral_centroids(S=lr.db_to_amplitude(spec))
  # calculate the mean spectral bandwidth
  bandwidths_all.append(np.mean(bandwidths))
  # calculate the mean spectral centroid
  centroids_all.append(np.mean(centroids))
  
# create our X matrix
X = np.column_stack([means, stds, maxs, tempo_mean, tempo_max, tempo_std, bandwidths_all, centroids_all])
```

#### Predicting Time Series Data
- Ridge regression useful if we have noisy or correlated variables.

#### Advanced time series prediction

##### Cleaning and Improving data
- Always understand the source of strange patterns in the data
- **Interpolation** : A common way to deal with missing data is to interpolate missing values
- Using a rolling window to transform data

##### Interpolation in Pandas

```python
# create a mask that return a boolean where missing values are
missing = prices.isna()

# interpolate linearly to fill missing values
prices_interp = prices.interpolate('linear')

# plot the interpolated data in red and the data with missing values in black
ax = prices_interp.plot(c='r')
prices.plot(c='k', ax=ax, lw=2)
```


#### Transforming data to standardize variance
- Using a rolling window, we'll calculate each timepoint's percent change over the mean of a window of previous timepoints. This standardizes the variance of our data and reduces long-term drift.
- A common transformation to apply to data is to standardize its mean and variance over time. There are many ways to do this.
- Convert dataset so that each point represents the % change over a previous window.
- this makes timepoints more comparable to one another if the absolute values of data change a lot.

##### Transforming to percent change with Pandas

```python
def percent_change(values):
    '''Calculate the %change between the last value and the mean of previous values'''
    # separate the last value and all previous values into variables
    previous_values = values[:-1]
    last_value = values[-1]
    
    # calculate the % diff betwn the last value and the mean of earlier values
    percent_change = (last_value - np.mean(previous_values)) / np.mean(previous_values)
    return percent_change
```

#### Applying this to our data
- We can apply this to our data using the `.aggregate()` method, passing our function as an input.

#### Finding outliers in data
- **Outlier** : A common definition is any datapoint that is more than three standard deviations away from the mean of the dataset
- They can have negative effects on the predictive power of our model, biasing it away from its "true" value

#### Plotting a threshold on our data

```python
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
for data, ax in zip([prices, prices_perc_change], axs):
    this_mean = data.mean()
    this_std = data.std()
    
    # plot the data, with a window that is 3 std dev around the mean
    data.plot(ax=ax)
    ax.axhline(this_mean + this_std * 3, ls='--', c='r')
    ax.axhline(this_mean - this_std * 3, ls='--', c='r')
```

##### Replacing outliers with the median

```python
# center the data so the mean is 0
prices_outlier_centered = prices_outlier_perc - prices_outlier_perc.mean()

# calculate std dev
std = prices_outlier_perc.std()

# use the abs value of each data point to make it easier to find outliers
outliers = np.abs(prices_outlier_centered) > (std *3)

# replace outliers with the median value
prices_outlier_fixed = prices_outlier_centered.copy()
prices_outlier_fixed[outliers] = np.nanmedian(prices_outlier_fixed) # calculates the median without being hindered by missing values
```

#### Creating features with windows
- Rolling window can be used to extract features as they change over time.
- In pandas, the `.aggregate()` method can be used to calculated many features of a window at once. By passing a list of functions to the method, each function will be called on the window, and collected in the output. 

```python
feats = prices.rolling(20).aggregate([np.std, np.max]).dropna()
```

#### Percentile summarizes data
- A useful tool for feature extraction is the percentile function. This is similar to calculating the mean or median of your data, but it gives more fine-grained control over what is extracted.
- For a given dataset, the Nth percentile is the value where N% of the data is below that datapoint, and 100-N% of the data is above that datapoint.

```print(np.percentile(np.linspace(0, 200), q=20))```

##### Combining np.percentile() with partial functions to calculate a range of percentiles

```python
data = np.linspace(0, 100)

# create a list of functions using a list comprehension
percentile_funcs = [partial(np.percentile, q=ii) for ii in [20, 40, 60]]

# calculate the output of each function in the same way
percentiles = [i_func(data) for i_func in percentile_funcs]

# calculate multiple percentiles of a rolling window
data.rolling(20).aggregate(percentiles)
```

##### Calculating "date-based" features


### Creating features from the past

#### Time-delayed features and auto-regressive models
- Defining high-quality and relevant features gives our model the best chance at finding useful patterns in the data.

##### The past is useful
- Perhaps the biggest difference between "time-series" data and "non-timeseries" data is the relationship between data points. Because the data has a linear flow(matching the progression of time), patterns will persists over a span of datapoints. As a result, we can use information from the past in order to predict values in the future.
- Timeseries data almost always have information that is shared between timepoints
- Often the features best-suited to predict a timeseries are previous values of the same timeseries

##### A note on smoothness and auto-correlation
- It's important to consider how "smooth" our data is when fitting models with time series.
- The smoothness of our data reflects how much correlation there is between one time point and those that come before and after it. AKA, how correlated is a timepoint with its neighboring timepoints(called autocorrelation)
- The amount of auto-correlation in data will impact our models. AKA, the extent to which previous timepoints are predictive of subsequent timepoints is often described as "autocorrelation", and can have a big impact on the performance of our model.

#### Creating time-lagged features
- Let's see how we can build a model that uses values in the past as input features 
- Remember that regression models will assign a "weight" to each input feature, and we can use these weights to determine how "smooth" or "autocorrelated" the signal is.

##### Time-shifting data with Pandas
- `df.shift(3)`

- **Creating a time-shifted dataframe**

```python
data = pd.Series(...)

# shifts
shifts = [0,1,2,3,4,5,6,7]

# create a dict of time-shifted data
many_shifts = {'lag_{}'.format(ii):data.shift(ii) for ii in shifts}

many_shifts = pd.DataFrame(many_shifts)  #convert to dataframe
```

#### Fitting a model with time-shifted features

```python
# fit the model using these input features
model = Ridge()
model.fit(many_shifts, data)
```

- fit a scikit learn regression model. `many_shifts` is simply a time shifted version of the timeseries contained in the 'data' variable.
- We'll fit the model using Ridge regression, which spreads out weights across features (if applicable) rather than assign it all to a single feature

#### Interpreting the auto-regressive model coefficents

```python
# visualize the fit model coefficients
fig, ax = plt.subplots()
ax.bar(many_shifts.columns, model.coef_)
ax.set(xlabel='Coefficent name', ylabel='Coefficient value')

plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
```

- Once we fit the model, we can investigate the coefficients it has found.
- **Larger absolute values of coefficients mean that a given feature has a large impact on the output variable**
- We can use a barplot the visualize the model's coefficients that were created after fitting the model.

### Stationarity and stability
- A stationary signal is one that does not change its statistical properties over time. It has the same mean, standard deviation, and general trend. A non-stationary signal does change its properties over time. Almost all real world data are non stationary.

#### Model stability
- Most models have an implicit assumption that the relationship between input and outputs is static. If this relationship changes (because the data is non stationary), then the model will generate predictions using an outdated relationship between inputs and outputs.
- Non-stationary data results in variability in our model. The statistical properties the model finds may change with the data. How can we quantify and correct for this?

#### Cross validation to quantify parameter stability
- One approach is to use cross-validation, which yields a set of model coefficents per iteration. We can quantify the variability of these coefficients across iterations.
- If a model's coefficients vary widely between cross-valiadtion splits, there's a good chance the data is non-stationary(or noisy)

#### Bootstrapping the mean
- Bootstrapping is a common way to assess variability
- Bootstrapping is a way to estimate the confidence in the mean of a collection of numbers.
- To bootstrap : 
- 1. Take a random sample of data with replacement
- 2. Calculate the mean of the sample
- 3. Repeat this process many times (1000s)
- 4. Calculate the percentiles of the result(usually 2.5, 97.5)
The result is a 95% confidence interval of the mean of each coefficient

- The lower and upper percentiles represent the variability of the mean.

```python
from sklearn.utils import resample

# cv_coefficients has shape (n_cv_folds, n_coefficients)
n_boots = 100
bootstrap_means = np.zeros(n_boots, n_coefficients)
for ii in range(n_boots):
    # generate random indices for our data with replacement
    # then take the sample mean
    random_sample = resample(cv_coefficients)
    bootstrap_means[ii] = random_sample.mean(axis=0)
    
# compute the percentiles of choice for the bootstrapped means
percentiles = np.percentile(bootstrap_means, (2.5, 97.5), axis=0)
```

#### Plotting the bootstrapped coefficients

```python
fig, ax = plt.subplots()
ax.scatter(many_shifts.columns, percentiles[0], marker='_', s=200)
ax.scatter(many_shifts.columns, percentiles[1], marker='_', s=200)
```

- Plot the lower and upper bounds of the 95% confidence intervals. This gives us and idea for the variability of the mean across all cross-validated iterations.

#### Assessing model performance stability
- It's also common to quantify the stability of a model's predictive power across cross-validation folds
- Calculate the predictive power of the model over cross-validation splits.

##### Model performance over time

```python
def my_corrcoef(est, X, y):
    """return the correlation coefficient betwn model predictions and      validation set"""
    return np.corrcoef(y, est.predict(X))[1,0]
    
# grab the date of the first index of each validation set
first_indices = [data.index[tt[0]] for tr, tt in cv.split(X, y)]

# calculate the CV scores and convert to a Pandas Series
cv_scores = cross_val_score(model, X, y, cv=cv, scoring=my_corrcoef)
cv_scores = pd.Series(cv_scores, index=first_indices)
```

- Because the cross-validation splits happen linearly over time, we can visualize the results as a time series.
- If we see large changes in the predictive power of a model at one moment in time, it could be because the statistics of the data have changed.
- Here we create a rolling mean of our cross-validation scores and plot it with matplotlib

```python
fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

# calculate a rolling mean of scores over time
cv_scores_mean = cv_scores.rolling(10, min_periods=1).mean()
cv_scores.plot(ax=axs[0])
axs[0].set(title='Validation scores (correlation)', ylim=[0,1])

# plot the raw data
data.plot(ax=axs[1])
axs[1].set(title='Validation data')
```

- There can be dips in middle, because statistics of data changes. To avoid this `one option is to restrict the size of the training window`
- This ensures that only the latest datapoints are used in training. We can control this with the `max_train_size` parameter.

```python
# only keep the last 100 datapoints in the training data
window=100

# initialize the cv with the window size
cv = TimeSeriesSplit(n_splits=10, max_train_size=window)
```

- Revisting our visualization from before, we see that restricting the training window slighlty improves the dip in performance in the middle of our validation data.


















































































































