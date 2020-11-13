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












