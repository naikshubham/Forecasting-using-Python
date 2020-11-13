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
