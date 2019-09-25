# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 18:49:52 2019

@author: 10664864
"""

from statsmodels.tsa.arima_model import ARMA
import pandas as pd

earthquake = pd.read_csv('earthquakes.csv')
#print(earthquake.head())

model = ARMA(earthquake['earthquakes_per_year'], order=(3, 1))

results = model.fit()

print(results.summary())