# Lets consider ARMA(1,1) model: if we want to generate and fit the data using following coefficients `yt = 0.5 * y(t-1) + 0.2 * e(t-1) + et`

from statsmodels.tsa.arima_process import arma_generate_sample
import matplotlib.pyplot as plt

ar_coefs = [1, -0.5]
ma_coefs = [1, 0.2]
# both the list starts with 1 which indicates `zero lag term` and is always set to 1. We set the lag 1 AR coef to -0.5 and MA coef to 0.2
# we generate the data passing in the coefs, the no of sample and standard deviation of the shocks. Here we pass the **negatives of the AR coefs** we desire and pass **MA coefs as it is**

y = arma_generate_sample(ar_coefs, ma_coefs, nsample = 100, sigma = 0.5)

plt.plot(y)

# Fitting an ARMA model

from statsmodels.tsa.arima_model import ARMA

# instantitate model object
model = ARMA(y, order=(1, 1))

#Fit model
results = model.fit()