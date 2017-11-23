#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
# Load data
url = 'cars.csv'
dat = pd.read_csv(url)
y = dat['dist']
X = dat[['speed']] # sklearn needs X to have 2 dim.

s = lm.LinearRegression(fit_intercept=False)
s.fit(X, y) # Fit regression model

f = plt.figure(figsize=(8, 6))
plt.plot(X, y, 'o', label="Data")
plt.plot(X, s.predict(X), label="OLS-no-intercept")
plt.legend(loc='upper left')
plt.show()
