#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm

# Load data
# url: https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/cars.csv
url = 'cars.csv'
dat = pd.read_csv(url)
print(dat.describe())
y = dat['dist']
X = dat[['speed']] # sklearn needs X to have 2 dim.

lr = lm.LinearRegression(fit_intercept=False)
lr.fit(X, y) # Fit regression model

ls = lm.LinearRegression(fit_intercept=True)
ls.fit(X, y) # Fit regression model

f = plt.figure(figsize=(8, 6))
plt.plot(X, y, 'o', label="Data")
plt.plot(X, lr.predict(X), label="OLS-no-intercept")
plt.plot(X, ls.predict(X), label="OLS-intercept")
plt.legend(loc='upper left')
plt.show()
