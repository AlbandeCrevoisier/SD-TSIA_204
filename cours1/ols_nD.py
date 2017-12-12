#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sklearn.linear_model as lm
import statsmodels.api as sm

# Load data
url = 'http://vincentarelbundock.github.io/Rdatasets/csv/datasets/trees.csv'
d = pd.read_csv(url)
# Fit regression model
X = d[['Girth', 'Height']]
X = sm.add_constant(X)
y = d['Volume']
res = sm.OLS(y, X).fit().params
XX = np.arange(8, 22, 0.5)
YY = np.arange(64, 90, 0.5)
xx, yy = np.meshgrid(XX, YY)
zz = res[0] + res[1]*xx + res[2]*yy
f = plt.figure()
ax = Axes3D(f)
ax.plot(X['Girth'], X['Height'], y, 'o')
ax.plot_wireframe(xx, yy, zz, rstride=10, cstride=10)
plt.show()
