#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 02:19:17 2020

@author: edith
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('china_gdp.csv')
X=dataset.iloc[:,0].values
y=dataset.iloc[:,1].values

from sklearn.preprocessing import PolynomialFeatures
poly_reg= PolynomialFeatures(degree=3)
X_poly=poly_reg.fit_transform(X.reshape(55,1))

from sklearn.linear_model import LinearRegression
lin_reg= LinearRegression()
lin_reg.fit(X_poly,y)
lin_reg.predict(X_poly)
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg.predict(X_poly))


# Applying Sigmoid Function for prediction
def sigmoid(x, Beta_1, Beta_2):
     y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
     return y
 
beta_1 = 0.10
beta_2 = 1990.0

#logistic function
Y_pred = sigmoid(X, beta_1 , beta_2)

#plot initial prediction against datapoints
plt.plot(X, Y_pred*15000000000000.)
plt.plot(X, y, 'ro')

# Lets normalize our data
xdata =X/max(X)
ydata =y/max(y)

# Optimixing the parameter
from scipy.optimize import curve_fit
popt, pcov = curve_fit(sigmoid, xdata, ydata)
#print the final parameters
print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))

x = np.linspace(1960, 2015, 55)
x = x/max(x)
plt.figure(figsize=(8,5))
y = sigmoid(x, *popt)
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(x,y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()
