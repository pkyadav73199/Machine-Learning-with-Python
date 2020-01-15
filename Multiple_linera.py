#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 01:13:53 2020

@author: edith
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('FuelConsumptionCo2.csv')
X=dataset.iloc[:,[4,5,8,9,10,11]].values
y=dataset.iloc[:,12].values

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X=sc_x.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_train)
np.mean(np.absolute(y_pred-y_train))

plt.scatter(X[:,5],y,color='red')
plt.plot(X_train[:,5],regressor.predict(X_train),color='blue')

import statsmodels.api as sm
X=np.append(arr=np.ones((1067,1)).astype(int), values= X,axis=1)
X_opt=X[:,[0,1,2,4,5]]
regressor_ols=sm.OLS(y,X_opt)
result=regressor_ols.fit()
print(result.summary())
































