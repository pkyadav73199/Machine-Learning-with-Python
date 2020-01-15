#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 02:02:05 2020

@author: edith
"""

# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

# Importing the dataset
dataset= pd.read_csv('FuelConsumptionCo2.csv')
X=dataset.iloc[:,8:12].values
y=dataset.iloc[:,12].values

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X=sc_x.fit_transform(X)"""

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X[:,3],y,test_size=0.2, random_state= 0 )
X_train=X_train.reshape(853,1)
y_train=y_train.reshape(853,1)
X_test=X_test.reshape(214,1)

from sklearn.linear_model import LinearRegression
linear= LinearRegression()
linear.fit(X_train,y_train)

y_pred=linear.predict(X_test)

plt.scatter(X_train,y_train, color='red')
plt.plot(X_test,linear.predict(X_test),color= 'blue')

from sklearn.metrics import r2_score
print(np.mean(np.absolute((y_pred-y_test))))

