#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 00:39:48 2020

@author: edith
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

dataset= pd.read_csv('ChurnData.csv')
X=dataset.iloc[:,[0,1,2,3,4,5,6]].values
y=dataset.iloc[:,-1].values

data_x=dataset.iloc[:,:-1].values
data_y=dataset.iloc[:,-1].values

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X=sc_x.fit_transform(X)
data_x=sc_x.fit_transform(data_x)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=0)
X_train.shape

from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression(C= 0.01,solver='liblinear')
log_reg.fit(X_train,y_train)

from sklearn.metrics import confusion_matrix
cm1=confusion_matrix(y_test,log_reg.predict(X_test))

from sklearn.metrics import jaccard_similarity_score
jss=jaccard_similarity_score(y_test,log_reg.predict(X_test))

from sklearn.metrics import classification_report, confusion_matrix
import itertools

yhat=log_reg.predict(X_test)

def plot_confusion_matrix(cm, classes, normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(y_test, yhat, labels=[1,0]))



# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')



































