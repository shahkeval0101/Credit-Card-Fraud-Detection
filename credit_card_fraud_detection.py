# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 22:46:26 2020

@author: keval
"""

import sys 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

dataset = pd.read_csv(r'C:\Users\keval\OneDrive\Desktop\machine learning course learning and important material\Machine Learning Project\Credit Card Fraud detection\creditcard.csv')
#v1-v28 are pca dimensionality reduction in order to protect the sensitive information

dataset.shape
dataset.describe()
dataset.info()

#Unsupervised learning algorithm
#dataset = dataset.sample(frac = 0.1, random_state = 1)
#to reduce the values in the dataset for computation purpose

#plot histogram
dataset.hist(figsize = (20,20))
plt.show()

fraud = dataset[dataset['Class'] == 1]
valid = dataset[dataset['Class'] == 0]
outlier_frac = len(fraud)/float(len(valid))
#huge disparity as fraud is less and less valid cases



corrmat =dataset.corr()
fig = plt.figure(figsize = (12,9))
sns.heatmap(corrmat, vmax = 0.8, square = True)
plt.show()



columns = dataset.columns.tolist()

columns = [c for c in columns if c not in ['Class']]

target = 'Class'

X = dataset[columns]
y  = dataset[target]


print(X.shape)
print(y.shape)


from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


classifiers = {
        "Isolation Forest":IsolationForest(max_samples = len(X), contamination = outlier_frac,random_state = 1),
        "local outlier factor":LocalOutlierFactor(n_neighbors = 20,contamination = outlier_frac)
        }

n_outliers = len(fraud)
for i , (clf_name,clf) in enumerate(classifiers.items()):
    if clf_name == 'local outlier factor':
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)
    
    #reshape the prediction
    #0 = valid and 1 = valid
    y_pred[y_pred == 1] = 0#valid
    y_pred[y_pred == -1]= 1#fraud

    n_errors = (y_pred != y).sum()
    
    
    print('{}: {}'.format(clf_name, n_errors))
    print(accuracy_score(y, y_pred))
    print(classification_report(y, y_pred))



"""
Results
Isolation Forest: 645
0.997735308472053
              precision    recall  f1-score   support

           0       1.00      1.00      1.00    284315
           1       0.34      0.35      0.35       492

    accuracy                           1.00    284807
   macro avg       0.67      0.67      0.67    284807
weighted avg       1.00      1.00      1.00    284807

local outlier factor: 935
0.9967170750718908
              precision    recall  f1-score   support

           0       1.00      1.00      1.00    284315
           1       0.05      0.05      0.05       492

    accuracy                           1.00    284807
   macro avg       0.52      0.52      0.52    284807
weighted avg       1.00      1.00   
"""