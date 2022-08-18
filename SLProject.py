# -*- coding: utf-8 -*-
"""
# GROUP PROJECT

## Welcome to the COMP 247 Project - Supervising Learning Project - KSI Collisions Toronto

Relevant Information:

    College: Centennial College
    Program: Software Engineering Technology - Artificial Intelligence
    Term: Summer 2022
    Course: 22M --Supervised Learning (SEC. 001) - COMP247001_2022MW

Group # 7

Group Members

    ., Ripudaman
    Maria, Karan
    Radmy, Mahpara Rafia
    Saenz Huerta, Ronald
    Sidhu, Manipal

COMP 247 Project

Group Project – Developing a predictive machine learning model (classifier) and deploy it as a web API for inference
Dataset

https://data.torontopolice.on.ca/datasets/TorontoPS::ksi/about
Models:

    Logistic Regression
    Random Forest Classifier
    Decision Tree Classifier
    KNeighbors Classifier
    SVC
    
"""

import SLProjectLib

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import *
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from scipy.stats import randint

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.utils import resample
from sklearn import tree

import joblib

#####################################################################################################
# Load & check the data: 
#####################################################################################################

data = pd.read_csv('ksi.csv')

# Analyze data - first step
### Analyze Data - Data Exploration Stats, Histogram, Graphs
SLProjectLib.analyze_data(data)
SLProjectLib.analyze_data_unique_values(data, False)


# preprocessing - clean data 
data = SLProjectLib.cleaning_data_initial(data)


# Analyze data - after the first cleaning
### Analyze Data - Data Exploration Stats, Histogram, Graphs
SLProjectLib.analyze_data(data)
SLProjectLib.analyze_data_unique_values(data, True)


#### Correlation in the dataset
# Trying to find corelation in the dataset
corr_matrix = data.corr()
sns.heatmap(corr_matrix)
plt.title(f'Correlation Matrix', fontsize = 20)
plt.show()

# preprocessing - clean data 
data = SLProjectLib.cleaning_data_values(data)


# Analyze data - after the second cleaning
### Analyze Data - Data Exploration Stats, Histogram, Graphs
SLProjectLib.analyze_data(data)
SLProjectLib.analyze_data_unique_values(data, True)


#### Correlation in the dataset
# Trying to find corelation in the dataset
corr_matrix = data.corr()
sns.heatmap(corr_matrix)
plt.title(f'Correlation Matrix', fontsize = 20)
plt.show()


corr_matrix = data.corr()
corr_matrix["ACCLASS"].sort_values(ascending=False)


#####################################################################################################
# Pre-processing Model
#####################################################################################################

### Get full pipeline transformer and data (train and test)

full_pipeline_transformer, X_group, Y_group, X_train, X_test, y_train, y_test = SLProjectLib.get_pipeline_x_y(data, 0.20)
print(full_pipeline_transformer, X_group, Y_group, X_train, X_test, y_train, y_test)


# Transformer
X_group_transformed = (full_pipeline_transformer.fit_transform(X_group))

# Train model
clf_0 = LogisticRegression().fit(X_group_transformed, Y_group)
 
# Predict on training set
pred_y_0 = clf_0.predict(X_group_transformed)
pred_y_0

# How's the accuracy, precision, recall, roc auc?
print("All Data Accuracy:",accuracy_score(pred_y_0, Y_group))
print("All Data Precision:",precision_score(pred_y_0, Y_group))
print("All Data Recall:",recall_score(pred_y_0, Y_group))
print("All Data ROC AUC Score:",roc_auc_score(pred_y_0, Y_group))

# -- With Injury field
#All Data Accuracy: 0.9163083030102547
#All Data Precision: 0.4180602006688963         -- We can improve in that stat
#All Data Recall: 0.9490238611713666
#All Data ROC AUC Score: 0.9316034545763829

# -- Without Injury field
#All Data Accuracy: 0.8646377770426729
#All Data Precision: 0.05207835642618251
#All Data Recall: 0.6374269005847953
#All Data ROC AUC Score: 0.7523322939754811


# Should we be excited?
print( np.unique( pred_y_0 ) )
# [0 1]

# Check the balance of our target values
Y_group.value_counts()
# 0    13022
# 1     2093
# Name: ACCLASS, dtype: int64

data.groupby('ACCLASS').size().plot(kind='pie',
                                       y = "ACCLASS",
                                       label = "ACCLASS",
                                       autopct='%1.1f%%')
plt.title(f'Balance Data group by Target Class', fontsize = 20)
plt.show()


##################################################################
## 1. Up-sample Minority Class
##################################################################


# Separate majority and minority classes
data_majority = data[data["ACCLASS"]==0]
data_minority = data[data["ACCLASS"]==1]
 
# Upsample minority class
data_minority_upsampled = resample(data_minority, 
                                     replace=True,     # sample with replacement
                                     n_samples=13022,  # to match majority class
                                     random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
data_upsampled = pd.concat([data_majority, data_minority_upsampled])
 
# Display new class counts
data_upsampled["ACCLASS"].value_counts()
# 0    13022
# 1    13022
#Name: ACCLASS, dtype: int64


data_upsampled.groupby('ACCLASS').size().plot(kind='pie',
                                               y = "ACCLASS",
                                               label = "ACCLASS",
                                               autopct='%1.1f%%')
plt.title(f'Balance Data with Up-sample Minority Class', fontsize = 20)
plt.show()


full_pipeline_transformer, X_group_upsampled, Y_group_upsampled, X_train, X_test, y_train, y_test = SLProjectLib.get_pipeline_x_y(data_upsampled, 0.20)
print(full_pipeline_transformer, X_group_upsampled, Y_group_upsampled, X_train, X_test, y_train, y_test)

# Get data transformed to fit after
X_group_upsampled_transformed = (full_pipeline_transformer.fit_transform(X_group_upsampled))

# Train model
clf_1 = LogisticRegression().fit(X_group_upsampled_transformed, Y_group_upsampled)
 
# Predict on training set
pred_y_1_upsampled = clf_1.predict(X_group_upsampled_transformed)
pred_y_1_upsampled

# How's the accuracy, precision, recall, roc auc?
print("All Data Accuracy:",accuracy_score(pred_y_1_upsampled, Y_group_upsampled))
print("All Data Precision:",precision_score(pred_y_1_upsampled, Y_group_upsampled))
print("All Data Recall:",recall_score(pred_y_1_upsampled, Y_group_upsampled))
print("All Data ROC AUC Score:",roc_auc_score(pred_y_1_upsampled, Y_group_upsampled))

# -- With Injury field
# All Data Accuracy: 0.7889725080632776
# All Data Precision: 0.7714636768545539   -- The precision increased compared than the unbalanced data  :) 
# All Data Recall: 0.7994588572338055
# All Data ROC AUC Score: 0.7893272918013514

# -- Without Injury field
# All Data Accuracy: 0.6719781907541085
# All Data Precision: 0.6896022116418369
# All Data Recall: 0.6661226911950152
# All Data ROC AUC Score: 0.6721921264619507


# Should we be excited?
print( np.unique( pred_y_1_upsampled ) )
# [0 1]

# Check the balance of our target values
Y_group_upsampled.value_counts()
# 0    13022
# 1     13022
# Name: ACCLASS, dtype: int64


##################################################################
## 2. Down-sample Minority Class
##################################################################


# Separate majority and minority classes
data_majority = data[data["ACCLASS"]==0]
data_minority = data[data["ACCLASS"]==1]
 
# Upsample minority class
data_majority_downsampled = resample(data_majority, 
                                     replace=True,     # sample with replacement
                                     n_samples=2093,  # to match majority class
                                     random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
data_downsampled = pd.concat([data_majority_downsampled, data_minority])
 
# Display new class counts
data_downsampled["ACCLASS"].value_counts()
# 0    2093
# 1    2093
#Name: ACCLASS, dtype: int64


data_downsampled.groupby('ACCLASS').size().plot(kind='pie',
                                               y = "ACCLASS",
                                               label = "Type",
                                               autopct='%1.1f%%')
plt.title(f'Balance Data with Down-sample Minority Class', fontsize = 20)
plt.show()


full_pipeline_transformer, X_group_downsampled, Y_group_downsampled, X_train, X_test, y_train, y_test = SLProjectLib.get_pipeline_x_y(data_downsampled, 0.20)
print(full_pipeline_transformer, X_group_downsampled, Y_group_downsampled, X_train, X_test, y_train, y_test)

# Get data transformed to fit after
X_group_downsampled_transformed = (full_pipeline_transformer.fit_transform(X_group_downsampled))

# Train model
clf_2 = LogisticRegression().fit(X_group_downsampled_transformed, Y_group_downsampled)
 
# Predict on training set
pred_y_2_downsampled = clf_2.predict(X_group_downsampled_transformed)
pred_y_2_downsampled

# How's the accuracy, precision, recall, roc auc?
print("All Data Accuracy:",accuracy_score(pred_y_2_downsampled, Y_group_downsampled))
print("All Data Precision:",precision_score(pred_y_2_downsampled, Y_group_downsampled))
print("All Data Recall:",recall_score(pred_y_2_downsampled, Y_group_downsampled))
print("All Data ROC AUC Score:",roc_auc_score(pred_y_2_downsampled, Y_group_downsampled))

# -- With Injury field
# All Data Accuracy: 0.7916865742952699
# All Data Precision: 0.7701863354037267  -- The precision increased compared than the unbalanced data  :) 
# All Data Recall: 0.8047928107838243
# All Data ROC AUC Score: 0.7922269138664885

# -- Without Injury field
# All Data Accuracy: 0.6731963688485427
# All Data Precision: 0.6779741997133302
# All Data Recall: 0.6715570279223853
# All Data ROC AUC Score: 0.6732121849693933


# Should we be excited?
print( np.unique( pred_y_2_downsampled ) )
# [0 1]

# Check the balance of our target values
Y_group_downsampled.value_counts()
# 0    2093
# 1     2093
# Name: ACCLASS, dtype: int64



#####################################################################################################
# Build Classification Models 
#####################################################################################################

### Get full pipeline transformer and data (train and test)


full_pipeline_transformer_u, X_group_u, Y_group_u, X_train_u, X_test_u, y_train_u, y_test_u = SLProjectLib.get_pipeline_x_y(data_upsampled, 0.20)
print(full_pipeline_transformer_u, X_group_u, Y_group_u, X_train_u, X_test_u, y_train_u, y_test_u)

full_pipeline_transformer_d, X_group_d, Y_group_d, X_train_d, X_test_d, y_train_d, y_test_d = SLProjectLib.get_pipeline_x_y(data_downsampled, 0.20)
print(full_pipeline_transformer_d, X_group_d, Y_group_d, X_train_d, X_test_d, y_train_d, y_test_d)


# Testing models

#### Logistic Regression
#  By Maria, Karan

gs_logistic = SLProjectLib.get_best_model(data_upsampled, 'LogisticRegression', full_pipeline_transformer_u, X_train_u, X_test_u, y_train_u, y_test_u)


#### Random Forest Classifier
# By Sidhu, Manipal

gs_random = SLProjectLib.get_best_model(data_downsampled, 'RandomForestClassifier', full_pipeline_transformer_d, X_train_d, X_test_d, y_train_d, y_test_d)


#### Decision Tree Classifier
# By ., Ripudaman

gs_decisiontree = SLProjectLib.get_best_model(data_upsampled, 'DecisionTreeClassifier', full_pipeline_transformer_u, X_train_u, X_test_u, y_train_u, y_test_u)


#### KNeighbors Classifier
# By Saenz Huerta, Ronald

gs_kneighbors = SLProjectLib.get_best_model(data_downsampled, 'KNeighborsClassifier', full_pipeline_transformer_d, X_train_d, X_test_d, y_train_d, y_test_d)


#### SVC
# By Radmy, Mahpara Rafia

gs_svc = SLProjectLib.get_best_model(data_downsampled, 'SVC', full_pipeline_transformer_d, X_train_d, X_test_d, y_train_d, y_test_d)


#### Hard Voting Classifier
# By Saenz Huerta, Ronald

gs_hardvoting = SLProjectLib.get_best_model(data_downsampled, 'HardVotingClassifier', full_pipeline_transformer_d, X_train_d, X_test_d, y_train_d, y_test_d)


#### Soft Voting Classifier
# By Saenz Huerta, Ronald

gs_softvoting = SLProjectLib.get_best_model(data_downsampled, 'SoftVotingClassifier', full_pipeline_transformer_d, X_train_d, X_test_d, y_train_d, y_test_d)
    


#####################################################################################################
# Save Models 
#####################################################################################################


# Save the splitted data in csv format
X_train_d.to_csv('X_train_data_ds_group7.csv',index=False)
X_test_d.to_csv('X_test_data_ds_group7.csv',index=False)
y_train_d.to_csv('y_train_data_ds_group7.csv',index=False)
y_test_d.to_csv('y_test_data_ds_group7.csv',index=False)

X_train_u.to_csv('X_train_data_us_group7.csv',index=False)
X_test_u.to_csv('X_test_data_us_group7.csv',index=False)
y_train_u.to_csv('y_train_data_us_group7.csv',index=False)
y_test_u.to_csv('y_test_data_us_group7.csv',index=False)


# Save the full pipeline using the joblib – (dump).
joblib.dump(full_pipeline_transformer, "full_pipeline_group7.pkl")

# Save the model Logistic Regression
joblib.dump(gs_logistic, "LR_model_group7.pkl")

# Save the model Random Forest Classifier
joblib.dump(gs_random, "RF_model_group7.pkl")

# Save the model Decision Tree Classifier
joblib.dump(gs_decisiontree, "DT_model_group7.pkl")

# Save the model KNeighbors Classifier
joblib.dump(gs_kneighbors, "KN_model_group7.pkl")

# Save the model SVC
joblib.dump(gs_svc, "SVC_model_group7.pkl")

# Save the model Hard Voting Classifier
joblib.dump(gs_hardvoting, "HV_model_group7.pkl")

# Save the model Soft Voting Classifier
joblib.dump(gs_softvoting, "SV_model_group7.pkl")



