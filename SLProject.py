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
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from scipy.stats import randint

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer

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

corr_matrix = data.corr()
corr_matrix["ACCLASS"].sort_values(ascending=False)



#####################################################################################################
# Build Classification Models 
#####################################################################################################

### Get full pipeline transformer and data (train and test)

full_pipeline_transformer, X_group, Y_group, X_train, X_test, y_train, y_test = SLProjectLib.get_pipeline_x_y(data, 0.20)

print(full_pipeline_transformer, X_group, Y_group, X_train, X_test, y_train, y_test)


# Testing models


#### Logistic Regression
#  By Maria, Karan

gs_logistic = SLProjectLib.get_best_model(data, 'LogisticRegression', full_pipeline_transformer, X_train, X_test, y_train, y_test)


#### Random Forest Classifier
# By Sidhu, Manipal

gs_random = SLProjectLib.get_best_model(data, 'RandomForestClassifier', full_pipeline_transformer, X_train, X_test, y_train, y_test)


#### Decision Tree Classifier
# By ., Ripudaman

gs_decisiontree = SLProjectLib.get_best_model(data, 'DecisionTreeClassifier', full_pipeline_transformer, X_train, X_test, y_train, y_test)


#### KNeighbors Classifier
# By Saenz Huerta, Ronald

gs_kneighbors = SLProjectLib.get_best_model(data, 'KNeighborsClassifier', full_pipeline_transformer, X_train, X_test, y_train, y_test)


#### SVC
# By Radmy, Mahpara Rafia

gs_svc = SLProjectLib.get_best_model(data, 'SVC', full_pipeline_transformer, X_train, X_test, y_train, y_test)


#### ALL MODELS - BEST MODEL 

#gs_all = SLProjectLib.get_best_model(data, 'ALL', full_pipeline_transformer, X_train, X_test, y_train, y_test)


#### Hard Voting Classifier
# By Saenz Huerta, Ronald

gs_hardvoting = SLProjectLib.get_best_model(data, 'HardVotingClassifier', full_pipeline_transformer, X_train, X_test, y_train, y_test)


#### Soft Voting Classifier
# By Saenz Huerta, Ronald

gs_softvoting = SLProjectLib.get_best_model(data, 'SoftVotingClassifier', full_pipeline_transformer, X_train, X_test, y_train, y_test)
    



#####################################################################################################
# Save Models 
#####################################################################################################

# Save the splitted data in csv format
X_train.to_csv('X_train_data_group7.csv',index=False)
X_test.to_csv('X_test_data_group7.csv',index=False)
y_train.to_csv('y_train_data_group7.csv',index=False)
y_test.to_csv('y_test_data_group7.csv',index=False)


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



    