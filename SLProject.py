# -*- coding: utf-8 -*-
"""
# GROUP PROJECT

## Welcome to the COMP 247 Project - Supervising Learning Project - KSI Collisions Toronto

Relevant Information:

    College: Centennial College
    Program: Software Engineering Technology - Artificial Intelligence
    Term: Summer 2022
    Course: 22M --Supervised Learning (SEC. 001) - COMP247001_2022MW

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer

from sklearn import tree


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


#### RandomForestClassifier
# By Sidhu, Manipal

gs_random = SLProjectLib.get_best_model(data, 'RandomForestClassifier', full_pipeline_transformer, X_train, X_test, y_train, y_test)


#### DecisionTreeClassifier
# By ., Ripudaman

gs_decisiontree = SLProjectLib.get_best_model(data, 'DecisionTreeClassifier', full_pipeline_transformer, X_train, X_test, y_train, y_test)


#### KNeighborsClassifier
# By Saenz Huerta, Ronald

gs_kneighbors = SLProjectLib.get_best_model(data, 'KNeighborsClassifier', full_pipeline_transformer, X_train, X_test, y_train, y_test)


#### SVC
# By Radmy, Mahpara Rafia

gs_svc = SLProjectLib.get_best_model(data, 'SVC', full_pipeline_transformer, X_train, X_test, y_train, y_test)


#### ALL MODELS - BEST MODEL 

gs_all = SLProjectLib.get_best_model(data, 'ALL', full_pipeline_transformer, X_train, X_test, y_train, y_test)


