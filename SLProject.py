# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 15:16:35 2022

@author: gmi_r
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
SLProjectLib.analyze_data(data)
SLProjectLib.analyze_data_unique_values(data, False)

# preprocessing - clean data 
data = SLProjectLib.cleaning_data_initial(data)


# Analyze data - after the first cleaning
SLProjectLib.analyze_data(data)
SLProjectLib.analyze_data_unique_values(data, True)


# preprocessing - clean data 
data = SLProjectLib.cleaning_data_values(data)


# Analyze data - after the second cleaning
SLProjectLib.analyze_data(data)
SLProjectLib.analyze_data_unique_values(data, True)



#####################################################################################################
# Build Classification Models 
#####################################################################################################



full_pipeline_transformer, X_group, Y_group, X_train, X_test, y_train, y_test = SLProjectLib.get_pipeline_x_y(data, 0.20)

print(full_pipeline_transformer, X_group, Y_group, X_train, X_test, y_train, y_test)




###############################################################################
# Testing models

gs_logistic = SLProjectLib.get_best_model(data, 'LogisticRegression', full_pipeline_transformer, X_train, X_test, y_train, y_test)

gs_random = SLProjectLib.get_best_model(data, 'RandomForestClassifier', full_pipeline_transformer, X_train, X_test, y_train, y_test)

gs_decisiontree = SLProjectLib.get_best_model(data, 'DecisionTreeClassifier', full_pipeline_transformer, X_train, X_test, y_train, y_test)

#gs_multinomialnb = SLProjectLib.get_best_model(data, 'MultinomialNB', full_pipeline_transformer, X_train, X_test, y_train, y_test)

gs_kneighbors = SLProjectLib.get_best_model(data, 'KNeighborsClassifier', full_pipeline_transformer, X_train, X_test, y_train, y_test)

gs_svc = SLProjectLib.get_best_model(data, 'SVC', full_pipeline_transformer, X_train, X_test, y_train, y_test)

