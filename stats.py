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

import sys
import json 
from os import path
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


features_columns_categorical = ["ROAD_CLASS", "DISTRICT", "LOCCOORD", "ACCLOC", "TRAFFCTL", "VISIBILITY", "LIGHT", "RDSFCOND", "IMPACTYPE", "INVTYPE", "VEHTYPE"]
features_columns_numbers = ['HOUR', 'CYCLIST','AUTOMOBILE','MOTORCYCLE','TRUCK','TRSN_CITY_VEH','EMERG_VEH','SPEEDING','AG_DRIV','REDLIGHT','ALCOHOL','DISABILITY','PEDESTRIAN','PASSENGER', 'POLICE_DIVISION', 'HOOD_ID', 'month']
features_columns = features_columns_categorical + features_columns_numbers
    
print(len(features_columns))


deploy_folder = r'deploy'
    
models = {
    "LogisticRegression": "LR_model_group7.pkl",
    "RandomForestClassifier": "RF_model_group7.pkl", 
    "DecisionTreeClassifier": "DT_model_group7.pkl",
    "KNeighborsClassifier": "KN_model_group7.pkl",
    "SVC": "SVC_model_group7.pkl",
    "HardVotingClassifier": "HV_model_group7.pkl",
    "SoftVotingClassifier": "SV_model_group7.pkl"
}

# Load data
X_test_ds = pd.read_csv(path.join(deploy_folder,"x_test_data_ds_group7.csv"))
y_test_ds = pd.read_csv(path.join(deploy_folder,"y_test_data_ds_group7.csv"))
X_train_ds = pd.read_csv(path.join(deploy_folder,"x_train_data_ds_group7.csv"))
y_train_ds = pd.read_csv(path.join(deploy_folder,"y_train_data_ds_group7.csv"))

X_test_us = pd.read_csv(path.join(deploy_folder,"x_test_data_us_group7.csv"))
y_test_us = pd.read_csv(path.join(deploy_folder,"y_test_data_us_group7.csv"))
X_train_us = pd.read_csv(path.join(deploy_folder,"x_train_data_us_group7.csv"))
y_train_us = pd.read_csv(path.join(deploy_folder,"y_train_data_us_group7.csv"))
    

# load all models:
models_loaded = {}
for model_name in (models):
    models_loaded[model_name] = joblib.load(path.join(deploy_folder, models[model_name]))
    print("\n" + "*"*100 )
    print(f'** Model {model_name} loaded')
    print("*"*100 + "\n"  )

    if model_name in ["DecisionTreeClassifier", "LogisticRegression"]:
        X_test = X_test_us
        y_test = y_test_us
    else:
        X_test = X_test_ds
        y_test = y_test_ds
        
        
    y_pred = models_loaded[model_name].predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)   
    class_report = classification_report(y_test, y_pred)      

    # print(f'accuracy={accuracy}  precision={precision}  recall={recall}  f1={f1}  roc_auc={roc_auc}  confussion_matrix={conf_matrix} ')
    
    print(models_loaded[model_name])
    
    
    # Test data performance
    print("Test Precision:",  precision)
    print("Test Recall:", recall)
    print("Test F1 Score:", f1)
    print("Test ROC AUC Score:", roc_auc)
    
    print("Test Accuracy Score = ", accuracy)
    print("Test Confusion Matrix = \n", conf_matrix)
    print("Test Classification Report = \n", class_report)

    # CONFUSION MATRIX PLOT
    ax = plt.subplot()
    sns.heatmap(conf_matrix, annot=True, ax = ax); #annot=True to annotate cells
    plt.title(f'Confusion Matrix - Model {model_name}', fontsize = 20)
    plt.show()


    if model_name != "HardVotingClassifier":
        # ROC AUC CURVE PLOT
        plot_roc_curve(models_loaded[model_name], X_test, y_test) 
            
        plt.title(f'ROC AUC CURVE - Model {model_name}', fontsize = 20)
        plt.show()

        
            
print("\n")





