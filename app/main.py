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

from flask import Flask, request, jsonify
import traceback
import pandas as pd
#from sklearn import preprocessing
# import pickle
import joblib
import sys
from os import path
from sklearn.metrics import *
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from flask_cors import CORS


# Your API definition
app = Flask(__name__)
CORS(app)

#####################################################################################################
# Load Files and models
#####################################################################################################

deploy_folder = r'deploy'

# Load data
X_test_ds = pd.read_csv(path.join(deploy_folder,"x_test_data_ds_group7.csv"))
y_test_ds = pd.read_csv(path.join(deploy_folder,"y_test_data_ds_group7.csv"))

X_test_us = pd.read_csv(path.join(deploy_folder,"x_test_data_us_group7.csv"))
y_test_us = pd.read_csv(path.join(deploy_folder,"y_test_data_us_group7.csv"))

# load all models:        
models = {
    "LogisticRegression": "LR_model_group7.pkl",
    "RandomForestClassifier": "RF_model_group7.pkl", 
    "DecisionTreeClassifier": "DT_model_group7.pkl",
    "KNeighborsClassifier": "KN_model_group7.pkl",
    "SVC": "SVC_model_group7.pkl",
    "HardVotingClassifier": "HV_model_group7.pkl",
    "SoftVotingClassifier": "SV_model_group7.pkl"
}

models_loaded = {}
for model_name in (models):
    models_loaded[model_name] = joblib.load(path.join(deploy_folder, models[model_name]))
    print(f'Model {model_name} loaded')

# name of columns
features_columns_categorical = ["ROAD_CLASS", "DISTRICT", "LOCCOORD", "ACCLOC", "TRAFFCTL", "VISIBILITY", "LIGHT", "RDSFCOND", "IMPACTYPE", "INVTYPE", "VEHTYPE"]
features_columns_numbers = ['HOUR', 'CYCLIST','AUTOMOBILE','MOTORCYCLE','TRUCK','TRSN_CITY_VEH','EMERG_VEH','SPEEDING','AG_DRIV','REDLIGHT','ALCOHOL','DISABILITY','PEDESTRIAN','PASSENGER', 'POLICE_DIVISION', 'HOOD_ID', 'month']
model_columns = features_columns_categorical + features_columns_numbers


#####################################################################################################
# METHOD PREDICT - RETURN THE PREDICTION USING EACH MODEL
#####################################################################################################

@app.route("/predict/<model_name>", methods=['GET','POST']) #use decorator pattern for the route
def predict(model_name):
    if models_loaded:
        try:
            print("***********************************************************")
            print(f'Get predictions for model {model_name}:')
            
            json_ = request.json
            print('Data JSON: \n', json_)
            
            query = pd.DataFrame(json_, columns=model_columns)
            
            prediction = list(models_loaded[model_name].predict(query))
            print('Prediction: ', prediction)
            
            res = jsonify({"prediction": str(prediction)})
            res.headers.add('Access-Control-Allow-Origin', '*')
            return res
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        return ('No model available.')
    
    
#####################################################################################################
# METHOD SCORES - RETURN STATS OF EACH MODEL 
#####################################################################################################

@app.route("/scores/<model_name>", methods=['GET','POST']) #use decorator pattern for the route
def scores(model_name):
    if models_loaded:
        try:
            print("***********************************************************")
            print(f'Get scores for model {model_name}:')
            
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
    
            print(f'accuracy={accuracy}  precision={precision}  recall={recall}  f1={f1}  roc_auc={roc_auc}  confussion_matrix={conf_matrix} ')
            
            res = jsonify({"accuracy": accuracy,
                            "precision": precision,
                            "recall":recall,
                            "f1": f1,
                            "roc_auc": roc_auc,
                            "confussion_matrix": str(conf_matrix)
                          })
            res.headers.add('Access-Control-Allow-Origin', '*')
            return res
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        return ('No model available.')
