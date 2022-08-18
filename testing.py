# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 23:21:42 2022

@author: gmi_r
"""


import pandas as pd
import joblib
import sys
from os import path
from sklearn.metrics import *
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import json 

features_columns_categorical = ["ROAD_CLASS", "DISTRICT", "LOCCOORD", "ACCLOC", "TRAFFCTL", "VISIBILITY", "LIGHT", "RDSFCOND", "IMPACTYPE", "INVTYPE", "VEHTYPE"]
features_columns_numbers = ['HOUR', 'CYCLIST','AUTOMOBILE','MOTORCYCLE','TRUCK','TRSN_CITY_VEH','EMERG_VEH','SPEEDING','AG_DRIV','REDLIGHT','ALCOHOL','DISABILITY','PEDESTRIAN','PASSENGER', 'POLICE_DIVISION', 'HOOD_ID', 'month']
features_columns = features_columns_categorical + features_columns_numbers
    
def data1():
        
    json_str_ = '''[{ 
        "ROAD_CLASS": "Collector", 
        "DISTRICT": "Toronto and East York", 
        "LOCCOORD": "Mid-Block",
        "ACCLOC": "At Intersection", 
        "TRAFFCTL": "No Control", 
        "VISIBILITY": "Clear", 
        "LIGHT": "Dark", 
        "RDSFCOND": "Dry", 
        "IMPACTYPE": "Pedestrian Collisions", 
        "INVTYPE": "Passenger", 
        "VEHTYPE": "Truck",
        "INJURY": "Non-Fatal Injury",
        "POLICE_DIVISION": 33,
        "HOOD_ID": 43,
        "month": 1,
        "HOUR": 12,
        "MOTORCYCLE": 0,
        "CYCLIST": 0,  
        "SPEEDING": 1, 
        "REDLIGHT": 0, 
        "DISABILITY": 0,
        "TRUCK": 0,
        "TRSN_CITY_VEH": 0,
        "EMERG_VEH": 0,
        "AG_DRIV": 1,
        "ALCOHOL": 0,
        "PEDESTRIAN": 0,
        "PASSENGER": 1
        }]'''
    
    json_ = json.loads(json_str_)
    query = pd.DataFrame(json_, columns=features_columns)
    return query


def data2():

    
    json_str_ = '''[{ 
        "ROAD_CLASS": "Expressway", 
        "DISTRICT": "Toronto and East York", 
        "LOCCOORD": "Mid-Block",
        "ACCLOC": "At Intersection", 
        "TRAFFCTL": "Pedestrian Crossover", 
        "VISIBILITY": "Fog, Mist, Smoke, Dust", 
        "LIGHT": "Daylight", 
        "RDSFCOND": "Dry", 
        "IMPACTYPE": "Pedestrian Collisions", 
        "INVTYPE": "Passenger", 
        "VEHTYPE": "Other",
        "POLICE_DIVISION": 33,
        "HOOD_ID": 43,
        "month": 1,
        "HOUR": 12,
        "MOTORCYCLE": 1,
        "CYCLIST": 0,  
        "SPEEDING": 1, 
        "REDLIGHT": 1, 
        "DISABILITY": 0,
        "TRUCK": 0,
        "TRSN_CITY_VEH": 0,
        "EMERG_VEH": 0,
        "AG_DRIV": 1,
        "ALCOHOL": 1,
        "PEDESTRIAN": 1,
        "PASSENGER": 0
        }]'''

    json_ = json.loads(json_str_)
    query = pd.DataFrame(json_, columns=features_columns)
    return query
  

def data3():  
    json_str_ = '''[{ 
        "ROAD_CLASS": "Collector", 
        "DISTRICT": "Toronto and East York", 
        "LOCCOORD": "Mid-Block",
        "ACCLOC": "At Intersection", 
        "TRAFFCTL": "No Control", 
        "VISIBILITY": "Clear", 
        "LIGHT": "Dark", 
        "RDSFCOND": "Dry", 
        "IMPACTYPE": "Pedestrian ", 
        "INVTYPE": "Passenger", 
        "VEHTYPE": "Truck",
        "POLICE_DIVISION": 33,
        "HOOD_ID": 43,
        "month": 1,
        "HOUR": 12,
        "MOTORCYCLE": 0,
        "CYCLIST": 0,  
        "SPEEDING": 0, 
        "REDLIGHT": 0, 
        "DISABILITY": 0,
        "TRUCK": 0,
        "TRSN_CITY_VEH": 0,
        "EMERG_VEH": 0,
        "AG_DRIV": 0,
        "ALCOHOL": 0,
        "PEDESTRIAN": 1,
        "PASSENGER": 0
        }]'''
        
    json_ = json.loads(json_str_)
    query = pd.DataFrame(json_, columns=features_columns)
    return query

def data4():  
    json_str_ = '''[{ 
        "ROAD_CLASS": "Collector", 
        "DISTRICT": "Toronto and East York", 
        "LOCCOORD": "Mid-Block",
        "ACCLOC": "At Intersection", 
        "TRAFFCTL": "No Control", 
        "VISIBILITY": "Clear", 
        "LIGHT": "Dark", 
        "RDSFCOND": "Dry", 
        "IMPACTYPE": "Angle", 
        "INVTYPE": "Driver", 
        "VEHTYPE": "Truck",
        "POLICE_DIVISION": 11,
        "HOOD_ID": 43,
        "month": 1,
        "HOUR": 12,
        "MOTORCYCLE": 0,
        "CYCLIST": 0,  
        "SPEEDING": 0, 
        "REDLIGHT": 0, 
        "DISABILITY": 0,
        "TRUCK": 0,
        "TRSN_CITY_VEH": 0,
        "EMERG_VEH": 0,
        "AG_DRIV": 0,
        "ALCOHOL": 0,
        "PEDESTRIAN": 0,
        "PASSENGER": 0
        }]'''
        
    json_ = json.loads(json_str_)
    query = pd.DataFrame(json_, columns=features_columns)
    return query



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

X_train = pd.read_csv(path.join(deploy_folder,"x_train_data_group7.csv"))
y_train = pd.read_csv(path.join(deploy_folder,"y_train_data_group7.csv"))
X_test = pd.read_csv(path.join(deploy_folder,"x_test_data_group7.csv"))
y_test = pd.read_csv(path.join(deploy_folder,"y_test_data_group7.csv"))


# load all models:
models_loaded = {}
for model_name in (models):
    models_loaded[model_name] = joblib.load(path.join(deploy_folder, models[model_name]))
    print(f'Model {model_name} loaded')
    
print("\n")
    
for model_name in (models):
    query = data1()
    prediction = list(models_loaded[model_name].predict(query))
    print(f'Model {model_name}: {prediction}')



print("\n")
    
for model_name in (models):
    query = data2()
    prediction = list(models_loaded[model_name].predict(query))
    print(f'Model {model_name}: {prediction}')
    
    
    
print("\n")
    
for model_name in (models):
    query = data3()
    prediction = list(models_loaded[model_name].predict(query))
    print(f'Model {model_name}: {prediction}')
    
    
print("\n")
    
for model_name in (models):
    query = data4()
    prediction = list(models_loaded[model_name].predict(query))
    print(f'Model {model_name}: {prediction}')
    