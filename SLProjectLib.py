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
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer


## Functions

### Analyze Data - Data Exploration, Stats, Plots

def analyze_data(data):
    
    print("\n","Data - 20 first rows")
    print(data.head(20))
    
    print("\n","Data info")
    print(data.info())
    
    print("\n","Data Shape")
    print(data.shape)
    
    print("\n","Data - Null Values")
    print(data.isnull().sum())
    
    print("\n","Data - Null Values String")
    for column in data.columns:
        print("Column:", column, " - Len:", len(data[data[column] == '<Null>']))
    
    print("\n","Data - Describe stats")
    stats = data.describe()
    print(data.describe())

    print("\n","Data - Plot histograms")
    hist = data.hist(bins=3, figsize=(9,10))
    
    print("\n","Data - Plot scatter matrix")
    pd.plotting.scatter_matrix(data, alpha=0.40, figsize=(13,8))
    

### Analyze Data - Unique Values

def analyze_data_unique_values(data, value_counts):
    
    print("\n","Data - Unique Values")
    for column in data:
        print("\n", "Column:", column, " - Len:", len(data[column].unique()), " - Values: ", data[column].unique(), "\n")
        if value_counts:
            print(data[column].value_counts())


### Cleaning Data - Replace values, Drop columns

def cleaning_data_initial(data):
  
    # Replace <null> with nan .
    data = data.replace('<Null>', np.nan)

    # Extract month from date and remove date column
    data['month'] = pd.DatetimeIndex(data['DATE']).month


    #BINARY COLUMNS 0: NULL 1: YES
    binary_columns=['CYCLIST','AUTOMOBILE','MOTORCYCLE','TRUCK','TRSN_CITY_VEH','EMERG_VEH','SPEEDING','AG_DRIV','REDLIGHT','ALCOHOL','DISABILITY','PEDESTRIAN','PASSENGER']
    
    for i in binary_columns:
        data[i].replace(np.nan, 0, inplace=True)
        data[i].replace("Yes", 1, inplace=True)
        data[i] = data[i].astype(int)

    # Drop columns that are not required 
    
    # A lot of different values 
    drop_columns=['INDEX_','ObjectId','ACCNUM', 'X', 'Y','STREET1', 'STREET2', 'LATITUDE', 'LONGITUDE']
    # Duplicated with HOOD_ID and POLICE_DIVISION         
    drop_columns+=['NEIGHBOURHOOD', 'DIVISION']
    # A lot of null values          
    drop_columns+=["OFFSET", "PEDTYPE", "PEDACT", "PEDCOND", "CYCLISTYPE", "CYCACT", "CYCCOND", "FATAL_NO"]
                   
    drop_columns+=['TIME', 'YEAR', 'DATE', 'WARDNUM', 'INITDIR', 'INVAGE']
      
    # DROP
    data = data.drop(drop_columns, axis=1)

    # Drop columns with null count greater than 40 % .
    data = data.dropna(axis=1,thresh=(data.shape[0]*0.6))

    # -----------------------------------------------------------------------------------------------------
    # Drop duplicates values - rows    
    data = data.drop_duplicates()
    data.nunique(axis=0)
    
    return data


### Cleaning Data - Replace values

def cleaning_data_values(data):
    
    '''
    Column: ROAD_CLASS  - Len: 11  - Values:  ['Major Arterial' 'Collector' 'Minor Arterial' 'Local' nan 'Other'
     'Pending' 'Laneway' 'Expressway' 'Expressway Ramp' 'Major Arterial Ramp'] 
    
    Major Arterial         11976
    Minor Arterial          2598
    Collector                929
    Local                    761
    Expressway                52
    Other                     25
    Laneway                   10
    Pending                    7
    Expressway Ramp            4
    Major Arterial Ramp        1
    
    
    Column: DISTRICT  - Len: 6  - Values:  ['Toronto and East York' 'Scarborough' 'Etobicoke York' 'North York' nan
     'Toronto East York'] 

    Toronto and East York    5617
    Etobicoke York           3884
    Scarborough              3798
    North York               3343
    Toronto East York          77


    Column: LOCCOORD  - Len: 8  - Values:  ['Intersection' 'Mid-Block' nan 'Exit Ramp Westbound'
     'Exit Ramp Southbound' 'Mid-Block (Abnormal)' 'Entrance Ramp Westbound'
     'Park, Private Property, Public Lane'] 
    
    Intersection                           11141
    Mid-Block                               5598
    Exit Ramp Westbound                        5
    Mid-Block (Abnormal)                       4
    Exit Ramp Southbound                       3
    Entrance Ramp Westbound                    2
    Park, Private Property, Public Lane        2
    
    
    Column: ACCLOC  - Len: 10  - Values:  ['At Intersection' nan 'Intersection Related' 'Non Intersection'
     'Private Driveway' 'At/Near Private Drive' 'Overpass or Bridge'
     'Underpass or Tunnel' 'Trail' 'Laneway'] 
    
    At Intersection          8060
    Non Intersection         1968
    Intersection Related     1019
    At/Near Private Drive     318
    Private Driveway           13
    Laneway                    13
    Overpass or Bridge         12
    Underpass or Tunnel         6
    Trail                       1
    
    
    Column: TRAFFCTL  - Len: 11  - Values:  ['Traffic Signal' 'No Control' 'Pedestrian Crossover' 'Stop Sign' nan
     'Yield Sign' 'Traffic Controller' 'School Guard' 'Police Control'
     'Traffic Gate' 'Streetcar (Stop for)'] 
    
    No Control              8092
    Traffic Signal          7104
    Stop Sign               1295
    Pedestrian Crossover     195
    Traffic Controller       104
    Yield Sign                16
    Streetcar (Stop for)      16
    Traffic Gate               5
    School Guard               2
    Police Control             2
    
    
    Column: VISIBILITY  - Len: 9  - Values:  ['Clear' 'Rain' 'Other' 'Snow' 'Strong wind' 'Fog, Mist, Smoke, Dust'
     'Drifting Snow' 'Freezing Rain' nan] 
    
    Clear                     14476
    Rain                       1819
    Snow                        332
    Other                        99
    Fog, Mist, Smoke, Dust       46
    Freezing Rain                43
    Drifting Snow                19
    Strong wind                   8
    
    
    Column: LIGHT  - Len: 9  - Values:  ['Daylight' 'Dark' 'Dawn, artificial' 'Dusk, artificial' 'Dusk'
     'Dark, artificial' 'Dawn' 'Daylight, artificial' 'Other'] 
    
    Daylight                9683
    Dark                    3582
    Dark, artificial        2854
    Dusk                     226
    Dusk, artificial         184
    Daylight, artificial     128
    Dawn                     104
    Dawn, artificial          93
    Other                      6
    
    
    Column: RDSFCOND  - Len: 10  - Values:  ['Dry' 'Wet' 'Other' 'Slush' 'Loose Snow' 'Ice' 'Packed Snow'
     'Spilled liquid' 'Loose Sand or Gravel' nan] 
    
    Dry                     13435
    Wet                      2870
    Loose Snow                166
    Other                     147
    Slush                      96
    Ice                        73
    Packed Snow                42
    Loose Sand or Gravel        7
    Spilled liquid              1
    
    
    Column: ACCLASS  - Len: 3  - Values:  ['Fatal' 'Non-Fatal Injury' 'Property Damage Only'] 
    
    Non-Fatal Injury        14561
    Fatal                    2297
    Property Damage Only        2
    
    
    Column: IMPACTYPE  - Len: 11  - Values:  ['Pedestrian Collisions' 'Turning Movement' 'Angle' 'Approaching'
     'SMV Other' 'Rear End' 'SMV Unattended Vehicle' 'Sideswipe'
     'Cyclist Collisions' 'Other' nan] 
    
    Pedestrian Collisions     6811
    Turning Movement          2552
    Cyclist Collisions        1674
    Rear End                  1602
    SMV Other                 1312
    Angle                     1206
    Approaching                870
    Sideswipe                  466
    Other                      182
    SMV Unattended Vehicle     181
    
    
    Column: INVTYPE  - Len: 19  - Values:  ['Driver' 'Pedestrian' 'Motorcycle Driver' 'Passenger' 'Vehicle Owner'
     'Other Property Owner' 'Other' 'Cyclist' 'Truck Driver'
     'Motorcycle Passenger' nan 'Driver - Not Hit' 'In-Line Skater'
     'Moped Driver' 'Wheelchair' 'Pedestrian - Not Hit' 'Trailer Owner'
     'Witness' 'Cyclist Passenger'] 

    Driver                  7618
    Pedestrian              2871
    Passenger               2543
    Vehicle Owner           1636
    Cyclist                  726
    Motorcycle Driver        607
    Truck Driver             316
    Other Property Owner     257
    Other                    174
    Motorcycle Passenger      32
    Moped Driver              27
    Driver - Not Hit          17
    Wheelchair                13
    In-Line Skater             5
    Trailer Owner              2
    Cyclist Passenger          2
    Pedestrian - Not Hit       1
    Witness                    1


    Column: INJURY  - Len: 6  - Values:  ['None' 'Fatal' 'Minor' 'Major' 'Minimal' nan] 
    
    None       6406
    Major      5668
    Minor      1311
    Minimal    1042
    Fatal       821
    
    
    Column: VEHTYPE  - Len: 28  - Values:  [
     'Automobile, Station Wagon' 
     'Other' 
     'Motorcycle' 
     'Bicycle' 
     nan
     'Municipal Transit Bus (TTC)' 
     'Truck - Open' 
     'Taxi' 
     'Passenger Van'
     'Delivery Van' 
     'Moped' 
     'Pick Up Truck' 
     'Police Vehicle' 
     'Truck-Tractor'
     'Truck - Closed (Blazer, etc)' 
     'Street Car'
     'Bus (Other) (Go Bus, Gray Coach)' 
     'Truck - Dump'
     'Construction Equipment'   --
     'Intercity Bus' 
     'Truck (other)' 
     'Truck - Tank'
     'Other Emergency Vehicle'  --
     'School Bus' 
     'Tow Truck' 
     'Off Road - 2 Wheels'
     'Fire Vehicle'             --
     'Truck - Car Carrier'] 
    
    Automobile, Station Wagon           6890
    Other                               4746
    Bicycle                              722
    Motorcycle                           608
    Municipal Transit Bus (TTC)          249
    Pick Up Truck                        179
    Truck - Open                         175
    Passenger Van                        118
    Delivery Van                          72
    Truck - Closed (Blazer, etc)          60
    Street Car                            42
    Truck - Dump                          34
    Truck-Tractor                         33
    Taxi                                  28
    Moped                                 22
    Truck (other)                         13
    Bus (Other) (Go Bus, Gray Coach)      13
    Truck - Tank                          10
    Intercity Bus                         10
    School Bus                             5
    Police Vehicle                         4
    Construction Equipment                 4
    Tow Truck                              4
    Fire Vehicle                           3
    Other Emergency Vehicle                1
    Off Road - 2 Wheels                    1
    Truck - Car Carrier                    1
    
    
    Column: POLICE_DIVISION  - Len: 17  - Values:  ['D11' 'D42' 'D41' 'D14' 'D23' 'D51' 'D32' 'D31' 'D43' 'D12' 'D55' 'D13'
     'D52' 'D54' 'D33' 'D22' 'D53'] 

    D42    1664
    D41    1297
    D22    1202
    D32    1193
    D23    1164
    D14    1163
    D43    1024
    D52     947
    D51     883
    D31     878
    D53     877
    D33     851
    D55     816
    D11     809
    D13     794
    D12     737
    D54     561

    '''
    
    
    
    # -----------------------------------------------------------------------------------------------------
    # Column: ROAD_CLASS 
    
    # Replace the Ramp type with an other existing category
    data['ROAD_CLASS'].replace('Expressway Ramp', 'Expressway', inplace=True)
    data['ROAD_CLASS'].replace('Major Arterial Ramp', 'Major Arterial', inplace=True)
    
    # Replace all null values with Other category
    data['ROAD_CLASS'].replace(np.nan, 'Other', inplace=True)
    
    
    # -----------------------------------------------------------------------------------------------------
    # Column: DISTRICT 
    
    # Replace in one category Toronto and East York
    data['DISTRICT'].replace('Toronto East York', 'Toronto and East York', inplace=True)
    
    # Replace all null values with Other category
    data['DISTRICT'].replace(np.nan, 'Other', inplace=True)
    
    
    # -----------------------------------------------------------------------------------------------------
    # Column: LOCCOORD 
    
    # Replace in existing categories
    data['LOCCOORD'].replace('Mid-Block (Abnormal)', 'Mid-Block', inplace=True)
    data['LOCCOORD'].replace('Entrance Ramp Westbound', 'Exit Ramp',inplace=True)
    data['LOCCOORD'].replace('Exit Ramp Westbound', 'Exit Ramp',inplace=True)
    data['LOCCOORD'].replace('Exit Ramp Southbound','Exit Ramp',inplace=True)
    data['LOCCOORD'].replace('Park, Private Property, Public Lane', 'Other',inplace=True)
    
    # Replace all null values with Other category
    data['LOCCOORD'].replace(np.nan, 'Other', inplace=True)
    
    
    # -----------------------------------------------------------------------------------------------------
    # Column: ACCLOC 
    
    # Replace in existing categories
    data['ACCLOC'].replace('Intersection Related', 'At Intersection', inplace=True)
    data['ACCLOC'].replace('Private Driveway', 'At/Near Private Drive', inplace=True)
    
    # Replace small values in Other category
    data['ACCLOC'].replace('Laneway', 'Other', inplace=True)
    data['ACCLOC'].replace('Overpass or Bridge', 'Other', inplace=True)
    data['ACCLOC'].replace('Underpass or Tunnel', 'Other', inplace=True)
    data['ACCLOC'].replace('Trail', 'Other', inplace=True)
    
    # Replace all null values with Other category
    data['ACCLOC'].replace(np.nan, 'Other', inplace=True)
    
    
    # -----------------------------------------------------------------------------------------------------
    # Column: TRAFFCTL 
    
    # Replace small values in Other category
    data['TRAFFCTL'].replace('Traffic Controller', 'Other', inplace=True)
    data['TRAFFCTL'].replace('Yield Sign', 'Other', inplace=True)
    data['TRAFFCTL'].replace('Streetcar (Stop for)', 'Other', inplace=True)
    data['TRAFFCTL'].replace('Traffic Gate', 'Other', inplace=True)
    data['TRAFFCTL'].replace('School Guard', 'Other', inplace=True)
    data['TRAFFCTL'].replace('Police Control', 'Other', inplace=True)
    
    # Replace all null values with Other category
    data['TRAFFCTL'].replace(np.nan, 'Other', inplace=True)
    

    # -----------------------------------------------------------------------------------------------------
    # Column: VISIBILITY 
    
    # Replace in existing categories
    data['VISIBILITY'].replace('Drifting Snow', 'Snow', inplace=True)
    data['VISIBILITY'].replace('Freezing Rain', 'Rain',inplace=True)
    
    # Replace small values in Other category
    data['VISIBILITY'].replace('Strong wind', 'Other',inplace=True)
    
    # Replace all null values with Other category
    data['VISIBILITY'].replace(np.nan, 'Other', inplace=True)
    
    
    # -----------------------------------------------------------------------------------------------------
    # Column: LIGHT 
    
    # Replace in existing categories
    data['LIGHT'].replace('Dark, artificial', 'Dark',inplace=True)
    data['LIGHT'].replace('Dusk, artificial', 'Dusk',inplace=True)
    data['LIGHT'].replace('Daylight, artificial', 'Daylight',inplace=True)
    data['LIGHT'].replace('Dawn, artificial', 'Dawn',inplace=True)
    
    # Replace all null values with Other category
    data['LIGHT'].replace(np.nan, 'Other', inplace=True)

    
    # -----------------------------------------------------------------------------------------------------
    # Column: RDSFCOND 
    
    # Replace in existing categories
    data['RDSFCOND'].replace('Loose Snow', 'Snow',inplace=True)
    data['RDSFCOND'].replace('Packed Snow', 'Snow',inplace=True)
    
    # Replace small values in Other category
    data['RDSFCOND'].replace('Loose Sand or Gravel', 'Other',inplace=True)
    data['RDSFCOND'].replace('Spilled liquid', 'Other',inplace=True)
    
    # Replace all null values with Other category
    data['RDSFCOND'].replace(np.nan, 'Other', inplace=True)
    
    
    # -----------------------------------------------------------------------------------------------------
    # Column: ACCLASS 
    
    # Replace in existing categories - Property Damage Only is Non-FAtal Injury
    data['ACCLASS'].replace('Property Damage Only', 'Non-Fatal Injury',inplace=True)
    
    # Replace all null values with Non-Fatal Injury category
    data['ACCLASS'].replace(np.nan, 'Non-Fatal Injury', inplace=True)
    
    # Replace values with binary classification
    data['ACCLASS'].replace('Non-Fatal Injury', 0, inplace=True)
    data['ACCLASS'].replace('Fatal', 1, inplace=True)
    
    
    # -----------------------------------------------------------------------------------------------------
    # Column: IMPACTYPE 
    
    # Replace all null values with Non-Fatal Injury category
    data['IMPACTYPE'].replace(np.nan, 'Other', inplace=True)
    
    
    # -----------------------------------------------------------------------------------------------------
    # Column: INVTYPE 
    
    # Replace in existing categories
    data['INVTYPE'].replace('Witness', 'Pedestrian', inplace=True)
    data['INVTYPE'].replace('Pedestrian - Not Hit', 'Pedestrian', inplace=True)
    data['INVTYPE'].replace('Driver - Not Hit', 'Driver', inplace=True)
    data['INVTYPE'].replace('Cyclist', 'Cyclist Passenger', inplace=True)
    data['INVTYPE'].replace('Cyclist Passenger', 'Cyclist Passenger', inplace=True)
    data['INVTYPE'].replace('Motorcycle Driver', 'Motorcycle Passenger', inplace=True)
    data['INVTYPE'].replace('Motorcycle Passenger', 'Motorcycle Passenger', inplace=True)
    data['INVTYPE'].replace('Trailer Owner', 'Truck Driver', inplace=True)
    
    # Replace small values in Other category
    data['INVTYPE'].replace('Other Property Owner', 'Other',inplace=True)
    data['INVTYPE'].replace('Moped Driver', 'Other',inplace=True)
    data['INVTYPE'].replace('Wheelchair', 'Other',inplace=True)
    data['INVTYPE'].replace('In-Line Skater', 'Other',inplace=True)
    
    # Replace all null values with Other category
    data['INVTYPE'].replace(np.nan, 'Other', inplace=True)
    
    
    # -----------------------------------------------------------------------------------------------------
    # Column: INJURY 
    
    # Replace in existing categories
    data['INJURY'].replace('Minimal', 'Minor', inplace=True)
    
    # Replace all null values with Non-Fatal Injury category
    data['INJURY'].replace(np.nan, 'None', inplace=True)
    
    
    # -----------------------------------------------------------------------------------------------------
    # Column: VEHTYPE 
    
    # Replace all types of truck in only group called 'Truck'
    data['VEHTYPE'].replace('Truck - Open', 'Truck', inplace=True)
    data['VEHTYPE'].replace('Truck-Tractor', 'Truck', inplace=True)
    data['VEHTYPE'].replace('Truck - Closed (Blazer, etc)', 'Truck', inplace=True)
    data['VEHTYPE'].replace('Truck - Dump', 'Truck', inplace=True)
    data['VEHTYPE'].replace('Truck (other)', 'Truck', inplace=True)
    data['VEHTYPE'].replace('Truck - Tank', 'Truck', inplace=True)
    data['VEHTYPE'].replace('Pick Up Truck', 'Truck', inplace=True)
    data['VEHTYPE'].replace('Tow Truck', 'Truck', inplace=True)
    data['VEHTYPE'].replace('Truck - Car Carrier', 'Truck', inplace=True)
    
    # Replace all types of 2 wheels in only group called 2 Wheels
    data['VEHTYPE'].replace('Motorcycle', '2 Wheels', inplace=True)
    data['VEHTYPE'].replace('Bicycle', '2 Wheels', inplace=True)
    data['VEHTYPE'].replace('Moped', '2 Wheels', inplace=True)
    data['VEHTYPE'].replace('Off Road - 2 Wheels', '2 Wheels', inplace=True)

    # Replace all types of automobiles in only group called Automobile
    data['VEHTYPE'].replace('Automobile, Station Wagon', 'Automobile', inplace=True)
    data['VEHTYPE'].replace('Taxi', 'Automobile', inplace=True)
    
    # Replace all types of Emergency Vehicles in only group called Emergency
    data['VEHTYPE'].replace('Police Vehicle', 'Emergency', inplace=True)
    data['VEHTYPE'].replace('Other Emergency Vehicle', 'Emergency', inplace=True)
    data['VEHTYPE'].replace('Fire Vehicle', 'Emergency', inplace=True)
    
    # Replace all types of Buses in only group called Bus
    data['VEHTYPE'].replace('Municipal Transit Bus (TTC)', 'Bus', inplace=True)
    data['VEHTYPE'].replace('Street Cars', 'Bus', inplace=True)
    data['VEHTYPE'].replace('Street Car', 'Bus', inplace=True)
    data['VEHTYPE'].replace('Bus (Other) (Go Bus, Gray Coach)', 'Bus', inplace=True)
    data['VEHTYPE'].replace('Intercity Bus', 'Bus', inplace=True)
    data['VEHTYPE'].replace('School Bus', 'Bus', inplace=True)
    
    # Replace all types of Vans in only group called Van
    data['VEHTYPE'].replace('Passenger Van', 'Van', inplace=True)
    data['VEHTYPE'].replace('Delivery Van', 'Van', inplace=True)
    
    # Replace small values in Other category
    data['VEHTYPE'].replace('Construction Equipment', 'Other',inplace=True)
    
    # Replace all null values with Other category
    data['VEHTYPE'].replace(np.nan, 'Other', inplace=True)


    # -----------------------------------------------------------------------------------------------------
    # Column: POLICE_DIVISION 
    
    #Police Division without 'D' character    
    data['POLICE_DIVISION'] = data['POLICE_DIVISION'].str.strip("D")
    data['POLICE_DIVISION'] = data['POLICE_DIVISION'].astype(int)
        
    
    # -----------------------------------------------------------------------------------------------------
    # Drop duplicates values - rows    
    data = data.drop_duplicates()
    data.nunique(axis=0)
    

    return data



### Data Preprocessing - Get pipeline transformer, and X, Y train ant test data.

def get_pipeline_x_y(data=None, test_size=0.20):
    
    features_columns_categorical = ["ROAD_CLASS", "DISTRICT", "LOCCOORD", "ACCLOC", "TRAFFCTL", "VISIBILITY", "LIGHT", "RDSFCOND", "IMPACTYPE", "INVTYPE", "INJURY", "VEHTYPE"]
    features_columns_numbers = ['HOUR', 'CYCLIST','AUTOMOBILE','MOTORCYCLE','TRUCK','TRSN_CITY_VEH','EMERG_VEH','SPEEDING','AG_DRIV','REDLIGHT','ALCOHOL','DISABILITY','PEDESTRIAN','PASSENGER', 'POLICE_DIVISION', 'HOOD_ID', 'month']


    #    ('imputer', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='missing')),
    #('imputer', KNNImputer(n_neighbors=2)),
    
        
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(missing_values=np.nan, strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

    # Full pipeline - Column Transformer
    full_pipeline_transformer = ColumnTransformer([
        ("num", num_pipeline, features_columns_numbers),
        ("cat", categorical_pipeline, features_columns_categorical),
    ])

    X_group = data[features_columns_categorical + features_columns_numbers]
    Y_group = data['ACCLASS']

            
    np.random.seed(2)

    # Divide data in train/test 
    X_train, X_test, y_train, y_test = train_test_split(X_group, Y_group, test_size=test_size, random_state=0)


    return full_pipeline_transformer, X_group, Y_group, X_train, X_test, y_train, y_test


### Get Best Model - Logistic Regression, Decision Tree, Random Forest Classifier, SVC, K-Neighbors Classifier

def get_best_model(data, classifier_model, full_pipeline_transformer, X_train, X_test, y_train, y_test):
    
    
    # Initialze the estimators
    estimator_lr = LogisticRegression(random_state=42)
    estimator_dt = DecisionTreeClassifier(random_state=42)
    estimator_rf = RandomForestClassifier(random_state=42)
    estimator_svc = SVC(probability=True, random_state=42)
    estimator_kn = KNeighborsClassifier()
    
    estimator_hard_vc = VotingClassifier(
        estimators=[('lr', estimator_lr), ('rf', estimator_rf), ('svc', estimator_svc), ('dt', estimator_dt), ('kn', estimator_kn)],
        voting='hard'
    )
    
    estimator_soft_vc = VotingClassifier(
        estimators=[('lr', estimator_lr), ('rf', estimator_rf), ('svc', estimator_svc), ('dt', estimator_dt), ('kn', estimator_kn)],
        voting='soft'
    )

    
    # Initiaze the hyperparameters for each dictionary
    if classifier_model == "LogisticRegression":

        param_grid = {
            'classifier__solver': ['lbfgs', 'saga'], 
            'classifier__max_iter': [100, 1000],
            'classifier__random_state': [0, 42], 
            'classifier__multi_class': ['auto', 'multinomial']
        }
        
        full_pipeline = Pipeline([
            ('preprocessing', full_pipeline_transformer),
            ('classifier', estimator_lr),
        ])
        
        
    elif classifier_model == "DecisionTreeClassifier":
        
        #'classifier__min_samples_split': [2, 5, 10, 20], 
        #'classifier__min_samples_leaf': [1, 5, 10], 
        #'classifier__max_leaf_nodes': [None, 5, 10, 20],
            
        param_grid = {
            'classifier__criterion': ['gini', 'entropy'], 
            'classifier__max_depth': [2 ,5, 10, 25, None],
            'classifier__min_samples_split': [2],
            'classifier__min_samples_leaf': [1],
            'classifier__max_leaf_nodes': [20],
            'classifier__class_weight': [None, {0:1,1:5}, {0:1,1:10}, {0:1,1:25}]
        }
        
        full_pipeline = Pipeline([
            ('preprocessing', full_pipeline_transformer),
            ('classifier', estimator_dt),
        ])

    elif classifier_model == "RandomForestClassifier":
        
        param_grid = {
            'classifier__n_estimators': [10, 50, 100, 250], 
            'classifier__max_depth': [5,10,20],
            'classifier__class_weight': [None, {0:1,1:5}, {0:1,1:10}, {0:1,1:25}]
        }
        
        full_pipeline = Pipeline([
            ('preprocessing', full_pipeline_transformer),
            ('classifier', estimator_rf),
        ])
        
    elif classifier_model == "SVC":
        
        param_grid = {
            'classifier__kernel': ['linear', 'rbf','poly'],
            #'classifier__C': [0.01, 0.1, 1, 10, 100],
            #'classifier__class_weight': [None, {0:1,1:5}, {0:1,1:10}, {0:1,1:25}]
            'classifier__gamma': ['auto'],
        }
                
        full_pipeline = Pipeline([
            ('preprocessing', full_pipeline_transformer),
            ('classifier', estimator_svc),
        ])
        
    elif classifier_model == "KNeighborsClassifier":
        
        param_grid = {
            'classifier__n_neighbors': [2,5,10,25,50]
        }
        
        full_pipeline = Pipeline([
            ('preprocessing', full_pipeline_transformer),
            ('classifier', estimator_kn),
        ])
        
    elif classifier_model == "HardVotingClassifier":
        
        param_grid = {}
        
        full_pipeline = Pipeline([
            ('preprocessing', full_pipeline_transformer),
            ('classifier', estimator_hard_vc),
        ])
        
    elif classifier_model == "SoftVotingClassifier":
        
        param_grid = {}
        
        full_pipeline = Pipeline([
            ('preprocessing', full_pipeline_transformer),
            ('classifier', estimator_soft_vc),
        ])


    else:
        param_grid = [
            {
                'classifier': [estimator_lr], 
                'classifier__solver': ['lbfgs', 'saga'], 
                'classifier__max_iter': [1000],
                'classifier__random_state': [0, 42], 
                'classifier__multi_class': ['auto', 'multinomial'] 
            },
            {
                'classifier': [estimator_dt], 
                'classifier__criterion': ['gini'], 
                'classifier__max_depth': [5,10,25,None],
                'classifier__min_samples_split': [2,5,10], 
                'classifier__class_weight': [None, {0:1,1:5}, {0:1,1:10}, {0:1,1:25}]
            },
            {
                'classifier': [estimator_rf], 
                'classifier__n_estimators': [10, 50, 100, 250], 
                'classifier__max_depth': [5,10,20],
                'classifier__class_weight': [None, {0:1,1:5}, {0:1,1:10}, {0:1,1:25}]
            },
            {
                'classifier': [estimator_svc],
                'classifier__kernel': ['linear', 'rbf', 'poly'],
                'classifier__gamma': ['auto'],
            },
            {
                'classifier': [estimator_kn], 
                'classifier__n_neighbors': [2,5,10,25,50]
            }
        ]        
        
        full_pipeline = Pipeline([
            ('preprocessing', full_pipeline_transformer),
            ('classifier', estimator_1),
        ])
    
    print("***********************************************************")
    print("Get Best Estimator/Params of the Model for ", classifier_model)   
    
    
    gs = GridSearchCV(full_pipeline, param_grid, cv=3, n_jobs=-1, scoring='roc_auc') 
    gs.fit(X_train, y_train)

        
    print("Best Estimator:", gs.best_estimator_)
    print("Best Params:", gs.best_params_)
    print("Best Score:", gs.best_score_)
    
    # Test data performance
    print("Test Precision:",precision_score(gs.predict(X_test), y_test))
    print("Test Recall:",recall_score(gs.predict(X_test), y_test))
    print("Test ROC AUC Score:",roc_auc_score(gs.predict(X_test), y_test))
    
    print("Test Accuracy Score = ", accuracy_score(gs.predict(X_test), y_test))
    print("Test Confusion Matrix = \n", confusion_matrix(gs.predict(X_test), y_test))
    print("Test Classification Report = \n", classification_report(gs.predict(X_test), y_test))

    # CONFUSION MATRIX PLOT   
    cm = confusion_matrix(gs.predict(X_test), y_test)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells

    if classifier_model != "HardVotingClassifier":
        # ROC AUC CURVE PLOT
        plot_roc_curve(gs, X_test, y_test) 
        plt.show()
        
    return gs.best_estimator_
