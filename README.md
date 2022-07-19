# ai-model-ksi

## Welcome to the COMP 247 Project - Supervising Learning Project - KSI Collisions Toronto

# Relevant Information:
- College: Centennial College
- Program: Software Engineering Technology - Artificial Intelligence
- Term: Summer 2022
- Course: 22M --Supervised Learning (SEC. 001) - COMP247001_2022MW

# Group Members
- ., Ripudaman
- Maria, Karan
- Radmy, Mahpara Rafia
- Saenz Huerta, Ronald
- Sidhu, Manipal

# COMP 247 Project 

Group Project – Developing a predictive machine learning model (classifier) and deploy it as a web API for inference

NLP:
- https://data.torontopolice.on.ca/datasets/TorontoPS::ksi/about


Models:
- Logistic Regression
- Random Forest Classifier
- Decision Tree Classifier
- KNeighbors Classifier
- SVC

Results:

- Logistic Regression
```
	Best Estimator: Pipeline(steps=[('preprocessing',
			 ColumnTransformer(transformers=[('num',
							  Pipeline(steps=[('imputer',
									   SimpleImputer(strategy='median')),
									  ('std_scaler',
									   StandardScaler())]),
							  ['HOUR', 'CYCLIST',
							   'AUTOMOBILE', 'MOTORCYCLE',
							   'TRUCK', 'TRSN_CITY_VEH',
							   'EMERG_VEH', 'SPEEDING',
							   'AG_DRIV', 'REDLIGHT',
							   'ALCOHOL', 'DISABILITY',
							   'PEDESTRIAN', 'PASSENGER',
							   'POLICE_DIVISION', 'HOOD...
							  Pipeline(steps=[('imputer',
									   SimpleImputer(fill_value='missing',
											 strategy='constant')),
									  ('encoder',
									   OneHotEncoder(handle_unknown='ignore'))]),
							  ['ROAD_CLASS', 'DISTRICT',
							   'LOCCOORD', 'ACCLOC',
							   'TRAFFCTL', 'VISIBILITY',
							   'LIGHT', 'RDSFCOND',
							   'IMPACTYPE', 'INVTYPE',
							   'INJURY', 'VEHTYPE'])])),
			('classifier',
			 LogisticRegression(max_iter=1000, multi_class='multinomial',
					    random_state=0))])
	
    Best Params: {'classifier__max_iter': 1000, 'classifier__multi_class': 'multinomial', 'classifier__random_state': 0, 'classifier__solver': 'lbfgs'}
	
    Best Score: 0.881947451783029

	Test Precision: 0.43705463182897863
	Test Recall: 0.9246231155778895
	Test ROC AUC Score: 0.9203498014150071
	Test Accuracy Score =  0.916639100231558
	Test Confusion Matrix = 
	 [[2587  237]
	 [  15  184]]

	Test Classification Report = 
		       precision    recall  f1-score   support

		   0       0.99      0.92      0.95      2824
		   1       0.44      0.92      0.59       199

	    accuracy                           0.92      3023
	   macro avg       0.72      0.92      0.77      3023
	weighted avg       0.96      0.92      0.93      3023
```

- Random Forest Classifier
```
    Best Estimator: Pipeline(steps=[('preprocessing',
                    ColumnTransformer(transformers=[('num',
                                                    Pipeline(steps=[('imputer',
                                                                    SimpleImputer(strategy='median')),
                                                                    ('std_scaler',
                                                                    StandardScaler())]),
                                                    ['HOUR', 'CYCLIST',
                                                    'AUTOMOBILE', 'MOTORCYCLE',
                                                    'TRUCK', 'TRSN_CITY_VEH',
                                                    'EMERG_VEH', 'SPEEDING',
                                                    'AG_DRIV', 'REDLIGHT',
                                                    'ALCOHOL', 'DISABILITY',
                                                    'PEDESTRIAN', 'PASSENGER',
                                                    'POLICE_DIVISION', 'HOOD...
                                                    Pipeline(steps=[('imputer',
                                                                    SimpleImputer(fill_value='missing',
                                                                                    strategy='constant')),
                                                                    ('encoder',
                                                                    OneHotEncoder(handle_unknown='ignore'))]),
                                                    ['ROAD_CLASS', 'DISTRICT',
                                                    'LOCCOORD', 'ACCLOC',
                                                    'TRAFFCTL', 'VISIBILITY',
                                                    'LIGHT', 'RDSFCOND',
                                                    'IMPACTYPE', 'INVTYPE',
                                                    'INJURY', 'VEHTYPE'])])),
                    ('classifier',
                    RandomForestClassifier(max_depth=20, n_estimators=250,
                                            random_state=42))])
    
    Best Params: {'classifier__class_weight': None, 'classifier__max_depth': 20, 'classifier__n_estimators': 250}
    
    Best Score: 0.9314571740975975
    
    Test Precision: 0.498812351543943
    Test Recall: 0.9905660377358491
    Test ROC AUC Score: 0.9577518911553666
    Test Accuracy Score =  0.9295401918623883
    Test Confusion Matrix = 
    [[2600  211]
    [   2  210]]
    
    Test Classification Report = 
                precision    recall  f1-score   support

            0       1.00      0.92      0.96      2811
            1       0.50      0.99      0.66       212

        accuracy                           0.93      3023
    macro avg       0.75      0.96      0.81      3023
    weighted avg       0.96      0.93      0.94      3023

```

- Decision Tree Classifier
```
    Best Estimator: Pipeline(steps=[('preprocessing',
                    ColumnTransformer(transformers=[('num',
                                                    Pipeline(steps=[('imputer',
                                                                    SimpleImputer(strategy='median')),
                                                                    ('std_scaler',
                                                                    StandardScaler())]),
                                                    ['HOUR', 'CYCLIST',
                                                    'AUTOMOBILE', 'MOTORCYCLE',
                                                    'TRUCK', 'TRSN_CITY_VEH',
                                                    'EMERG_VEH', 'SPEEDING',
                                                    'AG_DRIV', 'REDLIGHT',
                                                    'ALCOHOL', 'DISABILITY',
                                                    'PEDESTRIAN', 'PASSENGER',
                                                    'POLICE_DIVISION', 'HOOD...
                                                                    SimpleImputer(fill_value='missing',
                                                                                    strategy='constant')),
                                                                    ('encoder',
                                                                    OneHotEncoder(handle_unknown='ignore'))]),
                                                    ['ROAD_CLASS', 'DISTRICT',
                                                    'LOCCOORD', 'ACCLOC',
                                                    'TRAFFCTL', 'VISIBILITY',
                                                    'LIGHT', 'RDSFCOND',
                                                    'IMPACTYPE', 'INVTYPE',
                                                    'INJURY', 'VEHTYPE'])])),
                    ('classifier',
                    DecisionTreeClassifier(class_weight={0: 1, 1: 5}, max_depth=10,
                                            min_samples_split=10,
                                            random_state=42))])
    
    Best Params: {'classifier__class_weight': {0: 1, 1: 5}, 'classifier__criterion': 'gini', 'classifier__max_depth': 10, 'classifier__min_samples_split': 10}
    
    Best Score: 0.8575872195905258
    
    Test Precision: 0.7577197149643705
    Test Recall: 0.405852417302799
    Test ROC AUC Score: 0.6801278179495666
    Test Accuracy Score =  0.8117763810783989
    Test Confusion Matrix = 
    [[2135  102]
    [ 467  319]]
    
    Test Classification Report = 
                precision    recall  f1-score   support

            0       0.82      0.95      0.88      2237
            1       0.76      0.41      0.53       786

        accuracy                           0.81      3023
    macro avg       0.79      0.68      0.71      3023
    weighted avg       0.80      0.81      0.79      3023

```

- KNeighbors Classifier
```
    Best Estimator: Pipeline(steps=[('preprocessing',
                    ColumnTransformer(transformers=[('num',
                                                    Pipeline(steps=[('imputer',
                                                                    SimpleImputer(strategy='median')),
                                                                    ('std_scaler',
                                                                    StandardScaler())]),
                                                    ['HOUR', 'CYCLIST',
                                                    'AUTOMOBILE', 'MOTORCYCLE',
                                                    'TRUCK', 'TRSN_CITY_VEH',
                                                    'EMERG_VEH', 'SPEEDING',
                                                    'AG_DRIV', 'REDLIGHT',
                                                    'ALCOHOL', 'DISABILITY',
                                                    'PEDESTRIAN', 'PASSENGER',
                                                    'POLICE_DIVISION', 'HOOD_ID',
                                                    'month']),
                                                    ('cat',
                                                    Pipeline(steps=[('imputer',
                                                                    SimpleImputer(fill_value='missing',
                                                                                    strategy='constant')),
                                                                    ('encoder',
                                                                    OneHotEncoder(handle_unknown='ignore'))]),
                                                    ['ROAD_CLASS', 'DISTRICT',
                                                    'LOCCOORD', 'ACCLOC',
                                                    'TRAFFCTL', 'VISIBILITY',
                                                    'LIGHT', 'RDSFCOND',
                                                    'IMPACTYPE', 'INVTYPE',
                                                    'INJURY', 'VEHTYPE'])])),
                    ('classifier', KNeighborsClassifier())])
    
    Best Params: {'classifier__n_neighbors': 5}
    
    Best Score: 0.8504230246806003
    
    Test Precision: 0.42992874109263657
    Test Recall: 0.8379629629629629
    Test ROC AUC Score: 0.8762312142923115
    Test Accuracy Score =  0.9090307641415812
    Test Confusion Matrix = 
    [[2567  240]
    [  35  181]]
    
    Test Classification Report = 
                precision    recall  f1-score   support

            0       0.99      0.91      0.95      2807
            1       0.43      0.84      0.57       216

        accuracy                           0.91      3023
    macro avg       0.71      0.88      0.76      3023
    weighted avg       0.95      0.91      0.92      3023

```

- SVC
```
    Best Estimator: Pipeline(steps=[('preprocessing',
                    ColumnTransformer(transformers=[('num',
                                                    Pipeline(steps=[('imputer',
                                                                    SimpleImputer(strategy='median')),
                                                                    ('std_scaler',
                                                                    StandardScaler())]),
                                                    ['HOUR', 'CYCLIST',
                                                    'AUTOMOBILE', 'MOTORCYCLE',
                                                    'TRUCK', 'TRSN_CITY_VEH',
                                                    'EMERG_VEH', 'SPEEDING',
                                                    'AG_DRIV', 'REDLIGHT',
                                                    'ALCOHOL', 'DISABILITY',
                                                    'PEDESTRIAN', 'PASSENGER',
                                                    'POLICE_DIVISION', 'HOOD...
                                                    'month']),
                                                    ('cat',
                                                    Pipeline(steps=[('imputer',
                                                                    SimpleImputer(fill_value='missing',
                                                                                    strategy='constant')),
                                                                    ('encoder',
                                                                    OneHotEncoder(handle_unknown='ignore'))]),
                                                    ['ROAD_CLASS', 'DISTRICT',
                                                    'LOCCOORD', 'ACCLOC',
                                                    'TRAFFCTL', 'VISIBILITY',
                                                    'LIGHT', 'RDSFCOND',
                                                    'IMPACTYPE', 'INVTYPE',
                                                    'INJURY', 'VEHTYPE'])])),
                    ('classifier',
                    SVC(gamma='auto', probability=True, random_state=42))])
    
    Best Params: {'classifier__gamma': 'auto', 'classifier__kernel': 'rbf'}
    
    Best Score: 0.8857828761006091
    
    Test Precision: 0.40617577197149646
    Test Recall: 1.0
    Test ROC AUC Score: 0.95617110799439
    Test Accuracy Score =  0.9173006946741648
    Test Confusion Matrix = 
    [[2602  250]
    [   0  171]]
    
    Test Classification Report = 
                precision    recall  f1-score   support

            0       1.00      0.91      0.95      2852
            1       0.41      1.00      0.58       171

        accuracy                           0.92      3023
    macro avg       0.70      0.96      0.77      3023
    weighted avg       0.97      0.92      0.93      3023

```
