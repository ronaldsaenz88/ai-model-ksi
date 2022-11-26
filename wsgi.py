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

from app.main import app


if __name__ == "__main__":
    port = 12345 # If you don't provide any port the port will be set to 12345

    app.run(port=port, debug=True)
