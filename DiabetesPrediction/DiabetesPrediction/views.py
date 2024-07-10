from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

def home(request):
    return render(request, 'home.html')

def predict(request):
    return render(request, 'predict.html')

def result(request):
    df = pd.read_csv("C:/Users/athar/MachineLearning/diabetes.csv")
    X = df.drop('Outcome', axis=1)
    Y = df['Outcome']
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    X = X_std
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=2)
    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, y_train)
    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])
    pred=classifier.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])

    result1=""
    if(pred==1):
        result1="Positive"
    else:
        result1="Negative"

    return render(request, 'predict.html',{"result2":result1})