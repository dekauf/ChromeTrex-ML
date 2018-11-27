#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 13:15:35 2018

@author: omkar
"""


#%% Preprocess the data - PROBLEM 1

import pandas as pd
import numpy as np
from sklearn import tree
import xlsxwriter
import random

#Sheet for Prediction Output
workbook = xlsxwriter.Workbook('dinoPreds_vfast.xlsx')
worksheet = workbook.add_worksheet()

worksheet.write('A1', 'Speed')
worksheet.write('B1', 'Width')
worksheet.write('C1', 'Predictions')

combinedData = pd.read_csv('combined_dinoData_vfast.csv',header = 0)

dataY = list(combinedData['Distance'])
widths = list(combinedData['Width'])
speeds = list(combinedData['Speed'])

dataX = np.zeros([len(dataY),2])

for i in range(len(dataX)):
    dataX[i][0] = speeds[i]
    dataX[i][1] = widths[i]
    
#%% Preprocess the data - PROBLEM 2

import pandas as pd
import numpy as np
from sklearn import tree
import xlsxwriter
import random

#Sheet for Prediction Output
workbook = xlsxwriter.Workbook('dinoPreds_fast_multiple.xlsx')
worksheet = workbook.add_worksheet()

worksheet.write('A1', 'Previous_Speed')
worksheet.write('B1', 'Previous_Width')
worksheet.write('C1', 'Speed')
worksheet.write('D1', 'Width')
worksheet.write('E1', 'Previous_Distance')
worksheet.write('F1', 'Distance')

combinedData = pd.read_csv('dinoData_Multiple_Combined.csv',header = 0)

dataY = list(combinedData['Distance1'])

previousSpeeds = list(combinedData['Speed0'])
previousWidths = list(combinedData['Width0'])
speeds = list(combinedData['Speed1'])
widths = list(combinedData['Width1'])
previous_distances = list(combinedData['Distance0'])

dataX = np.zeros([len(dataY),5])

for i in range(len(dataX)):
    dataX[i][0] = previousSpeeds[i]
    dataX[i][1] = previousWidths[i]
    dataX[i][2] = speeds[i]
    dataX[i][3] = widths[i]
    dataX[i][4] = previous_distances[i]
    
    
#%% Run Decision Trees

#%%
#Unlimited Depth Tree
dataX_train = dataX
dataY_train = np.array(dataY)

tree_clf = tree.DecisionTreeClassifier(max_depth=None, criterion='entropy')
tree_clf.fit(dataX_train,dataY_train)

#%% Depth 10 Decision Tree

dataX_train = dataX
dataY_train = np.array(dataY)

tree_clf = tree.DecisionTreeClassifier(max_depth=10, criterion='entropy')
tree_clf.fit(dataX_train,dataY_train)

#%% Depth 5 Decision Tree

dataX_train = dataX
dataY_train = np.array(dataY)

tree_clf = tree.DecisionTreeClassifier(max_depth=5, criterion='entropy')
tree_clf.fit(dataX_train,dataY_train)

#%%Create test data

testLength = 500
testSpeeds = [0]*testLength
testWidths = [0]*testLength

for i in range (testLength):
    testSpeeds[i] = random.randint(18,25)
    testWidths[i] = random.randint(30,85)
    worksheet.write(i+1, 2, testSpeeds[i])
    worksheet.write(i+1, 3, testWidths[i])

testX = np.zeros([testLength,5])

testX[0][0] = 0
testX[0][1] = 0
testX[0][4] = 150
worksheet.write(1,0,0)
worksheet.write(1,1,0)
worksheet.write(1,4,150)

for i in range(len(testX)):
    testX[i][2] = testSpeeds[i]
    testX[i][3] = testWidths[i]
    if i>=1:
        testX[i][0] = testSpeeds[i-1]
        testX[i][1] = testWidths[i-1]
        worksheet.write(i+1, 0, testSpeeds[i-1])
        worksheet.write(i+1, 1, testWidths[i-1])

#%%Tree

# Predict on this data 
    
preds = [0]*testLength
for i in range(testLength):
    if i >= 1:
        testX[i][4] = preds[i-1]
        worksheet.write(i+1,4,preds[i-1])
    preds[i] = tree_clf.predict(testX[i].reshape(1,-1))
pred = list(preds)

for i in range(len(pred)):
    worksheet.write(i+1, 5, pred[i][0])

workbook.close()

#%% Linear Regression

dataX_train = dataX
dataY_train = np.array(dataY)

from sklearn import linear_model

clf = linear_model.LinearRegression()
clf.fit(dataX,dataY)

preds = [0]*testLength
for i in range(testLength):
    if i >= 1:
        testX[i][4] = preds[i-1]
        worksheet.write(i+1,4,preds[i-1])
    preds[i] = clf.predict(testX[i].reshape(1,-1))
pred = list(preds)

for i in range(len(pred)):
    worksheet.write(i+1, 5, pred[i][0])

workbook.close()
    
#%% Regression Trees

clf = tree.DecisionTreeRegressor()
clf.fit(dataX,dataY)

preds = [0]*testLength
for i in range(testLength):
    if i >= 1:
        testX[i][4] = preds[i-1]
        worksheet.write(i+1,4,preds[i-1])
    preds[i] = clf.predict(testX[i].reshape(1,-1))
pred = list(preds)

for i in range(len(pred)):
    worksheet.write(i+1, 5, pred[i][0])

workbook.close()

#%% Neural Net - Multi Layer Perceptron

from sklearn.neural_network import MLPRegressor
import numpy as np

nn = MLPRegressor(hidden_layer_sizes=(10), activation='tanh', solver='lbfgs')
#nn = MLPRegressor(hidden_layer_sizes=(3), activation='relu', solver='adam')

n = nn.fit(dataX, dataY)

preds = [0]*testLength
for i in range(testLength):
    if i >= 1:
        testX[i][4] = preds[i-1]
        worksheet.write(i+1,4,preds[i-1])
    preds[i] = nn.predict(testX[i].reshape(1,-1))
pred = list(preds)

for i in range(len(pred)):
    worksheet.write(i+1, 5, pred[i][0])

workbook.close()


#%% Visualize the trees

tree.export_graphviz(tree_clf,
    out_file='tree10.dot') 

