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
workbook = xlsxwriter.Workbook('dinoPreds_new_vfast.xlsx')
worksheet = workbook.add_worksheet()

worksheet.write('A1', 'Speed')
worksheet.write('B1', 'Width')
worksheet.write('C1', 'Distance')

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
workbook = xlsxwriter.Workbook('dinoPreds_new_fast.xlsx')
worksheet = workbook.add_worksheet()

worksheet.write('A1', 'Speed')
worksheet.write('B1', 'Width')
worksheet.write('C1', 'Distance') #This is the predicted distance that the dino should jump

combinedData = pd.read_csv('combined_dinoData_new_Multiple.csv',header = 0)

dataY = list(combinedData['Distance0']) #distance that the dino used for first set

previousSpeeds = list(combinedData['Speed0'])
previousWidths = list(combinedData['Width0'])

speeds = list(combinedData['Speed1'])
widths = list(combinedData['Width1'])

dataX = np.zeros([len(dataY),4])

for i in range(len(dataX)):
    dataX[i][0] = previousSpeeds[i]
    dataX[i][1] = previousWidths[i]
    dataX[i][2] = speeds[i]
    dataX[i][3] = widths[i]
    
#%%Create test data

testLength = 250
testSpeeds = [0]*testLength
testWidths = [0]*testLength

#ramp from 3 to 12 then go back down to 3
dist = 18
for i in range (testLength):
    if(dist == 26):
        dist = 18

    testSpeeds[i] = dist
    dist += 1

for i in range (testLength):
    testWidths[i] = random.randint(30,85)
    worksheet.write(i+1, 0, testSpeeds[i]) #current speed 
    worksheet.write(i+1, 1, testWidths[i]) #current width
    
testX = np.zeros([testLength,4])

for i in range(len(testX)):
    
    if i < len(testX) - 1: 
        testX[i][0] = testSpeeds[i]
        testX[i][1] = testWidths[i]
        testX[i][2] = testSpeeds[i+1] #future speed 
        testX[i][3] = testWidths[i+1] #future width
    
    else: 
        testX[i][0] = testSpeeds[i]
        testX[i][1] = testWidths[i]
        testX[i][2] = 0 #future speed 
        testX[i][3] = 0 #future width
    
    

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


#%%Tree

# Predict on this data 
    
preds = tree_clf.predict(testX)
pred = list(preds)

for i in range(len(pred)):
    worksheet.write(i+1, 2, pred[i])

workbook.close()

#%% Linear Regression

dataX_train = dataX
dataY_train = np.array(dataY)

from sklearn import linear_model

clf = linear_model.LinearRegression()
clf.fit(dataX,dataY)

predictions = clf.predict(testX)   
pred = list(predictions)

for i in range(len(pred)):
    worksheet.write(i+1, 2, pred[i])

#%% Regression Trees

clf = tree.DecisionTreeRegressor()
clf.fit(dataX,dataY)

preds = clf.predict(testX)
pred = list(preds)

for i in range(len(pred)):
    worksheet.write(i+1, 2, pred[i])

workbook.close()


#%% Neural Net- Single Representation

from sklearn.neural_network import MLPRegressor
import numpy as np

#nn = MLPRegressor(hidden_layer_sizes=(10), activation='tanh', solver='lbfgs')
nn = MLPRegressor(hidden_layer_sizes=(10), max_iter = 200, activation='relu', solver='adam')

n = nn.fit(dataX, dataY)

preds = nn.predict(testX)
pred = list(preds)

for i in range(len(pred)):
    worksheet.write(i+1, 2, pred[i])

workbook.close()


#%% Visualize the trees

tree.export_graphviz(tree_clf,
    out_file='tree10.dot') 

