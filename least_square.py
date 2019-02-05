#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 03:36:37 2018

@author: hitesh
"""

import sys
import random
import math

# Read Data 

datafile = 'ionosphere.data' #sys.argv[1]
dFile = open(datafile, 'r')
data=[]
for line in dFile:
    row = line.split()
    rVec = [float(item) for item in row]
    data.append(rVec)
dFile.close()
for i in range(0,len(data)):
    data[i].append(1.0)

rows = len(data)
cols = len(data[0])

labelfile = 'ionosphere.trainlabels.0' # sys.argv[1] 
lFile = open(labelfile, 'r')
trainlabels = {}
for line in lFile:
    row = line.split()
    if row[0] == '0':
        row[0] = '-1'
    trainlabels[int(row[1])] = int(row[0])
lFile.close()

# initialize w 
w = []
for j in range(0, cols, 1):
    w.append(0.02 * random.random() - 0.01)

def dot_product(a, b):
    dp = 0
    for i in range(0, cols, 1):
        dp += a[i] * b[i]
    return dp


def cost_function(w, data, label):
    error = 0
    for j in range(0, rows, 1):
        if (trainlabels.get(j) != None):
            error += (trainlabels.get(j) - dot_product(w, data[j])) ** 2
    return error

#gradient descent iteration 
eta = 0.0001
error = 0 #len(data) + 10
diff = 1
count = 0
while ((diff) > 0.0001):
    dellf = []
    for m in range(0, cols, 1):
        dellf.append(0)
    for j in range(0, rows, 1):
        if (trainlabels.get(j) != None):
            dp = dot_product(w, data[j])
            for k in range(0, cols, 1):
                dellf[k] += (trainlabels.get(j) - dp) * data[j][k]
    for j in range(0, cols, 1):
        w[j] = w[j] + eta * dellf[j]

    prev = error
    error = 0

    error =  cost_function(w,data, trainlabels)        
    if (prev > error):
        diff = prev - error
    else:
        diff = error - prev
    count = count + 1
    #if (count % 100 == 0):
    #    print(error)

normw = 0
for i in range(0, (cols - 1), 1):
    normw += w[i] ** 2
    
print("w", w)

normw = math.sqrt(normw)
d_orgin = abs(w[len(w) - 1] / normw)
print ("Distance to origin = " + str(d_orgin))

# Prediction 
for i in range(0, rows, 1):
    if (trainlabels.get(i) == None):
        dp = dot_product(w, data[i])
        if (dp > 0):
            print("1 " + str(i))
        else:
            print("0 " + str(i))