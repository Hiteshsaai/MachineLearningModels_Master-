#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 15:40:46 2018

@author: hitesh
"""

import sys
import random
import math


datafile = sys.argv[1]
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

labelfile = sys.argv[2] 
lFile = open(labelfile, 'r')
trainlabels = {}
for line in lFile:
    row = line.split()
    if row[0] == '0':
        row[0] = '-1'
    trainlabels[int(row[1])] = int(row[0])
lFile.close()


w = []
for j in range(0, cols, 1):
    w.append(0.02 * random.random() - 0.01)



def dot_product(a, b):
    dp = 0
    for i in range(0, cols, 1):
        dp += a[i] * b[i]
    return dp

# eta = 0.0001 and ( diff > 0.000000001 or diff > 0.001 )  for ionosphere 
# eta = 0.000000001      #### eta for breast cancer #####
eta = 0.0001
hingloss = rows * 10
diff = 1
count = 0

while ((diff) > 0.001):
    dellf = [0] * cols
    for j in range(0, rows, 1):
        if (trainlabels.get(j) != None):
            dp = dot_product(w, data[j])
            condition = (trainlabels.get(j) * (dot_product(w, data[j])))
            for k in range(0, cols, 1):
                if (condition < 1):
                    dellf[k] += -1 * ((trainlabels.get(j)) * data[j][k])
                else:
                    dellf[k] += 0      
    ## update w 
    for j in range(0, cols, 1):
        w[j] = w[j] - eta * dellf[j]
    prev = hingloss
    hingloss = 0
    ## compute hinge loss 
    for j in range(0, rows, 1):
        if (trainlabels.get(j) != None):

            hingloss += max(0, 1 - (trainlabels.get(j) * dot_product(w, data[j])))
        diff = abs(prev - hingloss)

normw = 0
for i in range(0, (cols - 1), 1):
    normw += w[i] ** 2
print ("w ",w[:2])
print("w0", w[2])


normw = math.sqrt(normw)

d_orgin = abs(w[len(w) - 1] / normw)

print ("Distance to origin = " + str(d_orgin))

for i in range(0, rows, 1):
    if (trainlabels.get(i) == None):
        dp = dot_product(w, data[i])
        if (dp > 0):
            print("1 " + str(i))
        else:
            print("0 " + str(i))
