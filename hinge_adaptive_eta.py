#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 13:54:54 2018

@author: hitesh
"""

import sys
import random

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

labelfile = sys.argv[2]
lFile = open(labelfile, 'r')
trainlabels = {}
for line in lFile:
    row = line.split()
    if row[0] == '0':
        row[0] = '-1'
    trainlabels[int(row[1])] = int(row[0])
lFile.close()

rows = len(data)
cols = len(data[0])

##print(trainlabels)
#print(trainlabels)

##initialize w

w = []

for j in range(cols):
    w.append(0)
    w[j] = (0.02 * random.uniform(0,1)) - 0.01

#    w[j] = 1
##define function dot_product

def dp(wt, data):
    dp = 0
    for i in range(0, cols, 1):
        dp += w[i] * data[i]
    return dp

##gradient descent iteration
##calculate error outside the loop
#initialize flag and iteration parameters

a = 0
k=0
while(a != 1):
    k+=1
    delf = []
    for i in range(cols):

        delf.append(0)
    for i in range(rows):
        if(trainlabels.get(i) != None):
            dot_product = dp(w, data[i])
            for j in range (cols):
                if(dot_product*trainlabels.get(i)<1):
                    delf[j]+=-1*data[i][j]*trainlabels.get(i)
                else:
                    delf[j]+=0

    eta_list = [1, .1, .01, .001, .0001, .00001, .000001, .0000001, .00000001, .000000001, .0000000001, .00000000001]
    bestobj = 1000000000000
    for k in range(0, len(eta_list), 1):
        eta = eta_list[k]
        for j in range(0,cols,1):
            w[j] = w[j] - eta*delf[j]
        error = 0.0
        for i in range(rows):
            if (trainlabels.get(i) != None):
                error += max(0, 1 - trainlabels.get(i) * dp(w, data[i]))
#compute gradient

        obj = error
        if (obj < bestobj):
            best_eta = eta
            bestobj = obj

        for j in range(cols):
            w[j] = w[j] + eta*delf[j]

    #print("Besteta:",best_eta)
    eta =best_eta
    for j in range(cols):
        w[j] =w[j] - eta*delf[j]

    curr_error = 0
    for i in range (rows):
        if(trainlabels.get(i) != None):
            curr_error += max( 0,1-trainlabels.get(i)*dp(w,data[i]))

    #print(error,k)
    if error - curr_error < 0.001:
        a = 1
    error = curr_error

## calculate differences in error:1
#print("w =",w)

normw = 0
for j in range((cols-1)):

    normw += w[j]**2

    #print(w[j])
normw = (normw)**0.5

#print("||w||=", normw)
d_origin = w[(len(w)-1)]/normw
#print(d_origin)

for i in range(rows):
    if(trainlabels.get(i) == None):
        dot_product = dp(w, data[i])
        if(dot_product > 0):
            print('1'+' '+str(i))
        else:
            print('0'+' '+str(i))
