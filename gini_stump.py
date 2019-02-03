#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 19:58:15 2018

@author: hitesh
"""

import sys
import random
import math

datafile = sys.argv[1] 
file = open(datafile)
data = []
l = file.readline()
while (l != ''):
    a = l.split()
    l2 = []
    for j in range(0, len(a), 1):
        l2.append(float(a[j]))
    data.append(l2)
    l = file.readline()
datarows = len(data)
datacols = len(data[0])
file.close()

labelfile = sys.argv[2]
labels = {}
file = open(labelfile)
l = file.readline()
classes = []
classes.append(0)
classes.append(0)
while (l != ''):
    a = l.split()
    labels[int(a[1])] = int(a[0])
    classes[int(a[0])] = classes[int(a[0])] + 1
    l = file.readline()
file.close()

lableFile = open(labelfile, 'r')
labels = {}
for line in lableFile:
    row = line.split()
    labels[int(row[1])] = int(row[0])
lableFile.close()

ginivals = []
split = 0
l3 = [0, 0]
for j in range(0, datacols, 1):
    ginivals.append(l3)
temp = 0
col = 0

for j in range(0, datacols, 1):
    listcol = [item[j] for item in data]
    keys = sorted(range(len(listcol)), key=lambda k: listcol[k])
    listcol.sort()
    ginival = []
    prevgini = 0
    prevrow = 0
    for k in range(1, len(labels), 1):
        lsize = k
        rsize = datarows - k
        lp = 0
        rp = 0
        for l in range(0, k, 1):
            if (labels[keys[l]] == 0):
                lp += 1
        for r in range(k, len(labels), 1):
            if (labels[keys[r]] == 0):
                rp += 1
        gini = (lsize / datarows) * (lp / lsize) * (1 - lp / lsize) + (rsize / datarows) * (rp / rsize) * (
            1 - rp / rsize)
        ginival.append(gini)

        prevgini = min(ginival)
        if (ginival[k - 1] == float(prevgini)):
            ginivals[j][0] = ginival[k - 1]
            ginivals[j][1] = k
    if (j == 0):
        temp = ginivals[j][0]
    if (ginivals[j][0] <= temp):
        temp = ginivals[j][0]
        col = j
        split = ginivals[j][1]
        if (split != 0):
            split = (listcol[split] + listcol[split - 1]) / 2
print("gini:", temp, "column:", col, "split:", split)

