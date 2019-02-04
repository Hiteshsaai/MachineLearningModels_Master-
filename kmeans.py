#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 15:11:22 2018

@author: hitesh
"""
import sys
import random

datafile = "ionosphere.data" #sys.argv[1]

f = open(datafile, 'r')
data = []
# i = 0
l = f.readline()
while (l != ''):
    a = l.split()
    l2 = []
    for j in range(0, len(a), 1):
        l2.append(float(a[j]))
    data.append(l2)
    l = f.readline()

rows = len(data)
cols = len(data[0])
f.close()

try:
    k = int(3)#sys.argv[2])
except IndexError:
    print(" Try this foramte python3 kmeans.py filename 3")
    sys.exit()

m_clust = []
col = []
for j in range(0, cols, 1):
    col.append(0)

for i in range(0, k, 1):
    m_clust.append(col)

random1 = 0
for p in range(0, k, 1):
    random1=random.randrange(0,(rows-1))
    m_clust[p] = data[random1]

trainlabels = {}
diff = 1

prev = [[0]*cols for x in range(k)]

dist =[]

m_dist =[]
for p in range(0, k, 1):
    m_dist.append(0)
n = []
for p in range(0, k, 1):
    dist.append(0.1)
for p in range(0, k, 1):
    n.append(0.1)
totaldist =1
classes=[]

while ((totaldist) > 0):
    for i in range(0,rows, 1):
        dist =[]
        for p in range(0, k, 1):
            dist.append(0)
        for p in range(0, k, 1):
            for j in range(0, cols, 1):
                dist[p] += ((data[i][j] - m_clust[p][j])**2)
        for p in range(0, k, 1):
            dist[p] = (dist[p])**0.5
        mindist=0
        mindist = min(dist)
        for p in range(0, k, 1):
            if(dist[p]==mindist):
                trainlabels[i] = p
                n[p]+=1
                break

    m_clust = [[0]*cols for x in range(k)]
    col = []

    for i in range(0, rows, 1):
        for p in range(0, k, 1):
            if(trainlabels.get(i) == p):
                for j in range(0, cols, 1):
                    temp =  m_clust[p][j]
                    temp1 =  data[i][j]
                    m_clust[p][j] = temp + temp1
    for j in range(0, cols, 1):
        for i in range(0, k, 1):
            m_clust[i][j] = m_clust[i][j]/n[i]

    classes = [int(x) for x in n]
    n=[0.1]*k
    m_dist = []
    for p in range(0, k, 1):
        m_dist.append(0)
    for p in range(0, k, 1):
        for j in range(0, cols, 1):
            m_dist[p]+=float((prev[p][j]-m_clust[p][j])**2)
        m_dist[p] = (m_dist[p])**0.5
    prev=m_clust
    totaldist = 0
    for b in range(0,len(m_dist),1):
        totaldist += m_dist[b]

    #print ("distance between means:",totaldist)
#print("data in each cluster for k =",k,"is",classes)
#print ("mdist = " + str(mdist))

for i in range(0,rows, 1):
    print(trainlabels[i],i)