#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 23:24:04 2018

@author: hitesh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 23:11:03 2018

@author: hitesh
"""

import math
from decimal import Decimal
import random as rnd
import sys 

def dot(W, data):
    res = [x * y for x, y in zip(W, data)]
    return sum(res)

def cross_entropy(W,data, labels):
    cost=0
    for i in range(0,len(data)):
        z = 0.
        z = dot(W,data[i])+1
        if(labels[i][0]!= None):
            try:
                p = Decimal(1 / (1 + Decimal(math.exp(-1 * z))))
            except OverflowError as e:
                p = float("inf")
                if p == 0 or math.isnan(p):
                    p=0.00001  
                if p == 1 or math.isinf(p):
                    p=0.99999
                cost += -1 * (labels[i][0] * math.log(p) + ((1 - labels[i][0]) * math.log(1 - p)))

    return cost



def cost_function(W,x, y):
    cost = sum([(dot(W, x[i]) - y[i][0])**2 for i in range(0, len(x))])
    
    return cost
    
def gradient_descent(data, labels, eta = 0.001, stp_val = 0.001, max_iter = 1000):
    #val = 1.0/len(data[0])
    #W = [val for i in range(0, len(data[0]))]
    W = []
    for i in range(0, 3):
        W.append(0.02 * rnd.uniform(0, 1) - 0.01)
    converged = False
    T = cross_entropy(W, data, labels)
    iteration = 0
    while not converged:
        pdict = [(dot(W, data[i]) - labels[i][0]) for i in range(0, len(data))]

        temp=[[(data[i][j]*pdict[i])for j in range(0,len(data[0]))] \
                for i in range(0,len(data))]

        grad =  [sum(l) for l in zip(*temp)]

        weight = [(W[i] - eta * grad[i]) for i in range(0, len(W))]
        W = weight

        error = cross_entropy(W, data, labels)

        if abs(T - error) <= stp_val:
            converged = True
        T = error
        iteration += 1
        if iteration == max_iter:
            converged = True
    return W
    
if __name__ == '__main__':
    file_name_data = sys.argv[1]
    file_name_labels = sys.argv[2]
    data_set = open(file_name_data, 'r')
    labels_set = open(file_name_labels, 'r')

    data = [line.split() for line in data_set]
    data = [[float(column) for column in row] for row in data]
    for i in range(0, len(data)):
        data[i].append(1)

    labels = [line.split() for line in labels_set]
    labels = [[int(column) for column in row] for row in labels]
    for i in range(0, len(labels)):
        if labels[i][0] == 0:
            labels[i][0] = -1
    
    
    W = gradient_descent(data, labels)
    print("Computed Vector Values [W1, W2] are: \n",W[:-1])
    W0 = W[len(W)-1]
    magnitude = math.sqrt(sum([W[i]**2 for i in range(0, len(W)-1)]))
    print("The Distance to Origin is: \n", abs(W0/magnitude))

    