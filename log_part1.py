#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 13:35:34 2018

@author: hitesh
"""
import sys
import random
import math

datafile= "climate.data"#sys.argv[1] 
trainfile= "climate.trainlabels.0"#sys.argv[2] 
        
dFile = open(datafile, 'r')
data=[]
for line in dFile:
    row = line.split()
    rVec = [float(item) for item in row]
    data.append(rVec)
dFile.close()
for i in range(0,len(data)):
    data[i].append(1)

train_f=open(trainfile,'r')

label={} 
l=train_f.readline()
while(l !=''):
    a=l.split()
    label[int(a[1])]=int(a[0])
    l=train_f.readline()
train_f.close()

try:


    def sigmoidfunction(wt,data):
        dp = dotproduct(wt, data)
        sigmoid = 1 / (1 + math.exp(-1 * dp))
        if (sigmoid >= 1):
            sigmoid = 0.999999
        return sigmoid

    def dotproduct(w,data):
        dp = [x * y for x, y in zip(w, data)]
        return sum(dp)

    def predict(w,data,train,eta,earlystopping,lambda1):

        print('W0 is ' + str(w[0]))
        print('W1 is ' + str(w[1]))

        normw=0;
        for i in range(0,len(data[0])-1):
            normw += w[i]**2

       
        normw=math.sqrt(normw)

        d_origin =(w[len(w)-1]/normw)
        print('The Distance to origin is ' +str(d_origin))
        print('w is ' + str(w))
        print("||W||", normw)
        
        #Prediction
        for i in range(0,len(data),1):
            if(train.get(i)== None):
                dp=dotproduct(w,data[i])
                if (dp < 0):
                    print("0 ",i)
        
                else:
                    print("1 ",i)
        
    def split_data(data,label,split):
        train_data = data[:int((len(data))*split)]
         
        test_data = data[int(len(data)*split):] 
        
        label_keys = list(label.keys())
        
        count1= []
        for i in range(0, len(label_keys[:int((len(label_keys))*split)]) ):
            count1.append(i)
            
        count2 = []
        for i in range(len(label_keys[:int((len(label_keys))*split)]),len(label) ):
            count2.append(i)
        
        tr_label = label_keys[:int((len(label_keys))*split)]
        tst_label = label_keys[int((len(label_keys))*split):]
        
        train_labels = {key: label.get(key) for key in tr_label}
        test_labels = {key: label.get(key) for key in tst_label}
    
    
        return train_data, test_data, train_labels, test_labels


    def gradiantdecesent(w,data, train,eta,earlystopping,lambda1):

        converged=False;
        h = []
        for j in range(0, len(data[0]), 1):
            h.append(0)
        h0 = []
        for j in range(0, len(data[0]), 1):
            h0.append(0)
            
        J = 100000000
        error = J - 10
        error = 0
        diff = 1

        while (diff > 0.001):
            J = error
            for j in range(0, len(data[0]), 1):
                h[j] = 0
                h0[j] = 0
            for i in range(0,len(data),1):
                if(train.get(i)!= None):
                    
                    pdict1 = (1 / (1 + (math.exp(-1 * dotproduct(w, data[i])))))
                    for j in range(0,len(data[0])):
                        h[j] += (( pdict1 - train.get(i) )*data[i][j]) + lambda1 * w[j]# for w
                        
            for j in range(0, len(data[0]), 1):
                w[j] += eta * -1 * (h[j] + h0[j] ) 

            error=0
            cost=0

            for j in range(0, len(data)): 
                if (train.get(j) != None):
                    cost += (-1 * (train[j] * math.log(sigmoidfunction(w, data[j])) + ((1 - train[j]) * math.log(1 - sigmoidfunction(w, data[j])))))
            for i in range(0,len(data[0]),1):
                cost += (lambda1/2) * w[i]**2
            
            error = cost
            diff = abs(J - error)        
           
        return w


    def main(earlystopping,data,train):
        
        w1 = []
        for j in range(0, len(data[0]), 1):
            w1.append(0)
        
        for j in range(0, len(data[0]), 1):
            w1[j] = w1[j] + .02 * random.random() - .01

        eta=0.001 #learning rate
        lambda1 = 0 #lambda 
        w2=gradiantdecesent(w1,data,train,eta,earlystopping,lambda1)
        prediction=predict(w2,data,train,eta,earlystopping,lambda1)
        
    earlystopping=False;

    main(earlystopping,data,label)


except OverflowError as error1:
    print("System not able to handle large dataset")
    main(earlystopping=True)
    


