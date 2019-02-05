#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 14:39:25 2018

@author: hitesh
"""
import sys
import random
import math

datafile= "climate.data"#sys.argv[1] 
trainfile= "climate.trainlabels.0"#sys.argv[2] 
labelfile = 'climate.labels'
      
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
train={} 
l=train_f.readline()
while(l !=''):
    a=l.split()
    train[int(a[1])]=int(a[0])
    l=train_f.readline()
train_f.close()

t=open(labelfile,'r')

labels={} 
l=t.readline()
while(l !=''):
    a=l.split()
    labels[int(a[1])]=int(a[0])
    l=t.readline()
t.close()      

try:


    def sigmoidfunction(wt, data):
        dp = dotproduct(wt, data)
        sigmoid = 1 / (1 + math.exp(-1 * dp))
        if (sigmoid >= 1):
            sigmoid = 0.999999
        return sigmoid

    def dotproduct(w,data):
        dp = [x * y for x, y in zip(w, data)]
        return sum(dp)



    def predictwithl2(w,data,train,eta,earlystopping,lambda1,labels):

        print('W :', w)
        normw=0;
        for i in range(0,len(data[0])-1):
            normw += w[i]**2
       
        normw=math.sqrt(normw)        
        dist = w[len(w) - 1] / normw
        print('Origin distance: ' + str(dist))
        print('||w|| :', normw)
        #Prediction
        count=0.0
        for i in range(0,len(data),1):
                dp=dotproduct(w,data[i])
                if (dp < 0):
              
                    if(labels.get(i)!=0):
                        count=count+1

                else:

                    if(labels.get(i)!=1):
                        count=count+1


        errorcount=count/len(data)

        print('Prediction of test data using Logistic Discrimination Gradiant Decesent with L2 regularization \n')
      
        return(errorcount*100)

    def predictwithoutl2(w,data,train,eta,earlystopping,lambda1,labels):

        print('W :', w)

        normw=0;
        for i in range(0,len(data[0])-1):
            normw += w[i]**2

        normw=math.sqrt(normw)
        print('||w|| :', normw)
        dist = w[len(w) - 1] / normw
        print('Origin distance: ' + str(dist))
        #print('Prediction of test data using Logistic Discrimination Gradiant Decesent with L2 regularization')
        #Prediction
        count=0
        for i in range(0,len(data),1):
            if(train.get(i)== None):
                dp=dotproduct(w,data[i])
                if (dp < 0):
                    print("0 ",i)

                else:
                    print("1 ",i)

    def minimum(a, n): 
  
        minpos = a.index(min(a)) 
    
        return minpos + 1


    def crossvalidation(w,data,train,eta,earlystopping,labels):
        lambda1array = [0.0,0.25,0.5,1.0,1.5,2.0]
        splitarray = [.25,.50,.75]
        #for lambda1 in lambda1array:
        #    print("lambda",lambda1)
        #    for split in splitarray:
        #        print("Split",split)
        lambda_position ={1 : 0.0, 2 :0.25, 3: 0.5, 4: 1.0, 5: 1.5, 6: 2.0}
        testingerror=0.0        
        min_dict = {}
        split_lambda= {}
        train_error = {}
        cross_error = {}
        for split in splitarray:
            trainingerror = []
            testerror = []
            totalerror = []
            print("split",split)
            split_lambda[split] = []
            train_error[split] = []
            cross_error[split] = []
            for lambda1 in lambda1array:
                print("lambda",lambda1)
                train_data,test_data,train_labels,test_labels=partition(data,train,split)
                w2=gradiantdecesent(w,train_data,train_labels,eta,earlystopping,lambda1)
             
                if not(len(train_data) == 0):
                    trainerror=predictwithl2(w2,train_data,train_labels,eta,earlystopping,lambda1,labels)

                if not(len(test_data) == 0):

                    testingerror=predictwithl2(w2,test_data,test_labels,eta,earlystopping,lambda1,labels)
                #print(testingerror)
                trainingerror.append(trainerror)
                testerror.append(testingerror)
                totalerror.append(trainerror + testingerror)
                
                #print("Training Error",trainerror)
                #print("Testing Error",testingerror)
                #print("Total Error",(testingerror+trainerror))
                train_error[split].append(trainerror)
                cross_error[split].append(testingerror)
                split_lambda[split].append(trainerror + testingerror)
                min_dict[split] = testingerror+trainerror
        print("----- Train Error for Different Split----- \n")
        print('Split 0.25 : \n', train_error[0.25])
        print('Split 0.5 : \n ', train_error[0.5])
        print('Split 0.75 : \n', train_error[0.75],'\n')
        print("----- Crossvalidation Error for Different Split-----", "\n")
        print('Split 0.25 : \n', cross_error[0.25])
        print('Split 0.5 : \n', cross_error[0.5])
        print('Split 0.75 : \n', cross_error[0.75],"\n")
        print("----- Total Error for Different Split-----", "\n")
        print('Split 0.25 :\n ', split_lambda[0.25])
        print('Split 0.5 : \n', split_lambda[0.5])
        print('Split 0.75 : \n', split_lambda[0.75], "\n")
        print("Best Lambdas for Different Split in Order of [0.25, 0.5, 0.75]")
        for i in splitarray:
            pos = minimum(split_lambda[i], len(split_lambda[i]))
            print(lambda_position[pos])
        #print("Best lambda",min(min_dict, key=lambda k: min_dict[k]))

    def truelabels(data,):
        indices = [i for i, x in enumerate(data)]
        for i in indices:
            print( i, labels[i])


    def partition(data,label,split):

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
                w[j] += eta * -1 * (h[j] + h0[j])
            
            error=0
            
            cost=0

            for j in range(0, len(data)): 
                if (train.get(j) != None):
                    cost += (-1 * (train[j] * math.log(sigmoidfunction(w, data[j])) + ((1 - train[j]) * math.log(1 - sigmoidfunction(w, data[j])))))
            for i in range(0,len(data[0]),1):
                cost += (lambda1/2) * w[i]**2
            
            error = cost
            print("error:",error)

            diff = abs(J - error)

        return w


    def main(earlystopping,data,train,labels):
        
        # Step1: Initialize inital w
        w1 = []
        for j in range(0, len(data[0]), 1):
            w1.append(0)
        
        for j in range(0, len(data[0]), 1):
            w1[j] = w1[j] + .02 * random.random() - .01
            
        eta=0.001
         
        crossvalidation(w1, data, train,eta,earlystopping,labels)


    earlystopping=False;

    main(earlystopping,data,train,labels)



except OverflowError as error1:
    print("System couldnt handle large dataset")
    main(earlystopping=True)



