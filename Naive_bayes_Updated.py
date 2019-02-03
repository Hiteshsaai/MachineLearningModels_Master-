"""
Created on Tue Sep 18 14:36:40 2018

@author: hitesh
"""
#------------------- Naive Bayes Classifier -----------------------------------
# -----------------------------------------------------------------------------
# ----In this Program I have predicted the Test labels using the Test data-----

import random as rnd
import math

def data_preparation(filename_data, filename_label):
    dataset= open(filename_data,'r')
    dataset = [val.split() for val in dataset]
    dataset = [[float(val) for val in num]for num in dataset]
    labels = open(filename_label,'r')
    labels = [lab.split() for lab in labels]
    labels = [[int(val) for val in num] for num in labels]
    return dataset , labels

def splitdata_based_on_trainlabel(data , trainlabel):
    train_label = open(trainlabel, 'r')
    train_label = [lab.split() for lab in train_label]
    train_label = [[int(val) for val in num] for num in train_label]
    
    train_data = []
    for i in range(0, len(train_label)):
        index = train_label[i][1]
        train_data.append(data[index])
    
    train_index= []
    for j in range(0, len(train_label)):
        train_index.append(train_label[j][1])
    tot_index = []
    for k in range(0, len(data)):
        tot_index.append(k)    
    
    for l in range(0, len(train_data)):
        if train_index[l] in tot_index:
            tot_index.remove(train_index[l])
        else:
            None
    test_data = []
    for m in range(0, len(tot_index)):
        test_data.append(data[tot_index[m]])
        
    return train_data, train_label,test_data


def split_dataset(data, labels, ratio):
    train_size = int(len(data) * ratio)
    train_set = []
    train_labels = []
    test_set = data.copy()
    test_labels = labels.copy()

    while len(train_set) != train_size:
        index = rnd.randrange(0, len(test_set))
        train_set.append(test_set.pop(index))
        train_labels.append(test_labels.pop(index))

    return train_set, train_labels, test_set, test_labels


def mean(data, classes):
    size = len(data)
    labels = [[row[i] for row in classes] for i in range(0, len(classes[0]))]
    labels = set(labels[0])
    mean = []
    for label in labels:
        temp = []
        for i in range(0, size):
            if classes[i][0] == label:
                temp.append(data[i])
        sz = len(temp)
        mn = [0.000000001 + (float(sum(l))/(sz))  for l in zip(*temp)]
        mean.append(mn)
    return mean
#labels = [[row[i] for row in classes] for i in range(0, len(classes[0]))]
#set(labels[0])
#classes[247][0] == set(labels[0])

def variance(data, classes):
    size = len(data)
    labels = [[row[i] for row in classes] for i in range(0, len(classes[0]))]
    labels = set(labels[0])
    mean_new = mean(data, classes)
    variance = []
    index = 0
    for label in labels:
        temp = []
        for i in range(0, size):
            if classes[i][0] == label:
                temp.append(data[i])
        mn = mean_new.pop(index)
        tvar = []
        for i in range(0, len(temp[0])):
            var = 0
            for j in range(0, len(temp)):
                var += (temp[j][i] - mn[i]) ** 2
            #var = float(var / ((len(temp)-1)))
            var = math.sqrt(float(var/ len(data)))
            tvar.append(var)
        variance.append(tvar)
    return variance

def prediction(data, test_data, train_label, cmean, cvariance):
    predicted_labels = []
    labels = [[row[i] for row in train_label] for i in range(0, len(train_label[0]))]
    labels = set(labels[0])
    lbl = list(labels)
    for i in range(0, len(test_data)):
        temp = test_data[i]
        minimum_dist = []
        for j in range(0, len(cmean)):
            dist = 0
            for k in range(0, len(temp)):
                dist += ((temp[k] - cmean[j][k]) / cvariance[j][k]) ** 2
            minimum_dist.append(dist)
        
        min_index = min(range(0, len(minimum_dist)), key=minimum_dist.__getitem__)
        predicted_labels.append(lbl[min_index])
    train_index= []
    for j in range(0, len(train_label)):
        train_index.append(train_label[j][1])
    tot_index = []
    for k in range(0, len(data)):
        tot_index.append(k)    
    
    for l in range(0, len(train_data)):
        if train_index[l] in tot_index:
            tot_index.remove(train_index[l])
        else:
            None
    predicted_labels = [[i] for i in predicted_labels]
    for m in range(0,len(tot_index)):
        predicted_labels[m].append(tot_index[m])
    return predicted_labels

def accuracy(true_labels, output_labels):
    size = len(true_labels)
    correct = 0
    for i in range(0, size):
        if true_labels[i][0] == output_labels[i]:
            correct += 1
    return float(correct/size) * 100


# ---------Enter the data name and labels name for the input-------------------
# You can enter the different file name and trainglabel data here for testing the Naive Bayes Classifier     
'''file_data = 'breast_cancer.data'# input("Enter the input data file name: ")
file_label = input("Enter the input lable file name: ")
file_train_labels = 'breast_cancer.trainlabels.0' #input("Enter the Respective Train Label: ")'''

import sys
file_data = sys.argv[1] 
file_train_labels = sys.argv[2] 


# Importing the datasets for the data preparation
datas , train_label = data_preparation(file_data, file_train_labels)

#Splitting the dataset in to the training and testing data
'''train_data, train_label,test_data, test_label = 
                                    split_dataset(datas, labels, 0.8)'''

train_data, train_label,test_data = splitdata_based_on_trainlabel(datas , file_train_labels)

# Mean Calculation 
cmean = mean(train_data, train_label)

# Variance Calculation
cvariance = variance(train_data, train_label)

'''# Prediction for the training data 
#train_output = prediction(train_data, train_label, cmean, cvariance )
print("Training Data Accuracy is: ", accuracy(train_label, test_output))'''

# Prediction for the test data
test_output = prediction(datas, test_data, train_label, cmean, cvariance)
print("The Predicted Test_label Output is:\n", test_output)


 #Importing the true labels
#labels = 'ionosphere.labels'
#labels = open(labels,'r')
#labels = [lab.split() for lab in labels]
#labels = [[int(val) for val in num] for num in labels]

# Accuracy Testing 

#correct = 0
#for i in range(len(test_output)):
#    if labels[test_output[i][1]][0] == test_output[i][0]:
#        correct += 1
#    else:
#        None 
#    acc = correct / len(test_output)  * 100  
    
    
    