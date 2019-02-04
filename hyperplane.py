import sys
from math import sqrt
from sklearn import svm
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import cross_val_score
import numpy as np

import random

def dotProduct(w, x):
    dp = 0.0
    for wi, xi in zip(w, x):
        dp += wi * xi
    return dp


def sign(x):
    if (x > 0):
        return 1
    elif (x < 0):
        return -1
    return 0

# Calculating best parameter for LinearSVC Classifier
def lsvm(training_data, training_target):
    print("Calculating best parameter for LinearSVC Classifier ...")
    clist = 2**np.array(range(-2, 10), dtype='float')
    cvscores = []
    for c in clist:
        print(c)
        clf= LinearSVC(C=c,dual=False)
        scores = cross_val_score(clf, training_data, training_target, cv=2)
        print("score", scores)
        cvscores.append(scores.mean()*100)
        bestscore, bestC = max([(val, clist[idx]) for (idx, val) in enumerate(cvscores)])
    print('Best CV accuracy =', round(bestscore,2), '% achieved at C =', bestC)

    # Retrain on whole trainning set using best C value obtained from Cross validation
    print("Retrain on whole trainning set using best C value obtained from Cross validation")
    clf = LinearSVC(C=bestC)
    clf.fit(training_data, training_target)
    #accu = clf.score(testing_data, testing_target)*100
    accu = clf.score(training_data, training_target)*100
    return [clf, accu, bestC]

#datafile= sys.argv[1] #"breastcancer.txt" #
#dataFile = "datasset.txt"
#tngLblFile = "traininglabels.txt"
#testDataFile = "input_data"

dataFile= "breast_cancer.data"
training_File = "breast_cancer.trainlabels.0"
#testDataFile = "breast_cancer.trainlabels.1"


#dataFile = sys.argv[1]
#tngLblFile = sys.argv[2]
#testDataFile = sys.argv[3]
#k = int(sys.argv[4])

# Read training labels file
'''labels = []
with open(tngLblFile) as infile:
    labels = list(map(lambda x: int(x.split()[0]), infile.readlines()))'''

dataSets = []
with open(dataFile) as f:
    for row in f:
        rowArray = list(map(float, row.split()))
        dataSets.append(rowArray)
f.close()

train_f=open(training_File,'r')
train_label={} 
l=train_f.readline()
while(l !=''):
    a=l.split()
    train_label[int(a[1])]=int(a[0])
    l=train_f.readline()
train_f.close()

train_label_ind = []
for i in train_label.keys():
    train_label_ind.append(i)

training_label = []
for i in train_label.values():
    training_label.append(i)
    
#train_label_ind = list(train_label.keys())

train_data = []
for i in train_label_ind:
    train_data.append(dataSets[i])    
    
test_data = []
for i in dataSets:
    if i not in train_data:
        test_data.append(i)

noCols = len(train_data[0])

planes=[10, 100,1000,10000]

for k in planes:

    w = []
    for i in range(0, k, 1):
        w.append([])
        for j in range(0, noCols, 1):
            w[i].append(random.uniform(-1, 1))

    #print ("random w " + str(w))

    z = []
    for i, data in enumerate(dataSets):
        z.append([])
        for j in range(0, k, 1):
            z[i].append(sign(dotProduct(w[j], data)))

    #print ("z " + str(z))
    #print ("tngLabels " + str(labels))

    z1 = []
    for i, data in enumerate(test_data):
        z1.append([])
        for j in range(0, k, 1):
            z1[i].append(sign(dotProduct(w[j], data)))

    #print ("z1 " + str(z1))

    

    print('\n ############## Using_RandomHyperPlanes k=',k,'###################\n')
    rresult= lsvm(z, training_label)
    rmodel = rresult[0]
    print('Training Accuracy with Linear SVM with Best C=',round(rresult[2],2),': ', rresult[1])
    prediction1=rmodel.predict(z1)
    print('\n ############## Predicted labels Using_RandomHyperPlanes ################### \n')
    for i in range(len(prediction1)):
        #print(str(int(prediction[i])) + ' ' + str(i) + '\n')
        print(int(prediction1[i]), i)




    print('\n ############## Using_OriginalDataPoints ################### \n')
    oresult= lsvm(dataSets, training_label)
    omodel = oresult[0]
    print('Training Accuracy with Linear SVM with Best C=',round(oresult[2],2),': ', oresult[1])
    prediction2=omodel.predict(test_data)

    print('\n ############## Predicted labels Using_OriginalDataPoints ################### \n')
    for i in range(len(prediction2)):
        #print(str(int(prediction[i])) + ' ' + str(i) + '\n')
        print(int(prediction2[i]), i)


