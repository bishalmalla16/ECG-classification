import numpy
import csv
import math
import operator
import random
import pickle

def accuracy(y,pred):
	count=0.0
	for i in range(0,len(y)):
		if(y[i]==pred[i]):
			count=count+1
	print (count)
	return count*100/len(y)




def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((x, dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

def getResponse(neighbors,Y):
	classVotes = {}
	for x in range(len(neighbors)):
		response =numpy.asscalar(Y[neighbors[x]]);
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def getAccuracy(testSet, predictions,Y):
		correct = 0
		for x in range(len(testSet)):
			if Y[x] == predictions[x]:
				correct += 1
				return (correct/float(len(testSet))) * 100.0


def loadDataset(split,X,Y, X_train=[] , Y_train=[],  X_test=[],  Y_test=[]):
    c=0
    for i in range(0,X.shape[0]):
        if c/X.shape[0] < 1-split and i%4==2 and Y[i]!=7 and Y[i]!=8 and Y[i]!=11 and Y[i]!=12:
            if c<15 and Y[i]==1:
                X_test.append(X[i])
                Y_test.append(Y[i])
                c=c+1
            elif c>=15:
                X_test.append(X[i])
                Y_test.append(Y[i])
                c=c+1
            else:
                X_train.append(X[i])
                Y_train.append(Y[i])
        else:
            X_train.append(X[i])
            Y_train.append(Y[i])
    return c

#create reduced feature matrix
reader=csv.reader(open("reduced_features.csv","r"),delimiter=",")
X=list(reader)
X=numpy.array(X)
X=X.astype(numpy.float)


#create result vector
reader=csv.reader(open("target_output.csv","r"),delimiter=",")
Y=list(reader)
Y=numpy.array(Y)
Y=Y.astype(numpy.int)

no_of_columns=X.shape[1]

X_train=[]
Y_train=[]
X_test=[]
Y_test=[]

c=loadDataset(0.8,X,Y, X_train , Y_train,  X_test,  Y_test)

predictions=[]
k = 4

#get k nearest neighbors and predict outcome
for i in range(0,len(X_test)):
	neighbors = getNeighbors(X_train,X_test[i],k)
	result = getResponse(neighbors,Y_train)
	predictions.append(result)
#print (predictions)

print ('Total test size = '+str(len(predictions)))
print ('Test Accuracy = ' +str(accuracy(Y_test,predictions)))

#for confusion matrix
from sklearn.metrics import confusion_matrix

y_obs=[]

for i in range(0,X.shape[0]):
	neighbors = getNeighbors(X_train,X[i],k)
	result = getResponse(neighbors,Y_train)
	y_obs.append(result)


cm = confusion_matrix(y_obs, Y)
print('...................Confusion Matrix....................')
print(cm)

'''
from sklearn.metrics import classification_report
print(classification_report(Y, y_obs))
