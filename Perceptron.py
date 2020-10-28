import numpy as np 
import operator
import Helper as hp

def perceptron(Xtrain, Ytrain, maxUpdates):
    allLabels = {}
    labelList = []
    for y in Ytrain:
        labelList.append(y)
    for label1 in labelList:
        for label2 in labelList:
            if label1 == label2 or (label1,label2) in allLabels:
                continue
            tempXtrain,tempYtrain = hp.makeLinerPredictorData(Xtrain, Ytrain, label1, label2)
            w = calculateW(tempXtrain,tempYtrain,maxUpdates)
            allLabels[(label1,label2)] = w
            allLabels[(label2,label1)] = (-1)*w
    return allLabels

def predictPerceptron(wDic, Xtest):
    result = []
    for x in Xtest:
        maxDotCount = {}
        for labelTu,w in wDic.items():
            if labelTu[0] not in maxDotCount:
                maxDotCount[labelTu[0]] = 0
            if np.dot(x,w) >= 0 :
                maxDotCount[labelTu[0]] += 1
        max1 = max(maxDotCount.items(), key=operator.itemgetter(1))
        del maxDotCount[max1[0]]
        max2 = max(maxDotCount.items(), key=operator.itemgetter(1))
        if max1[1] == max2[1]:
            result.append(max1[0] if np.dot(x,wDic[(max1[0],max2[0])]) >= 0 else max2[0])
        else:
            result.append(max1[0])
    return result

def calculateW(Xtrain,Ytrain,maxUpdates):
    w = [0] * len(Xtrain[0])
    update,i = 0, 0
    updatedIter = False
    while True:
        if Ytrain[i]*np.dot(Xtrain[i],w) <= 0:
            w = w + Ytrain[i]*Xtrain[i]
            update += 1
            updatedIter = True
            if update == maxUpdates:
                break
        i += 1
        if i == len(Ytrain): 
            if updatedIter == True:
                i = 0
                updatedIter = False
            else:
                break
    return w