import numpy as np 
import operator
import Helper as hp

def softSVM(Xtrain, Ytrain, lam):
    allLabels = {}
    labelList = []
    for y in Ytrain:
        labelList.append(y)
    for label1 in labelList:
        for label2 in labelList:
            if label1 == label2 or (label1,label2) in allLabels:
                continue
            tempXtrain,tempYtrain = hp.makeLinerPredictorData(Xtrain, Ytrain, label1, label2)
            w = calculateW(tempXtrain,tempYtrain,lam)
            allLabels[(label1,label2)] = w
            allLabels[(label2,label1)] = (-1)*w
    return allLabels

def predictSoftSVM(wDic, Xtest):
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

def calculateW(Xtrain,Ytrain,lam):
    step_size = 1. / lam
    w = np.array([0.] * len(Xtrain[0]))
    m = len(Ytrain)
    t = len(Ytrain) * 10
    result = np.array([0.] * len(Xtrain[0]))

    for i in range(t):
        x = np.random.randint(m)
        w = w - (step_size/(i+1))*(2*lam*w + (np.array([0.] * len(Xtrain[0]))  if 1.*Ytrain[x]*np.dot(w,Xtrain[x]) >= 1 else (-1.)*Ytrain[x]*Xtrain[x]))
        result += w

    return (1/t)*result