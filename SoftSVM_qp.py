import numpy as np 
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
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
    m = len(Ytrain)
    y = Ytrain.reshape(-1,1) * 1.
    x = y * Xtrain
    P = cvxopt_matrix(x @ x.T * lam * 1.)
    q = cvxopt_matrix(-np.ones((m, 1)))
    G = cvxopt_matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
    h = cvxopt_matrix(np.hstack((np.zeros(m),np.ones(m))))
    A = cvxopt_matrix(y.reshape(1, -1))
    b = cvxopt_matrix(np.zeros(1))

    sol = cvxopt_solvers.qp(P, q, G, h, A, b, options={'show_progress': False})
    alphas = np.array(sol['x'])
    return ((y * alphas).T @ Xtrain).reshape(-1,1)
