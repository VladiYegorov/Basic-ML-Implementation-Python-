import numpy as np 
import operator

def learnknn(Xtrain, Ytrain, k):
    def classfier(cord): 
        sampleSizeDic = {}
        distToSample = []  
        for x, y in zip(Xtrain, Ytrain):
            distToSample.append((np.linalg.norm(x-cord),y))
        distToSample.sort(key=lambda tu: tu[0])
        distToSample = distToSample[:k]
        for i in distToSample:
            if i[1] in sampleSizeDic:
                sampleSizeDic[i[1]] += 1
            else:
                sampleSizeDic[i[1]] = 1
        return max(sampleSizeDic.items(), key=operator.itemgetter(1))[0]
    return classfier

def predictknn(classfier,Xtest):
    return list(map(lambda row: classfier(row),Xtest))


