import numpy as np
import operator
import Helper as hp

def calculate_probability(x, mean, stdev):
    if stdev == 0:
        if x == mean:
            return 1
        else:
            return 0
    exponent = np.exp(-((x-mean)**2 / (2 * stdev**2)))
    return (1 / (np.sqrt(2 * np.pi) * stdev)) * exponent

def naiveBayes(Xtrain, Ytrain, param):
    dataByLabel = hp.separate(Xtrain,Ytrain)
    result = {}
    for label, infoList in dataByLabel.items():
        state = []
        infoList_dash = np.array(infoList).T
        for col in infoList_dash:
            state.append((np.mean(col),np.std(col)))
        result[label] = state
    return result
    
def predictNB(predictor, Xtest):
    result = []
    i = 0
    for row in Xtest:
        probDict = {}
        for label, states in predictor.items():
            calcProb = 1
            for x,state in zip(row,states):
                calcProb *= calculate_probability(x,state[0],state[1])
            probDict[label] = calcProb
        result.append(max(probDict.items(), key=operator.itemgetter(1))[0])
    return result