import numpy as np 
import operator

def calcError(real_values, predicted_values):
    correct = 0
    total = 0
    for i in range(len(real_values)):
        total += 1
        if predicted_values[i] == real_values[i]:
            correct += 1
    return (1.-1.*(correct/total))*100 

#find best parameter for the classifier using k-fold cross validation
def crossValidation(Xtrain, Ytrain, classifier, predictFunc, valuesSet, num_fold = 10):
    m = len(Ytrain)
    splitedXtrain = [[]] * num_fold
    setSize = m//num_fold
    minParamError = {}

    for param in valuesSet:
        prevIndex = 0
        avgFoldError = 0.
        extraSamples = m % num_fold
        for i in range(num_fold):
            currentIndex = prevIndex + setSize + (1 if extraSamples != 0 else 0)
            learn = classifier(np.concatenate((Xtrain[0:prevIndex],Xtrain[currentIndex:m]),axis=0),np.concatenate((Ytrain[0:prevIndex],Ytrain[currentIndex:m]),axis=0),param)
            result = predictFunc(learn,Xtrain[prevIndex:currentIndex])
            avgFoldError += calcError(Ytrain[prevIndex:currentIndex],result)

            prevIndex = currentIndex
            if extraSamples != 0:
                extraSamples -= 1
        minParamError[param] = avgFoldError/num_fold
    return classifier(np.array(Xtrain),np.array(Ytrain),min(minParamError, key=minParamError.get))
        





