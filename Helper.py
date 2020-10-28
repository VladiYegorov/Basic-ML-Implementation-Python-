from csv import reader
import numpy as np

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def load_csv(filename):
	dataset = []
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append([float(row[i].strip()) if isfloat(row[i].strip()) else hash(row[i].strip()) for i in range(len(row))])
	return dataset

#make new dataset only with the labels label1 and label2
def makeBinaryData(Xtrain, Ytrain, label1, label2):
    newXtrain = []
    newYtrain = []
    for i in range(len(Ytrain)):
        if Ytrain[i] == label1:
            newXtrain.append(Xtrain[i])
            newYtrain.append(label1)
        if Ytrain[i] == label2:
            newXtrain.append(Xtrain[i])
            newYtrain.append(label2)
    return (np.array(newXtrain),np.array(newYtrain))

#make new dataset only with the labels=1/-1 for label1/label2
def makeLinerPredictorData(Xtrain, Ytrain, label1, label2):
    newXtrain = []
    newYtrain = []
    for i in range(len(Ytrain)):
        if Ytrain[i] == label1:
            newXtrain.append(Xtrain[i])
            newYtrain.append(1)
        if Ytrain[i] == label2:
            newXtrain.append(Xtrain[i])
            newYtrain.append(-1)
    return (np.array(newXtrain),np.array(newYtrain))

#make dictinaory with labels as keys, and there value is a list of rows from 
#the data input (Xtrain) that correspond with the key label from (Ytrain).
def separate(Xtrain,Ytrain):
    separated = {}
    temp = False
    for rowX,label in zip(Xtrain,Ytrain):
        if (label not in separated):
            separated[label] = []
        separated[label].append(rowX)
    return separated