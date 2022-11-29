
#Required imports
import numpy as np
import random
import scipy.special
import pandas as pd

#Perceptron class - Contains the required methods and variables to perform multiple epochs
class perceptron:
    
    #Constructor - Contains weights and the learning rate
    def __init__(self, inputSize, learningRate):
        self._weights = abs(np.random.randn(inputSize+1)) #NOTE!!! set to inputSize+1 to account for the BIAS
        self._learningRate = learningRate
        
    #Activation Function - For this, I used a sigmoid function in combination with a threshold function
    def activation_Function(self, val):
        tempVal = (1/(1+np.exp(-val)))
        #print(tempVal)
        
        if tempVal > 0.66:
            return 2
        elif tempVal < 0.33:
            return 0
        else:
            return 1
        
        
    #Training method - Runs for a set number of epochs and edits weights when necessary
    def train(self, trainingSetInputs, trainingSetTargets, epochs):
        
        trainingSetInputs = np.c_[trainingSetInputs, np.ones((trainingSetInputs.shape[0]))] #NOTE!!! This adds the bias
        
        #Loop for each epoch
        for i in range(epochs):
            current = 0 #Tracks number of correct outputs
            #Loop for each input and output combination
            for (inputValues, targetValue) in zip(trainingSetInputs, trainingSetTargets):
                outputValue = self.activation_Function(np.dot(inputValues, self._weights)) #Calls activation function
                
                #If not target value, then edit the weights. Otherwise, current+1 since it was corret
                if outputValue != targetValue:
                    self._weights = self._weights + (-self._learningRate * (outputValue - targetValue) * inputValues)
                else:
                    current = current+1
            
            #Prints accuracy and will stop trianing if accuracy reachs 100%
            if (i % 1 == 0):
                temp = current/(len(trainingSetInputs))
                print("Epoch "+(str(i))+" accuracy is: "+(str(temp)))
                if (temp == 1):
                    break

    #Predict method - Takes in a value and predicts the output for it.
    def predict(self, inputValues):
        inputValues = np.append(inputValues,[1]) #NOTE!!! This adds the bias
        return self.activation_Function(np.dot(inputValues, self._weights))
    
#End of perceptron class

#Settings for the perceptron and batch size
sampleSize = 120
inputSize = 4
learningRate = 0.01
percep = perceptron(inputSize, learningRate)

#Reads CSV and creates necessary arrays
data=pd.read_csv('iris.csv')
data.columns=['Sepal_len_cm','Sepal_wid_cm','Petal_len_cm','Petal_wid_cm','Type']
Sepal_len_cm = data['Sepal_len_cm'].tolist()
Sepal_wid_cm = data['Sepal_wid_cm'].tolist()
Petal_len_cm = data['Petal_len_cm'].tolist()
Petal_wid_cm = data['Petal_wid_cm'].tolist()
Type = data['Type'].tolist()

fullInputArray = []
fullOutputArray = []
for i in range(len(Petal_wid_cm)):
    temp = [Sepal_len_cm[i],Sepal_wid_cm[i],Petal_len_cm[i],Petal_wid_cm[i]]
    fullInputArray.append(temp)
    temp2 = [Type[i]]
    if (Type[i] == "Setosa"):
        temp2 = 2
    elif (Type[i] == "Versicolor"):
        temp2 = 1
    elif (Type[i] == "Virginica"):
        temp2 = 0
    fullOutputArray.append(temp2)

temp = random.sample(range(0,len(Petal_wid_cm)), sampleSize)

inputArray = []
outputArray = []
for i in temp:
    inputArray.append(fullInputArray[i])
    outputArray.append(fullOutputArray[i])
inputArray = np.array(inputArray)
outputArray = np.array(outputArray)

#Train perceptron
percep.train(inputArray, outputArray, 1000)

#Prints total accuracy & prediction for each given inputs
total = 0
for (inputVals, targetVals) in zip(fullInputArray, fullOutputArray):
    p = percep.predict(inputVals)
    if (targetVals == p):
        total = total+1
    print("Data:{}, Target:{}, Prediction:{}".format(inputVals,targetVals,p))
print("ACCURACY: " + str((total/len(fullOutputArray))*100)+"%")
