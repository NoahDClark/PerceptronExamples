#Required imports
import numpy as np
import random
import scipy.special

#Perceptron class - Contains the required methods and variables to perform multiple epochs
class perceptron:
    
    #Constructor - Contains weights and the learning rate
    def __init__(self, inputSize, learningRate):
        self._weights = abs(np.random.randn(inputSize+1)) #NOTE!!! set to inputSize+1 to account for the BIAS
        self._learningRate = learningRate
        
    
    #Activation Function - For this, I used a simple threshold function
    def activation_Function(self, val):
        if val > 0.5:
            return 1
        else:
            return 0
        
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


#Recursive method to generate the 2^4 (16) boards used in part one of the assignment
def generateBoards(boards, boardBrightness, current = 0, array=None):
    if (array is None):
        generateBoards(boards,boardBrightness,(current),[0])
        generateBoards(boards,boardBrightness,(current+1),[1])
    elif (len(array) >= 4):
        boards.append(array)
        if (current > 1):
            boardBrightness.append([1])
        else:
            boardBrightness.append([0])
    else:
        temp = array.copy()
        temp.append(0)
        temp2 = array.copy()
        temp2.append(1)
        generateBoards(boards,boardBrightness,(current),temp)
        generateBoards(boards,boardBrightness,(current+1),temp2)
        
#Settings for the perceptron and batch size
sampleSize = 10
inputSize = 4
learningRate = 0.01
percep = perceptron(inputSize, learningRate)

#Creates the training values and testing values
boards = []
boardBrightnessValues = []
generateBoards(boards,boardBrightnessValues)
temp = random.sample(range(0,16), sampleSize)
inputArray = []
outputArray = []
for i in temp:
    inputArray.append(boards[i])
    outputArray.append(boardBrightnessValues[i])
inputArray = np.array(inputArray)
outputArray = np.array(outputArray)

#Train perceptron
percep.train(inputArray, outputArray, 1000)

#Prints total accuracy & prediction for each given inputs
total = 0
for (inputVals, targetVals) in zip(boards, boardBrightnessValues):
    p = percep.predict(inputVals)
    if (targetVals[0] == p):
        total = total+1
    print("Data: {}, Target: {}, Prediction: {}".format(inputVals,targetVals[0],p))
print("ACCURACY: " + str((total/16)*100)+"%")
