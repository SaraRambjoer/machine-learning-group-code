# Used to check if shallow150 neurons actually correspond to specific samples
from tensorflow import keras
import pandas
import numpy as np
from sklearn.preprocessing import LabelEncoder
def leakyRelu(x):
    return [ele if ele > 0 else ele*0.01 for ele in x]

def reLu(x):
    return [ele if ele > 0 else 0 for ele in x]

def sigmoid(x):
    return [1/(1+np.e*-ele) for ele in x]

keras.backend.set_floatx('float64')

# LeakyReLu
model = keras.models.load_model('models\\shallow150-leakyRelu')
neuronWeights = model.get_weights()  # Get first layer of weights
inputWeights = neuronWeights[0]
bias150 = neuronWeights[1]
# Get the training data
iris_flower_dataset = pandas.read_csv('Data/IRIS.csv')
x = iris_flower_dataset.drop(["species"], axis=1) # Create a dataset with only values
inputData = x.to_numpy()

# Check leakyReLu
for sample in inputData:
    # Find out how neurons activates
    neuronInputs = bias150 + np.dot(np.reshape(sample, (1, 4)), inputWeights)
    neuronActivations = leakyRelu(neuronInputs[0])  # 0 because it is a (150, 1) numpy array which is a double list so we need to unpack the data
    # Save result
    f = open("Logs\\shallow150-leakyRelu-activations-raw.csv", "a")
    f.write(str(neuronActivations))
    f.write("\n")
    f.close()

# reLu
model = keras.models.load_model('models\\shallow150-relu')
neuronWeights = model.get_weights()  # Get first layer of weights
inputWeights = neuronWeights[0]
bias150 = neuronWeights[1]
# Get the training data
iris_flower_dataset = pandas.read_csv('Data/IRIS.csv')
x = iris_flower_dataset.drop(["species"], axis=1) # Create a dataset with only values
inputData = x.to_numpy()

for sample in inputData:
    # Find out how neurons activates
    neuronInputs = bias150 + np.dot(np.reshape(sample, (1, 4)), inputWeights)
    neuronActivations = reLu(neuronInputs[0])  # 0 because it is a (150, 1) numpy array which is a double list so we need to unpack the data
    # Save result
    f = open("Logs\\shallow150-relu-activations-raw.csv", "a")
    f.write(str(neuronActivations))
    f.write("\n")
    f.close()

# sigmoid
model = keras.models.load_model('models\\shallow150-sigmoid')
neuronWeights = model.get_weights()  # Get first layer of weights
inputWeights = neuronWeights[0]
bias150 = neuronWeights[1]
# Get the training data
iris_flower_dataset = pandas.read_csv('Data/IRIS.csv')
x = iris_flower_dataset.drop(["species"], axis=1) # Create a dataset with only values
inputData = x.to_numpy()

for sample in inputData:
    # Find out how neurons activates
    neuronInputs = bias150 + np.dot(np.reshape(sample, (1, 4)), inputWeights)
    neuronActivations = sigmoid(neuronInputs[0])  # 0 because it is a (150, 1) numpy array which is a double list so we need to unpack the data
    # Save result
    f = open("Logs\\shallow150-sigmoid-activations-raw.csv", "a")
    f.write(str(neuronActivations))
    f.write("\n")
    f.close()

def process_log_file(inputPath, outputPath):
    f = open(inputPath, 'r')
    text = f.read()
    f.close()
    dataList = text.split("\n")
    dataList = dataList[0:-1]  # Last ele is empty
    dataList = [ele[1:-1] for ele in dataList] # Removes brakets in each string
    dataList = [ele.split(",") for ele in dataList] # Split into lists
    f = open(outputPath, 'a')
    for ele in dataList:
        count = len([ele2 for ele2 in ele if abs(float(ele2)) > 0.15]) # Get amount of activations with abs value greater than 0.15
        f.write(str(count))
        f.write("\n")
    f.close()

process_log_file("Logs\\shallow150-sigmoid-activations-raw.csv", "Logs\\shallow150-sigmoid-activations-processed.csv")
process_log_file("Logs\\shallow150-relu-activations-raw.csv", "Logs\\shallow150-relu-activations-processed.csv")
process_log_file("Logs\\shallow150-leakyRelu-activations-raw.csv", "Logs\\shallow150-leakyRelu-activations-processed.csv")


