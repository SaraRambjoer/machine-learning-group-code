import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas
import random
from sklearn.preprocessing import LabelEncoder, scale
tf.keras.backend.set_floatx('float64')

# perceptrons
iris_flower_dataset = pandas.read_csv('Data/IRIS.csv')
x = iris_flower_dataset.drop(["species"], axis=1) # Create a dataset with only values
x = scale(x)
y = iris_flower_dataset["species"].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}) # Create a dataset with only species and map them to integer values

# Encode y
target_encoder = LabelEncoder() # Label encoder is specialised for encoding target values
target_encoder.fit(y)
encoded_Y = target_encoder.transform(y)  # encodes y to an array'
# x and y into three classification problems


x1 = scale(iris_flower_dataset[iris_flower_dataset.species != "Iris-setosa"].drop(["species"], axis=1))
x2 = scale(iris_flower_dataset[iris_flower_dataset.species != "Iris-versicolor"].drop(["species"], axis=1))
x3 = scale(iris_flower_dataset[iris_flower_dataset.species != "Iris-virginica"].drop(["species"], axis=1))
y1 = np.asarray([0 if ele == 1 else 1 for ele in encoded_Y if ele != 0])
y2 = np.asarray([0 if ele == 0 else 1 for ele in encoded_Y if ele != 1])
y3 = np.asarray([0 if ele == 0 else 1 for ele in encoded_Y if ele != 2])
# encodes species to a list of lists containing one element for each category, to be used in cross entropy loss (better for classification)

# Results show that for x1 and y1, where iris versicolor and iris virginia are compared, it is not possible to seperate the classes with one
# perceptron, while in the other cases it is, which verifies the fact that perceptrons can only seperate linerarly seperable data
# LEGACY TEST CODE; NOT ACTUALLY PERCEPTRONS JUST A DIFFERENT TYPE OF SINGULAR ANN. Should produce the same results but why not be faithful to the original premise?
#perceptron_model1 = keras.models.Sequential()
#perceptron_model1.add(keras.layers.Dense(1, activation="sigmoid"))
#perceptron_model1.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy']) #use crossentropy for loss, adam optimizer (may change, must describe in report) and accuracy
## metric (Should also justify this in report)
#perceptron_model1.fit(x1, y1, verbose=0, epochs=5000, batch_size=5, callbacks=[tf.keras.callbacks.CSVLogger("Logs/perceptron-versicolor-virginicia.csv")])

#perceptron_model2 = keras.models.Sequential()
#perceptron_model2.add(keras.layers.Dense(1, activation="sigmoid"))
#perceptron_model2.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy']) #use crossentropy for loss, adam optimizer (may change, must describe in report) and accuracy
## metric (Should also justify this in report)
#perceptron_model2.fit(x2, y2, verbose=0, epochs=5000, batch_size=5, callbacks=[tf.keras.callbacks.CSVLogger("Logs/perceptron-setosa-virginicia.csv")])
#
## Create one layer NN with 3 perceptrons, one for each category
#perceptron_model3 = keras.models.Sequential()
#perceptron_model3.add(keras.layers.Dense(1, activation="sigmoid"))
#perceptron_model3.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy']) #use crossentropy for loss, adam optimizer (may change, must describe in report) and accuracy
## metric (Should also justify this in report)
#perceptron_model3.fit(x3, y3, verbose=0, epochs=5000, batch_size=5, callbacks=[tf.keras.callbacks.CSVLogger("Logs/perceptron-setosa-versicolor.csv")])
#
#perceptron_model1.save("models/perceptron-versicolor-virginicia")
#perceptron_model2.save("models/perceptron-setosa-virginicia")
#perceptron_model3.save("models/perceptron-setosa-versicolor")
##perceptron_model1.summary()
##perceptron_model2.summary()
##perceptron_model3.summary()
#

def perceptron_evaluate(inputs, weights):
    return 1 if sum(inputs*weights) > 0 else 0

def perceptron_update(weights, learningRate, output, desiredOutput, inputs):
    return weights + learningRate * (desiredOutput - output) * inputs

def perceptron_learning_step(inputs, desiredOutput, perceptronWeights, learningRate):
    output = perceptron_evaluate(inputs, perceptronWeights)
    return perceptron_update(perceptronWeights, learningRate, output, desiredOutput, inputs), output == desiredOutput

training1 = np.asarray(list(zip(x1, y1)))
training2 = np.asarray(list(zip(x2, y2)))
training3 = np.asarray(list(zip(x3, y3)))
def trainPerceptrons(epochs, training_data, logPath, savePath):
    weights = np.asarray([0.0, 0.0, 0.0, 0.0])
    learning_rate = 0.1
    f = open(logPath, 'a')
    f.write("epochs,accuracy,lossFunction\n")
    f.close()
    for count in range(0, epochs):
        data = list(np.copy(training_data))
        right = 0
        while len(data) > 0:
            ele = data.pop(random.randint(0, len(data)-1))
            weights, correct = perceptron_learning_step(ele[0], ele[1], weights, learning_rate)
            right = right + 1 if correct else right
        f = open(logPath, 'a')
        f.write(f"{count},{right/len(training_data)},{right/len(training_data)}\n")
        f.close()
    f = open(savePath, 'w')
    f.write(str(weights))
    f.close()

trainPerceptrons(1000, training1, "Logs/perceptron-versicolor-virginicia.csv", "models/perceptron-versicolor-virginicia.txt")
trainPerceptrons(1000, training2, "Logs/perceptron-setosa-virginicia.csv", "models/perceptron-setosa-virginicia.txt")
trainPerceptrons(1000, training3, "Logs/perceptron-setosa-versicolor.csv", "models/perceptron-setosa-versicolor.txt")
