import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas
import copy
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
tf.keras.backend.set_floatx('float64')

iris_flower_dataset = pandas.read_csv('Data/IRIS.csv')
x = iris_flower_dataset.drop(["species"], axis=1) # Create a dataset with only values
y = iris_flower_dataset["species"].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}) # Create a dataset with only species and map them to integer values

# Encode y
target_encoder = LabelEncoder() # Label encoder is specialised for encoding target values
target_encoder.fit(y)
encoded_Y = target_encoder.transform(y)  # encodes y to an array'
encoded_Y = keras.utils.to_categorical(encoded_Y) # Actually makes encoding into classification format readable by Keras
model01 = keras.models.Sequential()
model001 = keras.models.Sequential()
model0001 = keras.models.Sequential()
model00001 = keras.models.Sequential()
model000001 = keras.models.Sequential()
for num in range(10):
    model01.add(keras.layers.Dense(3, activation="relu"))
    model001.add(keras.layers.Dense(3, activation="relu"))
    model0001.add(keras.layers.Dense(3, activation="relu"))
    model00001.add(keras.layers.Dense(3, activation="relu"))
    model000001.add(keras.layers.Dense(3, activation="relu"))
model01.add(keras.layers.Dense(3, activation="softmax"))
model001.add(keras.layers.Dense(3, activation="softmax"))
model0001.add(keras.layers.Dense(3, activation="softmax"))
model00001.add(keras.layers.Dense(3, activation="softmax"))
model000001.add(keras.layers.Dense(3, activation="softmax"))



sgd01 = keras.optimizers.SGD(learning_rate=0.1)
sgd001 = keras.optimizers.SGD(learning_rate=0.01)
sgd0001 = keras.optimizers.SGD(learning_rate=0.001)
sgd00001 = keras.optimizers.SGD(learning_rate=0.0001)
sgd000001 = keras.optimizers.SGD(learning_rate=0.00001)
logCallback01relu = keras.callbacks.LambdaCallback(on_epoch_end=lambda batch, logs: layerWeightSave(model01, "deep_weight_logs/01relu.txt"))
logCallback001relu = keras.callbacks.LambdaCallback(on_epoch_end=lambda batch, logs: layerWeightSave(model001, "deep_weight_logs/001relu.txt"))
logCallback0001relu = keras.callbacks.LambdaCallback(on_epoch_end=lambda batch, logs: layerWeightSave(model0001, "deep_weight_logs/0001relu.txt"))
logCallback00001relu = keras.callbacks.LambdaCallback(on_epoch_end=lambda batch, logs: layerWeightSave(model00001, "deep_weight_logs/00001relu.txt"))
logCallback000001relu = keras.callbacks.LambdaCallback(on_epoch_end=lambda batch, logs: layerWeightSave(model000001, "deep_weight_logs/000001relu.txt"))



def layerWeightSave(model, fileName):
    f = open(fileName, "a")
    for ele in model.layers:
        f.write(str(ele.get_weights()[0]))
    f.write("\n|\n")
    f.close()


model01.compile(loss="categorical_crossentropy", optimizer=sgd01, metrics=['accuracy']) #use crossentropy for loss, sgd optimizer (may change, must describe in report) and accuracy metric (Should also justify this in report)
model01.fit(x, encoded_Y, verbose=0, epochs=5000, batch_size=32, callbacks=[tf.keras.callbacks.CSVLogger("Logs/deep12-0.1-relu.csv"), logCallback01relu])
model001.compile(loss="categorical_crossentropy", optimizer=sgd001, metrics=['accuracy']) #use crossentropy for loss, sgd optimizer (may change, must describe in report) and accuracy metric (Should also justify this in report)
model001.fit(x, encoded_Y, verbose=0, epochs=5000, batch_size=32, callbacks=[tf.keras.callbacks.CSVLogger("Logs/deep12-0.01-relu.csv"), logCallback001relu])
model0001.compile(loss="categorical_crossentropy", optimizer=sgd0001, metrics=['accuracy']) #use crossentropy for loss, sgd optimizer (may change, must describe in report) and accuracy metric (Should also justify this in report)
model0001.fit(x, encoded_Y, verbose=0, epochs=5000, batch_size=32, callbacks=[tf.keras.callbacks.CSVLogger("Logs/deep12-0.001-relu.csv"), logCallback0001relu])
model00001.compile(loss="categorical_crossentropy", optimizer=sgd00001, metrics=['accuracy']) #use crossentropy for loss, sgd optimizer (may change, must describe in report) and accuracy metric (Should also justify this in report)
model00001.fit(x, encoded_Y, verbose=0, epochs=5000, batch_size=32, callbacks=[tf.keras.callbacks.CSVLogger("Logs/deep12-0.0001-relu.csv"), logCallback00001relu])
model000001.compile(loss="categorical_crossentropy", optimizer=sgd000001, metrics=['accuracy']) #use crossentropy for loss, sgd optimizer (may change, must describe in report) and accuracy metric (Should also justify this in report)
model000001.fit(x, encoded_Y, verbose=0, epochs=5000, batch_size=32, callbacks=[tf.keras.callbacks.CSVLogger("Logs/deep12-0.00001-relu.csv"), logCallback000001relu])

model01.save("models/deep01-relu")
model001.save("models/deep001-relu")
model0001.save("models/deep0001-relu")
model00001.save("models/deep00001-relu")
model000001.save("models/deep000001-relu")

# Could consider also doing residual blocks but I don't uderstand them so I won't focus on it
#perceptron_model1.summary()
#perceptron_model2.summary()
#perceptron_model3.summary()