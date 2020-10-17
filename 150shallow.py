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
model01.add(keras.layers.Dense(150, activation="sigmoid"))
model001.add(keras.layers.Dense(150, activation="sigmoid"))
model0001.add(keras.layers.Dense(150, activation="sigmoid"))
model00001.add(keras.layers.Dense(150, activation="sigmoid"))
model000001.add(keras.layers.Dense(150, activation="sigmoid"))
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



model01.compile(loss="binary_crossentropy", optimizer=sgd01, metrics=['accuracy']) #use crossentropy for loss, sgd optimizer (may change, must describe in report) and accuracy metric (Should also justify this in report)
model01.fit(x, encoded_Y, verbose=0, epochs=5000, batch_size=32, callbacks=[tf.keras.callbacks.CSVLogger("Logs/shallow150-0.1-sigmoid.csv")])
model001.compile(loss="binary_crossentropy", optimizer=sgd001, metrics=['accuracy']) #use crossentropy for loss, sgd optimizer (may change, must describe in report) and accuracy metric (Should also justify this in report)
model001.fit(x, encoded_Y, verbose=0, epochs=5000, batch_size=32, callbacks=[tf.keras.callbacks.CSVLogger("Logs/shallow150-0.01-sigmoid.csv")])
model0001.compile(loss="binary_crossentropy", optimizer=sgd0001, metrics=['accuracy']) #use crossentropy for loss, sgd optimizer (may change, must describe in report) and accuracy metric (Should also justify this in report)
model0001.fit(x, encoded_Y, verbose=0, epochs=5000, batch_size=32, callbacks=[tf.keras.callbacks.CSVLogger("Logs/shallow150-0.001-sigmoid.csv")])
model00001.compile(loss="binary_crossentropy", optimizer=sgd00001, metrics=['accuracy']) #use crossentropy for loss, sgd optimizer (may change, must describe in report) and accuracy metric (Should also justify this in report)
model00001.fit(x, encoded_Y, verbose=0, epochs=5000, batch_size=32, callbacks=[tf.keras.callbacks.CSVLogger("Logs/shallow150-0.0001-sigmoid.csv")])
model000001.compile(loss="binary_crossentropy", optimizer=sgd000001, metrics=['accuracy']) #use crossentropy for loss, sgd optimizer (may change, must describe in report) and accuracy metric (Should also justify this in report)
model000001.fit(x, encoded_Y, verbose=0, epochs=5000, batch_size=32, callbacks=[tf.keras.callbacks.CSVLogger("Logs/shallow150-0.00001-sigmoid.csv")])

model01.save("models/shallow150-sigmoid")
model001.save("models/shallow150-sigmoid")
model0001.save("models/shallow150-sigmoid")
model00001.save("models/shallow150-sigmoid")
model000001.save("models/shallow150-sigmoid")
