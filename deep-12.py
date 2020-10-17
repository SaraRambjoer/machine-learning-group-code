import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas
import copy
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
tf.keras.backend.set_floatx('float64')
# Actually deep11 but whatever, refactor
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
# Uses leaky relu
model01leaky = keras.models.Sequential()
model001leaky = keras.models.Sequential()
model0001leaky = keras.models.Sequential()
model00001leaky = keras.models.Sequential()
model000001leaky = keras.models.Sequential()
for num in range(10):
    model01.add(keras.layers.Dense(3, activation="sigmoid"))
    model001.add(keras.layers.Dense(3, activation="sigmoid"))
    model0001.add(keras.layers.Dense(3, activation="sigmoid"))
    model00001.add(keras.layers.Dense(3, activation="sigmoid"))
    model000001.add(keras.layers.Dense(3, activation="sigmoid"))
    model01leaky.add(keras.layers.Dense(3, activation=keras.layers.LeakyReLU()))
    model001leaky.add(keras.layers.Dense(3, activation=keras.layers.LeakyReLU()))
    model0001leaky.add(keras.layers.Dense(3, activation=keras.layers.LeakyReLU()))
    model00001leaky.add(keras.layers.Dense(3, activation=keras.layers.LeakyReLU()))
    model000001leaky.add(keras.layers.Dense(3, activation=keras.layers.LeakyReLU()))
model01.add(keras.layers.Dense(3, activation="softmax"))
model001.add(keras.layers.Dense(3, activation="softmax"))
model0001.add(keras.layers.Dense(3, activation="softmax"))
model00001.add(keras.layers.Dense(3, activation="softmax"))
model000001.add(keras.layers.Dense(3, activation="softmax"))
model01leaky.add(keras.layers.Dense(3, activation="softmax"))
model001leaky.add(keras.layers.Dense(3, activation="softmax"))
model0001leaky.add(keras.layers.Dense(3, activation="softmax"))
model00001leaky.add(keras.layers.Dense(3, activation="softmax"))
model000001leaky.add(keras.layers.Dense(3, activation="softmax"))


sgd01 = keras.optimizers.SGD(learning_rate=0.1)
sgd001 = keras.optimizers.SGD(learning_rate=0.01)
sgd0001 = keras.optimizers.SGD(learning_rate=0.001)
sgd00001 = keras.optimizers.SGD(learning_rate=0.0001)
sgd000001 = keras.optimizers.SGD(learning_rate=0.00001)
logCallback01sigmoid = keras.callbacks.LambdaCallback(on_epoch_end=lambda batch, logs: layerWeightSave(model01, "deep_weight_logs/01sigmoid.txt"))
logCallback001sigmoid = keras.callbacks.LambdaCallback(on_epoch_end=lambda batch, logs: layerWeightSave(model001, "deep_weight_logs/001sigmoid.txt"))
logCallback0001sigmoid = keras.callbacks.LambdaCallback(on_epoch_end=lambda batch, logs: layerWeightSave(model0001, "deep_weight_logs/0001sigmoid.txt"))
logCallback00001sigmoid = keras.callbacks.LambdaCallback(on_epoch_end=lambda batch, logs: layerWeightSave(model00001, "deep_weight_logs/00001sigmoid.txt"))
logCallback000001sigmoid = keras.callbacks.LambdaCallback(on_epoch_end=lambda batch, logs: layerWeightSave(model000001, "deep_weight_logs/000001sigmoid.txt"))
logCallback01leakyrelu = keras.callbacks.LambdaCallback(on_epoch_end=lambda batch, logs: layerWeightSave(model01, "deep_weight_logs/01leakyrelu.txt"))
logCallback001leakyrelu = keras.callbacks.LambdaCallback(on_epoch_end=lambda batch, logs: layerWeightSave(model001, "deep_weight_logs/001leakyrelu.txt"))
logCallback0001leakyrelu = keras.callbacks.LambdaCallback(on_epoch_end=lambda batch, logs: layerWeightSave(model0001, "deep_weight_logs/0001leakyrelu.txt"))
logCallback00001leakyrelu = keras.callbacks.LambdaCallback(on_epoch_end=lambda batch, logs: layerWeightSave(model00001, "deep_weight_logs/00001leakyrelu.txt"))
logCallback000001leakyrelu = keras.callbacks.LambdaCallback(on_epoch_end=lambda batch, logs: layerWeightSave(model000001, "deep_weight_logs/000001leakyrelu.txt"))


def layerWeightSave(model, fileName):
    f = open(fileName, "a")
    for ele in model.layers:
        f.write(str(ele.get_weights()[0]))
    f.write("\n|\n")
    f.close()


model01.compile(loss="binary_crossentropy", optimizer=sgd01, metrics=['accuracy']) #use crossentropy for loss, sgd optimizer (may change, must describe in report) and accuracy metric (Should also justify this in report)
model01.fit(x, encoded_Y, verbose=0, epochs=5000, batch_size=32, callbacks=[tf.keras.callbacks.CSVLogger("Logs/deep12-0.1-sigmoid.csv"), logCallback01sigmoid])
model001.compile(loss="binary_crossentropy", optimizer=sgd001, metrics=['accuracy']) #use crossentropy for loss, sgd optimizer (may change, must describe in report) and accuracy metric (Should also justify this in report)
model001.fit(x, encoded_Y, verbose=0, epochs=5000, batch_size=32, callbacks=[tf.keras.callbacks.CSVLogger("Logs/deep12-0.01-sigmoid.csv"), logCallback001sigmoid])
model0001.compile(loss="binary_crossentropy", optimizer=sgd0001, metrics=['accuracy']) #use crossentropy for loss, sgd optimizer (may change, must describe in report) and accuracy metric (Should also justify this in report)
model0001.fit(x, encoded_Y, verbose=0, epochs=5000, batch_size=32, callbacks=[tf.keras.callbacks.CSVLogger("Logs/deep12-0.001-sigmoid.csv"), logCallback0001sigmoid])
model00001.compile(loss="binary_crossentropy", optimizer=sgd00001, metrics=['accuracy']) #use crossentropy for loss, sgd optimizer (may change, must describe in report) and accuracy metric (Should also justify this in report)
model00001.fit(x, encoded_Y, verbose=0, epochs=5000, batch_size=32, callbacks=[tf.keras.callbacks.CSVLogger("Logs/deep12-0.0001-sigmoid.csv"), logCallback00001sigmoid])
model000001.compile(loss="binary_crossentropy", optimizer=sgd000001, metrics=['accuracy']) #use crossentropy for loss, sgd optimizer (may change, must describe in report) and accuracy metric (Should also justify this in report)
model000001.fit(x, encoded_Y, verbose=0, epochs=5000, batch_size=32, callbacks=[tf.keras.callbacks.CSVLogger("Logs/deep12-0.00001-sigmoid.csv"), logCallback000001sigmoid])

model01.save("models/deep01-sigmoid")
model001.save("models/deep001-sigmoid")
model0001.save("models/deep0001-sigmoid")
model00001.save("models/deep00001-sigmoid")
model000001.save("models/deep000001-sigmoid")

model01leaky.compile(loss="binary_crossentropy", optimizer=sgd01, metrics=['accuracy']) #use crossentropy for loss, sgd optimizer (may change, must describe in report) and accuracy metric (Should also justify this in report)
model01leaky.fit(x, encoded_Y, verbose=0, epochs=5000, batch_size=32, callbacks=[tf.keras.callbacks.CSVLogger("Logs/deep12-0.1-leakyrelu.csv"), logCallback01leakyrelu])
model001leaky.compile(loss="binary_crossentropy", optimizer=sgd001, metrics=['accuracy']) #use crossentropy for loss, sgd optimizer (may change, must describe in report) and accuracy metric (Should also justify this in report)
model001leaky.fit(x, encoded_Y, verbose=0, epochs=5000, batch_size=32, callbacks=[tf.keras.callbacks.CSVLogger("Logs/deep12-0.01-leakyrelu.csv"), logCallback001leakyrelu])
model0001leaky.compile(loss="binary_crossentropy", optimizer=sgd0001, metrics=['accuracy']) #use crossentropy for loss, sgd optimizer (may change, must describe in report) and accuracy metric (Should also justify this in report)
model0001leaky.fit(x, encoded_Y, verbose=0, epochs=5000, batch_size=32, callbacks=[tf.keras.callbacks.CSVLogger("Logs/deep12-0.001-leakyrelu.csv"), logCallback0001leakyrelu])
model00001leaky.compile(loss="binary_crossentropy", optimizer=sgd00001, metrics=['accuracy']) #use crossentropy for loss, sgd optimizer (may change, must describe in report) and accuracy metric (Should also justify this in report)
model00001leaky.fit(x, encoded_Y, verbose=0, epochs=5000, batch_size=32, callbacks=[tf.keras.callbacks.CSVLogger("Logs/deep12-0.0001-leakyrelu.csv"), logCallback00001leakyrelu])
model000001leaky.compile(loss="binary_crossentropy", optimizer=sgd000001, metrics=['accuracy']) #use crossentropy for loss, sgd optimizer (may change, must describe in report) and accuracy metric (Should also justify this in report)
model000001leaky.fit(x, encoded_Y, verbose=0, epochs=5000, batch_size=32, callbacks=[tf.keras.callbacks.CSVLogger("Logs/deep12-0.00001-leakyrelu.csv"), logCallback000001leakyrelu])

model01leaky.save("models/deep01-leakyrelu")
model001leaky.save("models/deep001-leakyrelu")
model0001leaky.save("models/deep0001-leakyrelu")
model00001leaky.save("models/deep00001-leakyrelu")
model000001leaky.save("models/deep000001-leakyrelu")

# Could consider also doing residual blocks but I don't uderstand them so I won't focus on it
#perceptron_model1.summary()
#perceptron_model2.summary()
#perceptron_model3.summary()
