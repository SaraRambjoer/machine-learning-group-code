import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os


iris = pd.read_csv('Data/IRIS.csv', names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
print(iris.head())

colors = {'Iris-setosa':'r', 'Iris-versicolor':'g', 'Iris-virginica':'b'}

#plot the petals

fig, pet = plt.subplots()

for i in range(1,len(iris['petal_length'])):
    pet.scatter(iris['petal_length'][i], iris['petal_width'][i],color=colors[iris['species'][i]])

pet.set_title('Petals')
pet.set_xlabel('petal length')
pet.set_ylabel('petal width')

pet.axes.xaxis.set_ticks([])
pet.axes.yaxis.set_ticks([])

#plot the sepals

fig, sep = plt.subplots()

for i in range(1,len(iris['sepal_length'])):
    sep.scatter(iris['sepal_length'][i], iris['sepal_width'][i],color=colors[iris['species'][i]])

sep.set_title('Sepals')
sep.set_xlabel('Sepal length')
sep.set_ylabel('Sepal width')

sep.axes.xaxis.set_ticks([])
sep.axes.yaxis.set_ticks([])


plt.show()

