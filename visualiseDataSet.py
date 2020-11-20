import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os


iris = pd.read_csv('Data/IRIS.csv', names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
print(iris.head())

colors = {'Iris-setosa':'r', 'Iris-versicolor':'g', 'Iris-virginica':'b'}

#plot the petals

for i in range(1,len(iris['petal_length'])):
    plt.scatter(iris['petal_length'][i], iris['petal_width'][i],color=colors[iris['species'][i]])

plt.xlabel('petal length')
plt.ylabel('petal width')
plt.xticks(rotation=90)
for label in plt.gca().get_xaxis().get_ticklabels()[::2]:
    label.set_visible(False)
plt.savefig(os.path.join(os.path.abspath(os.getcwd()),
                                                "graphs\\petals_datasetvisualization.png"))

#plot the sepals

plt.clf()

for i in range(1,len(iris['sepal_length'])):
    plt.scatter(iris['sepal_length'][i], iris['sepal_width'][i],color=colors[iris['species'][i]])

plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xticks(rotation=90)
for label in plt.gca().get_xaxis().get_ticklabels()[::2]:
    label.set_visible(False)
plt.savefig(os.path.join(os.path.abspath(os.getcwd()),
                                                "graphs\\sepals_datasetvisualization.png"))

plt.clf()

for i in range(1,len(iris['sepal_length'])):
    plt.scatter(iris['sepal_length'][i], iris['petal_length'][i],color=colors[iris['species'][i]])

plt.xlabel('Sepal length')
plt.ylabel('Petal length')
plt.xticks(rotation=90)
for label in plt.gca().get_xaxis().get_ticklabels()[::2]:
    label.set_visible(False)
for label in plt.gca().get_yaxis().get_ticklabels()[::2]:
    label.set_visible(False)
plt.savefig(os.path.join(os.path.abspath(os.getcwd()),
                                                "graphs\\length_datasetvisualization.png"))


plt.clf()

for i in range(1,len(iris['sepal_length'])):
    plt.scatter(iris['sepal_width'][i], iris['petal_width'][i],color=colors[iris['species'][i]])

plt.xlabel('Sepal width')
plt.ylabel('Petal width')
plt.xticks(rotation=90)
for label in plt.gca().get_xaxis().get_ticklabels()[::2]:
    label.set_visible(False)
plt.savefig(os.path.join(os.path.abspath(os.getcwd()),
                                                "graphs\\width_datasetvisualization.png"))

plt.clf()

for i in range(1,len(iris['sepal_length'])):
    plt.scatter(iris['sepal_width'][i], iris['petal_length'][i],color=colors[iris['species'][i]])

plt.xlabel('Sepal width')
plt.ylabel('Petal length')
plt.xticks(rotation=90)
for label in plt.gca().get_xaxis().get_ticklabels()[::2]:
    label.set_visible(False)
for label in plt.gca().get_yaxis().get_ticklabels()[::2]:
    label.set_visible(False)
plt.savefig(os.path.join(os.path.abspath(os.getcwd()),
                                                "graphs\\sepal_width_petal_length_datasetvisualization.png"))

plt.clf()

for i in range(1,len(iris['sepal_length'])):
    plt.scatter(iris['sepal_length'][i], iris['petal_width'][i],color=colors[iris['species'][i]])

plt.xlabel('Sepal length')
plt.ylabel('Petal width')
plt.xticks(rotation=90)
for label in plt.gca().get_xaxis().get_ticklabels()[::2]:
    label.set_visible(False)
for label in plt.gca().get_yaxis().get_ticklabels()[::2]:
    label.set_visible(False)
plt.savefig(os.path.join(os.path.abspath(os.getcwd()),
                                                "graphs\\sepal_length_petal_width_datasetvisualization.png"))


