Graph naming conventions
- 

modelActivationFunc-smoothXX-activity/lossFunc:

* model: model name
* activation func: Activation func used in image
* smoothXX: How many log entries (epochs) are averaged in graph (for smoothness/legiability)
* activity/lossFunc: Whether or not the y-axis is the activation metric or the loss function metric (categorical cross-entropy)

perceptrons are named similary, only without learning rates and activation funcitons. 

x_datasetvisualization:

* A scatter plot visualizing the Iris Flower dataset according to x. For example: length_datavisualization
is a scatter plot of the Iris Flower Dataset where the axises are
sepal and petal length. 


All files containing word comparison compare different models with different hyperparameters
according to either accuracy or lossFunc. 