Logfile naming conventions: 
- 
Non-perceptron logfiles are follow this naming convention:
model-learningParameter-activationFunction

For example: depth3-0.01-leakyRelu is a logfile for a version of the depth3 model that has a learning rate of 0.01 and uses the leakyRelu activation function. 

Perceptron logfiles follow this naming convention: 

perceptron-classification1-classification2

For example: perceptron-setosa-versicolor is the logfile for the perceptron trained to
seperate between the Iris Setosa and iris Versicolor samples. 