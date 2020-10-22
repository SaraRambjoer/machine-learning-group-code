import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
matplotlib.use('TkAgg')
# Create graphs by epoch-lossFunc
def create_and_save_graph_lossFunc(inputFilepaths, functionNames, outputFilepath, xAxisName, yAxisName, graphTitle, moduloSmoothing):
    plt.clf()
    dirPath = os.path.abspath(os.getcwd())
    xList = []
    yList = []
    for filepath in inputFilepaths:
        f = open(os.path.join(dirPath, filepath), 'r')
        text = f.read()
        f.close()
        lines = text.split("\n") # Split logfile on newlines (each line is an epoch)
        lines = lines[1:]  # First line is just explanatory text
        lines = lines[:-1] # Last line is empty string so remove it
        lines = [ele.split(",") for ele in lines] # Map to 2d list s.t. each outer list is epoch, each inner list is epoch, accuracy, loss function as string values
        xvals = [float(ele[0]) for ele in lines] # Contains every epoch number as xes
        yvals = [float(ele[2]) for ele in lines] # contains accuracy for every epoch
        xToadd = []
        yToadd = []
        # Smooths out plot by averageing some datapoints
        xSum = 0
        xCount = 0
        for num in range(0, len(xvals)):
            xSum += xvals[num]
            xCount += 1
            if num % moduloSmoothing == 0:
                xToadd.append(xSum/xCount)
                xSum = 0
                xCount = 0
        ySum = 0
        yCount = 0
        for num in range(0, len(yvals)):
            ySum += yvals[num]
            yCount += 1
            if num % moduloSmoothing == 0:
                yToadd.append(ySum/yCount)
                ySum = 0
                yCount = 0
        xList.append(xToadd)
        yList.append(yToadd)
    for num in range(0, len(xList)): # For everly line to create
        plt.plot(xList[num], yList[num], label=functionNames[num])
    plt.title(graphTitle)
    plt.xlabel(xAxisName)
    plt.ylabel(yAxisName)
    plt.legend()
    plt.savefig(outputFilepath)
# Create graphs by epoch-accuracyMetric
def create_and_save_graph_accuracy(inputFilepaths, functionNames, outputFilepath, xAxisName, yAxisName, graphTitle,
                                   moduloSmoothing):
    plt.clf()
    dirPath = os.path.abspath(os.getcwd())
    xList = []
    yList = []
    for filepath in inputFilepaths:
        f = open(os.path.join(dirPath, filepath), 'r')
        text = f.read()
        f.close()
        lines = text.split("\n")  # Split logfile on newlines (each line is an epoch)
        lines = lines[1:]  # First line is just explanatory text
        lines = lines[:-1]  # Last line is empty string so remove it
        lines = [ele.split(",") for ele in
                 lines]  # Map to 2d list s.t. each outer list is epoch, each inner list is epoch, accuracy, loss function as string values
        xvals = [float(ele[0]) for ele in lines]  # Contains every epoch number as xes
        yvals = [float(ele[1]) for ele in lines]  # contains accuracy for every epoch
        xToadd = []
        yToadd = []
        # Smooths out plot by averageing some datapoints
        xSum = 0
        xCount = 0
        for num in range(0, len(xvals)):
            xSum += xvals[num]
            xCount += 1
            if num % moduloSmoothing == 0:
                xToadd.append(xSum / xCount)
                xSum = 0
                xCount = 0
        ySum = 0
        yCount = 0
        for num in range(0, len(yvals)):
            ySum += yvals[num]
            yCount += 1
            if num % moduloSmoothing == 0:
                yToadd.append(ySum / yCount)
                ySum = 0
                yCount = 0
        xList.append(xToadd)
        yList.append(yToadd)
    for num in range(0, len(xList)):  # For everly line to create
        plt.plot(xList[num], yList[num], label=functionNames[num])
    plt.title(graphTitle)
    plt.xlabel(xAxisName)
    plt.ylabel(yAxisName)
    plt.legend()
    plt.savefig(outputFilepath)

# Create graphs comparing learning rates for each model
def create_learning_rate_by_activation_function_deep12():
    create_and_save_graph_accuracy(["Logs\\deep12-0.1-leakyrelu.csv", "Logs\\deep12-0.01-leakyrelu.csv",
                           "Logs\\deep12-0.001-leakyrelu.csv", "Logs\\deep12-0.0001-leakyrelu.csv",
                           "Logs\\deep12-0.00001-leakyrelu.csv"],
                          ["a=0.1", "a=0.01", "a=0.001", "a=0.0001", "a=0.00001"],
                          os.path.join(os.path.abspath(os.getcwd()),
                                       "graphs\\deep11leakyrelu-smooth20-activity.png"), "epoch", "accuracy",
                          "Deep11 LeakyReLu learning rates", 20)
    create_and_save_graph_accuracy(["Logs\\deep12-0.1-relu.csv", "Logs\\deep12-0.01-relu.csv",
                           "Logs\\deep12-0.001-relu.csv", "Logs\\deep12-0.0001-relu.csv",
                           "Logs\\deep12-0.00001-relu.csv"],
                          ["a=0.1", "a=0.01", "a=0.001", "a=0.0001", "a=0.00001"],
                          os.path.join(os.path.abspath(os.getcwd()),
                                       "graphs\\deep11relu-smooth20-activity.png"), "epoch", "accuracy",
                          "Deep11 relu learning rates", 20)
    create_and_save_graph_accuracy(["Logs\\deep12-0.1-sigmoid.csv", "Logs\\deep12-0.01-sigmoid.csv",
                           "Logs\\deep12-0.001-sigmoid.csv", "Logs\\deep12-0.0001-sigmoid.csv",
                           "Logs\\deep12-0.00001-sigmoid.csv"],
                          ["a=0.1", "a=0.01", "a=0.001", "a=0.0001", "a=0.00001"],
                          os.path.join(os.path.abspath(os.getcwd()),
                                       "graphs\\deep11lsigmoid-smooth20-activity.png"), "epoch", "accuracy",
                          "Deep11 sigmoid learning rates", 20)
    create_and_save_graph_lossFunc(["Logs\\deep12-0.1-leakyrelu.csv", "Logs\\deep12-0.01-leakyrelu.csv",
                                    "Logs\\deep12-0.001-leakyrelu.csv", "Logs\\deep12-0.0001-leakyrelu.csv",
                                    "Logs\\deep12-0.00001-leakyrelu.csv"],
                                   ["a=0.1", "a=0.01", "a=0.001", "a=0.0001", "a=0.00001"],
                                   os.path.join(os.path.abspath(os.getcwd()),
                                                "graphs\\deep11leakyrelu-smooth20-lossFunc.png"), "epoch", "categorical crossentropy loss",
                                   "Deep11 LeakyReLu learning rates", 20)
    create_and_save_graph_lossFunc(["Logs\\deep12-0.1-relu.csv", "Logs\\deep12-0.01-relu.csv",
                                    "Logs\\deep12-0.001-relu.csv", "Logs\\deep12-0.0001-relu.csv",
                                    "Logs\\deep12-0.00001-relu.csv"],
                                   ["a=0.1", "a=0.01", "a=0.001", "a=0.0001", "a=0.00001"],
                                   os.path.join(os.path.abspath(os.getcwd()),
                                                "graphs\\deep11relu-smooth20-lossFunc.png"), "epoch", "categorical crossentropy loss",
                                   "Deep11 relu learning rates", 20)
    create_and_save_graph_lossFunc(["Logs\\deep12-0.1-sigmoid.csv", "Logs\\deep12-0.01-sigmoid.csv",
                                    "Logs\\deep12-0.001-sigmoid.csv", "Logs\\deep12-0.0001-sigmoid.csv",
                                    "Logs\\deep12-0.00001-sigmoid.csv"],
                                   ["a=0.1", "a=0.01", "a=0.001", "a=0.0001", "a=0.00001"],
                                   os.path.join(os.path.abspath(os.getcwd()),
                                                "graphs\\deep11lsigmoid-smooth20-lossFunc.png"), "epoch", "categorical crossentropy loss",
                                   "Deep11 sigmoid learning rates", 20)

def create_learning_rate_by_activation_function_shallow150():
    create_and_save_graph_accuracy(["Logs\\shallow150-0.1-leakyrelu.csv", "Logs\\shallow150-0.01-leakyrelu.csv",
                           "Logs\\shallow150-0.001-leakyrelu.csv", "Logs\\shallow150-0.0001-leakyrelu.csv",
                           "Logs\\shallow150-0.00001-leakyrelu.csv"], ["a=0.1", "a=0.01", "a=0.001", "a=0.0001", "a=0.00001"],
                          os.path.join(os.path.abspath(os.getcwd()),
                                       "graphs\\shallow150leakyrelu-smooth20-activity.png"), "epoch", "accuracy",
                          "Shallow150 leakyRelu learning rates", 20)
    create_and_save_graph_accuracy(["Logs\\shallow150-0.1-relu.csv", "Logs\\shallow150-0.01-relu.csv",
                           "Logs\\shallow150-0.001-relu.csv", "Logs\\shallow150-0.0001-relu.csv",
                           "Logs\\shallow150-0.00001-relu.csv"],
                          ["a=0.1", "a=0.01", "a=0.001", "a=0.0001", "a=0.00001"],
                          os.path.join(os.path.abspath(os.getcwd()),
                                       "graphs\\shallow150relu-smooth20-activity.png"), "epoch", "accuracy",
                          "Shallow150 relu learning rates", 20)
    create_and_save_graph_accuracy(["Logs\\shallow150-0.1-sigmoid.csv", "Logs\\shallow150-0.01-sigmoid.csv",
                           "Logs\\shallow150-0.001-sigmoid.csv", "Logs\\shallow150-0.0001-sigmoid.csv",
                           "Logs\\shallow150-0.00001-sigmoid.csv"],
                          ["a=0.1", "a=0.01", "a=0.001", "a=0.0001", "a=0.00001"],
                          os.path.join(os.path.abspath(os.getcwd()),
                                       "graphs\\shallow150sigmoid-smooth20-activity.png"), "epoch", "accuracy",
                          "Shallow150 sigmoid learning rates", 20)
    create_and_save_graph_lossFunc(["Logs\\shallow150-0.1-leakyrelu.csv", "Logs\\shallow150-0.01-leakyrelu.csv",
                           "Logs\\shallow150-0.001-leakyrelu.csv", "Logs\\shallow150-0.0001-leakyrelu.csv",
                           "Logs\\shallow150-0.00001-leakyrelu.csv"], ["a=0.1", "a=0.01", "a=0.001", "a=0.0001", "a=0.00001"],
                          os.path.join(os.path.abspath(os.getcwd()),
                                       "graphs\\shallow150leakyrelu-smooth20-lossFunc.png"), "epoch", "categorical crossentropy loss",
                          "Shallow150 leakyRelu learning rates", 20)
    create_and_save_graph_lossFunc(["Logs\\shallow150-0.1-relu.csv", "Logs\\shallow150-0.01-relu.csv",
                           "Logs\\shallow150-0.001-relu.csv", "Logs\\shallow150-0.0001-relu.csv",
                           "Logs\\shallow150-0.00001-relu.csv"],
                          ["a=0.1", "a=0.01", "a=0.001", "a=0.0001", "a=0.00001"],
                          os.path.join(os.path.abspath(os.getcwd()),
                                       "graphs\\shallow150relu-smooth20-lossFunc.png"), "epoch", "categorical crossentropy loss",
                          "Shallow150 relu learning rates", 20)
    create_and_save_graph_lossFunc(["Logs\\shallow150-0.1-sigmoid.csv", "Logs\\shallow150-0.01-sigmoid.csv",
                           "Logs\\shallow150-0.001-sigmoid.csv", "Logs\\shallow150-0.0001-sigmoid.csv",
                           "Logs\\shallow150-0.00001-sigmoid.csv"],
                          ["a=0.1", "a=0.01", "a=0.001", "a=0.0001", "a=0.00001"],
                          os.path.join(os.path.abspath(os.getcwd()),
                                       "graphs\\shallow150sigmoid-smooth20-lossFunc.png"), "epoch", "categorical crossentropy loss",
                          "Shallow150 sigmoid learning rates", 20)

def create_learning_rate_by_activation_function_depth3():
    create_and_save_graph_accuracy(["Logs\\depth3-0.1-leakyrelu.csv", "Logs\\depth3-0.01-leakyrelu.csv",
                           "Logs\\depth3-0.001-leakyrelu.csv", "Logs\\depth3-0.0001-leakyrelu.csv",
                           "Logs\\depth3-0.00001-leakyrelu.csv"], ["a=0.1", "a=0.01", "a=0.001", "a=0.0001", "a=0.00001"],
                          os.path.join(os.path.abspath(os.getcwd()),
                                       "graphs\\depth3leakyrelu-smooth20-activity.png"), "epoch", "accuracy",
                          "Depth3 leakyRelu learning rates", 20)
    create_and_save_graph_accuracy(["Logs\\depth3-0.1-relu.csv", "Logs\\depth3-0.01-relu.csv",
                           "Logs\\depth3-0.001-relu.csv", "Logs\\depth3-0.0001-relu.csv",
                           "Logs\\depth3-0.00001-relu.csv"],
                          ["a=0.1", "a=0.01", "a=0.001", "a=0.0001", "a=0.00001"],
                          os.path.join(os.path.abspath(os.getcwd()),
                                       "graphs\\depth3relu-smooth20-activity.png"), "epoch", "accuracy",
                          "Depth3 relu learning rates", 20)
    create_and_save_graph_accuracy(["Logs\\depth3-0.1-sigmoid.csv", "Logs\\depth3-0.01-sigmoid.csv",
                           "Logs\\depth3-0.001-sigmoid.csv", "Logs\\depth3-0.0001-sigmoid.csv",
                           "Logs\\depth3-0.00001-sigmoid.csv"],
                          ["a=0.1", "a=0.01", "a=0.001", "a=0.0001", "a=0.00001"],
                          os.path.join(os.path.abspath(os.getcwd()),
                                       "graphs\\depth3sigmoid-smooth20-activity.png"), "epoch", "accuracy",
                          "Depth3 sigmoid learning rates", 20)
    create_and_save_graph_lossFunc(["Logs\\depth3-0.1-leakyrelu.csv", "Logs\\depth3-0.01-leakyrelu.csv",
                                    "Logs\\depth3-0.001-leakyrelu.csv", "Logs\\depth3-0.0001-leakyrelu.csv",
                                    "Logs\\depth3-0.00001-leakyrelu.csv"],
                                   ["a=0.1", "a=0.01", "a=0.001", "a=0.0001", "a=0.00001"],
                                   os.path.join(os.path.abspath(os.getcwd()),
                                                "graphs\\depth3leakyrelu-smooth20-lossFunc.png"), "epoch", "categorical crossentropy loss",
                                   "Depth3 leakyRelu learning rates", 20)
    create_and_save_graph_lossFunc(["Logs\\depth3-0.1-relu.csv", "Logs\\depth3-0.01-relu.csv",
                                    "Logs\\depth3-0.001-relu.csv", "Logs\\depth3-0.0001-relu.csv",
                                    "Logs\\depth3-0.00001-relu.csv"],
                                   ["a=0.1", "a=0.01", "a=0.001", "a=0.0001", "a=0.00001"],
                                   os.path.join(os.path.abspath(os.getcwd()),
                                                "graphs\\depth3relu-smooth20-lossFunc.png"), "epoch", "categorical crossentropy loss",
                                   "Depth3 relu learning rates", 20)
    create_and_save_graph_lossFunc(["Logs\\depth3-0.1-sigmoid.csv", "Logs\\depth3-0.01-sigmoid.csv",
                                    "Logs\\depth3-0.001-sigmoid.csv", "Logs\\depth3-0.0001-sigmoid.csv",
                                    "Logs\\depth3-0.00001-sigmoid.csv"],
                                   ["a=0.1", "a=0.01", "a=0.001", "a=0.0001", "a=0.00001"],
                                   os.path.join(os.path.abspath(os.getcwd()),
                                                "graphs\\depth3sigmoid-smooth20-lossFunc.png"), "epoch", "categorical crossentropy loss",
                                   "Depth3 sigmoid learning rates", 20)

def create_learning_rate_by_activation_function_all_networks():
    create_learning_rate_by_activation_function_deep12()
    create_learning_rate_by_activation_function_depth3()
    create_learning_rate_by_activation_function_shallow150()

# A smoothing of 10 seems to be just right
# create_learning_rate_by_activation_function_all_networks()
# Much clearer by looking at loss function
# Deep11 leakyrelu: a=0.1
# Deep11 sigmoid: a=0.1 and a = 0.01 seem basically identical
# Deep11 relu: a = 0.001 only achieved good performance
# Deep3 leakyrelu: a = 0.1
# Deep3 relu: a=0.01
# Deep3 sigmoid: a=0.1
# Shallow150 leakyRelu: a=0.1
# Shallow150 relu: a = 0.1
# Shallow 150 sigmoid: a = 0.1

def create_activation_function_comparison_deep12():
    create_and_save_graph_accuracy(["Logs\\deep12-0.1-leakyrelu.csv", "Logs\\deep12-0.1-sigmoid.csv", "Logs\\deep12-0.001-relu.csv"],
                                   ["leakyReLu, a=0.1", "sigmoid, a=0.1", "relu, a=0.001"],
                                   os.path.join(os.path.abspath(os.getcwd()),
                                                "graphs\\deep11-acivation-funcs-smooth20-accuracy.png"), "epoch", "accuracy",
                                   "Deep11 LeakyReLu learning rates", 20)



