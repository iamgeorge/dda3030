import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
def gradiant_descend(W, matrix_X,train_X, matrix_Y,alpha = 0.6):
#calculate the gradient
    for i in range(100000):
        det = np.exp(np.dot(matrix_X, W))
        rowsum = -det.sum(axis = 1).repeat(3, axis = 1)
        det = det / rowsum
        for i in range(train_X.shape[0]):
            det[i, matrix_Y[i, 0]] += 1
        W = W + (alpha / len(matrix_X)) * np.dot(matrix_X.transpose(), det)
    return W
def main():
    # read data from csv file
    df = pd.read_excel("src/Classification iris.xlsx", header = None)
    X = df.iloc[:, :4]
    Y = df.iloc[:, -1:]
    #change the type of Y to int
    Y.loc[Y[4] == 'Iris-setosa', 4] = 0
    Y.loc[Y[4] == 'Iris-versicolor', 4] = 1
    Y.loc[Y[4] == 'Iris-virginica', 4] = 2
    #split the data into training set and test set
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size = 0.2)

    # create a class for logistic regression
    matrix_X = np.mat(train_X)
    matrix_Y = np.mat(train_Y)
    #create a matrix of 1
    W = np.mat(np.ones((train_X.shape[1], 3)))
    #calculate the sigmoid function
    W = gradiant_descend(W, matrix_X,train_X, matrix_Y)

    #make prediction
    pred_Y = np.dot(test_X, W).argmax(axis = 1)
    result_Y = np.array(test_Y)
    #calculate the accuracy
    error_rate = 0
    for i in range(0, len(pred_Y)):
        if (pred_Y[i] != result_Y[i]):
            error_rate = error_rate + 1
    error_rate = error_rate / len(pred_Y)
    print(error_rate)
for i in range(10):
    print('Iteration:' , i+1 )
    main()