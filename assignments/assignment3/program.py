import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn import ensemble, tree, metrics
from mlxtend.evaluate import bias_variance_decomp

import matplotlib.pyplot as plt



def decisionTree(maxDepth, lstNodeSize, X_train, Y_train):
        clf = tree.DecisionTreeRegressor(
            max_depth=maxDepth, min_samples_leaf=lstNodeSize, random_state=0)
        clf.fit(X_train, Y_train)
        print('accuracy: ', clf.score(X_train, Y_train))
        plt.figure(figsize=(20, 10))
        tree.plot_tree(clf, filled=True)
        print(" maximum depth: ",
              maxDepth, "least node size: ", lstNodeSize)
        print("The train error(SSE) is: ", metrics.mean_squared_error(
            clf.predict(X_train), Y_train)*300)
        print("The test error(SSE) is: ", metrics.mean_squared_error(
            clf.predict(X_test), Y_test)*100)

def baggingOfTrees(numOfTree, maxDepth,X_train, Y_train):
        regressor = tree.DecisionTreeRegressor(
            max_depth=maxDepth, random_state=0)
        clf = ensemble.BaggingRegressor(
            base_estimator=regressor, n_estimators=numOfTree, random_state=0)
        clf.fit(X_train, Y_train)
        print("bagging with ", numOfTree,
              " trees, maximum depth:  ", maxDepth)
        print("maximum depth: ", maxDepth)
        print("number of trees: ", numOfTree)
        print("train error(SSE): ", metrics.mean_squared_error(
            clf.predict(X_train), Y_train)*300)
        print("test error(SSE): ", metrics.mean_squared_error(
            clf.predict(X_test), Y_test)*100)
def randomForests(numOfTree, m, X_train, Y_train):
        clf = ensemble.RandomForestRegressor(
            n_estimators=numOfTree, max_features=m, random_state=0)
        clf.fit(X_train, Y_train)
        print("For the random forests with ", numOfTree, " trees and m = ", m)
        print("The train error(SSE) is: ", metrics.mean_squared_error(
            clf.predict(X_train), Y_train)*300)
        print("The test error(SSE) is: ", metrics.mean_squared_error(
            clf.predict(X_test), Y_test)*100)
def curves(X_train, Y_train, X_test, Y_test):
        numOfTrees = range(10, 101, 10)
        bias = []
        variance = []
        for i in numOfTrees:
            forest = ensemble.RandomForestRegressor(
                n_estimators=i, random_state=0)
            mse, b, var = bias_variance_decomp(
                forest, X_train, Y_train, X_test, Y_test, loss='mse')
            bias.append(b**2)
            variance.append(var)
            print(mse, b**2, var)
        plt.figure()
        plt.plot(numOfTrees, bias)
        plt.xlabel('The number of trees')
        plt.ylabel('square of bias')
        plt.figure()
        plt.plot(numOfTrees, variance)
        plt.xlabel('The number of trees')
        plt.ylabel('variance')
        plt.show()

data = pd.read_csv('/home/george/Course/dda3030/assignments/assignment3/Carseats.csv')
data = data.reset_index()
data.drop(data.columns[[0]], axis=1, inplace=True)
data.loc[data.ShelveLoc == 'Good', 'ShelveLoc'] = 2
data.loc[data.ShelveLoc == 'Medium', 'ShelveLoc'] = 1
data.loc[data.ShelveLoc == 'Bad', 'ShelveLoc'] = 0
data['ShelveLoc'] = data['ShelveLoc'].astype(int)
data.loc[data.Urban == 'Yes', 'Urban'] = 1
data.loc[data.Urban == 'No', 'Urban'] = 0
data['Urban'] = data['Urban'].astype(int)
data.loc[data.US == 'Yes', "US"] = 1
data.loc[data.US == 'No', "US"] = 0
data['US'] = data['US'].astype(int)
dataset = data.values
X_train = dataset[:300, 1:]
Y_train = dataset[:300, 0]
X_test = dataset[300:, 1:]
Y_test = dataset[300:, 0]

# print(dataset)

for maxD in range(4, 11, 2):
    for lstNd in range(2, 20, 5):
        decisionTree(maxD, lstNd,X_train,Y_train)
for numOfT in range(10, 51, 10):
    for maxD in range(4, 11, 2):
        baggingOfTrees(numOfT, maxD, X_train, Y_train)
for numOfT in range(10, 51, 10):
    for m in range(2, 10, 2):
        randomForests(numOfT, m, X_train, Y_train)
curves( X_train, Y_train, X_test, Y_test)
plt.show()








