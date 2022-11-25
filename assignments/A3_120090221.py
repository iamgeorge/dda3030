import pandas as pd
from sklearn import ensemble, tree, metrics
import matplotlib.pyplot as plt
from mlxtend.evaluate import bias_variance_decomp


class decisionTree(object):
    def __init__(self):
        self.data = None
        self.dataset = None
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None

    def loadData(self, dataset):
        data = pd.read_csv(dataset)
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
        self.data = data
        self.dataset = data.values
        self.X_train = self.dataset[:300, 1:]
        self.Y_train = self.dataset[:300, 0]
        self.X_test = self.dataset[300:, 1:]
        self.Y_test = self.dataset[300:, 0]

    def plotHistogram(self):
        self.data.hist(bins=20, figsize=(20, 15))

    def decisionTree(self, maxDepth, lstNodeSize):
        clf = tree.DecisionTreeRegressor(
            max_depth=maxDepth, min_samples_leaf=lstNodeSize, random_state=0)
        clf.fit(self.X_train, self.Y_train)
        # plt.figure(figsize=(20, 10))
        # tree.plot_tree(clf, filled=True)
        print("For the decision tree whose maximum depth is ",
              maxDepth, " and the least node size is", lstNodeSize)
        print("The train error(SSE) is: ", metrics.mean_squared_error(
            clf.predict(self.X_train), self.Y_train)*300)
        print("The test error(SSE) is: ", metrics.mean_squared_error(
            clf.predict(self.X_test), self.Y_test)*100)

    def baggingOfTrees(self, numOfTree, maxDepth):
        regressor = tree.DecisionTreeRegressor(
            max_depth=maxDepth, random_state=0)
        clf = ensemble.BaggingRegressor(
            base_estimator=regressor, n_estimators=numOfTree, random_state=0)
        clf.fit(self.X_train, self.Y_train)
        print("For the bagging of trees with ", numOfTree,
              " trees and the maximum depth  = ", maxDepth)
        print("The maximum depth is: ", maxDepth)
        print("The number of trees is", numOfTree)
        print("The train error(SSE) is: ", metrics.mean_squared_error(
            clf.predict(self.X_train), self.Y_train)*300)
        print("The test error(SSE) is: ", metrics.mean_squared_error(
            clf.predict(self.X_test), self.Y_test)*100)

    def randomForests(self, numOfTree, m):
        clf = ensemble.RandomForestRegressor(
            n_estimators=numOfTree, max_features=m, random_state=0)
        clf.fit(self.X_train, self.Y_train)
        print("For the random forests with ", numOfTree, " trees and m = ", m)
        print("The train error(SSE) is: ", metrics.mean_squared_error(
            clf.predict(self.X_train), self.Y_train)*300)
        print("The test error(SSE) is: ", metrics.mean_squared_error(
            clf.predict(self.X_test), self.Y_test)*100)

    def curves(self):
        numOfTrees = range(10, 101, 10)
        bias = []
        variance = []
        for i in numOfTrees:
            forest = ensemble.RandomForestRegressor(
                n_estimators=i, random_state=0)
            mse, b, var = bias_variance_decomp(
                forest, self.X_train, self.Y_train, self.X_test, self.Y_test, loss='mse')
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


DTclf = decisionTree()
DTclf.loadData("/home/george/Course/dda3030/assignments/assignment3/Carseats.csv")
DTclf.plotHistogram()
for maxD in range(4, 11, 2):
    for lstNd in range(2, 20, 5):
        DTclf.decisionTree(maxD, lstNd)
for numOfT in range(10, 51, 10):
    for maxD in range(4, 11, 2):
        DTclf.baggingOfTrees(numOfT, maxD)
for numOfT in range(10, 51, 10):
    for m in range(2, 10, 2):
        DTclf.randomForests(numOfT, m)
DTclf.curves()
plt.show()
