from tkinter.tix import COLUMN
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

def sklearn():
    for i in range(10): 
        df = pd.read_csv("/home/george/dda3030/src/Regression.csv") 
        df.head()
        df.drop(["station","Date"],axis=1,inplace=True)
        df.dropna(inplace=True)
        x = df.iloc[:,:21]
        y = df.iloc[:,-2:]
        print(i)
        train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=i)
        model = linear_model.LinearRegression()
        model.fit(train_X, train_y)
        print(model.coef_)
        print(model.intercept_)
        # print(model.score(train_X, train_y))
        # print(model.score(test_X, test_y))
        predict_y_test = model.predict(test_X)
        predict_y_train = model.predict(train_X)
        rms_train = sqrt(mean_squared_error(train_y, predict_y_train))
        rms_test = sqrt(mean_squared_error(test_y, predict_y_test))
        # print('train RMSE: ', rms_train)
        # print('test RMSE: ', rms_test)

def linearRegression(train_x, train_y):
    x = np.array(train_x)
    x_t = x.transpose()
    return np.dot(np.dot(np.linalg.inv(np.dot(x_t, x)), x_t), train_y)

def RMSE(y, y_hat):
    np_y = np.array(y)
    np_y_hat = np.array(y_hat)
    return np.sqrt(np.mean((np_y - np_y_hat)**2))
def main():
    df = pd.read_csv("/home/george/dda3030/src/Regression.csv")
    df.drop(["station","Date"],axis=1,inplace=True)
    df.dropna(inplace=True)
    x = df.iloc[:,:21]
    y = df.iloc[:,-2:]
    train_X, test_X, train_y, test_y = train_test_split(x, y, test_size=0.2)
    # print(train_X)
    model = linearRegression(train_X, train_y)
    # print(model)

    prediction_y_test = np.dot(test_X, model)
    prediction_y_train = np.dot(train_X, model)
    rms_train = RMSE(train_y, prediction_y_train)
    rms_test = RMSE(test_y, prediction_y_test)
    print('rms_train',rms_train)
    print('rms_test',rms_test)
    return 


for i in range(10):
    print('iteration',i)
    main()