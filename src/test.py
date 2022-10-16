from tkinter.tix import COLUMN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from math import sqrt
import seaborn as sns
import statsmodels.api   as sm
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
    # print(model.coef_)
    # print(model.intercept_)
    # print(model.score(train_X, train_y))
    # print(model.score(test_X, test_y))
    predict_y_test = model.predict(test_X)
    predict_y_train = model.predict(train_X)
    rms_train = sqrt(mean_squared_error(train_y, predict_y_train))
    rms_test = sqrt(mean_squared_error(test_y, predict_y_test))
    print('train RMSE: ', rms_train)
    print('test RMSE: ', rms_test)
