from tkinter.tix import COLUMN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import statsmodels.api   as sm
df = pd.read_csv("/home/george/dda3030/src/Regression.csv") 
df.head()
df.drop(["station","Date"],axis=1,inplace=True)
print(df)
