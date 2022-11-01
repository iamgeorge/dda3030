from numpy import append
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

df_train = pd.read_table("assignments/assignment2/train.txt") 
df_train.dropna(inplace=True)

df_test = pd.read_table("assignments/assignment2/train.txt") 
df_test.dropna(inplace=True)

X_train = df_train.iloc[:,1:]
y_train = df_train.iloc[:, 0].values

X_test = df_test.iloc[:,1:]
y_test = df_test.iloc[:, 0].values


classifier = SVC(kernel = 'linear', C=1e5)
classifier.fit(X_train, y_train)

#Make the prediction
y_pred = classifier.predict(X_test)


print("training error {:.2f} % ".format(1- classifier.score(X_train, y_train)))

print("testing error {:.2f} % ".format(1- classifier.score(X_test, y_test)))

print('w = ',classifier.coef_)
print('b = ',classifier.intercept_)
print('Indices of support vectors = ', classifier.support_)

# with open('readme.txt', 'w') as f:
#     f.writelines(lines)
