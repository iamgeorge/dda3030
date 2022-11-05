import pandas as pd
from sklearn.svm import SVC


def error(test,pred):
    error = 0
    for i in range(0,len(test)):
        if(test[i] != pred[i]):
            error += 1
    return error/len(test)

def svm_linear(X_train, y_train, X_test, y_test, c):
    classifier = SVC(kernel = 'linear', C=c)
    classifier.fit(X_train, y_train)

    #Make the prediction
    y_pred_test = classifier.predict(X_test)
    y_pred_train = classifier.predict(X_train)
    


    print("training error {:.2f} % ".format(error(y_test, y_pred_test)))

    print("testing error {:.2f} % ".format(error(y_pred_train, y_train)))

    print('w = ',classifier.coef_)
    print('b = ',classifier.intercept_)
    print('Indices of support vectors = ', classifier.support_)
    f = open("SVM_linear.txt", "a")
    f.write(str(error(y_test, y_pred_test))+ '%\n')
    f.write(str(error(y_pred_train, y_train))+ '%\n')
    f.write(str(classifier.coef_)+ '\n')
    f.write(str(classifier.intercept_)+ '\n')
    f.write(str(classifier.support_)+ '\n')
    f.close()


def svm_linear_slack(X_train, y_train, X_test, y_test, c):
    classifier = SVC(kernel = 'linear', C=c)
    classifier.fit(X_train, y_train)

    #Make the prediction
    y_pred_test = classifier.predict(X_test)
    y_pred_train = classifier.predict(X_train)
    


    print("training error {:.2f} % ".format(error(y_test, y_pred_test)))

    print("testing error {:.2f} % ".format(error(y_pred_train, y_train)))



    print('w = ',classifier.coef_)
    print('b = ',classifier.intercept_)
    print('Indices of support vectors = ', classifier.support_)
    slack = abs(1 - classifier.decision_function(X_train))
    if(c!=1e5):
        print("slack variable = ", slack)

    f = open("SVM_slack.txt", "a")
    f.write(str(error(y_test, y_pred_test))+ '%\n')
    f.write(str(error(y_pred_train, y_train))+ '%\n')
    f.write(str(classifier.coef_)+ '\n')
    f.write(str(classifier.intercept_)+ '\n')
    f.write(str(classifier.support_)+ '\n')
    f.write(str(slack)+ '\n')
    f.close()
def svm_poly_slack(X_train, y_train, X_test, y_test,dgr):
    classifier = SVC(kernel = 'poly',degree = dgr, C=1)
    classifier.fit(X_train, y_train)

    #Make the prediction
    y_pred_test = classifier.predict(X_test)
    y_pred_train = classifier.predict(X_train)
    


    print("training error {:.2f} % ".format(error(y_test, y_pred_test)))

    print("testing error {:.2f} % ".format(error(y_pred_train, y_train)))
    print('b = ',classifier.intercept_)
    print('Indices of support vectors = ', classifier.support_)

    f = open(("SVM_poly" + str(dgr) +".txt"), "a")
    f.write(str(error(y_test, y_pred_test))+ '%\n')
    f.write(str(error(y_pred_train, y_train))+ '%\n')
    f.write(str(classifier.support_)+ '\n')
    f.write(str(classifier.intercept_)+ '\n')

    f.close()
 
def svm_rbf(X_train, y_train, X_test, y_test, gamma):
    classifier = SVC(kernel = 'rbf', C=1, gamma=gamma)
    classifier.fit(X_train, y_train)

    #Make the prediction
    y_pred_test = classifier.predict(X_test)
    y_pred_train = classifier.predict(X_train)
    


    print("training error {:.2f} % ".format(error(y_test, y_pred_test)))

    print("testing error {:.2f} % ".format(error(y_pred_train, y_train)))

    print('b = ',classifier.intercept_)
    print('Indices of support vectors = ', classifier.support_)

    f = open("SVM_rbf.txt", "a")
    f.write(str(error(y_test, y_pred_test))+ '%\n')
    f.write(str(error(y_pred_train, y_train))+ '%\n')
    f.write(str(classifier.support_)+ '\n')
    f.write(str(classifier.intercept_)+ '\n')

    f.close()

def svm_sig(X_train, y_train, X_test, y_test, g):
    classifier = SVC(kernel = 'sigmoid', C=1, gamma=1)
    classifier.fit(X_train, y_train)

    #Make the prediction
    y_pred_test = classifier.predict(X_test)
    y_pred_train = classifier.predict(X_train)
    


    print("training error {:.2f} % ".format(error(y_test, y_pred_test)))

    print("testing error {:.2f} % ".format(error(y_pred_train, y_train)))

    print('b = ',classifier.intercept_)
    print('Indices of support vectors = ', classifier.support_)

    f = open("SVM_sigmoid.txt", "a")
    f.write(str(error(y_test, y_pred_test))+ '%\n')
    f.write(str(error(y_pred_train, y_train))+ '%\n')
    f.write(str(classifier.support_)+ '\n')
    f.write(str(classifier.intercept_)+ '\n')

    f.close()

def CalSVM(df_train, df_test, c,SVMType):
    

    X_train = df_train.iloc[0:80,1:]
    y_train = df_train.iloc[0:80:, 0].values

    X_test = df_test.iloc[0:20,1:]
    y_test = df_test.iloc[0:20, 0].values
    print("case 1")
    SVMType(X_train, y_train, X_test, y_test,c)

    X_train = df_train.iloc[42:,1:]
    y_train = df_train.iloc[42:, 0].values

    X_test = df_test.iloc[12:,1:]
    y_test = df_test.iloc[12:, 0].values
    print("case 2")
    SVMType(X_train, y_train, X_test, y_test,c)


    data_set_train = pd.read_table("/home/george/Course/dda3030/assignments/assignment2/train.txt", skiprows=range(41, 81))
    # print(data_set)
    X_train = data_set_train.iloc[:,1:]
    y_train = data_set_train.iloc[:, 0].values
    # print(y_train)
    data_set_test = pd.read_table("/home/george/Course/dda3030/assignments/assignment2/test.txt", skiprows=range(11, 21))
    X_test = data_set_test.iloc[:,1:]
    y_test = data_set_test.iloc[:, 0].values
    print("case 3")
    SVMType(X_train, y_train, X_test, y_test,c)


    # X_train = df_train.iloc[:,1:]
    # y_train = df_train.iloc[:, 0].values

    # X_test = df_test.iloc[:,1:]
    # y_test = df_test.iloc[:, 0].values
    # print("multiple")
    # SVMType(X_train, y_train, X_test, y_test,c)

    

df_train = pd.read_table("/home/george/Course/dda3030/assignments/assignment2/train.txt") 
df_train.dropna(inplace=True)

df_test = pd.read_table("/home/george/Course/dda3030/assignments/assignment2/test.txt") 
df_test.dropna(inplace=True)

CalSVM(df_train,df_test,1e5,svm_linear)
for i in range(1,10):
    CalSVM(df_train,df_test,0.1*i,svm_linear_slack)


CalSVM(df_train,df_test,2,svm_poly_slack)
CalSVM(df_train,df_test,3,svm_poly_slack)
CalSVM(df_train,df_test,1,svm_rbf)
CalSVM(df_train,df_test,1,svm_sig)