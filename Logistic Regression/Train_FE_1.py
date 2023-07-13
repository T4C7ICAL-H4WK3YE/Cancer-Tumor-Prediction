import pandas as pd
import numpy as np
from LR_mini import LogisticRegression_Mini
from LR_stoc import LogisticRegression_Stochastic
from LR import LogisticRegression_Batch

label_map = {'B': 0, 'M': 1}
df = pd.read_csv("Dsata Set for Assignment 1.csv",usecols=range(1,32),dtype={'diagnosis': 'category'}).replace({'diagnosis': label_map})

for _ in range(2):
    TrainingDS = df.sample(frac=0.67, random_state=1234).reset_index(drop=True)
    # print(TrainingDS)
    TestingDS = df.drop(TrainingDS.index)
    # print(TestingDS)

    X_train1 = TrainingDS.drop('diagnosis', axis=1)
    X_train1.fillna(X_train1.mean(), inplace=True)
    # print(X_train1)

    y_train1 = TrainingDS['diagnosis']
    # print(y_train1)

    X_train = X_train1.to_numpy(na_value=0)
    # print(X_train)

    y_train = y_train1.to_numpy(na_value=0)
    # print(y_train)

    X_test1 = TestingDS.drop('diagnosis', axis=1)
    X_test1.fillna(X_test1.mean(), inplace=True)
    # print(X_test1)

    y_test1 = TestingDS['diagnosis']
    # print(y_test1)

    X_test = X_test1.to_numpy(na_value=0)
    y_test = y_test1.to_numpy(na_value=0)


    clf = LogisticRegression_Batch(lr=0.001, epochs=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)


    def accuracy(y_pred, y_test):
        return np.sum(y_pred == y_test) / len(y_test)


    acc = accuracy(y_pred, y_test)
    print(acc)

    clf2 = LogisticRegression_Stochastic(lr=0.01, epochs=200)
    clf2.fit(X_train, y_train)
    y_pred = clf2.predict(X_test)


    def accuracy(y_pred, y_test):
        return np.sum(y_pred == y_test) / len(y_test)


    acc = accuracy(y_pred, y_test)
    print(acc)

    clf3 = LogisticRegression_Mini(lr=0.01, epochs=500, batch_size=10)
    clf3.fit(X_train, y_train)
    y_pred = clf3.predict(X_test)


    def accuracy(y_pred, y_test):
        return np.sum(y_pred == y_test) / len(y_test)


    acc = accuracy(y_pred, y_test)
    print(acc)
