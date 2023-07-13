import pandas as pd
import numpy as np
from LR_mini import LogisticRegression_Mini
from LR_stoc import LogisticRegression_Stochastic
from LR import LogisticRegression_Batch

label_map = {'B': 0, 'M': 1}
df = pd.read_csv("Dsata Set for Assignment 1.csv",usecols=range(1,32),dtype={'diagnosis': 'category'}).replace({'diagnosis': label_map})

print(df)

for _ in range(2):
    TrainingDS = df.sample(frac=0.67, random_state=1234).reset_index(drop=True)
    # print(TrainingDS)
    TestingDS = df.drop(TrainingDS.index)
    # print(TestingDS)

    X_train1 = TrainingDS.drop('diagnosis', axis=1)
    # print(X_train1)

    y_train1 = TrainingDS['diagnosis']
    # print(y_train1)

    X_train = X_train1.to_numpy(na_value=0)
    # print(X_train)

    y_train = y_train1.to_numpy(na_value=0)
    # print(y_train)

    X_test1 = TestingDS.drop('diagnosis', axis=1)
    # print(X_test1)

    y_test1 = TestingDS['diagnosis']
    # print(y_test1)

    X_test = X_test1.to_numpy(na_value=0)
    y_test = y_test1.to_numpy(na_value=0)

    Normalized_X_train = ((X_train - np.nanmean(X_train,axis=0))/np.nanstd(X_train,axis=0))
    Normalized_X_test = ((X_test - np.nanmean(X_test,axis=0))/ np.nanstd(X_test,axis=0))

    clf = LogisticRegression_Batch(lr=0.001, epochs=10000)
    clf.fit(Normalized_X_train, y_train)
    y_pred = clf.predict(Normalized_X_test)


    def accuracy(y_pred, y_test):
        return np.sum(y_pred == y_test) / len(y_test)


    acc = accuracy(y_pred, y_test)
    print(acc)

    clf2 = LogisticRegression_Stochastic(lr=0.01, epochs=100)
    clf2.fit(Normalized_X_train, y_train)
    y_pred = clf2.predict(Normalized_X_test)


    def accuracy(y_pred, y_test):
        return np.sum(y_pred == y_test) / len(y_test)


    acc = accuracy(y_pred, y_test)
    print(acc)

    clf3 = LogisticRegression_Mini(lr=0.01, epochs=500, batch_size=10)
    clf3.fit(Normalized_X_train, y_train)
    y_pred = clf3.predict(Normalized_X_test)


    def accuracy(y_pred, y_test):
        return np.sum(y_pred == y_test) / len(y_test)


    acc = accuracy(y_pred, y_test)
    print(acc)

