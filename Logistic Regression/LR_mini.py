import numpy as np
import matplotlib as mpl
import random
from matplotlib import pyplot as plt

def cost_func(m, t, h):
    np.seterr(divide='ignore')
    return ((-1/m)*(np.dot(t, np.log(h)) + np.dot(t,np.log(1-h))))

def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegression_Mini():

    def __init__(self, lr, epochs, batch_size):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.batch_size = batch_size

    def fit(self, X, y):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0
        c_ls = []

        for _ in range(self.epochs):
            for i in range(int(n_samples/self.batch_size)):
                idx = random.sample(range(n_samples), self.batch_size)

                linear_pred = np.dot(X[idx], self.weights) + self.bias
                predictions = sigmoid(linear_pred)

                dw = (1 / n_samples) * np.dot(X[idx].T, (predictions - y[idx]))
                db = (1 / n_samples) * np.sum(predictions - y[idx])

                self.weights = self.weights - self.lr*dw
                self.bias = self.bias - self.lr*db

            cost = cost_func(n_samples, y[:10], predictions)
            if _ % 1 == 0:
                c_ls.append(cost)
        plt.plot(range(self.epochs), c_ls)
        plt.show()


    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y <= 0.7 else 1 for y in y_pred]
        return class_pred