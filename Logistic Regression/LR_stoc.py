import numpy as np
from matplotlib import pyplot as plt

def cost_func(m, t, h):
    np.seterr(divide='ignore')
    return ((-1/m)*(np.dot(t, np.log(h)) + np.dot(t,np.log(1-h))))

def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegression_Stochastic():

    def __init__(self, lr, epochs):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples = X.shape[0]
        n_features = X.shape[1]

        self.weights = np.zeros(n_features)
        self.bias = 0
        c_ls = []

        for _ in range(self.epochs):
            for i in range(n_samples):
                idx = np.random.randint(0, n_samples)

                linear_pred = np.dot(X[idx], self.weights) + self.bias
                predictions = sigmoid(linear_pred)

                dw = np.dot(X[idx].T, (predictions - y[idx]))
                db = np.sum(predictions-y[idx])

                self.weights = self.weights - self.lr*dw
                self.bias = self.bias - self.lr*db


            cost = cost_func(n_samples, y, predictions)
            if _ % 1 == 0:
                c_ls.append(cost)
        plt.plot(range(self.epochs), c_ls)
        plt.show()


    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y <= 0.7 else 1 for y in y_pred]
        return class_pred