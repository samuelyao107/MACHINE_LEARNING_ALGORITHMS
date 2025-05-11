import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.002, number_of_iterations=2000):
        self.learning_rate = learning_rate
        self.number_of_iterations = number_of_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        number_of_samples, number_of_features = X.shape
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        #Gradient descent
        for iteration in range(self.number_of_iterations):
            y_predicted= np.dot(X, self.weights) + self.bias

            dw = (1/number_of_samples) * np.dot(X.T, (y_predicted-y))
            db = (1/number_of_samples) * np.sum(y_predicted-y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):

        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted
    
   