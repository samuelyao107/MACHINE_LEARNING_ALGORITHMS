import numpy as np

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1/(1 +np.exp(-x))

class LogisticRegression:
    def __init__(self, learning_rate=0.002, number_iterations=2000, threshold = 0.5):
        self.learning_rate=learning_rate
        self.number_iterations = number_iterations
        self.threshold = threshold
        self.weights= None
        self.bias = None

    def fit(self, X, y):
        number_samples, number_features = X.shape
        self.weights = np.zeros(number_features)
        self.bias = 0

        for _ in range(self.number_iterations):
            linear_predictions = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_predictions)

            dw = (1/number_samples) * np.dot(X.T, (predictions - y))
            db = (1/number_samples) * np.sum(predictions - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self,X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        class_predicted = [0 if y<= self.threshold else 1 for y in y_pred]
        return class_predicted
