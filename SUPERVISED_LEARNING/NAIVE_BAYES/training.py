import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from NaiveBayes import NaiveBayes

def accuracy(y_pred, y_test):
    return np.sum(y_pred == y_test)*100/ len(y_pred)

data = datasets.load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state= 1500)

model = NaiveBayes()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy(y_pred, y_test)

print(accuracy)
