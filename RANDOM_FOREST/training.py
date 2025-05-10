import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from RandomForest import RandomForest

data = datasets.load_breast_cancer()
X,y= data.data, data.target


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1500)


model = RandomForest()
model.fit(X_train, y_train)
predicted_classes = model.predict(X_test)

print(predicted_classes)

accuracy =np.sum (predicted_classes == y_test) * 100 / len(y_test)

print(accuracy)