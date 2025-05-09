import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

def mean_squared_error(y_test, y_predicted):
    return np.mean((y_test-y_predicted)**2)

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=25, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1000)

model = LinearRegression()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test,y_pred)

prediction = model.predict(X)
print(mse)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(10,7))
c1= plt.scatter(X_train, y_train, color= "r", s=10)
c2 = plt.scatter(X_test, y_test, color= "b", marker="o", s=10) 
plt.plot(X, prediction, color='orange',linewidth=2, label='Linear Prediction')
plt.show()