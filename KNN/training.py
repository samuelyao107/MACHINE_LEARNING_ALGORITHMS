import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from KNN import KNN
cmap= ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


data = datasets.load_iris()
X,y= data.data, data.target

#KNN is a lazy learner no need for splitting the data
#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1500)


plt.figure()
plt.scatter(X[:,2],X[:,3], c=y, cmap=cmap, edgecolor='k', s=20)
plt.show()

model = KNN(k=5)
model.fit(X, y)
predicted_classes = model.predict(X)

print(predicted_classes)

accuracy =np.sum (predicted_classes == y) * 100 / len(y)

print(accuracy)