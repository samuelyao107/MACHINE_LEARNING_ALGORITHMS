from sklearn import datasets
import matplotlib.pyplot as plt
from PCA import PCA


data = datasets.load_iris()
X = data.data
y = data.target


pca = PCA(2)

pca.fit(X)

X_projected = pca.transform(X)

x0= X_projected[:,0]
x1= X_projected[:,1]

plt.scatter(x0, x1, c=y)

plt.show()


