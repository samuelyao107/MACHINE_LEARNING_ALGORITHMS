import numpy as np


class PCA:

    def __init__(self,number_components):
        self.number_components = number_components
        self.components = None
        self.mean = None

    def fit(self, X):

        self.mean = np.mean(X, axis=0)  
        X -= self.mean  

        cov = np.cov(X.T)

        eigenvectors, eigenvalues = np.linalg.eig(cov)

        eigenvectors = eigenvectors.T

        indexes = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[indexes]
        eigenvectors = eigenvectors[indexes]

        self.components = eigenvectors[:self.number_components]

    def transform(self, X):

        X -= self.mean
        return np.dot(X, self.components.T)





