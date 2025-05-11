import numpy as np
from scipy.stats import norm

class NaiveBayes:

    def fit(self, X,y):
        number_samples, number_features = X.shape
        self._classes = np.unique(y)
        number_classes = len(self._classes)

        self._mean = np.zeros((number_classes, number_features), dtype=np.float64)
        self._variance = np.zeros((number_classes, number_features), dtype=np.float64)
        self._priors = np.zeros(number_classes, dtype=np.float64)

        for index, classe in enumerate(self._classes):
            X_classe = X[y == classe]
            self._mean[index, :] = X_classe.mean(axis=0)
            self._variance[index,:] = X_classe.var(axis = 0)
            self._priors[index] = X_classe.shape[0] / float(number_samples)



    def predict(self, X):
        y_pred= [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        posteriors = []

        for index, classe in enumerate(self._classes):
            prior = np.log(self._priors[index])
            posterior = np.sum(np.log(self._pdf(index,x)))
            posterior += prior
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)] 

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._variance
        std = np.sqrt(var)

        return norm.pdf(x,mean, std)




