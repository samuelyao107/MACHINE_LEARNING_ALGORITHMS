import sys
import os
sys.path.append(os.getenv("MY_PROJECT_PATH"))
from DECISION_TREES.DecisionTrees import DecisionTree
from collections import Counter
import numpy as np


class RandomForest:
    def __init__(self,number_trees = 15, max_depth=15, min_samples_split=2, number_features=None):
        self.number_trees = number_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.number_features = number_features
        self.trees = []

    def fit(self,X,y):
        self.trees = []
        for _ in range(self.number_trees):
            tree=DecisionTree(max_depth=self.max_depth, 
                               min_samples_split=self.min_samples_split,
                                 n_features =self.number_features)
            
            X_sample, y_sample = self._get_samples(X,y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)


    def _most_present_label(self, y):
    
        return  Counter(y).most_common(1)[0][0]       


    def _get_samples(self, X,y):
        number_samples = X.shape[0]
        indexes = np.random.choice(number_samples, number_samples, replace = True)
        return X[indexes], y[indexes]
    
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        all_predictions = np.swapaxes(predictions,0,1)
        return np.array([self._most_present_label(prediction) for prediction in all_predictions])



            



