import pandas as pd
import numpy as np
from collections import Counter

class my_KNN:

    def __init__(self, n_neighbors=5, metric="minkowski", p=2):
        # metric = {"minkowski", "euclidean", "manhattan", "cosine"}
        # p value only matters when metric = "minkowski"
        # notice that for "cosine", 1 is closest and -1 is furthest
        # therefore usually cosine_dist = 1- cosine(x,y)
        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        self.p = p

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, float
        # y: list, np.array or pd.Series, dependent variables, int or str
        self.classes_ = list(set(list(y)))
        self.X = X
        self.y = y
        return

    def dist(self,x):
        # Calculate distances of training data to a single input data point (distances from self.X to x)
        # Output np.array([distances to x])
        if self.metric == "minkowski":
            distances = np.sum(np.abs(self.X - x) ** self.p, axis=1) ** (1 / self.p)


        elif self.metric == "euclidean":
            self.p = 2
            distances = np.sum(np.abs(self.X - x) ** self.p, axis=1) ** (1 / self.p)


        elif self.metric == "manhattan":
            self.p = 1
            distances = np.sum(np.abs(self.X - x) ** self.p, axis=1) ** (1 / self.p)


        elif self.metric == "cosine":
            distances = 1 - np.sum(self.X * x, axis=1) / (
                    np.sqrt(np.sum(self.X ** 2, axis=1)) * np.sqrt(np.sum(x ** 2))
            )

        elif self.metric == "weighted_euclidean":
                 # Calculate the weighted distances
            distances = np.sum(
                np.abs(self.X - x) ** self.p * self.weights, axis=1
            ) ** (1 / self.p)


        else:
            raise Exception("Unknown criterion.")
        return distances

    def k_neighbors(self,x):
        # Return the stats of the labels of k nearest neighbors to a single input data point (np.array)
        # Output: Counter(labels of the self.n_neighbors nearest neighbors) e.g. {"Class A":3, "Class B":2}
        distances = self.dist(x)
        sorteddist = np.argsort(distances)
        Bestdistances = sorteddist[:self.n_neighbors]
        Labels =[]
        for label in Bestdistances:
              Labels.append(self.y[label])
        output = Counter(Labels)
        return output

    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        probs = self.predict_proba(X)
        predictions = [self.classes_[np.argmax(prob)] for prob in probs.to_numpy()]
        return predictions

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, float
        # prob is a dict of prediction probabilities belonging to each categories
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        probs = []
        try:
            X_feature = X[self.X.columns]
        except:
            raise Exception("Data Mismatch.")
        for val in X_feature.to_numpy():
                neighbors = self.k_neighbors(val)
                # Calculate the probability of data point x belonging to each class
                # e.g. prob = {"2": 1/3, "1": 2/3}
                prob = {}
                for x in neighbors:
                    prob[x] = neighbors[x] / self.n_neighbors
                probs.append(prob)
                # print(probs)
        probs = pd.DataFrame(probs, columns=self.classes_)
        probs = probs.replace(np.nan, 0)
        return probs



