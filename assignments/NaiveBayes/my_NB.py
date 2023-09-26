import pandas as pd
import numpy as np
from collections import Counter

class my_NB:

    def __init__(self, alpha=1):
        # alpha: smoothing factor
        # P(xi = t | y = c) = (N(t,c) + alpha) / (N(c) + n(i)*alpha)
        # where n(i) is the number of available categories (values) of feature i
        # Setting alpha = 1 is called Laplace smoothing
        self.alpha = alpha

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, str
        # y: list, np.array or pd.Series, dependent variables, int or str
        # list of classes for this model
        self.classes_ = list(set(list(y)))
        # for calculation of P(y)
        self.P_y = Counter(y)
        # self.P[yj][Xi][xi] = P(xi|yj) where Xi is the feature name and xi is the feature value, yj is a specific class label
        # make sure to use self.alpha in the __init__() function as the smoothing factor when calculating P(xi|yj)
        self.P = {}
        for label in self.classes_:
            if label not in self.P:
                self.P[label] = {}
            for feature in X:
                count= np.count_nonzero(X[feature][y==label])
                counter= Counter(X[feature][y==label])
                self.P[label][feature]={}
                nof_unique_feature= len(list(set(X[feature])))

                for temp in X[feature]:
                    if counter.get(temp)==None:
                        self.P[label][feature][temp]= self.alpha/(count+ (nof_unique_feature* self.alpha))
                    else:
                        self.P[label][feature][temp]= (counter.get(temp)+ self.alpha)/ (count+(nof_unique_feature*self.alpha))
        return

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, str
        # prob is a dict of prediction probabilities belonging to each categories
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        # P(yj|x) = P(x|yj)P(yj)/P(x)
        # P(x|yj) = P(x1|yj)P(x2|yj)...P(xk|yj) = self.P[yj][X1][x1]*self.P[yj][X2][x2]*...*self.P[yj][Xk][xk]
        probs = {}
        for label in self.classes_:
            p = self.P_y[label]
            for key in X:
                p *= X[key].apply(lambda value: self.P[label][key][value] if value in self.P[label][key] else 1)
            probs[label] = p
        probs = pd.DataFrame(probs, columns=self.classes_)
        sums = probs.sum(axis=1)
        probs = probs.apply(lambda v: v / sums)
        return probs

    def predict(self, X):
        # X: pd.DataFrame, independent variables, str
        # return predictions: list
        # Hint: predicted class is the class with highest prediction probability (from self.predict_proba)
        probs = self.predict_proba(X)
        key = probs.idxmax(axis=1, skipna=True)
        predictions = key.tolist()
        return predictions





