import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import sys
from pdb import set_trace
from sklearn.model_selection import GridSearchCV

##################################
sys.path.insert(0, '../..')
from assignments.Evaluation.my_evaluation import my_evaluation

class my_model():
    def fit(self, X, y):
        # do not exceed 29 mins
        text_cols = ["title", "location", "description", "requirements" ]
        for i in text_cols:
            X[i] = X[i].str.lower()
            X[i] = X[i].str.replace('[^a-zA-Z]', ' ')
            X[i] = X[i].str.replace('!', '')
        X['cleaned'] = X['description']
        self.preprocessor = TfidfVectorizer(stop_words='english', norm='l2',use_idf=False, smooth_idf=False)
        XX = self.preprocessor.fit_transform(X["description"])
        params = {"loss": ["hinge","log","perceptron"],
                  "alpha": [0.0001, 0.001, 0.01, 0.1 ],
                  "penalty": ["l2","l1"],
                  "max_iter": [1000,10000,100000],
                  "class_weight": ["balanced"],
                  "epsilon": [0.0001,0.001,0.01,0.1],
                  "validation_fraction": [0.2,0.6]}
        model = SGDClassifier()
        self.clf = GridSearchCV(model, param_grid=params)
        self.clf.fit(XX, y)

        return

    def predict(self, X):
        text_cols = [ "title", "location", "description", "requirements" ]
        for i in text_cols:
            X[i] = X[i].str.lower()
            X[i] = X[i].str.replace('[^a-zA-Z]', ' ')
            X[i] = X[i].str.replace('!', '')
        X['cleaned'] = X['description']
        XX = self.preprocessor.transform(X["description"])
        predictions = self.clf.predict(XX)
        return predictions



