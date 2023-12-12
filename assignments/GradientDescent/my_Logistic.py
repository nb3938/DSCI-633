import pandas as pd
import numpy as np

class my_Logistic:

    def __init__(self, learning_rate=0.1, batch_size=10, max_iter=100, shuffle=True):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.shuffle = shuffle

    def fit(self, X, y):
        data = X.to_numpy()
        self.w = np.zeros(data.shape[1])
        self.w0 = 0.0

        n = len(y)
        for epoch in range(self.max_iter):
            if self.shuffle:
                indices = np.random.permutation(n)
                data = data[indices]
                y = y[indices]

            for i in range(0, n, self.batch_size):
                X_batch = data[i:i+self.batch_size]
                y_batch = y[i:i+self.batch_size]
                self.update_weights(X_batch, y_batch)

    def update_weights(self, X_batch, y_batch):
        y_wht = self.predict_proba(X_batch)
        w = np.dot(X_batch.T, y_wht - y_batch)/len(y_batch)
        w0 = np.mean(y_wht - y_batch)
        self.w -= self.learning_rate * w
        self.w0 -= self.learning_rate * w0

    def predict_proba(self, X):
        wx = np.dot(X, self.w) + self.w0
        fx = 1.0 / (1 + np.exp(-wx))
        return fx

    def predict(self, X):
        probs = self.predict_proba(X)
        predictions = [1 if prob >= 0.5 else 0 for prob in probs]
        return predictions








