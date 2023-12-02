# Create sklearn compatible classifier for extreme classification
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss

import numpy as np

class ExtremeClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, model, n_classes, n_features):
        self.model = model
        self.n_classes = n_classes
        self.n_features = n_features

    def fit(self, X, y):
        '''
        Fits the model to the data.

        Args:
            X (np.ndarray): data, shape (n_samples, n_features)
            y (np.ndarray): labels, shape (n_samples, n_classes)
        '''
        X_norm = X / np.linalg.norm(X, axis=1)[:,np.newaxis]
        self.boundary = np.min(np.linalg.norm(X_norm, axis=1))
        X_proj = X / np.linalg.norm(X, axis=1)[:,np.newaxis]
        self.model.fit(X_proj, y)

    def predict(self, X):
        '''
        Predicts the labels for the data.

        Args:
            X (np.ndarray): data, shape (n_samples, n_features)

        Returns:
            labels (np.ndarray): predicted labels, shape (n_samples, n_classes)
        '''
        X_norm = np.linalg.norm(X, axis=1)
        X_valid = X_norm >= self.boundary
        
        if not np.all(X_valid):
            print("Warning: some samples are not extreme enough.")

        extreme_X = X / X_norm[:,np.newaxis]
        y_pred = self.model.predict(extreme_X)

        return y_pred

    def score(self, X, y):
        '''
        Returns the score of the model.

        Args:
            X (np.ndarray): data, shape (n_samples, n_features)
            y (np.ndarray): labels, shape (n_samples, n_classes)

        Returns:
            score (float): score
        '''
        y_pred = self.predict(X)
        score = accuracy_score(y, y_pred)
        return score

    def hamming_loss(self, X, y):
        '''
        Returns the hamming loss of the model.

        Args:
            X (np.ndarray): data, shape (n_samples, n_features)
            y (np.ndarray): labels, shape (n_samples, n_classes)

        Returns:
            score (float): hamming loss
        '''
        y_pred = self.predict(X)
        score = hamming_loss(y, y_pred)
        return score