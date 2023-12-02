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

    def fit(self, dataset, k=100):
        '''
        Fits the model to the data.

        Args:
            dataset (ExtremeDataset): dataset
            k (int): number of extreme samples to use
        '''
        X, y, boundary = dataset.get_extreme(k)
        X_proj = X / np.linalg.norm(X, axis=1)[:,np.newaxis]
        self.boundary = boundary

        self.model.fit(X_proj, y)

    def predict(self, dataset, percentage=0.1):
        '''
        Predicts the labels for the data.

        Args:
            dataset (ExtremeDataset): dataset

        Returns:
            labels (np.ndarray): predicted labels
        '''

        extreme_dataset = dataset.make_extreme(self.boundary['norm'])
        extreme_X, y_true, _ = extreme_dataset.get_extreme(percentage)

        extreme_X = extreme_X / np.linalg.norm(extreme_X, axis=1)[:,np.newaxis]
        y_pred = self.model.predict(extreme_X)

        return y_pred, y_true

    def score(self, dataset, percentage=0.1):
        '''
        Returns the score of the model.

        Args:
            dataset (ExtremeDataset): dataset

        Returns:
            score (float): score
        '''
        y_pred, y_true = self.predict(dataset, percentage)
        score = accuracy_score(y_true, y_pred > 0.5)
        return score

    def hamming_loss(self, dataset, percentage=0.1):
        '''
        Returns the hamming loss of the model.

        Args:
            dataset (ExtremeDataset): dataset

        Returns:
            score (float): hamming loss
        '''
        y_pred, y_true = self.predict(dataset, percentage)
        score = hamming_loss(y_true, y_pred > 0.5)
        return score