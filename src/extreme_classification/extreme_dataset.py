import numpy as np

import torch
from torch.utils.data import Dataset

from extreme_classification.distributions import multivariate_logistic_distribution

class ExtremeDataset(Dataset):

    def __init__(self, X, y, ranktransform=True):
        self.X = X
        self.y = y

        if ranktransform:
            self.rank_transform()

    def rank_transform(self):
        '''
        Rank transforms the labels in the dataset.

        The rank transformation is defined as follows:
        Given a sample x = (x1, x2, ..., xn), we transform it to
        v = (v1, v2, ..., vn) where vi = 1 / (1 + Fi(xi)) and Fi is the
        cumulative distribution function of the ith dimension.
        '''
        V = np.zeros(self.X.shape)
        for i in range(self.X.shape[1]):
            V[:, i] = 1 / (1 + np.argsort(np.argsort(self.X[:, i])))
        self.X = V

    def __getitem__(self, index):
        return torch.Tensor(self.X[index]), torch.Tensor(self.y[index])

    def __len__(self):
        return len(self.X)


def BivariateLogisticDataset(ExtremeDataset):

    def __init__(self, sizes, alphas, labels=None):
        """
        Creates a dataset with bivariate logistic distributions.

        Args:
            sizes (list): list of sizes of each distribution
            alphas (list): list of alphas of each distribution
            labels (list): list of labels of each distribution
        """
        self.X = []
        self.y = []
        for size, alpha, label in zip(sizes, alphas, labels):
            X = multivariate_logistic_distribution(size, alpha, label)
            self.X.append(X)
            self.y.append(np.ones(X.shape[0]) * label)
        self.X = np.concatenate(self.X, axis=0)
        self.y = np.concatenate(self.y, axis=0)

        self.rank_transform()