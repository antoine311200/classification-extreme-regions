import numpy as np
import pandas as pd

from extreme_classification.distributions import multivariate_logistic_distribution


class ExtremeDataset:
    def __init__(self, X, y, ranktransform=True):
        self.X = X
        self.y = y

        if ranktransform:
            self.rank_transform()

        self.X_norm = np.linalg.norm(self.X, axis=1)
        self.order = np.argsort(self.X_norm)[::-1]

    def rank_transform(self):
        """
        Rank transforms the labels in the dataset.

        The rank transformation is defined as follows:
        Given a sample x = (x1, x2, ..., xn), we transform it to
        v = (v1, v2, ..., vn) where vi = 1 / (1 + Fi(xi)) and Fi is the
        cumulative distribution function of the ith dimension.
        """
        V = np.zeros(self.X.shape)
        for i in range(self.X.shape[1]):
            V[:, i] = 1 / (
                1 - np.argsort(np.argsort(self.X[:, i])) / (self.X.shape[0] + 1)
            )
        self.X = V

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)

    def get_extreme(self, k):
        """
        Returns the k-th most extreme samples in the dataset according to the L1 norm.

        Args:
            k (int): number of samples to return
        """
        extreme_X = self.X[self.order[:k]]
        extreme_y = self.y[self.order[:k]]

        return extreme_X, extreme_y

    def get_boundary(self, k):
        """
        Returns the boundary between the k-th most extreme samples and the rest of the dataset.

        Args:
            k (int): number of samples to return
        """
        return self.X_norm[self.order[k]]

    def get_standard(self, k):
        """
        Returns the rest of the dataset that is not extreme.

        Args:
            k (int): numbers of extreme samples
        """
        standard_X = self.X[self.order[k:]]
        standard_y = self.y[self.order[k:]]

        return standard_X, standard_y

    def split_extreme(self, boundary):
        """
        Splits the dataset into extreme and standard.

        Args:
            boundary (float): boundary between extreme and standard
        """
        extreme_X = self.X[self.X_norm > boundary]
        extreme_y = self.y[self.X_norm > boundary]

        standard_X = self.X[self.X_norm <= boundary]
        standard_y = self.y[self.X_norm <= boundary]

        return extreme_X, extreme_y, standard_X, standard_y


class BivariateLogisticDataset(ExtremeDataset):
    def __init__(self, sizes, alphas, labels=None, ranktransform=True):
        """
        Creates a dataset with bivariate logistic distributions.

        Args:
            sizes (list): list of sizes of each distribution
            alphas (list): list of alphas of each distribution
            labels (list): list of labels of each distribution
        """
        self.X = []
        self.labels = []
        for size, alpha, label in zip(sizes, alphas, labels):
            X = multivariate_logistic_distribution(size, 2, alpha)
            self.X.append(X)
            self.labels.append(np.ones(X.shape[0]) * label)
        self.X = np.concatenate(self.X, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)

        if ranktransform:
            self.rank_transform()

        self.make_dataframe()

    def make_dataframe(self):
        """
        Creates a dataframe with the dataset and its labels.
        """
        dataframe = pd.DataFrame(self.X)
        dataframe["labels"] = self.labels
        dataframe["norm"] = np.linalg.norm(self.X, axis=1, ord=1)

        self.dataframe = dataframe

    def get_extreme(self, k, as_dataset=False):
        """
        Returns the k-th most extreme samples in the dataset according to the L1 norm.

        Args:
            X (np.ndarray): dataset of extreme samples
            labels (np.ndarray): labels of the dataset
            extreme_X (np.ndarray): less extreme sample
        """
        dataframe = self.dataframe.copy()
        dataframe = dataframe.sort_values(by="norm", ascending=False).reset_index(
            drop=True
        )

        if (
            not isinstance(k, int)
            and not isinstance(k, np.int64)
            and not isinstance(k, np.int32)
        ):
            boundary = dataframe.iloc[int(k * len(dataframe)) - 1]

            extreme_X = (
                dataframe.iloc[: int(k * len(dataframe)) - 1].iloc[:, :-2].values
            )
            extreme_labels = (
                dataframe.iloc[: int(k * len(dataframe)) - 1].iloc[:, -2].values
            )
        else:
            boundary = dataframe.iloc[k]

            # Get the k-th most extreme samples
            extreme_X = (
                dataframe[dataframe["norm"] >= boundary["norm"]].iloc[:, :-2].values
            )
            extreme_labels = (
                dataframe[dataframe["norm"] >= boundary["norm"]].iloc[:, -2].values
            )

        if as_dataset:
            dataset = BivariateLogisticDataset.from_data(extreme_X, extreme_labels)
            return dataset, boundary

        return extreme_X, extreme_labels, boundary

    def make_extreme(self, norm):
        """
        Makes the dataset more extreme by adding samples with the given norm.

        Args:
            norm (float): norm of the samples to be added
        """
        dataframe = self.dataframe.copy()

        # Get the samples with the given norm
        extreme_X = dataframe[dataframe["norm"] > norm].iloc[:, :-2].values
        extreme_labels = dataframe[dataframe["norm"] > norm].iloc[:, -2].values

        dataset = BivariateLogisticDataset.from_data(extreme_X, extreme_labels)

        return dataset

    def from_data(X, labels):
        """
        Creates a dataset from data.

        Args:
            X (np.ndarray): dataset
            labels (np.ndarray): labels of the dataset
        """
        dataset = BivariateLogisticDataset([1], [1], [1])
        dataset.X = X
        dataset.labels = labels

        dataset.make_dataframe()

        return dataset
