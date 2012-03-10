"""Implementation of the Nearest Centroid Classifier"""


import numpy as np
from ..base import BaseEstimator
from ..utils.validation import check_arrays
from ..base import ClassifierMixin
from ..metrics.pairwise import pairwise_distances


class NearestCentroidClassifier(BaseEstimator, ClassifierMixin):
    """Compute mean for each class for nearest centroid classification"""

    def __init__(self, metric='euclidean'):
        self.metric = metric

    def fit(self, X, y):
        """Compute the center of each class

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training data

        y : numpy array of shape [n_samples]
            Target values
        """
        X, y = check_arrays(X, y)
        n_samples, n_features = X.shape

        # TODO: check / validate input
        self.classes_ = classes = np.unique(y)
        n_classes = classes.shape[0]
        self.components_ = np.empty((n_classes, n_features), dtype=X.dtype)
        for i in range(n_classes):
            self.components_[i, :] = X[y == classes[i]].mean(axis=0)
        return self

    def predict(self, X):
        return self.classes_[pairwise_distances(
            X, self.components_, metric=self.metric).argmin(axis=1)]
