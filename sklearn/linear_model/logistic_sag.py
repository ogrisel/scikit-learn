"""
Learning a logistic regression using the SAG optimizer.
"""

# Author: Nicolas Le Roux <nicolas@le-roux.name>
#
# License: BSD Style.

import numpy as np
from .base import LinearClassifierMixin
from ..base import BaseEstimator
from ..utils.multiclass import unique_labels


def sigm(x):
    small_x = np.where(x < -20) # Avoid overflows.
    sigm_x = 1/(1 + np.exp(-x))
    sigm_x[small_x] = np.exp(small_x)
    return sigm_x


class LogisticSag(BaseEstimator, LinearClassifierMixin):
    """Logistic regression trained with SAG"""

    def __init__(self, n_iter, batch_size=50, l1reg=0., l2reg=0.):
        self.n_iter = n_iter # The equivalent number of passes through the data, since we are doing sampling with replacement.
        self.batch_size = batch_size # The size of the batches
        self.l1reg = l1reg # The strength of the l1 regularizer
        self.l2reg = l2reg # The strength of the l2 regularizer

    def fit(self, X, y):
        """Fit model."""

        self.classes_ = unique_labels(y)
        if len(self.classes_) > 2:
            # TODO: implement OvR reduction by default
            raise ValueError("LogisticSag only support binary classification, "
                             "got targets: %r" % list(self.classes_))

        # Initializing
        batch_size = self.batch_size
        l1reg = self.l1reg
        l2reg = self.l2reg
        n_iter = self.n_iter
        n_data, n_dims = X.shape
        stepsize = 0 # This value will actually not be used but I find it cleaner to initialize the variable outside an if statement.

        self.coef_ = np.zeros(n_dims)
        self.intercept_ = 0

        # The number of batches in the set. The last batch might be bigger.
        n_batches = np.ceil(n_data/batch_size)
        max_L = 0 # The maximum of all the maximum eigenvalues seen so far.

        all_gradients = np.zeros(n_data) # The list of all gradients with respect to the output.
        sum_gradient_weights = np.zeros(n_dims)
        sum_gradient_intercept = 0
        n_seen = 0 # Number of batches seen so far
        is_seen = np.zeros(n_batches) # Which batches have already been seen?
        n_updates = n_iter * n_batches # Total number of updates

        for update in range(n_updates):

            # Pick a minibatch at random
            batch = np.random.randint(n_batches)

            if batch < n_batches - 1:
                batch_indices = np.arange(batch_size * batch, batch_size * (batch + 1))
            else:
                # If this is the last batch, we use all the remaining points.
                batch_indices = np.arange(batch_size * batch, n_data)

            Xbatch = X[batch_indices]
            ybatch = y[batch_indices]

            # Compute the gradient with respect to the output for that batch.
            ybatch_pred = self.predict(Xbatch)
            gradient_batch_output = ybatch_pred - ybatch

            # Had we seen this batch already
            if is_seen[batch] == 0:
                # Add the gradient to the sum of gradients.
                sum_gradient_weights += np.dot(Xbatch.T, gradient_batch_output)
                sum_gradient_intercept += np.mean(gradient_batch_output)

                # This batch has now been seen.
                is_seen[batch] = 1
                n_seen += len(batch_indices)

                # Update the maximum of all top eigenvalues if needed.
                L_batch = np.mean(np.sum(Xbatch**2, axis=1))
                max_L = np.max(max_L, L_batch)
                stepsize = 1/(max_L/4 + l2reg)
            else:
                # Remove the old gradient and add in the new one.
                old_gradient_batch_output = all_gradients[batch_indices]
                diff_gradient = gradient_batch_output - old_gradient_batch_output
                sum_gradient_weights = np.dot(Xbatch.T, diff_gradient)
                sum_gradient_intercept += np.sum(diff_gradient)

            # Store the new gradients.
            all_gradients[batch_indices] = gradient_batch_output

            # Apply the l2 regularizer first.
            if l2reg > 0:
                self.coef_ *= 1 - l2reg * stepsize

            # Apply the gradient
            self.coef_ -= stepsize * sum_gradient_weights / n_seen
            self.intercept_ -= stepsize*sum_gradient_intercept / n_seen

            # Apply the l1 regularizer if needed.
            if l1reg > 0:
                update_scale = 1 - l1reg * stepsize / np.abs(self.coef_)
                self.coef_ *= np.fmax(0, update_scale)
        return self

    def predict_proba(self, X):
        """Predict using the linear model

        Parameters
        ----------
        X : numpy array of shape [n_samples, n_features]

        Returns
        -------
        C : array, shape = [n_samples]
            Returns predicted values.
        """
        return self._predict_proba_lr(X)
