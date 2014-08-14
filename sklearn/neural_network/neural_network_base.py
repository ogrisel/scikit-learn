"""Base utilities for the neural network modules
"""

# Author: Issam H. Laradji <issam.laradji@gmail.com>
# Licence: BSD 3 clause

import numpy as np

from ..utils.fixes import expit as logistic_sigmoid
from ..utils import check_random_state


def logistic(X):
    """Compute the logistic function."""
    return logistic_sigmoid(X, out=X)


def tanh(X):
    """Compute the hyperbolic tan function."""
    return np.tanh(X, out=X)


def relu(X):
    """Compute the rectified linear unit function."""
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X


def softmax(X):
    """Compute the K-way softmax function. """
    tmp = X - X.max(axis=1)[:, np.newaxis]
    np.exp(tmp, out=X)
    X /= X.sum(axis=1)[:, np.newaxis]

    return X


ACTIVATIONS = {'tanh': tanh, 'logistic': logistic, 'relu': relu}


def activation_functions():
    """This function simply returns the valid activation functions. """
    return ACTIVATIONS


def init_weights(weight_scale, n_features, n_outputs, random_state):
    """Initialize the parameter weights."""
    rng = check_random_state(random_state)

    coef = rng.uniform(-1, 1, (n_features, n_outputs))
    intercept = rng.uniform(-1, 1, n_outputs)

    if weight_scale != 1:
        coef *= weight_scale
        intercept *= weight_scale

    return coef, intercept
