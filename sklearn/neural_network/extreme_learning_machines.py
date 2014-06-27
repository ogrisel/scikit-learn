"""Extreme Learning Machines
"""

# Licence: BSD 3 clause

from abc import ABCMeta, abstractmethod

import numpy as np

from scipy.linalg import pinv2

from ..base import BaseEstimator, ClassifierMixin, RegressorMixin
from ..externals import six
from ..preprocessing import LabelBinarizer
from ..utils import atleast2d_or_csr, check_random_state, column_or_1d
from ..utils import check_random_state, atleast2d_or_csr
from ..utils.extmath import safe_sparse_dot
from ..utils.fixes import expit as logistic_sigmoid


def _identity(X):
    """Return the same input array."""
    return X


def _tanh(X):
    """Compute the hyperbolic tan function

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)

    Returns
    -------
    X_new : {array-like, sparse matrix}, shape (n_samples, n_features)
    """
    return np.tanh(X, X)


def _softmax(Z):
    """Compute the K-way softmax, (exp(Z).T / exp(Z).sum(axis=1)).T

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)

    Returns
    -------
    X_new : {array-like, sparse matrix}, shape (n_samples, n_features)
    """
    exp_Z = np.exp(Z.T - Z.max(axis=1)).T
    return (exp_Z.T / exp_Z.sum(axis=1)).T


class BaseELM(six.with_metaclass(ABCMeta, BaseEstimator)):

    """Base class for ELM classification and regression.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """
    _activation_functions = {
        'tanh': _tanh,
        'logistic': logistic_sigmoid
    }

    @abstractmethod
    def __init__(self, n_hidden, activation, random_state):
        self.n_hidden = n_hidden
        self.activation = activation
        self.random_state = random_state

    def _validate_params(self):
        """Validate input params. """
        if self.n_hidden <= 0:
            raise ValueError("n_hidden must be greater or equal zero")

        if self.activation not in self._activation_functions:
            raise ValueError("The activation %s"
                             " is not supported. " % self.activation)

    def _init_param(self):
        """Set the activation functions."""
        self._activation_func = self._activation_functions[self.activation]

    def _scaled_weight_init(self, fan_in, fan_out):
        """Scale the initial, random parameters for a specific layer."""
        if self.activation == 'tanh':
                interval = np.sqrt(6. / (fan_in + fan_out))

        elif self.activation == 'logistic':
                interval = 4. * np.sqrt(6. / (fan_in + fan_out))

        return interval

    def _init_random_weights(self):
        """Initialize weight and bias parameters."""
        rng = check_random_state(self.random_state)

        fan_in, fan_out = self._n_features, self.n_hidden
        interval = self._scaled_weight_init(fan_in, fan_out)

        self.coef_hidden_ = rng.uniform(
            -interval, interval, (fan_in, fan_out))
        self.intercept_hidden_ = rng.uniform(
            -interval, interval, (fan_out))

    def _get_hidden_activations(self, X):
        """Compute the values of the neurons in the hidden layer."""
        A = safe_sparse_dot(X, self.coef_hidden_)
        A += self.intercept_hidden_

        Z = self._activation_func(A)

        return Z

    def fit(self, X, y):
        """Fit the model to the data X and target y.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples)
             Subset of the target values.

        Returns
        -------
        self
        """
        X = atleast2d_or_csr(X)

        self._validate_params()

        n_samples, self._n_features = X.shape
        self.n_outputs_ = y.shape[1]
        self._init_param()

        self._init_random_weights()

        H = self._get_hidden_activations(X)
        self.coef_output_ = safe_sparse_dot(pinv2(H), y)

        return self

    def decision_function(self, X):
        """Fit the model to the data X and target y.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)

        Returns
        -------
        array, shape (n_samples)
        Predicted target values per element in X.
        """
        X = atleast2d_or_csr(X)

        self.hidden_activations_ = self._get_hidden_activations(X)
        output = safe_sparse_dot(self.hidden_activations_, self.coef_output_)

        return output


class ELMClassifier(BaseELM, ClassifierMixin):

    """Extreme learning machines classifier.

    The algorithm trains a single-hidden layer feedforward network by computing
    the hidden layer values using randomized parameters, then solving
    for the output weights using least-square solutions.

    This implementation works with data represented as dense and sparse numpy
    arrays of floating point values for the features.

    Parameters
    ----------
    n_hidden : int, default 100
       The number of neurons in the hidden layer.

    activation : {'logistic', 'tanh'}, default 'tanh'
        Activation function for the hidden layer.

         - 'logistic' for 1 / (1 + exp(x)).

         - 'tanh' for the hyperbolic tangent.

    random_state : int or RandomState, optional, default None
        State of or seed for random number generator.


    Attributes
    ----------
    `classes_` : array or list of array of shape = [n_classes]
        Class labels for each output.

    `n_outputs_` : int,
        Number of output neurons.

    """

    def __init__(self, n_hidden=100, activation='tanh', random_state=None):

        super(ELMClassifier, self).__init__(n_hidden, activation, random_state)

        self._lbin = LabelBinarizer(-1, 1)
        self.classes_ = None

    def fit(self, X, y):
        """Fit the model to the data X and target y.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples)

        Returns
        -------
        self
        """
        y = column_or_1d(y, warn=True)
        self.classes_ = np.unique(y)
        y = self._lbin.fit_transform(y)

        super(ELMClassifier, self).fit(X, y)

        return self

    def predict(self, X):
        """Predict using the extreme learning machines model

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)

        Returns
        -------
        array, shape (n_samples)
            Predicted target values per element in X.
        """
        X = atleast2d_or_csr(X)
        scores = self.decision_function(X)

        return self._lbin.inverse_transform(scores)

    def predict_proba(self, X):
        """Probability estimates.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)

        Returns
        -------
        array, shape (n_samples, n_outputs)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in `self.classes_`.
        """
        scores = self.decision_function(X)

        if len(self.classes_) == 2:
            scores = logistic_sigmoid(scores)
            return np.hstack([1 - scores, scores])
        else:
            return _softmax(scores)

    def predict_log_proba(self, X):
        """Return the log of probability estimates.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        T : array-like, shape (n_samples, n_outputs)
            Returns the log-probability of the sample for each class in the
            model, where classes are ordered as they are in
            `self.classes_`. Equivalent to log(predict_proba(X))
        """
        return np.log(self.predict_proba(X))


class ELMRegressor(BaseELM, RegressorMixin):

    """Extreme learning machines regressor.

    The algorithm trains a single-hidden layer feedforward network by computing
    the hidden layer values using randomized parameters, then solving
    for the output weights using least-square solutions.

    This implementation works with data represented as dense and sparse numpy
    arrays of floating point values for the features.

    Parameters
    ----------
    n_hidden : int, default 100
       The number of neurons in the hidden layer.

    activation : {'logistic', 'tanh'}, default 'tanh'
        Activation function for the hidden layer.

         - 'logistic' for 1 / (1 + exp(x)).

         - 'tanh' for the hyperbolic tangent.

    random_state : int or RandomState, optional, default None
        State of or seed for random number generator.


    Attributes
    ----------
    `classes_` : array or list of array of shape = [n_classes]
        Class labels for each output.

    `n_outputs_` : int,
        Number of output neurons.

    """

    def __init__(self, n_hidden=100, activation='tanh', random_state=None):

        super(ELMRegressor, self).__init__(n_hidden, activation, random_state)

        self.classes_ = None

    def fit(self, X, y):
        """Fit the model to the data X and target y.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples)
            Subset of the target values.

        Returns
        -------
        self
        """
        y = np.atleast_1d(y)

        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        super(ELMRegressor, self).fit(X, y)
        return self

    def predict(self, X):
        """Predict using the multi-layer perceptron model.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)

        Returns
        -------
        array, shape (n_samples)
            Predicted target values per element in X.
        """
        X = atleast2d_or_csr(X)

        return self.decision_function(X)
