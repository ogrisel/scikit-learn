"""Extreme Learning Machines
"""

# Author: Issam H. Laradji <issam.laradji@gmail.com>
# Licence: BSD 3 clause

from abc import ABCMeta, abstractmethod

import numpy as np

from ..base import BaseEstimator, ClassifierMixin, RegressorMixin
from ..externals import six
from ..preprocessing import LabelBinarizer
from ..metrics import mean_squared_error
from ..metrics.pairwise import pairwise_kernels
from ..linear_model.ridge import ridge_regression
from ..utils import gen_even_slices
from ..utils import check_array, check_X_y, check_random_state
from ..utils.extmath import safe_sparse_dot
from ..utils.fixes import expit as logistic_sigmoid
from ..utils.class_weight import get_sample_weight


def _inplace_logistic_sigmoid(X):
    """Compute the logistic function. """
    return logistic_sigmoid(X, out=X)


def _inplace_tanh(X):
    """Compute the hyperbolic tan function."""
    return np.tanh(X, out=X)


def _inplace_relu(X):
    """Compute the rectified linear unit function."""
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X


def _inplace_softmax(X):
    """Compute the K-way softmax function. """
    tmp = X - X.max(axis=1)[:, np.newaxis]
    np.exp(tmp, out=X)
    X /= X.sum(axis=1)[:, np.newaxis]

    return X


ACTIVATIONS = {'tanh': _inplace_tanh, 'logistic': _inplace_logistic_sigmoid,
               'relu': _inplace_relu}

ALGORITHMS = ['standard', 'recursion_lsqr']

KERNELS = ['random', 'linear', 'poly', 'rbf', 'sigmoid']


class BaseELM(six.with_metaclass(ABCMeta, BaseEstimator)):

    """Base class for ELM classification and regression.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    @abstractmethod
    def __init__(self, n_hidden, activation, algorithm, kernel, C, degree,
                 gamma, coef0, class_weight, weight_scale, batch_size, verbose,
                 random_state):
        self.C = C
        self.activation = activation
        self.algorithm = algorithm
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.class_weight = class_weight
        self.weight_scale = weight_scale
        self.batch_size = batch_size
        self.n_hidden = n_hidden
        self.verbose = verbose
        self.random_state = random_state

        self.classes_ = None
        self.coef_hidden_ = None
        self.coef_output_ = None

    def _init_param_validate_X_y(self, X, y):
        """Initialize parameters and validate X and y."""

        # Validate input params
        if self.n_hidden <= 0:
            raise ValueError("n_hidden must be >= 0. You entered %s." %
                             self.n_hidden)
        if self.C <= 0.0:
            raise ValueError("C must be > 0. You entered %s." % self.C)
        if self.activation not in ACTIVATIONS:
            raise ValueError("The activation %s"
                             " is not supported. " % self.activation)
        if self.algorithm not in ALGORITHMS:
            raise ValueError("The algorithm %s is not supported. Supported "
                             "algorithms are %s." % (self.algorithm,
                                                     ALGORITHMS))
        if self.kernel not in KERNELS:
            raise ValueError("The kernel %s is not supported. Supported "
                             "kernels are %s." % (self.kernel, KERNELS))
        if self.algorithm != 'standard' and self.class_weight is not None:
                raise NotImplementedError("class_weight is only supported "
                                          "when algorithm='standard'.")

        X, y = check_X_y(X, y, accept_sparse='csr', multi_output=True)
        n_samples, n_features = X.shape

        sample_weight = None

        # Classification
        if isinstance(self, ClassifierMixin):
            if self.classes_ is None:
                self.classes_ = np.unique(y)

            # Get sample weights
            if self.class_weight is not None and self.algorithm == 'standard':
                sample_weight = get_sample_weight(self.class_weight,
                                                  self.classes_, y)

            y = self.label_binarizer_.fit_transform(y)

        # Ensure y is 2D
        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        # Randomize and scale the input-to-hidden weight parameters
        if self.algorithm == 'standard' or self.coef_hidden_ is None:
            rng = check_random_state(self.random_state)

            if self.weight_scale == 'auto':
                if self.activation == 'tanh':
                    weight_scale = np.sqrt(6. / (n_features +
                                                 self.n_hidden))
                elif self.activation == 'logistic':
                    weight_scale = 4. * np.sqrt(6. / (n_features +
                                                      self.n_hidden))
                else:
                    weight_scale = np.sqrt(1. / n_features)
            else:
                weight_scale = self.weight_scale

            self.coef_hidden_ = rng.uniform(-weight_scale, weight_scale,
                                           (n_features, self.n_hidden))
            self.intercept_hidden_ = rng.uniform(-weight_scale, weight_scale,
                                                (self.n_hidden))

        if self.kernel in ['poly', 'rbf', 'sigmoid'] and self.gamma is None:
            # If custom gamma is not provided ...
            self.gamma = 1.0 / n_features

        if self.kernel != 'random':
            self._X_train = X

        return X, y, sample_weight

    def _compute_hidden_activations(self, X):
        """Compute the hidden activations using the set kernel."""
        if self.kernel == 'random':
            hidden_activations = safe_sparse_dot(X, self.coef_hidden_)
            hidden_activations += self.intercept_hidden_

            # Apply the activation method in an inline manner
            ACTIVATIONS[self.activation](hidden_activations)

        else:
            # Compute activation using a kernel
            hidden_activations = pairwise_kernels(X, self._X_train,
                                                  metric=self.kernel,
                                                  filter_params=True,
                                                  gamma=self.gamma,
                                                  degree=self.degree,
                                                  coef0=self.coef0)
        return hidden_activations

    def _fit(self, X, y, warm_start=False):
        """Fit the model to the data X and target y."""
        X, y, sample_weight = self._init_param_validate_X_y(X, y)

        if self.algorithm == 'standard':
            hidden_activations = self._compute_hidden_activations(X)

            # Compute hidden-to-output coefficients
            self.coef_output_ = ridge_regression(hidden_activations, y,
                                                 1.0 / self.C,
                                                 sample_weight=sample_weight).T
            if self.verbose:
                print("Training mean square error = %f" %
                      mean_squared_error(y, self._predict(X)))

        elif self.algorithm == 'recursion_lsqr':
            # Compute the least-square solutions in batches
            n_samples = X.shape[0]

            batch_size = np.clip(self.batch_size, 0, n_samples)
            n_batches = n_samples // batch_size
            batch_slices = list(gen_even_slices(n_batches * batch_size,
                                                n_batches))

            for batch, batch_slice in enumerate(batch_slices):

                hidden_activations = \
                    self._compute_hidden_activations(X[batch_slice])

                if batch == 0 and (warm_start is False or
                                   not hasattr(self, "_H_sub")):
                    # Initialize recursive least-square algorithm
                    self.coef_output_ = np.zeros((self.n_hidden,
                                                  self.n_outputs_))
                    # Compute hidden activation subset _H_sub
                    self._H_sub = safe_sparse_dot(hidden_activations.T,
                                                  hidden_activations)
                    # Compute y subset
                    y_sub = safe_sparse_dot(hidden_activations.T,
                                            y[batch_slice])

                else:
                    # Run recursive least-square algorithm
                    # Compute hidden activation subset _H_sub
                    self._H_sub += safe_sparse_dot(hidden_activations.T,
                                                   hidden_activations)
                    # Compute output y
                    y_pred = safe_sparse_dot(hidden_activations,
                                             self.coef_output_)
                    # Compute y subset
                    y_sub = safe_sparse_dot(hidden_activations.T,
                                           (y[batch_slice] - y_pred))

                # Update hidden-to-output coefficients
                self.coef_output_ += ridge_regression(self._H_sub, y_sub,
                                                      1.0 / self.C).T
                if self.verbose:
                    print("Batch %d," % batch),
                    print("Training mean square error = %f" %
                          mean_squared_error(y[batch_slice],
                                             self._predict(X[batch_slice])))
        return self

    def fit(self, X, y):
        """Fit the model to the data X and target y.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target values.

        Returns
        -------
        self : returns a trained elm usable for prediction.
        """
        self._fit(X, y, warm_start=False)

    def partial_fit(self, X, y):
        """Fit the model to the data X and target y.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Subset of training data.

        y : array-like, shape (n_samples,)
            Subset of target values.

        Returns
        -------
        self : returns a trained elm usable for prediction.
        """
        if self.algorithm != 'recursion_lsqr':
            raise ValueError("only 'recursion_lsqr' algorithm "
                             " supports partial fit")
        self._fit(X, y, warm_start=True)

        return self

    def _predict(self, X):
        """Predict using the trained model

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        y_pred : array-like, shape (n_samples,) or (n_samples, n_outputs)
                 The predicted values.
        """
        X = check_array(X, accept_sparse='csr')

        self.hidden_activations_ = self._compute_hidden_activations(X)
        y_pred = safe_sparse_dot(self.hidden_activations_, self.coef_output_)

        return y_pred


class ELMClassifier(BaseELM, ClassifierMixin):
    """Extreme learning machines classifier.

    The algorithm trains a single-hidden layer feedforward network by computing
    the hidden layer values using randomized parameters, then solving
    for the output weights using least-square solutions. For prediction,
    after computing the forward pass, the continuous output values pass
    through a gate function converting them to integers that represent classes.

    This implementation works with data represented as dense and sparse numpy
    arrays of floating point values for the features.

    Parameters
    ----------
    C : float, optional, default 1
        A regularization term that controls the linearity of the decision
        function. Smaller value of C makes the decision boundary more linear.

    class_weight : {dict, 'auto'}, optional
        Set the parameter C of class i to class_weight[i]*C for ELMClassifier.
        If not given, all classes are supposed to have weight one. The 'auto'
        mode uses the values of y to automatically adjust weights inversely
        proportional to class frequencies.

    weight_scale : float or 'auto', default 'auto'
        Scales the weights that initialize the outgoing weights of the first
        hidden layer. The weight values will range between plus and minus an
        interval based on the uniform distribution. That interval
        is 1. / n_features if weight_scale='auto'; otherwise,
        the interval is the value given to weight_scale.

    n_hidden: int, default 100
        The number of neurons in the hidden layer, it only applies to
        kernel='random'.

    activation : {'logistic', 'tanh', 'relu'}, default 'tanh'
        Activation function for the hidden layer. It only applies to
        kernel='random'.

         - 'logistic' returns f(x) = 1 / (1 + exp(x)).

         - 'tanh' returns f(x) = tanh(x).

         - 'relu' returns f(x) = max(0, x)

    algorithm : {'standard', 'recursion_lsqr'}, default 'standard'
        The algorithm for computing least-square solutions.
        Defaults to 'recursion_lsqr'

        - 'standard' computes the least-square solutions using the
          whole matrix at once.

        - 'recursion_lsqr' computes the least-square solutions by training
          on the dataset in batches using a recursive least-square
          algorithm.

    kernel : {'random', 'linear', 'poly', 'rbf', 'sigmoid'},
             optional, default 'random'
        Specifies the kernel type to be used in the algorithm.

    degree : int, optional, default 3
        Degree of the polynomial kernel function 'poly'.
        Ignored by all other kernels.

    gamma : float, optional, default None
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. If gamma is
        None then 1/n_features will be used instead.

    coef0 : float, optional default 0.0
        Independent term in kernel function. It only applies to
        'poly' and 'sigmoid'.

    batch_size : int, optional, default 200
        Size of minibatches for the 'recursion_lsqr' algoritm.
        Minibatches do not apply to the 'standard' ELM algorithm.

    verbose : bool, optional, default False
        Whether to print training score to stdout.

    random_state : int or RandomState, optional, default None
        State of or seed for random number generator.


    Attributes
    ----------
    `classes_` : array or list of array of shape = [n_classes]
        Class labels for each output.

    `n_outputs_` : int,
        Number of output neurons.

    References
    ----------
    Zong, Weiwei, Guang-Bin Huang, and Yiqiang Chen.
        "Weighted extreme learning machine for imbalance learning."
        Neurocomputing 101 (2013): 229-242.

    Liang, Nan-Ying, et al.
        "A fast and accurate online sequential learning algorithm for
        feedforward networks." Neural Networks, IEEE Transactions on
        17.6 (2006): 1411-1423.
        http://www.ntu.edu.sg/home/egbhuang/pdf/OS-ELM-TNN.pdf
    """
    def __init__(self, n_hidden=500, activation='tanh', algorithm='standard',
                 kernel='random', C=1, degree=3, gamma=None, coef0=0.0,
                 class_weight=None, weight_scale='auto',
                 batch_size=200, verbose=False, random_state=None):
        super(ELMClassifier, self).__init__(n_hidden=n_hidden,
                                            activation=activation,
                                            algorithm=algorithm, kernel=kernel,
                                            C=C, degree=degree, gamma=gamma,
                                            coef0=coef0,
                                            class_weight=class_weight,
                                            weight_scale=weight_scale,
                                            batch_size=batch_size,
                                            verbose=verbose,
                                            random_state=random_state)

        self.label_binarizer_ = LabelBinarizer(-1, 1)

    def partial_fit(self, X, y, classes=None):
        """Fit the model to the data X and target y.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,)
            Subset of the target values.

        classes : array-like, shape (n_classes,)
            List of all the classes that can possibly appear in the y vector.

            Must be provided at the first call to partial_fit, can be omitted
            in subsequent calls.

        Returns
        -------
        self : returns a trained elm usable for prediction.
        """
        if self.classes_ is None and classes is None:
            raise ValueError("classes must be passed on the first call "
                             "to partial_fit.")
        elif self.classes_ is not None and classes is not None:
            if np.any(self.classes_ != np.unique(classes)):
                raise ValueError("`classes` is not the same as on last call "
                                 "to partial_fit.")
        elif classes is not None:
            self.classes_ = classes

        super(ELMClassifier, self).partial_fit(X, y)

        return self

    def decision_function(self, X):
        """Decision function of the elm model

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        y : array-like, shape (n_samples,) or (n_samples, n_classes)
            The predict values.
        """
        scores = self._predict(X)

        if self.n_outputs_ == 1:
            return scores.ravel()
        else:
            return scores

    def predict(self, X):
        """Predict using the extreme learning machines model

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        y : array-like, shape (n_samples,) or (n_samples, n_classes)
            The predicted classes, or the predict values.
        """
        scores = self._predict(X)

        return self.label_binarizer_.inverse_transform(scores)

    def predict_proba(self, X):
        """Probability estimates.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        y_prob : array-like, shape (n_samples, n_classes)
                 The predicted probability of the sample for each class in the
                 model, where classes are ordered as they are in
                 `self.classes_`.
        """
        scores = self._predict(X)

        if len(self.classes_) == 2:
            scores = logistic_sigmoid(scores)
            return np.hstack([1 - scores, scores])
        else:
            return _inplace_softmax(scores)

    def predict_log_proba(self, X):
        """Return the log of probability estimates.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        y_prob : array-like, shape (n_samples, n_classes)
                 The predicted log-probability of the sample for each class
                 in the model, where classes are ordered as they are in
                 `self.classes_`. Equivalent to log(predict_proba(X))
        """
        y_prob = self.predict_proba(X)
        return np.log(y_prob, out=y_prob)


class ELMRegressor(BaseELM, RegressorMixin):
    """Extreme learning machines regressor.

    The algorithm trains a single-hidden layer feedforward network by computing
    the hidden layer values using randomized parameters, then solving
    for the output weights using least-square solutions. For prediction,
    ELMRegressor computes the forward pass resulting in contiuous output
    values.

    This implementation works with data represented as dense and sparse numpy
    arrays of floating point values for the features.

    Parameters
    ----------
    C: float, optional, default 10e5
        A regularization term that controls the linearity of the decision
        function. Smaller value of C makes the decision boundary more linear.

    weight_scale : float or 'auto', default 'auto'
        Scales the weights that initialize the outgoing weights of the first
        hidden layer. The weight values will range between plus and minus an
        interval based on the uniform distribution. That interval
        is 1. / n_features if weight_scale='auto'; otherwise,
        the interval is the value given to weight_scale.

    n_hidden: int, default 100
        The number of neurons in the hidden layer, it only applies to
        kernel='random'.

    activation : {'logistic', 'tanh', 'relu'}, default 'tanh'
        Activation function for the hidden layer. It only applies to
        kernel='random'.

         - 'logistic' returns f(x) = 1 / (1 + exp(x)).

         - 'tanh' returns f(x) = tanh(x).

         - 'relu' returns f(x) = max(0, x)

    algorithm : {'standard', 'recursion_lsqr'}, default 'standard'
        The algorithm for computing least-square solutions.
        Defaults to 'recursion_lsqr'

        - 'standard' computes the least-square solutions using the
          whole matrix at once.

        - 'recursion_lsqr' computes the least-square solutions by training
          on the dataset in batches using a recursive least-square
          algorithm.

    kernel : {'random', 'linear', 'poly', 'rbf', 'sigmoid'},
             optional, default 'random'
        Specifies the kernel type to be used in the algorithm.

    degree : int, optional, default 3
        Degree of the polynomial kernel function 'poly'.
        Ignored by all other kernels.

    gamma : float, optional, default None
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. If gamma is
        None then 1/n_features will be used instead.

    coef0 : float, optional default 0.0
        Independent term in kernel function. It only applies to
        'poly' and 'sigmoid'.

    batch_size : int, optional, default 200
        Size of minibatches for the 'recursion_lsqr' algoritm.
        Minibatches do not apply to the 'standard' ELM algorithm.

    verbose : bool, optional, default False
        Whether to print training score to stdout.

    random_state : int or RandomState, optional, default None
        State of or seed for random number generator.

    Attributes
    ----------
    `classes_` : array or list of array of shape = [n_classes]
        Class labels for each output.

    `n_outputs_` : int
        Number of output neurons.

    References
    ----------
    Zong, Weiwei, Guang-Bin Huang, and Yiqiang Chen.
        "Weighted extreme learning machine for imbalance learning."
        Neurocomputing 101 (2013): 229-242.

    Liang, Nan-Ying, et al.
        "A fast and accurate online sequential learning algorithm for
        feedforward networks." Neural Networks, IEEE Transactions on
        17.6 (2006): 1411-1423.
        http://www.ntu.edu.sg/home/egbhuang/pdf/OS-ELM-TNN.pdf
    """
    def __init__(self, n_hidden=100, activation='tanh', algorithm='standard',
                 weight_scale='auto', kernel='random', batch_size=200, C=10e5,
                 degree=3, gamma=None, coef0=0.0, verbose=False,
                 random_state=None):
        super(ELMRegressor, self).__init__(n_hidden=n_hidden,
                                           activation=activation,
                                           algorithm=algorithm, kernel=kernel,
                                           C=C, degree=degree, gamma=gamma,
                                           coef0=coef0, class_weight=None,
                                           weight_scale=weight_scale,
                                           batch_size=batch_size,
                                           verbose=verbose,
                                           random_state=random_state)

    def predict(self, X):
        """Predict using the elm model.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        y : array-like, shape (n_samples, n_outputs)
            The predicted classes, or the predict values.
        """
        return self._predict(X)
