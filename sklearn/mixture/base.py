import warnings
import numpy as np
from abc import ABCMeta, abstractmethod

from ..externals import six
from ..base import BaseEstimator
from ..base import DensityMixin
from .. import cluster
from ..utils import check_random_state, check_array
from ..utils.extmath import logsumexp
from sklearn.utils import ConvergenceWarning


def check_shape(param, param_shape, name):
    """Check the shape of the parameter"""
    param = np.array(param)
    if param.shape != param_shape:
        raise ValueError("The parameter '%s' should have the shape of %s, "
                         "but got %s" % (name, param_shape, param.shape))


def _check_X(X, n_components=None, n_features=None):
    """Validate the input data X

    Raise informative messages otherwise.

    Parameters
    ----------
    X : array-like, (n_samples, n_features)

    n_components : int

    Returns
    -------
    X : array, (n_samples, n_features)
    """
    X = check_array(X, dtype=[np.float64, np.float32], ensure_2d=False)
    if X.ndim != 2:
        raise ValueError("Expected the input data X have 2 dimensions, "
                         "but got %s dimension(s)" %
                         X.ndim)
    if n_components is not None and X.shape[0] < n_components:
        raise ValueError('Expected n_samples >= n_components'
                         'but got n_components = %d, n_samples = %d'
                         % (n_components, X.shape[0]))
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError("Expected the input data X have %d features, "
                         "but got %d features"
                         % (n_features, X.shape[1]))
    return X


def _check_weights(weights, desired_shape):
    """Check the user provided 'weights'

    Weights are the average responsibities for each component of the
    mixture.

    Parameters
    ----------
    weights : array-like, (n_components,)

    n_components : int

    Returns
    -------
    weights : array, (n_components,)
    """
    # check value
    weights = check_array(weights, dtype=[np.float64, np.float32],
                          ensure_2d=False)

    # check shape
    check_shape(weights, desired_shape, 'weights')

    # check range
    if (any(np.less(weights, 0)) or
            any(np.greater(weights, 1))):
        raise ValueError("The parameter 'weights' should be in the range "
                         "[0, 1], but got max value %.5f, min value %.5f"
                         % (np.min(weights), np.max(weights)))

    # check normalization
    if not np.allclose(np.abs(1 - np.sum(weights)), 0.0):
        raise ValueError("The parameter 'weights' should be normalized, "
                         "but got sum(weights) = %.5f" % np.sum(weights))
    return weights

# class cached_property(object):
#     def __init__(self, fget=None, doc=None):
#         self.fget = fget
#         if doc is None and fget is not None:
#             doc = fget.__doc__
#         self.__doc__ = doc
#
#     def __get__(self, obj, type=None):
#         if obj is None:
#             return self
#         value = obj.__dict__[self.fget.__name__] = self.fget(obj)
#         return value


class MixtureBase(six.with_metaclass(ABCMeta, DensityMixin, BaseEstimator)):
    def __init__(self, n_components, covariance_type, random_state, tol,
                 reg_covar, n_iter, n_init, params, init_params, verbose):
        self.n_components = n_components
        self.n_features = None
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.random_state_ = check_random_state(random_state)
        self.n_iter = n_iter
        self.n_init = n_init
        self.params = params
        self.init_params = init_params
        self.verbose = verbose
        self.converged_ = False

        if n_init < 1:
            raise ValueError("Invalid value for 'n_init': %d "
                             "Estimation requires at least one run" % n_init)

        if n_iter < 1:
            # TODO may have to adjust the iteration loop for old GMM
            raise ValueError("Invalid value for 'n_iter': %d "
                             "Estimation requires at least one iteration"
                             % n_iter)

        if reg_covar < 0:
            raise ValueError("Invalid value for 'reg_covar': %.5f "
                             "regularization on covariance must be "
                             "non-negative"
                             % reg_covar)

        if params is not None:
            # TODO deprecate 'params'
            pass

    # e-step functions
    @abstractmethod
    def _estimate_log_weights(self):
        """Estimate ln (w) in EM algorithm, estimate E[ ln(pi) ] in VB

        Returns
        -------
        log_weight : array-like, shape = (n_components, )
        """
        pass

    @abstractmethod
    def _estimate_log_prob(self, X):
        """Estimate log probabilities ln P(X | Z) for each sample in X with
        respect to the model.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        log_prob : array-like, shape = (n_samples, n_component)
        """
        pass

    def _estimate_log_prob_resp(self, X):
        """Weighted log probabilities and responsibilities

        Compute the weighted log probabilities and responsibilities for
        each sample in X with respect to current state of the model.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        log_prob_norm : array-like, shape = (n_samples,)
            ln p(X)

        log_prob : array-like, shape = (n_samples, n_components)
            ln (w) + ln p(X|Z)

        responsibilities : shape = (n_samples, n_components)
        """
        log_prob = self._estimate_log_weights() + self._estimate_log_prob(X)
        log_prob_norm = logsumexp(log_prob, axis=1)
        with np.errstate(under='ignore'):
            # ignore underflow
            resp = np.exp(log_prob - log_prob_norm[:, np.newaxis])
        resp += 10 * np.finfo(X.dtype).eps
        return log_prob_norm, log_prob, resp

    @abstractmethod
    def _e_step(self, X):
        """E step

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        log-likelihood : scalar

        responsibility : array-like, shape = (n_samples, n_components)
        """
        pass

    def _initialize_by_kmeans(self, X):
        """Compute the responsibilities for each sample in X using
        kmeans clustering

        Parameters
        ----------

        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        responsibilities : shape = (n_samples, n_components)
        """
        n_samples = X.shape[0]
        labels = cluster.KMeans(n_clusters=self.n_components,
                                random_state=self.random_state_).fit(X).labels_
        resp = np.zeros((n_samples, self.n_components))
        resp[range(n_samples), labels] = 1
        return resp

    def _initialize_resp(self, X):
        """Initialize the model parameters. If the initial parameters are
        not given, the model parameters are initialized by the method specified
        by `self.init_params`.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        resp : array-like, shape = (n_samples, n_features)
        """
        # use self.init_params to initialize
        if self.init_params == 'kmeans':
            resp = self._initialize_by_kmeans(X)
        elif self.init_params == 'random':
            resp = self.random_state_.rand(X.shape[0], self.n_components)
            resp = resp / resp.sum(axis=1)[:, np.newaxis]
        else:
            raise ValueError("Unimplemented initialization method %s"
                             % self.init_params)
        return resp

    def score_samples(self, X):
        """Compute the weighted log probabilities for each sample in X
        with respect to the model.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        log_prob : shape = (n_samples,)
            log probabilities
        """
        self._check_is_fitted()
        X = _check_X(X, None, self.n_features)

        log_prob = self._estimate_log_prob(X)
        return logsumexp(log_prob, axis=1)

    def _initialize(self, X):
        resp = self._initialize_resp(X)
        self._initialize_parameters(X, resp)

    def _fit(self, X, y=None):
        X = _check_X(X, self.n_components)
        self.n_features = X.shape[1]
        self._check_initial_parameters()

        max_log_likelihood = -np.infty
        best_params = self._get_parameters()

        #######################
        # for debug
        self._log_snapshot = []
        #######################

        for init in range(self.n_init):
            self._initialize(X)
            current_log_likelihood = -np.infty
            self.converged_ = False

            for i in range(self.n_iter):
                prev_log_likelihood = current_log_likelihood

                # e step
                current_log_likelihood, resp = self._e_step(X)

                #######################
                self._snapshot(X)
                #######################

                change = abs(current_log_likelihood - prev_log_likelihood)
                if change < self.tol:
                    self.converged_ = True
                    break

                # m step
                self._m_step(X, resp)

            if not self.converged_:
                # compute the log-likelihood of the last m-step
                warnings.warn('Initialization %d is not converged. '
                              'Try different init parameters, '
                              'or increase n_init, '
                              'or check for degenerate data.'
                              % (init + 1), ConvergenceWarning)
                current_log_likelihood, _ = self._e_step(X)

            if current_log_likelihood > max_log_likelihood:
                # max_log_likelihood is always updated,
                # since we compute the log-likelihood of the initialization
                max_log_likelihood = current_log_likelihood
                best_params = self._get_parameters()

        self._set_parameters(best_params)
        return self

    def predict(self, X):
        """Predict the labels for the data samples in X using trained model

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        C : array, shape = (n_samples,) component labels
        """
        self._check_is_fitted()
        X = _check_X(X, None, self.n_features)
        return self._estimate_log_prob(X).argmax(axis=1)

    def predict_proba(self, X):
        """Predict posterior probability of data under each Gaussian component
        in the model.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        responsibilities : array-like, shape = (n_samples, n_components)
            Returns the probability of the sample for each Gaussian component
            in the model.
        """
        self._check_is_fitted()
        X = _check_X(X, None, self.n_features)
        _, _, resp = self._estimate_log_prob_resp(X)
        return resp

    @abstractmethod
    def _check_initial_parameters(self):
        pass

    @abstractmethod
    def _initialize_parameters(self, X, resp):
        pass

    @abstractmethod
    def _estimate_suffstat(self, X, resp):
        """Estimate the sufficient statistics
        """
        pass

    @abstractmethod
    def _m_step(self, X, resp):
        """Estimate the parameters
        """
        pass

    @abstractmethod
    def _check_is_fitted(self):
        pass

    @abstractmethod
    def _get_parameters(self):
        pass

    @abstractmethod
    def _set_parameters(self, params):
        pass

    def sample(self):
        pass

    def aic(self, X):
        pass

    def bic(self, X):
        pass

    def _snapshot(self, X):
        pass
