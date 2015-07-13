import warnings
import numpy as np
from time import time
from abc import ABCMeta, abstractmethod

from ..externals import six
from ..base import BaseEstimator
from ..base import DensityMixin
from .. import cluster
from ..utils import check_random_state, check_array
from ..utils.extmath import logsumexp
from sklearn.utils import ConvergenceWarning
from sklearn.externals.six import print_


def check_shape(param, param_shape, name):
    param = np.array(param)
    if param.shape != param_shape:
        raise ValueError("The parameter '%s' should have the shape of %s, "
                         "but got %s" % (name, param_shape, param.shape))


def _check_X(X, n_components, n_features=None):
    """Check the input data X

    Parameters
    ----------
    X : array-like, (n_samples, n_features)

    n_components : int

    Returns
    -------
    X : array, (n_samples, n_features)
    """
    # remove 'ensure_2d=False' after #4511 is merged
    X = check_array(X, dtype=np.float64, ensure_2d=False)
    if X.ndim != 2:
        raise ValueError("Expected the input data X have 2 dimensions, "
                         "but got %s dimension(s)" %
                         X.ndim)
    if X.shape[0] < n_components:
        raise ValueError('Expected n_samples >= n_components'
                         'but got n_components = %d, n_samples = %d'
                         % (n_components, X.shape[0]))
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError("Expected the input data X have %d features, "
                         "but got %d features"
                         % (n_features, X.shape[1]))
    return X


def _sufficient_Sk_full(responsibilities, X, nk, xk, min_covar):
    """Compute the covariance matrices for the 'full' case

    Parameters
    ----------
    responsibilities : array-like, shape = (n_samples, n_components)

    X : array-like, shape = (n_samples, n_features)

    nk: array-like, shape = (n_components,)

    xk : array-like, shape = (n_components, n_features)

    min_covar : float

    Returns
    -------
    Sk : array, shape = (n_components, n_features, n_features)
    """
    # replace simplified equations, cov(X) = E[X^2]-E[X]^2 with
    # the definition equation since users may not estimate all of parameters
    n_features = X.shape[1]
    n_components = xk.shape[0]
    Sk = np.empty((n_components, n_features, n_features))
    for k in range(n_components):
        # remove + 10 * EPS
        diff = X - xk[k]
        Sk[k] = np.dot(responsibilities[:, k] * diff.T, diff) / nk[k]
        Sk[k].flat[::n_features+1] += min_covar
    return Sk


def _sufficient_Sk_tied(responsibilities, X, nk, xk, min_covar):
    """Compute the covariance matrices for the 'tied' case

    Parameters
    ----------
    responsibilities : array-like, shape = (n_samples, n_components)

    X : array-like, shape = (n_samples, n_features)

    nk: array-like, shape = (n_components,)

    xk : array-like, shape = (n_components, n_features)

    min_covar : float

    Returns
    -------
    Sk : array, shape = (n_components, n_features)
    """
    # TODO replace the simplified equation for GMM
    avg_X2 = np.dot(X.T, X)
    avg_means2 = np.dot(nk * xk.T, xk)
    covars = avg_X2 - avg_means2
    covars /= X.shape[0]
    covars.flat[::len(covars) + 1] += min_covar
    return covars


def _sufficient_Sk_diag(responsibilities, X, nk, xk, min_covar):
    """Compute the covariance matrices for the 'diag' case

    Parameters
    ----------
    responsibilities : array-like, shape = (n_samples, n_components)

    X : array-like, shape = (n_samples, n_features)

    nk: array-like, shape = (n_components,)

    xk : array-like, shape = (n_components, n_features)

    min_covar : float

    Returns
    -------
    Sk : array, shape = (n_components, n_features)
    """
    avg_X2 = np.dot(responsibilities.T, X * X) / nk[:, np.newaxis]
    avg_means2 = xk ** 2
    avg_X_means = xk * np.dot(responsibilities.T, X) / nk[:, np.newaxis]
    return avg_X2 - 2 * avg_X_means + avg_means2 + min_covar


def _sufficient_Sk_spherical(responsibilities, X, nk, xk, min_covar):
    """Compute the covariance matrices for the 'spherical' case

    Parameters
    ----------
    responsibilities : array-like, shape = (n_samples, n_components)

    X : array-like, shape = (n_samples, n_features)

    nk: array-like, shape = (n_components,)

    xk : array-like, shape = (n_components, n_features)

    min_covar : float

    Returns
    -------
    Sk : array, shape = (n_components,)
    """
    covars = _sufficient_Sk_diag(responsibilities, X, nk, xk, min_covar)
    return covars.mean(axis=1)


def _sufficient_Sk(responsibilities, X, nk, xk, min_covar, covariance_type):
    """Compute the covariance matrices

    Parameters
    ----------
    responsibilities : array-like, shape = (n_samples, n_components)

    X : array-like, shape = (n_samples, n_features)

    nk: array-like, shape = (n_components,)

    xk : array-like, shape = (n_components, n_features)

    min_covar : float

    Returns
    -------
    Sk : array,
        full : shape = (n_components, n_features, n_features)
        tied : shape = (n_components, n_features)
        diag : shape = (n_components, n_features)
        spherical : shape = (n_components,)
    """
    # TODO we could use some methods in sklearn.covariance
    # TODO degenerate cases
    sufficient_sk_functions = {"full": _sufficient_Sk_full,
                               "tied": _sufficient_Sk_tied,
                               "diag": _sufficient_Sk_diag,
                               "spherical": _sufficient_Sk_spherical}
    return sufficient_sk_functions[covariance_type](responsibilities, X, nk,
                                                    xk, min_covar)


def sufficient_statistics(responsibilities, X, min_covar, covariance_type):
    """Compute the sufficient statistics

    Parameters
    ----------
    responsibilities : array-like, shape = (n_samples, n_components)

    X : array-like, shape = (n_samples, n_features)

    min_covar : float

    covariance_type : string

    Returns
    -------
    nk: array-like, shape = (n_components,)

    xk : array-like, shape = (n_components, n_features)

    Sk : array,
        full : shape = (n_components, n_features, n_features)
        tied : shape = (n_components, n_features)
        diag : shape = (n_components, n_features)
        spherical : shape = (n_components,)
    """
    # compute three sufficient statistics
    nk = responsibilities.sum(axis=0)
    # remove + 10 * EPS
    xk = np.dot(responsibilities.T, X) / nk[:, np.newaxis]
    Sk = _sufficient_Sk(responsibilities, X, nk, xk, min_covar,
                        covariance_type)
    return nk, xk, Sk


def sample_gaussian(mean, covar, covariance_type='full', n_samples=1,
                    random_state=None):
    pass


class MixtureBase(six.with_metaclass(ABCMeta, DensityMixin,
                                     BaseEstimator)):
    def __init__(self, n_components, covariance_type, random_state, tol,
                 min_covar, n_iter, n_init, params, init_params, verbose):
        self.n_components = n_components
        self.n_features = None
        self.covariance_type = covariance_type
        self.tol = tol
        self.min_covar = min_covar
        self.random_state = random_state
        self.random_state_ = None
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

        if params is not None:
            # TODO deprecate 'params'
            pass

        if covariance_type not in ['spherical', 'tied', 'diag', 'full']:
                raise ValueError("Invalid value for 'covariance_type': %s "
                                 "'covariance_type' should be in "
                                 "['spherical', 'tied', 'diag', 'full']"
                                 % covariance_type)

    @abstractmethod
    def _check_initial_parameters(self):
        pass

    @abstractmethod
    def _initialize_parameters(self, X, resp, nk, xk, Sk):
        pass

    @abstractmethod
    def _e_step(self, X):
        pass

    @abstractmethod
    def _m_step(self, X, nk, xk, Sk):
        pass

    @abstractmethod
    def _estimate_weighted_log_probabilities(self, X):
        """Compute the weighted log probabilities for each sample in X with
        respect to the model.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        weighted_log_probabilities : shape = (n_samples, n_components)
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

    def _estimate_log_probabilities_responsibilities(self, X):
        """Compute the weighted log probabilities and responsibilities for
        each sample in X with respect to the model.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        log_probabilities : shape = (n_samples,)
            weighted log probabilities

        responsibilities : shape = (n_samples, n_components)
        """
        log_prob_comp = self._estimate_weighted_log_probabilities(X)
        log_prob = logsumexp(log_prob_comp, axis=1)
        with np.errstate(under='ignore'):
            # ignore underflow
            resp = np.exp(log_prob_comp - log_prob[:, np.newaxis])
        return log_prob_comp, log_prob, resp

    def score_samples(self, X):
        """Compute the weighted log probabilities for
        each sample in X with respect to the model.

        Parameters
        ----------

        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        log_probabilities : shape = (n_samples,)
            weighted log probabilities
        """
        self._check_is_fitted()
        X = _check_X(X, self.n_components, self.n_features)

        weighted_log_likelihood = self._estimate_weighted_log_probabilities(X)
        log_probabilities = logsumexp(weighted_log_likelihood, axis=1)
        return log_probabilities

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
        labels = cluster.KMeans(n_clusters=self.n_components,
                                random_state=self.random_state_).fit(X).labels_
        responsibilities = np.zeros((X.shape[0], self.n_components))
        responsibilities[range(X.shape[0]), labels] = 1
        return responsibilities

    def _initialize_sufficient_statistics(self, X):
        """Initialize the model parameters. If the initial parameters are
        not given, the model parameters are initialized by the method specified
        by `self.init_params`.

        Parameters
        ----------

        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        nk : array-like,

        xk : array-like,

        Sk : array-like
        """
        if self.verbose > 1:
            print_('\n\tInitializing parameters by ', end='')

        # use self.init_params to initialize
        if self.init_params == 'kmeans':
            if self.verbose > 1:
                print_('kmeans.', end='')
            responsibilities = self._initialize_by_kmeans(X)
            nk, xk, Sk = sufficient_statistics(responsibilities, X,
                                               self.min_covar,
                                               self.covariance_type)
        elif self.init_params == 'random':
            # other initialization methods as long as
            # they return responsibilities
            if self.verbose > 1:
                print_('random initialization.', end='')
            responsibilities = self.random_state_.rand(X.shape[0],
                                                       self.n_components)
            responsibilities = (
                responsibilities /
                responsibilities.sum(axis=1)[:, np.newaxis])
            nk, xk, Sk = sufficient_statistics(responsibilities, X,
                                               self.min_covar,
                                               self.covariance_type)
        else:
            raise ValueError("Unimplemented initialization method %s"
                             % self.init_params)
        return responsibilities, nk, xk, Sk

    def _initialize(self, X):
        resp, nk, xk, Sk = self._initialize_sufficient_statistics(X)
        self._initialize_parameters(X, resp, nk, xk, Sk)

    def fit(self, X, y=None):
        self.random_state_ = check_random_state(self.random_state)
        X = _check_X(X, self.n_components)
        self.n_features = X.shape[1]
        self._check_initial_parameters()

        max_log_likelihood = -np.infty

        if self.verbose > 0:
            print_('The estimation of %s started.' %
                   self.__class__.__name__, end='')

        for init in range(self.n_init):
            if self.verbose > 0:
                print_('Initialization %d' % (init + 1), end='')
                start_init_time = time()

            self._initialize(X)

            current_log_likelihood = -np.infty
            if self.verbose > 1:
                print_('\n\tUsed %.5fs' % (time() - start_init_time), end='')

            self.converged_ = False

            for i in range(self.n_iter):
                if self.verbose > 1:
                    start_iter_time = time()

                prev_log_likelihood = current_log_likelihood

                # e step
                current_log_likelihood, resp = self._e_step(X)

                if self.verbose > 1:
                    print_('\tLog-likelihood %.5f' % current_log_likelihood,
                           end=' ')

                change = abs(current_log_likelihood - prev_log_likelihood)
                if change < self.tol:
                    self.converged_ = True
                    break

                # m step
                nk, xk, Sk = sufficient_statistics(
                    resp, X, self.min_covar, self.covariance_type)
                self._m_step(resp, nk, xk, Sk)

                if self.verbose > 0:
                    print_('\n\tIteration %d' % (i + 1), end='')
                if self.verbose > 1:
                    print_('\tused %.5fs' % (time() - start_iter_time),
                           end=' ')
            if not self.converged_:
                # compute the log-likelihood of the last m-step
                warnings.warn('Initialization %d is not converged. '
                              'Try different init parameters, '
                              'or increase n_init, '
                              'or check for degenerate data.'
                              % (init + 1), ConvergenceWarning)
                current_log_likelihood, _ = self._e_step(X)
                if self.verbose > 1:
                    print_('\tLog-likelihood/lower bound %.5f' %
                           current_log_likelihood,
                           end='')
            else:
                if self.verbose > 0:
                    print_('\n\tInitialization %d is converged.' % (init + 1),
                           end='')

            if current_log_likelihood > max_log_likelihood:
                # max_log_likelihood is always updated,
                # since we compute the log-likelihood of the initialization
                max_log_likelihood = current_log_likelihood
                best_params = self._get_parameters()
                if self.verbose > 1:
                    print_('\n\tBetter parameters are found.', end='')

            if self.verbose > 1:
                print_('\n\tInitialization %s used %.5fs' %
                       (init + 1, time() - start_init_time), end='')
            if self.verbose > 0:
                print_('\n')

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
        X = _check_X(X, self.n_components, self.n_features)
        return self._estimate_weighted_log_probabilities(X).argmax(axis=1)

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
        X = _check_X(X, self.n_components, self.n_features)
        _, _, resp = self._estimate_log_probabilities_responsibilities(X)
        return resp

    def sample(self):
        pass

    def aic(self, X):
        pass

    def bic(self, X):
        pass