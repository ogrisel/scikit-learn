import warnings
import numpy as np
from scipy import linalg
from time import time
from abc import ABCMeta, abstractmethod

from ..externals import six
from ..base import BaseEstimator
from ..base import DensityMixin
from .. import cluster
from ..utils import check_random_state, check_array
from ..utils.validation import check_is_fitted
from ..utils.extmath import logsumexp
from sklearn.externals.six.moves import zip
from sklearn.externals.six import print_

EPS = np.finfo(np.float64).eps * 10


def _check_weights(n_components, n_features, weights):
    if weights is not None:
        # check value
        weights = check_array(weights, ensure_2d=False)

        # check shape
        if weights.shape != (n_components,):
            raise ValueError("'weights' must have shape (n_components, )")

        # check range
        if (any(np.less(weights, 0)) or
                any(np.greater(weights, 1))):
            raise ValueError("'weights' must be in the range [0, 1]")

        # check normalization
        if not np.allclose(np.abs(1 - np.sum(weights)), 0.0):
            raise ValueError("'weights' must be normalized")


def _check_means(n_components, n_features, means):
    if means is not None:
        # check value
        means = check_array(means)

        # check shape
        if means.shape != (n_components, n_features):
            raise ValueError("'means' must have shape (%s, %d)"
                             % (n_components, n_features))


def _check_covars_full(n_components, n_features, covars):
    # the shape of covars must be k x d x d
    if covars is not None:
        if (covars.ndim != 3 or covars.shape[0] != n_components or
                covars.shape[1] != n_features or
                covars.shape[2] != n_features):
            raise ValueError("'full' covariances must have shape "
                             "(n_components, n_features, n_features)")
        for k, cov in enumerate(covars):
            if (not np.allclose(cov, cov.T) or
                    np.any(np.less_equal(linalg.eigvalsh(cov), 0.0))):
                raise ValueError("The component %d of 'full' covars must be "
                                 "symmetric, positive-definite" % k)


def _check_covars_tied(n_components, n_features, covars):
    # the shape of covars must be d x d
    if covars is not None:
        if (covars.ndim != 2 or
                covars.shape[0] != n_features or
                covars.shape[1] != n_features):
            raise ValueError("'tied' covariances must have shape "
                             "(n_features, n_features)")
        if (not np.allclose(covars, covars.T) or
                np.any(np.less_equal(linalg.eigvalsh(covars), 0.0))):
            raise ValueError("'tied' covariance must be "
                             "symmetric, positive-definite")


def _check_covars_diag(n_components, n_features, covars):
    # the shape of covars must be k x d
    if covars is not None:
        if (covars.ndim != 2 or
                covars.shape[0] != n_components or
                covars.shape[1] != n_features):
            raise ValueError("'diag' covariances must have shape "
                             "(n_components, n_features)")
        if np.any(np.less_equal(covars, 0.0)):
            raise ValueError("'diag' covariance must be positive")


def _check_covars_spherical(n_components, n_features, covars):
    # the shape of covars must be (k, )
    if covars is not None:
        if covars.ndim != 1 or covars.shape[0] != n_components:
            raise ValueError("'spherical' covariances must have shape "
                             "(n_components, )")
        if np.any(np.less_equal(covars, 0.0)):
            raise ValueError("'spherical' covariance must be positive")


def _check_covars(n_components, n_features, covars, covariance_type):
    check_covars_functions = {"full": _check_covars_full,
                              "tied": _check_covars_tied,
                              "diag": _check_covars_tied,
                              "spherical": _check_covars_spherical}
    check_covars_functions[covariance_type](n_components, n_features, covars)


def _sufficient_Sk_full(responsibilities, X, xk, min_covar):
    # maybe we could use some methods in sklearn.covariance
    n_features = X.shape[1]
    n_components = xk.shape[0]
    covars = np.empty((n_components, n_features, n_features))
    for k in range(n_components):
        post = responsibilities[:, k]
        diff = X - xk[k]
        with np.errstate(under='ignore'):
            # remove + 10 * EPS
            covars[k] = np.dot(post * diff.T, diff) / post.sum()
            covars[k].flat[::n_features+1] += min_covar
    return covars


def _sufficient_Sk_diag(responsibilities, X, xk, min_covar):
    pass


def _sufficient_Sk_spherical(responsibilities, X, xk, min_covar):
    pass


def _sufficient_Sk_tied(responsibilities, X, xk, min_covar):
    pass


def _sufficient_Sk(responsibilities, X, xk, min_covar, covariance_type):
    sufficient_sk_functions = {"full": _sufficient_Sk_full,
                               "tied": _sufficient_Sk_tied,
                               "diag": _sufficient_Sk_diag,
                               "spherical": _sufficient_Sk_spherical}
    return sufficient_sk_functions[covariance_type](responsibilities, X, xk,
                                                    min_covar)


def _sufficient_statistics(responsibilities, X, min_covar, covariance_type):
    # compute three sufficient statistics
    nk = responsibilities.sum(axis=0)
    # remove + 10 * EPS
    xk = np.dot(responsibilities.T, X) / nk[:, np.newaxis]
    Sk = _sufficient_Sk(responsibilities, X, xk, min_covar, covariance_type)
    return nk, xk, Sk


def sample_gaussian(mean, covar, covariance_type='full', n_samples=1,
                    random_state=None):
    pass


def sample_gamma():
    pass


def sample_inv_wishart():
    pass


class _MixtureBase(six.with_metaclass(ABCMeta, DensityMixin,
                                      BaseEstimator)):
    def __init__(self, n_components=1, random_state=None, tol=1e-5,
                 min_covar=0, covariance_type='full',
                 n_init=1, n_iter=100, params='wmc', init_params='kmeans',
                 weights=None, means=None, covars=None,
                 verbose=0):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.min_covar = min_covar
        self.random_state = random_state
        self.n_iter = n_iter
        self.n_init = n_init
        self.params = params
        self.init_params = init_params
        self.verbose = verbose
        self.init_weights_ = weights
        self.weights_ = None
        self.init_means_ = means
        self.means_ = None
        self.init_covars_ = covars
        self.covars_ = None
        self.converged_ = False

        if covariance_type not in ['spherical', 'tied', 'diag', 'full']:
            raise ValueError('Invalid value for covariance_type: %s' %
                             covariance_type)
        if n_init < 1:
            raise ValueError('%s estimation requires at least one run' %
                             self.__class__.__name__)

        if params != 'wmc':
            warnings.warn("All of the weights, the means and the covariances "
                          "should be estimated, but got params = %s" % params)

        if covariance_type not in ['spherical', 'tied', 'diag', 'full']:
                raise ValueError('Invalid value for covariance_type: %s' %
                                 covariance_type)

    def _check(self, X):
        # check the input data X
        X = check_array(X, dtype=np.float64)
        if X.shape[0] < self.n_components:
            raise ValueError(
                '%s estimation with %s components, but got only %s samples.'
                % (self.__class__.__name__, self.n_components, X.shape[0]))

        # check weights
        _check_weights(self.n_components, X.shape[1], self.init_weights_)

        # check means
        _check_means(self.n_components, X.shape[1], self.init_means_)

        # check covars
        _check_covars(self.n_components, X.shape[1], self.init_covars_,
                      self.covariance_type)

    # m-step functions
    @abstractmethod
    def _estimate_weights(self, X, nk, xk, Sk):
        pass

    @abstractmethod
    def _estimate_means(self, X, nk, xk, Sk):
        pass

    @abstractmethod
    def _estimate_covariances_full(self, X, nk, xk, Sk):
        pass

    @abstractmethod
    def _estimate_covariances_tied(self, X, nk, xk, Sk):
        pass

    @abstractmethod
    def _estimate_covariances_diag(self, X, nk, xk, Sk):
        pass

    @abstractmethod
    def _estimate_covariances_spherical(self, X, nk, xk, Sk):
        pass

    def _estimate_covariances(self, responsibilities, nk, xk, Sk):
        estimate_covariances_functions = {
            "full": self._estimate_covariances_full,
            "tied": self._estimate_covariances_tied,
            "diag": self._estimate_covariances_diag,
            "spherical": self._estimate_covariances_spherical
        }
        return estimate_covariances_functions[self.covariance_type](
            responsibilities, nk, xk, Sk)

    def _m_step(self, responsibilities, nk, xk, Sk):
        self.weights_ = self._estimate_weights(responsibilities, nk, xk, Sk)
        self.means_ = self._estimate_means(responsibilities, nk, xk, Sk)
        self.covars_ = self._estimate_covariances(responsibilities, nk, xk, Sk)

    # e-step functions
    @abstractmethod
    def _estimate_log_weights(self):
        pass

    @abstractmethod
    def _estimate_log_likelihood_full(self, X):
        pass

    @abstractmethod
    def _estimate_log_likelihood_tied(self, X):
        pass

    @abstractmethod
    def _estimate_log_likelihood_diag(self, X):
        pass

    @abstractmethod
    def _estimate_log_likelihood_spherical(self, X):
        pass

    def _estimate_log_likelihood(self, X):
        estimate_log_likelihood_functions = {
            "full": self._estimate_log_likelihood_full,
            "tied": self._estimate_log_likelihood_tied,
            "diag": self._estimate_log_likelihood_diag,
            "spherical": self._estimate_log_likelihood_spherical
        }
        weighted_log_likelihood = (self._estimate_log_weights() +
                                   estimate_log_likelihood_functions
                                   [self.covariance_type](X))
        return weighted_log_likelihood

    def _estimate_responsibilities(self, X):
        weighted_log_likelihood = self._estimate_log_likelihood(X)
        log_probabilities = logsumexp(weighted_log_likelihood, axis=1)
        responsibilities = np.exp(weighted_log_likelihood -
                                  log_probabilities[:, np.newaxis])
        return responsibilities

    def score_samples(self, X):
        weighted_log_likelihood = self._estimate_log_likelihood(X)
        log_probabilities = logsumexp(weighted_log_likelihood, axis=1)
        return log_probabilities

    def _initialize_by_kmeans(self, X):
        labels = cluster.KMeans(n_clusters=self.n_components,
                                random_state=self.random_state).fit(X).labels_
        responsibilities = np.zeros((X.shape[0], self.n_components))
        responsibilities[range(X.shape[0]), labels] = 1
        return responsibilities

    def _initialize(self, X):
        if (self.init_weights_ is None or self.init_means_ is None or
           self.init_covars_ is None):
            if self.verbose > 1:
                print('\tInitializing parameters ... ')

            # use self.init_params to initialize
            if self.init_params == 'kmeans':
                responsibilities = self._initialize_by_kmeans(X)
            else:
                # other initialization methods as long as
                # they return responsibilities
                responsibilities = np.random.rand(X.shape[0],
                                                  self.n_components)
                responsibilities = (responsibilities /
                                    responsibilities.sum(axis=1)
                                    [:, np.newaxis])

            nk, xk, Sk = _sufficient_statistics(responsibilities, X,
                                                self.min_covar,
                                                self.covariance_type)
            if self.init_weights_ is None:
                self.weights_ = self._estimate_weights(responsibilities,
                                                       nk, xk, Sk)
                if self.verbose > 1:
                    print('\tWeights are initialized.')
            else:
                self.weights_ = self.init_weights_
                if self.verbose > 1:
                    print('\tWeights are provided.')

            if self.init_means_ is None:
                self.means_ = self._estimate_means(responsibilities,
                                                   nk, xk, Sk)
                if self.verbose > 1:
                    print('\tMeans are initialized.')
            else:
                self.means_ = self.init_means_
                if self.verbose > 1:
                    print('\tMeans are provided.')

            if self.init_covars_ is None:
                self.covars_ = self._estimate_covariances(responsibilities,
                                                          nk, xk, Sk)
                if self.verbose > 1:
                    print('\tCovariances are initialized.')
            else:
                self.covars_ = self.init_covars_
                if self.verbose > 1:
                    print('\tCovariances are provided.')

    def fit(self, X, y=None):
        # check the parameters
        self._check(X)
        max_log_likelihood = -np.infty

        if self.verbose > 0:
            print(('The estimation of %s is started.' %
                  self.__class__.__name__))

        for init in range(self.n_init):
            if self.verbose > 0:
                print('Initialization ' + str(init + 1))
                start_init_time = time()

            self._initialize(X)
            current_log_likelihood = self.score(X)
            if self.verbose > 1:
                print('\tInitial log-likelihood %s\tUsed %.5fs' %
                      (current_log_likelihood, time() - start_init_time))

            self.converged_ = False

            for i in range(self.n_iter):
                if self.verbose > 0:
                    print_('\tIteration %s ' % str(i + 1), end=' ')
                    start_iter_time = time()

                # e step
                responsibilities = self._estimate_responsibilities(X)

                # m step
                nk, xk, Sk = _sufficient_statistics(responsibilities, X,
                                                    self.min_covar,
                                                    self.covariance_type)
                self._m_step(responsibilities, nk, xk, Sk)

                # log_likelihood
                prev_log_likelihood = current_log_likelihood
                current_log_likelihood = self.score(X)

                if self.verbose > 1:
                    print_('Log-likelihood %s\tUsed %.5fs' %
                           (current_log_likelihood, time() - start_iter_time),
                           end=' ')
                if self.verbose > 0:
                    print

                change = abs(current_log_likelihood - prev_log_likelihood)
                if change < self.tol:
                    self.converged_ = True
                    if self.verbose > 0:
                            print('\tEstimation converged.')
                    break

            if self.verbose > 1:
                print_('\tInitialization %s used %.5fs ' %
                       (str(init + 1), time() - start_init_time), end=' ')

            if self.n_iter and current_log_likelihood > max_log_likelihood:
                max_log_likelihood = current_log_likelihood
                best_params = {'weights': self.weights_,
                               'means': self.means_,
                               'covars': self.covars_}
                if self.verbose > 1:
                    print('\tBetter parameters are found.')
                if self.verbose > 0:
                    print

        if np.isneginf(max_log_likelihood) and self.n_iter:
            raise RuntimeError(
                "%s was never able to compute a valid likelihood "
                "given initial parameters. Try different init parameters "
                "(or increasing n_init) or check for degenerate data."
                % self.__class__.__name__)

        if self.n_iter:
            self.covars_ = best_params['covars']
            self.means_ = best_params['means']
            self.weights_ = best_params['weights']

        return self

    def predict(self, X):
        check_is_fitted(self, 'weights_')
        check_is_fitted(self, 'means_')
        check_is_fitted(self, 'covars_')
        return self._estimate_log_likelihood(X).argmax(axis=1)

    def fit_predict(self, X, y=None):
        self.fit(X)
        self.predict(X)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, 'weights_')
        check_is_fitted(self, 'means_')
        check_is_fitted(self, 'covars_')
        return self._estimate_responsibilities(X)

    def sample(self):
        pass

    def aic(self, X):
        pass

    def bic(self, X):
        pass


class GaussianMixture(_MixtureBase):
    def _estimate_log_weights(self):
        return np.log(self.weights_)

    def _estimate_log_likelihood_full(self, X):
        n_samples, n_features = X.shape
        log_prob = np.empty((n_samples, self.n_components))
        for k, (mu, cov) in enumerate(zip(self.means_,  self.covars_)):
            try:
                cov_chol = linalg.cholesky(cov, lower=True)
            except:
                raise ValueError("'covars' must be symmetric, "
                                 "positive-definite")
            cv_log_det = 2 * np.sum(np.log(np.diagonal(cov_chol)))
            cv_sol = linalg.solve_triangular(cov_chol, (X - mu).T,
                                             lower=True).T
            log_prob[:, k] = - .5 * (np.sum(cv_sol ** 2, axis=1) +
                                     n_features * np.log(2 * np.pi) +
                                     cv_log_det)
        return log_prob

    def _estimate_log_likelihood_tied(self, X):
        pass

    def _estimate_log_likelihood_diag(self, X):
        pass

    def _estimate_log_likelihood_spherical(self, X):
        pass

    def _estimate_weights(self, X, nk, xk, Sk):
        return nk / X.shape[0]

    def _estimate_means(self, X, nk, xk, Sk):
        return xk

    def _estimate_covariances_full(self, X, nk, xk, Sk):
        return Sk

    def _estimate_covariances_tied(self, X, nk, xk, Sk):
        pass

    def _estimate_covariances_diag(self, X, nk, xk, Sk):
        pass

    def _estimate_covariances_spherical(self, X, nk, xk, Sk):
        pass


class BayesianGaussianMixture(_MixtureBase):
    def __init__(self, n_components=1, random_state=None, tol=1e-5,
                 min_covar=0, covariance_type='full',
                 n_init=1, n_iter=100, params='wmc', init_params='kmeans',
                 weights=None, means=None, covars=None,
                 alpha_0=None, m_0=None, beta_0=None, nu_0=None, W_0=None,
                 verbose=0):

        super(BayesianGaussianMixture, self).__init__(
            n_components, random_state, tol,
            min_covar, covariance_type, n_init, n_iter, params, init_params,
            weights, means, covars, verbose)
        self.alpha_0 = alpha_0
        self.m_0 = m_0
        self.beta_0 = beta_0
        self.nu_0 = nu_0
        self.W_0 = W_0

    def _estimate_weights(self, X, nk, xk, Sk):
        pass

    def _estimate_means(self, X, nk, xk, Sk):
        pass

    def _estimate_covariances_full(self, X, nk, xk, Sk):
        pass

    def _estimate_covariances_tied(self, X, nk, xk, Sk):
        pass

    def _estimate_covariances_diag(self, X, nk, xk, Sk):
        pass

    def _estimate_covariances_spherical(self, X, nk, xk, Sk):
        pass

    def _estimate_log_likelihood_full(self, X):
        pass

    def _estimate_log_likelihood_tied(self, X):
        pass

    def _estimate_log_likelihood_diag(self, X):
        pass

    def _estimate_log_likelihood_spherical(self, X):
        pass

    def _estimate_log_weights(self):
        pass


class DirichletProcessGaussianMixture(BayesianGaussianMixture):

    def _estimate_weights(self, X, nk, xk, Sk):
        pass

    def _estimate_log_weights(self):
        pass
