
import warnings
import numpy as np
from scipy import linalg
from abc import ABCMeta, abstractmethod
from ..externals import six
from ..base import BaseEstimator
from ..base import DensityMixin
from .. import cluster
from ..utils import check_random_state, check_array
from sklearn.externals.six.moves import zip

EPS = np.finfo(np.float64).eps*10

def _sufficient_Sk_full(responsibilities, X, xk, min_covar):
    # maybe we could use some methods in sklearn.covariance
    n_features = X.shape[1]
    n_components = xk.shape[0]
    covars = np.empty((n_components, n_features, n_features))
    for k in range(n_components):
        post = responsibilities[:, k]
        diff = X - xk[k]
        with np.errstate(under='ignore'):
            covars[k] = np.dot(post * diff.T, diff) / (post.sum())
            covars[k].flat[::n_features+1] += min_covar
    return covars


def _sufficient_Sk_diag(responsibilities, X, xk, min_covar):
    pass

def _sufficient_Sk_spherical(responsibilities, X, xk, min_covar):
    pass

def _sufficient_Sk_tied(responsibilities, X, xk, min_covar):
    pass

def _sufficient_Sk(responsibilities, X, xk, min_covar, covariance_type):
    sufficient_sk_functions = {"full":_sufficient_Sk_full,
                               "diag": _sufficient_Sk_diag,
                               "spherical":_sufficient_Sk_spherical,
                               "tied":_sufficient_Sk_tied}
    return sufficient_sk_functions[covariance_type](responsibilities, X, xk, min_covar)

class _MixtureModelBase(six.with_metaclass(ABCMeta, DensityMixin, BaseEstimator)):

    def __init__(self, n_components=1, random_state=None, tol=1e-5,
                 min_covar=0, covariance_type='full',
                 n_init=1, n_iter=100, init_params='kmeans',
                 weights=None, means=None, covars=None,
                 verbose=0):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.min_covar = min_covar
        self.random_state = random_state
        self.n_iter = n_iter
        self.n_init = n_init
        self.init_params = init_params
        self.verbose = verbose
        self.weights_ = weights
        self.means_ = means
        self.covars_ = covars
        self.converged_ = False

    def _check_weights(self, weights):
        if weights is not None:
            # check value
            weights = check_array(weights, ensure_2d=False)

            # check shape
            if weights.shape != (self.n_components,):
                raise ValueError(
                    "The GaussianMixtureModel parameter 'weight' must "
                    "have shape (%s, )." % self.n_components)

            # check range
            if (any(np.less(weights, 0)) or
                    any(np.greater(weights, 1))):
                raise ValueError(
                    "The GaussianMixtureModel parameter 'weight' must "
                    "be in the range [0, 1].")

            # check normalization
            if not np.less_equal(np.abs(1 - np.sum(weights)), EPS):
                raise ValueError(
                    "The GaussianMixtureModel parameter 'weight' must "
                    "be normalized.")

    def _check_means(self, means):
        if self.means_ is not None:
            # check value
            # check shape
            pass

    def _check_covars(self, covars, covars_type):
        pass

    def _check(self, X):
        # check the input data X
        X = check_array(X, dtype=np.float64)
        if X.shape[0] < self.n_components:
            raise ValueError(
                'GaussianMixtureModel estimation with %s components, '
                'but got only %s samples.' % (self.n_components, X.shape[0]))

        # check weights
        self._check_weights(self.weights_)

        # check means
        self._check_means(self.means_)

        # check covars
        self._check_covars(self.covars_, self.covariance_type)

    def _initialize_by_kmeans(self, X):
        labels = cluster.KMeans(n_clusters=self.n_components,
                                random_state=self.random_state).fit(X).labels_
        responsibilities = np.zeros((X.shape[0], self.n_components))
        responsibilities[range(X.shape[0]), labels] = 1
        return responsibilities

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
        return estimate_covariances_functions[self.covariance_type](responsibilities, nk, xk, Sk)

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

    def _estimate_responsibilities(self, X):
        estimate_log_likelihood_functions = {
            "full": self._estimate_log_likelihood_full,
            "tied": self._estimate_log_likelihood_tied,
            "diag": self._estimate_log_likelihood_diag,
            "spheircal": self._estimate_log_likelihood_spherical
        }
        weighted_log_likelihood = self._estimate_log_weights() \
                                 + estimate_log_likelihood_functions[self.covariance_type](X)
        return weighted_log_likelihood / weighted_log_likelihood.sum(axis=1)[:, np.newaxis]

    @abstractmethod
    def _estimate_log_weights(self):
        pass

    @abstractmethod
    def _estimate_log_likelihood(self, X):
        pass

    def _e_step(self, X):
        self._estimate_responsibilities(X)

    def _m_step(self, responsibilities, nk, xk, Sk):
        self.weights_ = self._estimate_weights(responsibilities, nk, xk, Sk)
        self.means_ = self._estimate_means(responsibilities, nk, xk, Sk)
        self.covars_ = self._estimate_covariances(responsibilities, nk, xk, Sk)

    def _initialize(self, X):
        if self.weights_ is None or self.means_ is None or self.covars_ is None:
            # use self.init_params to initialize
            if self.init_params == 'kmeans':
                responsibilities = self._initialize_by_kmeans(X)
            else:
                # other initialization methods as long as they provide responsibilities
                responsibilities = np.random.rand(X.shape[0], self.n_components)
                responsibilities = responsibilities / responsibilities.sum(axis=1)[:, np.newaxis]

            nk, xk, Sk = self._suffcient_statistics(X, responsibilities)
            if self.weights_ is None:
                self.weights_ = self._estimate_weights(responsibilities, nk, xk, Sk)
            if self.means_ is None:
                self.means_ = self._estimate_means(responsibilities, nk, xk, Sk)
            if self.covars_ is None:
                self.covars_ = self._estimate_covariances(responsibilities, nk, xk, Sk)

    def _suffcient_statistics(self, X, responsibilities):
        # compute three sufficient statistics
        nk = responsibilities.sum(axis=0)
        xk = np.dot(responsibilities.T, X) / nk[:, np.newaxis]
        Sk = _sufficient_Sk(responsibilities, X, xk, self.min_covar, self.covariance_type)
        return nk, xk, Sk

    def _fit(self, X):
        # check the parameters
        self._check(X)

        max_log_prob = -np.infty
        responsibilities = np.zeros((X.shape[0], self.n_components))
        for init in range(self.n_init):
            self._initialize(X)

            current_log_likelihood = None
            self.converged_ = False

            for i in range(self.n_iter):
                # e step
                responsibilities = self._estimate_responsibilities(X)

                # log_likelihood
                prev_log_likelihood = current_log_likelihood
                current_log_likelihood = self._estimate_log_likelihood(X)
                if prev_log_likelihood is not None:
                    change = abs(current_log_likelihood - prev_log_likelihood)
                    if change < self.tol:
                        self.converged_ = True

                # m step
                nk, xk, Sk = self._suffcient_statistics(X, responsibilities)
                self._m_step(nk, xk, Sk)

            if self.n_iter:
                if current_log_likelihood > max_log_prob:
                    max_log_prob = current_log_likelihood
                    best_params = {'weights': self.weights_,
                                   'means': self.means_,
                                   'covars': self.covars_}

        # check the existence of an init param that was not subject to
        # likelihood computation issue.
        if np.isneginf(max_log_prob) and self.n_iter:
            raise RuntimeError(
                "EM algorithm was never able to compute a valid likelihood "
                "given initial parameters. Try different init parameters "
                "(or increasing n_init) or check for degenerate data.")

        if self.n_iter:
            self.covars_ = best_params['covars']
            self.means_ = best_params['means']
            self.weights_ = best_params['weights']

        return responsibilities

    def fit(self, X, y=None):
        return self._fit(X)

    def score_samples(self, X):
        pass

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass

    def aic(self, X):
        pass

    def bic(self, X):
        pass


class GaussianMixtureModel(_MixtureModelBase):
    def _estimate_log_weight(self):
        ln_weights = np.log(self.weights_)

    def _estimate_log_likelihood_full(self, X):
        n_samples, n_features = X.shape
        log_prob = np.empty((n_samples, self.n_components))
        for c, (mu, cv) in enumerate(zip(self.means_,  self.covars_)):
            try:
                cv_chol = linalg.cholesky(cv, lower=True)
            except:
                raise ValueError("'covars' must be symmetric, "
                                 "positive-definite")
            cv_log_det = 2 * np.sum(np.log(np.diagonal(cv_chol)))
            cv_sol = linalg.solve_triangular(cv_chol, (X - mu).T, lower=True).T
            log_prob[:, c] = - .5 * (np.sum(cv_sol ** 2, axis=1) +
                                     n_features * np.log(2 * np.pi) + cv_log_det)
        return log_prob

    def _estimate_log_likelihood_tied(self, X):
        pass

    def _estimate_log_likelihood_diag(self, X):
        pass

    def _estimate_log_likelihood_spherical(self, X):
        pass

    def _estimate_weights(self, X, nk, xk, Sk):
        self.weights_ = nk.sum(axis=0) / X.shape[0]

    def _estimate_means(self, X, nk, xk, Sk):
        self.means_ = xk

    def _estimate_covariances_full(self, X, nk, xk, Sk):
        pass

    def _estimate_covariances_tied(self, X, nk, xk, Sk):
        pass

    def _estimate_covariances_diag(self, X, nk, xk, Sk):
        pass

    def _estimate_covariances_spherical(self, X, nk, xk, Sk):
        pass

    def _estimate_log_weights(self):
        pass

    def _estimate_log_likelihood(self, X):
        pass


class BayesianGaussianMixtureModel(_MixtureModelBase):
    def __init__(self, n_components=1, random_state=None, tol=1e-5,
                 min_covar=0, covariance_type='full',
                 n_init=1, n_iter=100, init_params='kmeans',
                 weights=None, means=None, covars=None,
                 alpha_0=None, m_0=None, beta_0=None, nu_0=None, W_0=None,
                 verbose=0):
        super(BayesianGaussianMixtureModel, self).__init__(
            n_components, random_state, tol,
            min_covar, covariance_type, n_init, n_iter, init_params,
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

    def _estimate_log_likelihood(self, X):
        pass


class DirichletProcessGaussianMixtureModel(BayesianGaussianMixtureModel):

    def _estimate_weights(self, X, nk, xk, Sk):
        pass
