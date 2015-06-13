
import warnings
import numpy as np
from scipy import linalg
from abc import ABCMeta, abstractmethod
from ..externals import six
from ..base import BaseEstimator
from ..base import DensityMixin
from .. import cluster
from ..utils import check_random_state, check_array

EPS = np.finfo(np.float64).eps*10

def _sufficient_Nk_Sk_full(responsibilities, X, Nk_xk):
    # return Nk_Sk
    pass

def _sufficient_Nk_Sk_diag(responsibilities, X, Nk_xk):
    # return Nk_Sk
    pass

def _sufficient_Nk_Sk_spherical(responsibilities, X, Nk_xk):
    # return Nk_Sk
    pass

def _sufficient_Nk_Sk_tied(responsibilities, X, Nk_xk):
    # return N_S
    pass

def _sufficient_Nk_Sk(responsibilities, X, Nk_xk, covariance_type):
    SUFFICIENT_NK_SK_FUNCTIONS = {"full":_sufficient_Nk_Sk_full,
                                   "diag": _sufficient_Nk_Sk_diag,
                                   "spherical":_sufficient_Nk_Sk_spherical,
                                   "tied":_sufficient_Nk_Sk_tied}
    return SUFFICIENT_NK_SK_FUNCTIONS[covariance_type](responsibilities, X, Nk_xk)

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

    def _check_weights(self):
        if self.weights_ is not None:
            # check value
            self.weights_ = check_array(self.weights_, ensure_2d=False)

            # check shape
            if self.weights_.shape != (self.n_components,):
                raise ValueError(
                    "The GaussianMixtureModel parameter 'weight' must "
                    "have shape (%s, )." % self.n_components)

            # check range
            if (any(np.less(self.weights_, 0)) or
                    any(np.greater(self.weights_, 1))):
                raise ValueError(
                    "The GaussianMixtureModel parameter 'weight' must "
                    "be in the range [0, 1].")

            # check normalization
            with np.errstate(invalid='ignore'):
                if not np.less_equal(np.abs(1 - np.sum(self.weights_)), EPS):
                    raise ValueError(
                        "The GaussianMixtureModel parameter 'weight' must "
                        "be normalized.")

    def _check_means(self):
        if self.means_ is not None:
            # check value
            pass
            # check shape

    def _check(self, X):
        # check the input data X
        X = check_array(X, dtype=np.float64)
        if X.shape[0] < self.n_components:
            raise ValueError(
                'GaussianMixtureModel estimation with %s components, '
                'but got only %s samples.' % (self.n_components, X.shape[0]))

        # check weights
        self._check_weights()

        # check means

        # check covars

    def _initialize_by_kmeans(self, X):
        labels = cluster.KMeans(n_clusters=self.n_components,
                                random_state=self.random_state).fit(X).labels_
        responsibilities = np.zeros((X.shape[0], self.n_components))
        responsibilities[range(X.shape[0]), labels] = 1
        return responsibilities

    @abstractmethod
    def _estimate_weights(self, Nk):
        pass

    @abstractmethod
    def _estimate_means(self, Nk, Nk_xk):
        pass

    @abstractmethod
    def _estimate_covariances(self, Nk, Nk_Sk):
        pass

    @abstractmethod
    def _estimate_responsibilities(self, X):
        pass
    
    @abstractmethod
    def _estimate_ln_likelihood(self, X):
        pass

    def _e_step(self, X):
        self._estimate_responsibilities(X)

    def _m_step(self, Nk, Nk_xk, Nk_Sk):
        self.weights_ = self._estimate_weights(Nk)
        self.means_ = self._estimate_means(Nk, Nk_xk)
        self.covars_ = self._estimate_covariances(Nk, Nk_Sk)

    def _initialize(self, X):
        if self.weights_ is None or self.means_ is None or self.covars_ is None:
            # use self.init_params to initialize
            if self.init_params == 'kmeans':
                responsibilities = self._initialize_by_kmeans(X)
            else:
                # other initialization methods as long as they provide responsibilities
                responsibilities = np.random.rand(X.shape[0], self.n_components)
                responsibilities = responsibilities / responsibilities.sum(1)[:, np.newaxis]

            Nk, Nk_xk, Nk_Sk = self._suffcient_statistics(X, responsibilities)
            if self.weights_ is None:
                self.weights_ = self.estimate_weight(Nk)
            if self.means_ is None:
                self.means_ = self.estimate_means(Nk, Nk_xk)
            if self.covars_ is None:
                self.covars_ = self.estimate_covars(Nk, Nk_Sk)

    def _suffcient_statistics(self, X, responsibilities):
        # compute three sufficient statistics
        Nk = responsibilities.sum(axis=0)
        Nk_xk = np.dot(X.T, responsibilities)
        Nk_Sk = _sufficient_Nk_Sk(responsibilities, X, Nk_xk, self.covariance_type)
        return Nk, Nk_xk, Nk_Sk

    def _fit(self, X):
        # check the parameters
        self._check(X)

        max_log_prob = -np.infty
        responsibilities = np.zeros((X.shape[0], self.n_components))
        for init in range(self.n_init):
            self._initialize(X)

            current_log_likelihood = None
            self.converged_ = False

            for iter in range(self.n_iter):
                # e step
                responsibilities = self._estimate_responsibilities(X)

                # log_likelihood
                prev_log_likelihood = current_log_likelihood
                current_log_likelihood = self._estimate_ln_likelihood(X)
                if prev_log_likelihood is not None:
                    change = abs(current_log_likelihood - prev_log_likelihood)
                    if change < self.tol:
                        self.converged_ = True

                # m step
                Nk, Nk_xk, Nk_Sk = self._suffcient_statistics(X, responsibilities)
                self._m_step(Nk, Nk_xk, Nk_Sk)

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
                "EM algorithm was never able to compute a valid likelihood " +
                "given initial parameters. Try different init parameters " +
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

    def _estimate_ln_weights(self, Nk):
        pass

    def _estimate_ln_precision(self):
        pass

    def _estimate_mahala_dist(self, X):
        pass

    def _estimate_responsibilities(self, X):
        return (self._estimate_ln_weights()
                + .5*self._estimate_ln_precision()
                - .5*self._estimate_mahala_dist(X)
                - X.shape[1]*.5*np.log(2*np.pi))

    def _estimate_weights(self, Nk):
        pass

    def _estimate_means(self, Nk, Nk_xk):
        pass

    def _estimate_covariances(self, Nk, Nk_Sk):
        # estimate four kinds of covariances
        pass

    def _estimate_ln_likelihood(self, X):
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

    def _estimate_ln_weights(self, Nk):
        pass

    def _estimate_ln_precision(self):
        pass

    def _estimate_mahala_dist(self, X):
        pass

    def _estimate_responsibilities(self, X):
        return (self._estimate_ln_weights()
                + .5*self._estimate_ln_precision()
                - .5*self._estimate_mahala_dist(X)
                - X.shape[1]*.5*np.log(2*np.pi))

    def _estimate_weights(self, Nk):
        pass

    def _estimate_means(self, Nk, Nk_xk):
        pass

    def _estimate_covariances(self, Nk, Nk_Sk):
        # estimate four kinds of covariances
        pass

    def _estimate_ln_likelihood(self, X):
        pass

class DirichletProcessGaussianMixtureModel(BayesianGaussianMixtureModel):

    def _estimate_weights(self, Nk):
        pass
