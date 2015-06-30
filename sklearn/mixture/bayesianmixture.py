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
from sklearn.utils import ConvergenceWarning
from sklearn.externals.six.moves import zip
from .gaussianmixture import _MixtureBase

def _check_alpha(alpha, n_components):
    """Check the 'alpha', the parameter of the Dirichlet distribution

    Parameters
    ----------
    alpha : array-like, [n_components, ]

    n_components : int

    Returns
    -------
    alpha : array, [n_components, ]
    """
    # check value
    alpha = check_array(alpha, dtype=np.float64, ensure_2d=False)

    # check shape
    if alpha.shape != (n_components,):
        raise ValueError("The 'alpha' should have the shape of "
                         "(n_components, ), "
                         "but got %s" % str(alpha.shape))
    return alpha

def _check_m(m, n_components, n_features):
    """Check the 'm', the parameter of the Gaussian distribution of the mean

    Parameters
    ----------
    m : array-like, [n_components, n_features]

    n_components : int

    n_features : int

    Returns
    -------
    m : array, [n_components, n_features]
    """
    # check value
    m = check_array(m, dtype=np.float64, ensure_2d=False)

    # check shape
    if m.shape != (n_components, n_features):
        raise ValueError("The 'm' should have shape (%s, %d), "
                         "but got %s"
                         % (n_components, n_features, str(m.shape)))
    return m

def _check_beta(beta, n_components):
    """Check the 'beta', the parameter of the Gaussian distribution of
    the means

    Parameters
    ----------
    beta : array-like, [n_components, ]

    n_components : int

    Returns
    -------
    beta : array, [n_components, ]
    """
    # check value
    beta = check_array(beta, dtype=np.float64, ensure_2d=False)

    # check shape
    if beta.shape != (n_components,):
        raise ValueError("The 'beta' should have the shape of "
                         "(n_components, ), "
                         "but got %s" % str(beta.shape))
    return beta


class BayesianGaussianMixture(_MixtureBase):
    def __init__(self, n_components=1, covariance_type='full',
                 random_state=None, tol=1e-6, min_covar=0,
                 n_iter=100, n_init=1, params=None, init_params='kmeans',
                 alpha_init=None, m_init=None, beta_init=None, nu_init=None,
                 W_init=None, verbose=0):
        super(BayesianGaussianMixture, self).__init__(
            n_components, covariance_type, random_state, tol, min_covar,
            n_iter, n_init, params, init_params, None, None, None, verbose)
        self.alpha_init = alpha_init
        self.m_init = m_init
        self.beta_init = beta_init
        self.nu_init = nu_init
        self.W_init = W_init

    def _estimate_weights(self, X, nk, xk, Sk):
        self.alpha = self.alpha_init + nk

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

    def _estimate_log_probabilities_full(self, X):
        pass

    def _estimate_log_probabilities_tied(self, X):
        pass

    def _estimate_log_probabilities_diag(self, X):
        pass

    def _estimate_log_probabilities_spherical(self, X):
        pass

    def _estimate_log_weights(self):
        pass


class DirichletProcessGaussianMixture(BayesianGaussianMixture):

    def _estimate_weights(self, X, nk, xk, Sk):
        pass

    def _estimate_log_weights(self):
        pass
