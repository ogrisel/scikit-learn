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
from sklearn.externals.six import print_

from .gaussianmixture import MixtureBase
from .gaussianmixture import sufficient_statistics, check_shape

def _define_checking_shape(n_components, n_features, precision_type):
    lambda_W = {'full': (n_components, n_features, n_features),
                 'tied': (n_features, n_features),
                 'diag': (n_components, n_features),
                 'spherical': (n_components, )}
    param_shape = {'weight_alpha': (n_components, ),
                   'mu_beta': (n_components,),
                   'mu_m': (n_components, n_features),
                   'lambda_nu': (n_components, ),
                   'lambda_W': lambda_W[precision_type]}
    return param_shape


def _check_weights(weight_alpha, desired_shape):
    """Check the parameter 'alpha' of the weight Dirichlet distribution

    Parameters
    ----------
    weight_alpha : array-like, (n_components,)

    n_components : int

    Returns
    -------
    weight_alpha : array, (n_components,)
    """
    # check value
    weight_alpha = check_array(weight_alpha, dtype=np.float64, ensure_2d=False)

    # check shape
    check_shape(weight_alpha, desired_shape, 'alpha')
    return weight_alpha

def _check_mu(mu_m, mu_beta, desired_shape_m, desired_shape_beta):
    """Check the 'm', the parameter of the Gaussian distribution of the mean

    Parameters
    ----------
    m : array-like, (n_components, n_features)

    beta : array-like, (n_components,)

    n_components : int

    n_features : int

    Returns
    -------
    m : array, (n_components, n_features)

    beta : array-like, (n_components,)
    """
    # check value
    mu_m = check_array(mu_m, dtype=np.float64, ensure_2d=False)

    # check shape
    check_shape(mu_m, desired_shape_m, 'mu_m')

    # check value
    mu_beta = check_array(mu_beta, dtype=np.float64, ensure_2d=False)

    # check shape
    check_shape(mu_beta, desired_shape_beta, 'mu_beta')
    return mu_m, mu_beta


def _check_lambda_nu(lambda_nu, desired_shape, n_features, name):
    lambda_nu = check_array(lambda_nu, dtype=np.float64, ensure_2d=False)
    check_shape(lambda_nu, desired_shape, name)

    if np.any(np.less_equal(lambda_nu, n_features - 1)):
        raise ValueError("The parameter '%s' "
                         "should be greater then %d, but got "
                         "minimal value %.5f"
                         % (name, n_features - 1, np.min(lambda_nu)))
    return lambda_nu


def _check_lambda_full(lambda_nu, lambda_W, desired_shape_nu, desired_shape_W):
    # Wishart distribution
    # check value
    lambda_nu = _check_lambda_nu(
        lambda_nu, desired_shape_nu, desired_shape_W[1], 'lambda_nu')

    # check value
    lambda_W = check_array(lambda_W, dtype=np.float64, ensure_2d=False,
                           allow_nd=True)
    # check dimension
    check_shape(lambda_W, desired_shape_W, 'lambda_W')

    for k, W in enumerate(lambda_W):
        if (not np.allclose(W, W.T) or
                np.any(np.less_equal(linalg.eigvalsh(W), 0.0))):
            raise ValueError("The component %d of 'full' Wishart distribution "
                             "parameter 'W' should be symmetric, "
                             "positive-definite" % k)
    return lambda_nu, lambda_W


def _check_lambda_tied(lambda_nu, lambda_W, desired_shape_nu, desired_shape_W):
    # Wishart distribution
    # check value
    lambda_nu = _check_lambda_nu(
        lambda_nu, desired_shape_nu, desired_shape_W[0], 'lambda_nu')

    # check value
    lambda_W = check_array(lambda_W, dtype=np.float64, ensure_2d=False)

    # check dimension
    check_shape(lambda_W, desired_shape_W, 'lambda_W')

    if (not np.allclose(lambda_W, lambda_W.T) or
            np.any(np.less_equal(linalg.eigvalsh(lambda_W), 0.0))):
        raise ValueError("The parameter 'W' of 'tied' Wishart distribution "
                         "should be symmetric, positive-definite")
    return lambda_nu, lambda_W


def _check_lambda_diag(lambda_nu, lambda_W, desired_shape_a, desired_shape_b):
    # Gamma distribution

    # check value
    lambda_nu = _check_lambda_nu(lambda_nu, desired_shape_a, 1, 'lambda_nu')

    # the shape of tau_b must be k x d
    check_shape(lambda_W, desired_shape_b, 'lambda_W')
    if np.any(np.less_equal(lambda_W, 0.0)):
        raise ValueError("The parameter 'W' of 'diag' Gamma distributions "
                         "should be positive")
    return lambda_nu, lambda_W


def _check_lambda_spherial(lambda_nu, lambda_W, desired_shape_a,
                           desired_shape_b):
    # Gamma distribution

    # check value
    lambda_nu = _check_lambda_nu(lambda_nu, desired_shape_a, 1, 'lambda_nu')

    # the shape of gamma_b must be (k, )
    lambda_W = check_array(lambda_W, dtype=np.float64, ensure_2d=False)

    check_shape(lambda_W, desired_shape_b, 'lambda_W')
    if np.any(np.less_equal(lambda_W, 0.0)):
        raise ValueError("The parameter 'W' of 'spherical' Gamma "
                         "distributions should be positive")
    return lambda_nu, lambda_W


def _check_lambda(lambda_nu, lambda_W, desired_shape_a,
                  desired_shape_b, covariance_type):
    """Check the parameter of the precision distributions

    Parameters
    ----------
    lambda_nu : array-like, shape of (n_components,)

    lambda_W : array-like,
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    n_components : int

    n_features : int

    covariance_type : string

    Returns
    -------
    lambda_nu : array-like

    lambda_W : array-like
    """

    check_covars_functions = {"full": _check_lambda_full,
                              "tied": _check_lambda_tied,
                              "diag": _check_lambda_diag,
                              "spherical": _check_lambda_spherial}
    return check_covars_functions[covariance_type](
        lambda_nu, lambda_W, desired_shape_a, desired_shape_b)


class BayesianGaussianMixture(MixtureBase):
    def __init__(self, n_components=1, precision_type='full',
                 random_state=None, tol=1e-6, min_covar=0,
                 n_iter=100, n_init=1, params=None, init_params='kmeans',
                 weight_alpha_prior=None,
                 mu_m_prior=None, mu_beta_prior=None,
                 lambda_nu_prior=None, lambda_W_prior=None,
                 verbose=0):
        super(BayesianGaussianMixture, self).__init__(
            n_components, precision_type, random_state, tol, min_covar,
            n_iter, n_init, params, init_params, verbose)
        self.precision_type = precision_type
        self.weight_alpha_prior = weight_alpha_prior
        self.mu_m_prior = mu_m_prior
        self.mu_beta_prior = mu_beta_prior
        self.lambda_nu_prior = lambda_nu_prior
        self.lambda_W_prior = lambda_W_prior

    def _check_initial_parameters(self):
        param_shape = _define_checking_shape(
            self.n_components, self.n_features, self.precision_type)
        if self.weight_alpha_prior is not None:
            _check_weights(self.weight_alpha_prior,
                           param_shape['weight_alpha'])

        if self.mu_beta_prior is not None or self.mu_m_prior is not None:
            _check_mu(self.mu_m_prior, self.mu_beta_prior,
                      param_shape['mu_m'], param_shape['mu_beta'])

        if self.lambda_nu_prior is not None or self.lambda_W_prior is not None:
            _check_lambda(self.lambda_nu_prior, self.lambda_W_prior,
                          param_shape['lambda_nu'], param_shape['lambda_W'])

    def _initialize_weight(self, weight_alpha_prior, X, nk, xk, Sk):
        if weight_alpha_prior is None:
            # TODO discuss default value
            self.weight_alpha_prior = 1e-3 * np.array(self.n_components)
            if self.verbose > 1:
                print_('\n\talpha_prior are initialized.', end='')
        else:
            self.weight_alpha_prior = weight_alpha_prior
            if self.verbose > 1:
                print_('\n\talpha_prior are provided.', end='')

        self._estimate_weights(X, nk, xk, Sk)
        if self.verbose > 1:
            print_('\talpha are initialized.', end='')

    def _initialize_mu(self, mu_beta_prior, mu_m_prior, X, nk, xk, Sk):
        if mu_beta_prior is None:
            self.mu_beta_prior = 1
            # TODO discuss default value
            if self.verbose > 1:
                print_('\n\tmu_beta_prior are initialized.', end='')
        else:
            self.mu_beta_prior = mu_beta_prior
            if self.verbose > 1:
                print_('\n\tmu_beta_prior are provided.', end='')

        if mu_m_prior is None:
            self.mu_m_prior = np.tile(X.mean(axis=0), (self.n_components, 1))
            # TODO discuss default value
            if self.verbose > 1:
                print_('\n\tmu_m_prior are initialized.', end='')
        else:
            self.mu_m_prior = mu_m_prior
            if self.verbose > 1:
                print_('\n\tmu_m_prior are provided.', end='')

        self._estimate_means(X, nk, xk, Sk)
        if self.verbose > 1:
            print_('\tmu are initialized.', end='')

    def _initialize_lambda(self, lambda_nu_prior, lambda_W_prior, X, nk, xk, Sk):
        if lambda_nu_prior is None:
            self.lambda_nu_prior = X.shape[1]
            # TODO discuss default value
            if self.verbose > 1:
                print_('\n\tlambda_nu_prior are initialized.', end='')
        else:
            self.lambda_nu_prior = lambda_nu_prior
            if self.verbose > 1:
                print_('\n\tlambda_nu_prior are provided.', end='')

        if lambda_W_prior is None:
            self.lambda_inv_W_prior = np.cov(X) * X.shape[0]
            try:
                self.lambda_W_prior = np.linalg.inv(self.lambda_inv_W_prior)
            except linalg.LinAlgError:
                raise ValueError("lambda_W_prior must be symmetric, "
                                 "positive-definite. Check data distribution")
            if self.verbose > 1:
                print_('\n\tlambda_W_prior are initialized.', end='')
        else:
            self.lambda_W_prior = lambda_W_prior
            try:
                self.lambda_inv_W_prior = np.linalg.inv(self.lambda_W_prior)
            except linalg.LinAlgError:
                raise ValueError("lambda_W_prior must be symmetric, "
                                 "positive-definite.")
            if self.verbose > 1:
                print_('\n\tlambda_W_prior are provided.', end='')

        self._estimate_lambda(X, nk, xk, Sk)
        if self.verbose > 1:
            print_('\tlambda_W are initialized.', end='')

    def _initialize_parameters(self, X, nk, xk, Sk,
                               weight_alpha_prior=None,
                               mu_beta_prior=None, mu_m_prior=None,
                               lambda_nu_prior=None, lambda_W_prior=None):
        self._initialize_weight(weight_alpha_prior, X, nk, xk, Sk)
        self._initialize_mu(mu_beta_prior, mu_m_prior, X, nk, xk, Sk)
        self._initialize_lambda(lambda_nu_prior, lambda_W_prior, X, nk, xk, Sk)

    def _e_step(self, X):
        pass

    def _m_step(self, responsibilities, nk, xk, Sk):
        pass

    def _estimate_weights(self, X, nk, xk, Sk):
        self.weight_alpha_ = self.weight_alpha_prior + nk

    def _estimate_means(self, X, nk, xk, Sk):
        self.mu_beta_ = self.mu_beta_prior + nk
        self.mu_m_ = (self.mu_beta_prior * self.mu_m_prior + nk * xk) / self.mu_beta_

    def _estimate_lambda_full(self, X, nk, xk, Sk):
        pass

    def _estimate_lambda_tied(self, X, nk, xk, Sk):
        pass

    def _estimate_lambda_diag(self, X, nk, xk, Sk):
        pass

    def _estimate_lambda_spherical(self, X, nk, xk, Sk):
        pass

    def _estimate_lambda(self, X, nk, xk, Sk):
        pass

    def _estimate_log_rho_full(self, X):
        pass

    def _estimate_log_rho_tied(self, X):
        pass

    def _estimate_log_rho_diag(self, X):
        pass

    def _estimate_log_rho_spherical(self, X):
        pass

    def _estimate_log_weights(self):
        pass


class DirichletProcessGaussianMixture(BayesianGaussianMixture):

    def _estimate_weights(self, X, nk, xk, Sk):
        pass

    def _estimate_log_weights(self):
        pass
