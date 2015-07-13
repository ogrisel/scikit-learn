import numpy as np
from scipy import linalg
from sklearn.externals.six.moves import zip

from ..utils.validation import check_is_fitted
from ..utils import check_array

from .base import MixtureBase, check_shape


def _define_parameter_shape(n_components, n_features, covariance_type):
    cov_shape = {'full': (n_components, n_features, n_features),
                 'tied': (n_features, n_features),
                 'diag': (n_components, n_features),
                 'spherical': (n_components, )}
    param_shape = {'weights': (n_components, ),
                   'means': (n_components, n_features),
                   'covariances': cov_shape[covariance_type]}
    return param_shape


def _check_weights(weights, desired_shape):
    """Check the 'weights'

    Parameters
    ----------
    weights : array-like, (n_components,)

    n_components : int

    Returns
    -------
    weights : array, (n_components,)
    """
    # check value
    weights = check_array(weights, dtype=np.float64, ensure_2d=False)

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


def _check_means(means, desired_shape):
    """Check the 'means'

    Parameters
    ----------
    means : array-like, (n_components, n_features)

    n_components : int

    n_features : int

    Returns
    -------
    means : array, (n_components, n_features)
    """
    # check value
    means = check_array(means, dtype=np.float64, ensure_2d=False)

    # check shape
    check_shape(means, desired_shape, 'means')
    return means


def _check_covars_full(covars, desired_shape):
    covars = check_array(covars, dtype=np.float64, ensure_2d=False,
                         allow_nd=True)
    check_shape(covars, desired_shape, 'full covariance')

    for k, cov in enumerate(covars):
        if (not np.allclose(cov, cov.T) or
                np.any(np.less_equal(linalg.eigvalsh(cov), 0.0))):
            raise ValueError("The component %d of 'full covariance' should be "
                             "symmetric, positive-definite" % k)
    return covars


def _check_covars_tied(covars, desired_shape):
    covars = check_array(covars, dtype=np.float64, ensure_2d=False)
    check_shape(covars, desired_shape, 'tied covariance')

    if (not np.allclose(covars, covars.T) or
            np.any(np.less_equal(linalg.eigvalsh(covars), 0.0))):
        raise ValueError("'tied covariance' should be "
                         "symmetric, positive-definite")
    return covars


def _check_covars_diag(covars, desired_shape):
    covars = check_array(covars, dtype=np.float64, ensure_2d=False)
    check_shape(covars, desired_shape, 'diag covariance')

    if np.any(np.less_equal(covars, 0.0)):
        raise ValueError("'diag covariance' should be positive")
    return covars


def _check_covars_spherical(covars, desired_shape):
    covars = check_array(covars, dtype=np.float64, ensure_2d=False)
    check_shape(covars, desired_shape, 'spherical covariance')

    if np.any(np.less_equal(covars, 0.0)):
        raise ValueError("'spherical covariance' should be positive")
    return covars


def _check_covars(covars, desired_shape, covariance_type):
    """Check the 'covariances'

    Parameters
    ----------
    covars : array-like,
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    n_components : int

    n_features : int

    covariance_type : string

    Returns
    -------
    covars : array
    """
    check_covars_functions = {"full": _check_covars_full,
                              "tied": _check_covars_tied,
                              "diag": _check_covars_diag,
                              "spherical": _check_covars_spherical}
    return check_covars_functions[covariance_type](covars, desired_shape)


class GaussianMixture(MixtureBase):
    """Gaussian Mixture Model

    Representation of a Gaussian mixture model probability distribution.
    This class allows for easy evaluation of, sampling from, and
    maximum-likelihood estimation of the parameters of a GMM distribution.

    Parameters
    ----------
    n_components : int, optional
        Number of mixture components. Defaults to 1.

    covariance_type : string, optional
        String describing the type of covariance parameters to
        use.  Must be one of 'spherical', 'tied', 'diag', 'full'.
        Defaults to 'diag'.

    random_state: RandomState or an int seed
        A random number generator instance. Defaults to None.

    min_covar : float, optional
        Floor on the diagonal of the covariance matrix to prevent
        overfitting.  Defaults to 1e-3.

    tol : float, optional
        Convergence threshold. EM iterations will stop when average
        gain in log-likelihood is below this threshold.  Defaults to 1e-6.

    n_iter : int, optional
        Number of EM iterations to perform.

    n_init : int, optional
        Number of initializations to perform. The best results is kept.

    params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 'w' for weights,
        'm' for means, and 'c' for covars.  Defaults to None.

    init_params : string, optional
        Controls how parameters are initialized unless the parameters are
        provided by users. It should be one of "kmeans", "random", None.
        Defaults to None. If it is not None, the variable responsibilities are
        initialized by the chosen method, which are used to further initialize
        weights, means, and covariances.

    weights : array-like, optional, shape (`n_components`, )
        User-provided initial weights. Defaults to None. If it None, weights
        are initialized by `init_params`.

    means: array-like, optional, shape (`n_components`, `n_features`)
        User-provided initial means. Defaults to None. If it None, means
        are initialized by `init_params`.

    covars: array-like, optional
        User-provided iitial covariances. Defaults to None. If it None, covars
        are initialized by `init_params`.

    verbose : int, default: 0
        Enable verbose output. If 1 then it always prints the current
        initialization and iteration step. If greater than 1 then
        it prints additionally the log probability and the time needed
        for each step.

    Attributes
    ----------
    weights_ : array, shape (`n_components`,)
        This attribute stores the mixing weights for each mixture component.

    means_ : array, shape (`n_components`, `n_features`)
        Mean parameters for each mixture component.

    covars_ : array
        Covariance parameters for each mixture component.  The shape
        depends on `covariance_type`::

            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

    converged_ : bool
        True when convergence was reached in fit(), False otherwise.
    """

    def __init__(self, n_components=1, covariance_type='full',
                 random_state=None, tol=1e-6, min_covar=0,
                 n_iter=100, n_init=1, params=None, init_params='kmeans',
                 weights=None, means=None, covars=None,
                 verbose=0):
        super(GaussianMixture, self).__init__(
            n_components=n_components, covariance_type=covariance_type,
            random_state=random_state, tol=tol, min_covar=min_covar,
            n_iter=n_iter, n_init=n_init, params=params,
            init_params=init_params, verbose=verbose)
        self.weights_init = weights
        self.weights_ = None
        self.means_init = means
        self.means_ = None
        self.covars_init = covars
        self.covars_ = None

    def _check_initial_parameters(self):
        # check the parameters
        param_shape = _define_parameter_shape(
            self.n_components, self.n_features, self.covariance_type)

        # check weights
        if self.weights_init is not None:
            self.weights_init = _check_weights(
                self.weights_init, param_shape['weights'])
        # check means
        if self.means_init is not None:
            self.means_init = _check_means(
                self.means_init, param_shape['means'])
        # check covars
        if self.covars_init is not None:
            self.covars_init = _check_covars(
                self.covars_init, param_shape['covariances'],
                self.covariance_type)

    def _initialize_parameters(self, X, responsibilities, nk, xk, Sk):
        if self.weights_init is None:
            self.weights_ = self._estimate_weights(
                responsibilities, nk, xk, Sk)
        else:
            self.weights_ = self.weights_init

        if self.means_init is None:
            self.means_ = self._estimate_means(
                responsibilities, nk, xk, Sk)
        else:
            self.means_ = self.means_init

        if self.covars_init is None:
            self.covars_ = self._estimate_covariances(
                responsibilities, nk, xk, Sk)
        else:
            self.covars_ = self.covars_init

    # e-step functions
    def _estimate_log_weights(self):
        return np.log(self.weights_)

    def _estimate_log_rho_full(self, X):
        n_samples, n_features = X.shape
        log_prob = np.empty((n_samples, self.n_components))
        for k, (mu, cov) in enumerate(zip(self.means_,  self.covars_)):
            try:
                cov_chol = linalg.cholesky(cov, lower=True)
            except linalg.LinAlgError:
                raise ValueError("'covars' must be symmetric, "
                                 "positive-definite")
            cv_log_det = 2. * np.sum(np.log(np.diagonal(cov_chol)))
            cv_sol = linalg.solve_triangular(cov_chol, (X - mu).T,
                                             lower=True).T
            log_prob[:, k] = - .5 * (cv_log_det +
                                     np.sum(np.square(cv_sol), axis=1))
        log_prob -= .5 * (n_features * np.log(2 * np.pi))
        return log_prob + self._estimate_log_weights()

    def _estimate_log_rho_tied(self, X):
        n_samples, n_features = X.shape
        log_prob = np.empty((n_samples, self.n_components))
        try:
            cov_chol = linalg.cholesky(self.covars_, lower=True)
        except linalg.LinAlgError:
            raise ValueError("'covars' must be symmetric, positive-definite")
        cv_log_det = 2. * np.sum(np.log(np.diagonal(cov_chol)))
        for k, mu in enumerate(self.means_):
            cv_sol = linalg.solve_triangular(cov_chol, (X - mu).T,
                                             lower=True).T
            log_prob[:, k] = np.sum(np.square(cv_sol), axis=1)
        log_prob = - .5 * (n_features * np.log(2 * np.pi)
                           + cv_log_det
                           + log_prob)
        return log_prob + self._estimate_log_weights()

    def _estimate_log_rho_diag(self, X):
        if np.any(np.less_equal(self.covars_, 0.0)):
            raise ValueError("'diag covariance' should be positive")
        n_samples, n_features = X.shape
        log_prob = - .5 * (n_features * np.log(2. * np.pi)
                           + np.sum(np.log(self.covars_), 1)
                           + np.sum((self.means_ ** 2 / self.covars_), 1)
                           - 2. * np.dot(X, (self.means_ / self.covars_).T)
                           + np.dot(X ** 2, (1. / self.covars_).T))
        return log_prob + self._estimate_log_weights()

    def _estimate_log_rho_spherical(self, X):
        if np.any(np.less_equal(self.covars_, 0.0)):
            raise ValueError("'spherical covariance' should be positive")
        n_samples, n_features = X.shape
        log_prob = - .5 * (n_features * np.log(2 * np.pi)
                           + n_features * np.log(self.covars_)
                           + np.sum(self.means_ ** 2, 1) / self.covars_
                           - 2 * np.dot(X, self.means_.T / self.covars_)
                           + np.outer(np.sum(X ** 2, axis=1), 1. / self.covars_))
        return log_prob + self._estimate_log_weights()

    def _estimate_weighted_log_probabilities(self, X):
        estimate_log_rho_functions = {
            "full": self._estimate_log_rho_full,
            "tied": self._estimate_log_rho_tied,
            "diag": self._estimate_log_rho_diag,
            "spherical": self._estimate_log_rho_spherical
        }
        weighted_log_prob = estimate_log_rho_functions[self.covariance_type](X)
        return weighted_log_prob

    def _e_step(self, X):
        _, log_prob, resp = self._estimate_log_probabilities_responsibilities(X)
        return np.sum(log_prob), resp

    # m-step functions
    def _estimate_weights(self, X, nk, xk, Sk):
        return nk / X.shape[0]

    def _estimate_means(self, X, nk, xk, Sk):
        return xk

    def _estimate_covariance_full(self, X, nk, xk, Sk):
        return Sk

    def _estimate_covariance_tied(self, X, nk, xk, Sk):
        return Sk

    def _estimate_covariance_diag(self, X, nk, xk, Sk):
        return Sk

    def _estimate_covariances_spherical(self, X, nk, xk, Sk):
        return Sk

    def _estimate_covariances(self, X, nk, xk, Sk):
        estimate_covariances_functions = {
            "full": self._estimate_covariance_full,
            "tied": self._estimate_covariance_tied,
            "diag": self._estimate_covariance_diag,
            "spherical": self._estimate_covariances_spherical
        }
        return estimate_covariances_functions[self.covariance_type](
            X, nk, xk, Sk)

    def _m_step(self, X, nk, xk, Sk):
        self.weights_ = self._estimate_weights(X, nk, xk, Sk)
        self.means_ = self._estimate_means(X, nk, xk, Sk)
        self.covars_ = self._estimate_covariances(X, nk, xk, Sk)

    def _check_is_fitted(self):
        check_is_fitted(self, 'weights_')
        check_is_fitted(self, 'means_')
        check_is_fitted(self, 'covars_')

    def _get_parameters(self):
        return self.weights_, self.means_, self.covars_

    def _set_parameters(self, params):
        self.weights_, self.means_, self.covars_ = params
