import numpy as np
from scipy import linalg
from scipy.special import digamma, gammaln

from time import time

from ..externals import six
from ..utils import check_array
from ..utils.validation import check_is_fitted
from ..utils.extmath import logsumexp
from sklearn.externals.six.moves import zip
from sklearn.externals.six import print_
from .gaussianmixture import MixtureBase, check_shape


def _define_prior_shape(n_components, n_features, precision_type):
    lambda_W_prior_shape = {
        'full': (n_features, n_features),
        'tied': (n_features, n_features),
        'diag': (n_features, ),
        'spherical': ()}
    param_shape = {'weight_alpha_prior': (),
                   'mu_beta_prior': (),
                   'mu_m_prior': (1, n_features),
                   'lambda_nu_prior': (),
                   'lambda_W_prior': lambda_W_prior_shape[precision_type]}
    return param_shape


def _check_weight_prior(weight_alpha_prior, desired_shape):
    """Check the parameter 'alpha' of the weight Dirichlet distribution

    Parameters
    ----------
    weight_alpha : scalar

    Returns
    -------
    weight_alpha : scalar
    """
    # check shape
    check_shape(weight_alpha_prior, desired_shape, 'alpha')
    if weight_alpha_prior <= 0:
        raise ValueError("The parameter 'weight_alpha_prior' should be "
                         "greater than 0, but got %.5f" % weight_alpha_prior)
    return weight_alpha_prior

def _check_mu_prior(mu_m, mu_beta, desired_shape_m, desired_shape_beta):
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
    check_shape(mu_m, desired_shape_m, 'mu_m_prior')

    # check shape
    check_shape(mu_beta, desired_shape_beta, 'mu_beta_prior')
    if mu_beta <= 0:
        raise ValueError("The parameter 'mu_beta_prior' should be "
                         "greater than 0, but got %.5f" % mu_beta)
    return mu_m, mu_beta


def _check_lambda_nu_prior(lambda_nu, desired_shape, n_features, name):
    check_shape(lambda_nu, desired_shape, name)
    if lambda_nu <= 0:
        raise ValueError("The parameter 'lambda_nu' should be "
                         "greater than 0, but got %.5f" % lambda_nu)
    if np.any(np.less_equal(lambda_nu, n_features - 1)):
        raise ValueError("The parameter '%s' "
                         "should be greater then %d, but got "
                         "minimal value %.5f"
                         % (name, n_features - 1, np.min(lambda_nu)))
    return lambda_nu

def _check_lambda_wishart_prior(lambda_nu, lambda_W,
                                desired_shape_nu, desired_shape_W):
    # Wishart distribution
    lambda_nu = _check_lambda_nu_prior(
        lambda_nu, desired_shape_nu, desired_shape_W[0], 'lambda_nu_prior')

    lambda_W = check_array(lambda_W, dtype=np.float64, ensure_2d=False)
    check_shape(lambda_W, desired_shape_W, 'lambda_W_prior')
    if (not np.allclose(lambda_W, lambda_W.T) or
            np.any(np.less_equal(linalg.eigvalsh(lambda_W), 0.0))):
        raise ValueError("The parameter 'W' of Wishart distribution "
                         "should be symmetric, positive-definite")
    return lambda_nu, lambda_W


def _check_lambda_gamma_prior(lambda_nu, lambda_W,
                              desired_shape_a, desired_shape_b):
    # Gamma distribution
    lambda_nu = _check_lambda_nu_prior(lambda_nu, desired_shape_a,
                                       1, 'lambda_nu_prior')

    lambda_W = check_array(lambda_W, dtype=np.float64, ensure_2d=False)
    check_shape(lambda_W, desired_shape_b, 'lambda_W_prior')
    if np.any(np.less_equal(lambda_W, 0.0)):
        raise ValueError("The parameter 'W' of Gamma distributions "
                         "should be positive")
    return lambda_nu, lambda_W


def _check_lambda_prior(lambda_nu, lambda_W, desired_shape_a,
                        desired_shape_b, covariance_type):
    """Check the parameter of the precision distributions

    Parameters
    ----------
    lambda_nu : array-like, shape of (n_components,)

    lambda_W : array-like,
        'full' : shape of (n_features, n_features)
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

    check_covars_functions = {"full": _check_lambda_wishart_prior,
                              "tied": _check_lambda_wishart_prior,
                              "diag": _check_lambda_gamma_prior,
                              "spherical": _check_lambda_gamma_prior}
    return check_covars_functions[covariance_type](
        lambda_nu, lambda_W, desired_shape_a, desired_shape_b)


def _dirichlet_log_C(alpha):
    """The log of the normalization term of Dirichlet distribution
    """
    return gammaln(np.sum(alpha)) - np.sum(gammaln(alpha))

def _wishart_log_B(n_dim, nu, inv_W_chol):
    """The log of the normalization term of Wishart distribution
    """
    log_det_W = - 2 * np.sum(np.log(np.diag(inv_W_chol)))
    temp1 = - nu * .5 * log_det_W
    temp2 = nu * n_dim * .5 * np.log(2)
    temp3 = n_dim * (n_dim - 1) * .25 * np.log(np.pi)
    temp4 = np.sum(gammaln((nu + 1 - np.arange(1, n_dim + 1)) * .5))
    return temp1 - temp2 - temp3 - temp4

def _wishart_entropy(n_dim, nu, inv_W_chol, log_W):
    return - np.log(_wishart_log_B(n_dim, nu, inv_W_chol)) - \
           .5 * (nu - n_dim - 1) * log_W + .5 * nu * n_dim

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
        param_shape = _define_prior_shape(
            self.n_components, self.n_features, self.precision_type)
        if self.weight_alpha_prior is not None:
            _check_weight_prior(self.weight_alpha_prior,
                                param_shape['weight_alpha_prior'])

        if self.mu_beta_prior is not None or self.mu_m_prior is not None:
            _check_mu_prior(self.mu_m_prior, self.mu_beta_prior,
                            param_shape['mu_m_prior'],
                            param_shape['mu_beta_prior'])

        if self.lambda_nu_prior is not None or self.lambda_W_prior is not None:
            _check_lambda_prior(
                self.lambda_nu_prior, self.lambda_W_prior,
                param_shape['lambda_nu_prior'], param_shape['lambda_W_prior'],
                self.precision_type)

    def _initialize_weight(self, X, nk, xk, Sk):
        if self.weight_alpha_prior is None:
            # TODO discuss default value
            self.weight_alpha_prior = 1e-3 * np.array(self.n_components)

        self.weight_alpha_ = self._estimate_weights(X, nk, xk, Sk)

        # for lower bound
        self.log_C_alpha_prior = _dirichlet_log_C(np.ones(self.n_components) *
                                                  self.weight_alpha_prior)

    def _initialize_mu(self, X, nk, xk, Sk):
        if self.mu_beta_prior is None:
            self.mu_beta_prior = 1
            # TODO discuss default value

        if self.mu_m_prior is None:
            self.mu_m_prior = X.mean(axis=0).reshape(1, -1)
            # TODO discuss default value

        self.mu_beta_, self.mu_m_ = self._estimate_means(X, nk, xk, Sk)
        self.log_beta_prior = self.n_features * np.log(self.mu_beta_prior / (2 * np.pi))

    def _initialize_lambda(self, X, nk, xk, Sk):
        if self.lambda_nu_prior is None:
            self.lambda_nu_prior = self.n_features
            # TODO discuss default value

        if self.lambda_W_prior is None:
            self.lambda_inv_W_prior = np.cov(X.T, bias=1) * self.n_features
            # TODO discuss default value
            try:
                self.lambda_W_prior = np.linalg.inv(self.lambda_inv_W_prior)
            except linalg.LinAlgError:
                raise ValueError("lambda_W_prior must be symmetric, "
                                 "positive-definite. Check data distribution")
        else:
            try:
                self.lambda_inv_W_prior = np.linalg.inv(self.lambda_W_prior)
            except linalg.LinAlgError:
                raise ValueError("lambda_W_prior must be symmetric, "
                                 "positive-definite.")

        self.lambda_nu_, self.lambda_inv_W_ = self._estimate_lambda(
            X, nk, xk, Sk)
        self.log_B_W_nu_prior = _wishart_log_B(
            self.n_features, self.lambda_nu_prior, self.lambda_W_prior)

    def _initialize_parameters(self, X, responsibilities, nk, xk, Sk):
        self._initialize_weight(X, nk, xk, Sk)
        self._initialize_mu(X, nk, xk, Sk)
        self._initialize_lambda(X, nk, xk, Sk)

    # m step
    def _estimate_weights(self, X, nk, xk, Sk):
        print 'alpha'
        print self.weight_alpha_prior + nk
        return self.weight_alpha_prior + nk

    def _estimate_means(self, X, nk, xk, Sk):
        mu_beta_ = self.mu_beta_prior + nk
        mu_m_ = (self.mu_beta_prior * self.mu_m_prior +
                      nk[:, np.newaxis] * xk) / mu_beta_[:, np.newaxis]
        print 'mu_beta'
        print mu_beta_
        print 'mu_m'
        print mu_m_
        return mu_beta_, mu_m_

    def _estimate_lambda_full(self, X, nk, xk, Sk):
        lambda_nu_ = self.lambda_nu_prior + nk
        lambda_inv_W_ = np.empty((self.n_components, self.n_features,
                                  self.n_features))
        for k in range(self.n_components):
            diff = xk[k] - self.mu_m_prior
            lambda_inv_W_[k] = (
                self.lambda_inv_W_prior + nk[k] * Sk[k] +
                (nk[k] * self.mu_beta_prior / self.mu_beta_[k]) *
                np.outer(diff, diff))
            print 'Wishart Mean'
            print np.linalg.inv(lambda_inv_W_[k]) * lambda_nu_[k]
        return lambda_nu_, lambda_inv_W_

    def _estimate_lambda_tied(self, X, nk, xk, Sk):
        pass

    def _estimate_lambda_diag(self, X, nk, xk, Sk):
        pass

    def _estimate_lambda_spherical(self, X, nk, xk, Sk):
        pass

    def _estimate_lambda(self, X, nk, xk, Sk):
        estimate_lambda_functions = {
            "full": self._estimate_lambda_full,
            "tied": self._estimate_lambda_tied,
            "diag": self._estimate_lambda_diag,
            "spherical": self._estimate_lambda_spherical
        }
        return estimate_lambda_functions[self.covariance_type](X, nk, xk, Sk)

    def _e_step(self, X):
        log_prob_comp, log_prob, resp = \
            self._estimate_log_probabilities_responsibilities(X)
        lower_bound = self._lower_bound(log_prob_comp, resp)
        return lower_bound, resp

    def _m_step(self, X, nk, xk, Sk):
        self.weight_alpha_ = self._estimate_weights(X, nk, xk, Sk)
        self.mu_beta_, self.mu_m_ = self._estimate_means(X, nk, xk, Sk)
        self.lambda_nu_, self.lambda_inv_W_ = self._estimate_lambda(
            X, nk, xk, Sk)

    # e step
    def _estimate_weighted_log_probabilities(self, X):
        # save ln_weights for computing lower bound
        log_weights = self._estimate_log_weights()
        _estimate_log_X = {
            "full": self._estimate_log_X_full,
            "tied": self._estimate_log_X_tied,
            "diag": self._estimate_log_X_diag,
            "spherical": self._estimate_log_X_spherical
        }
        # save ln_lambda, inv_W_chol for computing lower bound
        log_X = _estimate_log_X[self.precision_type](X)
        return log_weights + log_X

    def _estimate_log_weights(self):
        # first item in Equation 3.8
        log_weights = (digamma(self.weight_alpha_) -
                digamma(np.sum(self.weight_alpha_)))
        self.log_weights = log_weights
        return log_weights

    def _estimate_log_X_full(self, X):
        # second item in Equation 3.8
        n_samples = X.shape[0]
        n_features = self.n_features
        ln_W_digamma = np.arange(1, self.n_features + 1)
        log_X = np.empty((n_samples, self.n_components))
        log_lambda = np.empty((self.n_components, ))
        inv_W_chol = np.empty(self.lambda_inv_W_.shape)
        for k, (beta, m, nu, inv_W) in enumerate(
                zip(self.mu_beta_, self.mu_m_,
                    self.lambda_nu_, self.lambda_inv_W_)):
            try:
                inv_W_chol[k] = linalg.cholesky(inv_W, lower=True)
            except linalg.LinAlgError:
                raise ValueError("'lambda_inv_W_' must be symmetric, "
                                 "positive-definite")
            ln_inv_W_det = 2. * np.sum(np.log(np.diagonal(inv_W_chol[k])))
            log_lambda[k] = (np.sum(digamma(.5 * (nu + 1 - ln_W_digamma))) +
                             n_features * np.log(2) - ln_inv_W_det)

            W_sol = linalg.solve_triangular(inv_W_chol[k], (X - m).T,
                                            lower=True).T
            mahala_dist = np.sum(np.square(W_sol), axis=1)
            log_X[:, k] = - .5 * (- log_lambda[k] +
                                  n_features / beta + nu * mahala_dist)
        log_X -= .5 * (n_features * np.log(2 * np.pi))
        self.inv_W_chol = inv_W_chol
        self.log_lambda = log_lambda
        return log_X

    def _estimate_log_X_tied(self, X):
        pass

    def _estimate_log_X_diag(self, X):
        pass

    def _estimate_log_X_spherical(self, X):
        pass

    def _check_is_fitted(self):
        check_is_fitted(self, 'weight_alpha_')
        check_is_fitted(self, 'mu_beta_')
        check_is_fitted(self, 'mu_m_')
        check_is_fitted(self, 'lambda_nu_')
        check_is_fitted(self, 'lambda_inv_W_')

    def _get_parameters(self):
        return (self.weight_alpha_, self.mu_beta_, self.mu_m_, self.lambda_nu_,
                self.lambda_inv_W_)

    def _set_parameters(self, params):
        (self.weight_alpha_, self.mu_beta_, self.mu_m_, self.lambda_nu_,
         self.lambda_inv_W_) = params

    # lower bound methods
    def _lower_bound(self, log_prob_comp, resp):
        log_p_XZ = self._estimate_p_XZ(log_prob_comp, resp)
        log_p_weight = self._estimate_p_weight()
        log_p_mu_lambda = self._estimate_p_mu_lambda()
        log_q_z = self._estimate_q_Z(resp)
        log_q_weight = self._estimate_q_weight()
        log_q_mu_lambda = self._estimate_q_mu_lambda()
        print 'lower bound'
        print log_p_XZ
        print log_p_weight
        print log_p_mu_lambda
        print log_q_z
        print log_q_weight
        print log_q_mu_lambda
        print 'lower bound'
        return log_p_XZ + log_p_weight + log_p_mu_lambda + log_q_z + log_q_weight + log_q_mu_lambda

    def _estimate_p_XZ(self, log_prob_comp, resp):
        return np.sum(log_prob_comp * resp)

    def _estimate_p_weight(self):
        return self.log_C_alpha_prior + \
               (self.weight_alpha_prior - 1) * np.sum(self.log_weights)

    def _estimate_p_mu_lambda(self):
        temp1 = self.log_beta_prior + self.log_lambda - \
            self.n_features * self.mu_beta_prior / self.mu_beta_
        mk_sol = np.empty(self.n_components)
        for k in range(self.n_components):
            sol = linalg.solve_triangular(
                self.inv_W_chol[k],
                (self.mu_m_[k] - self.mu_m_prior).T, lower=True).T
            mk_sol[k] = np.sum(np.square(sol), axis=1)
        temp2 = self.mu_beta_prior * self.lambda_nu_ * mk_sol
        temp_mu = .5 * np.sum(temp1 + temp2)

        temp3 = self.n_components * self.log_B_W_nu_prior + \
                .5 * (self.lambda_nu_prior - self.n_features - 1) * np.sum(self.log_lambda)

        W0inv_Wk_tr = np.empty(self.n_components)
        for k in range(self.n_components):
            # another way to compute
            # sol = linalg.solve_triangular(
            #     self.inv_W_chol[k], self.lambda_inv_W_prior.T, lower=True)
            # W0inv_Wk_tr[k] = np.trace(linalg.solve_triangular(
            #     self.inv_W_chol[k], sol, lower=True, trans=1).T)
            inv_chol_sol = linalg.inv(self.inv_W_chol[k])
            W0inv_Wk_tr[k] = np.sum(self.lambda_inv_W_prior.T * np.dot(inv_chol_sol.T, inv_chol_sol))
        temp4 = -.5 * np.sum(self.lambda_nu_ * W0inv_Wk_tr)
        return temp_mu + temp3 + temp4

    def _estimate_q_Z(self, resp):
        # TODO underflow
        return np.sum(resp * np.log(resp))

    def _estimate_q_weight(self):
        return np.sum((self.weight_alpha_ - 1) * self.log_weights) + \
               _dirichlet_log_C(self.weight_alpha_)

    def _estimate_q_mu_lambda(self):
        wishart_entropy = np.empty(self.n_components)
        for k in range(self.n_components):
            wishart_entropy[k] = _wishart_entropy(
                self.n_features, self.lambda_nu_[k], self.lambda_inv_W_[k],
                self.log_lambda[k])
        return np.sum(.5 * self.log_lambda +
                      .5 * self.n_features * np.log(self.mu_beta_ / (2 * np.pi))
                      - self.n_features * .5 - wishart_entropy)


class DirichletProcessGaussianMixture(BayesianGaussianMixture):

    def _estimate_weights(self, X, nk, xk, Sk):
        pass

    def _estimate_log_weights(self):
        pass
