import numpy as np
from scipy import linalg
from scipy.special import digamma, gammaln

from time import time

from ..utils import check_array
from ..utils.validation import check_is_fitted
from sklearn.externals.six import print_
from .base import MixtureBase, check_shape
from .gaussianmixture import estimate_Gaussian_suffstat_Sk
from .gaussianmixture import estimate_Gaussian_suffstat_xk


EPS = np.finfo(float).eps


def _define_prior_shape(n_features, precision_type):
    """Define the shape of the prior parameters
    """
    lambda_W_prior_shape = {
        'full': (n_features, n_features),
        'tied': (n_features, n_features),
        'diag': (n_features, ),
        'spherical': ()}
    param_shape = {'weight_alpha_prior': (),
                   'mu_beta_prior': (),
                   'mu_m_prior': (n_features, ),
                   'lambda_nu_prior': (),
                   'lambda_W_prior': lambda_W_prior_shape[precision_type]}
    return param_shape


def _check_mu_prior(mu_m_prior, mu_beta_prior,
                    desired_shape_m, desired_shape_beta):
    """Check mu_m_prior, mu_beta_prior,
    the prior parameter of the mean Gaussian distribution

    Parameters
    ----------
    mu_m_prior : array-like, (n_components, n_features)

    mu_beta_prior : array-like, (n_components,)

    desired_shape_m : tuple

    desired_shape_beta : tuple

    Returns
    -------
    mu_m : array, shape (n_components, n_features)

    mu_beta : array-like, shape (n_components,)
    """
    # check value
    mu_m_prior = check_array(mu_m_prior, dtype=np.float64, ensure_2d=False)

    # check shape
    check_shape(mu_m_prior, desired_shape_m, 'mu_m_prior')

    # check shape
    check_shape(mu_beta_prior, desired_shape_beta, 'mu_beta_prior')
    if mu_beta_prior <= 0:
        raise ValueError("The parameter 'mu_beta_prior' should be "
                         "greater than 0, but got %.5f" % mu_beta_prior)
    return mu_m_prior, mu_beta_prior


def _check_lambda_nu_prior(lambda_nu_prior,
                           desired_shape, n_features):
    """Check lambda_nu_prior
    The prior parameter of Wishart or Gamma distribution

    Parameters
    ----------
    lambda_nu_prior : float

    desired_shape : tuple

    n_features : int

    Returns
    -------
    lambda_nu_prior : float
    """
    check_shape(lambda_nu_prior, desired_shape, 'lambda_nu_prior')
    if lambda_nu_prior <= 0:
        raise ValueError("The parameter 'lambda_nu_prior' should be "
                         "greater than 0, but got %.5f" % lambda_nu_prior)
    if np.any(np.less_equal(lambda_nu_prior, n_features - 1)):
        raise ValueError("The parameter 'lambda_nu_prior' "
                         "should be greater then %d, but got "
                         "minimal value %.5f"
                         % (n_features - 1, np.min(lambda_nu_prior)))
    return lambda_nu_prior


def _check_lambda_wishart_prior(lambda_nu_prior, lambda_W_prior,
                                desired_shape_nu, desired_shape_W):
    """Check lambda_nu_prior, lambda_W_prior
    The prior parameter of Wishart distribution

    Parameters
    ----------
    lambda_nu_prior : float

    lambda_W_prior : array-like

    desired_shape_nu : tuple

    desired_shape_W : int

    Returns
    -------
    lambda_nu_prior : float

    lambda_W_prior : array-like
    """
    lambda_nu_prior = _check_lambda_nu_prior(
        lambda_nu_prior, desired_shape_nu, desired_shape_W[0])

    lambda_W_prior = check_array(lambda_W_prior, dtype=np.float64,
                                 ensure_2d=False)

    check_shape(lambda_W_prior, desired_shape_W, 'lambda_W_prior')
    if (not np.allclose(lambda_W_prior, lambda_W_prior.T) or
            np.any(np.less_equal(linalg.eigvalsh(lambda_W_prior), 0.0))):
        raise ValueError("The parameter 'W' of Wishart distribution "
                         "should be symmetric, positive-definite")
    return lambda_nu_prior, lambda_W_prior


def _check_lambda_gamma_prior(lambda_nu_prior, lambda_W_prior,
                              desired_shape_a, desired_shape_b):
    """Check lambda_nu_prior, lambda_W_prior
    The prior parameter of Gamma distribution

    Parameters
    ----------
    lambda_nu_prior : float

    lambda_W_prior : array-like

    desired_shape_nu : tuple

    desired_shape_W : int

    Returns
    -------
    lambda_nu_prior : float

    lambda_W_prior : array-like
    """
    lambda_nu_prior = _check_lambda_nu_prior(lambda_nu_prior,
                                             desired_shape_a, 1)

    lambda_W_prior = check_array(lambda_W_prior, dtype=np.float64,
                                 ensure_2d=False)
    check_shape(lambda_W_prior, desired_shape_b, 'lambda_W_prior')
    if np.any(np.less_equal(lambda_W_prior, 0.0)):
        raise ValueError("The parameter 'W' of Gamma distributions "
                         "should be positive")
    return lambda_nu_prior, lambda_W_prior


def _check_lambda_prior(lambda_nu_prior, lambda_W_prior,
                        desired_shape_a, desired_shape_b, prec_type):
    """Check the parameter of the precision distributions

    Parameters
    ----------
    lambda_nu_prior : array-like, shape of (n_components,)

    lambda_W_prior : array-like,
        'full' : shape of (n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)

    n_components : int

    n_features : int

    prec_type : string

    Returns
    -------
    lambda_nu_prior : array-like

    lambda_W_prior : array-like
    """

    check_covars_functions = {"full": _check_lambda_wishart_prior,
                              "tied": _check_lambda_wishart_prior,
                              "diag": _check_lambda_gamma_prior,
                              "spherical": _check_lambda_gamma_prior}
    return check_covars_functions[prec_type](
        lambda_nu_prior, lambda_W_prior, desired_shape_a, desired_shape_b)


def _log_dirichlet_norm(alpha):
    """The log of the normalization term of Dirichlet distribution
    """
    return gammaln(np.sum(alpha)) - np.sum(gammaln(alpha))


def _log_wishart_norm(n_dim, nu, inv_W_chol):
    """The log of the normalization term of Wishart distribution
    """
    log_det_inv_W = 2 * np.sum(np.log(np.diag(inv_W_chol)))
    temp1 = nu * .5 * log_det_inv_W
    temp2 = nu * n_dim * .5 * np.log(2)
    temp3 = n_dim * (n_dim - 1) * .25 * np.log(np.pi)
    temp4 = np.sum(gammaln(.5 * (nu + 1 - np.arange(1, n_dim + 1))))
    return temp1 - temp2 - temp3 - temp4

def _log_gamma_norm_spherical(a, inv_b):
    """Compute one log of normalization of gamma distribution"""
    return a * np.log(inv_b) - gammaln(a)


def _log_gamma_norm_diag(a, inv_b):
    """Compute n_features log of normalization of gamma distribution"""
    return a * np.sum(np.log(inv_b)) - len(inv_b) * gammaln(a)


def _wishart_entropy(n_dim, nu, inv_W_chol, log_lambda):
    """The entropy of the Wishart distribution
    """
    return - _log_wishart_norm(n_dim, nu, inv_W_chol) - \
        .5 * (nu - n_dim - 1) * log_lambda + .5 * nu * n_dim


def _gamma_entropy_spherical(a, inv_b):
    return gammaln(a) - (a - 1) * digamma(a) - np.log(inv_b) + a


def _gamma_entropy_diag(a, inv_b):
    return (gammaln(a) - (a-1) * digamma(a) + a) * len(inv_b) - \
           np.sum(np.log(inv_b))


class BayesianGaussianMixture(MixtureBase):
    """
    Variational inference for a Bayesian Gaussian mixture model probability
    distribution. This class allows for easy and efficient inference
    of an approximate posterior distribution over the parameters of a
    Gaussian mixture model with a fixed number of components.

    Read more in the :ref:`User Guide <bgmm>`.

    Parameters
    ----------
    n_components: int, default to 1.
        Number of mixture components.

    precision_type: string, default to 'full'.
        String describing the type of covariance parameters to
        use.  Must be one of 'spherical', 'tied', 'diag', 'full'.

    random_state: RandomState or an int seed, defaults to None.
        A random number generator instance.

    tol : float, default 1e-3.
        Convergence threshold.

    reg_covar : float, defaults to 0.
        Non-negative regularization to the diagonal of covariance.

    n_iter : int, default to 100.
        Maximum number of iterations to perform before convergence.

    n_init : int, default to 1.
        Number of initializations to perform. The best results is kept.

    params : string, defaults to None.
        Controls how priorparameters are initialized unless the
        prior parameters are provided by users.
        It should be one of "kmeans", "random", None.
        Defaults to None. If it is not None, the variable responsibilities are
        initialized by the chosen method, which are used to further initialize
        weights, mu, and lambda.

    init_params : string, default to None.
        Controls which parameters are updated in the initialization
        process.  Can contain any combination of 'w' for weights,
        'm' for means, and 'c' for covars.  Defaults to 'wmc'.

    verbose : int, default to 0.
        Controls output verbosity.

    weight_alpha_prior : float, defaults to None.
        User-provided prior parameter of weights. If it is None,
        weight_alpha_prior is set to 1e-3 * n_components.

    mu_m_prior : array-like, shape (1, `n_components`),
        defaults to None.
        User-provided prior parameter of mu. If it None, mu_m_prior
        is set to the mean of X.

    mu_beta_prior : float, defaults to None.
        User-provided prior parameter of mu. If it is None,
        mu_beta_prior is set to 1.

    lambda_nu_prior : float, defaults to None.
        User-provided prior parameter of lambda. If it is None,
        mu_beta_prior is set to n_features.

    lambda_W_prior : array-like, defaults to None.
        User-provided prior parameter of lambda. If it is None,
        mu_beta_prior is set to inv(cov(X) * n_features).

    Attributes
    ----------
    weight_alpha_ : array, shape (`n_components`,)
        The parameters of weight Dirichlet distribution

    mu_beta_ : array, shape (`n_components`, )
        The parameters of mean Gaussian distribution

    mu_m_ : array, shape (`n_components`, `n_features`)
        The parameters of mean Gaussian distribution

    lambda_nu_ : array, shape (`n_components`, )
        The parameters of precision Wishart distribution

    lambda_W_ : array,
        The parameters of precision Wishart distribution
        The shape depends on `precision_type`::
            (`n_components`, `n_features`, `n_features`)  if 'full'
            (`n_features`, `n_features`)                  if 'tied',
            (`n_components`, `n_features`)                if 'diag',
            (`n_components`, )                            if 'spherical',

    converged_ : bool
        True when convergence was reached in fit(), False
        otherwise.

    See Also
    --------
    """
    def __init__(self, n_components=1, precision_type='full',
                 random_state=None, tol=1e-6, reg_covar=0,
                 n_iter=100, n_init=1, params=None, init_params='kmeans',
                 verbose=0,
                 weight_alpha_prior=None,
                 mu_m_prior=None, mu_beta_prior=None,
                 lambda_nu_prior=None, lambda_W_prior=None):
        super(BayesianGaussianMixture, self).__init__(
            n_components, precision_type, random_state, tol, reg_covar,
            n_iter, n_init, params, init_params, verbose)
        self.precision_type = precision_type
        self.weight_alpha_prior = weight_alpha_prior
        self.mu_m_prior = mu_m_prior
        self.mu_beta_prior = mu_beta_prior
        self.lambda_nu_prior = lambda_nu_prior
        self.lambda_W_prior = lambda_W_prior

        self.weight_alpha_ = None
        self.mu_beta_, self.mu_m_ = None, None
        self.lambda_nu_, self.lambda_inv_W_ = None, None

        if precision_type not in ['spherical', 'tied', 'diag', 'full']:
                raise ValueError("Invalid value for 'precision_type': %s "
                                 "'precision_type' should be in "
                                 "['spherical', 'tied', 'diag', 'full']"
                                 % precision_type)

    def _check_weight_prior(self, weight_alpha_prior,
                            desired_shape):
        """Check weight_alpha_prior,
        the prior parameter  of the weight Dirichlet distribution

        Parameters
        ----------
        weight_alpha_prior : float

        desired_shape : tuple

        Returns
        -------
        weight_alpha_prior : float
        """
        # check shape
        check_shape(weight_alpha_prior, desired_shape, 'alpha')
        if weight_alpha_prior <= 0:
            raise ValueError("The parameter 'weight_alpha_prior' should be "
                             "greater than 0, but got %.5f" % weight_alpha_prior)
        return weight_alpha_prior

    def _check_initial_parameters(self):
        param_shape = _define_prior_shape(self.n_features, self.precision_type)
        if self.weight_alpha_prior is not None:
            self._check_weight_prior(self.weight_alpha_prior,
                                     param_shape['weight_alpha_prior'])

        if self.mu_beta_prior is not None and self.mu_m_prior is not None:
            _check_mu_prior(self.mu_m_prior, self.mu_beta_prior,
                            param_shape['mu_m_prior'],
                            param_shape['mu_beta_prior'])
        elif self.mu_beta_prior is None and self.mu_m_prior is None:
            pass
        else:
            raise ValueError("mu_beta_prior and mu_m_prior should be None or "
                             "not at the same time")

        if (self.lambda_nu_prior is not None and
                self.lambda_W_prior is not None):
            _check_lambda_prior(
                self.lambda_nu_prior, self.lambda_W_prior,
                param_shape['lambda_nu_prior'], param_shape['lambda_W_prior'],
                self.precision_type)
        elif self.lambda_nu_prior is None and self.lambda_W_prior is None:
            pass
        else:
            raise ValueError("lambda_nu_prior and lambda_W_prior "
                             "should be None or not at the same time")

    def _estimate_suffstat(self, X, resp):
        """Compute the sufficient statistics for Gaussian distribution
        """
        nk = resp.sum(axis=0)
        xk = estimate_Gaussian_suffstat_xk(resp, X, nk)
        Sk = estimate_Gaussian_suffstat_Sk(resp, X, nk, xk, self.reg_covar,
                                           self.precision_type)
        return nk, xk, Sk

    def _initialize_weight(self, nk):
        """Initialize the prior parameter of weight Dirichlet distribution
        """
        if self.weight_alpha_prior is None:
            # TODO discuss default value
            self.weight_alpha_prior = 1e-3 * np.array(self.n_components)
        self.weight_alpha_ = self._estimate_weights(nk)

    def _initialize_mu(self, X, nk, xk):
        """Initialize the prior parameter of mean Gaussian distribution
        """
        if self.mu_beta_prior is None:
            self.mu_beta_prior = 1
            # TODO discuss default value

        if self.mu_m_prior is None:
            self.mu_m_prior = X.mean(axis=0)
            # TODO discuss default value

        self.mu_beta_, self.mu_m_ = self._estimate_mu(nk, xk)

    def _initialize_lambda(self, X, nk, xk, Sk):
        """Initialize the prior parameter of precision Wishart or Gamma
         distribution.
        """
        if self.precision_type in ['full', 'tied']:
            self._initialize_lambda_full_tied(X, nk, xk, Sk)
        elif self.precision_type in ['diag', 'tied']:
            self._initialize_lambda_diag_spherical(X, nk, xk, Sk)

    def _initialize_lambda_full_tied(self, X, nk, xk, Sk):
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

        self.lambda_nu_, self.lambda_inv_W_ = self._estimate_lambda(nk, xk, Sk)

    def _initialize_lambda_diag_spherical(self, X, nk, xk, Sk):
        """Gamma distribution"""
        if self.lambda_nu_prior is None:
            self.lambda_nu_prior = .5
            # TODO discuss default value

        if self.lambda_W_prior is None:
            self.lambda_inv_W_prior = .5 * np.diag(np.cov(X.T, bias=1))
            if self.precision_type == 'spherical':
                self.lambda_inv_W_prior = self.lambda_inv_W_prior.mean()
            if any(self.lambda_inv_W_prior <= 0):
                raise ValueError("lambda_W_prior must be greater than 0")
            self.lambda_W_prior = 1./self.lambda_inv_W_prior
        else:
            if self.lambda_W_prior <= 0:
                raise ValueError("lambda_W_prior must be greater than 0")
            self.lambda_inv_W_prior = 1 / self.lambda_W_prior

        self.lambda_nu_, self.lambda_inv_W_ = self._estimate_lambda(nk, xk, Sk)


    def _initialize_weight_prior(self):
        self._log_dirichlet_norm_alpha_prior = \
            _log_dirichlet_norm(np.ones(self.n_components) *
                                self.weight_alpha_prior)

    def _initialize_parameters(self, X, resp):
        nk, xk, Sk = self._estimate_suffstat(X, resp)
        self._initialize_weight(nk)
        self._initialize_mu(X, nk, xk)
        self._initialize_lambda(X, nk, xk, Sk)

        self._initialize_weight_prior()
        self._log_gaussian_norm_beta_prior = \
            .5 * self.n_features * np.log(self.mu_beta_prior / (2 * np.pi))

        if self.precision_type in ['full', 'tied']:
            self._log_wishart_norm_W_nu_prior = \
                _log_wishart_norm(self.n_features, self.lambda_nu_prior,
                                  self.lambda_inv_W_prior)
        elif self.precision_type == 'diag':
            # lambda_inv_W_prior has n_feature Gamma distribution
            self._log_gamma_norm_W_nu_prior = \
                _log_gamma_norm_diag(self.lambda_nu_prior, self.lambda_inv_W_prior)
            # lambda_inv_W_prior has only 1 Gamma distribution
        elif self.precision_type == 'spherical':
                _log_gamma_norm_spherical(self.lambda_nu_prior, self.lambda_inv_W_prior)


    # m step
    def _estimate_weights(self, nk):
        return self.weight_alpha_prior + nk

    def _estimate_mu(self, nk, xk):
        mu_beta_ = self.mu_beta_prior + nk
        mu_m_ = (self.mu_beta_prior * self.mu_m_prior +
                 nk[:, np.newaxis] * xk) / mu_beta_[:, np.newaxis]
        return mu_beta_, mu_m_

    def _estimate_lambda_full(self, nk, xk, Sk):
        lambda_nu_ = self.lambda_nu_prior + nk
        lambda_inv_W_ = np.empty((self.n_components, self.n_features,
                                  self.n_features))
        for k in range(self.n_components):
            diff = xk[k] - self.mu_m_prior
            lambda_inv_W_[k] = (
                self.lambda_inv_W_prior + nk[k] * Sk[k] +
                (nk[k] * self.mu_beta_prior / self.mu_beta_[k]) *
                np.outer(diff, diff))
        return lambda_nu_, lambda_inv_W_

    def _estimate_lambda_tied(self, nk, xk, Sk):
        lambda_nu_ = self.lambda_nu_prior + nk.sum()/self.n_components
        lambda_inv_W_ = self.lambda_inv_W_prior + Sk * nk.sum() / self.n_components
        diff = xk - self.mu_m_prior
        lambda_inv_W_ += self.mu_beta_prior / self.n_components * (
            np.dot((nk / self.mu_beta_) * diff.T, diff))
        return lambda_nu_, lambda_inv_W_

    def _estimate_lambda_diag(self, nk, xk, Sk):
        lambda_nu_ = self.lambda_nu_prior + .5 * nk
        diff = xk - self.mu_beta_prior
        lambda_inv_W_ = self.lambda_inv_W_prior + .5 * (
            nk[:, np.newaxis] * Sk +
            (nk * self.mu_beta_prior / self.mu_beta_)[:, np.newaxis] *
            np.square(diff))
        return lambda_nu_, lambda_inv_W_


    def _estimate_lambda_spherical(self, nk, xk, Sk):
        pass

    def _estimate_lambda(self, nk, xk, Sk):
        estimate_lambda_functions = {
            "full": self._estimate_lambda_full,
            "tied": self._estimate_lambda_tied,
            "diag": self._estimate_lambda_diag,
            "spherical": self._estimate_lambda_spherical
        }
        return estimate_lambda_functions[self.covariance_type](nk, xk, Sk)

    def _m_step(self, X, resp):
        nk, xk, Sk = self._estimate_suffstat(X, resp)
        self.weight_alpha_ = self._estimate_weights(nk)
        self.mu_beta_, self.mu_m_ = self._estimate_mu(nk, xk)
        self.lambda_nu_, self.lambda_inv_W_ = self._estimate_lambda(
            nk, xk, Sk)

    # e step
    def _e_step(self, X):
        _, log_prob, resp = self._estimate_log_prob_resp(X)
        self._lower_bound = self._estimate_lower_bound(log_prob, resp)
        return self._lower_bound, resp

    def _estimate_log_weights(self):
        """Equation 3.42
        """
        # save for computing the lower bound
        self._log_pi = (digamma(self.weight_alpha_) -
                digamma(np.sum(self.weight_alpha_)))
        return self._log_pi

    def _estimate_log_prob(self, X):
        _estimate_log_BGaussian_prob = {
            "full": self._estimate_log_BGuassian_prob_full,
            "tied": self._estimate_log_BGuassian_prob_tied,
            "diag": self._estimate_log_BGuassian_prob_diag,
            "spherical": self._estimate_log_BGuassian_prob_spherical
        }
        log_prob = _estimate_log_BGaussian_prob[self.precision_type](X)
        return log_prob

    def _estimate_log_BGuassian_prob_full(self, X):
        # second item in Equation 3.8
        n_samples = X.shape[0]
        n_features = self.n_features
        ln_W_digamma = np.arange(1, self.n_features + 1)
        log_prob = np.empty((n_samples, self.n_components))
        log_lambda = np.empty((self.n_components, ))
        inv_W_chol = np.empty(self.lambda_inv_W_.shape)
        for k in range(self.n_components):
            try:
                inv_W_chol[k] = linalg.cholesky(self.lambda_inv_W_[k],
                                                lower=True)
            except linalg.LinAlgError:
                raise ValueError("'lambda_inv_W_' must be symmetric, "
                                 "positive-definite")
            log_inv_W_det = 2. * np.sum(np.log(np.diagonal(inv_W_chol[k])))
            log_lambda[k] = np.sum(digamma(.5 * (self.lambda_nu_[k] + 1 -
                                                 ln_W_digamma))) + \
                            n_features * np.log(2) - log_inv_W_det

            W_sol = linalg.solve_triangular(inv_W_chol[k], (X - self.mu_m_[k]).T,
                                            lower=True).T
            mahala_dist = np.sum(np.square(W_sol), axis=1)
            log_prob[:, k] = - .5 * (- log_lambda[k] +
                                     n_features / self.mu_beta_[k] +
                                     self.lambda_nu_[k] * mahala_dist)
        log_prob -= .5 * n_features * np.log(2 * np.pi)

        # save inv_W_chol, log_lambda for computing lower bound
        self._inv_W_chol = inv_W_chol
        self._log_lambda = log_lambda
        return log_prob

    def _estimate_log_BGuassian_prob_tied(self, X):
        n_samples = X.shape[0]
        n_features = self.n_features
        ln_W_digamma = np.arange(1, self.n_features + 1)
        log_prob = np.empty((n_samples, self.n_components))
        try:
            inv_W_chol = linalg.cholesky(self.lambda_inv_W_, lower=True)
        except linalg.LinAlgError:
            raise ValueError("'lambda_inv_W_' must be symmetric, "
                             "positive-definite")
        log_inv_W_det = 2 * np.sum(np.log(np.diagonal(inv_W_chol)))
        log_lambda = np.sum(digamma(.5 * (self.lambda_nu_ + 1 - ln_W_digamma))) + \
                     n_features * np.log(2) - log_inv_W_det
        for k in range(self.n_components):
            W_sol = linalg.solve_triangular(inv_W_chol, (X - self.mu_m_[k]).T,
                                            lower=True).T
            mahala_dist = np.sum(np.square(W_sol), axis=1)
            log_prob[:, k] = -.5 * (- log_lambda + n_features / self.mu_beta_[k] +
                                    self.lambda_nu_ * mahala_dist)
        log_prob -= .5 * n_features * np.log(2 * np.pi)
        self._inv_W_chol = inv_W_chol
        self._log_lambda = log_lambda
        return log_prob

    def _estimate_log_BGuassian_prob_diag(self, X):
        n_features = self.n_features

        log_lambda = n_features * digamma(self.lambda_nu_) - \
                     np.sum(np.log(self.lambda_inv_W_), axis=1)
        log_prob = -.5 * (
            -log_lambda +
            (n_features / self.mu_beta_ +
             self.lambda_nu_ * (np.sum((self.mu_m_ ** 2 / self.lambda_inv_W_), 1)
                                - 2 * np.dot(X, (self.mu_m_ / self.lambda_inv_W_).T)
                                + np.dot(X ** 2, (1. / self.lambda_inv_W_).T))))
        log_prob -= .5 * n_features * np.log(2 * np.pi)
        self._inv_W_chol = None
        self._log_lambda = log_lambda
        return log_prob


    def _estimate_log_BGuassian_prob_spherical(self, X):
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
    def _estimate_lower_bound(self, log_prob, resp):
        log_p_XZ = self._estimate_p_XZ(log_prob, resp)
        log_p_weight = self._estimate_p_weight()
        log_p_mu_lambda = self._estimate_p_mu_lambda()
        log_q_z = self._estimate_q_Z(resp)
        log_q_weight = self._estimate_q_weight()
        log_q_mu_lambda = self._estimate_q_mu_lambda()
        print log_p_XZ, log_p_weight, log_p_mu_lambda, log_q_z, \
            log_q_weight, log_q_mu_lambda
        return log_p_XZ + log_p_weight + log_p_mu_lambda + log_q_z + \
            log_q_weight + log_q_mu_lambda

    def _estimate_p_XZ(self, log_prob, resp):
        """Equation 7.5, 7.6
        """
        return np.sum(log_prob * resp)

    def _estimate_p_weight(self):
        """Equation 7.7
        """
        return self._log_dirichlet_norm_alpha_prior + \
            (self.weight_alpha_prior - 1) * np.sum(self._log_pi)

    def _estimate_p_mu_lambda(self):
        if self.precision_type == 'full':
            return self._estimate_p_mu_lambda_full()
        elif self.precision_type == 'tied':
            return self._estimate_p_mu_lambda_tied()
        elif self.precision_type == 'diag':
            return self._estimate_p_mu_lambda_diag()

    def _estimate_p_mu_lambda_full(self):
        """Equation 7.9
        """
        temp1 = self._log_lambda - \
            self.n_features * self.mu_beta_prior / self.mu_beta_
        mk_sol = np.empty(self.n_components)
        for k in range(self.n_components):
            sol = linalg.solve_triangular(
                self._inv_W_chol[k],
                (self.mu_m_[k] - self.mu_m_prior).T, lower=True).T
            mk_sol[k] = np.sum(np.square(sol))
        temp2 = self.mu_beta_prior * self.lambda_nu_ * mk_sol
        temp_mu = .5 * np.sum(temp1 - temp2) + \
            self._log_gaussian_norm_beta_prior

        temp3 = self.n_components * self._log_wishart_norm_W_nu_prior + \
            .5 * (self.lambda_nu_prior - self.n_features - 1) * \
            np.sum(self._log_lambda)

        trace_W0inv_Wk = np.empty(self.n_components)
        for k in range(self.n_components):
            # another way to compute trace_W0inv_Wk
            # sol = linalg.solve_triangular(
            #     self.inv_W_chol[k], self.lambda_inv_W_prior.T, lower=True)
            # trace_W0inv_Wk[k] = np.trace(linalg.solve_triangular(
            #     self.inv_W_chol[k], sol, lower=True, trans=1).T)
            chol_sol = linalg.inv(self._inv_W_chol[k])
            trace_W0inv_Wk[k] = np.sum(self.lambda_inv_W_prior.T *
                                       np.dot(chol_sol.T, chol_sol))
        temp4 = -.5 * np.sum(self.lambda_nu_ * trace_W0inv_Wk)
        return temp_mu + temp3 + temp4

    def _estimate_p_mu_lambda_tied(self):
        temp1 = self.n_components * self.mu_beta_prior / self.mu_beta_
        mk_sol = np.empty(self.n_components)
        for k in range(self.n_components):
            sol = linalg.solve_triangular(
                self._inv_W_chol,
                (self.mu_m_[k] - self.mu_m_prior).T, lower=True).T
            mk_sol[k] = np.sum(np.square(sol))
        temp2 = self.mu_beta_prior * self.lambda_nu_ * mk_sol
        temp_mu = .5 * (self.n_components * self._log_lambda +
                        np.sum(temp1 + temp2)) + \
                  self._log_gaussian_norm_beta_prior

        temp3 = self.n_components * self._log_wishart_norm_W_nu_prior + \
            .5 * (self.lambda_nu_prior - self.n_features - 1) * \
            self.n_components * self._log_lambda

        chol_sol = linalg.inv(self._inv_W_chol)
        trace_W0inv_W = np.sum(self.lambda_inv_W_prior.T *
                               np.dot(chol_sol.T, chol_sol))
        temp4 = -.5 * self.n_components * self.lambda_nu_ * trace_W0inv_W
        return temp_mu + temp3 + temp4

    def _estimate_p_mu_lambda_diag(self):
        temp1 = self._log_lambda - \
            self.n_features * self.mu_beta_prior / self.mu_beta_
        diff = self.mu_m_ - self.mu_m_prior
        temp2 = self.mu_beta_prior * self.lambda_nu_ * \
            np.sum(np.square(diff) / self.lambda_inv_W_, axis=1)
        temp_mu = .5 * np.sum(temp1 - temp2) + \
            self._log_gaussian_norm_beta_prior

        temp3 = self.n_components * self._log_gamma_norm_W_nu_prior + \
            (self.lambda_nu_prior - 1) * np.sum(self._log_lambda)
        temp4 = np.sum(- self.lambda_nu_ * np.sum(self.lambda_inv_W_prior / self.lambda_inv_W_, axis=1))
        return temp_mu + temp3 + temp4

    def _estimate_q_Z(self, resp):
        """Equation 7.10
        """
        resp = resp[resp > 10 * EPS]
        return np.sum(resp * np.log(resp))

    def _estimate_q_weight(self):
        """Equation 7.11
        """
        return np.sum((self.weight_alpha_ - 1) * self._log_pi) + \
            _log_dirichlet_norm(self.weight_alpha_)

    def _estimate_q_mu_lambda(self):
        if self.precision_type == 'full':
            return self._estimate_q_mu_lambda_full()
        elif self.precision_type == 'tied':
            return self._estimate_q_mu_lambda_tied()
        elif self.precision_type == 'diag':
            return self._estimate_q_mu_lambda_diag()

    def _estimate_q_mu_lambda_full(self):
        wishart_entropy = np.empty(self.n_components)
        for k in range(self.n_components):
            wishart_entropy[k] = _wishart_entropy(
                self.n_features, self.lambda_nu_[k], self.lambda_inv_W_[k],
                self._log_lambda[k])
        return np.sum(.5 * self._log_lambda +
                      .5 * self.n_features * np.log(self.mu_beta_ / (2 * np.pi))
                      - .5 * self.n_features
                      - wishart_entropy)

    def _estimate_q_mu_lambda_tied(self):
        wishart_entropy = _wishart_entropy(self.n_features, self.lambda_nu_,
                                           self.lambda_inv_W_, self._log_lambda)
        return (.5 * self.n_components * self._log_lambda
                + np.sum(.5 * self.n_features * np.log(self.mu_beta_ / (2 * np.pi)))
                - .5 * self.n_components * self.n_features
                - self.n_components * wishart_entropy
                )

    def _estimate_q_mu_lambda_diag(self):
        return np.sum(
            .5 * self._log_lambda +
            .5 * self.n_features * np.log(self.mu_beta_ / (2 * np.pi)) -
            .5 * self.n_features -
            _log_gamma_norm_diag(self.lambda_nu_, self.lambda_inv_W_)
        )


    def fit(self, X, y=None):
        """Estimate model parameters with the VB algorithm.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        Returns
        -------
        self
        """
        return self._fit(X)

    def _snapshot(self, X):
        """ for debug
        """
        alpha = self.weight_alpha_ / np.sum(self.weight_alpha_)
        m = self.mu_m_
        if self.precision_type == 'full':
            covar = self.lambda_inv_W_ / self.lambda_nu_[:, np.newaxis, np.newaxis]
        elif self.precision_type == 'tied':
            covar = self.lambda_inv_W_ / self.lambda_nu_
        elif self.precision_type == 'diag':
            covar = self.lambda_inv_W_ / self.lambda_nu_[:, np.newaxis]
        self._log_snapshot.append((alpha, m, covar,
                                   self.predict(X), self._lower_bound))

