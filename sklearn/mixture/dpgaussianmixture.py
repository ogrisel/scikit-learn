import numpy as np
from scipy.special import digamma, gammaln

from time import time

from ..utils.validation import check_is_fitted
from sklearn.externals.six import print_
from .bayesianmixture import BayesianGaussianMixture, check_shape

def _log_beta_norm(a, b):
    """The log of the normalization of Beta distribution
    """
    return gammaln(a+b) - gammaln(a) - gammaln(b)


class DirichletProcessGaussianMixture(BayesianGaussianMixture):
    def __init__(self, n_components=1, precision_type='full',
                 random_state=None, tol=1e-6, reg_covar=0,
                 n_iter=100, n_init=1, params=None, init_params='kmeans',
                 verbose=0,
                 weight_gamma_prior=None,
                 mu_m_prior=None, mu_beta_prior=None,
                 lambda_nu_prior=None, lambda_W_prior=None):
        super(DirichletProcessGaussianMixture, self).__init__(
            n_components, precision_type, random_state, tol, reg_covar,
            n_iter, n_init, params, init_params, verbose,
            weight_gamma_prior, mu_m_prior, mu_beta_prior, lambda_nu_prior,
            lambda_W_prior)
        self.weight_gamma_prior = weight_gamma_prior
        self.weight_gamma_a_ = None
        self.weight_gamma_b_ = None
        self.mu_beta_, self.mu_m_ = None, None
        self.lambda_nu_, self.lambda_inv_W_ = None, None

        if precision_type not in ['spherical', 'tied', 'diag', 'full']:
                raise ValueError("Invalid value for 'precision_type': %s "
                                 "'precision_type' should be in "
                                 "['spherical', 'tied', 'diag', 'full']"
                                 % precision_type)

    def _check_weight_prior(self, weight_gamma_prior, desired_shape):
        """Check weight_gamma_prior,
        the prior parameter  of the weight Beta distribution

        Parameters
        ----------
        weight_gamma_prior : float

        desired_shape : tuple

        Returns
        -------
        weight_gamma_prior : float
        """
        # check shape
        check_shape(weight_gamma_prior, desired_shape, 'alpha')
        if weight_gamma_prior <= 0:
            raise ValueError("The parameter 'weight_gamma_prior' should be "
                             "greater than 0, but got %.5f"
                             % weight_gamma_prior)
        return weight_gamma_prior

    def _initialize_weight(self, nk):
        """Initialize the prior parameter of weight Dirichlet distribution
        """
        if self.weight_gamma_prior is None:
            # TODO discuss default value
            self.weight_gamma_prior = 0.1
        self.weight_gamma_a_, self.weight_gamma_b_ = self._estimate_weights(nk)

    def _initialize_weight_prior(self):
        self._log_beta_norm_gamma_prior = \
            _log_beta_norm(1, self.weight_gamma_prior)

    def _estimate_weights(self, nk):
        tmp = np.hstack((np.cumsum(nk[::-1])[-2::-1], 0))
        return 1 + nk, self.weight_gamma_prior + tmp

    def _m_step(self, X, resp):
        nk, xk, Sk = self._estimate_suffstat(X, resp)
        self.weight_gamma_a_, self.weight_gamma_b_ = self._estimate_weights(nk)
        self.mu_beta_, self.mu_m_ = self._estimate_mu(nk, xk)
        self.lambda_nu_, self.lambda_inv_W_ = self._estimate_lambda(nk, xk, Sk)

    def _estimate_log_weights(self):
        """Equation 9.22
        """
        digamma_sum = digamma(self.weight_gamma_a_ + self.weight_gamma_b_)
        digamma_a = digamma(self.weight_gamma_a_)
        digamma_b = digamma(self.weight_gamma_b_)
        self._log_pi_a = np.sum(digamma_a - digamma_sum)
        self._log_pi_b = np.sum(digamma_b - digamma_sum)
        return digamma_a - digamma_sum + np.hstack((0, np.cumsum(digamma_b - digamma_sum)[:-1]))

    def _estimate_p_weight(self):
        """Equation 9.31
        """
        return self.n_components * self._log_beta_norm_gamma_prior + \
               (self.weight_gamma_prior - 1) * self._log_pi_b

    def _estimate_q_weight(self):
        """Equation 9.32
        """
        log_norm = 0
        for k in range(self.n_components):
            log_norm += _log_beta_norm(self.weight_gamma_a_[k],
                                       self.weight_gamma_b_[k])
        return log_norm + np.sum((self.weight_gamma_a_ - 1) * self._log_pi_a +
                                 (self.weight_gamma_b_ - 1) * self._log_pi_b)

    def _check_is_fitted(self):
        check_is_fitted(self, 'weight_gamma_a_')
        check_is_fitted(self, 'weight_gamma_b_')
        check_is_fitted(self, 'mu_beta_')
        check_is_fitted(self, 'mu_m_')
        check_is_fitted(self, 'lambda_nu_')
        check_is_fitted(self, 'lambda_inv_W_')

    def _snapshot(self, X):
        """ for debug
        """
        tmp1 = self.weight_gamma_a_ / (self.weight_gamma_a_ + self.weight_gamma_b_)
        tmp2 = self.weight_gamma_b_ / (self.weight_gamma_a_ + self.weight_gamma_b_)
        log_pi = tmp1 * np.hstack((1, np.cumprod(tmp2[:-1])))

        log_m = self.mu_m_
        log_covar = self.lambda_inv_W_ / self.lambda_nu_[:, np.newaxis, np.newaxis]
        self._log_snapshot.append((log_pi, log_m, log_covar,
                                   self.predict(X), self._lower_bound))