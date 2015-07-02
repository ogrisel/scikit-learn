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


def _define_parameter_shape(n_components, n_features, covariance_type):
    cov_shape = {'full': (n_components, n_features, n_features),
                 'tied': (n_features, n_features),
                 'diag': (n_components, n_features),
                 'spherical': (n_components, )}
    param_shape = {'weights': (n_components, ),
                   'means': (n_components, n_features),
                   'covariances': cov_shape[covariance_type]}
    return param_shape


def check_shape(param, param_shape, name):
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
    def _m_step(self, responsibilities, nk, xk, Sk):
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
    def _estimate_responsibilities(self, X):
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
        X = _check_X(X, self.n_components, self.means_.shape[1])

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
                current_log_likelihood = self.score(X)
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
        X = _check_X(X, self.n_components, self.means_.shape[1])
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
        X = _check_X(X, self.n_components, self.means_.shape[1])
        return self._estimate_responsibilities(X)

    def sample(self):
        pass

    def aic(self, X):
        pass

    def bic(self, X):
        pass


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
            if self.verbose > 1:
                print_('\n\tWeights are initialized.', end='')
        else:
            self.weights_ = self.weights_init
            if self.verbose > 1:
                print_('\n\tWeights are provided.', end='')

        if self.means_init is None:
            self.means_ = self._estimate_means(responsibilities,
                                               nk, xk, Sk)
            if self.verbose > 1:
                print_('\n\tMeans are initialized.', end='')
        else:
            self.means_ = self.means_init
            if self.verbose > 1:
                print_('\n\tMeans are provided.', end='')

        if self.covars_init is None:
            self.covars_ = self._estimate_covariances(responsibilities,
                                                      nk, xk, Sk)
            if self.verbose > 1:
                print_('\n\tCovariances are initialized.', end='')
        else:
            self.covars_ = self.covars_init
            if self.verbose > 1:
                print_('\n\tCovariances are provided.', end='')

    # e-step functions
    def _estimate_log_weights(self):
        return np.log(self.weights_)

    def _estimate_log_rho_full(self, X):
        n_samples, n_features = X.shape
        log_prob = np.empty((n_samples, self.n_components))
        for k, (mu, cov) in enumerate(zip(self.means_,  self.covars_)):
            try:
                cov_chol = linalg.cholesky(cov, lower=True)
            except:
                raise ValueError("'covars' must be symmetric, "
                                 "positive-definite")
            cv_log_det = 2. * np.sum(np.log(np.diagonal(cov_chol)))
            cv_sol = linalg.solve_triangular(cov_chol, (X - mu).T,
                                             lower=True).T
            log_prob[:, k] = - .5 * (n_features * np.log(2 * np.pi)
                                     + cv_log_det
                                     + np.sum(np.square(cv_sol), axis=1))
        return log_prob + self._estimate_log_weights()

    def _estimate_log_rho_tied(self, X):
        n_samples, n_features = X.shape
        log_prob = np.empty((n_samples, self.n_components))
        try:
            cov_chol = linalg.cholesky(self.covars_, lower=True)
        except:
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

    def _estimate_responsibilities(self, X):
        _, resp = self._estimate_log_probabilities_responsibilities(X)
        return resp

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
        weighted_log_probabilities = self._estimate_weighted_log_probabilities(X)
        log_probabilities = logsumexp(weighted_log_probabilities, axis=1)
        with np.errstate(under='ignore'):
            # ignore underflow
            responsibilities = np.exp(weighted_log_probabilities -
                                      log_probabilities[:, np.newaxis])
        return log_probabilities, responsibilities

    def _e_step(self, X):
        log_prob, resp = self._estimate_log_probabilities_responsibilities(X)
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

    def _estimate_covariances(self, responsibilities, nk, xk, Sk):
        estimate_covariances_functions = {
            "full": self._estimate_covariance_full,
            "tied": self._estimate_covariance_tied,
            "diag": self._estimate_covariance_diag,
            "spherical": self._estimate_covariances_spherical
        }
        return estimate_covariances_functions[self.covariance_type](
            responsibilities, nk, xk, Sk)

    def _m_step(self, responsibilities, nk, xk, Sk):
        self.weights_ = self._estimate_weights(responsibilities, nk, xk, Sk)
        self.means_ = self._estimate_means(responsibilities, nk, xk, Sk)
        self.covars_ = self._estimate_covariances(responsibilities, nk, xk, Sk)

    def _check_is_fitted(self):
        check_is_fitted(self, 'weights_')
        check_is_fitted(self, 'means_')
        check_is_fitted(self, 'covars_')

    def _get_parameters(self):
        return self.weights_, self.means_, self.covars_

    def _set_parameters(self, params):
        self.weights_, self.means_, self.covars_ = params
