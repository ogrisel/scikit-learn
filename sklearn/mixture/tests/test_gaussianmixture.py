import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.datasets.samples_generator import make_spd_matrix
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_raise_message
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_warns
from sklearn.utils.testing import ignore_warnings
from sklearn.utils.validation import check_random_state

from sklearn import mixture
from sklearn.mixture.gaussianmixture import _check_X, _check_weights
from sklearn.mixture.gaussianmixture import _check_means, _check_covars

rng = np.random.RandomState(0)


COVARIANCE_TYPE = ['full', 'tied', 'diag', 'spherical']

def generate_data(n_samples, n_features, weights, means, covariances,
                  covariance_type):
    X = []
    if covariance_type == 'spherical':
        for k, (w, m, c) in enumerate(zip(weights, means,
                                          covariances['spherical'])):
            X.append(rng.multivariate_normal(m, c*np.eye(n_features),
                                             int(np.round(w*n_samples))))
    if covariance_type == 'diag':
        for k, (w, m, c) in enumerate(zip(weights, means,
                                          covariances['diag'])):
            X.append(rng.multivariate_normal(m, np.diag(c),
                                             int(np.round(w*n_samples))))
    if covariance_type == 'tied':
        for k, (w, m) in enumerate(zip(weights, means)):
            X.append(rng.multivariate_normal(m, covariances['tied'],
                                             int(np.round(w*n_samples))))
    if covariance_type == 'full':
        for k, (w, m, c) in enumerate(zip(weights, means,
                                          covariances['full'])):
            X.append(rng.multivariate_normal(m, c, int(np.round(w*n_samples))))
    X = np.vstack(X)
    return X


class RandData(object):
    n_samples = 400
    n_components = 2
    n_features = 2
    weights = rng.rand(n_components)
    weights = weights / weights.sum()
    means = rng.randint(-10, 10, (n_components, n_features))
    tol = 1e-6
    covariances = {'spherical': .5 + rng.rand(n_components),
                   'diag': (.5 + rng.rand(n_components, n_features)) ** 2,
                   'tied': make_spd_matrix(n_features, random_state=rng),
                   'full': np.array([make_spd_matrix(n_features,
                                                     random_state=rng)*.5
                                     for _ in range(n_components)])}
    X = dict(zip(COVARIANCE_TYPE, [generate_data(n_samples, n_features,
                                                 weights, means, covariances,
                                                 cov_type) for cov_type in
                                   COVARIANCE_TYPE]))


def test_GaussianMixture_parameters():
    # test bad parameters

    # n_init should be greater than 0
    n_init = rng.randint(-10, 1)
    assert_raise_message(ValueError,
                         "Invalid value for 'n_init': %d "
                         "Estimation requires at least one run"
                         % n_init,
                         mixture.GaussianMixture, n_init=n_init)

    # covariance_type should be in [spherical, diag, tied, full]
    covariance_type = 'bad_covariance_type'
    assert_raise_message(ValueError,
                         "Invalid value for 'covariance_type': %s "
                         "'covariance_type' should be in "
                         "['spherical', 'tied', 'diag', 'full']"
                         % covariance_type,
                         mixture.GaussianMixture,
                         covariance_type=covariance_type)

def test__check_X():
    n_samples = RandData.n_samples
    n_components = RandData.n_components
    n_features = RandData.n_features

    X_bad_dim = rng.rand(n_samples)
    assert_raise_message(ValueError,
                         'Expected the input data X have 2 dimensions, '
                         'but got %d dimension(s)' % X_bad_dim.ndim,
                         _check_X, X_bad_dim, n_components)

    X_bad_dim = rng.rand(n_components - 1, n_features)
    assert_raise_message(ValueError,
                         'Expected n_samples >= n_components'
                         'but got n_components = %d, n_samples = %d'
                         % (n_components, X_bad_dim.shape[0]),
                         _check_X, X_bad_dim, n_components)

    X_bad_dim = rng.rand(n_components, n_features + 1)
    assert_raise_message(ValueError,
                         'Expected the input data X have %d features, '
                         'but got %d features'
                         % (n_features, X_bad_dim.shape[1]),
                         _check_X, X_bad_dim, n_components, n_features)

def test__check_weights():
    n_components = RandData.n_components

    weights_bad = rng.rand(n_components, 1)
    assert_raise_message(ValueError,
                         "Expected the 'weights' have the shape of "
                         "(n_components, ), "
                         "but got %s" % str(weights_bad.shape),
                         _check_weights, weights_bad, n_components)

    weights_bad = rng.rand(n_components) + 1
    assert_raise_message(ValueError,
                         "The 'weights' should be in the range [0, 1]",
                         _check_weights, weights_bad, n_components)

    weights_bad = rng.rand(n_components)
    weights_bad = weights_bad/(weights_bad.sum() + 1)
    assert_raise_message(ValueError,
                         "The 'weights' should be normalized",
                         _check_weights, weights_bad, n_components)

def test__check_means():
    n_components = RandData.n_components
    n_features = RandData.n_features

    means_bad = rng.rand(n_components + 1, n_features)
    assert_raise_message(ValueError,
                         "The 'means' should have shape (%s, %d), "
                         "but got %s"
                         % (n_components, n_features, str(means_bad.shape)),
                         _check_means, means_bad, n_components, n_features)

def test__check_covars():
    n_samples = RandData.n_samples
    n_components = RandData.n_components
    n_features = RandData.n_features

    covars_type_bad = 'bad type'
    covars_any = rng.rand(n_components)
    assert_raise_message(ValueError, "Invalid value for 'covariance_type': %s "
                         "'covariance_type' should be in "
                         "['spherical', 'tied', 'diag', 'full']"
                         % covars_type_bad,
                         _check_covars, covars_any, n_components, n_features,
                         covars_type_bad)
    # TODO four kinds of covariances
