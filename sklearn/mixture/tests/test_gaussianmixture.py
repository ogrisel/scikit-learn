import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.datasets.samples_generator import make_spd_matrix
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_raise_message
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_warns
from sklearn.utils.testing import ignore_warnings
from sklearn.utils.validation import check_random_state

from sklearn import mixture
from sklearn.mixture.gaussianmixture import _sufficient_Sk
from sklearn.covariance import EmpiricalCovariance


rng = np.random.RandomState(0)


COVARIANCE_TYPE = ['full', 'tied', 'diag', 'spherical']

def generate_data(n_samples, n_features, weights, means, covariances,
                  covariance_type):
    X = []
    if covariance_type == 'spherical':
        for k, (w, m, c) in enumerate(zip(weights, means,
                                          covariances['spherical'])):
            X.append(rng.multivariate_normal(m, c * np.eye(n_features),
                                             int(np.round(w * n_samples))))
    if covariance_type == 'diag':
        for k, (w, m, c) in enumerate(zip(weights, means,
                                          covariances['diag'])):
            X.append(rng.multivariate_normal(m, np.diag(c),
                                             int(np.round(w * n_samples))))
    if covariance_type == 'tied':
        for k, (w, m) in enumerate(zip(weights, means)):
            X.append(rng.multivariate_normal(m, covariances['tied'],
                                             int(np.round(w * n_samples))))
    if covariance_type == 'full':
        for k, (w, m, c) in enumerate(zip(weights, means,
                                          covariances['full'])):
            X.append(rng.multivariate_normal(m, c, int(np.round(w *
                                                                n_samples))))

    X = np.vstack(X)
    return X


class RandData(object):
    n_samples = 500
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
    Y = np.hstack([k * np.ones(np.round(w * n_samples)) for k, w in
                  enumerate(weights)])


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
    from sklearn.mixture.gaussianmixture import _check_X
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

    X = rng.rand(n_samples, n_features)
    assert_array_equal(X, _check_X(X, n_components, n_features))


def test__check_weights():
    from sklearn.mixture.gaussianmixture import _check_weights
    n_components = RandData.n_components

    weights_bad = rng.rand(n_components, 1)
    assert_raise_message(ValueError,
                         "The 'weights' should have the shape of "
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

    weights = RandData.weights
    assert_array_equal(weights, _check_weights(weights, n_components))


def test__check_means():
    from sklearn.mixture.gaussianmixture import _check_means
    n_components = RandData.n_components
    n_features = RandData.n_features

    means_bad = rng.rand(n_components + 1, n_features)
    assert_raise_message(ValueError,
                         "The 'means' should have shape (%s, %d), "
                         "but got %s"
                         % (n_components, n_features, str(means_bad.shape)),
                         _check_means, means_bad, n_components, n_features)

    means = RandData.means
    assert_array_equal(means, _check_means(means, n_components, n_features))


def test__check_covars():
    from sklearn.mixture.gaussianmixture import _check_covars
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
    # full
    covars_bad = rng.rand(n_components + 1, n_features, n_features)
    assert_raise_message(ValueError, "'full' covariances should have shape "
                         "(%d, %d, %d), but got %s"
                         % (n_components, n_features, n_features,
                            str(covars_bad.shape)),
                         _check_covars, covars_bad, n_components, n_features,
                         'full')
    covars_bad = rng.rand(n_components, n_features, n_features)
    covars_bad[0] = np.eye(n_features)
    covars_bad[0, 0, 0] = -1
    assert_raise_message(ValueError, "The component %d of 'full' covars "
                                     "should be symmetric, positive-definite"
                         % 0, _check_covars, covars_bad, n_components,
                         n_features, 'full')
    covars = RandData.covariances['full']
    assert_array_equal(covars, _check_covars(covars, n_components,
                                             n_features, 'full'))

    # tied
    covars_bad = rng.rand(n_features + 1, n_features + 1)
    assert_raise_message(ValueError, "'tied' covariances should have shape "
                         "(%d, %d), but got %s"
                         % (n_features, n_features, str(covars_bad.shape)),
                         _check_covars, covars_bad, n_components, n_features,
                         'tied')
    covars_bad = np.eye(n_features)
    covars_bad[0, 0] = -1
    assert_raise_message(ValueError, "'tied' covariance should be "
                         "symmetric, positive-definite",
                         _check_covars, covars_bad, n_components, n_features,
                         'tied')
    covars = RandData.covariances['tied']
    assert_array_equal(covars, _check_covars(covars, n_components,
                                             n_features, 'tied'))
    # diag
    covars_bad = rng.rand(n_components + 1, n_features)
    assert_raise_message(ValueError, "'diag' covariances should have shape "
                         "(%d, %d), but got %s"
                         % (n_components, n_features, str(covars_bad.shape)),
                         _check_covars, covars_bad, n_components, n_features,
                         'diag')
    covars_bad = np.ones((n_components, n_features)) * -1
    assert_raise_message(ValueError, "'diag' covariance should be positive",
                         _check_covars, covars_bad, n_components, n_features,
                         'diag')
    covars = RandData.covariances['diag']
    assert_array_equal(covars, _check_covars(covars, n_components,
                                             n_features, 'diag'))

    # spherical
    covars_bad = rng.rand(n_components + 1)
    assert_raise_message(ValueError, "'spherical' covariances should have "
                                     "shape (%d, ), but got %s"
                         % (n_components, str(covars_bad.shape)),
                         _check_covars, covars_bad, n_components, n_features,
                         'spherical')
    covars_bad = np.ones(n_components)
    covars_bad[0] = -1
    assert_raise_message(ValueError, "'spherical' covariance should be "
                                     "positive",
                         _check_covars, covars_bad, n_components, n_features,
                         'spherical')
    covars = RandData.covariances['spherical']
    assert_array_equal(covars, _check_covars(covars, n_components,
                                             n_features, 'spherical'))


def test__sufficient_Sk_full():
    # compare the EmpiricalCovariance.covariance fitted on X*sqrt(resp)
    # with _sufficient_sk_full
    n_samples = RandData.n_samples
    n_features = RandData.n_features
    n_components = RandData.n_components

    # special case 1, assuming data is "centered"
    X = rng.rand(n_samples, n_features)
    resp = rng.rand(n_samples, 1)
    X_resp = np.sqrt(resp) * X
    nk = np.array([n_samples])
    xk = np.zeros((1, n_features))
    covars_pred = _sufficient_Sk(resp, X, nk, xk, 0, 'full')
    ecov = EmpiricalCovariance(assume_centered=True)
    ecov.fit(X_resp)
    assert_almost_equal(ecov.error_norm(covars_pred[0], norm='frobenius'), 0)
    assert_almost_equal(ecov.error_norm(covars_pred[0], norm='spectral'), 0)

    # special case 2, assuming resp are all ones
    resp = np.ones((n_samples, 1))
    nk = np.array([n_samples])
    xk = X.mean().reshape((1, -1))
    covars_pred = _sufficient_Sk(resp, X, nk, xk, 0, 'full')
    ecov = EmpiricalCovariance(assume_centered=False)
    ecov.fit(X)
    assert_almost_equal(ecov.error_norm(covars_pred[0], norm='frobenius'), 0)
    assert_almost_equal(ecov.error_norm(covars_pred[0], norm='spectral'), 0)


def test__sufficient_Sk_tied():
    # use equation Nk * Sk / N = S_tied
    n_samples = RandData.n_samples
    n_features = RandData.n_features
    n_components = RandData.n_components

    resp = rng.rand(n_samples, n_components)
    resp = resp / resp.sum(axis=1)[:, np.newaxis]
    X = rng.rand(n_samples, n_features)
    nk = resp.sum(axis=0)
    xk = np.dot(resp.T, X) / nk[:, np.newaxis]
    covars_pred_full = _sufficient_Sk(resp, X, nk, xk, 0, 'full')
    covars_pred_full = np.sum(nk[:, np.newaxis, np.newaxis] * covars_pred_full,
                              0) / n_samples

    covars_pred_tied = _sufficient_Sk(resp, X, nk, xk, 0, 'tied')
    ecov = EmpiricalCovariance()
    ecov.covariance_ = covars_pred_full
    assert_almost_equal(ecov.error_norm(covars_pred_tied, norm='frobenius'), 0)
    assert_almost_equal(ecov.error_norm(covars_pred_tied, norm='spectral'), 0)


def test__sufficient_Sk_diag():
    # test against 'full' case
    n_samples = RandData.n_samples
    n_features = RandData.n_features
    n_components = RandData.n_components

    resp = rng.rand(n_samples, n_components)
    resp = resp / resp.sum(axis=1)[:, np.newaxis]
    X = rng.rand(n_samples, n_features)
    nk = resp.sum(axis=0)
    xk = np.dot(resp.T, X) / nk[:, np.newaxis]
    covars_pred_full = _sufficient_Sk(resp, X, nk, xk, 0, 'full')
    covars_pred_full = np.array([np.diag(np.diag(d)) for d in covars_pred_full])
    covars_pred_diag = _sufficient_Sk(resp, X, nk, xk, 0, 'diag')
    covars_pred_diag = np.array([np.diag(d) for d in covars_pred_diag])
    ecov = EmpiricalCovariance()
    for (cov_full, cov_diag) in zip(covars_pred_full, covars_pred_diag):
        ecov.covariance_ = cov_full
        assert_almost_equal(ecov.error_norm(cov_diag, norm='frobenius'), 0)
        assert_almost_equal(ecov.error_norm(cov_diag, norm='spectral'), 0)

def test__sufficient_Sk_spherical():
    # computing spherical covariance equals to the variance of one-dimension
    # data after flattening
    n_samples = RandData.n_samples
    n_features = RandData.n_features
    n_components = 1

    X = rng.rand(n_samples, n_features)
    X = X - X.mean()
    resp = np.ones((n_samples, 1))
    nk = np.array([n_samples])
    xk = X.mean()
    covars_pred_spherical = _sufficient_Sk(resp, X, nk, xk, 0, 'spherical')
    covars_pred_spherical2 = np.dot(X.flatten().T, X.flatten()) / (n_features
                                                                   * n_samples)
    assert_almost_equal(covars_pred_spherical, covars_pred_spherical2)
