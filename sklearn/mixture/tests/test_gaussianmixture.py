import sys
import numpy as np
from scipy import stats

from sklearn.datasets.samples_generator import make_spd_matrix
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_raise_message
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_allclose
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import assert_warns_message
from sklearn.utils import ConvergenceWarning

from sklearn import mixture
from sklearn.mixture.base import _sufficient_Sk
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.externals.six.moves import cStringIO as StringIO


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
    means = rng.rand(n_components, n_features) * 50
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


def test_check_X():
    from sklearn.mixture.base import _check_X
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


def test_check_weights():
    from sklearn.mixture.base import _check_weights
    n_components = RandData.n_components

    weights_bad = rng.rand(n_components, 1)

    assert_raise_message(ValueError,
                         "The parameter 'weights' should have the shape of "
                         "(%d,), "
                         "but got %s" % (n_components, str(weights_bad.shape)),
                         _check_weights, weights_bad, (n_components, ))

    weights_bad = rng.rand(n_components) + 1
    assert_raise_message(ValueError,
                         "The parameter 'weights' should be in the range "
                         "[0, 1], but got max value %.5f, min value %.5f"
                         % (np.min(weights_bad), np.max(weights_bad)),
                         _check_weights, weights_bad, (n_components, ))

    weights_bad = rng.rand(n_components)
    weights_bad = weights_bad/(weights_bad.sum() + 1)
    assert_raise_message(ValueError,
                         "The parameter 'weights' should be normalized, "
                         "but got sum(weights) = %.5f" % np.sum(weights_bad),
                         _check_weights, weights_bad, (n_components, ))

    weights = RandData.weights
    assert_array_equal(weights, _check_weights(weights, (n_components, )))


def test_check_means():
    from sklearn.mixture.base import _check_means
    n_components = RandData.n_components
    n_features = RandData.n_features

    means_bad = rng.rand(n_components + 1, n_features)
    assert_raise_message(ValueError,
                         "The parameter 'means' should have the shape of "
                         "(%s, %d), but got %s"
                         % (n_components, n_features, str(means_bad.shape)),
                         _check_means, means_bad, (n_components, n_features))

    means = RandData.means
    assert_array_equal(means, _check_means(means, (n_components, n_features)))


def test_check_covars():
    from sklearn.mixture.base import (_check_covars,
                                                 _define_parameter_shape)
    # TODO integrate testing
    n_components = RandData.n_components
    n_features = RandData.n_features

    # full
    covars_bad = rng.rand(n_components + 1, n_features, n_features)
    desired_shape = _define_parameter_shape(n_components, n_features, 'full')
    desired_shape = desired_shape['covariances']
    assert_raise_message(
        ValueError,
        "The parameter 'full covariance' should have the "
        "shape of (%d, %d, %d), but got %s"
        % (n_components, n_features, n_features, str(covars_bad.shape)),
        _check_covars, covars_bad, desired_shape, 'full')
    covars_bad = rng.rand(n_components, n_features, n_features)
    covars_bad[0] = np.eye(n_features)
    covars_bad[0, 0, 0] = -1
    assert_raise_message(
        ValueError,
        "The component %d of 'full covariance' "
        "should be symmetric, positive-definite"
        % 0, _check_covars, covars_bad, desired_shape, 'full')
    covars = RandData.covariances['full']
    assert_array_equal(covars, _check_covars(covars, desired_shape, 'full'))

    # tied
    covars_bad = rng.rand(n_features + 1, n_features + 1)
    desired_shape = _define_parameter_shape(n_components, n_features, 'tied')
    desired_shape = desired_shape['covariances']
    assert_raise_message(
        ValueError,
        "The parameter 'tied covariance' should have the shape of "
        "(%d, %d), but got %s"
        % (n_features, n_features, str(covars_bad.shape)),
        _check_covars, covars_bad, desired_shape, 'tied')
    covars_bad = np.eye(n_features)
    covars_bad[0, 0] = -1
    assert_raise_message(ValueError, "'tied covariance' should be "
                         "symmetric, positive-definite",
                         _check_covars, covars_bad, desired_shape, 'tied')
    covars = RandData.covariances['tied']
    assert_array_equal(covars, _check_covars(covars, desired_shape, 'tied'))

    # diag
    covars_bad = rng.rand(n_components + 1, n_features)
    desired_shape = _define_parameter_shape(n_components, n_features, 'diag')
    desired_shape = desired_shape['covariances']
    assert_raise_message(
        ValueError,
        "The parameter 'diag covariance' should have the shape of "
        "(%d, %d), but got %s"
        % (n_components, n_features, str(covars_bad.shape)),
        _check_covars, covars_bad, desired_shape, 'diag')
    covars_bad = np.ones((n_components, n_features)) * -1
    assert_raise_message(ValueError, "'diag covariance' should be positive",
                         _check_covars, covars_bad, desired_shape, 'diag')
    covars = RandData.covariances['diag']
    assert_array_equal(covars, _check_covars(covars, desired_shape, 'diag'))

    # spherical
    covars_bad = rng.rand(n_components + 1)
    desired_shape = _define_parameter_shape(n_components, n_features,
                                           'spherical')
    desired_shape = desired_shape['covariances']
    assert_raise_message(
        ValueError,
        "The parameter 'spherical covariance' should have the "
        "shape of (%d,), but got %s"
        % (n_components, str(covars_bad.shape)),
        _check_covars, covars_bad, desired_shape, 'spherical')
    covars_bad = np.ones(n_components)
    covars_bad[0] = -1
    assert_raise_message(ValueError, "'spherical covariance' should be "
                                     "positive",
                         _check_covars, covars_bad, desired_shape, 'spherical')
    covars = RandData.covariances['spherical']
    assert_array_equal(covars, _check_covars(covars, desired_shape,
                                             'spherical'))


def test__sufficient_Sk_full():
    # compare the EmpiricalCovariance.covariance fitted on X*sqrt(resp)
    # with _sufficient_sk_full, n_components=1
    n_samples = RandData.n_samples
    n_features = RandData.n_features

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


def test_sufficient_Sk_diag():
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


def test_sufficient_Sk_spherical():
    # computing spherical covariance equals to the variance of one-dimension
    # data after flattening, n_components=1
    n_samples = RandData.n_samples
    n_features = RandData.n_features

    X = rng.rand(n_samples, n_features)
    X = X - X.mean()
    resp = np.ones((n_samples, 1))
    nk = np.array([n_samples])
    xk = X.mean()
    covars_pred_spherical = _sufficient_Sk(resp, X, nk, xk, 0, 'spherical')
    covars_pred_spherical2 = np.dot(X.flatten().T, X.flatten()) / (n_features
                                                                   * n_samples)
    assert_almost_equal(covars_pred_spherical, covars_pred_spherical2)


def _naive_lmvnpdf_diag(X, weights, means, covars):
    resp = np.empty((len(X), len(means)))
    stds = np.sqrt(covars)
    for i, (mean, std) in enumerate(zip(means, stds)):
        resp[:, i] = stats.norm.logpdf(X, mean, std).sum(axis=1)
    return resp + np.log(weights)


def test_GaussianMixture_log_probabilities():
    # test aginst with _naive_lmvnpdf_diag
    n_samples = RandData.n_samples
    n_features = RandData.n_features
    n_components = RandData.n_components

    weights = RandData.weights
    means = RandData.means
    covars_diag = rng.rand(n_components, n_features)
    X = rng.rand(n_samples, n_features)
    log_prob_naive = _naive_lmvnpdf_diag(X, weights, means, covars_diag)

    # full covariances
    covars_full = np.array([np.diag(x) for x in covars_diag])
    g = mixture.GaussianMixture(n_components=n_components,
                                weights=weights,
                                means=means, covars=covars_full,
                                random_state=rng, covariance_type='full')
    g._initialize(X)
    log_prob = g._estimate_log_rho_full(X)
    assert_array_almost_equal(log_prob, log_prob_naive)

    # diag covariances
    g = mixture.GaussianMixture(n_components=n_components,
                                weights=weights,
                                means=means, covars=covars_diag,
                                random_state=rng, covariance_type='diag')
    g._initialize(X)
    log_prob = g._estimate_log_rho_diag(X)
    assert_array_almost_equal(log_prob, log_prob_naive)

    # tied
    covars_tied = covars_full.mean(axis=0)
    g = mixture.GaussianMixture(n_components=n_components,
                                weights=weights,
                                means=means, covars=covars_tied,
                                random_state=rng, covariance_type='tied')
    g._initialize(X)
    log_prob_naive = _naive_lmvnpdf_diag(X, weights, means,
                                         [np.diag(covars_tied)] * n_components)
    log_prob = g._estimate_log_rho_tied(X)
    assert_array_almost_equal(log_prob, log_prob_naive)

    # spherical
    covars_spherical = covars_diag.mean(axis=1)
    g = mixture.GaussianMixture(n_components=n_components,
                                weights=weights,
                                means=means, covars=covars_spherical,
                                random_state=rng, covariance_type='spherical')
    g._initialize(X)
    log_prob_naive = _naive_lmvnpdf_diag(X, weights, means,
                                         [[k] * n_features for k in
                                          covars_spherical])
    log_prob = g._estimate_log_rho_spherical(X)
    assert_array_almost_equal(log_prob, log_prob_naive)

# skip tests on weighted_log_probabilities, log_weights

def test_GaussianMixture__estimate_log_probabilities_responsibilities():
    # test whether responsibilities are normalized
    n_samples = RandData.n_samples
    n_features = RandData.n_features
    n_components = RandData.n_components

    X = rng.rand(n_samples, n_features)
    for cov_type in COVARIANCE_TYPE:
        weights = RandData.weights
        means = RandData.means
        covariances = RandData.covariances[cov_type]
        g = mixture.GaussianMixture(n_components=n_components,
                                    random_state=rng,
                                    weights=weights,
                                    means=means, covars=covariances,
                                    covariance_type=cov_type)
        g._initialize(X)
        _, resp = g._estimate_log_probabilities_responsibilities(X)
        assert_array_almost_equal(resp.sum(axis=1), np.ones(n_samples))


def test_GaussianMixture_predict_predict_proba():
    for cov_type in COVARIANCE_TYPE:
        X = RandData.X[cov_type]
        Y = RandData.Y
        g = mixture.GaussianMixture(n_components=RandData.n_components,
                                    random_state=rng,
                                    weights=RandData.weights,
                                    means=RandData.means,
                                    covars=RandData.covariances[cov_type],
                                    covariance_type=cov_type)
        g._initialize(X)
        Y_pred = g.predict(X)
        Y_pred_proba = g.predict_proba(X).argmax(axis=1)
        assert_array_equal(Y_pred, Y_pred_proba)
        assert_greater(adjusted_rand_score(Y, Y_pred), .95)


def test_GaussianMixture_fit():
    # recover the ground truth
    n_features = RandData.n_features
    n_components = RandData.n_components

    for cov_type in COVARIANCE_TYPE:
        X = RandData.X[cov_type]
        g = mixture.GaussianMixture(n_components=n_components, n_init=20,
                                    n_iter=100, min_covar=0,
                                    random_state=rng,
                                    covariance_type=cov_type)
        g.fit(X)
        # needs more data to achieve rtol=1e-7
        assert_allclose(np.sort(g.weights_), np.sort(RandData.weights),
                        rtol=0.1, atol=1e-2)

        arg_idx1 = g.means_[:, 0].argsort()
        arg_idx2 = RandData.means[:, 0].argsort()
        assert_allclose(g.means_[arg_idx1], RandData.means[arg_idx2],
                        rtol=0.1, atol=1e-2)

        if cov_type == 'spherical':
            cov_pred = np.array([np.eye(n_features) * c for c in g.covars_])
            cov_test = np.array([np.eye(n_features) * c for c in
                                 RandData.covariances['spherical']])
        elif cov_type == 'diag':
            cov_pred = np.array([np.diag(d) for d in g.covars_])
            cov_test = np.array([np.diag(d) for d in
                                 RandData.covariances['diag']])
        elif cov_type == 'tied':
            cov_pred = np.array([g.covars_] * n_components)
            cov_test = np.array([RandData.covariances['tied']] * n_components)
        elif cov_type == 'full':
            cov_pred = g.covars_
            cov_test = RandData.covariances['full']
        arg_idx1 = np.trace(cov_pred, axis1=1, axis2=2).argsort()
        arg_idx2 = np.trace(cov_test, axis1=1, axis2=2).argsort()
        for k, h in zip(arg_idx1, arg_idx2):
            ecov = EmpiricalCovariance()
            ecov.covariance_ = cov_test[h]
            # the accuracy depends on the number of data and randomness, rng
            assert_allclose(ecov.error_norm(cov_pred[k]), 0, atol=0.1)


def test_GaussianMixture_fit_best_params():
    n_components = RandData.n_components
    n_init = 10
    for cov_type in COVARIANCE_TYPE:
        X = RandData.X[cov_type]
        g = mixture.GaussianMixture(n_components=n_components, n_init=1,
                                    n_iter=100, min_covar=0, random_state=rng,
                                    covariance_type=cov_type)
        ll = []
        for _ in range(n_init):
            g.fit(X)
            ll.append(g.score(X))
        ll = np.array(ll)
        g_best = mixture.GaussianMixture(n_components=n_components,
                                         n_init=n_init, n_iter=100,
                                         min_covar=0, random_state=rng,
                                         covariance_type=cov_type)
        g_best.fit(X)
        assert_almost_equal(ll.min(), g_best.score(X))


def test_GaussianMixture_fit_convergence_warning():
    n_components = RandData.n_components
    n_iter = 1
    for cov_type in COVARIANCE_TYPE:
        X = RandData.X[cov_type]
        g = mixture.GaussianMixture(n_components=n_components, n_init=1,
                                    n_iter=n_iter,
                                    min_covar=0, random_state=rng,
                                    covariance_type=cov_type)
        assert_warns_message(ConvergenceWarning,
                             'Initialization %d is not converged. '
                             'Try different init parameters, '
                             'or increase n_init, '
                             'or check for degenerate data.'
                             % n_iter, g.fit, X)


def test_GaussianMixture_verbose():
    n_components = RandData.n_components
    for cov_type in COVARIANCE_TYPE:
        X = RandData.X[cov_type]
        g = mixture.GaussianMixture(n_components=n_components, n_init=1,
                                    n_iter=100, min_covar=1, random_state=rng,
                                    covariance_type=cov_type,
                                    verbose=1)
        h = mixture.GaussianMixture(n_components=n_components, n_init=1,
                                    n_iter=100, min_covar=1, random_state=rng,
                                    covariance_type=cov_type,
                                    verbose=2)
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            g.fit(X)
            h.fit(X)
        finally:
            sys.stdout = old_stdout
