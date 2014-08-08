"""
Testing for Extreme Learning Machines module (sklearn.neural_network)
"""

# Author: Issam H. Laradji
# Licence: BSD 3 clause

import sys

import numpy as np

from numpy.testing import assert_almost_equal, assert_array_equal
from scipy.sparse import csr_matrix
from itertools import product

from sklearn import cross_validation
from sklearn.datasets import load_digits, load_boston
from sklearn.datasets import make_regression, make_multilabel_classification
from sklearn.externals.six.moves import cStringIO as StringIO
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import ELMClassifier
from sklearn.neural_network import ELMRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import gen_batches
from sklearn.utils.testing import assert_raises, assert_greater, assert_equal


np.seterr(all='warn')

random_state = 1

ACTIVATION_TYPES = ["logistic", "tanh", "relu"]
CLASSIFICATION_TYPES = ["binary", "multi-class"]
KERNELS = ["random", "linear", "poly", "rbf", "sigmoid"]
NONLINEAR_KERNELS = ["random", "poly", "rbf", "sigmoid"]

digits_dataset_multi = load_digits(n_class=3)

Xdigits_multi = MinMaxScaler().fit_transform(digits_dataset_multi.data[:200])
ydigits_multi = digits_dataset_multi.target[:200]

digits_dataset_binary = load_digits(n_class=2)

Xdigits_binary = MinMaxScaler().fit_transform(digits_dataset_binary.data[:200])
ydigits_binary = digits_dataset_binary.target[:200]


classification_datasets = {'binary': (Xdigits_binary, ydigits_binary),
                           'multi-class': (Xdigits_multi, ydigits_multi)}

boston = load_boston()

Xboston = StandardScaler().fit_transform(boston.data)[:200]
yboston = boston.target[:200]


def test_classification():
    """Test ELMClassifier.

    It should score higher than 0.95 for binary and multi-class
    classification digits datasets.
    """
    for X, y in classification_datasets.values():
        X_train = X[:150]
        y_train = y[:150]
        X_test = X[150:]

        expected_shape = X_test.shape[0]
        expected_dtype = y_train.dtype.kind

        for activation in ACTIVATION_TYPES:
            elm = ELMClassifier(n_hidden=50, activation=activation,
                                random_state=random_state)
            elm.fit(X_train, y_train)

            y_predict = elm.predict(X_test)
            assert_greater(elm.score(X_train, y_train), 0.95)

            assert_equal(y_predict.shape[0], expected_shape)
            assert_equal(y_predict.dtype.kind, expected_dtype)


def test_kernel_classification():
    """Test whether kernels work as intended for classification."""
    for (X, y), kernel in product(classification_datasets.values(), KERNELS):
            elm = ELMClassifier(kernel=kernel)
            elm.fit(X, y)
            assert_greater(elm.score(X, y), 0.9)


def test_kernel_regression():
    """Test whether kernels work as intended for regression."""
    for kernel in NONLINEAR_KERNELS:
        elm = ELMRegressor(kernel=kernel)
        elm.fit(Xboston, yboston)
        assert_greater(elm.score(Xboston, yboston), 0.9)


def test_regression():
    """Test ELMRegressor.

    It should achieve score higher than 0.95 for the boston dataset.
    """
    for activation in ACTIVATION_TYPES:
        elm = ELMRegressor(activation=activation)
        elm.fit(Xboston, yboston)
        assert_greater(elm.score(Xboston, yboston), 0.95)


def test_multilabel_classification():
    """Test that multi-label classification works as expected."""
    # test fit method
    X, y = make_multilabel_classification(n_samples=50, random_state=0,
                                          return_indicator=True)
    elm = ELMClassifier()
    elm.fit(X, y)
    assert_equal(elm.score(X, y), 1)


def test_multioutput_regression():
    """Test whether multi-output regression works as expected."""
    X, y = make_regression(n_samples=200, n_targets=5,
                           random_state=random_state)
    for activation in ACTIVATION_TYPES:
        elm = ELMRegressor(n_hidden=300, activation=activation,
                           random_state=random_state)
        elm.fit(X, y)
        assert_greater(elm.score(X, y), 0.95)


def test_overfitting():
    """Larger number of hidden neurons should increase training score."""
    X, y = classification_datasets['multi-class']

    for activation in ACTIVATION_TYPES:
        elm = ELMClassifier(n_hidden=5, activation=activation,
                            random_state=random_state)
        elm.fit(X, y)
        score_5_n_hidden = elm.score(X, y)

        elm = ELMClassifier(n_hidden=15, activation=activation,
                            random_state=random_state)
        elm.fit(X, y)
        score_15_n_hidden = elm.score(X, y)

        assert_greater(score_15_n_hidden, score_5_n_hidden)


def test_params_errors():
    """Test whether invalid parameters raise value error."""
    X = [[3, 2], [1, 6]]
    y = [1, 0]
    clf = ELMClassifier

    assert_raises(ValueError, clf(n_hidden=-1).fit, X, y)
    assert_raises(ValueError, clf(activation='ghost').fit, X, y)


def test_partial_fit_classes_error():
    """Test that passing different classes to partial_fit raises an error."""
    X = [3, 2]
    y = [0]
    clf = ELMClassifier
    elm = clf()
    # different classes passed
    assert_raises(ValueError, elm.partial_fit, X, y, classes=[1, 2])


def test_partial_fit_classification():
    """Test partial_fit for classification.

    It should output the same results as 'fit' for binary and
    multi-class classification.
    """
    for X, y in classification_datasets.values():
        batch_size = 200
        n_samples = X.shape[0]

        elm_fit = ELMClassifier(random_state=random_state,
                                batch_size=batch_size)
        elm_partial_fit = ELMClassifier(random_state=random_state)

        elm_fit.fit(X, y)
        for batch_slice in gen_batches(n_samples, batch_size):
            elm_partial_fit.partial_fit(X[batch_slice], y[batch_slice],
                                        classes=np.unique(y))

        pred1 = elm_fit.predict(X)
        pred2 = elm_partial_fit.predict(X)

        assert_array_equal(pred1, pred2)
        assert_greater(elm_fit.score(X, y), 0.95)


def test_partial_fit_regression():
    """Test partial_fit for regression.

    It should output the same results as 'fit' for regression on
    different activations functions.
    """
    X = Xboston
    y = yboston
    batch_size = 100
    n_samples = X.shape[0]

    for activation in ACTIVATION_TYPES:
        elm_fit = ELMRegressor(random_state=random_state,
                               activation=activation, batch_size=batch_size)

        elm_partial_fit = ELMRegressor(activation=activation,
                                       random_state=random_state,
                                       batch_size=batch_size)

        elm_fit.fit(X, y)
        for batch_slice in gen_batches(n_samples, batch_size):
            elm_partial_fit.partial_fit(X[batch_slice], y[batch_slice])

        pred1 = elm_fit.predict(X)
        pred2 = elm_partial_fit.predict(X)

        assert_almost_equal(pred1, pred2, decimal=2)
        assert_greater(elm_fit.score(X, y), 0.95)


def test_predict_proba_binary():
    """Test whether predict_proba works as expected for binary class."""
    X = Xdigits_binary[:50]
    y = ydigits_binary[:50]

    clf = ELMClassifier(n_hidden=10)
    clf.fit(X, y)
    y_proba = clf.predict_proba(X)
    y_log_proba = clf.predict_log_proba(X)

    (n_samples, n_classes) = y.shape[0], 2

    proba_max = y_proba.argmax(axis=1)
    proba_log_max = y_log_proba.argmax(axis=1)

    assert_equal(y_proba.shape, (n_samples, n_classes))
    assert_array_equal(proba_max, proba_log_max)
    assert_array_equal(y_log_proba, np.log(y_proba))

    assert_greater(roc_auc_score(y, y_proba[:, 1]), 0.95)


def test_predict_proba_multi():
    """Test whether predict_proba works as expected for multi class."""
    X = Xdigits_multi[:10]
    y = ydigits_multi[:10]

    clf = ELMClassifier(n_hidden=5)
    clf.fit(X, y)
    y_proba = clf.predict_proba(X)
    y_log_proba = clf.predict_log_proba(X)

    (n_samples, n_classes) = y.shape[0], np.unique(y).size

    proba_max = y_proba.argmax(axis=1)
    proba_log_max = y_log_proba.argmax(axis=1)

    assert_equal(y_proba.shape, (n_samples, n_classes))
    assert_array_equal(proba_max, proba_log_max)
    assert_array_equal(y_log_proba, np.log(y_proba))


def test_recursive_and_standard():
    """Test that recursive lsqr return the same result as standard lsqr."""
    batch_size = 50
    for dataset, class_weight in product(classification_datasets.values(),
                                         [None, 'auto']):
        X, y = dataset
        elm_standard = ELMClassifier(class_weight=class_weight,
                                     random_state=random_state)
        elm_recursive = ELMClassifier(class_weight=class_weight,
                                      random_state=random_state,
                                      batch_size=batch_size)
        elm_standard.fit(X, y)
        elm_recursive.fit(X, y)

        pred1 = elm_standard.predict(X)
        pred2 = elm_recursive.predict(X)

        assert_array_equal(pred1, pred2)
        assert_greater(elm_standard.score(X, y), 0.95)


def test_sparse_matrices():
    """Test that sparse and dense input matrices yield equal output."""
    X = Xdigits_binary[:50]
    y = ydigits_binary[:50]
    X_sparse = csr_matrix(X)
    elm = ELMClassifier(random_state=1, n_hidden=15, batch_size=50)
    elm.fit(X, y)
    pred1 = elm.decision_function(X)
    elm = ELMClassifier(random_state=1, n_hidden=15)
    elm.fit(X_sparse, y)
    pred2 = elm.decision_function(X_sparse)
    assert_almost_equal(pred1, pred2)
    pred1 = elm.predict(X)
    pred2 = elm.predict(X_sparse)
    assert_array_equal(pred1, pred2)


def test_verbose():
    """Test whether verbose works as intended."""
    X = Xboston
    y = yboston

    elm = ELMRegressor(verbose=True)
    old_stdout = sys.stdout
    sys.stdout = output = StringIO()

    elm.fit(X, y)
    sys.stdout = old_stdout

    assert output.getvalue() != ''


def test_weighted_elm():
    """Check class weight impact on AUC

    Re-weighting classes is expected to improve model performance
    for imbalanced classification problems.
    """
    rng = np.random.RandomState(random_state)
    n_samples_1 = 500
    n_samples_2 = 10
    X = np.r_[1.5 * rng.randn(n_samples_1, 20),
              1.2 * rng.randn(n_samples_2, 20) + [2] * 20]
    y = [0] * (n_samples_1) + [1] * (n_samples_2)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X, y, test_size=0.8, random_state=random_state)

    n_hidden = 20
    for activation in ACTIVATION_TYPES:
        elm_weightless = ELMClassifier(n_hidden=n_hidden,
                                       class_weight=None,
                                       random_state=random_state)
        elm_weightless.fit(X_train, y_train)

        elm_weight_auto = ELMClassifier(n_hidden=n_hidden,
                                        class_weight='auto',
                                        random_state=random_state)
        elm_weight_auto.fit(X_train, y_train)

        y_pred_weightless = elm_weightless.predict_proba(X_test)[:, 1]
        score_weightless = roc_auc_score(y_test, y_pred_weightless)

        y_pred_weighted = elm_weight_auto.predict_proba(X_test)[:, 1]
        score_weighted = roc_auc_score(y_test, y_pred_weighted)

        assert_greater(score_weighted, score_weightless)
