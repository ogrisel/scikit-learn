import numpy as np

from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_equal


def test_compute_class_weight():
    """Test (and demo) compute_class_weight."""
    y = np.asarray([2, 2, 2, 3, 3, 4])
    classes = np.unique(y)
    cw = compute_class_weight("auto", classes, y)
    assert_almost_equal(cw.sum(), classes.shape)
    assert_true(cw[0] < cw[1] < cw[2])


def test_compute_class_weight_not_present():
    """Raise error when y does not contain all class labels"""
    classes = np.arange(4)
    y = np.asarray([0, 0, 0, 1, 1, 2])
    assert_raises(ValueError, compute_class_weight, "auto", classes, y)


def test_compute_class_weight_auto_negative():
    """Test compute_class_weight when labels are negative"""
    # Test with balanced class labels.
    classes = np.array([-2, -1, 0])
    y = np.asarray([-1, -1, 0, 0, -2, -2])
    cw = compute_class_weight("auto", classes, y)
    assert_almost_equal(cw.sum(), classes.shape)
    assert_equal(len(cw), len(classes))
    assert_array_almost_equal(cw, np.array([1., 1., 1.]))

    # Test with unbalanced class labels.
    y = np.asarray([-1, 0, 0, -2, -2, -2])
    cw = compute_class_weight("auto", classes, y)
    assert_almost_equal(cw.sum(), classes.shape)
    assert_equal(len(cw), len(classes))
    assert_array_almost_equal(cw, np.array([0.545, 1.636, 0.818]), decimal=3)


def test_compute_class_weight_auto_unordered():
    """Test compute_class_weight when classes are unordered"""
    classes = np.array([1, 0, 3])
    y = np.asarray([1, 0, 0, 3, 3, 3])
    cw = compute_class_weight("auto", classes, y)
    assert_almost_equal(cw.sum(), classes.shape)
    assert_equal(len(cw), len(classes))
    assert_array_almost_equal(cw, np.array([1.636, 0.818, 0.545]), decimal=3)


def test_compute_sample_weight():
    """Test compute_sample_weight

    Make sure it returns the same unique values as compute_class_weight
    and in the same size as `y`.
    """
    classes = np.array([1, 0, 3, 5])
    y = np.asarray([1, 0, 0, 3, 3, 3, 5, 5, 5, 5, 5, 5, 5])
    sw = compute_sample_weight("auto", classes, y)
    cw = compute_class_weight("auto", classes, y)

    assert_array_equal(np.unique(sw), np.unique(cw))
    assert_equal(sw.shape[0], y.shape[0])


def test_compute_sample_weight_error_handling():
    """Test error handling of compute_sample_weight

    compute_sample_weight should raise an error when 'y' has a label not
    in classes.
    """
    classes = np.array([1, 0])
    y = np.asarray([1, 0, 0, 3, 3])

    assert_raises(ValueError, compute_sample_weight, "auto", classes, y)
