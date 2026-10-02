import numpy as np
import pytest
from scipy.sparse.linalg import lsqr as sp_lsqr

from sklearn.utils._lsqr import lsqr
from sklearn.utils._testing import assert_allclose


def _compare_lsqr(A, b, **kwargs):
    sp_result = sp_lsqr(A, b, **kwargs)
    result = lsqr(A, b, **kwargs)
    if A.dtype == np.float64:
        assert result[2] == sp_result[2]
        assert result[1] == sp_result[1]
        assert_allclose(result[0], sp_result[0], rtol=1e-10, atol=1e-10)
        for idx in range(3, 9):
            assert_allclose(result[idx], sp_result[idx], rtol=1e-8, atol=1e-8)
    else:
        # float32 follows SciPy's scalar recurrence, but a single ulp in the
        # matvecs can move the stopping test by one iteration.
        assert abs(int(result[2]) - int(sp_result[2])) <= 1
        assert_allclose(result[0], sp_result[0], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "shape", [(20, 5), (8, 15), (12, 12)], ids=["tall", "wide", "square"]
)
@pytest.mark.parametrize("damp", [0.0, 0.3])
def test_lsqr_matches_scipy(dtype, shape, damp):
    rng = np.random.RandomState(0)
    m, n = shape
    A = rng.randn(m, n).astype(dtype)
    b = rng.randn(m).astype(dtype)
    _compare_lsqr(A, b, damp=damp, atol=1e-8, btol=1e-8)


def test_lsqr_matches_scipy_iter_limit_and_x0():
    rng = np.random.RandomState(1)
    A = rng.randn(30, 10)
    b = rng.randn(30)
    x0 = rng.randn(10)
    _compare_lsqr(A, b, damp=0.1, iter_lim=3, atol=1e-12, btol=1e-12)
    _compare_lsqr(A, b, damp=0.0, x0=x0, atol=1e-10, btol=1e-10)


def test_lsqr_zero_rhs_returns_zero_solution():
    A = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    b = np.zeros(3)
    x, istop, itn = lsqr(A, b)[:3]
    assert istop == 0
    assert itn == 0
    assert_allclose(x, np.zeros(2))


def test_lsqr_scipy_documentation_examples():
    """Examples from ``scipy.sparse.linalg.lsqr`` on a dense matrix."""
    A = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

    b = np.array([1.0, 0.0, -1.0])
    x, istop, itn, r1norm = lsqr(A, b)[:4]
    sp_x, sp_istop, sp_itn, sp_r1norm = sp_lsqr(A, b)[:4]
    assert istop == sp_istop == 1
    assert itn == sp_itn
    assert_allclose(x, sp_x)
    assert_allclose(r1norm, sp_r1norm)

    b = np.array([1.0, 0.01, -1.0])
    x, istop, itn, r1norm = lsqr(A, b)[:4]
    sp_x, sp_istop, sp_itn, sp_r1norm = sp_lsqr(A, b)[:4]
    assert istop == sp_istop == 2
    assert itn == sp_itn
    assert_allclose(x, np.array([1.00333333, -0.99666667]), rtol=1e-6)
    assert_allclose(x, sp_x)
    assert_allclose(r1norm, sp_r1norm)
