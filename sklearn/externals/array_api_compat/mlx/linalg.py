"""
Array API compatible wrapper for ``mlx.core.linalg``.

Most MLX factorizations are only implemented on the CPU. As MLX arrays live
in unified memory, they are transparently evaluated on a CPU stream here,
without any data transfer. Results are evaluated eagerly so that numerical
failures are raised as :class:`LinAlgError` at the call site instead of when
the lazy graph is evaluated later.
"""
from __future__ import annotations

from typing import NamedTuple

import mlx.core as mx

from .._internal import clone_module
from ._aliases import matmul, matrix_transpose, sum, tensordot, vecdot
from ._typing import Array, DType

__all__ = clone_module("mlx.core.linalg", globals())


class LinAlgError(RuntimeError):
    """Error raised by MLX linear algebra functions."""


class EighResult(NamedTuple):
    eigenvalues: Array
    eigenvectors: Array


class QRResult(NamedTuple):
    Q: Array
    R: Array


class SlogdetResult(NamedTuple):
    sign: Array
    logabsdet: Array


class SVDResult(NamedTuple):
    U: Array
    S: Array
    Vh: Array


def _on_cpu(func, *args, **kwargs):
    # Evaluate the inputs first so that errors coming from upstream
    # operations are not reported as linear algebra errors.
    mx.eval(*[a for a in args if isinstance(a, mx.array)])
    out = func(*args, stream=mx.cpu, **kwargs)
    try:
        mx.eval(out)
    except RuntimeError as exc:
        raise LinAlgError(str(exc)) from exc
    return out


def cholesky(x: Array, /, *, upper: bool = False) -> Array:
    L = _on_cpu(mx.linalg.cholesky, x, upper=upper)
    # LAPACK stops on the first non-positive pivot and MLX returns the
    # partially factorized matrix without reporting the failure.
    diag = mx.diagonal(L, 0, -2, -1)
    if not mx.all(diag > 0).item():
        raise LinAlgError("Matrix is not positive definite")
    return L


def inv(x: Array, /) -> Array:
    return _on_cpu(mx.linalg.inv, x)


def solve(x1: Array, x2: Array, /) -> Array:
    return _on_cpu(mx.linalg.solve, x1, x2)


def eigh(x: Array, /) -> EighResult:
    return EighResult(*_on_cpu(mx.linalg.eigh, x))


def eigvalsh(x: Array, /) -> Array:
    return _on_cpu(mx.linalg.eigvalsh, x)


def qr(x: Array, /, *, mode: str = "reduced") -> QRResult:
    if mode != "reduced":
        raise NotImplementedError("MLX only supports mode='reduced' in qr")
    return QRResult(*_on_cpu(mx.linalg.qr, x))


def svd(x: Array, /, *, full_matrices: bool = True) -> SVDResult:
    if full_matrices:
        return SVDResult(*_on_cpu(mx.linalg.svd, x))
    # MLX always computes the full U and Vh factors, which needs
    # O(max(m, n)**2) memory. Compute the thin SVD from the SVD of the
    # triangular factor of a QR decomposition instead.
    m, n = x.shape[-2:]
    if m >= n:
        Q, R = qr(x)
        U, S, Vh = _on_cpu(mx.linalg.svd, R)
        return SVDResult(_on_cpu(mx.matmul, Q, U), S, Vh)
    Q, R = qr(matrix_transpose(x))
    U, S, Vh = _on_cpu(mx.linalg.svd, matrix_transpose(R))
    return SVDResult(U, S, _on_cpu(mx.matmul, Vh, matrix_transpose(Q)))


def svdvals(x: Array, /) -> Array:
    return _on_cpu(mx.linalg.svd, x, compute_uv=False)


def _default_rtol(x: Array) -> float:
    return max(x.shape[-2:]) * mx.finfo(x.dtype).eps


def pinv(x: Array, /, *, rtol: float | Array | None = None) -> Array:
    U, S, Vh = svd(x, full_matrices=False)
    if rtol is None:
        rtol = _default_rtol(x)
    cutoff = rtol * mx.max(S, axis=-1, keepdims=True)
    S_inv = mx.where(S > cutoff, 1 / mx.where(S > cutoff, S, 1), 0)
    return matrix_transpose(Vh) @ (S_inv[..., None] * matrix_transpose(U))


def matrix_rank(x: Array, /, *, rtol: float | Array | None = None) -> Array:
    S = svdvals(x)
    if rtol is None:
        rtol = _default_rtol(x)
    return mx.sum(S > rtol * mx.max(S, axis=-1, keepdims=True), axis=-1)


def _numpy_linalg(name, x):
    import numpy as np

    return getattr(np.linalg, name)(np.asarray(x))


def det(x: Array, /) -> Array:
    from ._aliases import _from_numpy

    return _from_numpy(_numpy_linalg("det", x))


def slogdet(x: Array, /) -> SlogdetResult:
    from ._aliases import _from_numpy

    sign, logabsdet = _numpy_linalg("slogdet", x)
    return SlogdetResult(_from_numpy(sign), _from_numpy(logabsdet))


def matrix_power(x: Array, n: int, /) -> Array:
    if n < 0:
        x, n = inv(x), -n
    result = mx.broadcast_to(mx.eye(x.shape[-1], dtype=x.dtype), x.shape)
    while n:
        if n & 1:
            result = result @ x
        n >>= 1
        if n:
            x = x @ x
    return result


def matrix_norm(
    x: Array, /, *, keepdims: bool = False, ord: float | str | None = "fro"
) -> Array:
    if ord in ("nuc", 2, -2):
        return _on_cpu(mx.linalg.norm, x, ord, axis=(-2, -1), keepdims=keepdims)
    return mx.linalg.norm(x, ord, axis=(-2, -1), keepdims=keepdims)


def vector_norm(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    ord: float = 2,
) -> Array:
    out_shape = None
    if axis is None:
        if keepdims:
            out_shape = (1,) * x.ndim
        x, axis = mx.reshape(x, (-1,)), 0
    elif isinstance(axis, tuple):
        axes = [a % x.ndim for a in axis]
        if keepdims:
            out_shape = tuple(1 if i in axes else s for i, s in enumerate(x.shape))
        rest = [i for i in range(x.ndim) if i not in axes]
        x = mx.transpose(x, rest + axes)
        x = mx.reshape(x, tuple(x.shape[: len(rest)]) + (-1,))
        axis = -1
    out = mx.linalg.norm(x, ord, axis=axis, keepdims=keepdims and out_shape is None)
    return out if out_shape is None else mx.reshape(out, out_shape)


def cross(x1: Array, x2: Array, /, *, axis: int = -1) -> Array:
    return mx.linalg.cross(x1, x2, axis=axis)


def outer(x1: Array, x2: Array, /) -> Array:
    return mx.outer(x1, x2)


def diagonal(x: Array, /, *, offset: int = 0) -> Array:
    return mx.diagonal(x, offset, -2, -1)


def trace(x: Array, /, *, offset: int = 0, dtype: DType | None = None) -> Array:
    return sum(diagonal(x, offset=offset), axis=-1, dtype=dtype)


__all__ += [
    "LinAlgError", "EighResult", "QRResult", "SlogdetResult", "SVDResult",
    "cholesky", "inv", "solve", "eigh", "eigvalsh", "qr", "svd", "svdvals",
    "pinv", "matrix_rank", "det", "slogdet", "matrix_power", "matrix_norm",
    "vector_norm", "cross", "outer", "diagonal", "trace", "matmul",
    "matrix_transpose", "tensordot", "vecdot",
]


def __dir__() -> list[str]:
    return __all__
