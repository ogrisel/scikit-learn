from __future__ import annotations

import builtins
from collections.abc import Sequence
from typing import Literal

import mlx.core as mx

from ..common._aliases import (
    UniqueAllResult,
    UniqueCountsResult,
    UniqueInverseResult,
)
from ._typing import Array, Device, DType

_py_scalars = (bool, int, float, complex)

# MLX does not expose the array API name for the boolean dtype.
bool = mx.bool_

_DTYPES_BY_NAME = {
    "bool": mx.bool_,
    "int8": mx.int8,
    "int16": mx.int16,
    "int32": mx.int32,
    "int64": mx.int64,
    "uint8": mx.uint8,
    "uint16": mx.uint16,
    "uint32": mx.uint32,
    "uint64": mx.uint64,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
    "float32": mx.float32,
    "float64": mx.float64,
    "complex64": mx.complex64,
}

_INDEX_DTYPE = mx.int32
_DEFAULT_FLOAT = mx.float32


# Device handling
#
# MLX arrays live in unified memory and carry no device information. The
# `device` argument of the array API is therefore mapped to the `stream`
# argument of MLX operations, and the device of an array is reported as the
# current default device (see `array_api_compat.device`).
#
# Devices are reported as the strings "cpu" and "gpu" rather than as
# mx.Device objects because the latter can neither be copied nor pickled, and
# raise TypeError when compared with None.


def _device_name(device: Device) -> str:
    return "gpu" if device.type == mx.gpu else "cpu"


def _default_device_name() -> str:
    return _device_name(mx.default_device())


def _normalize_device(device: Device | str | None) -> Device | None:
    if device is None or isinstance(device, mx.Device):
        return device
    if isinstance(device, mx.DeviceType):
        return mx.Device(device)
    if isinstance(device, str):
        name, _, index = device.partition(":")
        if name in ("cpu", "gpu"):
            return mx.Device(getattr(mx, name), int(index) if index else 0)
    raise ValueError(f"Unsupported device for MLX: {device!r}")


def _resolve_device(device: Device | str | None) -> Device:
    device = _normalize_device(device)
    return mx.default_device() if device is None else device


def _check_dtype_device(dtype: DType | None, device: Device | None) -> None:
    # Fail early, as MLX would otherwise only raise when evaluating the lazy
    # graph, far away from the faulty call.
    # Note: mx.Dtype.__eq__ raises TypeError when comparing with None.
    if (
        dtype is not None
        and dtype == mx.float64
        and _resolve_device(device).type == mx.gpu
    ):
        raise TypeError(
            "MLX does not support float64 on the GPU device. Use float32 or "
            "run on the CPU device, for instance with "
            "mlx.core.set_default_device(mlx.core.cpu)."
        )


def _dtype_from_foreign(obj: object) -> DType | None:
    # NumPy scalars such as np.float64 subclass Python scalars but, unlike
    # them, carry a dtype that must be preserved.
    dtype = getattr(obj, "dtype", None)
    if dtype is None:
        return None
    name = getattr(dtype, "name", None)
    if not isinstance(name, str):
        # e.g. torch.dtype objects: "torch.float64"
        name = str(dtype).rsplit(".", 1)[-1]
    return _DTYPES_BY_NAME.get(name)


def _default_dtype_for_scalar(value: complex) -> DType:
    if isinstance(value, builtins.bool):
        return mx.bool_
    if isinstance(value, int):
        return mx.int32
    if isinstance(value, float):
        return _DEFAULT_FLOAT
    if isinstance(value, complex):
        return mx.complex64
    return getattr(value, "dtype", _DEFAULT_FLOAT)


# Creation functions


def asarray(
    obj: object,
    /,
    *,
    dtype: DType | None = None,
    device: Device | str | None = None,
    copy: builtins.bool | None = None,
    **kwargs: object,
) -> Array:
    device = _normalize_device(device)
    if isinstance(obj, mx.array):
        if dtype is None or obj.dtype == dtype:
            _check_dtype_device(obj.dtype, device)
            return mx.array(obj) if copy else obj
        if copy is False:
            raise ValueError("Unable to avoid copy while changing the dtype")
        return astype(obj, dtype, device=device)

    if copy is False:
        raise ValueError(
            "MLX arrays cannot share memory with objects from other libraries: "
            "copy=False is not supported."
        )
    if dtype is None:
        # mx.array silently downcasts float64 inputs to float32: preserve the
        # dtype of the input as required by the array API.
        dtype = _dtype_from_foreign(obj)
    _check_dtype_device(dtype, device)
    if dtype is None:
        return mx.array(obj)
    return mx.array(obj, dtype=dtype)


def from_dlpack(
    x: object,
    /,
    *,
    device: Device | str | None = None,
    copy: builtins.bool | None = None,
) -> Array:
    if not hasattr(x, "__dlpack__"):
        raise AttributeError(f"{type(x).__name__} does not implement __dlpack__")
    if isinstance(x, mx.array):
        return asarray(x, device=device, copy=copy)
    if copy is False:
        raise BufferError("MLX cannot import foreign memory without a copy")
    try:
        return asarray(x, device=device)
    except TypeError:
        raise
    except Exception as exc:
        raise BufferError(f"Cannot import {type(x).__name__} into MLX") from exc


def _creation(func, shape, dtype, device, **kwargs):
    device = _normalize_device(device)
    _check_dtype_device(dtype, device)
    return func(shape, dtype=dtype, stream=device, **kwargs)


def zeros(
    shape: int | tuple[int, ...],
    *,
    dtype: DType | None = None,
    device: Device | str | None = None,
) -> Array:
    return _creation(mx.zeros, shape, dtype or _DEFAULT_FLOAT, device)


def ones(
    shape: int | tuple[int, ...],
    *,
    dtype: DType | None = None,
    device: Device | str | None = None,
) -> Array:
    return _creation(mx.ones, shape, dtype or _DEFAULT_FLOAT, device)


# MLX has no uninitialized allocation: mx.empty is an alias for mx.zeros.
empty = zeros


def full(
    shape: int | tuple[int, ...],
    fill_value: complex,
    *,
    dtype: DType | None = None,
    device: Device | str | None = None,
) -> Array:
    if dtype is None:
        dtype = _default_dtype_for_scalar(fill_value)
    device = _normalize_device(device)
    _check_dtype_device(dtype, device)
    return mx.full(shape, fill_value, dtype=dtype, stream=device)


def _like(func, x, dtype, device, *args):
    device = _normalize_device(device)
    dtype = x.dtype if dtype is None else dtype
    _check_dtype_device(dtype, device)
    return func(x, *args, dtype=dtype, stream=device)


def zeros_like(
    x: Array, /, *, dtype: DType | None = None, device: Device | str | None = None
) -> Array:
    return _like(mx.zeros_like, x, dtype, device)


def ones_like(
    x: Array, /, *, dtype: DType | None = None, device: Device | str | None = None
) -> Array:
    return _like(mx.ones_like, x, dtype, device)


empty_like = zeros_like


def full_like(
    x: Array,
    /,
    fill_value: complex,
    *,
    dtype: DType | None = None,
    device: Device | str | None = None,
) -> Array:
    return _like(mx.full_like, x, dtype, device, fill_value)


def arange(
    start: float,
    /,
    stop: float | None = None,
    step: float = 1,
    *,
    dtype: DType | None = None,
    device: Device | str | None = None,
) -> Array:
    device = _normalize_device(device)
    _check_dtype_device(dtype, device)
    if stop is None:
        start, stop = 0, start
    return mx.arange(start, stop, step, dtype=dtype, stream=device)


def eye(
    n_rows: int,
    n_cols: int | None = None,
    /,
    *,
    k: int = 0,
    dtype: DType | None = None,
    device: Device | str | None = None,
) -> Array:
    device = _normalize_device(device)
    dtype = dtype or _DEFAULT_FLOAT
    _check_dtype_device(dtype, device)
    return mx.eye(n_rows, n_cols, k, dtype=dtype, stream=device)


def linspace(
    start: float,
    stop: float,
    /,
    num: int,
    *,
    dtype: DType | None = None,
    device: Device | str | None = None,
    endpoint: builtins.bool = True,
) -> Array:
    device = _normalize_device(device)
    dtype = dtype or _DEFAULT_FLOAT
    _check_dtype_device(dtype, device)
    return mx.linspace(start, stop, num, endpoint=endpoint, dtype=dtype, stream=device)


def tril(x: Array, /, *, k: int = 0) -> Array:
    return mx.tril(x, k)


def triu(x: Array, /, *, k: int = 0) -> Array:
    return mx.triu(x, k)


# Data type functions


def astype(
    x: Array,
    dtype: DType,
    /,
    *,
    copy: builtins.bool = True,
    device: Device | str | None = None,
) -> Array:
    device = _normalize_device(device)
    _check_dtype_device(dtype, device)
    if x.dtype == dtype:
        return mx.array(x) if copy else x
    if x.dtype == mx.float64:
        # Down casting from float64 can only be evaluated on the CPU.
        device = mx.cpu
    return mx.astype(x, dtype, stream=device)


def result_type(*arrays_and_dtypes: Array | DType | complex) -> DType:
    others = [a for a in arrays_and_dtypes if not isinstance(a, _py_scalars)]
    scalars = [a for a in arrays_and_dtypes if isinstance(a, _py_scalars)]
    if not others:
        raise ValueError("At least one array or dtype must be provided")
    dtype = mx.result_type(*others)
    # Python scalars are weakly typed: they only matter if they belong to a
    # higher kind than the arrays and dtypes.
    for s in scalars:
        if isinstance(s, complex) and not mx.isdtype(dtype, "complex floating"):
            dtype = mx.complex64
        elif (
            isinstance(s, float)
            and not isinstance(s, builtins.bool)
            and not mx.isdtype(dtype, ("real floating", "complex floating"))
        ):
            dtype = _DEFAULT_FLOAT
    return dtype


def finfo(type_: DType | Array, /):
    return mx.finfo(type_.dtype if isinstance(type_, mx.array) else type_)


def iinfo(type_: DType | Array, /):
    return mx.iinfo(type_.dtype if isinstance(type_, mx.array) else type_)


# Element-wise functions not implemented by MLX


def _as_mlx(x: Array | complex, like: Array | None = None) -> Array:
    if isinstance(x, mx.array):
        return x
    if like is not None and isinstance(x, _py_scalars):
        return mx.array(x, dtype=result_type(like, x))
    return asarray(x)


def _bits_dtype(dtype: DType) -> DType:
    return {2: mx.int16, 4: mx.int32, 8: mx.int64}[dtype.size]


def signbit(x: Array, /) -> Array:
    if mx.isdtype(x.dtype, "real floating"):
        # Reinterpret the bits to correctly handle -0.0 and signed NaNs.
        return mx.view(x, _bits_dtype(x.dtype)) < 0
    if x.dtype == mx.bool_ or mx.isdtype(x.dtype, "unsigned integer"):
        return mx.zeros_like(x, dtype=mx.bool_)
    return x < 0


def copysign(x1: Array | complex, x2: Array | complex, /) -> Array:
    x1 = _as_mlx(x1, x2 if isinstance(x2, mx.array) else None)
    x2 = _as_mlx(x2, x1)
    abs_x1 = mx.abs(x1)
    return mx.where(signbit(x2), -abs_x1, abs_x1)


def hypot(x1: Array | complex, x2: Array | complex, /) -> Array:
    x1 = _as_mlx(x1, x2 if isinstance(x2, mx.array) else None)
    x2 = _as_mlx(x2, x1)
    a, b = mx.abs(x1), mx.abs(x2)
    hi, lo = mx.maximum(a, b), mx.minimum(a, b)
    # Rescale to avoid overflow and underflow of the squares.
    ratio = lo / mx.where(hi == 0, 1, hi)
    out = hi * mx.sqrt(1 + ratio * ratio)
    return mx.where(mx.isinf(a) | mx.isinf(b), mx.inf, out)


def nextafter(x1: Array | complex, x2: Array | complex, /) -> Array:
    import numpy as np

    x1 = _as_mlx(x1, x2 if isinstance(x2, mx.array) else None)
    x2 = _as_mlx(x2, x1)
    return asarray(np.nextafter(np.asarray(x1), np.asarray(x2)))


def _float64_accurate(name: str, np_name: str):
    # The float64 CPU kernels of these MLX functions only have float32
    # accuracy: compute them with NumPy instead.
    mx_func = getattr(mx, name)

    def func(*args):
        if not builtins.any(
            isinstance(a, mx.array) and a.dtype == mx.float64 for a in args
        ):
            return mx_func(*args)
        import numpy as np

        np_args = [_to_numpy(a) if isinstance(a, mx.array) else a for a in args]
        return _from_numpy(np.asarray(getattr(np, np_name)(*np_args)))

    func.__name__ = func.__qualname__ = name
    return func


exp = _float64_accurate("exp", "exp")
sin = _float64_accurate("sin", "sin")
cos = _float64_accurate("cos", "cos")
logaddexp = _float64_accurate("logaddexp", "logaddexp")


def clip(
    x: Array,
    /,
    min: float | Array | None = None,
    max: float | Array | None = None,
) -> Array:
    if min is None and max is None:
        return x
    out = mx.clip(x, min, max)
    if out.dtype != x.dtype:
        out = mx.astype(out, x.dtype)
    return out


def round(x: Array, /) -> Array:
    return mx.round(x)


def isin(
    x1: Array,
    x2: Array,
    /,
    *,
    assume_unique: builtins.bool = False,
    invert: builtins.bool = False,
) -> Array:
    sorted_x2 = mx.sort(mx.reshape(_as_mlx(x2), (-1,)))
    if sorted_x2.shape[0] == 0:
        out = mx.zeros(x1.shape, dtype=mx.bool_)
    else:
        idx = mx.clip(mx.searchsorted(sorted_x2, x1), 0, sorted_x2.shape[0] - 1)
        out = mx.take(sorted_x2, idx) == x1
    return mx.logical_not(out) if invert else out


# Statistical functions


def sum(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype: DType | None = None,
    keepdims: builtins.bool = False,
) -> Array:
    if dtype is not None and dtype != x.dtype:
        x = astype(x, dtype)
    return mx.sum(x, axis=axis, keepdims=keepdims)


def prod(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype: DType | None = None,
    keepdims: builtins.bool = False,
) -> Array:
    if dtype is not None and dtype != x.dtype:
        x = astype(x, dtype)
    return mx.prod(x, axis=axis, keepdims=keepdims)


def _var(x, axis, correction, keepdims):
    if float(correction).is_integer():
        return mx.var(x, axis=axis, keepdims=keepdims, ddof=int(correction))
    n = x.size if axis is None else _axis_size(x, axis)
    var0 = mx.var(x, axis=axis, keepdims=keepdims)
    return var0 * (n / (n - correction))


def _axis_size(x, axis):
    axes = (axis,) if isinstance(axis, int) else axis
    n = 1
    for a in axes:
        n *= x.shape[a]
    return n


def var(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    correction: float = 0.0,
    keepdims: builtins.bool = False,
) -> Array:
    return _var(x, axis, correction, keepdims)


def std(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    correction: float = 0.0,
    keepdims: builtins.bool = False,
) -> Array:
    return mx.sqrt(_var(x, axis, correction, keepdims))


def _cumulative(func, init, x, axis, dtype, include_initial):
    if axis is None:
        if x.ndim > 1:
            raise ValueError(
                "axis must be specified in cumulative functions for more than "
                "one dimension"
            )
        axis = 0
    if dtype is not None and dtype != x.dtype:
        x = astype(x, dtype)
    res = func(x, axis=axis)
    if include_initial:
        shape = list(res.shape)
        shape[axis] = 1
        res = mx.concat([mx.full(shape, init, dtype=res.dtype), res], axis=axis)
    return res


def cumulative_sum(
    x: Array,
    /,
    *,
    axis: int | None = None,
    dtype: DType | None = None,
    include_initial: builtins.bool = False,
) -> Array:
    return _cumulative(mx.cumsum, 0, x, axis, dtype, include_initial)


def cumulative_prod(
    x: Array,
    /,
    *,
    axis: int | None = None,
    dtype: DType | None = None,
    include_initial: builtins.bool = False,
) -> Array:
    return _cumulative(mx.cumprod, 1, x, axis, dtype, include_initial)


def diff(
    x: Array,
    /,
    *,
    axis: int = -1,
    n: int = 1,
    prepend: Array | None = None,
    append: Array | None = None,
) -> Array:
    to_concat = [a for a in (prepend, x, append) if a is not None]
    if len(to_concat) > 1:
        x = mx.concat(to_concat, axis=axis)
    return mx.diff(x, n=n, axis=axis)


# Searching and sorting functions: MLX returns uint32 indices which can
# silently underflow in index arithmetic.


def argmax(
    x: Array, /, *, axis: int | None = None, keepdims: builtins.bool = False
) -> Array:
    return mx.astype(mx.argmax(x, axis=axis, keepdims=keepdims), _INDEX_DTYPE)


def argmin(
    x: Array, /, *, axis: int | None = None, keepdims: builtins.bool = False
) -> Array:
    return mx.astype(mx.argmin(x, axis=axis, keepdims=keepdims), _INDEX_DTYPE)


def argsort(
    x: Array,
    /,
    *,
    axis: int = -1,
    descending: builtins.bool = False,
    stable: builtins.bool = True,
) -> Array:
    if not descending:
        return mx.astype(mx.argsort(x, axis=axis), _INDEX_DTYPE)
    # Stable descending sort: sort the reversed array and map the indices back.
    res = mx.astype(
        mx.flip(mx.argsort(mx.flip(x, axis=axis), axis=axis), axis=axis),
        _INDEX_DTYPE,
    )
    return (x.shape[axis] - 1) - res


def sort(
    x: Array,
    /,
    *,
    axis: int = -1,
    descending: builtins.bool = False,
    stable: builtins.bool = True,
) -> Array:
    res = mx.sort(x, axis=axis)
    return mx.flip(res, axis=axis) if descending else res


def searchsorted(
    x1: Array,
    x2: Array,
    /,
    *,
    side: Literal["left", "right"] = "left",
    sorter: Array | None = None,
) -> Array:
    if sorter is not None:
        x1 = mx.take(x1, sorter)
    return mx.astype(mx.searchsorted(x1, _as_mlx(x2, x1), side=side), _INDEX_DTYPE)


def count_nonzero(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: builtins.bool = False,
) -> Array:
    return mx.count_nonzero(x, axis=axis, keepdims=keepdims)


# Functions with data-dependent output shapes. MLX cannot express them in a
# lazy graph: evaluate the input and compute them with NumPy on the host. This
# is cheap on Apple silicon thanks to unified memory.


def _to_numpy(x: Array):
    import numpy as np

    return np.asarray(x)


def _from_numpy(x, dtype: DType | None = None) -> Array:
    return asarray(x, dtype=dtype, device=mx.cpu if x.dtype.name == "float64" else None)


def nonzero(x: Array, /) -> tuple[Array, ...]:
    if x.ndim == 0:
        raise ValueError("nonzero() does not support zero-dimensional arrays")
    return tuple(_from_numpy(i, _INDEX_DTYPE) for i in _to_numpy(x).nonzero())


def _np_unique(x, **kwargs):
    import numpy as np

    return np.unique(_to_numpy(x).ravel(), equal_nan=False, **kwargs)


def unique_values(x: Array, /) -> Array:
    return _from_numpy(_np_unique(x))


def unique_counts(x: Array, /) -> UniqueCountsResult:
    values, counts = _np_unique(x, return_counts=True)
    return UniqueCountsResult(_from_numpy(values), _from_numpy(counts, _INDEX_DTYPE))


def unique_inverse(x: Array, /) -> UniqueInverseResult:
    values, inverse = _np_unique(x, return_inverse=True)
    return UniqueInverseResult(
        _from_numpy(values),
        mx.reshape(_from_numpy(inverse, _INDEX_DTYPE), x.shape),
    )


def unique_all(x: Array, /) -> UniqueAllResult:
    values, indices, inverse, counts = _np_unique(
        x, return_index=True, return_inverse=True, return_counts=True
    )
    return UniqueAllResult(
        _from_numpy(values),
        _from_numpy(indices, _INDEX_DTYPE),
        mx.reshape(_from_numpy(inverse, _INDEX_DTYPE), x.shape),
        _from_numpy(counts, _INDEX_DTYPE),
    )


# Manipulation functions


def concat(
    arrays: tuple[Array, ...] | list[Array], /, *, axis: int | None = 0
) -> Array:
    if axis is None:
        arrays = [mx.reshape(a, (-1,)) for a in arrays]
        axis = 0
    return mx.concat(list(arrays), axis=axis)


def stack(arrays: tuple[Array, ...] | list[Array], /, *, axis: int = 0) -> Array:
    return mx.stack(list(arrays), axis=axis)


def permute_dims(x: Array, /, axes: tuple[int, ...]) -> Array:
    return mx.transpose(x, axes)


def moveaxis(
    x: Array,
    source: int | tuple[int, ...],
    destination: int | tuple[int, ...],
    /,
) -> Array:
    if isinstance(source, int) and isinstance(destination, int):
        return mx.moveaxis(x, source, destination)
    source = [s % x.ndim for s in source]
    destination = [d % x.ndim for d in destination]
    order = [n for n in range(x.ndim) if n not in source]
    for dest, src in sorted(zip(destination, source)):
        order.insert(dest, src)
    return mx.transpose(x, order)


def reshape(
    x: Array, /, shape: tuple[int, ...], *, copy: builtins.bool | None = None
) -> Array:
    out = mx.reshape(x, shape)
    return mx.array(out) if copy else out


def roll(
    x: Array,
    /,
    shift: int | tuple[int, ...],
    *,
    axis: int | tuple[int, ...] | None = None,
) -> Array:
    return mx.roll(x, shift, axis)


def repeat(
    x: Array, repeats: int | Array, /, *, axis: int | None = None
) -> Array:
    if isinstance(repeats, int):
        return mx.repeat(x, repeats, axis=axis)
    import numpy as np

    return _from_numpy(np.repeat(_to_numpy(x), _to_numpy(repeats), axis=axis))


def tile(x: Array, repetitions: tuple[int, ...], /) -> Array:
    return mx.tile(x, repetitions)


def squeeze(x: Array, /, axis: int | tuple[int, ...]) -> Array:
    return mx.squeeze(x, axis=axis)


def expand_dims(x: Array, /, axis: int | tuple[int, ...] = 0) -> Array:
    return mx.expand_dims(x, axis)


# Linear algebra functions of the main namespace


def matmul(x1: Array, x2: Array, /) -> Array:
    return mx.matmul(x1, x2)


def matrix_transpose(x: Array, /) -> Array:
    if x.ndim < 2:
        raise ValueError("x must be at least 2-dimensional for matrix_transpose")
    return mx.swapaxes(x, -1, -2)


def tensordot(
    x1: Array,
    x2: Array,
    /,
    *,
    axes: int | tuple[Sequence[int], Sequence[int]] = 2,
) -> Array:
    if not isinstance(axes, int):
        axes = [list(axes[0]), list(axes[1])]
    return mx.tensordot(x1, x2, axes=axes)


def vecdot(x1: Array, x2: Array, /, *, axis: int = -1) -> Array:
    if x1.shape[axis] != x2.shape[axis]:
        raise ValueError("x1 and x2 must have the same size along the given axis")
    x1_ = mx.moveaxis(x1, axis, -1)[..., None, :]
    x2_ = mx.moveaxis(x2, axis, -1)[..., None]
    if mx.isdtype(x1_.dtype, "complex floating"):
        x1_ = mx.conj(x1_)
    return mx.matmul(x1_, x2_)[..., 0, 0]


def take(
    x: Array, indices: Array, /, *, axis: int | None = None
) -> Array:
    if axis is None:
        if x.ndim != 1:
            raise ValueError("axis must be specified when x is not 1-dimensional")
        axis = 0
    return mx.take(x, _as_mlx(indices), axis=axis)


__all__ = [
    "asarray", "from_dlpack", "zeros", "ones", "empty", "full", "zeros_like",
    "ones_like", "empty_like", "full_like", "arange", "eye", "linspace", "tril",
    "triu", "astype", "result_type", "finfo", "iinfo", "bool", "signbit",
    "copysign", "hypot", "nextafter", "exp", "sin", "cos", "logaddexp",
    "clip", "round", "isin", "sum", "prod",
    "var", "std", "cumulative_sum", "cumulative_prod", "diff", "argmax",
    "argmin", "argsort", "sort", "searchsorted", "count_nonzero", "nonzero",
    "unique_values", "unique_counts", "unique_inverse", "unique_all",
    "UniqueAllResult", "UniqueCountsResult", "UniqueInverseResult", "concat",
    "stack", "permute_dims", "moveaxis", "reshape", "roll", "repeat", "tile",
    "squeeze", "expand_dims", "matmul", "matrix_transpose", "tensordot",
    "vecdot", "take",
]


def __dir__() -> list[str]:
    return __all__
