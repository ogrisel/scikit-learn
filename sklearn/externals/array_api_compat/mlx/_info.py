"""
Array API Inspection namespace

This is the namespace for inspection functions as defined by the array API
standard. See
https://data-apis.org/array-api/latest/API_specification/inspection.html for
more details.

"""
import mlx.core as mx

from ._aliases import _default_device_name, _normalize_device, _resolve_device


def _dtypes_for_device(device):
    dtypes = {
        "bool": mx.bool_,
        "int8": mx.int8,
        "int16": mx.int16,
        "int32": mx.int32,
        "int64": mx.int64,
        "uint8": mx.uint8,
        "uint16": mx.uint16,
        "uint32": mx.uint32,
        "uint64": mx.uint64,
        "float32": mx.float32,
        "float64": mx.float64,
        "complex64": mx.complex64,
    }
    if device.type == mx.gpu:
        # Metal kernels do not support double precision.
        del dtypes["float64"]
    return dtypes


_KINDS = {
    "bool": ("bool",),
    "signed integer": ("int8", "int16", "int32", "int64"),
    "unsigned integer": ("uint8", "uint16", "uint32", "uint64"),
    "integral": (
        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"
    ),
    "real floating": ("float32", "float64"),
    "complex floating": ("complex64",),
    "numeric": (
        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
        "float32", "float64", "complex64",
    ),
}


class __array_namespace_info__:
    """
    Get the array API inspection namespace for MLX.

    The array API inspection namespace defines the following functions:

    - capabilities()
    - default_device()
    - default_dtypes()
    - dtypes()
    - devices()

    See
    https://data-apis.org/array-api/latest/API_specification/inspection.html
    for more details.

    Notes
    -----
    MLX arrays live in unified memory and are not attached to any device. The
    device of an operation is the device of the stream used to evaluate it,
    which defaults to ``mlx.core.default_device()``. The list of supported
    dtypes depends on the device: ``float64`` is only supported on the CPU.

    Examples
    --------
    >>> info = xp.__array_namespace_info__()
    >>> info.default_dtypes()
    {'real floating': mlx.core.float32,
     'complex floating': mlx.core.complex64,
     'integral': mlx.core.int32,
     'indexing': mlx.core.int32}

    """

    __module__ = "mlx.core"

    def capabilities(self):
        """
        Return a dictionary of array API library capabilities.

        - **"boolean indexing"**: ``False``. ``mlx.core.array.__getitem__``
          and ``__setitem__`` reject boolean masks.

        - **"data-dependent shapes"**: ``True``. MLX has no native support
          for data-dependent output shapes, but the wrapped ``nonzero``,
          ``unique_*`` and ``repeat`` (with array repeats) functions evaluate
          their input eagerly to compute the output shape.

        """
        return {
            "boolean indexing": False,
            "data-dependent shapes": True,
            "max dimensions": None,
        }

    def default_device(self):
        """
        The default device used for new MLX arrays.

        This is ``mlx.core.default_device()`` (as a ``"cpu"`` or ``"gpu"``
        string) which can be changed with ``mlx.core.set_default_device`` or
        the ``mlx.core.stream`` context manager.
        """
        return _default_device_name()

    def default_dtypes(self, *, device=None):
        """
        The default data types used for new MLX arrays.

        MLX uses single precision floating point and 32-bit integers by
        default, irrespective of the device.
        """
        _normalize_device(device)
        return {
            "real floating": mx.float32,
            "complex floating": mx.complex64,
            "integral": mx.int32,
            "indexing": mx.int32,
        }

    def dtypes(self, *, device=None, kind=None):
        """
        The array API data types supported by MLX on the given device.

        ``float64`` is only reported for the CPU device.
        """
        dtypes = _dtypes_for_device(_resolve_device(device))
        if kind is None:
            return dtypes
        if isinstance(kind, str):
            kind = (kind,)
        names = set()
        for k in kind:
            if k not in _KINDS:
                raise ValueError(f"unsupported kind: {k!r}")
            names.update(_KINDS[k])
        return {name: dt for name, dt in dtypes.items() if name in names}

    def devices(self):
        """
        The devices supported by MLX.
        """
        devices = ["cpu"]
        if mx.metal.is_available():
            devices.append("gpu")
        return devices
