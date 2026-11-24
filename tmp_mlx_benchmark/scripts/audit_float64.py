# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

"""Measure the float64 accuracy of raw MLX CPU kernels against NumPy."""

import mlx.core as mx
import numpy as np

mx.set_default_device(mx.cpu)
rng = np.random.default_rng(0)
x = rng.uniform(0.1, 3.0, size=10_000)
y = rng.uniform(0.1, 3.0, size=10_000)
A = rng.normal(size=(200, 300))
B = rng.normal(size=(300, 100))
xm, ym, Am, Bm = (mx.array(a, dtype=mx.float64) for a in (x, y, A, B))
assert xm.dtype == mx.float64

unary = [
    "exp",
    "log",
    "log1p",
    "expm1",
    "sqrt",
    "rsqrt",
    "sin",
    "cos",
    "tan",
    "tanh",
    "sinh",
    "cosh",
    "arcsin",
    "arccos",
    "arctan",
    "arcsinh",
    "arctanh",
    "erf",
    "erfinv",
    "sigmoid",
    "square",
    "reciprocal",
    "log2",
    "log10",
    "abs",
    "exp2",
]
cases = {}
for name in unary:
    fn = getattr(mx, name, None)
    if fn is None:
        continue
    arg = xm / 4 if name in ("arcsin", "arccos", "arctanh", "erfinv") else xm
    ref_arg = x / 4 if name in ("arcsin", "arccos", "arctanh", "erfinv") else x
    ref = {
        "rsqrt": lambda a: 1 / np.sqrt(a),
        "sigmoid": lambda a: 1 / (1 + np.exp(-a)),
        "reciprocal": lambda a: 1 / a,
        "erf": None,
        "erfinv": None,
        "exp2": np.exp2,
    }.get(name, getattr(np, name, None))
    if ref is None:
        import scipy.special as sp

        ref = getattr(sp, name)
    cases[name] = (np.asarray(fn(arg)), ref(ref_arg))

cases["divide"] = (np.asarray(xm / ym), x / y)
cases["power"] = (np.asarray(mx.power(xm, ym)), x**y)
cases["sum"] = (np.asarray(mx.sum(xm)), np.sum(x))
cases["mean"] = (np.asarray(mx.mean(xm)), np.mean(x))
cases["var"] = (np.asarray(mx.var(xm)), np.var(x))
cases["cumsum"] = (np.asarray(mx.cumsum(xm)), np.cumsum(x))
cases["logsumexp"] = (np.asarray(mx.logsumexp(xm)), np.log(np.sum(np.exp(x))))
cases["logaddexp"] = (np.asarray(mx.logaddexp(xm, ym)), np.logaddexp(x, y))
cases["matmul"] = (np.asarray(Am @ Bm), A @ B)
cases["matvec"] = (np.asarray(Am @ Bm[:, 0]), A @ B[:, 0])
cases["outer_sum_sq"] = (np.asarray(mx.sum(Am * Am, axis=1)), np.sum(A * A, axis=1))
cases["softmax"] = (
    np.asarray(mx.softmax(Am, axis=1)),
    np.exp(A) / np.exp(A).sum(axis=1, keepdims=True),
)

bad = []
for name, (got, ref) in cases.items():
    rel = np.max(np.abs(got - ref) / np.maximum(np.abs(ref), 1e-300))
    flag = "BAD" if rel > 1e-12 else "ok"
    if flag == "BAD":
        bad.append(name)
    print(f"{flag:4s} {name:14s} max rel err {rel:.2e}")
print("inaccurate:", bad)
