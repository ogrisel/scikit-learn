# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

"""Time one (estimator, backend, n_samples) configuration in a fresh process."""

import argparse
import json
import os
import time

os.environ.setdefault("SCIPY_ARRAY_API", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np

import sklearn
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.utils._array_api import move_to

DEVICE_MEMORY_LIMIT = 6 * 1024**3
TIME_CAP_S = 5.0

ESTIMATORS = {
    "Ridge(cholesky)": lambda: Ridge(alpha=1.0, solver="cholesky"),
    "Ridge(svd)": lambda: Ridge(alpha=1.0, solver="svd"),
    "RidgeCV": lambda: RidgeCV(alphas=np.logspace(-3, 3, 10)),
    "LogisticRegression(lbfgs)": lambda: LogisticRegression(solver="lbfgs"),
    "LogisticRegression(newton-cg)": lambda: LogisticRegression(solver="newton-cg"),
}


def make_data(estimator_name, n_samples, n_features, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features), dtype=np.float32)
    if estimator_name.startswith("LogisticRegression"):
        W = rng.standard_normal((n_features, 3), dtype=np.float32) / np.sqrt(n_features)
        logits = X @ W + rng.gumbel(size=(n_samples, 3)).astype(np.float32)
        y = np.argmax(logits, axis=1).astype(np.float32)
    else:
        w = rng.standard_normal(n_features, dtype=np.float32)
        y = X @ w + rng.standard_normal(n_samples, dtype=np.float32)
    return X, y


def setup_backend(backend):
    if backend == "mlx-gpu":
        import mlx.core as mx

        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(DEVICE_MEMORY_LIMIT)
        mx.set_cache_limit(1024**3)

        def to_backend(a):
            return mx.array(a)

        def sync(est):
            mx.eval(*[v for v in vars(est).values() if isinstance(v, mx.array)])

        def peak_device_memory():
            return mx.get_peak_memory()

        return to_backend, sync, peak_device_memory, True

    if backend == "torch-mps":
        import torch

        torch.mps.set_per_process_memory_fraction(
            DEVICE_MEMORY_LIMIT / torch.mps.recommended_max_memory()
        )

        def to_backend(a):
            return torch.asarray(a, device="mps")

        def sync(est):
            torch.mps.synchronize()

        def peak_device_memory():
            return torch.mps.driver_allocated_memory()

        return to_backend, sync, peak_device_memory, True

    if backend == "numpy":
        return (lambda a: a), (lambda est: None), (lambda: 0), False

    raise ValueError(backend)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimator", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--n-samples", type=int, required=True)
    parser.add_argument("--n-features", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    to_backend, sync, peak_device_memory, dispatch = setup_backend(args.backend)
    make_estimator = ESTIMATORS[args.estimator]

    record = dict(
        estimator=args.estimator,
        backend=args.backend,
        n_samples=args.n_samples,
        n_features=args.n_features,
    )
    with sklearn.config_context(array_api_dispatch=dispatch):
        # Warm-up on small data to exclude kernel compilation and lazy imports.
        X, y = make_data(args.estimator, 2000, args.n_features, seed=1)
        est = make_estimator().fit(to_backend(X), to_backend(y))
        sync(est)

        X, y = make_data(args.estimator, args.n_samples, args.n_features)
        tic = time.perf_counter()
        X_b, y_b = to_backend(X), to_backend(y)
        record["transfer_s"] = time.perf_counter() - tic
        del X

        times = []
        for _ in range(args.repeats):
            est = make_estimator()
            tic = time.perf_counter()
            est.fit(X_b, y_b)
            sync(est)
            times.append(time.perf_counter() - tic)
            if times[-1] > TIME_CAP_S:
                break
        record["fit_times_s"] = times
        record["fit_time_s"] = float(np.min(times))
        n_iter = getattr(est, "n_iter_", None)
        record["n_iter"] = (
            None
            if n_iter is None
            else int(np.max(move_to(n_iter, xp=np, device="cpu")))
        )
        record["train_score"] = float(est.score(X_b, y_b))
        record["peak_device_memory_gb"] = peak_device_memory() / 1024**3

    with open(args.out, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
