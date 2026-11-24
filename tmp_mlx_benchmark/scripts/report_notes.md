## TL;DR

- **All 25 estimators tagged `array_api_support`** work with MLX arrays
  under `sklearn.set_config(array_api_dispatch=True)`, on both the
  `"gpu"` (float32) and `"cpu"` (float64) devices. So do
  `GaussianMixture`, the calibration helpers, and the array API metrics.
  The whole `pytest sklearn -k mlx` selection passes: 1647 passed,
  15 skipped. Peak memory during the run was 0.6 GB.
- The changes don't regress the other namespaces. All
  `array_api or torch or strict` tests pass, as do the full test files of
  every touched module, apart from 18 pre-existing torch MPS failures
  (`torch.unique` does not support unsigned dtypes on MPS).
- For matmul-bound solvers (`Ridge(solver="cholesky")`, `LogisticRegression`
  lbfgs and newton-cg, `RidgeCV` at p=1000), MLX on the GPU is
  **0.87–1.16x the speed of PyTorch MPS**, so roughly on par. Against
  NumPy on the 10-core CPU, both GPU backends are 1.8–4x faster for
  cholesky and logistic regression, and 6.6–8.7x faster for `RidgeCV`.
- For SVD-based solvers, MLX is **1.5–1.8x faster than torch MPS**:
  `Ridge(solver="svd")`, and `RidgeCV` at p=200 (1.23x). This is despite
  MLX linalg running on the CPU stream. The gain mostly comes from the
  QR-first thin SVD, see below.
- **Memory**: the MLX process baseline is about 0.2 GB, against about
  0.35–1.35 GB for torch MPS. MLX peaks are much lower for SVD-based
  solvers. For example, `Ridge(svd)` at p=200, n=1M peaks at 3.2 GB on MLX
  against 6.3 GB on torch. For the largest cholesky and logistic runs, the
  peaks are similar, and somewhat higher for MLX `Ridge(cholesky)`.
- MLX support required:
  - a new `array_api_compat.mlx` wrapper (about 1250 lines);
  - a new `array_api_extra.compress` helper and a few small
    `array_api_extra` fixes;
  - about 30 small, namespace-agnostic rewrites in scikit-learn, mostly
    replacing boolean-mask indexing, see below.
- **Correction to the first version of this report.** The NumPy baseline
  numbers in that version were wrong. The editable scikit-learn build was
  linked against Homebrew's `libomp`, while conda's OpenBLAS loaded conda's
  `libomp`. Having two OpenMP runtimes in one process slowed down the
  OpenMP-parallel NumPy path (`LogisticRegression` at p=200: 1.9 s, against
  0.7 s now) and segfaulted `PoissonRegressor`. Rebuilding with the
  conda-forge compilers fixed this, and all numbers below come from the
  rebuilt version.
- **PoissonRegressor** (`lbfgs`, `newton-cg`; see the dedicated section
  below): MLX is on par with torch MPS for `lbfgs` (0.73-1.29x, best at
  small and medium sizes, slower at p=200 with 1M samples) and
  1.0-1.9x faster than torch for `newton-cg` in most configurations, up to
  3.5x at p=1000 with 300k samples. Scores are identical across backends.

## Methodology

- Synthetic float32 data. Regression is a linear model plus noise.
  Classification has 3 classes, sampled from the argmax of logits plus
  Gumbel noise. Poisson targets are drawn from `Poisson(exp(0.5 + X @ w))`
  with `w ~ N(0, 1/n_features)`. `n_features` is 200 or 1000, and `n_samples` takes values
  from 10k up to 1M.
- Each (estimator, backend, n_samples) configuration runs in **its own
  process** under an external watchdog (`tmp_mlx_benchmark/scripts/watchdog.py`). The
  watchdog SIGKILLs the process tree if any of these happen:
  - the wall time exceeds 60 s;
  - the macOS *physical footprint* exceeds 8 GB. The footprint is read with
    `proc_pid_rusage(...).ri_phys_footprint`, which, unlike RSS, includes
    Metal/GPU buffers in unified memory;
  - system available memory drops below 5 GB.
- In-process caps: `mx.set_memory_limit(6 GB)`, `mx.set_cache_limit(1 GB)`,
  and `torch.mps.set_per_process_memory_fraction`.
- The driver skips sizes whose predicted fit time (extrapolated linearly
  from the previous size) exceeds 7 s, or whose raw `X` would exceed
  1.5 GB. It stops a backend's sweep as soon as one fit takes more than 5 s.
  No run was killed in either sweep.
- Each configuration does one warm-up fit on a 2000-sample problem
  (kernel compilation and caches), then up to 3 timed fits. The best time
  is reported. The fitted attributes are synchronised (`mx.eval` /
  `torch.mps.synchronize()`) before the clock stops. Host-to-device
  transfer time is measured separately (`transfer_s`) and is not included
  in the fit time.
- Solver settings: `RidgeCV(alphas=np.logspace(-3, 3, 10))` uses the
  default efficient leave-one-out (GCV) path. `LogisticRegression` uses the
  default `C=1.0` and `tol`.

## PoissonRegressor

Only `solver="lbfgs"` (the `"auto"` choice for `l1_ratio == 0`) and
`solver="newton-cg"` support array API inputs. `newton-cholesky`,
`newton-cd` and `newton-cd-gram` still convert to NumPy internally, and
fail on torch MPS arrays as well as on MLX arrays. They are not benchmarked.
Default parameters are used (`alpha=1`, `tol=1e-4`, `max_iter=100`).

Fit time in seconds (best of 3) and MLX speedup over torch MPS. "hessp" is
the number of Hessian-vector products, i.e. the number of inner conjugate
gradient iterations of `newton-cg`. It is identical across backends except
where noted, which confirms that the three backends follow the same
optimization path.

| solver | p | n_samples | NumPy | torch MPS | MLX | MLX vs MPS |
|---|---|---|---|---|---|---|
| lbfgs | 200 | 10k | 0.004 | 0.027 | 0.010 | 2.59x |
| lbfgs | 200 | 100k | 0.026 | 0.027 | 0.029 | 0.93x |
| lbfgs | 200 | 300k | 0.077 | 0.056 | 0.070 | 0.80x |
| lbfgs | 200 | 1M | 0.330 | 0.179 | 0.246 | 0.73x |
| lbfgs | 1000 | 10k | 0.016 | 0.027 | 0.021 | 1.29x |
| lbfgs | 1000 | 100k | 0.107 | 0.089 | 0.080 | 1.11x |
| lbfgs | 1000 | 300k | 0.344 | 0.247 | 0.223 | 1.11x |
| newton-cg | 200 | 10k | 0.016 | 0.036 | 0.027 | 1.34x |
| newton-cg | 200 | 100k | 0.136 | 0.115 | 0.095 | 1.22x |
| newton-cg | 200 | 300k | 0.450 | 0.250 | 0.257 | 0.97x |
| newton-cg | 200 | 1M | 1.380 | 0.638 | 0.637 | 1.00x |
| newton-cg | 1000 | 10k | 0.075 | 0.099 | 0.132 | 0.75x |
| newton-cg | 1000 | 100k | 0.689 | 0.503 | 0.419 | 1.20x |
| newton-cg | 1000 | 300k | 6.339 | 4.892 | 1.410 | 3.47x |

(All rows except the last: `n_iter_` is identical on the three backends.
Train scores agree to 4 digits. Intermediate sizes are in the raw results.)

Observations:

- `lbfgs` does very little work per iteration (one loss + gradient
  evaluation, 6-9 iterations), and scipy's L-BFGS-B runs on the host, so
  each iteration forces a device sync and a device-to-host copy of the
  gradient. Fit times are therefore dominated by dispatch and sync latency
  at small n, where MLX has lower overhead than torch MPS (2.6x faster at
  p=200, n=10k). At p=200 and large n (>= 300k) torch MPS is faster (MLX
  is 0.73x at 1M samples); at p=1000 MLX is 1.1x faster. We did not
  investigate the p=200 large-n gap.
- `newton-cg` is dominated by the Hessian-vector products (`X.T @ (w * (X @ s))`).
  MLX and torch MPS are on par at p=200 for n >= 300k, and MLX is 1.2x
  faster at 100k. It is slower at p=1000, n=10k (0.75x), where the problem
  is too small for the GPU to help over NumPy for either backend.
- **The 3.5x at p=1000, n=300k is partly an artifact.** NumPy and torch MPS
  needed 14 Hessian-vector products (8 outer iterations), while MLX needed
  11 (7 outer iterations), most likely because float32 rounding differences
  changed the CG trajectory. Normalized per Hessian-vector product, the
  speedup is 2.7x (128 ms for MLX, 350 ms for torch MPS, 453 ms for
  NumPy). The time per product for NumPy and torch MPS also increases
  sharply between 100k (46 ms for torch MPS) and 300k samples (350 ms),
  while MLX scales roughly linearly. This was reproduced in two separate
  sweeps. It is not explained by the iteration counts. The extra cost may
  come from line-search loss / gradient evaluations, which are not counted,
  but we did not verify it. The 100k point (1.2x vs torch MPS, 1.6x vs
  NumPy) is the more reliable comparison.
- Memory: MLX peaks are lower than or similar to torch MPS in all
  configurations (for example 2.7 GB vs 3.7 GB at p=1000, n=300k, and
  1.6 GB vs 2.6 GB for `lbfgs` at p=200, n=1M). The torch MPS baseline is
  about 1.3 GB as soon as its allocator is warmed up at 30k-100k samples.

## How MLX support works

MLX (`mlx.core`) is close to the array API but differs in several ways
that matter to scikit-learn. All of these are handled in the vendored
`array_api_compat.mlx` wrapper unless noted otherwise:

| MLX gap | Workaround |
|---|---|
| No boolean-mask indexing (`x[mask]`) | `xp.where` rewrites in sklearn; new `xpx.compress(cond, x, axis=)` (take + nonzero); `xpx.at(...).set` converts boolean masks |
| No device on arrays; devices are not picklable/deep-copyable; `Device == None` and `Dtype == None` raise | Devices exposed as the strings `"cpu"` / `"gpu"` (the current default device); mapped to `stream=` internally |
| float64 is CPU only, and the GPU fails late with an obscure error | Eager `TypeError` on GPU float64 (same as torch MPS); `info.dtypes(device="gpu")` omits float64 so `move_to` downcasts |
| float64 `exp`, `sin`, `cos`, `logaddexp` on the CPU device are not accurate to float64 precision | Route to NumPy for float64 inputs; this fixed a 5e-3 relative coefficient error in float64 `LogisticRegression` |
| `linalg` is CPU-stream only, lazy, and errors surface at `eval` time; non-PD `cholesky` silently returns garbage | Run on `mx.cpu`, `mx.eval` eagerly, raise `LinAlgError`; explicit positive-diagonal check after cholesky |
| `svd` has no `full_matrices=False` (full `U` is O(n²) memory: this is what crashed a previous iteration) | Thin SVD emulated via QR followed by SVD of the small R factor |
| No `nonzero`, `unique_*`, `det`, `slogdet`, `nextafter`; data-dependent shapes | Round-trip through NumPy on the host |
| Default integer dtype is int32; `argmax`/`argsort` return uint32 | Cast to int32 (index dtype reported as int32) |
| Missing or differently named keywords: `correction`, `include_initial`, `descending`, `stable`, `copy`, `device`, `dtype` on reductions, `prepend`/`append`, ... | Thin wrappers |
| `mx.array(x)` silently downcasts float64 inputs, including NumPy scalars such as `np.float64`; `from_dlpack` does too | Dtype-preserving `asarray` from foreign arrays and NumPy scalars |
| Basic indexing returns copies, not views | Fixed the one sklearn call site (`GaussianMixture`) that relied on mutating a view |
| Streams are thread-local: a lazy array built in one thread cannot be evaluated in another | Skip host threading for MLX in `_parallel_pairwise` (new `_supports_threading(xp)` helper) |
| `mlx.core.array.__array_namespace__` returns raw `mlx.core`, so SciPy's array API dispatch uses it and fails | New `_scipy_supports_namespace(xp)` helper; array API fallbacks for `trapezoid` (`auc`) and `gmean` (multinomial link); `_xlogy` for `HalfPoissonLoss` |

Changes needed in scikit-learn itself. All of them are namespace-agnostic,
and the NumPy, array-api-strict and torch tests still pass:

- **Boolean-mask indexing.** Rewritten with `xp.where` or `xpx.compress`
  in:
  - `_ridge._solve_svd`, `_encode._check_unknown`, and `label_binarize`;
  - `r2_score` / `d2_*`, `multilabel_confusion_matrix`, and
    `balanced_accuracy_score`;
  - `roc_curve` and the threshold computation shared by the ranking
    metrics;
  - `calinski_harabasz_score` and `davies_bouldin_score`;
  - `_weighted_percentile`, `_incremental_mean_and_var`, and `_nanaverage`;
  - `_safe_indexing` with boolean keys;
  - `GaussianNB.partial_fit`;
  - `LinearDiscriminantAnalysis`: class means, the `svd` solver, and
    `predict_log_proba`;
  - `LogisticRegressionCV`: the `Cs_` reciprocals.
- **0-d arrays used as slice bounds or Python ints.** MLX rejects them.
  Wrapped with `int(...)` in `det_curve`, partial `roc_auc_score`, the LDA
  rank, `PCA` with a float `n_components`, and `PCA(n_components="mle")`.
- **Float dtypes.** `confusion_matrix` used to pick its float accumulator
  with a hard-coded `str(device).startswith("mps")` check, and now uses
  `_max_precision_float_dtype`. The `_bincount` NumPy fallback for
  weighted counts uses the same helper.
- **Latent bug.** `LogisticRegressionCV.l1_ratio_` was created without
  `device` / `dtype`. This also affects torch.
- **Tests.**
  - `yield_namespaces` gains `("mlx.core", "gpu", "float32")` and
    `("mlx.core", "cpu", "float64")`.
  - A few tests compared `x.device` attributes and now use
    `array_device`.
  - `check_array_api_input` skips the dispatch-disabled NumPy-indexing
    smoke test for MLX, as it already does for array-api-strict.

## Things worth reporting upstream

- **MLX**:
  - float64 accuracy of `exp`, `sin`, `cos`, `erf`, `erfinv`, `sigmoid`,
    `logsumexp`, `logaddexp`, and `softmax` on the CPU: about 1e-7 relative
    error, see `tmp_mlx_benchmark/scripts/audit_float64.py`;
  - `mx.array` / `from_dlpack` downcasting float64;
  - `__array_namespace__` advertising non-compliant `mlx.core`;
  - `Device` / `Dtype` `__eq__(None)` raising;
  - `Device` not picklable;
  - `svd(full_matrices=False)` missing;
  - silent garbage from non-PD `cholesky`;
  - lazy linalg errors;
  - boolean-mask indexing missing.
- **array-api-compat**: the new `mlx` wrapper as a whole, plus
  `is_mlx_array` / `is_mlx_namespace`.
- **array-api-extra**:
  - `compress`;
  - the `at` boolean-mask path for namespaces without boolean indexing;
  - `setdiff1d` without boolean indexing.
- **SciPy**: array API dispatch for MLX would require SciPy's vendored
  array-api-compat to include the new wrapper.

## Caveats

- MLX has no per-array device. An MLX "device" in scikit-learn is the
  process-wide default device at the time of the call. Mixing CPU and GPU
  MLX arrays in one process is therefore not meaningfully supported.
- Linear algebra (cholesky, eigh, qr, svd) runs on the MLX CPU stream, not
  the GPU. The MLX `Ridge(svd)` and `RidgeCV` speedups come from the GPU
  matmuls around the decomposition and from the QR-first thin SVD trick.
- Three runs exceeded the 5 s budget: NumPy `RidgeCV` at p=200, n=1M
  (7.7 s), MLX `Ridge(svd)` at p=1000, n=300k (6.6 s), and NumPy
  `PoissonRegressor(newton-cg)` at p=1000, n=300k (6.3 s). The linear
  extrapolation from the previous size predicted less than 7 s, so the
  driver allowed them.
- At small n (≤ 30k), the GPU timings are dominated by dispatch overhead
  and are noisy. For example, torch newton-cg at p=200 is slower at
  n=30k (0.115 s) than at n=100k (0.066 s).
- Only 7 estimator configurations (Ridge cholesky / svd, RidgeCV,
  LogisticRegression lbfgs / newton-cg, PoissonRegressor lbfgs / newton-cg)
  were benchmarked. The other
  array API estimators were validated for correctness by the test suite,
  but not timed.
