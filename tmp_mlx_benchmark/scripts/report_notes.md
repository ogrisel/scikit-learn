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

## Methodology

- Synthetic float32 data. Regression is a linear model plus noise.
  Classification has 3 classes, sampled from the argmax of logits plus
  Gumbel noise. `n_features` is 200 or 1000, and `n_samples` takes values
  from 10k up to 1M.
- Each (estimator, backend, n_samples) configuration runs in **its own
  process** under an external watchdog (`scratch_mlx/watchdog.py`). The
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
    error, see `scratch_mlx/audit_float64.py`;
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
- Two runs exceeded the 5 s budget: NumPy `RidgeCV` at p=200, n=1M
  (7.7 s), and MLX `Ridge(svd)` at p=1000, n=300k (6.6 s). The linear
  extrapolation from the previous size predicted less than 7 s, so the
  driver allowed them.
- At small n (≤ 30k), the GPU timings are dominated by dispatch overhead
  and are noisy. For example, torch newton-cg at p=200 is slower at
  n=30k (0.115 s) than at n=100k (0.066 s).
- Only the 5 estimator configurations above were benchmarked. The other
  array API estimators were validated for correctness by the test suite,
  but not timed.
