# Suggested GitHub issue

Paste the sections below into a [feature request](https://github.com/scikit-learn/scikit-learn/issues/new?template=feature_request.yml). This is a prioritization and tracking issue. It is not a request to implement all ten items in one pull request.

**Title:** ENH Array API: prioritized low-hanging estimator and solver gaps

**Labels the template will add:** `New Feature`, `Needs Triage`. A maintainer can add `Array API` after triage.

**Related issues:**

- #35060 — `Ridge` `solver="cholesky"` (already open, labeled Needs Decision; do not start a PR until that label is cleared)
- #33584 — covariance module tracking issue
- #26585 — k-means
- #26586 — brute-force k-nearest neighbors
- #31869 — `CalibratedClassifierCV`

Audited tree: `scikit-learn/scikit-learn` `main` at `d0e22aee7b`.

---

## Describe the workflow you want to enable

Array API dispatch lets a subset of estimators run on PyTorch, CuPy, and dpnp, including CUDA and other accelerators, without a separate GPU library. Coverage today is 23 documented estimators, and 9 of those only for some hyperparameter settings. Several tags are wider than the code, so the common tests follow the default constructor and never exercise the missing solvers.

The workflow this issue is about: a user moves a dense array to a device and fits one of the estimators below under the configuration they already use on NumPy. The bottleneck of that configuration is a GEMM, a Cholesky factorization, a thin SVD, or an elementwise map, and the surrounding code is small enough that the port does not add a second algorithm.

Two tags currently overclaim support:

- `Ridge` and `RidgeClassifier` set `array_api_support = True` for every solver. `ridge_regression` then rejects every resolved solver except `"svd"`. `solver="auto"` is rewritten to `"svd"` and warns. On NumPy, `auto` resolves to `"cholesky"`.
- `LinearDiscriminantAnalysis` sets the tag unconditionally, but only `_solve_svd` dispatches. `shrinkage` is defined only for `solver="eigen"` and `solver="lsqr"`, so the shrinkage configurations cannot run under array API dispatch.

`GaussianMixture` is the opposite: the tag is conditional on `init_params in {"random", "random_from_data"}` and `warm_start=False`, and the other inits raise `NotImplementedError`.

## Describe your proposed solution

Track the following ten configurations, in this order. "User impact" combines whether the listed setting is the default with two popularity proxies: the number of files in scikit-learn's `examples/` tree that construct the estimator, and public GitHub code search. GitHub counts are matching Python files, not unique projects or users; forks, generated files, and tutorials inflate them, so they are useful for broad tiers rather than precise market share.

"Incremental cost" combines code complexity and future maintenance caused specifically by adding the missing array API path:

- **Very low:** almost all production changes are mechanical namespace substitutions such as `np.foo` to `xp.foo`, plus the tag and tests.
- **Low:** a few contained namespace/device branches or moves, while retaining one mathematical implementation.
- **Medium:** a persistent dual path, explicit host/device transfers, dtype/device handling, or numerical-equivalence work across namespace implementations.
- **High:** a new numerical reduction or algorithmic path whose rank-deficiency, precision, or convergence behavior must be maintained beside the SciPy/Cython CPU path.

The accelerator score is a 1–5 judgment of whether the bottleneck is a dense kernel a GPU already executes well.

| # | Configuration | User impact | In-tree examples | Accel. | Incremental cost |
| - | ------------- | ----------- | ---------------: | -----: | ---------------- |
| 1 | `Ridge` and `RidgeClassifier`, `solver="cholesky"` and dense `solver="auto"` | Very high; `auto` is the default | 17 | 5 | Medium |
| 2 | `LinearRegression`, `positive=False`, dense `X` | Very high; both are defaults | 17 | 5 | Medium |
| 3 | `RBFSampler`, dense `X` | Medium-low; estimator is specialized, settings are defaults | 1 | 5 | Low |
| 4 | `KernelRidge`, kernels already implemented for the array API | Medium; estimator is established, nonlinear kernels are non-default | 2 | 5 | Medium |
| 5 | `TruncatedSVD`, `algorithm="randomized"`, `power_iteration_normalizer="QR"` | High for dense input; randomized is the default, QR is non-default | 3 | 4 | Low |
| 6 | `EmpiricalCovariance` and `ShrunkCovariance` | Medium-low; small API surface, default settings | 4 | 4 | Very low |
| 7 | `LinearDiscriminantAnalysis`, `solver="eigen"`, including `shrinkage="auto"` | Medium; estimator is popular, this solver is non-default | 5 | 4 | High |
| 8 | `QuadraticDiscriminantAnalysis`, `solver="svd"` and `reg_param` | Medium-low; default solver on a less common estimator | 2 | 4 | Medium |
| 9 | `KernelPCA`, `eigen_solver="dense"`, same kernels as KernelRidge | Medium; dense is selected by `auto` only for some shapes | 2 | 4 | Medium |
| 10 | `GammaRegressor` and `TweedieRegressor`, `solver="lbfgs"`, `l1_ratio=0` | Medium-low; defaults on domain-specific estimators | 3 | 4 | Medium |

Each item should be its own pull request, with a conditional `array_api_support` tag wherever only one solver or kernel set dispatches.

### 1. Ridge and RidgeClassifier, `solver="cholesky"` and dense `solver="auto"`

**User impact: very high. Incremental Array API cost: medium.** Scikit-learn has 17 example files constructing `Ridge` or `RidgeClassifier`. A public GitHub code search for `"Ridge(" sklearn language:Python` matched about 212,000 Python files on October 15, 2026. That is an upper-bound popularity proxy, but it places Ridge in the highest-usage tier. More importantly, the missing `auto` configuration is the constructor default rather than an opt-in niche.

`solver="auto"` is the default and resolves to Cholesky for ordinary dense NumPy input. Dispatch forces SVD. On a GPU that SVD is often slower than Cholesky on CPU, so the supported configuration is the one users should not be running.

Form the Gram matrix on the device (`X.T @ X`, or `X @ X.T` when `n_features > n_samples`), add `alpha` on the diagonal, and solve with the existing SciPy path on the host. `_solve_cholesky` and `_solve_cholesky_kernel` already implement that solve. `fit_intercept` centering already goes through array-API `_preprocess_data`. Leave `positive=True` on `lbfgs`, and leave `sag`, `saga`, `lsqr`, and `sparse_cg` alone.

The host solve matches today's NumPy results, including the singular-matrix fallback, per-target `alpha`, and sample weights. `xp.linalg.solve` on device is faster only for large dual systems, and namespaces disagree on singular input (PyTorch raises, CuPy can return NaNs). This is the design in #35060. That issue is labeled Needs Decision.

On a 4-thread CPU (NumPy 2.5.4, SciPy 1.18.1, OpenBLAS), median float64 time after warmup:

| Shape | Gram + Cholesky, alpha = 1 | `scipy.linalg.lstsq` (gelsd) | Thin SVD |
| ----- | -------------------------: | ---------------------------: | -------: |
| 200,000 × 100 | 25 ms | 343 ms | 597 ms |
| 50,000 × 1,000 | 181 ms | 1,239 ms | 2,306 ms |

Float32 on the same machine, 200,000 × 100: Cholesky 21 ms, lstsq 182 ms, thin SVD 292 ms.

#35060 measured the GPU side on an NVIDIA L4, 500,000 × 100 float32: NumPy Cholesky 0.13 s, the current Torch SVD path 0.091 s, device Gram plus host Cholesky 0.011 s. Those GPU figures were not remeasured for this audit.

### 2. LinearRegression, `positive=False`, dense `X`

**User impact: very high. Incremental Array API cost: medium.** Scikit-learn has 17 example files constructing `LinearRegression`. A public GitHub code search for `"LinearRegression(" sklearn language:Python` matched about 379,000 Python files on October 15, 2026, the largest count among the sampled candidates. The missing dense, unconstrained configuration is the default.

Ordinary least squares has no array API path. `validate_data` and `_preprocess_data` already dispatch. The dense branch is one `scipy.linalg.lstsq`. The array API has `svd` and `qr`, and it does not have `lstsq`. A thin SVD, the same kernel Ridge already uses, preserves `rank_` and `singular_`. Keep the sparse `lsqr` branch and the `positive=True` `nnls` branch on NumPy.

The review point is the gelsd contract: `tol`, `rank_`, and `singular_` have to match closely enough for the common tests. Multi-target `y` is the shape to test first.

### 3. RBFSampler, dense `X`

**User impact: medium-low. Incremental Array API cost: low.** Only one scikit-learn example constructs `RBFSampler`. The listed dense path and `n_components` behavior are defaults, which makes this more useful than its raw estimator popularity suggests: it unlocks an otherwise on-device `RBFSampler` → linear-model pipeline.

Random Fourier features are the usual way to pair an approximate RBF map with a linear estimator. `Ridge` already has a GPU SVD path, and `Nystroem` already dispatches. `RBFSampler` is the missing piece. `transform` is a GEMM, a bias add, and `cos`.

Draw `random_weights_` and `random_offset_` with NumPy from `random_state`, then move them to the namespace and device of `X`, so the random stream does not depend on the device. `gamma="scale"` needs an array-API variance. The projection dtype must follow `X`. The sparse CSR branch can stay on `safe_sparse_dot`.

Same CPU environment: 200,000 × 128 → 512 took 151 ms in float32 and 1,683 ms in float64. 100,000 × 512 → 1,024 float32 took 244 ms.

### 4. KernelRidge, kernels that already dispatch

**User impact: medium. Incremental Array API cost: medium.** Two scikit-learn examples construct `KernelRidge`. The estimator is established, but the accelerator-relevant nonlinear kernels are non-default (`kernel="linear"` is the default). The cost is not a namespace rename: the project must maintain device kernel construction beside a host Cholesky solve and preserve mutation/fallback behavior.

For a few thousand samples, kernel ridge is the closed-form nonlinear regressor. The useful configurations are `rbf` and `polynomial` (the default kernel is `linear`). `pairwise_kernels` already dispatches `linear`, `rbf`, `polynomial`, `sigmoid`, `laplacian`, `cosine`, `chi2`, and `additive_chi2` when `n_jobs` resolves to 1. Callable kernels stay unsupported.

The blocker is `_solve_cholesky_kernel`: NumPy in-place diagonal update, SciPy solve, sample-weight outer product. `predict` is the second GEMM, `K @ dual_coef_`. A precomputed kernel is modified in place and must be restored, including when the solve falls back to `lstsq` on a singular matrix. The NumPy chi2 kernels have a Cython fast path that the array API branch must leave in place.

At n = 4,000, p = 64, float64: building the RBF kernel took 124 ms and the Cholesky solve of that kernel took 267 ms. Both pieces have to move.

### 5. TruncatedSVD, `algorithm="randomized"` and `power_iteration_normalizer="QR"`

**User impact: high for dense inputs. Incremental Array API cost: low.** Three scikit-learn examples construct `TruncatedSVD`. A public GitHub code search for `"TruncatedSVD(" sklearn language:Python` matched about 38,000 Python files on October 15, 2026. `algorithm="randomized"` is the default and is especially common in text and embedding workflows, although many real uses are SciPy sparse and would remain on CPU. Requiring `power_iteration_normalizer="QR"` narrows the immediate impact because its default is `"auto"`.

Randomized is the default algorithm. PCA already runs `_randomized_svd` on device when the normalizer is `"QR"`. `svd_flip` is already array-API aware. The remaining work is namespace-aware explained variance and `transform`.

Leave `algorithm="arpack"` (`scipy.sparse.linalg.svds`) on the NumPy and sparse path. `LU` and `"none"` normalizers are already rejected for array API inputs inside `randomized_svd`. Sparse TF–IDF callers keep the sparse path. The tag should be conditional, matching PCA.

### 6. EmpiricalCovariance and ShrunkCovariance

**User impact: medium-low. Incremental Array API cost: very low.** Four scikit-learn examples construct one of these estimators. Their public surface is much smaller than Ridge or linear regression, but the listed settings are defaults. Most of the numerical path already dispatches, making this the clearest near-mechanical port in the list.

`empirical_covariance` already forms the Gram matrix with the array API for non-NumPy namespaces. `LedoitWolf` and `OAS` are tagged. These two base estimators are not, so the common tests never certify them. #33584 still lists both as open.

Set the tag on `EmpiricalCovariance`. Replace `np.trace` and `np.eye` in `shrunk_covariance`, and `np.zeros` / `ndarray.mean` in `ShrunkCovariance.fit`, with the helpers Ledoit–Wolf already uses. `mahalanobis` and `error_norm` already branch on the namespace.

Keep the NumPy path of `empirical_covariance` on `np.cov`. That call returns float64, and `GraphicalLasso` depends on it. `MinCovDet`, `EllipticEnvelope`, and `GraphicalLasso` are separate changes: the first two are combinatorial, and graphical lasso is coordinate descent.

Centered Gram on the same CPU: 20,000 × 1,500 float32 took 72 ms; 60,000 × 300 float64 took 69 ms.

### 7. LinearDiscriminantAnalysis, `solver="eigen"`, including `shrinkage="auto"`

**User impact: medium. Incremental Array API cost: high.** Five scikit-learn examples construct LDA, but the missing `solver="eigen"` is not the default. Its value is concentrated in the important high-dimensional/small-class-sample regime where shrinkage is recommended. The implementation is a new reduction of a generalized eigenproblem, not an `np` → `xp` conversion.

The SVD solver is supported and is the default, and it cannot use shrinkage. `shrinkage="auto"` is the documented setting when a class has few rows relative to the number of features. Ledoit–Wolf, which implements that shrinkage, is already tagged.

`_class_means` already dispatches. `_class_cov` and `_solve_eigen` do not. `_solve_eigen` calls `scipy.linalg.eigh(Sb, Sw)`, a generalized problem that is not in the array API. Reduce it: Cholesky-whiten `Sw`, then call `xp.linalg.eigh` on the transformed between-class scatter. Leave `solver="lsqr"` for later; it is a SciPy iterative solver and a weaker accelerator fit.

The tag has to become conditional, or the common tests will keep passing on the default SVD solver while shrinkage stays broken. Rank-deficient `Sw` and `explained_variance_ratio_` ordering need explicit tests. `covariance_estimator` only works when that estimator itself dispatches (`LedoitWolf` and `OAS` do; `GraphicalLasso` does not).

### 8. QuadraticDiscriminantAnalysis, `solver="svd"` and `reg_param`

**User impact: medium-low. Incremental Array API cost: medium.** Two scikit-learn examples construct QDA. The SVD solver is the default, but QDA is less commonly used than LDA and scales poorly in the number of features. Per-class rank handling, dtype-sensitive warnings, and namespace-preserving prediction make the maintenance cost more than mechanical.

Fit and `predict_log_proba` are per-class covariances and quadratic forms, so a device helps training and inference. `GaussianMixture` already does the per-component version, and LDA's SVD path in the same file is the pattern to copy. Reuse `_class_means`. `solver="eigen"` can follow once item 7 exists, because that is where QDA shrinkage lives.

Rank handling, the `tol` warning, `store_covariance`, and `priors` have to round-trip in the input namespace. Keep this separate from the LDA generalized eigenproblem.

### 9. KernelPCA, `eigen_solver="dense"`

**User impact: medium. Incremental Array API cost: medium.** Two scikit-learn examples construct `KernelPCA`. `eigen_solver="auto"` selects dense only for some shapes; users who explicitly select dense are a subset. Kernel and centering code already dispatch, but spectrum slicing, rank/zero-eigenvalue decisions, inverse transform, and the conditional tag create a maintained dual path.

`KernelCenterer` and the same kernels as KernelRidge already dispatch. For `n_samples <= 200`, or whenever `n_components` is not small, `eigen_solver="auto"` already selects the dense eigensolver. `xp.linalg.eigh` has no `subset_by_index`; compute the full spectrum and slice. That is acceptable while the supported setting is the dense one.

`auto` switches to `arpack` when `n_samples > 200` and `n_components < 10`. The tag has to be conditional, the way PCA's tag already is. Inverse transform allocates a dense dual coefficient matrix and has to stay on device. Leave `arpack` and `randomized` on NumPy.

### 10. GammaRegressor and TweedieRegressor, `solver="lbfgs"` and `l1_ratio=0`

**User impact: medium-low. Incremental Array API cost: medium.** Three scikit-learn examples construct one of these estimators. The listed settings are defaults, but the estimators are domain-specific. Gamma alone would be low cost; treating Gamma and all Tweedie powers as one item raises the combined cost because the new loss code must maintain dtype-sensitive behavior at the `power` 0, 1, and 2 limits.

`PoissonRegressor` already runs `lbfgs` on device. Gamma and Tweedie use the same solver and the same GEMM (`X.T @ gradient`). The missing piece is the loss. `HalfPoissonLossArrayAPI` is the template: elementwise loss, gradient, and hessian, selected from `_get_loss` when the namespace is not NumPy. Gamma is the smaller slice. Tweedie adds `power` and the log versus identity link, including the `power` 0, 1, and 2 limits, which must match the Cython half-deviance that `score()` compares against.

Do not include `newton-cholesky` or `newton-cd` in that pull request.

## Describe alternatives you've considered, if relevant

These lack array API support and their bottlenecks are matrix multiplies or factorizations. They are worth doing later. They are not the first ten.

| Configuration | Why it is later |
| ------------- | --------------- |
| `NMF`, `solver="mu"`, `beta_loss="frobenius"` | The updates are GEMMs. The default solver is Cython coordinate descent. Ten iterations at 5,000 × 1,500, rank 32, float32, finished in 31 ms on the CPU above. KL and sparse branches make the diff much larger than a frobenius-only port. |
| `FastICA`, `algorithm="parallel"`, `whiten_solver="svd"` | Whitening is an SVD and the parallel update is a GEMM. The nonlinearity, decorrelation, and deflation are a second implementation. PCA already whitens on device. |
| `IncrementalPCA` | Batch SVD updates share code with PCA. The estimator exists for data that does not fit in memory; a device copy helps when each batch is large. |
| `PLSRegression` and `CCA` | NIPALS is a sequence of GEMMs. The deflation bookkeeping is easier to get subtly wrong than Ridge's single solve. |
| `HuberRegressor` | IRLS around weighted least squares. It becomes a small follow-on after `LinearRegression`. |
| `SkewedChi2Sampler` | Same projection shape as `RBFSampler`, fewer users. Natural to include in the `RBFSampler` patch or to skip. |
| `SimpleImputer(strategy="mean")`, `VarianceThreshold` | A few reductions. They remove a host sync in pipelines. They are not the compute bottleneck this ranking is about. |
| `GaussianProcessRegressor` | The kernel matrix and its Cholesky are a real target at the same sizes as `KernelRidge`. Hyperparameter optimization is L-BFGS over analytic kernel derivatives, and that derivative code is the maintainability cost. `KernelRidge` captures most of the speedup with a smaller surface. |

The following are what users ask for when they say GPU, and they are not low-hanging array API ports. The hot loop is Cython, libsvm, liblinear, or a tree histogram. An array API port would be a second implementation. #26585 and #26586 already record the spec holes (no scatter-add, no top-k) for the first two.

| Family | Blocker |
| ------ | ------- |
| `KMeans`, `MiniBatchKMeans`, `BisectingKMeans` | Lloyd and Elkan are Cython. A pure array API Lloyd would duplicate them. `GaussianMixture` `init_params="kmeans"` and `"k-means++"` are blocked on this. |
| `KNeighborsClassifier`, `KNeighborsRegressor`, `NearestNeighbors` with `algorithm="brute"` | The fast path is Cython `ArgKmin`. Euclidean, Manhattan, and cosine distances already dispatch. Row-wise top-k is not in the array API. |
| `MLPClassifier`, `MLPRegressor`, `solver="adam"` or `"sgd"` | Hand-written backpropagation. A large NumPy surface for a baseline. |
| `HistGradientBoosting`, `RandomForest`, `GradientBoosting`, `ExtraTrees` | Histograms and tree splits. |
| `SVC`, `SVR`, `LinearSVC`, `LinearSVR` | libsvm and liblinear. |
| `ElasticNet`, `Lasso`, `SGDClassifier`, logistic `sag` / `saga` / `newton-cd` / `liblinear` | Coordinate descent and SAG are Cython. The interesting inputs are often sparse. Dense L2 logistic regression is already covered by `lbfgs` and `newton-cg`. |

`LogisticRegression` `solver="newton-cholesky"` is a dense GEMM (the Hessian) followed by a small Cholesky factor, and the docs recommend it when `n_samples >> n_features`. It is still a poor first port: `NewtonCholeskySolver` is a large NumPy object with multinomial degree-of-freedom constraints and an LBFGS fallback. `lbfgs` and `newton-cg` already run that dense L2 problem on device.

`CalibratedClassifierCV(method="sigmoid")` is Platt scaling and is the default calibration method. `method="temperature"` already dispatches when the base estimator does. `method="isotonic"` is the Cython PAVA. #31869 marks the estimator as hard. Sigmoid is a follow-on once decision scores stay on device, not one of the ten above.

## Additional context

Popularity evidence was collected from two deliberately simple sources:

- constructor calls in distinct files under scikit-learn's `examples/` tree at `d0e22aee7b`;
- authenticated public GitHub code search on October 15, 2026, using exact constructor text, the term `sklearn`, and `language:Python`.

The sampled GitHub searches returned approximately 379,000 matching files for `LinearRegression`, 212,000 for `Ridge`, and 38,000 for `TruncatedSVD`. These are upper bounds rather than telemetry: one file can contain multiple calls, popular repositories can be forked many times, and the `sklearn` term need not refer to the constructor on the same line. They nevertheless separate the two baseline linear estimators from a still-common decomposition estimator by an order of magnitude. For the remaining candidates, default/non-default status and in-tree examples are used instead of manufacturing precision from noisy search counts.

Solver-level gap on estimators the documentation already lists. "Dispatches today" is the configuration the common tests can actually reach.

| Estimator | Dispatches today | Does not |
| --------- | ---------------- | -------- |
| `PCA` | `svd_solver` `"full"`, `"covariance_eigh"`, `"randomized"` with `power_iteration_normalizer="QR"` | `"arpack"`; randomized with `"LU"`, `"none"`, or `"auto"` (`auto` warns and uses QR) |
| `LogisticRegression`, `LogisticRegressionCV` | `"lbfgs"`, `"newton-cg"` | `"newton-cholesky"`, `"newton-cd"`, `"newton-cd-gram"`, `"sag"`, `"saga"`, `"liblinear"` |
| `PoissonRegressor` | `"lbfgs"` | `"newton-cg"`, `"newton-cholesky"`, `"newton-cd"`, `"newton-cd-gram"` |
| `GammaRegressor`, `TweedieRegressor` | nothing | all solvers; default is `"lbfgs"` (item 10) |
| `Ridge`, `RidgeClassifier` | `"svd"`, including `"auto"` rewritten to `"svd"` | `"cholesky"` (the NumPy default), `"lsqr"`, `"sparse_cg"`, `"sag"`, `"saga"`, `"lbfgs"` (item 1) |
| `RidgeCV`, `RidgeClassifierCV` | the LOO GCV path | not a solver knob; already tagged |
| `LinearDiscriminantAnalysis` | `solver="svd"` | `"eigen"` (required for shrinkage) and `"lsqr"` (item 7) |
| `GaussianMixture` | `init_params` `"random"` or `"random_from_data"`, `warm_start=False`, all `covariance_type` values | `"kmeans"`, `"k-means++"`, `warm_start=True` |
| `CalibratedClassifierCV` | `method="temperature"`, and only if the base estimator is tagged | `"sigmoid"` (the default), `"isotonic"` |
| `LabelBinarizer` | `sparse_output=False` | `sparse_output=True` (a SciPy sparse matrix) |

Also already tagged, with no solver knob involved: `LedoitWolf`, `OAS`, `Nystroem`, `GaussianNB`, `Binarizer`, `KernelCenterer`, `LabelEncoder`, `MaxAbsScaler`, `MinMaxScaler`, `Normalizer`, `PolynomialFeatures`, `StandardScaler`, plus `FeatureUnion` and the search meta-estimators when every step is tagged.

Review points if the items become pull requests:

- Ridge's host-side solve needs the maintainer decision already requested on #35060. Do not open that PR while the issue still has a "Needs …" label.
- `LinearRegression.rank_` and `singular_` should be compared against gelsd, not against a normal-equation solver.
- LDA's tag must become conditional so `solver="eigen"` is what the common tests run.
- `EmpiricalCovariance` must not change the float64 `np.cov` path.

CPU timings above are medians after one warmup, from a dedicated micromamba env (`array-api-audit`) with NumPy 2.5.4, SciPy 1.18.1, and OpenBLAS on 4 threads. That machine has no GPU. The L4 numbers are quoted from #35060.

---

This pull request includes code written with the assistance of AI.
The code has **not yet been reviewed** by a human.
