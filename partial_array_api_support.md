# Estimators with partial array API support

Estimators listed on the scikit-learn 1.10.dev0 array API page that accept array
API input only for some parameter values. Reviewed against `main` at
`e4b91c579` (17 Nov 2026).

| Estimator | Missing setting | Difficulty | User value | Tracking |
|---|---|---|---|---|
| `Ridge`, `RidgeClassifier` | `solver="cholesky"` (`auto` is forced to SVD and warns) | Medium. Gram matrix on device, `xp.linalg.solve`. Singular fallback differs per library. | High. SVD on GPU is often slower than NumPy Cholesky on CPU. | [#35060](https://github.com/scikit-learn/scikit-learn/issues/35060), open [PR #35418](https://github.com/scikit-learn/scikit-learn/pull/35418). Older draft [#29318](https://github.com/scikit-learn/scikit-learn/pull/29318). |
| `CalibratedClassifierCV` | `method="sigmoid"` (the default). Isotonic is separate. | Medium. Platt scaling is a two-scalar L-BFGS fit that still uses NumPy. | High. The default method fails under array API dispatch. | [#31869](https://github.com/scikit-learn/scikit-learn/issues/31869). Temperature is done. No PR for sigmoid or isotonic. |
| `GaussianMixture` | `init_params="kmeans"` (the default) and `"k-means++"` | High. Both call NumPy `KMeans`; Lloyd is Cython. | High. The default constructor is rejected. | No GMM issue. Blocked on [RFC #35238](https://github.com/scikit-learn/scikit-learn/issues/35238) and draft [PR #35382](https://github.com/scikit-learn/scikit-learn/pull/35382). |
| `LogisticRegression`, `PoissonRegressor` | `solver="newton-cholesky"` | Medium. Python solver, NumPy Hessian buffers, `scipy.linalg.solve`. Shared implementation. | High for logistic (large-sample L2). Medium for Poisson. | Still open on [meta-issue #33537](https://github.com/scikit-learn/scikit-learn/issues/33537). No dedicated PR. |
| `LogisticRegression` | `newton-cd` / `newton-cd-gram` (what `auto` picks when `l1_ratio > 0`) | High. Inner step is Cython coordinate descent. | High if you need L1 or elastic-net on device. The L2 default already works. | No issue. |

This pull request includes code written with the assistance of AI.
The content has **not yet been reviewed by a human**.
