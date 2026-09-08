# Independent pinball RSCV under a one-sided (half) coverage constraint

Each quantile regressor is tuned with `RandomizedSearchCV` and a custom
`refit` callable (same mechanism as `plot_grid_search_digits.py`):

- Lower α=0.05: feasible iff mean CV **P(Y ≥ q̂) ≥ 95%**, then min pinball.
- Upper α=0.95: feasible iff mean CV **P(Y ≤ q̂) ≥ 95%**, then min pinball.

The two independently selected models form the 90% interval. If both
one-sided constraints hold, fold-level interval miscoverage is at most
10% by a union bound.

Dataset: n=4000 synthetic example from PR #32903 (train 3000 / test 1000).

## Per-tail CV selection and test half-coverage

| family               | tail   | constraint_ok   |   n_feasible |   n_candidates |   cv_mean_half_coverage |   cv_min_fold_half_coverage |   cv_mean_pinball |   train_half_coverage |   test_half_coverage |   test_pinball |   elapsed_s | best_params                                                                                                |
|:---------------------|:-------|:----------------|-------------:|---------------:|------------------------:|----------------------------:|------------------:|----------------------:|---------------------:|---------------:|------------:|:-----------------------------------------------------------------------------------------------------------|
| RandomForest         | low    | True            |           16 |             24 |                   0.951 |                       0.947 |             0.087 |                 0.966 |                0.938 |          0.090 |      25.360 | {"n_estimators": 200, "min_samples_split": 10, "min_samples_leaf": 30, "max_depth": null}                  |
| RandomForest         | high   | True            |            8 |             24 |                   0.951 |                       0.939 |             0.440 |                 0.954 |                0.942 |          0.504 |      21.123 | {"n_estimators": 200, "min_samples_split": 2, "min_samples_leaf": 120, "max_depth": 6}                     |
| ExtraTrees           | low    | True            |           16 |             16 |                   0.989 |                       0.985 |             0.107 |                 0.993 |                0.988 |          0.103 |      18.548 | {"n_estimators": 200, "min_samples_split": 2, "min_samples_leaf": 9, "max_depth": 12}                      |
| ExtraTrees           | high   | True            |           16 |             16 |                   0.960 |                       0.952 |             0.438 |                 0.964 |                0.958 |          0.508 |      18.317 | {"n_estimators": 200, "min_samples_split": 2, "min_samples_leaf": 20, "max_depth": 12}                     |
| GradientBoosting     | low    | True            |            9 |             40 |                   0.952 |                       0.948 |             0.194 |                 0.971 |                0.948 |          0.187 |      11.386 | {"subsample": 1.0, "n_estimators": 30, "min_samples_leaf": 20, "max_depth": 3, "learning_rate": 0.1}       |
| GradientBoosting     | high   | True            |            9 |             40 |                   0.952 |                       0.945 |             0.438 |                 0.956 |                0.948 |          0.511 |      12.039 | {"subsample": 1.0, "n_estimators": 30, "min_samples_leaf": 20, "max_depth": 3, "learning_rate": 0.1}       |
| HistGradientBoosting | low    | True            |            9 |             36 |                   0.955 |                       0.950 |             0.111 |                 0.967 |                0.958 |          0.110 |       8.190 | {"min_samples_leaf": 20, "max_iter": 100, "max_depth": 6, "learning_rate": 0.05, "l2_regularization": 0.0} |
| HistGradientBoosting | high   | True            |            2 |             36 |                   0.957 |                       0.952 |             0.437 |                 0.958 |                0.949 |          0.511 |       8.884 | {"min_samples_leaf": 40, "max_iter": 50, "max_depth": 3, "learning_rate": 0.05, "l2_regularization": 0.0}  |
| HonestRF             | low    | True            |           13 |             16 |                   0.958 |                       0.945 |             0.092 |                 0.966 |                0.954 |          0.095 |       7.513 | {"n_estimators": 200, "min_samples_leaf": 9, "max_depth": 6, "honest_fraction": 0.5}                       |
| HonestRF             | high   | False           |            0 |             16 |                   0.935 |                       0.930 |             0.439 |                 0.934 |                0.918 |          0.519 |       7.451 | {"n_estimators": 200, "min_samples_leaf": 20, "max_depth": null, "honest_fraction": 0.5}                   |

## Combined 90% interval on the test set

| family               | constraint_ok   |   n_feasible_low |   n_feasible_high |   cv_half_low |   cv_half_high |   coverage |   mean_width |   width_oracle_spearman |   winkler | low_params                                                                                                 | high_params                                                                                               |
|:---------------------|:----------------|-----------------:|------------------:|--------------:|---------------:|-----------:|-------------:|------------------------:|----------:|:-----------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------|
| oracle               | True            |          nan     |           nan     |       nan     |        nan     |      0.900 |        5.515 |                   1.000 |    11.607 | nan                                                                                                        | nan                                                                                                       |
| constant_marginal    | True            |          nan     |           nan     |       nan     |        nan     |      0.896 |       13.156 |                   0.000 |    20.468 | nan                                                                                                        | nan                                                                                                       |
| RandomForest         | True            |           16.000 |             8.000 |         0.951 |          0.951 |      0.880 |        5.329 |                   0.956 |    11.883 | {"n_estimators": 200, "min_samples_split": 10, "min_samples_leaf": 30, "max_depth": null}                  | {"n_estimators": 200, "min_samples_split": 2, "min_samples_leaf": 120, "max_depth": 6}                    |
| ExtraTrees           | True            |           16.000 |            16.000 |         0.989 |          0.960 |      0.946 |        6.281 |                   0.972 |    12.215 | {"n_estimators": 200, "min_samples_split": 2, "min_samples_leaf": 9, "max_depth": 12}                      | {"n_estimators": 200, "min_samples_split": 2, "min_samples_leaf": 20, "max_depth": 12}                    |
| GradientBoosting     | True            |            9.000 |             9.000 |         0.952 |          0.952 |      0.896 |        7.410 |                   0.794 |    13.955 | {"subsample": 1.0, "n_estimators": 30, "min_samples_leaf": 20, "max_depth": 3, "learning_rate": 0.1}       | {"subsample": 1.0, "n_estimators": 30, "min_samples_leaf": 20, "max_depth": 3, "learning_rate": 0.1}      |
| HistGradientBoosting | True            |            9.000 |             2.000 |         0.955 |          0.957 |      0.907 |        6.122 |                   0.892 |    12.414 | {"min_samples_leaf": 20, "max_iter": 100, "max_depth": 6, "learning_rate": 0.05, "l2_regularization": 0.0} | {"min_samples_leaf": 40, "max_iter": 50, "max_depth": 3, "learning_rate": 0.05, "l2_regularization": 0.0} |
| HonestRF             | False           |           13.000 |             0.000 |         0.958 |          0.935 |      0.872 |        4.743 |                   0.927 |    12.275 | {"n_estimators": 200, "min_samples_leaf": 9, "max_depth": 6, "honest_fraction": 0.5}                       | {"n_estimators": 200, "min_samples_leaf": 20, "max_depth": null, "honest_fraction": 0.5}                  |

## Sharpest calibrated model class

The CV constraint is one-sided and slightly leaky on test (each tail near
95% compounds to interval coverage ≈ half_low + half_high − 1).

**Best class that is CV-feasible on both tails and ≥90% on test:**
**HistGradientBoosting** — test coverage 90.7%, mean width 6.12 (oracle 5.52),
Spearman 0.89. The 5% model is more flexible (`max_iter=100`, `max_depth=6`,
`min_samples_leaf=20`); the 95% model is more regularized (`max_iter=50`,
`max_depth=3`, `min_samples_leaf=40`).

**RandomForest** is sharper (width 5.33, Spearman 0.96, best Winkler 11.88)
but test coverage is 88.0%: both tails passed CV at 95.1% and slipped to
93.8%/94.2% on test (`msl=30` unbounded vs `msl=120`, `max_depth=6`).

**ExtraTrees** overcovers (94.6%) with a conservative lower tail (test
half-coverage 98.8%) so the band is wider (6.28) despite excellent Spearman
(0.97).

**GradientBoosting** meets the floor only with **30 trees**; the interval is
almost nominal (89.6%) but wide (7.41) and less discriminative (0.79). That
matches the staged-boosting result: pinball-feasible coverage lives at
early stopping, not at 200 trees.

**HonestRF** never found a 95% upper tail in this grid (best CV half-coverage
93.5%); the fallback pair is the sharpest of all (4.74) and undercovers
(87.2%).

Ranking by test mean width (both tails CV-feasible):

| family               |   coverage |   mean_width |   width_oracle_spearman |   winkler |
|:---------------------|-----------:|-------------:|------------------------:|----------:|
| RandomForest         |      0.880 |        5.329 |                   0.956 |    11.883 |
| HistGradientBoosting |      0.907 |        6.122 |                   0.892 |    12.414 |
| ExtraTrees           |      0.946 |        6.281 |                   0.972 |    12.215 |
| GradientBoosting     |      0.896 |        7.410 |                   0.794 |    13.955 |

Independent hparams let the 5% and 95% models differ (the gallery
example already suggested that). The half-coverage floor blocks the
pinball-only choice of an inward-biased tail; GBR is forced to few trees,
RF/HGB to larger leaves / less boosting on the upper tail.

This write-up includes work produced with the assistance of AI.
The code has **not yet been reviewed** by a human.
