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
| HonestRF             | low    | True            |           13 |             16 |                   0.957 |                       0.953 |             0.088 |                 0.961 |                0.935 |          0.090 |      16.245 | {"n_estimators": 200, "min_samples_leaf": 20, "max_depth": null, "honest_fraction": 0.4}                   |
| HonestRF             | high   | True            |            2 |             16 |                   0.953 |                       0.945 |             0.442 |                 0.952 |                0.942 |          0.502 |      13.940 | {"n_estimators": 200, "min_samples_leaf": 80, "max_depth": 12, "honest_fraction": 0.4}                     |

## Combined 90% interval on the test set

| family               | constraint_ok   |   n_feasible_low |   n_feasible_high |   cv_half_low |   cv_half_high |   coverage |   mean_width |   pinball_low |   pinball_high |   pinball_sum |   width_oracle_spearman |   winkler | low_params                                                                                                 | high_params                                                                                               |
|:---------------------|:----------------|-----------------:|------------------:|--------------:|---------------:|-----------:|-------------:|--------------:|---------------:|--------------:|------------------------:|----------:|:-----------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------|
| oracle               | True            |          nan     |           nan     |       nan     |        nan     |      0.900 |        5.515 |         0.083 |          0.497 |         0.580 |                   1.000 |    11.607 | nan                                                                                                        | nan                                                                                                       |
| constant_marginal    | True            |          nan     |           nan     |       nan     |        nan     |      0.896 |       13.156 |         0.349 |          0.674 |         1.023 |                   0.000 |    20.468 | nan                                                                                                        | nan                                                                                                       |
| RandomForest         | True            |           16.000 |             8.000 |         0.951 |          0.951 |      0.880 |        5.329 |         0.090 |          0.504 |         0.594 |                   0.956 |    11.883 | {"n_estimators": 200, "min_samples_split": 10, "min_samples_leaf": 30, "max_depth": null}                  | {"n_estimators": 200, "min_samples_split": 2, "min_samples_leaf": 120, "max_depth": 6}                    |
| ExtraTrees           | True            |           16.000 |            16.000 |         0.989 |          0.960 |      0.946 |        6.281 |         0.103 |          0.508 |         0.611 |                   0.972 |    12.215 | {"n_estimators": 200, "min_samples_split": 2, "min_samples_leaf": 9, "max_depth": 12}                      | {"n_estimators": 200, "min_samples_split": 2, "min_samples_leaf": 20, "max_depth": 12}                    |
| GradientBoosting     | True            |            9.000 |             9.000 |         0.952 |          0.952 |      0.896 |        7.410 |         0.187 |          0.511 |         0.698 |                   0.794 |    13.955 | {"subsample": 1.0, "n_estimators": 30, "min_samples_leaf": 20, "max_depth": 3, "learning_rate": 0.1}       | {"subsample": 1.0, "n_estimators": 30, "min_samples_leaf": 20, "max_depth": 3, "learning_rate": 0.1}      |
| HistGradientBoosting | True            |            9.000 |             2.000 |         0.955 |          0.957 |      0.907 |        6.122 |         0.110 |          0.511 |         0.621 |                   0.892 |    12.414 | {"min_samples_leaf": 20, "max_iter": 100, "max_depth": 6, "learning_rate": 0.05, "l2_regularization": 0.0} | {"min_samples_leaf": 40, "max_iter": 50, "max_depth": 3, "learning_rate": 0.05, "l2_regularization": 0.0} |
| HonestRF             | True            |           13.000 |             2.000 |         0.957 |          0.953 |      0.877 |        5.410 |         0.090 |          0.502 |         0.592 |                   0.980 |    11.842 | {"n_estimators": 200, "min_samples_leaf": 20, "max_depth": null, "honest_fraction": 0.4}                   | {"n_estimators": 200, "min_samples_leaf": 80, "max_depth": 12, "honest_fraction": 0.4}                    |

## Sharpest calibrated model class

Among classes that are CV-feasible on both tails **and** have test
coverage ≥ 90%, **HistGradientBoosting** is sharpest (mean width
**6.12**, test coverage 90.7%).

Among all CV-feasible classes (no test-coverage filter), **RandomForest**
is sharpest (mean width **5.33**, test coverage
88.0%).

Ranking by test mean width (CV-feasible only):

| family               |   coverage |   mean_width |   pinball_low |   pinball_high |   pinball_sum |   width_oracle_spearman |   winkler |
|:---------------------|-----------:|-------------:|--------------:|---------------:|--------------:|------------------------:|----------:|
| RandomForest         |      0.880 |        5.329 |         0.090 |          0.504 |         0.594 |                   0.956 |    11.883 |
| HonestRF             |      0.877 |        5.410 |         0.090 |          0.502 |         0.592 |                   0.980 |    11.842 |
| HistGradientBoosting |      0.907 |        6.122 |         0.110 |          0.511 |         0.621 |                   0.892 |    12.414 |
| ExtraTrees           |      0.946 |        6.281 |         0.103 |          0.508 |         0.611 |                   0.972 |    12.215 |
| GradientBoosting     |      0.896 |        7.410 |         0.187 |          0.511 |         0.698 |                   0.794 |    13.955 |

**HonestRF** (per-tree honesty + pinball splits) passed the CV half-coverage floor on both tails (13/16 lower, 2/16 upper). Test coverage 87.7%, mean width 5.41, pinball sum 0.592, Spearman 0.98. The previous global-split MSE HonestRF never found a 95% upper tail; pinball + per-tree honesty does, but the selected pair still slips below 90% on test (same leak as pinball-split RandomForest).

Independent hparams let the 5% and 95% models differ (the gallery
example already suggested that). The half-coverage floor blocks the
pinball-only choice of an inward-biased tail.

`HonestRF` grows each tree on a per-tree bootstrap/honesty split with
`DecisionTreeRegressor(criterion="quantile")` (pinball impurity and
grow-set leaf quantiles), then overwrites leaves with the honest-set
empirical quantile.

This write-up includes work produced with the assistance of AI.
The code has **not yet been reviewed** by a human.
