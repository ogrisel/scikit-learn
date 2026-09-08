# Quantile GB / RF interval tuning

Synthetic data as in the PR #32903 example: `y = x sin(x) + centered log-normal`
with `sigma = 0.5 + x/10` (heteroscedastic, right-skewed).

- Nominal coverage: **90%** (5th–95th percentile pair).
- Leaf-size rule: `min_samples_leaf >= 1/min(alpha, 1-alpha)` = **20**.
- Wall time: 320s.

Metrics: test coverage, mean width (sharpness), Spearman correlation of predicted
width vs oracle width (discriminative power), and Winkler/interval score
(proper scoring rule combining both).

## Example-default models from the PR

| setup     | family              |   coverage |   mean_width |   width_oracle_spearman |   winkler |
|:----------|:--------------------|-----------:|-------------:|------------------------:|----------:|
| n1k_seed0 | GB_example_defaults |      0.868 |        6.927 |                   0.843 |    10.043 |
| n1k_seed0 | RF_example_defaults |      0.832 |        6.547 |                   0.786 |     9.854 |
| n1k_seed0 | constant_marginal   |      0.888 |       12.446 |                   0.000 |    17.190 |
| n1k_seed0 | oracle              |      0.940 |        5.731 |                   1.000 |     8.626 |
| n4k_seed0 | GB_example_defaults |      0.862 |        6.052 |                   0.853 |    12.712 |
| n4k_seed0 | RF_example_defaults |      0.783 |        5.324 |                   0.778 |    13.301 |
| n4k_seed0 | constant_marginal   |      0.896 |       13.156 |                   0.000 |    20.468 |
| n4k_seed0 | oracle              |      0.900 |        5.515 |                   1.000 |    11.607 |

## Best config near nominal coverage (lowest Winkler among |cov-0.90|<=0.03)

### n1k_seed0

| family               |   coverage |   mean_width |   crossing_rate |   pinball_low |   pinball_high |   winkler |   width_oracle_spearman |   max_depth |   min_samples_leaf |   n_estimators |   learning_rate |   subsample |   l2_regularization |   max_iter |
|:---------------------|-----------:|-------------:|----------------:|--------------:|---------------:|----------:|------------------------:|------------:|-------------------:|---------------:|----------------:|------------:|--------------------:|-----------:|
| RandomForest         |      0.928 |        7.471 |           0.000 |         0.091 |          0.407 |     9.964 |                   0.882 |     nan     |             20.000 |        200.000 |         nan     |     nan     |             nan     |    nan     |
| ExtraTrees           |      0.944 |        7.412 |           0.000 |         0.122 |          0.359 |     9.632 |                   0.868 |       6.000 |              1.000 |        200.000 |         nan     |     nan     |             nan     |    nan     |
| GradientBoosting     |      0.872 |        7.368 |           0.000 |         0.150 |          0.375 |    10.494 |                   0.834 |       2.000 |              5.000 |        200.000 |           0.050 |       1.000 |             nan     |    nan     |
| HistGradientBoosting |      0.848 |        5.558 |           0.000 |         0.109 |          0.339 |     8.960 |                   0.772 |       3.000 |             40.000 |        nan     |           0.050 |     nan     |               0.000 |    200.000 |
| HonestRF             |      0.912 |        5.524 |           0.000 |         0.095 |          0.334 |     8.575 |                   0.600 |     nan     |              9.000 |        200.000 |         nan     |     nan     |             nan     |    nan     |
| GB_example_defaults  |      0.868 |        6.927 |           0.000 |         0.144 |          0.358 |    10.043 |                   0.843 |     nan     |            nan     |        nan     |         nan     |     nan     |             nan     |    nan     |
| RF_example_defaults  |      0.832 |        6.547 |           0.000 |         0.087 |          0.406 |     9.854 |                   0.786 |     nan     |            nan     |        nan     |         nan     |     nan     |             nan     |    nan     |
| constant_marginal    |      0.888 |       12.446 |           0.000 |         0.328 |          0.531 |    17.190 |                   0.000 |     nan     |            nan     |        nan     |         nan     |     nan     |             nan     |    nan     |
| oracle               |      0.940 |        5.731 |           0.000 |         0.074 |          0.357 |     8.626 |                   1.000 |     nan     |            nan     |        nan     |         nan     |     nan     |             nan     |    nan     |

### n4k_seed0

| family                     |   coverage |   mean_width |   crossing_rate |   pinball_low |   pinball_high |   winkler |   width_oracle_spearman |   max_depth |   min_samples_leaf |   n_estimators |   learning_rate |   subsample |   l2_regularization |   max_iter |
|:---------------------------|-----------:|-------------:|----------------:|--------------:|---------------:|----------:|------------------------:|------------:|-------------------:|---------------:|----------------:|------------:|--------------------:|-----------:|
| RandomForest               |      0.901 |        5.903 |           0.000 |         0.095 |          0.530 |    12.493 |                   0.908 |     nan     |             50.000 |        200.000 |         nan     |     nan     |             nan     |    nan     |
| ExtraTrees                 |      0.927 |        6.680 |           0.000 |         0.129 |          0.507 |    12.727 |                   0.958 |       6.000 |              1.000 |        200.000 |         nan     |     nan     |             nan     |    nan     |
| GradientBoosting           |      0.868 |        5.963 |           0.000 |         0.132 |          0.507 |    12.781 |                   0.865 |       2.000 |              9.000 |        200.000 |           0.050 |       0.800 |             nan     |    nan     |
| HistGradientBoosting       |      0.879 |        5.245 |           0.000 |         0.089 |          0.512 |    12.022 |                   0.931 |       3.000 |              9.000 |        nan     |           0.050 |     nan     |               1.000 |    200.000 |
| HonestRF                   |      0.881 |        4.745 |           0.000 |         0.093 |          0.519 |    12.238 |                   0.929 |     nan     |             20.000 |        200.000 |         nan     |     nan     |             nan     |    nan     |
| GB_example_defaults        |      0.862 |        6.052 |           0.000 |         0.132 |          0.504 |    12.712 |                   0.853 |     nan     |            nan     |        nan     |         nan     |     nan     |             nan     |    nan     |
| RF_example_defaults        |      0.783 |        5.324 |           0.000 |         0.092 |          0.573 |    13.301 |                   0.778 |     nan     |            nan     |        nan     |         nan     |     nan     |             nan     |    nan     |
| GB_search_pinball          |      0.835 |        5.124 |           0.000 |         0.089 |          0.503 |    11.847 |                   0.934 |     nan     |            nan     |        nan     |         nan     |     nan     |             nan     |    nan     |
| GB_search_pinball_msl_ge20 |      0.851 |        5.208 |           0.000 |         0.090 |          0.513 |    12.047 |                   0.930 |     nan     |            nan     |        nan     |         nan     |     nan     |             nan     |    nan     |
| RF_search_pinball          |      0.860 |        5.532 |           0.000 |         0.088 |          0.511 |    11.967 |                   0.937 |     nan     |            nan     |        nan     |         nan     |     nan     |             nan     |    nan     |
| RF_search_pinball_msl_ge20 |      0.858 |        5.614 |           0.000 |         0.088 |          0.524 |    12.237 |                   0.917 |     nan     |            nan     |        nan     |         nan     |     nan     |             nan     |    nan     |
| constant_marginal          |      0.896 |       13.156 |           0.000 |         0.349 |          0.674 |    20.468 |                   0.000 |     nan     |            nan     |        nan     |         nan     |     nan     |             nan     |    nan     |
| oracle                     |      0.900 |        5.515 |           0.000 |         0.083 |          0.497 |    11.607 |                   1.000 |     nan     |            nan     |        nan     |         nan     |     nan     |             nan     |    nan     |

## RandomForest: min_samples_leaf vs coverage (max_depth=None, n=4000 seed0)

|   min_samples_leaf |   coverage |   mean_width |   width_oracle_spearman |   winkler |
|-------------------:|-----------:|-------------:|------------------------:|----------:|
|              1.000 |      0.000 |        0.000 |                   0.000 |    37.861 |
|              5.000 |      0.695 |        4.027 |                   0.667 |    14.433 |
|              9.000 |      0.783 |        5.324 |                   0.778 |    13.301 |
|             20.000 |      0.849 |        5.878 |                   0.861 |    12.695 |
|             30.000 |      0.872 |        5.850 |                   0.877 |    12.713 |
|             50.000 |      0.901 |        5.903 |                   0.908 |    12.493 |

## Takeaways

See `RESULTS.md` for the full interpretation.

- RF pinball-split with `min_samples_leaf=50` (n=4000) hits 90.1% coverage
  with Spearman 0.91. The `1/α=20` rule is a floor; gallery `msl=9` undercovers.
- Honest RF `msl=40` is the best RF-family point (90.5%, Spearman 0.96).
- GBR never exceeded 86.8% coverage on this grid; pinball search made
  undercoverage worse while improving Winkler.
- HGBT reached 88.8% with shallow depth, L2, and large leaves.

This pull request includes code written with the assistance of AI.
The code has **not yet been reviewed** by a human.
