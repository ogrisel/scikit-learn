# Nested CV: half-coverage RSCV quantile interval pairs

Outer loop: shuffled `KFold` on the n=4000 synthetic example.
Inner loop: independent `RandomizedSearchCV` per tail with
the pinball-under-95%-half-coverage `refit` from `constrained_rscv.py`.
Reported metrics are mean ± std over **5 outer folds**.
This estimates the selection procedure, not a single 75/25 split.

## Outer-fold pair metrics (mean ± std)

| family               |   frac_feasible |   n_folds_cov≥90 | coverage      | mean_width     | pinball_sum   | spearman      | winkler        |
|:---------------------|----------------:|-----------------:|:--------------|:---------------|:--------------|:--------------|:---------------|
| oracle               |             1   |                2 | 0.896 ± 0.007 | 5.551 ± 0.135  | 0.524 ± 0.072 | 1.000 ± 0.000 | 10.486 ± 1.440 |
| constant_marginal    |             1   |                2 | 0.899 ± 0.010 | 13.152 ± 0.035 | 0.987 ± 0.060 | 0.000 ± 0.000 | 19.732 ± 1.206 |
| RandomForest         |             0.8 |                5 | 0.913 ± 0.005 | 5.713 ± 0.279  | 0.545 ± 0.072 | 0.967 ± 0.020 | 10.902 ± 1.437 |
| ExtraTrees           |             1   |                5 | 0.945 ± 0.004 | 6.256 ± 0.266  | 0.550 ± 0.066 | 0.973 ± 0.003 | 11.007 ± 1.317 |
| GradientBoosting     |             1   |                5 | 0.912 ± 0.013 | 7.780 ± 0.259  | 0.646 ± 0.073 | 0.833 ± 0.048 | 12.913 ± 1.453 |
| HistGradientBoosting |             1   |                5 | 0.915 ± 0.008 | 6.257 ± 0.375  | 0.565 ± 0.065 | 0.903 ± 0.012 | 11.305 ± 1.298 |
| HonestRF             |             1   |                3 | 0.902 ± 0.007 | 5.489 ± 0.179  | 0.540 ± 0.073 | 0.987 ± 0.003 | 10.794 ± 1.467 |

## Per-fold coverage and width

Coverage:

|   fold |   ExtraTrees |   GradientBoosting |   HistGradientBoosting |   HonestRF |   RandomForest |   constant_marginal |   oracle |
|-------:|-------------:|-------------------:|-----------------------:|-----------:|---------------:|--------------------:|---------:|
|      0 |        0.941 |              0.904 |                  0.906 |      0.891 |          0.911 |               0.886 |    0.902 |
|      1 |        0.941 |              0.906 |                  0.912 |      0.904 |          0.916 |               0.912 |    0.885 |
|      2 |        0.945 |              0.900 |                  0.915 |      0.909 |          0.905 |               0.895 |    0.901 |
|      3 |        0.949 |              0.934 |                  0.927 |      0.897 |          0.912 |               0.895 |    0.897 |
|      4 |        0.950 |              0.915 |                  0.915 |      0.906 |          0.919 |               0.905 |    0.894 |

Mean width:

|   fold |   ExtraTrees |   GradientBoosting |   HistGradientBoosting |   HonestRF |   RandomForest |   constant_marginal |   oracle |
|-------:|-------------:|-------------------:|-----------------------:|-----------:|---------------:|--------------------:|---------:|
|      0 |         6.39 |               7.90 |                   6.23 |       5.44 |           5.50 |               13.11 |     5.55 |
|      1 |         5.86 |               7.52 |                   5.65 |       5.25 |           5.34 |               13.19 |     5.33 |
|      2 |         6.11 |               7.50 |                   6.26 |       5.42 |           5.96 |               13.13 |     5.64 |
|      3 |         6.48 |               8.10 |                   6.53 |       5.67 |           5.95 |               13.15 |     5.68 |
|      4 |         6.44 |               7.88 |                   6.61 |       5.66 |           5.82 |               13.17 |     5.54 |

## Ranking from nested CV

Among classes with **mean** outer-fold coverage ≥ 90%,
**HonestRF** is sharpest (mean width **5.49**, coverage 90.2% ± 0.7%).

- **RandomForest**: 5/5 folds ≥ 90% (mean coverage 91.3% ± 0.5%, width 5.71 ± 0.28).
- **HonestRF**: 3/5 folds ≥ 90% (mean coverage 90.2% ± 0.7%, width 5.49 ± 0.18).
- **HistGradientBoosting**: 5/5 folds ≥ 90% (mean coverage 91.5% ± 0.8%, width 6.26 ± 0.38).

The single 75/25 split ranked HistGradientBoosting as the only CV-feasible class with test coverage ≥ 90%. Nested CV reverses that: RF, HonestRF, HGBT, ExtraTrees, and GBR all have mean coverage ≥ 90%. HonestRF is sharpest on average but undercovers on 2/5 folds; RF and HGBT are ≥ 90% on every fold, with RF narrower than HGBT.

Lowest mean Winkler: **HonestRF** (10.79 ± 1.47).

Selected hyperparameters can change across outer folds; see
`results/nested_cv_tail_metrics.csv`. Feasible inner-CV tails are
not a guarantee of outer-fold coverage ≥ 90%.

This write-up includes work produced with the assistance of AI.
The code has **not yet been reviewed** by a human.
