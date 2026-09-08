# Quantile interval tuning experiments

Scripts that extend the synthetic example in
`examples/ensemble/plot_gradient_boosting_quantile.py` from
[PR #32903](https://github.com/scikit-learn/scikit-learn/pull/32903).

The question is whether GradientBoosting and RandomForest (and ExtraTrees)
pairs of quantile regressors can be tuned to hit **nominal 90% coverage**
while still producing **discriminative** intervals (width tracking the
heteroscedastic noise scale), rather than collapsing or becoming a nearly
constant conservative band.

Run:

```bash
python experiments/quantile_interval_tuning/run_experiments.py
```

Outputs land in `experiments/quantile_interval_tuning/results/`.

See `RESULTS.md` for the first grid and `RESULTS_GBR_COVERAGE.md` for the
GBR-bug vs finite-sample pinball follow-up.

Constrained independent RSCV (pinball under 95% one-sided / "half" coverage):

```bash
python experiments/quantile_interval_tuning/constrained_rscv.py
python experiments/quantile_interval_tuning/constrained_rscv.py --kinds honest_rf
```

Nested CV of that same selection procedure (outer 5-fold, inner 3-fold RSCV):

```bash
python experiments/quantile_interval_tuning/nested_cv.py
```

`HonestRF` (`honest_forest.py`) is a custom forest: each tree draws its own
bootstrap sample, splits in-bag indices into grow vs honest, grows a pinball
(`criterion="quantile"`) tree on the grow set, and sets leaf values to the
honest-set empirical quantile.

`HonestRF` (`honest_forest.py`) is a custom forest: each tree draws its own
bootstrap sample, splits in-bag indices into grow vs honest, grows a pinball
(`criterion="quantile"`) tree on the grow set, and sets leaf values to the
honest-set empirical quantile.

