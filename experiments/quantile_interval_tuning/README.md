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
