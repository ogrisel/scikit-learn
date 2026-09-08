# GBR undercoverage: bug check vs finite-sample pinball

## Verdict

**Not an implementation bug.** sklearn GBR quantile does Friedman’s GBM line search:
trees are grown with `squared_error` on the negative pinball gradient, then each leaf
value is replaced by the sample quantile of residuals
(`PinballLoss.fit_intercept_only`), then multiplied by `learning_rate`
(`sklearn/ensemble/_gb.py` `_update_terminal_regions`). HGBT does the same
override (`_update_leaves_values`) because the pinball Hessian is zero a.e.

A one-step GBR (`n_estimators=1`, `learning_rate=1`) has train/test
`P(Y < q̂_{0.05}) ≈ 0.05`. If leaf values were wrong, that check would fail.

The severe test undercoverage is **overfitting past the coverage-optimal
complexity**, which **cross-validated pinball actively selects**.

## What the new diagnostics show

Gallery-like GBR (`lr=0.05`, `max_depth=2`, `min_samples_leaf=9`) on `n=4000`:

| trees | train cov | test cov | test width | test pinball sum |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.90 | 0.90 | 13.0 | 1.01 |
| 50 | 0.92 | **0.90** | 9.1 | 0.77 |
| 200 | 0.91 | 0.86 | 6.1 | 0.64 |
| 300 | 0.91 | 0.86 | 5.4 | **0.61** |

- After one tree the pair is a slightly adapted *marginal* interval: calibrated,
  not discriminative.
- Test coverage is still nominal around 50 trees.
- More trees keep **cutting test pinball** (especially the 5% model) while test
  coverage falls. Train coverage stays ~90% (`P(Y<q̂_{0.05})` and `P(Y>q̂_{0.95})`
  stay near 5%). On test both tails move **inward** (7.6% below the lower bound,
  6.2% above the upper at 200 trees).
- Pinball CV has no reason to stop at 50 trees: the proper score still improves.

Same GBR hparams vs `n`:

| n | train cov | test cov |
| ---: | ---: | ---: |
| 500 | 0.92 | 0.78 |
| 4000 | 0.91 | 0.86 |
| 16000 | 0.91 | **0.89** |

The train/test coverage gap shrinks with `n`. That is the finite-sample
signature, not a constant code bias. (RF with `min_samples_leaf=9` does **not**
recover with `n`; leaf size is the RF analogue of “too little mass per
quantile estimate”, and it does not grow with `n`.)

## Pinball-tuned RF also undercovers

The previous write-up overstated the RF vs GBR contrast for *pinball CV*.

On the same `n=4000` split, independent pinball `RandomizedSearchCV` for RF
chose `min_samples_leaf=20` (lower) and `50` + `max_depth=6` (upper) → **85.8%**
test coverage. Joint `min_samples_leaf=50` on both sides → **90.1%** coverage
and *worse* test pinball (0.625 vs 0.612). Pinball prefers the slightly
too-narrow pair.

So RF is **not** immune to pinball-CV undercoverage. What RF has, and GBR
does not, is a **monotone, interpretable capacity knob** (`min_samples_leaf`)
that moves coverage from collapse (`msl=1` → 0%) through nominal (`msl≈50` at
this `n`) to overcoverage. You can target 90% by that knob. GBR’s matching
knob is early stopping / small `n_estimators`; pinball CV will not pick it
because pinball is still falling when coverage has already left 90%.

RF leaf predictions are empirical quantiles of ≥`msl` observations (local CDF
weights, close to Meinshausen 2006). Averaging trees reduces variance of that
order statistic; it does not remove inward bias if `msl` is too small for
α=0.05 (need ≳1/α points, and in practice more). GBR is an *additive path*
that can keep fitting the 5% envelope after the interval is already too tight
on unseen data.

Honest RF (structure on one half, quantiles on the other) is the forest
analogue of not using the same samples to choose the partition and the
quantile — the same reason Athey, Tibshirani & Wager (2019) use honesty for
GRF.

## Literature (this is a known limitation)

1. **Romano, Patterson, Candès (NeurIPS 2019), “Conformalized Quantile Regression”.**
   Quantile-regression intervals from pinball minimization are valid only
   under regularity/asymptotics; in finite samples they routinely
   **undercover**. CQR exists specifically to restore finite-sample coverage
   around *any* quantile estimator (including GB and RF).
   https://arxiv.org/abs/1905.03222

2. **Koenker & Bassett (1978).** Pinball consistency is for the conditional
   quantile functional, not a finite-sample guarantee that
   `P(q̂_{α/2}(X) ≤ Y ≤ q̂_{1-α/2}(X)) = 1-α` for estimated `q̂`.

3. **Meinshausen (JMLR 2006), Quantile Regression Forests.** Consistency of
   forest-weighted empirical CDFs; no finite-sample coverage. sklearn’s
   pinball-split RF is a different estimator (quantile in each tree, then
   average) and inherits the usual order-statistic bias for small leaves.

4. **Athey, Tibshirani, Wager (Ann. Statist. 2019), Generalized Random Forests.**
   Honesty (separate samples for splits vs leaf estimates) to get
   asymptotically unbiased forest functionals, including quantiles.

5. **sklearn itself:** gallery example, issues
   [#8905](https://github.com/scikit-learn/scikit-learn/issues/8905) and
   [#18849](https://github.com/scikit-learn/scikit-learn/issues/18849).
   The same ~80% coverage on this example was reported in 2017; after
   checking train vs test and LightGBM, it was treated as usage/overfitting,
   not a bad leaf formula. LightGBM showed the same pattern.

6. Pinball is a **proper score for one quantile**. Independently minimizing
   pinball at 5% and 95% is *asymptotically* the right interval; with
   flexible models and small `n` it is a sharpness-first criterion and will
   select undercovering pairs (as in the staged table).

Practical implication for the example: tune GBR with **early stopping on
held-out coverage** (or conformalize, CQR), not pinball alone. For RF, put a
floor on `min_samples_leaf` well above `1/min(α,1-α)` and do not expect
independent pinball searches on the two tails to land there.

This write-up includes work produced with the assistance of AI.
The code has **not yet been reviewed** by a human.
