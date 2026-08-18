# LHS vs Uniform HPO — Final Report

_Generated 2026-08-18 14:47:10 UTC_

## Question

Can (log-)Latin Hypercube Sampling Pareto-dominate independent (log-)uniform
sampling on average for randomized hyperparameter search, on the fit-time vs
predictive-quality tradeoff (strictly proper scoring rules)?

## Method (short)

- Samplers: i.i.d. uniform/loguniform/discrete vs `scipy.stats.qmc.LatinHypercube`
  with inverse-CDF / stratified mapping.
- Scores: `neg_log_loss`, `neg_brier_score` (classification);
  `neg_mean_squared_error` (regression).
- Target: reach `u_ref + 0.95*(u* - u_ref)` where `u*` is the best score of either
  method on the instance and `u_ref` is the first uniform evaluation.
- Speedup: `T_uniform / T_lhs` (and companion n_eval ratio).
- Uncertainty: percentile bootstrap 95% CIs over problem instances (dataset×model×seed).

## Primary aggregate (Phase 1 + 2)

- Instances: **190** (with time-speedup: 97)
- Mean time-speedup (Uniform/LHS): **1.929** 95% CI [1.600, 2.291] (n=97)
- Median time-speedup: **1.468** 95% CI [1.068, 1.928]
- Mean n_eval-speedup: **2.018** 95% CI [1.649, 2.429]
- Fraction speedup>1: **0.660**
- Mean LHS−Uniform utility @25%/50%/100% budget: 0.21887 / 1.12949 / 0.15989
- Verdict: `lhs_pareto_dominates_on_average`

## All phases including Phase 0 smoke

- Instances: **200** (with time-speedup: 104)
- Mean time-speedup (Uniform/LHS): **1.883** 95% CI [1.568, 2.234] (n=104)
- Median time-speedup: **1.434** 95% CI [1.058, 1.895]
- Mean n_eval-speedup: **1.970** 95% CI [1.628, 2.356]
- Fraction speedup>1: **0.635**
- Mean LHS−Uniform utility @25%/50%/100% budget: 0.20787 / 1.07298 / 0.15184
- Verdict: `lhs_pareto_dominates_on_average`

## By scoring rule

## Scoring: `neg_brier_score`

- Instances: **20** (with time-speedup: 14)
- Mean time-speedup (Uniform/LHS): **2.175** 95% CI [1.579, 2.773] (n=14)
- Median time-speedup: **1.990** 95% CI [1.329, 3.009]
- Mean n_eval-speedup: **2.215** 95% CI [1.584, 2.848]
- Fraction speedup>1: **0.857**
- Mean LHS−Uniform utility @25%/50%/100% budget: -0.00000 / 0.00000 / 0.00000
- Verdict: `lhs_pareto_dominates_on_average`

## Scoring: `neg_log_loss`

- Instances: **130** (with time-speedup: 56)
- Mean time-speedup (Uniform/LHS): **2.061** 95% CI [1.549, 2.644] (n=56)
- Median time-speedup: **1.406** 95% CI [0.999, 1.896]
- Mean n_eval-speedup: **2.222** 95% CI [1.630, 2.887]
- Fraction speedup>1: **0.625**
- Mean LHS−Uniform utility @25%/50%/100% budget: -0.00077 / -0.00103 / 0.00033
- Verdict: `no_clear_average_dominance`

## Scoring: `neg_mean_squared_error`

- Instances: **40** (with time-speedup: 27)
- Mean time-speedup (Uniform/LHS): **1.527** 95% CI [1.153, 1.948] (n=27)
- Median time-speedup: **1.064** 95% CI [0.810, 2.042]
- Mean n_eval-speedup: **1.496** 95% CI [1.134, 1.897]
- Fraction speedup>1: **0.630**
- Mean LHS−Uniform utility @25%/50%/100% budget: 1.04214 / 5.36840 / 0.75842
- Verdict: `lhs_pareto_dominates_on_average`

## Verdict

**lhs_pareto_dominates_on_average**

On average, LHS reached the 95%-of-best target faster with a mean speedup of 1.93 (95% CI [1.60, 2.29]), and mid-budget anytime utility favored LHS. This supports average Pareto dominance under the studied budgets and models.

## Caveats

- Budgets are modest (≈20–40 evaluations); LHS benefits can depend on dimension and budget.
- CV noise affects time-to-target; companion n_eval-speedup is also reported.
- Search spaces mix continuous, log, integer, and categorical dimensions.
- OpenML problems may vary by download/cache; synthetic mixed-type problems are included.

## Artifact paths

- Results JSON: `/workspace/experiments/lhs_vs_uniform/results`
- Progress reports: `/workspace/experiments/lhs_vs_uniform/reports`

This pull request includes code written with the assistance of AI.
The code has **not yet been reviewed** by a human.
