# Experiment progress summary

## Phase 0 — complete
Sampler sanity checks passed (balanced LHS margins; LHS more space-filling). Smoke HPO returned finite proper scores.

## Phase 1 — complete (2026-08-18 ~14:34 UTC)

**Overall (140 instances = 7 problems × 20 seeds):**
- Mean time-speedup (Uniform/LHS): **2.00**, 95% CI **[1.61, 2.45]** (n=79 paired reaches)
- Median time-speedup: **1.47**, 95% CI **[1.06, 1.99]**
- Mean n_eval-speedup: **2.10**, 95% CI **[1.67, 2.60]**
- Fraction speedup>1: **0.67**
- Mean LHS−Uniform utility @25/50/100% budget: 0.30 / 1.53 / 0.22
- Verdict: `lhs_pareto_dominates_on_average`

| Problem | Mean speedup | 95% CI | Notes |
|---------|-------------:|--------|-------|
| breast_cancer_logreg | 1.78 | [1.31, 2.28] | dominates |
| breast_cancer_hgb | 3.15 | [1.15, 5.63] | sparse reaches; utility ≈ tied |
| digits_logreg | 2.52 | [1.12, 4.21] | mixed mid-budget utility |
| digits_hgb | (see JSON) | — | LHS often better at full budget |
| diabetes_ridge | 1.50 | [1.10, 1.98] | dominates; clear MSE utility gain |
| diabetes_hgb | 1.59 | [0.81, 2.40] | CI includes 1 |
| breast_cancer_logreg_brier | 2.18 | [1.58, 2.77] | dominates (secondary proper score) |

## Phase 2 — starting
Mixed numeric/categorical pipelines (synthetic + OpenML credit-g / adult subsample).

This pull request includes code written with the assistance of AI.
The code has **not yet been reviewed** by a human.
