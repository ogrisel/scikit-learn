# Experiment progress summary

## Phase 0 — complete
Sampler sanity checks passed. Smoke HPO returned finite proper scores.

## Phase 1 — complete
Overall (140 instances): mean time-speedup **2.00**, 95% CI **[1.61, 2.45]**; verdict `lhs_pareto_dominates_on_average`.

## Phase 2 — complete
Mixed numeric/categorical pipelines (50 instances): mean time-speedup **1.62**, 95% CI **[1.06, 2.29]**; anytime utility slightly favored uniform; verdict `no_clear_average_dominance` for Phase 2 alone.

| Problem | Mean speedup | Notes |
|---------|-------------:|-------|
| synth_mixed_hgb | 1.38 | mixed; utility slightly favors uniform |
| synth_mixed_logreg | 1.78 | dominates |
| credit_g_hgb | 1.94 | sparse reaches; utility ≈ tied/slightly uniform |
| adult_small_hgb | 1.30 | sparse reaches |

## Final pooled (Phase 1+2)
- Mean time-speedup: **1.93**, 95% CI **[1.60, 2.29]**
- Median: **1.47**, 95% CI **[1.07, 1.93]**
- Mean n_eval-speedup: **2.02**, 95% CI **[1.65, 2.43]**
- Fraction speedup>1: **0.66**
- Verdict: **`lhs_pareto_dominates_on_average`** (with Phase-2 caveats above)

See `FINAL_REPORT.md` for full write-up.

This pull request includes code written with the assistance of AI.
The code has **not yet been reviewed** by a human.
