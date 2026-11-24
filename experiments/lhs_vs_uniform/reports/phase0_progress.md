
# Phase 0 progress — started 2026-08-18 14:03:35 UTC
## Sampler sanity — 2026-08-18 14:03:35 UTC

- Continuous mean (a): LHS=0.500, Uniform=0.497 (expect ~0.5)
- Mean NN distance (a, log10 b): LHS=0.0984, Uniform=0.0880 (LHS typically ≥ Uniform)
- LHS int marginals: {1: 120, 2: 120, 3: 120, 4: 120, 5: 120}
- LHS choice marginals: {'y': 200, 'x': 200, 'z': 200}

### Progress — synth_clf_logreg (5/5 seeds) — 2026-08-18 14:03:38 UTC

## Running aggregate (synth_clf_logreg)

- Instances: **5** (with time-speedup: 2)
- Mean time-speedup (Uniform/LHS): **0.678** 95% CI [0.610, 0.746] (n=2)
- Median time-speedup: **0.678** 95% CI [0.610, 0.746]
- Mean n_eval-speedup: **0.857** 95% CI [0.714, 1.000]
- Fraction speedup>1: **0.000**
- Mean LHS−Uniform utility @25%/50%/100% budget: -0.00234 / -0.00112 / -0.00211
- Verdict: `uniform_better_on_average`

### Progress — breast_cancer_logreg (5/5 seeds) — 2026-08-18 14:03:57 UTC

## Running aggregate (breast_cancer_logreg)

- Instances: **5** (with time-speedup: 5)
- Mean time-speedup (Uniform/LHS): **1.476** 95% CI [0.426, 3.047] (n=5)
- Median time-speedup: **0.433** 95% CI [0.424, 4.358]
- Mean n_eval-speedup: **1.473** 95% CI [0.446, 2.884]
- Fraction speedup>1: **0.400**
- Mean LHS−Uniform utility @25%/50%/100% budget: -0.00000 / -0.00000 / 0.00000
- Verdict: `lhs_advantage_mixed_ci_includes_1`
## Phase 0 aggregate

- Instances: **10** (with time-speedup: 7)
- Mean time-speedup (Uniform/LHS): **1.248** 95% CI [0.499, 2.351] (n=7)
- Median time-speedup: **0.610** 95% CI [0.426, 1.737]
- Mean n_eval-speedup: **1.297** 95% CI [0.560, 2.308]
- Fraction speedup>1: **0.286**
- Mean LHS−Uniform utility @25%/50%/100% budget: -0.00117 / -0.00056 / -0.00106
- Verdict: `lhs_advantage_mixed_ci_includes_1`

Phase 0 complete — 2026-08-18 14:03:57 UTC
