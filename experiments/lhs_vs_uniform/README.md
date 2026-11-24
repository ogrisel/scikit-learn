# LHS vs Uniform Hyperparameter Sampling Experiments

Reproducible study comparing (log-)Latin Hypercube Sampling to independent
(log-)uniform sampling for randomized HPO in scikit-learn, under strictly
proper scoring rules.

## Quick start

```bash
pip install "scikit-learn>=1.5" scipy pandas matplotlib joblib
cd experiments/lhs_vs_uniform
python3 run_phase0.py
python3 run_phase1.py
python3 run_phase2.py
python3 run_aggregate.py
```

Scripts bootstrap away from the unbuilt in-repo `sklearn/` source tree and use
the installed package.

## Reports

- `reports/FINAL_REPORT.md` — verdict, speedups, CIs
- `reports/PROGRESS_SUMMARY.md` — running summary
- `reports/phase*_progress.md` — detailed progress logs
- `results/*.json` — per-instance metrics (gitignored; regenerable)

## No PR

This branch is for sharing experimental results; it is not intended as a
library change PR unless follow-up work decides otherwise.

## Tests

```bash
cd experiments/lhs_vs_uniform
python3 -m pytest test_samplers.py -v
```
