# HGB OpenMP scalability: PR 34935 vs `main` vs XGBoost / LightGBM / CatBoost

This run compares `HistGradientBoostingClassifier` on
[scikit-learn/scikit-learn#34935](https://github.com/scikit-learn/scikit-learn/pull/34935)
(`hgb/active_wait` @ `6ea3ffac5c`) to sklearn `main` @ `af6b2ded95`, and to
XGBoost 3.4.1, LightGBM 4.7.0 and CatBoost 1.2.10.

Each library is run with **five matched hyperparameter settings** (early
stopping off). For every dataset we plot **fit time vs held-out ROC AUC** and
draw the **Pareto front** (non-dominated: faster or more accurate) per model.

## Hardware and OpenMP

| Item | Value |
| --- | --- |
| Host | 4 vCPU KVM Intel Xeon, 1 socket, no SMT |
| sklearn | built with OpenMP (`libgomp`), active wait |
| `OMP_NUM_THREADS` | 32 |
| Threads measured | **4** (physical cores) and **16** (surplus / oversubscription) |
| BLAS | 1 thread |
| Timing | 1 warmup + 1 timed `fit` |
| Test metric | binary ROC AUC on a 50% hold-out (`n_test = n_train`) |

## Matched hyperparameter grid

| `hp_name` | `n_estimators` | `max_leaf_nodes` | `learning_rate` |
| --- | ---: | ---: | ---: |
| `fast_shallow` | 20 | 8 | 0.3 |
| `defaultish` | 40 | 31 | 0.1 |
| `wide_leaves` | 50 | 63 | 0.1 |
| `many_trees` | 200 | 15 | 0.05 |
| `slow_low_lr` | 100 | 63 | 0.03 |

Shared: `max_bin=255`, binary log-loss, `early_stopping=False`,
`random_state=0`. CatBoost still uses `l2_leaf_reg=3`, `depth=16`,
`grow_policy=Lossguide` (`max_leaves` is capped at 64 by CatBoost, hence 63
rather than 127).

## Fit time vs ROC AUC (Pareto)

Markers are HP settings; **lines are the Pareto front** of that model
(maximize AUC, minimize fit time). sklearn `main` is dashed so it remains
visible when it overlaps the PR.

### 4 threads (physical cores)

![pareto 4 threads](pareto_fit_vs_auc_threads_4.png)

- **sklearn `main` and PR 34935 overlap**: same AUC for every HP, fit times
  within a few percent. The thread-cap heuristic does not change the trees.
- **LightGBM** usually owns the upper-left front on small/narrow data
  (`fast_shallow` is cheapest; `wide_leaves` is the high-AUC corner).
- **XGBoost** sits next to LightGBM / sklearn; slightly slower than LightGBM
  on most fronts, similar AUC.
- **sklearn HGB** is competitive on medium-wide / 50k×50: `fast_shallow` is
  among the fastest, `wide_leaves` matches the others at ~0.985 AUC.
- **CatBoost** is shifted right (slower) at matched trees/leaves. On tiny
  data it can sit slightly above the others in AUC; on 2k×128 it is both
  slower and weaker (`fast_shallow` 0.881 vs ~0.90–0.91).
- `slow_low_lr` and often `many_trees` are **dominated**: more fit time
  without a useful AUC gain vs `wide_leaves` or `defaultish`.

### 16 threads (surplus OpenMP)

![pareto 16 threads](pareto_fit_vs_auc_threads_16.png)

Same AUCs as at 4 threads (trees do not change). What moves is **fit time**:

- **sklearn `main` shifts right** on every shape — same quality, much more
  time. Tiny 1k×10 `wide_leaves`: 0.080 s @ 4 threads → **1.03 s @ 16**.
  `defaultish`: 0.059 s → **0.66 s**.
- **PR 34935 stays put on tiny data** (`fast_shallow` 0.014 s, `wide_leaves`
  0.054 s at both 4 and 16) because the heuristic pins tiny problems to 1
  thread. On larger shapes it still oversubscribes this 4-core VM, but less
  than `main`, so **its front strictly dominates `main`** (equal AUC, lower
  time).
- **XGBoost** fronts barely move (tiny `fast_shallow` 0.0075 → 0.0074 s).
- **LightGBM** degrades on small data (tiny `defaultish` 0.021 s @ 4 → 0.277 s
  @ 16) but not as badly as sklearn `main`.
- **CatBoost** is again almost unchanged from 4 to 16 threads.

## Thread scaling at `defaultish` (40 / 31 / 0.1)

![fit time vs threads](fit_time_vs_threads.png)

| shape | threads | sklearn main | sklearn PR | XGBoost | LightGBM | CatBoost | AUC (sklearn) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny 1k×10 | 4 | 0.059 | 0.039 | 0.041 | **0.021** | 0.092 | 0.958 |
| tiny 1k×10 | 16 | 0.663 | **0.039** | 0.041 | 0.277 | 0.090 | 0.958 |
| small 5k×20 | 4 | 0.079 | 0.064 | 0.070 | **0.043** | 0.189 | 0.984 |
| small 5k×20 | 16 | 0.837 | 0.353 | **0.071** | 0.351 | 0.190 | 0.984 |
| medium 50k×50 | 4 | 0.231 | **0.228** | 0.327 | 0.276 | 0.692 | 0.969 |
| medium 50k×50 | 16 | 1.069 | 0.790 | **0.327** | 0.657 | 0.710 | 0.969 |

AUC is identical for sklearn `main` and the PR at every HP. XGBoost/LightGBM
stay within ~0.007 of sklearn except CatBoost on 2k×128.

## How to rerun

```bash
export PATH=/path/to/venv/bin:$PATH CC=gcc CXX=g++
bash benchmarks/run_hgb_omp_scalability.sh
```

This pull request includes code written with the assistance of AI.
The code has **not yet been reviewed** by a human.
