# Results: can GB/RF quantile pairs hit 90% coverage *and* be discriminative?

Setup is the PR #32903 gallery example: \(y = x\sin(x)\) plus centered log-normal
noise with \(\sigma(x)=0.5+x/10\) (heteroscedastic, right-skewed). Nominal
interval is the 5th–95th percentile pair. Oracle quantiles are known in closed
form.

Two sample sizes: \(n=1000\) (gallery) and \(n=4000\) (coverage SE ≈ 1pp).
Metrics on a 25% test split:

- **coverage** (target 90%)
- **mean width** (sharpness; oracle ≈ 5.5 on \(n=4000\))
- **Spearman(pred width, oracle width)** (does the band track heteroscedasticity?)
- **Winkler / interval score** (proper score; pinball@5% + pinball@95% is equivalent)

The constant marginal 5%/95% quantiles of \(y\) (ignore \(X\)) are a coverage
baseline with **zero** discriminative power.

## Headline

**Random forests: yes, if leaves are large enough — larger than `1/min(α,1-α)`.**
On \(n_\text{train}=3000\), pinball-split RF with `min_samples_leaf=50` hits
90.1% coverage, mean width 5.90 (oracle 5.52), Spearman 0.91.

**`min_samples_leaf >= 20` is a necessary floor, not a sufficient target.**
With unbounded depth:

| min_samples_leaf | coverage | mean width | Spearman | Winkler |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.000 | 0.00 | 0.00 | 37.9 |
| 9 (gallery default) | 0.783 | 5.32 | 0.78 | 13.3 |
| 20 (`1/α` rule) | 0.849 | 5.88 | 0.86 | 12.7 |
| 30 | 0.872 | 5.85 | 0.88 | 12.7 |
| 50 | **0.901** | 5.90 | **0.91** | **12.5** |

`min_samples_leaf=1` collapses the interval (each leaf quantile is a single
training point; averaging trees does not remove that bias). The gallery
`min_samples_leaf=9` undercovers badly once \(n\) is large enough to see it.

**Honest RF** (MSE splits on one half, empirical leaf quantiles on the other)
is the cleanest RF-family point: `min_samples_leaf=40` → 90.5% coverage, width
5.41, Spearman **0.96**. Width vs \(x\) is a stable increasing step function;
pinball-split RF@50 matches coverage but the upper band is much more jagged
(log-normal tail + same-sample quantile estimates).

**Gradient boosting: not with tree-growth / pinball tuning alone.** Every
GBR grid point on \(n=4000\) sat in **81.5–86.8%** coverage. Slowing the
learning rate (`lr=0.01`, 800 trees, `min_samples_leaf=40`) stayed at 86.8%.
One-sided rates show the 5% model is the main problem (`P(y ≥ q̂_0.05) ≈ 0.93`
vs target 0.95); the 95% model is only slightly low. That matches the gallery
remark that the lower quantile underfits the sinusoid.

**Pinball CV makes coverage worse, not better.** Independent
`RandomizedSearchCV` on pinball (the gallery strategy) selected sharper
models:

| method | coverage | width | Spearman | Winkler |
| --- | ---: | ---: | ---: | ---: |
| oracle | 0.900 | 5.52 | 1.00 | 11.61 |
| GB pinball search | 0.835 | 5.12 | 0.93 | **11.85** |
| GB pinball + `msl≥20` | 0.851 | 5.21 | 0.93 | 12.05 |
| RF pinball search | 0.860 | 5.53 | 0.94 | 11.97 |
| HGBT `max_depth=3`, `msl=40`, `l2=1` | 0.888 | 5.25 | 0.90 | 12.15 |
| RF `msl=50` | 0.901 | 5.90 | 0.91 | 12.49 |
| Honest RF `msl=40` | 0.905 | 5.41 | 0.96 | 12.51 |
| constant marginal | 0.896 | 13.16 | 0.00 | 20.47 |

Winkler *prefers* the undercovering GB search (11.85, near oracle 11.61)
because the interval is tighter. Nominal coverage and pinball/Winkler are
not the same objective here. Constraining `min_samples_leaf ≥ 20` in the
search only recovered ~2pp of coverage.

Buying GB coverage by fitting 3%/97% instead of 5%/95% reached 91.1% but
blew the mean width up to 8.37 — discriminative but not sharp.

**HistGradientBoosting** is the best boosting cousin (88.8% with shallow
trees + L2 + large leaves) but still short of nominal.

**ExtraTrees** with `min_samples_leaf≥9` *overcovers* (~96%) and is wider;
`max_depth=6`, `min_samples_leaf=1` can sit near 93% with high Spearman but
that leaf size is unsafe in general (unbounded depth + `msl=1` collapses).

## Implications for PR #32903 docs / example

1. Recommend `min_samples_leaf ≥ 1/min(α,1-α)` as a **minimum**, then tune
   **upward** until test coverage is close to nominal. On this problem the
   operating point was ~2.5× that floor.
2. Do **not** imply that pinball CV yields calibrated intervals. It yields
   good Winkler scores and good width-vs-\(x\) correlation, with systematic
   undercoverage for GBR and for RF with gallery defaults.
3. Honest leaf estimates are worth mentioning (or exemplifying): same
   coverage as a large-leaf pinball forest, smoother scale function.
4. GBR quantile pairs on this example should be described as *approximately*
   calibrated even after a grid; HGBT is closer; RF/honest RF can be put on
   the nominal line.

Plots: `results/pareto_coverage_width.png`, `results/width_vs_x.png`,
`results/intervals_RF_msl50.png`, `results/intervals_HonestRF_msl40.png`,
`results/intervals_GB_best_grid.png`. Full grid: `results/metrics.csv`.

This write-up includes work produced with the assistance of AI.
The code has **not yet been reviewed** by a human.
