# Reproduce the HGB OpenMP scalability benchmark

This pixi workspace installs compilers, OpenMP, scikit-learn build deps, and
XGBoost / LightGBM / CatBoost from conda-forge, then builds **sklearn `main`**
and **[PR 34935](https://github.com/scikit-learn/scikit-learn/pull/34935)**
from git and runs `benchmarks/hgb_omp_scalability.py`.

## One-time setup

Install [pixi](https://pixi.sh/) (`curl -fsSL https://pixi.sh/install.sh | bash`).

From a clone of this branch:

```bash
cd benchmarks/hgb_omp_scalability
pixi install
pixi run bench
```

That will:

1. Clone sklearn `main` and `cakedev0/scikit-learn` branch `hgb/active_wait`
   into `./trees/`
2. `pip install -e` each tree **with OpenMP** using the pixi compilers
3. Sweep HPs × data shapes × thread counts × ``KMP_BLOCKTIME`` in
   ``{0, 200}`` for sklearn main, XGBoost, LightGBM, CatBoost, then sklearn
   PR 34935 (one process per ``KMP_BLOCKTIME`` so llvm-openmp sees it)
4. Write CSV + Pareto plots to `../hgb_omp_scalability_out/`

Re-plot an existing CSV:

```bash
pixi run plot
```

## Useful knobs

| Variable / flag | Meaning | Default |
| --- | --- | --- |
| `THREADS` | comma-separated OpenMP / GBDT thread counts | `4,10` (M4 P-cores, all physical cores) |
| `KMP_BLOCKTIMES` | llvm-openmp spin waits (ms), one process each | `0,200` (`0` is the Apple Silicon default) |
| `OMP_NUM_THREADS` | upper bound seen by sklearn | `10` (set in `pixi.toml`) |
| `HGB_BENCH_OUT` | output directory | `../hgb_omp_scalability_out` |
| `SKLEARN_MAIN_URL` / `SKLEARN_MAIN_REF` | sklearn main tree | GitHub `scikit-learn/scikit-learn` `main` |
| `SKLEARN_PR_URL` / `SKLEARN_PR_REF` | PR tree | `cakedev0/scikit-learn` `hgb/active_wait` |

Forward extra argparse flags after `--`:

```bash
THREADS=4,10 pixi run bench -- --repeats 2 --warmup 1
KMP_BLOCKTIMES=0 pixi run bench
pixi run bench -- --shapes tiny_1k_x_10,small_5k_x_20 --hps tiny_stumps,fast_medium
```

## `KMP_BLOCKTIME`

`pixi run bench` always runs **two** llvm-openmp wait settings, in separate
processes (libomp reads `KMP_BLOCKTIME` at init):

| Value | Meaning |
| --- | --- |
| `0` | Apple Silicon / hybrid-CPU llvm-openmp default (no post-region spin) |
| `200` | historical 200 ms spin wait |

Plot legends show the **effective** `KMP_BLOCKTIME` recorded for that process
(`KMP_BLOCKTIME=0` vs `KMP_BLOCKTIME=200`). Override the pair with
`KMP_BLOCKTIMES=0` or `KMP_BLOCKTIMES=200`.

## Active vs passive OpenMP wait

conda-forge `libgomp` often exports `OMP_WAIT_POLICY=PASSIVE` from
`etc/conda/activate.d`. PR 34935 branches its thread heuristic on that.

```bash
pixi run -e active-wait bench
pixi run -e passive-wait bench
```

`HGB_BENCH_OUT` should be different for each so CSVs do not overwrite:

```bash
HGB_BENCH_OUT=$PWD/out-active pixi run -e active-wait bench
HGB_BENCH_OUT=$PWD/out-passive pixi run -e passive-wait bench
```

## Other machines / platforms

The committed `pixi.lock` is for **linux-64**. On macOS or Windows, from this
directory:

```bash
pixi project platform add osx-arm64   # or osx-64, win-64
pixi install
pixi run bench
```

Building sklearn needs a working C/C++ toolchain; pixi pulls
`c-compiler` / `cxx-compiler` from conda-forge (`libgomp` on Linux,
`llvm-openmp` on macOS). Confirm OpenMP after a build:

```text
.../sklearn/__init__.py 1.10.dev0 openmp True n_threads 10
```

If `openmp False`, the HGB thread sweep is meaningless.

## Partial runs

```bash
pixi run prepare          # git clone/update only
pixi run build-main       # editable install sklearn main
pixi run sklearn-main     # sklearn main sweep only
pixi run others           # XGBoost / LightGBM / CatBoost only
pixi run sklearn-pr       # sklearn PR 34935 sweep only
```

This pull request includes code written with the assistance of AI.
The code has **not yet been reviewed** by a human.
