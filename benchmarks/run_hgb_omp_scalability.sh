#!/usr/bin/env bash
# Thin wrapper. Prefer: cd benchmarks/hgb_omp_scalability && pixi run bench
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HERE="$(cd "$(dirname "$0")" && pwd)"
PIXI_DIR="$HERE/hgb_omp_scalability"

if [[ -f "$PIXI_DIR/pixi.toml" ]] && command -v pixi >/dev/null 2>&1; then
  cd "$PIXI_DIR"
  exec pixi run bench "$@"
fi

# Fallback: existing Python on PATH and pre-fetched sklearn trees.
OUT="${HGB_BENCH_OUT:-$HERE/hgb_omp_scalability_out}"
export PYTHONUNBUFFERED=1
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
export SKLEARN_MAIN="${SKLEARN_MAIN:-$PIXI_DIR/trees/main}"
export SKLEARN_PR="${SKLEARN_PR:-$PIXI_DIR/trees/pr34935}"
python "$PIXI_DIR/prepare.py" bench "$@"
echo "Results in $OUT"
