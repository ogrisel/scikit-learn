#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV:-/tmp/venv-gbdt}"
OUT="${OUT:-$ROOT/benchmarks/hgb_omp_scalability_out}"
export PATH="$VENV/bin:$PATH"
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"
export PYTHONUNBUFFERED=1
# Keep BLAS single-threaded so GBDT thread sweeps are not confounded.
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg
# Allow OpenMP to exceed os.cpu_count() so we can probe the "too many
# default threads on a big machine" regime on this 4-vCPU VM.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}"
# Default libgomp wait policy is ACTIVE; leave unset unless the caller sets it.

mkdir -p "$OUT"

build_sklearn() {
  local src="$1"
  echo "Installing scikit-learn from $src"
  pip install -e "$src" --no-build-isolation
  python -c "import sklearn; from sklearn.utils._openmp_helpers import _openmp_parallelism_enabled; print(sklearn.__file__, sklearn.__version__, 'openmp', _openmp_parallelism_enabled())"
}

CSV="$OUT/results.csv"
rm -f "$CSV"

build_sklearn /tmp/sklearn-main
python "$ROOT/benchmarks/hgb_omp_scalability.py" \
  --sklearn-label main \
  --out-csv "$CSV" \
  --out-meta "$OUT/meta_main.json" \
  --libs sklearn \
  --threads "${THREADS:-4,16}" \
  "$@"

# Other libraries are independent of the sklearn build; run them once.
python "$ROOT/benchmarks/hgb_omp_scalability.py" \
  --sklearn-label main \
  --out-csv "$CSV" \
  --out-meta "$OUT/meta_others.json" \
  --libs xgboost,lightgbm,catboost \
  --threads "${THREADS:-4,16}" \
  "$@"

build_sklearn /tmp/sklearn-pr
python "$ROOT/benchmarks/hgb_omp_scalability.py" \
  --sklearn-label pr34935 \
  --out-csv "$CSV" \
  --out-meta "$OUT/meta_pr34935.json" \
  --libs sklearn \
  --threads "${THREADS:-4,16}" \
  "$@"

python "$ROOT/benchmarks/hgb_omp_scalability_plot.py" "$CSV" --out-dir "$OUT"
echo "Results in $OUT"
