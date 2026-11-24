# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

"""Sweep benchmark sizes, one watchdog-guarded process per configuration."""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
SIZES = [10_000, 30_000, 100_000, 300_000, 1_000_000]
TIME_CAP_S = 5.0
EXTRAPOLATION_CAP_S = 7.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/tmp/mlx_bench")
    parser.add_argument("--estimators", nargs="+", required=True)
    parser.add_argument(
        "--backends", nargs="+", default=["numpy", "torch-mps", "mlx-gpu"]
    )
    parser.add_argument("--n-features", type=int, default=200)
    parser.add_argument("--max-footprint-gb", type=float, default=8)
    parser.add_argument("--min-available-gb", type=float, default=5)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--max-data-gb", type=float, default=1.5)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    results = os.path.join(args.out_dir, "results.jsonl")
    failures = os.path.join(args.out_dir, "failures.jsonl")

    for estimator in args.estimators:
        for backend in args.backends:
            prev = None
            for n_samples in SIZES:
                data_gb = n_samples * args.n_features * 4 / 1024**3
                if data_gb > args.max_data_gb:
                    print(
                        f"skip {estimator} {backend} n={n_samples}: "
                        f"X would use {data_gb:.1f} GB",
                        flush=True,
                    )
                    break
                if prev is not None:
                    predicted = prev[1] * n_samples / prev[0]
                    if predicted > EXTRAPOLATION_CAP_S:
                        print(
                            f"skip {estimator} {backend} n={n_samples}: "
                            f"predicted {predicted:.1f}s",
                            flush=True,
                        )
                        break
                report = os.path.join(args.out_dir, "watchdog_last.json")
                cmd = [
                    PY,
                    os.path.join(HERE, "watchdog.py"),
                    "--timeout",
                    str(args.timeout),
                    "--max-footprint-gb",
                    str(args.max_footprint_gb),
                    "--min-available-gb",
                    str(args.min_available_gb),
                    "--report",
                    report,
                    "--",
                    PY,
                    os.path.join(HERE, "bench_worker.py"),
                    "--estimator",
                    estimator,
                    "--backend",
                    backend,
                    "--n-samples",
                    str(n_samples),
                    "--n-features",
                    str(args.n_features),
                    "--out",
                    results,
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                with open(report) as f:
                    wd = json.load(f)
                if proc.returncode != 0:
                    err = (proc.stderr or "").strip().splitlines()[-3:]
                    rec = dict(
                        estimator=estimator,
                        backend=backend,
                        n_samples=n_samples,
                        watchdog=wd,
                        stderr=err,
                    )
                    with open(failures, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    print(
                        f"FAIL {estimator} {backend} n={n_samples}: "
                        f"{wd['killed'] or err}",
                        flush=True,
                    )
                    break
                rec = json.loads(proc.stdout.strip().splitlines()[-1])
                # Attach the host-side peak footprint to the last result line.
                with open(results) as f:
                    lines = f.readlines()
                last = json.loads(lines[-1])
                last["peak_footprint_gb"] = wd["peak_footprint_gb"]
                lines[-1] = json.dumps(last) + "\n"
                with open(results, "w") as f:
                    f.writelines(lines)
                print(
                    f"ok   {estimator:30s} {backend:10s} n={n_samples:>9,d} "
                    f"fit={rec['fit_time_s']:.3f}s footprint="
                    f"{wd['peak_footprint_gb']:.2f}GB score={rec['train_score']:.4f}",
                    flush=True,
                )
                prev = (n_samples, rec["fit_time_s"])
                if rec["fit_time_s"] > TIME_CAP_S:
                    break


if __name__ == "__main__":
    main()
