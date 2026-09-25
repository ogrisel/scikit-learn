"""Fetch sklearn trees, build them in the pixi env, and run the HGB bench.

Intended to be invoked via ``pixi run`` from this directory so compilers,
OpenMP, and GBDT libraries come from the lockfile.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH_DIR = HERE.parent
REPO_ROOT = BENCH_DIR.parent
TREES = HERE / "trees"
MAIN_DIR = TREES / "main"
PR_DIR = TREES / "pr34935"
OUT_DIR = Path(os.environ.get("HGB_BENCH_OUT", BENCH_DIR / "hgb_omp_scalability_out"))
BENCH_PY = BENCH_DIR / "hgb_omp_scalability.py"
PLOT_PY = BENCH_DIR / "hgb_omp_scalability_plot.py"

MAIN_URL = os.environ.get("SKLEARN_MAIN_URL", "https://github.com/scikit-learn/scikit-learn.git")
MAIN_REF = os.environ.get("SKLEARN_MAIN_REF", "main")
PR_URL = os.environ.get("SKLEARN_PR_URL", "https://github.com/cakedev0/scikit-learn.git")
PR_REF = os.environ.get("SKLEARN_PR_REF", "hgb/active_wait")


DEFAULT_KMP_BLOCKTIMES = "0,200"


def run(cmd, **kwargs):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.check_call(cmd, **kwargs)


def git_clone_or_update(url: str, ref: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not (dest / ".git").exists():
        if dest.exists():
            shutil.rmtree(dest)
        run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--single-branch",
                "--branch",
                ref,
                url,
                str(dest),
            ]
        )
        return
    run(["git", "remote", "set-url", "origin", url], cwd=dest)
    env = os.environ.copy()
    env.setdefault("GIT_CONFIG_COUNT", "1")
    env.setdefault("GIT_CONFIG_KEY_0", "advice.detachedHead")
    env.setdefault("GIT_CONFIG_VALUE_0", "false")
    run(["git", "fetch", "--depth", "1", "origin", ref], cwd=dest, env=env)
    run(["git", "checkout", "-f", "FETCH_HEAD"], cwd=dest, env=env)


def cmd_prepare(_args) -> None:
    git_clone_or_update(MAIN_URL, MAIN_REF, MAIN_DIR)
    git_clone_or_update(PR_URL, PR_REF, PR_DIR)
    print(f"sklearn main: {MAIN_DIR}")
    print(f"sklearn PR:   {PR_DIR}")


def _verify_openmp() -> None:
    code = (
        "import sklearn; "
        "from sklearn.utils._openmp_helpers import _openmp_parallelism_enabled, "
        "_openmp_effective_n_threads; "
        "print(sklearn.__file__, sklearn.__version__, "
        "'openmp', _openmp_parallelism_enabled(), "
        "'n_threads', _openmp_effective_n_threads())"
    )
    run([sys.executable, "-c", code])


def cmd_build(args) -> None:
    which = args.target
    src = MAIN_DIR if which == "main" else PR_DIR
    if not (src / "pyproject.toml").exists():
        raise SystemExit(f"{src} is missing; run: pixi run prepare")
    env = os.environ.copy()
    # Prefer conda compilers already on PATH from pixi activation.
    run(
        [sys.executable, "-m", "pip", "install", "-e", str(src), "--no-build-isolation"],
        env=env,
    )
    _verify_openmp()


def _kmp_blocktimes():
    raw = os.environ.get("KMP_BLOCKTIMES", DEFAULT_KMP_BLOCKTIMES)
    return [x.strip() for x in raw.split(",") if x.strip()]


def _bench_cmd(sklearn_label: str, libs: str, extra: list[str], kmp_blocktime: str) -> list[str]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv = OUT_DIR / "results.csv"
    meta = OUT_DIR / f"meta_{sklearn_label}_kmp{kmp_blocktime}.json"
    threads = os.environ.get("THREADS", "4,10")
    cmd = [
        sys.executable,
        str(BENCH_PY),
        "--sklearn-label",
        sklearn_label,
        "--out-csv",
        str(csv),
        "--out-meta",
        str(meta),
        "--libs",
        libs,
    ]
    if not any(a == "--threads" or a.startswith("--threads=") for a in extra):
        cmd.extend(["--threads", threads])
    cmd.extend(extra)
    return cmd


def _run_bench(sklearn_label: str, libs: str, extra: list[str]) -> None:
    """One subprocess per KMP_BLOCKTIME so libomp picks it up at init."""
    for bt in _kmp_blocktimes():
        env = os.environ.copy()
        env["KMP_BLOCKTIME"] = bt
        print(f"+ KMP_BLOCKTIME={bt} (effective for this process)", flush=True)
        run(_bench_cmd(sklearn_label, libs, extra, bt), env=env)


def cmd_bench(args) -> None:
    if not args.others_only:
        cmd_prepare(args)
    extra = args.extra or []
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv = OUT_DIR / "results.csv"
    if args.sklearn_only is None and not args.others_only:
        csv.unlink(missing_ok=True)

    if args.others_only:
        _run_bench("others", "xgboost,lightgbm,catboost", extra)
        return

    if args.sklearn_only in (None, "main"):
        cmd_build(argparse.Namespace(target="main"))
        _run_bench("main", "sklearn", extra)

    if args.sklearn_only is None:
        _run_bench("main", "xgboost,lightgbm,catboost", extra)

    if args.sklearn_only in (None, "pr"):
        cmd_build(argparse.Namespace(target="pr"))
        _run_bench("pr34935", "sklearn", extra)

    if args.sklearn_only is None:
        cmd_plot(args)


def cmd_plot(_args) -> None:
    csv = OUT_DIR / "results.csv"
    if not csv.exists():
        raise SystemExit(f"missing {csv}; run: pixi run bench")
    run([sys.executable, str(PLOT_PY), str(csv), "--out-dir", str(OUT_DIR)])
    print("Results in", OUT_DIR)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cmd", choices=["prepare", "build", "bench", "plot"])
    p.add_argument("target", nargs="?", choices=["main", "pr"])
    p.add_argument("--sklearn-only", choices=["main", "pr"], default=None)
    p.add_argument("--others-only", action="store_true")
    args, extra = p.parse_known_args()
    extra = [e for e in extra if e != "--"]

    if args.cmd == "prepare":
        cmd_prepare(args)
    elif args.cmd == "build":
        if args.target is None:
            raise SystemExit("build requires target: main or pr")
        cmd_build(args)
    elif args.cmd == "bench":
        args.extra = extra
        cmd_bench(args)
    else:
        cmd_plot(args)


if __name__ == "__main__":
    main()
