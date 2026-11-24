# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

"""Build the markdown + HTML benchmark report from the JSONL results."""

import glob
import json
import os
import platform
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
RESULTS = sorted(
    glob.glob(os.path.join(OUT, "results", "results_*.jsonl")),
    key=lambda p: ("poisson" in p, p),
)
FAILURES = sorted(glob.glob(os.path.join(OUT, "results", "failures_*.jsonl")))
BACKENDS = ["numpy", "torch-mps", "mlx-gpu"]
LABELS = {"numpy": "NumPy (CPU)", "torch-mps": "PyTorch MPS", "mlx-gpu": "MLX GPU"}
COLORS = {"numpy": "tab:gray", "torch-mps": "tab:orange", "mlx-gpu": "tab:blue"}


def load():
    return pd.concat([pd.read_json(p, lines=True) for p in RESULTS], ignore_index=True)


def load_failures():
    rows = []
    for path in FAILURES:
        with open(path) as f:
            rows += [json.loads(line) for line in f]
    return rows


def plot_times(df, n_features):
    sub = df[df.n_features == n_features]
    estimators = list(dict.fromkeys(sub.estimator))
    ncols = 3
    nrows = -(-len(estimators) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4 * nrows), sharey=True)
    axes = axes.ravel()
    for ax in axes[len(estimators) :]:
        ax.set_visible(False)
    for ax, est in zip(axes, estimators):
        for backend in BACKENDS:
            d = sub[(sub.estimator == est) & (sub.backend == backend)]
            d = d.sort_values("n_samples")
            ax.plot(
                d.n_samples,
                d.fit_time_s,
                "o-",
                label=LABELS[backend],
                color=COLORS[backend],
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axhline(5, color="red", lw=0.8, ls="--")
        ax.set_title(est, fontsize=10)
        ax.set_xlabel("n_samples")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("fit time (s, best of 3)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Fit time, float32, n_features={n_features} (red: 5 s budget)")
    fig.tight_layout()
    path = os.path.join(OUT, f"fit_time_p{n_features}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return os.path.basename(path)


def speedup_table(df):
    rows = []
    for (est, p), g in df.groupby(["estimator", "n_features"], sort=False):
        common = set.intersection(*[set(g[g.backend == b].n_samples) for b in BACKENDS])
        if not common:
            continue
        n = max(common)
        t = {
            b: float(g[(g.backend == b) & (g.n_samples == n)].fit_time_s.iloc[0])
            for b in BACKENDS
        }
        fp = {
            b: float(g[(g.backend == b) & (g.n_samples == n)].peak_footprint_gb.iloc[0])
            for b in BACKENDS
        }
        rows.append(
            dict(
                estimator=est,
                n_features=p,
                n_samples=n,
                numpy_s=t["numpy"],
                torch_mps_s=t["torch-mps"],
                mlx_s=t["mlx-gpu"],
                mlx_vs_mps=t["torch-mps"] / t["mlx-gpu"],
                mlx_vs_numpy=t["numpy"] / t["mlx-gpu"],
                mps_footprint_gb=fp["torch-mps"],
                mlx_footprint_gb=fp["mlx-gpu"],
            )
        )
    return pd.DataFrame(rows)


def plot_family(df, family):
    """Fit time and peak memory vs. n_samples for one estimator family."""
    sub = df[df.estimator.str.startswith(family)]
    combos = [
        (est, p)
        for est in dict.fromkeys(sub.estimator)
        for p in sorted(sub[sub.estimator == est].n_features.unique())
    ]
    fig, axes = plt.subplots(
        2, len(combos), figsize=(3.4 * len(combos), 6), sharex=True
    )
    for j, (est, p) in enumerate(combos):
        for backend in BACKENDS:
            d = sub[
                (sub.estimator == est)
                & (sub.n_features == p)
                & (sub.backend == backend)
            ].sort_values("n_samples")
            kw = dict(color=COLORS[backend], label=LABELS[backend])
            axes[0, j].plot(d.n_samples, d.fit_time_s, "o-", **kw)
            axes[1, j].plot(d.n_samples, d.peak_footprint_gb, "o-", **kw)
        axes[0, j].set_title(f"{est}\np={p}", fontsize=9)
        axes[0, j].set_xscale("log")
        axes[0, j].set_yscale("log")
        axes[0, j].set_ylabel("fit time (s)" if j == 0 else "")
        axes[1, j].set_ylabel("peak footprint (GB)" if j == 0 else "")
        axes[1, j].set_xlabel("n_samples")
        for ax in axes[:, j]:
            ax.grid(True, which="both", alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(OUT, f"{family}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return os.path.basename(path)


def plot_speedup(table):
    fig, ax = plt.subplots(figsize=(12, 4.2))
    labels = [
        f"{r.estimator}\np={r.n_features}, n={r.n_samples:,}"
        for r in table.itertuples()
    ]
    x = range(len(table))
    w = 0.38
    ax.bar(
        [i - w / 2 for i in x],
        table.numpy_s / table.torch_mps_s,
        w,
        label=LABELS["torch-mps"],
        color=COLORS["torch-mps"],
    )
    ax.bar(
        [i + w / 2 for i in x],
        table.mlx_vs_numpy,
        w,
        label=LABELS["mlx-gpu"],
        color=COLORS["mlx-gpu"],
    )
    ax.axhline(1, color="k", lw=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("speedup vs NumPy (CPU)")
    ax.set_title("Speedup over NumPy at the largest size run by all backends")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT, "speedup.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return os.path.basename(path)


def md_table(df, floatfmt):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        cells = []
        for c, v in zip(cols, row):
            if isinstance(v, float):
                cells.append(format(v, floatfmt.get(c, ".3f")))
            elif isinstance(v, int) and c == "n_samples":
                cells.append(f"{v:,}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def versions():
    import mlx.core as mx
    import numpy as np
    import scipy
    import torch

    import sklearn

    return (
        f"Apple M4, 32 GB unified memory, macOS {platform.mac_ver()[0]}; "
        f"Python {platform.python_version()}, NumPy {np.__version__}, "
        f"SciPy {scipy.__version__}, PyTorch {torch.__version__}, "
        f"MLX {mx.__version__}, scikit-learn {sklearn.__version__} (this branch)"
    )


def main():
    os.makedirs(OUT, exist_ok=True)
    df = load()
    table = speedup_table(df)
    figs = {p: plot_times(df, p) for p in sorted(df.n_features.unique())}
    speedup_fig = plot_speedup(table)
    poisson_fig = plot_family(df, "PoissonRegressor")
    failures = load_failures()
    notes_path = os.path.join(os.path.dirname(__file__), "report_notes.md")
    notes = open(notes_path).read() if os.path.exists(notes_path) else ""

    md = [
        "---",
        'title: "MLX array API support in scikit-learn: benchmarks"',
        "---",
        "",
        "> This report includes code written with the assistance of AI.",
        "> The code has **not yet been reviewed** by a human.",
        "",
        f"**Setup**: {versions()}.",
        "",
        notes,
        "",
        "## Summary at the largest size run by all backends",
        "",
        "Times are best-of-3 `fit` wall-clock seconds, after a warm-up fit and "
        "with an explicit device sync (`mx.eval` / `torch.mps.synchronize`). "
        "Footprint is the peak physical footprint of the benchmark process "
        "(includes GPU allocations in unified memory) sampled every 100 ms by "
        "the external watchdog.",
        "",
        md_table(
            table,
            {
                "mlx_vs_mps": ".2f",
                "mlx_vs_numpy": ".2f",
                "mps_footprint_gb": ".2f",
                "mlx_footprint_gb": ".2f",
            },
        ),
        "",
        f"![]({speedup_fig})",
        "",
        "## Fit time vs. number of samples",
        "",
    ]
    for p, fig in figs.items():
        md += [f"![]({fig})", ""]
    md += ["## PoissonRegressor: time and memory", "", f"![]({poisson_fig})", ""]
    md += ["## Raw results", ""]
    raw = df[
        [
            "estimator",
            "backend",
            "n_features",
            "n_samples",
            "fit_time_s",
            "transfer_s",
            "n_iter",
            "n_hessp",
            "train_score",
            "peak_footprint_gb",
        ]
    ].copy()
    for col in ("n_iter", "n_hessp"):
        raw[col] = raw[col].astype("object").where(raw[col].notna(), "")
    md += [
        md_table(
            raw,
            {
                "fit_time_s": ".3f",
                "transfer_s": ".3f",
                "train_score": ".4f",
                "peak_footprint_gb": ".2f",
            },
        ),
        "",
    ]
    if failures:
        md += ["## Killed or failed runs", ""]
        for f in failures:
            md.append(
                f"- {f['estimator']} / {f['backend']} / n={f['n_samples']:,}: "
                f"{f['watchdog'].get('killed') or f['stderr']}"
            )
    md_path = os.path.join(OUT, "report.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    html_path = os.path.join(OUT, "report.html")
    pandoc = os.path.join(os.path.dirname(sys.executable), "pandoc")
    subprocess.run(
        [
            pandoc,
            "report.md",
            "-o",
            "report.html",
            "--standalone",
            "--toc",
            "--metadata",
            "lang=en",
            "-c",
            "style.css",
        ],
        cwd=OUT,
        check=True,
    )
    print(md_path)
    print(html_path)


if __name__ == "__main__":
    main()
