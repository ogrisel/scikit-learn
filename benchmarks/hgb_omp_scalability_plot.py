"""Summarize HGB OpenMP / HP-sweep results and plot Pareto fronts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {
    "sklearn main": "#2ca02c",
    "sklearn pr34935": "#d62728",
    "xgboost": "#9467bd",
    "lightgbm": "#ff7f0e",
    "catboost": "#1f77b4",
}
MARKERS = {
    "fast_shallow": "o",
    "defaultish": "s",
    "wide_leaves": "D",
    "many_trees": "^",
    "slow_low_lr": "v",
}


def lib_label(row):
    if row["lib"] == "sklearn":
        return f"sklearn {row['sklearn_label']}"
    return row["lib"]


def pareto_front(g, time_col="fit_seconds_median", auc_col="test_roc_auc_median"):
    """Non-dominated points: lower fit time, higher ROC AUC."""
    pts = g.sort_values([time_col, auc_col], ascending=[True, False])
    keep = []
    best_auc = -np.inf
    for idx, row in pts.iterrows():
        auc = row[auc_col]
        if auc > best_auc:
            keep.append(idx)
            best_auc = auc
    return pts.loc[keep].sort_values(time_col)


def _plot_pareto(df, out_path, title):
    shapes = list(df["shape"].unique())
    n = len(shapes)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.8 * rows), squeeze=False)
    for ax, shape in zip(axes.ravel(), shapes):
        sub = df[df["shape"] == shape]
        n_samples = int(sub["n_samples"].iloc[0])
        n_features = int(sub["n_features"].iloc[0])
        for label, g in sub.groupby("label"):
            color = COLORS.get(label, "gray")
            ax.scatter(
                g["fit_seconds_median"],
                g["test_roc_auc_median"],
                c=color,
                s=36,
                alpha=0.35,
                zorder=2,
            )
            for _, row in g.iterrows():
                m = MARKERS.get(row.get("hp_name", ""), "o")
                ax.scatter(
                    row["fit_seconds_median"],
                    row["test_roc_auc_median"],
                    c=color,
                    marker=m,
                    s=42,
                    edgecolors="k",
                    linewidths=0.4,
                    zorder=3,
                )
            front = pareto_front(g)
            ax.plot(
                front["fit_seconds_median"],
                front["test_roc_auc_median"],
                color=color,
                lw=1.6,
                label=label,
                zorder=4,
            )
        ax.set_title(f"{shape}\n({n_samples} x {n_features})")
        ax.set_xlabel("fit time (s)")
        ax.set_ylabel("test ROC AUC")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, loc="lower right")
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    hp_handles = [
        plt.Line2D(
            [0],
            [0],
            color="k",
            marker=m,
            linestyle="None",
            label=name,
            markersize=6,
        )
        for name, m in MARKERS.items()
        if name in set(df.get("hp_name", pd.Series(dtype=str)))
    ]
    if hp_handles:
        fig.legend(
            handles=hp_handles,
            loc="upper center",
            ncol=len(hp_handles),
            fontsize=8,
            title="HP setting (markers); lines = Pareto front per model",
            bbox_to_anchor=(0.5, 1.04),
        )
    fig.suptitle(title, y=1.08)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=Path)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if "error" in df.columns:
        failed = df[df["error"].notna() & (df["error"].astype(str) != "")]
        if len(failed):
            print("Failed rows:")
            cols = [c for c in ["sklearn_label", "lib", "hp_name", "shape", "n_threads", "error"] if c in failed]
            print(failed[cols].to_string(index=False))
        df = df[df["error"].isna() | (df["error"].astype(str) == "")]
    df["label"] = df.apply(lib_label, axis=1)
    if "hp_name" not in df.columns:
        df["hp_name"] = "defaultish"

    group_keys = [c for c in ["label", "shape", "hp_name"] if c in df.columns]
    if 1 in set(df["n_threads"]):
        one = (
            df[df["n_threads"] == 1][group_keys + ["fit_seconds_median"]]
            .rename(columns={"fit_seconds_median": "t1"})
        )
        df = df.merge(one, on=group_keys, how="left")
        df["speedup_vs_1"] = df["t1"] / df["fit_seconds_median"]

    index_cols = [c for c in ["shape", "hp_name", "n_threads"] if c in df.columns]
    pivot = df.pivot_table(
        index=index_cols,
        columns="label",
        values="fit_seconds_median",
    )
    pivot.to_csv(out / "fit_seconds_pivot.csv")
    print(pivot.to_string())

    if "test_roc_auc_median" in df.columns:
        auc_pivot = df.pivot_table(
            index=index_cols,
            columns="label",
            values="test_roc_auc_median",
        )
        auc_pivot.to_csv(out / "test_roc_auc_pivot.csv")
        print("\nTest ROC AUC:")
        print(auc_pivot.to_string())

    fronts = []
    for (shape, n_threads, label), g in df.groupby(["shape", "n_threads", "label"]):
        front = pareto_front(g)
        front = front.copy()
        front["on_pareto"] = True
        fronts.append(front)
    if fronts:
        pd.concat(fronts).to_csv(out / "pareto_points.csv", index=False)

    thread_levels = sorted(df["n_threads"].unique())
    for n_threads in thread_levels:
        sub = df[df["n_threads"] == n_threads]
        _plot_pareto(
            sub,
            out / f"pareto_fit_vs_auc_threads_{n_threads}.png",
            f"Fit time vs test ROC AUC (n_threads={n_threads}); lines = Pareto front",
        )

    # Thread-scaling plots for the defaultish HP if present, else first HP.
    hp_for_threads = "defaultish" if "defaultish" in set(df["hp_name"]) else df["hp_name"].iloc[0]
    tdf = df[df["hp_name"] == hp_for_threads]
    if tdf["n_threads"].nunique() > 1:
        shapes = list(tdf["shape"].unique())
        n = len(shapes)
        cols = 3
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows), squeeze=False)
        for ax, shape in zip(axes.ravel(), shapes):
            sub = tdf[tdf["shape"] == shape]
            for label, g in sub.groupby("label"):
                g = g.sort_values("n_threads")
                ax.plot(
                    g["n_threads"],
                    g["fit_seconds_median"],
                    marker="o",
                    label=label,
                    color=COLORS.get(label),
                )
            ax.set_title(f"{shape}\n({int(sub['n_samples'].iloc[0])} x {int(sub['n_features'].iloc[0])})")
            ax.set_xlabel("n_threads")
            ax.set_ylabel("fit time (s)")
            ax.set_xticks(sorted(sub["n_threads"].unique()))
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
        for ax in axes.ravel()[n:]:
            ax.set_visible(False)
        fig.suptitle(f"Fit time vs threads ({hp_for_threads})", y=1.01)
        fig.tight_layout()
        fig.savefig(out / "fit_time_vs_threads.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        if "speedup_vs_1" in tdf.columns and tdf["speedup_vs_1"].notna().any():
            fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows), squeeze=False)
            for ax, shape in zip(axes.ravel(), shapes):
                sub = tdf[tdf["shape"] == shape]
                for label, g in sub.groupby("label"):
                    g = g.sort_values("n_threads")
                    ax.plot(
                        g["n_threads"],
                        g["speedup_vs_1"],
                        marker="o",
                        label=label,
                        color=COLORS.get(label),
                    )
                ax.axhline(1.0, color="k", lw=0.8, ls="--")
                ax.set_title(f"{shape}")
                ax.set_xlabel("n_threads")
                ax.set_ylabel("speedup vs 1 thread")
                ax.set_xticks(sorted(sub["n_threads"].unique()))
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=7)
            for ax in axes.ravel()[n:]:
                ax.set_visible(False)
            fig.suptitle(f"Speedup vs 1 thread ({hp_for_threads})", y=1.01)
            fig.tight_layout()
            fig.savefig(out / "speedup_vs_threads.png", dpi=140, bbox_inches="tight")
            plt.close(fig)

    df.to_csv(out / "results_with_speedup.csv", index=False)
    print("Wrote", out)


if __name__ == "__main__":
    main()
