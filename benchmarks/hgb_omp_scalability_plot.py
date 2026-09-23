"""Summarize and plot HGB OpenMP scalability CSV results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def lib_label(row):
    if row["lib"] == "sklearn":
        return f"sklearn {row['sklearn_label']}"
    return row["lib"]


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
            print(failed[["sklearn_label", "lib", "shape", "n_threads", "error"]].to_string(index=False))
        df = df[df["error"].isna() | (df["error"].astype(str) == "")]
    df["label"] = df.apply(lib_label, axis=1)
    df["speedup"] = df.groupby(["label", "shape"], sort=False)["fit_seconds_median"].transform(
        lambda s: s.iloc[0] / s if len(s) else s
    )
    # speedup vs 1-thread of same label+shape
    one = (
        df[df["n_threads"] == 1][["label", "shape", "fit_seconds_median"]]
        .rename(columns={"fit_seconds_median": "t1"})
    )
    df = df.merge(one, on=["label", "shape"], how="left")
    df["speedup_vs_1"] = df["t1"] / df["fit_seconds_median"]

    pivot = df.pivot_table(
        index=["shape", "n_samples", "n_features", "n_threads"],
        columns="label",
        values="fit_seconds_median",
    )
    pivot.to_csv(out / "fit_seconds_pivot.csv")
    print(pivot.to_string())

    shapes = list(df["shape"].unique())
    n = len(shapes)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows), squeeze=False)
    for ax, shape in zip(axes.ravel(), shapes):
        sub = df[df["shape"] == shape]
        for label, g in sub.groupby("label"):
            g = g.sort_values("n_threads")
            ax.plot(g["n_threads"], g["fit_seconds_median"], marker="o", label=label)
        n_samples = int(sub["n_samples"].iloc[0])
        n_features = int(sub["n_features"].iloc[0])
        ax.set_title(f"{shape}\n({n_samples} x {n_features})")
        ax.set_xlabel("n_threads")
        ax.set_ylabel("fit time (s)")
        ax.set_xticks(sorted(sub["n_threads"].unique()))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.suptitle("HistGradientBoosting / GBDT fit time vs thread count", y=1.01)
    fig.tight_layout()
    fig.savefig(out / "fit_time_vs_threads.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows), squeeze=False)
    for ax, shape in zip(axes.ravel(), shapes):
        sub = df[df["shape"] == shape]
        for label, g in sub.groupby("label"):
            g = g.sort_values("n_threads")
            ax.plot(g["n_threads"], g["speedup_vs_1"], marker="o", label=label)
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        n_samples = int(sub["n_samples"].iloc[0])
        n_features = int(sub["n_features"].iloc[0])
        ax.set_title(f"{shape}\n({n_samples} x {n_features})")
        ax.set_xlabel("n_threads")
        ax.set_ylabel("speedup vs 1 thread")
        ax.set_xticks(sorted(sub["n_threads"].unique()))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.suptitle("Speedup vs single-thread (same library)", y=1.01)
    fig.tight_layout()
    fig.savefig(out / "speedup_vs_threads.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # Relative to sklearn main at the same shape/thread
    sk_main = df[(df["lib"] == "sklearn") & (df["sklearn_label"] == "main")][
        ["shape", "n_threads", "fit_seconds_median"]
    ].rename(columns={"fit_seconds_median": "main_t"})
    if len(sk_main):
        rel = df.merge(sk_main, on=["shape", "n_threads"], how="left")
        rel["vs_sklearn_main"] = rel["main_t"] / rel["fit_seconds_median"]
        rel.to_csv(out / "relative_to_sklearn_main.csv", index=False)
        at_default = rel[rel["n_threads"] == rel["cpu_count"]] if "cpu_count" in rel.columns else rel
        print("\nRelative to sklearn main (values >1 are faster than main):")
        print(
            rel.pivot_table(
                index=["shape", "n_threads"],
                columns="label",
                values="vs_sklearn_main",
            ).to_string()
        )

    df.to_csv(out / "results_with_speedup.csv", index=False)
    print("Wrote", out)


if __name__ == "__main__":
    main()
