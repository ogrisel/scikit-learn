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
LINESTYLES = {
    "sklearn main": "--",
    "sklearn pr34935": "-",
    "xgboost": "-",
    "lightgbm": "-",
    "catboost": "-",
}
MARKERS = {
    "tiny_stumps": "P",
    "fast_medium": "X",
    "more_trees": ">",
    "wide_boosted": "<",
}
THREAD_HP_PREFERENCE = ("fast_medium", "tiny_stumps", "more_trees", "wide_boosted")


def lib_label(row):
    if row["lib"] == "sklearn":
        return f"sklearn {row['sklearn_label']}"
    return row["lib"]


def format_kmp_blocktime(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return None
    s = s.lower().replace("milliseconds", "").replace("ms", "").strip()
    try:
        n = float(s)
        return str(int(n)) if n == int(n) else str(n)
    except ValueError:
        return str(val).strip()


def row_kmp_blocktime(row):
    """Effective KMP_BLOCKTIME recorded for this measurement, if any."""
    for col in ("kmp_blocktime", "kmp_blocktime_env"):
        if col in row.index:
            value = format_kmp_blocktime(row[col])
            if value is not None:
                return value
    return None


def legend_title(kmp):
    if kmp is None:
        return None
    return f"KMP_BLOCKTIME={kmp}"


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


def hypervolume_min_time_max_auc(front, t_ref, a_ref):
    """2D hypervolume of a min-time / max-AUC staircase vs (t_ref, a_ref)."""
    if front.empty:
        return 0.0
    pts = front.sort_values("fit_seconds_median")
    times = np.concatenate([pts["fit_seconds_median"].to_numpy(), [t_ref]])
    aucs = pts["test_roc_auc_median"].to_numpy()
    hv = 0.0
    for i, auc in enumerate(aucs):
        hv += max(float(auc) - a_ref, 0.0) * max(float(times[i + 1]) - float(times[i]), 0.0)
    return hv


def top_k_frontiers(sub, k=3):
    """Return labels of the k Pareto fronts with largest hypervolume."""
    t_ref = float(sub["fit_seconds_median"].max()) * 1.01 + 1e-9
    a_ref = float(sub["test_roc_auc_median"].min()) - 1e-6
    scores = []
    for label, g in sub.groupby("label"):
        hv = hypervolume_min_time_max_auc(pareto_front(g), t_ref, a_ref)
        scores.append((hv, label))
    scores.sort(reverse=True)
    return [label for _, label in scores[:k]]


def zoom_limits_for_fronts(sub, labels):
    """Smallest axis box that contains the selected models' Pareto fronts."""
    fronts = []
    for label in labels:
        g = sub[sub["label"] == label]
        if g.empty:
            continue
        fronts.append(pareto_front(g))
    if not fronts:
        return None
    pts = pd.concat(fronts)
    t_max = float(pts["fit_seconds_median"].max())
    a_min = float(pts["test_roc_auc_median"].min())
    a_max = float(pts["test_roc_auc_median"].max())
    t_pad = max(t_max * 0.08, 1e-3)
    a_span = max(a_max - a_min, 1e-3)
    return {
        "xlim": (0.0, t_max + t_pad),
        "ylim": (max(0.0, a_min - 0.15 * a_span), min(1.0, a_max + 0.15 * a_span)),
    }


def _plot_pareto(df, out_path, title, zoom=False, top_k=3, kmp=None):
    shapes = list(df["shape"].unique())
    n = len(shapes)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.8 * rows), squeeze=False)
    zoom_notes = []
    for ax, shape in zip(axes.ravel(), shapes):
        sub = df[df["shape"] == shape]
        n_samples = int(sub["n_samples"].iloc[0])
        n_features = int(sub["n_features"].iloc[0])
        top_labels = top_k_frontiers(sub, k=top_k) if zoom else None
        limits = zoom_limits_for_fronts(sub, top_labels) if zoom else None
        if zoom and top_labels:
            zoom_notes.append(f"{shape}: {', '.join(top_labels)}")
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
                ls=LINESTYLES.get(label, "-"),
                lw=2.0 if (top_labels and label in top_labels) else 1.6,
                label=label,
                zorder=4,
            )
        if limits:
            ax.set_xlim(*limits["xlim"])
            ax.set_ylim(*limits["ylim"])
        subtitle = f"{shape}\n({n_samples} x {n_features})"
        if zoom and top_labels:
            subtitle += f"\nzoom: {', '.join(top_labels)}"
        ax.set_title(subtitle)
        ax.set_xlabel("fit time (s)")
        ax.set_ylabel("test ROC AUC")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, loc="lower right", title=legend_title(kmp), title_fontsize=6)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    present = set(df.get("hp_name", pd.Series(dtype=str)))
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
        if name in present
    ]
    if hp_handles:
        fig.legend(
            handles=hp_handles,
            loc="upper center",
            ncol=min(5, len(hp_handles)),
            fontsize=7,
            title="HP setting (markers); lines = Pareto front per model",
            bbox_to_anchor=(0.5, 1.06),
        )
    fig.suptitle(title, y=1.10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    if zoom_notes:
        print("Zoom windows (top-3 hypervolume fronts):")
        for note in zoom_notes:
            print(" ", note)


def _split_by_kmp(df):
    """Yield (kmp_blocktime, frame) pairs, one per effective KMP_BLOCKTIME."""
    if "kmp" not in df.columns:
        return [(None, df)]
    values = sorted(df["kmp"].dropna().unique(), key=lambda v: (len(v), v))
    groups = [(v, df[df["kmp"] == v]) for v in values]
    missing = df[df["kmp"].isna()]
    if len(missing):
        groups.append((None, missing))
    return groups


def _plot_thread_scaling(df, out, suffix, kmp):
    present_hps = set(df["hp_name"])
    hp_for_threads = next(
        (h for h in THREAD_HP_PREFERENCE if h in present_hps),
        df["hp_name"].iloc[0],
    )
    tdf = df[df["hp_name"] == hp_for_threads]
    if tdf["n_threads"].nunique() <= 1:
        return
    kmp_title = "" if kmp is None else f", KMP_BLOCKTIME={kmp}"
    shapes = list(tdf["shape"].unique())
    n = len(shapes)
    cols = 3
    rows = (n + cols - 1) // cols

    def _panels(value_col, ylabel, suptitle, out_name, hline=None):
        fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows), squeeze=False)
        for ax, shape in zip(axes.ravel(), shapes):
            sub = tdf[tdf["shape"] == shape]
            for label, g in sub.groupby("label"):
                g = g.sort_values("n_threads")
                ax.plot(
                    g["n_threads"],
                    g[value_col],
                    marker="o",
                    label=label,
                    color=COLORS.get(label),
                    ls=LINESTYLES.get(label, "-"),
                )
            if hline is not None:
                ax.axhline(hline, color="k", lw=0.8, ls="--")
            ax.set_title(
                f"{shape}\n({int(sub['n_samples'].iloc[0])} x {int(sub['n_features'].iloc[0])})"
            )
            ax.set_xlabel("n_threads")
            ax.set_ylabel(ylabel)
            ax.set_xticks(sorted(sub["n_threads"].unique()))
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, title=legend_title(kmp), title_fontsize=7)
        for ax in axes.ravel()[n:]:
            ax.set_visible(False)
        fig.suptitle(suptitle, y=1.01)
        fig.tight_layout()
        fig.savefig(out / out_name, dpi=140, bbox_inches="tight")
        plt.close(fig)

    _panels(
        "fit_seconds_median",
        "fit time (s)",
        f"Fit time vs threads ({hp_for_threads}{kmp_title})",
        f"fit_time_vs_threads{suffix}.png",
    )
    if "speedup_vs_1" in tdf.columns and tdf["speedup_vs_1"].notna().any():
        _panels(
            "speedup_vs_1",
            "speedup vs 1 thread",
            f"Speedup vs 1 thread ({hp_for_threads}{kmp_title})",
            f"speedup_vs_threads{suffix}.png",
            hline=1.0,
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=Path)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--hps",
        default="",
        help="Optional comma-separated hp_name filter (default: all rows in CSV).",
    )
    args = p.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if "error" in df.columns:
        failed = df[df["error"].notna() & (df["error"].astype(str) != "")]
        if len(failed):
            print("Failed rows:")
            cols = [
                c
                for c in ["sklearn_label", "lib", "hp_name", "shape", "n_threads", "kmp_blocktime", "error"]
                if c in failed
            ]
            print(failed[cols].to_string(index=False))
        df = df[df["error"].isna() | (df["error"].astype(str) == "")]
    df["label"] = df.apply(lib_label, axis=1)
    kmp = df.apply(row_kmp_blocktime, axis=1)
    # Older CSVs have no KMP_BLOCKTIME column: keep a single unlabelled group.
    if kmp.notna().any():
        df["kmp"] = kmp
    if "hp_name" not in df.columns:
        df["hp_name"] = THREAD_HP_PREFERENCE[0]
    if args.hps:
        wanted = {x.strip() for x in args.hps.split(",") if x.strip()}
        df = df[df["hp_name"].isin(wanted)]

    group_keys = [c for c in ["label", "shape", "hp_name", "kmp"] if c in df.columns]
    if 1 in set(df["n_threads"]):
        one = (
            df[df["n_threads"] == 1][group_keys + ["fit_seconds_median"]]
            .rename(columns={"fit_seconds_median": "t1"})
        )
        df = df.merge(one, on=group_keys, how="left")
        df["speedup_vs_1"] = df["t1"] / df["fit_seconds_median"]

    index_cols = [c for c in ["shape", "hp_name", "n_threads", "kmp"] if c in df.columns]
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

    front_keys = [c for c in ["shape", "n_threads", "kmp", "label"] if c in df.columns]
    fronts = []
    for _, g in df.groupby(front_keys):
        front = pareto_front(g).copy()
        front["on_pareto"] = True
        fronts.append(front)
    if fronts:
        pd.concat(fronts).to_csv(out / "pareto_points.csv", index=False)

    # One figure set per KMP_BLOCKTIME: overlaying both values is unreadable.
    for kmp_value, kdf in _split_by_kmp(df):
        suffix = "" if kmp_value is None else f"_kmp{kmp_value}"
        kmp_title = "" if kmp_value is None else f", KMP_BLOCKTIME={kmp_value}"
        for n_threads in sorted(kdf["n_threads"].unique()):
            sub = kdf[kdf["n_threads"] == n_threads]
            _plot_pareto(
                sub,
                out / f"pareto_fit_vs_auc{suffix}_threads_{n_threads}.png",
                f"Fit time vs test ROC AUC (n_threads={n_threads}{kmp_title})"
                "; lines = Pareto front",
                kmp=kmp_value,
            )
            _plot_pareto(
                sub,
                out / f"pareto_fit_vs_auc{suffix}_threads_{n_threads}_zoom.png",
                "Zoomed to top-3 Pareto fronts by hypervolume "
                f"(n_threads={n_threads}{kmp_title})",
                zoom=True,
                top_k=3,
                kmp=kmp_value,
            )
        _plot_thread_scaling(kdf, out, suffix, kmp_value)

    df.to_csv(out / "results_with_speedup.csv", index=False)
    print("Wrote", out)


if __name__ == "__main__":
    main()
