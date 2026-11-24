#!/usr/bin/env python3
"""Phase 0: correctness smoke tests for LHS vs uniform HPO."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401

import numpy as np

from metrics_agg import aggregate_metrics, format_agg_markdown, instance_metrics
from problems import phase0_problems
from run_utils import RESULTS, append_report, run_problem_seeds, save_json, utc_now
from samplers import ParamSpec, sample_search_space


def sanity_check_samplers() -> str:
    specs = [
        ParamSpec("a", "uniform", low=0.0, high=1.0),
        ParamSpec("b", "loguniform", low=1e-3, high=1e3),
        ParamSpec("c", "int", low=1, high=5),
        ParamSpec("d", "choice", choices=("x", "y", "z")),
    ]
    n = 600
    lhs = sample_search_space(specs, n, "lhs", random_state=0)
    uni = sample_search_space(specs, n, "uniform", random_state=1)

    a_lhs = np.array([p["a"] for p in lhs])
    a_uni = np.array([p["a"] for p in uni])
    # Marginals should be roughly Uniform(0,1)
    lhs_mean, uni_mean = float(a_lhs.mean()), float(a_uni.mean())

    # Space-filling: mean nearest-neighbor distance in 2D continuous subspace should be
    # higher for LHS (more spread) for same n — use a,log(b).
    def nn_mean(samples):
        pts = np.column_stack(
            [
                np.array([p["a"] for p in samples]),
                np.log10(np.array([p["b"] for p in samples])),
            ]
        )
        # subsample for speed
        pts = pts[:200]
        dsum = 0.0
        for i in range(len(pts)):
            diff = pts - pts[i]
            dist = np.sqrt((diff**2).sum(axis=1))
            dist[i] = np.inf
            dsum += dist.min()
        return dsum / len(pts)

    nn_lhs, nn_uni = nn_mean(lhs), nn_mean(uni)

    # Discrete marginals roughly balanced for LHS
    from collections import Counter

    c_counts = Counter(p["c"] for p in lhs)
    d_counts = Counter(p["d"] for p in lhs)

    lines = [
        f"## Sampler sanity — {utc_now()}",
        "",
        f"- Continuous mean (a): LHS={lhs_mean:.3f}, Uniform={uni_mean:.3f} (expect ~0.5)",
        f"- Mean NN distance (a, log10 b): LHS={nn_lhs:.4f}, Uniform={nn_uni:.4f} "
        f"(LHS typically ≥ Uniform)",
        f"- LHS int marginals: {dict(sorted(c_counts.items()))}",
        f"- LHS choice marginals: {dict(d_counts)}",
        "",
    ]
    assert abs(lhs_mean - 0.5) < 0.08
    assert abs(uni_mean - 0.5) < 0.08
    assert min(c_counts.values()) >= n // 5 - 30
    return "\n".join(lines)


def main() -> None:
    report = "phase0_progress.md"
    append_report(report, f"\n# Phase 0 progress — started {utc_now()}\n")
    sanity = sanity_check_samplers()
    append_report(report, sanity)
    print(sanity)

    all_rows = []
    for problem in phase0_problems():
        print(f"Running {problem.problem_id}...")
        rows = run_problem_seeds(
            problem,
            seeds=list(range(5)),
            progress_every=5,
            report_name=report,
        )
        all_rows.extend(rows)
        for r in rows:
            assert r["ok"], r
            assert np.isfinite(r["final_uniform"]) and np.isfinite(r["final_lhs"])

    agg = aggregate_metrics(all_rows)
    body = format_agg_markdown(agg, title="Phase 0 aggregate")
    append_report(report, body)
    print(body)
    save_json(RESULTS / "phase0_metrics.json", {"rows": all_rows, "aggregate": agg})
    append_report(report, f"\nPhase 0 complete — {utc_now()}\n")


if __name__ == "__main__":
    main()
