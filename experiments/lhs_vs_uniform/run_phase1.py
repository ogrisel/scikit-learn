#!/usr/bin/env python3
"""Phase 1: multi-seed systematic experiments on simple models/datasets."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401

from metrics_agg import aggregate_metrics, format_agg_markdown
from problems import phase1_problems
from run_utils import RESULTS, append_report, run_problem_seeds, save_json, utc_now


def main() -> None:
    report = "phase1_progress.md"
    append_report(report, f"\n# Phase 1 progress — started {utc_now()}\n")
    seeds = list(range(20))
    all_rows = []
    for problem in phase1_problems():
        print(f"\n=== {problem.problem_id} ({problem.scoring}) ===", flush=True)
        rows = run_problem_seeds(
            problem,
            seeds=seeds,
            progress_every=5,
            report_name=report,
        )
        all_rows.extend(rows)
        agg_p = aggregate_metrics(rows)
        body = format_agg_markdown(agg_p, title=f"Phase 1 done: {problem.problem_id}")
        append_report(report, body)
        print(body)
        save_json(RESULTS / f"phase1_{problem.problem_id}.json", {"rows": rows, "aggregate": agg_p})

    agg = aggregate_metrics(all_rows)
    body = format_agg_markdown(agg, title="Phase 1 overall aggregate")
    append_report(report, body)
    print(body)
    save_json(RESULTS / "phase1_metrics.json", {"rows": all_rows, "aggregate": agg})
    append_report(report, f"\nPhase 1 complete — {utc_now()}\n")


if __name__ == "__main__":
    main()
