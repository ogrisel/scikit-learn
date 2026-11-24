#!/usr/bin/env python3
"""Phase 2: mixed numeric/categorical pipelines."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401

from metrics_agg import aggregate_metrics, format_agg_markdown
from problems import phase2_problems
from run_utils import RESULTS, append_report, run_pipeline_problem, save_json, utc_now


def main() -> None:
    report = "phase2_progress.md"
    append_report(report, f"\n# Phase 2 progress — started {utc_now()}\n")
    # Fewer seeds on slower OpenML problems; more on synthetic.
    all_rows = []
    for cfg in phase2_problems():
        pid = cfg["problem_id"]
        if pid.startswith("synth"):
            seeds = list(range(15))
        elif pid.startswith("adult"):
            seeds = list(range(8))
        else:
            seeds = list(range(12))
        print(f"\n=== {pid} seeds={len(seeds)} ===", flush=True)
        try:
            rows = run_pipeline_problem(
                cfg,
                seeds=seeds,
                progress_every=3,
                report_name=report,
            )
        except Exception as exc:
            msg = f"\n### FAILED {pid}: {exc!r} — {utc_now()}\n"
            append_report(report, msg)
            print(msg)
            continue
        all_rows.extend(rows)
        agg_p = aggregate_metrics(rows)
        body = format_agg_markdown(agg_p, title=f"Phase 2 done: {pid}")
        append_report(report, body)
        print(body)
        save_json(RESULTS / f"phase2_{pid}.json", {"rows": rows, "aggregate": agg_p})

    agg = aggregate_metrics(all_rows)
    body = format_agg_markdown(agg, title="Phase 2 overall aggregate")
    append_report(report, body)
    print(body)
    save_json(RESULTS / "phase2_metrics.json", {"rows": all_rows, "aggregate": agg})
    append_report(report, f"\nPhase 2 complete — {utc_now()}\n")


if __name__ == "__main__":
    main()
