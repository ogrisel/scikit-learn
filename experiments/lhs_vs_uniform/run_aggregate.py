#!/usr/bin/env python3
"""Aggregate Phase 0–2 results into a final verdict report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401

from metrics_agg import aggregate_metrics, format_agg_markdown
from run_utils import REPORTS, RESULTS, append_report, save_json, utc_now


def load_rows(path: Path) -> list:
    if not path.exists():
        return []
    with path.open() as f:
        data = json.load(f)
    return data.get("rows", [])


def main() -> None:
    p0 = load_rows(RESULTS / "phase0_metrics.json")
    p1 = load_rows(RESULTS / "phase1_metrics.json")
    p2 = load_rows(RESULTS / "phase2_metrics.json")
    all_rows = p0 + p1 + p2
    # Primary analysis excludes tiny phase-0 smoke if desired; include all but note.
    primary = p1 + p2
    agg_all = aggregate_metrics(all_rows)
    agg_primary = aggregate_metrics(primary)

    # Per-scoring breakdown
    by_scoring = {}
    for sc in sorted({r.get("scoring") for r in primary if r.get("scoring")}):
        by_scoring[sc] = aggregate_metrics([r for r in primary if r.get("scoring") == sc])

    report = "FINAL_REPORT.md"
    # overwrite-style: write fresh file
    path = REPORTS / report
    path.parent.mkdir(parents=True, exist_ok=True)

    mt = agg_primary["mean_speedup_time"]
    lines = [
        f"# LHS vs Uniform HPO — Final Report",
        f"",
        f"_Generated {utc_now()}_",
        f"",
        f"## Question",
        f"",
        f"Can (log-)Latin Hypercube Sampling Pareto-dominate independent (log-)uniform",
        f"sampling on average for randomized hyperparameter search, on the fit-time vs",
        f"predictive-quality tradeoff (strictly proper scoring rules)?",
        f"",
        f"## Method (short)",
        f"",
        f"- Samplers: i.i.d. uniform/loguniform/discrete vs `scipy.stats.qmc.LatinHypercube`",
        f"  with inverse-CDF / stratified mapping.",
        f"- Scores: `neg_log_loss`, `neg_brier_score` (classification);",
        f"  `neg_mean_squared_error` (regression).",
        f"- Target: reach `u_ref + 0.95*(u* - u_ref)` where `u*` is the best score of either",
        f"  method on the instance and `u_ref` is the first uniform evaluation.",
        f"- Speedup: `T_uniform / T_lhs` (and companion n_eval ratio).",
        f"- Uncertainty: percentile bootstrap 95% CIs over problem instances (dataset×model×seed).",
        f"",
        format_agg_markdown(agg_primary, title="Primary aggregate (Phase 1 + 2)"),
        format_agg_markdown(agg_all, title="All phases including Phase 0 smoke"),
        f"## By scoring rule",
        f"",
    ]
    for sc, agg in by_scoring.items():
        lines.append(format_agg_markdown(agg, title=f"Scoring: `{sc}`"))

    # Interpretation
    verdict = agg_primary["verdict"]
    if verdict == "lhs_pareto_dominates_on_average":
        interp = (
            "On average, LHS reached the 95%-of-best target faster with a mean speedup "
            f"of {mt['estimate']:.2f} (95% CI [{mt['low']:.2f}, {mt['high']:.2f}]), and "
            "mid-budget anytime utility favored LHS. This supports average Pareto dominance "
            "under the studied budgets and models."
        )
    elif verdict == "lhs_advantage_mixed_ci_includes_1":
        interp = (
            f"LHS showed a point-estimate mean time-speedup of {mt['estimate']:.2f}, but the "
            f"95% CI [{mt['low']:.2f}, {mt['high']:.2f}] includes 1, so average Pareto "
            "dominance is **not** established at this sample size / setting mix."
        )
    elif verdict == "uniform_better_on_average":
        interp = (
            "Uniform sampling was faster to target on average; LHS did not Pareto-dominate."
        )
    else:
        interp = (
            "Results are inconclusive: no clear average dominance of either sampler "
            "under the defined criteria."
        )

    lines.extend(
        [
            "## Verdict",
            "",
            f"**{verdict}**",
            "",
            interp,
            "",
            "## Caveats",
            "",
            "- Budgets are modest (≈20–40 evaluations); LHS benefits can depend on dimension and budget.",
            "- CV noise affects time-to-target; companion n_eval-speedup is also reported.",
            "- Search spaces mix continuous, log, integer, and categorical dimensions.",
            "- OpenML problems may vary by download/cache; synthetic mixed-type problems are included.",
            "",
            "## Artifact paths",
            "",
            f"- Results JSON: `{RESULTS}`",
            f"- Progress reports: `{REPORTS}`",
            "",
            "This pull request includes code written with the assistance of AI.",
            "The code has **not yet been reviewed** by a human.",
            "",
        ]
    )
    # Note: user asked no PR — disclosure still required for work summaries per AGENTS.md

    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    print(text)
    save_json(
        RESULTS / "final_aggregate.json",
        {"primary": agg_primary, "all": agg_all, "by_scoring": by_scoring},
    )


if __name__ == "__main__":
    main()
