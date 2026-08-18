"""Shared runner utilities for phased HPO experiments."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from evaluate import SearchRun, run_paired_search
from metrics_agg import aggregate_metrics, format_agg_markdown, instance_metrics
from problems import Problem, build_mixed_pipeline
from samplers import ParamSpec

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def append_report(name: str, body: str) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / name
    with path.open("a", encoding="utf-8") as f:
        f.write(body)
        if not body.endswith("\n"):
            f.write("\n")
    return path


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def run_problem_seeds(
    problem: Problem,
    seeds: list[int],
    *,
    n_jobs: int = 1,
    progress_every: int = 5,
    report_name: str = "progress.md",
) -> list[dict[str, Any]]:
    X, y = problem.load_xy()
    rows: list[dict[str, Any]] = []
    t_report = time.time()
    for i, seed in enumerate(seeds, start=1):
        paired = run_paired_search(
            problem.make_estimator(),
            X,
            y,
            problem.specs,
            n_iter=problem.n_iter,
            scoring=problem.scoring,
            cv=problem.cv,
            seed=seed,
            problem_id=problem.problem_id,
            n_jobs=n_jobs,
        )
        metrics = instance_metrics(paired["uniform"], paired["lhs"])
        metrics["phase_problem"] = problem.problem_id
        rows.append(metrics)

        if i % progress_every == 0 or i == len(seeds) or (time.time() - t_report) > 300:
            agg = aggregate_metrics(rows)
            body = (
                f"\n### Progress — {problem.problem_id} ({i}/{len(seeds)} seeds) — {utc_now()}\n\n"
                + format_agg_markdown(agg, title=f"Running aggregate ({problem.problem_id})")
            )
            append_report(report_name, body)
            t_report = time.time()
            print(body)
    return rows


def run_pipeline_problem(
    cfg: dict[str, Any],
    seeds: list[int],
    *,
    n_jobs: int = 1,
    progress_every: int = 3,
    report_name: str = "progress.md",
) -> list[dict[str, Any]]:
    X, y = cfg["load_xy"]()
    est = build_mixed_pipeline(X, model=cfg["model"])
    specs: list[ParamSpec] = cfg["specs"]
    rows: list[dict[str, Any]] = []
    t_report = time.time()
    for i, seed in enumerate(seeds, start=1):
        paired = run_paired_search(
            est,
            X,
            y,
            specs,
            n_iter=cfg["n_iter"],
            scoring=cfg["scoring"],
            cv=cfg["cv"],
            seed=seed,
            problem_id=cfg["problem_id"],
            n_jobs=n_jobs,
        )
        metrics = instance_metrics(paired["uniform"], paired["lhs"])
        metrics["phase_problem"] = cfg["problem_id"]
        rows.append(metrics)
        if i % progress_every == 0 or i == len(seeds) or (time.time() - t_report) > 300:
            agg = aggregate_metrics(rows)
            body = (
                f"\n### Progress — {cfg['problem_id']} ({i}/{len(seeds)} seeds) — {utc_now()}\n\n"
                + format_agg_markdown(agg, title=f"Running aggregate ({cfg['problem_id']})")
            )
            append_report(report_name, body)
            print(body)
            t_report = time.time()
    return rows


def runs_to_serializable(paired: dict[str, SearchRun]) -> dict[str, Any]:
    return {k: v.to_dict() for k, v in paired.items()}
