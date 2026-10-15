#!/usr/bin/env python3
"""Regenerate all neval plots from saved summary with fixed x ticks [3,5,10,30]."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_neval_budgets as m
from samplers import ParamSpec


def parse_spec_line(line: str) -> ParamSpec:
    # e.g. "alpha ~ loguniform(0.0001, 10000)"
    name, rest = line.split(" ~ ", 1)
    if rest.startswith("loguniform("):
        a, b = rest[len("loguniform(") : -1].split(", ")
        return ParamSpec(name, "loguniform", low=float(a), high=float(b))
    if rest.startswith("uniform("):
        a, b = rest[len("uniform(") : -1].split(", ")
        return ParamSpec(name, "uniform", low=float(a), high=float(b))
    if rest.startswith("randint("):
        a, b = rest[len("randint(") : -1].split(", ")
        return ParamSpec(name, "int", low=float(a), high=float(b))
    raise ValueError(line)


def main():
    summary = json.loads((m.RESULTS / "neval_budget_summary.json").read_text())
    assert m.N_ITERS == (3, 5, 10, 30), m.N_ITERS
    for rec in summary:
        specs = [parse_spec_line(s) for s in rec["search_space"]]
        cfg = {"problem_id": rec["problem_id"], "scoring": None}
        # recover scoring from problem id heuristics
        if "ridge" in rec["problem_id"] or rec["problem_id"].endswith("_hgb") and "diabetes" in rec["problem_id"]:
            cfg["scoring"] = (
                "neg_mean_squared_error"
                if "diabetes" in rec["problem_id"]
                else "neg_log_loss"
            )
        elif "diabetes_hgb" in rec["problem_id"]:
            cfg["scoring"] = "neg_mean_squared_error"
        else:
            cfg["scoring"] = "neg_log_loss"
        if rec["problem_id"] == "diabetes_ridge":
            cfg["scoring"] = "neg_mean_squared_error"
        if rec["problem_id"] == "diabetes_hgb":
            cfg["scoring"] = "neg_mean_squared_error"

        by_method = {"uniform": rec["uniform"], "lhs": rec["lhs"]}
        for method, rows in by_method.items():
            got = sorted(r["n_iter"] for r in rows)
            if got != list(m.N_ITERS):
                raise SystemExit(
                    f"{rec['problem_id']}/{method} has n_iter={got}, expected {list(m.N_ITERS)}"
                )
        stem = Path(rec["figure"]).stem
        m.plot_problem(cfg, specs, by_method, stem)
        print("replotted", stem, "x=", list(m.N_ITERS))
    print("OK")


if __name__ == "__main__":
    main()
