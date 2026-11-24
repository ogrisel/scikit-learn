"""Anytime metrics, time-to-95%, speedups, and bootstrap CIs."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from evaluate import SearchRun


def time_to_target(
    run: SearchRun,
    u_target: float,
) -> tuple[float | None, int | None]:
    """Return (cum_time, n_eval) when best-so-far first reaches u_target."""
    for t in run.trials:
        if t.best_score_so_far >= u_target:
            return t.cum_time, t.n_eval
    return None, None


def instance_metrics(
    uniform_run: SearchRun,
    lhs_run: SearchRun,
    *,
    fraction: float = 0.95,
) -> dict[str, Any]:
    """Compute paired speedup metrics for one problem instance."""
    all_scores = np.concatenate([uniform_run.scores, lhs_run.scores])
    finite = all_scores[np.isfinite(all_scores)]
    if finite.size == 0:
        return {"ok": False, "reason": "no_finite_scores"}

    u_star = float(np.max(finite))
    # Reference: first evaluated config on uniform run (protocol anchor).
    u_ref = float(uniform_run.trials[0].score) if uniform_run.trials else float(finite.min())
    if not np.isfinite(u_ref):
        u_ref = float(np.min(finite))

    # If already at/above star or flat landscape, target is u_star.
    gap = u_star - u_ref
    if gap <= 1e-12:
        u_target = u_star
    else:
        u_target = u_ref + fraction * gap

    t_u, n_u = time_to_target(uniform_run, u_target)
    t_l, n_l = time_to_target(lhs_run, u_target)

    out: dict[str, Any] = {
        "ok": True,
        "problem_id": uniform_run.problem_id,
        "seed": uniform_run.seed,
        "scoring": uniform_run.scoring,
        "u_star": u_star,
        "u_ref": u_ref,
        "u_target": u_target,
        "t_uniform": t_u,
        "t_lhs": t_l,
        "n_uniform": n_u,
        "n_lhs": n_l,
        "final_uniform": float(uniform_run.best_scores[-1]) if len(uniform_run.trials) else np.nan,
        "final_lhs": float(lhs_run.best_scores[-1]) if len(lhs_run.trials) else np.nan,
        "reached_uniform": t_u is not None,
        "reached_lhs": t_l is not None,
    }

    if t_u is not None and t_l is not None and t_l > 0:
        out["speedup_time"] = t_u / t_l
    else:
        out["speedup_time"] = None

    if n_u is not None and n_l is not None and n_l > 0:
        out["speedup_neval"] = n_u / n_l
    else:
        out["speedup_neval"] = None

    # Utility at fixed evaluation budgets (fraction of N).
    n_total = len(uniform_run.trials)
    for frac in (0.1, 0.25, 0.5, 1.0):
        k = max(1, int(round(frac * n_total)))
        out[f"best_uniform_@{frac}"] = float(uniform_run.best_scores[k - 1])
        out[f"best_lhs_@{frac}"] = float(lhs_run.best_scores[k - 1])
        out[f"lhs_minus_uniform_@{frac}"] = (
            out[f"best_lhs_@{frac}"] - out[f"best_uniform_@{frac}"]
        )

    return out


def bootstrap_ci(
    values: Iterable[float],
    *,
    n_boot: int = 5000,
    alpha: float = 0.05,
    random_state: int = 0,
    statistic: str = "mean",
) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "estimate": np.nan, "low": np.nan, "high": np.nan}

    rng = np.random.default_rng(random_state)

    def _stat(x: np.ndarray) -> float:
        if statistic == "mean":
            return float(np.mean(x))
        if statistic == "median":
            return float(np.median(x))
        raise ValueError(statistic)

    estimate = _stat(arr)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = arr[rng.integers(0, arr.size, size=arr.size)]
        boots[i] = _stat(sample)
    low, high = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {
        "n": int(arr.size),
        "estimate": estimate,
        "low": float(low),
        "high": float(high),
    }


def aggregate_metrics(instance_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [r for r in instance_rows if r.get("ok")]
    speedups_t = [r["speedup_time"] for r in ok_rows if r.get("speedup_time") is not None]
    speedups_n = [r["speedup_neval"] for r in ok_rows if r.get("speedup_neval") is not None]
    deltas_25 = [r.get("lhs_minus_uniform_@0.25") for r in ok_rows]
    deltas_50 = [r.get("lhs_minus_uniform_@0.5") for r in ok_rows]
    deltas_100 = [r.get("lhs_minus_uniform_@1.0") for r in ok_rows]

    mean_t = bootstrap_ci(speedups_t, statistic="mean")
    med_t = bootstrap_ci(speedups_t, statistic="median")
    mean_n = bootstrap_ci(speedups_n, statistic="mean")

    frac_speedup_gt1 = (
        float(np.mean([s > 1.0 for s in speedups_t])) if speedups_t else float("nan")
    )
    frac_lhs_better_25 = (
        float(np.mean([d > 0 for d in deltas_25 if d is not None and np.isfinite(d)]))
        if deltas_25
        else float("nan")
    )

    # Average dominance heuristic: mean utility gap > 0 at mid budget and mean speedup > 1
    # with CI mostly above 1.
    dominates = (
        mean_t["n"] > 0
        and mean_t["estimate"] > 1.0
        and mean_t["low"] > 1.0
        and np.nanmean(deltas_50) > 0
    )
    mixed = (
        mean_t["n"] > 0
        and mean_t["estimate"] > 1.0
        and mean_t["low"] <= 1.0
    )

    if dominates:
        verdict = "lhs_pareto_dominates_on_average"
    elif mixed:
        verdict = "lhs_advantage_mixed_ci_includes_1"
    elif mean_t["n"] > 0 and mean_t["estimate"] < 1.0 and mean_t["high"] < 1.0:
        verdict = "uniform_better_on_average"
    else:
        verdict = "no_clear_average_dominance"

    return {
        "n_instances": len(ok_rows),
        "n_with_speedup_time": len(speedups_t),
        "mean_speedup_time": mean_t,
        "median_speedup_time": med_t,
        "mean_speedup_neval": mean_n,
        "frac_speedup_time_gt1": frac_speedup_gt1,
        "frac_lhs_better_at_25pct_budget": frac_lhs_better_25,
        "mean_lhs_minus_uniform_at_25": float(np.nanmean(deltas_25)) if deltas_25 else np.nan,
        "mean_lhs_minus_uniform_at_50": float(np.nanmean(deltas_50)) if deltas_50 else np.nan,
        "mean_lhs_minus_uniform_at_100": float(np.nanmean(deltas_100)) if deltas_100 else np.nan,
        "verdict": verdict,
    }


def format_agg_markdown(agg: dict[str, Any], title: str = "Aggregate results") -> str:
    mt = agg["mean_speedup_time"]
    med = agg["median_speedup_time"]
    mn = agg["mean_speedup_neval"]
    lines = [
        f"## {title}",
        "",
        f"- Instances: **{agg['n_instances']}** (with time-speedup: {agg['n_with_speedup_time']})",
        f"- Mean time-speedup (Uniform/LHS): **{mt['estimate']:.3f}** "
        f"95% CI [{mt['low']:.3f}, {mt['high']:.3f}] (n={mt['n']})",
        f"- Median time-speedup: **{med['estimate']:.3f}** "
        f"95% CI [{med['low']:.3f}, {med['high']:.3f}]",
        f"- Mean n_eval-speedup: **{mn['estimate']:.3f}** "
        f"95% CI [{mn['low']:.3f}, {mn['high']:.3f}]",
        f"- Fraction speedup>1: **{agg['frac_speedup_time_gt1']:.3f}**",
        f"- Mean LHS−Uniform utility @25%/50%/100% budget: "
        f"{agg['mean_lhs_minus_uniform_at_25']:.5f} / "
        f"{agg['mean_lhs_minus_uniform_at_50']:.5f} / "
        f"{agg['mean_lhs_minus_uniform_at_100']:.5f}",
        f"- Verdict: `{agg['verdict']}`",
        "",
    ]
    return "\n".join(lines)
