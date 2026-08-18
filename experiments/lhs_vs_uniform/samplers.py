"""Independent uniform and Latin Hypercube samplers for HPO search spaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np
from scipy.stats import qmc

ParamKind = Literal["uniform", "loguniform", "int", "choice"]


@dataclass(frozen=True)
class ParamSpec:
    name: str
    kind: ParamKind
    low: float | None = None
    high: float | None = None
    choices: tuple[Any, ...] | None = None

    def __post_init__(self) -> None:
        if self.kind in ("uniform", "loguniform", "int"):
            if self.low is None or self.high is None:
                raise ValueError(f"{self.name}: low/high required for kind={self.kind}")
            if self.kind == "loguniform" and not (self.low > 0 and self.high > 0):
                raise ValueError(f"{self.name}: loguniform bounds must be > 0")
            if self.low >= self.high:
                raise ValueError(f"{self.name}: require low < high")
        elif self.kind == "choice":
            if not self.choices:
                raise ValueError(f"{self.name}: choices required for kind=choice")
        else:
            raise ValueError(f"Unknown kind: {self.kind}")


def _unit_to_value(u: float, spec: ParamSpec, rng: np.random.Generator) -> Any:
    """Map a unit-interval sample u in [0, 1) to a concrete parameter value."""
    u = float(np.clip(u, 0.0, 1.0 - 1e-12))
    if spec.kind == "uniform":
        return spec.low + u * (spec.high - spec.low)
    if spec.kind == "loguniform":
        log_low = np.log(spec.low)
        log_high = np.log(spec.high)
        return float(np.exp(log_low + u * (log_high - log_low)))
    if spec.kind == "int":
        # Stratified integer bins over [low, high] inclusive.
        n_bins = int(spec.high - spec.low) + 1
        idx = min(int(u * n_bins), n_bins - 1)
        return int(spec.low) + idx
    # choice
    n = len(spec.choices)
    idx = min(int(u * n), n - 1)
    return spec.choices[idx]


def sample_independent(
    specs: Sequence[ParamSpec],
    n_samples: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """i.i.d. (log-)uniform / discrete sampling (RandomizedSearchCV-style)."""
    out: list[dict[str, Any]] = []
    for _ in range(n_samples):
        params: dict[str, Any] = {}
        for spec in specs:
            u = float(rng.random())
            params[spec.name] = _unit_to_value(u, spec, rng)
        out.append(params)
    return out


def sample_lhs(
    specs: Sequence[ParamSpec],
    n_samples: int,
    rng: np.random.Generator,
    *,
    scramble: bool = True,
) -> list[dict[str, Any]]:
    """Latin Hypercube samples mapped through inverse CDFs / stratified bins."""
    d = len(specs)
    seed = int(rng.integers(0, 2**31 - 1))
    sampler = qmc.LatinHypercube(d=d, scramble=scramble, seed=seed)
    unit = sampler.random(n=n_samples)
    out: list[dict[str, Any]] = []
    for row in unit:
        params: dict[str, Any] = {}
        for j, spec in enumerate(specs):
            params[spec.name] = _unit_to_value(float(row[j]), spec, rng)
        out.append(params)
    return out


def sample_search_space(
    specs: Sequence[ParamSpec],
    n_samples: int,
    method: Literal["uniform", "lhs"],
    random_state: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(random_state)
    if method == "uniform":
        return sample_independent(specs, n_samples, rng)
    if method == "lhs":
        return sample_lhs(specs, n_samples, rng)
    raise ValueError(f"Unknown method: {method}")
