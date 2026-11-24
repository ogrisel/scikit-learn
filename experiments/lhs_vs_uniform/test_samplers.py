"""Unit tests for every public function/method in samplers.py."""

from __future__ import annotations

import numpy as np
import pytest

from samplers import (
    ParamSpec,
    _unit_to_value,
    logint_grid,
    sample_independent,
    sample_lhs,
    sample_search_space,
)


# ---------------------------------------------------------------------------
# logint_grid
# ---------------------------------------------------------------------------


class TestLogintGrid:
    def test_endpoints_preserved(self):
        grid = logint_grid(3, 29)
        assert grid[0] == 3
        assert grid[-1] == 29

    def test_strictly_increasing_unique_ints(self):
        grid = logint_grid(50, 300)
        assert all(isinstance(v, int) for v in grid)
        assert list(grid) == sorted(set(grid))
        assert all(a < b for a, b in zip(grid, grid[1:]))

    def test_all_values_within_bounds(self):
        grid = logint_grid(3, 29)
        assert all(3 <= v <= 29 for v in grid)

    def test_sparse_not_dense_range(self):
        # Must not return every integer in a multi-decade span.
        grid = logint_grid(50, 300)
        assert len(grid) < (300 - 50 + 1)
        assert len(grid) >= 5

    def test_known_poly_reg_grids(self):
        assert logint_grid(3, 29) == (3, 4, 5, 7, 9, 12, 16, 22, 29)
        assert logint_grid(50, 300) == (50, 67, 91, 122, 165, 223, 300)

    def test_explicit_n_steps(self):
        grid = logint_grid(1, 1000, n_steps=4)
        assert grid[0] == 1
        assert grid[-1] == 1000
        assert len(grid) <= 4 + 2  # rounding may dedupe; endpoints forced

    def test_rejects_non_positive(self):
        with pytest.raises(ValueError, match="positive"):
            logint_grid(0, 10)
        with pytest.raises(ValueError, match="positive"):
            logint_grid(-1, 10)

    def test_rejects_low_ge_high(self):
        with pytest.raises(ValueError, match="low < high"):
            logint_grid(5, 5)
        with pytest.raises(ValueError, match="low < high"):
            logint_grid(10, 3)

    def test_narrow_range(self):
        grid = logint_grid(2, 3)
        assert grid == (2, 3)


# ---------------------------------------------------------------------------
# ParamSpec construction / validation
# ---------------------------------------------------------------------------


class TestParamSpecInit:
    def test_uniform_ok(self):
        s = ParamSpec("x", "uniform", low=0.0, high=1.0)
        assert s.name == "x"

    def test_loguniform_ok(self):
        s = ParamSpec("alpha", "loguniform", low=1e-3, high=1e3)
        assert s.kind == "loguniform"

    def test_int_ok(self):
        s = ParamSpec("d", "int", low=2, high=5)
        assert s.kind == "int"

    def test_logint_ok(self):
        s = ParamSpec("k", "logint", low=3, high=29)
        assert s.kind == "logint"

    def test_choice_ok(self):
        s = ParamSpec("kernel", "choice", choices=("poly", "rbf"))
        assert s.choices == ("poly", "rbf")

    def test_requires_low_high(self):
        with pytest.raises(ValueError, match="low/high"):
            ParamSpec("x", "uniform")
        with pytest.raises(ValueError, match="low/high"):
            ParamSpec("x", "logint", low=1)

    def test_log_bounds_must_be_positive(self):
        with pytest.raises(ValueError, match="bounds must be > 0"):
            ParamSpec("x", "loguniform", low=0.0, high=1.0)
        with pytest.raises(ValueError, match="bounds must be > 0"):
            ParamSpec("x", "logint", low=0, high=10)

    def test_low_must_be_lt_high(self):
        with pytest.raises(ValueError, match="low < high"):
            ParamSpec("x", "uniform", low=1.0, high=1.0)
        with pytest.raises(ValueError, match="low < high"):
            ParamSpec("x", "int", low=5, high=3)

    def test_choice_requires_choices(self):
        with pytest.raises(ValueError, match="choices"):
            ParamSpec("k", "choice")
        with pytest.raises(ValueError, match="choices"):
            ParamSpec("k", "choice", choices=())

    def test_unknown_kind(self):
        with pytest.raises(ValueError, match="Unknown kind"):
            ParamSpec("x", "gaussian", low=0.0, high=1.0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ParamSpec.discrete_values
# ---------------------------------------------------------------------------


class TestDiscreteValues:
    def test_logint(self):
        s = ParamSpec("k", "logint", low=3, high=29)
        assert s.discrete_values() == logint_grid(3, 29)

    def test_int(self):
        s = ParamSpec("d", "int", low=2, high=5)
        assert s.discrete_values() == (2, 3, 4, 5)

    def test_choice(self):
        s = ParamSpec("k", "choice", choices=("a", "b"))
        assert s.discrete_values() == ("a", "b")

    def test_continuous_returns_none(self):
        assert ParamSpec("x", "uniform", low=0.0, high=1.0).discrete_values() is None
        assert ParamSpec("x", "loguniform", low=1e-3, high=1.0).discrete_values() is None


# ---------------------------------------------------------------------------
# ParamSpec.format
# ---------------------------------------------------------------------------


class TestFormat:
    def test_uniform(self):
        assert ParamSpec("x", "uniform", low=0.0, high=1.0).format() == "x ~ uniform(0, 1)"

    def test_loguniform(self):
        s = ParamSpec("alpha", "loguniform", low=1e-4, high=1e4)
        assert s.format() == "alpha ~ loguniform(0.0001, 10000)"

    def test_int(self):
        assert ParamSpec("d", "int", low=2, high=5).format() == "d ~ randint(2, 5)"

    def test_logint_includes_grid(self):
        s = ParamSpec("k", "logint", low=3, high=29)
        text = s.format()
        assert text.startswith("k ~ lograndint(3, 29)")
        assert "[3, 4, 5, 7, 9, 12, 16, 22, 29]" in text

    def test_choice(self):
        s = ParamSpec("kernel", "choice", choices=("poly", "rbf"))
        assert s.format() == "kernel ∈ ['poly', 'rbf']"


# ---------------------------------------------------------------------------
# ParamSpec.near_low_boundary / near_high_boundary
# ---------------------------------------------------------------------------


class TestNearBoundary:
    def test_choice_never_boundary(self):
        s = ParamSpec("k", "choice", choices=("a", "b"))
        assert s.near_low_boundary("a") is False
        assert s.near_high_boundary("b") is False

    def test_uniform_frac(self):
        s = ParamSpec("x", "uniform", low=0.0, high=100.0)
        assert s.near_low_boundary(0.0) is True
        assert s.near_low_boundary(4.0) is True  # 4% of span
        assert s.near_low_boundary(10.0) is False
        assert s.near_high_boundary(100.0) is True
        assert s.near_high_boundary(96.0) is True
        assert s.near_high_boundary(90.0) is False

    def test_loguniform_frac(self):
        s = ParamSpec("a", "loguniform", low=1e-2, high=1e2)
        # log span = 4 decades; 5% ≈ 0.2 decades from each end
        assert s.near_low_boundary(1e-2) is True
        assert s.near_high_boundary(1e2) is True
        assert s.near_low_boundary(1.0) is False
        assert s.near_high_boundary(1.0) is False
        assert type(s.near_low_boundary(1e-2)) is bool

    def test_int_uses_frac_like_other_kinds(self):
        """Integer bounds should respect frac over the discrete index range."""
        s = ParamSpec("d", "int", low=0, high=100)
        # With frac=0.05, first ~5% of indices (0..5) are near low.
        assert s.near_low_boundary(0, frac=0.05) is True
        assert s.near_low_boundary(5, frac=0.05) is True
        assert s.near_low_boundary(10, frac=0.05) is False
        assert s.near_high_boundary(100, frac=0.05) is True
        assert s.near_high_boundary(95, frac=0.05) is True
        assert s.near_high_boundary(90, frac=0.05) is False

    def test_logint_uses_grid_index(self):
        s = ParamSpec("k", "logint", low=3, high=29)
        grid = s.discrete_values()
        assert s.near_low_boundary(grid[0]) is True
        assert s.near_high_boundary(grid[-1]) is True
        mid = grid[len(grid) // 2]
        assert s.near_low_boundary(mid) is False
        assert s.near_high_boundary(mid) is False

    def test_logint_off_grid_snaps_to_nearest_not_endpoint(self):
        s = ParamSpec("k", "logint", low=3, high=29)
        # 6 is between grid points 5 and 7 — mid-range, not an endpoint.
        assert 6 not in s.discrete_values()
        assert s.near_low_boundary(6) is False
        assert s.near_high_boundary(6) is False


# ---------------------------------------------------------------------------
# ParamSpec.widened
# ---------------------------------------------------------------------------


class TestWidened:
    def test_choice_unchanged(self):
        s = ParamSpec("k", "choice", choices=("a", "b"))
        assert s.widened(side="both") is s

    def test_loguniform_low_high_both(self):
        s = ParamSpec("a", "loguniform", low=1e-3, high=1e3)
        assert s.widened(side="low").low == pytest.approx(1e-4)
        assert s.widened(side="low").high == pytest.approx(1e3)
        assert s.widened(side="high").low == pytest.approx(1e-3)
        assert s.widened(side="high").high == pytest.approx(1e4)
        both = s.widened(side="both")
        assert both.low == pytest.approx(1e-4)
        assert both.high == pytest.approx(1e4)

    def test_uniform_expands_by_span(self):
        s = ParamSpec("x", "uniform", low=10.0, high=20.0)
        both = s.widened(side="both")
        assert both.low == pytest.approx(0.0)
        assert both.high == pytest.approx(30.0)

    def test_unit_interval_uniform_not_widened(self):
        s = ParamSpec("p", "uniform", low=0.0, high=1.0)
        assert s.widened(side="both") is s

    def test_int_widens_by_span(self):
        s = ParamSpec("d", "int", low=2, high=5)
        both = s.widened(side="both")
        # span=3 → low max(0, 2-3)=0, high=5+3=8 (no depth/leaf/nodes in name)
        assert both.low == 0.0
        assert both.high == 8.0

    def test_int_depth_name_floors_at_one(self):
        s = ParamSpec("max_depth", "int", low=2, high=5)
        low = s.widened(side="low")
        assert low.low == 1.0

    def test_logint_widens_multiplicatively(self):
        s = ParamSpec("k", "logint", low=50, high=300)
        both = s.widened(side="both")
        assert both.low == 25.0
        assert both.high == 600.0
        assert both.kind == "logint"

    def test_logint_low_floors_at_one(self):
        s = ParamSpec("k", "logint", low=1, high=10)
        assert s.widened(side="low").low == 1.0


# ---------------------------------------------------------------------------
# _unit_to_value
# ---------------------------------------------------------------------------


class TestUnitToValue:
    @pytest.fixture
    def rng(self):
        return np.random.default_rng(0)

    def test_uniform_endpoints(self, rng):
        s = ParamSpec("x", "uniform", low=2.0, high=5.0)
        assert _unit_to_value(0.0, s, rng) == pytest.approx(2.0)
        assert _unit_to_value(1.0 - 1e-15, s, rng) == pytest.approx(5.0, abs=1e-9)

    def test_loguniform_endpoints(self, rng):
        s = ParamSpec("a", "loguniform", low=1e-2, high=1e2)
        assert _unit_to_value(0.0, s, rng) == pytest.approx(1e-2)
        hi = _unit_to_value(1.0 - 1e-15, s, rng)
        assert hi == pytest.approx(1e2, rel=1e-6)

    def test_int_covers_full_range(self, rng):
        s = ParamSpec("d", "int", low=2, high=5)
        seen = {_unit_to_value(u, s, rng) for u in np.linspace(0, 0.999, 400)}
        assert seen == {2, 3, 4, 5}

    def test_int_binning_edges(self, rng):
        s = ParamSpec("d", "int", low=0, high=3)  # 4 bins
        assert _unit_to_value(0.0, s, rng) == 0
        assert _unit_to_value(0.249, s, rng) == 0
        assert _unit_to_value(0.25, s, rng) == 1
        assert _unit_to_value(0.999, s, rng) == 3

    def test_logint_only_returns_grid_values(self, rng):
        s = ParamSpec("k", "logint", low=3, high=29)
        grid = set(s.discrete_values())
        for u in np.linspace(0, 0.999, 200):
            assert _unit_to_value(u, s, rng) in grid

    def test_logint_covers_all_grid_points(self, rng):
        s = ParamSpec("k", "logint", low=3, high=29)
        grid = s.discrete_values()
        seen = {_unit_to_value(u, s, rng) for u in np.linspace(0, 0.999, 500)}
        assert seen == set(grid)

    def test_choice(self, rng):
        s = ParamSpec("k", "choice", choices=("poly", "rbf", "sigmoid"))
        # 3 equal bins on [0, 1): [0, 1/3), [1/3, 2/3), [2/3, 1)
        assert _unit_to_value(0.0, s, rng) == "poly"
        assert _unit_to_value(0.333, s, rng) == "poly"
        assert _unit_to_value(1.0 / 3.0, s, rng) == "rbf"
        assert _unit_to_value(0.5, s, rng) == "rbf"
        assert _unit_to_value(0.999, s, rng) == "sigmoid"

    def test_clips_out_of_range_u(self, rng):
        s = ParamSpec("x", "uniform", low=0.0, high=1.0)
        assert 0.0 <= _unit_to_value(-1.0, s, rng) <= 1.0
        assert 0.0 <= _unit_to_value(2.0, s, rng) <= 1.0


# ---------------------------------------------------------------------------
# sample_independent
# ---------------------------------------------------------------------------


class TestSampleIndependent:
    def test_shape_and_keys(self):
        specs = [
            ParamSpec("a", "loguniform", low=1e-3, high=1.0),
            ParamSpec("d", "int", low=1, high=5),
            ParamSpec("k", "choice", choices=("x", "y")),
        ]
        out = sample_independent(specs, 20, np.random.default_rng(0))
        assert len(out) == 20
        assert all(set(p) == {"a", "d", "k"} for p in out)

    def test_values_in_bounds(self):
        specs = [
            ParamSpec("u", "uniform", low=-1.0, high=1.0),
            ParamSpec("a", "loguniform", low=1e-4, high=1e4),
            ParamSpec("i", "int", low=2, high=8),
            ParamSpec("li", "logint", low=50, high=300),
            ParamSpec("c", "choice", choices=("poly", "rbf")),
        ]
        out = sample_independent(specs, 100, np.random.default_rng(1))
        grid = set(logint_grid(50, 300))
        for p in out:
            assert -1.0 <= p["u"] <= 1.0
            assert 1e-4 <= p["a"] <= 1e4
            assert p["i"] in range(2, 9)
            assert p["li"] in grid
            assert p["c"] in ("poly", "rbf")

    def test_reproducible_with_same_rng_seed(self):
        specs = [ParamSpec("a", "loguniform", low=1e-2, high=1e2)]
        a = sample_independent(specs, 10, np.random.default_rng(42))
        b = sample_independent(specs, 10, np.random.default_rng(42))
        assert a == b

    def test_empty_n_samples(self):
        specs = [ParamSpec("x", "uniform", low=0.0, high=1.0)]
        assert sample_independent(specs, 0, np.random.default_rng(0)) == []


# ---------------------------------------------------------------------------
# sample_lhs
# ---------------------------------------------------------------------------


class TestSampleLhs:
    def test_shape_and_keys(self):
        specs = [
            ParamSpec("a", "loguniform", low=1e-3, high=1.0),
            ParamSpec("d", "int", low=1, high=4),
        ]
        out = sample_lhs(specs, 8, np.random.default_rng(0))
        assert len(out) == 8
        assert all(set(p) == {"a", "d"} for p in out)

    def test_latin_property_on_int_bins(self):
        """With n_samples == n_bins, each integer appears exactly once (1D LHS)."""
        specs = [ParamSpec("d", "int", low=0, high=9)]  # 10 values
        out = sample_lhs(specs, 10, np.random.default_rng(0), scramble=False)
        vals = [p["d"] for p in out]
        assert sorted(vals) == list(range(10))

    def test_latin_property_multidim_strata(self):
        """Each dimension's unit strata are occupied once when n is the bin count."""
        specs = [
            ParamSpec("x", "uniform", low=0.0, high=1.0),
            ParamSpec("y", "uniform", low=0.0, high=1.0),
        ]
        n = 16
        # Probe via re-deriving unit coords from values (uniform inverse).
        out = sample_lhs(specs, n, np.random.default_rng(7), scramble=True)
        for name in ("x", "y"):
            units = [p[name] for p in out]  # already in [0,1]
            strata = [min(int(u * n), n - 1) for u in units]
            assert sorted(strata) == list(range(n))

    def test_logint_values_on_grid(self):
        specs = [ParamSpec("k", "logint", low=3, high=29)]
        grid = set(logint_grid(3, 29))
        out = sample_lhs(specs, 20, np.random.default_rng(3))
        assert all(p["k"] in grid for p in out)

    def test_reproducible(self):
        specs = [ParamSpec("a", "loguniform", low=1e-3, high=1.0)]
        a = sample_lhs(specs, 5, np.random.default_rng(99))
        b = sample_lhs(specs, 5, np.random.default_rng(99))
        assert a == b


# ---------------------------------------------------------------------------
# sample_search_space
# ---------------------------------------------------------------------------


class TestSampleSearchSpace:
    def test_uniform_method(self):
        specs = [ParamSpec("x", "uniform", low=0.0, high=1.0)]
        out = sample_search_space(specs, 5, "uniform", random_state=0)
        assert len(out) == 5
        assert all(0.0 <= p["x"] <= 1.0 for p in out)

    def test_lhs_method(self):
        specs = [ParamSpec("x", "uniform", low=0.0, high=1.0)]
        out = sample_search_space(specs, 5, "lhs", random_state=0)
        assert len(out) == 5

    def test_methods_differ(self):
        specs = [
            ParamSpec("a", "loguniform", low=1e-4, high=1e4),
            ParamSpec("b", "uniform", low=0.0, high=1.0),
        ]
        u = sample_search_space(specs, 30, "uniform", random_state=0)
        l = sample_search_space(specs, 30, "lhs", random_state=0)
        assert u != l

    def test_reproducible(self):
        specs = [ParamSpec("x", "uniform", low=0.0, high=1.0)]
        a = sample_search_space(specs, 10, "lhs", random_state=123)
        b = sample_search_space(specs, 10, "lhs", random_state=123)
        assert a == b

    def test_unknown_method(self):
        specs = [ParamSpec("x", "uniform", low=0.0, high=1.0)]
        with pytest.raises(ValueError, match="Unknown method"):
            sample_search_space(specs, 3, "sobol", random_state=0)  # type: ignore[arg-type]

    def test_mixed_space_smoke(self):
        specs = [
            ParamSpec("splinetransformer__n_knots", "logint", low=3, high=29),
            ParamSpec("nystroem__kernel", "choice", choices=("poly", "rbf")),
            ParamSpec("nystroem__degree", "int", low=2, high=5),
            ParamSpec("nystroem__gamma", "loguniform", low=1e-6, high=1e6),
            ParamSpec("nystroem__n_components", "logint", low=50, high=300),
        ]
        for method in ("uniform", "lhs"):
            out = sample_search_space(specs, 40, method, random_state=0)
            assert len(out) == 40
            knots = {p["splinetransformer__n_knots"] for p in out}
            comps = {p["nystroem__n_components"] for p in out}
            assert knots <= set(logint_grid(3, 29))
            assert comps <= set(logint_grid(50, 300))
