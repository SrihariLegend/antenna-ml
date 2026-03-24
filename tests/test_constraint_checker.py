"""Tests for constraint_checker module."""

import pytest

from constraint_checker import ConstraintResult, apply_constraints, validate_constraints


# ---------------------------------------------------------------------------
# validate_constraints
# ---------------------------------------------------------------------------


class TestValidateConstraints:
    """Unit tests for validate_constraints."""

    def test_empty_constraints_returns_no_errors(self):
        assert validate_constraints({}) == []

    def test_valid_constraints_returns_no_errors(self):
        constraints = {
            "length of patch in mm": (5.0, 20.0),
            "S11(dB)": (None, -10.0),
            "width of patch in mm": (3.0, None),
        }
        assert validate_constraints(constraints) == []

    def test_both_bounds_none_is_valid(self):
        constraints = {"length of patch in mm": (None, None)}
        assert validate_constraints(constraints) == []

    def test_equal_bounds_is_valid(self):
        constraints = {"length of patch in mm": (10.0, 10.0)}
        assert validate_constraints(constraints) == []

    def test_min_greater_than_max_returns_error(self):
        constraints = {"length of patch in mm": (25.0, 10.0)}
        errors = validate_constraints(constraints)
        assert len(errors) == 1
        assert "length of patch in mm" in errors[0]

    def test_multiple_invalid_constraints(self):
        constraints = {
            "length of patch in mm": (25.0, 10.0),
            "S11(dB)": (0.0, -20.0),
        }
        errors = validate_constraints(constraints)
        assert len(errors) == 2


# ---------------------------------------------------------------------------
# apply_constraints
# ---------------------------------------------------------------------------


class TestApplyConstraints:
    """Unit tests for apply_constraints."""

    def test_no_constraints_all_in_bounds(self):
        predictions = {"length of patch in mm": 15.0, "S11(dB)": -12.0}
        results = apply_constraints(predictions, {})
        assert all(r.in_bounds for r in results)
        assert len(results) == 2

    def test_value_within_bounds(self):
        predictions = {"length of patch in mm": 15.0}
        constraints = {"length of patch in mm": (10.0, 20.0)}
        results = apply_constraints(predictions, constraints)
        assert results[0].in_bounds is True

    def test_value_below_min_bound(self):
        predictions = {"length of patch in mm": 3.0}
        constraints = {"length of patch in mm": (10.0, 20.0)}
        results = apply_constraints(predictions, constraints)
        assert results[0].in_bounds is False

    def test_value_above_max_bound(self):
        predictions = {"length of patch in mm": 25.0}
        constraints = {"length of patch in mm": (10.0, 20.0)}
        results = apply_constraints(predictions, constraints)
        assert results[0].in_bounds is False

    def test_value_at_exact_min_bound(self):
        predictions = {"length of patch in mm": 10.0}
        constraints = {"length of patch in mm": (10.0, 20.0)}
        results = apply_constraints(predictions, constraints)
        assert results[0].in_bounds is True

    def test_value_at_exact_max_bound(self):
        predictions = {"length of patch in mm": 20.0}
        constraints = {"length of patch in mm": (10.0, 20.0)}
        results = apply_constraints(predictions, constraints)
        assert results[0].in_bounds is True

    def test_only_min_bound(self):
        predictions = {"S11(dB)": -5.0}
        constraints = {"S11(dB)": (-10.0, None)}
        results = apply_constraints(predictions, constraints)
        assert results[0].in_bounds is True
        assert results[0].max_bound is None

    def test_only_max_bound_violated(self):
        predictions = {"S11(dB)": -5.0}
        constraints = {"S11(dB)": (None, -10.0)}
        results = apply_constraints(predictions, constraints)
        assert results[0].in_bounds is False

    def test_only_max_bound_satisfied(self):
        predictions = {"S11(dB)": -15.0}
        constraints = {"S11(dB)": (None, -10.0)}
        results = apply_constraints(predictions, constraints)
        assert results[0].in_bounds is True

    def test_unconstrained_parameter_in_bounds(self):
        predictions = {"length of patch in mm": 15.0, "S11(dB)": -12.0}
        constraints = {"length of patch in mm": (10.0, 20.0)}
        results = apply_constraints(predictions, constraints)
        s11_result = next(r for r in results if r.parameter == "S11(dB)")
        assert s11_result.in_bounds is True
        assert s11_result.min_bound is None
        assert s11_result.max_bound is None

    def test_result_fields_populated(self):
        predictions = {"length of patch in mm": 15.0}
        constraints = {"length of patch in mm": (10.0, 20.0)}
        results = apply_constraints(predictions, constraints)
        r = results[0]
        assert r.parameter == "length of patch in mm"
        assert r.predicted_value == 15.0
        assert r.min_bound == 10.0
        assert r.max_bound == 20.0
        assert r.in_bounds is True


# ---------------------------------------------------------------------------
# ConstraintResult dataclass
# ---------------------------------------------------------------------------


class TestConstraintResult:
    """Unit tests for ConstraintResult dataclass."""

    def test_creation(self):
        r = ConstraintResult(
            parameter="test",
            predicted_value=5.0,
            min_bound=1.0,
            max_bound=10.0,
            in_bounds=True,
        )
        assert r.parameter == "test"
        assert r.predicted_value == 5.0
        assert r.min_bound == 1.0
        assert r.max_bound == 10.0
        assert r.in_bounds is True

    def test_none_bounds(self):
        r = ConstraintResult(
            parameter="test",
            predicted_value=5.0,
            min_bound=None,
            max_bound=None,
            in_bounds=True,
        )
        assert r.min_bound is None
        assert r.max_bound is None
