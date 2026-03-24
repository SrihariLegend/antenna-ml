"""Post-prediction constraint checking for antenna design parameters.

Applies user-supplied dimension bounds to prediction results and flags
out-of-bounds values.
"""

import logging
from dataclasses import dataclass

import config

config.setup_logging()
logger = logging.getLogger(__name__)


@dataclass
class ConstraintResult:
    """Result of checking a single predicted parameter against user-supplied bounds."""

    parameter: str
    predicted_value: float
    min_bound: float | None
    max_bound: float | None
    in_bounds: bool


def validate_constraints(
    constraints: dict[str, tuple[float | None, float | None]],
) -> list[str]:
    """Check that min <= max for each constraint.

    Args:
        constraints: Dict mapping parameter names to (min_bound, max_bound) tuples.
            Either bound may be None (unconstrained on that side).

    Returns:
        List of error messages. Empty list means all constraints are valid.
    """
    errors: list[str] = []
    for param, (lo, hi) in constraints.items():
        if lo is not None and hi is not None and lo > hi:
            errors.append(
                f"Constraint for '{param}': min ({lo}) is greater than max ({hi})."
            )
    for error in errors:
        logger.error("Constraint validation: %s", error)
    return errors


def apply_constraints(
    predictions: dict[str, float],
    constraints: dict[str, tuple[float | None, float | None]],
) -> list[ConstraintResult]:
    """Check each prediction against its bounds.

    Parameters with no constraint entry are marked ``in_bounds=True``.

    Args:
        predictions: Dict mapping parameter names to predicted float values.
        constraints: Dict mapping parameter names to (min_bound, max_bound) tuples.
            Either bound may be None (unconstrained on that side).

    Returns:
        List of ConstraintResult, one per predicted parameter.
    """
    results: list[ConstraintResult] = []
    for param, value in predictions.items():
        lo, hi = constraints.get(param, (None, None))
        in_bounds = True
        if lo is not None and value < lo:
            in_bounds = False
        if hi is not None and value > hi:
            in_bounds = False

        if not in_bounds:
            logger.warning(
                "Parameter '%s' predicted %.4f is out of bounds [%s, %s].",
                param,
                value,
                lo,
                hi,
            )

        results.append(
            ConstraintResult(
                parameter=param,
                predicted_value=value,
                min_bound=lo,
                max_bound=hi,
                in_bounds=in_bounds,
            )
        )
    return results
