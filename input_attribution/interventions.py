"""Small, strict helpers for fixed-value input interventions.

An intervention is applied to a copy of the already prepared policy input.
The module intentionally knows nothing about MetaDrive, SB3, or the meaning of
an input index.  That boundary makes it useful for both the synthetic adapter
and a ported environment adapter.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Integral, Real
import re

import numpy as np


class InterventionError(ValueError):
    """The requested intervention cannot be applied safely."""


_SAFE_PATTERN_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
_RESERVED_PATTERN_IDS = frozenset(
    {
        "p00",
        "p0",
        "p00_baseline",
        "baseline",
        "noop",
        "none",
        "identity",
        "data",
    }
)


@dataclass(frozen=True, slots=True)
class Intervention:
    """A named set of indices and the value sent to the policy at those indices."""

    id: str
    name: str
    indices: tuple[int, ...]
    fixed_value: float = -1.0

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise InterventionError("pattern id must be a non-empty string")
        if self.id != self.id.strip():
            raise InterventionError("pattern id must not have leading or trailing whitespace")
        if self.id in {".", ".."} or _SAFE_PATTERN_ID.fullmatch(self.id) is None:
            raise InterventionError(
                "pattern id must be a safe ASCII path component "
                "matching [A-Za-z0-9_.-]+"
            )
        normalized_id = self.id.casefold()
        if normalized_id in _RESERVED_PATTERN_IDS or normalized_id.endswith("_offline"):
            raise InterventionError(
                f"pattern id {self.id!r} is reserved or collides with a raw/offline artifact"
            )
        if not isinstance(self.name, str) or not self.name.strip():
            raise InterventionError("pattern name must be a non-empty string")
        # Validate here as well as at the array boundary, so config-created
        # patterns fail before an environment is started.
        validate_indices(self.indices)
        _finite_real(self.fixed_value, "fixed_value")


@dataclass(frozen=True, slots=True)
class InterventionResult:
    """Result of applying one pattern to one prepared input."""

    original: np.ndarray
    modified: np.ndarray
    applied: bool
    changed: bool
    changed_dimensions: int

    @property
    def changed_count(self) -> int:
        """Whether this application changed at least one target dimension."""

        return int(self.changed)


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise InterventionError(f"{name} must be a finite numeric value")
    result = float(value)
    if not math.isfinite(result):
        raise InterventionError(f"{name} must be finite")
    return result


def validate_indices(
    indices: Sequence[object],
    *,
    dimension: int | None = None,
) -> tuple[int, ...]:
    """Validate and normalize zero-based, unique input indices.

    Duplicate indices are rejected explicitly.  Silently deduplicating them
    makes a malformed schema look like a successful intervention and obscures
    the requested group size.
    """

    if isinstance(indices, (str, bytes)):
        raise InterventionError("indices must be a sequence of integers")
    try:
        values = tuple(indices)
    except TypeError as error:
        raise InterventionError("indices must be a sequence of integers") from error
    normalized: list[int] = []
    seen: set[int] = set()
    for offset, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise InterventionError(
                f"indices[{offset}] must be an integer, found {type(value).__name__}"
            )
        index = int(value)
        if index < 0:
            raise InterventionError(f"indices[{offset}] must be non-negative")
        if dimension is not None and index >= dimension:
            raise InterventionError(
                f"indices[{offset}]={index} is outside dimension {dimension}"
            )
        if index in seen:
            raise InterventionError(f"duplicate intervention index: {index}")
        seen.add(index)
        normalized.append(index)
    if not normalized:
        raise InterventionError("an intervention must target at least one index")
    return tuple(normalized)


def coerce_intervention(
    pattern: Intervention | Mapping[str, object],
    *,
    dimension: int | None = None,
) -> Intervention:
    """Accept the plain-dict config form without coupling config to this type."""

    if isinstance(pattern, Intervention):
        if dimension is not None:
            validate_indices(pattern.indices, dimension=dimension)
        return pattern
    if not isinstance(pattern, Mapping):
        raise InterventionError("pattern must be an Intervention or mapping")
    missing = [key for key in ("id", "name", "indices") if key not in pattern]
    if missing:
        raise InterventionError(f"pattern is missing key(s): {', '.join(missing)}")
    fixed_value = pattern.get("fixed_value", -1.0)
    indices = validate_indices(pattern["indices"], dimension=dimension)
    return Intervention(
        id=str(pattern["id"]),
        name=str(pattern["name"]),
        indices=indices,
        fixed_value=_finite_real(fixed_value, "fixed_value"),
    )


def apply_intervention(
    original: object,
    pattern: Intervention | Mapping[str, object],
    *,
    dimension: int | None = None,
) -> InterventionResult:
    """Copy a prepared 1-D input and set exactly the requested dimensions.

    No range clipping, normalization, or repair is performed here.  The
    caller has already crossed the normal observation preprocessing boundary;
    this function is the explicit stress input sent to the policy.
    """

    array = np.asarray(original)
    if array.ndim != 1:
        raise InterventionError(f"policy input must be 1-D, found shape {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise InterventionError(f"policy input must be numeric, found {array.dtype}")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise InterventionError("policy input must be real, found complex values")
    # Work in a real floating representation.  Assigning -1.0 directly to an
    # integer/unsigned array can silently truncate or wrap the requested value.
    try:
        array = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise InterventionError("policy input cannot be represented as float64") from error
    if not np.all(np.isfinite(array)):
        raise InterventionError("policy input contains NaN or Inf")
    if dimension is not None and array.shape[0] != dimension:
        raise InterventionError(
            f"policy input shape {array.shape} does not match dimension {dimension}"
        )
    resolved = coerce_intervention(pattern, dimension=int(array.shape[0]))
    modified = np.array(array, copy=True)
    target_before = modified[list(resolved.indices)].copy()
    modified[list(resolved.indices)] = resolved.fixed_value
    if not np.all(np.isfinite(modified)):
        # This mainly protects unusual numpy dtypes and future changes to the
        # assignment path.  A non-finite value must never be sent to a model.
        raise InterventionError("fixed-value intervention produced NaN or Inf")
    changed_dimensions = int(np.count_nonzero(target_before != resolved.fixed_value))
    return InterventionResult(
        original=np.array(array, copy=True),
        modified=modified,
        applied=True,
        changed=changed_dimensions > 0,
        changed_dimensions=changed_dimensions,
    )


def apply_fixed_value(
    original: object,
    indices: Sequence[object],
    fixed_value: object = -1.0,
    *,
    dimension: int | None = None,
) -> np.ndarray:
    """Compatibility convenience returning only the modified copy."""

    result = apply_intervention(
        original,
        {
            "id": "inline",
            "name": "inline",
            "indices": indices,
            "fixed_value": fixed_value,
        },
        dimension=dimension,
    )
    return result.modified


def apply_to_batch(
    inputs: object,
    pattern: Intervention | Mapping[str, object],
    *,
    dimension: int | None = None,
) -> tuple[np.ndarray, tuple[InterventionResult, ...]]:
    """Apply one pattern independently to every row of a prepared batch."""

    array = np.asarray(inputs)
    if array.ndim != 2:
        raise InterventionError(f"batch policy input must be 2-D, found {array.shape}")
    results = tuple(
        apply_intervention(row, pattern, dimension=dimension)
        for row in array
    )
    if not results:
        # Empty batches are useful for callers but still need a validated
        # pattern and known width.
        width = int(array.shape[1])
        coerce_intervention(pattern, dimension=dimension or width)
        return np.array(array, copy=True), results
    return np.stack([result.modified for result in results]), results


# Descriptive aliases used by small adapters and downstream porting scripts.
intervene = apply_intervention
validate_intervention_indices = validate_indices


__all__ = [
    "Intervention",
    "InterventionError",
    "InterventionResult",
    "apply_fixed_value",
    "apply_intervention",
    "apply_to_batch",
    "coerce_intervention",
    "intervene",
    "validate_indices",
    "validate_intervention_indices",
]
