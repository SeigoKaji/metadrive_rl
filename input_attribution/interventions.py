"""Explicit, copy-based observation interventions.

An intervention never mutates the source observation.  It records both the
requested indices and the indices whose values actually changed, which keeps a
no-op replacement distinguishable from a replacement that produced no policy
change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .schema import InputSchema, SchemaError


class InterventionError(ValueError):
    """Raised when a configured intervention is unsafe or incomplete."""


_METHOD_ALIASES = {
    "noop": "identity",
    "none": "identity",
    "unchanged": "identity",
    "no-change": "identity",
    "no_change": "identity",
    "fixed-value": "fixed",
    "fixed_value": "fixed_level",
    "fixedlevel": "fixed_level",
    "mirror": "reflection",
}
_SUPPORTED_METHODS = frozenset({"fixed", "fixed_level", "reference", "identity", "neutral", "reflection"})
_SCOPES = frozenset({"full_episode", "explicitly_conditional"})
_INAPPLICABLE_ACTIONS = frozenset({"abort_pattern", "continue_unmodified_with_warning"})


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _as_index_value_map(
    values: Mapping[int | str, Any] | Sequence[Any] | None,
    schema: InputSchema,
    indices: tuple[int, ...],
    *,
    label: str,
) -> dict[int, Any]:
    if values is None:
        return {}
    if isinstance(values, Mapping):
        result: dict[int, Any] = {}
        for key, value in values.items():
            try:
                index = schema.spec(key).index
            except (SchemaError, TypeError, ValueError) as exc:
                raise InterventionError(f"{label} contains unknown input {key!r}") from exc
            result[index] = value
        return result
    if isinstance(values, (str, bytes)):
        raise InterventionError(f"{label} must be a mapping or sequence")
    sequence = list(values)
    if len(sequence) != len(indices):
        raise InterventionError(
            f"{label} sequence has length {len(sequence)}, expected {len(indices)}"
        )
    return dict(zip(indices, sequence, strict=True))


@dataclass(frozen=True, slots=True)
class InterventionPattern:
    """A named intervention pattern resolved against an :class:`InputSchema`.

    ``method='fixed'`` requires one replacement value per target index.
    ``method='reference'`` obtains those values from one saved observation.
    ``method='identity'`` is the explicit P00/no-change pattern.
    """

    pattern_id: str
    indices: tuple[int | str, ...] = ()
    method: str = "fixed"
    values: Mapping[int | str, Any] | Sequence[Any] | None = None
    reference_id: str | None = None
    description: str = ""
    replacement_kind: str = "diagnostic"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    # ``scope`` and ``on_inapplicable`` are explicit because closed-loop
    # execution must distinguish a full-episode experiment from a legacy
    # reference that may continue on unmodified input after a mismatch.
    scope: str | None = None
    on_inapplicable: str | None = None
    variant_id: str | None = None
    variant: Mapping[str, Any] | str | None = None
    center: Any | None = None
    tolerance: Any | None = None
    clip: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.pattern_id, str) or not self.pattern_id.strip():
            raise InterventionError("pattern_id must be a non-empty string")
        if not isinstance(self.method, str):
            raise InterventionError("method must be a string")
        method = self.method.lower().strip()
        method = _METHOD_ALIASES.get(method, method)
        if method not in _SUPPORTED_METHODS:
            raise InterventionError(
                "method must be fixed, fixed_level, reference, identity, neutral, or reflection"
            )
        object.__setattr__(self, "method", method)
        if isinstance(self.indices, (str, bytes)):
            raise InterventionError("indices must be a sequence of indices")
        object.__setattr__(self, "indices", tuple(self.indices))
        if len(set(self.indices)) != len(self.indices):
            raise InterventionError("pattern indices must be unique")
        scope = self.scope
        if scope is None:
            # Existing reference patterns are compatibility experiments unless
            # their config opts into full_episode explicitly.  Typed local
            # operations have no donor context and therefore cover the whole
            # episode by default.
            scope = "explicitly_conditional" if method == "reference" else "full_episode"
        if not isinstance(scope, str):
            raise InterventionError("scope must be full_episode or explicitly_conditional")
        scope = scope.strip().lower().replace("-", "_")
        if scope not in _SCOPES:
            raise InterventionError("scope must be full_episode or explicitly_conditional")
        object.__setattr__(self, "scope", scope)
        on_inapplicable = self.on_inapplicable
        if on_inapplicable is None:
            on_inapplicable = (
                "continue_unmodified_with_warning"
                if scope == "explicitly_conditional"
                else "abort_pattern"
            )
        if not isinstance(on_inapplicable, str):
            raise InterventionError(
                "on_inapplicable must be abort_pattern or continue_unmodified_with_warning"
            )
        on_inapplicable = on_inapplicable.strip().lower().replace("-", "_")
        if on_inapplicable not in _INAPPLICABLE_ACTIONS:
            raise InterventionError(
                "on_inapplicable must be abort_pattern or continue_unmodified_with_warning"
            )
        object.__setattr__(self, "on_inapplicable", on_inapplicable)
        if self.variant_id is not None and (
            not isinstance(self.variant_id, str) or not self.variant_id.strip()
        ):
            raise InterventionError("variant_id must be a non-empty string when provided")
        if self.variant is not None and not isinstance(self.variant, (str, Mapping)):
            raise InterventionError("variant must be a variant id or mapping")
        if self.tolerance is not None:
            if isinstance(self.tolerance, Mapping):
                for key, value in self.tolerance.items():
                    try:
                        number = float(value)
                    except (TypeError, ValueError) as exc:
                        raise InterventionError(f"tolerance[{key!r}] must be finite and non-negative") from exc
                    if not np.isfinite(number) or number < 0:
                        raise InterventionError(f"tolerance[{key!r}] must be finite and non-negative")
            else:
                try:
                    number = float(self.tolerance)
                except (TypeError, ValueError) as exc:
                    raise InterventionError("tolerance must be finite and non-negative") from exc
                if not np.isfinite(number) or number < 0:
                    raise InterventionError("tolerance must be finite and non-negative")
        if not isinstance(self.clip, (bool, np.bool_)):
            raise InterventionError("clip must be boolean")
        object.__setattr__(self, "clip", bool(self.clip))
        if method == "reference" and not self.reference_id:
            # A reference may be passed directly to ``apply``; in that case the
            # id is optional.  Keep this pattern serializable and defer the
            # presence check until execution.
            pass

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InterventionPattern":
        if not isinstance(value, Mapping):
            raise InterventionError("pattern must be a mapping")
        indices = value.get("indices", value.get("targets", ()))
        method = value.get("method", value.get("operation", value.get("kind", "fixed")))
        values = value.get(
            "values",
            value.get("replacement_values", value.get("replacement")),
        )
        metadata = value.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, Mapping):
            raise InterventionError("pattern.metadata must be a mapping")
        variant_id = value.get("variant_id", value.get("variant_name"))
        variant = value.get("variant")
        if variant_id is None and isinstance(variant, str):
            variant_id = variant
            variant = None
        if variant_id is None:
            variant_id = metadata.get("variant_id")
        return cls(
            pattern_id=str(value.get("pattern_id", value.get("id", ""))),
            indices=tuple(indices),
            method=str(method),
            values=values,
            reference_id=value.get("reference_id"),
            description=str(value.get("description", "")),
            replacement_kind=str(value.get("replacement_kind", "diagnostic")),
            metadata=metadata,
            scope=value.get("scope"),
            on_inapplicable=value.get("on_inapplicable"),
            variant_id=variant_id,
            variant=variant,
            center=value.get("center"),
            tolerance=value.get("tolerance", value.get("meaningful_tolerance")),
            clip=value.get("clip", False),
        )

    def resolve_indices(self, schema: InputSchema) -> tuple[int, ...]:
        try:
            return schema.resolve_indices(self.indices)
        except SchemaError as exc:
            raise InterventionError(str(exc)) from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "indices": list(self.indices),
            "method": self.method,
            "values": _json_value(self.values),
            "reference_id": self.reference_id,
            "description": self.description,
            "replacement_kind": self.replacement_kind,
            "metadata": _json_value(self.metadata),
            "scope": self.scope,
            "on_inapplicable": self.on_inapplicable,
            "variant_id": self.variant_id,
            "variant": _json_value(self.variant),
            "center": _json_value(self.center),
            "tolerance": _json_value(self.tolerance),
            "clip": self.clip,
        }


@dataclass(frozen=True, slots=True)
class InterventionResult:
    """Result of applying one pattern to one observation."""

    observation: np.ndarray
    requested_indices: tuple[int, ...]
    changed_indices: tuple[int, ...]
    no_op_indices: tuple[int, ...]
    pattern_id: str
    method: str
    reference_id: str | None = None
    skipped: bool = False
    skip_reason: str | None = None
    skip_category: str | None = None
    eligible: bool = True
    scope: str = "full_episode"
    on_inapplicable: str = "abort_pattern"
    # Values are recorded after the observation dtype conversion.  This makes
    # exact changes auditable for float32 models while ``tolerance`` supports a
    # separate meaningful-change count.
    original_values: Mapping[int, Any] = field(default_factory=dict)
    applied_values: Mapping[int, Any] = field(default_factory=dict)
    delta_values: Mapping[int, float] = field(default_factory=dict)
    delta_abs_values: Mapping[int, float] = field(default_factory=dict)
    meaningful_changed_indices: tuple[int, ...] = ()
    tolerance: Mapping[int, float] = field(default_factory=dict)
    clipped_indices: tuple[int, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.changed_indices)

    @property
    def applied(self) -> bool:
        """Whether this target was eligible and received an intervention."""

        return bool(self.eligible and not self.skipped)

    @property
    def meaningful_changed(self) -> bool:
        """Whether at least one exact change exceeded its declared tolerance."""

        return bool(self.meaningful_changed_indices)

    @property
    def meaningful(self) -> bool:
        """Compatibility alias for :attr:`meaningful_changed`."""

        return self.meaningful_changed

    @property
    def applied_count(self) -> int:
        return 0 if self.skipped else len(self.requested_indices)

    @property
    def actual_change_count(self) -> int:
        return len(self.changed_indices)

    @property
    def no_op_count(self) -> int:
        return len(self.no_op_indices)

    @property
    def changed_count_exact(self) -> int:
        return self.actual_change_count

    @property
    def meaningful_changed_count(self) -> int:
        return len(self.meaningful_changed_indices)

    @property
    def skipped_count(self) -> int:
        return 0 if not self.skipped else len(self.requested_indices)

    @property
    def clip_count(self) -> int:
        return len(self.clipped_indices)

    @property
    def warning(self) -> str | None:
        return self.skip_reason if self.skipped and self.on_inapplicable == "continue_unmodified_with_warning" else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "method": self.method,
            "reference_id": self.reference_id,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "skip_category": self.skip_category,
            "applied": self.applied,
            "requested_indices": list(self.requested_indices),
            "changed_indices": list(self.changed_indices),
            "no_op_indices": list(self.no_op_indices),
            "changed": self.changed,
            "applied_count": self.applied_count,
            "actual_change_count": self.actual_change_count,
            "no_op_count": self.no_op_count,
            "changed_count_exact": self.changed_count_exact,
            "meaningful_changed_indices": list(self.meaningful_changed_indices),
            "meaningful_changed_count": self.meaningful_changed_count,
            "meaningful_changed": self.meaningful_changed,
            "eligible": self.eligible,
            "scope": self.scope,
            "on_inapplicable": self.on_inapplicable,
            "skipped_count": self.skipped_count,
            "original_values": _json_value(self.original_values),
            "applied_values": _json_value(self.applied_values),
            "delta_values": _json_value(self.delta_values),
            "delta_abs_values": _json_value(self.delta_abs_values),
            "tolerance": _json_value(self.tolerance),
            "clipped_indices": list(self.clipped_indices),
            "clip_count": self.clip_count,
        }


def _is_bool_spec(spec: Any) -> bool:
    return str(getattr(spec, "value_type", "")).casefold() in {"bool", "boolean"}


def _coupled_group(schema: InputSchema, index: int) -> set[int]:
    """Return the symmetric closure of one schema coupling declaration.

    Older schemas declared the group only on the validity flag while newer
    ones put ``related_valid_flags`` on each value.  Treat both forms as the
    same relation for safety checks, without requiring every independent
    value to be changed whenever its flag is valid.
    """

    group = {index}
    changed = True
    while changed:
        changed = False
        for item in schema.inputs:
            declared = {item.index, *item.coupled_indices, *item.related_valid_flags}
            if group.intersection(declared):
                before = len(group)
                group.update(declared)
                changed = changed or len(group) != before
    return group


def _related_flag_indices(schema: InputSchema, index: int) -> set[int]:
    spec = schema.spec(index)
    flags = set(spec.related_valid_flags)
    for candidate in schema.inputs:
        if not _is_bool_spec(candidate):
            continue
        if index in candidate.coupled_indices or candidate.index in spec.coupled_indices:
            flags.add(candidate.index)
    flags.discard(index)
    return flags


def _related_value_indices(schema: InputSchema, index: int) -> set[int]:
    spec = schema.spec(index)
    if not _is_bool_spec(spec):
        return set()
    values = {
        member
        for member in spec.coupled_indices
        if not _is_bool_spec(schema.spec(member)) and member != index
    }
    for candidate in schema.inputs:
        if _is_bool_spec(candidate):
            continue
        if index in candidate.related_valid_flags or index in candidate.coupled_indices:
            values.add(candidate.index)
    return values


def _flag_is_valid(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(np.isfinite(float(value)) and float(value) == 1.0)
    return False


def _flag_is_invalid(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return not bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(np.isfinite(float(value)) and float(value) == 0.0)
    return False


def _values_equal(left: Any, right: Any) -> bool:
    try:
        return bool(np.asarray(left == right).item())
    except Exception:
        return False


def _validate_coupling(
    pattern_indices: tuple[int, ...],
    schema: InputSchema,
    *,
    source: np.ndarray | None = None,
) -> str | None:
    selected = set(pattern_indices)
    for index in pattern_indices:
        spec = schema.spec(index)
        group = _coupled_group(schema, index)
        # A declared non-independent member must be evaluated with its whole
        # coherent group.  Independent offset/heading values are deliberately
        # exempt: they may be replaced alone while their valid flag remains 1.
        if not spec.independently_replaceable:
            missing = sorted(group - selected)
            if missing:
                return (
                    f"pattern changes non-independent input {index} without "
                    f"coupled indices {missing}"
                )
        if _is_bool_spec(spec):
            continue
        for flag_index in sorted(_related_flag_indices(schema, index)):
            if flag_index in selected:
                continue
            if source is not None and not _flag_is_valid(source[flag_index]):
                return (
                    f"pattern changes input {index} while related valid flag "
                    f"{flag_index} is not valid; change a coherent group or use a valid reference"
                )
    return None


def _reference_array(
    reference: Any,
    schema: InputSchema,
) -> np.ndarray:
    try:
        return schema.validate_observation(reference, copy=True)
    except SchemaError as exc:
        raise InterventionError(f"reference observation is invalid: {exc}") from exc


def _context_mismatch(
    pattern: InterventionPattern,
    context: Mapping[str, Any] | None,
    observation_context: Mapping[str, Any] | None = None,
) -> str | None:
    """Return a human-readable reason when a reference precondition is absent."""

    requirements = pattern.metadata.get(
        "reference_requirements",
        pattern.metadata.get("required_context", {}),
    )
    if requirements is None:
        requirements = {}
    if not isinstance(requirements, Mapping):
        return "reference requirements must be a mapping"
    if requirements and context is None:
        return "reference context is required but was not provided"
    for key, expected in requirements.items():
        if key not in context:  # type: ignore[operator]
            return f"reference context is missing {key!r}"
        actual = context[key]  # type: ignore[index]
        if actual != expected:
            return f"reference context {key!r}={actual!r} does not match required {expected!r}"
    compatibility_keys = pattern.metadata.get("compatibility_keys", ())
    if compatibility_keys is None:
        compatibility_keys = ()
    if not isinstance(compatibility_keys, Sequence) or isinstance(compatibility_keys, (str, bytes)):
        return "compatibility_keys must be a sequence"
    if compatibility_keys:
        if context is None or observation_context is None:
            return "observation and reference contexts are required for compatibility_keys"
        for key in compatibility_keys:
            if key not in context or key not in observation_context:
                return f"compatibility context is missing {key!r}"
            reference_value = context[key]
            observation_value = observation_context[key]
            if reference_value is None or observation_value is None:
                return f"compatibility context {key!r} is null"
            if reference_value != observation_value:
                return (
                    f"compatibility context {key!r} differs: "
                    f"reference={reference_value!r}, observation={observation_value!r}"
                )
    return None


def _precondition_mismatch(
    pattern: InterventionPattern,
    source: np.ndarray,
    schema: InputSchema,
) -> str | None:
    preconditions = pattern.metadata.get("preconditions", {})
    if preconditions is None:
        return None
    if not isinstance(preconditions, Mapping):
        return "preconditions must be a mapping"
    for key, expected in preconditions.items():
        try:
            index = schema.spec(key).index
        except (SchemaError, TypeError, ValueError) as exc:
            return f"precondition references unknown input {key!r}: {exc}"
        actual = source[index]
        try:
            equal = bool(np.asarray(actual == expected).item())
        except Exception:
            equal = False
        if not equal:
            return f"precondition {key!r}={expected!r} does not match observation value {actual!r}"
    return None


def _fixed_value_confirmation(
    pattern: InterventionPattern,
    index: int,
    value: Any,
    schema: InputSchema,
    *,
    allow_invalid_variant: bool = False,
) -> str | None:
    """Check schema-provided replacement metadata when present.

    A fully explicit pattern is allowed when a schema has no replacement value
    (ports commonly resolve it in an adapter).  If the schema does declare a
    replacement, however, an unconfirmed fixed value must never silently
    replace it.
    """

    spec = schema.spec(index)
    declared = spec.replacement
    if not declared:
        confirmed = pattern.metadata.get("replacement_confirmed")
        if confirmed is False:
            return "replacement is explicitly marked unconfirmed"
        return None
    if not isinstance(declared, Mapping):
        return "schema replacement metadata is not a mapping"
    kind = declared.get("kind", declared.get("method", ""))
    if kind in {"reference", "real_observation", "saved_reference"}:
        return "schema requires a saved reference replacement"
    if kind not in {"fixed", "fixed_value", "confirmed_fixed"}:
        return f"schema replacement kind {kind!r} is unresolved for a fixed intervention"
    if declared.get("confirmed") is False or declared.get("status") in {"unset", "unknown", "unconfirmed"}:
        return "schema replacement value is not confirmed"
    if allow_invalid_variant:
        # The complete joint, including dtype conversion and clipping, is
        # checked once the final values have been assigned.  Do not compare a
        # pre-clip value here: an explicit ``clip=True`` may intentionally
        # produce the confirmed sentinel.  Returning here also permits every
        # member of the joint to pass this legacy fixed-value check; the
        # post-assignment joint validator then rejects any member mismatch.
        return None
    if "value" in declared or "fixed_value" in declared:
        expected = declared.get("value", declared.get("fixed_value"))
        if not _values_equal(expected, value):
            return f"fixed value differs from schema-confirmed replacement {expected!r}"
    return None


def _skipped_result(
    source: np.ndarray,
    pattern: InterventionPattern,
    indices: tuple[int, ...],
    reason: str,
    *,
    eligible: bool = True,
    skip_category: str = "failed",
) -> InterventionResult:
    return InterventionResult(
        observation=source,
        requested_indices=indices,
        changed_indices=(),
        no_op_indices=(),
        pattern_id=pattern.pattern_id,
        method=pattern.method,
        reference_id=pattern.reference_id,
        skipped=True,
        skip_reason=reason,
        skip_category=skip_category,
        eligible=eligible,
        scope=pattern.scope or "full_episode",
        on_inapplicable=pattern.on_inapplicable or "abort_pattern",
    )


def _variant_record(
    pattern: InterventionPattern,
    schema: InputSchema,
    index: int,
) -> Mapping[str, Any] | None:
    """Resolve one pattern variant without guessing from an input index."""

    candidate = pattern.variant
    if candidate is None:
        candidate = pattern.metadata.get("variant")
    if isinstance(candidate, str):
        try:
            return schema.variant(index, candidate)
        except SchemaError as exc:
            raise InterventionError(str(exc)) from exc
    if isinstance(candidate, Mapping):
        return candidate
    variant_id = pattern.variant_id or pattern.metadata.get("variant_id")
    if variant_id is None:
        return None
    if not isinstance(variant_id, str) or not variant_id.strip():
        raise InterventionError("variant_id must be a non-empty string")
    try:
        return schema.variant(index, variant_id)
    except SchemaError as exc:
        raise InterventionError(str(exc)) from exc


def _variant_operation(record: Mapping[str, Any], pattern_id: str) -> str:
    operation = record.get("operation", record.get("method", record.get("kind")))
    if not isinstance(operation, str) or not operation.strip():
        raise InterventionError(f"pattern {pattern_id} variant must declare operation")
    operation = operation.strip().casefold().replace("-", "_")
    if operation not in {"identity", "neutral", "reflection", "fixed_level", "fixed", "reference"}:
        raise InterventionError(f"pattern {pattern_id} variant operation {operation!r} is unsupported")
    return operation


def _variant_confirmation_reason(
    record: Mapping[str, Any],
    pattern_id: str,
    operation: str,
) -> str | None:
    confirmed = record.get("confirmed", False)
    if type(confirmed) is not bool:
        return "variant.confirmed must be a boolean"
    if not confirmed:
        return "variant is not source-confirmed"
    meaning = record.get("meaning", record.get("description"))
    evidence = record.get(
        "evidence",
        record.get("source", record.get("reference", record.get("provenance"))),
    )
    if not isinstance(meaning, str) or not meaning.strip():
        return "confirmed variant requires meaning/description"
    if not isinstance(evidence, str) or not evidence.strip():
        return "confirmed variant requires evidence/source provenance"
    if operation in {"neutral", "fixed_level", "fixed"} and not any(
        key in record for key in ("value", "values", "level", "levels")
    ):
        return f"{operation} variant requires value/level"
    if operation == "reflection" and not any(key in record for key in ("center", "centers")):
        return "reflection variant requires center"
    return None


def _variant_mapping_for_indices(
    value: Any,
    schema: InputSchema,
    indices: tuple[int, ...],
    *,
    label: str,
) -> dict[int, Any]:
    """Accept one scalar, per-input mapping, or target-ordered sequence."""

    if isinstance(value, Mapping):
        result = _as_index_value_map(value, schema, indices, label=label)
        missing = sorted(set(indices) - set(result))
        if missing:
            raise InterventionError(f"{label} is missing values for indices {missing}")
        return result
    if isinstance(value, (str, bytes)) or np.isscalar(value):
        return {index: value for index in indices}
    return _as_index_value_map(value, schema, indices, label=label)


def _variant_declared_range(
    record: Mapping[str, Any],
    pattern_id: str,
) -> tuple[float | None, float | None] | None:
    """Read a variant's optional generation domain for inline records."""

    raw = record.get("range", record.get("value_range"))
    if raw is None:
        return None
    if isinstance(raw, (str, bytes)):
        raise InterventionError(f"pattern {pattern_id} variant range must contain two bounds")
    try:
        pair = tuple(raw)
    except TypeError as exc:
        raise InterventionError(f"pattern {pattern_id} variant range must contain two bounds") from exc
    if len(pair) != 2:
        raise InterventionError(f"pattern {pattern_id} variant range must contain two bounds")
    result: list[float | None] = []
    for bound in pair:
        if bound is None:
            result.append(None)
            continue
        try:
            number = float(bound)
        except (TypeError, ValueError) as exc:
            raise InterventionError(f"pattern {pattern_id} variant range must be numeric") from exc
        if not np.isfinite(number):
            raise InterventionError(f"pattern {pattern_id} variant range must be finite")
        result.append(number)
    if result[0] is not None and result[1] is not None and result[0] > result[1]:
        raise InterventionError(f"pattern {pattern_id} variant range minimum exceeds maximum")
    return result[0], result[1]


def _variant_domain_reason(
    value: Any,
    bounds: tuple[float | None, float | None],
    *,
    label: str,
) -> str | None:
    """Return a reason when a generated value leaves the variant domain."""

    minimum, maximum = bounds
    try:
        number = float(value)
    except (TypeError, ValueError):
        return f"{label} must be numeric for the declared variant range"
    if not np.isfinite(number):
        return f"{label} must be finite for the declared variant range"
    if minimum is not None and number < minimum:
        return f"{label}={number} is below the variant range minimum {minimum}"
    if maximum is not None and number > maximum:
        return f"{label}={number} exceeds the variant range maximum {maximum}"
    return None


def _contains_unsigned_marker(spec: Any) -> bool:
    fields = (
        getattr(spec, "id", ""),
        getattr(spec, "name_ja", ""),
        getattr(spec, "description", ""),
        getattr(spec, "physical_quantity", ""),
        getattr(spec, "model_representation", ""),
        getattr(spec, "normalization", ""),
    )
    def _structured_text(value: Any) -> list[str]:
        if isinstance(value, Mapping):
            result: list[str] = []
            for key, nested in value.items():
                result.append(str(key))
                result.extend(_structured_text(nested))
            return result
        if isinstance(value, (list, tuple)):
            result = []
            for nested in value:
                result.extend(_structured_text(nested))
            return result
        return [str(value)]

    structured = [
        getattr(spec, "replacement", None),
        getattr(spec, "variants", ()),
    ]
    text = " ".join(
        [str(value) for value in fields]
        + [part for value in structured for part in _structured_text(value)]
    ).casefold()
    return "unsigned" in text or "符号なし" in text or "non-negative" in text


def _variant_compatibility_reason(
    record: Mapping[str, Any],
    spec: Any,
    index: int,
    indices: tuple[int, ...],
) -> str | None:
    allowed_indices = record.get("allowed_indices")
    if allowed_indices is not None:
        try:
            allowed_index_set = {int(value) for value in allowed_indices}
        except (TypeError, ValueError):
            return "variant allowed_indices must contain integers"
        if index not in allowed_index_set:
            return f"variant is not declared for input index {index}"
    allowed_ids = record.get("allowed_ids", record.get("semantic_ids"))
    if allowed_ids is not None and spec.id not in set(allowed_ids):
        return f"variant is not declared for semantic input {spec.id!r}"
    allowed_groups = record.get("allowed_groups")
    if allowed_groups is not None and spec.group not in set(allowed_groups):
        return f"variant is not declared for input group {spec.group!r}"
    declared_type = record.get("value_type", record.get("dtype"))
    if declared_type is not None:
        if not isinstance(declared_type, str):
            return f"variant input {index} value_type must be a string"
        normalized = declared_type.casefold()
        if normalized == "boolean":
            normalized = "bool"
        if normalized in {"category", "categorical"}:
            normalized = "categorical"
        expected = "categorical" if spec.value_type in {"category", "categorical"} else spec.value_type
        if normalized != expected:
            return f"variant input {index} value_type={declared_type!r} conflicts with schema {expected!r}"
    for key in ("coupled_indices", "related_valid_flags"):
        declared = record.get(key, ())
        if declared is None:
            continue
        try:
            declared_set = {int(value) for value in declared}
        except (TypeError, ValueError) as exc:
            return f"variant input {index} {key} must contain integer indices"
        if key == "coupled_indices" and not declared_set.issubset(set(indices)):
            missing = sorted(declared_set - set(indices))
            return f"variant input {index} omits coupled indices {missing}"
    categories = record.get("categories", record.get("category_values"))
    if categories is not None and not spec.is_discrete:
        return f"variant input {index} declares categories for a continuous input"
    variant_range = _variant_declared_range(record, f"input {index}")
    if variant_range is not None and spec.value_range is not None:
        variant_minimum, variant_maximum = variant_range
        spec_minimum, spec_maximum = spec.value_range
        if (
            variant_minimum is not None
            and spec_minimum is not None
            and variant_minimum < spec_minimum
        ):
            return f"variant input {index} range extends below the schema range"
        if (
            variant_maximum is not None
            and spec_maximum is not None
            and variant_maximum > spec_maximum
        ):
            return f"variant input {index} range extends above the schema range"
    return None


def _typed_replacement_values(
    pattern: InterventionPattern,
    source: np.ndarray,
    schema: InputSchema,
    indices: tuple[int, ...],
) -> tuple[dict[int, Any], dict[int, Mapping[str, Any]], set[int]]:
    """Resolve neutral/reflection/fixed_level into explicit values.

    The returned values are still uncast.  Casting, exact-change accounting,
    and optional clipping happen in :func:`apply_intervention`.
    """

    records: dict[int, Mapping[str, Any]] = {}
    values: dict[int, Any] = {}
    if pattern.values is not None:
        explicit = _as_index_value_map(
            pattern.values,
            schema,
            indices,
            label=f"pattern {pattern.pattern_id}.values",
        )
    else:
        explicit = {}
    for index in indices:
        spec = schema.spec(index)
        if pattern.method == "reflection" and getattr(spec, "is_discrete", False):
            raise InterventionError(
                f"pattern {pattern.pattern_id}[{index}] cannot reflect discrete input {spec.id!r}"
            )
        record = _variant_record(pattern, schema, index)
        if pattern.method == "reflection" and _contains_unsigned_marker(spec):
            raise InterventionError(
                f"pattern {pattern.pattern_id}[{index}] cannot reflect unsigned input {spec.id!r}"
            )
        if record is None:
            raise InterventionError(
                f"pattern {pattern.pattern_id}[{index}] requires a registered or inline confirmed variant"
            )
        if record is not None:
            operation = _variant_operation(record, pattern.pattern_id)
            reason = _variant_confirmation_reason(record, pattern.pattern_id, operation)
            if reason is not None:
                raise InterventionError(f"pattern {pattern.pattern_id}[{index}] rejected: {reason}")
            compatibility = _variant_compatibility_reason(record, spec, index, indices)
            if compatibility is not None:
                raise InterventionError(f"pattern {pattern.pattern_id}[{index}] rejected: {compatibility}")
            records[index] = record
            expected_operations = {
                "neutral": {"neutral"},
                "reflection": {"reflection"},
                "fixed_level": {"fixed_level", "fixed"},
            }
            if operation not in expected_operations[pattern.method]:
                return_value = (
                    f"variant operation {operation!r} does not match method {pattern.method!r}"
                )
                raise InterventionError(f"pattern {pattern.pattern_id}[{index}] rejected: {return_value}")
            declared_range = _variant_declared_range(record, pattern.pattern_id)
        else:  # pragma: no cover - guarded above; keeps type checkers honest
            declared_range = None
        if pattern.method == "reflection":
            record_center = None if record is None else record.get("centers", record.get("center"))
            center = pattern.center if pattern.center is not None else record_center
            if center is None:
                # A metadata center is accepted for compact legacy mappings.
                center = pattern.metadata.get("center")
            if center is None:
                raise InterventionError(
                    f"pattern {pattern.pattern_id} reflection requires a source-confirmed center"
                )
            center_map = _variant_mapping_for_indices(
                center, schema, indices, label=f"pattern {pattern.pattern_id}.center"
            )
            declared_center = record.get("centers", record.get("center"))
            if declared_center is not None and pattern.center is not None:
                declared_center_map = _variant_mapping_for_indices(
                    declared_center,
                    schema,
                    indices,
                    label=f"pattern {pattern.pattern_id}.variant.center",
                )
                if not _values_equal(center_map[index], declared_center_map[index]):
                    raise InterventionError(
                        f"pattern {pattern.pattern_id}[{index}] center conflicts with the confirmed variant"
                    )
            try:
                center_value = float(center_map[index])
            except (TypeError, ValueError) as exc:
                raise InterventionError(
                    f"pattern {pattern.pattern_id}[{index}] center must be finite"
                ) from exc
            if not np.isfinite(center_value):
                raise InterventionError(f"pattern {pattern.pattern_id}[{index}] center must be finite")
            if declared_range is not None:
                reason = _variant_domain_reason(
                    center_value,
                    declared_range,
                    label=f"pattern {pattern.pattern_id}[{index}] center",
                )
                if reason is not None:
                    raise InterventionError(reason)
            values[index] = 2.0 * center_value - float(source[index])
            if declared_range is not None:
                reason = _variant_domain_reason(
                    values[index],
                    declared_range,
                    label=f"pattern {pattern.pattern_id}[{index}] reflected value",
                )
                if reason is not None:
                    raise InterventionError(reason)
        elif record is not None:
            raw = record.get("values", record.get("levels"))
            if raw is None:
                raw = record.get("level", record.get("value"))
            if raw is None:
                raise InterventionError(
                    f"pattern {pattern.pattern_id}[{index}] variant has no replacement value"
                )
            declared_values = _variant_mapping_for_indices(
                raw,
                schema,
                indices,
                label=f"pattern {pattern.pattern_id}.variant.values",
            )
            if index in explicit and not _values_equal(explicit[index], declared_values[index]):
                raise InterventionError(
                    f"pattern {pattern.pattern_id}[{index}] value conflicts with the confirmed variant"
                )
            values[index] = explicit[index] if index in explicit else declared_values[index]
            if declared_range is not None:
                reason = _variant_domain_reason(
                    values[index],
                    declared_range,
                    label=f"pattern {pattern.pattern_id}[{index}] value",
                )
                if reason is not None:
                    raise InterventionError(reason)
    # Scalar records may have populated more than the current index; retain
    # only requested targets and fail closed on omissions.
    if set(values) != set(indices):
        missing = sorted(set(indices) - set(values))
        raise InterventionError(f"pattern {pattern.pattern_id} has no replacement value for indices {missing}")
    for index, record in records.items():
        categories = record.get("categories", record.get("category_values"))
        if categories is None:
            continue
        candidate = values[index]
        if isinstance(candidate, (float, np.floating)) and not np.isfinite(float(candidate)):
            raise InterventionError(
                f"pattern {pattern.pattern_id}[{index}] category replacement is non-finite"
            )
        if not any(_values_equal(candidate, allowed) for allowed in categories):
            raise InterventionError(
                f"pattern {pattern.pattern_id}[{index}] value {candidate!r} is outside declared categories"
            )
    return values, records, set()


def _tolerance_map(pattern: InterventionPattern, indices: tuple[int, ...]) -> dict[int, float]:
    raw = pattern.tolerance
    if raw is None:
        raw = pattern.metadata.get("meaningful_tolerance", pattern.metadata.get("tolerance", 0.0))
    if isinstance(raw, Mapping):
        values: dict[int, float] = {}
        # Mapping keys may be semantic ids, but integer and decimal-index keys
        # cover the portable result format; semantic mappings are handled by
        # the pattern constructor before execution.
        for index in indices:
            candidate = raw.get(index, raw.get(str(index), 0.0))
            if candidate is None:
                candidate = 0.0
            number = float(candidate)
            if not np.isfinite(number) or number < 0:
                raise InterventionError(f"tolerance[{index}] must be finite and non-negative")
            values[index] = number
        return values
    number = float(raw or 0.0)
    if not np.isfinite(number) or number < 0:
        raise InterventionError("tolerance must be finite and non-negative")
    return {index: number for index in indices}


def _invalid_joint_evidence_entry(
    evidence: Mapping[Any, Any],
    schema: InputSchema,
    index: int,
) -> Mapping[str, Any] | None:
    """Resolve explicit per-member evidence for a confirmed invalid group."""

    spec = schema.spec(index)
    for key in (index, str(index), spec.id):
        candidate = evidence.get(key)
        if isinstance(candidate, Mapping):
            return candidate
    return None


def _invalid_joint_evidence_reason(
    pattern: InterventionPattern,
    replacement_values: Mapping[int, Any],
    group: set[int],
    schema: InputSchema,
) -> str | None:
    """Validate explicit evidence when no invalid_value sentinel is registered."""

    raw = pattern.metadata.get("invalid_joint_evidence")
    if not isinstance(raw, Mapping):
        return None
    for member in sorted(group):
        entry = _invalid_joint_evidence_entry(raw, schema, member)
        if entry is None:
            return (
                f"invalid flag/value group lacks explicit confirmed evidence for input {member}"
            )
        if entry.get("confirmed") is not True:
            return (
                f"invalid flag/value group has unconfirmed evidence for input {member}"
            )
        provenance = entry.get(
            "evidence",
            entry.get("source", entry.get("provenance")),
        )
        if not isinstance(provenance, str) or not provenance.strip():
            return (
                f"invalid flag/value group lacks evidence provenance for input {member}"
            )
        expected = entry.get("invalid_value", entry.get("value"))
        if expected is None:
            return (
                f"invalid flag/value group evidence lacks a value for input {member}"
            )
        if not _values_equal(replacement_values.get(member), expected):
            return (
                f"invalid flag/value group evidence value mismatch at input {member}: "
                f"expected {expected!r}"
            )
    return ""


def _invalid_value_provenance(spec: Any, declared: Mapping[str, Any]) -> str | None:
    """Return one explicit source string for a schema invalid sentinel."""

    for key in ("evidence", "source", "provenance", "reference"):
        value = declared.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    # Existing portable schemas keep the source citation on the input row
    # rather than repeating it inside ``replacement``.  Keep that format
    # valid while still rejecting a bare ``confirmed`` boolean with no source
    # anywhere in the row.
    for key in ("source", "reference"):
        value = getattr(spec, key, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _invalid_joint_member_reason(
    pattern: InterventionPattern,
    replacement_values: Mapping[int, Any],
    member: int,
    schema: InputSchema,
) -> str | None:
    """Validate one final member against an explicit invalid-state source."""

    spec = schema.spec(member)
    declared = spec.replacement
    if isinstance(declared, Mapping) and "invalid_value" in declared:
        if declared.get("confirmed") is not True:
            return f"invalid flag/value group has unconfirmed invalid_value at input {member}"
        if _invalid_value_provenance(spec, declared) is None:
            return f"invalid flag/value group lacks evidence provenance for input {member}"
        expected = declared["invalid_value"]
        if not _values_equal(replacement_values.get(member), expected):
            return (
                f"invalid flag/value group has an invalid sentinel mismatch at input {member}: "
                f"expected {expected!r}"
            )
        return None

    raw = pattern.metadata.get("invalid_joint_evidence")
    if isinstance(raw, Mapping):
        entry = _invalid_joint_evidence_entry(raw, schema, member)
        if entry is None:
            return f"invalid flag/value group lacks explicit confirmed evidence for input {member}"
        if entry.get("confirmed") is not True:
            return f"invalid flag/value group has unconfirmed evidence for input {member}"
        provenance = entry.get(
            "evidence",
            entry.get("source", entry.get("provenance")),
        )
        if not isinstance(provenance, str) or not provenance.strip():
            return f"invalid flag/value group lacks evidence provenance for input {member}"
        expected = entry.get("invalid_value", entry.get("value"))
        if expected is None:
            return f"invalid flag/value group evidence lacks a value for input {member}"
        if not _values_equal(replacement_values.get(member), expected):
            return (
                f"invalid flag/value group evidence value mismatch at input {member}: "
                f"expected {expected!r}"
            )
        return None
    return (
        f"invalid flag/value group at input {member} requires a source-confirmed "
        "invalid_value or explicit confirmed evidence"
    )


def _invalid_flag_candidate(
    pattern: InterventionPattern,
    spec: Any,
    value: Any,
) -> bool:
    """Recognize an invalid flag after an explicitly requested clip."""

    if _flag_is_invalid(value):
        return True
    if not (pattern.clip or bool(pattern.metadata.get("clip", False))):
        return False
    value_range = getattr(spec, "value_range", None)
    if value_range is None or not isinstance(value, (int, float, np.integer, np.floating)):
        return False
    candidate = float(value)
    if not np.isfinite(candidate):
        return False
    minimum, maximum = value_range
    outside = (minimum is not None and candidate < minimum) or (
        maximum is not None and candidate > maximum
    )
    if not outside:
        return False
    low = -np.inf if minimum is None else minimum
    high = np.inf if maximum is None else maximum
    return _flag_is_invalid(float(np.clip(candidate, low, high)))


def _invalid_joint_reason(
    pattern: InterventionPattern,
    replacement_values: Mapping[int, Any],
    indices: tuple[int, ...],
    schema: InputSchema,
    *,
    validate_values: bool = True,
) -> str | None:
    """Require explicit confirmation before creating a flag-invalid joint state."""

    for index in indices:
        spec = schema.spec(index)
        if spec.value_type not in {"bool", "boolean"}:
            continue
        value = replacement_values.get(index)
        is_invalid = _invalid_flag_candidate(pattern, spec, value)
        related_values = _related_value_indices(schema, index)
        if not is_invalid or not related_values:
            continue
        group = _coupled_group(schema, index)
        missing_values = sorted(group - set(indices))
        if missing_values:
            return (
                f"invalid flag/value group at input {index} is missing related "
                f"values {missing_values}"
            )
        if not validate_values:
            continue
        for member in sorted(group):
            reason = _invalid_joint_member_reason(
                pattern, replacement_values, member, schema
            )
            if reason is not None:
                return reason
    return None


def apply_intervention(
    observation: Any,
    pattern: InterventionPattern | Mapping[str, Any],
    schema: InputSchema,
    *,
    reference_observation: Any | None = None,
    reference_observations: Mapping[str, Any] | None = None,
    reference_context: Mapping[str, Any] | None = None,
    reference_contexts: Mapping[str, Mapping[str, Any]] | None = None,
    observation_context: Mapping[str, Any] | None = None,
    strict: bool = True,
) -> InterventionResult:
    """Apply a pattern to a copy of ``observation``.

    ``reference_observation`` is a single saved vector.  For convenience a
    mapping of ``reference_id`` to vectors can be supplied as well.  A
    reference replacement always reads all target values from the same vector;
    this prevents accidental mixing of independently observed states.
    """

    if not isinstance(pattern, InterventionPattern):
        pattern = InterventionPattern.from_dict(pattern)
    try:
        source = schema.validate_observation(observation, copy=True)
    except SchemaError as exc:
        raise InterventionError(f"source observation is invalid: {exc}") from exc
    indices = pattern.resolve_indices(schema)
    configured_skip = pattern.metadata.get("skip_reason")
    if configured_skip:
        return _skipped_result(
            source,
            pattern,
            indices,
            str(configured_skip),
            eligible=False,
            skip_category=str(pattern.metadata.get("skip_category", "declared_out_of_scope")),
        )
    coupling_reason = _validate_coupling(indices, schema, source=source)
    if coupling_reason is not None:
        if strict:
            raise InterventionError(coupling_reason)
        return _skipped_result(source, pattern, indices, coupling_reason)
    precondition_reason = _precondition_mismatch(pattern, source, schema)
    if precondition_reason is not None:
        if strict:
            raise InterventionError(precondition_reason)
        return _skipped_result(source, pattern, indices, precondition_reason)
    context = reference_context
    if context is None and pattern.reference_id and reference_contexts is not None:
        context = reference_contexts.get(pattern.reference_id)
    context_reason = _context_mismatch(pattern, context, observation_context)
    if context_reason is not None:
        if strict:
            raise InterventionError(f"pattern {pattern.pattern_id} skipped: {context_reason}")
        return _skipped_result(source, pattern, indices, context_reason)
    if pattern.method == "identity":
        return InterventionResult(
            observation=source,
            requested_indices=indices,
            changed_indices=(),
            no_op_indices=indices,
            pattern_id=pattern.pattern_id,
            method=pattern.method,
            reference_id=pattern.reference_id,
            scope=pattern.scope or "full_episode",
            on_inapplicable=pattern.on_inapplicable or "abort_pattern",
            original_values={index: source[index] for index in indices},
            applied_values={index: source[index] for index in indices},
            delta_values={index: 0.0 for index in indices},
            delta_abs_values={index: 0.0 for index in indices},
            tolerance={index: 0.0 for index in indices},
        )

    typed_records: dict[int, Mapping[str, Any]] = {}
    if pattern.method in {"neutral", "reflection", "fixed_level"}:
        try:
            replacement_values, typed_records, _ = _typed_replacement_values(
                pattern, source, schema, indices
            )
        except InterventionError as exc:
            if strict:
                raise
            return _skipped_result(source, pattern, indices, str(exc))
    elif pattern.method == "reference":
        ref = reference_observation
        if ref is None and pattern.reference_id and reference_observations is not None:
            try:
                ref = reference_observations[pattern.reference_id]
            except KeyError as exc:
                reason = f"unknown reference observation id: {pattern.reference_id}"
                if strict:
                    raise InterventionError(reason) from exc
                return _skipped_result(source, pattern, indices, reason)
        if ref is None:
            reason = f"pattern {pattern.pattern_id} requires a saved reference observation"
            if strict:
                raise InterventionError(reason)
            return _skipped_result(source, pattern, indices, reason)
        try:
            ref_array = _reference_array(ref, schema)
        except InterventionError as exc:
            if strict:
                raise
            return _skipped_result(source, pattern, indices, str(exc))
        replacement_values = {index: ref_array[index] for index in indices}
    else:
        # Legacy ``method=fixed`` remains intentionally strict against a
        # schema-confirmed replacement value.  This is separate from the new
        # fixed_level variant operation, which can register multiple levels.
        if pattern.values is None:
            replacement_values = {}
            for index in indices:
                declared = schema.spec(index).replacement
                if isinstance(declared, Mapping) and (
                    "value" in declared or "fixed_value" in declared
                ):
                    replacement_values[index] = declared.get(
                        "value", declared.get("fixed_value")
                    )
        else:
            replacement_values = _as_index_value_map(
                pattern.values,
                schema,
                indices,
                label=f"pattern {pattern.pattern_id}.values",
            )
        if set(replacement_values) != set(indices):
            missing = sorted(set(indices) - set(replacement_values))
            raise InterventionError(
                f"pattern {pattern.pattern_id} has no replacement value for indices {missing}"
            )

    # Check group membership before any assignment.  Value equality is checked
    # again below against the values the model actually receives, after dtype
    # conversion and optional clipping.
    invalid_joint_reason = _invalid_joint_reason(
        pattern,
        replacement_values,
        indices,
        schema,
        validate_values=False,
    )
    if invalid_joint_reason is not None:
        if strict:
            raise InterventionError(invalid_joint_reason)
        return _skipped_result(source, pattern, indices, invalid_joint_reason)

    invalid_variant_indices: set[int] = set()
    for index in indices:
        spec = schema.spec(index)
        value = replacement_values.get(index)
        is_invalid_flag = _is_bool_spec(spec) and _invalid_flag_candidate(pattern, spec, value)
        if is_invalid_flag:
            invalid_variant_indices.update(_coupled_group(schema, index) & set(indices))

    tolerance = _tolerance_map(pattern, indices)
    result = np.array(source, copy=True)
    changed: list[int] = []
    no_op: list[int] = []
    meaningful_changed: list[int] = []
    original_values: dict[int, Any] = {}
    applied_values: dict[int, Any] = {}
    delta_values: dict[int, float] = {}
    delta_abs_values: dict[int, float] = {}
    clipped_indices: list[int] = []
    for index in indices:
        spec = schema.spec(index)
        value = replacement_values[index]
        if pattern.method == "fixed":
            confirmation_reason = _fixed_value_confirmation(
                pattern,
                index,
                value,
                schema,
                allow_invalid_variant=index in invalid_variant_indices,
            )
            if confirmation_reason is not None:
                if strict:
                    raise InterventionError(
                        f"pattern {pattern.pattern_id}[{index}] rejected: {confirmation_reason}"
                    )
                return _skipped_result(source, pattern, indices, confirmation_reason)
        variant = typed_records.get(index)
        clip = pattern.clip or bool(pattern.metadata.get("clip", False))
        if variant is not None and isinstance(variant.get("clip"), bool):
            clip = clip or bool(variant.get("clip"))
        if spec.value_range is not None and isinstance(value, (int, float, np.integer, np.floating)):
            minimum, maximum = spec.value_range
            candidate = float(value)
            if not np.isfinite(candidate):
                raise InterventionError(
                    f"pattern {pattern.pattern_id}[{index}] is non-finite; clip cannot repair NaN/Inf"
                )
            outside = (minimum is not None and candidate < minimum) or (
                maximum is not None and candidate > maximum
            )
            if outside and clip:
                low = -np.inf if minimum is None else minimum
                high = np.inf if maximum is None else maximum
                value = float(np.clip(candidate, low, high))
                clipped_indices.append(index)
        try:
            spec.validate_value(value, location=f"pattern {pattern.pattern_id}[{index}]")
        except SchemaError as exc:
            raise InterventionError(str(exc)) from exc
        # Assigning a float replacement into an integer source array can lose
        # information silently, so reject it unless numpy can represent the
        # value exactly.  Standard SB3 vectors are float arrays.
        old = result[index].item() if isinstance(result[index], np.generic) else result[index]
        cast_value = np.asarray(value, dtype=result.dtype).item()
        if strict and isinstance(value, (float, np.floating)) and np.issubdtype(result.dtype, np.integer):
            if float(cast_value) != float(value):
                raise InterventionError(
                    f"replacement at index {index} cannot be represented by observation dtype {result.dtype}"
                )
        result[index] = cast_value
        final_value = result[index].item() if isinstance(result[index], np.generic) else result[index]
        try:
            # Validate the value that the model actually receives.  This
            # catches float32 overflow and category/range changes introduced
            # by dtype conversion.
            spec.validate_value(final_value, location=f"pattern {pattern.pattern_id}[{index}] after dtype cast")
        except SchemaError as exc:
            raise InterventionError(str(exc)) from exc
        original_values[index] = old
        applied_values[index] = final_value
        try:
            delta = float(final_value) - float(old)
            delta_values[index] = delta
            delta_abs_values[index] = abs(delta)
        except (TypeError, ValueError):
            delta_values[index] = float("nan")
            delta_abs_values[index] = float("nan")
        if bool(np.asarray(old != final_value).item()):
            changed.append(index)
            threshold = tolerance.get(index, 0.0)
            if np.isfinite(delta_abs_values[index]) and delta_abs_values[index] > threshold:
                meaningful_changed.append(index)
        else:
            no_op.append(index)
    invalid_joint_reason = _invalid_joint_reason(
        pattern,
        applied_values,
        indices,
        schema,
    )
    if invalid_joint_reason is not None:
        if strict:
            raise InterventionError(invalid_joint_reason)
        return _skipped_result(source, pattern, indices, invalid_joint_reason)
    return InterventionResult(
        observation=result,
        requested_indices=indices,
        changed_indices=tuple(changed),
        no_op_indices=tuple(no_op),
        pattern_id=pattern.pattern_id,
        method=pattern.method,
        reference_id=pattern.reference_id,
        scope=pattern.scope or "full_episode",
        on_inapplicable=pattern.on_inapplicable or "abort_pattern",
        original_values=original_values,
        applied_values=applied_values,
        delta_values=delta_values,
        delta_abs_values=delta_abs_values,
        meaningful_changed_indices=tuple(meaningful_changed),
        tolerance=tolerance,
        clipped_indices=tuple(clipped_indices),
    )


def apply_pattern(*args: Any, **kwargs: Any) -> InterventionResult:
    """Backward-friendly alias for :func:`apply_intervention`."""

    return apply_intervention(*args, **kwargs)


def patterns_from_config(values: Iterable[Mapping[str, Any] | InterventionPattern]) -> tuple[InterventionPattern, ...]:
    result: list[InterventionPattern] = []
    for value in values:
        result.append(value if isinstance(value, InterventionPattern) else InterventionPattern.from_dict(value))
    if len({item.pattern_id for item in result}) != len(result):
        raise InterventionError("pattern ids must be unique")
    return tuple(result)


def _schema_fixed_value(spec: Any) -> tuple[bool, Any | None]:
    declared = getattr(spec, "replacement", None)
    if not isinstance(declared, Mapping):
        return False, None
    kind = declared.get("kind", declared.get("method", ""))
    if kind not in {"fixed", "fixed_value", "confirmed_fixed"}:
        return False, None
    if declared.get("confirmed") is False or declared.get("status") in {"unset", "unknown", "unconfirmed"}:
        return False, None
    if "value" not in declared and "fixed_value" not in declared:
        return False, None
    return True, declared.get("value", declared.get("fixed_value"))


def _preferred_typed_variant(spec: Any) -> Mapping[str, Any] | None:
    """Pick one source-confirmed local variant for the generated A detail row.

    The generator deliberately emits one row per input for compatibility with
    existing reports.  Additional stress variants belong in explicit config
    entries, where their IDs and scope are visible to reviewers.
    """

    variants = getattr(spec, "variants", ())
    for preferred in ("neutral", "fixed_level", "reflection"):
        for variant in variants:
            operation = str(variant.get("operation", variant.get("method", ""))).casefold().replace("-", "_")
            if variant.get("confirmed") is True and operation == preferred:
                return variant
    return None


def generate_individual_patterns(
    schema: InputSchema,
    *,
    reference_id: str | None = None,
    include_non_independent: bool = True,
) -> tuple[InterventionPattern, ...]:
    """Build the complete ①-A input table without guessing neutral values.

    Independent dimensions become individual patterns.  A source-confirmed
    fixed replacement is used when the schema supplies one; otherwise an
    explicit saved reference id is required.  Dimensions that cannot be
    independently changed still receive a skipped per-index row and one
    coherent coupled-group pattern, preserving full index coverage without
    pretending that a flag-only intervention is valid.
    """

    schema.validate()
    result: list[InterventionPattern] = []
    emitted_groups: set[tuple[int, ...]] = set()
    for spec in sorted(schema.inputs, key=lambda item: item.index):
        fixed, fixed_value = _schema_fixed_value(spec)
        preferred_variant = _preferred_typed_variant(spec)
        coupled = tuple(sorted(set(spec.coupled_indices) | {spec.index}))
        if not spec.independently_replaceable:
            if include_non_independent:
                result.append(
                    InterventionPattern(
                        pattern_id=f"input_{spec.index:03d}",
                        indices=(spec.index,),
                        method="identity",
                        description=f"{spec.name_ja}: individual replacement is not valid",
                        metadata={
                            "skip_reason": "non-independent input; evaluate its coherent coupled group",
                            "coherent_group_indices": list(coupled),
                            "input_index": spec.index,
                        },
                    )
                )
            if coupled and coupled not in emitted_groups:
                emitted_groups.add(coupled)
                group_fixed: dict[int, Any] = {}
                all_fixed = True
                for index in coupled:
                    member_fixed, member_value = _schema_fixed_value(schema.spec(index))
                    if not member_fixed:
                        all_fixed = False
                        break
                    group_fixed[index] = member_value
                if all_fixed:
                    result.append(
                        InterventionPattern(
                            pattern_id="group_" + "_".join(str(index) for index in coupled),
                            indices=coupled,
                            method="fixed",
                            values=group_fixed,
                            description="coherent coupled replacement",
                            metadata={"coherent_group": True, "input_indices": list(coupled)},
                        )
                    )
                elif reference_id:
                    result.append(
                        InterventionPattern(
                            pattern_id="group_" + "_".join(str(index) for index in coupled),
                            indices=coupled,
                            method="reference",
                            reference_id=reference_id,
                            description="coherent saved-reference group replacement",
                            metadata={
                                "coherent_group": True,
                                "input_indices": list(coupled),
                                "compatibility_keys": ["road_segment_id", "target_lane_ordinal"],
                            },
                        )
                    )
                else:
                    result.append(
                        InterventionPattern(
                            pattern_id="group_" + "_".join(str(index) for index in coupled),
                            indices=coupled,
                            method="identity",
                            description="coherent group requires a saved reference",
                            metadata={
                                "skip_reason": "no confirmed group replacement and no saved reference_id",
                                "coherent_group": True,
                                "input_indices": list(coupled),
                            },
                        )
                    )
            continue
        if preferred_variant is not None:
            operation = str(
                preferred_variant.get("operation", preferred_variant.get("method", "fixed_level"))
            ).casefold().replace("-", "_")
            result.append(
                InterventionPattern(
                    pattern_id=f"input_{spec.index:03d}",
                    indices=(spec.index,),
                    method=operation,
                    variant_id=str(preferred_variant["id"]),
                    description=f"{spec.name_ja}: source-confirmed {operation} variant",
                    metadata={"input_index": spec.index, "variant_generated": True},
                )
            )
        elif fixed:
            result.append(
                InterventionPattern(
                    pattern_id=f"input_{spec.index:03d}",
                    indices=(spec.index,),
                    method="fixed",
                    values={spec.index: fixed_value},
                    description=f"{spec.name_ja}: schema-confirmed fixed replacement",
                    metadata={"input_index": spec.index, "replacement_confirmed": True},
                )
            )
        elif reference_id:
            result.append(
                InterventionPattern(
                    pattern_id=f"input_{spec.index:03d}",
                    indices=(spec.index,),
                    method="reference",
                    reference_id=reference_id,
                    description=f"{spec.name_ja}: saved-reference replacement",
                    metadata={
                        "input_index": spec.index,
                        "compatibility_keys": ["road_segment_id", "target_lane_ordinal"],
                    },
                )
            )
        else:
            result.append(
                InterventionPattern(
                    pattern_id=f"input_{spec.index:03d}",
                    indices=(spec.index,),
                    method="identity",
                    description=f"{spec.name_ja}: no confirmed replacement configured",
                    metadata={
                        "skip_reason": "no schema-confirmed fixed replacement and no saved reference_id",
                        "input_index": spec.index,
                    },
                )
            )
    return patterns_from_config(result)


def generate_safe_patterns(*args: Any, **kwargs: Any) -> tuple[InterventionPattern, ...]:
    """Alias for :func:`generate_individual_patterns`."""

    return generate_individual_patterns(*args, **kwargs)


def dump_patterns(patterns: Iterable[InterventionPattern], path: str) -> None:
    with open(path, "w", encoding="utf-8") as stream:
        json.dump([pattern.to_dict() for pattern in patterns], stream, ensure_ascii=False, indent=2)
        stream.write("\n")
