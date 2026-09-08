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

    def __post_init__(self) -> None:
        if not isinstance(self.pattern_id, str) or not self.pattern_id.strip():
            raise InterventionError("pattern_id must be a non-empty string")
        method = self.method.lower().replace("_", "-")
        if method in {"noop", "none", "unchanged", "no-change"}:
            method = "identity"
        if method not in {"fixed", "reference", "identity"}:
            raise InterventionError("method must be fixed, reference, or identity")
        object.__setattr__(self, "method", method)
        if isinstance(self.indices, (str, bytes)):
            raise InterventionError("indices must be a sequence of indices")
        object.__setattr__(self, "indices", tuple(self.indices))
        if len(set(self.indices)) != len(self.indices):
            raise InterventionError("pattern indices must be unique")
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
        method = value.get("method", value.get("kind", "fixed"))
        values = value.get(
            "values",
            value.get("replacement_values", value.get("replacement")),
        )
        metadata = value.get("metadata", {})
        return cls(
            pattern_id=str(value.get("pattern_id", value.get("id", ""))),
            indices=tuple(indices),
            method=str(method),
            values=values,
            reference_id=value.get("reference_id"),
            description=str(value.get("description", "")),
            replacement_kind=str(value.get("replacement_kind", "diagnostic")),
            metadata=metadata,
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

    @property
    def changed(self) -> bool:
        return bool(self.changed_indices)

    @property
    def applied_count(self) -> int:
        return 0 if self.skipped else len(self.requested_indices)

    @property
    def actual_change_count(self) -> int:
        return len(self.changed_indices)

    @property
    def no_op_count(self) -> int:
        return len(self.no_op_indices)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "method": self.method,
            "reference_id": self.reference_id,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "requested_indices": list(self.requested_indices),
            "changed_indices": list(self.changed_indices),
            "no_op_indices": list(self.no_op_indices),
            "changed": self.changed,
            "applied_count": self.applied_count,
            "actual_change_count": self.actual_change_count,
            "no_op_count": self.no_op_count,
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
        if pattern.metadata.get("invalid_joint_confirmed") is not True:
            return "invalid replacement requires metadata.invalid_joint_confirmed=true"
        if "invalid_value" not in declared:
            return (
                f"input {index} has no source-confirmed invalid_value for the "
                "invalid coupled replacement"
            )
        if declared.get("confirmed") is not True:
            return f"input {index} invalid_value is not source-confirmed"
        invalid_value = declared["invalid_value"]
        if not _values_equal(invalid_value, value):
            return (
                f"input {index} differs from schema-confirmed invalid_value "
                f"{invalid_value!r}"
            )
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
) -> InterventionResult:
    return InterventionResult(
        observation=source,
        requested_indices=indices,
        changed_indices=(),
        no_op_indices=indices,
        pattern_id=pattern.pattern_id,
        method=pattern.method,
        reference_id=pattern.reference_id,
        skipped=True,
        skip_reason=reason,
    )


def _invalid_joint_reason(
    pattern: InterventionPattern,
    replacement_values: Mapping[int, Any],
    indices: tuple[int, ...],
    schema: InputSchema,
) -> str | None:
    """Require explicit confirmation before creating a flag-invalid joint state."""

    for index in indices:
        spec = schema.spec(index)
        if spec.value_type not in {"bool", "boolean"}:
            continue
        value = replacement_values.get(index)
        is_invalid = _flag_is_invalid(value)
        related_values = _related_value_indices(schema, index)
        if not is_invalid or not related_values:
            continue
        missing_values = sorted(related_values - set(indices))
        if missing_values:
            return (
                f"invalid flag/value group at input {index} is missing related "
                f"values {missing_values}"
            )
        invalid_joint_confirmed = pattern.metadata.get("invalid_joint_confirmed") is True
        if invalid_joint_confirmed:
            group = related_values | {index}
            declared_members = [schema.spec(member).replacement for member in group]
            # A schema may omit replacement metadata entirely and let an
            # explicitly confirmed diagnostic pattern provide its values.  If
            # it declares even one neutral replacement, however, every member
            # must also declare a source-confirmed invalid variant; otherwise
            # the neutral sentinel could be mistaken for the invalid state.
            if any(isinstance(declared, Mapping) for declared in declared_members):
                for member in sorted(group):
                    declared = schema.spec(member).replacement
                    if not isinstance(declared, Mapping) or "invalid_value" not in declared:
                        return (
                            f"invalid flag/value group at input {index} lacks a "
                            f"source-confirmed invalid_value for input {member}"
                        )
                    if declared.get("confirmed") is not True:
                        return (
                            f"invalid flag/value group at input {index} has an "
                            f"unconfirmed invalid_value at input {member}"
                        )
                    expected = declared["invalid_value"]
                    if not _values_equal(replacement_values.get(member), expected):
                        return (
                            f"invalid flag/value group at input {index} has an "
                            f"invalid sentinel mismatch at input {member}: "
                            f"expected {expected!r}"
                        )
            continue
        all_declared_confirmed = True
        for coupled_index in related_values | {index}:
            declared = schema.spec(coupled_index).replacement
            if not isinstance(declared, Mapping) or declared.get("confirmed") is not True:
                all_declared_confirmed = False
                break
        if not all_declared_confirmed:
            return (
                f"invalid flag/value group at input {index} requires "
                "metadata.invalid_joint_confirmed=true or source-confirmed group replacements"
            )
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
        return _skipped_result(source, pattern, indices, str(configured_skip))
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
        )

    if pattern.method == "reference":
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

    invalid_joint_reason = _invalid_joint_reason(pattern, replacement_values, indices, schema)
    if invalid_joint_reason is not None:
        if strict:
            raise InterventionError(invalid_joint_reason)
        return _skipped_result(source, pattern, indices, invalid_joint_reason)

    invalid_variant_indices: set[int] = set()
    if pattern.metadata.get("invalid_joint_confirmed") is True:
        for index in indices:
            spec = schema.spec(index)
            value = replacement_values.get(index)
            is_invalid_flag = _is_bool_spec(spec) and _flag_is_invalid(value)
            if not is_invalid_flag:
                continue
            group = _related_value_indices(schema, index) | {index}
            if any(
                isinstance(schema.spec(member).replacement, Mapping)
                and "invalid_value" in schema.spec(member).replacement
                for member in group
            ):
                invalid_variant_indices.update(group)

    result = np.array(source, copy=True)
    changed: list[int] = []
    no_op: list[int] = []
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
        try:
            spec.validate_value(value, location=f"pattern {pattern.pattern_id}[{index}]")
        except SchemaError as exc:
            raise InterventionError(str(exc)) from exc
        # Assigning a float replacement into an integer source array can lose
        # information silently, so reject it unless numpy can represent the
        # value exactly.  Standard SB3 vectors are float arrays.
        old = result[index]
        cast_value = np.asarray(value, dtype=result.dtype).item()
        if strict and isinstance(value, (float, np.floating)) and np.issubdtype(result.dtype, np.integer):
            if float(cast_value) != float(value):
                raise InterventionError(
                    f"replacement at index {index} cannot be represented by observation dtype {result.dtype}"
                )
        result[index] = cast_value
        if bool(np.asarray(old != result[index]).item()):
            changed.append(index)
        else:
            no_op.append(index)
    return InterventionResult(
        observation=result,
        requested_indices=indices,
        changed_indices=tuple(changed),
        no_op_indices=tuple(no_op),
        pattern_id=pattern.pattern_id,
        method=pattern.method,
        reference_id=pattern.reference_id,
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
        if fixed:
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
