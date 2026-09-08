"""Input schemas used by the portable attribution analysis.

The analysis package deliberately does not infer the meaning of an observation
from its length.  A :class:`InputSchema` is an explicit, checked mapping from
an observation index to a stable input description.  This is useful for both
the 259 element official observation and for a port whose three target-lane
features appear at arbitrary (and potentially non-tail) indices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


class SchemaError(ValueError):
    """Raised when a schema or an observation does not satisfy its contract."""


def _json_value(value: Any) -> Any:
    """Convert common numpy values to JSON-compatible values."""

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


def _finite_number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise SchemaError(f"{location} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise SchemaError(f"{location} must be a finite number")
    return result


def _as_tuple_ints(values: Iterable[Any], location: str) -> tuple[int, ...]:
    result: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise SchemaError(f"{location} must contain integer indices")
        result.append(int(value))
    return tuple(result)


def _strict_mapping_bool(
    value: Mapping[str, Any],
    key: str,
    *,
    default: bool,
) -> bool:
    """Read a JSON boolean without applying Python's broad truthiness rules.

    ``bool("false")`` and ``bool(0)`` are both legal Python expressions but
    are dangerous when reading an experiment schema.  A schema flag is a
    wire-format value, so only the actual JSON/Python ``bool`` type is
    accepted.  In particular, numpy scalar booleans are rejected too: they
    indicate that a caller has already coerced the schema rather than loaded
    it from its portable representation.
    """

    if key not in value:
        return default
    raw = value[key]
    if type(raw) is not bool:
        raise SchemaError(f"input {key} must be a boolean (true/false), got {type(raw).__name__}")
    return raw


_UNRESOLVED_MARKERS = (
    "unresolved",
    "unknown",
    "unset",
    "not verified",
    "not_verified",
    "source verification required",
    "source_verification_required",
    "reference verification required",
    "reference_verification_required",
    "未確定",
    "未設定",
    "未確認",
    "不明",
)


_VARIANT_OPERATIONS = frozenset(
    {
        "identity",
        "neutral",
        "reflection",
        "fixed_level",
        "fixed",
        "reference",
    }
)


def _variant_operation(value: Mapping[str, Any], location: str) -> str:
    operation = value.get("operation", value.get("method", value.get("kind")))
    if not isinstance(operation, str) or not operation.strip():
        raise SchemaError(f"{location} must declare operation/method")
    operation = operation.strip().casefold().replace("-", "_")
    if operation not in _VARIANT_OPERATIONS:
        allowed = ", ".join(sorted(_VARIANT_OPERATIONS))
        raise SchemaError(f"{location}.operation={operation!r} is unsupported; use {allowed}")
    return operation


def _variant_id(value: Mapping[str, Any], location: str) -> str:
    identifier = value.get("id", value.get("variant_id", value.get("name")))
    if not isinstance(identifier, str) or not identifier.strip():
        raise SchemaError(f"{location} must declare a non-empty id")
    return identifier.strip()


def _variant_scalar_values(value: Any) -> tuple[Any, ...]:
    """Flatten scalar values carried by a variant value/center mapping."""

    if isinstance(value, Mapping):
        result: list[Any] = []
        for nested in value.values():
            result.extend(_variant_scalar_values(nested))
        return tuple(result)
    if isinstance(value, (list, tuple, np.ndarray)):
        result = []
        for nested in value:
            result.extend(_variant_scalar_values(nested))
        return tuple(result)
    return (value,)


def _validate_variant_domain(
    value: Any,
    bounds: tuple[Any, Any],
    location: str,
) -> None:
    """Require declared values and centers to lie in a variant's range.

    A variant ``range`` describes the values that this variant is allowed to
    generate.  It is narrower than or equal to the owning input's range; it is
    not a comment that can be ignored after checking only the input schema.
    """

    minimum, maximum = bounds
    for scalar in _variant_scalar_values(value):
        if scalar is None:
            continue
        number = float(scalar) if isinstance(scalar, (bool, np.bool_)) else _finite_number(scalar, location)
        if minimum is not None and number < float(minimum):
            raise SchemaError(
                f"{location}={number} is below the variant range minimum {minimum}"
            )
        if maximum is not None and number > float(maximum):
            raise SchemaError(
                f"{location}={number} exceeds the variant range maximum {maximum}"
            )


def _validate_variant_mapping(
    value: Mapping[str, Any],
    location: str,
    *,
    require_confirmation_details: bool = True,
) -> dict[str, Any]:
    """Validate a portable typed intervention variant record.

    The record remains a JSON mapping so adapters can carry it across machines.
    This check intentionally does not infer a value, center, category, or
    validity sentinel from an input's index or dimension.
    """

    if not isinstance(value, Mapping):
        raise SchemaError(f"{location} must be a mapping")
    result = {str(key): _json_value(item) for key, item in value.items()}
    identifier = _variant_id(result, location)
    operation = _variant_operation(result, location)
    result.setdefault("id", identifier)
    result.setdefault("operation", operation)
    if "confirmed" in result and type(result["confirmed"]) is not bool:
        raise SchemaError(f"{location}.confirmed must be a boolean")
    confirmed = result.get("confirmed", False)
    if confirmed and require_confirmation_details:
        # A boolean is not provenance.  Require a semantic statement and an
        # evidence/source pointer before a variant can be used by execution.
        meaning = result.get("meaning", result.get("description"))
        evidence = result.get(
            "evidence",
            result.get("source", result.get("reference", result.get("provenance"))),
        )
        if not isinstance(meaning, str) or not meaning.strip():
            raise SchemaError(f"{location} confirmed=true requires meaning/description")
        if not isinstance(evidence, str) or not evidence.strip():
            raise SchemaError(f"{location} confirmed=true requires evidence/source provenance")
        if operation in {"neutral", "fixed_level", "fixed"}:
            has_value = any(key in result for key in ("value", "values", "level", "levels"))
            if not has_value:
                raise SchemaError(f"{location} {operation} variant requires value/level")
        elif operation == "reflection" and not any(key in result for key in ("center", "centers")):
            raise SchemaError(f"{location} reflection variant requires center")
    for key in ("range", "value_range"):
        if key in result:
            raw_range = result[key]
            if isinstance(raw_range, (str, bytes)):
                raise SchemaError(f"{location}.{key} must be a two-element sequence")
            try:
                pair = tuple(raw_range)
            except TypeError as exc:
                raise SchemaError(f"{location}.{key} must be a two-element sequence") from exc
            if len(pair) != 2:
                raise SchemaError(f"{location}.{key} must be a two-element sequence")
            for bound in pair:
                if bound is not None:
                    _finite_number(bound, f"{location}.{key}")
            if pair[0] is not None and pair[1] is not None and float(pair[0]) > float(pair[1]):
                raise SchemaError(f"{location}.{key} minimum cannot exceed maximum")
            result["range"] = list(pair)
            break
    if "range" in result:
        bounds = (result["range"][0], result["range"][1])
        for key in ("value", "values", "level", "levels", "center", "centers"):
            if key in result and result[key] is not None:
                _validate_variant_domain(result[key], bounds, f"{location}.{key}")
    for key in ("coupled_indices", "related_valid_flags"):
        if key in result:
            raw = result[key]
            if isinstance(raw, (str, bytes)):
                raise SchemaError(f"{location}.{key} must be an array of indices")
            result[key] = list(_as_tuple_ints(raw, f"{location}.{key}"))
    for key in ("allowed_indices",):
        if key in result:
            raw = result[key]
            if isinstance(raw, (str, bytes)):
                raise SchemaError(f"{location}.{key} must be an array of indices")
            result[key] = list(_as_tuple_ints(raw, f"{location}.{key}"))
    for key in ("allowed_ids", "allowed_groups", "semantic_ids"):
        if key in result:
            raw = result[key]
            if isinstance(raw, (str, bytes)):
                raise SchemaError(f"{location}.{key} must be an array of strings")
            if not isinstance(raw, Sequence) or any(not isinstance(item, str) or not item.strip() for item in raw):
                raise SchemaError(f"{location}.{key} must be an array of non-empty strings")
            result[key] = [item.strip() for item in raw]
    return result


def _unresolved_paths(value: Any, path: str = "schema") -> list[str]:
    """Find unresolved provenance markers at any nesting depth.

    Templates often put a valid numeric index next to an unresolved source or
    replacement mapping.  Looking only at the top-level input id therefore
    lets a schema pass while its actual meaning is still unknown.  Traverse
    JSON-like values recursively and retain paths for a useful error message.
    Mapping keys are checked as well because a nested key such as
    ``unknown_reference`` is itself a common unresolved marker.
    """

    found: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            key_lower = key_text.casefold()
            if any(marker in key_lower for marker in _UNRESOLVED_MARKERS):
                found.append(f"{path}.{key_text}")
            found.extend(_unresolved_paths(nested, f"{path}.{key_text}"))
        return found
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            found.extend(_unresolved_paths(nested, f"{path}[{index}]"))
        return found
    if isinstance(value, str):
        lowered = value.casefold()
        if any(marker in lowered for marker in _UNRESOLVED_MARKERS):
            found.append(path)
    return found


@dataclass(frozen=True, slots=True)
class InputSpec:
    """Description of one model input dimension.

    ``replacement`` is intentionally a JSON-compatible mapping rather than a
    callable.  The schema is portable across machines; adapters resolve any
    environment-specific replacement values before an experiment is run.
    Common keys are ``kind`` (``fixed`` or ``reference``), ``value`` and
    ``value_method``.
    """

    index: int
    id: str
    name_ja: str
    description: str = ""
    group: str = ""
    physical_quantity: str = ""
    unit: str = ""
    model_representation: str = ""
    value_range: tuple[float | None, float | None] | None = None
    normalization: str = ""
    clip_method: str = ""
    value_type: str = "continuous"
    reference: str = ""
    replacement: Mapping[str, Any] | None = None
    related_valid_flags: tuple[int, ...] = ()
    coupled_indices: tuple[int, ...] = ()
    independently_replaceable: bool = True
    ig_interpolation_allowed: bool = True
    source: str = ""
    # New typed intervention variants.  Mappings are retained verbatim (after
    # JSON-safe normalization) so portable bundles do not depend on Python
    # classes or adapter code.
    variants: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int):
            raise SchemaError("input index must be an integer")
        if self.index < 0:
            raise SchemaError("input index must be non-negative")
        for field_name in ("id", "name_ja", "group", "value_type"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise SchemaError(f"{field_name} must be a non-empty string")
        value_type = self.value_type.lower()
        if value_type not in {"continuous", "bool", "boolean", "categorical", "category"}:
            raise SchemaError(
                "value_type must be continuous, bool, or categorical"
            )
        if self.value_range is not None:
            if len(self.value_range) != 2:
                raise SchemaError("value_range must contain (minimum, maximum)")
            minimum, maximum = self.value_range
            if minimum is not None:
                minimum = _finite_number(minimum, f"input {self.index} minimum")
            if maximum is not None:
                maximum = _finite_number(maximum, f"input {self.index} maximum")
            if minimum is not None and maximum is not None and minimum > maximum:
                raise SchemaError("value_range minimum cannot exceed maximum")
            object.__setattr__(self, "value_range", (minimum, maximum))
        object.__setattr__(self, "value_type", value_type)
        object.__setattr__(
            self,
            "related_valid_flags",
            _as_tuple_ints(self.related_valid_flags, "related_valid_flags"),
        )
        object.__setattr__(
            self,
            "coupled_indices",
            _as_tuple_ints(self.coupled_indices, "coupled_indices"),
        )
        if self.replacement is not None and not isinstance(self.replacement, Mapping):
            raise SchemaError("replacement must be a JSON-compatible mapping")
        if isinstance(self.variants, (str, bytes)):
            raise SchemaError("variants must be an array of mappings")
        normalized_variants: list[Mapping[str, Any]] = []
        seen_variant_ids: set[str] = set()
        for variant_index, variant in enumerate(self.variants):
            normalized = _validate_variant_mapping(
                variant,
                f"input {self.index}.variants[{variant_index}]",
            )
            identifier = str(normalized["id"])
            if identifier in seen_variant_ids:
                raise SchemaError(f"input {self.index} has duplicate variant id {identifier!r}")
            seen_variant_ids.add(identifier)
            normalized_variants.append(normalized)
        object.__setattr__(self, "variants", tuple(normalized_variants))
        if not isinstance(self.independently_replaceable, bool):
            raise SchemaError("independently_replaceable must be boolean")
        if not isinstance(self.ig_interpolation_allowed, bool):
            raise SchemaError("ig_interpolation_allowed must be boolean")

    @property
    def is_discrete(self) -> bool:
        return self.value_type in {"bool", "boolean", "categorical", "category"}

    @property
    def stable_id(self) -> str:
        """Compatibility alias for the schema's stable English id."""

        return self.id

    @property
    def interpolation_allowed(self) -> bool:
        return bool(self.ig_interpolation_allowed) and not self.is_discrete

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "id": self.id,
            "name_ja": self.name_ja,
            "description": self.description,
            "group": self.group,
            "physical_quantity": self.physical_quantity,
            "unit": self.unit,
            "model_representation": self.model_representation,
            "range": list(self.value_range) if self.value_range is not None else None,
            "normalization": self.normalization,
            "clip_method": self.clip_method,
            "value_type": self.value_type,
            "reference": self.reference,
            "replacement": _json_value(self.replacement),
            "related_valid_flags": list(self.related_valid_flags),
            "coupled_indices": list(self.coupled_indices),
            "independently_replaceable": self.independently_replaceable,
            "ig_interpolation_allowed": self.ig_interpolation_allowed,
            "source": self.source,
            "variants": [_json_value(item) for item in self.variants],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InputSpec":
        if not isinstance(value, Mapping):
            raise SchemaError("each schema input must be a mapping")
        raw_range = value.get("range", value.get("value_range"))
        if raw_range is not None:
            if isinstance(raw_range, (str, bytes)):
                raise SchemaError("input range must be a two-element sequence")
            try:
                value_range = tuple(raw_range)
            except TypeError as exc:
                raise SchemaError("input range must be a two-element sequence") from exc
        else:
            value_range = None
        raw_index = value.get("index")
        if isinstance(raw_index, bool) or not isinstance(raw_index, (int, np.integer)):
            raise SchemaError("input index must be an integer")
        raw_id = value.get("id", value.get("stable_id"))
        raw_name = value.get("name_ja", value.get("name"))
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise SchemaError("input id must be a non-empty string")
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise SchemaError("input name_ja must be a non-empty string")
        raw_group = value.get("group")
        if not isinstance(raw_group, str) or not raw_group.strip():
            raise SchemaError("input group must be a non-empty string")
        raw_type = value.get("value_type", "continuous")
        if not isinstance(raw_type, str):
            raise SchemaError("input value_type must be a string")
        return cls(
            index=int(raw_index),
            id=raw_id,
            name_ja=raw_name,
            description=str(value.get("description", "")),
            group=raw_group,
            physical_quantity=str(value.get("physical_quantity", "")),
            unit=str(value.get("unit", "")),
            model_representation=str(value.get("model_representation", "")),
            value_range=value_range,
            normalization=str(value.get("normalization", "")),
            clip_method=str(value.get("clip_method", "")),
            value_type=raw_type,
            reference=str(value.get("reference", "")),
            replacement=value.get("replacement"),
            related_valid_flags=tuple(value.get("related_valid_flags", ())),
            coupled_indices=tuple(value.get("coupled_indices", ())),
            independently_replaceable=_strict_mapping_bool(
                value, "independently_replaceable", default=True
            ),
            ig_interpolation_allowed=_strict_mapping_bool(
                value, "ig_interpolation_allowed", default=True
            ),
            source=str(value.get("source", "")),
            variants=tuple(
                value.get("variants", value.get("intervention_variants", ())) or ()
            ),
        )

    def validate_value(self, value: Any, *, location: str | None = None) -> None:
        """Validate a replacement or observation value without clipping it."""

        where = location or f"input {self.index}"
        if self.is_discrete:
            if self.value_type in {"bool", "boolean"}:
                if isinstance(value, (bool, np.bool_)):
                    numeric = int(value)
                elif isinstance(value, (int, np.integer, float, np.floating)) and float(value) in (0.0, 1.0):
                    numeric = int(value)
                else:
                    raise SchemaError(f"{where} expects a boolean or 0/1 value")
                del numeric
            elif isinstance(value, (bool, np.bool_)):
                raise SchemaError(f"{where} expects a categorical value")
            elif not isinstance(value, (int, np.integer, float, np.floating, str)):
                raise SchemaError(f"{where} expects a scalar categorical value")
            elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
                raise SchemaError(f"{where} must be a finite categorical value")
        else:
            _finite_number(value, where)
        if self.value_range is not None and isinstance(value, (int, float, np.integer, np.floating)):
            number = float(value)
            minimum, maximum = self.value_range
            if minimum is not None and number < minimum:
                raise SchemaError(f"{where}={number} is below the declared minimum {minimum}")
            if maximum is not None and number > maximum:
                raise SchemaError(f"{where}={number} exceeds the declared maximum {maximum}")


@dataclass(frozen=True, slots=True)
class InputSchema:
    """Complete explicit schema for a one-dimensional vector observation."""

    dimension: int
    inputs: tuple[InputSpec, ...]
    schema_id: str = ""
    version: str = "1"
    preprocessing: Mapping[str, Any] = field(default_factory=dict)
    # Variants that apply to a declared group of inputs (for example the
    # source-verified LiDAR no-detection default) can live once at schema
    # level.  Per-input variants remain on ``InputSpec.variants``.
    intervention_variants: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int) or self.dimension <= 0:
            raise SchemaError("dimension must be a positive integer")
        object.__setattr__(self, "inputs", tuple(self.inputs))
        if not isinstance(self.intervention_variants, Mapping):
            raise SchemaError("intervention_variants must be a mapping")
        normalized_variants: dict[str, Mapping[str, Any]] = {}
        for key, value in self.intervention_variants.items():
            if not isinstance(key, str) or not key.strip():
                raise SchemaError("intervention_variants keys must be non-empty strings")
            normalized = _validate_variant_mapping(
                value,
                f"intervention_variants[{key!r}]",
            )
            declared_id = str(normalized["id"])
            if declared_id != key:
                raise SchemaError(
                    f"intervention_variants key {key!r} does not match variant id {declared_id!r}"
                )
            normalized_variants[key] = normalized
        object.__setattr__(self, "intervention_variants", normalized_variants)
        self.validate()

    @property
    def by_index(self) -> dict[int, InputSpec]:
        return {item.index: item for item in self.inputs}

    @property
    def entries(self) -> tuple[InputSpec, ...]:
        """Compatibility alias for callers that call schema rows ``entries``."""

        return self.inputs

    @property
    def n_features(self) -> int:
        return self.dimension

    @property
    def by_id(self) -> dict[str, InputSpec]:
        return {item.id: item for item in self.inputs}

    def variant(self, index_or_id: int | str, variant_id: str) -> Mapping[str, Any]:
        """Resolve an input-local or schema-level typed variant.

        Input-local records take precedence, which permits the same variant id
        to have input-specific centers or values.  A schema-level record is
        useful for a group whose semantics and replacement value are shared.
        """

        spec = self.spec(index_or_id)
        for value in spec.variants:
            if value.get("id") == variant_id:
                return value
        try:
            return self.intervention_variants[variant_id]
        except KeyError as exc:
            raise SchemaError(
                f"unknown intervention variant {variant_id!r} for input {spec.id!r}"
            ) from exc

    def validate(self, *, require_provenance: bool = False) -> None:
        if len(self.inputs) != self.dimension:
            raise SchemaError(
                f"schema has {len(self.inputs)} entries but dimension is {self.dimension}"
            )
        indices = [item.index for item in self.inputs]
        if len(set(indices)) != len(indices):
            raise SchemaError("schema input indices must be unique")
        expected = set(range(self.dimension))
        if set(indices) != expected:
            missing = sorted(expected - set(indices))
            extra = sorted(set(indices) - expected)
            raise SchemaError(f"schema must cover every index 0..{self.dimension - 1}; missing={missing}, extra={extra}")
        ids = [item.id for item in self.inputs]
        if len(set(ids)) != len(ids):
            raise SchemaError("schema input ids must be unique")
        for item in self.inputs:
            for related in (*item.related_valid_flags, *item.coupled_indices):
                if related not in expected:
                    raise SchemaError(f"input {item.index} references unknown index {related}")
            for variant_index, variant in enumerate(item.variants):
                self._validate_variant_contract(
                    variant,
                    f"input {item.index}.variants[{variant_index}]",
                    item=item,
                    expected=expected,
                )
        for identifier, variant in self.intervention_variants.items():
            self._validate_variant_contract(
                variant,
                f"intervention_variants[{identifier!r}]",
                item=None,
                expected=expected,
            )
            if (
                variant.get("confirmed") is True
                and _variant_operation(variant, f"intervention_variants[{identifier!r}]")
                == "reflection"
            ):
                allowed_indices = set(variant.get("allowed_indices", ()))
                allowed_ids = set(variant.get("allowed_ids", variant.get("semantic_ids", ())))
                allowed_groups = set(variant.get("allowed_groups", ()))
                candidates = [
                    item
                    for item in self.inputs
                    if (
                        not allowed_indices
                        and not allowed_ids
                        and not allowed_groups
                    )
                    or item.index in allowed_indices
                    or item.id in allowed_ids
                    or item.group in allowed_groups
                ]
                if any(item.is_discrete for item in candidates):
                    raise SchemaError(
                        f"intervention_variants[{identifier!r}] reflection cannot target discrete inputs"
                    )
        for item in self.inputs:
            if item.index in item.coupled_indices:
                continue
        if require_provenance:
            unresolved = _unresolved_paths(self.to_dict())
            if unresolved:
                preview = ", ".join(unresolved[:8])
                suffix = "..." if len(unresolved) > 8 else ""
                raise SchemaError(
                    "schema contains unresolved provenance metadata at "
                    f"{preview}{suffix}"
                )
            for item in self.inputs:
                fields = {
                    "source": item.source,
                    "description": item.description,
                    "physical_quantity": item.physical_quantity,
                    "model_representation": item.model_representation,
                    "normalization": item.normalization,
                    "reference": item.reference,
                }
                missing = [name for name, value in fields.items() if not str(value).strip()]
                if missing:
                    detail = []
                    if missing:
                        detail.append(f"missing={missing}")
                    raise SchemaError(
                        f"input {item.index} ({item.id}) has incomplete provenance: {', '.join(detail)}"
                    )
                replacement = item.replacement
                if isinstance(replacement, Mapping):
                    kind = replacement.get("kind", replacement.get("method", ""))
                    if str(kind).casefold() in {"unresolved", "unknown", "unset"}:
                        raise SchemaError(
                            f"input {item.index} ({item.id}) has unresolved replacement metadata"
                        )

    @staticmethod
    def _validate_variant_contract(
        variant: Mapping[str, Any],
        location: str,
        *,
        item: InputSpec | None,
        expected: set[int],
    ) -> None:
        """Validate variant range/type/category/coupling declarations."""

        operation = _variant_operation(variant, location)
        for key in ("coupled_indices", "related_valid_flags"):
            for related in variant.get(key, ()):
                if related not in expected:
                    raise SchemaError(f"{location}.{key} references unknown index {related}")
        for related in variant.get("allowed_indices", ()):
            if related not in expected:
                raise SchemaError(f"{location}.allowed_indices references unknown index {related}")
        if item is not None:
            declared_type = variant.get("value_type", variant.get("dtype"))
            if declared_type is not None:
                if not isinstance(declared_type, str):
                    raise SchemaError(f"{location}.value_type must be a string")
                normalized_type = declared_type.casefold()
                if normalized_type in {"boolean", "bool"}:
                    normalized_type = "bool"
                if normalized_type in {"category", "categorical"}:
                    normalized_type = "categorical"
                if normalized_type != item.value_type:
                    item_type = "categorical" if item.value_type in {"category", "categorical"} else item.value_type
                    raise SchemaError(
                        f"{location}.value_type={declared_type!r} conflicts with input {item.index} type {item_type!r}"
                    )
            variant_range = variant.get("range", variant.get("value_range"))
            if variant_range is not None:
                pair = tuple(variant_range)
                minimum, maximum = pair
                if item.value_range is not None:
                    item_min, item_max = item.value_range
                    if item_min is not None and minimum is not None and float(minimum) < float(item_min):
                        raise SchemaError(f"{location} range extends below input {item.index} range")
                    if item_max is not None and maximum is not None and float(maximum) > float(item_max):
                        raise SchemaError(f"{location} range extends above input {item.index} range")
            categories = variant.get("categories", variant.get("category_values"))
            if categories is not None:
                if not isinstance(categories, Sequence) or isinstance(categories, (str, bytes)):
                    raise SchemaError(f"{location}.categories must be an array")
                if not item.is_discrete:
                    raise SchemaError(f"{location}.categories requires a discrete input")
        confirmed = variant.get("confirmed", False)
        if confirmed is True and operation == "reflection" and item is not None and item.is_discrete:
            raise SchemaError(f"{location} reflection is not valid for discrete input {item.index}")
        if confirmed is True and operation == "reflection" and item is None:
            # A schema-level variant may target a group, so the concrete
            # InputSpec is checked again at execution.  Reject an explicitly
            # discrete group declaration here instead of allowing it to pass
            # solely because this validation has no ``item`` argument.
            declared_type = variant.get("value_type", variant.get("dtype"))
            if isinstance(declared_type, str) and declared_type.casefold() in {
                "bool",
                "boolean",
                "category",
                "categorical",
            }:
                raise SchemaError(
                    f"{location} reflection requires a continuous group variant"
                )

    def validate_for_execution(self) -> None:
        """Fail closed on unresolved source/order metadata before an experiment."""

        self.validate(require_provenance=True)

    def spec(self, index_or_id: int | str) -> InputSpec:
        if isinstance(index_or_id, str):
            try:
                return self.by_id[index_or_id]
            except KeyError:
                # JSON object keys are strings; accept a decimal index as a
                # convenience while keeping non-numeric strings semantic IDs.
                if index_or_id.strip().lstrip("+").isdigit():
                    try:
                        return self.by_index[int(index_or_id)]
                    except KeyError:
                        pass
                raise SchemaError(f"unknown input id or index: {index_or_id}")
        try:
            return self.by_index[int(index_or_id)]
        except (KeyError, ValueError, TypeError) as exc:
            raise SchemaError(f"unknown input index: {index_or_id}") from exc

    def resolve_indices(self, values: Iterable[int | str]) -> tuple[int, ...]:
        result = tuple(self.spec(value).index for value in values)
        if len(set(result)) != len(result):
            raise SchemaError("intervention indices must be unique")
        return result

    def validate_observation(
        self,
        observation: Any,
        *,
        check_range: bool = True,
        copy: bool = True,
    ) -> np.ndarray:
        array = np.asarray(observation)
        if array.ndim != 1:
            raise SchemaError(f"observation must be one-dimensional, got shape {array.shape}")
        if array.shape[0] != self.dimension:
            raise SchemaError(
                f"observation dimension {array.shape[0]} does not match schema dimension {self.dimension}"
            )
        if not np.issubdtype(array.dtype, np.number) and array.dtype != np.bool_:
            raise SchemaError(f"observation dtype {array.dtype} is not numeric")
        if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
            raise SchemaError("observation contains non-finite values")
        if check_range:
            for item in self.inputs:
                item.validate_value(array[item.index], location=f"observation[{item.index}]")
        return np.array(array, copy=copy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "version": self.version,
            "dimension": self.dimension,
            "preprocessing": _json_value(self.preprocessing),
            "intervention_variants": _json_value(self.intervention_variants),
            "inputs": [item.to_dict() for item in sorted(self.inputs, key=lambda x: x.index)],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InputSchema":
        if not isinstance(value, Mapping):
            raise SchemaError("schema must be a mapping")
        raw_inputs = value.get("inputs", value.get("entries"))
        if not isinstance(raw_inputs, Sequence) or isinstance(raw_inputs, (str, bytes)):
            raise SchemaError("schema.inputs must be a list")
        entries = tuple(InputSpec.from_dict(item) for item in raw_inputs)
        dimension = value.get("dimension", len(entries))
        if isinstance(dimension, bool) or not isinstance(dimension, (int, np.integer)):
            raise SchemaError("schema dimension must be an integer")
        preprocessing = value.get("preprocessing", {})
        if not isinstance(preprocessing, Mapping):
            raise SchemaError("schema preprocessing must be a mapping")
        schema_id = value.get("schema_id", value.get("id", ""))
        version = value.get("version", "1")
        if not isinstance(schema_id, str) or not isinstance(version, str):
            raise SchemaError("schema_id and version must be strings")
        return cls(
            dimension=int(dimension),
            inputs=entries,
            schema_id=schema_id,
            version=version,
            preprocessing=preprocessing,
            intervention_variants=value.get(
                "intervention_variants", value.get("variants", {})
            ) or {},
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "InputSchema":
        with Path(path).open("r", encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))

    def to_json(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, ensure_ascii=False, indent=2)
            stream.write("\n")


def schema_from_dict(value: Mapping[str, Any]) -> InputSchema:
    """Functional alias useful to adapters and configuration loaders."""

    return InputSchema.from_dict(value)


def load_schema(path: str | Path) -> InputSchema:
    return InputSchema.from_json(path)
