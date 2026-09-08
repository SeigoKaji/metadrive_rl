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

    def __post_init__(self) -> None:
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int) or self.dimension <= 0:
            raise SchemaError("dimension must be a positive integer")
        object.__setattr__(self, "inputs", tuple(self.inputs))
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
