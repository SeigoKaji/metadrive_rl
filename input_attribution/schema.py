"""Portable, strict observation-schema expansion for vector observations.

Schema files describe semantics outside Python so a model with a different
observation dimension can be analysed without editing the attribution code.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import io
import math
from pathlib import Path
from types import MappingProxyType
import tomllib
from typing import Final


class ObservationSchemaError(ValueError):
    """Raised when an observation schema is incomplete or inconsistent."""


# A shorter compatibility name is convenient for callers and error handling.
SchemaError = ObservationSchemaError


@dataclass(frozen=True, slots=True)
class Feature:
    """One semantically named scalar in a flat observation vector."""

    index: int
    name: str
    block: str
    kind: str = "scalar"
    angle_deg: float | None = None
    groups: tuple[str, ...] = ()
    description: str | None = None
    constant: float | None = None
    resolved: bool = True

    @property
    def group(self) -> str | None:
        """Return the first group for simple consumers, if one exists."""

        return self.groups[0] if self.groups else None


@dataclass(frozen=True, slots=True)
class ObservationSchema:
    """Expanded schema whose indices exactly cover one observation vector."""

    schema_version: int
    name: str
    observation_dim: int
    features: tuple[Feature, ...]
    groups: Mapping[str, tuple[int, ...]]
    group_descriptions: Mapping[str, str | None] = MappingProxyType({})
    source_path: Path | None = None

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ObservationSchemaError("schema_version: 現在対応しているversionは1です")
        if not self.name.strip():
            raise ObservationSchemaError("name: 空でない文字列で指定してください")
        if self.observation_dim <= 0:
            raise ObservationSchemaError("observation_dim: 1以上で指定してください")
        features = tuple(sorted(self.features, key=lambda feature: feature.index))
        if len(features) != self.observation_dim:
            raise ObservationSchemaError(
                "features: observation_dimと同じ数のfeatureが必要です "
                f"({self.observation_dim} expected, {len(features)} found)"
            )
        indices = [feature.index for feature in features]
        expected = list(range(self.observation_dim))
        if indices != expected:
            duplicates = sorted({index for index in indices if indices.count(index) > 1})
            missing = sorted(set(expected).difference(indices))
            outside = sorted(set(indices).difference(expected))
            details: list[str] = []
            if duplicates:
                details.append(f"重複index={duplicates}")
            if missing:
                details.append(f"欠落index={missing}")
            if outside:
                details.append(f"範囲外index={outside}")
            detail = "; ".join(details) if details else "index集合が不正です"
            raise ObservationSchemaError(f"features: 0..D-1を一度ずつ定義してください ({detail})")
        names = [feature.name for feature in features]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ObservationSchemaError(f"features: feature名が重複しています: {', '.join(duplicates)}")
        for feature in features:
            if not feature.name.strip() or not feature.block.strip() or not feature.kind.strip():
                raise ObservationSchemaError("features: name/block/kindは空にできません")
            if feature.angle_deg is not None and not math.isfinite(feature.angle_deg):
                raise ObservationSchemaError(f"features[{feature.index}]: angle_degは有限値で指定してください")
            if feature.constant is not None and not math.isfinite(feature.constant):
                raise ObservationSchemaError(f"features[{feature.index}]: constantは有限値で指定してください")
        normalized_groups: dict[str, tuple[int, ...]] = {}
        for group_name, raw_indices in self.groups.items():
            if not isinstance(group_name, str) or not group_name.strip():
                raise ObservationSchemaError("groups: 空でないgroup名で指定してください")
            group_indices = tuple(dict.fromkeys(raw_indices))
            if not group_indices:
                raise ObservationSchemaError(f"groups.{group_name}: 少なくとも1つのmemberが必要です")
            invalid = sorted(set(group_indices).difference(expected))
            if invalid:
                raise ObservationSchemaError(
                    f"groups.{group_name}: 範囲外indexがあります: {invalid}"
                )
            normalized_groups[group_name] = tuple(sorted(group_indices))
        descriptions = dict(self.group_descriptions)
        unknown_descriptions = sorted(set(descriptions).difference(normalized_groups))
        if unknown_descriptions:
            raise ObservationSchemaError(
                "group_descriptions: 未定義groupがあります: "
                + ", ".join(unknown_descriptions)
            )
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "groups", MappingProxyType(normalized_groups))
        object.__setattr__(self, "group_descriptions", MappingProxyType(descriptions))

    @property
    def unresolved_features(self) -> tuple[Feature, ...]:
        """Features still marked as placeholders in a template schema."""

        return tuple(feature for feature in self.features if not feature.resolved)

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Feature names in vector index order."""

        return tuple(feature.name for feature in self.features)

    @property
    def lidar_features(self) -> tuple[Feature, ...]:
        """Features labelled as LiDAR by the external schema."""

        return tuple(feature for feature in self.features if feature.kind == "lidar")

    def validate_dimension(self, dimension: int) -> None:
        """Fail with both dimensions when a runtime vector does not match."""

        if dimension != self.observation_dim:
            raise ObservationSchemaError(
                "observation dimension mismatch: "
                f"schema={self.observation_dim}, runtime={dimension}"
            )

    def feature_at(self, index: int) -> Feature:
        """Return the feature at ``index`` with range validation."""

        if isinstance(index, bool) or not isinstance(index, int):
            raise ObservationSchemaError("feature indexは整数で指定してください")
        if index < 0 or index >= self.observation_dim:
            raise ObservationSchemaError(
                f"feature indexが範囲外です: {index} (0..{self.observation_dim - 1})"
            )
        return self.features[index]

    def feature_named(self, name: str) -> Feature:
        """Find one feature by its globally unique external name."""

        for feature in self.features:
            if feature.name == name:
                return feature
        raise ObservationSchemaError(f"未定義featureです: {name}")

    # A pair of readable aliases avoids callers depending on implementation
    # details while retaining the concise names often used in notebooks.
    get_feature = feature_at
    get_feature_by_name = feature_named

    def indices_for_group(self, name: str) -> tuple[int, ...]:
        """Return sorted feature indices belonging to a named semantic group."""

        try:
            return self.groups[name]
        except KeyError as error:
            raise ObservationSchemaError(f"未定義groupです: {name}") from error

    def lidar_sector_groups(self, sector_degrees: float) -> dict[str, tuple[int, ...]]:
        """Build non-empty angular LiDAR groups from angle metadata.

        Angles are interpreted in the direction declared by the schema.  The
        group labels only identify their angular intervals; they do not claim
        road-relative semantics such as "front" or "left".
        """

        if isinstance(sector_degrees, bool) or not isinstance(sector_degrees, (int, float)):
            raise ObservationSchemaError("lidar sector幅は有限の数値で指定してください")
        width = float(sector_degrees)
        if not math.isfinite(width) or not 0.0 < width <= 360.0:
            raise ObservationSchemaError("lidar sector幅は0より大きく360以下で指定してください")
        lidar = self.lidar_features
        if not lidar:
            return {}
        missing_angle = [feature.name for feature in lidar if feature.angle_deg is None]
        if missing_angle:
            raise ObservationSchemaError(
                "LiDAR sectorを作るにはangle metadataが必要です: "
                + ", ".join(missing_angle[:5])
            )
        grouped: dict[int, list[int]] = {}
        for feature in lidar:
            assert feature.angle_deg is not None
            sector = min(int((feature.angle_deg % 360.0) / width), int(math.ceil(360.0 / width)) - 1)
            grouped.setdefault(sector, []).append(feature.index)
        result: dict[str, tuple[int, ...]] = {}
        for sector, indices in sorted(grouped.items()):
            start = sector * width
            end = min((sector + 1) * width, 360.0)
            result[_sector_name(start, end)] = tuple(sorted(indices))
        return result

    # Explicit verb form for config/CLI code.
    build_lidar_sector_groups = lidar_sector_groups

    def expanded_rows(self) -> list[dict[str, object]]:
        """Return serializable per-feature rows for CSV/DataFrame export."""

        rows: list[dict[str, object]] = []
        for feature in self.features:
            rows.append(
                {
                    "index": feature.index,
                    "name": feature.name,
                    "block": feature.block,
                    "kind": feature.kind,
                    "angle_deg": feature.angle_deg,
                    "groups": ",".join(feature.groups),
                    "description": feature.description,
                    "constant": feature.constant,
                    "resolved": feature.resolved,
                }
            )
        return rows


_ROOT_KEYS: Final[frozenset[str]] = frozenset(
    {"schema_version", "name", "observation_dim", "blocks", "ranges", "groups"}
)
_BLOCK_KEYS: Final[frozenset[str]] = frozenset(
    {
        "name",
        "start",
        "feature_names",
        "kind",
        "resolved",
        "description",
        "group",
        "groups",
        "constants",
    }
)
_RANGE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "name",
        "start",
        "count",
        "feature_name_template",
        "kind",
        "angle_start_deg",
        "angle_step_deg",
        "angle_direction",
        "resolved",
        "description",
        "group",
        "groups",
        "constants",
    }
)
_GROUP_KEYS: Final[frozenset[str]] = frozenset({"name", "members", "description"})


@dataclass(frozen=True, slots=True)
class _Source:
    """Intermediate block/range expansion used to resolve group members."""

    name: str
    kind: str
    indices: tuple[int, ...]


def _error(location: str, message: str) -> ObservationSchemaError:
    return ObservationSchemaError(f"{location}: {message}")


def _table(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise _error(location, "TOML tableで指定してください")
    return value


def _reject_unknown(table: Mapping[str, object], allowed: frozenset[str], location: str) -> None:
    unknown = sorted(set(table).difference(allowed))
    if unknown:
        raise _error(location, f"未対応のkeyがあります: {', '.join(unknown)}")


def _required(table: Mapping[str, object], key: str, location: str) -> object:
    if key not in table:
        raise _error(location, f"必須key '{key}' がありません")
    return table[key]


def _string(value: object, location: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str):
        raise _error(location, "文字列で指定してください")
    if nonempty and not value.strip():
        raise _error(location, "空でない文字列で指定してください")
    if "\x00" in value:
        raise _error(location, "NUL文字は指定できません")
    return value


def _integer(value: object, location: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _error(location, "boolではない整数で指定してください")
    if minimum is not None and value < minimum:
        raise _error(location, f"{minimum} 以上の整数で指定してください")
    return value


def _real(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error(location, "boolではない有限の数値で指定してください")
    result = float(value)
    if not math.isfinite(result):
        raise _error(location, "有限の数値で指定してください")
    return result


def _boolean(value: object, location: str) -> bool:
    if not isinstance(value, bool):
        raise _error(location, "boolで指定してください")
    return value


def _array(value: object, location: str) -> list[object]:
    if not isinstance(value, list):
        raise _error(location, "arrayで指定してください")
    return value


def _groups_from_entry(table: Mapping[str, object], location: str) -> tuple[str, ...]:
    if "group" in table and "groups" in table:
        raise _error(location, "groupとgroupsを同時に指定できません")
    if "group" in table:
        return (_string(table["group"], f"{location}.group"),)
    if "groups" not in table:
        return ()
    groups = tuple(
        _string(value, f"{location}.groups[{index}]")
        for index, value in enumerate(_array(table["groups"], f"{location}.groups"))
    )
    if not groups:
        raise _error(f"{location}.groups", "少なくとも1つのgroupを指定してください")
    if len(set(groups)) != len(groups):
        raise _error(f"{location}.groups", "同じgroupを重複指定できません")
    return groups


def _constants(value: object | None, count: int, location: str) -> tuple[float | None, ...]:
    if value is None:
        return (None,) * count
    if not isinstance(value, list):
        # Accepting a scalar is useful for a whole semantic block, while still
        # retaining an exact, visible constant in the expanded schema.
        return (_real(value, location),) * count
    if len(value) != count:
        raise _error(location, f"feature数と同じ{count}件のconstantsが必要です")
    return tuple(_real(item, f"{location}[{index}]") for index, item in enumerate(value))


def _description(table: Mapping[str, object], location: str) -> str | None:
    return _string(table["description"], f"{location}.description") if "description" in table else None


def _sector_name(start: float, end: float) -> str:
    def format_angle(value: float) -> str:
        text = f"{value:g}".replace("-", "m").replace(".", "p")
        return text.zfill(3) if text.isdigit() else text

    return f"lidar_sector_{format_angle(start)}_{format_angle(end)}_deg"


def _format_range_name(template: str, *, offset: int, index: int, source_name: str, location: str) -> str:
    try:
        name = template.format(offset=offset, index=index, name=source_name)
    except (KeyError, IndexError, ValueError) as error:
        raise _error(location, f"feature_name_templateを展開できません: {error}") from error
    return _string(name, location)


def _expand_block(
    entry: Mapping[str, object], entry_index: int
) -> tuple[list[Feature], _Source, dict[str, list[int]]]:
    location = f"blocks[{entry_index}]"
    _reject_unknown(entry, _BLOCK_KEYS, location)
    name = _string(_required(entry, "name", location), f"{location}.name")
    start = _integer(_required(entry, "start", location), f"{location}.start", minimum=0)
    names = [
        _string(value, f"{location}.feature_names[{index}]")
        for index, value in enumerate(_array(_required(entry, "feature_names", location), f"{location}.feature_names"))
    ]
    if not names:
        raise _error(f"{location}.feature_names", "少なくとも1つのfeature名が必要です")
    if len(set(names)) != len(names):
        raise _error(f"{location}.feature_names", "feature名が重複しています")
    kind = _string(entry.get("kind", "scalar"), f"{location}.kind")
    resolved = _boolean(entry.get("resolved", True), f"{location}.resolved")
    description = _description(entry, location)
    constants = _constants(entry.get("constants"), len(names), f"{location}.constants")
    groups = _groups_from_entry(entry, location)
    indices = tuple(range(start, start + len(names)))
    features = [
        Feature(
            index=index,
            name=feature_name,
            block=name,
            kind=kind,
            groups=groups,
            description=description,
            constant=constants[offset],
            resolved=resolved,
        )
        for offset, (index, feature_name) in enumerate(zip(indices, names, strict=True))
    ]
    implicit_groups = {name: list(indices)}
    for group_name in groups:
        implicit_groups.setdefault(group_name, []).extend(indices)
    return features, _Source(name=name, kind="block", indices=indices), implicit_groups


def _expand_range(
    entry: Mapping[str, object], entry_index: int
) -> tuple[list[Feature], _Source, dict[str, list[int]]]:
    location = f"ranges[{entry_index}]"
    _reject_unknown(entry, _RANGE_KEYS, location)
    name = _string(_required(entry, "name", location), f"{location}.name")
    start = _integer(_required(entry, "start", location), f"{location}.start", minimum=0)
    count = _integer(_required(entry, "count", location), f"{location}.count", minimum=1)
    template = _string(
        _required(entry, "feature_name_template", location),
        f"{location}.feature_name_template",
    )
    kind = _string(_required(entry, "kind", location), f"{location}.kind")
    resolved = _boolean(entry.get("resolved", True), f"{location}.resolved")
    description = _description(entry, location)
    constants = _constants(entry.get("constants"), count, f"{location}.constants")
    groups = _groups_from_entry(entry, location)
    angle_keys = {"angle_start_deg", "angle_step_deg", "angle_direction"}.intersection(entry)
    if angle_keys and angle_keys != {"angle_start_deg", "angle_step_deg", "angle_direction"}:
        raise _error(location, "angle_start_deg/angle_step_deg/angle_directionをすべて指定してください")
    angle_start: float | None = None
    angle_step: float | None = None
    direction: str | None = None
    if angle_keys:
        angle_start = _real(entry["angle_start_deg"], f"{location}.angle_start_deg")
        angle_step = _real(entry["angle_step_deg"], f"{location}.angle_step_deg")
        if angle_step <= 0.0:
            raise _error(f"{location}.angle_step_deg", "0より大きい数値で指定してください")
        direction = _string(entry["angle_direction"], f"{location}.angle_direction")
        if direction not in {"clockwise", "counterclockwise"}:
            raise _error(f"{location}.angle_direction", "clockwiseまたはcounterclockwiseで指定してください")
    indices = tuple(range(start, start + count))
    features: list[Feature] = []
    # Schema angles are reported in the declared sensor direction.  Thus a
    # clockwise scan with start=0, step=1.5 has 0, 1.5, 3.0, ... metadata;
    # the direction field preserves its physical orientation.
    sign = 1.0 if direction == "clockwise" else -1.0
    for offset, index in enumerate(indices):
        angle = None if angle_start is None or angle_step is None else (angle_start + sign * offset * angle_step) % 360.0
        features.append(
            Feature(
                index=index,
                name=_format_range_name(
                    template,
                    offset=offset,
                    index=index,
                    source_name=name,
                    location=f"{location}.feature_name_template",
                ),
                block=name,
                kind=kind,
                angle_deg=angle,
                groups=groups,
                description=description,
                constant=constants[offset],
                resolved=resolved,
            )
        )
    implicit_groups = {name: list(indices)}
    for group_name in groups:
        implicit_groups.setdefault(group_name, []).extend(indices)
    return features, _Source(name=name, kind="range", indices=indices), implicit_groups


def _resolve_group_member(
    member: str,
    *,
    source_indices: Mapping[str, tuple[int, ...]],
    feature_indices: Mapping[str, int],
    location: str,
) -> tuple[int, ...]:
    """Resolve a bare or explicitly typed group member reference."""

    prefix, separator, reference = member.partition(":")
    if separator:
        if prefix == "feature":
            if reference not in feature_indices:
                raise _error(location, f"未定義featureです: {reference}")
            return (feature_indices[reference],)
        if prefix in {"block", "range"}:
            if reference not in source_indices:
                raise _error(location, f"未定義{prefix}です: {reference}")
            return source_indices[reference]
        raise _error(location, "member prefixはfeature/block/rangeだけを指定できます")
    in_source = member in source_indices
    in_feature = member in feature_indices
    if in_source and in_feature:
        raise _error(location, f"曖昧なmemberです: {member}。feature:またはblock:/range:を付けてください")
    if in_source:
        return source_indices[member]
    if in_feature:
        return (feature_indices[member],)
    raise _error(location, f"未定義feature/block/rangeです: {member}")


def _expand_root_groups(
    value: object,
    *,
    source_indices: Mapping[str, tuple[int, ...]],
    feature_indices: Mapping[str, int],
) -> tuple[dict[str, list[int]], dict[str, str | None]]:
    groups: dict[str, list[int]] = {}
    descriptions: dict[str, str | None] = {}
    for number, raw_group in enumerate(_array(value, "groups")):
        location = f"groups[{number}]"
        group = _table(raw_group, location)
        _reject_unknown(group, _GROUP_KEYS, location)
        name = _string(_required(group, "name", location), f"{location}.name")
        if name in groups:
            raise _error(location, f"group名が重複しています: {name}")
        members = _array(_required(group, "members", location), f"{location}.members")
        if not members:
            raise _error(f"{location}.members", "少なくとも1つのmemberが必要です")
        indices: list[int] = []
        for member_index, raw_member in enumerate(members):
            member = _string(raw_member, f"{location}.members[{member_index}]")
            indices.extend(
                _resolve_group_member(
                    member,
                    source_indices=source_indices,
                    feature_indices=feature_indices,
                    location=f"{location}.members[{member_index}]",
                )
            )
        groups[name] = list(dict.fromkeys(indices))
        descriptions[name] = _description(group, location)
    return groups, descriptions


def load_observation_schema(
    path: str | Path, *, allow_unresolved: bool = False
) -> ObservationSchema:
    """Expand and validate a portable observation schema TOML.

    A template may contain ``resolved = false`` blocks/ranges for human
    editing.  Such a template can only be inspected using
    ``allow_unresolved=True``; normal analysis must reject it before a model
    is evaluated.
    """

    source_path = Path(path).expanduser().resolve()
    if source_path.suffix.lower() != ".toml":
        raise ObservationSchemaError("observation schema: .tomlファイルを指定してください")
    try:
        source = source_path.read_bytes()
    except OSError as error:
        raise ObservationSchemaError(f"observation schemaを読めません: {source_path}") from error
    try:
        raw = tomllib.load(io.BytesIO(source))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
        raise ObservationSchemaError(f"observation schemaのTOML形式が不正です: {source_path}") from error
    root = _table(raw, "root")
    _reject_unknown(root, _ROOT_KEYS, "root")
    schema_version = _integer(_required(root, "schema_version", "root"), "schema_version", minimum=1)
    if schema_version != 1:
        raise _error("schema_version", "現在対応しているversionは1です")
    name = _string(_required(root, "name", "root"), "name")
    observation_dim = _integer(
        _required(root, "observation_dim", "root"), "observation_dim", minimum=1
    )
    raw_blocks = _array(root.get("blocks", []), "blocks")
    raw_ranges = _array(root.get("ranges", []), "ranges")
    if not raw_blocks and not raw_ranges:
        raise _error("root", "blocksまたはrangesを少なくとも1つ指定してください")

    features: list[Feature] = []
    sources: list[_Source] = []
    implicit_groups: dict[str, list[int]] = {}
    for number, raw_block in enumerate(raw_blocks):
        block_features, source, groups = _expand_block(_table(raw_block, f"blocks[{number}]"), number)
        features.extend(block_features)
        sources.append(source)
        for group_name, indices in groups.items():
            implicit_groups.setdefault(group_name, []).extend(indices)
    for number, raw_range in enumerate(raw_ranges):
        range_features, source, groups = _expand_range(_table(raw_range, f"ranges[{number}]"), number)
        features.extend(range_features)
        sources.append(source)
        for group_name, indices in groups.items():
            implicit_groups.setdefault(group_name, []).extend(indices)

    source_names = [source.name for source in sources]
    if len(set(source_names)) != len(source_names):
        duplicates = sorted({name for name in source_names if source_names.count(name) > 1})
        raise _error("blocks/ranges", f"block/range名が重複しています: {', '.join(duplicates)}")
    source_indices = {source.name: source.indices for source in sources}
    feature_indices: dict[str, int] = {}
    for feature in features:
        if feature.name in feature_indices:
            raise _error("blocks/ranges", f"feature名が重複しています: {feature.name}")
        feature_indices[feature.name] = feature.index

    explicit_groups, descriptions = _expand_root_groups(
        root.get("groups", []),
        source_indices=source_indices,
        feature_indices=feature_indices,
    )
    all_groups: dict[str, list[int]] = {
        name: list(dict.fromkeys(indices)) for name, indices in implicit_groups.items()
    }
    for group_name, indices in explicit_groups.items():
        # A root group with the same name extends an implicit block/group. It
        # makes schemas concise while preserving a deterministic union.
        all_groups.setdefault(group_name, []).extend(indices)
        all_groups[group_name] = list(dict.fromkeys(all_groups[group_name]))

    index_groups: dict[int, list[str]] = {feature.index: list(feature.groups) for feature in features}
    for group_name, indices in all_groups.items():
        for index in indices:
            index_groups.setdefault(index, []).append(group_name)
    enriched_features = tuple(
        replace(feature, groups=tuple(dict.fromkeys(index_groups.get(feature.index, ()))))
        for feature in features
    )
    schema = ObservationSchema(
        schema_version=schema_version,
        name=name,
        observation_dim=observation_dim,
        features=enriched_features,
        groups={name: tuple(indices) for name, indices in all_groups.items()},
        group_descriptions=descriptions,
        source_path=source_path,
    )
    if schema.unresolved_features and not allow_unresolved:
        unresolved = ", ".join(feature.name for feature in schema.unresolved_features[:8])
        suffix = "..." if len(schema.unresolved_features) > 8 else ""
        raise ObservationSchemaError(
            "schema contains unresolved placeholder features; "
            f"resolve them before analysis: {unresolved}{suffix}"
        )
    return schema


def load_schema(path: str | Path, *, allow_unresolved: bool = False) -> ObservationSchema:
    """Compatibility alias for :func:`load_observation_schema`."""

    return load_observation_schema(path, allow_unresolved=allow_unresolved)
