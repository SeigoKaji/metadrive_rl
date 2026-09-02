"""Unit tests for portable observation-schema expansion and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from input_attribution.schema import ObservationSchemaError, load_observation_schema


def _write_schema(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "schema.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_official_schema_covers_every_index_and_expands_clockwise_lidar() -> None:
    schema = load_observation_schema("observation_schemas/metadrive_default_259.toml")

    assert schema.observation_dim == 259
    assert [feature.index for feature in schema.features] == list(range(schema.observation_dim))
    assert schema.features[19].name == "lidar_000"
    assert schema.features[19].angle_deg == 0.0
    assert schema.features[20].angle_deg == 1.5
    assert schema.indices_for_group("checkpoint_1") == (9, 10, 11, 12, 13)
    sectors = schema.lidar_sector_groups(15)
    assert sectors["lidar_sector_000_015_deg"] == tuple(range(19, 29))


def test_schema_rejects_overlap_missing_indices_and_unresolved_template(tmp_path: Path) -> None:
    overlap = _write_schema(
        tmp_path,
        """\
schema_version = 1
name = "overlap"
observation_dim = 3
[[blocks]]
name = "first"
start = 0
feature_names = ["a", "b"]
[[blocks]]
name = "second"
start = 1
feature_names = ["c"]
""",
    )
    with pytest.raises(ObservationSchemaError, match="0..D-1"):
        load_observation_schema(overlap)

    missing = _write_schema(
        tmp_path,
        """\
schema_version = 1
name = "missing"
observation_dim = 3
[[blocks]]
name = "only"
start = 0
feature_names = ["a", "b"]
""",
    )
    with pytest.raises(ObservationSchemaError, match="observation_dim"):
        load_observation_schema(missing)

    unresolved = _write_schema(
        tmp_path,
        """\
schema_version = 1
name = "template"
observation_dim = 2
[[blocks]]
name = "known"
start = 0
feature_names = ["a"]
[[blocks]]
name = "placeholder"
start = 1
resolved = false
feature_names = ["UNRESOLVED_b"]
""",
    )
    with pytest.raises(ObservationSchemaError, match="unresolved"):
        load_observation_schema(unresolved)
    inspected = load_observation_schema(unresolved, allow_unresolved=True)
    assert [feature.name for feature in inspected.unresolved_features] == ["UNRESOLVED_b"]


def test_custom_dimension_uses_blocks_ranges_and_overlapping_groups_without_python_changes(
    tmp_path: Path,
) -> None:
    schema = load_observation_schema(
        _write_schema(
            tmp_path,
            """\
schema_version = 1
name = "portable_custom"
observation_dim = 7
[[blocks]]
name = "state"
start = 0
kind = "state"
groups = ["state", "custom"]
constants = [0.0, 1.0]
feature_names = ["left", "right"]
[[ranges]]
name = "laser"
start = 2
count = 5
kind = "lidar"
group = "lidar_all"
feature_name_template = "ray_{index:03d}"
angle_start_deg = 10
angle_step_deg = 20
angle_direction = "counterclockwise"
[[groups]]
name = "combined"
members = ["block:state", "range:laser", "feature:left"]
""",
        )
    )

    assert schema.feature_names == ("left", "right", "ray_002", "ray_003", "ray_004", "ray_005", "ray_006")
    assert schema.features[2].angle_deg == 10.0
    assert schema.features[3].angle_deg == 350.0
    assert schema.indices_for_group("combined") == tuple(range(7))
    assert schema.feature_at(0).constant == 0.0
    assert schema.feature_at(1).constant == 1.0
    assert schema.expanded_rows()[2]["angle_deg"] == 10.0
