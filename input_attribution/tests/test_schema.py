"""Source-backed observation schema and pattern contract tests."""

from __future__ import annotations

import math

import pytest

from input_attribution.schema import (
    LIDAR_GROUP_INDICES,
    SchemaError,
    default_patterns,
    lidar_partition,
    load_schema,
    schema_document,
    standard_259_schema,
    synthetic_259_schema,
    synthetic_262_schema,
    template_262_schema,
    validate_patterns,
)


def test_standard_schema_has_complete_verified_per_index_metadata() -> None:
    rows = standard_259_schema()

    assert len(rows) == 259
    assert [row["index"] for row in rows] == list(range(259))
    required = {
        "index",
        "name",
        "group",
        "dtype",
        "normalization",
        "normal_range",
        "zero_meaning",
        "one_meaning",
        "source",
        "source_version",
        "source_commit",
        "source_path",
        "source_lines",
        "status",
    }
    assert all(required <= set(row) for row in rows)
    assert all(row["dtype"] == "float32" for row in rows)
    assert all(row["status"] == "verified" for row in rows)
    assert all(row["source_version"] == "0.4.3" for row in rows)
    assert all(row["source_commit"] == "85e5dadc6c7436d324348f6e3d8f8e680c06b4db" for row in rows)
    assert rows[0]["name"] == "road_edge_left_distance_norm"
    assert rows[18]["name"] == "navigation_next_next_checkpoint_curve_angle_norm"
    assert rows[19]["name"] == "lidar_beam_000"
    assert rows[258]["name"] == "lidar_beam_239"


def test_lidar_partition_is_exactly_four_disjoint_sixty_beam_groups() -> None:
    partition = lidar_partition()
    groups = list(partition.values())

    assert list(partition) == ["lidar_front", "lidar_left", "lidar_rear", "lidar_right"]
    assert [len(indices) for indices in groups] == [60, 60, 60, 60]
    assert len({index for indices in groups for index in indices}) == 240
    assert sorted(index for indices in groups for index in indices) == list(range(240))
    assert partition["lidar_front"] == list(range(30)) + list(range(210, 240))
    assert partition["lidar_right"] == list(range(30, 90))
    assert partition["lidar_rear"] == list(range(90, 150))
    assert partition["lidar_left"] == list(range(150, 210))

    rows = standard_259_schema()
    for group_name, beams in LIDAR_GROUP_INDICES.items():
        expected = {19 + beam for beam in beams}
        actual = {row["index"] for row in rows if row["group"] == group_name}
        assert actual == expected
    assert rows[19]["relative_angle_deg"] == pytest.approx(0.0)
    assert rows[79]["relative_angle_deg"] == pytest.approx(90.0)
    assert rows[139]["relative_angle_deg"] == pytest.approx(180.0)
    assert rows[199]["relative_angle_deg"] == pytest.approx(270.0)


def test_observation_rows_preserve_executable_formulas_and_provenance() -> None:
    rows = standard_259_schema()

    assert "MAX_LANE_NUM" in rows[0]["normalization"]
    assert "last_current_action[1][0]" in rows[5]["normalization"]
    assert "last_current_action[1][1]" in rows[6]["normalization"]
    assert "arccos" in rows[7]["normalization"]
    assert "(dir + 1) / 2" in rows[12]["normalization"]
    assert "returns 0.5" in rows[13]["notes"]
    assert "degrees(ref_lane.angle)" in rows[13]["normalization"]
    assert "returns 0.5" in rows[18]["notes"]
    assert rows[19]["lidar_distance_m"] == pytest.approx(50.0)
    assert rows[19]["source_file_sha256"]["metadrive/utils/math.py"]
    assert rows[19]["source_file_sha256"]["metadrive/component/sensors/distance_detector.py"]


def test_262_host_template_is_unknown_at_every_position() -> None:
    rows = template_262_schema()

    assert len(rows) == 262
    assert [row["index"] for row in rows] == list(range(262))
    assert {row["group"] for row in rows} == {"host_input_unverified"}
    assert all(row["status"] == "unverified" for row in rows)
    assert all(row["normal_range"] is None for row in rows)
    assert all(row["source_version"] is None for row in rows)
    assert all(row["source_commit"] is None for row in rows)
    assert all("position" in row["host_verification_fields"] for row in rows)
    assert rows[0]["name"] != standard_259_schema()[0]["name"]
    assert rows[259]["name"] == "host_input_259_unverified"
    assert rows[261]["name"] == "host_input_261_unverified"

    with pytest.raises(SchemaError, match="lacks a verified normal_range"):
        load_schema("host_262_migration_template", require_verified=True)
    document = schema_document(262)
    assert document["verified"] is False
    assert document["real_host_verified"] is False
    assert document["groups"] == [
        {
            "id": "host_input_unverified",
            "name": "target host positions requiring verification",
            "indices": list(range(262)),
        }
    ]


def test_synthetic_widths_are_explicit_and_carry_no_real_host_claim() -> None:
    rows_259 = synthetic_259_schema()
    rows_262 = synthetic_262_schema()

    assert len(rows_259) == 259
    assert len(rows_262) == 262
    assert all(row["group"] == "synthetic_features" for row in rows_259 + rows_262)
    assert all(row["status"] == "verified" for row in rows_259 + rows_262)
    assert all(row["source_version"] == "input_attribution.schema.v1" for row in rows_259 + rows_262)
    assert all(row["source_commit"] is None for row in rows_259 + rows_262)
    assert load_schema("synthetic_259_contract")[0]["name"] == "synthetic_feature_000"
    assert load_schema("synthetic_262_contract")[-1]["index"] == 261
    assert load_schema("input_schema_synthetic_262.toml")[-1]["name"] == "synthetic_feature_261"


def test_dimension_only_schema_is_rejected_and_generated_files_are_loadable() -> None:
    with pytest.raises(SchemaError, match="dimension alone is insufficient"):
        load_schema({"dimension": 259})
    assert len(load_schema("input_schema_259.toml")) == 259
    assert len(load_schema("input_schema_262_template.toml")) == 262


def test_default_patterns_are_plain_dicts_in_verified_group_order() -> None:
    patterns = default_patterns(259)

    assert len(patterns) == 10
    assert [pattern["id"] for pattern in patterns] == [f"P{number:02d}" for number in range(1, 11)]
    assert [pattern["name"] for pattern in patterns] == [
        "road_edges",
        "lane_heading",
        "speed",
        "controls_history_yaw",
        "lane_lateral",
        "navigation",
        "lidar_front",
        "lidar_left",
        "lidar_rear",
        "lidar_right",
    ]
    assert all(set(pattern) == {"id", "name", "indices", "fixed_value"} for pattern in patterns)
    assert patterns[6]["indices"] == list(range(19, 49)) + list(range(229, 259))
    assert patterns[6]["fixed_value"] == -1.0
    assert default_patterns(259, fixed_value=-100.0)[0]["fixed_value"] == -100.0
    assert len(default_patterns(2)) == 1
    with pytest.raises(SchemaError, match="wider than the verified 259"):
        default_patterns(262)


def test_patterns_reject_nonfinite_duplicate_and_out_of_range_values() -> None:
    valid = [{"id": "P", "name": "test", "indices": [0], "fixed_value": -100.0}]
    assert validate_patterns(valid, dimension=1)[0]["fixed_value"] == -100.0
    with pytest.raises(SchemaError, match="finite"):
        validate_patterns([{**valid[0], "fixed_value": math.nan}], dimension=1)
    with pytest.raises(SchemaError, match="duplicate pattern id"):
        validate_patterns(valid + valid, dimension=1)
    with pytest.raises(SchemaError, match="duplicate index"):
        validate_patterns([{**valid[0], "indices": [0, 0]}], dimension=1)
    with pytest.raises(SchemaError, match="outside"):
        validate_patterns([{**valid[0], "indices": [1]}], dimension=1)
