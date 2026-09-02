"""Artifact safety and staging publication behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from input_attribution.results import (
    ArtifactError,
    StagedRunDirectory,
    atomic_write_text,
    expanded_schema_rows,
    safe_child,
    validate_basename,
    write_expanded_schema_csv,
)
from input_attribution.schema import Feature, ObservationSchema


def test_staging_failure_keeps_previous_successful_run(tmp_path: Path) -> None:
    target = tmp_path / "official_left_curve"
    target.mkdir()
    (target / "result.txt").write_text("previous success", encoding="utf-8")

    with pytest.raises(RuntimeError):
        with StagedRunDirectory(target) as staging:
            (staging / "result.txt").write_text("partial new run", encoding="utf-8")
            raise RuntimeError("injected failure")

    assert (target / "result.txt").read_text(encoding="utf-8") == "previous success"
    assert not list(tmp_path.glob(".official_left_curve.staging-*"))

    holder = StagedRunDirectory(target)
    with holder as staging:
        (staging / "result.txt").write_text("new success", encoding="utf-8")
        published = holder.publish()
    assert published == target
    assert (target / "result.txt").read_text(encoding="utf-8") == "new success"


def test_staging_rejects_existing_regular_file_without_side_effects(tmp_path: Path) -> None:
    target = tmp_path / "official_left_curve"
    target.write_text("previous file", encoding="utf-8")

    with pytest.raises(ArtifactError, match="not a regular directory"):
        with StagedRunDirectory(target):
            pytest.fail("a staging directory must not be created for a file target")

    assert target.is_file()
    assert target.read_text(encoding="utf-8") == "previous file"
    assert not list(tmp_path.glob(".official_left_curve.staging-*"))
    assert not list(tmp_path.glob(".official_left_curve.previous-*"))


@pytest.mark.parametrize("name", ("../bad", "sub/path", r"sub\path", "", ".", ".."))
def test_basename_validation_rejects_path_traversal(name: str) -> None:
    with pytest.raises(ArtifactError):
        validate_basename(name)


@pytest.mark.parametrize(
    "name",
    (
        "CON",
        "NUL.txt",
        "COM1",
        "ends-with-dot.",
        "ends-with-space ",
        "has:colon",
        "has*asterisk",
        "has?question",
        "has|pipe",
        'has"quote',
        "has<less",
        "has>greater",
        "has\x1fcontrol",
    ),
)
def test_basename_validation_rejects_windows_reserved_and_invalid_components(name: str) -> None:
    with pytest.raises(ArtifactError):
        validate_basename(name)


def test_basename_validation_keeps_portable_regular_component() -> None:
    assert validate_basename("official_left_curve.v2") == "official_left_curve.v2"


def test_expanded_schema_artifact_preserves_feature_block(tmp_path: Path) -> None:
    schema = ObservationSchema(
        schema_version=1,
        name="blocks",
        observation_dim=1,
        # Direct construction intentionally omits Feature.groups; the schema
        # mapping must still be reflected in the exported row.
        features=(Feature(0, "vehicle_speed", "ego_state"),),
        groups={"ego_state": (0,), "vehicle_state": (0,)},
    )

    rows = expanded_schema_rows(schema)
    output = tmp_path / "feature_schema_expanded.csv"
    write_expanded_schema_csv(output, schema)

    assert rows == [
        {
            "index": 0,
            "name": "vehicle_speed",
            "block": "ego_state",
            "group": "ego_state",
            "groups": "ego_state,vehicle_state",
            "kind": "scalar",
            "angle_deg": None,
            "resolved": True,
            "description": None,
            "constant": None,
        }
    ]
    assert output.read_text(encoding="utf-8").splitlines()[0].split(",")[2] == "block"


def test_safe_child_rejects_parent_traversal(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError):
        safe_child(tmp_path / "artifacts", "../outside.json")


def test_atomic_writer_rejects_a_broken_symlink_destination(tmp_path: Path) -> None:
    destination = tmp_path / "artifact.txt"
    destination.symlink_to(tmp_path / "does_not_exist")

    with pytest.raises(ArtifactError, match="symlink"):
        atomic_write_text(destination, "unsafe")
