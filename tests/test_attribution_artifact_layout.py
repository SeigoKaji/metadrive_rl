"""Small pure-filesystem checks for the numbered result resolver."""

from __future__ import annotations

from pathlib import Path

import pytest

from input_attribution.artifact_layout import (
    ArtifactLayoutError,
    detect_layout,
    resolve_artifact,
)


def test_resolve_artifact_supports_flat_and_numbered_full_layouts(tmp_path: Path) -> None:
    flat = tmp_path / "flat"
    flat.mkdir()
    (flat / "analysis_metadata.json").write_text("{}", encoding="utf-8")
    assert detect_layout(flat).is_flat
    assert resolve_artifact(flat, "analysis_metadata.json") == flat / "analysis_metadata.json"

    numbered = tmp_path / "numbered"
    shared = numbered / "shared"
    shared.mkdir(parents=True)
    (shared / "analysis_metadata.json").write_text("{}", encoding="utf-8")
    assert detect_layout(numbered).is_numbered
    assert resolve_artifact(numbered, "analysis_metadata.json") == shared / "analysis_metadata.json"


def test_mixed_flat_and_numbered_markers_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "shared").mkdir()
    (tmp_path / "shared" / "analysis_metadata.json").write_text("{}", encoding="utf-8")
    (tmp_path / "analysis_metadata.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ArtifactLayoutError, match="mixes flat and numbered"):
        detect_layout(tmp_path)


def test_numbered_symlink_directory_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "analysis_metadata.json").write_text("{}", encoding="utf-8")
    (tmp_path / "shared").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ArtifactLayoutError, match="unsafe symlink"):
        detect_layout(tmp_path)
