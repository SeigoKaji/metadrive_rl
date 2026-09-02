"""Strict analysis-config parsing independent from MetaDrive."""

from __future__ import annotations

from pathlib import Path

import pytest

from input_attribution.config import AnalysisConfigError, load_analysis_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_official_analysis_config_loads_with_expected_defaults() -> None:
    config = load_analysis_config(
        PROJECT_ROOT / "attribution_configs" / "official_left_curve.toml"
    )

    assert config.baseline.strategy == "episode_start"
    assert config.perturbation.lidar_sector_degrees == 15.0
    assert config.integrated_gradients.steps == 64
    assert config.aggregation.progress_bins == 3
    assert config.closed_loop.enabled is True
    assert config.source_sha256 is not None


def test_analysis_config_rejects_unknown_typo_key(tmp_path: Path) -> None:
    source = (
        PROJECT_ROOT / "attribution_configs" / "official_left_curve.toml"
    ).read_text(encoding="utf-8")
    path = tmp_path / "typo.toml"
    path.write_text(
        source.replace("batch_size = 128", "batch_size = 128\nstepz = 64"),
        encoding="utf-8",
    )

    with pytest.raises(AnalysisConfigError, match="stepz"):
        load_analysis_config(path)


def test_analysis_config_rejects_integrated_gradients_steps_below_two(
    tmp_path: Path,
) -> None:
    source = (
        PROJECT_ROOT / "attribution_configs" / "official_left_curve.toml"
    ).read_text(encoding="utf-8")
    path = tmp_path / "invalid_steps.toml"
    path.write_text(source.replace("steps = 64", "steps = 1"), encoding="utf-8")

    with pytest.raises(AnalysisConfigError, match="2 以上"):
        load_analysis_config(path)


@pytest.mark.parametrize(
    ("old", "new", "match"),
    (
        ("record_visualization = false", "record_visualization = true", "record_visualization"),
        ("save_observations = true", "save_observations = false", "save_observations"),
    ),
)
def test_analysis_config_rejects_unimplemented_boolean_modes(
    tmp_path: Path,
    old: str,
    new: str,
    match: str,
) -> None:
    source = (
        PROJECT_ROOT / "attribution_configs" / "official_left_curve.toml"
    ).read_text(encoding="utf-8")
    path = tmp_path / "unsupported_mode.toml"
    path.write_text(source.replace(old, new), encoding="utf-8")

    with pytest.raises(AnalysisConfigError, match=match):
        load_analysis_config(path)
