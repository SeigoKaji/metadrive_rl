"""TOML正本化後のinspect/factory境界を軽量に検証する。"""

from __future__ import annotations

from pathlib import Path

import pytest

from configs.experiment_config import canonical_config_path
from env_factory import make_env, make_evaluation_env, make_training_env
from inspect_env import _inspection_log_path, parse_args as parse_inspection_args
from project_paths import OUTPUT_DIR


def test_inspect_selects_the_requested_profile_and_stage() -> None:
    """inspectのpreflightも学習・評価と同じcanonical TOMLを解決する。"""

    default_args = parse_inspection_args([])
    evaluation_args = parse_inspection_args(
        ["--profile", "generalization", "--stage", "evaluation"]
    )

    assert default_args.config == canonical_config_path("official").resolve()
    assert default_args.stage == "train"
    assert evaluation_args.config == canonical_config_path("generalization").resolve()
    assert evaluation_args.profile == "generalization"
    assert evaluation_args.stage == "evaluation"
    assert evaluation_args.experiment.profile.evaluation_env_config["start_seed"] == 0
    assert _inspection_log_path(
        evaluation_args.experiment,
        evaluation_args.stage,
    ) == (OUTPUT_DIR / "inspect_env" / "generalization" / "evaluation.log")


def test_inspect_accepts_an_arbitrary_toml_source(tmp_path: Path) -> None:
    """--configはprofile aliasと排他的に任意のTOMLをpreflightできる。"""

    source = canonical_config_path("official")
    custom = tmp_path / "copied_official.toml"
    custom.write_bytes(source.read_bytes())

    args = parse_inspection_args(["--config", str(custom), "--stage", "evaluation"])

    assert args.config == custom.resolve()
    assert args.stage == "evaluation"
    assert args.experiment.source_path == custom.resolve()

    with pytest.raises(SystemExit):
        parse_inspection_args(
            ["--profile", "official", "--config", str(custom)]
        )


def test_environment_factories_require_resolved_arguments() -> None:
    """factoryはofficial global fallbackを持たず、呼出側の設定漏れを即座に示す。"""

    with pytest.raises(TypeError):
        make_env()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        make_training_env(0)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        make_evaluation_env()  # type: ignore[call-arg]
