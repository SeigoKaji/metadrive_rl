"""TOML正本化後のinspect/factory境界を軽量に検証する。"""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

import inspect_env
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


def test_inspect_displays_nested_effective_config_without_changing_defaults(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Config/list/boolをJSONで表示し、未指定defaultと専用seed一覧を保つ。"""

    from metadrive.utils.config import Config

    actual = Config({
        "map_config": {"config": "SSS", "lane_num": 2, "length": 100, "runtime_type": object},
        "lead_vehicles": {"count": [1, 2], "enabled": False, "lane": "ego"},
        "variants": [Config({"enabled": True, "default": 3})],
        "random_spawn_lane_index": False,
        "unrequested": 7,
    })
    variant = actual["variants"][0]
    requested = {
        "map_config": {"config": "SSS", "lane_num": 2},
        "lead_vehicles": {"count": [1, 2], "enabled": False},
        "variants": [{"enabled": True}],
        "random_spawn_lane_index": False,
        "scenario_seeds": [9, 5],
    }

    inspect_env._print_effective_environment_config(actual, requested)
    output = capsys.readouterr().out
    displayed = json.loads(output.split("== Effective values in env.config ==", 1)[1])

    assert displayed["map_config"] == {
        "config": "SSS", "lane_num": 2, "length": 100, "runtime_type": "<class 'object'>"
    }
    assert displayed["lead_vehicles"] == {"count": [1, 2], "enabled": False, "lane": "ego"}
    assert displayed["variants"] == [{"enabled": True, "default": 3}]
    assert displayed["random_spawn_lane_index"] is False
    assert "scenario_seeds" not in displayed
    assert "unrequested" not in displayed
    assert actual["map_config"]["length"] == 100
    assert actual["map_config"]["runtime_type"] is object
    assert actual["variants"][0] is variant
    assert requested["map_config"] == {"config": "SSS", "lane_num": 2}
    assert requested["scenario_seeds"] == [9, 5]


@pytest.mark.parametrize("requested", [
    {"map_config": {"lane_num": 3}},
    {"lead_vehicles": {"count": [1]}},
    {"lead_vehicles": {"enabled": True}},
])
def test_inspect_still_rejects_different_nested_effective_values(requested: dict) -> None:
    """defaultの追加だけを許し、指定値・配列長・boolの違いは見逃さない。"""

    from metadrive.utils.config import Config

    actual = Config({
        "map_config": {"lane_num": 2, "length": 100},
        "lead_vehicles": {"count": [1, 2], "enabled": False},
    })
    with pytest.raises(AssertionError):
        inspect_env._print_effective_environment_config(actual, requested)


@pytest.mark.parametrize("scenario_seeds", [None, [9, 5], [9, 0]])
def test_inspect_initial_reset_and_checker_use_selected_scenarios(
    scenario_seeds: list[int] | None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """実機なしで既定reset・一覧先頭・checkerの範囲衝突分岐を検証する。"""

    from metadrive.utils.config import Config

    start = 5 if scenario_seeds is None else min(scenario_seeds)
    stop = 6 if scenario_seeds is None else max(scenario_seeds) + 1
    requested = {
        "start_seed": start, "num_scenarios": stop - start,
        "map_config": {"config": "SSS"},
    }
    if scenario_seeds is not None:
        requested["scenario_seeds"] = scenario_seeds

    class InspectionEnv(gym.Env):
        def __init__(self) -> None:
            self.observation_space = gym.spaces.Box(0, 1, shape=(1,), dtype=np.float32)
            self.action_space = gym.spaces.Discrete(9)
            self.config = Config({
                "start_seed": start, "num_scenarios": stop - start,
                "map_config": {"config": "SSS", "lane_num": 2},
            })
            self.reset_calls = []
            self.current_seed = None
            self.closed = False

        def reset(self, **kwargs):
            self.reset_calls.append(kwargs)
            seed = kwargs.get("seed", start)
            assert start <= seed < stop, (
                f"scenario_index (seed) should be in [{start}:{stop})"
            )
            self.current_seed = seed
            return np.array([0.5], dtype=np.float32), {}

        def step(self, action):
            return np.array([0.5], dtype=np.float32), 0.0, False, False, {}

        def close(self):
            self.closed = True

    env = InspectionEnv()
    checker_calls = []
    random_targets = []

    def fake_check_env(checked_env, *, warn):
        assert warn is True
        checker_calls.append(checked_env)
        checked_env.reset(seed=0)
        checked_env.reset()

    def fake_random_actions(checked_env, max_steps):
        assert max_steps == 50
        random_targets.append(checked_env)
        # random走行の開始とdone後の無引数resetを模擬する。
        checked_env.reset()
        checked_env.reset()

    monkeypatch.setattr(inspect_env, "environment_config_for_stage", lambda *_: requested)
    monkeypatch.setattr(inspect_env, "make_env", lambda config: env)
    monkeypatch.setattr(inspect_env, "_print_versions", lambda: None)
    monkeypatch.setattr(inspect_env, "_print_action_conversion", lambda env: None)
    monkeypatch.setattr(inspect_env, "_run_random_actions", fake_random_actions)
    monkeypatch.setattr(inspect_env, "check_env", fake_check_env)

    inspect_env._run_inspection(parse_inspection_args([]).experiment, "train")
    output = capsys.readouterr().out
    inspection_seed = start if scenario_seeds is None else scenario_seeds[0]

    assert env.reset_calls[0] == ({} if scenario_seeds is None else {"seed": inspection_seed})
    if scenario_seeds is None:
        assert env.reset_calls[1] == {"seed": 0}
        assert checker_calls[0] is env
        assert len(checker_calls) == 2
        assert isinstance(checker_calls[1], inspect_env._CheckEnvFixedScenarioAdapter)
        assert env.reset_calls[2] == {"seed": inspection_seed}
        assert "SB3 seed=0 conflicts with MetaDrive's configured scenario range" in output
        assert f"[{start}:{stop})" in output
        assert f"stage start scenario={inspection_seed}" in output
        assert random_targets == [env]
        assert env.reset_calls[-2:] == [{}, {}]
    else:
        assert len(checker_calls) == 1
        assert isinstance(checker_calls[0], inspect_env._CheckEnvFixedScenarioAdapter)
        assert random_targets == checker_calls
        assert env.reset_calls == [{"seed": inspection_seed}] * 5
        assert f"selected-list inspection adapter (scenario={inspection_seed})" in output
        assert "selected-list adapted check_env: PASS" in output
        assert "compatibility finding" not in output
    assert env.closed
    assert "scenario_seeds" not in env.config
    assert requested["map_config"] == {"config": "SSS"}
    if scenario_seeds is not None:
        assert requested["scenario_seeds"] == scenario_seeds


def test_environment_factories_require_resolved_arguments() -> None:
    """factoryはofficial global fallbackを持たず、呼出側の設定漏れを即座に示す。"""

    with pytest.raises(TypeError):
        make_env()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        make_training_env(0)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        make_evaluation_env()  # type: ignore[call-arg]
