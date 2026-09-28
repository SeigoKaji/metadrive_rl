"""Small boundary tests for the optional lookahead environment connection."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import env_factory
from lookahead_learning.checkpoint import resolve_lookahead_config


def test_make_env_wraps_both_raw_host_variants_only_when_config_is_present(
    monkeypatch: Any,
) -> None:
    """The wrapper is after raw host construction, including start-lane tasks."""

    class FakeRaw:
        def __init__(self, config: dict[str, object]) -> None:
            self.config = config
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class FakeStartLaneRaw(FakeRaw):
        pass

    metadrive_envs = ModuleType("metadrive.envs")
    metadrive_envs.MetaDriveEnv = FakeRaw  # type: ignore[attr-defined]
    metadrive = ModuleType("metadrive")
    metadrive.envs = metadrive_envs  # type: ignore[attr-defined]
    start_lane = ModuleType("start_lane_env")
    start_lane.StartLaneMetaDriveEnv = FakeStartLaneRaw  # type: ignore[attr-defined]

    calls: list[tuple[object, dict[str, object]]] = []
    adapter = ModuleType("lookahead_learning.adapter")

    def wrap(raw: object, **config: object) -> object:
        calls.append((raw, config))
        return ("wrapped", raw, config)

    adapter.wrap_lookahead_env = wrap  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "metadrive", metadrive)
    monkeypatch.setitem(sys.modules, "metadrive.envs", metadrive_envs)
    monkeypatch.setitem(sys.modules, "start_lane_env", start_lane)
    monkeypatch.setitem(sys.modules, "lookahead_learning.adapter", adapter)

    lookahead = resolve_lookahead_config({
        "lateral_accel_reward_enabled": True, "lookahead_time_s": 1.2,
        "prediction_reward_enabled": True, "prediction_reward_weight": 0.3,
        "prediction_error_scale_m": 2.0,
    })
    wrapped = env_factory.make_env(
        {"start_lane_objective": "return"},
        lookahead_config=lookahead,
    )
    assert isinstance(wrapped, tuple) and wrapped[0] == "wrapped"
    assert isinstance(calls[0][0], FakeStartLaneRaw)
    assert calls[0][1] == lookahead

    canonical_wrapped = env_factory.make_env(
        {"map": "C"},
        lookahead_config=lookahead,
    )
    assert isinstance(canonical_wrapped, tuple)
    assert isinstance(canonical_wrapped[1], FakeRaw)
    assert canonical_wrapped[1].config == {"map": "C"}
    assert canonical_wrapped[2] == lookahead
    assert isinstance(calls[1][0], FakeRaw)
    assert calls[1][0].config == {"map": "C"}
    assert calls[1][1] == lookahead

    raw = env_factory.make_env({"map": "C"})
    assert isinstance(raw, FakeRaw)
    assert len(calls) == 2  # no wrapper call is made when [lookahead] is absent


def test_stage_factories_forward_the_same_lookahead_config_and_keep_monitor_outer(
    monkeypatch: Any,
) -> None:
    """train/evaluate share one factory setting; training adds Monitor outside it."""

    class FakeSpace:
        def __init__(self) -> None:
            self.seeds: list[int] = []

        def seed(self, value: int) -> None:
            self.seeds.append(value)

    class FakeEnv:
        def __init__(self) -> None:
            self.action_space = FakeSpace()
            self.observation_space = FakeSpace()
            self.closed = False

        def close(self) -> None:
            self.closed = True

    raw_envs: list[FakeEnv] = []
    forwarded: list[dict[str, object] | None] = []

    def fake_make_env(
        _config: object,
        *,
        lookahead_config: dict[str, object] | None = None,
    ) -> FakeEnv:
        forwarded.append(lookahead_config)
        env = FakeEnv()
        raw_envs.append(env)
        return env

    monitored: list[object] = []

    class FakeMonitor:
        def __init__(self, env: object) -> None:
            self.env = env
            monitored.append(self)

    monkeypatch.setattr(env_factory, "make_env", fake_make_env)
    monkeypatch.setattr(env_factory, "Monitor", FakeMonitor)
    lookahead = resolve_lookahead_config({
        "lateral_accel_reward_enabled": True, "lookahead_time_s": 1.2,
        "prediction_reward_enabled": True, "prediction_reward_weight": 0.3,
        "prediction_error_scale_m": 2.0,
    })

    training_env = env_factory.make_training_env(
        rank=0,
        seed=11,
        env_config={"map": "C"},
        lookahead_config=lookahead,
    )
    evaluation_env = env_factory.make_evaluation_env(
        seed=12,
        env_config={"map": "C"},
        lookahead_config=lookahead,
    )

    assert forwarded == [lookahead, lookahead]
    assert training_env is monitored[0]
    assert monitored[0].env is raw_envs[0]  # type: ignore[attr-defined]
    assert evaluation_env is raw_envs[1]
    assert raw_envs[0].action_space.seeds == [11]
    assert raw_envs[1].observation_space.seeds == [12]


def test_portable_examples_use_normal_cli_and_identical_experiment_conditions():
    from pathlib import Path
    from configs.experiment_config import load_experiment_config
    from train import parse_args as train_args
    from evaluate import parse_args as evaluate_args

    base = load_experiment_config("configs/official_start_lane_return_lookahead.toml").profile
    for filename in ("distance.toml", "time_only.toml", "time_prediction.toml"):
        path = str(Path("lookahead_learning/examples") / filename)
        training = train_args(["--config", path])
        evaluation = evaluate_args(["--config", path])
        config = training.experiment.profile.lookahead_config
        assert config == evaluation.experiment.profile.lookahead_config
        assert config["lookahead_time_s"] == (None if filename == "distance.toml" else 1)
        assert config["prediction_reward_enabled"] == (filename == "time_prediction.toml")
        assert config["pp_weight"] == 0
        assert config["lateral_accel_reward_enabled"] is False
        for key in ("train_env_config", "evaluation_env_config", "training_config"):
            assert getattr(training.experiment.profile, key) == getattr(base, key)
