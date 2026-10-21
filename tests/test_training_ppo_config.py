"""TOMLからPPO constructorへ渡す解決済みscalarを検証する。"""

from __future__ import annotations

from contextlib import closing
from pathlib import Path
import pickle
import random

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.save_util import save_to_zip_file
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import train as train_module
import env_factory
from configs.experiment_config import PPO_COMMON_SCALAR_KEYS, canonical_config_path
from lookahead_learning.checkpoint import resolve_lookahead_config
from evaluate import parse_args as parse_evaluation_args


@pytest.mark.parametrize("experiment_name", ["official", "custom_run", "custom.v2", "custom.zip"])
@pytest.mark.parametrize("scenario_seeds", [None, [5, 0, 3], [11], "groups"])
def test_training_forwards_all_resolved_ppo_scalars_and_records_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    experiment_name: str,
    scenario_seeds: list[int] | str | None,
) -> None:
    """SB3をmockし、10 scalarがconstructorとmetadataの双方へ届くことを確認する。"""

    captured_constructor: dict[str, object] = {}
    captured_metadata: dict[str, object] = {}

    class FakeVecEnv:
        def __init__(self, factories: object) -> None:
            self.factories = factories
            self.closed = False

        def close(self) -> None:
            self.closed = True

        def get_attr(self, name):
            if scenario_seeds == "groups":
                assert name == "scene_group_reset_counts"
                return [
                    {"curve__C": {5: 1, 1: 2}, "traffic__SSS": {5: 3}},
                    {"curve__C": {5: 4, 1: 1}, "traffic__SSS": {5: 0}},
                ]
            assert scenario_seeds is not None
            assert name == "scenario_reset_counts"
            if len(scenario_seeds) == 1:
                return [{scenario_seeds[0]: 1}, {scenario_seeds[0]: 2}]
            return [{5: 1, 0: 2, 3: 3}, {5: 2, 0: 0, 3: 1}]

    class FakePPO:
        def __init__(self, _policy: str, _env: object, **kwargs: object) -> None:
            captured_constructor.update(kwargs)
            self.device = "cpu"
            self.num_timesteps = 0

        def set_logger(self, logger: Logger) -> None:
            self.logger = logger

        def learn(self, *, total_timesteps: int, log_interval: int) -> None:
            assert log_interval == 4
            assert self.logger.get_dir() == str(tmp_path / "tensorboard" / experiment_name)
            self.num_timesteps = total_timesteps

        def save(self, path: str) -> None:
            save_to_zip_file(path, data={"test": True})

        @staticmethod
        def load(*_args: object, **_kwargs: object) -> object:
            return type("ReloadedPPO", (), {"device": "cpu"})()

    monkeypatch.setattr(train_module, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(train_module, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(train_module, "TENSORBOARD_LOG_DIR", tmp_path / "tensorboard")
    monkeypatch.setattr(train_module, "OUTPUT_DIR", tmp_path / "outputs")
    monkeypatch.setattr(train_module, "SubprocVecEnv", FakeVecEnv)
    monkeypatch.setattr(train_module, "PPO", FakePPO)
    monkeypatch.setattr(
        train_module,
        "_write_json",
        lambda _path, payload: captured_metadata.update(payload),
    )

    config_path = tmp_path / "experiment.toml"
    text = canonical_config_path("official").read_text(encoding="utf-8").replace(
        'name = "official"', f'name = "{experiment_name}"'
    )
    if scenario_seeds == "groups":
        text = text.split("[environment.common]", 1)[0] + '''[environment.common]
discrete_action = true
discrete_throttle_dim = 3
discrete_steering_dim = 3
log_level = 50
[[environment.scene_groups]]
name = "curve__C"
train_seeds = [5, 1]
evaluation_seeds = [1]
[environment.scene_groups.common]
map = "C"
horizon = 500
[[environment.scene_groups]]
name = "traffic__SSS"
train_seeds = [5]
[environment.scene_groups.common]
map = "SSS"
traffic_density = 0.1
[environment.train]
[environment.evaluation]
'''
    elif scenario_seeds is not None:
        text = text.replace("start_seed = 5\nnum_scenarios = 1", f"scenario_seeds = {scenario_seeds}")
    config_path.write_text(text, encoding="utf-8")
    args = train_module.parse_args(["--config", str(config_path)])
    model_path = train_module._run_training(args, tmp_path / "train.log")

    expected = {
        "n_steps": 4096,
        "learning_rate": 0.0003,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "normalize_advantage": True,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    }
    assert {key: captured_constructor[key] for key in expected} == expected
    assert "seed" not in captured_constructor
    tensorboard_log_dir = tmp_path / "tensorboard" / experiment_name
    assert captured_constructor["tensorboard_log"] == str(tensorboard_log_dir)
    assert list(tensorboard_log_dir.glob("events.out.tfevents.*"))
    assert list((tmp_path / "tensorboard").iterdir()) == [tensorboard_log_dir]
    evaluation_args = parse_evaluation_args(["--config", str(config_path)])
    assert model_path == tmp_path / "models" / f"{experiment_name}.zip"
    assert model_path == tmp_path / evaluation_args.model
    assert model_path.is_file()
    artifacts = captured_metadata["artifacts"]
    assert artifacts["model_path"] == str(model_path)
    assert artifacts["metadata_path"] == str(
        tmp_path / "outputs" / experiment_name / "training" / "training_metadata.json"
    )
    assert artifacts["tensorboard_log_directory"] == str(tensorboard_log_dir)

    training_metadata = captured_metadata["training"]
    assert isinstance(training_metadata, dict)
    resolved = training_metadata["resolved_ppo_config"]
    assert isinstance(resolved, dict)
    assert {key: resolved[key] for key in PPO_COMMON_SCALAR_KEYS} == {
        key: expected[key] for key in PPO_COMMON_SCALAR_KEYS
    }
    assert resolved["n_steps"] == args.n_steps
    assert resolved["tensorboard_log"] == str(tensorboard_log_dir)
    assert training_metadata["ppo_seed_argument"] is None
    if scenario_seeds == "groups":
        assert training_metadata["scene_groups"] == args.experiment.profile.train_env_config["scene_groups"]
        assert training_metadata["scene_group_reset_counts"] == {
            "curve__C": {"5": 5, "1": 3}, "traffic__SSS": {"5": 3}
        }
        assert training_metadata["scenario_sampling_reproducible_from_rl_seed"] is True
        assert "uniformly from all (scene_group, seed) pairs" in training_metadata["scenario_sampling_note"]
        assert "scenario_seed" not in training_metadata
        assert "scenario_seeds" not in training_metadata
        assert "scenario_reset_counts" not in training_metadata
    elif scenario_seeds is not None:
        assert training_metadata["scenario_seeds"] == scenario_seeds
        expected_counts = {"11": 3} if len(scenario_seeds) == 1 else {"5": 3, "0": 2, "3": 4}
        assert training_metadata["scenario_reset_counts"] == expected_counts
        assert training_metadata["scenario_sampling_reproducible_from_rl_seed"] is True
        assert training_metadata["scenario_seed_range"] == {
            "start": min(scenario_seeds), "stop_exclusive": max(scenario_seeds) + 1
        }
        if experiment_name == "official" and len(scenario_seeds) == 1:
            assert training_metadata["scenario_seed"] == scenario_seeds[0]
        else:
            assert "scenario_seed" not in training_metadata
    else:
        assert "scenario_seeds" not in training_metadata
        assert "scenario_reset_counts" not in training_metadata


def test_factory_consumes_scenario_list_before_metadrive_construction(monkeypatch):
    from types import SimpleNamespace
    import metadrive.envs

    monkeypatch.setattr(metadrive.envs, "MetaDriveEnv", lambda config: SimpleNamespace(config=config))
    config = {"scenario_seeds": [5, 0, 3], "start_seed": 100, "num_scenarios": 1}
    env = env_factory.make_env(config)
    assert env.config == {"start_seed": 0, "num_scenarios": 6}
    assert config == {"scenario_seeds": [5, 0, 3], "start_seed": 100, "num_scenarios": 1}


def test_training_sampler_is_repeatable_local_and_counts_actual_resets(monkeypatch):
    class FakeEnv(gym.Env):
        def __init__(self):
            self.action_space = gym.spaces.Discrete(2)
            self.observation_space = gym.spaces.Box(0, 20, shape=(1,))
            self.current_seed = -1

        def reset(self, *, seed=None):
            self.current_seed = seed
            return np.asarray([seed], dtype=np.float32), {"scenario_seed": seed}

    monkeypatch.setattr(env_factory, "make_env", lambda *_args, **_kwargs: FakeEnv())
    before = pickle.dumps((random.getstate(), np.random.get_state(), torch.get_rng_state().numpy().tobytes()))
    envs = [env_factory.make_training_env(rank, 11, {"scenario_seeds": [2, 5, 11]})
            for rank in [0, 0, 1]]
    try:
        samples = [[env.reset()[1]["scenario_seed"] for _ in range(30)] for env in envs]
        assert samples[0] == samples[1]
        assert samples[0] != samples[2]
        assert all(set(sample) <= {2, 5, 11} for sample in samples)
        assert envs[0].reset(seed=5)[1]["scenario_seed"] == 5
        assert envs[0].get_wrapper_attr("scenario_reset_counts") == {
            seed: samples[0].count(seed) + int(seed == 5) for seed in [2, 5, 11]
        }
        with pytest.raises(ValueError):
            envs[0].reset(seed=3)
        counts = envs[0].get_wrapper_attr("scenario_reset_counts").copy()
        monkeypatch.setattr(envs[0].unwrapped, "reset", lambda **_kwargs: (np.zeros(1), {}))
        with pytest.raises(RuntimeError, match="requested scenario seed"):
            envs[0].reset(seed=11)
        assert envs[0].get_wrapper_attr("scenario_reset_counts") == counts
        assert pickle.dumps((random.getstate(), np.random.get_state(), torch.get_rng_state().numpy().tobytes())) == before
    finally:
        for env in envs:
            env.close()


def test_training_logger_replaces_only_the_selected_experiments_events(
    tmp_path: Path,
) -> None:
    """再実行後の曲線は最新分だけになり、他の実験・旧ログ・メモを保持する。"""

    tensorboard_root = tmp_path / "tensorboard"
    log_dir = tensorboard_root / "official"
    log_dir.mkdir(parents=True)
    note_path = log_dir / "notes.txt"
    note_path.write_text("keep this note", encoding="utf-8")
    preserved_paths = []
    for name in ("another_experiment", "PPO_1"):
        sibling_dir = tensorboard_root / name
        sibling_dir.mkdir()
        event_path = sibling_dir / "events.out.tfevents.keep"
        event_path.write_bytes(b"keep these events")
        preserved_paths.append(event_path)

    for step, value in ((1, 10.0), (2, 20.0)):
        with closing(train_module._configure_training_logger(log_dir)) as logger:
            logger.record("test/reward", value)
            logger.dump(step=step)
        events = EventAccumulator(str(log_dir)).Reload()
        assert [(event.step, event.value) for event in events.Scalars("test/reward")] == [
            (step, value)
        ]
        assert len(list(log_dir.glob("events.out.tfevents.*"))) == 1

    assert note_path.read_text(encoding="utf-8") == "keep this note"
    assert all(path.read_bytes() == b"keep these events" for path in preserved_paths)
    assert {path.name for path in tensorboard_root.iterdir()} == {
        "official", "another_experiment", "PPO_1"
    }


def test_training_logger_does_not_clear_events_through_a_directory_symlink(
    tmp_path: Path,
) -> None:
    target_dir = tmp_path / "other_experiment"
    target_dir.mkdir()
    event_path = target_dir / "events.out.tfevents.keep"
    event_path.write_bytes(b"keep these events")
    log_dir = tmp_path / "official"
    log_dir.symlink_to(target_dir, target_is_directory=True)

    with pytest.raises(ValueError, match="シンボリックリンク"):
        train_module._configure_training_logger(log_dir)

    assert event_path.read_bytes() == b"keep these events"


@pytest.mark.parametrize("motion_model", ["constant_speed", "constant_acceleration"])
@pytest.mark.parametrize("time_s", [None, 1.0])
@pytest.mark.parametrize("lateral_enabled", [False, True])
def test_training_serializes_resolved_lookahead_config_into_ppo_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lateral_enabled: bool,
    time_s: float | None,
    motion_model: str,
) -> None:
    """active TOML値はPPO ZIPのcustom attributeと通常metadataへ届く。"""

    source = canonical_config_path("official")
    config_path = tmp_path / "lookahead.toml"
    config_path.write_text(
        source.read_text(encoding="utf-8")
        + "\n[lookahead]\nlookahead_m = 6.0\npp_weight = 0.25\n"
        + f"lateral_accel_reward_enabled = {str(lateral_enabled).lower()}\n"
        + "max_lateral_accel = 1.2\nlateral_accel_weight = 0.07\n"
        + ("" if time_s is None else f"lookahead_time_s = {time_s}\n")
        + f"prediction_reward_enabled = {str(time_s is not None).lower()}\n"
        + f'prediction_motion_model = "{motion_model}"\n'
        + "prediction_reward_weight = 0.13\nprediction_error_scale_m = 2.0\n",
        encoding="utf-8",
    )
    captured_metadata: dict[str, object] = {}
    saved_attributes: dict[str, object] = {}
    worker_configs: list[object] = []

    class FakeVecEnv:
        def __init__(self, factories: object) -> None:
            self.factories = factories
            worker_configs.extend(factory.keywords["lookahead_config"] for factory in factories)

        def close(self) -> None:
            pass

    class FakePPO:
        def __init__(self, _policy: str, _env: object, **_kwargs: object) -> None:
            self.device = "cpu"
            self.num_timesteps = 0

        def set_logger(self, logger: Logger) -> None:
            self.logger = logger

        def learn(self, *, total_timesteps: int, log_interval: int) -> None:
            del log_interval
            self.num_timesteps = total_timesteps

        def save(self, path: str) -> None:
            saved_attributes.update(
                {
                    "lookahead_schema_version": self.lookahead_schema_version,
                    "lookahead_config": self.lookahead_config,
                }
            )
            save_to_zip_file(path, data={"test": True})

        @staticmethod
        def load(*_args: object, **_kwargs: object) -> object:
            return type("ReloadedPPO", (), {
                "device": "cpu",
                **saved_attributes,
            })()

    monkeypatch.setattr(train_module, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(train_module, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(train_module, "TENSORBOARD_LOG_DIR", tmp_path / "tensorboard")
    monkeypatch.setattr(train_module, "OUTPUT_DIR", tmp_path / "outputs")
    monkeypatch.setattr(train_module, "SubprocVecEnv", FakeVecEnv)
    monkeypatch.setattr(train_module, "PPO", FakePPO)
    monkeypatch.setattr(
        train_module,
        "_write_json",
        lambda _path, payload: captured_metadata.update(payload),
    )

    args = train_module.parse_args(["--config", str(config_path)])
    model_path = train_module._run_training(args, tmp_path / "train.log")

    expected = resolve_lookahead_config({
        "lookahead_m": 6.0, "pp_weight": 0.25,
        "lateral_accel_reward_enabled": lateral_enabled,
        "max_lateral_accel": 1.2, "lateral_accel_weight": 0.07,
        "lookahead_time_s": time_s, "prediction_reward_enabled": time_s is not None,
        "prediction_reward_weight": 0.13, "prediction_error_scale_m": 2.0,
        "prediction_motion_model": motion_model,
    })
    assert saved_attributes == {
        "lookahead_schema_version": 4,
        "lookahead_config": expected,
    }
    assert captured_metadata["lookahead"] == expected
    assert worker_configs == [expected] * args.num_envs
    assert model_path.is_file()
