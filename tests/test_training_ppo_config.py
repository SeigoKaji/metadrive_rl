"""TOMLからPPO constructorへ渡す解決済みscalarを検証する。"""

from __future__ import annotations

from contextlib import closing
from pathlib import Path

import pytest
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.save_util import save_to_zip_file
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import train as train_module
from configs.experiment_config import PPO_COMMON_SCALAR_KEYS, canonical_config_path
from lookahead_learning.checkpoint import resolve_lookahead_config
from evaluate import parse_args as parse_evaluation_args


@pytest.mark.parametrize("experiment_name", ["official", "custom_run", "custom.v2", "custom.zip"])
def test_training_forwards_all_resolved_ppo_scalars_and_records_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    experiment_name: str,
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
    config_path.write_text(
        canonical_config_path("official").read_text(encoding="utf-8").replace(
            'name = "official"', f'name = "{experiment_name}"'
        ),
        encoding="utf-8",
    )
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


@pytest.mark.parametrize("lateral_enabled", [False, True])
def test_training_serializes_resolved_lookahead_config_into_ppo_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lateral_enabled: bool,
) -> None:
    """active TOML値はPPO ZIPのcustom attributeと通常metadataへ届く。"""

    source = canonical_config_path("official")
    config_path = tmp_path / "lookahead.toml"
    config_path.write_text(
        source.read_text(encoding="utf-8")
        + "\n[lookahead]\nlookahead_m = 6.0\npp_weight = 0.25\n"
        + f"lateral_accel_reward_enabled = {str(lateral_enabled).lower()}\n"
        + "max_lateral_accel = 1.2\nlateral_accel_weight = 0.07\n",
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
    })
    assert saved_attributes == {
        "lookahead_schema_version": 2,
        "lookahead_config": expected,
    }
    assert captured_metadata["lookahead"] == expected
    assert worker_configs == [expected] * args.num_envs
    assert model_path.is_file()
