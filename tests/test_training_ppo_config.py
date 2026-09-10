"""TOMLからPPO constructorへ渡す解決済みscalarを検証する。"""

from __future__ import annotations

from pathlib import Path

import pytest

import train as train_module
from configs.experiment_config import PPO_COMMON_SCALAR_KEYS, canonical_config_path


def test_training_forwards_all_resolved_ppo_scalars_and_records_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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

        def learn(self, *, total_timesteps: int, log_interval: int) -> None:
            assert log_interval == 4
            self.num_timesteps = total_timesteps

        def save(self, path: str) -> None:
            Path(f"{path}.zip").write_bytes(b"fake PPO model")

        @staticmethod
        def load(*_args: object, **_kwargs: object) -> object:
            return type("ReloadedPPO", (), {"device": "cpu"})()

    monkeypatch.setattr(train_module, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(train_module, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(train_module, "MONITOR_LOG_DIR", tmp_path / "monitor")
    monkeypatch.setattr(train_module, "TENSORBOARD_LOG_DIR", tmp_path / "tensorboard")
    monkeypatch.setattr(train_module, "OUTPUT_DIR", tmp_path / "outputs")
    monkeypatch.setattr(train_module, "SubprocVecEnv", FakeVecEnv)
    monkeypatch.setattr(train_module, "PPO", FakePPO)
    monkeypatch.setattr(
        train_module,
        "_write_json",
        lambda _path, payload: captured_metadata.update(payload),
    )

    args = train_module.parse_args(
        ["--config", str(canonical_config_path("official"))]
    )
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
    assert model_path.is_file()

    training_metadata = captured_metadata["training"]
    assert isinstance(training_metadata, dict)
    resolved = training_metadata["resolved_ppo_config"]
    assert isinstance(resolved, dict)
    assert {key: resolved[key] for key in PPO_COMMON_SCALAR_KEYS} == {
        key: expected[key] for key in PPO_COMMON_SCALAR_KEYS
    }
    assert resolved["n_steps"] == args.n_steps
    assert training_metadata["ppo_seed_argument"] is None


def test_training_serializes_resolved_lookahead_config_into_ppo_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """active TOML値はPPO ZIPのcustom attributeと通常metadataへ届く。"""

    source = canonical_config_path("official")
    config_path = tmp_path / "lookahead.toml"
    config_path.write_text(
        source.read_text(encoding="utf-8")
        + "\n[lookahead]\nlookahead_m = 6.0\npp_weight = 0.25\n",
        encoding="utf-8",
    )
    captured_metadata: dict[str, object] = {}
    saved_attributes: dict[str, object] = {}

    class FakeVecEnv:
        def __init__(self, factories: object) -> None:
            self.factories = factories

        def close(self) -> None:
            pass

    class FakePPO:
        def __init__(self, _policy: str, _env: object, **_kwargs: object) -> None:
            self.device = "cpu"
            self.num_timesteps = 0

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
            Path(f"{path}.zip").write_bytes(b"fake PPO model")

        @staticmethod
        def load(*_args: object, **_kwargs: object) -> object:
            return type("ReloadedPPO", (), {
                "device": "cpu",
                **saved_attributes,
            })()

    monkeypatch.setattr(train_module, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(train_module, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(train_module, "MONITOR_LOG_DIR", tmp_path / "monitor")
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

    expected = {"lookahead_m": 6.0, "pp_weight": 0.25}
    assert saved_attributes == {
        "lookahead_schema_version": 1,
        "lookahead_config": expected,
    }
    assert captured_metadata["lookahead"] == expected
    assert model_path.is_file()
