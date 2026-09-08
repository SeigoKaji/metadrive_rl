"""Pre-action rollout collection and durable offline reload tests."""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import numpy as np

from input_attribution.rollout import collect_rollout, load_rollout, save_rollout
from input_attribution.compact import finalize_result_directory


class _Space:
    def __init__(self, *, shape: tuple[int, ...] | None = None, n: int | None = None) -> None:
        self.shape = shape
        self.n = n


class _Env:
    def __init__(self) -> None:
        self.observation_space = _Space(shape=(2,))
        self.action_space = _Space(n=2)
        self.current_seed: int | None = None
        self.current: np.ndarray | None = None
        self.step_number = 0
        self.config: dict[str, object] = {}
        self.seen_before_step: list[np.ndarray] = []

    def reset(self, *, seed: int) -> tuple[np.ndarray, dict[str, object]]:
        self.current_seed = seed
        self.step_number = 0
        self.current = np.asarray([float(seed), 0.0], dtype=np.float32)
        return self.current, {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        assert self.current is not None
        self.seen_before_step.append(self.current.copy())
        self.step_number += 1
        self.current = np.asarray(
            [float(self.current_seed), float(self.step_number)], dtype=np.float32
        )
        return self.current, float(action), self.step_number == 2, False, {}


class _MutatingAdapter:
    """Confirms collector does not share saved/env observations with policy input."""

    def evaluate(self, observations: np.ndarray) -> SimpleNamespace:
        copied = observations.copy()
        observations[...] = -123.0
        logits = np.tile(np.asarray([[0.0, 1.0]], dtype=np.float32), (copied.shape[0], 1))
        return SimpleNamespace(
            logits=logits,
            log_probabilities=np.log(np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)),
            probabilities=np.full((copied.shape[0], 2), 0.5, dtype=np.float32),
            deterministic_actions=np.ones(copied.shape[0], dtype=np.int64),
            values=np.zeros(copied.shape[0], dtype=np.float32),
        )


def test_collector_saves_pre_action_copies_and_loads_standard_artifacts(tmp_path: Path) -> None:
    env = _Env()
    rollout = collect_rollout(
        env=env,
        adapter=_MutatingAdapter(),
        model=None,
        schema={"observation_dim": 2},
        scenario_start=5,
        scenario_count=1,
        deterministic=True,
        rl_seed=0,
    )

    assert rollout.metadata["runtime_contract"]["reset_observation_dim"] == 2

    assert np.array_equal(rollout.observations, [[5.0, 0.0], [5.0, 1.0]])
    assert np.array_equal(env.seen_before_step, [[5.0, 0.0], [5.0, 1.0]])
    assert np.array_equal(rollout.selected_actions, [1, 1])
    assert np.array_equal(rollout.env_actions, [1, 1])
    assert rollout.step_records[0]["step"] == 1
    assert rollout.step_records[0]["done"] is False
    assert rollout.step_records[1]["done"] is True

    paths = save_rollout(tmp_path, rollout)
    assert {path.name for path in paths.values()} == {
        "rollout_arrays.npz",
        "rollout_steps.jsonl",
        "rollout_metadata.json",
    }
    loaded = load_rollout(tmp_path)
    assert np.array_equal(loaded.observations, rollout.observations)
    assert np.array_equal(loaded.log_probabilities, rollout.log_probabilities)
    assert np.array_equal(loaded.dones, rollout.dones)
    assert len(loaded.step_records) == 2


def test_loader_resolves_numbered_shared_rollout_artifacts(tmp_path: Path) -> None:
    rollout = collect_rollout(
        env=_Env(),
        adapter=_MutatingAdapter(),
        model=None,
        schema={"observation_dim": 2},
        scenario_start=5,
        scenario_count=1,
        deterministic=True,
        rl_seed=0,
    )
    save_rollout(tmp_path, rollout)
    (tmp_path / "feature_schema_expanded.csv").write_text("index,name\n0,speed\n", encoding="utf-8")
    finalize_result_directory(tmp_path, output_mode="full", result_kind="collect")

    loaded = load_rollout(tmp_path)
    assert np.array_equal(loaded.observations, rollout.observations)
    assert (tmp_path / "shared" / "rollout_arrays.npz").is_file()
