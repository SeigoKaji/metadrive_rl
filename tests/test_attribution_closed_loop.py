"""Copy/seed invariants for paired closed-loop policy-input replacement."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from input_attribution.closed_loop import (
    ClosedLoopError,
    ClosedLoopTarget,
    replace_policy_observation,
    run_paired_closed_loop,
    schema_targets,
)
from input_attribution.schema import load_observation_schema


class _Space:
    def __init__(self, *, shape: tuple[int, ...] | None = None, n: int | None = None) -> None:
        self.shape = shape
        self.n = n
        self.seeds: list[int] = []

    def seed(self, value: int) -> None:
        self.seeds.append(value)


class _FakeEnv:
    instances: list["_FakeEnv"] = []

    def __init__(self) -> None:
        self.observation_space = _Space(shape=(3,))
        self.action_space = _Space(n=2)
        self.config: dict[str, object] = {}
        self.current_seed: int | None = None
        self.returned: np.ndarray | None = None
        self.returned_before_policy: list[np.ndarray] = []
        self.received_actions: list[int] = []
        self.closed = False
        self.step_number = 0
        type(self).instances.append(self)

    def reset(self, *, seed: int) -> tuple[np.ndarray, dict[str, object]]:
        self.current_seed = seed
        self.step_number = 0
        self.returned = np.asarray([float(seed), 10.0, 20.0], dtype=np.float32)
        return self.returned, {"seed": seed}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        assert self.returned is not None
        self.returned_before_policy.append(self.returned.copy())
        self.received_actions.append(action)
        self.step_number += 1
        self.returned = np.asarray(
            [float(self.current_seed), 10.0 + self.step_number, 20.0], dtype=np.float32
        )
        return self.returned, float(action), self.step_number >= 2, False, {
            "arrive_dest": action == 1,
            "route_completion": 0.5 * self.step_number,
            "target_lane_valid": True,
            "time_in_target_lane_ratio": 0.75,
        }

    def close(self) -> None:
        self.closed = True


class _MutatingAdapter:
    """Deliberately mutates its argument; only a policy-input copy may change."""

    def __init__(self) -> None:
        self.inputs: list[np.ndarray] = []

    def evaluate(self, observations: np.ndarray) -> SimpleNamespace:
        self.inputs.append(observations.copy())
        action = int(observations[0, 1] < 0)
        observations[...] = -999.0
        return SimpleNamespace(deterministic_actions=np.asarray([action], dtype=np.int64))


class _MismatchedResetEnv(_FakeEnv):
    reset_count = 0

    def reset(self, *, seed: int) -> tuple[np.ndarray, dict[str, object]]:
        type(self).reset_count += 1
        self.current_seed = seed
        self.step_number = 0
        # Vary only a policy-irrelevant coordinate to prove that the paired
        # reset invariant is checked independently of the chosen target/action.
        self.returned = np.asarray(
            [float(type(self).reset_count), 10.0, 20.0], dtype=np.float32
        )
        return self.returned, {"seed": seed}


def test_replacement_returns_an_independent_policy_input() -> None:
    observation = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    reference = np.asarray([4.0, 5.0, 6.0], dtype=np.float32)

    changed = replace_policy_observation(observation, (1,), reference)

    assert np.array_equal(changed, [1.0, 5.0, 3.0])
    assert np.array_equal(observation, [1.0, 2.0, 3.0])
    assert np.array_equal(reference, [4.0, 5.0, 6.0])
    assert not np.shares_memory(changed, observation)
    assert not np.shares_memory(changed, reference)


def test_paired_closed_loop_preserves_env_observations_actions_and_seeds() -> None:
    _FakeEnv.instances.clear()
    adapter = _MutatingAdapter()

    result = run_paired_closed_loop(
        env_factory=_FakeEnv,
        adapter=adapter,
        scenario_seeds=[17],
        targets=[ClosedLoopTarget(name="state_1", indices=(1,))],
        replacement_strategy="episode_start_constant",
        rl_seed=123,
    )

    assert len(result.records) == 1
    record = result.records[0]
    assert record["scenario_seed"] == 17
    assert record["target_indices"] == [1]
    assert record["baseline"]["policy_input_was_copied"] is True
    assert record["intervention"]["policy_input_was_copied"] is True
    assert record["baseline"]["total_reward"] == 0.0
    assert record["intervention"]["total_reward"] == 0.0
    assert record["delta"]["total_reward"] == 0.0
    assert record["delta"]["success"] == 0.0
    assert result.summary_rows[0]["mean_baseline_success"] == 0.0
    assert result.summary_rows[0]["mean_intervention_success"] == 0.0
    assert result.summary_rows[0]["mean_delta_success"] == 0.0
    assert result.summary_rows[0]["mean_baseline_arrive_dest"] == 0.0
    assert result.summary_rows[0]["baseline_termination_reason_counts_json"] == (
        '{"other_termination": 1}'
    )
    assert result.summary_rows[0]["mean_baseline_target_lane_target_lane_valid"] == 1.0
    assert result.summary_rows[0]["mean_delta_target_lane_time_in_target_lane_ratio"] == 0.0

    # One probe plus baseline/intervention environments.  The paired two each
    # receive the same scenario reset and same RL-space seeds.
    paired = [env for env in _FakeEnv.instances if env.current_seed == 17]
    assert len(paired) == 2
    for env in paired:
        assert env.action_space.seeds == [123]
        assert env.observation_space.seeds == [123]
        assert env.closed is True
        assert env.received_actions == [0, 0]
        assert np.array_equal(env.returned_before_policy[0], [17.0, 10.0, 20.0])
        assert np.array_equal(env.returned_before_policy[1], [17.0, 11.0, 20.0])

    # The mutating adapter sees copies; none of the internal env buffers has
    # been overwritten with its sentinel value.
    assert adapter.inputs
    assert all(not np.any(env.returned_before_policy[0] == -999) for env in paired)


def test_paired_closed_loop_rejects_unequal_reset_observations() -> None:
    _MismatchedResetEnv.reset_count = 0

    with pytest.raises(ClosedLoopError, match="reset observations differ"):
        run_paired_closed_loop(
            env_factory=_MismatchedResetEnv,
            adapter=_MutatingAdapter(),
            scenario_seeds=[7],
            targets=[ClosedLoopTarget(name="state_1", indices=(1,))],
            replacement_strategy="episode_start_constant",
            rl_seed=123,
        )


def test_schema_group_targets_use_full_group_membership_not_primary_display_group() -> None:
    schema = load_observation_schema("observation_schemas/metadrive_default_259.toml")

    targets = schema_targets(
        schema,
        group_names=("road_boundaries", "heading", "checkpoint_1"),
    )

    assert {target.name: target.indices for target in targets} == {
        "road_boundaries": (0, 1),
        "heading": (2,),
        "checkpoint_1": (9, 10, 11, 12, 13),
    }
