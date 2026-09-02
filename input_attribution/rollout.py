"""Collection and durable storage of pre-action policy rollouts.

The collector intentionally stores the observation *before* ``env.step`` and
uses a new float32 copy for every policy decision.  This makes offline
perturbation and Integrated Gradients reproducible without starting MetaDrive
again, while preventing analysis code from accidentally changing an
environment-owned observation buffer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from evaluation_visualization import (
    ACTION_HISTORY_SECONDS,
    ActionSwitchTracker,
    decode_discrete_action,
    derive_timing,
    make_step_telemetry,
    read_runtime_road_metrics,
)

from .results import (
    ArtifactError,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_npz,
    json_value,
    safe_child,
)


class RolloutError(ValueError):
    """Raised when the supported single-agent rollout contract is violated."""


@dataclass(frozen=True, slots=True)
class RolloutData:
    """Arrays and scalar records for one or more ordinary policy episodes."""

    observations: np.ndarray
    logits: np.ndarray
    log_probabilities: np.ndarray
    probabilities: np.ndarray
    values: np.ndarray
    selected_actions: np.ndarray
    env_actions: np.ndarray
    episode_ids: np.ndarray
    scenario_seeds: np.ndarray
    steps: np.ndarray
    rewards: np.ndarray
    cumulative_rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    dones: np.ndarray
    step_records: tuple[dict[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def observation_dim(self) -> int:
        if self.observations.ndim != 2:
            raise RolloutError(
                f"rollout observations must be two-dimensional, got {self.observations.shape}"
            )
        return int(self.observations.shape[1])

    @property
    def row_count(self) -> int:
        return int(self.observations.shape[0])

    def validate(self) -> None:
        """Check that every per-decision field aligns with observations."""

        rows = self.row_count
        if self.logits.ndim != 2:
            raise RolloutError(f"logits must be N×A, got {self.logits.shape}")
        if self.log_probabilities.shape != self.logits.shape:
            raise RolloutError(
                "log_probabilities shape must match logits: "
                f"{self.log_probabilities.shape} != {self.logits.shape}"
            )
        if self.probabilities.shape != self.logits.shape:
            raise RolloutError(
                "probabilities shape must match logits: "
                f"{self.probabilities.shape} != {self.logits.shape}"
            )
        arrays = {
            "values": self.values,
            "selected_actions": self.selected_actions,
            "env_actions": self.env_actions,
            "episode_ids": self.episode_ids,
            "scenario_seeds": self.scenario_seeds,
            "steps": self.steps,
            "rewards": self.rewards,
            "cumulative_rewards": self.cumulative_rewards,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "dones": self.dones,
        }
        if self.logits.shape[0] != rows:
            raise RolloutError("logits row count must match observations")
        for name, array in arrays.items():
            if np.asarray(array).reshape(-1).shape[0] != rows:
                raise RolloutError(
                    f"{name} row count must match observations: "
                    f"{np.asarray(array).shape} vs {rows}"
                )
        if len(self.step_records) not in {0, rows}:
            raise RolloutError(
                "step_records must be absent or contain one record per observation"
            )


def _as_flat_observation(value: Any, *, expected_dim: int | None = None) -> np.ndarray:
    observation = np.asarray(value, dtype=np.float32)
    if observation.ndim != 1:
        raise RolloutError(
            "only a single flat one-dimensional Box observation is supported; "
            f"got reset observation shape={observation.shape}"
        )
    if expected_dim is not None and observation.shape[0] != expected_dim:
        raise RolloutError(
            f"reset observation dimension mismatch: expected {expected_dim}, "
            f"got {observation.shape[0]}"
        )
    return observation.copy()


def _flat_space_dimension(space: Any, label: str) -> int:
    shape = getattr(space, "shape", None)
    if shape is None:
        raise RolloutError(f"{label} must be a flat Box-like space with shape, got {space!r}")
    shape_tuple = tuple(shape)
    if len(shape_tuple) != 1 or not isinstance(shape_tuple[0], (int, np.integer)):
        raise RolloutError(
            f"{label} must have one flat dimension, got shape={shape_tuple!r}"
        )
    dimension = int(shape_tuple[0])
    if dimension <= 0:
        raise RolloutError(f"{label} dimension must be positive, got {dimension}")
    return dimension


def _discrete_dimension(space: Any, label: str) -> int:
    action_count = getattr(space, "n", None)
    if isinstance(action_count, bool) or not isinstance(action_count, (int, np.integer)):
        raise RolloutError(
            f"{label} must be a single Discrete action space; got {space!r}"
        )
    if int(action_count) <= 0:
        raise RolloutError(f"{label} has invalid Discrete size: {action_count!r}")
    return int(action_count)


def _schema_dimension(schema: Any) -> int | None:
    for name in ("observation_dim", "dimension", "dim"):
        value = getattr(schema, name, None)
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            return int(value)
    if isinstance(schema, Mapping):
        value = schema.get("observation_dim")
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            return int(value)
    return None


def validate_runtime_contract(
    *,
    env: Any,
    model: Any | None,
    schema: Any | None,
    reset_observation: Any | None = None,
) -> dict[str, int]:
    """Fail closed unless env/model/schema all describe the same flat input.

    The caller may pass a reset observation to additionally verify the runtime
    object.  This helper is deliberately independent from MetaDrive classes so
    it can be unit-tested with a tiny fake environment.
    """

    env_observation_dim = _flat_space_dimension(
        getattr(env, "observation_space", None), "env.observation_space"
    )
    env_action_dim = _discrete_dimension(
        getattr(env, "action_space", None), "env.action_space"
    )
    values: dict[str, int] = {
        "env_observation_dim": env_observation_dim,
        "env_action_dim": env_action_dim,
    }
    if model is not None:
        model_observation_dim = _flat_space_dimension(
            getattr(model, "observation_space", None), "model.observation_space"
        )
        model_action_dim = _discrete_dimension(
            getattr(model, "action_space", None), "model.action_space"
        )
        values["model_observation_dim"] = model_observation_dim
        values["model_action_dim"] = model_action_dim
        if model_observation_dim != env_observation_dim:
            raise RolloutError(
                "observation dimensions disagree: "
                f"env={env_observation_dim}, model={model_observation_dim}"
            )
        if model_action_dim != env_action_dim:
            raise RolloutError(
                f"action dimensions disagree: env={env_action_dim}, model={model_action_dim}"
            )
    schema_dim = _schema_dimension(schema) if schema is not None else None
    if schema_dim is not None:
        values["schema_observation_dim"] = schema_dim
        if schema_dim != env_observation_dim:
            raise RolloutError(
                "observation dimensions disagree: "
                f"env={env_observation_dim}, schema={schema_dim}"
            )
    if reset_observation is not None:
        reset_dim = _as_flat_observation(
            reset_observation, expected_dim=env_observation_dim
        ).shape[0]
        values["reset_observation_dim"] = int(reset_dim)
    return values


def _normalise_policy_vector(value: Any, *, name: str, action_dim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array[np.newaxis, :]
    if array.ndim != 2 or array.shape[0] != 1 or array.shape[1] != action_dim:
        raise RolloutError(
            f"adapter {name} must be a 1×{action_dim} batch, got {array.shape}"
        )
    return array[0].copy()


def _policy_field(output: Any, *names: str) -> Any:
    for name in names:
        if isinstance(output, Mapping) and name in output:
            return output[name]
        if hasattr(output, name):
            return getattr(output, name)
    joined = ", ".join(names)
    raise RolloutError(f"policy adapter output is missing required field: {joined}")


def _evaluate_action(
    adapter: Any,
    observation: np.ndarray,
    *,
    action_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    # Adapter implementations are expected to be pure.  Pass a second copy
    # nevertheless so a third-party/debug adapter cannot corrupt the saved
    # pre-action observation used for later offline analysis.
    output = adapter.evaluate(observation[np.newaxis, :].copy())
    logits = _normalise_policy_vector(
        _policy_field(output, "logits", "raw_logits"),
        name="logits",
        action_dim=action_dim,
    )
    log_probabilities = _normalise_policy_vector(
        _policy_field(output, "log_probabilities", "log_probs"),
        name="log_probabilities",
        action_dim=action_dim,
    )
    probabilities = _normalise_policy_vector(
        _policy_field(output, "probabilities", "probs"),
        name="probabilities",
        action_dim=action_dim,
    )
    values = np.asarray(_policy_field(output, "values", "value"), dtype=np.float32)
    if values.size != 1:
        raise RolloutError(f"adapter values must contain one value, got shape={values.shape}")
    action_values = np.asarray(
        _policy_field(output, "deterministic_actions", "deterministic_action", "actions")
    )
    if action_values.size != 1:
        raise RolloutError(
            "adapter deterministic action must contain one single-agent action, "
            f"got shape={action_values.shape}"
        )
    action = int(action_values.reshape(-1)[0])
    if not 0 <= action < action_dim:
        raise RolloutError(
            f"adapter action {action} is outside environment Discrete({action_dim})"
        )
    return logits, log_probabilities, probabilities, float(values.reshape(-1)[0]), action


def _reset(env: Any, scenario_seed: int) -> tuple[np.ndarray, Mapping[str, Any]]:
    result = env.reset(seed=scenario_seed)
    if isinstance(result, tuple) and len(result) == 2:
        observation, info = result
    else:  # Gym legacy compatibility; MetaDrive currently returns the new API.
        observation, info = result, {}
    if not isinstance(info, Mapping):
        raise RolloutError(f"env.reset info must be a mapping, got {type(info)!r}")
    return np.asarray(observation, dtype=np.float32), info


def _step(env: Any, action: int) -> tuple[np.ndarray, float, bool, bool, Mapping[str, Any]]:
    result = env.step(action)
    if not isinstance(result, tuple):
        raise RolloutError("env.step must return a tuple")
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
    elif len(result) == 4:  # legacy Gym makes a truncated distinction unavailable.
        observation, reward, done, info = result
        terminated, truncated = bool(done), False
    else:
        raise RolloutError(f"env.step returned {len(result)} values instead of 5")
    if not isinstance(info, Mapping):
        raise RolloutError(f"env.step info must be a mapping, got {type(info)!r}")
    return (
        np.asarray(observation, dtype=np.float32),
        float(reward),
        bool(terminated),
        bool(truncated),
        info,
    )


def _action_id(action: int, env: Any) -> int:
    config = getattr(env, "config", None)
    if isinstance(config, Mapping):
        try:
            return decode_discrete_action(action, config).action_id
        except (KeyError, TypeError, ValueError):
            pass
    return int(action)


def _runtime_timing(env: Any) -> tuple[Any | None, int]:
    config = getattr(env, "config", None)
    if not isinstance(config, Mapping):
        return None, 1
    try:
        timing = derive_timing(config)
    except (KeyError, TypeError, ValueError):
        return None, 1
    history = max(1, math.ceil(ACTION_HISTORY_SECONDS * timing.control_hz))
    return timing, history


def _optional_info(info: Mapping[str, Any], key: str) -> Any:
    value = info.get(key)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return None


def _scalar_step_record(
    *,
    episode: int,
    scenario_seed: int,
    step: int,
    action: int,
    value: float,
    reward: float,
    cumulative_reward: float,
    terminated: bool,
    truncated: bool,
    info: Mapping[str, Any],
    switch_count: int,
    switch_rate: float | None,
) -> dict[str, Any]:
    return {
        "episode": episode,
        "scenario_seed": scenario_seed,
        "step": step,
        "selected_action": action,
        "env_action": action,
        "critic_value": value,
        "reward": reward,
        "cumulative_reward": cumulative_reward,
        "terminated": terminated,
        "truncated": truncated,
        "done": bool(terminated or truncated),
        "action_switch_count": switch_count,
        "action_switches_per_second": switch_rate,
        "route_completion": _optional_info(info, "route_completion"),
        "out_of_road": _optional_info(info, "out_of_road"),
        "crash": _optional_info(info, "crash"),
        "arrive_dest": _optional_info(info, "arrive_dest"),
        "target_lane_valid": _optional_info(info, "target_lane_valid"),
        "target_lane_ordinal": _optional_info(info, "target_lane_ordinal"),
    }


def collect_rollout(
    *,
    env: Any,
    adapter: Any,
    model: Any | None,
    schema: Any | None,
    scenario_start: int,
    scenario_count: int,
    deterministic: bool = True,
    rl_seed: int | None = None,
) -> RolloutData:
    """Run ordinary deterministic policy episodes and retain pre-step inputs.

    ``deterministic=False`` is intentionally rejected for now: the adapter's
    public result is a policy-mode action and recording a second random action
    would make the offline actor outputs no longer describe the driven path.
    """

    if not deterministic:
        raise RolloutError(
            "only deterministic policy rollout collection is supported by the initial analyzer"
        )
    if isinstance(scenario_count, bool) or not isinstance(scenario_count, int) or scenario_count <= 0:
        raise RolloutError(f"scenario_count must be a positive integer, got {scenario_count!r}")
    if isinstance(scenario_start, bool) or not isinstance(scenario_start, int):
        raise RolloutError(f"scenario_start must be an integer, got {scenario_start!r}")

    contract = validate_runtime_contract(env=env, model=model, schema=schema)
    observation_dim = contract["env_observation_dim"]
    action_dim = contract["env_action_dim"]
    timing, history_length = _runtime_timing(env)

    observations: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    log_probabilities: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    values: list[float] = []
    selected_actions: list[int] = []
    env_actions: list[int] = []
    episode_ids: list[int] = []
    scenario_seeds: list[int] = []
    steps: list[int] = []
    rewards: list[float] = []
    cumulative_rewards: list[float] = []
    terminated_values: list[bool] = []
    truncated_values: list[bool] = []
    done_values: list[bool] = []
    records: list[dict[str, Any]] = []
    episode_start_observations: list[list[float]] = []

    for episode in range(1, scenario_count + 1):
        requested_seed = scenario_start + episode - 1
        observation, _reset_info = _reset(env, requested_seed)
        actual_seed_value = getattr(env, "current_seed", None)
        if actual_seed_value is not None and int(actual_seed_value) != requested_seed:
            raise RolloutError(
                "requested scenario seed does not match env.current_seed: "
                f"requested={requested_seed}, actual={actual_seed_value}"
            )
        observation = _as_flat_observation(observation, expected_dim=observation_dim)
        # Preserve explicit evidence that the runtime object returned by
        # reset(), not only the declared observation space, matched the model
        # and schema dimension contract.
        contract["reset_observation_dim"] = int(observation.shape[0])
        episode_start_observations.append(observation.astype(float).tolist())
        cumulative_reward = 0.0
        decision_step = 0
        switch_tracker = ActionSwitchTracker(history_length=history_length)

        while True:
            obs_before = observation.astype(np.float32, copy=True)
            policy_logits, policy_log_probs, policy_probs, value, action = _evaluate_action(
                adapter, obs_before, action_dim=action_dim
            )
            next_observation, reward, terminated, truncated, info = _step(env, action)
            decision_step += 1
            cumulative_reward += reward
            action_id = _action_id(action, env)
            if timing is not None:
                switch_count, switch_rate = switch_tracker.observe(
                    action_id,
                    sim_time_seconds=decision_step * timing.action_duration_seconds,
                )
            else:
                # ActionSwitchTracker only needs a positive time to report a
                # rate.  Without known timing we retain the count and leave
                # the physical-unit rate explicitly unavailable.
                if (
                    switch_tracker.previous_action_id is not None
                    and action_id != switch_tracker.previous_action_id
                ):
                    switch_tracker.switch_count += 1
                switch_tracker.previous_action_id = action_id
                switch_tracker.history.append(action_id)
                switch_count, switch_rate = switch_tracker.switch_count, None

            scalar_record = _scalar_step_record(
                episode=episode,
                scenario_seed=requested_seed,
                step=decision_step,
                action=action,
                value=value,
                reward=reward,
                cumulative_reward=cumulative_reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
                switch_count=switch_count,
                switch_rate=switch_rate,
            )
            if timing is not None and "velocity" in info:
                try:
                    decoded = decode_discrete_action(action, getattr(env, "config", {}))
                    telemetry = make_step_telemetry(
                        episode=episode,
                        scenario_seed=requested_seed,
                        step=decision_step,
                        horizon=(
                            None
                            if getattr(env, "config", {}).get("horizon") is None
                            else int(getattr(env, "config", {})["horizon"])
                        ),
                        timing=timing,
                        decoded_action=decoded,
                        info=info,
                        reward=reward,
                        cumulative_reward=cumulative_reward,
                        terminated=terminated,
                        truncated=truncated,
                        road=read_runtime_road_metrics(getattr(env, "agent", None)),
                        action_switch_count=switch_count,
                        action_switches_per_second=(
                            0.0 if switch_rate is None else switch_rate
                        ),
                    )
                    telemetry.update(scalar_record)
                    scalar_record = telemetry
                except (AttributeError, KeyError, TypeError, ValueError):
                    # Telemetry is supplemental; observations and policy
                    # outputs remain the primary reproducible data.
                    pass

            observations.append(obs_before)
            logits.append(policy_logits)
            log_probabilities.append(policy_log_probs)
            probabilities.append(policy_probs)
            values.append(value)
            selected_actions.append(action)
            env_actions.append(action)
            episode_ids.append(episode)
            scenario_seeds.append(requested_seed)
            steps.append(decision_step)
            rewards.append(reward)
            cumulative_rewards.append(cumulative_reward)
            terminated_values.append(terminated)
            truncated_values.append(truncated)
            done_values.append(bool(terminated or truncated))
            records.append(json_value(scalar_record))

            observation = _as_flat_observation(next_observation, expected_dim=observation_dim)
            if terminated or truncated:
                break

    row_count = len(observations)
    if row_count == 0:
        raise RolloutError("environment produced no policy decisions")
    rollout = RolloutData(
        observations=np.stack(observations).astype(np.float32, copy=False),
        logits=np.stack(logits).astype(np.float32, copy=False),
        log_probabilities=np.stack(log_probabilities).astype(np.float32, copy=False),
        probabilities=np.stack(probabilities).astype(np.float32, copy=False),
        values=np.asarray(values, dtype=np.float32),
        selected_actions=np.asarray(selected_actions, dtype=np.int64),
        env_actions=np.asarray(env_actions, dtype=np.int64),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        scenario_seeds=np.asarray(scenario_seeds, dtype=np.int64),
        steps=np.asarray(steps, dtype=np.int64),
        rewards=np.asarray(rewards, dtype=np.float32),
        cumulative_rewards=np.asarray(cumulative_rewards, dtype=np.float32),
        terminated=np.asarray(terminated_values, dtype=np.bool_),
        truncated=np.asarray(truncated_values, dtype=np.bool_),
        dones=np.asarray(done_values, dtype=np.bool_),
        step_records=tuple(records),
        metadata={
            "record_semantics": "pre-action observation; one row per policy decision",
            "deterministic": True,
            "rl_seed": rl_seed,
            "scenario_start": scenario_start,
            "scenario_count": scenario_count,
            "row_count": row_count,
            "observation_dim": observation_dim,
            "action_dim": action_dim,
            "episode_start_observations": episode_start_observations,
            "runtime_contract": contract,
        },
    )
    rollout.validate()
    return rollout


def save_rollout(
    run_directory: Path,
    rollout: RolloutData,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Persist the standard rollout NPZ, scalar JSONL, and metadata JSON."""

    rollout.validate()
    arrays_path = safe_child(run_directory, "rollout_arrays.npz")
    steps_path = safe_child(run_directory, "rollout_steps.jsonl")
    metadata_path = safe_child(run_directory, "rollout_metadata.json")
    atomic_write_npz(
        arrays_path,
        observations=rollout.observations,
        logits=rollout.logits,
        log_probabilities=rollout.log_probabilities,
        log_probs=rollout.log_probabilities,
        probabilities=rollout.probabilities,
        values=rollout.values,
        selected_actions=rollout.selected_actions,
        env_actions=rollout.env_actions,
        episode_ids=rollout.episode_ids,
        scenario_seeds=rollout.scenario_seeds,
        steps=rollout.steps,
        rewards=rollout.rewards,
        cumulative_rewards=rollout.cumulative_rewards,
        terminated=rollout.terminated,
        truncated=rollout.truncated,
        dones=rollout.dones,
    )
    row_count = atomic_write_jsonl(steps_path, rollout.step_records)
    combined_metadata = dict(rollout.metadata)
    combined_metadata.update(dict(metadata or {}))
    combined_metadata["step_record_count"] = row_count
    atomic_write_json(metadata_path, combined_metadata)
    return {
        "arrays": arrays_path,
        "steps": steps_path,
        "metadata": metadata_path,
    }


def _load_required(archive: Any, *names: str) -> np.ndarray:
    for name in names:
        if name in archive:
            return np.asarray(archive[name])
    raise RolloutError(f"rollout archive is missing required array: {' or '.join(names)}")


def load_rollout(run_directory: Path) -> RolloutData:
    """Load an offline rollout from its standard three artifact files."""

    if run_directory.is_symlink() or not run_directory.is_dir():
        raise RolloutError(f"rollout directory must be a regular directory: {run_directory}")
    arrays_path = run_directory / "rollout_arrays.npz"
    steps_path = run_directory / "rollout_steps.jsonl"
    metadata_path = run_directory / "rollout_metadata.json"
    for path in (arrays_path, steps_path, metadata_path):
        if path.is_symlink() or not path.is_file():
            raise RolloutError(f"rollout artifact is missing or unsafe: {path}")
    try:
        with np.load(arrays_path, allow_pickle=False) as archive:
            rollout = RolloutData(
                observations=np.asarray(_load_required(archive, "observations"), dtype=np.float32),
                logits=np.asarray(_load_required(archive, "logits"), dtype=np.float32),
                log_probabilities=np.asarray(
                    _load_required(archive, "log_probabilities", "log_probs"),
                    dtype=np.float32,
                ),
                probabilities=np.asarray(_load_required(archive, "probabilities"), dtype=np.float32),
                values=np.asarray(_load_required(archive, "values"), dtype=np.float32),
                selected_actions=np.asarray(
                    _load_required(archive, "selected_actions"), dtype=np.int64
                ),
                env_actions=np.asarray(_load_required(archive, "env_actions"), dtype=np.int64),
                episode_ids=np.asarray(_load_required(archive, "episode_ids"), dtype=np.int64),
                scenario_seeds=np.asarray(
                    _load_required(archive, "scenario_seeds"), dtype=np.int64
                ),
                steps=np.asarray(_load_required(archive, "steps"), dtype=np.int64),
                rewards=np.asarray(_load_required(archive, "rewards"), dtype=np.float32),
                cumulative_rewards=np.asarray(
                    _load_required(archive, "cumulative_rewards"), dtype=np.float32
                ),
                terminated=np.asarray(_load_required(archive, "terminated"), dtype=np.bool_),
                truncated=np.asarray(_load_required(archive, "truncated"), dtype=np.bool_),
                dones=np.asarray(_load_required(archive, "dones", "done"), dtype=np.bool_),
            )
    except (OSError, ValueError) as error:
        raise RolloutError(f"could not read rollout archive {arrays_path}: {error}") from error

    records: list[dict[str, Any]] = []
    try:
        with steps_path.open("r", encoding="utf-8") as file_obj:
            for line_number, line in enumerate(file_obj, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise RolloutError(
                        f"rollout JSONL row {line_number} must be an object"
                    )
                records.append(item)
        with metadata_path.open("r", encoding="utf-8") as file_obj:
            metadata = json.load(file_obj)
    except (OSError, json.JSONDecodeError) as error:
        raise RolloutError(f"could not read rollout metadata: {error}") from error
    if not isinstance(metadata, Mapping):
        raise RolloutError("rollout metadata JSON must contain an object")
    rollout = RolloutData(
        **{
            **asdict(rollout),
            "step_records": tuple(records),
            "metadata": dict(metadata),
        }
    )
    rollout.validate()
    return rollout
