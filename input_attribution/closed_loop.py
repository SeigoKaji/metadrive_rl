"""Paired closed-loop input replacement for a fixed policy.

This module never wraps or mutates MetaDrive itself.  It changes only a fresh
copy of the observation passed to the policy, then gives the resulting action
to the environment unchanged.  Every intervention is compared to a separately
reset ordinary rollout using the same scenario and RL-space seeds.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
import json
import math
from pathlib import Path
import statistics
from typing import Any

import numpy as np

from evaluation_visualization import ACTION_HISTORY_SECONDS, ActionSwitchTracker, derive_timing

from .results import atomic_write_csv, atomic_write_jsonl, expanded_schema_rows, safe_child


class ClosedLoopError(ValueError):
    """Raised when a paired intervention would not preserve its invariants."""


@dataclass(frozen=True, slots=True)
class ClosedLoopTarget:
    """A named set of policy-input dimensions to replace together."""

    name: str
    indices: tuple[int, ...]
    kind: str = "feature"


@dataclass(frozen=True, slots=True)
class ClosedLoopResult:
    """Per-scenario paired outcomes and their per-target aggregate."""

    records: tuple[dict[str, Any], ...]
    summary_rows: tuple[dict[str, Any], ...]
    replacement_strategy: str


def replace_policy_observation(
    observation: Any,
    indices: Sequence[int],
    replacement: Any,
) -> np.ndarray:
    """Return a changed policy-input copy while leaving both inputs untouched."""

    source = np.asarray(observation, dtype=np.float32)
    reference = np.asarray(replacement, dtype=np.float32)
    if source.ndim != 1 or reference.ndim != 1 or source.shape != reference.shape:
        raise ClosedLoopError(
            "observation and replacement must be same-shape one-dimensional arrays: "
            f"observation={source.shape}, replacement={reference.shape}"
        )
    normalised = tuple(int(index) for index in indices)
    if not normalised:
        raise ClosedLoopError("closed-loop target must contain at least one index")
    if len(set(normalised)) != len(normalised):
        raise ClosedLoopError(f"closed-loop target has duplicate indices: {normalised}")
    if any(index < 0 or index >= source.size for index in normalised):
        raise ClosedLoopError(
            f"closed-loop indices are outside observation dimension {source.size}: {normalised}"
        )
    result = source.copy()
    result[list(normalised)] = reference[list(normalised)]
    return result


# An explicit alias makes the copy-only contract easy to find from external
# callers and tests without exposing a Gym wrapper.
replace_policy_input = replace_policy_observation


def _target_value(target: Any, name: str, default: Any = None) -> Any:
    if isinstance(target, Mapping):
        return target.get(name, default)
    return getattr(target, name, default)


def normalize_targets(
    targets: Iterable[Any],
    *,
    observation_dim: int,
) -> tuple[ClosedLoopTarget, ...]:
    """Normalize core aggregation/perturbation targets without mutating them."""

    normalised: list[ClosedLoopTarget] = []
    seen_names: set[str] = set()
    for position, target in enumerate(targets):
        name_value = _target_value(target, "name", None)
        if name_value is None:
            name_value = _target_value(target, "label", f"target_{position}")
        name = str(name_value)
        if not name or name in seen_names:
            raise ClosedLoopError(f"closed-loop targets need unique non-empty names: {name!r}")
        indices_value = _target_value(target, "indices", None)
        if indices_value is None:
            indices_value = _target_value(target, "feature_indices", None)
        if indices_value is None:
            raise ClosedLoopError(f"closed-loop target {name!r} does not define indices")
        indices = tuple(int(index) for index in indices_value)
        if not indices or len(indices) != len(set(indices)):
            raise ClosedLoopError(f"closed-loop target {name!r} has invalid indices: {indices}")
        if any(index < 0 or index >= observation_dim for index in indices):
            raise ClosedLoopError(
                f"closed-loop target {name!r} is outside observation dimension "
                f"{observation_dim}: {indices}"
            )
        kind = str(_target_value(target, "kind", "feature"))
        normalised.append(ClosedLoopTarget(name=name, indices=indices, kind=kind))
        seen_names.add(name)
    if not normalised:
        raise ClosedLoopError("at least one closed-loop feature or group must be selected")
    return tuple(normalised)


def schema_targets(
    schema: Any,
    *,
    feature_indices: Iterable[int] = (),
    group_names: Iterable[str] = (),
) -> tuple[ClosedLoopTarget, ...]:
    """Build explicit targets from schema feature indices and semantic groups."""

    rows = expanded_schema_rows(schema)
    by_index = {int(row["index"]): row for row in rows}
    targets: list[ClosedLoopTarget] = []
    for index in feature_indices:
        index = int(index)
        try:
            row = by_index[index]
        except KeyError as error:
            raise ClosedLoopError(f"schema has no feature index {index}") from error
        targets.append(
            ClosedLoopTarget(name=str(row["name"]), indices=(index,), kind="feature")
        )
    for group in group_names:
        # A feature can belong to several root groups (for example ego_state
        # and road_boundaries).  The schema's group mapping is authoritative;
        # the flattened CSV's primary group is only a display convenience.
        try:
            getter = getattr(schema, "indices_for_group", None)
            if callable(getter):
                indices = tuple(int(index) for index in getter(str(group)))
            elif isinstance(getattr(schema, "groups", None), Mapping):
                indices = tuple(
                    int(index) for index in getattr(schema, "groups")[str(group)]
                )
            else:
                indices = tuple(
                    int(row["index"])
                    for row in rows
                    if str(row.get("group")) == str(group)
                )
        except KeyError as error:
            raise ClosedLoopError(f"schema has no group {group!r}") from error
        if not indices:
            raise ClosedLoopError(f"schema has no group {group!r}")
        targets.append(ClosedLoopTarget(name=str(group), indices=indices, kind="group"))
    return normalize_targets(targets, observation_dim=len(rows))


def _as_reference(value: Any, observation_dim: int, *, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 2:
        if array.shape[0] != 1:
            raise ClosedLoopError(f"{label} must contain exactly one reference row, got {array.shape}")
        array = array[0]
    if array.ndim != 1 or array.shape[0] != observation_dim:
        raise ClosedLoopError(
            f"{label} must be a {observation_dim}-dimensional observation, got {array.shape}"
        )
    return array.copy()


def _schema_constant_vector(schema: Any, observation_dim: int) -> np.ndarray:
    """Read only schema-declared constants; unknown entries stay NaN."""

    constants = np.full(observation_dim, np.nan, dtype=np.float32)
    for row in expanded_schema_rows(schema):
        index = int(row["index"])
        value = row.get("constant")
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise ClosedLoopError(
                f"schema constant for feature {row['name']!r} must be numeric"
            )
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ClosedLoopError(
                f"schema constant for feature {row['name']!r} must be finite"
            )
        constants[index] = numeric
    return constants


def _reference_for_episode(
    *,
    strategy: str,
    episode_start: np.ndarray,
    observation_dim: int,
    specified_reference: Any | None,
    dataset_observations: Any | None,
    schema: Any | None,
) -> np.ndarray:
    if strategy == "episode_start_constant":
        return episode_start.copy()
    if strategy == "specified_reference_constant":
        if specified_reference is None:
            raise ClosedLoopError(
                "specified_reference_constant requires a specified_reference observation"
            )
        return _as_reference(
            specified_reference, observation_dim, label="specified_reference"
        )
    if strategy == "dataset_median_constant":
        if dataset_observations is None:
            raise ClosedLoopError(
                "dataset_median_constant requires dataset_observations from a saved rollout"
            )
        dataset = np.asarray(dataset_observations, dtype=np.float32)
        if dataset.ndim != 2 or dataset.shape[1] != observation_dim or dataset.shape[0] == 0:
            raise ClosedLoopError(
                "dataset_observations must be a non-empty N×D observation matrix; "
                f"got {dataset.shape}"
            )
        return np.median(dataset, axis=0).astype(np.float32, copy=False)
    if strategy == "schema_constant":
        if schema is None:
            raise ClosedLoopError("schema_constant requires an observation schema")
        return _schema_constant_vector(schema, observation_dim)
    raise ClosedLoopError(
        "unknown closed-loop replacement strategy; choose one of "
        "episode_start_constant, specified_reference_constant, "
        "dataset_median_constant, schema_constant"
    )


def _single_action(adapter: Any, policy_input: np.ndarray, action_dim: int) -> int:
    output = adapter.evaluate(policy_input[np.newaxis, :])
    for name in ("deterministic_actions", "deterministic_action", "actions"):
        value = output.get(name) if isinstance(output, Mapping) else getattr(output, name, None)
        if value is not None:
            action_values = np.asarray(value)
            break
    else:
        raise ClosedLoopError("policy adapter output is missing deterministic action")
    if action_values.size != 1:
        raise ClosedLoopError(
            f"policy adapter must return one single-agent action, got {action_values.shape}"
        )
    action = int(action_values.reshape(-1)[0])
    if not 0 <= action < action_dim:
        raise ClosedLoopError(f"policy action {action} outside Discrete({action_dim})")
    return action


def _seed_spaces(env: Any, rl_seed: int | None) -> None:
    """Apply the same RL-space seed before each side of a paired comparison."""

    if rl_seed is None:
        return
    for space in (getattr(env, "action_space", None), getattr(env, "observation_space", None)):
        seed_method = getattr(space, "seed", None)
        if callable(seed_method):
            seed_method(int(rl_seed))


def _termination_reason(
    *, terminated: bool, truncated: bool, info: Mapping[str, Any]
) -> str:
    if bool(info.get("arrive_dest", False)):
        return "success"
    if bool(info.get("out_of_road", False)):
        return "out_of_road"
    if bool(info.get("crash_vehicle", False)):
        return "crash_vehicle"
    if bool(info.get("crash_object", False)):
        return "crash_object"
    if bool(info.get("crash", False)):
        return "crash"
    if bool(info.get("start_lane_departure", False)):
        return "start_lane_departure"
    if bool(info.get("wrong_lane_arrival", False)):
        return "wrong_lane_arrival"
    if truncated or bool(info.get("max_step", False)):
        return "max_step_truncation"
    if terminated:
        return "other_termination"
    return "not_terminated"


def _optional_number(value: Any) -> float | None:
    # Episode outcome flags are aggregated as 0/1 rates and their paired
    # differences as indicator deltas.  Keeping bool support here ensures
    # success/out-of-road/crash are present in all-scenario summaries.
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if not isinstance(value, (int, float, np.number)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _episode_outcome(
    *,
    env: Any,
    adapter: Any,
    scenario_seed: int,
    replacement: np.ndarray | None,
    target: ClosedLoopTarget | None,
    rl_seed: int | None,
    initial_reset: tuple[Any, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run one fresh episode, altering only a policy-input copy when asked."""

    action_space = getattr(env, "action_space", None)
    observation_space = getattr(env, "observation_space", None)
    action_dim_value = getattr(action_space, "n", None)
    observation_shape = getattr(observation_space, "shape", None)
    if not isinstance(action_dim_value, (int, np.integer)) or isinstance(action_dim_value, bool):
        raise ClosedLoopError("closed-loop mode requires a single Discrete action space")
    if tuple(observation_shape or ())[:1] != tuple(observation_shape or ()) or len(tuple(observation_shape or ())) != 1:
        raise ClosedLoopError("closed-loop mode requires a flat one-dimensional observation space")
    action_dim = int(action_dim_value)
    observation_dim = int(tuple(observation_shape)[0])
    if initial_reset is None:
        _seed_spaces(env, rl_seed)
        reset = env.reset(seed=scenario_seed)
        if isinstance(reset, tuple) and len(reset) == 2:
            observation, reset_info = reset
        else:
            observation, reset_info = reset, {}
    else:
        observation, reset_info = initial_reset
    if not isinstance(reset_info, Mapping):
        raise ClosedLoopError("env.reset info must be a mapping")
    actual_seed = getattr(env, "current_seed", None)
    if actual_seed is not None and int(actual_seed) != scenario_seed:
        raise ClosedLoopError(
            "requested scenario seed does not match env.current_seed: "
            f"requested={scenario_seed}, actual={actual_seed}"
        )
    observation = _as_reference(observation, observation_dim, label="reset observation")
    episode_start = observation.copy()
    if replacement is not None:
        replacement = _as_reference(replacement, observation_dim, label="replacement")
        if target is None:
            raise AssertionError("intervention replacement requires a target")
        if np.any(np.isnan(replacement[list(target.indices)])):
            raise ClosedLoopError(
                "schema_constant has no explicit finite constant for every selected "
                f"feature in target {target.name!r}"
            )

    timing = None
    try:
        timing = derive_timing(getattr(env, "config", {}))
    except (KeyError, TypeError, ValueError):
        pass
    history_length = (
        max(1, math.ceil(ACTION_HISTORY_SECONDS * timing.control_hz))
        if timing is not None
        else 1
    )
    switches = ActionSwitchTracker(history_length=history_length)
    total_reward = 0.0
    step = 0
    final_info: Mapping[str, Any] = {}
    terminated = False
    truncated = False
    policy_input_was_copied = True
    while True:
        environment_observation = observation.copy()
        if target is None:
            policy_input = environment_observation.copy()
        else:
            policy_input = replace_policy_observation(
                environment_observation, target.indices, replacement
            )
        # A malicious/mistaken adapter can mutate its argument; the env-owned
        # observation still must be intact before env.step.
        action = _single_action(adapter, policy_input, action_dim)
        if not np.array_equal(observation, environment_observation):
            policy_input_was_copied = False
            raise ClosedLoopError(
                "policy evaluation modified the environment observation; refusing "
                "to continue a non-isolated closed-loop run"
            )
        result = env.step(action)
        if not isinstance(result, tuple):
            raise ClosedLoopError("env.step must return a tuple")
        if len(result) == 5:
            next_observation, reward, terminated, truncated, info = result
        elif len(result) == 4:
            next_observation, reward, done, info = result
            terminated, truncated = bool(done), False
        else:
            raise ClosedLoopError(f"env.step returned {len(result)} values instead of 5")
        if not isinstance(info, Mapping):
            raise ClosedLoopError("env.step info must be a mapping")
        step += 1
        total_reward += float(reward)
        final_info = info
        if timing is not None:
            switch_count, switch_rate = switches.observe(
                action,
                sim_time_seconds=step * timing.action_duration_seconds,
            )
        else:
            if switches.previous_action_id is not None and switches.previous_action_id != action:
                switches.switch_count += 1
            switches.previous_action_id = action
            switches.history.append(action)
            switch_count, switch_rate = switches.switch_count, None
        observation = _as_reference(
            next_observation, observation_dim, label="next observation"
        )
        if bool(terminated) or bool(truncated):
            break

    target_lane = {
        key: final_info.get(key)
        for key in (
            "target_lane_valid",
            "ever_departed_target_lane",
            "lane_departure_count",
            "off_target_duration_seconds",
            "time_in_target_lane_ratio",
        )
        if key in final_info
    }
    return {
        "scenario_seed": int(scenario_seed),
        "total_reward": float(total_reward),
        "episode_length": int(step),
        "success": bool(final_info.get("arrive_dest", False)),
        "arrive_dest": bool(final_info.get("arrive_dest", False)),
        "out_of_road": bool(final_info.get("out_of_road", False)),
        "crash": bool(final_info.get("crash", False)),
        "route_completion": _optional_number(final_info.get("route_completion")),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "termination_reason": _termination_reason(
            terminated=bool(terminated), truncated=bool(truncated), info=final_info
        ),
        "action_switch_count": int(switch_count),
        "action_switches_per_second": switch_rate,
        "target_lane": target_lane or None,
        "reset_info_keys": sorted(str(key) for key in reset_info),
        "policy_input_was_copied": policy_input_was_copied,
        "episode_start_observation": episode_start,
    }


def _numeric_delta(intervention: Any, baseline: Any) -> float | None:
    first = _optional_number(intervention)
    second = _optional_number(baseline)
    return None if first is None or second is None else first - second


def _outcome_delta(intervention: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key in (
        "total_reward",
        "episode_length",
        "success",
        "arrive_dest",
        "out_of_road",
        "crash",
        "route_completion",
        "action_switch_count",
        "action_switches_per_second",
    ):
        values[key] = _numeric_delta(intervention.get(key), baseline.get(key))
    intervention_lane = intervention.get("target_lane")
    baseline_lane = baseline.get("target_lane")
    if isinstance(intervention_lane, Mapping) and isinstance(baseline_lane, Mapping):
        values["target_lane"] = {
            key: _numeric_delta(intervention_lane.get(key), baseline_lane.get(key))
            for key in set(intervention_lane) | set(baseline_lane)
        }
    else:
        values["target_lane"] = None
    return values


def _factory_env(env_factory: Callable[[], Any]) -> Any:
    env = env_factory()
    if env is None:
        raise ClosedLoopError("env_factory returned None")
    return env


def run_paired_closed_loop(
    *,
    env_factory: Callable[[], Any],
    adapter: Any,
    scenario_seeds: Iterable[int],
    targets: Iterable[Any],
    replacement_strategy: str,
    specified_reference: Any | None = None,
    dataset_observations: Any | None = None,
    schema: Any | None = None,
    rl_seed: int | None = None,
) -> ClosedLoopResult:
    """Run baseline/intervention pairs for every selected scenario and target."""

    seeds = tuple(int(seed) for seed in scenario_seeds)
    if not seeds:
        raise ClosedLoopError("at least one scenario seed is required")
    probe = _factory_env(env_factory)
    try:
        shape = tuple(getattr(getattr(probe, "observation_space", None), "shape", ()))
        if len(shape) != 1:
            raise ClosedLoopError("closed-loop mode requires flat observations")
        observation_dim = int(shape[0])
    finally:
        close = getattr(probe, "close", None)
        if callable(close):
            close()
    normalised_targets = normalize_targets(targets, observation_dim=observation_dim)

    records: list[dict[str, Any]] = []
    for target in normalised_targets:
        for scenario_seed in seeds:
            baseline_env = _factory_env(env_factory)
            try:
                baseline = _episode_outcome(
                    env=baseline_env,
                    adapter=adapter,
                    scenario_seed=scenario_seed,
                    replacement=None,
                    target=None,
                    rl_seed=rl_seed,
                )
            finally:
                close = getattr(baseline_env, "close", None)
                if callable(close):
                    close()

            intervention_env = _factory_env(env_factory)
            try:
                # Resolve the reference from the intervention reset's actual
                # start state.  Episode-start replacement is therefore tied to
                # the same scenario rather than an unrelated rollout row.
                _seed_spaces(intervention_env, rl_seed)
                reset = intervention_env.reset(seed=scenario_seed)
                if isinstance(reset, tuple) and len(reset) == 2:
                    intervention_initial, reset_info = reset
                else:
                    intervention_initial, reset_info = reset, {}
                if not isinstance(reset_info, Mapping):
                    raise ClosedLoopError("env.reset info must be a mapping")
                # _episode_outcome performs its own reset so this preliminary
                # reset is only for reference construction.  MetaDrive reset
                # with an explicit seed recreates the same scenario.
                initial = _as_reference(
                    intervention_initial,
                    observation_dim,
                    label="intervention reset observation",
                )
                replacement = _reference_for_episode(
                    strategy=replacement_strategy,
                    episode_start=initial,
                    observation_dim=observation_dim,
                    specified_reference=specified_reference,
                    dataset_observations=dataset_observations,
                    schema=schema,
                )
                intervention = _episode_outcome(
                    env=intervention_env,
                    adapter=adapter,
                    scenario_seed=scenario_seed,
                    replacement=replacement,
                    target=target,
                    rl_seed=rl_seed,
                    initial_reset=(intervention_initial, reset_info),
                )
            finally:
                close = getattr(intervention_env, "close", None)
                if callable(close):
                    close()
            baseline_start = np.asarray(
                baseline.get("episode_start_observation"), dtype=np.float32
            )
            intervention_start = np.asarray(
                intervention.get("episode_start_observation"), dtype=np.float32
            )
            if not np.array_equal(baseline_start, intervention_start):
                maximum = (
                    float(np.max(np.abs(baseline_start - intervention_start)))
                    if baseline_start.shape == intervention_start.shape
                    else float("nan")
                )
                raise ClosedLoopError(
                    "paired baseline/intervention reset observations differ despite "
                    f"the same scenario/RL seeds: scenario_seed={scenario_seed}, "
                    f"baseline_shape={baseline_start.shape}, "
                    f"intervention_shape={intervention_start.shape}, "
                    f"max_abs_difference={maximum}"
                )
            # ``episode_start_observation`` is a runtime helper, not a scalar
            # result; retaining it in the metadata would bloat JSONL.
            baseline.pop("episode_start_observation", None)
            intervention.pop("episode_start_observation", None)
            delta = _outcome_delta(intervention, baseline)
            record = {
                "target_name": target.name,
                "target_kind": target.kind,
                "target_indices": list(target.indices),
                "scenario_seed": scenario_seed,
                "replacement_strategy": replacement_strategy,
                "baseline": baseline,
                "intervention": intervention,
                "delta": delta,
                "baseline_total_reward": baseline["total_reward"],
                "intervention_total_reward": intervention["total_reward"],
                "delta_total_reward": delta["total_reward"],
                "baseline_episode_length": baseline["episode_length"],
                "intervention_episode_length": intervention["episode_length"],
                "delta_episode_length": delta["episode_length"],
                "delta_route_completion": delta["route_completion"],
            }
            records.append(record)

    summary_rows: list[dict[str, Any]] = []
    for target in normalised_targets:
        matching = [record for record in records if record["target_name"] == target.name]
        row: dict[str, Any] = {
            "target_name": target.name,
            "target_kind": target.kind,
            "target_indices": ";".join(str(index) for index in target.indices),
            "replacement_strategy": replacement_strategy,
            "scenario_count": len(matching),
        }
        for metric in (
            "total_reward",
            "episode_length",
            "success",
            "arrive_dest",
            "out_of_road",
            "crash",
            "route_completion",
            "action_switch_count",
            "action_switches_per_second",
        ):
            baseline_values = [
                _optional_number(record["baseline"].get(metric)) for record in matching
            ]
            intervention_values = [
                _optional_number(record["intervention"].get(metric)) for record in matching
            ]
            delta_values = [_optional_number(record["delta"].get(metric)) for record in matching]
            for prefix, values in (
                ("baseline", baseline_values),
                ("intervention", intervention_values),
                ("delta", delta_values),
            ):
                finite = [value for value in values if value is not None]
                row[f"mean_{prefix}_{metric}"] = (
                    statistics.fmean(finite) if finite else None
                )
        for prefix in ("baseline", "intervention"):
            reason_counts: dict[str, int] = {}
            for record in matching:
                reason = str(record[prefix].get("termination_reason", "unknown"))
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            row[f"{prefix}_termination_reason_counts_json"] = json.dumps(
                reason_counts, ensure_ascii=False, sort_keys=True
            )

        lane_metric_names = sorted(
            {
                str(metric_name)
                for record in matching
                for side in ("baseline", "intervention")
                for metric_name in (
                    record[side].get("target_lane", {}).keys()
                    if isinstance(record[side].get("target_lane"), Mapping)
                    else ()
                )
            }
        )
        for metric in lane_metric_names:
            baseline_values = [
                _optional_number(record["baseline"]["target_lane"].get(metric))
                for record in matching
                if isinstance(record["baseline"].get("target_lane"), Mapping)
            ]
            intervention_values = [
                _optional_number(record["intervention"]["target_lane"].get(metric))
                for record in matching
                if isinstance(record["intervention"].get("target_lane"), Mapping)
            ]
            delta_values = [
                _optional_number(record["delta"]["target_lane"].get(metric))
                for record in matching
                if isinstance(record["delta"].get("target_lane"), Mapping)
            ]
            for prefix, values in (
                ("baseline", baseline_values),
                ("intervention", intervention_values),
                ("delta", delta_values),
            ):
                finite = [value for value in values if value is not None]
                row[f"mean_{prefix}_target_lane_{metric}"] = (
                    statistics.fmean(finite) if finite else None
                )
        summary_rows.append(row)
    return ClosedLoopResult(
        records=tuple(records),
        summary_rows=tuple(summary_rows),
        replacement_strategy=replacement_strategy,
    )


def save_closed_loop(run_directory: Path, result: ClosedLoopResult) -> dict[str, Path]:
    """Persist the required JSONL and CSV closed-loop artifacts atomically."""

    runs_path = safe_child(run_directory, "closed_loop_runs.jsonl")
    summary_path = safe_child(run_directory, "closed_loop_summary.csv")
    atomic_write_jsonl(runs_path, result.records)
    atomic_write_csv(summary_path, result.summary_rows)
    return {"runs": runs_path, "summary": summary_path}


# Public name requested by the CLI design; retain the descriptive paired name
# above for callers that need the baseline guarantee to be obvious.
run_closed_loop = run_paired_closed_loop
