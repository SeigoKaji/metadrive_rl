"""Baseline and closed-loop collection (①-B) at explicit observation boundaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import hashlib
import math
from numbers import Integral, Real
from typing import Any

import numpy as np

from .interventions import Intervention, InterventionError, apply_intervention, coerce_intervention
from .policy_comparison import policy_identity
from .reward_adapter import RewardTermsResult, reward_terms_result, validate_reward_terms
from .storage import save_frame


class CollectionError(RuntimeError):
    """A baseline or intervention rollout cannot continue."""


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise CollectionError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CollectionError(f"{name} must be finite")
    return result


def _jsonish(value: object) -> object:
    """Capture common state values without importing storage internals deeply."""

    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise CollectionError("state contains NaN or Inf")
        return value
    if isinstance(value, np.generic):
        return _jsonish(value.item())
    if isinstance(value, np.ndarray):
        return _jsonish(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _jsonish(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonish(item) for item in value]
    # A state object from a real adapter should be converted by its snapshot
    # implementation.  Refusing unknown objects avoids hidden repr state in
    # raw data.
    raise CollectionError(f"state value {type(value).__name__} is not JSONable")


def _observation_hash(observation: object) -> str:
    array = np.asarray(observation)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.number):
        raise CollectionError("observation hash requires a numeric 1-D observation")
    if np.issubdtype(array.dtype, np.complexfloating) or not np.all(np.isfinite(array)):
        raise CollectionError("observation contains NaN/Inf or complex values")
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _call_reset(env: object, seed: int) -> tuple[object, dict[str, object]]:
    reset = getattr(env, "reset", None)
    if not callable(reset):
        raise CollectionError("environment does not provide reset")
    try:
        result = reset(seed=seed)
    except TypeError:
        # A ported old Gym adapter may expose reset() only.  We still record
        # that the requested seed could not be passed through.
        result = reset()
    if isinstance(result, tuple) and len(result) == 2:
        observation, info = result
        if info is None:
            info = {}
        if not isinstance(info, Mapping):
            raise CollectionError("reset info must be a mapping")
        return observation, dict(info)
    return result, {}


def _call_step(env: object, action: int) -> object:
    step = getattr(env, "step", None)
    if not callable(step):
        raise CollectionError("environment does not provide step")
    # Only the invocation belongs in this helper.  Return validation happens
    # after the caller knows that env.step itself returned, allowing a
    # malformed tuple or non-finite reward to be recorded as an executed
    # ``step_result`` failure instead of being mislabeled as a step exception.
    return step(action)


def _predict(adapter: object, values: np.ndarray, *, deterministic: bool) -> tuple[np.ndarray, int]:
    predict = getattr(adapter, "predict", None)
    if not callable(predict):
        raise CollectionError("adapter does not provide predict")
    # SB3's Box/MLP boundary is float32.  Synthetic adapters also accept this
    # representation, while keeping a float64 intervention here would make a
    # valid real adapter reject the exact same saved policy input.
    values = np.asarray(values, dtype=np.float32)
    result = predict(values, deterministic=deterministic)
    if not isinstance(result, tuple) or len(result) != 2:
        raise CollectionError("adapter.predict must return (probabilities, action)")
    probabilities, actions = result
    probability_array = np.asarray(probabilities, dtype=np.float64)
    if probability_array.ndim != 1 or probability_array.size == 0:
        raise CollectionError(f"probabilities must be 1-D, found {probability_array.shape}")
    if not np.all(np.isfinite(probability_array)) or np.any(probability_array < 0):
        raise CollectionError("probabilities contain invalid values")
    total = float(np.sum(probability_array))
    if not math.isclose(total, 1.0, abs_tol=1e-6, rel_tol=1e-6):
        raise CollectionError(f"probabilities do not sum to one: {total}")
    values = np.asarray(actions)
    if values.ndim == 0:
        action = int(values)
    elif values.size == 1:
        action = int(values.reshape(-1)[0])
    else:
        raise CollectionError("single-input action output has an unexpected batch shape")
    action_count = getattr(adapter, "action_count", None)
    if isinstance(action_count, Integral) and not 0 <= action < int(action_count):
        raise CollectionError(f"action {action} outside action_count={action_count}")
    return probability_array, action


def _seed_status(env: object, reset_info: Mapping[str, object], requested_seed: int) -> tuple[int | None, bool]:
    actual = reset_info.get("seed")
    if actual is None:
        actual = getattr(env, "current_seed", None)
    if actual is None:
        return None, False
    try:
        actual_int = int(actual)
    except (TypeError, ValueError):
        return None, False
    return actual_int, actual_int == requested_seed


def _termination_reason(
    info: Mapping[str, object],
    *,
    terminated: bool,
    truncated: bool,
    horizon_hit: bool,
) -> str | None:
    reason = info.get("termination_reason")
    if reason is not None:
        return str(reason)
    for key in ("arrive_dest", "out_of_road", "crash_vehicle", "crash_object", "crash"):
        if bool(info.get(key, False)):
            return key
    if terminated:
        return "terminated"
    if truncated:
        return "truncated"
    if horizon_hit:
        return "horizon"
    return None


def _reward_result(
    info: Mapping[str, object],
    reward: float,
    *,
    terminated: bool,
    truncated: bool,
    reward_terms_provider: object | None,
    strict: bool,
    atol: float,
    rtol: float,
) -> RewardTermsResult:
    """Read one provider's already-computed reward terms exactly once.

    The common path reads the explicit ``info['reward_terms']`` contract.  A
    port can pass a small provider object/function when its wrapper exposes a
    different key.  Providers must return terms (or ``None``), and are never
    called again to repair a malformed or mismatching result.
    """

    if reward_terms_provider is None:
        result = reward_terms_result(
            info,
            reward,
            terminated=terminated,
            truncated=truncated,
            atol=atol,
            rtol=rtol,
        )
    else:
        extractor = reward_terms_provider
        if not callable(extractor):
            extractor = getattr(reward_terms_provider, "extract_reward_terms", None)
        if not callable(extractor):
            raise CollectionError(
                "reward_terms_provider must be callable or expose "
                "extract_reward_terms"
            )
        # The provider contract mirrors extract_reward_terms.  Do not retry
        # on TypeError: an internal provider error must remain observable and
        # must not consume state or invoke the reward path twice.
        terms = extractor(
            info,
            reward,
            terminated=terminated,
            truncated=truncated,
        )
        if isinstance(terms, RewardTermsResult):
            # A provider result is a convenient transport object, not a
            # proof.  Recompute status/residual from its terms at this step's
            # returned-reward boundary so a fabricated ``verified`` flag or
            # stale residual cannot enter raw data.
            result = validate_reward_terms(
                terms.terms,
                reward,
                terminated=terminated,
                truncated=truncated,
                atol=atol,
                rtol=rtol,
            )
        elif terms is None or isinstance(terms, Mapping):
            result = validate_reward_terms(
                terms,
                reward,
                terminated=terminated,
                truncated=truncated,
                atol=atol,
                rtol=rtol,
            )
        else:
            raise CollectionError(
                "reward_terms_provider must return a mapping, None, or "
                "RewardTermsResult"
            )
    # Strict-mode escalation is performed by the caller after assigning the
    # result to the raw record.  That ordering preserves a mismatch/nonfinite
    # status together with the already returned reward.
    return result


def _reward_stage(adapter: object) -> str | None:
    """Return the explicit reward boundary used for A/B pairing."""

    candidate = getattr(adapter, "reward_stage", None)
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
    metadata = getattr(adapter, "metadata", None)
    if isinstance(metadata, Mapping):
        candidate = metadata.get("reward_stage")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    # The collector stores the value returned directly by env.step.  This is
    # a known boundary even when the host cannot expose a deeper reward
    # breakdown; callers can override it when a wrapper has another stage.
    return "environment.step.returned_reward"


def collect_rollout(
    adapter: object,
    *,
    pattern: Intervention | Mapping[str, object] | None = None,
    scenario_seed: int = 5,
    policy_seed: int = 5,
    horizon: int = 127,
    run_dir: str | None = None,
    record_gif: bool = True,
    deterministic: bool = True,
    reward_atol: float = 1e-6,
    reward_rtol: float = 1e-6,
    reward_terms_provider: object | None = None,
    reward_provider: object | None = None,
    strict_reward_terms: bool = False,
) -> dict[str, object]:
    """Collect one baseline or one independent closed-loop intervention run.

    The raw record is appended immediately after a successful ``env.step``.
    Snapshot and frame failures therefore cannot erase an already returned
    reward or make an executed step appear unexecuted.
    """

    if isinstance(scenario_seed, bool) or int(scenario_seed) < 0:
        raise CollectionError("scenario_seed must be a non-negative integer")
    if isinstance(policy_seed, bool) or int(policy_seed) < 0:
        raise CollectionError("policy_seed must be a non-negative integer")
    if isinstance(horizon, bool) or int(horizon) < 0:
        raise CollectionError("horizon must be non-negative")
    scenario_seed = int(scenario_seed)
    policy_seed = int(policy_seed)
    horizon = int(horizon)
    if reward_terms_provider is not None and reward_provider is not None:
        raise CollectionError(
            "provide only one of reward_terms_provider and reward_provider"
        )
    if reward_terms_provider is None:
        reward_terms_provider = reward_provider
    if not isinstance(strict_reward_terms, bool):
        raise CollectionError("strict_reward_terms must be a boolean")
    dimension = None
    for dimension_name in ("dimension", "observation_dim", "object_dimension"):
        try:
            candidate_dimension = getattr(adapter, dimension_name)
        except Exception:
            continue
        if isinstance(candidate_dimension, Integral) and not isinstance(candidate_dimension, bool):
            dimension = int(candidate_dimension)
            break
    if not isinstance(dimension, Integral) or int(dimension) <= 0:
        raise CollectionError("adapter must expose positive integer dimension")
    dimension = int(dimension)
    intervention: Intervention | None
    if pattern is None:
        intervention = None
        rollout_id, name = "P00", "baseline"
    else:
        try:
            intervention = coerce_intervention(pattern, dimension=dimension)
        except InterventionError as error:
            return _failed_rollout(
                pattern,
                failure={"stage": "intervention", "step": 0, "error": str(error)},
            )
        rollout_id, name = intervention.id, intervention.name
    result: dict[str, object] = {
        "id": rollout_id,
        "name": name,
        "pattern": (
            None
            if intervention is None
            else {
                "id": intervention.id,
                "name": intervention.name,
                "indices": list(intervention.indices),
                "fixed_value": intervention.fixed_value,
            }
        ),
        "indices": [] if intervention is None else list(intervention.indices),
        "fixed_value": None if intervention is None else intervention.fixed_value,
        "status": "failed",
        "failure": None,
        "comparable": False,
        "initial": {
            "observation_hash": None,
            "observation": None,
            "state": None,
            "seed": None,
            "requested_seed": scenario_seed,
            "seed_verified": False,
        },
        "adapter_identity": policy_identity(adapter),
        "reward_stage": _reward_stage(adapter),
        "action_dt": None,
        "records": [],
        "applied_count": 0,
        "changed_count": 0,
        "unchanged_count": 0,
        "termination": {"step": None, "reason": None, "terminated": False, "truncated": False},
        "requested": {
            "scenario_seed": scenario_seed,
            "policy_seed": policy_seed,
            "horizon": horizon,
            "deterministic": bool(deterministic),
            "pattern": (
                None
                if intervention is None
                else {
                    "id": intervention.id,
                    "name": intervention.name,
                    "indices": list(intervention.indices),
                    "fixed_value": intervention.fixed_value,
                }
            ),
        },
    }
    env = None
    failure: dict[str, object] | None = None
    try:
        seed_policy = getattr(adapter, "seed_policy", None)
        if callable(seed_policy):
            seed_policy(policy_seed)
        make_env = getattr(adapter, "make_env", None)
        if not callable(make_env):
            raise CollectionError("adapter does not provide make_env")
        env = make_env()
        observation, reset_info = _call_reset(env, scenario_seed)
        actual_seed, seed_verified = _seed_status(env, reset_info, scenario_seed)
        initial = result["initial"]
        assert isinstance(initial, dict)
        initial["seed"] = actual_seed
        initial["seed_verified"] = seed_verified
        initial["observation"] = _jsonish(observation)
        initial["observation_hash"] = _observation_hash(observation)
        try:
            snapshot = getattr(adapter, "snapshot")
            initial["state"] = _jsonish(snapshot(env))
        except Exception as error:
            # Initial state is optional on a ported adapter, but preserve the
            # reason so the run cannot claim a fully verified initial state.
            failure = {"stage": "initial_snapshot", "step": 0, "error": f"{type(error).__name__}: {error}"}
        if not seed_verified:
            failure = failure or {
                "stage": "reset_seed",
                "step": 0,
                "error": f"requested scenario seed {scenario_seed} was not verified (actual={actual_seed!r})",
            }
        try:
            action_dt_fn = getattr(adapter, "action_dt")
            action_dt = _finite_float(action_dt_fn(env), name="action_dt")
            if action_dt <= 0:
                raise CollectionError("action_dt must be positive")
            result["action_dt"] = action_dt
        except Exception as error:
            failure = failure or {"stage": "action_dt", "step": 0, "error": f"{type(error).__name__}: {error}"}
        step_index = 0
        while step_index < horizon:
            raw_observation = observation
            record: dict[str, object] = {
                "step": step_index,
                "raw_observation": _jsonish(raw_observation),
                "next_observation": None,
                "next_observation_hash": None,
                "input": None,
                "modified_input": None,
                "probabilities": None,
                "action": None,
                "executed": False,
                "reward": None,
                "terminated": False,
                "truncated": False,
                "info": {},
                "state": None,
                "reward_terms": {"status": "unavailable", "terms": None, "residual": None, "atol": reward_atol, "rtol": reward_rtol},
                "frame": None,
                "sim_time": None,
            }
            try:
                prepare = getattr(adapter, "prepare")
                prepared = np.asarray(prepare(raw_observation))
                if prepared.ndim != 1 or prepared.shape[0] != dimension:
                    raise CollectionError(
                        f"prepared input shape {prepared.shape} does not match ({dimension},)"
                    )
                if np.issubdtype(prepared.dtype, np.complexfloating) or not np.all(np.isfinite(prepared)):
                    raise CollectionError("prepared input contains invalid values")
                prepared = np.asarray(prepared, dtype=np.float32)
                record["input"] = prepared.tolist()
                if intervention is None:
                    modified = np.array(prepared, copy=True)
                    record["modified_input"] = modified.tolist()
                else:
                    applied = apply_intervention(prepared, intervention, dimension=dimension)
                    # Keep the adapter's ordinary float32 policy boundary;
                    # apply_intervention computes in float64 so fixed values
                    # cannot be silently truncated, then this cast preserves
                    # the representable requested value.
                    modified = np.asarray(applied.modified, dtype=np.float32)
                    record["modified_input"] = modified.tolist()
                    result["applied_count"] = int(result["applied_count"]) + 1
                    if applied.changed:
                        result["changed_count"] = int(result["changed_count"]) + 1
                    else:
                        result["unchanged_count"] = int(result["unchanged_count"]) + 1
                probabilities, action = _predict(adapter, modified, deterministic=deterministic)
                record["probabilities"] = probabilities.tolist()
                record["action"] = action
            except Exception as error:
                failure = failure or {"stage": "predict" if record["input"] is not None else "prepare_or_intervention", "step": step_index, "error": f"{type(error).__name__}: {error}"}
                cast_records = result["records"]
                assert isinstance(cast_records, list)
                cast_records.append(record)
                break
            try:
                step_result = _call_step(env, int(record["action"]))
            except Exception as error:
                # No transition was returned, so this is an env.step
                # exception and the action remains unexecuted in raw data.
                failure = failure or {
                    "stage": "step",
                    "step": step_index,
                    "error": f"{type(error).__name__}: {error}",
                }
                cast_records = result["records"]
                assert isinstance(cast_records, list)
                cast_records.append(record)
                break
            if not isinstance(step_result, tuple) or len(step_result) != 5:
                # env.step returned, but no usable reward can be identified.
                # Retain the executed call as a distinct step-result failure.
                record["executed"] = True
                record["reward"] = None
                cast_records = result["records"]
                assert isinstance(cast_records, list)
                cast_records.append(record)
                failure = failure or {
                    "stage": "step_result",
                    "step": step_index,
                    "error": "environment.step must return observation,reward,terminated,truncated,info",
                }
                result["termination"] = {
                    "step": step_index,
                    "reason": "step_result_failure",
                    "terminated": False,
                    "truncated": False,
                }
                break
            next_observation, reward_raw, terminated_raw, truncated_raw, info = step_result
            try:
                reward = _finite_float(reward_raw, name="returned reward")
            except Exception as error:
                # The call returned, but its reward is not representable as a
                # finite experiment value.  Keep executed=True and reward=None
                # so this cannot be confused with an exception in env.step.
                record["executed"] = True
                record["reward"] = None
                cast_records = result["records"]
                assert isinstance(cast_records, list)
                cast_records.append(record)
                failure = failure or {
                    "stage": "step_result",
                    "step": step_index,
                    "error": f"{type(error).__name__}: {error}",
                }
                result["termination"] = {
                    "step": step_index,
                    "reason": "step_result_failure",
                    "terminated": False,
                    "truncated": False,
                }
                break
            if not isinstance(terminated_raw, (bool, np.bool_)) or not isinstance(
                truncated_raw,
                (bool, np.bool_),
            ):
                # The transition returned a finite reward, but malformed end
                # flags cannot safely drive the next loop.  Keep this
                # distinct from an exception raised inside env.step.
                record["executed"] = True
                record["reward"] = reward
                try:
                    record["next_observation"] = _jsonish(next_observation)
                    record["next_observation_hash"] = _observation_hash(next_observation)
                except Exception:
                    # The returned reward remains committed even if this
                    # optional transition telemetry is itself malformed.
                    pass
                cast_records = result["records"]
                assert isinstance(cast_records, list)
                cast_records.append(record)
                failure = failure or {
                    "stage": "step_result",
                    "step": step_index,
                    "error": "terminated and truncated must be booleans",
                }
                result["termination"] = {
                    "step": step_index,
                    "reason": "step_result_failure",
                    "terminated": False,
                    "truncated": False,
                }
                break
            terminated = bool(terminated_raw)
            truncated = bool(truncated_raw)
            # Commit the successful env.step boundary first.  Everything
            # after this point is optional telemetry and must not be allowed
            # to erase the returned reward or executed flag.
            record["executed"] = True
            record["reward"] = reward
            record["terminated"] = terminated
            record["truncated"] = truncated
            records = result["records"]
            assert isinstance(records, list)
            records.append(record)
            step_index += 1
            # The post-step observation is part of the returned transition.
            # Store it before any optional info/reward/snapshot/frame work so
            # a later measurement failure still leaves the transition intact.
            try:
                record["next_observation"] = _jsonish(next_observation)
                record["next_observation_hash"] = _observation_hash(next_observation)
            except Exception as error:
                failure = failure or {
                    "stage": "telemetry",
                    "step": step_index - 1,
                    "error": f"{type(error).__name__}: {error}",
                }
                result["termination"] = {
                    "step": step_index - 1,
                    "reason": "telemetry_failure",
                    "terminated": terminated,
                    "truncated": truncated,
                }
                record["termination_reason"] = "telemetry_failure"
                break
            action_dt_value = result.get("action_dt")
            if isinstance(action_dt_value, (int, float)) and math.isfinite(float(action_dt_value)):
                record["sim_time"] = float(step_index * float(action_dt_value))
            info_failure = False
            try:
                if info is None:
                    info_mapping: Mapping[str, object] = {}
                elif isinstance(info, Mapping):
                    info_mapping = info
                else:
                    raise CollectionError("step info must be a mapping")
                record["info"] = _jsonish(info_mapping)
                reward_result = _reward_result(
                    info_mapping,
                    reward,
                    terminated=terminated,
                    truncated=truncated,
                    reward_terms_provider=reward_terms_provider,
                    strict=strict_reward_terms,
                    atol=reward_atol,
                    rtol=reward_rtol,
                )
                # Assign before strict-mode escalation so a mismatch remains
                # visible in the retained successful-step record.
                record["reward_terms"] = reward_result.as_dict()
                if strict_reward_terms and reward_result.status in {
                    "mismatch",
                    "nonfinite",
                    "malformed",
                }:
                    raise CollectionError(
                        f"reward terms validation {reward_result.status}"
                    )
                if terminated or truncated:
                    record["termination_reason"] = _termination_reason(
                        info_mapping,
                        terminated=terminated,
                        truncated=truncated,
                        horizon_hit=False,
                    )
            except Exception as error:
                info_failure = True
                failure = failure or {
                    "stage": "telemetry",
                    "step": step_index - 1,
                    "error": f"{type(error).__name__}: {error}",
                }
            if info_failure:
                # A successful step is retained above, but the pattern stops
                # at this measurement boundary as required by the contract.
                result["termination"] = {
                    "step": step_index - 1,
                    "reason": _termination_reason(
                        info if isinstance(info, Mapping) else {},
                        terminated=terminated,
                        truncated=truncated,
                        horizon_hit=False,
                    ),
                    "terminated": terminated,
                    "truncated": truncated,
                }
                record["termination_reason"] = "telemetry_failure"
                break
            try:
                record["state"] = _jsonish(getattr(adapter, "snapshot")(env))
            except Exception as error:
                failure = failure or {"stage": "telemetry", "step": step_index - 1, "error": f"{type(error).__name__}: {error}"}
                result["termination"] = {
                    "step": step_index - 1,
                    "reason": "telemetry_failure",
                    "terminated": terminated,
                    "truncated": truncated,
                }
                break
            if record_gif:
                try:
                    frame = getattr(adapter, "frame")(env)
                    if run_dir is None:
                        raise CollectionError("run_dir is required when record_gif=True")
                    record["frame"] = save_frame(run_dir, rollout_id, step_index - 1, frame)
                except Exception as error:
                    failure = failure or {"stage": "frame", "step": step_index - 1, "error": f"{type(error).__name__}: {error}"}
                    result["termination"] = {
                        "step": step_index - 1,
                        "reason": "frame_failure",
                        "terminated": terminated,
                        "truncated": truncated,
                    }
                    record["termination_reason"] = "frame_failure"
                    break
            observation = next_observation
            if terminated or truncated:
                result["termination"] = {
                    "step": step_index - 1,
                    "reason": _termination_reason(info if isinstance(info, Mapping) else {}, terminated=terminated, truncated=truncated, horizon_hit=False),
                    "terminated": terminated,
                    "truncated": truncated,
                }
                break
        else:
            # Explicit horizon is a collection boundary.  It is a normal
            # completed run when no environment failure occurred.
            if step_index >= horizon:
                result["termination"] = {
                    "step": step_index - 1 if step_index else None,
                    "reason": "horizon",
                    "terminated": False,
                    "truncated": True,
                }
    except Exception as error:
        failure = failure or {"stage": "setup", "step": 0, "error": f"{type(error).__name__}: {error}"}
    finally:
        if env is not None:
            try:
                close = getattr(env, "close", None)
                if callable(close):
                    close()
            except Exception as error:
                failure = failure or {"stage": "close", "step": len(result.get("records", [])), "error": f"{type(error).__name__}: {error}"}
    result["failure"] = failure
    result["status"] = "complete" if failure is None else "failed"
    result["comparable"] = bool(failure is None and result["initial"].get("seed_verified"))
    return result


def _failed_rollout(pattern: object, *, failure: dict[str, object]) -> dict[str, object]:
    pattern_id = "unknown"
    pattern_name = "unknown"
    if isinstance(pattern, Mapping):
        pattern_id = str(pattern.get("id", pattern_id))
        pattern_name = str(pattern.get("name", pattern_name))
    else:
        pattern_id = str(getattr(pattern, "id", pattern_id))
        pattern_name = str(getattr(pattern, "name", pattern_name))
    return {
        "id": pattern_id,
        "name": pattern_name,
        "status": "failed",
        "failure": failure,
        "comparable": False,
        "initial": {"observation_hash": None, "observation": None, "state": None, "seed": None, "seed_verified": False},
        "action_dt": None,
        "records": [],
        "applied_count": 0,
        "changed_count": 0,
        "unchanged_count": 0,
    }


def collect_baseline(adapter: object, **kwargs: object) -> dict[str, object]:
    """Collect the common unmodified baseline episode."""

    return collect_rollout(adapter, pattern=None, **kwargs)


def collect_closed_loop(
    adapter: object,
    pattern: Intervention | Mapping[str, object],
    **kwargs: object,
) -> dict[str, object]:
    """Collect one independent ①-B episode with the fixed input pattern."""

    return collect_rollout(adapter, pattern=pattern, **kwargs)


run_closed_loop = collect_closed_loop
collect_intervention = collect_closed_loop


__all__ = [
    "CollectionError",
    "collect_baseline",
    "collect_closed_loop",
    "collect_intervention",
    "collect_rollout",
    "run_closed_loop",
]
