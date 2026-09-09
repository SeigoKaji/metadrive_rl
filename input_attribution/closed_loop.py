"""重点入力介入を毎step実環境へ適用するclosed-loop走行。

この層は環境の状態を評価する責務を持ち、観測置換は ``interventions`` の
schema-aware APIへ委譲する。P00は必ず先に走らせ、各patternは同じscenario/RL
seedで逐次的にreset・closeし、初期条件と保存済みreferenceを検証する。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
import copy
import hashlib
import inspect
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import RunArtifacts, build_manifest
from .collection import (
    CollectionError,
    _call_variants,
    close_environment,
    decode_action,
    finalize_video,
    infer_video_fps,
    make_environment,
    model_input,
    policy_predict,
    policy_probabilities,
    render_frame,
    reset_environment,
    save_video_frame,
    seed_runtime,
    step_environment,
    telemetry,
    telemetry_time,
)
from .interventions import (
    InterventionPattern,
    InterventionResult,
    apply_intervention,
    patterns_from_config,
)
from .metrics import compare_distributions, summarize_intervention_records
from .schema import InputSchema
from .trajectory_metrics import paired_deltas, summarize_trajectory


class ClosedLoopError(RuntimeError):
    """closed-loop走行を安全に完了できなかった。"""


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return str(value)


def _copy(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _find_hook(adapter: Any, names: Sequence[str]):
    for name in names:
        hook = getattr(adapter, name, None)
        if callable(hook):
            return hook
        if isinstance(adapter, Mapping) and callable(adapter.get(name)):
            return adapter[name]
    return None


def _pattern_mapping(config: Any, patterns: Sequence[Any] | None) -> list[Any]:
    values = list(patterns) if patterns is not None else list(getattr(config, "patterns", ()) or ())
    if not values:
        return [{"id": "P00", "method": "identity", "indices": []}]
    return values


def apply_pattern(
    observation: Any,
    pattern: InterventionPattern | Mapping[str, Any],
    schema: InputSchema,
    *,
    reference_observation: Any | None = None,
    reference_observations: Mapping[str, Any] | None = None,
    reference_context: Mapping[str, Any] | None = None,
    reference_contexts: Mapping[str, Mapping[str, Any]] | None = None,
    observation_context: Mapping[str, Any] | None = None,
    strict: bool = True,
) -> InterventionResult:
    """Apply the common intervention implementation exactly once.

    Closed-loop patterns may be intentionally tied to a saved road segment.
    A mismatch is recorded as a skipped intervention so the physical episode
    remains valid and the pattern can be reported without aborting later
    patterns.
    """

    return apply_intervention(
        _copy(observation),
        pattern,
        schema,
        reference_observation=reference_observation,
        reference_observations=reference_observations,
        reference_context=reference_context,
        reference_contexts=reference_contexts,
        observation_context=observation_context,
        strict=strict,
    )


def _observation_hash(observation: Any) -> str:
    array = np.ascontiguousarray(np.asarray(observation))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _state_snapshot(adapter: Any, env: Any, pre: Mapping[str, Any]) -> dict[str, Any]:
    """Read optional position/lane state without inventing nearest-lane values."""

    result: dict[str, Any] = {"telemetry": _plain(pre)}
    hook = _find_hook(adapter, ("initial_state", "state_snapshot", "episode_state", "snapshot_state"))
    if hook is not None:
        try:
            value = _call_variants(hook, (((env,), {}), ((env, dict(pre)), {}), ((), {})))
            if isinstance(value, Mapping):
                result["state"] = _plain(value)
        except Exception as exc:
            result["state_error"] = f"{type(exc).__name__}: {exc}"
    # A vehicle position may already be included in telemetry; preserve the
    # exact key/value instead of deriving a coordinate from another lane.
    for key in (
        "position", "position_xy", "position_xy_m", "vehicle_position", "vehicle_x", "vehicle_y",
        "target_lane_ordinal", "current_lane_ordinal", "target_lane_valid",
        "target_lane_offset_m", "normalized_target_lane_error",
        "road_segment_id",
        "actual_scenario_seed", "scenario_seed", "seed",
    ):
        if key in pre:
            result[key] = _plain(pre[key])
    return result


def _required_initial_fields(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    state = snapshot.get("state") if isinstance(snapshot.get("state"), Mapping) else {}
    telemetry_data = snapshot.get("telemetry") if isinstance(snapshot.get("telemetry"), Mapping) else {}
    merged = dict(telemetry_data)
    merged.update(state)
    fields: dict[str, Any] = {}
    position_aliases = ("position", "position_xy", "position_xy_m", "vehicle_position")
    for key in position_aliases:
        value = merged.get(key)
        if value is not None:
            try:
                position_values = np.asarray(value, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                position_values = np.empty((0,), dtype=float)
            if position_values.size < 2 or not np.all(np.isfinite(position_values[:2])):
                continue
            fields["position"] = merged[key]
            break
    if "vehicle_x" in merged and "vehicle_y" in merged:
        try:
            vehicle_position = [float(merged["vehicle_x"]), float(merged["vehicle_y"])]
        except (TypeError, ValueError):
            vehicle_position = None
        if vehicle_position is not None and all(np.isfinite(value) for value in vehicle_position):
            fields.setdefault("position", vehicle_position)
    for key in ("target_lane_ordinal", "current_lane_ordinal", "target_lane_valid"):
        if key in merged and merged[key] is not None:
            fields[key] = merged[key]
    for key in ("target_lane_offset_m", "normalized_target_lane_error"):
        if key in merged and merged[key] is not None:
            fields[key] = merged[key]
    if merged.get("road_segment_id") is not None:
        fields["road_segment_id"] = merged["road_segment_id"]
    return fields


def _coerce_seed(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    try:
        if float(value) != float(parsed):
            return None
    except (TypeError, ValueError, OverflowError):
        pass
    return parsed


def _actual_scenario_seed(
    env: Any,
    reset_info: Mapping[str, Any] | None,
    pre_telemetry: Mapping[str, Any] | None,
) -> int | None:
    """Read the seed that the environment actually applied after reset."""

    # Prefer an explicitly reported actual seed over a generic reset ``seed``;
    # the latter can be the requested value echoed by a wrapper even when the
    # underlying environment ignored it.  Conflicting explicit sources are
    # treated as unavailable so the caller fails closed instead of choosing
    # whichever wrapper happened to be visited first.
    explicit: list[int] = []
    for source in (pre_telemetry, reset_info):
        if not isinstance(source, Mapping):
            continue
        key = "actual_scenario_seed"
        if key in source and source[key] is not None:
            parsed = _coerce_seed(source[key])
            if parsed is not None:
                explicit.append(parsed)
    for name in ("current_seed", "actual_scenario_seed", "scenario_seed", "_active_scenario_seed"):
        value = getattr(env, name, None)
        if value is not None and not callable(value):
            parsed = _coerce_seed(value)
            if parsed is not None:
                explicit.append(parsed)
    if explicit:
        return explicit[0] if all(value == explicit[0] for value in explicit) else None
    return None


def _compare_initial(
    baseline: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    baseline_observation: np.ndarray,
    current_observation: np.ndarray,
    scenario_seed: int | None,
    rl_seed: int | None,
) -> dict[str, Any]:
    """Compare requested seeds and explicit initial geometry.

    A missing position or target-lane ordinal/valid flag is an unverified
    comparison.  It is never promoted to a match merely because the model
    observation happens to be equal.
    """

    reasons: list[str] = []
    if not np.array_equal(baseline_observation, current_observation):
        reasons.append("model_observation_mismatch")
    baseline_requested_scenario = baseline.get("requested_scenario_seed", baseline.get("scenario_seed"))
    current_requested_scenario = current.get("requested_scenario_seed", scenario_seed)
    if baseline_requested_scenario != current_requested_scenario:
        reasons.append("scenario_seed_mismatch")
    baseline_requested_rl = baseline.get("requested_rl_seed", baseline.get("rl_seed"))
    current_requested_rl = current.get("requested_rl_seed", rl_seed)
    if baseline_requested_rl != current_requested_rl:
        reasons.append("rl_seed_mismatch")
    baseline_actual = _coerce_seed(baseline.get("actual_scenario_seed"))
    current_actual = _coerce_seed(current.get("actual_scenario_seed"))
    if (
        baseline_requested_scenario is not None
        and baseline_actual is not None
        and current_actual is not None
        and baseline_actual != current_actual
    ):
        reasons.append("actual_scenario_seed_mismatch")
    baseline_fields = _required_initial_fields(baseline.get("snapshot", {}))
    current_fields = _required_initial_fields(current.get("snapshot", {}))
    unavailable: list[str] = []
    for key in sorted(set(baseline_fields) | set(current_fields)):
        if key not in baseline_fields or key not in current_fields:
            unavailable.append(key)
            continue
        if baseline_fields[key] != current_fields[key]:
            reasons.append(f"{key}_mismatch")
    required_fields = ("position", "target_lane_ordinal", "target_lane_valid")
    missing_required = [
        key
        for key in required_fields
        if key not in baseline_fields or key not in current_fields
    ]
    unavailable.extend(key for key in missing_required if key not in unavailable)
    if baseline.get("requested_scenario_seed", baseline.get("scenario_seed")) is not None and (
        baseline_actual is None or current_actual is None
    ):
        unavailable.append("actual_scenario_seed")
    if reasons:
        status = "mismatch"
    elif unavailable:
        status = "unverified"
    else:
        status = "matched"
    return {
        "status": status,
        "matched": True if status == "matched" else False if status == "mismatch" else None,
        "reasons": reasons,
        "unavailable_fields": unavailable,
        "unverified_fields": unavailable,
        "checked_fields": sorted(set(baseline_fields) | set(current_fields)),
    }


def _safe_stack(values: Sequence[np.ndarray], expected_dim: int | None = None) -> np.ndarray:
    if not values:
        width = expected_dim or 0
        return np.empty((0, width), dtype=np.float32)
    try:
        result = np.stack([np.asarray(value) for value in values], axis=0)
    except (TypeError, ValueError) as exc:
        raise ClosedLoopError(f"trajectory observations are not fixed-shape arrays: {exc}") from exc
    if result.dtype.kind not in "biufc":
        raise ClosedLoopError(f"trajectory observation dtype is not numeric: {result.dtype}")
    return np.array(result, copy=True)


def _reference_records_by_episode(reference_records: Any) -> dict[int, list[Mapping[str, Any]]]:
    if reference_records is None:
        return {}
    rows: list[Mapping[str, Any]] = []
    if isinstance(reference_records, Mapping):
        nested = reference_records.get("records")
        if isinstance(nested, Sequence):
            rows = [row for row in nested if isinstance(row, Mapping)]
        else:
            for value in reference_records.values():
                if isinstance(value, Sequence):
                    rows.extend(row for row in value if isinstance(row, Mapping))
    elif isinstance(reference_records, Sequence):
        rows = [row for row in reference_records if isinstance(row, Mapping)]
    result: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        try:
            episode = int(row.get("episode", 0))
        except (TypeError, ValueError):
            episode = 0
        result.setdefault(episode, []).append(row)
    for episode_rows in result.values():
        episode_rows.sort(key=lambda row: int(row.get("step", 0)))
    return result


def _reference_observation_for_row(
    reference_observations: Mapping[str, Any] | None,
    row: Mapping[str, Any],
) -> Any | None:
    """Resolve a saved observation using the collection's stable row IDs.

    The CLI emits both episode-N:step and episode:step keys. Older callers
    may provide a nested {episode_id: {step: observation}} mapping or an
    observation-index key, so resolution remains explicit and deterministic.
    """

    if not isinstance(reference_observations, Mapping):
        return None
    try:
        episode = int(row.get("episode", 0))
    except (TypeError, ValueError):
        episode = 0
    try:
        step = int(row.get("step", 0))
    except (TypeError, ValueError):
        step = 0
    episode_id = str(row.get("episode_id", f"episode-{episode}"))
    keys: list[Any] = [
        f"{episode_id}:{step}",
        f"{episode}:{step}",
        f"{episode_id}:step-{step}",
        f"episode-{episode}:step-{step}",
    ]
    if row.get("observation_index") is not None:
        keys.extend((str(row["observation_index"]), row["observation_index"]))
    for key in keys:
        if key in reference_observations:
            return reference_observations[key]
    for parent_key in (episode_id, f"episode-{episode}", str(episode)):
        nested = reference_observations.get(parent_key)
        if not isinstance(nested, Mapping):
            continue
        for key in (step, str(step), f"step-{step}"):
            if key in nested:
                return nested[key]
    return None


def _values_equal(left: Any, right: Any, *, numeric_tolerance: float = 1e-8) -> bool:
    """Compare JSON-like trace values without turning missing fields into zero."""

    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        if set(left) != set(right):
            return False
        return all(_values_equal(left[key], right[key], numeric_tolerance=numeric_tolerance) for key in left)
    if isinstance(left, (list, tuple, np.ndarray)) or isinstance(right, (list, tuple, np.ndarray)):
        try:
            return bool(
                np.allclose(
                    np.asarray(left, dtype=float),
                    np.asarray(right, dtype=float),
                    atol=numeric_tolerance,
                    rtol=numeric_tolerance,
                    equal_nan=False,
                )
            )
        except (TypeError, ValueError):
            try:
                return list(left) == list(right)
            except TypeError:
                return False
    if isinstance(left, (float, np.floating)) or isinstance(right, (float, np.floating)):
        try:
            return bool(np.isclose(float(left), float(right), atol=numeric_tolerance, rtol=numeric_tolerance))
        except (TypeError, ValueError):
            return False
    return left == right


def _trace_compare(
    reference_rows: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    *,
    reference_observations: Mapping[str, Any] | None = None,
    current_observations: Sequence[np.ndarray] | None = None,
) -> dict[str, Any]:
    """Compare all recorded P00 fields when an explicit reference is supplied."""

    fields = ("probabilities", "action", "reward", "terminated", "truncated", "post_telemetry")
    observation_check = reference_observations is not None
    initial_fields = (
        "position",
        "position_xy",
        "position_xy_m",
        "vehicle_position",
        "vehicle_x",
        "vehicle_y",
        "target_lane_ordinal",
        "current_lane_ordinal",
        "target_lane_valid",
        "target_lane_offset_m",
        "normalized_target_lane_error",
        "road_segment_id",
    )
    mismatches: list[dict[str, Any]] = []
    if len(reference_rows) != len(rows):
        mismatches.append({"field": "record_count", "expected": len(reference_rows), "actual": len(rows)})
    for index, (expected, actual) in enumerate(zip(reference_rows, rows, strict=False)):
        for field in fields:
            left = expected.get(field)
            right = actual.get(field)
            if field == "post_telemetry" and isinstance(left, Mapping) and isinstance(right, Mapping):
                # Newer collectors add explicit phase/step fields.  Require
                # every saved reference value while allowing such additive
                # provenance keys in the current run.
                equal = all(
                    key in right and _values_equal(value, right[key])
                    for key, value in left.items()
                )
            else:
                equal = _values_equal(left, right)
            if not equal:
                mismatches.append({"step": actual.get("step", index), "field": field, "expected": _plain(left), "actual": _plain(right)})
        if index == 0:
            for field in ("scenario_seed", "rl_seed"):
                if field in expected and not _values_equal(expected.get(field), actual.get(field)):
                    mismatches.append({
                        "step": actual.get("step", index),
                        "field": field,
                        "expected": _plain(expected.get(field)),
                        "actual": _plain(actual.get(field)),
                    })
            expected_pre = expected.get("pre_telemetry")
            actual_pre = actual.get("pre_telemetry")
            if isinstance(expected_pre, Mapping):
                actual_pre = actual_pre if isinstance(actual_pre, Mapping) else {}
                for field in initial_fields:
                    if field not in expected_pre:
                        continue
                    if field not in actual_pre or not _values_equal(expected_pre[field], actual_pre[field]):
                        mismatches.append({
                            "step": actual.get("step", index),
                            "field": f"initial_{field}",
                            "expected": _plain(expected_pre.get(field)),
                            "actual": _plain(actual_pre.get(field)),
                        })
        if observation_check:
            expected_observation = _reference_observation_for_row(reference_observations, expected)
            current_observation = None
            if current_observations is not None and index < len(current_observations):
                current_observation = current_observations[index]
            if expected_observation is None:
                mismatches.append({
                    "step": actual.get("step", index),
                    "field": "observation",
                    "reason": "reference_observation_missing",
                })
            elif current_observation is None or not np.array_equal(
                np.asarray(expected_observation), np.asarray(current_observation)
            ):
                mismatches.append({
                    "step": actual.get("step", index),
                    "field": "observation",
                    "expected": _plain(expected_observation),
                    "actual": _plain(current_observation),
                })
    checked_fields = list(fields)
    if observation_check:
        checked_fields.append("observation")
    checked_fields.extend(("initial_scenario_seed", "initial_rl_seed", "initial_geometry"))
    return {
        "status": "matched" if not mismatches else "mismatch",
        "matched": not mismatches,
        "checked_fields": checked_fields,
        "checked_records": min(len(reference_rows), len(rows)),
        "mismatches": mismatches[:50],
        "mismatch_count": len(mismatches),
    }


def _trace_alignment(baseline_rows: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]], baseline_obs: Sequence[np.ndarray], current_obs: Sequence[np.ndarray]) -> dict[str, Any]:
    """Compare the unmodified path until the intervention causes divergence."""

    compared = 0
    first_divergence: int | None = None
    fields = ("original_probabilities", "original_action", "reward", "post_telemetry", "terminated", "truncated")
    for index, (baseline, current) in enumerate(zip(baseline_rows, rows, strict=False)):
        equal = True
        if index >= len(baseline_obs) or index >= len(current_obs) or not np.array_equal(baseline_obs[index], current_obs[index]):
            equal = False
        for field in fields:
            left = baseline.get("probabilities") if field == "original_probabilities" else baseline.get(field)
            right = current.get(field)
            if field == "original_probabilities":
                left = baseline.get("probabilities")
            if field == "post_telemetry":
                equal = equal and left == right
            elif field == "original_probabilities":
                try:
                    equal = equal and bool(np.allclose(np.asarray(left, dtype=float), np.asarray(right, dtype=float), atol=1e-8, rtol=1e-8))
                except (TypeError, ValueError):
                    equal = False
            else:
                equal = equal and left == right
        if equal:
            compared += 1
        elif first_divergence is None:
            first_divergence = int(current.get("step", index))
    status = "matched" if first_divergence is None and len(rows) == len(baseline_rows) else "diverged_after_step"
    return {
        "status": status,
        "comparable_prefix_steps": compared,
        "first_divergence_step": first_divergence,
        "baseline_steps": len(baseline_rows),
        "current_steps": len(rows),
        "fields": list(fields),
    }


@dataclass
class ClosedLoopResult:
    store: RunArtifacts | None
    patterns: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    baseline_references: dict[int, dict[str, Any]] = field(default_factory=dict)
    policy_fingerprint_before: str | None = None
    policy_fingerprint_after: str | None = None
    policy_unchanged: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        def numeric(value: Any) -> float | None:
            if value is None or isinstance(value, bool) or isinstance(value, (Mapping, list, tuple)):
                return None
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                return None
            return parsed if np.isfinite(parsed) else None

        def metric_summary(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
            metric_rows = [
                item.get("metrics", {}) if isinstance(item.get("metrics", {}), Mapping) else {}
                for item in episodes
            ]
            names = sorted({
                str(name)
                for row in metric_rows
                for name, value in row.items()
                if numeric(value) is not None
            })
            episode_mean: dict[str, float | None] = {}
            step_weighted: dict[str, float | None] = {}
            weighted_episode_mean: dict[str, float | None] = {}
            metric_sum: dict[str, float | None] = {}
            aggregation: dict[str, str] = {}
            metric_counts: dict[str, dict[str, int]] = {}
            pooled_rms_names = {
                "lane_rms_m",
                "lane_error_rms_m",
                "target_lane_rms_m",
            }
            for name in names:
                values: list[float] = []
                weighted_values: list[tuple[float, int]] = []
                valid_weighted_values: list[tuple[float, int]] = []
                for index, row in enumerate(metric_rows):
                    value = numeric(row.get(name))
                    if value is None:
                        continue
                    values.append(value)
                    episode = episodes[index] if index < len(episodes) else {}
                    weight_value = row.get("poststep_count", episode.get("record_count", 0))
                    try:
                        weight = max(0, int(weight_value))
                    except (TypeError, ValueError):
                        weight = 0
                    weighted_values.append((value, weight))
                    try:
                        valid_weight = max(0, int(row.get("valid_count", 0)))
                    except (TypeError, ValueError):
                        valid_weight = 0
                    valid_weighted_values.append((value, valid_weight))
                episode_mean[name] = float(np.mean(values)) if values else None
                total_weight = sum(weight for _, weight in weighted_values)
                weighted_episode_mean[name] = (
                    float(sum(value * weight for value, weight in weighted_values) / total_weight)
                    if total_weight
                    else None
                )
                metric_sum[name] = float(sum(values)) if values else None
                if name in pooled_rms_names:
                    valid_weight = sum(weight for _, weight in valid_weighted_values)
                    step_weighted[name] = (
                        float(
                            np.sqrt(
                                sum(value * value * weight for value, weight in valid_weighted_values)
                                / valid_weight
                            )
                        )
                        if valid_weight
                        else None
                    )
                    aggregation[name] = "pooled_valid_samples_rms"
                else:
                    step_weighted[name] = weighted_episode_mean[name]
                    aggregation[name] = "weighted_episode_mean_by_poststeps"
                metric_counts[name] = {
                    "episodes": len(values),
                    "poststeps": int(total_weight),
                    "valid_samples": int(sum(weight for _, weight in valid_weighted_values)),
                }
            poststeps = 0
            for item in episodes:
                metrics = item.get("metrics", {})
                value = metrics.get("poststep_count") if isinstance(metrics, Mapping) else None
                if value is None:
                    value = item.get("record_count", 0)
                try:
                    poststeps += max(0, int(value))
                except (TypeError, ValueError):
                    pass
            return {
                "episode_mean": episode_mean,
                "step_weighted": step_weighted,
                "weighted_episode_mean": weighted_episode_mean,
                "sum": metric_sum,
                "aggregation": aggregation,
                "counts": {
                    "episodes": len(episodes),
                    "poststeps": int(poststeps),
                    "records": int(sum(int(item.get("record_count", 0)) for item in episodes)),
                    "metrics": metric_counts,
                },
            }

        def intervention_summary(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
            rows = [
                item.get("intervention_counts")
                for item in episodes
                if isinstance(item.get("intervention_counts"), Mapping)
            ]
            integer_names = (
                "target_step_count",
                "planned_target_step_count",
                "attempted_step_count",
                "executed_env_step_count",
                "actual_target_step_count",
                "aborted_before_env_step_count",
                "eligible_count",
                "applied_count",
                "changed_count_exact",
                "applied_element_count",
                "changed_element_count_exact",
                "noop_element_count",
                "noop_count",
                "skipped_count",
                "out_of_scope_count",
                "failed_count",
                "unknown_record_count",
            )
            aggregate: dict[str, Any] = {
                name: int(sum(int(row.get(name, 0) or 0) for row in rows))
                for name in integer_names
            }
            aggregate["episodes"] = len(episodes)
            aggregate["recorded_episodes"] = len(rows)
            aggregate["skip_reasons"] = {}
            for row in rows:
                reasons = row.get("skip_reasons", {})
                if not isinstance(reasons, Mapping):
                    continue
                for reason, count in reasons.items():
                    aggregate["skip_reasons"][str(reason)] = (
                        aggregate["skip_reasons"].get(str(reason), 0) + int(count or 0)
                    )
            per_input: dict[str, dict[str, Any]] = {}
            for row in rows:
                row_delta = row.get("per_input_delta")
                if not isinstance(row_delta, Mapping):
                    continue
                for input_index, values in row_delta.items():
                    if not isinstance(values, Mapping):
                        continue
                    key = str(input_index)
                    target = per_input.setdefault(
                        key,
                        {
                            "applied_count": 0,
                            "delta_abs_count": 0,
                            "delta_abs_weighted_sum": 0.0,
                            "delta_abs_max": None,
                            "tolerance_values": set(),
                            "tolerance_observed_count": 0,
                            "clip_count": 0,
                        },
                    )
                    applied_count = int(values.get("applied_count", 0) or 0)
                    delta_count = int(values.get("delta_abs_count", 0) or 0)
                    target["applied_count"] += applied_count
                    target["delta_abs_count"] += delta_count
                    delta_mean = values.get("delta_abs_mean")
                    if delta_mean is not None and delta_count:
                        target["delta_abs_weighted_sum"] += float(delta_mean) * delta_count
                    delta_max = values.get("delta_abs_max")
                    if delta_max is not None:
                        target["delta_abs_max"] = (
                            float(delta_max)
                            if target["delta_abs_max"] is None
                            else max(float(target["delta_abs_max"]), float(delta_max))
                        )
                    for tolerance in values.get("tolerance_values", ()) or ():
                        try:
                            target["tolerance_values"].add(float(tolerance))
                        except (TypeError, ValueError):
                            continue
                    target["tolerance_observed_count"] += int(
                        values.get("tolerance_observed_count", 0) or 0
                    )
                    target["clip_count"] += int(values.get("clip_count", 0) or 0)
            aggregate["per_input_delta"] = {}
            for input_index in sorted(per_input, key=lambda value: int(value)):
                values = per_input[input_index]
                tolerance_values = sorted(values["tolerance_values"])
                delta_count = values["delta_abs_count"]
                aggregate["per_input_delta"][input_index] = {
                    "applied_count": values["applied_count"],
                    "delta_abs_count": delta_count,
                    "delta_abs_mean": (
                        values["delta_abs_weighted_sum"] / delta_count
                        if delta_count
                        else None
                    ),
                    "delta_abs_max": values["delta_abs_max"],
                    "tolerance_observed_count": values["tolerance_observed_count"],
                    "tolerance_values": tolerance_values,
                    "tolerance": tolerance_values[0] if len(tolerance_values) == 1 else None,
                    "clip_count": values["clip_count"],
                }
            aggregate["applied_rate_over_target"] = (
                aggregate["applied_count"] / aggregate["target_step_count"]
                if aggregate["target_step_count"]
                else None
            )
            aggregate["changed_rate_over_applied"] = (
                aggregate["changed_count_exact"] / aggregate["applied_count"]
                if aggregate["applied_count"]
                else None
            )
            return aggregate

        def paired_summary(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
            verified_rows: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
            missing_reasons: dict[str, int] = {}
            for episode in episodes:
                status = episode.get("paired_p00_status")
                paired = episode.get("metrics", {}).get("paired_p00") if isinstance(episode.get("metrics", {}), Mapping) else None
                if (
                    status == "matched"
                    and isinstance(paired, Mapping)
                    and _pair_execution_eligible(episode)
                ):
                    verified_rows.append((paired, episode))
                    continue
                if status is None:
                    # P00 itself has no pair; for other legacy rows this means
                    # pairing was not recorded and must remain N/A.
                    reason = "paired_status_missing"
                else:
                    reason = str(episode.get("paired_p00_missing_reason") or status)
                # A pair can have more than one independent missing condition
                # (for example both P00 and the intervention were budget
                # censored). Preserve each reason as its own aggregate key so
                # consumers can count the affected side without parsing a
                # concatenated human-readable field.
                reasons = [item.strip() for item in reason.split(";") if item.strip()]
                for missing_reason in reasons or [reason]:
                    missing_reasons[missing_reason] = missing_reasons.get(missing_reason, 0) + 1
            fields = (
                "lane_rms_m",
                "lane_max_abs_m",
                "departure_count",
                "departure_time_s",
                "progress_m",
                "duration_s",
                "speed_mean_m_s",
                "low_speed_duration_s",
                "action_switch_count",
                "steering_variation_sum",
                "cumulative_reward",
            )
            episode_mean: dict[str, float | None] = {}
            step_weighted: dict[str, float | None] = {}
            for field_name in fields:
                values: list[float] = []
                weighted: list[tuple[float, int]] = []
                for row, episode in verified_rows:
                    value = row.get(field_name)
                    try:
                        parsed = float(value)
                    except (TypeError, ValueError):
                        continue
                    if not np.isfinite(parsed):
                        continue
                    values.append(parsed)
                    try:
                        weight = max(1, int(episode.get("record_count", 1)))
                    except (TypeError, ValueError):
                        weight = 1
                    weighted.append((parsed, weight))
                episode_mean[field_name] = float(np.mean(values)) if values else None
                total_weight = sum(weight for _, weight in weighted)
                step_weighted[field_name] = (
                    float(sum(value * weight for value, weight in weighted) / total_weight)
                    if total_weight
                    else None
                )
            return {
                "status": "available" if verified_rows else "unavailable",
                "verified_pair_count": len(verified_rows),
                "missing_pair_count": len(episodes) - len(verified_rows),
                "missing_reasons": missing_reasons,
                "episode_mean": episode_mean,
                "step_weighted": step_weighted,
            }

        def video_summary(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
            rows = [
                item.get("video")
                for item in episodes
                if isinstance(item.get("video"), Mapping)
            ]
            reason_counts: dict[str, int] = {}
            for row in rows:
                reason = row.get("reason")
                if reason:
                    reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1
            return {
                "episodes": len(episodes),
                "requested_episodes": sum(bool(row.get("requested")) for row in rows),
                "enabled_episodes": sum(bool(row.get("enabled")) for row in rows),
                "generated_episodes": sum(
                    str(row.get("status", "")).casefold()
                    in {"saved", "generated", "frames_saved_without_store"}
                    for row in rows
                ),
                "frame_count": int(sum(int(row.get("frame_count", 0) or 0) for row in rows)),
                "not_generated_reasons": reason_counts,
            }

        def execution_group_summary(
            label: str,
            episodes: Sequence[Mapping[str, Any]],
            episode_indices: Sequence[int],
        ) -> dict[str, Any]:
            """Summarize one termination class without changing legacy totals."""

            status_counts: dict[str, int] = {}
            terminal_counts: dict[str, int] = {}
            assessment_counts: dict[str, int] = {}
            episode_ids: list[Any] = []
            for episode in episodes:
                episode_id = episode.get("episode_id", episode.get("episode"))
                episode_ids.append(episode_id)
                for field_name, counts in (
                    ("execution_status", status_counts),
                    ("terminal_reason", terminal_counts),
                    ("assessment_status", assessment_counts),
                ):
                    value = episode.get(field_name)
                    if value in (None, ""):
                        value = "unknown"
                    text_value = str(value)
                    counts[text_value] = counts.get(text_value, 0) + 1
            aggregate = metric_summary(episodes)
            aggregate_counts = aggregate["counts"]
            return {
                "classification": label,
                "episode_indices": [int(index) for index in episode_indices],
                "episode_ids": episode_ids,
                "episodes": len(episodes),
                "records": int(aggregate_counts["records"]),
                "metrics": [
                    episode.get("metrics", {})
                    for episode in episodes
                ],
                "metric_summary": aggregate,
                "intervention_counts": intervention_summary(episodes),
                "counts": {
                    "episodes": len(episodes),
                    "records": int(aggregate_counts["records"]),
                    "poststeps": int(aggregate_counts["poststeps"]),
                    "execution_status": status_counts,
                    "terminal_reason": terminal_counts,
                    "assessment_status": assessment_counts,
                },
                "reasons": {
                    "execution_status": status_counts,
                    "terminal_reason": terminal_counts,
                    "assessment_status": assessment_counts,
                },
            }

        pattern_summary: dict[str, Any] = {}
        for pattern_id, episodes in self.patterns.items():
            metrics = [episode.get("metrics", {}) for episode in episodes]
            diagnostic_aggregate = metric_summary(episodes)
            natural_indices = [
                index
                for index, episode in enumerate(episodes)
                if _pair_execution_eligible(episode)
            ]
            partial_indices = [
                index
                for index, episode in enumerate(episodes)
                if not _pair_execution_eligible(episode)
            ]
            partial_unknown_indices = [
                index
                for index, episode in enumerate(episodes)
                if _pair_execution_is_partial_unknown(episode)
            ]
            natural_episodes = [episodes[index] for index in natural_indices]
            partial_episodes = [episodes[index] for index in partial_indices]
            partial_unknown_episodes = [episodes[index] for index in partial_unknown_indices]
            # Runtime-produced episodes carry execution metadata.  Once that
            # metadata exists, the pattern-level metrics are the primary
            # evaluation of naturally completed episodes only; raw metrics
            # from failed/aborted/budget-censored episodes remain available in
            # the execution groups and diagnostic aggregate below.  Preserve
            # the legacy direct-constructed result shape when no episode has
            # execution metadata at all, since such records cannot be
            # classified without inventing a failure state.
            has_execution_metadata = any(
                episode.get("execution_status") not in (None, "")
                or episode.get("terminal_reason") not in (None, "")
                for episode in episodes
            )
            primary_indices = (
                natural_indices
                if has_execution_metadata
                else list(range(len(episodes)))
            )
            primary_episodes = [episodes[index] for index in primary_indices]
            aggregate = metric_summary(primary_episodes)
            execution_group_counts = {
                "natural_completion": len(natural_episodes),
                "partial_or_interrupted": len(partial_episodes),
            }
            execution_groups = {
                "natural_completion": execution_group_summary(
                    "natural_completion",
                    natural_episodes,
                    natural_indices,
                ),
                "partial_or_interrupted": execution_group_summary(
                    "partial_or_interrupted",
                    partial_episodes,
                    partial_indices,
                ),
            }
            # Keep the legacy two-group shape when there are no ambiguous
            # records, while exposing an explicit subset for missing/unknown
            # execution metadata when it occurs.
            if partial_unknown_episodes:
                execution_group_counts["partial_unknown"] = len(partial_unknown_episodes)
                execution_groups["partial_unknown"] = execution_group_summary(
                    "partial_unknown",
                    partial_unknown_episodes,
                    partial_unknown_indices,
                )
            pattern_summary[pattern_id] = {
                "episodes": len(episodes),
                "records": sum(int(item.get("record_count", 0)) for item in episodes),
                "metrics": metrics,
                "metric_summary": aggregate,
                "diagnostic_metric_summary": diagnostic_aggregate,
                "metric_summary_scope": (
                    "natural_completion"
                    if has_execution_metadata
                    else "legacy_all_episodes"
                ),
                "primary_metric_episode_indices": [int(index) for index in primary_indices],
                "excluded_metric_episode_indices": [
                    int(index)
                    for index in range(len(episodes))
                    if index not in primary_indices
                ],
                # Keep the two weighting choices explicit at the pattern level
                # for report readers that do not need the nested counts.
                "episode_mean": aggregate["episode_mean"],
                "step_weighted": aggregate["step_weighted"],
                "weighted_episode_mean": aggregate["weighted_episode_mean"],
                "sum": aggregate["sum"],
                "aggregation": aggregate["aggregation"],
                "counts": aggregate["counts"],
                "intervention_counts": intervention_summary(episodes),
                "execution_group_counts": execution_group_counts,
                "execution_groups": execution_groups,
                "video": video_summary(episodes),
                "execution_status": sorted({str(item.get("execution_status", "unknown")) for item in episodes}),
                "assessment_status": sorted({str(item.get("assessment_status", "unknown")) for item in episodes}),
                "paired_p00": (
                    {
                        "status": "baseline_control",
                        "verified_pair_count": 0,
                        "missing_pair_count": 0,
                        "missing_reasons": {},
                        "episode_mean": {},
                        "step_weighted": {},
                    }
                    if pattern_id == "P00"
                    else paired_summary(episodes)
                ),
            }
        all_episodes = [
            (pattern_id, episode)
            for pattern_id, episodes in self.patterns.items()
            for episode in episodes
        ]
        execution_counts = {
            "completed": 0,
            "failed": 0,
            "aborted": 0,
        }
        for _, episode in all_episodes:
            status = str(episode.get("execution_status") or "unknown").casefold()
            if status in execution_counts:
                execution_counts[status] += 1
        runtime_failure_count = execution_counts["failed"] + execution_counts["aborted"]
        pattern_status_ids: dict[str, list[str]] = {
            "completed": [],
            "failed": [],
            "aborted": [],
            "unknown": [],
        }
        for pattern_id, episodes in self.patterns.items():
            statuses = {
                str(episode.get("execution_status") or "unknown").casefold()
                for episode in episodes
            }
            if statuses and "aborted" in statuses:
                classification = "aborted"
            elif statuses and "failed" in statuses:
                classification = "failed"
            elif statuses and statuses <= {"completed"}:
                classification = "completed"
            else:
                classification = "unknown"
            pattern_status_ids[classification].append(pattern_id)
        for values in pattern_status_ids.values():
            values.sort()
        counts = {
            "pattern_count": len(self.patterns),
            "episode_count": len(all_episodes),
            "completed_episode_count": execution_counts["completed"],
            "failed_episode_count": execution_counts["failed"],
            "aborted_episode_count": execution_counts["aborted"],
            "runtime_failure_count": runtime_failure_count,
            "execution_failure_count": runtime_failure_count,
            "unexpected_interruption_count": execution_counts["aborted"],
            "completed_pattern_count": len(pattern_status_ids["completed"]),
            "failed_pattern_count": len(pattern_status_ids["failed"]),
            "aborted_pattern_count": len(pattern_status_ids["aborted"]),
            "unknown_pattern_count": len(pattern_status_ids["unknown"]),
            "completed_pattern_ids": pattern_status_ids["completed"],
            "failed_pattern_ids": pattern_status_ids["failed"],
            "aborted_pattern_ids": pattern_status_ids["aborted"],
            "runtime_failure_pattern_count": (
                len(pattern_status_ids["failed"]) + len(pattern_status_ids["aborted"])
            ),
            "runtime_failure_pattern_ids": sorted(
                pattern_status_ids["failed"] + pattern_status_ids["aborted"]
            ),
            "required_experiment_failed": bool(runtime_failure_count),
        }
        overall_status = (
            "success"
            if runtime_failure_count == 0
            else "failed"
            if execution_counts["completed"] == 0
            else "partial_failure"
        )
        return {
            "store": str(self.store.run_dir) if self.store else None,
            "p00_reference_episodes": len(self.baseline_references),
            "policy_fingerprint_before": self.policy_fingerprint_before,
            "policy_fingerprint_after": self.policy_fingerprint_after,
            "policy_unchanged": self.policy_unchanged,
            "status": overall_status,
            "counts": counts,
            "patterns": pattern_summary,
        }


def _thresholds(config: Any) -> dict[str, Any]:
    section = getattr(config, "closed_loop", {}) or {}
    return {
        "departure_tolerance_ratio": float(section.get("departure_tolerance_ratio", 0.05)),
        "departure_consecutive_steps": int(section.get("departure_consecutive_steps", 1)),
        "low_speed_m_s": float(section.get("low_speed_m_s", 0.5)),
    }


def _flag_true(sources: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    for source in sources:
        for name in names:
            value = source.get(name)
            if value is True or (isinstance(value, np.bool_) and bool(value)):
                return True
            if isinstance(value, (int, np.integer, float, np.floating)) and not isinstance(value, bool) and value == 1:
                return True
            if isinstance(value, str) and value.strip().lower() in {"true", "yes", "1"}:
                return True
    return False


def _terminal_reason(
    terminated: bool,
    truncated: bool,
    step_info: Mapping[str, Any],
    post_telemetry: Mapping[str, Any],
) -> str:
    """Resolve a reportable end reason while retaining raw Gymnasium flags."""

    sources = (step_info, post_telemetry)
    if _flag_true(
        sources,
        ("wrong_lane_arrival", "wrong_lane_goal", "arrive_wrong_lane", "goal_wrong_lane"),
    ):
        return "wrong_lane_arrival"
    if _flag_true(sources, ("arrive_dest", "arrived", "success", "goal_reached")):
        return "arrive_dest"
    if _flag_true(sources, ("out_of_road", "out_of_drivable", "road_out", "crash_out_of_road")):
        return "out_of_road"
    if _flag_true(sources, ("crash", "crashed", "collision", "collision_occurred")):
        return "crash"
    if _flag_true(
        sources,
        ("start_lane_departure", "departed_start_lane", "start_lane_departed"),
    ):
        return "start_lane_departure"
    if truncated:
        if _flag_true(
            sources,
            ("horizon", "horizon_reached", "time_limit", "time_limit_reached", "timeout", "TimeLimit.truncated"),
        ):
            return "horizon"
        return "truncated"
    if terminated:
        return "terminated"
    return "unknown"


def _pattern_execution_policy(pattern: InterventionPattern) -> tuple[str, str]:
    """Return the declared B scope and inapplicable handling policy.

    Legacy/reference patterns remain explicitly conditional for compatibility.
    A pattern opts into the stricter full-episode contract with
    ``scope = "full_episode"``; unless it explicitly declares a continuation
    policy, an unexpected skip aborts that pattern before ``env.step``.
    """

    metadata = pattern.metadata if isinstance(pattern.metadata, Mapping) else {}
    scope = getattr(pattern, "scope", None) or metadata.get("scope") or "explicitly_conditional"
    scope = str(scope).casefold().replace("-", "_")
    if scope not in {"full_episode", "explicitly_conditional"}:
        scope = "explicitly_conditional"
    configured = getattr(pattern, "on_inapplicable", None) or metadata.get("on_inapplicable")
    if configured is None:
        configured = "abort_pattern" if scope == "full_episode" else "continue_unmodified_with_warning"
    action = str(configured).casefold().replace("-", "_")
    if action not in {"abort_pattern", "continue_unmodified_with_warning"}:
        action = "abort_pattern" if scope == "full_episode" else "continue_unmodified_with_warning"
    return scope, action


_VIDEO_PATTERNS_UNSET = object()


def _video_pattern_selector(video_config: Mapping[str, Any]) -> tuple[frozenset[str] | None, str]:
    """Resolve optional per-pattern B video selection.

    ``None`` means the legacy behaviour: when video is enabled, every selected
    closed-loop pattern captures frames.  An explicit ``false`` disables all
    B videos, while a list limits capture to those pattern identifiers.  The
    config validator owns the public shape; this runtime check keeps direct
    callers fail-closed when they bypass config loading.
    """

    configured = video_config.get("patterns", _VIDEO_PATTERNS_UNSET)
    if configured is _VIDEO_PATTERNS_UNSET or configured is True:
        return None, "all_patterns"
    if configured is False:
        return frozenset(), "disabled_by_config"
    if isinstance(configured, (str, bytes)):
        values = [configured]
    elif isinstance(configured, Sequence) and not isinstance(configured, Mapping):
        values = list(configured)
    else:
        raise ClosedLoopError("video.patterns must be false or an array of pattern identifiers")
    identifiers: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ClosedLoopError("video.patterns entries must be non-empty strings")
        identifiers.add(value.strip())
    return frozenset(identifiers), "configured_patterns"


def _video_state_for_pattern(
    identifier: str,
    *,
    requested: bool,
    selector: frozenset[str] | None,
    selector_status: str,
) -> tuple[bool, str | None]:
    """Return whether one B pattern captures video and, if not, why."""

    if not requested:
        return False, "video_disabled"
    if selector is None:
        return True, None
    if identifier in selector:
        return True, None
    if selector_status == "disabled_by_config":
        return False, "disabled_by_video_patterns"
    return False, "pattern_not_selected_by_video_patterns"


def _declared_out_of_scope(
    pattern: InterventionPattern,
    intervention: InterventionResult,
) -> bool:
    """Identify a predeclared inapplicable state without hiding failures."""

    for name in ("out_of_scope", "inapplicable", "declared_inapplicable"):
        value = getattr(intervention, name, None)
        if value is not None:
            return bool(value)
    category = getattr(intervention, "skip_category", None)
    if category is not None and str(category).casefold().replace("-", "_") in {
        "out_of_scope",
        "inapplicable",
        "declared_inapplicable",
    }:
        return True
    metadata = pattern.metadata if isinstance(pattern.metadata, Mapping) else {}
    if metadata.get("declared_inapplicable") is True or metadata.get("out_of_scope") is True:
        return True
    reason = str(getattr(intervention, "skip_reason", "") or "").casefold()
    configured_reasons = metadata.get(
        "out_of_scope_reasons",
        metadata.get("declared_inapplicable_reasons", ()),
    )
    if isinstance(configured_reasons, str):
        configured_reasons = (configured_reasons,)
    if isinstance(configured_reasons, Sequence) and not isinstance(configured_reasons, (str, bytes)):
        if any(str(value).casefold() in reason for value in configured_reasons):
            return True
    if any(token in reason for token in ("out_of_scope", "out-of-scope", "declared inapplicable")):
        return True
    # A legacy reference pattern is conditional by definition.  A donor road
    # or reference context mismatch is an expected out-of-scope step, while a
    # typed full-episode operation failing its precondition remains a failure.
    if getattr(pattern, "scope", None) == "explicitly_conditional" and (
        "compatibility context" in reason
        or "reference context" in reason
        or "reference observation" in reason
    ):
        return True
    return False


def _pattern_target_step_count(config: Any, max_steps: int | None) -> int | None:
    if max_steps is not None:
        return max(0, int(max_steps))
    scenario = getattr(config, "scenario", {}) or {}
    if isinstance(scenario, Mapping) and scenario.get("horizon") is not None:
        try:
            return max(0, int(scenario["horizon"]))
        except (TypeError, ValueError):
            return None
    return None


def _pair_execution_eligible(episode: Mapping[str, Any]) -> bool:
    """Allow P00 pairing only for complete, naturally terminated episodes."""

    if str(episode.get("execution_status", "")).casefold() != "completed":
        return False
    terminal_reason = episode.get("terminal_reason")
    if str(terminal_reason or "").casefold() not in {
        "terminated",
        "horizon",
        "arrive_dest",
        "wrong_lane_arrival",
        "out_of_road",
        "crash",
        "start_lane_departure",
    }:
        return False
    if any(
        isinstance(record, Mapping) and bool(record.get("budget_truncated"))
        for record in episode.get("records", ())
    ):
        return False
    return True


def _pair_execution_is_partial_unknown(episode: Mapping[str, Any]) -> bool:
    """Identify missing/ambiguous execution metadata separately from aborts."""

    status = str(episode.get("execution_status") or "").casefold()
    terminal_reason = str(episode.get("terminal_reason") or "").casefold()
    if not status or not terminal_reason:
        return True
    if terminal_reason in {"unknown", "truncated"}:
        return True
    if status == "completed" and terminal_reason not in {
        "terminated",
        "horizon",
        "arrive_dest",
        "wrong_lane_arrival",
        "out_of_road",
        "crash",
        "start_lane_departure",
        "budget_censored",
        "intervention_abort",
        "runtime_error",
        "failed",
    }:
        return True
    return False


def _selected_policy_change(
    original_probabilities: np.ndarray,
    probabilities: np.ndarray,
    original_action: int,
) -> dict[str, Any]:
    comparison = compare_distributions(
        np.asarray(original_probabilities, dtype=np.float64),
        np.asarray(probabilities, dtype=np.float64),
        selected_actions=np.asarray([original_action], dtype=np.int64),
    )
    return {
        "selected_action": int(comparison.selected_actions[0]),
        "selected_probability_original": float(comparison.selected_probability_original[0]),
        "selected_probability_changed": float(comparison.selected_probability_changed[0]),
        "selected_probability_delta": float(comparison.selected_probability_delta[0]),
        "selected_probability_delta_pp": float(comparison.selected_probability_delta_pp[0]),
        "selected_probability_abs_delta": float(comparison.selected_probability_abs_delta[0]),
        "selected_probability_abs_delta_pp": float(comparison.selected_probability_abs_delta_pp[0]),
        "js_divergence": float(comparison.js[0]),
        "action_changed": bool(comparison.action_changed[0]),
    }


def _policy_fingerprint(policy: Any, adapter: Any) -> str | None:
    """Read a stable policy fingerprint when the runtime exposes one."""

    seen: set[int] = set()
    for target in (policy, adapter):
        if target is None or id(target) in seen:
            continue
        seen.add(id(target))
        fingerprint = getattr(target, "fingerprint", None)
        if not callable(fingerprint):
            continue
        try:
            value = fingerprint()
        except Exception as exc:
            raise ClosedLoopError(
                f"failed to fingerprint policy before/after closed-loop execution: {exc}"
            ) from exc
        return str(value)
    return None


def run_closed_loop(
    policy: Any,
    adapter: Any,
    config: Any,
    *,
    patterns: Sequence[Any] | None = None,
    store: RunArtifacts | None = None,
    episodes: int | None = None,
    max_steps: int | None = None,
    scenario_seed: int | None = None,
    rl_seed: int | None = None,
    deterministic: bool = True,
    schema: InputSchema | None = None,
    reference_observation: Any | None = None,
    reference_observations: Mapping[str, Any] | None = None,
    reference_context: Mapping[str, Any] | None = None,
    reference_contexts: Mapping[str, Mapping[str, Any]] | None = None,
    reference_records: Any | None = None,
    video: bool | None = None,
    video_dir: str | Path | None = None,
    video_fps: float | None = None,
) -> ClosedLoopResult:
    """Run selected patterns in separate, sequentially closed environments."""

    selected = _pattern_mapping(config, patterns)
    try:
        resolved_patterns = list(patterns_from_config(selected))
    except Exception as exc:
        raise ClosedLoopError(f"pattern configuration is invalid: {exc}") from exc
    if not any(pattern.pattern_id == "P00" for pattern in resolved_patterns):
        raise ClosedLoopError("closed-loop patterns must explicitly include P00 baseline")
    resolved_patterns.sort(key=lambda pattern: (pattern.pattern_id != "P00", pattern.pattern_id))
    if schema is None:
        schema = getattr(config, "schema_object", None)
    if schema is None:
        schema_path = getattr(config, "schema_path", None)
        if schema_path is not None:
            try:
                from .schema import load_schema

                schema = load_schema(schema_path)
            except Exception as exc:
                raise ClosedLoopError(f"schemaを読み込めません: {exc}") from exc
    if schema is None:
        raise ClosedLoopError("closed-loopにはInputSchemaが必要です")
    if episodes is None:
        episodes = int((getattr(config, "closed_loop", {}) or {}).get("episodes", 1))
    if episodes <= 0:
        raise ClosedLoopError("episodesは1以上で指定してください")
    if video is None:
        video = bool((getattr(config, "video", {}) or {}).get("enabled", False))
    video_config = getattr(config, "video", {}) or {}
    if not isinstance(video_config, Mapping):
        raise ClosedLoopError("video configuration must be a mapping")
    video_pattern_selector, video_pattern_selector_status = _video_pattern_selector(video_config)
    configured_video_fps = video_config.get("fps") if isinstance(video_config, Mapping) else None
    if video_fps is None and configured_video_fps is not None:
        try:
            video_fps = float(configured_video_fps)
        except (TypeError, ValueError):
            raise ClosedLoopError(f"video.fps must be numeric: {configured_video_fps!r}") from None
    thresholds = _thresholds(config)
    policy_fingerprint_before = _policy_fingerprint(policy, adapter)
    result = ClosedLoopResult(
        store,
        policy_fingerprint_before=policy_fingerprint_before,
    )
    explicit_reference_rows = _reference_records_by_episode(reference_records)
    if store:
        store.update_status("closed_loop", "running", pattern_ids=[pattern.pattern_id for pattern in resolved_patterns])
    try:
        for pattern in resolved_patterns:
            identifier = pattern.pattern_id
            pattern_scope, pattern_on_inapplicable = _pattern_execution_policy(pattern)
            pattern_video, pattern_video_reason = _video_state_for_pattern(
                identifier,
                requested=bool(video),
                selector=video_pattern_selector,
                selector_status=video_pattern_selector_status,
            )
            result.patterns.setdefault(identifier, [])
            for episode_index in range(episodes):
                episode_dir = store.closed_loop_dir / identifier / f"episode-{episode_index}" if store else None
                if episode_dir is not None:
                    if episode_dir.exists():
                        raise ClosedLoopError(f"closed-loop episode output already exists: {episode_dir}")
                    episode_dir.mkdir(parents=True, exist_ok=False)
                if pattern_video:
                    if video_dir is not None:
                        episode_video_dir = Path(video_dir) / identifier / f"episode-{episode_index}"
                    elif episode_dir is not None:
                        episode_video_dir = episode_dir / "video"
                    else:
                        episode_video_dir = Path("outputs/input_attribution_video") / identifier / f"episode-{episode_index}"
                else:
                    episode_video_dir = None
                env = make_environment(adapter, config)
                seed = scenario_seed if scenario_seed is not None else getattr(config, "scenario_seed", None)
                episode_seed = int(seed) + episode_index if seed is not None else None
                episode_rl_seed = int(rl_seed) if rl_seed is not None else getattr(config, "rl_seed", None)
                observations: list[np.ndarray] = []
                modified_observations: list[np.ndarray] = []
                records: list[dict[str, Any]] = []
                frames: list[dict[str, Any]] = []
                terminal_reason = "unknown"
                current_phase = "episode_start"
                runtime_failure_phase: str | None = None
                runtime_failure_reason: str | None = None
                episode_result_saved = False
                try:
                    current_phase = "runtime_seed"
                    seed_metadata = seed_runtime(adapter, env, episode_rl_seed)
                    current_phase = "reset"
                    raw_obs, reset_info = reset_environment(adapter, env, episode_seed)
                    done = False
                    step = 0
                    seed_verified = episode_seed is None
                    initial_snapshot: dict[str, Any] | None = None
                    step_context: dict[str, Any] = {}
                    while not done:
                        step_context = {
                            "step": step,
                            "raw_observation": _copy(raw_obs),
                            "reset_info": _plain(reset_info),
                            "record_saved": False,
                        }
                        current_phase = "pre_telemetry"
                        pre = telemetry(adapter, env, reset_info, phase="pre", step=step)
                        step_context["pre_telemetry"] = pre
                        current_phase = "seed_verification"
                        actual_scenario_seed = _actual_scenario_seed(env, reset_info, pre)
                        step_context["actual_scenario_seed"] = actual_scenario_seed
                        if not seed_verified:
                            if actual_scenario_seed is None or actual_scenario_seed != episode_seed:
                                raise ClosedLoopError(
                                    f"environment did not apply requested scenario seed {episode_seed}: "
                                    f"actual={actual_scenario_seed!r}"
                                )
                            seed_verified = True
                        current_phase = "preprocess"
                        original_input, preprocess_info = model_input(adapter, raw_obs, reset_info)
                        step_context["original_input"] = np.array(original_input, copy=True)
                        step_context["preprocess_info"] = preprocess_info
                        try:
                            current_phase = "schema_validation"
                            schema.validate_observation(original_input, copy=False)
                        except Exception as exc:
                            raise ClosedLoopError(f"original model input violates schema at step {step}: {exc}") from exc
                        current_snapshot = {
                            "scenario_seed": episode_seed,
                            "rl_seed": episode_rl_seed,
                            "requested_scenario_seed": episode_seed,
                            "requested_rl_seed": episode_rl_seed,
                            "actual_scenario_seed": actual_scenario_seed,
                            "observation_hash": _observation_hash(original_input),
                            "snapshot": _state_snapshot(adapter, env, pre),
                        }
                        if initial_snapshot is None:
                            initial_snapshot = current_snapshot
                            if identifier == "P00":
                                # Store only metadata here; arrays remain in the
                                # dedicated reference npy file below.  A P00
                                # self-check still requires explicit geometry.
                                current_snapshot["initial_match"] = _compare_initial(
                                    current_snapshot,
                                    current_snapshot,
                                    baseline_observation=original_input,
                                    current_observation=original_input,
                                    scenario_seed=episode_seed,
                                    rl_seed=episode_rl_seed,
                                )
                            elif episode_index in result.baseline_references:
                                baseline_reference = result.baseline_references[episode_index]
                                current_snapshot["initial_match"] = _compare_initial(
                                    baseline_reference["initial_snapshot"],
                                    current_snapshot,
                                    baseline_observation=np.asarray(baseline_reference["initial_observation"]),
                                    current_observation=original_input,
                                    scenario_seed=episode_seed,
                                    rl_seed=episode_rl_seed,
                                )
                                if current_snapshot["initial_match"]["status"] == "mismatch":
                                    raise ClosedLoopError(
                                        f"pattern {identifier} episode {episode_index} initial state differs from P00: "
                                        f"{current_snapshot['initial_match']['reasons']}"
                                    )
                            else:
                                current_snapshot["initial_match"] = {
                                    "status": "unavailable",
                                    "matched": None,
                                    "reasons": ["P00 baseline has not completed"],
                                }
                        current_phase = "intervention"
                        intervention = apply_pattern(
                            original_input,
                            pattern,
                            schema,
                            reference_observation=reference_observation,
                            reference_observations=reference_observations,
                            reference_context=reference_context,
                            reference_contexts=reference_contexts,
                            observation_context=pre,
                            strict=False,
                        )
                        step_context["intervention"] = intervention
                        current_phase = "policy"
                        changed_input = np.array(intervention.observation, copy=True)
                        step_context["changed_input"] = np.array(changed_input, copy=True)
                        probabilities = policy_probabilities(policy, changed_input)
                        step_context["probabilities"] = np.array(probabilities, copy=True)
                        original_probabilities = policy_probabilities(policy, original_input)
                        step_context["original_probabilities"] = np.array(original_probabilities, copy=True)
                        action = policy_predict(policy, changed_input, deterministic=deterministic)
                        step_context["action"] = _plain(action)
                        original_action = policy_predict(policy, original_input, deterministic=deterministic)
                        step_context["original_action"] = _plain(original_action)
                        if not 0 <= int(action) < probabilities.size:
                            raise ClosedLoopError(f"modified policy action is outside Discrete({probabilities.size}): {action}")
                        if not 0 <= int(original_action) < original_probabilities.size:
                            raise ClosedLoopError(
                                f"original policy action is outside Discrete({original_probabilities.size}): {original_action}"
                            )
                        intervention_skipped = bool(getattr(intervention, "skipped", False))
                        declared_out_of_scope = _declared_out_of_scope(pattern, intervention)
                        abort_intervention = bool(
                            intervention_skipped
                            and pattern_on_inapplicable == "abort_pattern"
                            and not declared_out_of_scope
                        )
                        # Keep the env.step call, its successful return, post
                        # telemetry, and action decoding as separate phases.
                        # A simulator may have advanced before raising, so an
                        # env-step exception is recorded as physically unknown
                        # and is never counted as a returned step.
                        current_phase = "env_step"
                        env_step_attempted = not abort_intervention
                        env_step_returned = False
                        env_step_error: str | None = None
                        post_telemetry_attempted = False
                        post_telemetry_error: str | None = None
                        action_decode_error: str | None = None
                        step_failure_phase: str | None = None
                        step_failure_reason: str | None = None
                        next_raw = raw_obs
                        reward: Any = None
                        terminated: bool | None = False if abort_intervention else None
                        truncated: bool | None = False if abort_intervention else None
                        step_info: Mapping[str, Any] = {}
                        post: Mapping[str, Any] | None = None
                        decoded_action: Any = None
                        if abort_intervention:
                            # Preserve the attempted input and reason while
                            # guaranteeing that this pattern never advances
                            # the physical environment after an unexpected
                            # full-episode inapplicability.
                            pass
                        else:
                            try:
                                next_raw, reward, terminated, truncated, step_info = step_environment(env, action)
                                env_step_returned = True
                                step_context.update(
                                    {
                                        "next_raw": _copy(next_raw),
                                        "reward": _plain(reward),
                                        "terminated": bool(terminated),
                                        "truncated": bool(truncated),
                                        "step_info": _plain(step_info),
                                        "env_step_returned": True,
                                    }
                                )
                            except Exception as exc:
                                env_step_error = f"{type(exc).__name__}: {exc}"
                                step_context["env_step_error"] = env_step_error
                                step_failure_phase = "env_step"
                                step_failure_reason = env_step_error
                            if env_step_returned:
                                current_phase = "post_telemetry"
                                post_telemetry_attempted = True
                                try:
                                    post = telemetry(adapter, env, step_info, phase="post", step=step + 1)
                                    step_context["post_telemetry"] = post
                                except Exception as exc:
                                    post_telemetry_error = f"{type(exc).__name__}: {exc}"
                                    step_failure_phase = "post_telemetry"
                                    step_failure_reason = post_telemetry_error
                                if post_telemetry_error is None:
                                    current_phase = "action_decode"
                                    try:
                                        decoded_action = decode_action(adapter, action, config=config)
                                        step_context["decoded_action"] = decoded_action
                                    except Exception as exc:
                                        action_decode_error = f"{type(exc).__name__}: {exc}"
                                        step_failure_phase = "action_decode"
                                        step_failure_reason = action_decode_error
                        current_phase = "record"
                        pre_time = telemetry_time(pre, step)
                        post_time = telemetry_time(post, step + 1) if post is not None else None
                        selected_change = _selected_policy_change(
                            original_probabilities,
                            probabilities,
                            int(original_action),
                        )
                        requested_indices = tuple(
                            int(value) for value in intervention.requested_indices
                        )
                        original_values = {
                            index: _plain(original_input[index]) for index in requested_indices
                        }
                        modified_values = {
                            index: _plain(changed_input[index]) for index in requested_indices
                        }
                        delta_values = {
                            index: _plain(
                                np.asarray(changed_input[index], dtype=np.float64)
                                - np.asarray(original_input[index], dtype=np.float64)
                            )
                            for index in requested_indices
                        }
                        intervention_payload = intervention.to_dict()
                        # Result objects from older cores default ``eligible``
                        # to true even when a conditional reference is outside
                        # its declared context.  Normalize the serialized B
                        # record from the execution decision so denominators
                        # describe this run rather than the donor schema.
                        intervention_payload["eligible"] = bool(
                            not intervention_skipped or not declared_out_of_scope
                        )
                        intervention_payload["applied"] = bool(not intervention_skipped)
                        intervention_payload["scope"] = pattern_scope
                        intervention_payload["on_inapplicable"] = pattern_on_inapplicable
                        if getattr(intervention, "skip_category", None) is not None:
                            intervention_payload["skip_category"] = _plain(
                                getattr(intervention, "skip_category")
                            )
                        record = {
                            "pattern_id": identifier,
                            "episode": episode_index,
                            "episode_id": f"episode-{episode_index}",
                            "scenario_seed": episode_seed,
                            "rl_seed": episode_rl_seed,
                            "requested_scenario_seed": episode_seed,
                            "requested_rl_seed": episode_rl_seed,
                            "actual_scenario_seed": actual_scenario_seed,
                            "seed_metadata": seed_metadata,
                            "step": step,
                            "observation_index": len(observations),
                            "observation_hash": _observation_hash(original_input),
                            "modified_observation_hash": _observation_hash(changed_input),
                            "observation_shape": list(original_input.shape),
                            "observation_dtype": str(original_input.dtype),
                            "pre_time": _plain(pre_time),
                            "post_time": _plain(post_time),
                            "probabilities": probabilities.tolist(),
                            "original_probabilities": original_probabilities.tolist(),
                            "action": None if abort_intervention else _plain(action),
                            "action_forwarded": None if abort_intervention else _plain(action),
                            "original_action": _plain(original_action),
                            "decoded_action": decoded_action,
                            "intervention": intervention_payload,
                            "intervention_scope": pattern_scope,
                            "intervention_on_inapplicable": pattern_on_inapplicable,
                            "intervention_scope_status": (
                                "aborted"
                                if abort_intervention
                                else "out_of_scope"
                                if intervention_skipped and declared_out_of_scope
                                else "skipped_conditional"
                                if intervention_skipped
                                else "applied"
                            ),
                            "intervention_declared_out_of_scope": declared_out_of_scope,
                            "original_values": original_values,
                            "modified_values": modified_values,
                            "delta_values": delta_values,
                            "delta_abs_values": {
                                index: _plain(abs(float(value)))
                                for index, value in delta_values.items()
                                if isinstance(value, (int, float, np.integer, np.floating))
                            },
                            "preprocess": preprocess_info,
                            "pre_telemetry": pre,
                            "road_segment_id": _plain(pre.get("road_segment_id")) if isinstance(pre, Mapping) else None,
                            "post_telemetry": post,
                            "reward": _plain(reward),
                            "terminated": terminated,
                            "truncated": truncated,
                            "termination_flags_status": (
                                "not_observed" if not env_step_returned else "observed"
                            ),
                            "done": (
                                bool(terminated or truncated)
                                if env_step_returned or abort_intervention
                                else None
                            ),
                            "env_step_attempted": env_step_attempted,
                            # ``called`` records the invocation itself;
                            # ``returned`` is the executable-step count.
                            "env_step_called": env_step_attempted,
                            "env_step_returned": env_step_returned,
                            "env_step_status": (
                                "not_called"
                                if not env_step_attempted
                                else "returned"
                                if env_step_returned
                                else "raised"
                            ),
                            "env_step_error": env_step_error,
                            "physical_state_unknown": bool(
                                env_step_attempted and not env_step_returned
                            ),
                            "post_telemetry_attempted": post_telemetry_attempted,
                            "post_telemetry_status": (
                                "not_attempted"
                                if not post_telemetry_attempted
                                else "available"
                                if post_telemetry_error is None
                                else "missing"
                            ),
                            "post_telemetry_error": post_telemetry_error,
                            "info_status": (
                                "not_observed" if not env_step_returned else "observed"
                            ),
                            "action_decode_status": (
                                "not_attempted"
                                if abort_intervention or not env_step_returned or post_telemetry_error is not None
                                else "decoded"
                                if action_decode_error is None
                                else "missing"
                            ),
                            "action_decode_error": action_decode_error,
                            "phase": (
                                "intervention_abort"
                                if abort_intervention
                                else step_failure_phase or "completed"
                            ),
                            "failure_phase": step_failure_phase,
                            "failure_reason": step_failure_reason,
                            "info": _plain(step_info),
                        }
                        record.update(selected_change)
                        if abort_intervention:
                            record["intervention_abort"] = True
                            record["intervention_abort_reason"] = getattr(
                                intervention, "skip_reason", None
                            )
                        if step_failure_phase is not None:
                            record["runtime_failure"] = True
                        observations.append(np.array(original_input, copy=True))
                        modified_observations.append(changed_input)
                        records.append(record)
                        step_context["record_saved"] = True
                        current_phase = "video"
                        if pattern_video:
                            frame, frame_status = render_frame(
                                adapter,
                                env,
                                step=step,
                                simulation_time=post_time,
                                pattern_id=identifier,
                                phase="post",
                            )
                            if frame is not None:
                                frame_status = save_video_frame(
                                    frame,
                                    episode_video_dir,
                                    step=step,
                                    simulation_time=post_time,
                                    pattern_id=identifier,
                                    phase="post",
                                )
                            frame_status["episode"] = episode_index
                            frames.append(frame_status)
                            record["frame_status"] = frame_status
                        if abort_intervention:
                            terminal_reason = "intervention_abort"
                            record["terminal_reason"] = terminal_reason
                            done = True
                            continue
                        if step_failure_phase is not None:
                            runtime_failure_phase = step_failure_phase
                            runtime_failure_reason = step_failure_reason
                            terminal_reason = "runtime_error"
                            record["terminal_reason"] = terminal_reason
                            done = True
                            continue
                        raw_obs = next_raw
                        reset_info = step_info
                        step += 1
                        done = terminated or truncated
                        if done:
                            terminal_reason = _terminal_reason(terminated, truncated, step_info, post)
                            record["terminal_reason"] = terminal_reason
                        elif max_steps is not None and step >= max_steps:
                            terminal_reason = "budget_censored"
                            record["budget_truncated"] = True
                            record["terminal_reason"] = terminal_reason
                            done = True
                    if terminal_reason == "unknown":
                        terminal_reason = "unknown"
                    current_phase = "metrics"
                    metrics = summarize_trajectory(records, terminal_reason=terminal_reason, **thresholds)
                    planned_target_steps = _pattern_target_step_count(config, max_steps)
                    executed_step_count = sum(
                        bool(record.get("env_step_returned", record.get("env_step_called")))
                        for record in records
                    )
                    intervention_count = summarize_intervention_records(
                        records=records,
                        target_step_count=len(records),
                        declared_out_of_scope=[
                            bool(record.get("intervention_declared_out_of_scope", False))
                            for record in records
                        ],
                    )
                    intervention_count.update(
                        {
                            "planned_target_step_count": planned_target_steps,
                            "attempted_step_count": len(records),
                            "executed_env_step_count": int(executed_step_count),
                            "actual_target_step_count": int(executed_step_count),
                            "aborted_before_env_step_count": int(
                                sum(not bool(record.get("env_step_attempted", record.get("env_step_called"))) for record in records)
                            ),
                            "target_step_count": len(records),
                            "planned_range_complete": (
                                None
                                if planned_target_steps is None
                                else len(records) >= planned_target_steps
                            ),
                        }
                    )
                    skipped_interventions = [
                        record.get("intervention", {})
                        for record in records
                        if isinstance(record.get("intervention"), Mapping)
                        and record["intervention"].get("skipped") is True
                    ]
                    skip_reasons = sorted({
                        str(item.get("skip_reason"))
                        for item in skipped_interventions
                        if item.get("skip_reason")
                    })
                    execution_status = (
                        "failed"
                        if runtime_failure_phase is not None
                        else "aborted"
                        if terminal_reason == "intervention_abort"
                        else "completed"
                    )
                    if runtime_failure_phase is not None:
                        assessment_status = "not_evaluable_runtime_error"
                    elif terminal_reason == "intervention_abort":
                        assessment_status = "not_evaluable_intervention_abort"
                    elif identifier == "P00":
                        assessment_status = "control_baseline"
                    elif skipped_interventions:
                        assessment_status = (
                            "conditional_out_of_scope"
                            if all(
                                bool(record.get("intervention_declared_out_of_scope", False))
                                for record in records
                                if isinstance(record.get("intervention"), Mapping)
                                and record["intervention"].get("skipped") is True
                            )
                            else "conditional_or_partial_application"
                        )
                    elif (
                        intervention_count.get("changed_count_exact", 0) > 0
                        and intervention_count.get("meaningful_changed_count") == 0
                    ):
                        assessment_status = "numerical_only"
                    elif intervention_count.get("changed_count_exact", 0) == 0:
                        assessment_status = "no_exact_input_change"
                    else:
                        assessment_status = "evaluated"
                    episode_result: dict[str, Any] = {
                        "pattern_id": identifier,
                        "episode": episode_index,
                        "episode_id": f"episode-{episode_index}",
                        "scenario_seed": episode_seed,
                        "rl_seed": episode_rl_seed,
                        "record_count": len(records),
                        "terminal_reason": terminal_reason,
                        "initial_snapshot": initial_snapshot,
                        "records": records,
                        "metrics": metrics,
                        "execution_status": execution_status,
                        "assessment_status": assessment_status,
                        "failure_phase": runtime_failure_phase,
                        "failure_reason": runtime_failure_reason,
                        "scope": pattern_scope,
                        "on_inapplicable": pattern_on_inapplicable,
                        "video": {
                            "requested": bool(video),
                            "enabled": bool(pattern_video),
                            "selector": video_pattern_selector_status,
                            "configured_patterns": (
                                None
                                if video_pattern_selector is None
                                else sorted(video_pattern_selector)
                            ),
                            "status": "pending" if pattern_video else "not_generated",
                            "reason": pattern_video_reason,
                            "frame_count": 0,
                        },
                        "target_step_count": len(records),
                        "planned_target_step_count": planned_target_steps,
                        "actual_target_step_count": int(executed_step_count),
                        "eligible_count": intervention_count.get("eligible_count"),
                        "applied_count": intervention_count.get("applied_count"),
                        "changed_count_exact": intervention_count.get("changed_count_exact"),
                        "changed_count": intervention_count.get("changed_count_exact"),
                        "changed_element_count_exact": intervention_count.get("changed_element_count_exact"),
                        "applied_element_count": intervention_count.get("applied_element_count"),
                        "noop_element_count": intervention_count.get("noop_element_count"),
                        "noop_count": intervention_count.get("noop_count"),
                        "skipped_count": intervention_count.get("skipped_count"),
                        "per_input_delta": intervention_count.get("per_input_delta", {}),
                        "intervention_counts": intervention_count,
                        # Pairing is filled only after the per-episode initial
                        # and execution checks below.  Explicit nulls keep a
                        # control episode from being mistaken for a verified
                        # intervention pair by legacy readers.
                        "paired_p00_status": None,
                        "paired_p00_verified": None,
                        "paired_p00_missing_reason": "baseline_control" if identifier == "P00" else None,
                        "intervention_skipped": bool(skipped_interventions),
                        "intervention_skip_count": len(skipped_interventions),
                        "intervention_skip_reasons": skip_reasons,
                    }
                    current_phase = "pairing"
                    if identifier == "P00":
                        reference = {
                            "episode": episode_index,
                            "initial_snapshot": initial_snapshot,
                            "initial_observation": np.array(observations[0], copy=True) if observations else np.empty((schema.dimension,), dtype=np.float32),
                            "observations": [np.array(value, copy=True) for value in observations],
                            "records": [dict(record) for record in records],
                        }
                        result.baseline_references[episode_index] = reference
                        if episode_index in explicit_reference_rows:
                            episode_result["reference_trace_validation"] = _trace_compare(
                                explicit_reference_rows[episode_index],
                                records,
                                reference_observations=reference_observations,
                                current_observations=observations,
                            )
                            if episode_result["reference_trace_validation"]["status"] == "mismatch":
                                raise ClosedLoopError(
                                    f"P00 episode {episode_index} does not match saved reference trace"
                                )
                        else:
                            episode_result["reference_trace_validation"] = {
                                "status": "self_reference",
                                "matched": True,
                                "checked_records": len(records),
                            }
                    else:
                        baseline_reference = result.baseline_references.get(episode_index)
                        if baseline_reference is not None:
                            episode_result["p00_trace_alignment"] = _trace_alignment(
                                baseline_reference["records"],
                                records,
                                baseline_reference["observations"],
                                observations,
                            )
                            baseline_initial = baseline_reference.get("initial_snapshot") or {}
                            current_initial = initial_snapshot or {}
                            baseline_initial_status = (
                                (baseline_initial.get("initial_match") or {}).get("status")
                            )
                            current_initial_status = (
                                (current_initial.get("initial_match") or {}).get("status")
                            )
                            baseline_episode = (
                                result.patterns.get("P00", [])[episode_index]
                                if episode_index < len(result.patterns.get("P00", []))
                                else {}
                            )
                            baseline_execution_status = baseline_episode.get("execution_status")
                            pair_reasons: list[str] = []
                            if baseline_initial_status != "matched":
                                pair_reasons.append(
                                    f"baseline_initial_{baseline_initial_status or 'missing'}"
                                )
                            if current_initial_status != "matched":
                                pair_reasons.append(
                                    f"pattern_initial_{current_initial_status or 'missing'}"
                                )
                            if baseline_execution_status == "aborted":
                                pair_reasons.append("baseline_intervention_abort")
                            elif not _pair_execution_eligible(baseline_episode):
                                pair_reasons.append(
                                    f"baseline_terminal_{baseline_episode.get('terminal_reason') or 'incomplete'}"
                                )
                            if execution_status == "aborted":
                                pair_reasons.append("pattern_intervention_abort")
                            elif not _pair_execution_eligible(episode_result):
                                pair_reasons.append(
                                    f"pattern_terminal_{episode_result.get('terminal_reason') or 'incomplete'}"
                                )
                            if (
                                baseline_initial_status == "matched"
                                and current_initial_status == "matched"
                                and _pair_execution_eligible(baseline_episode)
                                and _pair_execution_eligible(episode_result)
                            ):
                                episode_result["metrics"]["paired_p00"] = paired_deltas(
                                    metrics,
                                    baseline_episode.get("metrics", {}),
                                )
                                episode_result["paired_p00_status"] = "matched"
                                episode_result["paired_p00_verified"] = True
                                episode_result["paired_p00_missing_reason"] = None
                            else:
                                episode_result["metrics"]["paired_p00"] = None
                                episode_result["paired_p00_status"] = (
                                    "interrupted"
                                    if execution_status == "aborted"
                                    else "unverified"
                                    if baseline_initial_status != "matched"
                                    or current_initial_status != "matched"
                                    else "incomplete"
                                    if not _pair_execution_eligible(episode_result)
                                    or not _pair_execution_eligible(baseline_episode)
                                    else "unverified"
                                )
                                episode_result["paired_p00_verified"] = False
                                episode_result["paired_p00_missing_reason"] = "; ".join(pair_reasons)
                        else:
                            episode_result["paired_p00_status"] = "unavailable"
                            episode_result["paired_p00_verified"] = False
                            episode_result["paired_p00_missing_reason"] = "p00_reference_missing"
                    video_result: dict[str, Any] | None = None
                    if pattern_video and store:
                        video_result = finalize_video(
                            frames,
                            episode_video_dir,
                            fps=video_fps if video_fps is not None else infer_video_fps(records),
                        )
                        episode_result["video"] = {
                            **episode_result["video"],
                            **video_result,
                            "status": str(video_result.get("status", "generated")),
                            "frame_count": len(frames),
                            "reason": video_result.get("reason"),
                        }
                    elif pattern_video:
                        # Without a RunArtifacts store the frame files may
                        # still be written to the requested/default directory,
                        # but there is no durable video.json to point to.
                        episode_result["video"] = {
                            **episode_result["video"],
                            "status": "frames_saved_without_store",
                            "frame_count": len(frames),
                            "reason": "store_not_provided",
                        }
                    result.patterns[identifier].append(episode_result)
                    episode_result_saved = True
                    current_phase = "artifact"
                    if store:
                        np.save(episode_dir / "observations.npy", _safe_stack(observations, schema.dimension), allow_pickle=False)
                        np.save(episode_dir / "modified_observations.npy", _safe_stack(modified_observations, schema.dimension), allow_pickle=False)
                        store.write_json(episode_dir.relative_to(store.run_dir) / "trajectory.json", episode_result)
                        if pattern_video and video_result is not None:
                            store.write_json(
                                episode_dir.relative_to(store.run_dir) / "video.json",
                                video_result,
                            )
                    if identifier == "P00" and store:
                        reference_dir = store.closed_loop_dir / "P00" / f"episode-{episode_index}"
                        store.write_json(reference_dir.relative_to(store.run_dir) / "reference.json", {
                            "episode": episode_index,
                            "record_count": len(records),
                            "records": records,
                            "initial_snapshot": initial_snapshot,
                        })
                        np.save(reference_dir / "reference_observations.npy", _safe_stack(observations, schema.dimension), allow_pickle=False)
                except Exception as exc:
                    # Preserve an attempted episode when a runtime hook fails
                    # after records have already been collected.  An
                    # intervention-resolution failure in a full-episode
                    # pattern is a pattern abort; policy/decode/schema/env
                    # failures remain explicit runtime failures instead of
                    # being silently reclassified as intervention skips.
                    if episode_result_saved or current_phase in {
                        "runtime_seed",
                        "reset",
                        "seed_verification",
                    }:
                        raise
                    failure_reason = f"{type(exc).__name__}: {exc}"
                    # Materialize the current attempt before building the
                    # episode-level failure.  This keeps a policy/intermediate
                    # hook error from silently dropping an already obtained
                    # pre/modified observation.  A missing intervention result
                    # is represented explicitly while preserving the current
                    # observation slot and pre-step telemetry.
                    if (
                        not step_context.get("record_saved", False)
                    ):
                        partial_original = step_context.get("original_input")
                        partial_changed = step_context.get("changed_input")
                        original_missing = partial_original is None
                        if original_missing:
                            partial_original_array = np.full(
                                (schema.dimension,),
                                np.nan,
                                dtype=np.float32,
                            )
                        else:
                            partial_original_array = np.array(partial_original, copy=True)
                        modified_missing = partial_changed is None
                        if modified_missing:
                            # Keep the array slot aligned while recording that
                            # no modified model input was produced.
                            partial_changed_array = np.array(partial_original_array, copy=True)
                        else:
                            partial_changed_array = np.array(partial_changed, copy=True)
                        partial_intervention = step_context.get("intervention")
                        try:
                            partial_intervention_payload = (
                                partial_intervention.to_dict()
                                if partial_intervention is not None
                                else None
                            )
                        except Exception:
                            partial_intervention_payload = None
                        partial_observation_index = len(observations)
                        observations.append(partial_original_array)
                        modified_observations.append(partial_changed_array)
                        records.append(
                            {
                                "pattern_id": identifier,
                                "episode": episode_index,
                                "episode_id": f"episode-{episode_index}",
                                "scenario_seed": episode_seed,
                                "rl_seed": episode_rl_seed,
                                "requested_scenario_seed": episode_seed,
                                "requested_rl_seed": episode_rl_seed,
                                "actual_scenario_seed": step_context.get("actual_scenario_seed"),
                                "step": int(step_context.get("step", len(records))),
                                "observation_index": partial_observation_index,
                                "observation_missing": original_missing,
                                "modified_observation_missing": modified_missing,
                                "observation_hash": (
                                    _observation_hash(partial_original_array)
                                    if not original_missing
                                    else None
                                ),
                                "modified_observation_hash": (
                                    _observation_hash(partial_changed_array)
                                    if not modified_missing
                                    else None
                                ),
                                "observation_shape": list(partial_original_array.shape),
                                "observation_dtype": str(partial_original_array.dtype),
                                "pre_time": telemetry_time(
                                    step_context.get("pre_telemetry") or {},
                                    step_context.get("step"),
                                ),
                                "post_time": telemetry_time(
                                    step_context.get("post_telemetry") or {},
                                    None,
                                ),
                                "probabilities": _plain(step_context.get("probabilities")),
                                "original_probabilities": _plain(
                                    step_context.get("original_probabilities")
                                ),
                                "action": _plain(step_context.get("action")),
                                "action_forwarded": (
                                    _plain(step_context.get("action"))
                                    if step_context.get("env_step_returned", False)
                                    else None
                                ),
                                "original_action": _plain(step_context.get("original_action")),
                                "decoded_action": _plain(step_context.get("decoded_action")),
                                "intervention": partial_intervention_payload,
                                "preprocess": _plain(step_context.get("preprocess_info")),
                                "pre_telemetry": _plain(step_context.get("pre_telemetry")),
                                "post_telemetry": _plain(step_context.get("post_telemetry")),
                                "reward": _plain(step_context.get("reward")),
                                "terminated": step_context.get("terminated"),
                                "truncated": step_context.get("truncated"),
                                "termination_flags_status": (
                                    "observed"
                                    if step_context.get("env_step_returned", False)
                                    else "not_observed"
                                ),
                                "done": None,
                                "env_step_attempted": bool(
                                    step_context.get("env_step_returned", False)
                                ),
                                "env_step_called": bool(
                                    step_context.get("env_step_returned", False)
                                ),
                                "env_step_returned": bool(
                                    step_context.get("env_step_returned", False)
                                ),
                                "env_step_status": (
                                    "returned"
                                    if step_context.get("env_step_returned", False)
                                    else "not_called"
                                ),
                                "physical_state_unknown": False,
                                "post_telemetry_attempted": bool(
                                    step_context.get("post_telemetry") is not None
                                ),
                                "post_telemetry_status": (
                                    "available"
                                    if step_context.get("post_telemetry") is not None
                                    else "not_attempted"
                                ),
                                "action_decode_status": (
                                    "decoded"
                                    if "decoded_action" in step_context
                                    else "not_attempted"
                                ),
                                "info": _plain(step_context.get("step_info")),
                                "info_status": (
                                    "observed"
                                    if step_context.get("env_step_returned", False)
                                    else "not_observed"
                                ),
                                "phase": current_phase,
                                "failure_phase": current_phase,
                                "failure_reason": failure_reason,
                                "partial_record": True,
                            }
                        )
                    intervention_abort_failure = (
                        pattern_scope == "full_episode"
                        and current_phase == "intervention"
                    )
                    failure_terminal_reason = (
                        "intervention_abort"
                        if intervention_abort_failure
                        else "runtime_error"
                    )
                    failure_execution_status = (
                        "aborted" if intervention_abort_failure else "failed"
                    )
                    failure_assessment_status = (
                        "not_evaluable_intervention_abort"
                        if intervention_abort_failure
                        else "not_evaluable_runtime_error"
                    )
                    failure_metrics = summarize_trajectory(
                        records,
                        terminal_reason=failure_terminal_reason,
                        **thresholds,
                    )
                    failure_planned_steps = _pattern_target_step_count(config, max_steps)
                    failure_executed_steps = sum(
                        bool(record.get("env_step_returned", record.get("env_step_called")))
                        for record in records
                    )
                    failure_counts = summarize_intervention_records(
                        records=records,
                        target_step_count=len(records),
                        declared_out_of_scope=[
                            bool(record.get("intervention_declared_out_of_scope", False))
                            for record in records
                        ],
                    )
                    failure_counts.update(
                        {
                            "planned_target_step_count": failure_planned_steps,
                            "attempted_step_count": len(records),
                            "executed_env_step_count": int(failure_executed_steps),
                            "actual_target_step_count": int(failure_executed_steps),
                            "aborted_before_env_step_count": int(
                                sum(not bool(record.get("env_step_attempted", record.get("env_step_called"))) for record in records)
                            ),
                            "target_step_count": len(records),
                            "planned_range_complete": False,
                        }
                    )
                    failed_episode: dict[str, Any] = {
                        "pattern_id": identifier,
                        "episode": episode_index,
                        "episode_id": f"episode-{episode_index}",
                        "scenario_seed": episode_seed,
                        "rl_seed": episode_rl_seed,
                        "record_count": len(records),
                        "terminal_reason": failure_terminal_reason,
                        "initial_snapshot": initial_snapshot,
                        "records": records,
                        "metrics": failure_metrics,
                        "execution_status": failure_execution_status,
                        "assessment_status": failure_assessment_status,
                        "scope": pattern_scope,
                        "on_inapplicable": pattern_on_inapplicable,
                        "failure_phase": current_phase,
                        "failure_reason": failure_reason,
                        "video": {
                            "requested": bool(video),
                            "enabled": bool(pattern_video),
                            "selector": video_pattern_selector_status,
                            "configured_patterns": (
                                None
                                if video_pattern_selector is None
                                else sorted(video_pattern_selector)
                            ),
                            "status": "partial" if pattern_video and frames else "not_generated",
                            "reason": failure_reason,
                            "frame_count": len(frames),
                        },
                        "target_step_count": len(records),
                        "planned_target_step_count": failure_planned_steps,
                        "actual_target_step_count": int(failure_executed_steps),
                        "eligible_count": failure_counts.get("eligible_count"),
                        "applied_count": failure_counts.get("applied_count"),
                        "changed_count_exact": failure_counts.get("changed_count_exact"),
                        "changed_count": failure_counts.get("changed_count_exact"),
                        "changed_element_count_exact": failure_counts.get("changed_element_count_exact"),
                        "applied_element_count": failure_counts.get("applied_element_count"),
                        "noop_element_count": failure_counts.get("noop_element_count"),
                        "noop_count": failure_counts.get("noop_count"),
                        "skipped_count": failure_counts.get("skipped_count"),
                        "per_input_delta": failure_counts.get("per_input_delta", {}),
                        "intervention_counts": failure_counts,
                        "paired_p00_status": (
                            "interrupted" if intervention_abort_failure else "unverified"
                        ) if identifier != "P00" else None,
                        "paired_p00_verified": False if identifier != "P00" else None,
                        "paired_p00_missing_reason": (
                            f"pattern_{failure_terminal_reason}"
                            if identifier != "P00"
                            else "baseline_control"
                        ),
                        "intervention_skipped": False,
                        "intervention_skip_count": failure_counts.get("skipped_count"),
                        "intervention_skip_reasons": sorted(
                            failure_counts.get("skip_reasons", {})
                        ),
                    }
                    result.patterns[identifier].append(failed_episode)
                    episode_result_saved = True
                    if store:
                        np.save(
                            episode_dir / "observations.npy",
                            _safe_stack(observations, schema.dimension),
                            allow_pickle=False,
                        )
                        np.save(
                            episode_dir / "modified_observations.npy",
                            _safe_stack(modified_observations, schema.dimension),
                            allow_pickle=False,
                        )
                        store.write_json(
                            episode_dir.relative_to(store.run_dir) / "trajectory.json",
                            failed_episode,
                        )
                finally:
                    close_environment(adapter, env)
        policy_fingerprint_after = _policy_fingerprint(policy, adapter)
        result.policy_fingerprint_after = policy_fingerprint_after
        if policy_fingerprint_before is not None and policy_fingerprint_after is not None:
            result.policy_unchanged = policy_fingerprint_before == policy_fingerprint_after
            if not result.policy_unchanged:
                raise ClosedLoopError(
                    "policy parameters/buffers changed during closed-loop analysis"
                )
        if store:
            summary = result.as_dict()
            store.write_json("02_closed_loop/summary.json", summary)
            manifest = store.load_manifest() if store.manifest_path.is_file() else build_manifest(config=config, command="closed-loop")
            manifest["closed_loop"] = summary
            manifest.setdefault("stages", {})["closed_loop"] = {
                "patterns": list(result.patterns),
                "p00_reference_episodes": len(result.baseline_references),
                "video": bool(video),
                "status": summary["status"],
                "counts": summary["counts"],
                "video_pattern_selector": video_pattern_selector_status,
                "video_patterns": (
                    None
                    if video_pattern_selector is None
                    else sorted(video_pattern_selector)
                ),
            }
            store.save_manifest(manifest)
            store.update_status(
                "closed_loop",
                "failed" if summary["counts"]["required_experiment_failed"] else "success",
                patterns=list(result.patterns),
                status=summary["status"],
                counts=summary["counts"],
            )
        return result
    except Exception as exc:
        if store:
            store.update_status("closed_loop", "failed", error=f"{type(exc).__name__}: {exc}")
        raise


closed_loop = run_closed_loop


__all__ = [
    "ClosedLoopError",
    "ClosedLoopResult",
    "apply_pattern",
    "closed_loop",
    "run_closed_loop",
]
