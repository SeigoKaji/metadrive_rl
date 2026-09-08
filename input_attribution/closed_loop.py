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

        pattern_summary: dict[str, Any] = {}
        for pattern_id, episodes in self.patterns.items():
            metrics = [episode.get("metrics", {}) for episode in episodes]
            aggregate = metric_summary(episodes)
            pattern_summary[pattern_id] = {
                "episodes": len(episodes),
                "records": sum(int(item.get("record_count", 0)) for item in episodes),
                "metrics": metrics,
                "metric_summary": aggregate,
                # Keep the two weighting choices explicit at the pattern level
                # for report readers that do not need the nested counts.
                "episode_mean": aggregate["episode_mean"],
                "step_weighted": aggregate["step_weighted"],
                "weighted_episode_mean": aggregate["weighted_episode_mean"],
                "sum": aggregate["sum"],
                "aggregation": aggregate["aggregation"],
                "counts": aggregate["counts"],
            }
        return {
            "store": str(self.store.run_dir) if self.store else None,
            "p00_reference_episodes": len(self.baseline_references),
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
    configured_video_fps = video_config.get("fps") if isinstance(video_config, Mapping) else None
    if video_fps is None and configured_video_fps is not None:
        try:
            video_fps = float(configured_video_fps)
        except (TypeError, ValueError):
            raise ClosedLoopError(f"video.fps must be numeric: {configured_video_fps!r}") from None
    thresholds = _thresholds(config)
    result = ClosedLoopResult(store)
    explicit_reference_rows = _reference_records_by_episode(reference_records)
    if store:
        store.update_status("closed_loop", "running", pattern_ids=[pattern.pattern_id for pattern in resolved_patterns])
    try:
        for pattern in resolved_patterns:
            identifier = pattern.pattern_id
            result.patterns.setdefault(identifier, [])
            for episode_index in range(episodes):
                episode_dir = store.closed_loop_dir / identifier / f"episode-{episode_index}" if store else None
                if episode_dir is not None:
                    if episode_dir.exists():
                        raise ClosedLoopError(f"closed-loop episode output already exists: {episode_dir}")
                    episode_dir.mkdir(parents=True, exist_ok=False)
                if video:
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
                try:
                    seed_metadata = seed_runtime(adapter, env, episode_rl_seed)
                    raw_obs, reset_info = reset_environment(adapter, env, episode_seed)
                    done = False
                    step = 0
                    seed_verified = episode_seed is None
                    initial_snapshot: dict[str, Any] | None = None
                    while not done:
                        pre = telemetry(adapter, env, reset_info, phase="pre", step=step)
                        actual_scenario_seed = _actual_scenario_seed(env, reset_info, pre)
                        if not seed_verified:
                            if actual_scenario_seed is None or actual_scenario_seed != episode_seed:
                                raise ClosedLoopError(
                                    f"environment did not apply requested scenario seed {episode_seed}: "
                                    f"actual={actual_scenario_seed!r}"
                                )
                            seed_verified = True
                        original_input, preprocess_info = model_input(adapter, raw_obs, reset_info)
                        try:
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
                        changed_input = np.array(intervention.observation, copy=True)
                        probabilities = policy_probabilities(policy, changed_input)
                        original_probabilities = policy_probabilities(policy, original_input)
                        action = policy_predict(policy, changed_input, deterministic=deterministic)
                        original_action = policy_predict(policy, original_input, deterministic=deterministic)
                        if not 0 <= int(action) < probabilities.size:
                            raise ClosedLoopError(f"modified policy action is outside Discrete({probabilities.size}): {action}")
                        if not 0 <= int(original_action) < original_probabilities.size:
                            raise ClosedLoopError(
                                f"original policy action is outside Discrete({original_probabilities.size}): {original_action}"
                            )
                        next_raw, reward, terminated, truncated, step_info = step_environment(env, action)
                        post = telemetry(adapter, env, step_info, phase="post", step=step + 1)
                        pre_time = telemetry_time(pre, step)
                        post_time = telemetry_time(post, step + 1)
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
                            "action": _plain(action),
                            "action_forwarded": _plain(action),
                            "original_action": _plain(original_action),
                            "decoded_action": decode_action(adapter, action, config=config),
                            "intervention": intervention.to_dict(),
                            "preprocess": preprocess_info,
                            "pre_telemetry": pre,
                            "post_telemetry": post,
                            "reward": _plain(reward),
                            "terminated": terminated,
                            "truncated": truncated,
                            "done": bool(terminated or truncated),
                            "info": _plain(step_info),
                        }
                        observations.append(np.array(original_input, copy=True))
                        modified_observations.append(changed_input)
                        records.append(record)
                        if video:
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
                    metrics = summarize_trajectory(records, terminal_reason=terminal_reason, **thresholds)
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
                        "intervention_skipped": bool(skipped_interventions),
                        "intervention_skip_count": len(skipped_interventions),
                        "intervention_skip_reasons": skip_reasons,
                    }
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
                            if baseline_initial_status == "matched" and current_initial_status == "matched":
                                episode_result["metrics"]["paired_p00"] = paired_deltas(
                                    metrics,
                                    result.patterns["P00"][episode_index]["metrics"],
                                )
                            else:
                                episode_result["metrics"]["paired_p00"] = None
                                episode_result["paired_p00_status"] = "unverified"
                    result.patterns[identifier].append(episode_result)
                    if store:
                        np.save(episode_dir / "observations.npy", _safe_stack(observations, schema.dimension), allow_pickle=False)
                        np.save(episode_dir / "modified_observations.npy", _safe_stack(modified_observations, schema.dimension), allow_pickle=False)
                        store.write_json(episode_dir.relative_to(store.run_dir) / "trajectory.json", episode_result)
                        if video:
                            store.write_json(
                                episode_dir.relative_to(store.run_dir) / "video.json",
                                finalize_video(
                                    frames,
                                    episode_video_dir,
                                    fps=video_fps if video_fps is not None else infer_video_fps(records),
                                ),
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
                finally:
                    close_environment(adapter, env)
        if store:
            store.write_json("02_closed_loop/summary.json", result.as_dict())
            manifest = store.load_manifest() if store.manifest_path.is_file() else build_manifest(config=config, command="closed-loop")
            manifest["closed_loop"] = result.as_dict()
            manifest.setdefault("stages", {})["closed_loop"] = {
                "patterns": list(result.patterns),
                "p00_reference_episodes": len(result.baseline_references),
                "video": bool(video),
            }
            store.save_manifest(manifest)
            store.update_status("closed_loop", "success", patterns=list(result.patterns))
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
