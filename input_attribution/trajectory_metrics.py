"""Closed-loop物理テレメトリの明示的なepisode集計。

介入済み観測を評価値へ再利用せず、``record['post_telemetry']`` と環境が返した
終了flagだけを読み取る。欠測は0で埋めず、``None`` と ``status`` で区別する。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import math
from typing import Any

import numpy as np


_MISSING = object()


def _first(mapping: Mapping[str, Any], names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return default


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _boolean(value: Any) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    return None


def _telemetry(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record.get("post_telemetry")
    return value if isinstance(value, Mapping) else {}


def _time(
    record: Mapping[str, Any],
    telemetry: Mapping[str, Any],
    fallback: float | None = None,
) -> float | None:
    value = _first(
        telemetry,
        ("simulation_time_s", "simulation_time_seconds", "simulation_time", "sim_time_seconds", "sim_time_s", "sim_time", "time_s", "time"),
        _first(record, ("post_time", "simulation_time"), fallback),
    )
    parsed = _number(value)
    return parsed if parsed is not None else fallback


def _lane_error(telemetry: Mapping[str, Any]) -> float | None:
    return _number(
        _first(
            telemetry,
            (
                "target_lane_lateral_error_m",
                "target_lane_offset_m",
                "target_lane_error_m",
                "signed_target_lane_offset_m",
            ),
        )
    )


def _lane_valid(telemetry: Mapping[str, Any]) -> bool | None:
    return _boolean(_first(telemetry, ("target_lane_valid", "lane_target_valid", "target_valid")))


def _departure(telemetry: Mapping[str, Any], *, tolerance_ratio: float) -> bool | None:
    if _lane_valid(telemetry) is False:
        return None
    explicit = _first(
        telemetry,
        ("target_lane_departed", "departed_target_lane", "lane_departed"),
    )
    error = _lane_error(telemetry)
    width = _number(_first(telemetry, ("lane_width_m", "target_lane_width_m")))
    if error is None or width is None or width <= 0:
        value = _boolean(explicit)
        if value is not None:
            return value
        in_lane = _boolean(_first(telemetry, ("in_target_lane", "target_lane_in_bounds")))
        return None if in_lane is None else not in_lane
    # Apply this run's declared tolerance to the measured geometry even when
    # an adapter also reports in_target_lane using a different threshold.
    return abs(error) > (width / 2.0) * (1.0 + float(tolerance_ratio))


def _speed_m_s(telemetry: Mapping[str, Any]) -> float | None:
    speed = _number(_first(telemetry, ("speed_m_s", "speed_mps", "velocity_m_s", "velocity_mps")))
    if speed is not None:
        return max(0.0, speed)
    speed_km_h = _number(_first(telemetry, ("speed_km_h", "velocity_km_h")))
    return max(0.0, speed_km_h / 3.6) if speed_km_h is not None else None


def _progress_m(telemetry: Mapping[str, Any]) -> float | None:
    return _number(
        _first(
            telemetry,
            (
                "route_progress_m",
                "progress_m",
                "route_progress",
                "distance_travelled_m",
                "distance_traveled_m",
            ),
        )
    )


def _termination_flags(record: Mapping[str, Any]) -> dict[str, bool | None]:
    sources: list[Mapping[str, Any]] = []
    for key in ("post_telemetry", "info"):
        value = record.get(key)
        if isinstance(value, Mapping):
            sources.append(value)
    def read(names: Sequence[str]) -> bool | None:
        for source in sources:
            value = _boolean(_first(source, names))
            if value is not None:
                return value
        return None
    wrong_lane_arrival = read(
        ("wrong_lane_arrival", "wrong_lane_goal", "arrive_wrong_lane", "goal_wrong_lane")
    )
    start_lane_departure = read(
        ("start_lane_departure", "departed_start_lane", "start_lane_departed")
    )
    arrived = read(("arrive_dest", "arrived", "success", "goal_reached"))
    # MetaDrive's destination flag can be true for a vehicle that reaches the
    # destination from the wrong lane.  Preserve that explicit failure flag and
    # do not expose the raw destination bit as a successful arrival.
    if wrong_lane_arrival is True:
        arrived = False
    return {
        "arrived": arrived,
        "road_out": read(("out_of_road", "out_of_drivable", "road_out", "crash_out_of_road")),
        "crash": read(("crash", "crashed", "collision", "collision_occurred")),
        "wrong_lane_arrival": wrong_lane_arrival,
        "start_lane_departure": start_lane_departure,
    }


def _steering(record: Mapping[str, Any]) -> float | None:
    decoded = record.get("decoded_action")
    for source in (decoded, _telemetry(record), record):
        if isinstance(source, Mapping):
            value = _number(_first(source, ("steering", "steering_command", "steer", "steer_command")))
            if value is not None:
                return value
    return None


def _action(record: Mapping[str, Any]) -> Any:
    return record.get("action", _MISSING)


def _pre_time(record: Mapping[str, Any], telemetry: Mapping[str, Any], fallback: float | None = None) -> float | None:
    value = _first(
        telemetry,
        ("simulation_time_s", "simulation_time_seconds", "simulation_time", "sim_time_seconds", "sim_time_s", "sim_time", "time_s", "time"),
        _first(record, ("pre_time", "pre_simulation_time")),
    )
    if value is None:
        return fallback
    return _number(value)


def _step_durations(
    rows: Sequence[Mapping[str, Any]],
    post_times: Sequence[float | None],
) -> list[float | None]:
    """Return one duration for each post-step state.

    Collection records explicitly carry pre/post times.  Using those pairs
    preserves the reset-to-first-step interval.  If a pre-time is absent, an
    adjacent pair of declared post-times is still a verified interval.  There
    is no synthetic one-second or observation-index fallback: an interval
    without a simulator clock remains ``None``.
    """

    result: list[float | None] = []
    for index, post_time in enumerate(post_times):
        pre_telemetry = rows[index].get("pre_telemetry", {})
        if not isinstance(pre_telemetry, Mapping):
            pre_telemetry = {}
        pre_time = _pre_time(rows[index], pre_telemetry, None)
        delta: float | None = None
        if post_time is not None:
            if pre_time is not None:
                candidate = post_time - pre_time
            elif index > 0 and post_times[index - 1] is not None:
                candidate = post_time - float(post_times[index - 1])
            else:
                candidate = None
            if candidate is not None and math.isfinite(candidate) and candidate >= 0.0:
                delta = float(candidate)
        result.append(delta)
    return result


def summarize_trajectory(
    records: Iterable[Mapping[str, Any]],
    *,
    terminal_reason: str | None = None,
    departure_tolerance_ratio: float = 0.05,
    departure_consecutive_steps: int = 1,
    low_speed_m_s: float = 0.5,
) -> dict[str, Any]:
    """Return physical metrics for one closed-loop episode.

    RMS uses only valid post-step target-lane samples as its denominator. The
    ``valid_rate`` is time-weighted when timestamps exist; ``valid_count`` and
    ``poststep_count`` remain available so a reader can inspect sample coverage.
    """

    rows = [dict(record) for record in records]
    telemetry_rows = [_telemetry(record) for record in rows]
    times = [_time(record, telemetry_rows[index], None) for index, record in enumerate(rows)]
    deltas = _step_durations(rows, times)
    errors: list[float] = []
    valid_mask: list[bool] = []
    lane_validity: list[bool | None] = []
    departures: list[bool | None] = []
    speeds: list[float] = []
    progress: list[float] = []
    cumulative_reward = 0.0
    reward_count = 0
    for telemetry in telemetry_rows:
        error = _lane_error(telemetry)
        valid = _lane_valid(telemetry)
        lane_validity.append(valid)
        if valid is True and error is not None:
            errors.append(error)
            valid_mask.append(True)
        elif valid is False:
            valid_mask.append(False)
        else:
            valid_mask.append(False)
        departures.append(_departure(telemetry, tolerance_ratio=departure_tolerance_ratio))
        speed = _speed_m_s(telemetry)
        if speed is not None:
            speeds.append(speed)
        distance = _progress_m(telemetry)
        if distance is not None:
            progress.append(distance)
    for record in rows:
        reward = _number(record.get("reward"))
        if reward is not None:
            cumulative_reward += reward
            reward_count += 1

    valid_count = len(errors)
    lane_reference_known = any(
        value is False or (value is True and _lane_error(telemetry) is not None)
        for value, telemetry in zip(lane_validity, telemetry_rows, strict=False)
    )
    poststep_count = len(rows)
    known_deltas = [delta for delta in deltas if delta is not None]
    verified_time_interval_count = len(known_deltas)
    total_duration = sum(known_deltas) if known_deltas else None
    valid_durations = [
        delta for delta, valid in zip(deltas, valid_mask, strict=False)
        if valid and delta is not None
    ]
    valid_time = (
        sum(valid_durations)
        if known_deltas and (valid_durations or valid_count == 0)
        else None
    )
    sample_valid_rate = (
        valid_count / poststep_count
        if lane_reference_known and poststep_count
        else None
    )
    valid_rate = (
        valid_time / total_duration
        if valid_time is not None and total_duration is not None and total_duration > 0
        else None
    )

    departure_events: list[int] = []
    consecutive = max(1, int(departure_consecutive_steps))
    index = 0
    while index < len(departures):
        if departures[index] is not True:
            index += 1
            continue
        start = index
        while index < len(departures) and departures[index] is True:
            index += 1
        if index - start >= consecutive:
            departure_events.append(start)
    departure_durations = [
        delta for delta, value in zip(deltas, departures, strict=False)
        if value is True and delta is not None
    ]
    departure_observations = [value for value in departures if value is True]
    departure_time = (
        (sum(departure_durations) if departure_durations else None)
        if departure_observations and known_deltas
        else 0.0
        if known_deltas
        else None
    )
    first_departure_time = (
        times[departure_events[0]]
        if departure_events and known_deltas
        else None
    )

    speeds_for_low = [_speed_m_s(telemetry) for telemetry in telemetry_rows]
    low_speed_observations = [
        speed for speed in speeds_for_low
        if speed is not None and speed < float(low_speed_m_s)
    ]
    low_speed_durations = [
        delta for delta, speed in zip(deltas, speeds_for_low, strict=False)
        if speed is not None and speed < float(low_speed_m_s) and delta is not None
    ]
    low_speed_duration = (
        (sum(low_speed_durations) if low_speed_durations else None)
        if low_speed_observations and known_deltas
        else 0.0
        if known_deltas
        else None
    )

    actions = [_action(record) for record in rows]
    action_switch_count = sum(
        1 for previous, current in zip(actions, actions[1:], strict=False)
        if previous is not _MISSING and current is not _MISSING and previous != current
    )
    steering_values = [_steering(record) for record in rows]
    steering_values = [value for value in steering_values if value is not None]
    steering_variation = [abs(current - previous) for previous, current in zip(steering_values, steering_values[1:], strict=False)]
    row_flags = [_termination_flags(row) for row in rows]
    flags: dict[str, bool | None] = {}
    for name in ("arrived", "road_out", "crash", "wrong_lane_arrival", "start_lane_departure"):
        values = [item[name] for item in row_flags]
        flags[name] = (
            True
            if any(value is True for value in values)
            else False
            if any(value is False for value in values)
            else None
        )
    if flags["wrong_lane_arrival"] is True:
        flags["arrived"] = False
    terminal_text = terminal_reason or (str(rows[-1].get("terminal_reason")) if rows and rows[-1].get("terminal_reason") else None)
    if terminal_text:
        lowered = terminal_text.lower()
        if flags["wrong_lane_arrival"] is None and "wrong_lane" in lowered:
            flags["wrong_lane_arrival"] = True
            flags["arrived"] = False
        if flags["start_lane_departure"] is None and "start_lane_departure" in lowered:
            flags["start_lane_departure"] = True
        if flags["arrived"] is None and any(token in lowered for token in ("arrive", "success", "goal", "dest")):
            flags["arrived"] = True
        if flags["road_out"] is None and any(token in lowered for token in ("out_of_road", "out-of-road", "road_out")):
            flags["road_out"] = True
        if flags["crash"] is None and any(token in lowered for token in ("crash", "collision")):
            flags["crash"] = True

    metric_status = "available" if valid_count else "not_available"
    progress_value = progress[-1] if progress else None
    result: dict[str, Any] = {
        "status": metric_status,
        "poststep_count": poststep_count,
        "valid_count": valid_count,
        "valid_poststep_count": valid_count,
        "verified_time_interval_count": verified_time_interval_count,
        "valid_time_s": valid_time if lane_reference_known else None,
        "valid_time_seconds": valid_time if lane_reference_known else None,
        "valid_time": valid_time if lane_reference_known else None,
        "valid_poststep_time_s": valid_time if lane_reference_known else None,
        "valid_rate": valid_rate if lane_reference_known else None,
        "sample_valid_rate": sample_valid_rate,
        "valid_poststep_rate": sample_valid_rate,
        "lane_metric_denominator": "valid_poststep_target_lane_samples",
        "lane_rms_m": float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
        "lane_max_abs_m": float(np.max(np.abs(errors))) if errors else None,
        "lane_error_rms_m": float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
        "lane_error_max_abs_m": float(np.max(np.abs(errors))) if errors else None,
        "target_lane_rms_m": float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
        "target_lane_max_abs_m": float(np.max(np.abs(errors))) if errors else None,
        "departure_count": len(departure_events) if any(value is not None for value in departures) else None,
        "departure_time_s": departure_time if any(value is not None for value in departures) else None,
        "first_departure_time_s": first_departure_time,
        "departure_tolerance_ratio": float(departure_tolerance_ratio),
        "departure_consecutive_steps": consecutive,
        "ever_departed": bool(departure_events) if any(value is not None for value in departures) else None,
        "ever_departed_target_lane": bool(departure_events) if any(value is not None for value in departures) else None,
        "arrived": flags["arrived"],
        "arrival": flags["arrived"],
        "wrong_lane_arrival": flags["wrong_lane_arrival"],
        "start_lane_departure": flags["start_lane_departure"],
        "progress_m": progress_value,
        "route_progress_m": progress_value,
        "duration_s": total_duration,
        "speed_mean_m_s": float(np.mean(speeds)) if speeds else None,
        "speed_mean_mps": float(np.mean(speeds)) if speeds else None,
        "low_speed_threshold_m_s": float(low_speed_m_s),
        "low_speed_duration_s": low_speed_duration if any(speed is not None for speed in speeds_for_low) else None,
        "road_out": flags["road_out"],
        "crash": flags["crash"],
        "termination_reason": terminal_text,
        "cumulative_reward": cumulative_reward if reward_count else None,
        "return": cumulative_reward if reward_count else None,
        "episode_return": cumulative_reward if reward_count else None,
        "action_switch_count": action_switch_count if any(action is not _MISSING for action in actions) else None,
        "action_switches": action_switch_count if any(action is not _MISSING for action in actions) else None,
        "steering_variation_sum": float(np.sum(steering_variation)) if steering_values else None,
        "steering_variation_max": float(np.max(steering_variation)) if steering_variation else None,
        "steering_variation": float(np.sum(steering_variation)) if steering_values else None,
        "termination_flags": flags,
    }
    return result


def paired_deltas(metrics: Mapping[str, Any], p00_metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Return pattern-minus-P00 deltas for declared numeric metrics."""

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
    result: dict[str, Any] = {}
    for field in fields:
        current = _number(metrics.get(field))
        baseline = _number(p00_metrics.get(field))
        result[field] = None if current is None or baseline is None else current - baseline
        result[f"p00_{field}"] = baseline
        result[f"p00_delta_{field}"] = result[field]
    return result


def summarize_pattern_episodes(
    episodes: Mapping[str, Mapping[str, Any]],
    *,
    p00_id: str = "P00",
    **kwargs: Any,
) -> dict[str, Any]:
    """Add paired P00 deltas to per-pattern episode payloads."""

    baseline = episodes.get(p00_id)
    baseline_metrics = baseline.get("metrics", {}) if isinstance(baseline, Mapping) else {}
    result: dict[str, Any] = {}
    for pattern_id, episode in episodes.items():
        current = dict(episode)
        metrics = dict(current.get("metrics", {}))
        if pattern_id != p00_id and baseline_metrics:
            metrics["paired_p00"] = paired_deltas(metrics, baseline_metrics)
        current["metrics"] = metrics
        result[pattern_id] = current
    return result


__all__ = ["paired_deltas", "summarize_pattern_episodes", "summarize_trajectory"]
