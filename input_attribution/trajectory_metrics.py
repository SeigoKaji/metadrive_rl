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


def _env_step_returned(record: Mapping[str, Any]) -> bool:
    """Return whether a record represents a physically returned env step.

    New closed-loop records carry ``env_step_returned`` and it is authoritative:
    a called step that raised must not become a post-step observation.  Older
    saved records only carry ``env_step_called``; retain that legacy fallback.
    Records with neither marker predate the distinction and remain physical
    rows for backward compatibility.
    """

    if "env_step_returned" in record:
        return _boolean(record.get("env_step_returned")) is True
    if "env_step_called" in record:
        return _boolean(record.get("env_step_called")) is True
    return True


def _env_step_attempted(record: Mapping[str, Any]) -> bool:
    """Return whether execution attempted to invoke the environment."""

    if "env_step_attempted" in record:
        return _boolean(record.get("env_step_attempted")) is True
    if "env_step_called" in record:
        return _boolean(record.get("env_step_called")) is True
    if "env_step_returned" in record:
        returned = _boolean(record.get("env_step_returned"))
        # A standalone false return marker is the legacy/minimal encoding of
        # an env.step exception.  New policy-abort records also carry
        # ``env_step_attempted=False`` and are handled by the first branch.
        return returned is not None
    # Legacy records have no execution marker; they are already persisted as
    # completed post-step rows and count as attempted physical steps.
    return True


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


def _measurement_info(
    values: Sequence[Any],
    times: Sequence[float | None],
) -> dict[str, Any]:
    """Describe a telemetry series without converting missing samples to zero."""

    expected = len(values)
    known_steps = [index for index, value in enumerate(values) if value is not None]
    known_times = [
        float(times[index])
        for index in known_steps
        if index < len(times) and times[index] is not None
    ]
    missing_steps = [index for index, value in enumerate(values) if value is None]
    missing_ranges: list[dict[str, Any]] = []
    if missing_steps:
        start = previous = missing_steps[0]
        for index in missing_steps[1:]:
            if index == previous + 1:
                previous = index
                continue
            missing_ranges.append(
                {
                    "first_step": start,
                    "last_step": previous,
                    "count": previous - start + 1,
                    "first_time_s": times[start] if start < len(times) else None,
                    "last_time_s": times[previous] if previous < len(times) else None,
                }
            )
            start = previous = index
        missing_ranges.append(
            {
                "first_step": start,
                "last_step": previous,
                "count": previous - start + 1,
                "first_time_s": times[start] if start < len(times) else None,
                "last_time_s": times[previous] if previous < len(times) else None,
            }
        )
    return {
        "measured_count": len(known_steps),
        "expected_count": expected,
        "missing_count": max(0, expected - len(known_steps)),
        "coverage_rate": (len(known_steps) / expected) if expected else None,
        "first_step": known_steps[0] if known_steps else None,
        "last_step": known_steps[-1] if known_steps else None,
        "first_time_s": known_times[0] if known_times else None,
        "last_time_s": known_times[-1] if known_times else None,
        "missing_ranges": missing_ranges,
        "missing_spans": missing_ranges,
    }


def _state_measurement(
    values: Sequence[bool | None],
    times: Sequence[float | None],
) -> dict[str, Any]:
    """Describe a tri-state event series without treating unknown as false.

    Event metrics need two independent kinds of coverage.  ``values`` tells us
    whether the event state was observed at each step, while the interval
    measurement (computed separately from timestamps) tells us how long that
    state lasted.  Keep both the known true/false samples and unknown samples
    here so an empty list of true samples cannot be mistaken for a confirmed
    absence of the event.
    """

    info = _measurement_info(values, times)
    true_indices = [index for index, value in enumerate(values) if value is True]
    false_indices = [index for index, value in enumerate(values) if value is False]
    missing_indices = [index for index, value in enumerate(values) if value is None]
    expected_count = len(values)
    if expected_count == 0:
        status = "not_applicable"
    elif missing_indices and (true_indices or false_indices):
        status = "partial"
    elif missing_indices:
        status = "unavailable"
    else:
        status = "complete"
    info.update(
        {
            "status": status,
            "true_count": len(true_indices),
            "false_count": len(false_indices),
            "known_true_indices": true_indices,
            "known_false_indices": false_indices,
            "missing_indices": missing_indices,
            "ever_true": (
                True
                if true_indices
                else None
                if missing_indices
                else False
            ),
        }
    )
    return info


def _event_duration_measurement(
    event_indices: Sequence[int],
    deltas: Sequence[float | None],
) -> dict[str, Any]:
    """Report duration coverage for the intervals belonging to one event type.

    Summing only known intervals can make a partially measured event look
    complete.  Keep that partial sum for diagnostics, while callers use the
    status/coverage fields to decide whether the public duration is valid.
    """

    indices = [int(index) for index in event_indices]
    measured_indices = [index for index in indices if index < len(deltas) and deltas[index] is not None]
    missing_indices = [index for index in indices if index >= len(deltas) or deltas[index] is None]
    measured_duration = (
        float(sum(float(deltas[index]) for index in measured_indices))
        if measured_indices
        else None
    )
    expected_count = len(indices)
    measured_count = len(measured_indices)
    if expected_count == 0:
        status = "not_applicable"
        reason = "no_events"
    elif measured_count == expected_count:
        status = "complete"
        reason = None
    elif measured_count == 0:
        status = "unavailable"
        reason = "missing_event_interval"
    else:
        status = "partial"
        reason = "missing_event_interval"
    return {
        "status": status,
        "measured_duration_s": measured_duration,
        "measured_count": measured_count,
        "expected_count": expected_count,
        "missing_count": len(missing_indices),
        "coverage_rate": (measured_count / expected_count) if expected_count else None,
        "measured_event_indices": measured_indices,
        "missing_event_indices": missing_indices,
        "reason": reason,
    }


def _event_measurement(
    event_indices: Sequence[int],
    deltas: Sequence[float | None],
    state_measurement: Mapping[str, Any],
) -> dict[str, Any]:
    """Combine event-interval and event-state coverage.

    A duration can be fully measured for the observed true samples while the
    event state is still unknown at another step.  In that case retain the
    observed duration as a diagnostic, but mark the public measurement
    partial so callers do not present it as the episode total.
    """

    result = _event_duration_measurement(event_indices, deltas)
    known_state_indices = [
        *state_measurement["known_true_indices"],
        *state_measurement["known_false_indices"],
    ]
    known_state_deltas = [
        deltas[index]
        for index in known_state_indices
        if index < len(deltas) and deltas[index] is not None
    ]
    known_event_duration = result["measured_duration_s"]
    if not event_indices and state_measurement["false_count"]:
        # No observed event samples is a measured zero only when at least one
        # state was explicitly observed as false.  Keep this diagnostic
        # separate from the public total, which remains N/A if another state
        # is unknown.
        known_event_duration = 0.0
    result.update(
        {
            "state_status": state_measurement["status"],
            "state_measured_count": state_measurement["measured_count"],
            "state_expected_count": state_measurement["expected_count"],
            "state_missing_count": state_measurement["missing_count"],
            "state_coverage_rate": state_measurement["coverage_rate"],
            "state_missing_indices": state_measurement["missing_indices"],
            "state_known_interval_count": len(known_state_deltas),
            "state_expected_interval_count": state_measurement["expected_count"],
            "state_missing_interval_count": max(
                0,
                state_measurement["expected_count"] - len(known_state_deltas),
            ),
            "state_known_duration_s": (
                float(sum(float(value) for value in known_state_deltas))
                if known_state_deltas
                else None
            ),
            "known_event_duration_s": known_event_duration,
        }
    )
    if state_measurement["missing_count"]:
        if result["status"] in {"complete", "not_applicable"}:
            result["status"] = "partial"
        if result["reason"] is None or result["reason"] == "no_events":
            result["reason"] = "missing_event_state"
        else:
            result["reason"] = "missing_event_state_and_interval"
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

    Only records whose environment step returned are included in physical
    post-step metrics.  Non-returned attempts remain in the execution counts
    so a runtime failure cannot be mistaken for a telemetry sample.
    RMS uses only valid post-step target-lane samples as its denominator. The
    ``valid_rate`` is time-weighted when timestamps exist; ``valid_count`` and
    ``poststep_count`` remain available so a reader can inspect sample coverage.
    """

    raw_rows = [dict(record) for record in records]
    returned_mask = [_env_step_returned(record) for record in raw_rows]
    attempted_mask = [_env_step_attempted(record) for record in raw_rows]
    rows = [record for record, returned in zip(raw_rows, returned_mask, strict=False) if returned]
    record_count = len(raw_rows)
    attempted_step_count = int(sum(attempted_mask))
    env_step_returned_count = int(sum(returned_mask))
    nonreturned_step_count = int(
        sum(attempted and not returned for attempted, returned in zip(attempted_mask, returned_mask, strict=False))
    )
    excluded_step_count = record_count - len(rows)
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
    measured_duration = float(sum(known_deltas)) if known_deltas else None
    interval_expected = len(rows)
    duration_complete = bool(interval_expected) and verified_time_interval_count == interval_expected
    total_duration = measured_duration if duration_complete else None
    lane_validity_state_measurement = _state_measurement(lane_validity, times)
    lane_validity_complete = bool(lane_validity) and lane_validity_state_measurement["missing_count"] == 0
    known_valid_durations = [
        delta for delta, valid in zip(deltas, lane_validity, strict=False)
        if valid is True and delta is not None
    ]
    known_valid_time = (
        float(sum(known_valid_durations))
        if known_valid_durations
        else 0.0
        if lane_validity_state_measurement["false_count"]
        else None
    )
    valid_time = known_valid_time if duration_complete else None
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
    public_valid_time = (
        valid_time
        if lane_reference_known and lane_validity_complete
        else None
    )
    public_valid_rate = (
        valid_rate
        if lane_reference_known and lane_validity_complete
        else None
    )

    lane_error_values = [_lane_error(telemetry) for telemetry in telemetry_rows]
    lane_valid_values = [_lane_valid(telemetry) for telemetry in telemetry_rows]
    speed_values = [_speed_m_s(telemetry) for telemetry in telemetry_rows]
    progress_values = [_progress_m(telemetry) for telemetry in telemetry_rows]
    # Each closed-loop record represents one pre→post simulator interval when
    # explicit pre/post clocks are present.  A collector without a first
    # pre-time may leave only the first interval unknown, but its denominator
    # remains the number of attempted steps.
    interval_coverage = (
        verified_time_interval_count / interval_expected
        if interval_expected
        else None
    )
    interval_measurement = _measurement_info(deltas, times)
    duration_measurement = {
        "status": (
            "unavailable"
            if interval_expected == 0 or verified_time_interval_count == 0
            else "partial"
            if verified_time_interval_count < interval_expected
            else "complete"
        ),
        "measured_count": verified_time_interval_count,
        "expected_count": interval_expected,
        "missing_count": max(0, interval_expected - verified_time_interval_count),
        "coverage_rate": interval_coverage,
        "measured_duration_s": measured_duration,
        "missing_ranges": interval_measurement["missing_ranges"],
        "reason": (
            "no_records"
            if interval_expected == 0
            else "missing_pre_or_post_time"
            if verified_time_interval_count < interval_expected
            else None
        ),
    }
    known_times = [value for value in times if value is not None]
    measurement_range = {
        "first_time_s": float(known_times[0]) if known_times else None,
        "last_time_s": float(known_times[-1]) if known_times else None,
        "span_s": (
            float(known_times[-1] - known_times[0])
            if len(known_times) >= 2
            else None
        ),
        "timestamp_count": len(known_times),
        "expected_timestamp_count": len(rows),
        "missing_timestamp_count": max(0, len(rows) - len(known_times)),
        "interval_count": verified_time_interval_count,
        "expected_interval_count": interval_expected,
        "interval_coverage_rate": interval_coverage,
        "missing_interval_count": duration_measurement["missing_count"],
        "missing_interval_ranges": duration_measurement["missing_ranges"],
        "measured_duration_s": duration_measurement["measured_duration_s"],
        "duration_measurement_reason": duration_measurement["reason"],
    }
    telemetry_coverage = {
        "lane_error": _measurement_info(lane_error_values, times),
        "lane_valid": _measurement_info(lane_valid_values, times),
        "speed_m_s": _measurement_info(speed_values, times),
        "progress_m": _measurement_info(progress_values, times),
        "clock": _measurement_info(times, times),
    }
    known_state_interval_count = sum(
        delta is not None
        for delta, state in zip(deltas, lane_validity, strict=False)
        if state is not None
    )
    valid_time_measurement = {
        "status": (
            "unavailable"
            if not rows or lane_validity_state_measurement["measured_count"] == 0
            else "complete"
            if lane_validity_complete and duration_complete
            else "partial"
        ),
        "known_valid_time_s": known_valid_time,
        "known_duration_s": measured_duration,
        "known_valid_rate": (
            known_valid_time / measured_duration
            if known_valid_time is not None
            and measured_duration is not None
            and measured_duration > 0
            else None
        ),
        "known_valid_interval_count": len(known_valid_durations),
        "known_state_interval_count": known_state_interval_count,
        "expected_interval_count": len(rows),
        "missing_interval_count": max(0, len(rows) - known_state_interval_count),
        "state_measured_count": lane_validity_state_measurement["measured_count"],
        "state_expected_count": lane_validity_state_measurement["expected_count"],
        "state_missing_count": lane_validity_state_measurement["missing_count"],
        "state_coverage_rate": lane_validity_state_measurement["coverage_rate"],
        "interval_coverage_rate": interval_coverage,
        "reason": (
            "no_records"
            if not rows
            else "missing_lane_validity_state"
            if not lane_validity_complete
            else "missing_pre_or_post_time"
            if not duration_complete
            else None
        ),
    }

    departure_state_measurement = _state_measurement(departures, times)
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
    departure_event_indices = [index for index, value in enumerate(departures) if value is True]
    departure_duration_measurement = _event_measurement(
        departure_event_indices,
        deltas,
        departure_state_measurement,
    )
    departure_observations = departure_event_indices
    departure_state_complete = bool(departures) and departure_state_measurement["missing_count"] == 0
    departure_time = (
        None
        if not departures
        else None
        if not departure_state_complete
        else departure_duration_measurement["measured_duration_s"]
        if departure_observations and departure_duration_measurement["status"] == "complete"
        else None
        if departure_observations
        else 0.0
        if departure_state_complete
        else None
    )
    first_departure_index = departure_events[0] if departure_events else None
    first_departure_has_unknown_prefix = bool(
        first_departure_index is not None
        and any(value is None for value in departures[:first_departure_index])
    )
    first_departure_time = (
        times[first_departure_index]
        if first_departure_index is not None
        and not first_departure_has_unknown_prefix
        and times[first_departure_index] is not None
        else None
    )

    speeds_for_low = [_speed_m_s(telemetry) for telemetry in telemetry_rows]
    low_speed_state_values = [
        None if speed is None else speed < float(low_speed_m_s)
        for speed in speeds_for_low
    ]
    low_speed_state_measurement = _state_measurement(low_speed_state_values, times)
    low_speed_observations = [
        speed for speed in speeds_for_low
        if speed is not None and speed < float(low_speed_m_s)
    ]
    low_speed_event_indices = [
        index
        for index, value in enumerate(low_speed_state_values)
        if value is True
    ]
    low_speed_events: list[int] = []
    index = 0
    while index < len(low_speed_state_values):
        if low_speed_state_values[index] is not True:
            index += 1
            continue
        low_speed_events.append(index)
        while index < len(low_speed_state_values) and low_speed_state_values[index] is True:
            index += 1
    low_speed_duration_measurement = _event_measurement(
        low_speed_event_indices,
        deltas,
        low_speed_state_measurement,
    )
    low_speed_state_complete = bool(low_speed_state_values) and low_speed_state_measurement["missing_count"] == 0
    low_speed_duration = (
        None
        if not low_speed_state_values
        else None
        if not low_speed_state_complete
        else low_speed_duration_measurement["measured_duration_s"]
        if low_speed_observations and low_speed_duration_measurement["status"] == "complete"
        else None
        if low_speed_observations
        else 0.0
        if low_speed_state_complete
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
    lane_rms = float(np.sqrt(np.mean(np.square(errors)))) if errors else None
    lane_max_abs = float(np.max(np.abs(errors))) if errors else None
    lane_rms_measurement = {
        "status": (
            "unavailable"
            if valid_count == 0
            else "complete"
            if valid_count == poststep_count
            else "partial"
        ),
        "measured_count": valid_count,
        "expected_count": poststep_count,
        "missing_count": max(0, poststep_count - valid_count),
        "coverage_rate": (valid_count / poststep_count) if poststep_count else None,
        "denominator": valid_count,
        "reason": "no_valid_target_lane_samples" if valid_count == 0 else None,
    }
    ever_departed = (
        None
        if not departures
        else True
        if departure_events
        else None
        if departure_state_measurement["missing_count"]
        else False
    )
    ever_departed_observed = (
        departure_state_measurement["ever_true"]
        if departures
        else None
    )
    ever_low_speed = (
        None
        if not low_speed_state_values
        else True
        if low_speed_state_measurement["true_count"]
        else None
        if low_speed_state_measurement["missing_count"]
        else False
    )
    departure_count = (
        len(departure_events)
        if departures and departure_state_measurement["missing_count"] == 0
        else None
    )
    low_speed_count = (
        len(low_speed_events)
        if low_speed_state_values and low_speed_state_measurement["missing_count"] == 0
        else None
    )
    result: dict[str, Any] = {
        "status": metric_status,
        "telemetry_status": (
            "unavailable"
            if not known_times and not any(value is not None for value in lane_error_values)
            else "partial"
            if any(
                item["missing_count"] > 0
                for item in telemetry_coverage.values()
                if isinstance(item, Mapping)
            )
            else "complete"
        ),
        "poststep_count": poststep_count,
        "record_count": record_count,
        "attempted_step_count": attempted_step_count,
        "executed_step_count": poststep_count,
        "env_step_returned_count": env_step_returned_count,
        "nonreturned_step_count": nonreturned_step_count,
        "not_returned_step_count": nonreturned_step_count,
        "physical_state_unknown_count": nonreturned_step_count,
        "excluded_step_count": excluded_step_count,
        "valid_count": valid_count,
        "valid_poststep_count": valid_count,
        "verified_time_interval_count": verified_time_interval_count,
        "duration_measurement": duration_measurement,
        "interval_measurement": duration_measurement,
        "interval_coverage_rate": interval_coverage,
        "departure_duration_measurement": departure_duration_measurement,
        "low_speed_duration_measurement": low_speed_duration_measurement,
        "departure_state_measurement": departure_state_measurement,
        "low_speed_state_measurement": low_speed_state_measurement,
        "event_state_coverage": {
            "departure": departure_state_measurement,
            "low_speed": low_speed_state_measurement,
        },
        "measurement_range": measurement_range,
        "telemetry_measurement_range": measurement_range,
        "valid_time_measurement": valid_time_measurement,
        "lane_validity_state_measurement": lane_validity_state_measurement,
        "measured_time_span_s": measurement_range["span_s"],
        "measured_first_time_s": measurement_range["first_time_s"],
        "measured_last_time_s": measurement_range["last_time_s"],
        "telemetry_coverage": telemetry_coverage,
        "partial_telemetry": bool(
            any(
                item["missing_count"] > 0
                for item in telemetry_coverage.values()
                if isinstance(item, Mapping)
            )
        ),
        "measured_step_count": {
            "poststep": poststep_count,
            "lane_error": int(sum(value is not None for value in lane_error_values)),
            "speed_m_s": int(sum(value is not None for value in speed_values)),
            "progress_m": int(sum(value is not None for value in progress_values)),
            "clock": len(known_times),
        },
        "valid_time_s": public_valid_time,
        "valid_time_seconds": public_valid_time,
        "valid_time": public_valid_time,
        "valid_poststep_time_s": public_valid_time,
        "valid_rate": public_valid_rate,
        "sample_valid_rate": sample_valid_rate,
        "valid_poststep_rate": sample_valid_rate,
        "lane_metric_denominator": "valid_poststep_target_lane_samples",
        "lane_metric_denominator_count": valid_count,
        "lane_metric_measurement": lane_rms_measurement,
        "lane_rms_measurement": lane_rms_measurement,
        "lane_rms_m": lane_rms,
        "lane_max_abs_m": lane_max_abs,
        "lane_error_rms_m": lane_rms,
        "lane_error_max_abs_m": lane_max_abs,
        "target_lane_rms_m": lane_rms,
        "target_lane_max_abs_m": lane_max_abs,
        "departure_count": departure_count,
        "departure_observed_count": len(departure_events),
        "departure_time_s": departure_time,
        "first_departure_time_s": first_departure_time,
        "departure_tolerance_ratio": float(departure_tolerance_ratio),
        "departure_consecutive_steps": consecutive,
        "ever_departed": ever_departed,
        "ever_departed_target_lane": ever_departed,
        "ever_departed_observed": ever_departed_observed,
        "ever_departed_raw": ever_departed_observed,
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
        "low_speed_count": low_speed_count,
        "low_speed_observed_count": len(low_speed_events),
        "ever_low_speed": ever_low_speed,
        "low_speed_duration_s": low_speed_duration,
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
