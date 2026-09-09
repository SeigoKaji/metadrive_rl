"""Focused regression checks for partial closed-loop telemetry metrics.

These fixtures intentionally use a tiny synthetic trajectory.  They exercise
the public ``summarize_trajectory`` API and keep event-state coverage separate
from timestamp/interval coverage.
"""

from __future__ import annotations

import pytest

from input_attribution.trajectory_metrics import paired_deltas, summarize_trajectory


def _row(
    index: int,
    *,
    lane_offset: float | None = 0.0,
    speed_m_s: float | None = 1.0,
    include_pre_time: bool = True,
    include_post_time: bool = True,
) -> dict[str, object]:
    telemetry: dict[str, object] = {}
    if include_post_time:
        telemetry["simulation_time_s"] = (index + 1) * 0.1
    if lane_offset is not None:
        telemetry.update(
            {
                "target_lane_valid": True,
                "target_lane_offset_m": lane_offset,
                "lane_width_m": 2.0,
            }
        )
    if speed_m_s is not None:
        telemetry["speed_m_s"] = speed_m_s
    row: dict[str, object] = {"step": index, "post_telemetry": telemetry}
    if include_pre_time:
        row["pre_time"] = index * 0.1
    if include_post_time:
        row["post_time"] = (index + 1) * 0.1
    return row


def test_all_known_event_states_report_zero_and_complete_coverage() -> None:
    metrics = summarize_trajectory([_row(0), _row(1)])

    assert metrics["duration_s"] == pytest.approx(0.2)
    assert metrics["interval_coverage_rate"] == 1.0
    assert metrics["departure_time_s"] == 0.0
    assert metrics["departure_count"] == 0
    assert metrics["ever_departed"] is False
    assert metrics["low_speed_duration_s"] == 0.0
    assert metrics["low_speed_count"] == 0
    assert metrics["ever_low_speed"] is False
    assert metrics["departure_state_measurement"]["status"] == "complete"
    assert metrics["low_speed_state_measurement"]["status"] == "complete"


def test_partial_event_state_keeps_known_rms_but_does_not_claim_zero_events() -> None:
    metrics = summarize_trajectory(
        [
            _row(0),
            _row(1, lane_offset=None, speed_m_s=None),
        ]
    )

    assert metrics["duration_s"] == pytest.approx(0.2)
    assert metrics["lane_rms_m"] == 0.0
    assert metrics["lane_metric_denominator_count"] == 1
    assert metrics["departure_count"] is None
    assert metrics["departure_time_s"] is None
    assert metrics["ever_departed"] is None
    assert metrics["low_speed_count"] is None
    assert metrics["low_speed_duration_s"] is None
    assert metrics["ever_low_speed"] is None
    assert metrics["departure_state_measurement"]["status"] == "partial"
    assert metrics["departure_state_measurement"]["missing_count"] == 1
    assert metrics["low_speed_state_measurement"]["status"] == "partial"
    assert metrics["interval_measurement"]["status"] == "complete"
    departure_measurement = metrics["departure_duration_measurement"]
    assert departure_measurement["known_event_duration_s"] == 0.0
    assert departure_measurement["state_known_interval_count"] == 1
    assert departure_measurement["state_known_duration_s"] == pytest.approx(0.1)
    assert departure_measurement["state_missing_interval_count"] == 1


def test_all_unknown_event_states_are_unavailable_without_zero_filling() -> None:
    metrics = summarize_trajectory(
        [
            _row(0, lane_offset=None, speed_m_s=None),
            _row(1, lane_offset=None, speed_m_s=None),
        ]
    )

    assert metrics["duration_s"] == pytest.approx(0.2)
    assert metrics["lane_rms_m"] is None
    assert metrics["lane_metric_denominator_count"] == 0
    assert metrics["departure_count"] is None
    assert metrics["departure_time_s"] is None
    assert metrics["ever_departed"] is None
    assert metrics["low_speed_count"] is None
    assert metrics["low_speed_duration_s"] is None
    assert metrics["ever_low_speed"] is None
    assert metrics["departure_state_measurement"]["status"] == "unavailable"
    assert metrics["low_speed_state_measurement"]["status"] == "unavailable"


def test_known_true_event_with_unknown_interval_reports_occurrence_only() -> None:
    metrics = summarize_trajectory(
        [
            _row(
                0,
                lane_offset=1.2,
                speed_m_s=0.1,
                include_pre_time=False,
            ),
            _row(1),
        ]
    )

    assert metrics["ever_departed"] is True
    assert metrics["departure_count"] == 1
    assert metrics["departure_time_s"] is None
    assert metrics["departure_duration_measurement"]["status"] == "unavailable"
    assert metrics["departure_duration_measurement"]["missing_event_indices"] == [0]
    assert metrics["ever_low_speed"] is True
    assert metrics["low_speed_count"] == 1
    assert metrics["low_speed_duration_s"] is None
    assert metrics["low_speed_duration_measurement"]["status"] == "unavailable"
    assert metrics["duration_s"] is None
    assert metrics["interval_measurement"]["status"] == "partial"


def test_known_event_and_unknown_state_keep_partial_event_sum_as_detail_only() -> None:
    metrics = summarize_trajectory(
        [
            _row(0, lane_offset=1.2, speed_m_s=0.1),
            _row(1, lane_offset=None, speed_m_s=None),
            _row(2, lane_offset=0.0, speed_m_s=1.0),
        ]
    )

    assert metrics["ever_departed"] is True
    assert metrics["departure_count"] is None
    assert metrics["departure_time_s"] is None
    assert metrics["departure_duration_measurement"]["status"] == "partial"
    assert metrics["departure_duration_measurement"]["measured_duration_s"] == pytest.approx(0.1)
    assert metrics["departure_duration_measurement"]["state_missing_count"] == 1
    assert metrics["ever_low_speed"] is True
    assert metrics["low_speed_count"] is None
    assert metrics["low_speed_duration_s"] is None
    assert metrics["low_speed_duration_measurement"]["status"] == "partial"
    assert metrics["low_speed_duration_measurement"]["measured_duration_s"] == pytest.approx(0.1)
    assert metrics["lane_rms_m"] == pytest.approx((1.2**2 / 2.0) ** 0.5)
    assert metrics["lane_metric_denominator_count"] == 2


def test_paired_delta_does_not_turn_missing_values_into_zero() -> None:
    deltas = paired_deltas(
        {"lane_rms_m": 1.0, "departure_time_s": None},
        {"lane_rms_m": None, "departure_time_s": 0.2},
    )

    assert deltas["lane_rms_m"] is None
    assert deltas["p00_delta_lane_rms_m"] is None
    assert deltas["departure_time_s"] is None
    assert deltas["p00_delta_departure_time_s"] is None


def test_empty_trajectory_keeps_event_metrics_unavailable() -> None:
    metrics = summarize_trajectory([])

    assert metrics["duration_s"] is None
    assert metrics["departure_count"] is None
    assert metrics["departure_time_s"] is None
    assert metrics["ever_departed"] is None
    assert metrics["low_speed_count"] is None
    assert metrics["low_speed_duration_s"] is None
    assert metrics["ever_low_speed"] is None
    assert metrics["departure_state_measurement"]["status"] == "not_applicable"
    assert metrics["low_speed_state_measurement"]["status"] == "not_applicable"


def test_departure_threshold_stays_separate_from_raw_observed_true_state() -> None:
    metrics = summarize_trajectory(
        [_row(0, lane_offset=1.2)],
        departure_consecutive_steps=2,
    )

    assert metrics["departure_count"] == 0
    assert metrics["ever_departed"] is False
    assert metrics["ever_departed_observed"] is True
    assert metrics["departure_state_measurement"]["ever_true"] is True


def test_partial_lane_validity_does_not_treat_unknown_as_invalid_time() -> None:
    metrics = summarize_trajectory(
        [
            _row(0, lane_offset=0.0, speed_m_s=1.0),
            _row(1, lane_offset=None, speed_m_s=1.0),
        ]
    )

    assert metrics["poststep_count"] == 2
    assert metrics["valid_count"] == 1
    assert metrics["valid_time_s"] is None
    assert metrics["valid_rate"] is None
    assert metrics["lane_validity_state_measurement"]["status"] == "partial"
    assert metrics["lane_validity_state_measurement"]["missing_count"] == 1
    valid_time = metrics["valid_time_measurement"]
    assert valid_time["status"] == "partial"
    assert valid_time["known_valid_time_s"] == pytest.approx(0.1)
    assert valid_time["known_valid_rate"] == pytest.approx(0.5)
    assert valid_time["known_valid_interval_count"] == 1
    assert valid_time["expected_interval_count"] == 2


def test_policy_failure_attempt_is_excluded_from_poststep_metrics() -> None:
    metrics = summarize_trajectory(
        [
            {
                "step": 0,
                "env_step_attempted": False,
                "env_step_called": False,
                "env_step_returned": False,
                "failure_phase": "policy",
                "post_telemetry": None,
            }
        ]
    )

    assert metrics["record_count"] == 1
    assert metrics["poststep_count"] == 0
    assert metrics["executed_step_count"] == 0
    assert metrics["attempted_step_count"] == 0
    assert metrics["nonreturned_step_count"] == 0
    assert metrics["env_step_returned_count"] == 0
    assert metrics["duration_s"] is None
    assert metrics["lane_rms_m"] is None


def test_env_step_exception_is_excluded_but_nonreturned_attempt_is_counted() -> None:
    metrics = summarize_trajectory(
        [
            {
                "step": 0,
                "env_step_attempted": True,
                "env_step_called": True,
                "env_step_returned": False,
                "failure_phase": "env_step",
                "post_telemetry": None,
            }
        ]
    )

    assert metrics["record_count"] == 1
    assert metrics["poststep_count"] == 0
    assert metrics["executed_step_count"] == 0
    assert metrics["attempted_step_count"] == 1
    assert metrics["nonreturned_step_count"] == 1
    assert metrics["env_step_returned_count"] == 0
    assert metrics["physical_state_unknown_count"] == 1
    assert metrics["duration_s"] is None


def test_returned_step_with_missing_post_telemetry_remains_in_physical_denominator() -> None:
    metrics = summarize_trajectory(
        [
            {
                "step": 0,
                "pre_time": 0.0,
                "post_time": 0.1,
                "env_step_attempted": True,
                "env_step_called": True,
                "env_step_returned": True,
                "post_telemetry": {
                    "simulation_time_s": 0.1,
                    "target_lane_valid": True,
                    "target_lane_offset_m": 0.0,
                    "lane_width_m": 2.0,
                    "speed_m_s": 1.0,
                },
            },
            {
                "step": 1,
                "pre_time": 0.1,
                "post_time": 0.2,
                "env_step_attempted": True,
                "env_step_called": True,
                "env_step_returned": True,
                "post_telemetry": None,
            },
        ]
    )

    assert metrics["record_count"] == 2
    assert metrics["poststep_count"] == 2
    assert metrics["executed_step_count"] == 2
    assert metrics["attempted_step_count"] == 2
    assert metrics["nonreturned_step_count"] == 0
    assert metrics["env_step_returned_count"] == 2
    assert metrics["duration_s"] == pytest.approx(0.2)
    assert metrics["telemetry_coverage"]["lane_error"]["measured_count"] == 1
    assert metrics["lane_rms_m"] == 0.0
    assert metrics["lane_metric_denominator_count"] == 1
    assert metrics["lane_validity_state_measurement"]["missing_count"] == 1


def test_known_false_events_with_unknown_intervals_keep_event_zero_and_episode_na() -> None:
    metrics = summarize_trajectory(
        [
            {
                "step": 0,
                "env_step_returned": True,
                "post_telemetry": {
                    "target_lane_valid": True,
                    "target_lane_offset_m": 0.0,
                    "lane_width_m": 2.0,
                    "speed_m_s": 1.0,
                },
            },
            {
                "step": 1,
                "env_step_returned": True,
                "post_telemetry": {
                    "target_lane_valid": True,
                    "target_lane_offset_m": 0.0,
                    "lane_width_m": 2.0,
                    "speed_m_s": 1.0,
                },
            },
        ]
    )

    assert metrics["duration_s"] is None
    assert metrics["duration_measurement"]["status"] == "unavailable"
    assert metrics["departure_time_s"] == 0.0
    assert metrics["low_speed_duration_s"] == 0.0
    assert metrics["departure_state_measurement"]["status"] == "complete"
    assert metrics["low_speed_state_measurement"]["status"] == "complete"


def test_legacy_records_without_step_flags_remain_physical_steps() -> None:
    metrics = summarize_trajectory([_row(0), _row(1)])

    assert metrics["record_count"] == 2
    assert metrics["poststep_count"] == 2
    assert metrics["executed_step_count"] == 2
    assert metrics["attempted_step_count"] == 2
    assert metrics["nonreturned_step_count"] == 0
