"""Focused regression checks for A/B applicability and telemetry provenance."""

from __future__ import annotations

import numpy as np
import pytest

from input_attribution.adapters.fake import FakePolicyAdapter
from input_attribution.artifacts import create_run
from input_attribution.closed_loop import ClosedLoopError, ClosedLoopResult, run_closed_loop
from input_attribution.interventions import InterventionError
from input_attribution.offline import analyze_offline
from input_attribution.tests.test_runtime import _TrackingAdapter, _config, _schema
from input_attribution.schema import InputSchema, InputSpec
from input_attribution.metrics import (
    compare_distributions,
    summarize_distribution_comparison,
    summarize_intervention_records,
)
from input_attribution.trajectory_metrics import summarize_trajectory


def _provenance_schema(dimension: int = 3) -> InputSchema:
    common = {
        "description": "synthetic scalar",
        "physical_quantity": "synthetic value",
        "model_representation": "raw scalar",
        "normalization": "none",
        "reference": "synthetic fixture",
        "source": "input_attribution/tests/test_revision_runtime.py",
        "variants": (
            {
                "id": "neutral",
                "operation": "neutral",
                "value": 0.5,
                "meaning": "synthetic central diagnostic level",
                "evidence": "synthetic fixture source",
                "confirmed": True,
            },
            {
                "id": "reflect",
                "operation": "reflection",
                "center": 0.5,
                "meaning": "synthetic reflection around a confirmed center",
                "evidence": "synthetic fixture source",
                "confirmed": True,
            },
            {
                "id": "low",
                "operation": "fixed_level",
                "value": 0.4,
                "meaning": "synthetic lower diagnostic level",
                "evidence": "synthetic fixture source",
                "confirmed": True,
            },
        ),
    }
    return InputSchema(
        dimension=dimension,
        inputs=tuple(
            InputSpec(index, f"x{index}", f"x{index}", group=f"x{index}", **common)
            for index in range(dimension)
        ),
    )


class _RoadSwitchAdapter(_TrackingAdapter):
    """Synthetic road fixture whose context changes during a live episode."""

    switch_at = 2

    def telemetry(self, env, info=None, *, phase="post", step=0):
        result = super().telemetry(env, info, phase=phase, step=step)
        result["road_segment_id"] = ["synthetic", 0 if step < self.switch_at else 1]
        return result


def test_offline_counts_target_eligible_applied_and_out_of_scope_independently():
    schema = _provenance_schema()
    observations = np.zeros((127, schema.dimension), dtype=np.float32)
    original = observations.copy()
    contexts = [
        {"road_segment_id": ["synthetic", 0], "simulation_time_s": float(index) * 0.1}
        for index in range(19)
    ] + [{"road_segment_id": ["synthetic", 1]} for _ in range(108)]
    result = analyze_offline(
        observations,
        FakePolicyAdapter(dimension=schema.dimension, action_count=3),
        schema,
        patterns=[
            {
                "id": "reference",
                "method": "reference",
                "indices": [0],
                "reference_id": "donor",
                "metadata": {"compatibility_keys": ["road_segment_id"]},
            }
        ],
        reference_observations={"donor": np.ones(schema.dimension, dtype=np.float32)},
        reference_contexts={"donor": {"road_segment_id": ["synthetic", 0]}},
        observation_contexts=contexts,
        strict=False,
    )

    pattern = result.patterns[0]
    counts = pattern.summary["intervention_counts"]
    assert counts["target_step_count"] == 127
    assert counts["eligible_count"] == 19
    assert counts["applied_count"] == 19
    assert counts["changed_count_exact"] == 19
    assert counts["noop_count"] == 0
    assert counts["skipped_count"] == 108
    assert counts["out_of_scope_count"] == 108
    assert pattern.rows[0]["road_segment_id"] == ["synthetic", 0]
    assert pattern.rows[19]["road_segment_id"] == ["synthetic", 1]
    assert pattern.rows[0]["time_s"] == pytest.approx(0.0)
    assert pattern.rows[19]["time_s"] is None
    assert pattern.rows[19]["skipped"] is True
    assert pattern.rows[0]["selected_probability_abs_delta"] >= 0.0
    np.testing.assert_array_equal(observations, original)


def test_offline_typed_variants_cover_all_127_steps_with_exact_counts():
    """Typed A patterns share one original 127-step fixture and full denominator."""

    schema = _provenance_schema()
    observations = np.zeros((127, schema.dimension), dtype=np.float32)
    policy = FakePolicyAdapter(dimension=schema.dimension, action_count=3)
    before = policy.fingerprint()
    result = analyze_offline(
        observations,
        policy,
        schema,
        patterns=[
            {
                "id": "typed_neutral",
                "method": "neutral",
                "indices": [0],
                "variant_id": "neutral",
            },
            {
                "id": "typed_reflect",
                "method": "reflection",
                "indices": [0],
                "variant_id": "reflect",
            },
            {
                "id": "typed_low",
                "method": "fixed_level",
                "indices": [0],
                "variant_id": "low",
            },
        ],
        strict=True,
    )

    assert [item.pattern.pattern_id for item in result.patterns] == [
        "typed_neutral",
        "typed_reflect",
        "typed_low",
    ]
    expected_delta = {
        "typed_neutral": 0.5,
        "typed_reflect": 1.0,
        "typed_low": 0.4,
    }
    for item in result.patterns:
        counts = item.summary["intervention_counts"]
        assert counts["target_step_count"] == 127
        assert counts["eligible_count"] == 127
        assert counts["applied_count"] == 127
        assert counts["skipped_count"] == 0
        assert counts["changed_count_exact"] == 127
        assert counts["changed_element_count_exact"] == 127
        assert item.summary["assessment_status"] == "evaluated"
        assert item.summary["step_masks"]["changed_exact"] == [True] * 127
        assert all(row["changed_exact"] is True for row in item.rows)
        delta = counts["per_input_delta"]["0"]
        assert delta["applied_count"] == 127
        assert delta["delta_abs_count"] == 127
        assert delta["delta_abs_mean"] == pytest.approx(expected_delta[item.pattern.pattern_id])
        assert delta["delta_abs_max"] == pytest.approx(expected_delta[item.pattern.pattern_id])
        assert delta["tolerance_values"] == [0.0]
        assert delta["clip_count"] == 0
    assert result.metadata["policy_unchanged"] is True
    assert policy.fingerprint() == before


def test_closed_loop_full_episode_abort_keeps_attempt_record_without_env_step():
    adapter = _TrackingAdapter(dimension=3, action_count=3, horizon=3)
    config = _config()
    config.scenario["horizon"] = 3
    result = run_closed_loop(
        adapter,
        adapter,
        config,
        schema=_schema(),
        patterns=[
            {"id": "P00", "method": "identity", "indices": []},
            {
                "id": "full_reference",
                "method": "reference",
                "indices": [0],
                "reference_id": "donor",
                "scope": "full_episode",
                "metadata": {"compatibility_keys": ["road_segment_id"]},
            },
        ],
        episodes=1,
        max_steps=3,
        scenario_seed=11,
        rl_seed=23,
        reference_observations={"donor": np.ones(3, dtype=np.float32)},
        reference_contexts={"donor": {"road_segment_id": ["synthetic", 99]}},
    )

    episode = result.patterns["full_reference"][0]
    assert episode["execution_status"] == "aborted"
    assert episode["assessment_status"] == "not_evaluable_intervention_abort"
    assert episode["terminal_reason"] == "intervention_abort"
    assert episode["records"][0]["env_step_called"] is False
    assert episode["records"][0]["post_telemetry"] is None
    assert episode["intervention_counts"]["executed_env_step_count"] == 0
    assert episode["intervention_counts"]["aborted_before_env_step_count"] == 1
    assert adapter.created[-1].step_count == 0


def test_closed_loop_full_episode_intervention_exception_is_saved_as_pattern_abort(monkeypatch):
    import input_attribution.closed_loop as closed_loop_module

    original_apply_pattern = closed_loop_module.apply_pattern

    def raise_for_pattern(*args, **kwargs):
        pattern = args[1]
        if getattr(pattern, "pattern_id", None) == "boom":
            raise InterventionError("unexpected confirmed intervention condition")
        return original_apply_pattern(*args, **kwargs)

    monkeypatch.setattr(closed_loop_module, "apply_pattern", raise_for_pattern)
    adapter = _TrackingAdapter(dimension=3, action_count=3, horizon=3)
    config = _config()
    config.scenario["horizon"] = 3
    result = run_closed_loop(
        adapter,
        adapter,
        config,
        schema=_schema(),
        patterns=[
            {"id": "P00", "method": "identity", "indices": []},
            {
                "id": "boom",
                "method": "fixed",
                "indices": [0],
                "values": {0: 1.0},
                "scope": "full_episode",
            },
        ],
        episodes=1,
        max_steps=3,
        scenario_seed=11,
        rl_seed=23,
    )
    episode = result.patterns["boom"][0]
    assert episode["execution_status"] == "aborted"
    assert episode["assessment_status"] == "not_evaluable_intervention_abort"
    assert episode["terminal_reason"] == "intervention_abort"
    assert episode["failure_phase"] == "intervention"
    assert "unexpected confirmed intervention condition" in episode["failure_reason"]
    assert episode["records"] == []
    assert adapter.created[-1].step_count == 0


def test_closed_loop_policy_exception_is_saved_as_failed_episode_and_next_pattern_runs():
    class _FailingPolicyAdapter(_TrackingAdapter):
        probability_calls = 0

        def probabilities(self, observations):
            self.probability_calls += 1
            if self.probability_calls > 6:
                raise RuntimeError("synthetic policy evaluation failure")
            return super().probabilities(observations)

    adapter = _FailingPolicyAdapter(dimension=3, action_count=3, horizon=2)
    config = _config()
    config.scenario["horizon"] = 2
    result = run_closed_loop(
        adapter,
        adapter,
        config,
        schema=_schema(),
        patterns=[
            {"id": "P00", "method": "identity", "indices": []},
            {"id": "failed", "method": "fixed", "indices": [0], "values": {0: 1.0}},
            {"id": "after_failure", "method": "fixed", "indices": [1], "values": {1: 1.0}},
        ],
        episodes=1,
        max_steps=2,
        scenario_seed=11,
        rl_seed=23,
    )
    failed = result.patterns["failed"][0]
    assert failed["execution_status"] == "failed"
    assert failed["assessment_status"] == "not_evaluable_runtime_error"
    assert failed["failure_phase"] == "policy"
    assert "synthetic policy evaluation failure" in failed["failure_reason"]
    assert result.patterns["after_failure"][0]["execution_status"] == "failed"


def test_closed_loop_rejects_policy_fingerprint_mutation():
    class _MutatingPolicyAdapter(_TrackingAdapter):
        mutated = False

        def probabilities(self, observations):
            result = super().probabilities(observations)
            if not self.mutated:
                self.weights[0, 0] += np.float32(0.01)
                self.mutated = True
            return result

    adapter = _MutatingPolicyAdapter(dimension=3, action_count=3, horizon=1)
    config = _config()
    config.scenario["horizon"] = 1
    with pytest.raises(ClosedLoopError, match="policy parameters/buffers changed"):
        run_closed_loop(
            adapter,
            adapter,
            config,
            schema=_schema(),
            patterns=[{"id": "P00", "method": "identity", "indices": []}],
            episodes=1,
            max_steps=1,
            scenario_seed=11,
            rl_seed=23,
        )


def test_closed_loop_budget_censored_episode_is_excluded_from_verified_pair():
    adapter = _TrackingAdapter(dimension=3, action_count=3, horizon=3)
    config = _config()
    config.scenario["horizon"] = 3
    result = run_closed_loop(
        adapter,
        adapter,
        config,
        schema=_schema(),
        patterns=[
            {"id": "P00", "method": "identity", "indices": []},
            {"id": "local", "method": "fixed", "indices": [0], "values": {0: 1.0}},
        ],
        episodes=1,
        max_steps=1,
        scenario_seed=11,
        rl_seed=23,
    )
    episode = result.patterns["local"][0]
    assert episode["terminal_reason"] == "budget_censored"
    assert episode["paired_p00_status"] == "incomplete"
    assert episode["paired_p00_verified"] is False
    assert episode["metrics"]["paired_p00"] is None
    aggregate = result.as_dict()["patterns"]["local"]["paired_p00"]
    assert aggregate["verified_pair_count"] == 0
    assert aggregate["missing_reasons"]["pattern_terminal_budget_censored"] == 1


def test_closed_loop_records_each_latest_observation_after_road_switch():
    adapter = _RoadSwitchAdapter(dimension=3, action_count=3, horizon=4)
    config = _config()
    config.scenario["horizon"] = 4
    result = run_closed_loop(
        adapter,
        adapter,
        config,
        schema=_schema(),
        patterns=[
            {"id": "P00", "method": "identity", "indices": []},
            {"id": "local", "method": "fixed", "indices": [0], "values": {0: 1.0}},
        ],
        episodes=1,
        max_steps=4,
        scenario_seed=11,
        rl_seed=23,
    )
    records = result.patterns["local"][0]["records"]
    assert [row["road_segment_id"] for row in records] == [
        ["synthetic", 0],
        ["synthetic", 0],
        ["synthetic", 1],
        ["synthetic", 1],
    ]
    assert all(row["env_step_called"] is True for row in records)
    assert all(row["intervention"]["changed_count_exact"] == 1 for row in records)
    assert adapter.created[-1].forwarded_actions == [int(row["action_forwarded"]) for row in records]
    # The post-switch decision is based on this run's latest observation, not
    # the P00 trace or a cumulatively modified vector.
    assert records[2]["observation_hash"] != records[1]["observation_hash"]
    assert records[2]["original_values"][0] == pytest.approx(0.5)


def test_closed_loop_typed_reflection_forwards_modified_action_after_19_step_road_switch():
    """Typed B uses each fresh observation and records all 19/108 road steps."""

    adapter = _RoadSwitchAdapter(
        dimension=3,
        action_count=3,
        horizon=127,
        weights=np.asarray(
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        bias=np.zeros(3, dtype=np.float32),
    )
    adapter.switch_at = 19
    config = _config()
    config.scenario["horizon"] = 127
    before = adapter.fingerprint()
    result = run_closed_loop(
        adapter,
        adapter,
        config,
        schema=_provenance_schema(),
        patterns=[
            {"id": "P00", "method": "identity", "indices": []},
            {
                "id": "typed_reflect",
                "method": "reflection",
                "indices": [0],
                "variant_id": "reflect",
                "scope": "full_episode",
                "on_inapplicable": "abort_pattern",
            },
        ],
        episodes=1,
        max_steps=127,
        scenario_seed=11,
        rl_seed=23,
    )

    episode = result.patterns["typed_reflect"][0]
    records = episode["records"]
    assert len(records) == 127
    assert [row["road_segment_id"] for row in records[:19]] == [["synthetic", 0]] * 19
    assert [row["road_segment_id"] for row in records[19:]] == [["synthetic", 1]] * 108
    assert episode["intervention_counts"]["target_step_count"] == 127
    assert episode["intervention_counts"]["eligible_count"] == 127
    assert episode["intervention_counts"]["applied_count"] == 127
    assert episode["intervention_counts"]["changed_count_exact"] == 127
    assert episode["intervention_counts"]["skipped_count"] == 0
    assert episode["per_input_delta"]["0"]["delta_abs_count"] == 127
    assert episode["per_input_delta"]["0"]["delta_abs_max"] == pytest.approx(1.0)
    assert episode["assessment_status"] == "evaluated"
    assert all(row["env_step_called"] is True for row in records)
    assert all(row["intervention_scope_status"] == "applied" for row in records)
    assert adapter.created[-1].forwarded_actions == [int(row["action_forwarded"]) for row in records]
    assert any(row["action_forwarded"] != row["original_action"] for row in records)
    assert result.policy_unchanged is True
    # The post-switch decision was evaluated from the current B trajectory.
    assert records[19]["original_values"][0] == pytest.approx(19 / 127)
    assert records[20]["original_values"][0] == pytest.approx(20 / 127)
    assert adapter.fingerprint() == before


def test_partial_telemetry_exposes_missing_range_without_zero_filling():
    rows = [
        {
            "step": 0,
            "post_time": 0.1,
            "post_telemetry": {
                "simulation_time_s": 0.1,
                "target_lane_valid": True,
                "target_lane_offset_m": 0.5,
            },
        },
        {
            "step": 1,
            "post_time": 0.2,
            "post_telemetry": {"simulation_time_s": 0.2},
        },
    ]
    metrics = summarize_trajectory(rows)
    assert metrics["lane_rms_m"] == 0.5
    assert metrics["lane_error_rms_m"] != 0.0
    assert metrics["telemetry_status"] == "partial"
    lane_range = metrics["telemetry_coverage"]["lane_error"]
    assert lane_range["measured_count"] == 1
    assert lane_range["missing_count"] == 1
    assert lane_range["missing_ranges"][0]["first_step"] == 1
    assert metrics["measurement_range"]["span_s"] == 0.1


def test_trajectory_duration_and_progress_require_measured_units_and_intervals():
    rows = [
        {
            "step": 0,
            "post_time": None,
            "post_telemetry": {
                "target_lane_valid": True,
                "target_lane_offset_m": 2.0,
                "lane_width_m": 2.0,
                "speed_m_s": 0.1,
                "route_progress": 100.0,
            },
        },
        {
            "step": 1,
            "pre_time": 0.1,
            "post_time": 0.2,
            "post_telemetry": {
                "target_lane_valid": True,
                "target_lane_offset_m": 0.0,
                "lane_width_m": 2.0,
                "speed_m_s": 1.0,
                "route_progress": 101.0,
            },
        },
    ]
    metrics = summarize_trajectory(rows)
    assert metrics["progress_m"] is None
    assert metrics["departure_time_s"] is None
    assert metrics["low_speed_duration_s"] is None
    assert metrics["duration_measurement"]["status"] == "partial"
    assert metrics["duration_measurement"]["reason"] == "missing_pre_or_post_time"
    assert metrics["duration_measurement"]["missing_count"] == 1


def test_event_duration_keeps_partial_sum_separate_from_main_duration():
    rows = [
        {
            "step": 0,
            "post_time": None,
            "post_telemetry": {
                "target_lane_valid": True,
                "target_lane_offset_m": 2.0,
                "lane_width_m": 2.0,
                "speed_m_s": 0.1,
            },
        },
        {
            "step": 1,
            "pre_time": 0.1,
            "post_time": 0.2,
            "post_telemetry": {
                "target_lane_valid": True,
                "target_lane_offset_m": 0.0,
                "lane_width_m": 2.0,
                "speed_m_s": 1.0,
            },
        },
        {
            "step": 2,
            "pre_time": 0.2,
            "post_time": 0.3,
            "post_telemetry": {
                "target_lane_valid": True,
                "target_lane_offset_m": 2.0,
                "lane_width_m": 2.0,
                "speed_m_s": 0.1,
            },
        },
    ]
    metrics = summarize_trajectory(rows)

    assert metrics["duration_s"] is None
    assert metrics["duration_measurement"]["measured_duration_s"] == pytest.approx(0.2)
    assert metrics["departure_time_s"] is None
    departure = metrics["departure_duration_measurement"]
    assert departure["status"] == "partial"
    assert departure["expected_count"] == 2
    assert departure["measured_count"] == 1
    assert departure["measured_duration_s"] == pytest.approx(0.1)
    assert departure["missing_event_indices"] == [0]
    assert metrics["low_speed_duration_s"] is None
    low_speed = metrics["low_speed_duration_measurement"]
    assert low_speed["status"] == "partial"
    assert low_speed["expected_count"] == 2
    assert low_speed["measured_count"] == 1
    assert low_speed["measured_duration_s"] == pytest.approx(0.1)


def test_probability_delta_keeps_signed_and_absolute_changed_only_views():
    original = np.asarray([[0.6, 0.4], [0.4, 0.6], [0.5, 0.5]])
    changed = np.asarray([[0.7, 0.3], [0.5, 0.5], [0.5, 0.5]])
    comparison = compare_distributions(original, changed, selected_actions=[0, 1, 0])
    summary = summarize_distribution_comparison(
        comparison,
        changed_input_mask=[True, True, False],
        episodes=[0, 0, 0],
    )
    actual = summary["actual_input_changes_only"]
    assert actual["count"] == 2
    assert actual["selected_probability_delta"]["mean"] == 0.0
    assert actual["selected_probability_abs_delta"]["mean"] == pytest.approx(0.1)
    assert summary["episode_mean_abs"]["actual_input_changes_only"] == pytest.approx(0.1)


def test_intervention_counts_leave_missing_legacy_records_unknown_not_noop():
    counts = summarize_intervention_records(
        records=[
            {"step": 0},
            {"step": 1, "intervention": {}},
            {
                "step": 2,
                "intervention": {
                    "requested_indices": [0],
                    "changed_indices": [],
                    "no_op_indices": [0],
                    "skipped": False,
                    "eligible": True,
                    "applied": True,
                    "delta_values": {0: 0.0},
                    "delta_abs_values": {0: 0.0},
                    "tolerance": {0: 0.0},
                    "clipped_indices": [],
                },
            },
        ],
        target_step_count=3,
    )
    assert counts["status"] == "unavailable"
    assert counts["unknown_record_count"] == 2
    assert counts["applied_count"] == 1
    assert counts["noop_count"] == 1
    assert counts["changed_count_exact"] == 0
    assert counts["per_input_delta"]["0"] == {
        "applied_count": 1,
        "delta_abs_count": 1,
        "delta_abs_mean": 0.0,
        "delta_abs_max": 0.0,
        "tolerance_observed_count": 1,
        "tolerance_values": [0.0],
        "tolerance": 0.0,
        "clip_count": 0,
    }


def test_closed_loop_video_patterns_select_per_pattern_and_record_skip_reason(tmp_path):
    config = _config()
    config.scenario["horizon"] = 1
    config.video = {"enabled": True, "patterns": ["local"]}
    adapter = _TrackingAdapter(dimension=3, action_count=3, horizon=1)
    store = create_run(tmp_path, "runtime", "fake", run_id="video-select")
    store.save_manifest({"manifest_version": 1})
    result = run_closed_loop(
        adapter,
        adapter,
        config,
        schema=_schema(),
        patterns=[
            {"id": "P00", "method": "identity", "indices": []},
            {"id": "local", "method": "fixed", "indices": [0], "values": {0: 1.0}},
        ],
        episodes=1,
        max_steps=1,
        scenario_seed=11,
        rl_seed=23,
        store=store,
    )
    p00_video = result.patterns["P00"][0]["video"]
    local_video = result.patterns["local"][0]["video"]
    assert p00_video["enabled"] is False
    assert p00_video["reason"] == "pattern_not_selected_by_video_patterns"
    assert local_video["enabled"] is True
    assert local_video["status"] == "generated"
    assert local_video["frame_count"] == 1
    assert (
        store.closed_loop_dir / "local" / "episode-0" / "video.json"
    ).is_file()
    assert not (
        store.closed_loop_dir / "P00" / "episode-0" / "video.json"
    ).exists()

    disabled_config = _config()
    disabled_config.scenario["horizon"] = 1
    disabled_config.video = {"enabled": True, "patterns": False}
    disabled_adapter = _TrackingAdapter(dimension=3, action_count=3, horizon=1)
    disabled_store = create_run(tmp_path, "runtime", "fake", run_id="video-disabled")
    disabled_store.save_manifest({"manifest_version": 1})
    disabled = run_closed_loop(
        disabled_adapter,
        disabled_adapter,
        disabled_config,
        schema=_schema(),
        patterns=[
            {"id": "P00", "method": "identity", "indices": []},
            {"id": "local", "method": "fixed", "indices": [0], "values": {0: 1.0}},
        ],
        episodes=1,
        max_steps=1,
        scenario_seed=11,
        rl_seed=23,
        store=disabled_store,
    )
    for pattern_id in ("P00", "local"):
        info = disabled.patterns[pattern_id][0]["video"]
        assert info["enabled"] is False
        assert info["reason"] == "disabled_by_video_patterns"
        assert info["frame_count"] == 0


def test_closed_loop_summary_separates_natural_and_partial_episode_groups():
    def episode(index, *, execution_status, terminal_reason, lane_rms):
        return {
            "episode": index,
            "episode_id": f"episode-{index}",
            "record_count": 1,
            "terminal_reason": terminal_reason,
            "execution_status": execution_status,
            "assessment_status": "evaluated",
            "metrics": {
                "lane_rms_m": lane_rms,
                "poststep_count": 1,
                "valid_count": 1,
            },
            "intervention_counts": {
                "target_step_count": 1,
                "planned_target_step_count": 1,
                "attempted_step_count": 1,
                "executed_env_step_count": 1 if execution_status != "aborted" else 0,
                "actual_target_step_count": 1 if execution_status != "aborted" else 0,
                "aborted_before_env_step_count": 1 if execution_status == "aborted" else 0,
                "eligible_count": 1,
                "applied_count": 1,
                "changed_count_exact": 1,
                "noop_count": 0,
                "skipped_count": 0,
                "skip_reasons": {},
            },
            "records": [],
        }

    result = ClosedLoopResult(
        None,
        patterns={
            "P01": [
                episode(0, execution_status="completed", terminal_reason="terminated", lane_rms=1.0),
                episode(1, execution_status="aborted", terminal_reason="intervention_abort", lane_rms=2.0),
                episode(2, execution_status="completed", terminal_reason="budget_censored", lane_rms=3.0),
            ]
        },
    )
    summary = result.as_dict()["patterns"]["P01"]
    assert summary["metric_summary"]["counts"]["episodes"] == 3
    assert summary["execution_group_counts"] == {
        "natural_completion": 1,
        "partial_or_interrupted": 2,
    }
    natural = summary["execution_groups"]["natural_completion"]
    partial = summary["execution_groups"]["partial_or_interrupted"]
    assert natural["episode_indices"] == [0]
    assert natural["metric_summary"]["episode_mean"]["lane_rms_m"] == pytest.approx(1.0)
    assert partial["episode_indices"] == [1, 2]
    assert partial["metric_summary"]["episode_mean"]["lane_rms_m"] == pytest.approx(2.5)
    assert partial["reasons"]["execution_status"] == {"aborted": 1, "completed": 1}
    assert partial["reasons"]["terminal_reason"] == {
        "intervention_abort": 1,
        "budget_censored": 1,
    }


def test_execution_groups_require_completed_status_and_known_terminal_reason():
    def episode(index, *, status=None, terminal_reason):
        result = {
            "episode": index,
            "episode_id": f"episode-{index}",
            "record_count": 1,
            "terminal_reason": terminal_reason,
            "assessment_status": "evaluated",
            "metrics": {"lane_rms_m": float(index), "poststep_count": 1, "valid_count": 1},
            "intervention_counts": {"target_step_count": 1, "applied_count": 1},
            "records": [],
        }
        if status is not None:
            result["execution_status"] = status
        return result

    result = ClosedLoopResult(
        None,
        patterns={
            "P01": [
                episode(0, status="completed", terminal_reason="out_of_road"),
                episode(1, terminal_reason="horizon"),
                episode(2, status="completed", terminal_reason="truncated"),
            ]
        },
    )
    summary = result.as_dict()["patterns"]["P01"]

    assert summary["execution_group_counts"]["natural_completion"] == 1
    assert summary["execution_group_counts"]["partial_or_interrupted"] == 2
    assert summary["execution_group_counts"]["partial_unknown"] == 2
    assert summary["execution_groups"]["natural_completion"]["episode_indices"] == [0]
    assert summary["execution_groups"]["partial_unknown"]["episode_indices"] == [1, 2]
