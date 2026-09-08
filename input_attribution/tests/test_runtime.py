"""Bounded fake runtime tests for collection and closed-loop invariants."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from input_attribution.adapters.fake import FakePolicyAdapter, FakeVectorEnv
from input_attribution.artifacts import create_run
from input_attribution.collection import (
    CollectionError,
    collect_reference,
    finalize_video,
    policy_probabilities,
    save_video_frame,
    telemetry_time,
)
from input_attribution.closed_loop import ClosedLoopError, ClosedLoopResult, run_closed_loop
from input_attribution.schema import InputSchema, InputSpec
from input_attribution.trajectory_metrics import summarize_trajectory


def _schema() -> InputSchema:
    return InputSchema(
        dimension=3,
        inputs=(
            InputSpec(0, "x", "x", group="x"),
            InputSpec(1, "y", "y", group="y"),
            InputSpec(2, "z", "z", group="z"),
        ),
    )


class _TrackingEnv(FakeVectorEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forwarded_actions: list[int] = []

    def step(self, action):
        self.forwarded_actions.append(int(np.asarray(action).reshape(-1)[0]))
        return super().step(action)


class _TrackingAdapter(FakePolicyAdapter):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.created: list[_TrackingEnv] = []
        self.reset_seeds: list[int | None] = []
        self.closed: list[_TrackingEnv] = []
        self.omit_lane = False
        self.terminal_flags: dict[str, bool] = {}
        self.force_actual_seed: int | None = None

    def make_environment(self, config=None, *, seed=None):
        horizon = int((getattr(config, "scenario", {}) or {}).get("horizon", self.horizon))
        env = _TrackingEnv(self.dimension, self.action_count, horizon=horizon)
        self.env = env
        self.created.append(env)
        return env

    make_env = make_environment

    def reset_env(self, env=None, *, seed=None):
        if env is not None:
            self.env = env
        self.reset_seeds.append(seed)
        return self.env.reset(seed=seed)

    def close(self, env=None):
        target = env if env is not None else self.env
        if target is not None:
            self.closed.append(target)
            target.close()
        if target is self.env:
            self.env = None

    close_env = close

    def telemetry(self, env, info=None, *, phase="post", step=0):
        result = dict(info or {})
        result.update(
            {
                "simulation_time_s": step * 0.1,
                "speed_m_s": 1.0,
                "actual_scenario_seed": (
                    self.force_actual_seed if self.force_actual_seed is not None else env.current_seed
                ),
                "position_xy": [float(step), 0.0],
                "road_segment_id": ["synthetic", 0],
            }
        )
        if not self.omit_lane:
            result.update(
                {
                    "target_lane_ordinal": 0,
                    "target_lane_valid": True,
                    "target_lane_offset_m": 0.6 * step,
                    "lane_width_m": 2.0,
                    "in_target_lane": step < 2,
                }
            )
        result.update(self.terminal_flags if phase == "post" else {})
        return result

    def render_frame(self, env, **kwargs):
        del env, kwargs
        return np.zeros((4, 4, 3), dtype=np.uint8)


def _config():
    return SimpleNamespace(
        scenario={"episodes": 1, "horizon": 3},
        scenario_seed=11,
        rl_seed=23,
        closed_loop={
            "episodes": 1,
            "departure_tolerance_ratio": 0.05,
            "departure_consecutive_steps": 1,
            "low_speed_m_s": 0.5,
        },
        video={"enabled": False},
    )


def test_collection_seeds_each_episode_closes_and_does_not_pad_after_budget():
    adapter = _TrackingAdapter(dimension=3, action_count=3, horizon=3)
    result = collect_reference(
        adapter,
        adapter,
        _config(),
        schema=_schema(),
        episodes=1,
        max_steps=2,
        scenario_seed=11,
        rl_seed=23,
    )
    assert len(result.records) == 2
    assert result.records[-1]["budget_truncated"] is True
    assert adapter.reset_seeds == [11]
    assert len(adapter.closed) == 1 and adapter.closed[0].closed
    assert result.records[0]["post_time"] == 0.1
    assert result.records[0]["post_telemetry"]["simulation_time_s"] == 0.1


def test_collection_rejects_unnormalized_and_multi_batch_policy_outputs():
    class _BadPolicy:
        def probabilities(self, observations):
            del observations
            return np.asarray([[0.2, 0.2], [0.8, 0.8]], dtype=np.float64)

    with pytest.raises(CollectionError, match="1次元|合計"):
        policy_probabilities(_BadPolicy(), np.zeros(3, dtype=np.float32))


def test_closed_loop_uses_latest_own_observation_and_forwards_modified_action():
    adapter = _TrackingAdapter(dimension=3, action_count=3, horizon=3)
    config = _config()
    patterns = [
        {"id": "P00", "method": "identity", "indices": []},
        {"id": "P01", "method": "fixed", "indices": [0], "values": {0: 1.0}},
    ]
    result = run_closed_loop(
        adapter,
        adapter,
        config,
        schema=_schema(),
        patterns=patterns,
        episodes=1,
        max_steps=2,
        scenario_seed=11,
        rl_seed=23,
    )
    assert set(result.patterns) == {"P00", "P01"}
    altered = result.patterns["P01"][0]
    assert altered["terminal_reason"] == "budget_censored"
    assert len(altered["records"]) == 2
    assert altered["records"][1]["intervention"]["changed_indices"] == [0]
    # The second decision sees this run's own post-step observation (0.333...)
    # rather than the P00 trajectory or a cumulative modified vector.
    assert altered["records"][1]["observation_hash"] != altered["records"][0]["observation_hash"]
    assert all("action_forwarded" in row for row in altered["records"])
    assert adapter.created[-1].forwarded_actions == [
        int(row["action_forwarded"]) for row in altered["records"]
    ]
    assert altered["metrics"]["valid_count"] == 2
    assert altered["metrics"]["departure_count"] == 1
    assert altered["metrics"]["lane_rms_m"] is not None
    assert altered["metrics"]["duration_s"] == 0.2
    assert result.baseline_references[0]["records"]


def test_closed_loop_rejects_a_reset_that_did_not_apply_requested_seed():
    adapter = _TrackingAdapter(dimension=3, action_count=3, horizon=1)
    adapter.force_actual_seed = 999
    config = _config()
    config.scenario["horizon"] = 1
    with pytest.raises(ClosedLoopError, match="requested scenario seed"):
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


def test_closed_loop_validates_saved_p00_records_and_observations():
    adapter = _TrackingAdapter(dimension=3, action_count=3, horizon=2)
    config = _config()
    collected = collect_reference(
        adapter,
        adapter,
        config,
        schema=_schema(),
        episodes=1,
        max_steps=2,
        scenario_seed=11,
        rl_seed=23,
    )
    references = {
        f"{row['episode_id']}:{row['step']}": collected.observations[index].copy()
        for index, row in enumerate(collected.records)
    }
    result = run_closed_loop(
        adapter,
        adapter,
        config,
        schema=_schema(),
        patterns=[{"id": "P00", "method": "identity", "indices": []}],
        episodes=1,
        max_steps=2,
        scenario_seed=11,
        rl_seed=23,
        reference_records=collected.records,
        reference_observations=references,
    )
    validation = result.patterns["P00"][0]["reference_trace_validation"]
    assert validation["status"] == "matched"
    assert "observation" in validation["checked_fields"]
    assert validation["checked_records"] == 2


def test_closed_loop_reference_context_mismatch_skips_but_matching_context_applies():
    pattern = {
        "id": "P01_reference",
        "method": "reference",
        "indices": [1],
        "reference_id": "r1",
        "metadata": {"compatibility_keys": ["road_segment_id"]},
    }
    reference_observations = {"r1": np.asarray([0.0, 1.0, 0.0], dtype=np.float32)}
    config = _config()
    config.scenario["horizon"] = 1

    incompatible = run_closed_loop(
        _TrackingAdapter(dimension=3, action_count=3, horizon=1),
        _TrackingAdapter(dimension=3, action_count=3, horizon=1),
        config,
        schema=_schema(),
        patterns=[{"id": "P00", "method": "identity", "indices": []}, pattern],
        episodes=1,
        max_steps=1,
        scenario_seed=11,
        rl_seed=23,
        reference_observations=reference_observations,
        reference_contexts={"r1": {"road_segment_id": ["synthetic", 60]}},
    )
    skipped_episode = incompatible.patterns["P01_reference"][0]
    assert skipped_episode["records"][0]["intervention"]["skipped"] is True
    assert skipped_episode["records"][0]["intervention"]["changed_indices"] == []
    assert skipped_episode["intervention_skipped"] is True
    assert "compatibility context 'road_segment_id' differs" in skipped_episode["intervention_skip_reasons"][0]
    assert (
        skipped_episode["records"][0]["action"]
        == incompatible.patterns["P00"][0]["records"][0]["action"]
    )

    compatible = run_closed_loop(
        _TrackingAdapter(dimension=3, action_count=3, horizon=1),
        _TrackingAdapter(dimension=3, action_count=3, horizon=1),
        config,
        schema=_schema(),
        patterns=[{"id": "P00", "method": "identity", "indices": []}, pattern],
        episodes=1,
        max_steps=1,
        scenario_seed=11,
        rl_seed=23,
        reference_observations=reference_observations,
        reference_contexts={"r1": {"road_segment_id": ["synthetic", 0]}},
    )
    applied_episode = compatible.patterns["P01_reference"][0]
    assert applied_episode["records"][0]["intervention"]["skipped"] is False
    assert applied_episode["records"][0]["intervention"]["changed_indices"] == [1]
    assert (
        applied_episode["records"][0]["action"]
        != compatible.patterns["P00"][0]["records"][0]["action"]
    )


def test_missing_target_lane_is_reported_as_unavailable_not_zero():
    adapter = _TrackingAdapter(dimension=3, action_count=3, horizon=1)
    adapter.omit_lane = True
    result = run_closed_loop(
        adapter,
        adapter,
        _config(),
        schema=_schema(),
        patterns=[{"id": "P00", "method": "identity", "indices": []}],
        episodes=1,
        max_steps=1,
        scenario_seed=11,
        rl_seed=23,
    )
    metrics = result.patterns["P00"][0]["metrics"]
    assert metrics["status"] == "not_available"
    assert metrics["lane_rms_m"] is None
    assert metrics["departure_count"] is None
    assert metrics["valid_rate"] is None


def test_missing_initial_geometry_is_unverified_and_pair_delta_is_na():
    adapter = _TrackingAdapter(dimension=3, action_count=3, horizon=1)
    adapter.omit_lane = True
    result = run_closed_loop(
        adapter,
        adapter,
        _config(),
        schema=_schema(),
        patterns=[
            {"id": "P00", "method": "identity", "indices": []},
            {"id": "P01", "method": "fixed", "indices": [0], "values": {0: 1.0}},
        ],
        episodes=1,
        max_steps=1,
        scenario_seed=11,
        rl_seed=23,
    )
    assert result.patterns["P00"][0]["initial_snapshot"]["initial_match"]["status"] == "unverified"
    p01 = result.patterns["P01"][0]
    assert p01["initial_snapshot"]["initial_match"]["status"] == "unverified"
    assert p01["metrics"]["paired_p00"] is None
    assert p01["paired_p00_status"] == "unverified"


def test_video_failure_is_separate_and_episode_frame_paths_do_not_collide(tmp_path):
    adapter = _TrackingAdapter(dimension=3, action_count=3, horizon=1)
    config = _config()
    store = create_run(tmp_path, "runtime", "fake", run_id="video")
    store.save_manifest({"manifest_version": 1})
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
        store=store,
        video=True,
    )
    episode_dir = store.closed_loop_dir / "P00" / "episode-0"
    video = store.read_json(episode_dir.relative_to(store.run_dir) / "video.json")
    assert video["saved_count"] == 1
    assert video["failure_count"] == 0
    assert video["frame_map"][0]["episode"] == 0
    assert (episode_dir / "video" / "frame-000000.png").is_file()
    assert (episode_dir / "video" / "trajectory.gif").is_file()


def test_saved_video_frame_contains_ascii_runtime_overlay_without_mutating_capture(tmp_path):
    frame = np.full((64, 240, 3), 127, dtype=np.uint8)
    original = frame.copy()
    saved = save_video_frame(
        frame,
        tmp_path / "P01" / "episode-2" / "video",
        step=7,
        simulation_time=0.25,
        pattern_id="P01",
        phase="post",
    )
    assert saved["status"] == "saved"
    assert "pattern=P01" in saved["overlay"]
    assert "path=episode-2" in saved["overlay"]
    assert "step=000007" in saved["overlay"]
    assert "post t=0.250s" in saved["overlay"]
    assert np.array_equal(frame, original)

    from PIL import Image

    with Image.open(saved["path"]) as image:
        rendered = np.asarray(image.convert("RGB"))
    assert rendered.shape == frame.shape
    assert not np.array_equal(rendered, original)
    saved_next = save_video_frame(
        frame,
        tmp_path / "P01" / "episode-2" / "video",
        step=8,
        simulation_time=0.30,
        pattern_id="P01",
        phase="post",
    )
    video = finalize_video(
        [saved, saved_next],
        tmp_path / "P01" / "episode-2" / "video",
        fps=20.0,
    )
    assert video["fps"] == pytest.approx(20.0)
    assert video["frame_duration_ms"] == 50
    assert video["gif"] is not None
    with Image.open(video["gif"]) as gif:
        assert gif.n_frames == 2


def test_trajectory_metrics_deduplicates_continuous_departure_and_preserves_na():
    rows = [
        {
            "pre_time": index * 0.1,
            "post_time": (index + 1) * 0.1,
            "post_telemetry": {
                "target_lane_valid": True,
                "target_lane_offset_m": 1.2,
                "lane_width_m": 2.0,
                "speed_m_s": 1.0,
            },
        }
        for index in range(3)
    ]
    metrics = summarize_trajectory(rows, departure_consecutive_steps=2)
    assert metrics["departure_count"] == 1
    assert metrics["departure_time_s"] == pytest.approx(0.3)

    known_in_lane = [
        {"pre_time": 0.0, "post_time": 0.1, "post_telemetry": {"target_lane_valid": True, "target_lane_offset_m": 0.0, "lane_width_m": 2.0}},
        {"pre_time": 0.1, "post_time": 0.2, "post_telemetry": {"target_lane_valid": True, "target_lane_offset_m": 0.0, "lane_width_m": 2.0}},
    ]
    no_departure = summarize_trajectory(known_in_lane)
    assert no_departure["departure_count"] == 0
    assert no_departure["departure_time_s"] == 0.0

    invalid = summarize_trajectory(
        [{"pre_time": 0.0, "post_time": 0.1, "post_telemetry": {"target_lane_valid": False}}]
    )
    assert invalid["valid_count"] == 0
    assert invalid["valid_rate"] == 0.0
    assert invalid["valid_time_s"] == 0.0

    unavailable = summarize_trajectory([{"post_telemetry": {}}])
    assert unavailable["valid_rate"] is None


def test_time_metrics_never_fabricate_clock_from_sample_index():
    rows = [
        {
            "action": 0,
            "post_telemetry": {
                "target_lane_valid": True,
                "target_lane_offset_m": 1.2,
                "lane_width_m": 2.0,
                "speed_m_s": 0.1,
            },
        },
        {
            "action": 1,
            "post_telemetry": {
                "target_lane_valid": True,
                "target_lane_offset_m": 0.0,
                "lane_width_m": 2.0,
                "speed_m_s": 0.1,
            },
        },
    ]
    metrics = summarize_trajectory(rows)
    assert telemetry_time({}, 123) is None
    assert metrics["duration_s"] is None
    assert metrics["valid_time_s"] is None
    assert metrics["valid_rate"] is None
    assert metrics["low_speed_duration_s"] is None
    assert metrics["first_departure_time_s"] is None
    assert metrics["sample_valid_rate"] == 1.0
    assert metrics["poststep_count"] == 2


def test_current_lane_offset_cannot_be_used_as_target_lane_error():
    metrics = summarize_trajectory(
        [{
            "pre_time": 0.0,
            "post_time": 0.1,
            "post_telemetry": {
                "target_lane_valid": True,
                "lane_center_offset_m": 1.5,
                "lane_width_m": 2.0,
            },
        }]
    )
    assert metrics["lane_rms_m"] is None
    assert metrics["valid_count"] == 0
    assert metrics["valid_rate"] is None


def test_closed_loop_preserves_raw_flags_and_reports_explicit_arrival_reason():
    adapter = _TrackingAdapter(dimension=3, action_count=3, horizon=1)
    adapter.terminal_flags = {"arrive_dest": True}
    config = _config()
    config.scenario["horizon"] = 1
    result = run_closed_loop(
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
    episode = result.patterns["P00"][0]
    row = episode["records"][-1]
    assert row["terminated"] is True
    assert row["truncated"] is False
    assert row["terminal_reason"] == "arrive_dest"
    assert episode["terminal_reason"] == "arrive_dest"


def test_wrong_lane_arrival_overrides_destination_success_and_preserves_flags():
    adapter = _TrackingAdapter(dimension=3, action_count=3, horizon=1)
    adapter.terminal_flags = {"arrive_dest": True, "wrong_lane_arrival": True}
    config = _config()
    config.scenario["horizon"] = 1
    result = run_closed_loop(
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
    episode = result.patterns["P00"][0]
    assert episode["terminal_reason"] == "wrong_lane_arrival"
    assert episode["metrics"]["arrived"] is False
    assert episode["metrics"]["wrong_lane_arrival"] is True
    later_success = summarize_trajectory(
        [
            {"post_telemetry": {"wrong_lane_arrival": True, "arrive_dest": True}},
            {"post_telemetry": {"arrive_dest": True}},
        ]
    )
    assert later_success["arrived"] is False


def test_closed_loop_summary_exposes_episode_and_step_weighted_metrics():
    result = ClosedLoopResult(
        store=None,
        patterns={
            "P01": [
                {"record_count": 1, "metrics": {"lane_rms_m": 1.0, "poststep_count": 1, "valid_count": 1}},
                {"record_count": 3, "metrics": {"lane_rms_m": 3.0, "poststep_count": 3, "valid_count": 3}},
            ]
        },
    )
    summary = result.as_dict()["patterns"]["P01"]
    assert summary["episode_mean"]["lane_rms_m"] == pytest.approx(2.0)
    assert summary["step_weighted"]["lane_rms_m"] == pytest.approx(np.sqrt(7.0))
    assert summary["counts"]["episodes"] == 2
    assert summary["counts"]["poststeps"] == 4
