"""Regression tests for closed-loop failures after the environment step."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from input_attribution.adapters.fake import FakePolicyAdapter, FakeVectorEnv
from input_attribution.artifacts import create_run
from input_attribution.closed_loop import ClosedLoopResult, run_closed_loop
from input_attribution.schema import InputSchema, InputSpec


def _schema() -> InputSchema:
    return InputSchema(
        dimension=3,
        inputs=(
            InputSpec(0, "x", "x", group="x"),
            InputSpec(1, "y", "y", group="y"),
            InputSpec(2, "z", "z", group="z"),
        ),
    )


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        scenario={"horizon": 3},
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


class _RaisesOnStepEnv(FakeVectorEnv):
    def step(self, action):
        # Increment first to model the uncertainty that a simulator may have
        # advanced before raising from its step implementation.
        self.step_count += 1
        raise RuntimeError("synthetic env.step failure")


class _FailureAdapter(FakePolicyAdapter):
    """Fake adapter that fails only for the second (non-P00) pattern."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.created: list[FakeVectorEnv] = []
        self.closed: list[FakeVectorEnv] = []
        self.active_pattern_index = -1
        self.failure_mode: str | None = None

    def make_environment(self, config=None, *, seed=None):
        del seed
        horizon = int((getattr(config, "scenario", {}) or {}).get("horizon", self.horizon))
        index = len(self.created)
        if self.failure_mode == "env_step":
            env = _RaisesOnStepEnv(self.dimension, self.action_count, horizon=horizon)
        else:
            env = FakeVectorEnv(self.dimension, self.action_count, horizon=horizon)
        env.pattern_index = index
        self.created.append(env)
        self.env = env
        self.active_pattern_index = index
        return env

    make_env = make_environment

    def reset_env(self, env=None, *, seed=None):
        if env is not None:
            self.env = env
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
        if (
            self.failure_mode == "pre_telemetry"
            and getattr(env, "pattern_index", -1) == 1
            and phase == "pre"
            and step == 0
        ):
            raise RuntimeError("synthetic pre telemetry failure")
        if (
            self.failure_mode == "post_telemetry"
            and getattr(env, "pattern_index", -1) == 1
            and phase == "post"
            and step == 2
        ):
            raise RuntimeError("synthetic post telemetry failure")
        result = dict(info or {})
        result.update(
            {
                "simulation_time_s": step * 0.1,
                "speed_m_s": 1.0,
                "actual_scenario_seed": env.current_seed,
                "position_xy": [float(step), 0.0],
                "road_segment_id": ["synthetic", 0],
                "target_lane_ordinal": 0,
                "target_lane_valid": True,
                "target_lane_offset_m": 0.0,
                "lane_width_m": 2.0,
                "in_target_lane": True,
            }
        )
        return result

    def probabilities(self, observations):
        if self.failure_mode == "policy" and self.active_pattern_index == 1:
            raise RuntimeError("synthetic policy failure")
        return super().probabilities(observations)

    def decode_action(self, action, env_config=None):
        del env_config
        if (
            self.failure_mode == "action_decode"
            and self.active_pattern_index == 1
            and self.env is not None
            and self.env.step_count == 2
        ):
            raise RuntimeError("synthetic action decode failure")
        return super().decode_action(action)


def _run_failure(mode: str, *, store=None):
    adapter = _FailureAdapter(dimension=3, action_count=3, horizon=3)
    adapter.failure_mode = mode
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
        max_steps=3,
        scenario_seed=11,
        rl_seed=23,
        store=store,
    )
    return adapter, result


def test_post_telemetry_failure_keeps_step_return_and_does_not_advance_again():
    adapter, result = _run_failure("post_telemetry")

    baseline = result.patterns["P00"][0]
    failed = result.patterns["P01"][0]
    assert baseline["execution_status"] == "completed"
    assert failed["execution_status"] == "failed"
    assert failed["failure_phase"] == "post_telemetry"
    assert len(failed["records"]) == 2
    row = failed["records"][1]
    assert row["phase"] == "post_telemetry"
    assert row["failure_phase"] == "post_telemetry"
    assert row["post_telemetry"] is None
    assert row["reward"] is not None
    assert row["terminated"] is False
    assert row["truncated"] is False
    assert row["info"]["simulation_time"] == 2 * 0.1
    assert row["env_step_returned"] is True
    assert row["observation_index"] == 1
    assert len(adapter.created[1]._observation) == len(failed["records"][1]["modified_values"]) + 2
    assert adapter.created[1].step_count == 2
    counts = result.as_dict()["counts"]
    assert counts["runtime_failure_count"] == 1
    assert counts["completed_pattern_count"] == 1
    assert counts["failed_pattern_count"] == 1
    assert counts["aborted_pattern_count"] == 0


def test_action_decode_failure_keeps_post_telemetry_and_step_alignment(tmp_path: Path):
    store = create_run(tmp_path, "runtime", "fake", run_id="decode")
    store.save_manifest({"manifest_version": 1})
    adapter, result = _run_failure("action_decode", store=store)

    failed = result.patterns["P01"][0]
    assert failed["execution_status"] == "failed"
    assert failed["failure_phase"] == "action_decode"
    assert len(failed["records"]) == 2
    row = failed["records"][1]
    assert row["phase"] == "action_decode"
    assert row["failure_phase"] == "action_decode"
    assert row["post_telemetry"]["simulation_time_s"] == 0.2
    assert row["reward"] is not None
    assert row["env_step_returned"] is True
    assert row["decoded_action"] is None
    assert row["observation_index"] == 1
    assert adapter.created[1].step_count == 2
    assert result.patterns["P00"][0]["execution_status"] == "completed"
    saved = store.read_json(Path("02_closed_loop/P01/episode-0/trajectory.json"))
    saved_observations = np.load(
        store.closed_loop_dir / "P01" / "episode-0" / "observations.npy",
        allow_pickle=False,
    )
    saved_modified = np.load(
        store.closed_loop_dir / "P01" / "episode-0" / "modified_observations.npy",
        allow_pickle=False,
    )
    assert len(saved["records"]) == 2
    assert saved_observations.shape == (2, 3)
    assert saved_modified.shape == (2, 3)
    assert saved["records"][1]["observation_index"] == 1
    status = store.read_json(Path("status.json"))
    assert status["stages"]["closed_loop"]["state"] == "failed"
    assert status["stages"]["closed_loop"]["counts"]["runtime_failure_count"] == 1


def test_env_step_failure_is_unknown_and_not_counted_as_returned_step():
    adapter, result = _run_failure("env_step")

    failed = result.patterns["P01"][0]
    assert failed["execution_status"] == "failed"
    assert failed["failure_phase"] == "env_step"
    assert len(failed["records"]) == 1
    row = failed["records"][0]
    assert row["phase"] == "env_step"
    assert row["env_step_attempted"] is True
    assert row["env_step_returned"] is False
    assert row["env_step_called"] is True
    assert row["physical_state_unknown"] is True
    assert row["reward"] is None
    assert row["terminated"] is None
    assert row["truncated"] is None
    assert row["termination_flags_status"] == "not_observed"
    assert row["info_status"] == "not_observed"
    assert row["post_telemetry"] is None
    assert failed["intervention_counts"]["executed_env_step_count"] == 0
    assert adapter.created[1].step_count == 1


def test_policy_failure_keeps_pre_and_modified_inputs_in_aligned_slots():
    adapter, result = _run_failure("policy")

    failed = result.patterns["P01"][0]
    assert failed["execution_status"] == "failed"
    assert failed["failure_phase"] == "policy"
    assert len(failed["records"]) == 1
    row = failed["records"][0]
    assert row["phase"] == "policy"
    assert row["partial_record"] is True
    assert row["observation_missing"] is False
    assert row["modified_observation_missing"] is False
    assert row["observation_index"] == 0
    assert row["env_step_attempted"] is False
    assert row["post_telemetry"] is None
    assert len(result.patterns["P01"][0]["records"]) == len(adapter.created[1]._observation[:1])
    assert adapter.created[1].step_count == 0


def test_pre_telemetry_failure_is_saved_with_explicit_missing_observation():
    adapter, result = _run_failure("pre_telemetry")

    failed = result.patterns["P01"][0]
    assert failed["execution_status"] == "failed"
    assert failed["failure_phase"] == "pre_telemetry"
    assert len(failed["records"]) == 1
    row = failed["records"][0]
    assert row["phase"] == "pre_telemetry"
    assert row["partial_record"] is True
    assert row["observation_missing"] is True
    assert row["modified_observation_missing"] is True
    assert row["observation_index"] == 0
    assert row["env_step_attempted"] is False
    assert adapter.created[1].step_count == 0


def test_primary_metric_summary_excludes_failed_episode_but_keeps_partial_detail():
    result = ClosedLoopResult(
        store=None,
        patterns={
            "P01": [
                {
                    "episode": 0,
                    "record_count": 1,
                    "execution_status": "completed",
                    "terminal_reason": "arrive_dest",
                    "metrics": {
                        "lane_rms_m": 1.0,
                        "poststep_count": 1,
                        "valid_count": 1,
                    },
                    "records": [],
                },
                {
                    "episode": 1,
                    "record_count": 1,
                    "execution_status": "failed",
                    "terminal_reason": "runtime_error",
                    "metrics": {
                        "lane_rms_m": 99.0,
                        "poststep_count": 1,
                        "valid_count": 1,
                    },
                    "records": [],
                },
            ]
        },
    )

    summary = result.as_dict()["patterns"]["P01"]

    assert summary["metric_summary"]["episode_mean"]["lane_rms_m"] == pytest.approx(1.0)
    assert summary["episode_mean"]["lane_rms_m"] == pytest.approx(1.0)
    assert summary["step_weighted"]["lane_rms_m"] == pytest.approx(1.0)
    assert summary["metric_summary"]["counts"]["episodes"] == 1
    assert summary["diagnostic_metric_summary"]["episode_mean"]["lane_rms_m"] == pytest.approx(50.0)
    partial = summary["execution_groups"]["partial_or_interrupted"]
    assert partial["episode_indices"] == [1]
    assert partial["metric_summary"]["episode_mean"]["lane_rms_m"] == pytest.approx(99.0)


def test_primary_metric_summary_excludes_budget_censored_episode():
    result = ClosedLoopResult(
        store=None,
        patterns={
            "P01": [
                {
                    "episode": 0,
                    "record_count": 1,
                    "execution_status": "completed",
                    "terminal_reason": "horizon",
                    "metrics": {
                        "lane_rms_m": 1.0,
                        "poststep_count": 1,
                        "valid_count": 1,
                    },
                    "records": [],
                },
                {
                    "episode": 1,
                    "record_count": 1,
                    "execution_status": "completed",
                    "terminal_reason": "budget_censored",
                    "metrics": {
                        "lane_rms_m": 77.0,
                        "poststep_count": 1,
                        "valid_count": 1,
                    },
                    "records": [{"budget_truncated": True}],
                },
            ]
        },
    )

    summary = result.as_dict()["patterns"]["P01"]

    assert summary["metric_summary"]["episode_mean"]["lane_rms_m"] == pytest.approx(1.0)
    assert summary["metric_summary"]["counts"]["episodes"] == 1
    assert summary["diagnostic_metric_summary"]["episode_mean"]["lane_rms_m"] == pytest.approx(39.0)
    partial = summary["execution_groups"]["partial_or_interrupted"]
    assert partial["episode_indices"] == [1]
    assert partial["metric_summary"]["episode_mean"]["lane_rms_m"] == pytest.approx(77.0)
