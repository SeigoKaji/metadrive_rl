"""Dependency-light acceptance tests for input_attribution's core paths."""

from __future__ import annotations

from pathlib import Path
import copy
import hashlib
import tempfile

import numpy as np
import pytest

from input_attribution.collection import collect_baseline, collect_closed_loop
from input_attribution.interventions import Intervention, InterventionError, apply_intervention
from input_attribution.policy_comparison import PolicyComparisonError, compare_saved_baseline, js_divergence
from input_attribution.reward_adapter import (
    RewardTermsResult,
    reward_terms_result,
    validate_reward_terms,
)
from input_attribution.runner import run_experiment
from input_attribution.storage import read_manifest, read_rollout, write_rollout
from input_attribution.synthetic import SyntheticAdapter, SyntheticEnv


def _pattern(*indices: int, identifier: str = "P01") -> Intervention:
    return Intervention(identifier, "synthetic test", tuple(indices), -1.0)


def test_intervention_is_copy_and_replaces_every_step_without_touching_other_values() -> None:
    adapter = SyntheticAdapter(dimension=8, action_count=3, steps=127)
    with tempfile.TemporaryDirectory() as directory:
        baseline = collect_baseline(
            adapter,
            scenario_seed=5,
            policy_seed=5,
            horizon=127,
            run_dir=directory,
            record_gif=False,
        )
        write_rollout(directory, baseline)
        baseline_path = Path(directory) / "data" / "P00_baseline.json"
        baseline_hash_before = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
        baseline_snapshot = copy.deepcopy(baseline)
        changed = collect_closed_loop(
            adapter,
            _pattern(0, 1),
            scenario_seed=5,
            policy_seed=5,
            horizon=127,
            run_dir=directory,
            record_gif=False,
        )
        assert hashlib.sha256(baseline_path.read_bytes()).hexdigest() == baseline_hash_before
    assert baseline["status"] == "complete"
    assert changed["status"] == "complete"
    assert changed["applied_count"] == 127
    assert changed["changed_count"] == 127
    assert changed["unchanged_count"] == 0
    assert baseline == baseline_snapshot
    assert all(record["modified_input"][0:2] == [-1.0, -1.0] for record in changed["records"])
    assert changed["records"][0]["info"]["road_section_id"] == 0
    assert changed["records"][-1]["info"]["road_section_id"] == 1
    assert all(
        record["input"][2:] == record["modified_input"][2:]
        for record in changed["records"]
    )
    assert all(
        record["input"] == record["modified_input"]
        for record in baseline["records"]
    )
    assert all(record["next_observation"] is not None for record in baseline["records"])
    assert all(record["next_observation_hash"] for record in baseline["records"])


def test_intervention_rejects_duplicate_out_of_range_nonfinite_and_complex_values() -> None:
    with pytest.raises(InterventionError):
        apply_intervention(np.zeros(4), {"id": "P", "name": "p", "indices": [1, 1]})
    with pytest.raises(InterventionError):
        apply_intervention(np.zeros(4), {"id": "P", "name": "p", "indices": [4]})
    with pytest.raises(InterventionError):
        apply_intervention(np.zeros(4), {"id": "P", "name": "p", "indices": [1], "fixed_value": np.inf})
    with pytest.raises(InterventionError):
        apply_intervention(np.ones(4, dtype=np.complex128), {"id": "P", "name": "p", "indices": [1]})


@pytest.mark.parametrize(
    "identifier",
    [
        "P00",
        "p0",
        "P00_baseline",
        "baseline",
        "noop",
        "none",
        "identity",
        "data",
        "P01_offline",
        "../../outside",
        ".",
        "..",
        " P01",
        "P01 ",
        "P/01",
        "P 01",
    ],
)
def test_intervention_rejects_reserved_or_unsafe_artifact_ids(identifier: str) -> None:
    with pytest.raises(InterventionError):
        Intervention(identifier, "unsafe id", (0,))


def test_js_divergence_has_required_boundaries_and_zero_terms_are_finite() -> None:
    assert js_divergence([0.5, 0.5], [0.5, 0.5]) == 0.0
    assert np.isclose(js_divergence([1.0, 0.0], [0.0, 1.0]), np.log(2.0))
    assert np.isclose(js_divergence([0.0, 1.0, 0.0], [0.0, 1.0, 0.0]), 0.0)
    with pytest.raises(PolicyComparisonError):
        js_divergence([0.6, 0.6], [0.5, 0.5])


def test_offline_a_does_not_construct_or_step_an_environment_and_batch_matches_sequential() -> None:
    adapter = SyntheticAdapter(dimension=8, action_count=3, steps=7)
    with tempfile.TemporaryDirectory() as directory:
        baseline = collect_baseline(
            adapter,
            scenario_seed=5,
            policy_seed=5,
            horizon=7,
            run_dir=directory,
            record_gif=False,
        )
        baseline_snapshot = copy.deepcopy(baseline)
        made_before = adapter.make_env_calls
        reset_env = adapter.last_env
        assert reset_env is not None
        assert reset_env.reset_calls == 1
        assert reset_env.step_calls == 7
        assert reset_env.close_calls == 1
        env_counts_before_a = (
            reset_env.reset_calls,
            reset_env.step_calls,
            reset_env.close_calls,
        )
        predict_before = adapter.predict_calls
        patterns = [
            _pattern(0, identifier="P01"),
            _pattern(1, identifier="P02"),
        ]
        sequential = compare_saved_baseline(adapter, baseline, patterns, batch=False)
        assert adapter.predict_calls - predict_before == len(baseline["records"]) * len(patterns)
        predict_after_sequential = adapter.predict_calls
        batched = compare_saved_baseline(adapter, baseline, patterns, batch=True)
        assert adapter.predict_calls - predict_after_sequential == len(patterns)
        assert (reset_env.reset_calls, reset_env.step_calls, reset_env.close_calls) == env_counts_before_a
    assert adapter.make_env_calls == made_before
    assert baseline == baseline_snapshot
    assert len(sequential) == len(patterns)
    assert len(batched) == len(patterns)
    assert all(
        len(value["records"]) == len(baseline["records"])
        for value in sequential + batched
    )
    assert sum(len(value["records"]) for value in sequential) == len(baseline["records"]) * len(patterns)
    for sequential_value, batched_value in zip(sequential, batched):
        for sequential_record, batched_record in zip(
            sequential_value["records"], batched_value["records"]
        ):
            assert sequential_record["step"] == batched_record["step"]
            assert sequential_record["action"] == batched_record["action"]
            assert sequential_record["argmax_changed"] == batched_record["argmax_changed"]
            np.testing.assert_allclose(
                sequential_record["p"], batched_record["p"], rtol=1e-12, atol=1e-12
            )
            np.testing.assert_allclose(
                sequential_record["q"], batched_record["q"], rtol=1e-12, atol=1e-12
            )
            assert sequential_record["js"] == pytest.approx(
                batched_record["js"], rel=1e-12, abs=1e-12
            )
    assert len(adapter.prediction_inputs) == len(baseline["records"]) * 3 + len(patterns)
    assert adapter.prediction_inputs[-1].shape == (len(baseline["records"]), 8)
    assert all("reward" not in record for record in sequential[0]["records"])


def test_offline_requires_saved_identity() -> None:
    adapter = SyntheticAdapter(dimension=4, action_count=2, steps=1)
    baseline = {
        "records": [{"input": [0.1, 0.2, 0.3, 0.4], "probabilities": [0.5, 0.5]}]
    }
    with pytest.raises(PolicyComparisonError, match="identity"):
        compare_saved_baseline(adapter, baseline, [_pattern(0)])


def test_offline_argmax_changed_uses_q_argmax_when_sampled_action_differs() -> None:
    class SampledActionAdapter(SyntheticAdapter):
        def predict(self, inputs: object, deterministic: bool = True) -> tuple[np.ndarray, int]:
            values = np.asarray(inputs)
            if values.ndim == 1:
                return np.asarray([0.75, 0.25]), 0 if deterministic else 1
            if values.ndim == 2:
                probabilities = np.tile(np.asarray([0.75, 0.25]), (values.shape[0], 1))
                actions = np.full(values.shape[0], 0 if deterministic else 1, dtype=np.int64)
                return probabilities, actions
            raise ValueError(f"unexpected input shape {values.shape}")

    adapter = SampledActionAdapter(dimension=4, action_count=2, steps=1)
    with tempfile.TemporaryDirectory() as directory:
        baseline = collect_baseline(
            adapter,
            scenario_seed=5,
            policy_seed=5,
            horizon=1,
            run_dir=directory,
            record_gif=False,
        )
    result = compare_saved_baseline(
        adapter,
        baseline,
        [_pattern(0)],
        deterministic=False,
    )
    record = result[0]["records"][0]
    assert result[0]["status"] == "complete"
    assert record["baseline_action"] == 0
    assert record["action"] == 1
    assert record["argmax_changed"] is False
    assert record["p"] == record["q"]


@pytest.mark.parametrize("case", ["failed", "empty", "missing_input", "missing_probabilities"])
def test_offline_rejects_failed_or_partial_baseline_without_inference(case: str) -> None:
    adapter = SyntheticAdapter(dimension=4, action_count=2, steps=1)
    with tempfile.TemporaryDirectory() as directory:
        baseline = collect_baseline(
            adapter,
            scenario_seed=5,
            policy_seed=5,
            horizon=1,
            run_dir=directory,
            record_gif=False,
        )
    invalid = copy.deepcopy(baseline)
    if case == "failed":
        invalid["status"] = "failed"
    elif case == "empty":
        invalid["records"] = []
    elif case == "missing_input":
        invalid["records"][0]["input"] = None
    else:
        invalid["records"][0]["probabilities"] = None
    predict_before = adapter.predict_calls
    make_env_before = adapter.make_env_calls
    with pytest.raises(PolicyComparisonError):
        compare_saved_baseline(adapter, invalid, [_pattern(0)])
    assert adapter.predict_calls == predict_before
    assert adapter.make_env_calls == make_env_before


def test_forged_verified_reward_terms_are_revalidated_and_strict_failure_is_logged() -> None:
    def forged_provider(
        _info: object,
        _returned_reward: float,
        *,
        terminated: bool,
        truncated: bool,
    ) -> RewardTermsResult:
        del terminated, truncated
        return RewardTermsResult(
            status="verified",
            terms={"forged": 999.0},
            residual=0.0,
            atol=1e-6,
            rtol=1e-6,
        )

    adapter = SyntheticAdapter(dimension=4, action_count=2, steps=1)
    with tempfile.TemporaryDirectory() as directory:
        result = collect_baseline(
            adapter,
            scenario_seed=5,
            policy_seed=5,
            horizon=1,
            run_dir=directory,
            record_gif=False,
            reward_terms_provider=forged_provider,
        )
    assert result["status"] == "complete"
    terms = result["records"][0]["reward_terms"]
    assert terms["status"] == "mismatch"
    assert terms["residual"] != 0.0

    strict_adapter = SyntheticAdapter(dimension=4, action_count=2, steps=1)
    with tempfile.TemporaryDirectory() as directory:
        strict_result = collect_baseline(
            strict_adapter,
            scenario_seed=5,
            policy_seed=5,
            horizon=1,
            run_dir=directory,
            record_gif=False,
            reward_terms_provider=forged_provider,
            strict_reward_terms=True,
        )
    assert strict_result["status"] == "failed"
    assert strict_result["failure"]["stage"] == "telemetry"
    strict_record = strict_result["records"][0]
    assert strict_record["executed"] is True
    assert strict_record["reward"] is not None
    assert strict_record["reward_terms"]["status"] == "mismatch"
    assert strict_record["reward_terms"]["residual"] != 0.0


class _StepResultFaultEnv(SyntheticEnv):
    def __init__(self, fault: str) -> None:
        super().__init__(dimension=4, action_count=2, steps=1)
        self.fault = fault

    def step(self, action: int) -> object:
        observation, reward, terminated, truncated, info = super().step(action)
        if self.fault == "nonfinite_reward":
            return observation, np.inf, terminated, truncated, info
        return observation, reward, terminated, truncated


class _StepResultFaultAdapter(SyntheticAdapter):
    def __init__(self, fault: str) -> None:
        super().__init__(dimension=4, action_count=2, steps=1)
        self.fault = fault

    def make_env(self) -> _StepResultFaultEnv:
        self.make_env_calls += 1
        self.last_env = _StepResultFaultEnv(self.fault)
        return self.last_env


@pytest.mark.parametrize("fault", ["nonfinite_reward", "malformed_tuple"])
def test_returned_nonfinite_or_malformed_step_result_is_executed_step_result_failure(fault: str) -> None:
    adapter = _StepResultFaultAdapter(fault)
    with tempfile.TemporaryDirectory() as directory:
        result = collect_baseline(
            adapter,
            scenario_seed=5,
            policy_seed=5,
            horizon=1,
            run_dir=directory,
            record_gif=False,
        )
    assert result["status"] == "failed"
    assert result["failure"]["stage"] == "step_result"
    assert result["failure"]["step"] == 0
    assert len(result["records"]) == 1
    assert result["records"][0]["executed"] is True
    assert result["records"][0]["reward"] is None
    assert adapter.last_env is not None
    assert adapter.last_env.step_calls == 1


def test_closed_loop_natural_end_uses_actual_length_and_runs_intervened_action() -> None:
    # Both episodes use the same synthetic environment configuration.  The
    # fixed input changes the action-derived trajectory and reaches the
    # configured natural goal at step 50; baseline remains on its 127-step
    # horizon.  No collector flag changes the environment by pattern.
    baseline_adapter = SyntheticAdapter(
        dimension=259,
        action_count=9,
        steps=127,
        natural_end_step=50,
    )
    adapter = SyntheticAdapter(
        dimension=259,
        action_count=9,
        steps=127,
        natural_end_step=50,
    )
    with tempfile.TemporaryDirectory() as directory:
        baseline = collect_baseline(
            baseline_adapter,
            scenario_seed=5,
            policy_seed=5,
            horizon=127,
            run_dir=directory,
            record_gif=False,
        )
        result = collect_closed_loop(
            adapter,
            _pattern(0),
            scenario_seed=5,
            policy_seed=5,
            horizon=127,
            run_dir=directory,
            record_gif=False,
        )
    assert result["status"] == "complete"
    assert len(result["records"]) == 50
    assert result["applied_count"] == 50
    assert result["termination"]["step"] == 49
    assert result["termination"]["terminated"] is True
    assert all(record["executed"] for record in result["records"])
    assert len(baseline["records"]) == 127
    assert adapter.last_env is not None and adapter.last_env.step_calls == 50
    assert baseline_adapter.last_env is not None and baseline_adapter.last_env.step_calls == 127
    assert [record["action"] for record in baseline["records"][:50]] != [
        record["action"] for record in result["records"]
    ]
    assert baseline["records"][1]["input"] != result["records"][1]["input"]
    assert adapter.last_env.actions == [record["action"] for record in result["records"]]
    assert all(record["next_observation"] is not None for record in result["records"])


def test_successful_step_reward_survives_frame_failure_and_pattern_stops() -> None:
    adapter = SyntheticAdapter(dimension=8, action_count=3, steps=10, fail_frame_at=2)
    with tempfile.TemporaryDirectory() as directory:
        result = collect_closed_loop(
            adapter,
            _pattern(0),
            scenario_seed=5,
            policy_seed=5,
            horizon=10,
            run_dir=directory,
            record_gif=True,
        )
    assert result["status"] == "failed"
    assert result["failure"]["stage"] == "frame"
    assert len(result["records"]) == 2
    assert result["records"][1]["executed"] is True
    assert isinstance(result["records"][1]["reward"], float)
    assert adapter.make_env_calls == 1


def test_reward_terms_keep_signs_and_distinguish_unavailable_mismatch_nonfinite() -> None:
    verified = validate_reward_terms({"progress": 1.5, "penalty": -0.5}, 1.0)
    assert verified.status == "verified"
    assert verified.residual == 0.0
    assert reward_terms_result(None, 1.0, terminated=False, truncated=False).status == "unavailable"
    assert validate_reward_terms({"progress": 1.0}, 2.0).status == "mismatch"
    assert validate_reward_terms({"progress": np.inf}, 1.0).status == "nonfinite"
    assert validate_reward_terms({"progress": "bad"}, 1.0).status == "malformed"


def test_synthetic_n2_run_writes_separate_raw_and_visual_artifacts() -> None:
    config = {
        "backend": "synthetic",
        "synthetic_dimension": 8,
        "synthetic_action_count": 3,
        "synthetic_steps": 3,
        "horizon": 3,
        "record_gif": True,
        "patterns": [
            {"id": "P01", "name": "one", "indices": [0], "fixed_value": -1.0},
            {"id": "P02", "name": "two", "indices": [1], "fixed_value": -1.0},
        ],
    }
    with tempfile.TemporaryDirectory() as directory:
        result = run_experiment(config, run_dir=Path(directory) / "run")
        run_dir = result.run_dir
        assert result.status == "complete", result.errors
        assert (run_dir / "data" / "P00_baseline.json").is_file()
        assert (run_dir / "data" / "P01.json").is_file()
        assert (run_dir / "data" / "P02_offline.json").is_file()
        assert (run_dir / "report.html").is_file()
        assert len(list((run_dir / "patterns").glob("*/rollout.gif"))) == 2
        assert len(list((run_dir / "patterns").glob("*/policy_change.png"))) == 2
        manifest = read_manifest(run_dir)
        assert manifest["base_main_sha"] == "7849aad80ac353fd616c1a1398c11dd3497eed05"
        assert read_rollout(run_dir, "P00")["status"] == "complete"
