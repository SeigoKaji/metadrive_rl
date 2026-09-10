"""Independent acceptance checks for reward and porting contracts.

The tests here deliberately exercise the public collection, comparison,
runner, schema, and report boundaries.  They use short deterministic synthetic
episodes so a failure in telemetry or reward accounting remains observable
without starting a MetaDrive engine.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from input_attribution.collection import collect_baseline, collect_closed_loop
from input_attribution.interventions import Intervention
from input_attribution.policy_comparison import compare_saved_baseline, policy_identity
from input_attribution.reporting import generate_report
from input_attribution.reward_adapter import reward_terms_result
from input_attribution.runner import run_experiment
from input_attribution.schema import load_schema, synthetic_262_schema_document
from input_attribution.storage import ensure_run_layout, write_manifest, write_rollout
from input_attribution.synthetic import SyntheticAdapter, SyntheticEnv


def _pattern(*indices: int, identifier: str = "P01") -> Intervention:
    return Intervention(identifier, "acceptance intervention", tuple(indices), -1.0)


def _good_event(
    *,
    reward: float = 1.25,
    terminated: bool = False,
    truncated: bool = False,
    info: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "info": dict(
            info
            if info is not None
            else {"reward_terms": {"progress": 1.5, "penalty": -0.25}}
        ),
    }


class EventEnv(SyntheticEnv):
    """Synthetic environment whose transitions are explicit test events."""

    def __init__(self, events: list[dict[str, object]], *, dimension: int = 4) -> None:
        super().__init__(
            dimension=dimension,
            action_count=3,
            steps=max(1, len(events)),
            include_reward_terms=False,
        )
        self.events = events

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, object]:
        if self.step_index >= len(self.events):
            raise RuntimeError("event stream exhausted")
        event = self.events[self.step_index]
        failure = event.get("raise")
        if failure is not None:
            raise RuntimeError(str(failure))
        observation, _ignored_reward, _ignored_terminated, _ignored_truncated, _ignored_info = super().step(action)
        info = event.get("info", {})
        if info is None:
            info = {}
        # Keep malformed return metadata intact so the collector's boundary
        # validation is tested.  Normal events still carry actual booleans.
        terminated = event.get("terminated", False)
        truncated = event.get("truncated", False)
        return (
            observation,
            float(event["reward"]),
            terminated,  # type: ignore[return-value]
            truncated,  # type: ignore[return-value]
            info,
        )


class EventAdapter(SyntheticAdapter):
    """Synthetic policy adapter with an injectable transition stream/fault."""

    def __init__(
        self,
        events: list[dict[str, object]],
        *,
        dimension: int = 4,
        snapshot_fail_after: int | None = None,
        frame_failure: bool = False,
    ) -> None:
        super().__init__(dimension=dimension, action_count=3, steps=max(1, len(events)))
        self.events = events
        self.snapshot_fail_after = snapshot_fail_after
        self.frame_failure = frame_failure

    def make_env(self) -> EventEnv:
        self.make_env_calls += 1
        self.last_env = EventEnv(self.events, dimension=self.dimension)
        return self.last_env

    def snapshot(self, env: EventEnv) -> dict[str, object]:
        if self.snapshot_fail_after is not None and env.step_index >= self.snapshot_fail_after:
            raise RuntimeError(f"snapshot failure at step {env.step_index}")
        return env.snapshot()

    def frame(self, env: EventEnv) -> np.ndarray:
        if self.frame_failure:
            raise RuntimeError("frame failure after successful step")
        return env.frame()


def test_reward_terms_preserve_signs_and_terminal_overwrite_order() -> None:
    events = [
        _good_event(
            reward=1.5,
            info={"reward_terms": {"progress": 2.0, "penalty": -0.5}},
        ),
        _good_event(
            reward=4.25,
            terminated=True,
            info={
                # These are prior components retained only as diagnostic info;
                # the returned terminal terms are the post-overwrite values.
                "pre_terminal_terms": {"progress": 10.0, "penalty": -2.0},
                "reward_terms": {
                    "terminal_overwrite": 5.0,
                    "post_overwrite_adjustment": -0.75,
                },
            },
        ),
    ]
    adapter = EventAdapter(events)
    result = collect_baseline(
        adapter,
        scenario_seed=5,
        policy_seed=5,
        horizon=5,
        record_gif=False,
    )

    assert result["status"] == "complete"
    records = result["records"]
    assert len(records) == 2
    first_terms = records[0]["reward_terms"]
    assert first_terms["status"] == "verified"
    assert first_terms["terms"] == {"progress": 2.0, "penalty": -0.5}
    terminal_terms = records[1]["reward_terms"]
    assert terminal_terms["status"] == "verified"
    assert terminal_terms["terms"] == {
        "terminal_overwrite": 5.0,
        "post_overwrite_adjustment": -0.75,
    }
    assert "progress" not in terminal_terms["terms"]
    for record in records:
        terms = record["reward_terms"]["terms"]
        assert np.isclose(sum(terms.values()), record["reward"])
    assert result["termination"]["step"] == 1
    assert result["termination"]["terminated"] is True


def test_reward_provider_hook_and_strict_error_states(monkeypatch: pytest.MonkeyPatch) -> None:
    import input_attribution.reward_adapter as reward_adapter_module

    original_extract = reward_adapter_module.extract_reward_terms
    calls: list[dict[str, object]] = []

    def alternate_extract(
        info: Mapping[str, object] | None,
        returned_reward: float,
        *,
        terminated: bool,
        truncated: bool,
    ) -> Mapping[str, object] | None:
        assert info is not None
        calls.append(
            {
                "info": dict(info),
                "reward": returned_reward,
                "terminated": terminated,
                "truncated": truncated,
            }
        )
        return info.get("alternate_terms", info.get("reward_terms"))  # type: ignore[return-value]

    monkeypatch.setattr(reward_adapter_module, "extract_reward_terms", alternate_extract)
    adapter = EventAdapter(
        [
            _good_event(
                reward=1.0,
                info={"alternate_terms": {"forward": 1.25, "drag": -0.25}},
            ),
            _good_event(
                reward=0.5,
                truncated=True,
                info={"alternate_terms": {"forward": 0.75, "drag": -0.25}},
            ),
        ]
    )
    result = collect_baseline(
        adapter,
        scenario_seed=5,
        policy_seed=5,
        horizon=2,
        record_gif=False,
    )
    assert result["status"] == "complete"
    assert len(calls) == 2
    assert all(record["reward_terms"]["status"] == "verified" for record in result["records"])
    assert result["records"][0]["reward_terms"]["terms"] == {
        "forward": 1.25,
        "drag": -0.25,
    }

    # The alternate source check above intentionally monkeypatches the
    # package-level extractor.  Restore the common extractor before asserting
    # its independent unavailable/mismatch/nonfinite classifications.
    monkeypatch.setattr(
        reward_adapter_module,
        "extract_reward_terms",
        original_extract,
    )

    assert reward_terms_result(None, 1.0, terminated=False, truncated=False).status == "unavailable"
    assert (
        reward_terms_result(
            {"reward_terms_status": "unavailable"},
            1.0,
            terminated=False,
            truncated=False,
        ).status
        == "unavailable"
    )
    assert (
        reward_terms_result(
            {"reward_terms": {"component": 2.0}},
            1.0,
            terminated=False,
            truncated=False,
        ).status
        == "mismatch"
    )
    assert (
        reward_terms_result(
            {"reward_terms": {"component": np.inf}},
            1.0,
            terminated=False,
            truncated=False,
        ).status
        == "nonfinite"
    )
    assert (
        reward_terms_result(
            {"reward_terms": {"component": "bad"}},
            1.0,
            terminated=False,
            truncated=False,
        ).status
        == "malformed"
    )

    strict_adapter = EventAdapter([_good_event(reward=1.0, info={"reward_terms": {"x": 2.0}})])
    strict_result = collect_baseline(
        strict_adapter,
        scenario_seed=5,
        policy_seed=5,
        horizon=2,
        record_gif=False,
        strict_reward_terms=True,
    )
    assert strict_result["status"] == "failed"
    assert strict_result["failure"]["stage"] == "telemetry"
    assert len(strict_result["records"]) == 1
    assert strict_result["records"][0]["executed"] is True
    assert strict_result["records"][0]["reward"] == 1.0
    strict_terms = strict_result["records"][0]["reward_terms"]
    assert strict_terms["status"] == "mismatch"
    assert strict_terms["residual"] == pytest.approx(1.0)


def test_invalid_returned_step_metadata_keeps_reward_and_is_distinct_from_step_exception() -> None:
    """A returned transition is retained before validating its end flags.

    ``env.step`` itself did return a reward here, but its termination metadata
    is invalid.  The failure must therefore remain distinguishable from an
    exception raised inside ``env.step`` while retaining the acquired reward
    and executed-step count.
    """

    adapter = EventAdapter(
        [
            _good_event(
                reward=2.5,
                terminated="malformed-terminated",
                info={"reward_terms": {"progress": 2.5}},
            ),
            _good_event(reward=0.25),
        ]
    )
    result = collect_baseline(
        adapter,
        scenario_seed=5,
        policy_seed=5,
        horizon=3,
        record_gif=False,
    )

    assert result["status"] == "failed"
    failure = result["failure"]
    assert isinstance(failure, Mapping)
    assert failure["stage"] == "step_result"
    assert failure["step"] == 0
    records = result["records"]
    assert len(records) == 1
    assert records[0]["executed"] is True
    assert records[0]["reward"] == 2.5
    assert adapter.last_env is not None
    assert adapter.last_env.step_calls == 1
    assert adapter.last_env.close_calls == 1


@pytest.mark.parametrize("fault", ["step", "provider", "info", "snapshot", "frame"])
def test_post_step_failures_retain_reward_and_stop_pattern(
    fault: str,
    tmp_path: Path,
) -> None:
    if fault == "step":
        events = [_good_event(), {"raise": "step exception"}]
        adapter = EventAdapter(events)
        kwargs: dict[str, object] = {"record_gif": False}
        expected_stage = "step"
        expected_records = 2
    elif fault == "provider":
        adapter = EventAdapter([_good_event()])

        def provider(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("provider exception")

        kwargs = {"record_gif": False, "reward_terms_provider": provider}
        expected_stage = "telemetry"
        expected_records = 1
    elif fault == "info":
        adapter = EventAdapter([_good_event(info={"unserializable": object()})])
        kwargs = {"record_gif": False}
        expected_stage = "telemetry"
        expected_records = 1
    elif fault == "snapshot":
        adapter = EventAdapter([_good_event()], snapshot_fail_after=1)
        kwargs = {"record_gif": False}
        expected_stage = "telemetry"
        expected_records = 1
    else:
        adapter = EventAdapter([_good_event()], frame_failure=True)
        kwargs = {"record_gif": True, "run_dir": str(tmp_path)}
        expected_stage = "frame"
        expected_records = 1

    result = collect_closed_loop(
        adapter,
        _pattern(0),
        scenario_seed=5,
        policy_seed=5,
        horizon=3,
        **kwargs,
    )
    assert result["status"] == "failed"
    assert result["failure"]["stage"] == expected_stage
    assert len(result["records"]) == expected_records
    assert result["records"][0]["executed"] is True
    assert result["records"][0]["reward"] == 1.25
    assert adapter.last_env is not None
    assert adapter.last_env.step_calls == 1
    assert adapter.last_env.close_calls == 1
    if fault == "step":
        assert result["records"][1]["executed"] is False
        assert result["records"][1]["reward"] is None


class PatternFaultAdapter(EventAdapter):
    """Baseline succeeds while the independent intervention run fails."""

    def __init__(self) -> None:
        super().__init__([_good_event(), _good_event()])

    def make_env(self) -> EventEnv:
        self.make_env_calls += 1
        events = (
            [_good_event(), {"raise": "intervention step exception"}]
            if self.make_env_calls > 1
            else [_good_event(), _good_event(truncated=True)]
        )
        self.last_env = EventEnv(events)
        return self.last_env


def test_failed_pattern_makes_run_partial_and_keeps_executed_prefix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import input_attribution.runner as runner_module

    adapter = PatternFaultAdapter()
    monkeypatch.setattr(runner_module, "build_adapter", lambda _config: adapter)
    result = run_experiment(
        {
            "backend": "synthetic",
            "synthetic_dimension": 4,
            "synthetic_action_count": 3,
            "synthetic_steps": 2,
            "horizon": 2,
            "record_gif": False,
            "patterns": [
                {"id": "P01", "name": "fault", "indices": [0], "fixed_value": -1.0}
            ],
        },
        run_dir=tmp_path / "partial-run",
        report=False,
    )
    assert result.status == "partial"
    assert result.baseline["status"] == "complete"
    assert result.patterns[0]["status"] == "failed"
    assert result.patterns[0]["failure"]["stage"] == "step"
    assert result.patterns[0]["records"][0]["executed"] is True
    assert result.patterns[0]["records"][1]["executed"] is False
    assert result.errors


class AlternateTermsAdapter(EventAdapter):
    """D=262 adapter whose environment exposes a port-specific reward key."""

    def __init__(self, events: list[dict[str, object]]) -> None:
        super().__init__(events, dimension=262)


def test_synthetic_262_high_index_and_alternate_provider_connect_a_b_js_report(
    tmp_path: Path,
) -> None:
    schema_document = synthetic_262_schema_document()
    features = load_schema(schema_document, dimension=262, require_verified=True)
    assert len(features) == 262
    assert features[261]["index"] == 261
    assert all(
        row["source_import"] == "input_attribution.synthetic.SyntheticEnv"
        for row in features
    )

    pattern = _pattern(261)
    pattern_mapping = {
        "id": pattern.id,
        "name": pattern.name,
        "indices": list(pattern.indices),
        "fixed_value": pattern.fixed_value,
    }
    events = [
        _good_event(
            reward=1.0,
            info={"alternate_terms": {"forward": 1.3, "drag": -0.3}},
        ),
        _good_event(
            reward=0.5,
            info={"alternate_terms": {"forward": 0.8, "drag": -0.3}},
        ),
        _good_event(
            reward=0.25,
            truncated=True,
            info={"alternate_terms": {"forward": 0.55, "drag": -0.3}},
        ),
    ]
    adapter = AlternateTermsAdapter(events)
    provider_calls: list[Mapping[str, object]] = []

    def alternate_provider(
        info: Mapping[str, object],
        returned_reward: float,
        *,
        terminated: bool,
        truncated: bool,
    ) -> Mapping[str, object]:
        del returned_reward, terminated, truncated
        provider_calls.append(info)
        return info["alternate_terms"]  # type: ignore[return-value]

    baseline = collect_baseline(
        adapter,
        scenario_seed=5,
        policy_seed=5,
        horizon=3,
        record_gif=False,
        reward_terms_provider=alternate_provider,
    )
    changed = collect_closed_loop(
        adapter,
        pattern,
        scenario_seed=5,
        policy_seed=5,
        horizon=3,
        record_gif=False,
        reward_terms_provider=alternate_provider,
    )
    assert baseline["status"] == "complete"
    assert changed["status"] == "complete"
    assert changed["changed_count"] == 3
    assert len(provider_calls) == 6
    assert all(record["reward_terms"]["status"] == "verified" for record in changed["records"])

    offline = compare_saved_baseline(adapter, baseline, [pattern])
    assert offline[0]["status"] == "complete"
    assert len(offline[0]["records"]) == 3
    assert all(np.isfinite(record["js"]) for record in offline[0]["records"])
    assert all(len(record["q"]) == 3 for record in offline[0]["records"])

    # The normal runner also accepts the explicit high index for D=262; this
    # path uses its regular synthetic reward provider and writes no GIFs.
    core_run = run_experiment(
        {
            "backend": "synthetic",
            "synthetic_dimension": 262,
            "synthetic_action_count": 3,
            "synthetic_steps": 3,
            "horizon": 3,
            "record_gif": False,
            "patterns": [pattern_mapping],
        },
        run_dir=tmp_path / "core-262-run",
        report=False,
    )
    assert core_run.status == "complete", core_run.errors

    report_dir = ensure_run_layout(tmp_path / "alternate-report")
    write_manifest(
        report_dir,
        {
            "format_version": "input-attribution.v1",
            "base_main_sha": "7849aad80ac353fd616c1a1398c11dd3497eed05",
            "backend": "synthetic",
            "config": {
                "backend": "synthetic",
                "synthetic_dimension": 262,
                "record_gif": False,
                "patterns": [pattern_mapping],
            },
            "adapter": policy_identity(adapter),
            "patterns": [pattern_mapping],
            "status": "complete",
        },
    )
    write_rollout(report_dir, baseline)
    write_rollout(report_dir, changed)
    write_rollout(report_dir, offline[0], offline=True)
    report = generate_report(report_dir)
    assert report.status == "success"
    assert (report_dir / "report.html").is_file()
    assert (report_dir / "summary.csv").is_file()
    assert (report_dir / "patterns" / "P01" / "policy_change.png").is_file()
    assert "261" in (report_dir / "report_input_details.csv").read_text(encoding="utf-8")
