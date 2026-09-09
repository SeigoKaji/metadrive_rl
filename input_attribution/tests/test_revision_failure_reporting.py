"""回帰: runtime failure の保存診断、stage pointer、report再生成境界。"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
import tomllib

import numpy as np

from input_attribution.adapters.fake import FakePolicyAdapter
from input_attribution.cli import _closed_loop_runtime_diagnostics, command_report
from input_attribution.cli import main as cli_main
from input_attribution.closed_loop import ClosedLoopResult
from input_attribution.reporting import (
    _aggregate_closed_steps,
    _closed_assessment_status,
    _closed_measurement_details,
    _stage_roots,
    _status_text,
    generate_report,
)
from input_attribution.trajectory_metrics import summarize_trajectory


class _CLIFailingAdapter(FakePolicyAdapter):
    """Fake adapter whose changed reference input fails once.

    Pbad points at the second saved observation while Pafter points at the
    first.  Looking at the current environment's first step and input value
    identifies Pbad without relying on the number of setup environments the
    CLI happens to create.
    """

    def probabilities(self, observations):
        environment = getattr(self, "env", None)
        values = np.asarray(observations)
        if (
            environment is not None
            and int(getattr(environment, "step_count", -1)) == 0
            and values.size
            and float(values.reshape(-1)[0]) > 0.4
        ):
            raise RuntimeError("synthetic CLI policy failure")
        return super().probabilities(observations)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _trajectory(pattern_id: str, *, execution_status: str = "completed") -> dict[str, object]:
    return {
        "pattern_id": pattern_id,
        "episode_id": "episode-0",
        "execution_status": execution_status,
        "assessment_status": "evaluated" if execution_status == "completed" else "not_evaluable_runtime_error",
        "terminal_reason": "arrive_dest" if execution_status == "completed" else "runtime_error",
        "failure_phase": None if execution_status == "completed" else "policy",
        "failure_reason": None if execution_status == "completed" else "synthetic policy failure",
        "metrics": {"status": "success", "lane_rms_m": 1.0},
        "records": [{"step": 0, "intervention": {"applied": False}, "env_step_called": False}],
    }


def test_failed_stage_pointer_is_reported_without_falling_back_to_stale_success(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    current = run_dir / "02_closed_loop" / "analysis-current"
    stale = run_dir / "02_closed_loop" / "analysis-stale"
    _write_json(current / "Pfailed" / "episode-0" / "trajectory.json", _trajectory("Pfailed", execution_status="failed"))
    _write_json(stale / "Pold" / "episode-0" / "trajectory.json", _trajectory("Pold"))
    _write_json(
        run_dir / "status.json",
        {
            "stages": {
                "closed_loop": {
                    "state": "failed",
                    "analysis_id": "analysis-current",
                    "relative_dir": "02_closed_loop/analysis-current",
                    "error": "RuntimeError: synthetic policy failure",
                }
            }
        },
    )

    roots = _stage_roots(run_dir, json.loads((run_dir / "status.json").read_text(encoding="utf-8")))
    assert roots["closed_loop"] == current
    result = generate_report(run_dir)
    identifiers = {row["pattern_id"] for row in result.summary_rows}
    assert "Pfailed" in identifiers
    assert "Pold" not in identifiers
    assert "失敗" in result.status
    markdown = (result.output_dir / "report.md").read_text(encoding="utf-8")
    assert "Pfailed" in markdown
    assert "Pold" not in markdown


def test_failed_stage_without_pointer_does_not_scan_canonical_success(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_json(
        run_dir / "02_closed_loop" / "stale-success" / "Pold" / "episode-0" / "trajectory.json",
        _trajectory("Pold"),
    )
    status = {"stages": {"closed_loop": {"state": "failed", "error": "missing pointer"}}}
    _write_json(run_dir / "status.json", status)
    roots = _stage_roots(run_dir, status)
    assert roots["closed_loop"] is False
    assert generate_report(run_dir).summary_rows == ()


def test_error_stage_without_pointer_does_not_scan_canonical_success(tmp_path: Path) -> None:
    for state in ("error", "ERROR"):
        run_dir = tmp_path / state
        _write_json(
            run_dir / "02_closed_loop" / "stale-success" / "Pold" / "episode-0" / "trajectory.json",
            _trajectory("Pold"),
        )
        status = {"stages": {"closed_loop": {"state": state, "error": "missing pointer"}}}
        _write_json(run_dir / "status.json", status)
        roots = _stage_roots(run_dir, status)
        assert roots["closed_loop"] is False
        assert generate_report(run_dir).summary_rows == ()


def test_saved_mixed_closed_loop_summary_keeps_episode_specific_status(tmp_path: Path) -> None:
    """A pattern summary's aggregate status list must not taint every episode."""

    run_dir = tmp_path / "mixed-summary"
    closed_root = run_dir / "02_closed_loop"
    _write_json(
        run_dir / "status.json",
        {
            "stages": {
                "closed_loop": {
                    "state": "success",
                    "relative_dir": "02_closed_loop",
                }
            }
        },
    )
    completed = {
        "pattern_id": "Pmixed",
        "episode_id": "episode-0",
        "execution_status": "completed",
        "assessment_status": "evaluated",
        "terminal_reason": "horizon",
        "metrics": {
            "status": "success",
            "lane_rms_m": 1.0,
            "poststep_count": 1,
            "valid_count": 1,
        },
        "records": [{"step": 0}],
    }
    failed = {
        "pattern_id": "Pmixed",
        "episode_id": "episode-1",
        "execution_status": "failed",
        "assessment_status": "not_evaluable_runtime_error",
        "terminal_reason": "runtime_error",
        "failure_phase": "policy",
        "failure_reason": "synthetic policy failure",
        "metrics": {
            "status": "success",
            "lane_rms_m": 9.0,
            "poststep_count": 1,
            "valid_count": 1,
        },
        "records": [{"step": 0}],
    }
    _write_json(closed_root / "Pmixed" / "episode-0" / "trajectory.json", completed)
    _write_json(closed_root / "Pmixed" / "episode-1" / "trajectory.json", failed)
    # Use the actual producer so the report reader sees the same pattern
    # summary shape as a saved ClosedLoopResult.  Its status arrays describe
    # the collection as a whole; they have no episode-to-status correspondence
    # and must not be copied into each metric row.
    summary = ClosedLoopResult(store=None, patterns={"Pmixed": [completed, failed]}).as_dict()
    _write_json(closed_root / "summary.json", summary)

    result = generate_report(run_dir)
    row = next(item for item in result.summary_rows if item["pattern_id"] == "Pmixed")
    episodes = {item["episode_id"]: item for item in row["closed_loop_episode_rows"]}
    assert row["closed_loop_episode_count"] == 2
    assert row["closed_loop_natural_episode_count"] == 1
    assert row["closed_loop_failed_episode_count"] == 1
    assert row["closed_loop_interrupted_episode_count"] == 1
    assert row["closed_loop_lane_rms_m"] == 1.0
    assert episodes["episode-0"]["closed_loop_execution_status"] == "completed"
    assert episodes["episode-0"]["closed_loop_episode_natural"] is True
    assert episodes["episode-1"]["closed_loop_execution_status"] == "failed"
    assert episodes["episode-1"]["closed_loop_episode_interrupted"] is True


def test_report_regeneration_does_not_mutate_run_status(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    status = {
        "run_id": "immutable-status",
        "stages": {
            "run": {"state": "failed", "error": "preserve me"},
            "report": {"state": "pending", "diagnostic": "preserve me too"},
        },
    }
    _write_json(run_dir / "status.json", status)
    before = (run_dir / "status.json").read_bytes()
    assert command_report(Namespace(run_dir=run_dir, output_dir=None)) == 0
    assert (run_dir / "status.json").read_bytes() == before


def test_cli_diagnostic_counts_runtime_failure_and_abort_separately() -> None:
    class Result:
        patterns = {
            "P00": [{"execution_status": "completed", "assessment_status": "control_baseline"}],
            "Pfailed": [{"execution_status": "failed", "assessment_status": "not_evaluable_runtime_error"}],
            "Pabort": [{"execution_status": "aborted", "assessment_status": "not_evaluable_intervention_abort"}],
            "Pcrash": [{"execution_status": "completed", "assessment_status": "evaluated", "terminal_reason": "crash"}],
        }

    diagnostics = _closed_loop_runtime_diagnostics(Result())
    assert diagnostics["failed_episode_count"] == 1
    assert diagnostics["aborted_episode_count"] == 1
    assert diagnostics["completed_episode_count"] == 2
    assert diagnostics["runtime_failure_count"] == 2


def test_cli_diagnostic_prefers_current_result_counts_over_summary_aliases() -> None:
    class Result:
        patterns = {"P00": [{"execution_status": "completed"}]}

        def as_dict(self):
            return {
                "status": "partial_failure",
                "counts": {
                    "episode_count": 3,
                    "completed_episode_count": 1,
                    "failed_episode_count": 1,
                    "aborted_episode_count": 1,
                    "runtime_failure_count": 2,
                    "required_experiment_failed": True,
                },
                # A legacy summary alias must not invent two more episodes.
                "execution_status": ["completed", "failed", "aborted"],
            }

    diagnostics = _closed_loop_runtime_diagnostics(Result())
    assert diagnostics["episode_count"] == 3
    assert diagnostics["completed_episode_count"] == 1
    assert diagnostics["failed_episode_count"] == 1
    assert diagnostics["aborted_episode_count"] == 1
    assert diagnostics["runtime_failure_count"] == 2


def test_status_text_includes_partial_execution_failure_counts() -> None:
    closed = [
        {
            "closed_loop_episode_count": 2,
            "closed_loop_natural_episode_count": 1,
            "closed_loop_interrupted_episode_count": 1,
            "closed_loop_episode_rows": [
                {"closed_loop_execution_status": "completed"},
                {"closed_loop_execution_status": "failed"},
            ],
        }
    ]
    status = _status_text(
        {"stages": {"closed_loop": {"state": "success"}}},
        offline=(),
        closed=closed,
        ig=(),
    )
    assert "一部失敗" in status


def test_status_text_uses_closed_loop_stage_counts_without_double_counting() -> None:
    status = _status_text(
        {
            "stages": {
                "closed_loop": {
                    "state": "failed",
                    "counts": {
                        "episode_count": 3,
                        "completed_episode_count": 1,
                        "failed_episode_count": 1,
                        "aborted_episode_count": 1,
                        "runtime_failure_count": 2,
                    },
                }
            }
        },
        offline=(),
        closed=[
            {
                "closed_loop_episode_rows": [
                    {"closed_loop_execution_status": "failed"},
                    {"closed_loop_execution_status": "completed"},
                ]
            }
        ],
        ig=(),
    )
    assert status == "一部失敗（completed=1、failed=1、aborted=1）"


def test_status_text_episode_count_only_is_legacy_fallback() -> None:
    status = _status_text(
        {
            "stages": {
                "closed_loop": {
                    "state": "success",
                    "counts": {"episode_count": 2},
                }
            }
        },
        offline=(),
        closed=[
            {
                "closed_loop_episode_rows": [
                    {"closed_loop_execution_status": "failed"},
                ]
            }
        ],
        ig=(),
    )
    assert "failed=1" in status
    assert "completed=0" in status


def test_report_excludes_env_step_attempt_without_returned_telemetry() -> None:
    rows = [
        {
            "env_step_called": True,
            "env_step_returned": False,
            "post_telemetry": {"target_lane_valid": True, "target_lane_offset_m": 9.0, "speed_mps": 9.0},
        },
        {
            "env_step_called": True,
            "env_step_returned": True,
            "post_telemetry": {"target_lane_valid": None, "target_lane_offset_m": 9.0, "speed_mps": 9.0},
        },
        {
            "env_step_called": True,
            "env_step_returned": True,
            "post_telemetry": {"target_lane_valid": True, "target_lane_offset_m": 1.0, "speed_mps": 2.0},
        },
    ]
    aggregates = _aggregate_closed_steps(rows)
    assert aggregates["closed_loop_poststep_count"] == 2
    assert aggregates["closed_loop_lane_rms_m"] == 1.0
    assert aggregates["closed_loop_speed_mean_mps"] == 5.5


def test_failed_baseline_is_not_reported_as_control_only() -> None:
    assert _closed_assessment_status(
        {
            "pattern_id": "P00",
            "closed_loop_status": "failed",
            "closed_loop_failed_episode_count": 1,
        }
    ) == "実行失敗"


def test_measurement_details_mark_known_event_with_unknown_interval_partial() -> None:
    details = _closed_measurement_details(
        {
            "event_state_coverage": {
                "departure": {"status": "complete", "measured_count": 2, "expected_count": 2},
                "low_speed": {"status": "complete", "measured_count": 2, "expected_count": 2},
            },
            "departure_duration_measurement": {
                "status": "unavailable",
                "state_measured_count": 2,
                "state_expected_count": 2,
            },
            "low_speed_duration_measurement": {
                "status": "unavailable",
                "state_measured_count": 2,
                "state_expected_count": 2,
            },
            "lane_metric_measurement": {"status": "complete", "measured_count": 2, "expected_count": 2},
            "duration_measurement": {"status": "unavailable"},
            "interval_measurement": {"status": "unavailable"},
        }
    )
    assert details["closed_loop_measurement_status"] == "一部未計測"
    assert "一部未計測" in details["closed_loop_departure_measurement_detail"]


def test_cli_run_saves_partial_failure_and_returns_nonzero(tmp_path: Path, capsys) -> None:
    source = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "configs" / "fake_259.toml").read_text(encoding="utf-8")
    )
    source["analysis"]["individual_inputs"] = False
    source["schema"]["path"] = str(Path(__file__).resolve().parents[1] / "schemas" / "official_259.json")
    source["adapter"] = {"factory": f"{__name__}:_CLIFailingAdapter"}
    source["scenario"]["episodes"] = 1
    source["scenario"]["horizon"] = 2
    source["closed_loop"].update(
        {
            "episodes": 1,
            "max_steps": 2,
            "patterns": ["P00", "Pbad", "Pafter"],
        }
    )
    source["patterns"] = [
        {"id": "P00", "description": "control", "indices": [], "method": "identity"},
        {"id": "Pbad", "description": "synthetic failure", "indices": [0], "method": "reference", "reference_id": "episode-0:1"},
        {"id": "Pafter", "description": "continues after failure", "indices": [1], "method": "reference", "reference_id": "episode-0:0"},
    ]
    source["output"].update(
        {
            "root": str(tmp_path / "results"),
            "experiment": "cli_failure",
            "model": "fake_policy",
        }
    )
    config_path = tmp_path / "failure.json"
    config_path.write_text(json.dumps(source), encoding="utf-8")

    exit_code = cli_main(["run", "--config", str(config_path)])
    assert exit_code != 0
    first_cli_output = capsys.readouterr()
    assert "closed-loop counts: completed=2, failed=1, aborted=0" in first_cli_output.err
    run_parent = tmp_path / "results" / "cli_failure" / "fake_policy"
    run_dirs = [path for path in run_parent.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["stages"]["run"]["state"] == "failed"
    assert status["stages"]["run"]["counts"]["runtime_failure_count"] == 1
    closed_status = status["stages"]["closed_loop"]
    assert closed_status["state"] == "failed"
    assert closed_status["counts"]["failed_episode_count"] == 1
    assert closed_status["counts"]["completed_episode_count"] == 2

    closed_root = run_dir / closed_status["relative_dir"]
    failed = json.loads((closed_root / "Pbad" / "episode-0" / "trajectory.json").read_text(encoding="utf-8"))
    continued = json.loads((closed_root / "Pafter" / "episode-0" / "trajectory.json").read_text(encoding="utf-8"))
    assert failed["execution_status"] == "failed"
    assert failed["records"]
    assert continued["execution_status"] == "completed"
    assert (closed_root / "Pbad" / "episode-0" / "observations.npy").is_file()
    assert (closed_root / "Pbad" / "episode-0" / "modified_observations.npy").is_file()

    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "一部失敗" in report or "失敗（" in report
    assert "Pbad" in report
    assert "Pafter" in report
    detail_csv = (run_dir / "closed_loop_details.csv").read_text(encoding="utf-8-sig")
    assert "failure_phase" in detail_csv
    assert "policy" in detail_csv

    # The same saved run is also exercised through the standalone command.
    # It allocates a fresh closed-loop analysis pointer and must preserve the
    # nonzero runtime exit contract plus the current failure counts.
    assert cli_main(["closed-loop", "--run-dir", str(run_dir)]) != 0
    rerun_status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    rerun_closed_status = rerun_status["stages"]["closed_loop"]
    assert rerun_closed_status["state"] == "failed"
    assert rerun_closed_status["counts"]["failed_episode_count"] == 1
    assert rerun_closed_status["counts"]["completed_episode_count"] == 2


def test_cli_report_keeps_partial_measurement_as_na(tmp_path: Path) -> None:
    records = [
        {
            "step": 0,
            "pre_time": 0.0,
            "post_time": 0.1,
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
            "post_telemetry": {"simulation_time_s": 0.2},
        },
    ]
    metrics = dict(summarize_trajectory(records))
    metrics.update(
        {
            "status": "success",
            "execution_status": "completed",
            "assessment_status": "evaluated",
            "terminal_reason": "horizon",
        }
    )
    run_dir = tmp_path / "partial-measurement"
    _write_json(
        run_dir / "status.json",
        {
            "stages": {
                "closed_loop": {
                    "state": "success",
                    "relative_dir": "02_closed_loop/analysis",
                }
            }
        },
    )
    _write_json(
        run_dir / "02_closed_loop" / "analysis" / "Ppartial" / "episode-0" / "trajectory.json",
        {
            "pattern_id": "Ppartial",
            "episode_id": "episode-0",
            "execution_status": "completed",
            "assessment_status": "evaluated",
            "terminal_reason": "horizon",
            "metrics": metrics,
            "records": records,
        },
    )

    assert cli_main(["report", "--run-dir", str(run_dir)]) == 0
    summary = (run_dir / "summary.csv").read_text(encoding="utf-8-sig")
    assert "closed_loop_departure_time_s" in summary
    report_text = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "逸脱時間(s)" in report_text
    assert "N/A" in report_text
    partial_rows = [line for line in report_text.splitlines() if line.startswith("| Ppartial |")]
    assert any(
        "一部未計測" in line
        and "RMS既知 n=1/2" in line
        and "状態既知 n=1/2" in line
        and "既知逸脱時間=0 s" in line
        for line in partial_rows
    )
    details_text = (run_dir / "report_details.md").read_text(encoding="utf-8")
    assert "①-B 計測完全性" in details_text
    assert "Ppartial: 一部未計測" in details_text
    details = json.loads((run_dir / "report_details.json").read_text(encoding="utf-8"))
    summary_row = details["summary_rows"][0]
    assert summary_row["closed_loop_departure_time_s"] is None
    assert summary_row["closed_loop_low_speed_duration_s"] is None
    assert summary_row["closed_loop_valid_rate"] is None
    raw = json.dumps(details["raw_closed_loop"], ensure_ascii=False)
    assert "event_state_coverage" in raw
    assert "known_event_duration_s" in raw
    assert metrics["departure_time_s"] is None
    assert metrics["low_speed_duration_s"] is None
