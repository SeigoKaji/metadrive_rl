"""回帰: A/Bの成立範囲を分けた主表・詳細・coverageを確認する。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from input_attribution.reporting import (
    _aggregate_closed_steps,
    _closed_intervention_stats,
    _findings,
    _normalise_closed,
    _offline_detail_records,
    _svg_bars,
    _svg_coverage,
    generate_report,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _offline_pattern(pattern_id: str, *, applied: int, changed: int, skipped: int, rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "pattern": {"pattern_id": pattern_id, "indices": [2], "method": "fixed", "replacement": "neutral"},
        "rows": rows,
        "summary": {
            "requested_dimension_count": 1,
            "target_step_count": applied + skipped,
            "eligible_count": applied,
            "applied_row_count": applied,
            "actual_changed_row_count": changed,
            "no_op_row_count": applied - changed,
            "skipped_row_count": skipped,
            "valid_row_count": applied,
            "all_timestamps": {"count": applied, "action_change_rate": 0.0, "selected_probability_delta": {"mean": 0.0}, "js": {"mean": 0.0}},
            "actual_input_changes_only": {"count": changed, "action_change_count": 0, "action_change_rate": 0.0 if changed else None, "selected_probability_delta": {"mean": 1e-12 if changed else None}, "js": {"mean": 6e-16 if changed else None}},
            "skip_reasons": ["road mismatch"] if skipped else [],
        },
    }


def _trajectory(pattern_id: str, *, paired: bool, skipped_after: bool = False) -> dict[str, object]:
    records = []
    for step in range(6):
        skipped = skipped_after and step >= 3
        records.append(
            {
                "step": step,
                "episode_id": "episode-0",
                "post_time": float(step),
                "intervention": {
                    "applied_count": 0 if skipped else 1,
                    "changed": not skipped and step % 2 == 0,
                    "changed_indices": [2] if not skipped and step % 2 == 0 else [],
                    "skipped": skipped,
                    "skip_reason": "unexpected context" if skipped else None,
                },
                "post_telemetry": {
                    "target_lane_valid": False if step == 5 else True,
                    "target_lane_offset_m": 1.0 + step / 10,
                    "speed_mps": 1.0,
                    "route_progress_m": float(step),
                    "simulation_time_s": float(step),
                    "road_segment_id": ["road-a"] if step < 3 else ["road-b"],
                },
                "action_forwarded": 1,
            }
        )
    metrics: dict[str, object] = {
        "status": "success",
        "lane_rms_m": 1.5,
        "arrived": pattern_id == "P00",
        "progress_m": 5.0,
        "duration_s": 5.0,
        "valid_count": 5,
        "terminal_reason": "arrive_dest" if pattern_id == "P00" else "horizon",
    }
    if paired:
        metrics["paired_p00"] = {"status": "matched", "p00_delta_lane_rms_m": 0.25, "p00_delta_progress_m": 1.0}
    return {"pattern_id": pattern_id, "episode_id": "episode-0", "metrics": metrics, "records": records}


def test_report_revision_has_independent_coverage_and_eight_column_tables(tmp_path: Path) -> None:
    run_dir = tmp_path / "fixture"
    _write_json(run_dir / "status.json", {"stages": {"offline": {"state": "success"}, "closed_loop": {"state": "success"}}})
    old_rows = [
        {"episode": 0, "step": step, "actual_input_changed": False, "skipped": step >= 19, "skip_reason": "road mismatch" if step >= 19 else None, "road_segment_id": ["road-a"] if step < 19 else ["road-b"]}
        for step in range(127)
    ]
    local_rows = [
        {"episode": 0, "step": step, "actual_input_changed": True, "skipped": False, "selected_probability_delta": 1e-12, "js_divergence": 6e-16, "road_segment_id": ["road-a"] if step < 19 else ["road-b"]}
        for step in range(127)
    ]
    _write_json(run_dir / "01_offline" / "result.json", {"patterns": [
        _offline_pattern("P01_reference", applied=19, changed=0, skipped=108, rows=old_rows),
        _offline_pattern("P02_local", applied=127, changed=127, skipped=0, rows=local_rows),
    ]})
    _write_json(run_dir / "02_closed_loop" / "P00" / "episode-0" / "trajectory.json", _trajectory("P00", paired=False))
    _write_json(run_dir / "02_closed_loop" / "P02_local" / "episode-0" / "trajectory.json", _trajectory("P02_local", paired=True, skipped_after=True))

    result = generate_report(run_dir)
    rows = {row["pattern_id"]: row for row in result.summary_rows}
    assert rows["P01_reference"]["offline_target_count"] == 127
    assert rows["P01_reference"]["offline_applied_count"] == 19
    assert rows["P01_reference"]["offline_skipped_count"] == 108
    assert rows["P01_reference"]["offline_assessment_status"] == "評価不能：変更なし"
    assert rows["P02_local"]["offline_target_count"] == 127
    assert rows["P02_local"]["offline_changed_count"] == 127
    assert rows["P02_local"]["offline_actual_selected_probability_delta_abs_pp"] is not None
    assert rows["P02_local"]["closed_loop_applied_count"] == 3
    assert rows["P02_local"]["closed_loop_changed_count"] == 2
    assert rows["P02_local"]["closed_loop_skipped_count"] == 3
    assert rows["P02_local"]["p00_delta_lane_rms_m"] == 0.25
    assert rows["P02_local"]["closed_loop_paired_p00_verified"] is True

    markdown = (run_dir / "report.md").read_text(encoding="utf-8")
    html = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "実変更件数/適用" in markdown
    assert "道路区間" in markdown or "road_segment_id" in html
    assert "評価不能：変更なし" in markdown
    assert "①-Aでは P00" not in markdown
    assert "report_details.json" in markdown
    assert "report_offline_coverage.svg" in markdown
    assert "report_closed_loop_coverage.svg" in html
    assert (run_dir / "report_details.md").is_file()
    assert (run_dir / "report_details.json").is_file()
    assert (run_dir / "offline_details.csv").is_file()
    assert (run_dir / "closed_loop_details.csv").is_file()
    assert "road-a" in (run_dir / "offline_details.csv").read_text(encoding="utf-8-sig")
    assert "skipped" in (run_dir / "report_offline_coverage.svg").read_text(encoding="utf-8")

    # Both principal tables have eight or fewer cells per header row.
    for line in markdown.splitlines():
        if line.startswith("| パターン(ID・対象) |"):
            assert line.count("|") - 1 <= 8

    with (run_dir / "summary.csv").open(newline="", encoding="utf-8-sig") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert summary_rows[0]["offline_actual_js_divergence"] in {"", "N/A", "6e-16"}


def test_report_does_not_synthesize_p00_delta_without_episode_pair_evidence(tmp_path: Path) -> None:
    run_dir = tmp_path / "unknown-pair"
    _write_json(run_dir / "02_closed_loop" / "P00" / "episode-0" / "trajectory.json", _trajectory("P00", paired=False))
    _write_json(run_dir / "02_closed_loop" / "P01" / "episode-0" / "trajectory.json", {
        **_trajectory("P01", paired=False),
        "metrics": {"status": "success", "lane_rms_m": 9.0, "progress_m": 100.0, "valid_count": 5},
    })
    result = generate_report(run_dir)
    row = next(row for row in result.summary_rows if row["pattern_id"] == "P01")
    assert row["closed_loop_paired_p00_verified"] is None
    assert row["p00_delta_lane_rms_m"] is None
    assert row["closed_loop_paired_missing_count"] == 1


def test_report_does_not_assign_metres_to_unitless_legacy_progress(tmp_path: Path) -> None:
    run_dir = tmp_path / "unitless-progress"
    for pattern_id in ("P00", "P01"):
        _write_json(
            run_dir / "02_closed_loop" / pattern_id / "episode-0" / "trajectory.json",
            {
                "pattern_id": pattern_id,
                "episode_id": "episode-0",
                "metrics": {
                    "status": "success",
                    "lane_rms_m": 1.0,
                    # This legacy key has no declared physical unit and must
                    # not become a metre-labelled report value.
                    "progress": 42.0,
                    "target_step_count": 1,
                    "eligible_count": 1,
                    "applied_count": 1,
                    "changed_count": 1,
                    **({"paired_p00": {"status": "matched", "p00_delta_lane_rms_m": 0.0}} if pattern_id == "P01" else {}),
                },
                "records": [
                    {
                        "step": 0,
                        "intervention": {"eligible": True, "applied": True, "changed": True},
                        "post_telemetry": {"target_lane_valid": True, "target_lane_offset_m": 1.0, "progress": 42.0},
                    }
                ],
            },
        )

    result = generate_report(run_dir)
    row = next(row for row in result.summary_rows if row["pattern_id"] == "P01")
    assert row["closed_loop_progress"] is None
    assert row["p00_delta_progress"] is None
    assert "42 m" not in (run_dir / "report.md").read_text(encoding="utf-8")


def test_report_revision_fixture_keeps_scope_counts_and_outcome_states(tmp_path: Path) -> None:
    """Regression fixture for the P00127/P02/no-op/abort report boundary."""

    run_dir = tmp_path / "acceptance-fixture"
    _write_json(
        run_dir / "status.json",
        {"stages": {"offline": {"state": "success"}, "closed_loop": {"state": "success"}}},
    )

    # P02 retains all 127 target steps, with 19 applied, 12 exact changes and
    # a tiny distribution-only effect on the changed rows.  The first twelve
    # rows keep argmax unchanged so JS is tested independently of action flips.
    p02_rows = []
    for step in range(127):
        skipped = step >= 19
        changed = step < 12
        p02_rows.append(
            {
                "episode": 0,
                "step": step,
                "time_s": float(step) * 0.1,
                "actual_input_changed": changed,
                "skipped": skipped,
                "skip_reason": "road mismatch" if skipped else None,
                "action_changed": False,
                "selected_probability_delta": 1e-12 if changed else 0.0,
                "js_divergence": 6e-16 if changed else 0.0,
                "original_values": {"2": 0.0},
                "modified_values": {"2": 1.0 if changed else 0.0},
                "road_segment_id": ["road-a" if step < 19 else "road-b"],
            }
        )
    p02 = _offline_pattern("P02_old", applied=19, changed=12, skipped=108, rows=p02_rows)
    p02["summary"].update(
        {
            "target_step_count": 127,
            "eligible_count": 19,
            "applied_row_count": 19,
            "actual_changed_row_count": 12,
            "no_op_row_count": 7,
            "skipped_row_count": 108,
            "all_timestamps": {
                "count": 127,
                "action_change_rate": 0.0,
                "selected_probability_delta": {"mean": 1e-12},
                "js": {"mean": 6e-16},
            },
            "actual_input_changes_only": {
                "count": 12,
                "action_change_count": 0,
                "action_change_rate": 0.0,
                "selected_probability_delta_pp": {"mean": 0.01},
                "selected_probability_delta": {"mean": 1e-12},
                "selected_probability_delta_abs_pp": {"mean": 1e-10},
                "js": {"mean": 6e-16},
            },
        }
    )

    patterns = [
        {
            "pattern": {"pattern_id": "P00", "indices": [], "method": "identity"},
            "rows": [{"episode": 0, "step": 0, "actual_input_changed": False, "skipped": False}],
            "summary": {
                "target_step_count": 127,
                "eligible_count": 127,
                "applied_row_count": 127,
                "actual_changed_row_count": 0,
                "no_op_row_count": 127,
                "all_timestamps": {"count": 127, "action_change_rate": 0.0, "selected_probability_delta": {"mean": 0.0}, "js": {"mean": 0.0}},
            },
        },
        p02,
        {
            "pattern": {"pattern_id": "group_lidar", "indices": [4], "method": "group"},
            "rows": [{"episode": 0, "step": 0, "actual_input_changed": True, "skipped": False}],
            "summary": {"target_step_count": 1, "eligible_count": 1, "applied_row_count": 1, "actual_changed_row_count": 1, "no_op_row_count": 0, "all_timestamps": {"count": 1, "action_change_rate": 0.0, "selected_probability_delta": {"mean": 0.01}, "js": {"mean": 0.001}}},
        },
        {
            "pattern": {"pattern_id": "P00127", "indices": [4], "method": "fixed"},
            "rows": [{"episode": 0, "step": 0, "actual_input_changed": True, "skipped": False}],
            "summary": {"target_step_count": 1, "eligible_count": 1, "applied_row_count": 1, "actual_changed_row_count": 1, "no_op_row_count": 0, "all_timestamps": {"count": 1, "action_change_rate": 0.1, "selected_probability_delta": {"mean": 0.1}, "js": {"mean": 0.01}}},
        },
    ]
    # A dense individual LiDAR block is deliberately no-op.  It must remain
    # in details and summary.csv while staying out of the ranking figure.
    for index in range(240):
        patterns.append(
            {
                "pattern": {"pattern_id": f"input_{index:03d}", "indices": [index], "method": "reference"},
                "rows": [{"episode": 0, "step": 0, "actual_input_changed": False, "skipped": False}],
                "summary": {"target_step_count": 1, "eligible_count": 1, "applied_row_count": 1, "actual_changed_row_count": 0, "no_op_row_count": 1, "all_timestamps": {"count": 1, "action_change_rate": 0.0, "selected_probability_delta": {"mean": 0.0}, "js": {"mean": 0.0}}},
            }
        )
    _write_json(run_dir / "01_offline" / "result.json", {"patterns": patterns})

    def closed(pattern_id: str, *, metrics: dict[str, object], episode: str = "episode-0") -> None:
        _write_json(
            run_dir / "02_closed_loop" / pattern_id / episode / "trajectory.json",
            {
                "pattern_id": pattern_id,
                "episode_id": episode,
                "metrics": metrics,
                "records": [
                    {
                        "step": 0,
                        "episode_id": episode,
                        "post_time": 6.0,
                        "intervention": {"eligible": True, "applied": True, "changed": True, "skipped": False},
                        "post_telemetry": {"target_lane_offset_m": float(metrics.get("lane_rms_m", 0.0)), "target_lane_valid": True, "road_segment_id": ["road-a"]},
                    }
                ],
            },
        )

    closed(
        "P00",
        metrics={"status": "success", "lane_rms_m": 1.0, "arrived": True, "progress_m": 100.0, "duration_s": 8.0, "terminal_reason": "arrive_dest", "target_step_count": 1, "eligible_count": 1, "applied_count": 1, "changed_count": 1},
    )
    closed(
        "P00127",
        metrics={"status": "success", "lane_rms_m": 2.762, "departure_count": 1, "departure_time_s": 6.0, "first_departure_time_s": 6.0, "arrived": True, "progress_m": 127.0, "duration_s": 8.0, "terminal_reason": "arrive_dest", "target_step_count": 1, "eligible_count": 1, "applied_count": 1, "changed_count": 1, "paired_p00": {"status": "matched", "p00_delta_lane_rms_m": 1.762}},
    )
    closed(
        "Pstop",
        metrics={"status": "success", "lane_rms_m": 0.01, "arrived": False, "progress_m": 3.0, "duration_s": 6.0, "terminal_reason": "stopped", "target_step_count": 1, "eligible_count": 1, "applied_count": 1, "changed_count": 1},
    )
    closed(
        "Pabort",
        metrics={"status": "success", "lane_rms_m": 0.02, "arrived": False, "progress_m": 1.0, "duration_s": 0.1, "terminal_reason": "intervention_abort", "execution_status": "aborted", "target_step_count": 1, "eligible_count": 1, "applied_count": 1, "changed_count": 1},
    )
    closed(
        "Pnum",
        metrics={"status": "success", "assessment_status": "numerical_only", "meaningful_changed_count": 0, "lane_rms_m": 1.0, "arrived": True, "progress_m": 20.0, "terminal_reason": "arrive_dest", "target_step_count": 1, "eligible_count": 1, "applied_count": 1, "changed_count": 1},
    )
    # Unequal episodes: only episode-0 is verified, so the aggregate delta is
    # based on that episode and the missing episode remains counted separately.
    closed("Punequal", metrics={"status": "success", "lane_rms_m": 2.0, "arrived": True, "progress_m": 10.0, "paired_p00": {"status": "matched", "p00_delta_lane_rms_m": 1.0}}, episode="episode-0")
    closed("Punequal", metrics={"status": "success", "lane_rms_m": 9.0, "arrived": False, "progress_m": 1.0}, episode="episode-1")

    result = generate_report(run_dir)
    rows = {row["pattern_id"]: row for row in result.summary_rows}
    assert rows["P02_old"]["offline_target_count"] == 127
    assert rows["P02_old"]["offline_applied_count"] == 19
    assert rows["P02_old"]["offline_changed_count"] == 12
    assert rows["P02_old"]["offline_actual_action_change_rate"] == 0.0
    assert rows["P02_old"]["offline_actual_selected_probability_delta_pp"] == 0.01
    assert rows["P02_old"]["offline_actual_js_divergence"] == 6e-16
    assert rows["P00127"]["closed_loop_arrival_rate"] == 1.0
    assert rows["P00127"]["closed_loop_lane_rms_m"] == 2.762
    assert rows["P00127"]["closed_loop_departure_count"] == 1
    assert rows["P00127"]["closed_loop_departure_time_s"] == 6.0
    assert rows["Pstop"]["closed_loop_lane_rms_m"] == 0.01
    assert rows["Pabort"]["closed_loop_assessment_status"] == "実行失敗／中断"
    assert rows["Pnum"]["closed_loop_meaningful_changed_count"] == 0
    assert rows["Pnum"]["closed_loop_assessment_status"] == "評価対象あり（数値変化のみ）"
    assert rows["Punequal"]["closed_loop_paired_episode_count"] == 1
    assert rows["Punequal"]["closed_loop_paired_missing_count"] == 1
    assert rows["Punequal"]["p00_delta_lane_rms_m"] == 1.0
    assert rows["P00"]["closed_loop_assessment_status"] == "対照"
    assert rows["P00127"]["closed_loop_arrival_count"] == 1
    assert rows["P00127"]["closed_loop_arrival_episode_count"] == 1
    assert rows["Pabort"]["closed_loop_interrupted_episode_count"] == 1

    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "①-Aでは P00" not in report
    assert "Pstop" not in next((line for line in report.splitlines() if "横ずれRMS最小値" in line), "")
    assert "③ IG は未実行です。①-A/①-Bの主結果には影響しません。" in report
    assert "report_ig.svg" not in report
    assert not (run_dir / "report_ig.svg").exists()
    assert "P00127" in report
    assert "group_lidar" in report
    assert "input_000" not in report
    assert "P00対照" in report
    assert "通常観測=127 unique step" in report
    assert "①-A延べ=" in report and "①-B延べ=" in report
    assert "個別LiDAR 240入力" in report
    details = (run_dir / "offline_details.csv").read_text(encoding="utf-8-sig")
    assert "input_239" in details
    assert "original_value" in details and "modified_value" in details
    assert "6e-16" in (run_dir / "summary.csv").read_text(encoding="utf-8-sig")


def test_coverage_series_keeps_episode_and_time_identity() -> None:
    svg = _svg_coverage(
        [
            {"pattern_id": "P02", "episode": "episode-0", "step": 0, "simulation_time_s": 0.0, "applied": True, "changed_exact": True, "road_segment_id": ["road-a"]},
            {"pattern_id": "P02", "episode": "episode-0", "step": 1, "simulation_time_s": 1.0, "applied": False, "changed_exact": False, "skipped": True, "road_segment_id": ["road-b"]},
            {"pattern_id": "P02", "episode": "episode-1", "step": 0, "simulation_time_s": 0.0, "applied": True, "changed_exact": False, "road_segment_id": ["road-c"]},
            {"pattern_id": "P02", "episode": "episode-1", "step": 1, "simulation_time_s": 2.0, "applied": True, "changed_exact": True, "road_segment_id": ["road-d"]},
        ],
        title="coverage",
        mode="closed_loop",
    )
    assert "P02 / episode-0" in svg
    assert "P02 / episode-1" in svg
    assert "time=1.0" in svg and "time=2.0" in svg
    assert "road-a" in svg and "road-d" in svg


def test_findings_uses_saved_variant_classification_and_precision_state() -> None:
    base = {
        "pattern_id": "P02",
        "offline_actual_action_change_rate": 0.0,
        "offline_changed_count": 1,
        "offline_actual_js_divergence": 1e-13,
    }
    saved = {
        **base,
        "variant_classification": "sensor",
        "raw_offline": {"pattern": {"tolerance": 1e-4}},
    }
    saved_findings = _findings([saved], closed=[], ig=[])
    assert any("sensor" in finding for finding in saved_findings)
    assert any("精度・許容幅は保存されていますが" in finding for finding in saved_findings)

    missing_findings = _findings([base], closed=[], ig=[])
    assert any("精度・許容幅が未記録" in finding for finding in missing_findings)


def test_pair_evidence_requires_explicit_success_and_legacy_reference_checks(tmp_path: Path) -> None:
    run_dir = tmp_path / "pair-evidence"
    p00 = _trajectory("P00", paired=False)
    p00["metrics"]["reference_trace_validation"] = {"status": "matched", "mismatch_count": 0}
    _write_json(run_dir / "02_closed_loop" / "P00" / "episode-0" / "trajectory.json", p00)
    # Alignment alone is not a pair proof.
    _write_json(
        run_dir / "02_closed_loop" / "Ptrace" / "episode-0" / "trajectory.json",
        {
            **_trajectory("Ptrace", paired=False),
            "metrics": {
                "status": "success",
                "lane_rms_m": 2.0,
                "arrived": True,
                "terminal_reason": "arrive_dest",
                "p00_trace_alignment": {"status": "matched"},
                "paired_p00": {"p00_delta_lane_rms_m": 1.0},
            },
        },
    )
    # Numeric legacy deltas become valid only with both explicit checks.
    _write_json(
        run_dir / "02_closed_loop" / "Plegacy" / "episode-0" / "trajectory.json",
        {
            **_trajectory("Plegacy", paired=False),
            "metrics": {
                "status": "success",
                "lane_rms_m": 2.0,
                "arrived": True,
                "terminal_reason": "arrive_dest",
                "initial_snapshot": {"initial_match": {"matched": True}},
                "reference_trace_validation": {"status": "matched", "mismatch_count": 0},
                "paired_p00": {"p00_delta_lane_rms_m": 1.0},
                "paired_p00_verified": False,
            },
        },
    )
    # A legacy intervention episode may rely on the P00-side validation when
    # its own trajectory omitted that field.
    _write_json(
        run_dir / "02_closed_loop" / "Pcross" / "episode-0" / "trajectory.json",
        {
            **_trajectory("Pcross", paired=False),
            "metrics": {
                "status": "success",
                "lane_rms_m": 2.0,
                "arrived": True,
                "terminal_reason": "arrive_dest",
                "initial_snapshot": {"initial_match": {"matched": True}},
                "paired_p00": {"p00_delta_lane_rms_m": 1.5},
            },
        },
    )
    result = generate_report(run_dir)
    rows = {row["pattern_id"]: row for row in result.summary_rows}
    assert rows["Ptrace"]["closed_loop_paired_p00_verified"] is None
    assert rows["Ptrace"]["p00_delta_lane_rms_m"] is None
    assert rows["Plegacy"]["closed_loop_paired_p00_verified"] is False
    assert rows["Plegacy"]["p00_delta_lane_rms_m"] is None
    assert rows["Pcross"]["closed_loop_paired_p00_verified"] is True
    assert rows["Pcross"]["p00_delta_lane_rms_m"] == 1.5


def test_mixed_natural_and_abort_uses_natural_metrics_and_exposes_abort(tmp_path: Path) -> None:
    run_dir = tmp_path / "mixed-abort"
    _write_json(run_dir / "02_closed_loop" / "P02" / "episode-0" / "trajectory.json", _trajectory("P02", paired=True))
    aborted = _trajectory("P02", paired=False)
    aborted["episode_id"] = "episode-1"
    aborted["metrics"] = {
        "status": "success",
        "execution_status": "aborted",
        "terminal_reason": "intervention_abort",
        "lane_rms_m": 99.0,
        "arrived": False,
        "progress_m": 1.0,
        "paired_p00": {"status": "matched", "p00_delta_lane_rms_m": 99.0},
        "paired_p00_verified": True,
    }
    for record in aborted["records"]:
        record["episode_id"] = "episode-1"
        record["env_step_called"] = False
        record.pop("post_telemetry", None)
    _write_json(run_dir / "02_closed_loop" / "P02" / "episode-1" / "trajectory.json", aborted)
    rows = {row["pattern_id"]: row for row in generate_report(run_dir).summary_rows}
    row = rows["P02"]
    assert row["closed_loop_natural_episode_count"] == 1
    assert row["closed_loop_interrupted_episode_count"] == 1
    assert row["closed_loop_paired_episode_count"] == 1
    assert row["closed_loop_paired_missing_count"] == 0
    assert row["p00_delta_lane_rms_m"] == 0.25
    assert any("pairing excluded" in reason for reason in row["closed_loop_paired_missing_reasons"])
    assert row["closed_loop_lane_rms_m"] == 1.5
    assert row["closed_loop_arrival_rate"] == 0.0
    assert "中断" in row["closed_loop_status"]
    assert "中断" in row["closed_loop_assessment_status"]
    assert row["closed_loop_poststep_count"] == 6
    assert not any("横ずれRMS範囲" in text for text in _findings([row], closed=[row], ig=[]))


def test_abort_attempt_without_env_step_has_no_poststep_metrics() -> None:
    aggregate = _aggregate_closed_steps(
        [
            {
                "step": 0,
                "env_step_called": False,
                "intervention": {"applied": True, "changed": True},
                "post_telemetry": None,
                "target_lane_offset_m": 12.0,
            }
        ]
    )
    assert aggregate["closed_loop_poststep_count"] == 0
    assert aggregate.get("closed_loop_lane_rms_m") is None


def test_missing_post_telemetry_does_not_fall_back_to_pre_or_partial_aggregate() -> None:
    row = {
        "step": 0,
        "env_step_called": True,
        "pre_telemetry": {
            "target_lane_offset_m": 9.0,
            "speed_mps": 9.0,
            "simulation_time_s": 0.0,
        },
        "post_telemetry": {
            "target_lane_offset_m": None,
            "speed_mps": None,
            "simulation_time_s": None,
        },
    }
    aggregate = _aggregate_closed_steps([row])
    assert aggregate["closed_loop_poststep_count"] == 1
    assert aggregate.get("closed_loop_lane_rms_m") is None
    assert aggregate.get("closed_loop_speed_mean_mps") is None
    assert aggregate.get("closed_loop_duration_s") is None

    normalised = _normalise_closed(
        [
            {
                "pattern_id": "Ppartial",
                "episode_id": "episode-0",
                "metrics": {
                    "status": "success",
                    "lane_rms_m": None,
                    "duration_s": None,
                    "poststep_count": None,
                    "arrived": None,
                },
                "records": [row],
            }
        ]
    )
    item = normalised[0]
    assert item["closed_loop_lane_rms_m"] is None
    assert item["closed_loop_duration_s"] is None
    assert item["closed_loop_poststep_count"] is None
    assert item["closed_loop_arrived"] is None


def test_report_surfaces_saved_video_omission_reasons_compactly(tmp_path: Path) -> None:
    run_dir = tmp_path / "video-reasons"
    _write_json(
        run_dir / "manifest.json",
        {"run_id": "video-reasons", "video": {"enabled": True, "patterns": ["P00", "Pframe"]}},
    )
    _write_json(
        run_dir / "02_closed_loop" / "summary.json",
        {
            "patterns": {
                "P00": {"video": {"status": "not_generated", "enabled": False, "reason": "video_disabled"}},
                "Pnotselected": {"video": {"status": "not_generated", "enabled": False, "reason": "pattern_not_selected_by_video_patterns"}},
                "Pframe": {"video": {"status": "partial", "enabled": True, "frame_count": 0, "reason": "frame_unavailable"}},
            }
        },
    )

    result = generate_report(run_dir)
    markdown = (result.output_dir / "report.md").read_text(encoding="utf-8")
    html = (result.output_dir / "report.html").read_text(encoding="utf-8")
    assert "動画未生成理由" in markdown
    assert "video_disabled" in markdown
    assert "pattern_not_selected_by_video_patterns" in markdown
    assert "frame_unavailable" in markdown
    assert "video_disabled" in html
    assert "pattern_not_selected_by_video_patterns" in html
    assert "frame_unavailable" in html
    # One reason line per distinct saved code, regardless of summary nesting.
    assert markdown.count("動画未生成理由:") == 3


def test_b_intervention_unknown_and_record_count_only_stay_na() -> None:
    unknown = _closed_intervention_stats([{"intervention": {}}])
    assert unknown["target"] == 1
    assert unknown["applied"] is None
    assert unknown["changed"] is None
    applied_without_change = _closed_intervention_stats([{"intervention": {"applied": True}}])
    assert applied_without_change["applied"] == 1
    assert applied_without_change["changed"] is None
    identity_control = _closed_intervention_stats(
        [{"intervention": {"method": "identity", "applied_count": 0, "changed": False, "skipped": False}}]
    )
    assert identity_control["applied"] == 1
    assert identity_control["changed"] == 0
    assert identity_control["noop"] == 1


def test_detail_keeps_delta_units_and_meaning_fields() -> None:
    records = _offline_detail_records(
        [
            {
                "pattern": {"pattern_id": "P02"},
                "rows": [
                    {
                        "episode": 0,
                        "step": 1,
                        "selected_probability_delta": 0.0001,
                        "selected_probability_delta_pp": 0.01,
                        "delta_values": {"2": 0.0001},
                        "delta_abs_values": {"2": 0.0001},
                        "tolerance": 1e-7,
                        "clipped_indices": [2],
                        "clip_count": 1,
                        "meaningful_changed": True,
                        "meaningful_changed_count": 1,
                    }
                ],
            }
        ]
    )
    row = records[0]
    assert row["selected_probability_delta"] == 0.0001
    assert row["selected_probability_delta_pp"] == 0.01
    assert row["tolerance"] == 1e-7
    assert row["clip_count"] == 1
    assert row["meaningful_changed"] is True


def test_bars_exclude_numeric_only_and_do_not_call_sample_top20() -> None:
    svg = _svg_bars(
        [
            {"pattern_id": "P02", "offline_changed_count": 2, "offline_actual_action_change_rate": 0.8, "offline_assessment_status": "評価対象あり"},
            {"pattern_id": "P03", "offline_changed_count": 2, "offline_actual_action_change_rate": 0.9, "offline_assessment_status": "評価対象あり（数値変化のみ）"},
            {"pattern_id": "input_000", "offline_changed_count": 2, "offline_actual_action_change_rate": 0.1, "offline_assessment_status": "評価対象あり"},
        ]
    )
    assert "P02" in svg
    assert "P03" not in svg
    assert "top 20" not in svg.lower()
    assert "ID order" in svg
