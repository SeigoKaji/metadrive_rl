"""保存結果だけから日本語レポートと SVG を再生成できることを確認する。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from input_attribution.pack import create_portable_zip, portable_relative_paths
from input_attribution.reporting import (
    _collect,
    _normalise_closed,
    _normalise_offline,
    generate_report,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_report_reads_runtime_fixture_and_keeps_p00_pairing(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_json(run_dir / "manifest.json", {"run_id": "fixture", "status": "success"})
    _write_json(run_dir / "status.json", {"stages": {"offline": {"state": "success"}, "closed_loop": {"state": "success"}}})
    _write_json(
        run_dir / "01_offline" / "result.json",
        {
            "schema_dimension": 3,
            "patterns": [
                {
                    "pattern": {"pattern_id": "P00", "indices": [], "method": "identity", "description": "変更なし"},
                    "summary": {"requested_dimension_count": 0, "all_timestamps": {"count": 2, "action_change_rate": 0.0, "selected_probability_delta": {"mean": 0.0}, "js": {"mean": 0.0}}},
                },
                {
                    "pattern": {"pattern_id": "P01", "indices": [1, 2], "method": "fixed", "description": "目標レーン中心相当値"},
                    "summary": {"requested_dimension_count": 2, "actual_changed_row_count": 1, "no_op_row_count": 1, "applied_row_count": 2, "all_timestamps": {"count": 2, "action_change_rate": 0.5, "selected_probability_delta": {"mean": -0.2}, "js": {"mean": 0.1}}},
                },
            ],
        },
    )
    for pattern, errors, progress in (("P00", (0.1, 0.2), 10.0), ("P01", (0.4, 0.6), 8.0)):
        _write_json(
            run_dir / "02_closed_loop" / pattern / "episode-0" / "trajectory.json",
            {
                "pattern_id": pattern,
                "terminal_reason": "arrive_dest" if pattern == "P00" else "out_of_road",
                "records": [
                    {"step": index, "post_telemetry": {"target_lane_lateral_error_m": error, "speed_mps": 2.0, "route_progress_m": progress * (index + 1), "simulation_time_s": 0.1 * (index + 1)}}
                    for index, error in enumerate(errors)
                ],
            },
        )
    _write_json(
        run_dir / "02_closed_loop" / "summary.json",
        {"patterns": {"P00": {"episodes": 1, "records": 2}, "P01": {"episodes": 1, "records": 2}}},
    )

    result = generate_report(run_dir)
    assert result.status == "success"
    assert {row["pattern_id"] for row in result.summary_rows} == {"P00", "P01"}
    rows = {row["pattern_id"]: row for row in result.summary_rows}
    assert rows["P01"]["offline_applied_count"] == 2
    assert rows["P01"]["offline_changed_count"] == 1
    assert rows["P01"]["offline_noop_count"] == 1
    assert rows["P01"]["offline_action_change_rate"] == 0.5
    assert rows["P00"]["closed_loop_lane_rms_m"] is not None
    assert rows["P01"]["p00_delta_lane_rms_m"] is not None
    assert rows["P01"]["closed_loop_termination"] == "out_of_road"
    assert "N/A" in (run_dir / "report.md").read_text(encoding="utf-8") or "未実行" in (run_dir / "report.md").read_text(encoding="utf-8")
    html = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "<svg" in html
    assert "外部" not in html
    assert (run_dir / "report_closed_loop_speed.svg").is_file()
    assert (run_dir / "report_closed_loop_steering.svg").is_file()
    assert "post_time" in html and "pre_time" in html
    with (run_dir / "summary.csv").open("rb") as handle:
        assert handle.read(3) == b"\xef\xbb\xbf"
    with (run_dir / "summary.csv").open(newline="", encoding="utf-8-sig") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 2
    first_report = (run_dir / "report.md").read_bytes()
    second = generate_report(run_dir)
    assert second.output_dir.parent == run_dir / "reports"
    assert (run_dir / "report.md").read_bytes() == first_report
    assert (second.output_dir / "report.html").is_file()


def test_report_without_optional_ig_marks_it_as_unexecuted(tmp_path: Path) -> None:
    run_dir = tmp_path / "empty-run"
    _write_json(run_dir / "manifest.json", {"run_id": "empty"})
    result = generate_report(run_dir)
    assert result.summary_rows == ()
    text = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "③ IG は未実行" in text
    assert "report_ig.svg" in text
    assert "未実行または結果未記録" in text


def test_report_accepts_column_like_npz_records(tmp_path: Path) -> None:
    run_dir = tmp_path / "npz-run"
    _write_json(run_dir / "manifest.json", {"run_id": "npz"})
    (run_dir / "01_offline").mkdir(parents=True)
    np.savez(
        run_dir / "01_offline" / "steps.npz",
        pattern_id=np.asarray(["P01", "P01"]),
        action_changed=np.asarray([False, True]),
        selected_probability_delta=np.asarray([0.0, -0.1]),
        js_divergence=np.asarray([0.0, 0.05]),
        actual_input_changed=np.asarray([False, True]),
    )
    result = generate_report(run_dir)
    row = result.summary_rows[0]
    assert row["pattern_id"] == "P01"
    assert row["offline_applied_count"] == 2
    assert row["offline_changed_count"] == 1
    assert row["offline_noop_count"] == 1


def test_report_uses_status_selected_analysis_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "analysis-ids"
    _write_json(
        run_dir / "status.json",
        {"stages": {"offline": {"state": "success", "analysis_id": "new", "relative_dir": "01_offline/new"}}},
    )
    for analysis_id, pattern_id in (("old", "P_OLD"), ("new", "P_NEW")):
        _write_json(
            run_dir / "01_offline" / analysis_id / "result.json",
            {"patterns": [{"pattern": {"pattern_id": pattern_id, "indices": [1]}, "summary": {"all_timestamps": {"count": 1, "action_change_rate": 0.0}}}]},
        )
    result = generate_report(run_dir)
    assert [row["pattern_id"] for row in result.summary_rows] == ["P_NEW"]


def test_report_uses_canonical_patterns_and_not_statistics_or_reference_containers(tmp_path: Path) -> None:
    """Canonical result summaries must define the pattern row set and pp metric."""

    run_dir = tmp_path / "canonical-shape"
    _write_json(
        run_dir / "status.json",
        {
            "stages": {
                "offline": {"state": "success", "relative_dir": "01_offline/analysis"},
                "closed_loop": {"state": "success", "relative_dir": "02_closed_loop/analysis"},
            }
        },
    )
    canonical_ids = [
        "P00",
        "P01_road_boundaries_reference",
        "P02_heading_reference",
        "P03_speed_history_reference",
        "P04_current_lane_lateral_reference",
        "P05_navigation_reference",
        "P06_lidar_all_no_detection",
        *(f"input_{index:03d}" for index in range(265)),
    ]
    assert len(canonical_ids) == 272
    patterns = []
    for pattern_id in canonical_ids:
        summary = {
            "requested_dimension_count": 1,
            "applied_row_count": 1,
            "actual_changed_row_count": 0,
            "no_op_row_count": 1,
            "all_timestamps": {
                "count": 1,
                "action_change_rate": 0.0,
                "selected_probability_delta": {"mean": 0.0},
                "js": {"mean": 0.0},
            },
        }
        if pattern_id == "P00":
            summary["requested_dimension_count"] = 0
        if pattern_id == "P01_road_boundaries_reference":
            summary.update(
                {
                    "applied_row_count": 40,
                    "actual_changed_row_count": 39,
                    "no_op_row_count": 1,
                    "all_timestamps": {
                        "count": 40,
                        "action_change_rate": 0.4,
                        # This is the canonical raw probability fraction.
                        "selected_probability_delta": {"mean": -0.15523287991746582},
                        "js": {"mean": 0.0435039160018241},
                    },
                    # Diagnostic rows contain default pp zeros; they must not
                    # override the authoritative all-timestamps summary.
                    "episode_summaries": {
                        "0": {
                            "all_timestamps": {
                                "action_change_rate": 0.4,
                                "selected_probability_delta": {"mean": -0.15523287991746582},
                            }
                        }
                    },
                }
            )
        patterns.append(
            {
                "pattern": {"pattern_id": pattern_id, "indices": [] if pattern_id == "P00" else [0]},
                "rows": [{"selected_probability_delta_pp": 0.0, "js_divergence": 0.0}],
                "summary": summary,
            }
        )
    _write_json(run_dir / "01_offline" / "analysis" / "result.json", {"patterns": patterns, "schema_dimension": 259})

    statistics = run_dir / "01_offline" / "analysis" / "input_statistics.csv"
    statistics.parent.mkdir(parents=True, exist_ok=True)
    with statistics.open("w", encoding="utf-8", newline="") as handle:
        handle.write("index,id,name_ja,min,max,mean,std,constant\n")
        for index in range(259):
            handle.write(f"{index},semantic_{index},入力{index},0,1,0.5,0.1,False\n")

    closed_patterns = {
        pattern_id: {"metrics": {"status": "success", "valid_count": 1, "lane_rms_m": 0.1}}
        for pattern_id in (
            "P00",
            "P01_road_boundaries_reference",
            "P02_heading_reference",
            "P04_current_lane_lateral_reference",
            "P05_navigation_reference",
            "P06_lidar_all_no_detection",
        )
    }
    _write_json(run_dir / "02_closed_loop" / "analysis" / "summary.json", {"patterns": closed_patterns})
    _write_json(
        run_dir / "02_closed_loop" / "analysis" / "P00" / "episode-0" / "video.json",
        {"fps": 10.0, "frame_count": 1, "frame_map": [{"step": 0, "simulation_time": 0.1}]},
    )
    # Structural P00 trace container has records but no closed-loop metric;
    # it must not become an ``unknown`` summary row.
    _write_json(
        run_dir / "02_closed_loop" / "analysis" / "P00" / "episode-0" / "reference.json",
        {"records": [{"step": 0, "post_telemetry": {"target_lane_valid": True}}]},
    )

    metadata, offline_raw, closed_raw, _ig, _artifacts = _collect(run_dir)
    offline = _normalise_offline(offline_raw, metadata.get("_pattern_labels"))
    closed = _normalise_closed(closed_raw, metadata.get("_pattern_labels"))
    assert len(offline) == 272
    assert {row["pattern_id"] for row in offline} == set(canonical_ids)
    p01 = next(row for row in offline if row["pattern_id"] == "P01_road_boundaries_reference")
    assert abs(p01["offline_selected_probability_delta_pp"] - (-15.523287991746582)) < 1e-12
    assert len(closed) == 6
    assert {row["pattern_id"] for row in closed} == set(closed_patterns)
    assert all(row["pattern_id"] != "unknown" for row in closed)


def test_report_preserves_runtime_skip_nulls_names_and_paired_na(tmp_path: Path) -> None:
    """Regression fixture for the canonical result/summary artifact contract."""

    run_dir = tmp_path / "runtime-contract"
    _write_json(
        run_dir / "status.json",
        {
            "stages": {
                "offline": {"state": "success", "relative_dir": "01_offline/new"},
                "closed_loop": {"state": "success", "relative_dir": "02_closed_loop/new"},
            }
        },
    )
    _write_json(
        run_dir / "patterns.json",
        [
            {"pattern_id": "P00", "indices": [], "names_ja": [], "method": "identity"},
            {"pattern_id": "input_000", "indices": [0], "names_ja": ["左道路端までの距離"], "method": "reference"},
        ],
    )
    _write_json(
        run_dir / "01_offline" / "new" / "result.json",
        {
            "patterns": [
                {
                    "pattern": {"pattern_id": "P00", "indices": [], "method": "identity"},
                    "summary": {
                        "applied_row_count": 4,
                        "actual_changed_row_count": 0,
                        "no_op_row_count": 4,
                        "skipped_row_count": 0,
                        "all_timestamps": {
                            "count": 4,
                            "action_change_rate": 0.0,
                            "selected_probability_delta": {"mean": 0.0},
                            "js": {"mean": 0.0},
                        },
                        "episode_summaries": {"0": {"all_timestamps": {"action_change_rate": 0.0}}},
                    },
                },
                {
                    "pattern": {"pattern_id": "input_000", "indices": [0], "method": "reference"},
                    "rows": [
                        {"pattern_id": "input_000", "skipped": True, "skip_reason": "compatibility context is missing 'road_segment_id'", "selected_probability_delta": 0.0, "js_divergence": 0.0}
                    ],
                    "summary": {
                        "applied_row_count": 0,
                        "actual_changed_row_count": 0,
                        "no_op_row_count": 0,
                        "skipped_row_count": 4,
                        "valid_row_count": 0,
                        "skip_reasons": ["compatibility context is missing 'road_segment_id'"],
                        "all_timestamps": {"count": 0, "action_change_rate": None, "selected_probability_delta": {"mean": None}, "js": {"mean": None}},
                    },
                },
            ]
        },
    )
    _write_json(
        run_dir / "02_closed_loop" / "new" / "summary.json",
        {
            "patterns": {
                "P00": {"metrics": {"status": "not_available", "valid_count": 0, "valid_rate": None, "lane_rms_m": None, "progress_m": None}, "records": 1}
            }
        },
    )

    result = generate_report(run_dir)
    rows = {row["pattern_id"]: row for row in result.summary_rows}
    assert rows["P00"]["pattern_name"] == "変更なし"
    assert rows["P00"]["offline_applied_count"] == 4
    assert rows["P00"]["offline_noop_count"] == 4
    assert rows["P00"]["offline_episode_mean_action_change_rate"] == 0.0
    assert rows["P00"]["p00_delta_lane_rms_m"] is None
    assert rows["input_000"]["pattern_name"] == "左道路端までの距離"
    assert rows["input_000"]["offline_status"] == "未実行（skip）"
    assert rows["input_000"]["offline_reason"] == "道路区間情報がない4件を除外。0/4件で比較"
    assert rows["input_000"]["offline_selected_probability_delta_pp"] is None
    assert rows["P00"]["closed_loop_status"] == "not_available"
    assert "valid_count=0" in rows["P00"]["closed_loop_reason"]


def test_report_formats_canonical_partial_skips_and_keeps_mixed_reasons(tmp_path: Path) -> None:
    run_dir = tmp_path / "partial-skip"
    _write_json(
        run_dir / "status.json",
        {"stages": {"offline": {"state": "success", "relative_dir": "01_offline/new"}}},
    )
    road_reasons = [
        "compatibility context 'road_segment_id' differs: reference=['>>>', '1C0_0_'], observation=['1C0_0_', '1C0_1_']",
        "compatibility context 'road_segment_id' differs: reference=['>>>', '1C0_0_'], observation=['>', '>>']",
    ]
    patterns = []
    for pattern_id, include_valid, reasons in (
        ("partial_with_valid", True, road_reasons),
        ("partial_without_valid", False, [road_reasons[0], "unexpected policy failure"]),
        ("complete", True, []),
        ("total_skip", True, ["compatibility context is missing 'road_segment_id'"]),
    ):
        summary = {
            "applied_row_count": 2 if pattern_id.startswith("partial") or pattern_id == "complete" else 0,
            "actual_changed_row_count": 1 if pattern_id.startswith("partial") else 0,
            "no_op_row_count": 1 if pattern_id.startswith("partial") or pattern_id == "complete" else 0,
            "skipped_row_count": 2 if pattern_id.startswith("partial") or pattern_id == "total_skip" else 0,
            "all_timestamps": {
                "count": 2 if pattern_id.startswith("partial") or pattern_id == "complete" else 0,
                "action_change_rate": 0.5 if pattern_id.startswith("partial") else 0.0 if pattern_id == "complete" else None,
                "selected_probability_delta": {"mean": -0.1 if pattern_id.startswith("partial") else 0.0 if pattern_id == "complete" else None},
                "js": {"mean": 0.01 if pattern_id.startswith("partial") else 0.0 if pattern_id == "complete" else None},
            },
            "skip_reasons": reasons,
        }
        if include_valid:
            summary["valid_row_count"] = summary["applied_row_count"]
        patterns.append({"pattern": {"pattern_id": pattern_id, "indices": [0]}, "summary": summary})
    patterns.append(
        {
            "pattern": {"pattern_id": "partial_failed", "indices": [0]},
            "status": "failed",
            "summary": {
                "applied_row_count": 1,
                "actual_changed_row_count": 1,
                "no_op_row_count": 0,
                "skipped_row_count": 1,
                "valid_row_count": 1,
                "all_timestamps": {
                    "count": 1,
                    "action_change_rate": 1.0,
                    "selected_probability_delta": {"mean": 0.1},
                    "js": {"mean": 0.02},
                },
                "skip_reasons": ["unexpected policy failure"],
            },
        }
    )
    _write_json(run_dir / "01_offline" / "new" / "result.json", {"patterns": patterns})

    result = generate_report(run_dir)
    rows = {row["pattern_id"]: row for row in result.summary_rows}
    expected = "参照観測と道路区間が異なる2件を除外。2/4件で比較"
    assert rows["partial_with_valid"]["offline_status"] == "完了（一部スキップ）"
    assert rows["partial_without_valid"]["offline_status"] == "完了（一部スキップ）"
    assert rows["partial_with_valid"]["offline_reason"].startswith(expected)
    assert rows["partial_without_valid"]["offline_reason"].startswith("参照観測と道路区間が異なる。2件を除外。2/4件で比較。")
    assert "unexpected policy failure" in rows["partial_without_valid"]["offline_reason"]
    assert rows["partial_with_valid"]["offline_action_change_rate"] == 0.5
    assert rows["partial_without_valid"]["offline_action_change_rate"] == 0.5
    assert rows["partial_with_valid"]["offline_selected_probability_delta_pp"] == -10.0
    assert rows["partial_without_valid"]["offline_selected_probability_delta_pp"] == -10.0
    assert rows["partial_with_valid"]["offline_js_divergence"] == 0.01
    assert rows["partial_without_valid"]["offline_js_divergence"] == 0.01
    assert rows["complete"]["offline_status"] == "完了"
    assert rows["complete"]["offline_reason"] == ""
    assert rows["partial_failed"]["offline_status"] == "failed"
    assert rows["total_skip"]["offline_status"] == "未実行（skip）"
    assert rows["total_skip"]["offline_selected_probability_delta_pp"] is None
    assert rows["total_skip"]["offline_reason"] == "道路区間情報がない2件を除外。0/2件で比較"
    raw_summary = rows["partial_without_valid"]["raw_offline"]["summary"]
    assert raw_summary["skip_reasons"][-1] == "unexpected policy failure"

    markdown = (run_dir / "report.md").read_text(encoding="utf-8")
    html = (run_dir / "report.html").read_text(encoding="utf-8")
    assert expected in markdown
    assert "スキップ・欠測理由" in markdown
    assert "未実行/欠測理由" not in markdown
    assert expected in html
    with (run_dir / "summary.csv").open(newline="", encoding="utf-8-sig") as handle:
        csv_rows = {row["pattern_id"]: row for row in csv.DictReader(handle)}
    assert csv_rows["partial_without_valid"]["offline_status"] == "完了（一部スキップ）"
    assert "2件を除外。2/4件で比較" in csv_rows["partial_without_valid"]["offline_reason"]


def test_report_formats_per_step_partial_skips_consistently(tmp_path: Path) -> None:
    run_dir = tmp_path / "per-step-partial-skip"
    steps_path = run_dir / "01_offline" / "steps.csv"
    steps_path.parent.mkdir(parents=True, exist_ok=True)
    steps_path.write_text(
        "pattern_id,episode,step,actual_input_changed,skipped,skip_reason,action_changed,selected_probability_delta,js_divergence\n"
        "P01,0,0,True,False,,True,-0.1,0.01\n"
        "P01,0,1,False,False,,False,0.0,0.0\n"
        "P01,0,2,False,True,\"compatibility context 'road_segment_id' differs: reference=['a'], observation=['b']\",False,0.0,0.0\n"
        "P01,0,3,False,True,\"compatibility context 'road_segment_id' differs: reference=['a'], observation=['c']\",False,0.0,0.0\n",
        encoding="utf-8",
    )

    result = generate_report(run_dir)
    row = next(row for row in result.summary_rows if row["pattern_id"] == "P01")
    assert row["offline_status"] == "完了（一部スキップ）"
    assert row["offline_applied_count"] == 2
    assert row["offline_changed_count"] == 1
    assert row["offline_noop_count"] == 1
    assert row["offline_reason"] == "参照観測と道路区間が異なる2件を除外。2/4件で比較"


def test_report_aggregates_distinct_closed_loop_episodes(tmp_path: Path) -> None:
    run_dir = tmp_path / "multi-episode"
    _write_json(
        run_dir / "status.json",
        {"stages": {"closed_loop": {"state": "success", "relative_dir": "02_closed_loop/new"}}},
    )
    for pattern, values, arrivals in (
        ("P00", (1.0, 2.0), (False, False)),
        ("P02", (3.0, 5.0), (True, False)),
    ):
        for episode, (rms, arrived) in enumerate(zip(values, arrivals, strict=True)):
            paired = {"p00_delta_lane_rms_m": rms - values[episode] if pattern == "P00" else rms - (1.0 + episode), "p00_delta_progress_m": 1.0}
            _write_json(
                run_dir / "02_closed_loop" / "new" / pattern / f"episode-{episode}" / "trajectory.json",
                {
                    "pattern_id": pattern,
                    "episode_id": f"episode-{episode}",
                    "metrics": {
                        "status": "success",
                        "lane_rms_m": rms,
                        "lane_max_abs_m": rms + 0.5,
                        "arrived": arrived,
                        "progress_m": float(episode + 1),
                        "paired_p00": paired,
                    },
                    "records": [{"step": 0, "post_time": 0.1, "post_telemetry": {"target_lane_offset_m": rms}}],
                },
            )

    result = generate_report(run_dir)
    rows = {row["pattern_id"]: row for row in result.summary_rows}
    assert rows["P02"]["closed_loop_episode_count"] == 2
    assert rows["P02"]["closed_loop_lane_rms_m"] == 4.0
    assert rows["P02"]["closed_loop_lane_max_abs_m"] == 5.5
    assert rows["P02"]["closed_loop_arrival_rate"] == 0.5
    assert rows["P02"]["p00_delta_lane_rms_m"] == 2.5
    assert "episode平均" in (run_dir / "report.md").read_text(encoding="utf-8")


def test_report_keeps_canonical_nulls_and_unverified_pair_na(tmp_path: Path) -> None:
    run_dir = tmp_path / "canonical-null"
    _write_json(
        run_dir / "status.json",
        {"stages": {"closed_loop": {"state": "success", "relative_dir": "02_closed_loop/new"}}},
    )
    for pattern, metrics, offset, valid in (
        ("P00", {"status": "not_available", "lane_rms_m": None, "valid_count": 0}, 9.0, False),
        ("P01", {"status": "success", "lane_rms_m": 2.0, "valid_count": 1, "paired_p00": {"p00_delta_lane_rms_m": 1.0, "status": "unverified"}}, 2.0, True),
    ):
        _write_json(
            run_dir / "02_closed_loop" / "new" / pattern / "episode-0" / "trajectory.json",
            {
                "pattern_id": pattern,
                "episode_id": "episode-0",
                "metrics": metrics,
                "records": [{"step": 0, "post_time": 0.1, "post_telemetry": {"target_lane_offset_m": offset, "target_lane_valid": valid}}],
            },
        )

    result = generate_report(run_dir)
    rows = {row["pattern_id"]: row for row in result.summary_rows}
    assert rows["P00"]["closed_loop_lane_rms_m"] is None
    assert rows["P01"]["closed_loop_paired_p00_verified"] is False
    assert rows["P01"]["p00_delta_lane_rms_m"] is None
    assert "P00 pairing was not verified" in rows["P01"]["closed_loop_reason"]


def test_report_prioritises_clips_and_frame_maps_over_png_lists(tmp_path: Path) -> None:
    run_dir = tmp_path / "media-run"
    _write_json(run_dir / "manifest.json", {"run_id": "media"})
    video_root = run_dir / "00_reference"
    media_dir = video_root / "video"
    frame_dir = media_dir / "episode-0"
    frame_dir.mkdir(parents=True)
    (media_dir / "trajectory.gif").write_bytes(b"GIF89a")
    for index in range(4):
        (frame_dir / f"frame-{index:06d}.png").write_bytes(b"PNG")
    _write_json(
        video_root / "video.json",
        {"gif": "video/trajectory.gif", "fps": 10.0, "frame_count": 100, "frame_map": []},
    )

    result = generate_report(run_dir)
    html = (result.output_dir / "report.html").read_text(encoding="utf-8")
    assert "trajectory.gif" in html
    assert "video.json" in html
    assert "frame-000000.png" not in html


def test_report_surfaces_nonconverged_ig_completeness(tmp_path: Path) -> None:
    run_dir = tmp_path / "ig-quality"
    _write_json(
        run_dir / "status.json",
        {"stages": {"ig": {"state": "success", "relative_dir": "03_ig/new"}}},
    )
    _write_json(
        run_dir / "03_ig" / "new" / "result.json",
        {
            "target_action": 2,
            "attributions": [0.2, -0.1],
            "completeness_delta": 0.01,
            "absolute_completeness_error": 0.01,
            "completeness_tolerance": 1e-4,
            "retry_count": 1,
            "completeness_status": "nonconverged",
            "warnings": ["retry limit reached"],
        },
    )

    result = generate_report(run_dir)
    assert result.summary_rows == ()
    markdown = (run_dir / "report.md").read_text(encoding="utf-8")
    html = (run_dir / "report.html").read_text(encoding="utf-8")
    manifest = json.loads((run_dir / "report_manifest.json").read_text(encoding="utf-8"))
    assert "completeness非収束" in markdown
    assert "retry limit reached" in markdown
    assert "completeness非収束" in html
    assert manifest["sections"]["integrated_gradients"]["quality"]["status"] == "警告（completeness非収束）"


def test_report_keeps_ig_context_for_each_step_and_baseline(tmp_path: Path) -> None:
    run_dir = tmp_path / "ig-context"
    _write_json(
        run_dir / "status.json",
        {"stages": {"ig": {"state": "success", "relative_dir": "03_ig/analysis"}}},
    )
    results = []
    for episode, step, action, baseline_id, fx, fb, attributions in (
        (0, 75, 8, "episode-0:70", 0.8, 0.2, [0.6, -0.1, 0.0]),
        (1, 80, 7, "episode-1:70", 0.7, 0.1, [0.4, -0.2, 0.1]),
    ):
        output_difference = fx - fb
        attribution_sum = sum(attributions)
        results.append(
            {
                "episode": episode,
                "step": step,
                "target_action": action,
                "baseline_id": baseline_id,
                "observation_score": fx,
                "baseline_score": fb,
                "attributions": attributions,
                "absolute_attributions": [abs(value) for value in attributions],
                "attribution_sum": attribution_sum,
                "output_difference": output_difference,
                "completeness_delta": attribution_sum - output_difference,
                "integration_points": 64,
                "completeness_status": "converged",
                "tolerance": 1e-4,
                "retry_count": 0,
            }
        )
    _write_json(run_dir / "03_ig" / "analysis" / "result.json", {"analysis_id": "ig-context-run", "results": results})

    result = generate_report(run_dir)
    markdown = (result.output_dir / "report.md").read_text(encoding="utf-8")
    html = (result.output_dir / "report.html").read_text(encoding="utf-8")
    svg = (result.output_dir / "report_ig.svg").read_text(encoding="utf-8")
    manifest = json.loads((result.output_dir / "report_manifest.json").read_text(encoding="utf-8"))
    contexts = manifest["sections"]["integrated_gradients"]["contexts"]
    assert len(contexts) == 2
    assert {context["ig_step"] for context in contexts} == {75, 80}
    assert {context["ig_target_action"] for context in contexts} == {8, 7}
    assert "ig-context-run" in markdown and "baseline ID" in markdown
    assert "F_x" in html and "F_b" in html and "output difference" in html and "residual" in html
    assert "step=75" in svg and "step=80" in svg
    assert "baseline=episode-0:70" in svg and "baseline=episode-1:70" in svg


def test_optional_ig_failure_does_not_fail_primary_report(tmp_path: Path) -> None:
    run_dir = tmp_path / "ig-failure"
    _write_json(
        run_dir / "status.json",
        {
            "status": "failed",
            "stages": {
                "offline": {"state": "success"},
                "ig": {"state": "failed", "reason": "Captum is unavailable"},
            },
        },
    )
    _write_json(
        run_dir / "01_offline" / "result.json",
        {"patterns": [{"pattern": {"pattern_id": "P00"}, "summary": {"all_timestamps": {"count": 1, "action_change_rate": 0.0}}}]},
    )
    result = generate_report(run_dir)
    assert result.status == "完了（③ IGは任意）"
    markdown = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "Captum is unavailable" in markdown


def test_portable_zip_is_deterministic_and_excludes_outputs(tmp_path: Path) -> None:
    (tmp_path / "input_attribution").mkdir()
    (tmp_path / "input_attribution" / "docs").mkdir()
    (tmp_path / "input_attribution" / "reporting.py").write_text("# source\n", encoding="utf-8")
    (tmp_path / "input_attribution" / "docs" / "usage.md").write_text("# usage\n", encoding="utf-8")
    (tmp_path / "input_attribution" / "PORTABLE_FILES.txt").write_text("input_attribution/reporting.py\n", encoding="utf-8")
    (tmp_path / "input_attribution" / "weights.pt").write_bytes(b"model")
    (tmp_path / "input_attribution" / "weights.pth").write_bytes(b"model")
    (tmp_path / "input_attribution" / "build").mkdir()
    (tmp_path / "input_attribution" / "build" / "generated.py").write_text("generated\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "root-only.md").write_text("do not package\n", encoding="utf-8")
    (tmp_path / "requirements-ig.txt").write_text("captum\n", encoding="utf-8")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "model.zip").write_bytes(b"model")
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "result.json").write_text("{}", encoding="utf-8")
    assert portable_relative_paths(tmp_path) == ("input_attribution/PORTABLE_FILES.txt", "input_attribution/docs/usage.md", "input_attribution/reporting.py")
    first = create_portable_zip(tmp_path, tmp_path / "one.zip").read_bytes()
    second = create_portable_zip(tmp_path, tmp_path / "two.zip").read_bytes()
    assert first == second
    with ZipFile(tmp_path / "one.zip") as archive:
        assert archive.namelist() == ["input_attribution/PORTABLE_FILES.txt", "input_attribution/docs/usage.md", "input_attribution/reporting.py"]
        assert all("model" not in name and "outputs" not in name for name in archive.namelist())
        assert all(not name.startswith("docs/") for name in archive.namelist())
