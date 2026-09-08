"""Report-first packaging tests that do not start MetaDrive or load PPO."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

import analyze_input_attribution as cli
from input_attribution.compact import (
    CompactError,
    build_compact_report,
    compact_existing_result,
    compact_in_place,
    finalize_result_directory,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _full_result(
    tmp_path: Path,
    *,
    name: str = "toy",
    scenario_count: int = 2,
    replacement_strategy: str = "episode_start_constant",
    closed_loop_targets: list[str] | None = None,
) -> Path:
    result = tmp_path / name
    result.mkdir(parents=True)
    (result / "report.md").write_text("# original detailed report\n", encoding="utf-8")
    (result / "feature_schema_expanded.csv").write_text("index,name\n0,speed\n", encoding="utf-8")
    (result / "rollout_arrays.npz").write_bytes(b"rollout")
    (result / "plots").mkdir()
    (result / "plots" / "overview.png").write_bytes(b"png")
    (result / "analysis_metadata.json").write_text(
        json.dumps(
            {
                "experiment": {"name": name},
                "model": {"path": "models/toy.zip", "sha256": "model-sha"},
                "schema": {"name": "toy_schema", "observation_dim": 2},
                "scenario_seed_range": {"start": 5, "count": scenario_count},
                "rollout": {"row_count": 12},
                "baseline": {"strategy": "episode_start", "count": 1},
                "integrated_gradients": {
                    "targets": ["selected_log_probability", "critic_value"],
                    "steps": 8,
                },
                "closed_loop_provenance": {
                    "strategy": replacement_strategy,
                    "reference_source": "per_intervention_episode_reset",
                },
                "closed_loop_targets": (
                    ["navigation", "speed"] if closed_loop_targets is None else closed_loop_targets
                ),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (result / "perturbation_feature_summary.csv").write_text(
        "scope,slice_name,baseline_scope,target_kind,target_name,mean_js_divergence\n"
        "full_episode,all,mean_over_baselines,feature,speed,0.9\n",
        encoding="utf-8",
    )
    _write_csv(
        result / "perturbation_group_summary.csv",
        [
            {
                "scope": "full_episode",
                "slice_name": "all",
                "baseline_scope": "mean_over_baselines",
                "target_kind": "group",
                "target_name": "navigation",
                "target_size": 10,
                "mean_js_divergence": 0.8,
            },
            {
                "scope": "full_episode",
                "slice_name": "all",
                "baseline_scope": "mean_over_baselines",
                "target_kind": "lidar_sector",
                "target_name": "lidar_000.0_015.0_deg",
                "target_size": 10,
                "mean_js_divergence": 0.0,
            },
        ],
    )
    _write_csv(
        result / "ig_feature_summary.csv",
        [
            {
                "scope": "full_episode",
                "slice_name": "all",
                "baseline_scope": "mean_over_baselines",
                "target": "selected_log_probability",
                "feature_name": "speed",
                "mean_absolute_ig": 0.7,
            },
            {
                "scope": "full_episode",
                "slice_name": "all",
                "baseline_scope": "mean_over_baselines",
                "target": "critic_value",
                "feature_name": "speed",
                "mean_absolute_ig": 0.2,
            },
        ],
    )
    _write_csv(
        result / "ig_group_summary.csv",
        [
            {
                "scope": "full_episode",
                "slice_name": "all",
                "baseline_scope": "mean_over_baselines",
                "target": "selected_log_probability",
                "group_kind": "group",
                "group_name": "navigation",
                "group_size": 10,
                "group_absolute_mass": 0.6,
            },
            {
                "scope": "full_episode",
                "slice_name": "all",
                "baseline_scope": "mean_over_baselines",
                "target": "selected_log_probability",
                "group_kind": "lidar_sector",
                "group_name": "lidar_000.0_015.0_deg",
                "group_size": 10,
                "group_absolute_mass": 0.0,
            },
            {
                "scope": "full_episode",
                "slice_name": "all",
                "baseline_scope": "mean_over_baselines",
                "target": "critic_value",
                "group_kind": "group",
                "group_name": "navigation",
                "group_size": 10,
                "group_absolute_mass": 0.3,
            },
        ],
    )
    _write_csv(
        result / "ig_completeness.csv",
        [
            {
                "scope": "full_episode",
                "slice_name": "all",
                "baseline_scope": "mean_over_baselines",
                "target": "selected_log_probability",
                "mean_relative_completeness_error": 0.01,
                "max_relative_completeness_error": 0.02,
                "mean_completeness_residual": 0.001,
            },
            {
                "scope": "full_episode",
                "slice_name": "all",
                "baseline_scope": "mean_over_baselines",
                "target": "critic_value",
                "mean_relative_completeness_error": 0.03,
                "max_relative_completeness_error": 0.04,
                "mean_completeness_residual": 0.002,
            },
        ],
    )
    _write_csv(
        result / "closed_loop_summary.csv",
        [
            {
                "target_name": "navigation",
                "target_kind": "group",
                "scenario_count": scenario_count,
                "replacement_strategy": replacement_strategy,
                "mean_baseline_total_reward": 10.0,
                "mean_intervention_total_reward": 7.0,
                "mean_delta_total_reward": -3.0,
                "mean_baseline_success": 1.0,
                "mean_intervention_success": 0.0,
                "mean_baseline_out_of_road": 0.0,
                "mean_intervention_out_of_road": 1.0,
                "mean_delta_success": -0.5,
                "mean_delta_out_of_road": 0.5,
            },
            {
                "target_name": "speed",
                "target_kind": "feature",
                "scenario_count": scenario_count,
                "replacement_strategy": replacement_strategy,
                "mean_baseline_total_reward": 10.0,
                "mean_intervention_total_reward": 11.0,
                "mean_delta_total_reward": 1.0,
                "mean_baseline_success": 1.0,
                "mean_intervention_success": 1.0,
                "mean_baseline_out_of_road": 0.0,
                "mean_intervention_out_of_road": 0.0,
                "mean_delta_success": 0.0,
                "mean_delta_out_of_road": 0.0,
            },
        ],
    )
    return result


def test_compact_report_centers_the_four_column_evaluation_table(tmp_path: Path) -> None:
    result = _full_result(tmp_path, scenario_count=1)

    # A single scenario is represented as a categorical outcome, never as a
    # rounded percentage inferred from an aggregate.
    rows = [
        {
            "target_name": "navigation",
            "target_kind": "group",
            "scenario_count": 1,
            "replacement_strategy": "schema_constant",
            "mean_baseline_total_reward": 10.0,
            "mean_intervention_total_reward": 7.0,
            "mean_delta_total_reward": -3.0,
            "mean_baseline_success": 1.0,
            "mean_intervention_success": 0.0,
            "mean_baseline_out_of_road": 0.0,
            "mean_intervention_out_of_road": 1.0,
        }
    ]
    _write_csv(result / "closed_loop_summary.csv", rows)

    report = build_compact_report(result)

    assert "| 走らせ方 | 総報酬 (total reward) | 完走 | 道路外への逸脱 |" in report
    assert "進路案内をスキーマ定数で固定（グループ、スキーマ定数, n=1）" in report
    assert "7.00（Δ -3.00）" in report
    assert "できなかった" in report
    assert "あり" in report
    assert "episode開始時" not in report
    assert "摂動 feature" not in report
    assert "model: toy.zip" in report
    assert "入力: toy\\_schema（2 次元）" in report


def test_compact_report_keeps_all_multiscenario_interventions_and_unknown_names(
    tmp_path: Path,
) -> None:
    result = _full_result(
        tmp_path,
        scenario_count=2,
        replacement_strategy="specified_reference_constant",
        closed_loop_targets=[f"custom|target_{index}`" for index in range(6)],
    )
    rows = []
    for index in range(6):
        rows.append(
            {
                "target_name": "normalized_speed" if index == 0 else f"custom|target_{index}`",
                "target_kind": "feature" if index % 2 == 0 else "group",
                "scenario_count": 2,
                "replacement_strategy": "specified_reference_constant",
                "mean_baseline_total_reward": 10.0,
                "mean_intervention_total_reward": 9.0 - index,
                "mean_baseline_success": 1.0,
                "mean_intervention_success": 0.5,
                "mean_baseline_out_of_road": 0.0,
                "mean_intervention_out_of_road": 0.5,
            }
        )
    _write_csv(result / "closed_loop_summary.csv", rows)

    report = build_compact_report(result)

    assert "総報酬 (平均 total reward)" in report
    assert "完走率 (%)" in report
    assert "道路外への逸脱率 (%)" in report
    assert "50.0%" in report
    assert "n=2" in report
    assert "速度計／車速を指定基準値で固定（1項目、指定基準値, n=2）" in report
    assert "custom\\|target\\_1\\`" in report
    assert report.count("で固定（") == 6
    assert "episode開始時" not in report


def test_compact_report_proves_common_baseline_only_from_paired_raw_records(
    tmp_path: Path,
) -> None:
    result = _full_result(tmp_path, scenario_count=2)
    records = []
    for target_name, target_kind, intervention_reward in (
        ("navigation", "group", 7.0),
        ("normalized_speed", "feature", 8.0),
    ):
        for seed, baseline_reward in ((5, 10.0), (6, 11.0)):
            records.append(
                {
                    "target_name": target_name,
                    "target_kind": target_kind,
                    "replacement_strategy": "episode_start_constant",
                    "scenario_seed": seed,
                    "baseline": {
                        "total_reward": baseline_reward,
                        "success": True,
                        "out_of_road": False,
                    },
                    "intervention": {
                        "total_reward": intervention_reward,
                        "success": seed == 5,
                        "out_of_road": seed == 6,
                    },
                }
            )
    _write_jsonl(result / "closed_loop_runs.jsonl", records)

    report = build_compact_report(result)

    assert report.count("| 通常走行 |") == 1
    assert report.count("で固定（") == 2
    assert "総報酬 (平均 total reward)" in report
    assert "75.0%" not in report
    assert "50.0%" in report
    assert "速度計／車速を開始時の値で固定（1項目）" in report


def test_compact_report_does_not_claim_common_baseline_when_seed_or_outcome_differs(
    tmp_path: Path,
) -> None:
    result = _full_result(tmp_path, scenario_count=2)
    records = [
        {
            "target_name": "navigation",
            "target_kind": "group",
            "replacement_strategy": "schema_constant",
            "scenario_seed": 5,
            "baseline": {"total_reward": 10.0, "success": True, "out_of_road": False},
            "intervention": {"total_reward": 8.0, "success": True, "out_of_road": False},
        },
        {
            "target_name": "normalized_speed",
            "target_kind": "feature",
            "replacement_strategy": "schema_constant",
            "scenario_seed": 6,
            "baseline": {"total_reward": 10.0, "success": True, "out_of_road": False},
            "intervention": {"total_reward": 8.0, "success": True, "out_of_road": False},
        },
    ]
    _write_jsonl(result / "closed_loop_runs.jsonl", records)

    report = build_compact_report(result)

    assert "通常走行: 全 target 共通" not in report
    assert report.count("通常走行（対応:") == 2
    assert "n=1" in report


def test_compact_report_explains_unexecuted_closed_loop_without_zero_rows(tmp_path: Path) -> None:
    result = _full_result(tmp_path, closed_loop_targets=[])
    (result / "closed_loop_summary.csv").write_text(
        "target_name,target_kind,scenario_count,mean_baseline_total_reward\n",
        encoding="utf-8",
    )
    (result / "closed_loop_runs.jsonl").write_text("", encoding="utf-8")

    report = build_compact_report(result)

    assert "closed-loop は未実施" in report
    assert "実行が必要" in report
    assert "| 走らせ方 |" not in report
    assert "摂動 feature" not in report


def test_compact_report_marks_missing_outcomes_instead_of_treating_them_as_zero(
    tmp_path: Path,
) -> None:
    result = _full_result(tmp_path, scenario_count=2)
    _write_csv(
        result / "closed_loop_summary.csv",
        [
            {
                "target_name": "navigation",
                "target_kind": "group",
                "scenario_count": 2,
                "replacement_strategy": "dataset_median_constant",
                "mean_baseline_total_reward": "",
                "mean_intervention_total_reward": "",
                "mean_baseline_success": "",
                "mean_intervention_success": "",
                "mean_baseline_out_of_road": "",
                "mean_intervention_out_of_road": "",
            }
        ],
    )

    report = build_compact_report(result)

    assert report.count("未記録") >= 6
    assert "データ中央値で固定" in report
    assert "episode開始時" not in report


@pytest.mark.parametrize("invalid_count", ["", None, "0", "-1", "1.5", "unknown"])
def test_summary_invalid_scenario_count_hides_zero_means(
    tmp_path: Path, invalid_count: object
) -> None:
    result = _full_result(tmp_path, scenario_count=1)
    _write_csv(
        result / "closed_loop_summary.csv",
        [
            {
                "target_name": "navigation",
                "target_kind": "group",
                "scenario_count": invalid_count,
                "replacement_strategy": "schema_constant",
                "mean_baseline_total_reward": 0.0,
                "mean_intervention_total_reward": 0.0,
                "mean_delta_total_reward": 0.0,
                "mean_baseline_success": 0.0,
                "mean_intervention_success": 0.0,
                "mean_baseline_out_of_road": 0.0,
                "mean_intervention_out_of_road": 0.0,
            }
        ],
    )

    report = build_compact_report(result)

    assert "0.00" not in report
    assert "0.0%" not in report
    assert "Δ" not in report
    assert report.count("未記録") >= 6


def test_summary_positive_scenario_count_preserves_measured_zero_outcomes(
    tmp_path: Path,
) -> None:
    result = _full_result(tmp_path, scenario_count=1)
    _write_csv(
        result / "closed_loop_summary.csv",
        [
            {
                "target_name": "navigation",
                "target_kind": "group",
                "scenario_count": 1,
                "replacement_strategy": "schema_constant",
                "mean_baseline_total_reward": 0.0,
                "mean_intervention_total_reward": 0.0,
                "mean_delta_total_reward": 0.0,
                "mean_baseline_success": 0.0,
                "mean_intervention_success": 0.0,
                "mean_baseline_out_of_road": 0.0,
                "mean_intervention_out_of_road": 0.0,
            }
        ],
    )

    report = build_compact_report(result)

    assert "0.00" in report
    assert "0.00（Δ +0.00）" in report
    assert "できなかった" in report
    assert "なし" in report


def _compact_layout_files(result: Path) -> set[str]:
    return {path.relative_to(result).as_posix() for path in result.rglob("*") if path.is_file()}


def _archive_members(result: Path) -> dict[str, bytes]:
    members: dict[str, bytes] = {}
    for archive_path in result.rglob("details.zip"):
        with ZipFile(archive_path) as archive:
            members.update({name: archive.read(name) for name in archive.namelist()})
    return members


def test_compact_in_place_publishes_numbered_sections_and_archives_full_tree(tmp_path: Path) -> None:
    result = _full_result(tmp_path)

    compact_in_place(result)

    assert {
        "report.md",
        "experiment_01_perturbation/report.md",
        "experiment_01_perturbation/details.zip",
        "experiment_02_integrated_gradients/report.md",
        "experiment_02_integrated_gradients/details.zip",
        "experiment_03_closed_loop/report.md",
        "experiment_03_closed_loop/details.zip",
        "shared/details.zip",
    } == _compact_layout_files(result)
    members = _archive_members(result)
    assert members["shared/report.md"] == b"# original detailed report\n"
    assert members["shared/analysis_metadata.json"]
    assert members["shared/plots/overview.png"] == b"png"
    report = (result / "report.md").read_text(encoding="utf-8")
    assert "experiment_01_perturbation/report.md" in report
    assert "experiment_03_closed_loop/report.md" in report


@pytest.mark.parametrize("output_mode", ["compact", "full"])
def test_numbered_reports_explain_experiment_mapping_and_link_to_own_details(
    tmp_path: Path, output_mode: str
) -> None:
    result = _full_result(tmp_path)
    finalize_result_directory(result, output_mode=output_mode)
    root_report = (result / "report.md").read_text(encoding="utf-8")
    assert root_report.startswith("# 入力寄与解析結果\n")
    assert root_report.index("| 実験01") < root_report.index("## 実験03")
    assert "通常走行の収集は共通の準備" in root_report
    expected_detail = (
        "experiment_03_closed_loop/details.zip"
        if output_mode == "compact"
        else "experiment_03_closed_loop/"
    )
    assert f"[実験03の詳細データ]({expected_detail})" in root_report

    section_report = (result / "experiment_03_closed_loop/report.md").read_text(
        encoding="utf-8"
    )
    assert section_report.startswith("# 実験03 入力固定での走行比較\n")
    assert "model: toy.zip" in section_report
    assert "評価条件:" in section_report
    assert "総報酬は評価点の合計、差分は介入 − 通常" in section_report
    assert "変化なしは不要性の証明ではありません" in section_report
    assert "[3実験の一覧に戻る](../report.md)" in section_report

    for report_path in [result / "report.md", *sorted(result.glob("experiment_*/report.md"))]:
        text = report_path.read_text(encoding="utf-8")
        for link in re.findall(r"\]\(([^)]+)\)", text):
            assert (report_path.parent / link).exists(), (report_path, link)
        if report_path.parent != result:
            detail = "details.zip" if output_mode == "compact" else "./"
            assert f"]({detail})" in text
            assert "](../shared/)" not in text


def test_compact_existing_result_does_not_modify_source_and_rejects_compact_input(
    tmp_path: Path,
) -> None:
    source = _full_result(tmp_path, name="source")
    target = tmp_path / "target"
    source_before = (source / "report.md").read_bytes()

    compact_existing_result(source, target)

    assert (source / "report.md").read_bytes() == source_before
    assert (target / "experiment_03_closed_loop/report.md").is_file()
    members = _archive_members(target)
    assert members["shared/report.md"] == source_before
    assert members["shared/analysis_metadata.json"]
    with pytest.raises(CompactError, match="extract the full"):
        compact_existing_result(target, tmp_path / "other")


def test_compact_cli_success_does_not_load_runtime_and_preserves_archive_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = _full_result(tmp_path / "outputs" / "toy" / "attribution", name="source")
    source_bytes = {
        path.relative_to(source).as_posix(): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(
        cli,
        "_load_runtime",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("runtime loaded")),
    )

    output = cli._compact_command(SimpleNamespace(results=source, output_prefix="compact"))

    assert output == source.parent / "compact"
    assert (output / "experiment_01_perturbation/report.md").is_file()
    members = _archive_members(output)
    assert members["shared/report.md"] == source_bytes["report.md"]
    assert members["shared/analysis_metadata.json"] == source_bytes["analysis_metadata.json"]
    assert members["shared/plots/overview.png"] == source_bytes["plots/overview.png"]
    assert (source / "report.md").read_bytes() == source_bytes["report.md"]


def test_numbered_full_source_preserves_root_and_section_reports_in_archives(tmp_path: Path) -> None:
    source = _full_result(tmp_path, name="numbered_source")
    finalize_result_directory(source, output_mode="full")
    root_report = (source / "report.md").read_bytes()
    shared_report = (source / "shared" / "report.md").read_bytes()
    experiment_report = (source / "experiment_01_perturbation" / "report.md").read_bytes()

    target = tmp_path / "numbered_compact"
    compact_existing_result(source, target)

    members = _archive_members(target)
    assert members["shared/report.md"] == root_report
    assert members["shared/full_report.md"] == shared_report
    assert members["experiment_01_perturbation/report.md"] == experiment_report


def test_output_mode_is_compact_by_default_and_full_is_explicit() -> None:
    common = [
        "--config",
        "configs/official.toml",
        "--schema",
        "observation_schemas/metadrive_default_259.toml",
        "--analysis-config",
        "attribution_configs/official_left_curve.toml",
        "--output-prefix",
        "compact_parser_test",
    ]
    for command in ("run", "analyze", "closed-loop"):
        command_args = [command, *common]
        if command == "analyze":
            command_args.extend(["--rollout", "saved-rollout"])
        args = cli.parse_args(command_args)
        assert args.output_mode == "compact"
        full_args = cli.parse_args([*command_args, "--output-mode", "full"])
        assert full_args.output_mode == "full"
    with pytest.raises(SystemExit):
        cli.parse_args(["run", *common, "--output-mode", "unexpected"])


def test_compact_rejects_same_source_and_destination(tmp_path: Path) -> None:
    source = _full_result(tmp_path, name="same")

    with pytest.raises(CompactError, match="separate paths"):
        compact_existing_result(source, source)


def test_disabled_summary_is_explained_without_inventing_zero(tmp_path: Path) -> None:
    result = _full_result(tmp_path)
    (result / "perturbation_feature_summary.csv").write_text(
        "scope,slice_name,baseline_scope,target_kind,target_name,mean_js_divergence\n",
        encoding="utf-8",
    )

    report = build_compact_report(result)

    assert "| 走らせ方 |" in report
    assert "摂動 feature" not in report


def test_compact_cli_failure_keeps_existing_target(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    source = output_root / "toy" / "attribution" / "source"
    _full_result(source.parent, name="source")
    target = output_root / "toy" / "attribution" / "compact"
    target.mkdir(parents=True)
    (target / "old.txt").write_text("old", encoding="utf-8")
    monkeypatch.setattr(cli, "OUTPUT_DIR", output_root)
    monkeypatch.setattr(
        cli,
        "compact_existing_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(CompactError("injected")),
    )

    with pytest.raises(CompactError, match="injected"):
        cli._compact_command(SimpleNamespace(results=source, output_prefix="compact"))

    assert (target / "old.txt").read_text(encoding="utf-8") == "old"
