from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from input_attribution.compact import _target_display
from input_attribution.experiment_reports import build_method_report


def _write_json(root: Path, name: str, value: object) -> None:
    (root / name).write_text(json.dumps(value), encoding="utf-8")


def _write_csv(root: Path, name: str, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with (root / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _metadata(**sections: object) -> dict[str, object]:
    value: dict[str, object] = {
        "model": {"path": "/models/example.zip"},
        "schema": {"name": "schema", "observation_dim": 259},
        "baseline": {"strategy": "episode_start", "count": 1},
    }
    value.update(sections)
    return value


def _label(name: str, indices: tuple[str, ...]) -> str:
    suffix = f" [{','.join(indices)}]" if indices else ""
    return f"{name}{suffix}"


def _common(scope: str = "full_episode", slice_name: str = "all", baseline_index: str = "") -> dict[str, str]:
    return {
        "scope": scope,
        "slice_name": slice_name,
        "baseline_scope": "mean_over_baselines",
        "baseline_index": baseline_index,
    }


def test_perturbation_filters_slices_and_preserves_zero_and_group_size(tmp_path: Path) -> None:
    _write_json(tmp_path, "analysis_metadata.json", _metadata(perturbation={"enabled": True}))
    feature = {
        **_common(),
        "feature_name": "bad|feature_*",
        "feature_index": "7",
        "target_kind": "feature",
        "target_size": "1",
        "mean_js_divergence": "0.0",
        "action_flip_rate": "0.0",
        "mean_value_delta": "999",
    }
    feature_time = {
        **_common(slice_name="progress_0"),
        "feature_name": "time-only",
        "feature_index": "8",
        "target_kind": "feature",
        "mean_js_divergence": "99",
        "action_flip_rate": "1",
    }
    feature_baseline = {
        **_common(baseline_index="0"),
        "feature_name": "other-baseline",
        "feature_index": "9",
        "target_kind": "feature",
        "mean_js_divergence": "88",
        "action_flip_rate": "1",
    }
    _write_csv(tmp_path, "perturbation_feature_summary.csv", [feature, feature_time, feature_baseline])
    semantic = {
        **_common(),
        "group_name": "semantic|group",
        "group_indices": "[0, 1, 2]",
        "target_kind": "group",
        "target_size": "3",
        "mean_js_divergence": "0.2",
        "action_flip_rate": "0.5",
    }
    lidar = {
        **_common(),
        "group_name": "sector",
        "group_indices": "[3, 4]",
        "target_kind": "lidar_sector",
        "target_size": "2",
        "mean_js_divergence": "0",
        "action_flip_rate": "0",
    }
    _write_csv(tmp_path, "perturbation_group_summary.csv", [semantic, lidar])

    report = build_method_report(tmp_path, "01", label_formatter=_label, details_link="details.zip")

    assert "# 実験01 入力置換" in report
    assert "bad\\|feature\\_\\* \\[7\\]" in report
    assert "semantic\\|group \\[0,1,2\\]" in report
    assert "| semantic\\|group \\[0,1,2\\] | 3 | 0.2 | 50.0% |" in report
    assert "| sector \\[3,4\\] | 2 | 0 | 0.0% |" in report
    assert "time-only" not in report
    assert "other-baseline" not in report
    assert "999" not in report
    assert "| bad\\|feature\\_\\* \\[7\\] | 1 | 0 | 0.0% |" in report
    assert "[詳細な全件・rawデータ](details.zip)" in report
    assert "[全体の走行比較](../report.md)" in report


def test_perturbation_disabled_is_not_inferred_from_zero_rows(tmp_path: Path) -> None:
    _write_json(tmp_path, "analysis_metadata.json", _metadata(perturbation={"enabled": False}))
    row = {
        **_common(),
        "feature_name": "stored-but-disabled",
        "feature_index": "0",
        "mean_js_divergence": "0",
        "action_flip_rate": "0",
    }
    _write_csv(tmp_path, "perturbation_feature_summary.csv", [row])

    report = build_method_report(tmp_path, "01", label_formatter=lambda name, _: name, details_link="details.zip")

    assert "- 未実施" in report
    assert "stored-but-disabled" not in report
    assert "JSD (nats)" not in report


def test_executed_false_overrides_enabled_true(tmp_path: Path) -> None:
    _write_json(
        tmp_path,
        "analysis_metadata.json",
        _metadata(perturbation={"enabled": True, "executed": False}),
    )
    row = {
        **_common(),
        "feature_name": "closed-loop-only",
        "feature_index": "0",
        "mean_js_divergence": "0.1",
        "action_flip_rate": "0.1",
    }
    _write_csv(tmp_path, "perturbation_feature_summary.csv", [row])

    report = build_method_report(tmp_path, "01", label_formatter=lambda name, _: name, details_link="details.zip")

    assert "- 未実施" in report
    assert "closed-loop-only" not in report
    assert "実施済みデータあり" not in report


def test_unknown_execution_state_can_show_valid_legacy_rows(tmp_path: Path) -> None:
    # Historical metadata has no enabled field.  Existing aggregate rows are
    # useful evidence, but the report must not call them "実施済み".
    _write_json(tmp_path, "analysis_metadata.json", _metadata(perturbation={}))
    row = {
        **_common(),
        "feature_name": "legacy",
        "feature_index": "0",
        "mean_js_divergence": "0.1",
        "action_flip_rate": "0.25",
    }
    _write_csv(tmp_path, "perturbation_feature_summary.csv", [row])

    report = build_method_report(tmp_path, "01", label_formatter=lambda name, _: name, details_link="details.zip")

    assert "保存済み結果を表示（実行状態の項目は旧metadataに未記録）" in report
    assert "legacy" in report
    assert "実施済みデータあり" not in report


def test_ig_separates_targets_units_and_ignores_nan_or_slices(tmp_path: Path) -> None:
    _write_json(
        tmp_path,
        "analysis_metadata.json",
        _metadata(
            integrated_gradients={
                "enabled": True,
                "targets": ["selected_log_probability", "critic_value"],
                "steps": 64,
            }
        ),
    )
    feature_rows = [
        {
            **_common(),
            "feature_name": "speed|feature",
            "feature_index": "1",
            "target": "selected_log_probability",
            "mean_absolute_ig": "0.5",
            "mean_signed_ig": "-0.25",
        },
        {
            **_common(),
            "feature_name": "zero-feature",
            "feature_index": "2",
            "target": "critic_value",
            "mean_absolute_ig": "0",
            "mean_signed_ig": "0",
        },
        {
            **_common(slice_name="progress_0"),
            "feature_name": "time-only",
            "feature_index": "3",
            "target": "selected_log_probability",
            "mean_absolute_ig": "99",
            "mean_signed_ig": "99",
        },
        {
            **_common(),
            "feature_name": "nan-feature",
            "feature_index": "4",
            "target": "critic_value",
            "mean_absolute_ig": "nan",
            "mean_signed_ig": "nan",
        },
    ]
    _write_csv(tmp_path, "ig_feature_summary.csv", feature_rows)
    group_rows = [
        {
            **_common(),
            "group_name": "navigation|group",
            "group_indices": "[4, 5]",
            "group_kind": "group",
            "group_size": "2",
            "target": "selected_log_probability",
            "group_absolute_mass": "0.4",
            "group_signed_ig": "0.3",
        },
        {
            **_common(),
            "group_name": "lidar-sector",
            "group_indices": "[6]",
            "group_kind": "lidar_sector",
            "group_size": "1",
            "target": "critic_value",
            "group_absolute_mass": "0.0",
            "group_signed_ig": "0.0",
        },
    ]
    _write_csv(tmp_path, "ig_group_summary.csv", group_rows)

    report = build_method_report(tmp_path, "02", label_formatter=_label, details_link="details.zip")

    assert "# 実験02 Integrated Gradients" in report
    assert "## target: selected\\_log\\_probability（単位: nats）" in report
    assert "## target: critic\\_value（単位: critic value / return尺度）" in report
    assert "speed\\|feature \\[1\\]" in report
    assert "navigation\\|group \\[4,5\\]" in report
    assert "| navigation\\|group \\[4,5\\] | 2 | 0.4 | 0.3 |" in report
    assert "| zero-feature \\[2\\] | 1 | 0 | 0 |" in report
    assert "time-only" not in report
    assert "nan-feature" not in report
    assert "target間では大小を比較しません" in report


def test_injected_target_formatter_is_not_html_escaped_twice(tmp_path: Path) -> None:
    _write_json(tmp_path, "analysis_metadata.json", _metadata(perturbation={"enabled": True}))
    row = {
        **_common(),
        "feature_name": "<custom>&name",
        "feature_index": "0",
        "mean_js_divergence": "0.1",
        "action_flip_rate": "0.2",
    }
    _write_csv(tmp_path, "perturbation_feature_summary.csv", [row])

    report = build_method_report(tmp_path, "01", label_formatter=_target_display, details_link="details.zip")

    assert "&lt;custom&gt;&amp;name" in report
    assert "&amp;lt;" not in report


def test_ig_enabled_without_rows_is_unrecorded_and_unknown_id_fails(tmp_path: Path) -> None:
    _write_json(tmp_path, "analysis_metadata.json", _metadata(integrated_gradients={"enabled": True}))
    # Keep a recognized raw-artifact marker so the resolver can read metadata
    # even though this method has no saved rows.
    _write_csv(tmp_path, "ig_feature_summary.csv", [])

    report = build_method_report(tmp_path, "02", label_formatter=lambda name, _: name, details_link="details.zip")

    assert "- 未記録" in report
    assert "比較可能な全体行は未記録です。" in report
    with pytest.raises(ValueError, match="experiment_id"):
        build_method_report(tmp_path, "03", label_formatter=lambda name, _: name, details_link="details.zip")
