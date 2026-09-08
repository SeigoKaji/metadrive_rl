"""Compact, report-first packaging for completed input-attribution results.

The analysis itself intentionally keeps the existing full artifact contract.  A
compact result is only a distribution view: the generated full files are
placed in ``details.zip`` and a short report is left at the result root.  This
module is deliberately independent from MetaDrive, SB3, and the attribution
calculation modules so an existing result can be compacted without loading a
model or starting an environment.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from .artifact_layout import (
    EXPERIMENT_01,
    EXPERIMENT_02,
    EXPERIMENT_03,
    EXPERIMENT_FILES,
    EXPERIMENT_IDS,
    PLOT_EXPERIMENT,
    SHARED,
    SHARED_FILES,
    ArtifactLayoutError,
    detect_layout,
    resolve_artifact,
)
from .results import ArtifactError, atomic_write_text, validate_basename


class CompactError(ValueError):
    """Raised when a full result cannot be safely compacted."""


def _regular_directory(path: Path, *, label: str) -> Path:
    candidate = path.expanduser()
    if candidate.is_symlink() or not candidate.is_dir():
        raise CompactError(f"{label} must be a regular directory: {path}")
    return candidate.resolve()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CompactError(f"compact result is missing {label}: {path.name}")
    try:
        with path.open("r", encoding="utf-8") as file_obj:
            value = json.load(file_obj)
    except (OSError, json.JSONDecodeError) as error:
        raise CompactError(f"could not read {label}: {error}") from error
    if not isinstance(value, dict):
        raise CompactError(f"{label} must contain a JSON object")
    return value


def read_analysis_metadata(result_directory: Path) -> dict[str, Any]:
    """Read the analysis provenance needed by compact report generation."""

    directory = _regular_directory(result_directory, label="result directory")
    try:
        metadata_path = resolve_artifact(directory, "analysis_metadata.json")
    except ArtifactLayoutError as error:
        raise CompactError(str(error)) from error
    return _read_json(metadata_path, label="analysis_metadata.json")


def result_experiment_name(result_directory: Path) -> str:
    """Return the safe experiment name recorded by a full result."""

    metadata = read_analysis_metadata(result_directory)
    experiment = metadata.get("experiment")
    if not isinstance(experiment, Mapping):
        raise CompactError("analysis_metadata.json has no experiment object")
    name = experiment.get("name")
    if not isinstance(name, str) or not name:
        raise CompactError("analysis_metadata.json has no safe experiment name")
    try:
        return validate_basename(name, label="experiment name")
    except (ArtifactError, TypeError) as error:
        raise CompactError("analysis_metadata.json has no safe experiment name") from error


def _read_csv(path: Path) -> list[dict[str, str]]:
    if path.is_symlink() or not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as file_obj:
            return [dict(row) for row in csv.DictReader(file_obj)]
    except (OSError, csv.Error) as error:
        raise CompactError(f"could not read summary CSV {path.name}: {error}") from error


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class _Outcome:
    """One ordinary or intervention outcome used by the compact table."""

    target_name: str
    target_kind: str
    target_indices: tuple[str, ...]
    replacement_strategy: str | None
    scenario_count: int | None
    reward_values: tuple[float, ...] = ()
    success_values: tuple[bool | None, ...] = ()
    out_of_road_values: tuple[bool | None, ...] = ()
    summary_reward: float | None = None
    summary_delta_reward: float | None = None
    summary_success: float | None = None
    summary_out_of_road: float | None = None
    source: str = "raw"


@dataclass(frozen=True)
class _ReportRow:
    """A row in the four-column compact evaluation table."""

    label: str
    outcome: _Outcome
    baseline: _Outcome | None = None
    intervention: bool = False


_TARGET_LABELS = {
    "ego_state": "自車状態",
    "navigation": "進路案内",
    "lidar": "LiDAR",
    "lidar_all": "LiDAR全周",
    "road_boundaries": "道路境界",
    "heading": "基準車線との方位差",
    "speed": "車速",
    "steering_and_action_history": "ハンドルと前回の操作",
    "yaw_rate": "車の向きが変わる速さ",
    "current_lane_lateral_position": "現在車線の横位置",
    "checkpoint_1": "通過点1",
    "checkpoint_2": "通過点2",
    "distance_to_left_road_boundary": "左道路境界までの距離",
    "distance_to_right_road_boundary": "右道路境界までの距離",
    "heading_difference_to_reference_lane": "基準車線との方位差",
    "normalized_speed": "速度計／車速",
    "current_steering": "現在のハンドルの切れ角",
    "previous_steering_action": "前回のハンドル操作",
    "previous_throttle_brake_action": "前回の加減速操作",
    "lateral_position_in_current_lane": "現在車線内の横位置",
    "checkpoint_1_forward_projection": "通過点1の前後方向の位置",
    "checkpoint_1_right_projection": "通過点1の左右方向の位置",
    "checkpoint_1_curve_radius": "通過点1 曲率半径",
    "checkpoint_1_curve_direction": "通過点1の道が曲がる向き",
    "checkpoint_1_curve_angle": "通過点1の道が曲がる角度",
    "checkpoint_2_forward_projection": "通過点2の前後方向の位置",
    "checkpoint_2_right_projection": "通過点2の左右方向の位置",
    "checkpoint_2_curve_radius": "通過点2 曲率半径",
    "checkpoint_2_curve_direction": "通過点2の道が曲がる向き",
    "checkpoint_2_curve_angle": "通過点2の道が曲がる角度",
}


def _markdown_text(value: Any, *, missing: str = "未記録") -> str:
    """Return unformatted, table-safe text for metadata and target names."""

    if value is None:
        return missing
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return missing
    # The table intentionally does not wrap labels in inline-code markers.
    # Escape HTML and Markdown metacharacters so custom names cannot alter the
    # report structure when they come from metadata or JSONL.
    text = html.escape(text, quote=False)
    text = text.replace("\r", " ").replace("\n", " ")
    for character in ("\\", "`", "|", "*", "_", "[", "]", "#"):
        text = text.replace(character, "\\" + character)
    return text


def _target_display(name: Any, indices: tuple[str, ...] = ()) -> str:
    """Give known schema names a short Japanese gloss without hiding raw names."""

    raw = _markdown_text(name, missing="unknown")
    key = str(name).strip().lower() if name is not None else ""
    gloss = _TARGET_LABELS.get(key)
    if gloss is None:
        checkpoint = re.fullmatch(r"checkpoint[_ -]?(\d+)", key)
        if checkpoint is not None:
            gloss = f"チェックポイント{checkpoint.group(1)}"
    display = raw if gloss is None else _markdown_text(gloss)
    if indices:
        display += f" [indices: {_markdown_text(','.join(indices))}]"
    return display


def _target_value(row: Mapping[str, Any], *fields: str) -> str:
    for field in fields:
        value = row.get(field)
        if value is not None and str(value).strip():
            return str(value)
    return "unknown"


def _target_indices(value: Any) -> tuple[str, ...]:
    if value is None or str(value).strip() == "":
        return ()
    if isinstance(value, str):
        values = value.split(";")
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, Mapping)):
        values = list(value)
    else:
        values = [value]
    return tuple(str(item).strip() for item in values if str(item).strip())


def _bool_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number) and number in {0.0, 1.0}:
            return bool(number)
        return None
    text = str(value).strip().lower() if value is not None else ""
    if text in {"true", "1", "yes", "y", "あり", "できた"}:
        return True
    if text in {"false", "0", "no", "n", "なし", "できなかった"}:
        return False
    return None


def _positive_count(value: Any) -> int | None:
    number = _number(value)
    if number is None or number < 1 or not number.is_integer():
        return None
    return int(number)


def _scenario_seed(value: Any) -> int | str | None:
    if value is None or str(value).strip() == "":
        return None
    number = _number(value)
    if number is not None and number.is_integer():
        return int(number)
    return str(value)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read paired records when present; an absent/empty file means no run."""

    if path.is_symlink() or not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as file_obj:
            for line_number, line in enumerate(file_obj, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise CompactError(
                        f"could not read {path.name} line {line_number}: {error}"
                    ) from error
                if not isinstance(value, Mapping):
                    raise CompactError(
                        f"{path.name} line {line_number} must contain a JSON object"
                    )
                records.append(dict(value))
    except OSError as error:
        raise CompactError(f"could not read {path.name}: {error}") from error
    return records


def _outcome_mapping(record: Mapping[str, Any], side: str) -> Mapping[str, Any]:
    value = record.get(side)
    return value if isinstance(value, Mapping) else {}


def _raw_outcome(
    target_name: str,
    target_kind: str,
    target_indices: tuple[str, ...],
    strategy: str | None,
    records: list[Mapping[str, Any]],
    *,
    side: str,
) -> _Outcome:
    rewards: list[float] = []
    successes: list[bool | None] = []
    out_of_road: list[bool | None] = []
    for record in records:
        side_values = _outcome_mapping(record, side)
        reward = _number(side_values.get("total_reward"))
        if reward is None:
            reward = _number(record.get(f"{side}_total_reward"))
        if reward is not None:
            rewards.append(reward)
        successes.append(_bool_value(side_values.get("success")))
        out_of_road.append(_bool_value(side_values.get("out_of_road")))
    return _Outcome(
        target_name=target_name,
        target_kind=target_kind,
        target_indices=target_indices,
        replacement_strategy=strategy,
        scenario_count=len(records),
        reward_values=tuple(rewards),
        success_values=tuple(successes),
        out_of_road_values=tuple(out_of_road),
        source="raw",
    )


def _summary_outcome(
    row: Mapping[str, Any],
    *,
    side: str,
    default_strategy: str | None,
) -> _Outcome:
    target_name = _target_value(row, "target_name", "target")
    target_kind = _target_value(row, "target_kind", "kind")
    scenario_count = _positive_count(row.get("scenario_count"))
    measured = scenario_count is not None
    return _Outcome(
        target_name=target_name,
        target_kind=target_kind,
        target_indices=_target_indices(row.get("target_indices")),
        replacement_strategy=(
            str(row.get("replacement_strategy"))
            if row.get("replacement_strategy") not in {None, ""}
            else default_strategy
        ),
        scenario_count=scenario_count,
        summary_reward=(
            _number(row.get(f"mean_{side}_total_reward")) if measured else None
        ),
        summary_delta_reward=(
            _number(row.get("mean_delta_total_reward"))
            if measured and side == "intervention"
            else None
        ),
        summary_success=(
            _number(row.get(f"mean_{side}_success")) if measured else None
        ),
        summary_out_of_road=(
            _number(row.get(f"mean_{side}_out_of_road")) if measured else None
        ),
        source="summary",
    )


def _raw_groups(
    records: Iterable[Mapping[str, Any]],
) -> list[tuple[str, str, tuple[str, ...], str | None, list[dict[str, Any]]]]:
    groups: dict[tuple[str, str, tuple[str, ...], str | None], list[dict[str, Any]]] = {}
    order: list[tuple[str, str, tuple[str, ...], str | None]] = []
    for record in records:
        target_name = _target_value(record, "target_name", "target")
        target_kind = _target_value(record, "target_kind", "kind")
        target_indices = _target_indices(record.get("target_indices"))
        strategy = (
            str(record.get("replacement_strategy"))
            if record.get("replacement_strategy") not in {None, ""}
            else None
        )
        key = (target_name, target_kind, target_indices, strategy)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(dict(record))
    return [
        (name, kind, indices, strategy, groups[(name, kind, indices, strategy)])
        for name, kind, indices, strategy in order
    ]


def _baseline_signature(
    group: tuple[str, str, tuple[str, ...], str | None, list[dict[str, Any]]],
) -> tuple[Any, ...] | None:
    """Return a strict signature suitable for proving a common baseline."""

    _, _, _, _, records = group
    values: list[tuple[Any, float, bool, bool]] = []
    for record in records:
        seed = _scenario_seed(record.get("scenario_seed"))
        baseline = _outcome_mapping(record, "baseline")
        reward = _number(baseline.get("total_reward"))
        success = _bool_value(baseline.get("success"))
        out_of_road = _bool_value(baseline.get("out_of_road"))
        if seed is None or reward is None or success is None or out_of_road is None:
            return None
        values.append((seed, reward, success, out_of_road))
    # A duplicate seed is retained in the sequence, so a malformed target with
    # a different number of records cannot be mistaken for the same baseline.
    return tuple(sorted(values, key=lambda value: repr(value[0])))


def _common_baseline(
    groups: list[tuple[str, str, tuple[str, ...], str | None, list[dict[str, Any]]]],
) -> bool:
    if not groups:
        return False
    first = _baseline_signature(groups[0])
    if first is None:
        return False
    for group in groups[1:]:
        other = _baseline_signature(group)
        if other is None or len(other) != len(first):
            return False
        for left, right in zip(first, other):
            if left[0] != right[0] or left[2:] != right[2:]:
                return False
            if left[1] != right[1]:
                return False
    return True


def _strategy_short(strategy: Any) -> str:
    raw = "未記録" if strategy is None or str(strategy).strip() == "" else str(strategy)
    labels = {
        "episode_start_constant": "開始時の値",
        "schema_constant": "スキーマ定数",
        "dataset_median_constant": "データ中央値",
        "specified_reference_constant": "指定基準値",
    }
    return _markdown_text(labels.get(raw, raw), missing="未記録")


def _count_label(count: int | None) -> str:
    return f"n={count}" if count is not None else "n=不明"


def _mean_outcome(outcome: _Outcome, field: str) -> float | None:
    summary_value = getattr(outcome, f"summary_{field}")
    if summary_value is not None:
        return summary_value
    values = getattr(outcome, f"{field}_values")
    if not values:
        return None
    return sum(values) / len(values)


def _binary_rate(outcome: _Outcome, field: str) -> tuple[float | None, int | None]:
    summary_value = getattr(outcome, f"summary_{field}")
    if summary_value is not None:
        return (summary_value if 0.0 <= summary_value <= 1.0 else None, None)
    values = [value for value in getattr(outcome, f"{field}_values") if value is not None]
    if not values:
        return None, 0
    return sum(1.0 if value else 0.0 for value in values) / len(values), len(values)


def _format_reward(outcome: _Outcome, *, baseline: _Outcome | None = None) -> str:
    value = _mean_outcome(outcome, "reward")
    if value is None:
        return "未記録"
    if outcome.source == "raw" and (
        outcome.scenario_count is None
        or len(outcome.reward_values) != outcome.scenario_count
    ):
        # A partial average would conceal which scenario was missing and could
        # make a non-paired delta look like a paired result.
        return "未記録"
    result = f"{value:.2f}"
    if baseline is not None and _counts_compatible(outcome, baseline):
        if outcome.source == "summary":
            delta = outcome.summary_delta_reward
            if delta is not None:
                result += f"（Δ {delta:+.2f}）"
        elif (
            baseline.source != "raw"
            or (
                baseline.scenario_count is not None
                and len(baseline.reward_values) == baseline.scenario_count
            )
        ):
            baseline_value = _mean_outcome(baseline, "reward")
            if baseline_value is not None:
                result += f"（Δ {value - baseline_value:+.2f}）"
    return result


def _counts_compatible(first: _Outcome, second: _Outcome) -> bool:
    return (
        first.scenario_count is not None
        and second.scenario_count is not None
        and first.scenario_count == second.scenario_count
    )


def _format_binary(outcome: _Outcome, *, field: str, positive: str, negative: str) -> str:
    if outcome.scenario_count == 1:
        values = getattr(outcome, f"{field}_values")
        if values and values[0] is not None:
            return positive if values[0] else negative
        summary_value = getattr(outcome, f"summary_{field}")
        if summary_value == 1.0:
            return positive
        if summary_value == 0.0:
            return negative
        return "未記録"
    rate, valid_count = _binary_rate(outcome, field)
    if rate is None:
        return "未記録"
    if valid_count is not None and outcome.scenario_count is not None and valid_count != outcome.scenario_count:
        return f"{rate * 100:.1f}%（有効 {valid_count}/{outcome.scenario_count}）"
    if outcome.scenario_count is None:
        return f"{rate * 100:.1f}%（n不明）"
    return f"{rate * 100:.1f}%"


def _normal_label(outcome: _Outcome, *, common: bool, include_indices: bool = False) -> str:
    if common:
        return "通常走行"
    else:
        description = (
            f"対応: {_target_display(outcome.target_name)} "
            f"({_markdown_text(outcome.target_kind)}, {_count_label(outcome.scenario_count)})"
        )
        if include_indices:
            description = (
                f"対応: {_target_display(outcome.target_name, outcome.target_indices)} "
                f"({_markdown_text(outcome.target_kind)}, {_count_label(outcome.scenario_count)})"
            )
    return f"通常走行（{description}）"


def _intervention_label(
    outcome: _Outcome,
    *,
    detailed: bool = False,
    include_indices: bool = False,
) -> str:
    strategy = _strategy_short(outcome.replacement_strategy)
    kind = "グループ" if outcome.target_kind == "group" else "1項目"
    suffix = f"（{kind}"
    if detailed:
        suffix += f"、{strategy}, {_count_label(outcome.scenario_count)}"
    suffix += "）"
    return f"{_target_display(outcome.target_name, outcome.target_indices if include_indices else ())}を{strategy}で固定{suffix}"


def _render_evaluation_table(rows: list[_ReportRow]) -> str:
    counts = [row.outcome.scenario_count for row in rows]
    all_single = bool(counts) and all(count == 1 for count in counts)
    all_multi = bool(counts) and all(count is None or count > 1 for count in counts)
    if all_single:
        reward_header = "総報酬 (total reward)"
        success_header = "完走"
        road_header = "道路外への逸脱"
    elif all_multi:
        reward_header = "総報酬 (平均 total reward)"
        success_header = "完走率 (%)"
        road_header = "道路外への逸脱率 (%)"
    else:
        reward_header = "総報酬 (単一値 / 複数平均)"
        success_header = "完走 (単一: 可否 / 複数: 率%)"
        road_header = "道路外への逸脱 (単一: 有無 / 複数: 率%)"
    lines = [
        f"| 走らせ方 | {reward_header} | {success_header} | {road_header} |",
        "| --- | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {label} | {reward} | {success} | {road} |".format(
                label=row.label,
                reward=_format_reward(
                    row.outcome,
                    baseline=row.baseline if row.intervention else None,
                ),
                success=_format_binary(
                    row.outcome,
                    field="success",
                    positive="できた",
                    negative="できなかった",
                ),
                road=_format_binary(
                    row.outcome,
                    field="out_of_road",
                    positive="あり",
                    negative="なし",
                ),
            )
        )
    return "\n".join(lines)


def _closed_loop_report_section(
    directory: Path,
    metadata: Mapping[str, Any],
) -> str:
    try:
        records_path = resolve_artifact(directory, "closed_loop_runs.jsonl")
        summary_path = resolve_artifact(directory, "closed_loop_summary.csv")
    except ArtifactLayoutError as error:
        raise CompactError(str(error)) from error
    records = _read_jsonl(records_path)
    summary_rows = _read_csv(summary_path)
    closed_provenance = metadata.get("closed_loop_provenance")
    default_strategy = (
        str(closed_provenance.get("strategy"))
        if isinstance(closed_provenance, Mapping)
        and closed_provenance.get("strategy") not in {None, ""}
        else None
    )
    if not records and not summary_rows:
        targets = metadata.get("closed_loop_targets")
        if isinstance(targets, list) and targets:
            reason = "対象は記録されていますが、paired 結果ファイルが空です。closed-loop の実行が必要です。"
        else:
            reason = "この結果では closed-loop は未実施、または結果が記録されていません。評価を表示するには closed-loop の実行が必要です。"
        return f"## Paired closed-loop 評価\n\n{reason}"

    rows: list[_ReportRow] = []
    common_baseline_verified = False
    if records:
        groups = _raw_groups(records)
        common = _common_baseline(groups)
        common_baseline_verified = common
        duplicate_target_keys = {
            (name, kind)
            for name, kind, _, _, _ in groups
            if sum(1 for other_name, other_kind, _, _, _ in groups if (other_name, other_kind) == (name, kind)) > 1
        }
        strategies = {strategy for _, _, _, strategy, _ in groups if strategy not in {None, ""}}
        strategy_common = len(strategies) <= 1
        common_baseline: _Outcome | None = None
        if common:
            name, kind, indices, strategy, group_records = groups[0]
            common_baseline = _raw_outcome(
                name,
                kind,
                indices,
                strategy or default_strategy,
                group_records,
                side="baseline",
            )
            rows.append(
                _ReportRow(
                    label=_normal_label(common_baseline, common=True),
                    outcome=common_baseline,
                )
            )
        for name, kind, indices, strategy, group_records in groups:
            strategy = strategy or default_strategy
            intervention = _raw_outcome(
                name,
                kind,
                indices,
                strategy,
                group_records,
                side="intervention",
            )
            baseline = _raw_outcome(
                name,
                kind,
                indices,
                strategy,
                group_records,
                side="baseline",
            )
            if not common:
                rows.append(
                    _ReportRow(
                        label=_normal_label(
                            baseline,
                            common=False,
                            include_indices=(name, kind) in duplicate_target_keys,
                        ),
                        outcome=baseline,
                    )
                )
            rows.append(
                _ReportRow(
                    label=_intervention_label(
                        intervention,
                        detailed=(not common or not strategy_common),
                        include_indices=(name, kind) in duplicate_target_keys,
                    ),
                    outcome=intervention,
                    baseline=common_baseline if common else baseline,
                    intervention=True,
                )
            )
    else:
        # Summary CSVs have means but do not prove that the ordinary outcome is
        # shared across targets.  Keep an ordinary row per target instead of
        # presenting it as one common baseline.
        for row in summary_rows:
            baseline = _summary_outcome(row, side="baseline", default_strategy=default_strategy)
            intervention = _summary_outcome(
                row,
                side="intervention",
                default_strategy=default_strategy,
            )
            rows.append(_ReportRow(label=_normal_label(baseline, common=False), outcome=baseline))
            rows.append(
                _ReportRow(
                    label=_intervention_label(intervention, detailed=True),
                    outcome=intervention,
                    baseline=baseline,
                    intervention=True,
                )
            )

    if not rows:
        return "## Paired closed-loop 評価\n\n結果行がありません。評価を表示するには closed-loop の実行が必要です。"
    strategies = {
        row.outcome.replacement_strategy
        for row in rows
        if row.intervention and row.outcome.replacement_strategy not in {None, ""}
    }
    notes: list[str] = []
    if not common_baseline_verified:
        notes.append("通常走行は target 別に記録しています（共通 baseline と検証できないため）。")
    if len(strategies) > 1:
        notes.append("置換方法は target ごとに異なります。")
    prefix = " ".join(notes)
    if prefix:
        prefix += "\n\n"
    return prefix + _render_evaluation_table(rows)


def build_compact_report(
    result_directory: Path,
    *,
    details_link: str | None = None,
    heading_level: int = 1,
) -> str:
    """Build the experiment-03 comparison without rerunning the experiment."""

    directory = _regular_directory(result_directory, label="result directory")
    metadata = read_analysis_metadata(directory)
    model = metadata.get("model") if isinstance(metadata.get("model"), Mapping) else {}
    schema = metadata.get("schema") if isinstance(metadata.get("schema"), Mapping) else {}
    rollout = metadata.get("rollout") if isinstance(metadata.get("rollout"), Mapping) else {}
    seed_range = metadata.get("scenario_seed_range")
    scenario_count = seed_range.get("count") if isinstance(seed_range, Mapping) else None
    scenario_start = seed_range.get("start") if isinstance(seed_range, Mapping) else None
    if scenario_count is None:
        scenario_count = rollout.get("scenario_count")

    closed_provenance = metadata.get("closed_loop_provenance")
    closed_strategy = (
        closed_provenance.get("strategy")
        if isinstance(closed_provenance, Mapping)
        else None
    )
    model_path = model.get("path")
    model_basename = (
        str(model_path).replace("\\", "/").rsplit("/", 1)[-1]
        if model_path not in {None, ""}
        else None
    )
    observation_dim = schema.get("observation_dim", metadata.get("model_observation_dim"))
    details_note = f"\n\n[実験03の詳細データ]({details_link})" if details_link else ""
    return f"""{'#' * heading_level} 実験03 入力固定での走行比較

同じ開始条件の通常走行と、指定した入力だけを置き換え続ける走行を比較します。以下の表は実験03の結果であり、実験01のJSDや実験02のIGとは別です。

- model: {_markdown_text(model_basename)}
- 評価条件: seed {_markdown_text(scenario_start)} から {_markdown_text(scenario_count)} 件、RL seed {_markdown_text(metadata.get('rl_seed'))}
- 入力: {_markdown_text(schema.get('name'))}（{_markdown_text(observation_dim)} 次元）
- 置換値: {_strategy_short(closed_strategy)}

{_closed_loop_report_section(directory, metadata)}

総報酬は評価点の合計、差分は介入 − 通常です。複数 scenario では報酬を平均し、完走と道路外への逸脱を率（%）で表示します。欠損は `未記録` とし、0 には置き換えません。
この結果は指定モデル・評価条件・置換方法に限った手掛かりであり、変化なしは不要性の証明ではありません。{details_note}
"""


def _iter_archive_files(source: Path) -> list[Path]:
    """Return regular files below a section, rejecting unsafe input."""

    files: list[Path] = []
    for path in sorted(source.rglob("*")):
        if path.name == "details.zip":
            raise CompactError(
                "compact input already contains details.zip; extract the full result archive "
                "into a new directory and retry"
            )
        if path.is_symlink():
            raise CompactError(f"compact input contains an unsafe symlink: {path}")
        if path.is_file():
            files.append(path)
    return files


def _write_archive(root: Path, section: Path, archive_path: Path) -> None:
    """Archive one numbered section with run-root-relative member names."""

    if root.is_symlink() or section.is_symlink() or archive_path.is_symlink():
        raise CompactError(f"archive path contains an unsafe symlink: {section}")
    root = root.resolve()
    section = section.resolve()
    archive_path = archive_path.resolve()
    if not section.is_dir():
        raise CompactError(f"archive section must be a regular directory: {section}")
    files = _iter_archive_files(section)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{archive_path.name}.", suffix=".tmp", dir=archive_path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
            for path in files:
                try:
                    member = path.relative_to(root).as_posix()
                except ValueError as error:
                    raise CompactError(f"archive file escaped result root: {path}") from error
                archive.write(path, member)
        os.replace(temporary, archive_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _move_checked(source: Path, target: Path) -> None:
    """Move one staging artifact without overwriting a sibling."""

    if source.is_symlink():
        raise CompactError(f"result contains an unsafe symlink: {source}")
    if not source.exists():
        return
    if target.exists() or target.is_symlink():
        raise CompactError(f"artifact destination collision: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(target))


def _organize_flat_result(directory: Path) -> None:
    """Move the legacy flat staging tree into the numbered raw layout."""

    try:
        layout = detect_layout(directory)
    except ArtifactLayoutError as error:
        raise CompactError(str(error)) from error
    if not layout.is_flat:
        raise CompactError(f"expected a flat full result, found {layout.kind}")

    for child in list(directory.iterdir()):
        if child.name in {SHARED, *EXPERIMENT_IDS}:
            raise CompactError(f"numbered artifact directory already exists: {child}")
        if child.name == "report.md":
            _move_checked(child, directory / SHARED / "report.md")
            continue
        owner: str | None = None
        if child.name in SHARED_FILES:
            owner = SHARED
        else:
            for experiment_id, names in EXPERIMENT_FILES.items():
                if child.name in names:
                    owner = experiment_id
                    break
        if owner is not None:
            _move_checked(child, directory / owner / child.name)
            continue
        if child.name == "plots" and child.is_dir():
            if child.is_symlink():
                raise CompactError(f"result contains an unsafe symlink: {child}")
            for plot in list(child.iterdir()):
                if plot.name in PLOT_EXPERIMENT:
                    owner = PLOT_EXPERIMENT[plot.name]
                    _move_checked(plot, directory / owner / "plots" / plot.name)
                else:
                    _move_checked(plot, directory / SHARED / "plots" / plot.name)
            try:
                child.rmdir()
            except OSError as error:
                raise CompactError(f"could not clear legacy plots directory: {child}") from error
            continue
        # Preserve unknown files and directories in shared/ rather than
        # silently dropping a future artifact.  The collision check makes an
        # ambiguous classification a safe failure.
        _move_checked(child, directory / SHARED / child.name)


def _prepare_numbered_full(directory: Path) -> None:
    """Validate numbered directories and retain a prior root report."""

    for name in (SHARED, *EXPERIMENT_IDS):
        child = directory / name
        if child.exists() and (child.is_symlink() or not child.is_dir()):
            raise CompactError(f"numbered artifact directory is unsafe: {child}")
    shared = directory / SHARED
    shared.mkdir(parents=True, exist_ok=True)
    root_report = directory / "report.md"
    if root_report.is_symlink():
        raise CompactError(f"result contains an unsafe symlink: {root_report}")
    if root_report.is_file():
        # Keep the prior root overview at the canonical shared/report.md path.
        # A numbered full result may already have a preserved flat report in
        # shared/report.md; retain that byte-for-byte under an explicit name
        # before moving the root overview into the canonical slot.
        if (shared / "report.md").exists():
            _move_checked(shared / "report.md", shared / "full_report.md")
        _move_checked(root_report, shared / "report.md")
    # A numbered source may carry a newer/unknown top-level artifact. Keep it
    # in shared so compacting never leaves an unarchived root file behind.
    for child in list(directory.iterdir()):
        if child.name in {SHARED, *EXPERIMENT_IDS, "report.md"}:
            continue
        _move_checked(child, shared / child.name)


def _organize_collect_result(directory: Path) -> None:
    """Keep collector output in shared/ while retaining every artifact."""

    try:
        layout = detect_layout(directory)
    except ArtifactLayoutError as error:
        raise CompactError(str(error)) from error
    if layout.is_numbered:
        return
    if not layout.is_flat:
        raise CompactError(f"expected a flat collector result, found {layout.kind}")
    for child in list(directory.iterdir()):
        if child.name == "report.md":
            _move_checked(child, directory / SHARED / "report.md")
        else:
            _move_checked(child, directory / SHARED / child.name)


def _method_report(
    directory: Path,
    experiment_id: str,
    *,
    compact: bool,
) -> str:
    """Build one dedicated method report through the shared report API."""

    details_link = "details.zip" if compact else "./"
    method_number = "01" if experiment_id == EXPERIMENT_01 else "02"
    from .experiment_reports import build_method_report

    return build_method_report(
        directory,
        method_number,
        label_formatter=_target_display,
        details_link=details_link,
    )


def _experiment03_report(directory: Path, *, compact: bool) -> str:
    return build_compact_report(
        directory, details_link="details.zip" if compact else "./"
    ).rstrip() + "\n\n[3実験の一覧に戻る](../report.md)\n"


def _root_report_text(directory: Path, *, compact: bool) -> str:
    """Put the experiment mapping before the requested driving-result table."""

    detail_suffix = "details.zip" if compact else ""
    comparison = build_compact_report(
        directory,
        details_link=f"{EXPERIMENT_03}/{detail_suffix}",
        heading_level=2,
    )
    return f"""# 入力寄与解析結果

実験は次の3つです。通常走行の収集は共通の準備であり、別の実験には数えません。

| 番号・実験 | 調べること | 結果を見る |
| --- | --- | --- |
| 実験01 入力置換 | 入力を基準値へ置き換えると、操作の選択確率が変わるか | [実験01の表]({EXPERIMENT_01}/report.md) |
| 実験02 Integrated Gradients | 基準値での出力から現在の出力までの変化を、各入力へどう割り当てるか | [実験02の表]({EXPERIMENT_02}/report.md) |
| 実験03 入力固定での走行比較 | 入力を固定して走ると、総報酬・完走・逸脱が変わるか | [実験03の表]({EXPERIMENT_03}/report.md) |

走行結果だけを確認する場合は、下の実験03の表を読めば十分です。実験01・02の主要表もリンク先にあり、CSVを自分で集計する必要はありません。
共通の設定・入力の対応表・通常走行の記録は [共通の準備データ]({SHARED}/{detail_suffix}) に保存しています。

{comparison.rstrip()}
"""


def _write_method_reports(directory: Path, *, compact: bool) -> None:
    for experiment_id in (EXPERIMENT_01, EXPERIMENT_02):
        target = directory / experiment_id
        target.mkdir(parents=True, exist_ok=True)
        atomic_write_text(target / "report.md", _method_report(directory, experiment_id, compact=compact))
    target = directory / EXPERIMENT_03
    target.mkdir(parents=True, exist_ok=True)
    atomic_write_text(target / "report.md", _experiment03_report(directory, compact=compact))


def _write_root_report(directory: Path, *, compact: bool) -> None:
    atomic_write_text(directory / "report.md", _root_report_text(directory, compact=compact))


def _remove_raw_sections(directory: Path, *, preserve_shared_report: bool = False) -> None:
    for name in (SHARED, *EXPERIMENT_IDS):
        section = directory / name
        for child in list(section.iterdir()):
            if child.name == "details.zip":
                continue
            if child.name == "report.md" and (name != SHARED or preserve_shared_report):
                continue
            if child.is_symlink() or child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
            else:
                raise CompactError(f"cannot remove unsupported compact artifact: {child}")


def _compact_numbered_result(directory: Path, *, already_numbered: bool) -> None:
    """Write section archives and reduce a numbered full tree to reports."""

    sections = [directory / name for name in (EXPERIMENT_01, EXPERIMENT_02, EXPERIMENT_03, SHARED)]
    report_texts = {
        EXPERIMENT_01: _method_report(directory, EXPERIMENT_01, compact=True),
        EXPERIMENT_02: _method_report(directory, EXPERIMENT_02, compact=True),
        EXPERIMENT_03: _experiment03_report(directory, compact=True),
    }
    root_report = _root_report_text(directory, compact=True)
    if already_numbered:
        # Archive the original reports/raw artifacts before replacing reports
        # with the concise numbered views.  This preserves a numbered full
        # source's root and per-experiment reports byte-for-byte in details.
        for section in sections:
            section.mkdir(parents=True, exist_ok=True)
            _write_archive(directory, section, section / "details.zip")
    for experiment_id, text in report_texts.items():
        atomic_write_text(directory / experiment_id / "report.md", text)
    atomic_write_text(directory / "report.md", root_report.rstrip() + "\n")
    if not already_numbered:
        for section in sections:
            _write_archive(directory, section, section / "details.zip")
    _remove_raw_sections(directory)


def finalize_result_directory(
    result_directory: Path,
    *,
    output_mode: str = "full",
    result_kind: str = "analysis",
) -> Path:
    """Organize a flat staging result and optionally publish compact views."""

    directory = _regular_directory(result_directory, label="result directory")
    if output_mode not in {"full", "compact"}:
        raise CompactError(f"unknown output mode: {output_mode}")
    try:
        layout = detect_layout(directory)
    except ArtifactLayoutError as error:
        raise CompactError(str(error)) from error
    if result_kind == "collect":
        _organize_collect_result(directory)
        atomic_write_text(
            directory / "report.md",
            "# 共通準備\n\n"
            "rollout と schema を保存しました。実験01〜03の解析は未実施です。"
            " 詳細なデータは `shared/` を参照してください。\n",
        )
        return directory
    if layout.is_flat:
        _organize_flat_result(directory)
        already_numbered = False
    elif layout.is_numbered:
        _prepare_numbered_full(directory)
        already_numbered = True
    else:
        raise CompactError(
            "result is already compact; extract the four details.zip archives into a new "
            "full result directory before regrouping it"
        )
    if output_mode == "compact":
        _compact_numbered_result(directory, already_numbered=already_numbered)
    else:
        _write_method_reports(directory, compact=False)
        _write_root_report(directory, compact=False)
    return directory


def compact_in_place(result_directory: Path) -> Path:
    """Organize a full result and leave one report/archive per section."""

    return finalize_result_directory(result_directory, output_mode="compact")


def _overlap(first: Path, second: Path) -> bool:
    try:
        first.relative_to(second)
        return True
    except ValueError:
        pass
    try:
        second.relative_to(first)
        return True
    except ValueError:
        return False


def _copy_full_result(source: Path, target: Path) -> None:
    """Copy a source full tree into an empty staging directory safely."""

    target.mkdir(parents=True, exist_ok=True)
    for source_path in sorted(source.rglob("*")):
        relative = source_path.relative_to(source)
        target_path = target / relative
        if source_path.is_symlink():
            raise CompactError(f"compact source contains an unsafe symlink: {source_path}")
        if source_path.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
        elif source_path.is_file():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
        else:
            raise CompactError(f"compact source contains unsupported artifact: {source_path}")


def compact_existing_result(source_directory: Path, target_directory: Path) -> Path:
    """Create a compact sibling from a flat or numbered full result."""

    source = _regular_directory(source_directory, label="source result directory")
    target = target_directory.expanduser().resolve()
    if _overlap(source, target):
        raise CompactError(
            "compact source and destination must be separate paths; choose a new output prefix"
        )
    try:
        detect_layout(source)
    except ArtifactLayoutError as error:
        raise CompactError(str(error)) from error
    _copy_full_result(source, target)
    return finalize_result_directory(target, output_mode="compact")
