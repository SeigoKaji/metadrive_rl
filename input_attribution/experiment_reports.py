"""Small, method-specific reports for input-attribution result directories.

The full analysis report is intentionally left as the source of truth for all
rows and diagnostics.  This module only selects the one, comparable slice
needed by the short experiment-01/02 reports.  In particular, it does not
infer an execution state from a configuration flag: an old result may contain
valid rows while its metadata has no ``enabled`` field.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


LabelFormatter = Callable[[str, tuple[str, ...]], str]

_FULL_SCOPE = "full_episode"
_FULL_SLICE = "all"
_MEAN_BASELINE = "mean_over_baselines"
_TOP_N = 5

_TARGET_UNITS = {
    "selected_log_probability": "nats",
    "selected_vs_runner_up_margin": "raw logit差",
    "critic_value": "critic value / return尺度",
}


def _artifact_path(root: Path, legacy_relative_path: str) -> Path:
    """Resolve a legacy artifact through the shared output-layout contract."""

    from .artifact_layout import resolve_artifact

    return Path(resolve_artifact(root, legacy_relative_path))


def _load_json(root: Path, relative_path: str) -> dict[str, Any]:
    try:
        path = _artifact_path(root, relative_path)
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError, TypeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _load_csv(root: Path, relative_path: str) -> list[dict[str, str]]:
    try:
        path = _artifact_path(root, relative_path)
        with path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except (OSError, ValueError, csv.Error, UnicodeError):
        return []


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _method_metadata(metadata: Mapping[str, Any], section: str) -> Mapping[str, Any]:
    direct = _mapping(metadata.get(section))
    grouped = _mapping(_mapping(metadata.get("methods")).get(section))
    if grouped:
        return {**direct, **grouped}
    return direct


def _enabled(metadata: Mapping[str, Any], section: str) -> bool | None:
    explicit_key = f"{section}_executed"
    explicit = metadata.get(explicit_key)
    if isinstance(explicit, bool):
        return explicit
    section_metadata = _method_metadata(metadata, section)
    executed = section_metadata.get("executed")
    if isinstance(executed, bool):
        return executed
    enabled = section_metadata.get("enabled")
    # Without an execution field, retain the documented configuration state:
    # disabled is unexecuted and enabled-but-missing rows are unrecorded.  An
    # absent enabled field (as in old metadata) remains an unknown state.
    return enabled if isinstance(enabled, bool) else None


def _finite(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None or not str(value).strip():
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _is_full_row(row: Mapping[str, Any]) -> bool:
    """Select exactly the report's comparable, all-episode aggregate row."""

    if row.get("scope") != _FULL_SCOPE:
        return False
    if row.get("slice_name") != _FULL_SLICE:
        return False
    if row.get("baseline_scope") != _MEAN_BASELINE:
        return False
    # A missing column is not equivalent to an explicitly empty baseline.
    return "baseline_index" in row and not str(row.get("baseline_index") or "").strip()


def _indices(row: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = row.get(key)
    if value is None:
        return ()
    text = str(value).strip()
    if not text:
        return ()
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        parsed = None
    if isinstance(parsed, Sequence) and not isinstance(parsed, (str, bytes)):
        return tuple(str(item) for item in parsed)
    stripped = text.strip("[]()")
    if not stripped:
        return ()
    delimiter = ";" if ";" in stripped else ","
    return tuple(part.strip() for part in stripped.split(delimiter) if part.strip())


def _escape_cell(value: Any) -> str:
    """Escape a formatter result for a Markdown table cell.

    ``label_formatter`` already returns display-ready text from the compact
    writer.  In particular, it may contain entities such as ``&lt;``; do not
    HTML-escape it a second time.  The character-wise pass preserves existing
    Markdown escapes while protecting labels supplied by simpler callers.
    """

    text = str(value).replace("\r", " ").replace("\n", " ")
    specials = set("\\`|*_[]#")
    output: list[str] = []
    for index, character in enumerate(text):
        if character == "\\":
            following = text[index + 1] if index + 1 < len(text) else ""
            output.append("\\" if following in specials else "\\\\")
        elif character in specials:
            previous = text[index - 1] if index else ""
            output.append(character if previous == "\\" else f"\\{character}")
        else:
            output.append(character)
    return "".join(output)


def _label(
    row: Mapping[str, Any],
    *,
    name_key: str,
    indices_key: str,
    label_formatter: LabelFormatter,
) -> str:
    name = str(row.get(name_key) or row.get("target_name") or "未記録")
    formatted = label_formatter(name, _indices(row, indices_key))
    return _escape_cell(formatted)


def _format_number(value: float | None) -> str:
    return "未記録" if value is None else f"{value:.6g}"


def _format_fraction(value: float | None) -> str:
    if value is None:
        return "未記録"
    # The stored metric is a fraction.  Keep an unexpected value visible
    # rather than silently treating a percentage as a fraction.
    return f"{value * 100:.1f}%" if 0.0 <= value <= 1.0 else _format_number(value)


def _dimension(row: Mapping[str, Any], *, group: bool) -> str:
    key = "group_size" if group and row.get("group_size") not in (None, "") else "target_size"
    value = row.get(key)
    if value is None or not str(value).strip():
        # Each feature-summary row represents one named feature.  This is a
        # property of the artifact schema, not a guessed observation size.
        return "1" if not group else "未記録"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _escape_cell(value)
    return str(int(number)) if math.isfinite(number) and number.is_integer() else _format_number(number)


def _sort_by(rows: Sequence[dict[str, str]], key: str, label_key: str) -> list[dict[str, str]]:
    def order(row: dict[str, str]) -> tuple[int, float, str]:
        value = _finite(row, key)
        label = (
            row.get(label_key)
            or row.get("target_name")
            or row.get("feature_name")
            or row.get("group_name")
            or ""
        )
        return (0 if value is not None else 1, -(value or 0.0), str(label))

    return sorted(rows, key=order)[:_TOP_N]


def _condition_lines(metadata: Mapping[str, Any], experiment_id: str) -> list[str]:
    lines: list[str] = []
    model = _mapping(metadata.get("model"))
    model_path = model.get("path")
    if model_path:
        lines.append(f"- model: {_escape_cell(Path(str(model_path)).name)}")

    schema = _mapping(metadata.get("schema"))
    schema_parts: list[str] = []
    if schema.get("name"):
        schema_parts.append(str(schema["name"]))
    if schema.get("observation_dim") is not None:
        schema_parts.append(f"dim={schema['observation_dim']}")
    if schema_parts:
        lines.append(f"- schema: {_escape_cell(', '.join(schema_parts))}")

    baseline = _mapping(metadata.get("baseline"))
    baseline_parts: list[str] = []
    for key in ("strategy", "count"):
        if baseline.get(key) is not None and str(baseline.get(key)).strip():
            baseline_parts.append(f"{key}={baseline[key]}")
    if baseline_parts:
        lines.append(f"- baseline: {_escape_cell(', '.join(baseline_parts))}")

    rollout = _mapping(metadata.get("rollout"))
    seed_range = _mapping(metadata.get("scenario_seed_range"))
    scenario_count = seed_range.get("count")
    if scenario_count is None:
        scenario_count = rollout.get("scenario_count")
    if scenario_count is not None and str(scenario_count).strip():
        lines.append(f"- scenarios: {_escape_cell(scenario_count)}")

    if experiment_id == "02":
        ig = _method_metadata(metadata, "integrated_gradients")
        targets = ig.get("targets")
        if isinstance(targets, Sequence) and not isinstance(targets, (str, bytes)):
            target_text = ", ".join(str(target) for target in targets)
            if target_text:
                lines.append(f"- IG targets: {_escape_cell(target_text)}")
        if ig.get("steps") is not None and str(ig.get("steps")).strip():
            lines.append(f"- IG steps: {_escape_cell(ig['steps'])}")
    return lines


def _state_text(enabled: bool | None, has_rows: bool) -> str:
    if enabled is False:
        return "未実施"
    if enabled is True:
        return "実施済みデータあり" if has_rows else "未記録"
    return "保存済み結果を表示（実行状態の項目は旧metadataに未記録）" if has_rows else "実施状態不明（保存データなし）"


def _state_section(enabled: bool | None, has_rows: bool) -> str:
    return "## 状態\n\n- " + _state_text(enabled, has_rows) + "\n"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _perturbation_table(
    title: str,
    rows: Sequence[dict[str, str]],
    *,
    group: bool,
    label_formatter: LabelFormatter,
) -> str:
    if not rows:
        return f"### {title}\n\n未記録（比較可能な全体行がありません）。\n"
    rankable = [row for row in rows if _finite(row, "mean_js_divergence") is not None]
    if not rankable:
        return f"### {title}\n\n未記録（JSDがありません）。\n"
    chosen = _sort_by(rankable, "mean_js_divergence", "target_name" if group else "feature_name")
    body: list[list[str]] = []
    for row in chosen:
        body.append(
            [
                _label(
                    row,
                    name_key="group_name" if group else "feature_name",
                    indices_key="group_indices" if group else "feature_index",
                    label_formatter=label_formatter,
                ),
                _dimension(row, group=group),
                _format_number(_finite(row, "mean_js_divergence")),
                _format_fraction(_finite(row, "action_flip_rate")),
            ]
        )
    return (
        f"### {title}\n\n"
        + _table(("対象", "次元数", "JSD (nats)", "action変化率"), body)
        + "\n"
    )


def _unit(target: str) -> str:
    return _TARGET_UNITS.get(target, "単位不明（metadata/詳細を確認）")


def _ig_table(
    title: str,
    rows: Sequence[dict[str, str]],
    *,
    group: bool,
    label_formatter: LabelFormatter,
) -> str:
    if not rows:
        return f"### {title}\n\n未記録（比較可能な全体行がありません）。\n"
    absolute_key = "group_absolute_mass" if group else "mean_absolute_ig"
    signed_key = "group_signed_ig" if group else "mean_signed_ig"
    rankable = [row for row in rows if _finite(row, absolute_key) is not None]
    if not rankable:
        return f"### {title}\n\n未記録（寄与値がありません）。\n"
    ordered = _sort_by(rankable, absolute_key, "group_name" if group else "feature_name")
    body: list[list[str]] = []
    for row in ordered:
        body.append(
            [
                _label(
                    row,
                    name_key="group_name" if group else "feature_name",
                    indices_key="group_indices" if group else "feature_index",
                    label_formatter=label_formatter,
                ),
                _dimension(row, group=group),
                _format_number(_finite(row, absolute_key)),
                _format_number(_finite(row, signed_key)),
            ]
        )
    return (
        f"### {title}\n\n"
        + _table(("対象", "次元数", "絶対値寄与", "符号付き寄与"), body)
        + "\n"
    )


def _footer(details_link: str) -> str:
    link = str(details_link).strip()
    details = f"[詳細な全件・rawデータ]({link})" if link else "詳細な全件・rawデータ（リンク未指定）"
    return f"## 参照\n\n- {details}\n- [全体の走行比較](../report.md)\n"


def _build_perturbation_report(
    root: Path,
    metadata: Mapping[str, Any],
    *,
    label_formatter: LabelFormatter,
    details_link: str,
) -> str:
    feature_rows = [
        row for row in _load_csv(root, "perturbation_feature_summary.csv") if _is_full_row(row)
    ]
    group_rows = [
        row for row in _load_csv(root, "perturbation_group_summary.csv") if _is_full_row(row)
    ]
    enabled = _enabled(metadata, "perturbation")
    data_rows = feature_rows + group_rows
    parts = [
        "# 実験01 入力置換",
        "",
        "## 目的と方法",
        "",
        "観測の入力1項目（feature）または入力群（group）を比較の基準値（baseline）に置換し、行動の選択確率（actor出力分布）の変化を調べます。JSDは分布差（nats）、action変化率は選択actionが変わった割合です。以下は全走行・baseline平均の集計で、報酬の因果効果ではありません。",
        "",
        "## 条件",
        "",
    ]
    conditions = _condition_lines(metadata, "01")
    parts.append("\n".join(conditions) if conditions else "条件はmetadataに記録されていません。")
    parts.extend(
        [
            "",
            _state_section(enabled, bool(data_rows)).rstrip("\n"),
        ]
    )
    if enabled is not False and data_rows:
        semantic_groups = [row for row in group_rows if row.get("target_kind") == "group"]
        lidar_groups = [row for row in group_rows if row.get("target_kind") == "lidar_sector"]
        parts.extend(
            [
                "",
                _perturbation_table("単一feature（入力1項目）上位5件", feature_rows, group=False, label_formatter=label_formatter).rstrip("\n"),
                "",
                _perturbation_table("semantic group（入力群）上位5件", semantic_groups, group=True, label_formatter=label_formatter).rstrip("\n"),
                "",
                _perturbation_table("LiDAR sector（入力群）上位5件", lidar_groups, group=True, label_formatter=label_formatter).rstrip("\n"),
            ]
        )
    elif enabled is not False:
        parts.extend(["", "比較可能な全体行は未記録です。"])
    parts.extend(["", _footer(details_link).rstrip("\n")])
    return "\n".join(parts).rstrip() + "\n"


def _target_order(metadata: Mapping[str, Any], rows: Sequence[dict[str, str]]) -> list[str]:
    configured = _method_metadata(metadata, "integrated_gradients").get("targets")
    result: list[str] = []
    if isinstance(configured, Sequence) and not isinstance(configured, (str, bytes)):
        result.extend(str(target) for target in configured if str(target).strip())
    for row in rows:
        target = str(row.get("target") or "").strip()
        if target and target not in result:
            result.append(target)
    return result


def _build_ig_report(
    root: Path,
    metadata: Mapping[str, Any],
    *,
    label_formatter: LabelFormatter,
    details_link: str,
) -> str:
    feature_rows = [row for row in _load_csv(root, "ig_feature_summary.csv") if _is_full_row(row)]
    group_rows = [row for row in _load_csv(root, "ig_group_summary.csv") if _is_full_row(row)]
    all_rows = feature_rows + group_rows
    enabled = _enabled(metadata, "integrated_gradients")
    parts = [
        "# 実験02 Integrated Gradients",
        "",
        "## 目的と方法",
        "",
        "基準観測での出力と現在の観測での出力の差を、各入力へ割り当てます。調べる出力（IG target）ごとに寄与を分けます。絶対値寄与は大きさ、符号付き寄与は増減方向を表し、＋は調べる出力を増やす向き、−は減らす向きです（運転の良し悪しではありません）。以下は全走行・baseline平均の集計で、targetごとに単位が異なるためtarget間では大小を比較しません。",
        "",
        "## 条件",
        "",
    ]
    conditions = _condition_lines(metadata, "02")
    parts.append("\n".join(conditions) if conditions else "条件はmetadataに記録されていません。")
    parts.extend(["", _state_section(enabled, bool(all_rows)).rstrip("\n")])
    if enabled is not False and all_rows:
        for target in _target_order(metadata, all_rows):
            target_features = [row for row in feature_rows if row.get("target") == target]
            target_groups = [row for row in group_rows if row.get("target") == target]
            semantic_groups = [row for row in target_groups if row.get("group_kind") == "group"]
            lidar_groups = [row for row in target_groups if row.get("group_kind") == "lidar_sector"]
            parts.extend(
                [
                    "",
                    f"## target: {_escape_cell(target)}（単位: {_escape_cell(_unit(target))}）",
                    "",
                    _ig_table("feature（入力1項目）上位5件", target_features, group=False, label_formatter=label_formatter).rstrip("\n"),
                    "",
                    _ig_table("semantic group（入力群）上位5件", semantic_groups, group=True, label_formatter=label_formatter).rstrip("\n"),
                    "",
                    _ig_table("LiDAR sector（入力群）上位5件", lidar_groups, group=True, label_formatter=label_formatter).rstrip("\n"),
                ]
            )
    elif enabled is not False:
        parts.extend(["", "比較可能な全体行は未記録です。"])
    parts.extend(["", _footer(details_link).rstrip("\n")])
    return "\n".join(parts).rstrip() + "\n"


def build_method_report(
    result_directory: str | Path,
    experiment_id: str,
    *,
    label_formatter: LabelFormatter,
    details_link: str,
) -> str:
    """Build the short report for experiment ``01`` or ``02``.

    ``label_formatter`` is injected by the compact writer so this module does
    not depend on its target-label vocabulary.  ``details_link`` is also
    caller-supplied because compact and full layouts place the archive at
    different relative paths.
    """

    normalized_id = str(experiment_id).zfill(2)
    if normalized_id not in {"01", "02"}:
        raise ValueError("experiment_id must be '01' or '02'")
    root = Path(result_directory)
    metadata = _load_json(root, "analysis_metadata.json")
    if normalized_id == "01":
        return _build_perturbation_report(
            root,
            metadata,
            label_formatter=label_formatter,
            details_link=details_link,
        )
    return _build_ig_report(
        root,
        metadata,
        label_formatter=label_formatter,
        details_link=details_link,
    )


__all__ = ["build_method_report"]
