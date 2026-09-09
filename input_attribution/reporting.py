"""保存済み入力帰属結果から日本語レポートを作る小さな標準ライブラリ層。

このモジュールは MetaDrive、Stable-Baselines3、PyTorch、Captum を import
しません。``run`` の実行時に作られた JSON/JSONL/CSV/NPZ を読むだけなので、
環境を再起動できない PC でも ``report`` を再生成できます。実験側の保存形式は
段階的に拡張されるため、ここではファイル名と列名を少数の別名で受け付けますが、
値が無い項目を 0 と推測することはしません。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote


class ReportingError(ValueError):
    """保存結果をレポートに変換できない場合のエラー。"""


_MISSING = object()
_NA = "N/A"
_BASELINE_PATTERN_IDS = frozenset({"P00", "P0", "BASELINE", "NOOP", "NONE"})
_SUMMARY_FIELDS = (
    "pattern_id",
    "pattern_name",
    "changed_indices",
    "changed_dimension_count",
    "replacement",
    "offline_status",
    "offline_reason",
    "offline_applied_count",
    "offline_changed_count",
    "offline_noop_count",
    "offline_action_change_rate",
    "offline_episode_mean_action_change_rate",
    "offline_step_weighted_action_change_rate",
    "offline_selected_probability_delta_pp",
    "offline_js_divergence",
    "closed_loop_status",
    "closed_loop_reason",
    "closed_loop_episode_count",
    "closed_loop_natural_episode_count",
    "closed_loop_evaluable_episode_count",
    "closed_loop_interrupted_episode_count",
    "closed_loop_unevaluable_episode_count",
    "closed_loop_interrupted_reasons",
    "closed_loop_lane_rms_m",
    "closed_loop_lane_max_abs_m",
    "closed_loop_departure_count",
    "closed_loop_departure_time_s",
    "closed_loop_first_departure_time_s",
    "closed_loop_ever_departed",
    "closed_loop_arrived",
    "closed_loop_arrival_rate",
    "closed_loop_arrival_count",
    "closed_loop_arrival_episode_count",
    "closed_loop_wrong_lane_arrival",
    "closed_loop_start_lane_departure",
    "closed_loop_initial_comparison_verified",
    "closed_loop_paired_p00_verified",
    "closed_loop_road_out",
    "closed_loop_crash",
    "closed_loop_progress",
    "closed_loop_speed_mean_mps",
    "closed_loop_low_speed_duration_s",
    "closed_loop_duration_s",
    "closed_loop_valid_count",
    "closed_loop_valid_time_s",
    "closed_loop_poststep_count",
    "closed_loop_valid_rate",
    "closed_loop_stop_fraction",
    "closed_loop_termination",
    "closed_loop_cumulative_reward",
    "closed_loop_action_switch_count",
    "p00_delta_lane_rms_m",
    "p00_delta_progress",
    # New report-facing fields are appended so old ``summary.csv`` readers
    # continue to find the legacy columns in the same order.
    "offline_target_count",
    "offline_eligible_count",
    "offline_skipped_count",
    "offline_actual_action_change_count",
    "offline_actual_action_change_rate",
    "offline_actual_selected_probability_delta_pp",
    "offline_actual_selected_probability_delta_abs_pp",
    "offline_actual_js_divergence",
    "offline_assessment_status",
    "offline_execution_status",
    "offline_meaningful_changed_count",
    "offline_declared_tolerance",
    "offline_runtime_assessment_status",
    "offline_runtime_execution_status",
    "offline_scope",
    "offline_on_inapplicable",
    "variant_id",
    "variant_ids",
    "variant_classification",
    "variant_evidence",
    "resolved_values",
    "resolved_expression",
    "closed_loop_target_step_count",
    "closed_loop_eligible_count",
    "closed_loop_applied_count",
    "closed_loop_changed_count",
    "closed_loop_noop_count",
    "closed_loop_skipped_count",
    "closed_loop_meaningful_changed_count",
    "closed_loop_paired_episode_count",
    "closed_loop_paired_missing_count",
    "closed_loop_paired_missing_reasons",
    "closed_loop_assessment_status",
    "closed_loop_execution_status",
    "closed_loop_failed_episode_count",
    "closed_loop_aborted_episode_count",
    "closed_loop_measurement_status",
    "closed_loop_lane_measurement_detail",
    "closed_loop_departure_measurement_detail",
    "closed_loop_low_speed_measurement_detail",
)


# The human-facing tables intentionally stay small.  Every field not shown
# here is retained in summary.csv, report_details.json and the detail CSVs.
_OFFLINE_MAIN_COLUMNS = (
    ("pattern_id", "パターン(ID・対象)"),
    ("replacement", "変更ルール・値"),
    ("offline_coverage", "適用件数/全対象"),
    ("offline_change_coverage", "実変更件数/適用"),
    ("offline_action_change_coverage", "行動変更件数/実変更"),
    ("offline_actual_selected_probability_delta_abs_pp", "選択確率差平均絶対値(pp、実変更時)"),
    ("offline_actual_js_divergence", "JS平均(nats、実変更時)"),
    ("offline_assessment_status", "評価状態"),
)

_CLOSED_MAIN_COLUMNS = (
    ("pattern_id", "パターン(ID・対象)"),
    ("closed_loop_coverage", "適用/全対象・実変更/適用"),
    ("closed_loop_outcome", "到達件数/走行数・終了理由"),
    ("closed_loop_lane_summary", "横ずれRMS(m、検証済P00差)"),
    ("closed_loop_departure_time_s", "逸脱時間(s)"),
    ("closed_loop_progress", "進行距離(m)"),
    ("closed_loop_duration_s", "走行時間(s)"),
    ("closed_loop_assessment_status", "評価状態"),
)


@dataclass(frozen=True)
class ReportResult:
    """レポート生成の成果物と機械可読な集計。"""

    run_dir: Path
    output_dir: Path
    status: str
    summary_rows: tuple[dict[str, Any], ...]
    files: tuple[Path, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "output_dir": str(self.output_dir),
            "status": self.status,
            "summary_rows": [dict(row) for row in self.summary_rows],
            "files": [str(path) for path in self.files],
        }


def _plain(value: Any) -> Any:
    """NumPy 値を任意依存なしで JSON/表示可能な値へ変換する。"""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        try:
            return _plain(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        try:
            return _plain(value.tolist())
        except (TypeError, ValueError):
            pass
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return str(value)


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _is_baseline_pattern(value: Any) -> bool:
    return str(value or "").strip().upper() in _BASELINE_PATTERN_IDS


def _first(mapping: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None and str(mapping[name]).strip() != "":
            return mapping[name]
    return default


def _first_nested(mapping: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    direct = _first(mapping, *names, default=_MISSING)
    if direct is not _MISSING:
        return direct
    wanted = set(names)
    queue: list[Mapping[str, Any]] = [mapping]
    visited: set[int] = set()
    while queue:
        current = queue.pop(0)
        if id(current) in visited:
            continue
        visited.add(id(current))
        for key, value in current.items():
            if key in wanted and value is not None:
                return value
            if isinstance(value, Mapping):
                queue.append(value)
            elif isinstance(value, list):
                queue.extend(item for item in value if isinstance(item, Mapping))
    return default


def _as_rows(value: Any) -> list[dict[str, Any]]:
    """Mapping/list/column-oriented valuesを rows に統一する。"""

    if value is None:
        return []
    if isinstance(value, Mapping):
        # A named ``rows``/``records`` member is the most common JSON shape.
        for key in ("rows", "records", "results", "patterns", "features", "data"):
            nested = value.get(key)
            if isinstance(nested, list):
                rows = _as_rows(nested)
                if rows:
                    return rows
            if key == "patterns" and isinstance(nested, Mapping):
                rows = []
                for identifier, item in nested.items():
                    if isinstance(item, Mapping):
                        rows.append({"pattern_id": str(identifier), **dict(item)})
                if rows:
                    return rows
        sequence_values = [
            item for item in value.values()
            if isinstance(item, (list, tuple)) and not isinstance(item, (str, bytes))
        ]
        if sequence_values and all(len(item) == len(sequence_values[0]) for item in sequence_values):
            size = len(sequence_values[0])
            rows: list[dict[str, Any]] = []
            for index in range(size):
                rows.append({
                    str(key): item[index] if isinstance(item, (list, tuple)) else item
                    for key, item in value.items()
                })
            return rows
        return [dict(value)]
    if isinstance(value, (list, tuple)):
        rows = []
        for item in value:
            if isinstance(item, Mapping):
                rows.append(dict(item))
            elif hasattr(item, "_asdict"):
                rows.append(dict(item._asdict()))
        return rows
    return []


def _read_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows.extend(_as_rows(value))
    except (OSError, UnicodeError):
        return []
    return rows


def _read_csv(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except (OSError, UnicodeError, csv.Error):
        return []


def _read_npz(path: Path) -> dict[str, Any]:
    """Read only small metadata arrays when NumPy is present.

    The report remains usable without NumPy; in that case NPZ is simply
    mentioned as an available artifact and no unsupported metric is invented.
    """

    try:
        import numpy as np  # type: ignore
    except ImportError:
        return {}
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {str(key): archive[key] for key in archive.files}
    except (OSError, ValueError, TypeError):
        return {}


def _npz_rows(arrays: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Convert column-like NPZ arrays into records without padding/truncation."""

    lengths = []
    for value in arrays.values():
        shape = getattr(value, "shape", ())
        if len(shape) >= 1:
            lengths.append(int(shape[0]))
    if not lengths or len(set(lengths)) != 1:
        return []
    count = lengths[0]
    rows: list[dict[str, Any]] = []
    for index in range(count):
        row: dict[str, Any] = {}
        for key, value in arrays.items():
            try:
                item = value[index]
                if hasattr(item, "shape") and len(item.shape) > 0:
                    item = item.tolist()
                elif hasattr(item, "item"):
                    item = item.item()
                row[str(key)] = _plain(item)
            except (IndexError, TypeError, ValueError):
                continue
        rows.append(row)
    return rows


def _safe_relative(path: Path, root: Path) -> str | None:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    if any(part in {"", ".", ".."} for part in relative.parts):
        return None
    return PurePosixPath(*relative.parts).as_posix()


def _parse_indices(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        result: list[int] = []
        for item in value:
            try:
                result.append(int(item))
            except (TypeError, ValueError):
                continue
        return result
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        parsed = None
    if isinstance(parsed, list):
        return _parse_indices(parsed)
    return [int(item) for item in re.findall(r"-?\d+", text)]


def _json_value(value: Any) -> Any:
    """Parse JSON-shaped CSV cells while leaving ordinary text unchanged."""

    if isinstance(value, (list, tuple, Mapping)):
        return value
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return value


def _pattern_label_from_names(identifier: str, names: Sequence[Any]) -> str | None:
    clean = [str(name).strip() for name in names if str(name).strip()]
    if len(clean) == 1:
        return clean[0]
    if len(clean) > 1:
        # Group patterns carry one label per changed dimension.  A compact
        # group label keeps the report readable while retaining the dimension
        # count in the table; the full names remain in patterns.csv/json.
        prefixes = {re.sub(r"\s*\d+\s*$", "", name).strip() for name in clean}
        prefixes.discard("")
        if len(prefixes) == 1:
            return f"{next(iter(prefixes))}（{len(clean)}次元）"
        return f"{clean[0]} 等（{len(clean)}次元）"
    if identifier.upper() in {"P00", "P0", "BASELINE", "NOOP", "NONE"}:
        return "変更なし"
    return None


def _pattern_labels(run_dir: Path) -> dict[str, str]:
    """Read the saved pattern/input snapshots used by the runtime.

    The analysis result intentionally stores a compact pattern object and may
    omit Japanese labels.  ``patterns.json``/``patterns.csv`` and
    ``input_schema.csv`` are the immutable snapshots that complete that
    presentation metadata.
    """

    schema_by_index: dict[int, str] = {}
    schema_path = run_dir / "input_schema.csv"
    if schema_path.is_file():
        for row in _read_csv(schema_path):
            try:
                index = int(row.get("index", ""))
            except (TypeError, ValueError):
                continue
            name = str(row.get("name_ja", "")).strip()
            if name:
                schema_by_index[index] = name

    catalog: list[dict[str, Any]] = []
    json_path = run_dir / "patterns.json"
    if json_path.is_file():
        payload = _read_json(json_path)
        catalog = _as_rows(payload)
    if not catalog:
        csv_path = run_dir / "patterns.csv"
        if csv_path.is_file():
            catalog = _read_csv(csv_path)

    labels: dict[str, str] = {}
    for row in catalog:
        identifier = str(_first(row, "pattern_id", "id", default="")).strip()
        if not identifier:
            continue
        indices = _parse_indices(_first(row, "indices", "requested_indices", "target_indices"))
        raw_names = _json_value(_first(row, "names_ja", "pattern_names_ja", "names", default=None))
        names = raw_names if isinstance(raw_names, (list, tuple)) else []
        names = [str(name) for name in names if str(name).strip()]
        if not names and indices:
            names = [schema_by_index[index] for index in indices if index in schema_by_index]
        label = _pattern_label_from_names(identifier, names)
        if label is None:
            description = str(_first(row, "description", "label", default="")).strip()
            if description:
                label = description.split(":", 1)[0].strip()
        if label is None:
            label = identifier
        labels[identifier] = label

    return labels


def _label(row: Mapping[str, Any], labels: Mapping[str, str] | None = None) -> str:
    value = _first(
        row,
        "pattern_name_ja",
        "name_ja",
        "label_ja",
        "display_name",
        "target_name_ja",
        "pattern_name",
        "target_name",
        "group_name",
        "name",
        "label",
        default=None,
    )
    identifier = _pattern_id(row)
    if value is None or str(value).strip() in {"", "未記録", "unknown"}:
        if labels and identifier in labels:
            return str(labels[identifier])
        return "未記録"
    return str(value)


def _pattern_id(row: Mapping[str, Any]) -> str:
    value = _first(row, "pattern_id", "target_id", "id", "pattern", "target", default="unknown")
    if isinstance(value, Mapping):
        value = _first(value, "pattern_id", "target_id", "id", default="unknown")
    return str(value)


def _nested(row: Mapping[str, Any], *names: str) -> Mapping[str, Any]:
    for name in names:
        value = row.get(name)
        if isinstance(value, Mapping):
            return value
    return {}


def _metric(row: Mapping[str, Any], *names: str) -> float | None:
    value = _first(row, *names)
    if value is not None:
        number = _number(value)
        if number is not None:
            return number
        if isinstance(value, Mapping):
            for statistic in ("mean", "average", "median", "p95", "value"):
                number = _number(value.get(statistic))
                if number is not None:
                    return number
    # Runtime artifacts intentionally retain nested ``summary`` and telemetry
    # sections.  Search their mappings without treating absent values as zero.
    wanted = set(names)
    stack: list[Mapping[str, Any]] = [row]
    visited: set[int] = set()
    while stack:
        current = stack.pop(0)
        if id(current) in visited:
            continue
        visited.add(id(current))
        for key, child_value in current.items():
            if key in wanted:
                number = _number(child_value)
                if number is not None:
                    return number
                if isinstance(child_value, Mapping):
                    for statistic in ("mean", "average", "median", "p95", "value"):
                        number = _number(child_value.get(statistic))
                        if number is not None:
                            return number
            if isinstance(child_value, Mapping):
                stack.append(child_value)
            elif isinstance(child_value, list):
                stack.extend(item for item in child_value if isinstance(item, Mapping))
    return None


def _status(row: Mapping[str, Any], default: str = "未記録") -> str:
    value = _first(row, "status", "state", "result_status", default=default)
    if isinstance(value, Mapping):
        value = _first(value, "status", "state", default=default)
    return str(value)


def _reason_parts(row: Mapping[str, Any]) -> list[str]:
    """Return saved execution/telemetry reasons as de-duplicated parts."""

    values: list[str] = []

    def add(value: Any) -> None:
        if isinstance(value, (list, tuple)):
            values.extend(str(item).strip() for item in value if str(item).strip())
        elif value not in (None, ""):
            values.append(str(value).strip())

    for name in ("skip_reason", "reason", "message", "error"):
        add(row.get(name))
    for key in ("skip_reasons", "reasons", "missing_reasons", "unavailable_reasons"):
        add(_first_nested(row, key, default=None))

    return list(dict.fromkeys(value for value in values if value))


def _reason(row: Mapping[str, Any]) -> str:
    """Return a saved execution/telemetry reason without inventing one."""

    direct = _first(row, "skip_reason", "reason", "message", "error", default=None)
    if direct not in (None, ""):
        if isinstance(direct, (list, tuple)):
            values = [str(item).strip() for item in direct if str(item).strip()]
            if values:
                return "; ".join(dict.fromkeys(values))
        else:
            return str(direct)
    for key in ("skip_reasons", "reasons", "missing_reasons", "unavailable_reasons"):
        value = _first_nested(row, key, default=None)
        if isinstance(value, (list, tuple)):
            values = [str(item).strip() for item in value if str(item).strip()]
            if values:
                return "; ".join(dict.fromkeys(values))
        elif value not in (None, ""):
            return str(value)
    return ""


def _summary_mapping(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("summary")
    return value if isinstance(value, Mapping) else {}


def _count_text(value: Any) -> str | None:
    number = _number(value)
    if number is None:
        return None
    if number.is_integer():
        return str(int(number))
    return _fmt(number)


def _offline_reason_parts(row: Mapping[str, Any], extra_rows: Iterable[Mapping[str, Any]] = ()) -> list[str]:
    """Collect canonical and per-step reasons while retaining unknown errors."""

    values = _reason_parts(row)
    for child in row.get("rows", ()) if isinstance(row.get("rows"), list) else ():
        if isinstance(child, Mapping):
            values.extend(_reason_parts(child))
    for child in extra_rows:
        values.extend(_reason_parts(child))
    return list(dict.fromkeys(value for value in values if value))


def _offline_coverage_text(applied: Any, skipped: Any, valid: Any) -> str:
    """Describe the measured/skipped coverage when both sides are known."""

    skipped_text = _count_text(skipped)
    if skipped_text is None or _number(skipped) in (None, 0):
        return ""
    compared = _number(applied)
    if compared is None:
        compared = _number(valid)
    if compared is None:
        return f"{skipped_text}件を除外（比較件数不明）"
    compared_text = _count_text(compared) or str(compared)
    total = compared + float(_number(skipped) or 0.0)
    total_text = _count_text(total) or str(total)
    return f"{skipped_text}件を除外。{compared_text}/{total_text}件で比較"


def _format_offline_reason(
    reasons: Sequence[str],
    *,
    applied: Any,
    skipped: Any,
    valid: Any,
) -> str:
    """Render skip reasons compactly without hiding unrelated errors.

    Runtime summaries retain one raw reason for every distinct incompatible
    context.  Those raw values are useful in ``raw_offline`` but make the
    human-facing report unreadable, so repeated known compatibility reasons
    are collapsed and the measured/skipped coverage is shown explicitly.
    """

    values = list(dict.fromkeys(str(reason).strip() for reason in reasons if str(reason).strip()))
    if not values:
        return ""
    categories: dict[str, list[str]] = {
        "road_mismatch": [],
        "road_missing": [],
        "lane_mismatch": [],
        "lane_missing": [],
    }
    known: set[str] = set()
    for value in values:
        # A saved reason may itself contain multiple messages.  Treat such a
        # value as unknown so the original diagnostic is not hidden by a
        # compatibility summary.
        if ";" in value:
            continue
        if value.startswith("compatibility context 'road_segment_id' differs:"):
            categories["road_mismatch"].append(value)
            known.add(value)
        elif value in {
            "compatibility context is missing 'road_segment_id'",
            "compatibility context 'road_segment_id' is null",
        }:
            categories["road_missing"].append(value)
            known.add(value)
        elif value.startswith("compatibility context 'target_lane_ordinal' differs:"):
            categories["lane_mismatch"].append(value)
            known.add(value)
        elif value in {
            "compatibility context is missing 'target_lane_ordinal'",
            "compatibility context 'target_lane_ordinal' is null",
        }:
            categories["lane_missing"].append(value)
            known.add(value)
    coverage = _offline_coverage_text(applied, skipped, valid)
    labels = []
    if categories["road_mismatch"]:
        labels.append("参照観測と道路区間が異なる")
    if categories["road_missing"]:
        labels.append("道路区間情報がない")
    if categories["lane_mismatch"]:
        labels.append("参照観測と目標レーンが異なる")
    if categories["lane_missing"]:
        labels.append("目標レーン情報がない")
    unknown = [value for value in values if value not in known]
    rendered: list[str] = []
    if labels:
        if len(labels) == 1 and not unknown:
            rendered.append(labels[0] + coverage)
            coverage = ""
        else:
            rendered.append("、".join(labels))
    if coverage:
        rendered.append(coverage)
    if unknown:
        rendered.append("理由: " + "；".join(unknown))
    return "。".join(rendered) if rendered else "; ".join(values)


def _mapping_stat(value: Any) -> float | None:
    if isinstance(value, Mapping):
        for key in ("mean", "average", "value", "median", "p95"):
            number = _number(value.get(key))
            if number is not None:
                return number
        return None
    return _number(value)


def _summary_top_metric(row: Mapping[str, Any], *names: str) -> float | None:
    """Read an aggregate field before traversing per-step records.

    ``result.json`` contains both ``rows`` and ``summary``.  A breadth-first
    generic lookup can encounter a skipped row's zero-valued probability
    before the summary's explicit null.  Aggregate fields therefore get a
    narrow, ordered lookup first.
    """

    summary = _summary_mapping(row)
    for mapping in (summary, row):
        for name in names:
            if name in mapping and mapping[name] is not None:
                value = _mapping_stat(mapping[name])
                if value is not None:
                    return value
    return None


def _summary_intervention_metric(row: Mapping[str, Any], *names: str) -> float | None:
    """Read canonical intervention counts before legacy summary aliases.

    New offline summaries keep the authoritative A counts under
    ``summary.intervention_counts``.  Looking there first prevents a
    diagnostic ``valid_count`` or a per-step fallback from changing the
    denominator.  Older artifacts have no such section and fall through to
    the top-level summary aliases.
    """

    summary = _summary_mapping(row)
    candidates: list[Mapping[str, Any]] = []
    for container in (summary.get("intervention_counts"), row.get("intervention_counts")):
        if isinstance(container, Mapping):
            candidates.append(container)
    candidates.extend((summary, row))
    for candidate in candidates:
        for name in names:
            if name in candidate and candidate[name] is not None:
                value = _mapping_stat(candidate[name])
                if value is not None:
                    return value
    return None


def _summary_text_value(row: Mapping[str, Any], *names: str) -> Any:
    """Read a scalar status/definition from the canonical summary first."""

    summary = _summary_mapping(row)
    for mapping in (summary, row):
        value = _first(mapping, *names, default=_MISSING)
        if value is not _MISSING:
            return value
    return None


def _declared_tolerance(row: Mapping[str, Any]) -> Any:
    """Return the saved intervention tolerance without inventing one."""

    # The pattern definition is authoritative.  ``_first_nested`` is used
    # only as a legacy fallback for artifacts that kept it under metadata.
    value = _first(row, "tolerance", "meaningful_tolerance", default=_MISSING)
    if value is not _MISSING:
        return value
    value = _first_nested(row, "tolerance", "meaningful_tolerance", default=None)
    return None if isinstance(value, Mapping) and not value else value


def _summary_section_metric(row: Mapping[str, Any], section: str, *names: str) -> float | None:
    summary = _summary_mapping(row)
    value = summary.get(section)
    if not isinstance(value, Mapping):
        return None
    for name in names:
        if name in value and value[name] is not None:
            number = _mapping_stat(value[name])
            if number is not None:
                return number
    return None


def _summary_section_mapping(row: Mapping[str, Any], section: str) -> Mapping[str, Any]:
    """Return one aggregate section without traversing diagnostic rows."""

    value = _summary_mapping(row).get(section)
    return value if isinstance(value, Mapping) else {}


def _probability_delta_pp(value: Any, *, fraction: bool = False) -> float | None:
    """Normalize a saved probability delta to percentage points by field key.

    ``selected_probability_delta`` is a probability fraction in runtime
    artifacts, while ``selected_probability_delta_pp`` already uses percentage
    points.  The caller supplies the key semantics explicitly; magnitude is
    never used to guess a unit (a tiny 0.01 pp value must stay 0.01 pp).
    """

    number = _number(value)
    if number is None:
        return None
    return number * 100.0 if fraction else number


def _offline_child_rows(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract per-step offline rows from canonical and legacy containers."""

    for key in ("rows", "records", "steps"):
        value = row.get(key)
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, Mapping)]
    return []


def _offline_row_changed(row: Mapping[str, Any]) -> bool | None:
    intervention = row.get("intervention")
    if isinstance(intervention, Mapping):
        value = _bool_value(_first(intervention, "changed", "actual_input_changed", default=None))
        if value is not None:
            return value
        indices = _first(intervention, "changed_indices", "actual_changed_indices", default=None)
        if indices is not None:
            return bool(_parse_indices(indices))
    value = _bool_value(_first(row, "actual_input_changed", "changed", "input_changed", default=None))
    if value is not None:
        return value
    indices = _first(row, "changed_indices", "actual_changed_indices", default=None)
    if indices is not None:
        return bool(_parse_indices(indices))
    return None


def _offline_row_skipped(row: Mapping[str, Any]) -> bool | None:
    intervention = row.get("intervention")
    if isinstance(intervention, Mapping):
        value = _bool_value(_first(intervention, "skipped", "skip", default=None))
        if value is not None:
            return value
    return _bool_value(_first(row, "skipped", "skip", default=None))


def _offline_raw_stats(row: Mapping[str, Any]) -> dict[str, Any]:
    """Calculate changed-only effects from saved per-step rows.

    The aggregate summary is authoritative for counts.  This fallback is for
    legacy rows and adds absolute/signed values without turning missing data
    into zero.
    """

    changed_rows = []
    for child in _offline_child_rows(row):
        skipped = _offline_row_skipped(child)
        changed = _offline_row_changed(child)
        if skipped is True or changed is not True:
            continue
        changed_rows.append(child)
    probability: list[float] = []
    js: list[float] = []
    action_changed = 0
    action_known = 0
    for child in changed_rows:
        value = _probability_delta_pp(_first(child, "selected_probability_delta_pp", default=None))
        if value is None:
            value = _probability_delta_pp(_first(child, "selected_probability_delta", "selected_action_probability_delta", default=None), fraction=True)
        if value is not None:
            probability.append(value)
        js_value = _number(_first(child, "js_divergence", "js", "mean_js_divergence", default=None))
        if js_value is not None:
            js.append(js_value)
        action = _bool_value(_first(child, "action_changed", "argmax_changed", default=None))
        if action is not None:
            action_known += 1
            action_changed += int(action)
    return {
        "count": len(changed_rows),
        "probability_signed_pp": sum(probability) / len(probability) if probability else None,
        "probability_abs_pp": sum(abs(value) for value in probability) / len(probability) if probability else None,
        "js": sum(js) / len(js) if js else None,
        "action_changed_count": action_changed if action_known else None,
        "action_known_count": action_known,
    }


def _offline_derived_fields(
    row: Mapping[str, Any],
    *,
    applied: float | None,
    changed: float | None,
    noop: float | None,
    skipped: float | None,
    valid: float | None,
) -> dict[str, Any]:
    """Build independent A coverage and changed-only metrics."""

    all_section = _summary_section_mapping(row, "all_timestamps")
    actual_section = _summary_section_mapping(row, "actual_input_changes_only")
    raw = _offline_raw_stats(row)
    target = _summary_intervention_metric(row, "target_step_count", "target_count", "total_target_count")
    if target is None:
        target = _summary_top_metric(
            row,
            "target_step_count",
            "target_count",
            "total_target_count",
        )
    if target is None and applied is not None and skipped is not None:
        target = applied + skipped
    if target is None:
        target = _mapping_stat(all_section.get("count"))
    if target is None:
        target = valid
    eligible = _summary_intervention_metric(row, "eligible_count", "eligible_row_count", "applicable_count")
    if eligible is None:
        eligible = _summary_top_metric(row, "eligible_row_count", "eligible_count", "valid_row_count", "applicable_count")
    if eligible is None:
        eligible = valid if valid is not None else applied
    actual_count = _mapping_stat(actual_section.get("count"))
    if actual_count is None:
        actual_count = _summary_intervention_metric(row, "changed_count_exact", "actual_changed_row_count", "changed_count")
    if actual_count is None:
        actual_count = changed
    actual_rate = _summary_section_metric(actual_section and row or {}, "actual_input_changes_only", "action_change_rate")
    action_count = _summary_section_metric(row, "actual_input_changes_only", "action_change_count", "action_changed_count")
    if action_count is None:
        action_count = raw["action_changed_count"]
    if actual_rate is None and action_count is not None and actual_count not in (None, 0):
        actual_rate = action_count / actual_count
    signed = _summary_section_metric(
        row,
        "actual_input_changes_only",
        "selected_probability_delta_pp",
        "selected_action_probability_delta_pp",
    )
    if signed is None:
        signed = _probability_delta_pp(_summary_section_mapping(row, "actual_input_changes_only").get("selected_probability_delta", {}).get("mean"), fraction=True) if isinstance(_summary_section_mapping(row, "actual_input_changes_only").get("selected_probability_delta"), Mapping) else None
    if signed is None:
        signed = raw["probability_signed_pp"]
    absolute = _summary_section_metric(
        row,
        "actual_input_changes_only",
        "selected_probability_delta_abs_pp",
        "selected_probability_abs_delta_pp",
        "mean_absolute_selected_probability_delta_pp",
    )
    if absolute is None:
        absolute = raw["probability_abs_pp"]
    js = _summary_section_metric(row, "actual_input_changes_only", "js_divergence", "js", "mean_js_divergence", "js_mean")
    if js is None:
        js = raw["js"]
    if actual_count == 0:
        # Preserve raw measured zeros in legacy fields, while the changed-only
        # assessment explicitly remains unavailable.
        actual_rate = None
        signed = None
        absolute = None
        js = None
    meaningful = _summary_intervention_metric(
        row,
        "meaningful_changed_count",
        "meaningful_change_count",
        "meaningful_changed_element_count",
    )
    runtime_assessment = _summary_text_value(row, "assessment_status", "runtime_assessment_status")
    runtime_execution = _summary_text_value(row, "execution_status", "runtime_execution_status")
    scope = _summary_text_value(row, "scope")
    on_inapplicable = _summary_text_value(row, "on_inapplicable")
    return {
        "offline_target_count": _int_or_number(target),
        "offline_eligible_count": _int_or_number(eligible),
        "offline_skipped_count": _int_or_number(skipped),
        "offline_actual_action_change_count": _int_or_number(action_count),
        "offline_actual_action_change_rate": actual_rate,
        "offline_actual_selected_probability_delta_pp": signed,
        "offline_actual_selected_probability_delta_abs_pp": absolute,
        "offline_actual_js_divergence": js,
        "offline_execution_status": runtime_execution or _status(row, "完了"),
        "offline_meaningful_changed_count": _int_or_number(meaningful),
        "offline_declared_tolerance": _plain(_declared_tolerance(row)),
        "offline_runtime_assessment_status": runtime_assessment,
        "offline_runtime_execution_status": runtime_execution,
        "offline_scope": scope,
        "offline_on_inapplicable": on_inapplicable,
        "offline_assessment_status": _offline_assessment_status(
            applied=applied,
            changed=changed,
            skipped=skipped,
            target=target,
            status=_status(row, "完了"),
            runtime_assessment=runtime_assessment,
            meaningful=meaningful,
        ),
    }


def _int_or_number(value: float | None) -> int | float | None:
    if value is None:
        return None
    return int(value) if float(value).is_integer() else value


def _offline_assessment_status(
    *,
    applied: float | None,
    changed: float | None,
    skipped: float | None,
    target: float | None,
    status: str,
    runtime_assessment: Any = None,
    meaningful: float | None = None,
) -> str:
    lower = str(status).strip().lower()
    if lower in {"failed", "error", "failure", "失敗"}:
        return "実行失敗"
    runtime_lower = str(runtime_assessment or "").strip().lower()
    if runtime_lower in {"failed", "error", "failure", "実行失敗"}:
        return "実行失敗"
    if applied in (None, 0):
        if target not in (None, 0) or skipped not in (None, 0):
            return "評価不能：適用なし"
        return "未実行／未記録"
    if changed in (None, 0):
        return "評価不能：変更なし"
    if runtime_lower in {"numerical_only", "numeric_only", "数値変化のみ", "数値差のみ"}:
        return "評価対象あり（数値変化のみ）"
    if meaningful == 0 and changed > 0:
        return "評価対象あり（数値変化のみ）"
    if skipped not in (None, 0) and target is not None and applied < target:
        return "条件付き／一部適用"
    return "評価対象あり"


def _episode_mean_metric(row: Mapping[str, Any], *names: str) -> float | None:
    summary = _summary_mapping(row)
    episodes = summary.get("episode_summaries")
    if not isinstance(episodes, Mapping):
        episodes = summary.get("episodes")
    if not isinstance(episodes, Mapping):
        return None
    values: list[float] = []
    for episode in episodes.values():
        if not isinstance(episode, Mapping):
            continue
        section = episode.get("all_timestamps")
        if not isinstance(section, Mapping):
            section = episode.get("summary")
        if not isinstance(section, Mapping):
            continue
        for name in names:
            if name not in section:
                continue
            number = _mapping_stat(section[name])
            if number is not None:
                values.append(number)
                break
    if not values:
        return None
    return sum(values) / len(values)


def _bool_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1", "はい"}:
            return True
        if lowered in {"false", "no", "0", "いいえ"}:
            return False
    return None


def _source_priority(row: Mapping[str, Any]) -> int:
    value = _number(row.get("_source_priority"))
    return int(value) if value is not None else 0


def _method_status(
    payload: Mapping[str, Any],
    method: str,
    has_rows: bool,
    rows: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    if method == "closed_loop":
        # A legacy stage may still say success while its saved episode rows
        # contain an explicit runtime failure.  Current stage counts have
        # priority; raw rows are the fallback for pre-count runs.
        if _metadata_has_execution_failure_counts(payload):
            execution_counts = _metadata_execution_failure_counts(payload)
        else:
            execution_counts = _closed_execution_failure_counts(
                [row for row in (rows or ()) if isinstance(row, Mapping)]
            )
        if execution_counts["failed"] or execution_counts["aborted"]:
            return "一部失敗" if execution_counts["completed"] else "失敗"
    stages = payload.get("stages")
    stage_candidates = (method, "ig") if method == "integrated_gradients" else (method,)
    if isinstance(stages, Mapping):
        stage = next((stages.get(candidate) for candidate in stage_candidates if isinstance(stages.get(candidate), Mapping)), None)
    else:
        stage = None
    if isinstance(stage, Mapping):
        state = _first(stage, "state", "status")
        if state in {"failed", "error"}:
            return "失敗"
        if state in {"running", "pending"}:
            return "実行中"
        if state == "skipped":
            return "未実行"
        if state == "success":
            return "完了" if has_rows else "未記録"
    method_data = payload.get(method)
    if isinstance(method_data, Mapping):
        status = _first(method_data, "status", "state", "result_status")
        if status:
            return str(status)
        executed = method_data.get("executed")
        if executed is False:
            return "未実行"
        if executed is True and has_rows:
            return "完了"
    for key in (f"{method}_status", f"{method}_executed"):
        if key in payload:
            value = payload[key]
            if isinstance(value, bool):
                return "完了" if value and has_rows else ("未記録" if value else "未実行")
            if value is not None:
                return str(value)
    return "完了" if has_rows else "未実行または結果未記録"


def _method_reason(payload: Mapping[str, Any], method: str) -> str:
    """Expose a saved stage/dependency error when a section has no rows."""

    stages = payload.get("stages")
    candidates = (method, "ig") if method == "integrated_gradients" else (method,)
    if isinstance(stages, Mapping):
        for candidate in candidates:
            stage = stages.get(candidate)
            if isinstance(stage, Mapping):
                reason = _reason(stage)
                if reason:
                    return reason
                value = _first(stage, "error_type", "dependency", "detail", default=None)
                if value not in (None, ""):
                    return str(value)
    method_data = payload.get(method)
    if isinstance(method_data, Mapping):
        return _reason(method_data)
    return ""


def _classify(path: Path) -> str | None:
    parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    if any(token in name for token in ("closed_loop", "closed-loop", "closedloop")) or any(
        token in parts for token in ("02_closed_loop", "experiment_03_closed_loop")
    ):
        return "closed_loop"
    if any(token in name for token in ("offline", "perturb", "pattern")) or any(
        token in parts for token in ("01_offline", "experiment_01_perturbation")
    ):
        return "offline"
    if any(token in name for token in ("integrated_gradient", "ig_", "attribution")) or any(
        token in parts for token in ("03_ig", "experiment_02_integrated_gradients")
    ):
        return "ig"
    return None


def _stage_roots(run_dir: Path, metadata: Mapping[str, Any]) -> dict[str, Path | None | bool]:
    """Resolve the explicit analysis directory selected by ``status.json``.

    ``None`` means the old flat/canonical layout has no pointer and may be
    scanned.  ``False`` means a stage has an explicit state but no safe
    directory pointer.  A failed stage may still contain the diagnostics that
    explain the failure, so a validated failed pointer is retained.  The
    caller must never fall back to another analysis directory after a current
    pointer has been recorded.
    """

    stages = metadata.get("stages")
    if not isinstance(stages, Mapping):
        return {"offline": None, "closed_loop": None, "ig": None}
    result: dict[str, Path | None | bool] = {"offline": None, "closed_loop": None, "ig": None}
    for kind, stage_name in (("offline", "offline"), ("closed_loop", "closed_loop"), ("ig", "ig")):
        stage = stages.get(stage_name)
        if not isinstance(stage, Mapping):
            continue
        # Status values are persisted by different writers/versions; treat
        # their spelling and case uniformly before deciding whether a stage
        # may fall back to its legacy canonical directory.
        state = str(stage.get("state", "")).strip().casefold()
        if state in {"pending", "running", "skipped"}:
            result[kind] = False
            continue
        relative = stage.get("relative_dir")
        if relative is None:
            analysis_id = stage.get("analysis_id")
            if analysis_id:
                relative = {"offline": "01_offline", "closed_loop": "02_closed_loop", "ig": "03_ig"}[kind] + "/" + str(analysis_id)
        if relative is None:
            # A failed stage without a current pointer must not fall back to a
            # canonical directory that may contain an older successful run.
            # Successful legacy flat layouts remain readable for compatibility.
            if state in {"failed", "error"}:
                result[kind] = False
            continue
        relative_path = Path(str(relative))
        expected_prefix = {"offline": "01_offline", "closed_loop": "02_closed_loop", "ig": "03_ig"}[kind]
        if relative_path.is_absolute() or ".." in relative_path.parts or not relative_path.parts or relative_path.parts[0] != expected_prefix:
            result[kind] = False
            continue
        candidate = (run_dir / relative_path).resolve()
        try:
            candidate.relative_to(run_dir.resolve())
            candidate.relative_to((run_dir / expected_prefix).resolve())
        except ValueError:
            result[kind] = False
            continue
        result[kind] = candidate if candidate.is_dir() else False
    return result


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _infer_pattern_id_from_path(path: Path, kind: str | None) -> str | None:
    if kind != "offline":
        return None
    for part in reversed(path.parts):
        if part == "P00" or part == "LIDAR" or re.fullmatch(r"input_\d+", part):
            return part
    return None


def _collect(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[Path]]:
    metadata: dict[str, Any] = {}
    offline: list[dict[str, Any]] = []
    closed: list[dict[str, Any]] = []
    ig: list[dict[str, Any]] = []
    artifacts: list[Path] = []

    for candidate in ("manifest.json", "resolved_config.json", "status.json", "analysis_metadata.json"):
        payload = _read_json(run_dir / candidate)
        if isinstance(payload, Mapping):
            metadata.update(dict(payload))
            artifacts.append(run_dir / candidate)

    # These snapshots are report metadata, not additional analysis rows.  In
    # particular, patterns.json used to be classified as an offline file by
    # its filename and produced a spurious ``unknown`` row.
    labels = _pattern_labels(run_dir)
    if labels:
        metadata["_pattern_labels"] = labels
    for metadata_name in ("input_schema.csv", "patterns.csv", "patterns.json"):
        metadata_path = run_dir / metadata_name
        if metadata_path.is_file():
            artifacts.append(metadata_path)

    selected_roots = _stage_roots(run_dir, metadata)

    # Include files in deterministic order.  Reports produced on a previous
    # call are excluded from the data scan to avoid recursively reporting
    # their own summary.csv as fresh experiment data.
    ignored_names = {
        "report.html",
        "report.md",
        "report_manifest.json",
        "report_details.json",
        "report_details.md",
        "offline_details.csv",
        "closed_loop_details.csv",
        "summary.csv",
        "input_schema.csv",
        "patterns.csv",
        "patterns.json",
        # These are metadata/diagnostic tables, not one offline result row
        # per intervention pattern.  In particular, input_statistics.csv has
        # one row per semantic input and must not add 259 pseudo-patterns.
        "input_statistics.csv",
        "input_statistics.json",
        "statistics.csv",
        "statistics.json",
        "feature_statistics.csv",
        "feature_statistics.json",
        # P00 reference traces are consumed by the explicit reference scanner;
        # their record container has no closed-loop pattern summary.
        "reference.json",
    }
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("report_"):
            continue
        kind = _classify(path)
        if kind in selected_roots and selected_roots[kind] is False:
            continue
        selected_root = selected_roots.get(kind) if kind is not None else None
        if isinstance(selected_root, Path) and not _inside(path, selected_root):
            continue
        # video.json is an artifact for media linking, not a closed-loop
        # metric container.  Its frame_map is often a list, so feeding it to
        # _as_rows would create one ``unknown`` row per video frame.
        if path.name == "video.json":
            artifacts.append(path)
            continue
        if path.name in ignored_names:
            continue
        if path.suffix.lower() == ".json":
            value = _read_json(path)
            if isinstance(value, Mapping) and path.name in {"manifest.json", "resolved_config.json", "status.json", "analysis_metadata.json"}:
                continue
            if isinstance(value, Mapping) and isinstance(value.get("records"), list):
                # Retain episode-level terminal metadata when expanding a
                # trajectory container into step rows.
                container = {key: item for key, item in value.items() if key != "records"}
                rows = [{**container, **dict(record)} for record in value["records"] if isinstance(record, Mapping)]
            elif kind == "ig" and isinstance(value, Mapping) and isinstance(value.get("results"), list):
                # IG result.json keeps an analysis ID beside its per-step
                # result objects.  Preserve that context on every row so the
                # rendered table can distinguish multiple IG evaluations.
                container = {key: item for key, item in value.items() if key != "results"}
                rows = [{**container, **dict(record)} for record in value["results"] if isinstance(record, Mapping)]
            else:
                rows = _as_rows(value)
        elif path.suffix.lower() == ".jsonl":
            rows = _read_jsonl(path)
        elif path.suffix.lower() == ".csv":
            rows = _read_csv(path)
        elif path.suffix.lower() == ".npz":
            rows = _npz_rows(_read_npz(path))
        else:
            rows = []
        inferred_pattern_id = _infer_pattern_id_from_path(path, kind)
        priority = 0
        if path.name == "result.json":
            priority = 3
        elif path.name in {"summary.json", "summary.csv"}:
            priority = 2
        elif path.suffix.lower() in {".csv", ".jsonl"}:
            priority = 1
        for row in rows:
            row.setdefault("_source_path", _safe_relative(path, run_dir) or path.name)
            row.setdefault("_source_priority", priority)
            if inferred_pattern_id and _pattern_id(row) == "unknown":
                row["pattern_id"] = inferred_pattern_id
        if rows and kind == "offline":
            offline.extend(rows)
        elif rows and kind == "closed_loop":
            closed.extend(rows)
        elif rows and kind == "ig":
            ig.extend(rows)
        if path.suffix.lower() in {".json", ".jsonl", ".csv", ".npz", ".svg", ".png", ".gif", ".mp4", ".webm"}:
            artifacts.append(path)

    # Canonical flat names may have no identifying token in the filename.
    # Read them only when the corresponding method directory exists.
    for directory, kind in (("01_offline", "offline"), ("02_closed_loop", "closed_loop"), ("03_ig", "ig")):
        base = run_dir / directory
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".csv"} and path.name not in ignored_names:
                # The previous walk already classified these files.  A set
                # prevents duplicate rows when a future classifier changes.
                pass
    return metadata, offline, closed, ig, artifacts


def _normalise_offline_legacy(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for source in rows:
        row = dict(source)
        # ``OfflineAnalysisResult.to_dict`` stores each pattern as
        # ``{"pattern": {...}, "summary": {...}, "rows": [...]}``.
        # Keep both mappings visible so the report can show the declared
        # indices and the measured all-timestamps aggregates.
        pattern_data = row.get("pattern")
        if isinstance(pattern_data, Mapping):
            row = {**dict(pattern_data), **row}
        identifier = _pattern_id(row)
        indices = _parse_indices(_first(row, "changed_indices", "requested_indices", "target_indices", "indices", "feature_indices", "index"))
        applied = _metric(row, "applied_count", "n_applied", "sample_count", "applicable_count", "all_timestamps_count", "count", "steps")
        changed = _metric(row, "changed_count", "n_changed", "actual_change_count", "actual_changed_row_count", "modified_count", "change_count")
        noop = _metric(row, "noop_count", "no_op_count", "no_op_row_count", "n_noop", "unchanged_count")
        if noop is None and applied is not None and changed is not None:
            noop = max(applied - changed, 0.0)
        action_rate = _metric(row, "action_change_rate", "action_flip_rate", "action_changed_fraction", "argmax_change_rate", "flip_rate")
        if action_rate is None:
            flips = _metric(row, "action_changed_count", "argmax_changed_count")
            if flips is not None and applied not in (None, 0):
                action_rate = flips / applied
        selected_delta = _metric(
            row,
            "selected_probability_delta_pp",
            "selected_action_probability_delta_pp",
            "probability_delta_pp",
            "selected_probability_change_pp",
        )
        if selected_delta is None:
            raw = _metric(row, "selected_probability_delta", "selected_action_probability_delta", "probability_delta")
            if raw is not None:
                selected_delta = raw * 100.0
        js = _metric(row, "js_divergence", "js", "mean_js_divergence", "js_mean")
        replacement = _replacement_text(row)
        item = {
            "pattern_id": identifier,
            "pattern_name": _label(row),
            "changed_indices": indices,
            "changed_dimension_count": len(indices) if indices else _metric(row, "changed_dimension_count", "requested_dimension_count", "target_size", "group_size"),
            "replacement": str(replacement),
            "offline_status": _status(row, "完了"),
            "offline_applied_count": int(applied) if applied is not None and applied.is_integer() else applied,
            "offline_changed_count": int(changed) if changed is not None and changed.is_integer() else changed,
            "offline_noop_count": int(noop) if noop is not None and noop.is_integer() else noop,
            "offline_action_change_rate": action_rate,
            "offline_selected_probability_delta_pp": selected_delta,
            "offline_js_divergence": js,
            "offline_episode_mean_action_change_rate": _metric(row, "episode_mean_action_change_rate", "episode_action_change_rate"),
            "offline_step_weighted_action_change_rate": _metric(row, "step_weighted_action_change_rate", "step_action_change_rate"),
            "variant_id": _first(row, "variant_id", default=None),
            "variant_ids": _first(row, "variant_ids", default=None),
            "variant_classification": _first(row, "variant_classification", default=None),
            "variant_evidence": _first(row, "variant_evidence", default=None),
            "resolved_values": _first(row, "resolved_values", default=None),
            "resolved_expression": _first(row, "resolved_expression", default=None),
            "offline_reason": _first(row, "reason", "skip_reason", "message", default=""),
            "raw_offline": row,
        }
        item.update(_offline_derived_fields(row, applied=applied, changed=changed, noop=noop, skipped=None, valid=applied))
        if _is_baseline_pattern(identifier):
            item["offline_assessment_status"] = "対照"
        grouped.setdefault(identifier, []).append(item)

    result: list[dict[str, Any]] = []
    for _identifier, items in grouped.items():
        # A pattern summary is preferred.  If only per-step records are
        # available, aggregate their directly stored booleans/probabilities;
        # the report must still distinguish no-op from a changed input whose
        # policy output happened to remain unchanged.
        item = max(items, key=lambda value: sum(entry is not None for entry in value.values()))
        if len(items) > 1:
            if all(value.get("offline_applied_count") is None for value in items):
                item["offline_applied_count"] = len(items)
            if all(value.get("offline_changed_count") is None for value in items):
                changed_count = sum(bool(value.get("raw_offline", {}).get("actual_input_changed", value.get("raw_offline", {}).get("changed_indices"))) for value in items)
                item["offline_changed_count"] = changed_count
            if all(value.get("offline_noop_count") is None for value in items):
                applied_count = _number(item.get("offline_applied_count"))
                changed_count = _number(item.get("offline_changed_count"))
                if applied_count is not None and changed_count is not None:
                    item["offline_noop_count"] = max(applied_count - changed_count, 0.0)
            if all(value.get("offline_action_change_rate") is None for value in items):
                action_values = [
                    float(raw.get("action_changed"))
                    for raw in (value.get("raw_offline", {}) for value in items)
                    if isinstance(raw.get("action_changed"), bool)
                ]
                if action_values:
                    item["offline_action_change_rate"] = sum(action_values) / len(action_values)
            if all(value.get("offline_selected_probability_delta_pp") is None for value in items):
                probability_values = [_number(value.get("raw_offline", {}).get("selected_probability_delta")) for value in items]
                probability_values = [value for value in probability_values if value is not None]
                if probability_values:
                    item["offline_selected_probability_delta_pp"] = 100.0 * sum(probability_values) / len(probability_values)
            if all(value.get("offline_js_divergence") is None for value in items):
                js_values = [_number(value.get("raw_offline", {}).get("js_divergence")) for value in items]
                js_values = [value for value in js_values if value is not None]
                if js_values:
                    item["offline_js_divergence"] = sum(js_values) / len(js_values)
        result.append(item)
    return result


def _replacement_text(row: Mapping[str, Any]) -> str:
    """Render the declared operation together with resolved typed values."""

    base = _first(
        row,
        "replacement_description",
        "description",
        "replacement",
        "replacement_value",
        "method",
        default="未記録",
    )
    variant = _first(row, "variant_id", default=None)
    resolved = _first(row, "resolved_values", default=None)
    expression = _first(row, "resolved_expression", default=None)
    if variant is None and resolved is None and expression is None:
        return str(base)
    parts = [str(base)]
    if variant is not None:
        parts.append(f"variant={variant}")
    if resolved is not None:
        parts.append("値=" + json.dumps(_plain(resolved), ensure_ascii=False, sort_keys=True))
    if expression is not None:
        parts.append(f"式={expression}")
    return " / ".join(parts)


def _normalise_offline(rows: Iterable[Mapping[str, Any]], labels: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    """Normalize canonical offline summaries, with per-step fallback.

    The current runtime writes one authoritative ``result.json`` per analysis
    and also writes diagnostic ``steps.csv``/``arrays.npz`` files.  Selecting
    the result summary first prevents a skipped row's zero-valued policy
    fields from being mistaken for a measured comparison.
    """

    grouped: dict[str, list[dict[str, Any]]] = {}
    for source in rows:
        row = dict(source)
        pattern_data = row.get("pattern")
        if isinstance(pattern_data, Mapping):
            row = {**dict(pattern_data), **row}
        identifier = _pattern_id(row)
        if identifier == "unknown":
            continue
        indices = _parse_indices(_first(row, "changed_indices", "requested_indices", "target_indices", "indices", "feature_indices", "index"))
        applied = _summary_intervention_metric(row, "applied_count", "applied_row_count", "n_applied", "sample_count", "applicable_count")
        if applied is None:
            applied = _summary_top_metric(row, "applied_row_count", "applied_count", "n_applied", "sample_count", "applicable_count")
        if applied is None:
            applied = _summary_section_metric(row, "all_timestamps", "count")
        if applied is None:
            applied = _metric(row, "all_timestamps_count", "count", "steps")
        changed = _summary_intervention_metric(row, "changed_count_exact", "changed_count", "actual_changed_row_count", "n_changed", "actual_change_count", "modified_count", "change_count")
        if changed is None:
            changed = _summary_top_metric(row, "actual_changed_row_count", "changed_count", "n_changed", "actual_change_count", "modified_count", "change_count")
        noop = _summary_intervention_metric(row, "noop_count", "no_op_row_count", "no_op_count", "n_noop", "unchanged_count")
        if noop is None:
            noop = _summary_top_metric(row, "no_op_row_count", "noop_count", "no_op_count", "n_noop", "unchanged_count")
        skipped_count = _summary_intervention_metric(row, "skipped_count", "skipped_row_count", "n_skipped")
        if skipped_count is None:
            skipped_count = _summary_top_metric(row, "skipped_row_count", "skipped_count", "n_skipped")
        valid_count = _summary_intervention_metric(row, "eligible_count", "eligible_row_count", "valid_count")
        if valid_count is None:
            valid_count = _summary_top_metric(row, "valid_row_count", "valid_count", "eligible_row_count")
        row_skipped = _bool_value(_first(row, "skipped", "skip", default=None)) is True
        is_skipped = row_skipped or (
            skipped_count is not None
            and skipped_count > 0
            and applied in (None, 0)
            and valid_count in (None, 0)
        )
        partial_skip = not is_skipped and skipped_count is not None and skipped_count > 0
        if noop is None and applied is not None and changed is not None and not is_skipped:
            noop = max(applied - changed, 0.0)

        action_rate = None if is_skipped else _summary_section_metric(row, "all_timestamps", "action_change_rate")
        if action_rate is None and not is_skipped:
            action_rate = _metric(row, "action_change_rate", "action_flip_rate", "action_changed_fraction", "argmax_change_rate", "flip_rate")
        if action_rate is None and not is_skipped:
            flips = _summary_section_metric(row, "all_timestamps", "action_change_count")
            if flips is None:
                flips = _metric(row, "action_changed_count", "argmax_changed_count")
            if flips is not None and applied not in (None, 0):
                action_rate = flips / applied

        selected_delta = None if is_skipped else _summary_section_metric(
            row,
            "all_timestamps",
            "selected_probability_delta_pp",
            "selected_action_probability_delta_pp",
            "probability_delta_pp",
        )
        if selected_delta is None and not is_skipped:
            raw = _summary_section_metric(row, "all_timestamps", "selected_probability_delta", "selected_action_probability_delta", "probability_delta")
            if raw is not None:
                selected_delta = raw * 100.0
            # A canonical result summary is authoritative.  Its nested
            # per-step rows may contain default zero pp fields, which must
            # not override a valid all-timestamps mean or turn a missing
            # summary value into a fabricated number.
            elif not isinstance(_summary_mapping(row).get("all_timestamps"), Mapping):
                selected_delta = _metric(
                    row,
                    "selected_probability_delta_pp",
                    "selected_action_probability_delta_pp",
                    "probability_delta_pp",
                    "selected_probability_change_pp",
                )
        if selected_delta is None and not is_skipped and not isinstance(_summary_mapping(row).get("all_timestamps"), Mapping):
            # Legacy per-step rows may store the raw fraction directly.
            raw = _metric(row, "selected_probability_delta", "selected_action_probability_delta", "probability_delta")
            if raw is not None:
                selected_delta = raw * 100.0
        js = None if is_skipped else _summary_section_metric(row, "all_timestamps", "js_divergence", "js", "mean_js_divergence", "js_mean")
        if js is None and not is_skipped:
            js = _metric(row, "js_divergence", "js", "mean_js_divergence", "js_mean")

        episode_rate = None if is_skipped else _episode_mean_metric(row, "action_change_rate")
        if episode_rate is None and not is_skipped:
            episode_rate = _metric(row, "episode_mean_action_change_rate", "episode_action_change_rate")
        step_rate = None if is_skipped else _summary_section_metric(row, "all_timestamps", "action_change_rate")
        if step_rate is None and not is_skipped:
            step_rate = _metric(row, "step_weighted_action_change_rate", "step_action_change_rate")

        replacement = _replacement_text(row)
        base_status = _status(row, "完了")
        base_status_lower = base_status.strip().lower()
        explicit_failure = base_status_lower in {"failed", "error", "failure", "失敗"}
        success_status = base_status_lower in {"完了", "success", "successful", "ok", "passed", "成功"}
        offline_status = (
            base_status
            if explicit_failure
            else "未実行（skip）"
            if is_skipped
            else "完了（一部スキップ）"
            if partial_skip and success_status
            else base_status
        )
        item = {
            "pattern_id": identifier,
            "pattern_name": _label(row, labels),
            "changed_indices": indices,
            "changed_dimension_count": len(indices) if indices else _summary_top_metric(row, "changed_dimension_count", "requested_dimension_count", "target_size", "group_size"),
            "replacement": str(replacement),
            "offline_status": offline_status,
            "offline_applied_count": int(applied) if applied is not None and applied.is_integer() else applied,
            "offline_changed_count": int(changed) if changed is not None and changed.is_integer() else changed,
            "offline_noop_count": int(noop) if noop is not None and noop.is_integer() else noop,
            "offline_action_change_rate": action_rate,
            "offline_selected_probability_delta_pp": selected_delta,
            "offline_js_divergence": js,
            "offline_episode_mean_action_change_rate": episode_rate,
            "offline_step_weighted_action_change_rate": step_rate,
            "variant_id": _first(row, "variant_id", default=None),
            "variant_ids": _first(row, "variant_ids", default=None),
            "variant_classification": _first(row, "variant_classification", default=None),
            "variant_evidence": _first(row, "variant_evidence", default=None),
            "resolved_values": _first(row, "resolved_values", default=None),
            "resolved_expression": _first(row, "resolved_expression", default=None),
            "offline_reason": _format_offline_reason(
                _offline_reason_parts(row),
                applied=applied,
                skipped=skipped_count,
                valid=valid_count,
            ),
            "raw_offline": row,
        }
        item.update(_offline_derived_fields(row, applied=applied, changed=changed, noop=noop, skipped=skipped_count, valid=valid_count))
        if _is_baseline_pattern(identifier):
            item["offline_assessment_status"] = "対照"
        grouped.setdefault(identifier, []).append(item)

    result: list[dict[str, Any]] = []
    for _identifier, items in grouped.items():
        canonical = [value for value in items if _source_priority(value.get("raw_offline", {})) >= 3]
        item = max(canonical or items, key=lambda value: sum(entry is not None for entry in value.values()))
        if canonical:
            result.append(item)
            continue

        raw_rows = [value.get("raw_offline", {}) for value in items]
        skipped_rows = [raw for raw in raw_rows if _bool_value(raw.get("skipped", raw.get("skip"))) is True]
        valid_rows = [raw for raw in raw_rows if raw not in skipped_rows]
        if skipped_rows and not valid_rows:
            item["offline_status"] = "未実行（skip）"
            item["offline_applied_count"] = 0
            item["offline_changed_count"] = 0
            item["offline_noop_count"] = 0
            item["offline_action_change_rate"] = None
            item["offline_episode_mean_action_change_rate"] = None
            item["offline_step_weighted_action_change_rate"] = None
            item["offline_selected_probability_delta_pp"] = None
            item["offline_js_divergence"] = None
        else:
            if skipped_rows:
                item["offline_status"] = "完了（一部スキップ）"
            changed_values = [_bool_value(raw.get("actual_input_changed")) for raw in valid_rows]
            changed_values = [value for value in changed_values if value is not None]
            if changed_values:
                item["offline_applied_count"] = len(valid_rows)
                item["offline_changed_count"] = sum(changed_values)
                item["offline_noop_count"] = len(changed_values) - sum(changed_values)
            action_values = [_bool_value(raw.get("action_changed")) for raw in valid_rows]
            action_values = [value for value in action_values if value is not None]
            if action_values:
                step_rate = sum(action_values) / len(action_values)
                item["offline_action_change_rate"] = step_rate
                item["offline_step_weighted_action_change_rate"] = step_rate
                by_episode: dict[str, list[bool]] = {}
                for raw in valid_rows:
                    value = _bool_value(raw.get("action_changed"))
                    if value is not None:
                        by_episode.setdefault(str(raw.get("episode", "_single")), []).append(value)
                episode_values = [sum(values) / len(values) for values in by_episode.values() if values]
                item["offline_episode_mean_action_change_rate"] = sum(episode_values) / len(episode_values) if episode_values else None
            probability_values: list[float] = []
            for raw in valid_rows:
                value = _number(raw.get("selected_probability_delta_pp"))
                if value is None:
                    value = _number(raw.get("selected_probability_delta"))
                    if value is not None:
                        value *= 100.0
                if value is not None:
                    probability_values.append(value)
            if probability_values:
                item["offline_selected_probability_delta_pp"] = sum(probability_values) / len(probability_values)
            js_values = [_number(raw.get("js_divergence")) for raw in valid_rows]
            js_values = [value for value in js_values if value is not None]
            if js_values:
                item["offline_js_divergence"] = sum(js_values) / len(js_values)

        reasons = _offline_reason_parts({}, skipped_rows)
        if reasons:
            applied_for_reason = item.get("offline_applied_count")
            if applied_for_reason is None and valid_rows:
                applied_for_reason = len(valid_rows)
            item["offline_reason"] = _format_offline_reason(
                reasons,
                applied=applied_for_reason,
                skipped=len(skipped_rows),
                valid=applied_for_reason,
            )
        result.append(item)
    return result


def _bool_metric(row: Mapping[str, Any], *names: str) -> bool | None:
    value = _first_nested(row, *names)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1", "はい"}:
            return True
        if lowered in {"false", "no", "0", "いいえ"}:
            return False
    return None


def _closed_step_values(rows: Sequence[Mapping[str, Any]], *names: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _post_metric(row, *names)
        if value is not None:
            values.append(value)
    return values


def _direct_metric_value(mapping: Mapping[str, Any], *names: str) -> tuple[float | None, bool]:
    """Read a metric from one mapping without descending into records."""

    for name in names:
        if name not in mapping:
            continue
        value = mapping[name]
        return _mapping_stat(value), True
    return None, False


def _post_metric(row: Mapping[str, Any], *names: str) -> float | None:
    """Read only explicit post-step/top-level legacy telemetry.

    Generic ``_metric`` traversal also visits ``pre_telemetry``.  That is
    useful for report metadata, but it would turn a missing post-step value
    into a fabricated measurement during physical aggregation.
    """

    post = row.get("post_telemetry")
    if isinstance(post, Mapping):
        value, present = _direct_metric_value(post, *names)
        if present:
            return value
    value, present = _direct_metric_value(row, *names)
    return value if present else None


def _post_bool(row: Mapping[str, Any], *names: str) -> bool | None:
    post = row.get("post_telemetry")
    candidates: list[Mapping[str, Any]] = []
    if isinstance(post, Mapping):
        candidates.append(post)
    candidates.append(row)
    for mapping in candidates:
        for name in names:
            if name not in mapping:
                continue
            return _bool_value(mapping[name])
    return None


def _post_value(row: Mapping[str, Any], *names: str) -> Any:
    """Read a scalar from post telemetry or a top-level legacy row only."""

    post = row.get("post_telemetry")
    candidates: list[Mapping[str, Any]] = []
    if isinstance(post, Mapping):
        candidates.append(post)
    candidates.append(row)
    for mapping in candidates:
        for name in names:
            if name in mapping:
                return mapping[name]
    return None


def _canonical_metric(row: Mapping[str, Any], *names: str) -> tuple[float | None, bool]:
    """Prefer a saved aggregate key, including an explicit null value."""

    candidates: list[Mapping[str, Any]] = [row]
    for key in ("metrics", "closed_loop", "performance", "summary"):
        value = row.get(key)
        if isinstance(value, Mapping):
            candidates.append(value)
    for candidate in candidates:
        value, present = _direct_metric_value(candidate, *names)
        if present:
            return value, True
    # Absence is resolved by the caller from explicit post-step telemetry.
    # Do not recurse into ``records`` here: a pre-step value or a partial
    # record must not override a missing aggregate field.
    return None, False


def _canonical_value(row: Mapping[str, Any], *names: str) -> tuple[Any, bool]:
    """Prefer a saved aggregate scalar, including an explicit null value."""

    candidates: list[Mapping[str, Any]] = [row]
    for key in ("metrics", "closed_loop", "performance", "summary"):
        value = row.get(key)
        if isinstance(value, Mapping):
            candidates.append(value)
    for candidate in candidates:
        for name in names:
            if name in candidate:
                return candidate[name], True
    return None, False


def _closed_intervention(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("intervention")
    return value if isinstance(value, Mapping) else {}


def _closed_intervention_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Count B interventions from this run's own trajectory records.

    A and B deliberately use separate records.  In particular, target-lane
    telemetry validity is never used as an intervention denominator.
    """

    if not rows:
        # A summary-only artifact without canonical intervention counts does
        # not establish even the target denominator.  Keep every B count N/A
        # instead of turning the absence of records into a zero-application
        # run.
        return {
            "target": None,
            "eligible": None,
            "applied": None,
            "changed": None,
            "noop": None,
            "skipped": None,
            "reasons": [],
        }
    target = len(rows)
    eligible = 0
    applied = 0
    changed = 0
    skipped = 0
    reasons: list[str] = []
    unknown_application = False
    unknown_change = False
    for row in rows:
        intervention = _closed_intervention(row)
        has_intervention_fields = any(
            key in intervention
            for key in (
                "eligible", "applicable", "applied", "applied_count", "intervention_applied",
                "changed", "changed_count", "actual_input_changed", "changed_indices", "actual_changed_indices",
                "skipped", "skip", "skip_reason", "reason",
            )
        )
        if not has_intervention_fields:
            unknown_application = True
            continue
        raw_skipped = _bool_value(_first(intervention, "skipped", "skip", default=None))
        if raw_skipped is None:
            raw_skipped = _bool_value(_first(row, "intervention_skipped", "skipped", default=None))
        eligible_value = _bool_value(_first(intervention, "eligible", "applicable", default=None))
        if eligible_value is False:
            skipped += 1
            reason = _first(intervention, "skip_reason", "reason", default=None)
            if reason not in (None, ""):
                reasons.append(str(reason))
            continue
        if raw_skipped is True:
            skipped += 1
            reason = _first(intervention, "skip_reason", "reason", default=None)
            if reason not in (None, ""):
                reasons.append(str(reason))
            continue
        eligible += 1
        explicit_applied = _first(intervention, "applied", "intervention_applied", default=_MISSING)
        applied_value = _bool_value(explicit_applied) if explicit_applied is not _MISSING else None
        method = str(_first(intervention, "method", "intervention_method", default="")).strip().lower()
        if applied_value is None and explicit_applied is _MISSING and method in {"identity", "control", "baseline", "noop", "no_op"}:
            # Legacy P00 records used applied_count=0 because zero input
            # elements were replaced.  With skipped=false and an explicit
            # identity method, the trajectory step itself was still an
            # applied control observation.  This exception does not infer
            # application from an empty intervention mapping.
            applied_value = True
        if applied_value is None:
            applied_count = _number(_first(intervention, "applied_count", default=None))
            if applied_count is not None:
                applied_value = applied_count > 0
        if applied_value is None:
            unknown_application = True
            continue
        if applied_value is False:
            eligible -= 1
            skipped += 1
            reason = _first(intervention, "skip_reason", "reason", default=None)
            if reason not in (None, ""):
                reasons.append(str(reason))
            continue
        applied += 1
        changed_value = _bool_value(_first(intervention, "changed", "actual_input_changed", default=None))
        if changed_value is None:
            changed_indices = _first(intervention, "changed_indices", "actual_changed_indices", default=None)
            changed_value = bool(_parse_indices(changed_indices)) if changed_indices is not None else None
        if changed_value is None:
            unknown_change = True
        if changed_value is True:
            changed += 1
        reason = _first(intervention, "skip_reason", "reason", default=None)
        if reason not in (None, ""):
            reasons.append(str(reason))
    # A record with no intervention metadata is still a trajectory target, but
    # its application status is unknown.  Keep N/A instead of treating it as
    # an applied no-op; current runtime records always carry the mapping.
    if target and (unknown_application or not any("intervention" in row for row in rows)):
        return {
            "target": target,
            "eligible": None,
            "applied": None,
            "changed": None,
            "noop": None,
            "skipped": None,
            "reasons": [],
        }
    return {
        "target": target,
        "eligible": eligible,
        "applied": applied,
        "changed": None if unknown_change else changed,
        "noop": None if unknown_change else applied - changed,
        "skipped": skipped,
        "reasons": list(dict.fromkeys(reasons)),
    }


def _direct_metric(mapping: Mapping[str, Any], *names: str) -> float | None:
    """Read episode aggregate fields without descending into step records."""

    candidates: list[Mapping[str, Any]] = [mapping]
    for key in ("metrics", "intervention_counts", "closed_loop", "performance", "summary"):
        value = mapping.get(key)
        if isinstance(value, Mapping):
            candidates.append(value)
    for candidate in candidates:
        for name in names:
            if name not in candidate or candidate[name] is None:
                continue
            value = candidate[name]
            number = _mapping_stat(value)
            if number is not None:
                return number
    return None


def _measurement_status(mapping: Mapping[str, Any] | None) -> str | None:
    if not isinstance(mapping, Mapping):
        return None
    status = str(mapping.get("status", "")).strip().casefold().replace("-", "_")
    if status in {"partial", "incomplete", "一部未計測"}:
        return "partial"
    if status in {"unavailable", "not_available", "not_applicable", "未計測", "n/a"}:
        return "unavailable"
    if status in {"complete", "completed", "完了"}:
        return "complete"
    missing = _number(mapping.get("missing_count"))
    expected = _number(mapping.get("expected_count"))
    if missing is not None and missing > 0:
        return "partial"
    if expected is not None and expected > 0 and missing == 0:
        return "complete"
    return None


def _measurement_fraction(mapping: Mapping[str, Any] | None) -> tuple[int, int] | None:
    if not isinstance(mapping, Mapping):
        return None
    measured = _number(mapping.get("measured_count"))
    expected = _number(mapping.get("expected_count"))
    if measured is None:
        measured = _number(mapping.get("state_measured_count"))
    if expected is None:
        expected = _number(mapping.get("state_expected_count"))
    if measured is None or expected is None:
        return None
    return max(0, int(measured)), max(0, int(expected))


def _closed_measurement_details(source: Mapping[str, Any]) -> dict[str, Any]:
    """Expose measurement completeness beside the execution assessment.

    The public trajectory metrics intentionally keep a partial event duration
    as ``None``.  Reports still need to show what was known, so retain the
    measured denominator and the known event-time sum in a compact, readable
    detail string.  The nested raw dictionaries remain available in
    ``report_details.json`` for full auditing.
    """

    event_coverage = source.get("event_state_coverage")
    event_coverage = event_coverage if isinstance(event_coverage, Mapping) else {}
    departure_state = event_coverage.get("departure")
    if not isinstance(departure_state, Mapping):
        departure_state = source.get("departure_state_measurement")
    low_speed_state = event_coverage.get("low_speed")
    if not isinstance(low_speed_state, Mapping):
        low_speed_state = source.get("low_speed_state_measurement")
    departure_duration = source.get("departure_duration_measurement")
    departure_duration = departure_duration if isinstance(departure_duration, Mapping) else None
    low_speed_duration = source.get("low_speed_duration_measurement")
    low_speed_duration = low_speed_duration if isinstance(low_speed_duration, Mapping) else None
    lane_measurement = source.get("lane_metric_measurement")
    if not isinstance(lane_measurement, Mapping):
        lane_measurement = source.get("lane_rms_measurement")
    lane_measurement = lane_measurement if isinstance(lane_measurement, Mapping) else None

    statuses = [
        _measurement_status(value)
        for value in (
            departure_state,
            low_speed_state,
            departure_duration,
            low_speed_duration,
            lane_measurement,
            source.get("duration_measurement"),
            source.get("interval_measurement"),
        )
    ]
    statuses = [value for value in statuses if value is not None]
    if any(value == "partial" for value in statuses):
        overall = "一部未計測"
    elif statuses and all(value == "unavailable" for value in statuses):
        overall = "未計測"
    elif statuses and all(value == "complete" for value in statuses):
        overall = "完全計測"
    else:
        # A complete clock/interval series beside an unavailable event state,
        # or a complete state beside an unavailable interval, is still a
        # partial measurement from the report reader's perspective.
        overall = "一部未計測"

    lane_detail: str | None = None
    lane_fraction = _measurement_fraction(lane_measurement)
    if lane_fraction is not None:
        measured, expected = lane_fraction
        lane_value = _number(source.get("lane_rms_m"))
        value_text = f"、RMS既知値={_fmt(lane_value, suffix=' m')}" if lane_value is not None else ""
        denominator = _number(source.get("lane_metric_denominator_count"))
        denominator_text = f"、分母={int(denominator)}" if denominator is not None else ""
        prefix = "一部未計測; " if _measurement_status(lane_measurement) == "partial" else ""
        lane_detail = f"{prefix}RMS既知 n={measured}/{expected}{value_text}{denominator_text}"

    def event_detail(
        state: Mapping[str, Any] | None,
        duration: Mapping[str, Any] | None,
        value_name: str,
    ) -> str | None:
        fraction = _measurement_fraction(state)
        if fraction is None and isinstance(duration, Mapping):
            fraction = _measurement_fraction(
                {
                    "state_measured_count": duration.get("state_measured_count"),
                    "state_expected_count": duration.get("state_expected_count"),
                }
            )
        if fraction is None and not isinstance(duration, Mapping):
            return None
        measured, expected = fraction or (0, 0)
        known = _number(duration.get("known_event_duration_s")) if isinstance(duration, Mapping) else None
        known_text = f"、既知{value_name}時間={_fmt(known, suffix=' s')}" if known is not None else ""
        event_statuses = [
            value
            for value in (_measurement_status(state), _measurement_status(duration))
            if value is not None
        ]
        if "partial" in event_statuses or (
            "unavailable" in event_statuses and "complete" in event_statuses
        ):
            status = "partial"
        elif event_statuses and all(value == "unavailable" for value in event_statuses):
            status = "unavailable"
        else:
            status = event_statuses[0] if event_statuses else None
        prefix = "一部未計測; " if status == "partial" else ""
        return f"{prefix}状態既知 n={measured}/{expected}{known_text}"

    return {
        "closed_loop_measurement_status": overall,
        "closed_loop_lane_measurement_detail": lane_detail,
        "closed_loop_departure_measurement_detail": event_detail(
            departure_state, departure_duration, "逸脱"
        ),
        "closed_loop_low_speed_measurement_detail": event_detail(
            low_speed_state, low_speed_duration, "低速"
        ),
    }


def _combined_measurement_status(items: Sequence[Mapping[str, Any]]) -> str | None:
    statuses = [
        str(item.get("closed_loop_measurement_status"))
        for item in items
        if item.get("closed_loop_measurement_status") not in (None, "")
    ]
    if not statuses:
        return None
    if "一部未計測" in statuses:
        return "一部未計測"
    if all(value == "未計測" for value in statuses):
        return "未計測"
    if all(value == "完全計測" for value in statuses):
        return "完全計測"
    return "一部未計測"


def _combined_measurement_detail(items: Sequence[Mapping[str, Any]], key: str) -> str | None:
    values = [
        str(item[key])
        for item in items
        if item.get(key) not in (None, "")
    ]
    return " / ".join(dict.fromkeys(values)) if values else None


def _closed_assessment_status(item: Mapping[str, Any]) -> str:
    status = str(item.get("closed_loop_status", "")).strip().lower()
    failed_count = _number(item.get("closed_loop_failed_episode_count")) or 0.0
    aborted_count = _number(item.get("closed_loop_aborted_episode_count")) or 0.0
    interrupted_count = _number(item.get("closed_loop_interrupted_episode_count")) or 0.0
    if status in {"failed", "error", "failure", "実行失敗"}:
        return "実行失敗"
    if failed_count > 0:
        natural_count = _number(item.get("closed_loop_natural_episode_count")) or 0.0
        if natural_count > 0:
            suffix = f"実行失敗{_int_or_number(failed_count)}episode"
            if aborted_count > 0:
                suffix += f"・中断{_int_or_number(aborted_count)}episode"
            return f"評価対象あり（{suffix}）"
        return "実行失敗"
    if _is_baseline_pattern(item.get("pattern_id")) and aborted_count == 0 and interrupted_count == 0:
        return "対照"
    if interrupted_count > 0:
        natural_count = _number(item.get("closed_loop_natural_episode_count")) or 0.0
        if natural_count > 0:
            return f"評価対象あり（中断{_int_or_number(interrupted_count)}episode）"
        return "実行失敗／中断"
    execution_text = str(item.get("closed_loop_execution_status", "")).strip().lower()
    if any(token in execution_text for token in ("abort", "aborted", "中断", "budget", "censor", "truncat", "timeout", "failed", "error")):
        return "実行失敗／中断"
    runtime_assessment = str(item.get("closed_loop_runtime_assessment_status", "")).strip().lower()
    if runtime_assessment in {"failed", "error", "failure", "実行失敗"}:
        return "実行失敗"
    if runtime_assessment in {"aborted", "intervention_abort", "not_evaluable_intervention_abort", "中断", "実行失敗／中断"}:
        return "実行失敗／中断"
    termination = str(item.get("closed_loop_termination", "")).strip().lower()
    if any(token in termination for token in ("stop", "stopped", "停止")) and _bool_value(item.get("closed_loop_arrived")) is not True:
        return "停止／未到達"
    if _bool_value(item.get("closed_loop_crash")) is True:
        return "衝突"
    if _bool_value(item.get("closed_loop_road_out")) is True:
        return "道路外"
    applied = _number(item.get("closed_loop_applied_count"))
    changed = _number(item.get("closed_loop_changed_count"))
    skipped = _number(item.get("closed_loop_skipped_count"))
    target = _number(item.get("closed_loop_target_step_count"))
    if applied is None:
        return "適用件数未記録"
    if applied == 0:
        return "評価不能：適用なし"
    if changed == 0:
        return "評価不能：変更なし"
    meaningful = _number(item.get("closed_loop_meaningful_changed_count"))
    if runtime_assessment in {"numerical_only", "numeric_only", "数値変化のみ"} or (meaningful == 0 and changed > 0):
        return "評価対象あり（数値変化のみ）"
    if skipped not in (None, 0) and target not in (None, 0) and applied < target:
        return "条件付き／一部適用"
    if item.get("closed_loop_paired_missing_count") not in (None, 0):
        return "評価対象あり（P00ペア一部不明）"
    return "評価対象あり"


def _aggregate_closed_steps(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Derive only directly measured aggregates from per-step telemetry.

    A closed-loop trajectory stores pre/post telemetry beside each action.  The
    report may calculate RMS and maxima from those physical values, but it does
    not invent a departure threshold or replace missing target-lane telemetry
    with current-lane values.
    """

    # ``records`` may contain an intervention attempt for which env.step was
    # never called (for example, an intervention abort).  Such a row is a
    # target record, but it is not a measured post-step.  Prefer the explicit
    # runtime marker and retain legacy rows when post telemetry is present.
    measured_rows: list[Mapping[str, Any]] = []
    for row in rows:
        env_step_called = _post_bool(row, "env_step_called", "step_called")
        env_step_returned = _post_bool(row, "env_step_returned", "step_returned")
        post_telemetry = row.get("post_telemetry")
        if env_step_called is False:
            continue
        # ``env_step_called`` records the attempt.  A runtime exception can
        # therefore leave it true while no post-step telemetry exists.  The
        # explicit return marker is authoritative whenever present; retain the
        # old called/telemetry inference only for legacy trajectories.
        if env_step_returned is False:
            continue
        if env_step_returned is True or env_step_called is True or isinstance(post_telemetry, Mapping):
            measured_rows.append(row)

    lateral: list[float] = []
    for row in measured_rows:
        target_valid = _post_bool(row, "target_lane_valid", "lane_reference_valid", "target_lane_reference_valid")
        telemetry = row.get("post_telemetry")
        telemetry = telemetry if isinstance(telemetry, Mapping) else {}
        validity_names = ("target_lane_valid", "lane_reference_valid", "target_lane_reference_valid")
        validity_declared = any(name in row or name in telemetry for name in validity_names)
        # An explicit unknown validity is not a measured target-lane sample.
        # Legacy rows without any validity key retain the old metric fallback.
        if validity_declared and target_valid is not True:
            continue
        value = _post_metric(row, "target_lane_lateral_error_m", "target_lane_offset_m", "target_lane_error_m", "signed_target_lane_offset_m")
        if value is not None:
            lateral.append(value)
    speed = _closed_step_values(measured_rows, "speed_mps", "speed_m_s", "vehicle_speed_mps", "speed")
    # ``progress``/``route_progress`` are legacy keys without a declared
    # physical unit.  They must not be rendered with an ``m`` suffix: only
    # the explicit metre fields are valid report inputs.
    progress = _closed_step_values(measured_rows, "route_progress_m", "progress_m")
    times = _closed_step_values(measured_rows, "simulation_time_s", "sim_time_s", "sim_time_seconds", "time_s", "elapsed_seconds")
    rewards = _closed_step_values(measured_rows, "reward")
    result: dict[str, Any] = {}
    if lateral:
        result["closed_loop_lane_rms_m"] = math.sqrt(sum(value * value for value in lateral) / len(lateral))
        result["closed_loop_lane_max_abs_m"] = max(abs(value) for value in lateral)
        result["closed_loop_lateral_valid_count"] = len(lateral)
        result["closed_loop_valid_count"] = len(lateral)
    result["closed_loop_poststep_count"] = len(measured_rows)
    validity_values: list[bool | None] = []
    validity_declared = False
    for row in measured_rows:
        post = row.get("post_telemetry")
        post = post if isinstance(post, Mapping) else {}
        names = ("target_lane_valid", "lane_reference_valid", "target_lane_reference_valid")
        validity_declared = validity_declared or any(name in row or name in post for name in names)
        validity_values.append(_post_bool(row, *names))
    validity_complete = not validity_declared or all(value is not None for value in validity_values)
    if lateral and measured_rows and validity_complete:
        result["closed_loop_valid_rate"] = len(lateral) / len(measured_rows)
    if speed:
        result["closed_loop_speed_mean_mps"] = sum(speed) / len(speed)
        result["closed_loop_speed_valid_count"] = len(speed)
    if progress:
        # Progress is a measured scalar; the final available value is used,
        # preserving the fact that early-terminated runs may be shorter.
        result["closed_loop_progress"] = progress[-1]
    if times:
        result["closed_loop_duration_s"] = max(times) - min(times)
    if rewards:
        result["closed_loop_reward"] = sum(rewards)
        result["closed_loop_cumulative_reward"] = sum(rewards)
    departure_flags = [_post_bool(row, "target_lane_departure", "lane_departure", "departed_target_lane") for row in measured_rows]
    if any(value is not None for value in departure_flags):
        result["closed_loop_departure_count"] = sum(1 for value in departure_flags if value is True)
        result["closed_loop_ever_departed"] = any(value is True for value in departure_flags)
    arrivals = [_post_bool(row, "arrived", "success", "reached_goal", "arrival", "arrive_dest") for row in measured_rows]
    if any(value is not None for value in arrivals):
        result["closed_loop_arrived"] = any(value is True for value in arrivals)
    road_flags = [_post_bool(row, "road_out", "out_of_road", "out_of_drivable") for row in measured_rows]
    crash_flags = [_post_bool(row, "crash", "crashed", "collision") for row in measured_rows]
    if any(value is not None for value in road_flags):
        result["closed_loop_road_out"] = any(value is True for value in road_flags)
    if any(value is not None for value in crash_flags):
        result["closed_loop_crash"] = any(value is True for value in crash_flags)
    wrong_lane = [_post_bool(row, "wrong_lane_arrival", "wrong_lane_goal", "arrive_wrong_lane", "goal_wrong_lane") for row in measured_rows]
    start_lane = [_post_bool(row, "start_lane_departure", "departed_start_lane", "start_lane_departed") for row in measured_rows]
    if any(value is not None for value in wrong_lane):
        result["closed_loop_wrong_lane_arrival"] = any(value is True for value in wrong_lane)
    if any(value is not None for value in start_lane):
        result["closed_loop_start_lane_departure"] = any(value is True for value in start_lane)
    terminations = [_post_value(row, "termination_reason", "terminal_reason", "end_reason", "termination") for row in measured_rows]
    terminations = [str(value) for value in terminations if value not in (None, "")]
    if terminations:
        result["closed_loop_termination"] = terminations[-1]
    return result


def _closed_is_step(row: Mapping[str, Any]) -> bool:
    """Identify trajectory records without treating summary metrics as steps."""

    direct_names = {
        "step", "action", "action_forwarded", "probabilities", "original_action",
        "post_telemetry", "pre_telemetry", "post_time", "pre_time", "reward",
        "observation_index", "intervention", "decoded_action",
    }
    if any(name in row for name in direct_names):
        return True
    return any(isinstance(row.get(name), Mapping) for name in ("post_telemetry", "pre_telemetry"))


def _closed_episode_id(row: Mapping[str, Any], default: str = "__summary__") -> str:
    value = _first(row, "episode_id", "episode", "episode_index", "episode_number", default=None)
    if value is None:
        return default
    return str(value)


_CLOSED_EPISODE_STATUS_FIELDS = (
    "status",
    "execution_status",
    "closed_loop_execution_status",
    "assessment_status",
    "runtime_assessment_status",
    "closed_loop_runtime_assessment_status",
    "terminal_reason",
    "termination_reason",
    "end_reason",
    "termination",
    "failure_phase",
    "failure_reason",
    "error",
)


def _closed_single_status(value: Any) -> Any:
    """Return one aggregate status only when it is unambiguous."""

    if isinstance(value, (list, tuple)):
        values = [item for item in value if item not in (None, "")]
        if not values:
            return _MISSING
        unique = {str(item).strip().casefold() for item in values}
        return values[0] if len(unique) == 1 else _MISSING
    if isinstance(value, Mapping) or value in (None, ""):
        return _MISSING
    return value


def _closed_metric_source(
    row: Mapping[str, Any],
    metric: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Keep mixed aggregate status arrays from becoming episode metadata."""

    source = dict(row)
    if metric is not None:
        source.update(dict(metric))
    status_unknown = bool(source.get("_closed_episode_status_unknown"))
    for name in _CLOSED_EPISODE_STATUS_FIELDS:
        if name not in source:
            continue
        value = _closed_single_status(source[name])
        if value is _MISSING:
            source.pop(name, None)
            if name in {
                "execution_status",
                "closed_loop_execution_status",
                "assessment_status",
                "runtime_assessment_status",
                "closed_loop_runtime_assessment_status",
            }:
                status_unknown = True
        else:
            source[name] = value
    if status_unknown:
        source["_closed_episode_status_unknown"] = True
    return source


def _closed_metric_rows(row: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any], int]]:
    """Extract explicit episode metrics from summary/trajectory containers."""

    metrics = row.get("metrics")
    if isinstance(metrics, list):
        result: list[tuple[str, Mapping[str, Any], int]] = []
        for index, value in enumerate(metrics):
            if not isinstance(value, Mapping):
                continue
            episode = _closed_episode_id(value, _closed_episode_id(row, f"episode-{index}"))
            result.append((episode, _closed_metric_source(row, value), index))
        return result
    if isinstance(metrics, Mapping):
        episode = _closed_episode_id(metrics, _closed_episode_id(row))
        return [(episode, _closed_metric_source(row, metrics), 0)]
    if not _closed_is_step(row):
        return [(_closed_episode_id(row), _closed_metric_source(row), 0)]
    return []


def _pair_status_is_success(value: Any) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"matched", "verified", "success", "ok", "passed", "true"}:
        return True
    if token in {"unmatched", "unverified", "failed", "failure", "error", "false", "missing", "mismatch", "not_matched", "not-matched", "incompatible"}:
        return False
    return None


def _closed_pair_evidence(
    merged: Mapping[str, Any],
    p00: Mapping[str, Any],
    legacy_reference: Mapping[str, Any] | None = None,
) -> tuple[bool | None, bool | None]:
    """Return (verified, initial-comparison-verified) without guessing pairs.

    A trace alignment record by itself is not a pair proof.  Current runtime
    records carry an explicit pair status/boolean.  Legacy records are
    accepted only when their initial snapshot matched and the reference trace
    validation matched as well.  When a legacy intervention episode omitted
    that validation, the matching P00 reference record may supply it.  A
    numeric delta with no such evidence stays unknown so an aggregate cannot
    be presented as a verified comparison.
    """

    initial_verified = _bool_metric(merged, "initial_comparison_verified")
    initial_match = _first_nested(merged, "initial_match", default=None)
    if initial_verified is None and isinstance(initial_match, Mapping):
        initial_verified = _bool_value(initial_match.get("matched"))
        if initial_verified is None:
            initial_verified = _pair_status_is_success(_first(initial_match, "status", "comparison_status", default=None))
    snapshot = _first_nested(merged, "initial_snapshot", default=None)
    if initial_verified is None and isinstance(snapshot, Mapping):
        nested_match = snapshot.get("initial_match")
        if isinstance(nested_match, Mapping):
            initial_verified = _bool_value(nested_match.get("matched"))
            if initial_verified is None:
                initial_verified = _pair_status_is_success(_first(nested_match, "status", "comparison_status", default=None))

    pair_success = _bool_metric(merged, "paired_p00_verified", "p00_pair_verified")
    if pair_success is None:
        pair_status = _first_nested(merged, "paired_p00_status", "p00_pair_status", default=None)
        pair_success = _pair_status_is_success(pair_status)
    # ``paired_p00.status`` is an explicit pair result.  A status on a trace
    # alignment object is only supporting evidence, never the pair result.
    if pair_success is None:
        pair_success = _pair_status_is_success(_first(p00, "status", "pair_status", default=None))

    reference_validation = _first_nested(
        merged,
        "reference_trace_validation",
        "p00_reference_validation",
        "reference_validation",
        default=None,
    )
    if not isinstance(reference_validation, Mapping) and isinstance(legacy_reference, Mapping):
        reference_validation = _first_nested(
            legacy_reference,
            "reference_trace_validation",
            "p00_reference_validation",
            "reference_validation",
            default=None,
        )
    trace_alignment = _first_nested(merged, "p00_trace_alignment", "paired_trace_alignment", default=None)
    reference_validation_matched = False
    for evidence in (reference_validation, trace_alignment):
        if isinstance(evidence, Mapping):
            status = _pair_status_is_success(_first(evidence, "status", "alignment_status", default=None))
            if status is True:
                if evidence is reference_validation:
                    reference_validation_matched = True

    # Old artifacts did not save a pair status, but did save both explicit
    # initial and reference-trace checks beside the numeric delta.
    legacy_evidence = bool(p00) and any(
        _metric(p00, key) is not None for key in ("p00_delta_lane_rms_m", "p00_delta_progress_m")
    ) and initial_verified is True and reference_validation_matched
    # An explicit negative pair result is authoritative.  The legacy
    # restoration path is only for artifacts that omitted the pair boolean and
    # status entirely.
    if legacy_evidence and pair_success is None:
        pair_success = True

    if initial_verified is False:
        return False if pair_success is not None or legacy_evidence else None, initial_verified
    if pair_success is True:
        return True, initial_verified
    if pair_success is False:
        return False, initial_verified
    return None, initial_verified


def _closed_episode_outcome(merged: Mapping[str, Any], status: str, reason: str) -> tuple[bool, bool, list[str]]:
    """Classify a closed-loop episode while retaining its raw termination."""

    if merged.get("_closed_episode_status_unknown") is True:
        # A mixed aggregate status list has no episode-to-status mapping.  It
        # remains unevaluable until an episode-specific trajectory supplies a
        # scalar status; it must not be promoted to either failure or success.
        return False, False, []

    values = [
        status,
        reason,
        str(_first(merged, "execution_status", "closed_loop_execution_status", default="")),
        str(_first(merged, "terminal_reason", "termination_reason", "end_reason", "termination", default="")),
    ]
    text = " ".join(values).strip().lower()
    interruption_tokens = (
        "abort",
        "中断",
        "budget_censor",
        "budget-censor",
        "budget_censored",
        "censor",
        "truncat",
        "timeout",
        "failed",
        "failure",
        "error",
    )
    interrupted = any(token in text for token in interruption_tokens)
    natural = not interrupted and str(status).strip().lower() not in {"failed", "error", "failure", "未実行", "not_available", "unavailable"}
    reasons: list[str] = []
    if interrupted:
        termination = _first(merged, "terminal_reason", "termination_reason", "end_reason", "termination", default=None)
        execution = _first(merged, "execution_status", "closed_loop_execution_status", default=None)
        for value in (termination, execution, reason):
            if value not in (None, "") and str(value) not in reasons:
                reasons.append(str(value))
    return natural, interrupted, reasons


def _closed_item_from_metric(
    identifier: str,
    source: Mapping[str, Any],
    step_rows: Sequence[Mapping[str, Any]],
    labels: Mapping[str, str] | None,
    legacy_reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one episode row, filling only missing fields from telemetry."""

    nested = _nested(source, "closed_loop", "performance", "summary")
    merged = {**nested, **dict(source)}
    def canonical_metric(*names: str) -> float | None:
        return _canonical_metric(merged, *names)[0]

    def canonical_value(*names: str) -> tuple[Any, bool]:
        return _canonical_value(merged, *names)

    def canonical_or_fallback(fallback: Any, *names: str) -> Any:
        value, present = _canonical_metric(merged, *names)
        return value if present else fallback

    status = _status(merged, "完了")
    reason = _reason(merged)
    status_unknown = merged.get("_closed_episode_status_unknown") is True
    valid_count = canonical_metric("valid_count")
    if not reason and status.strip().lower() in {"not_available", "unavailable", "na", "n/a"}:
        reason = "target-lane telemetry is unavailable"
        if valid_count == 0:
            reason += " (valid_count=0)"
    p00 = _first_nested(merged, "paired_p00", "p00", default={})
    p00 = p00 if isinstance(p00, Mapping) else {}
    paired_verified, initial_verified = _closed_pair_evidence(merged, p00, legacy_reference)
    natural_episode, interrupted_episode, interruption_reasons = _closed_episode_outcome(merged, status, reason)
    termination_value, termination_present = canonical_value(
        "termination_reason", "terminal_reason", "end_reason", "termination", "reason"
    )
    execution_value, execution_present = canonical_value("execution_status", "closed_loop_execution_status")
    assessment_value, assessment_present = canonical_value("assessment_status", "runtime_assessment_status")
    item: dict[str, Any] = {
        "pattern_id": identifier,
        "episode_id": _closed_episode_id(merged),
        "pattern_name": _label(merged, labels),
        "closed_loop_status": status,
        "closed_loop_lane_rms_m": canonical_metric("lane_rms_m", "lateral_error_rms_m", "target_lane_rms_m", "lateral_rms_m"),
        "closed_loop_lane_max_abs_m": canonical_metric("lane_max_abs_m", "max_abs_lateral_error_m", "target_lane_max_abs_m", "lateral_max_abs_m"),
        "closed_loop_departure_count": canonical_metric("departure_count", "lane_departure_count", "target_lane_departure_count"),
        "closed_loop_departure_time_s": canonical_metric("departure_time_s"),
        "closed_loop_first_departure_time_s": canonical_metric("first_departure_time_s"),
        "closed_loop_ever_departed": canonical_value("ever_departed", "ever_departed_target_lane")[0],
        "closed_loop_arrived": canonical_value("arrived", "success", "reached_goal", "arrival", "arrive_dest")[0],
        "closed_loop_wrong_lane_arrival": canonical_value("wrong_lane_arrival", "wrong_lane_goal", "arrive_wrong_lane", "goal_wrong_lane")[0],
        "closed_loop_start_lane_departure": canonical_value("start_lane_departure", "departed_start_lane", "start_lane_departed")[0],
        "closed_loop_road_out": canonical_value("road_out", "out_of_road", "out_of_drivable")[0],
        "closed_loop_crash": canonical_value("crash", "crashed", "collision")[0],
        # Preserve the declared unit.  Unitless legacy progress values are
        # retained in raw_closed but are unavailable to the metre-labelled
        # report fields.
        "closed_loop_progress": canonical_metric("progress_m", "route_progress_m"),
        "closed_loop_speed_mean_mps": canonical_metric("speed_mean_mps", "speed_mean_m_s", "mean_speed_mps", "average_speed_mps", "speed_mean", "speed_m_s"),
        "closed_loop_low_speed_duration_s": canonical_metric("low_speed_duration_s"),
        "closed_loop_duration_s": canonical_metric("duration_s", "simulation_time_s", "sim_time_seconds", "elapsed_seconds", "time_s"),
        "closed_loop_termination": termination_value if termination_present else "未記録",
        "closed_loop_reward": canonical_metric("cumulative_reward", "return", "episode_return", "reward"),
        "closed_loop_cumulative_reward": canonical_metric("cumulative_reward", "return", "episode_return", "reward"),
        "closed_loop_valid_count": canonical_metric("valid_count"),
        "closed_loop_valid_time_s": canonical_metric("valid_time_s", "valid_time_seconds", "valid_time"),
        "closed_loop_poststep_count": canonical_metric("poststep_count"),
        "closed_loop_valid_rate": canonical_metric("valid_rate"),
        "closed_loop_stop_fraction": canonical_metric("stop_fraction", "low_speed_fraction", "stopped_fraction"),
        "closed_loop_action_switch_count": canonical_metric("action_switch_count", "steering_switch_count"),
        # Values from a paired object are accepted below only when its
        # verification state is known.  This prevents an aggregate P00
        # subtraction from masquerading as an episode-level paired result.
        "p00_delta_lane_rms_m": _metric(merged, "p00_delta_lane_rms_m") if paired_verified is True else None,
        "p00_delta_progress": _metric(merged, "p00_delta_progress_m") if paired_verified is True else None,
        "closed_loop_paired_p00_verified": paired_verified,
        "closed_loop_reason": reason,
        "closed_loop_episode_natural": natural_episode,
        "closed_loop_episode_interrupted": interrupted_episode,
        "closed_loop_interruption_reasons": interruption_reasons,
        "raw_closed": merged,
    }
    item["closed_loop_initial_comparison_verified"] = initial_verified
    if initial_verified is False:
        item["closed_loop_reason"] = "; ".join(value for value in (reason, "initial comparison was not verified") if value)
    if paired_verified is False:
        item["closed_loop_reason"] = "; ".join(value for value in (item.get("closed_loop_reason", ""), "P00 pairing was not verified") if value)
    intervention_stats = _closed_intervention_stats(step_rows)
    item.update(
        {
            "closed_loop_target_step_count": _int_or_number(
                canonical_or_fallback(intervention_stats["target"], "target_step_count", "target_count", "record_count")
            ),
            "closed_loop_eligible_count": _int_or_number(
                canonical_or_fallback(intervention_stats["eligible"], "eligible_count", "applicable_count")
            ),
            "closed_loop_applied_count": _int_or_number(
                canonical_or_fallback(intervention_stats["applied"], "applied_count", "intervention_applied_count")
            ),
            "closed_loop_changed_count": _int_or_number(
                canonical_or_fallback(intervention_stats["changed"], "changed_count", "changed_count_exact", "intervention_changed_count")
            ),
            "closed_loop_noop_count": _int_or_number(
                canonical_or_fallback(intervention_stats["noop"], "noop_count", "no_op_count", "intervention_noop_count")
            ),
            "closed_loop_skipped_count": _int_or_number(
                canonical_or_fallback(intervention_stats["skipped"], "skipped_count", "intervention_skip_count", "skip_count")
            ),
            "closed_loop_meaningful_changed_count": _int_or_number(
                canonical_or_fallback(None, "meaningful_changed_count", "meaningful_change_count", "meaningful_changed_element_count")
            ),
            "closed_loop_intervention_skip_reasons": list(intervention_stats["reasons"]),
            "closed_loop_execution_status": execution_value if execution_present else ("unknown" if status_unknown else "completed"),
            "closed_loop_runtime_assessment_status": assessment_value if assessment_present else None,
        }
    )
    item.update(_closed_measurement_details(merged))
    # Pair status is separate from execution status.  Mark the old explicit
    # object as a legacy source so downstream consumers can distinguish it.
    if not _first_nested(merged, "paired_p00_status", "p00_pair_status", default=None) and p00:
        item["closed_loop_pair_status_source"] = "legacy_paired_p00"
    if paired_verified is True:
        if item.get("p00_delta_lane_rms_m") is None:
            item["p00_delta_lane_rms_m"] = _metric(p00, "p00_delta_lane_rms_m")
        if item.get("p00_delta_progress") is None:
            item["p00_delta_progress"] = _metric(p00, "p00_delta_progress_m")
    aggregate = _aggregate_closed_steps(step_rows)
    authoritative: set[str] = set()
    for output_key, names in {
        "closed_loop_lane_rms_m": ("lane_rms_m", "lateral_error_rms_m", "target_lane_rms_m", "lateral_rms_m"),
        "closed_loop_lane_max_abs_m": ("lane_max_abs_m", "max_abs_lateral_error_m", "target_lane_max_abs_m", "lateral_max_abs_m"),
        "closed_loop_departure_count": ("departure_count", "lane_departure_count", "target_lane_departure_count"),
        "closed_loop_departure_time_s": ("departure_time_s",),
        "closed_loop_first_departure_time_s": ("first_departure_time_s",),
        "closed_loop_valid_count": ("valid_count",),
        "closed_loop_valid_time_s": ("valid_time_s", "valid_time_seconds", "valid_time"),
        "closed_loop_valid_rate": ("valid_rate",),
        "closed_loop_ever_departed": ("ever_departed", "ever_departed_target_lane"),
        "closed_loop_arrived": ("arrived", "success", "reached_goal", "arrival", "arrive_dest"),
        "closed_loop_wrong_lane_arrival": ("wrong_lane_arrival", "wrong_lane_goal", "arrive_wrong_lane", "goal_wrong_lane"),
        "closed_loop_start_lane_departure": ("start_lane_departure", "departed_start_lane", "start_lane_departed"),
        "closed_loop_road_out": ("road_out", "out_of_road", "out_of_drivable"),
        "closed_loop_crash": ("crash", "crashed", "collision"),
        "closed_loop_progress": ("progress_m", "route_progress_m"),
        "closed_loop_speed_mean_mps": ("speed_mean_mps", "speed_mean_m_s", "mean_speed_mps", "average_speed_mps", "speed_mean", "speed_m_s"),
        "closed_loop_low_speed_duration_s": ("low_speed_duration_s",),
        "closed_loop_duration_s": ("duration_s", "simulation_time_s", "sim_time_seconds", "elapsed_seconds", "time_s"),
        "closed_loop_reward": ("cumulative_reward", "return", "episode_return", "reward"),
        "closed_loop_cumulative_reward": ("cumulative_reward", "return", "episode_return", "reward"),
        "closed_loop_poststep_count": ("poststep_count",),
        "closed_loop_termination": ("termination_reason", "terminal_reason", "end_reason", "termination", "reason"),
    }.items():
        if any(canonical_value(name)[1] for name in names):
            authoritative.add(output_key)
    for key, value in aggregate.items():
        if key in authoritative:
            continue
        if item.get(key) in (None, "", "未記録"):
            item[key] = value
    return item


def _mean_numbers(values: Iterable[Any]) -> float | None:
    numbers = [value for value in (_number(item) for item in values) if value is not None]
    return sum(numbers) / len(numbers) if numbers else None


def _sum_numbers(values: Iterable[Any]) -> float | None:
    numbers = [value for value in (_number(item) for item in values) if value is not None]
    return sum(numbers) if numbers else None


def _closed_metric_candidate_key(item: Mapping[str, Any]) -> tuple[int, int, int, int, int]:
    """Prefer episode-specific metadata over a richer aggregate summary."""

    has_episode_id = any(
        item.get(name) not in (None, "")
        for name in ("episode_id", "episode", "episode_index", "episode_number")
    )
    has_scalar_execution = any(
        name in item and _closed_single_status(item.get(name)) is not _MISSING
        for name in (
            "execution_status",
            "closed_loop_execution_status",
            "assessment_status",
            "runtime_assessment_status",
            "closed_loop_runtime_assessment_status",
            "terminal_reason",
            "termination_reason",
            "end_reason",
            "termination",
        )
    )
    richness = sum(value not in (None, "") for value in item.values())
    source_priority = _number(item.get("_source_priority"))
    return (
        int(has_episode_id),
        int(has_scalar_execution),
        int(item.get("_closed_episode_status_unknown") is not True),
        richness,
        int(source_priority or 0),
    )


def _normalise_closed(rows: Iterable[Mapping[str, Any]], labels: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for source in rows:
        row = dict(source)
        # Direct callers may provide an unexpanded trajectory container.
        inputs: list[dict[str, Any]] = []
        records = row.get("records")
        if isinstance(records, list):
            base = {key: value for key, value in row.items() if key != "records"}
            inputs.append(base)
            inputs.extend({**base, **dict(record)} for record in records if isinstance(record, Mapping))
        else:
            inputs.append(row)
        for item in inputs:
            nested = _nested(item, "closed_loop", "performance", "summary")
            merged = {**nested, **item}
            identifier = _pattern_id(merged)
            grouped.setdefault(identifier, []).append(merged)

    # Some legacy runs saved the reference-trace validation only on the P00
    # trajectory.  Keep that evidence keyed by episode so a corresponding
    # intervention episode can restore its statusless numeric delta without
    # treating an unrelated episode as paired.
    legacy_references: dict[str, Mapping[str, Any]] = {}
    for identifier, group in grouped.items():
        if not _is_baseline_pattern(identifier):
            continue
        for row in group:
            reference = _first_nested(
                row,
                "reference_trace_validation",
                "p00_reference_validation",
                "reference_validation",
                default=None,
            )
            if not isinstance(reference, Mapping):
                continue
            if _pair_status_is_success(_first(reference, "status", "alignment_status", default=None)) is not True:
                continue
            legacy_references[_closed_episode_id(row)] = {"reference_trace_validation": dict(reference)}

    result: list[dict[str, Any]] = []
    for identifier, group in grouped.items():
        metric_groups: dict[str, list[Mapping[str, Any]]] = {}
        step_groups: dict[str, list[Mapping[str, Any]]] = {}
        seen_metrics: set[tuple[str, str, int]] = set()
        for row in group:
            source_key = str(row.get("_source_path", "")) or repr(row.get("metrics", row))
            for episode, metric, metric_index in _closed_metric_rows(row):
                key = (source_key, episode, metric_index)
                if key in seen_metrics:
                    continue
                seen_metrics.add(key)
                metric_groups.setdefault(episode, []).append(metric)
            if _closed_is_step(row):
                # Avoid recursively selecting a pattern summary while deriving
                # values from post/pre-step telemetry.
                step = {key: value for key, value in row.items() if key not in {"metrics", "summary", "performance", "closed_loop"}}
                step_groups.setdefault(_closed_episode_id(row), []).append(step)

        episode_ids = sorted(set(metric_groups) | set(step_groups))
        if not episode_ids:
            episode_ids = ["__summary__"]
        episode_items: list[dict[str, Any]] = []
        for episode in episode_ids:
            candidates = metric_groups.get(episode, [])
            source = max(candidates, key=_closed_metric_candidate_key) if candidates else {"pattern_id": identifier}
            episode_items.append(
                _closed_item_from_metric(
                    identifier,
                    source,
                    step_groups.get(episode, ()),
                    labels,
                    legacy_references.get(episode),
                )
            )

        natural_items = [item for item in episode_items if item.get("closed_loop_episode_natural") is True and item.get("closed_loop_episode_interrupted") is not True]
        interrupted_items = [item for item in episode_items if item.get("closed_loop_episode_interrupted") is True]
        unevaluable_items = [item for item in episode_items if item.get("closed_loop_episode_natural") is not True and item.get("closed_loop_episode_interrupted") is not True]
        failed_items = [
            item
            for item in episode_items
            if any(
                _execution_failure_kind(item.get(key)) == "failed"
                for key in (
                    "closed_loop_execution_status",
                    "closed_loop_runtime_assessment_status",
                )
            )
            or str(item.get("closed_loop_termination", "")).strip().casefold() == "runtime_error"
        ]
        aborted_items = [
            item
            for item in interrupted_items
            if item not in failed_items
            and (
                any(
                    _execution_failure_kind(item.get(key)) == "aborted"
                    for key in (
                        "closed_loop_execution_status",
                        "closed_loop_runtime_assessment_status",
                    )
                )
                or str(item.get("closed_loop_termination", "")).strip().casefold() == "intervention_abort"
            )
        ]
        # Preserve the episode-level rows even when every run was interrupted;
        # only natural/comparable runs contribute to the main performance
        # aggregates below.
        aggregate_items = natural_items
        statuses = list(dict.fromkeys(str(item.get("closed_loop_status", "未記録")) for item in episode_items))
        reasons = list(dict.fromkeys(str(item.get("closed_loop_reason", "")).strip() for item in episode_items if str(item.get("closed_loop_reason", "")).strip()))
        bool_values = {
            key: [_bool_value(item.get(key)) for item in episode_items]
            for key in ("closed_loop_arrived", "closed_loop_road_out", "closed_loop_crash", "closed_loop_ever_departed", "closed_loop_wrong_lane_arrival", "closed_loop_start_lane_departure", "closed_loop_initial_comparison_verified")
        }
        # Pair counts describe only natural/evaluable episodes.  An aborted
        # or budget-censored episode may still carry a positive pair flag in
        # legacy output, but it cannot contribute a verified delta or make a
        # missing natural pair look complete.
        paired_values = [_bool_value(item.get("closed_loop_paired_p00_verified")) for item in natural_items]
        if _is_baseline_pattern(identifier):
            # P00 is the reference/control itself; pair coverage is not an
            # incomplete comparison and must not appear as a missing pair.
            paired_values = []
        paired_count = sum(value is True for value in paired_values)
        paired_missing = sum(value is not True for value in paired_values)
        paired_reasons = list(dict.fromkeys(
            reason
            for item in episode_items
            for reason in (
                _reason_parts(item.get("raw_closed", {}) if isinstance(item.get("raw_closed"), Mapping) else {})
                + ([str(_first_nested(item.get("raw_closed", {}), "paired_p00_missing_reason", default=""))] if isinstance(item.get("raw_closed"), Mapping) and _first_nested(item.get("raw_closed", {}), "paired_p00_missing_reason", default=None) not in (None, "") else [])
            )
            if reason
        ))
        for interrupted in interrupted_items:
            episode_label = interrupted.get("episode_id", "unknown")
            details = interrupted.get("closed_loop_interruption_reasons") or ["中断・打切り"]
            paired_reasons.extend(
                f"episode {episode_label}: P00 pairing excluded ({reason})"
                for reason in details
                if reason
            )
        paired_reasons = list(dict.fromkeys(paired_reasons))
        paired_delta_items = [] if _is_baseline_pattern(identifier) else [item for item in natural_items if item.get("closed_loop_paired_p00_verified") is True]
        aggregate_bool_values = {
            key: [_bool_value(item.get(key)) for item in aggregate_items]
            for key in bool_values
        }
        aggregate_paired_values = [_bool_value(item.get("closed_loop_paired_p00_verified")) for item in aggregate_items]
        item: dict[str, Any] = {
            "pattern_id": identifier,
            "pattern_name": _label(episode_items[0], labels),
            "closed_loop_status": statuses[0] if len(statuses) == 1 else "複数episode",
            "closed_loop_reason": "; ".join(reasons),
            "closed_loop_episode_count": len(episode_items),
            "closed_loop_natural_episode_count": len(natural_items),
            "closed_loop_evaluable_episode_count": len(natural_items),
            "closed_loop_interrupted_episode_count": len(interrupted_items),
            "closed_loop_unevaluable_episode_count": len(unevaluable_items),
            "closed_loop_failed_episode_count": len(failed_items),
            "closed_loop_aborted_episode_count": len(aborted_items),
            "closed_loop_measurement_status": _combined_measurement_status(episode_items),
            "closed_loop_lane_measurement_detail": _combined_measurement_detail(
                episode_items, "closed_loop_lane_measurement_detail"
            ),
            "closed_loop_departure_measurement_detail": _combined_measurement_detail(
                episode_items, "closed_loop_departure_measurement_detail"
            ),
            "closed_loop_low_speed_measurement_detail": _combined_measurement_detail(
                episode_items, "closed_loop_low_speed_measurement_detail"
            ),
            "closed_loop_interrupted_reasons": list(dict.fromkeys(reason for episode in interrupted_items for reason in episode.get("closed_loop_interruption_reasons", []) if reason)),
            "closed_loop_lane_rms_m": _mean_numbers(item.get("closed_loop_lane_rms_m") for item in aggregate_items),
            "closed_loop_lane_max_abs_m": max((value for value in (_number(item.get("closed_loop_lane_max_abs_m")) for item in aggregate_items) if value is not None), default=None),
            "closed_loop_departure_count": _sum_numbers(item.get("closed_loop_departure_count") for item in aggregate_items),
            "closed_loop_departure_time_s": _mean_numbers(item.get("closed_loop_departure_time_s") for item in aggregate_items),
            "closed_loop_first_departure_time_s": _mean_numbers(item.get("closed_loop_first_departure_time_s") for item in aggregate_items),
            "closed_loop_ever_departed": (any(value is True for value in aggregate_bool_values["closed_loop_ever_departed"]) if any(value is not None for value in aggregate_bool_values["closed_loop_ever_departed"]) else None),
            "closed_loop_arrived": all(value is True for value in aggregate_bool_values["closed_loop_arrived"]) if aggregate_bool_values["closed_loop_arrived"] and all(value is not None for value in aggregate_bool_values["closed_loop_arrived"]) else None,
            "closed_loop_arrival_rate": (sum(1.0 for value in aggregate_bool_values["closed_loop_arrived"] if value is True) / sum(value is not None for value in aggregate_bool_values["closed_loop_arrived"]) if any(value is not None for value in aggregate_bool_values["closed_loop_arrived"]) else None),
            "closed_loop_arrival_count": sum(1 for value in aggregate_bool_values["closed_loop_arrived"] if value is True),
            "closed_loop_arrival_episode_count": sum(value is not None for value in aggregate_bool_values["closed_loop_arrived"]),
            "closed_loop_wrong_lane_arrival": (any(value is True for value in aggregate_bool_values["closed_loop_wrong_lane_arrival"]) if any(value is not None for value in aggregate_bool_values["closed_loop_wrong_lane_arrival"]) else None),
            "closed_loop_start_lane_departure": (any(value is True for value in aggregate_bool_values["closed_loop_start_lane_departure"]) if any(value is not None for value in aggregate_bool_values["closed_loop_start_lane_departure"]) else None),
            "closed_loop_initial_comparison_verified": (all(value is True for value in aggregate_bool_values["closed_loop_initial_comparison_verified"]) if aggregate_bool_values["closed_loop_initial_comparison_verified"] and all(value is not None for value in aggregate_bool_values["closed_loop_initial_comparison_verified"]) else None),
            "closed_loop_paired_p00_verified": (True if aggregate_paired_values and all(value is True for value in aggregate_paired_values) else False if any(value is False for value in paired_values) else None),
            "closed_loop_road_out": (any(value is True for value in aggregate_bool_values["closed_loop_road_out"]) if any(value is not None for value in aggregate_bool_values["closed_loop_road_out"]) else None),
            "closed_loop_crash": (any(value is True for value in aggregate_bool_values["closed_loop_crash"]) if any(value is not None for value in aggregate_bool_values["closed_loop_crash"]) else None),
            "closed_loop_progress": _mean_numbers(item.get("closed_loop_progress") for item in aggregate_items),
            "closed_loop_speed_mean_mps": _mean_numbers(item.get("closed_loop_speed_mean_mps") for item in aggregate_items),
            "closed_loop_low_speed_duration_s": _mean_numbers(item.get("closed_loop_low_speed_duration_s") for item in aggregate_items),
            "closed_loop_duration_s": _mean_numbers(item.get("closed_loop_duration_s") for item in aggregate_items),
            "closed_loop_reward": _mean_numbers(item.get("closed_loop_reward") for item in aggregate_items),
            "closed_loop_cumulative_reward": _mean_numbers(item.get("closed_loop_cumulative_reward") for item in aggregate_items),
            "closed_loop_valid_count": _sum_numbers(item.get("closed_loop_valid_count") for item in aggregate_items),
            "closed_loop_valid_time_s": _sum_numbers(item.get("closed_loop_valid_time_s") for item in aggregate_items),
            "closed_loop_poststep_count": _sum_numbers(item.get("closed_loop_poststep_count") for item in aggregate_items),
            "closed_loop_valid_rate": _mean_numbers(item.get("closed_loop_valid_rate") for item in aggregate_items),
            "closed_loop_stop_fraction": _mean_numbers(item.get("closed_loop_stop_fraction") for item in aggregate_items),
            "closed_loop_action_switch_count": _mean_numbers(item.get("closed_loop_action_switch_count") for item in aggregate_items),
            "closed_loop_termination": (str(episode_items[0].get("closed_loop_termination")) if len(set(str(item.get("closed_loop_termination", "未記録")) for item in episode_items)) == 1 else f"複数（{len(episode_items)} episode）"),
            "p00_delta_lane_rms_m": _mean_numbers(item.get("p00_delta_lane_rms_m") for item in paired_delta_items),
            "p00_delta_progress": _mean_numbers(item.get("p00_delta_progress") for item in paired_delta_items),
            "closed_loop_target_step_count": _sum_numbers(item.get("closed_loop_target_step_count") for item in episode_items),
            "closed_loop_eligible_count": _sum_numbers(item.get("closed_loop_eligible_count") for item in episode_items),
            "closed_loop_applied_count": _sum_numbers(item.get("closed_loop_applied_count") for item in episode_items),
            "closed_loop_changed_count": _sum_numbers(item.get("closed_loop_changed_count") for item in episode_items),
            "closed_loop_noop_count": _sum_numbers(item.get("closed_loop_noop_count") for item in episode_items),
            "closed_loop_skipped_count": _sum_numbers(item.get("closed_loop_skipped_count") for item in episode_items),
            "closed_loop_meaningful_changed_count": _sum_numbers(item.get("closed_loop_meaningful_changed_count") for item in episode_items),
            "closed_loop_paired_episode_count": paired_count,
            "closed_loop_paired_missing_count": paired_missing,
            "closed_loop_paired_missing_reasons": paired_reasons,
            "closed_loop_execution_status": "; ".join(dict.fromkeys(str(item.get("closed_loop_execution_status", "completed")) for item in episode_items)),
            "closed_loop_runtime_assessment_status": "; ".join(
                dict.fromkeys(
                    str(item.get("closed_loop_runtime_assessment_status"))
                    for item in episode_items
                    if item.get("closed_loop_runtime_assessment_status") not in (None, "")
                )
            ) or None,
            "closed_loop_episode_rows": episode_items,
            "raw_closed": {"episodes": [item.get("raw_closed", {}) for item in episode_items]},
        }
        if interrupted_items:
            item["closed_loop_status"] = (item.get("closed_loop_status") or "複数episode") + "（中断あり）"
        item["closed_loop_assessment_status"] = _closed_assessment_status(item)
        result.append(item)
    return result


def _ig_context(row: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the execution context that gives an IG vector its meaning."""

    def number(*names: str) -> float | None:
        return _number(_first(row, *names, default=None))

    attribution_sum = number("attribution_sum", "sum_ig", "integrated_gradients_sum")
    output_difference = number("output_difference", "output_delta", "f_x_minus_f_b")
    residual = number("completeness_delta", "residual", "completeness_residual")
    if residual is None and attribution_sum is not None and output_difference is not None:
        residual = attribution_sum - output_difference
    return {
        "ig_analysis_id": _first(row, "analysis_id", "ig_analysis_id", default=None),
        "ig_episode": _first(row, "episode", "episode_id", "episode_index", default=None),
        "ig_step": _first(row, "step", "step_index", default=None),
        "ig_target_action": _first(row, "target_action", "fixed_action", "selected_action", default=None),
        "ig_baseline_id": _first(row, "baseline_id", "baseline_reference_id", default=None),
        "ig_fx": number("observation_score", "fx", "f_x"),
        "ig_fb": number("baseline_score", "fb", "f_b"),
        "ig_attribution_sum": attribution_sum,
        "ig_output_difference": output_difference,
        "ig_residual": residual,
        "ig_integration_points": number("integration_points", "n_points", "n_steps", "integration_n_steps"),
    }


def _ig_context_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate the per-feature rows into one row per IG evaluation."""

    contexts: dict[tuple[Any, ...], dict[str, Any]] = {}
    fields = (
        "ig_analysis_id",
        "ig_episode",
        "ig_step",
        "ig_target_action",
        "ig_baseline_id",
        "ig_fx",
        "ig_fb",
        "ig_attribution_sum",
        "ig_output_difference",
        "ig_residual",
        "ig_integration_points",
    )
    for row in rows:
        context = {field: row.get(field) for field in fields}
        key = tuple(_plain(context[field]) for field in fields)
        contexts.setdefault(key, context)
    return list(contexts.values())


def _ig_context_label(row: Mapping[str, Any], index: Any) -> str:
    """Stable ASCII SVG label for one feature attribution and its context."""

    def display(value: Any) -> str:
        return "N/A" if value in (None, "") else str(value)

    return (
        f"step={display(row.get('ig_step'))} "
        f"action={display(row.get('ig_target_action'))} "
        f"baseline={display(row.get('ig_baseline_id'))} "
        f"index={display(index)}"
    )


def _normalise_ig(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    def quality(row: Mapping[str, Any]) -> dict[str, Any]:
        tolerance = _metric(row, "completeness_tolerance", "tolerance", "ig_completeness_tolerance")
        retry_count = _metric(row, "retry_count", "ig_retry_count", "completeness_retry_count")
        delta = _metric(row, "completeness_delta", "ig_completeness_delta")
        absolute_error = _metric(row, "absolute_completeness_error", "ig_absolute_completeness_error")
        status_value = _first_nested(row, "completeness_status", "ig_completeness_status", default=None)
        status = str(status_value).strip() if status_value not in (None, "") else ""
        if not status:
            if absolute_error is None and delta is not None:
                absolute_error = abs(delta)
            if absolute_error is not None and tolerance is not None:
                status = "converged" if absolute_error <= tolerance else "nonconverged"
            else:
                status = "unknown"
        raw_warnings = _first_nested(row, "warnings", "ig_warnings", "completeness_warnings", default=[])
        if isinstance(raw_warnings, (list, tuple)):
            warnings = [str(value) for value in raw_warnings if str(value).strip()]
        elif raw_warnings in (None, ""):
            warnings = []
        else:
            warnings = [str(raw_warnings)]
        if status.lower() in {"nonconverged", "not_converged", "failed", "warning", "warn"} and not warnings:
            warnings = ["Integrated Gradients completeness が許容誤差内に収束していません"]
        return {
            "ig_completeness_status": status,
            "ig_completeness_tolerance": tolerance,
            "ig_retry_count": int(retry_count) if retry_count is not None and retry_count.is_integer() else retry_count,
            "ig_completeness_delta": delta,
            "ig_absolute_completeness_error": absolute_error,
            "ig_warnings": warnings,
        }

    result = []
    for source in rows:
        row = dict(source)
        quality_values = quality(row)
        context_values = _ig_context(row)
        attribution_vector = _first(row, "attributions", "ig_values", "integrated_gradients")
        if isinstance(attribution_vector, list) and attribution_vector and all(_number(item) is not None for item in attribution_vector):
            for index, item in enumerate(attribution_vector):
                value = _number(item)
                result.append({
                    "pattern_id": _pattern_id(row),
                    "pattern_name": _label(row),
                    "index": index,
                    "feature_name": f"index {index}",
                    "ig": value,
                    "absolute_ig": abs(value) if value is not None else None,
                    **context_values,
                    **quality_values,
                    "raw_ig": row,
                })
            continue
        indices = _parse_indices(_first(row, "indices", "feature_indices", "index"))
        value = _metric(row, "ig", "attribution", "signed_ig", "mean_ig", "absolute_ig", "abs_ig")
        result.append({
            "pattern_id": _pattern_id(row),
            "pattern_name": _label(row),
            "index": indices[0] if indices else _first(row, "index", "feature_index", default=None),
            "feature_name": _first(row, "feature_name", "name", "label", default="未記録"),
            "ig": value,
            "absolute_ig": _metric(row, "absolute_ig", "abs_ig", "mean_abs_ig", "mean_absolute_ig") or (abs(value) if value is not None else None),
            **context_values,
            **quality_values,
            "raw_ig": row,
        })
    return result


def _ig_quality(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    statuses = [str(row.get("ig_completeness_status", "")).strip().lower() for row in rows]
    warnings: list[str] = []
    for row in rows:
        values = row.get("ig_warnings", [])
        if isinstance(values, (list, tuple)):
            warnings.extend(str(value) for value in values if str(value).strip())
        elif values not in (None, ""):
            warnings.append(str(values))
    warnings = list(dict.fromkeys(warnings))
    if any(status in {"nonconverged", "not_converged", "failed", "warning", "warn"} for status in statuses):
        status = "警告（completeness非収束）"
    elif any(status in {"converged", "ok", "success", "passed"} for status in statuses):
        status = "収束"
    elif rows:
        status = "未確認"
    else:
        status = "未実行"
    sample = next((row for row in rows if row.get("ig_completeness_status")), {})
    return {
        "status": status,
        "raw_statuses": sorted(set(statuses)),
        "tolerance": sample.get("ig_completeness_tolerance"),
        "retry_count": sample.get("ig_retry_count"),
        "delta": sample.get("ig_completeness_delta"),
        "absolute_error": sample.get("ig_absolute_completeness_error"),
        "warnings": warnings,
    }


def _join_rows(offline: Sequence[Mapping[str, Any]], closed: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in offline:
        by_id[str(row["pattern_id"])] = dict(row)
    for row in closed:
        identifier = str(row["pattern_id"])
        by_id.setdefault(identifier, {"pattern_id": identifier, "pattern_name": row.get("pattern_name", identifier)})
        current = by_id[identifier]
        for key, value in row.items():
            # Closed-loop trajectory files commonly omit the catalog label.
            # Do not overwrite a label recovered from patterns/input_schema.
            if key == "pattern_name" and str(value).strip() in {"", "未記録", "unknown"} and current.get("pattern_name") not in (None, "", "未記録", "unknown"):
                continue
            current[key] = value
    return sorted(by_id.values(), key=lambda row: (str(row.get("pattern_id", "")), str(row.get("pattern_name", ""))))


def _p00_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    for row in rows:
        value = str(row.get("pattern_id", "")).upper()
        if value in {"P00", "P0", "BASELINE", "NOOP", "NONE"} or str(row.get("pattern_name", "")).upper() == "P00":
            return row
    return None


def _detail_scope(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count unique observation steps separately from pattern/run rows."""

    step_keys: set[tuple[str, str]] = set()
    episodes: set[str] = set()
    runs: set[tuple[str, str]] = set()
    for record in records:
        episode = str(_first(record, "episode", "episode_id", "episode_index", default="__episode__"))
        pattern = str(record.get("pattern_id", "__pattern__"))
        episodes.add(episode)
        runs.add((pattern, episode))
        step = _first(record, "step", "observation_index", "time_step", default=None)
        if step is None:
            step = _first(record, "simulation_time_s", "time_s", "post_time", "timestamp_s", default=None)
        if step is not None:
            step_keys.add((episode, str(step)))
    return {
        "unique_steps": len(step_keys),
        "unique_episodes": len(episodes),
        "runs": len(runs),
    }


def _report_scope(
    *,
    metadata: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    offline: Sequence[Mapping[str, Any]],
    closed: Sequence[Mapping[str, Any]],
    offline_details: Sequence[Mapping[str, Any]],
    closed_details: Sequence[Mapping[str, Any]],
) -> dict[str, int | float | None]:
    """Return unique source scope and pattern/run totals for the header."""

    offline_scope = _detail_scope(offline_details)
    closed_scope = _detail_scope(closed_details)
    stages = metadata.get("stages")
    collect = stages.get("collect") if isinstance(stages, Mapping) else None
    if not isinstance(collect, Mapping):
        collect = {}
    if offline_scope["unique_steps"] == 0:
        offline_scope["unique_steps"] = int(_number(collect.get("records")) or 0)
    if offline_scope["unique_episodes"] == 0:
        offline_scope["unique_episodes"] = int(_number(collect.get("episodes")) or 0)
    offline_total = _sum_numbers(row.get("offline_target_count") for row in offline)
    closed_total = _sum_numbers(row.get("closed_loop_target_step_count") for row in closed)
    if closed_scope["runs"] == 0:
        closed_scope["runs"] = int(sum(_number(row.get("closed_loop_episode_count")) or 0 for row in closed))
    if closed_scope["unique_episodes"] == 0 and closed:
        closed_scope["unique_episodes"] = int(sum(_number(row.get("closed_loop_episode_count")) or 0 for row in closed))
    return {
        "offline_unique_steps": offline_scope["unique_steps"] or None,
        "offline_unique_episodes": offline_scope["unique_episodes"] or None,
        "offline_total_steps": offline_total,
        "offline_pattern_count": len(offline),
        "closed_unique_steps": closed_scope["unique_steps"] or None,
        "closed_unique_episodes": closed_scope["unique_episodes"] or None,
        "closed_run_count": closed_scope["runs"] or None,
        "closed_total_steps": closed_total,
        "closed_pattern_count": len(closed),
    }


def _scope_text(scope: Mapping[str, Any]) -> str:
    def integer(name: str) -> str:
        value = scope.get(name)
        return _fmt(value, digits=0)

    offline_text = (
        f"①-A延べ={integer('offline_total_steps')} pattern-step・{integer('offline_pattern_count')} patterns"
        if scope.get("offline_pattern_count")
        else "①-A=未実行"
    )
    if scope.get("closed_run_count"):
        closed_text = (
            f"①-B延べ={integer('closed_run_count')} runs・{integer('closed_unique_episodes')} episode・{integer('closed_total_steps')} step"
        )
    else:
        closed_text = "①-B=未実行"
    return (
        f"通常観測={integer('offline_unique_steps')} unique step・{integer('offline_unique_episodes')} episode / "
        f"{offline_text} / {closed_text}"
    )


def _stage_present(row: Mapping[str, Any], kind: str) -> bool:
    prefix = "offline_" if kind == "offline" else "closed_loop_"
    target_key = "offline_target_count" if kind == "offline" else "closed_loop_target_step_count"
    keys = (
        target_key,
        f"{prefix}applied_count",
        f"{prefix}status",
        f"{prefix}episode_count",
        f"{prefix}assessment_status",
    )
    return any(row.get(key) not in (None, "", "N/A", "未記録") for key in keys)


def _coverage_class_counts(rows: Sequence[Mapping[str, Any]], kind: str) -> dict[str, int]:
    prefix = "offline_" if kind == "offline" else "closed_loop_"
    counts = {"changed": 0, "unchanged": 0, "partial": 0, "all_skipped": 0, "unknown": 0}
    changed_key = f"{prefix}changed_count"
    if kind == "offline":
        changed_key = "offline_changed_count"
    target_key = "offline_target_count" if kind == "offline" else "closed_loop_target_step_count"
    for row in rows:
        if not _stage_present(row, kind) or _is_baseline_pattern(row.get("pattern_id")):
            continue
        changed = _number(row.get(changed_key))
        applied = _number(row.get(f"{prefix}applied_count"))
        target = _number(row.get(target_key))
        skipped = _number(row.get(f"{prefix}skipped_count"))
        all_skipped = target is not None and target > 0 and applied in (None, 0) and skipped not in (None, 0)
        if all_skipped:
            counts["all_skipped"] += 1
        if not all_skipped and skipped not in (None, 0) and target not in (None, 0) and applied is not None and applied < target:
            counts["partial"] += 1
        # Changed/no-op is an independent coverage axis.  A partially
        # applicable pattern can therefore contribute to both its partial
        # count and its changed/no-op count.
        if changed is not None and changed > 0:
            counts["changed"] += 1
        elif changed == 0 and applied not in (None, 0):
            counts["unchanged"] += 1
        elif not all_skipped and skipped in (None, 0) and target in (None, 0) and applied in (None, 0):
            counts["unknown"] += 1
    return counts


def _individual_lidar_noop_count(rows: Sequence[Mapping[str, Any]]) -> int:
    count = 0
    for row in rows:
        identifier = str(row.get("pattern_id", "")).lower()
        name = str(row.get("pattern_name", "")).lower()
        # Canonical 259-input runs use ``input_###`` IDs and may omit the
        # catalog label in legacy summaries.  Count those individual rows as
        # LiDAR only when the label says so or when no conflicting label was
        # saved; the full mapping remains in details for audit.
        if not identifier.startswith("input_") or (name not in {"", "未記録", "unknown"} and "lidar" not in name):
            continue
        changed = _number(row.get("offline_changed_count"))
        if changed == 0:
            count += 1
    return count


def _conclusion_lines(
    *,
    rows: Sequence[Mapping[str, Any]],
    offline: Sequence[Mapping[str, Any]],
    closed: Sequence[Mapping[str, Any]],
) -> list[str]:
    lines: list[str] = []
    for kind, label in (("offline", "①-A"), ("closed_loop", "①-B")):
        counts = _coverage_class_counts(rows, kind)
        if kind == "closed_loop" and not closed:
            lines.append(f"- {label}: 未実行（走行記録なし）。主表のB指標はN/Aです。")
            continue
        lines.append(
            f"- {label}: 実変更あり{counts['changed']}件、変更なし{counts['unchanged']}件、"
            f"一部適用{counts['partial']}件、全skip{counts['all_skipped']}件"
            + (f"、条件不足/未記録{counts['unknown']}件" if counts["unknown"] else "")
            + "。"
        )
        if counts["partial"] or counts["all_skipped"] or counts["unknown"]:
            lines.append(f"- {label}条件不足警告: skip・欠測・適用不能を含むため、全域適用や一般化された重要度とは解釈しません。")
    lidar_count = _individual_lidar_noop_count(rows)
    if lidar_count:
        lines.append(
            f"- 個別LiDAR {lidar_count}入力は変更なしです。全stepの値と判定は [offline_details.csv](offline_details.csv) と [report_details.json](report_details.json) に保存しています。"
        )
    return lines


def _p00_reference_sentence(rows: Sequence[Mapping[str, Any]]) -> str:
    row = _p00_row(rows)
    if row is None:
        return "P00対照（保存通常走行）: 到達・横ずれ・逸脱・衝突・終了理由は未記録です。"
    episodes = _number(row.get("closed_loop_arrival_episode_count"))
    if episodes is None:
        episodes = _number(row.get("closed_loop_evaluable_episode_count")) or _number(row.get("closed_loop_episode_count"))
    arrival_count = row.get("closed_loop_arrival_count")
    if arrival_count is None:
        arrival_rate = _number(row.get("closed_loop_arrival_rate"))
        arrival_count = None if episodes is None or arrival_rate is None else _int_or_number(episodes * arrival_rate)
    def state(name: str) -> str:
        value = _bool_value(row.get(name))
        return "あり" if value is True else "なし" if value is False else _NA
    return (
        f"P00対照（保存通常走行）: 到達{_pair_text(arrival_count, episodes)}、"
        f"横ずれRMS {_fmt(row.get('closed_loop_lane_rms_m'), digits=4)} m、"
        f"逸脱 {_fmt(row.get('closed_loop_departure_count'), digits=0)}回/{_fmt(row.get('closed_loop_departure_time_s'), suffix=' s')}、"
        f"衝突{state('closed_loop_crash')}、道路外{state('closed_loop_road_out')}、"
        f"終了理由={row.get('closed_loop_termination') or _NA}。"
    )


def _add_p00_deltas(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for source in rows:
        row = dict(source)
        is_baseline = str(row.get("pattern_id", "")).upper() in {"P00", "P0", "BASELINE", "NOOP", "NONE"}
        for key, delta_key in (
            ("closed_loop_lane_rms_m", "p00_delta_lane_rms_m"),
            ("closed_loop_progress", "p00_delta_progress"),
        ):
            # P00 is a baseline row, not a paired intervention comparison.
            # A missing/unknown status is deliberately N/A; aggregate values
            # are not evidence that episode-level pairing succeeded.
            verified_episode_count = _number(row.get("closed_loop_paired_episode_count"))
            has_verified_episode_delta = verified_episode_count is not None and verified_episode_count > 0
            if is_baseline or (
                row.get("closed_loop_initial_comparison_verified") is False
                and not has_verified_episode_delta
            ) or (
                row.get("closed_loop_paired_p00_verified") is not True
                and not has_verified_episode_delta
            ):
                row[delta_key] = None
                continue
            # Per-episode normalization is the only source of a P00 delta.
            # Never subtract aggregate current/reference values here: a
            # verified episode may have a missing metric, and another
            # unpaired episode may still make the aggregate value available.
            if _number(row.get(delta_key)) is None:
                row[delta_key] = None
        output.append(row)
    return output


def _fmt(value: Any, *, digits: int = 4, suffix: str = "") -> str:
    if value is None or value == "" or (isinstance(value, float) and not math.isfinite(value)):
        return _NA
    if isinstance(value, bool):
        return "はい" if value else "いいえ"
    number = _number(value)
    if number is None:
        return escape(str(value))
    if float(number).is_integer():
        return f"{int(number)}{suffix}"
    return f"{number:.{digits}g}{suffix}"


def _pct(value: Any) -> str:
    number = _number(value)
    return _NA if number is None else f"{number * 100:.2f}%"


_IG_CONTEXT_COLUMNS = (
    ("ig_analysis_id", "analysis_id"),
    ("ig_episode", "episode"),
    ("ig_step", "step"),
    ("ig_target_action", "fixed action"),
    ("ig_baseline_id", "baseline ID"),
    ("ig_fx", "F_x"),
    ("ig_fb", "F_b"),
    ("ig_attribution_sum", "sum IG"),
    ("ig_output_difference", "output difference"),
    ("ig_residual", "residual"),
    ("ig_integration_points", "npoints"),
)


def _ig_context_value(context: Mapping[str, Any], key: str) -> str:
    value = context.get(key)
    if key in {"ig_fx", "ig_fb", "ig_attribution_sum", "ig_output_difference", "ig_residual"}:
        return _fmt(value, digits=8)
    if key == "ig_integration_points":
        return _fmt(value, digits=0)
    return _fmt(value)


def _ig_context_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    contexts = _ig_context_rows(rows)
    if not contexts:
        return "IG 実行 context は保存されていません。"
    headers = [label for _key, label in _IG_CONTEXT_COLUMNS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for context in contexts:
        lines.append("| " + " | ".join(_ig_context_value(context, key) for key, _label in _IG_CONTEXT_COLUMNS) + " |")
    return "\n".join(lines)


def _ig_context_html(rows: Sequence[Mapping[str, Any]]) -> str:
    contexts = _ig_context_rows(rows)
    if not contexts:
        return "<p>IG 実行 context は保存されていません。</p>"
    header_html = "".join(f"<th>{escape(label)}</th>" for _key, label in _IG_CONTEXT_COLUMNS)
    row_html = "".join(
        "<tr>" + "".join(
            f"<td>{_ig_context_value(context, key)}</td>" for key, _label in _IG_CONTEXT_COLUMNS
        ) + "</tr>"
        for context in contexts
    )
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{row_html}</tbody></table>"


def _pair_text(numerator: Any, denominator: Any, *, percent: bool = False) -> str:
    left = _count_text(numerator)
    right = _count_text(denominator)
    if left is None or right is None:
        return _NA
    if percent and _number(denominator) not in (None, 0):
        return f"{left}/{right}（{_number(numerator) / _number(denominator) * 100:.2f}%）"
    return f"{left}/{right}"


def _main_pattern_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Keep configured groups in the main table and move individual inputs to details."""

    identifiers = [str(row.get("pattern_id", "")) for row in rows]
    if len(rows) <= 32:
        return list(rows)
    selected = [
        row for row in rows
        if str(row.get("pattern_id", "")).upper().startswith("P")
        or str(row.get("pattern_id", "")).lower().startswith("group_")
    ]
    return selected or list(rows[:32])


def _main_replacement_text(row: Mapping[str, Any]) -> str:
    """Keep high-dimensional replacement cells readable in the main table."""

    text = str(row.get("replacement", _NA))
    resolved = row.get("resolved_values")
    if not isinstance(resolved, Mapping) or len(resolved) <= 16:
        return text
    # ``replacement`` was built from the same mapping.  Strip that expanded
    # suffix before adding the compact representation; raw values remain in
    # report_details.json and offline_details.csv.
    base = text.split(" / 値=", 1)[0]
    values = [_plain(value) for value in resolved.values()]
    encoded_values: list[str] = []
    for value in values:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True)
        if encoded not in encoded_values:
            encoded_values.append(encoded)
    if len(encoded_values) == 1:
        compact = f"全{len(values)}入力={encoded_values[0]}"
    else:
        representative = list(resolved.items())[:4]
        compact = (
            "代表値=" + json.dumps({str(key): _plain(value) for key, value in representative}, ensure_ascii=False, sort_keys=True)
            + f"（全{len(values)}入力・異なる値あり）"
        )
    return f"{base} / 値={compact}（詳細: offline_details.csv）"


def _offline_main_value(row: Mapping[str, Any], key: str) -> str:
    if key == "pattern_id":
        identifier = str(row.get("pattern_id", _NA))
        label = str(row.get("pattern_name", "")).strip()
        return f"{identifier}（{label}）" if label and label not in {identifier, _NA, "未記録"} else identifier
    if key == "replacement":
        return _main_replacement_text(row)
    if key == "offline_coverage":
        return _pair_text(row.get("offline_applied_count"), row.get("offline_target_count"))
    if key == "offline_change_coverage":
        return _pair_text(row.get("offline_changed_count"), row.get("offline_applied_count"))
    if key == "offline_action_change_coverage":
        return _pair_text(row.get("offline_actual_action_change_count"), row.get("offline_changed_count"))
    if key == "offline_actual_selected_probability_delta_abs_pp":
        if row.get("offline_assessment_status") == "評価対象あり（数値変化のみ）":
            return _NA
        return _fmt(row.get(key), suffix=" pp")
    if key == "offline_actual_js_divergence":
        if row.get("offline_assessment_status") == "評価対象あり（数値変化のみ）":
            return _NA
        return _fmt(row.get(key))
    if key == "offline_assessment_status":
        return str(row.get(key) or _NA)
    return _fmt(row.get(key))


def _closed_main_value(row: Mapping[str, Any], key: str) -> str:
    if key == "pattern_id":
        identifier = str(row.get("pattern_id", _NA))
        label = str(row.get("pattern_name", "")).strip()
        return f"{identifier}（{label}）" if label and label not in {identifier, _NA, "未記録"} else identifier
    if key == "closed_loop_coverage":
        return f"{_pair_text(row.get('closed_loop_applied_count'), row.get('closed_loop_target_step_count'))}・{_pair_text(row.get('closed_loop_changed_count'), row.get('closed_loop_applied_count'))}"
    if key == "closed_loop_outcome":
        arrival_count = row.get("closed_loop_arrival_count")
        arrival_episodes = row.get("closed_loop_arrival_episode_count")
        if arrival_count is None or arrival_episodes is None:
            arrived = _number(row.get("closed_loop_arrival_rate"))
            episodes = _number(row.get("closed_loop_evaluable_episode_count"))
            arrival_count = None if arrived is None or episodes is None else arrived * episodes
            arrival_episodes = episodes
        arrival_text = (
            _NA
            if arrival_episodes is None or _number(arrival_episodes) == 0
            else _pair_text(_int_or_number(arrival_count), arrival_episodes)
        )
        interrupted = _number(row.get("closed_loop_interrupted_episode_count")) or 0.0
        suffix = f"・中断{_int_or_number(interrupted)}" if interrupted else ""
        return f"{arrival_text}・{row.get('closed_loop_termination') or _NA}{suffix}"
    if key == "closed_loop_lane_summary":
        rms = _fmt(row.get("closed_loop_lane_rms_m"), suffix=" m")
        if row.get("closed_loop_measurement_status") == "一部未計測":
            detail = row.get("closed_loop_lane_measurement_detail")
            if detail:
                rms += f"（{detail}）"
        verified_episode_count = _number(row.get("closed_loop_paired_episode_count"))
        if row.get("closed_loop_paired_p00_verified") is True or (verified_episode_count is not None and verified_episode_count > 0):
            delta = _fmt(row.get("p00_delta_lane_rms_m"), suffix=" m")
            if verified_episode_count not in (None, 0) and row.get("closed_loop_paired_p00_verified") is not True:
                delta += f"（検証済み{_int_or_number(verified_episode_count)}episode）"
        else:
            delta = _NA
        return f"RMS {rms}; P00差 {delta}"
    if key == "closed_loop_departure_time_s":
        value = _fmt(row.get(key), suffix=" s")
        if row.get("closed_loop_measurement_status") == "一部未計測":
            detail = row.get("closed_loop_departure_measurement_detail")
            if detail:
                value += f"（{detail}）"
        return value
    if key == "closed_loop_progress":
        return _fmt(row.get(key), suffix=" m")
    if key == "closed_loop_duration_s":
        return _fmt(row.get(key), suffix=" s")
    if key == "closed_loop_assessment_status":
        value = str(row.get(key) or _NA)
        if row.get("closed_loop_measurement_status") == "一部未計測" and "一部未計測" not in value:
            value += "・一部未計測"
        return value
    return _fmt(row.get(key))


def _summary_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[tuple[str, str]], *, kind: str) -> str:
    visible = _main_pattern_rows(rows)
    value_fn = _offline_main_value if kind == "offline" else _closed_main_value
    lines = [
        "| " + " | ".join(label for _key, label in columns) + " |",
        "|" + "|".join("---" for _key, _label in columns) + "|",
    ]
    for row in visible:
        values = [value_fn(row, key).replace("|", "\\|").replace("\n", " ") for key, _label in columns]
        lines.append("| " + " | ".join(values) + " |")
    omitted = len(rows) - len(visible)
    if omitted > 0:
        lines.append(f"| 詳細へ移動（{omitted}行） | " + " | ".join(_NA for _ in columns[1:]) + " |")
    return "\n".join(lines) if lines else "結果行はありません。"


def _summary_table_html(rows: Sequence[Mapping[str, Any]], columns: Sequence[tuple[str, str]], *, kind: str) -> str:
    visible = _main_pattern_rows(rows)
    value_fn = _offline_main_value if kind == "offline" else _closed_main_value
    header = "".join(f"<th>{escape(label)}</th>" for _key, label in columns)
    body = []
    for row in visible:
        body.append("<tr>" + "".join(f"<td>{escape(value_fn(row, key))}</td>" for key, _label in columns) + "</tr>")
    omitted = len(rows) - len(visible)
    if omitted > 0:
        body.append("<tr><td>詳細へ移動（%d行）</td>%s</tr>" % (omitted, "".join(f"<td>{_NA}</td>" for _ in columns[1:])))
    return f"<table class=summary-table><thead><tr>{header}</tr></thead><tbody>{''.join(body) or f'<tr><td colspan="{len(columns)}">結果行はありません。</td></tr>'}</tbody></table>"


def _table(rows: Sequence[Mapping[str, Any]]) -> str:
    headers = [
        ("pattern_id", "ID"),
        ("pattern_name", "対象入力"),
        ("changed_dimension_count", "変更次元数"),
        ("replacement", "置換内容"),
        ("offline_status", "①-A状態"),
        ("offline_reason", "①-A理由"),
        ("offline_applied_count", "適用件数"),
        ("offline_changed_count", "実変更件数"),
        ("offline_noop_count", "no-op件数"),
        ("offline_action_change_rate", "行動変更割合"),
        ("offline_episode_mean_action_change_rate", "episode平均"),
        ("offline_step_weighted_action_change_rate", "step加重平均"),
        ("offline_selected_probability_delta_pp", "選択行動確率差(pp)"),
        ("offline_js_divergence", "JS(nats)"),
        ("closed_loop_status", "①-B状態"),
        ("closed_loop_reason", "①-B理由"),
        ("closed_loop_episode_count", "episode数"),
        ("closed_loop_lane_rms_m", "横ずれRMS(episode平均,m)"),
        ("closed_loop_lane_max_abs_m", "横ずれ最大(m)"),
        ("closed_loop_departure_count", "逸脱回数"),
        ("closed_loop_departure_time_s", "逸脱時間(episode平均,s)"),
        ("closed_loop_first_departure_time_s", "初回逸脱時刻(episode平均,s)"),
        ("closed_loop_ever_departed", "逸脱有無"),
        ("closed_loop_arrived", "到達"),
        ("closed_loop_arrival_rate", "到達率(episode)"),
        ("closed_loop_wrong_lane_arrival", "誤レーン到達"),
        ("closed_loop_start_lane_departure", "開始レーン逸脱"),
        ("closed_loop_initial_comparison_verified", "初期対照確認"),
        ("closed_loop_paired_p00_verified", "P00対応確認"),
        ("closed_loop_road_out", "道路外"),
        ("closed_loop_crash", "衝突"),
        ("closed_loop_progress", "進行度"),
        ("closed_loop_speed_mean_mps", "平均速度(m/s)"),
        ("closed_loop_low_speed_duration_s", "低速時間(s)"),
        ("closed_loop_stop_fraction", "停車割合"),
        ("closed_loop_duration_s", "走行時間(s)"),
        ("closed_loop_cumulative_reward", "累積報酬"),
        ("closed_loop_action_switch_count", "Action切替回数"),
        ("closed_loop_valid_count", "target-lane有効件数"),
        ("closed_loop_valid_time_s", "target-lane有効時間(s)"),
        ("closed_loop_valid_rate", "target-lane有効率"),
        ("closed_loop_termination", "終了理由"),
        ("p00_delta_lane_rms_m", "P00差 横ずれRMS"),
        ("p00_delta_progress", "P00差 進行度"),
    ]
    lines = ["| " + " | ".join(label for _, label in headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        values = []
        for key, _label_text in headers:
            value = row.get(key)
            if key.endswith("rate") or key == "closed_loop_stop_fraction":
                text = _pct(value)
            elif key == "offline_selected_probability_delta_pp":
                text = _fmt(value, suffix=" pp")
            elif key == "closed_loop_arrived":
                text = _fmt(value)
            else:
                text = _fmt(value)
            text = str(text).replace("|", "\\|").replace("\n", " ")
            values.append(text)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _road_id(row: Mapping[str, Any]) -> Any:
    value = _first_nested(
        row,
        "road_segment_id",
        "road_id",
        "road_segment",
        "road_section_id",
        "road_id_tuple",
        default=None,
    )
    return _plain(value)


def _detail_row_json(value: Any) -> str:
    return json.dumps(_plain(value), ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _offline_detail_records(raw_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Flatten saved A rows while preserving values and context for auditing."""

    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for container in raw_rows:
        identifier = _pattern_id(container)
        children = _offline_child_rows(container)
        if not children:
            # A legacy per-step file is itself the row.
            children = [dict(container)] if any(key in container for key in ("step", "actual_input_changed", "skipped", "action_changed")) else []
        for position, child in enumerate(children):
            child_identifier = _pattern_id(child)
            if child_identifier == "unknown":
                child_identifier = identifier
            episode = _first(child, "episode_id", "episode", default=_first(container, "episode_id", "episode", default="episode-unknown"))
            step = _first(child, "step", "observation_index", default=position)
            source = str(container.get("_source_path", child.get("_source_path", "")))
            key = (str(child_identifier), str(episode), str(step), source)
            if key in seen:
                continue
            seen.add(key)
            intervention = child.get("intervention") if isinstance(child.get("intervention"), Mapping) else {}
            skipped = _offline_row_skipped(child)
            changed = _offline_row_changed(child)
            requested = _first(child, "requested_indices", default=_first(intervention, "requested_indices", default=None))
            changed_indices = _first(child, "changed_indices", default=_first(intervention, "changed_indices", default=None))
            time_value = _first(
                child,
                "simulation_time_s",
                "time_s",
                "timestamp_s",
                "timestamp",
                "time",
                default=_first(container, "simulation_time_s", "time_s", "timestamp_s", "timestamp", "time", default=None),
            )
            original_value = _first(
                child,
                "original_value",
                "original_values",
                "source_value",
                "source_values",
                "before_value",
                "before_values",
                default=_first(
                    intervention,
                    "original_value",
                    "original_values",
                    "source_value",
                    "source_values",
                    "before_value",
                    "before_values",
                    default=_first(container, "original_value", "original_values", default=None),
                ),
            )
            modified_value = _first(
                child,
                "modified_value",
                "modified_values",
                "replacement_value",
                "replacement_values",
                "after_value",
                "after_values",
                default=_first(
                    intervention,
                    "modified_value",
                    "modified_values",
                    "replacement_value",
                    "replacement_values",
                    "after_value",
                    "after_values",
                    default=_first(container, "modified_value", "modified_values", default=None),
                ),
            )
            selected_probability = _first(child, "selected_probability_delta", default=None)
            selected_probability_pp = _first(child, "selected_probability_delta_pp", default=None)
            if selected_probability_pp is None:
                selected_probability_pp = _probability_delta_pp(selected_probability, fraction=True)
            result.append(
                {
                    "pattern_id": child_identifier,
                    "episode": _plain(episode),
                    "step": _plain(step),
                    "time_s": _plain(time_value),
                    "road_segment_id": _road_id(child) if _road_id(child) is not None else _road_id(container),
                    "applied": None if skipped is None else not skipped,
                    "changed_exact": changed,
                    "noop": (changed is False and skipped is not True),
                    "skipped": skipped,
                    "skip_reason": _first(child, "skip_reason", default=_first(intervention, "skip_reason", default=None)),
                    "requested_indices": _plain(requested),
                    "changed_indices": _plain(changed_indices),
                    "reference_id": _first(child, "reference_id", default=_first(intervention, "reference_id", default=_first(container, "reference_id", default=None))),
                    "variant_id": _first(child, "variant_id", default=_first(container, "variant_id", default=None)),
                    "variant_ids": _first(child, "variant_ids", default=_first(container, "variant_ids", default=None)),
                    "variant_classification": _first(child, "variant_classification", default=_first(container, "variant_classification", default=None)),
                    "variant_evidence": _first(child, "variant_evidence", default=_first(container, "variant_evidence", default=None)),
                    "resolved_values": _first(child, "resolved_values", default=_first(container, "resolved_values", default=None)),
                    "resolved_expression": _first(child, "resolved_expression", default=_first(container, "resolved_expression", default=None)),
                    "original_value": _plain(original_value),
                    "modified_value": _plain(modified_value),
                    # Preserve the saved key's unit.  The fraction field is
                    # converted only when a separate pp column is requested;
                    # a tiny already-pp value must never be multiplied by 100.
                    "selected_probability_delta": selected_probability,
                    "selected_probability_delta_pp": selected_probability_pp,
                    "js_divergence": _first(child, "js_divergence", "js", default=None),
                    "delta_values": _first(child, "delta_values", default=_first(intervention, "delta_values", default=None)),
                    "delta_abs_values": _first(child, "delta_abs_values", default=_first(intervention, "delta_abs_values", default=None)),
                    "tolerance": _first(child, "tolerance", "meaningful_tolerance", default=_first(intervention, "tolerance", "meaningful_tolerance", default=_first(container, "tolerance", "meaningful_tolerance", default=None))),
                    "clipped_indices": _first(child, "clipped_indices", "clip_indices", default=_first(intervention, "clipped_indices", "clip_indices", default=None)),
                    "clip_count": _first(child, "clip_count", "clipped_count", default=_first(intervention, "clip_count", "clipped_count", default=None)),
                    "meaningful_changed": _first(child, "meaningful_changed", "meaningful_change", default=_first(intervention, "meaningful_changed", "meaningful_change", default=None)),
                    "meaningful_changed_count": _first(child, "meaningful_changed_count", "meaningful_change_count", default=_first(intervention, "meaningful_changed_count", "meaningful_change_count", default=None)),
                    "raw": child,
                    "source": source,
                    "source_priority": _plain(container.get("_source_priority", child.get("_source_priority", 0))),
                }
            )
    return result


def _coverage_records(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Select one authoritative mask cell per pattern/episode/step."""

    selected: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for record in records:
        key = (
            str(record.get("pattern_id", "unknown")),
            str(record.get("episode", record.get("episode_id", "episode-unknown"))),
            str(record.get("step", "")),
        )
        current = selected.get(key)
        priority = _number(record.get("source_priority")) or 0.0
        current_priority = _number(current.get("source_priority")) if current is not None else None
        if current is None or priority >= (current_priority or 0.0):
            selected[key] = record
    return list(selected.values())


def _closed_detail_records(step_records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in step_records:
        intervention = _closed_intervention(row)
        selected_probability = _first(row, "selected_probability_delta", default=_first(intervention, "selected_probability_delta", default=None))
        selected_probability_pp = _first(row, "selected_probability_delta_pp", default=_first(intervention, "selected_probability_delta_pp", default=None))
        if selected_probability_pp is None:
            selected_probability_pp = _probability_delta_pp(selected_probability, fraction=True)
        result.append(
            {
                "pattern_id": _first(row, "pattern_id", "pattern", default="unknown"),
                "episode": _first(row, "episode_id", "episode", default="episode-unknown"),
                "step": _first(row, "step", "observation_index", default=None),
                "simulation_time_s": _first_nested(row, "post_time", "simulation_time_s", "sim_time_s", "time_s", default=None),
                "road_segment_id": _road_id(row) or _road_id(row.get("post_telemetry", {}) if isinstance(row.get("post_telemetry"), Mapping) else {}),
                "applied": None if not intervention else not bool(_bool_value(_first(intervention, "skipped", "skip", default=False))),
                "changed_exact": _bool_value(_first(intervention, "changed", "actual_input_changed", default=None)),
                "skipped": _bool_value(_first(intervention, "skipped", "skip", default=None)),
                "skip_reason": _first(intervention, "skip_reason", "reason", default=None),
                "requested_indices": _first(intervention, "requested_indices", default=None),
                "changed_indices": _first(intervention, "changed_indices", default=None),
                "original_values": _first(row, "original_values", default=_first(intervention, "original_values", default=None)),
                "modified_values": _first(row, "modified_values", default=_first(intervention, "modified_values", default=None)),
                "delta_values": _first(row, "delta_values", default=_first(intervention, "delta_values", default=None)),
                "delta_abs_values": _first(row, "delta_abs_values", default=_first(intervention, "delta_abs_values", default=None)),
                "tolerance": _first(row, "tolerance", "meaningful_tolerance", default=_first(intervention, "tolerance", "meaningful_tolerance", default=None)),
                "clipped_indices": _first(row, "clipped_indices", "clip_indices", default=_first(intervention, "clipped_indices", "clip_indices", default=None)),
                "clip_count": _first(row, "clip_count", "clipped_count", default=_first(intervention, "clip_count", "clipped_count", default=None)),
                "meaningful_changed": _first(row, "meaningful_changed", "meaningful_change", default=_first(intervention, "meaningful_changed", "meaningful_change", default=None)),
                "meaningful_changed_count": _first(row, "meaningful_changed_count", "meaningful_change_count", default=_first(intervention, "meaningful_changed_count", "meaningful_change_count", default=None)),
                "execution_status": _first(row, "execution_status", "closed_loop_execution_status", default=None),
                "assessment_status": _first(row, "assessment_status", "closed_loop_runtime_assessment_status", default=None),
                "failure_phase": _first(row, "failure_phase", "runtime_failure_phase", default=None),
                "failure_reason": _first(row, "failure_reason", "runtime_failure_reason", default=None),
                "terminal_reason": _first(row, "terminal_reason", "termination_reason", default=None),
                "selected_probability_delta": selected_probability,
                "selected_probability_delta_pp": selected_probability_pp,
                "action": _first(row, "action_forwarded", "action", default=None),
                "original_action": _first(row, "original_action", default=None),
                "original_observation_hash": _first(row, "observation_hash", default=None),
                "modified_observation_hash": _first(row, "modified_observation_hash", default=None),
                "target_lane_valid": _first_nested(row, "target_lane_valid", default=None),
                "raw": row,
                "source": row.get("_plot_source", ""),
            }
        )
    return result


def _write_detail_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields and key != "raw":
                fields.append(str(key))
    if not fields:
        fields = ["pattern_id", "episode", "step"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            output = {}
            for field in fields:
                value = row.get(field)
                output[field] = "" if value is None else _detail_row_json(value) if isinstance(value, (Mapping, list, tuple)) else value
            writer.writerow(output)


def _compact_raw_containers(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Remove expanded trajectory duplicates while retaining each source/pattern."""

    compact: list[Mapping[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        source = str(row.get("_source_path", ""))
        identifier = _pattern_id(row)
        key = (source, identifier)
        if key in seen:
            continue
        seen.add(key)
        compact.append(row)
    return compact


def _compact_offline_containers(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Keep one authoritative raw A container per pattern for report JSON.

    The collector also reads legacy ``arrays.npz``/``steps.csv`` diagnostics.
    Those files are useful for the flattened detail CSV, but retaining every
    expanded diagnostic row under ``raw_offline`` multiplies a large official
    run's JSON without adding information.  Prefer the highest-priority
    ``result.json`` row while keeping all collected step values in
    ``offline_details``.
    """

    selected: dict[str, Mapping[str, Any]] = {}

    def score(row: Mapping[str, Any]) -> tuple[int, int, int, str]:
        source = str(row.get("_source_path", ""))
        return (
            _source_priority(row),
            int(isinstance(row.get("summary"), Mapping)),
            int(isinstance(row.get("rows"), list)),
            source,
        )

    for row in rows:
        identifier = _pattern_id(row)
        current = selected.get(identifier)
        if current is None or score(row) > score(current):
            selected[identifier] = row
    compact: list[Mapping[str, Any]] = []
    for key in sorted(selected):
        row = dict(selected[key])
        # Every per-step row, including its raw donor/source values, is kept in
        # ``offline_details``.  The pattern-level JSON keeps the declaration
        # and aggregate counts here without copying those rows a second time.
        for child_key in ("rows", "records", "steps"):
            if isinstance(row.get(child_key), list):
                row.pop(child_key, None)
        compact.append(row)
    return compact


def _without_detail_raw(value: Any) -> Any:
    """Remove duplicated raw blobs from normalized summary snapshots."""

    if isinstance(value, Mapping):
        return {
            str(key): _without_detail_raw(item)
            for key, item in value.items()
            if key not in {"raw", "raw_offline", "raw_closed"}
        }
    if isinstance(value, list):
        return [_without_detail_raw(item) for item in value]
    if isinstance(value, tuple):
        return [_without_detail_raw(item) for item in value]
    return _plain(value)


def _details_markdown(
    *,
    rows: Sequence[Mapping[str, Any]],
    offline_details: Sequence[Mapping[str, Any]],
    closed_details: Sequence[Mapping[str, Any]],
    ig: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# 入力依存度分析 詳細",
        "",
        "主レポートの要約に含めていない定義、母数、元値・変更値、skip理由、道路区間、episode/step、P00ペア状態を保存した詳細です。値が保存されていない項目を0へ補完していません。",
        "",
        "## 参照ファイル",
        "",
        "- `summary.csv`: legacy列を含むパターン集計",
        "- `offline_details.csv`: ①-Aの全保存step、適用/実変更/no-op/skip、道路区間と元データ",
        "- `closed_loop_details.csv`: ①-Bの全保存step、介入と環境テレメトリ",
        "- `report_details.json`: 上記集計とraw/resultの完全なJSON表現",
        "- `report_offline_coverage.svg`: ①-Aの適用/実変更mask（色はcoverage診断）",
        "- `report_closed_loop_coverage.svg`: ①-Bの適用/skip区間（色はcoverage診断）",
        "",
        "## パターン定義と集計",
        "",
        "各行のraw定義（method、indices、replacement、reference/donor、compatibility context、演算、元値分布）は `report_details.json` と `summary.csv` の対応パターンから確認できます。",
        "",
        f"- パターン数: {len(rows)}",
        f"- ①-A詳細step数: {len(offline_details)}",
        f"- ①-B詳細step数: {len(closed_details)}",
        f"- ③ IG保存行数: {len(ig)}",
        "",
        "## ペア比較の扱い",
        "",
        "P00との差は、episode単位で `paired_p00_verified=true` または保存された検証済み状態があるものだけを集約しています。検証状態が無い旧出力はN/Aであり、集約P00値から差を捏造していません。",
        "",
        "## 解釈上の注意",
        "",
        "到達、レーン逸脱、衝突、停止、途中中断は別の終了・評価状態です。進行距離は保存された `progress_m` の単位(m)を維持し、target-lane telemetryのvalid率を介入適用率へ流用していません。",
        "",
        "IG未実行の場合、IGに関する詳細は1行の未実行表示だけで、①-A/①-Bの結果や順位へ混ぜていません。",
        "",
    ]
    measurement_rows = [
        row
        for row in rows
        if row.get("closed_loop_measurement_status") in {"一部未計測", "未計測"}
    ]
    if measurement_rows:
        lines.extend(
            [
                "## ①-B 計測完全性",
                "",
                "execution/assessment の状態と物理テレメトリの計測完全性は別に表示しています。N/A（一部未計測）は未知区間を0へ補完せず、既知の値と分母だけを併記します。",
                "",
            ]
        )
        for row in measurement_rows:
            details = [
                row.get("closed_loop_lane_measurement_detail"),
                row.get("closed_loop_departure_measurement_detail"),
                row.get("closed_loop_low_speed_measurement_detail"),
            ]
            details = [str(value) for value in details if value not in (None, "")]
            lines.append(
                f"- {row.get('pattern_id', _NA)}: {row.get('closed_loop_measurement_status')}"
                + ("; " + " / ".join(details) if details else "")
            )
        lines.append("")
    return "\n".join(lines)


def _episode_detail_table(rows: Sequence[Mapping[str, Any]]) -> str:
    details: list[str] = []
    for row in rows:
        episodes = row.get("closed_loop_episode_rows")
        if not isinstance(episodes, list) or len(episodes) <= 1:
            continue
        for episode in episodes:
            if not isinstance(episode, Mapping):
                continue
            details.append(
                "| " + " | ".join(
                    (
                        str(row.get("pattern_id", _NA)),
                        str(episode.get("episode_id", _NA)),
                        _fmt(episode.get("closed_loop_lane_rms_m")),
                        _fmt(episode.get("closed_loop_lane_max_abs_m")),
                        _fmt(episode.get("closed_loop_arrived")),
                        _fmt(episode.get("closed_loop_progress")),
                        str(episode.get("closed_loop_status", _NA)),
                    )
                ) + " |"
            )
    if not details:
        return ""
    return "\n".join(
        [
            "| pattern | episode | lane RMS (m) | lane max (m) | arrived | progress | status |",
            "| --- | --- | --- | --- | --- | --- | --- |",
            *details,
        ]
    )


def _svg_placeholder(title: str, reason: str, *, width: int = 760, height: int = 230) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">'
        f'<rect width="100%" height="100%" fill="#fafafa" stroke="#ccd6e0"/>'
        f'<text x="{width / 2:.0f}" y="88" text-anchor="middle" font-size="18" fill="#17212b">{escape(title)}</text>'
        f'<text x="{width / 2:.0f}" y="130" text-anchor="middle" font-size="14" fill="#5e6b76">{escape(reason)}</text></svg>'
    )


def _svg_coverage(
    records: Sequence[Mapping[str, Any]],
    *,
    title: str,
    mode: str,
) -> str:
    """Render a compact applied/changed/skip coverage map.

    The map is diagnostic only.  It intentionally does not label phases or
    infer a curve segment; road IDs are shown exactly as saved beside each
    step so a reviewer can inspect context mismatches in the detail CSV.
    """

    # A trajectory may contain several episodes for one pattern.  Grouping
    # by pattern alone would put every episode's step 0 beside one another and
    # make a coverage cell look like a continuous trace.  Keep the episode in
    # the series key even when the label is omitted from the principal table.
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for record in records:
        pattern = str(record.get("pattern_id", "unknown"))
        episode = str(_first(record, "episode", "episode_id", default="episode-unknown"))
        grouped.setdefault((pattern, episode), []).append(record)
    if not grouped:
        return _svg_placeholder(title, "No per-step coverage records were saved")
    # Keep the main figure readable for the official 272-pattern run; detail
    # CSV/JSON still contain every pattern and every step.
    if len(grouped) > 32:
        allowed = {
            key for key in grouped
            if key[0].upper().startswith("P") or key[0].lower().startswith("group_")
        }
        if allowed:
            grouped = {key: grouped[key] for key in sorted(allowed)}
        else:
            # Keep the main diagnostic bounded for an unconventional catalog;
            # the detail CSV/JSON remains the complete source of coverage.
            grouped = dict(sorted(grouped.items())[:32])

    def series_order(record: Mapping[str, Any], position: int) -> tuple[float, float, int]:
        time = _number(_first(record, "simulation_time_s", "time_s", "timestamp_s", "post_time", default=None))
        step = _number(_first(record, "step", "observation_index", default=None))
        return (
            time if time is not None else float("inf"),
            step if step is not None else float("inf"),
            position,
        )

    grouped = {
        key: [record for position, record in sorted(enumerate(values), key=lambda item: series_order(item[1], item[0]))]
        for key, values in sorted(grouped.items())
    }
    width, left, right = 1180, 156, 24
    row_height, top, bottom = 30, 42, 50
    height = top + bottom + row_height * len(grouped)
    max_steps = max((len(values) for values in grouped.values()), default=1)
    chart_w = max(width - left - right, 260)
    cell_w = chart_w / max(max_steps, 1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="24" font-size="16" fill="#17212b">{escape(title)}</text>',
        f'<text x="12" y="24" font-size="11" fill="#5e6b76">road_segment_id is shown in each step title</text>',
    ]
    for row_index, ((pattern, episode), values) in enumerate(grouped.items()):
        y = top + row_index * row_height
        series_label = f"{pattern} / {episode}"
        parts.append(f'<text x="{left - 8}" y="{y + 18}" text-anchor="end" font-size="11" fill="#17212b">{escape(series_label[:30])}</text>')
        for index, record in enumerate(values):
            applied = _bool_value(record.get("applied"))
            changed = _bool_value(record.get("changed_exact"))
            skipped = _bool_value(record.get("skipped"))
            if skipped is True or applied is False:
                color = "#a0aec0"  # skipped / inapplicable
                state = "skipped"
            elif changed is True:
                color = "#2563eb"  # exact model-input change
                state = "changed"
            elif applied is True:
                color = "#f6c344"  # applied no-op
                state = "applied-no-op"
            else:
                color = "#edf2f7"
                state = "unknown"
            x = left + index * cell_w
            road = record.get("road_segment_id", _NA)
            step = record.get("step", index)
            time = _first(record, "simulation_time_s", "time_s", "timestamp_s", "post_time", default=None)
            parts.append(f'<rect x="{x:.2f}" y="{y + 6}" width="{max(cell_w - 0.5, 0.5):.2f}" height="18" fill="{color}"><title>{escape(series_label)} step={escape(str(step))} time={escape(str(time))} state={escape(state)} road_segment_id={escape(str(road))}</title></rect>')
        # A short road-ID trace provides a visible indication when the
        # compatibility context changes; exact full values remain in CSV.
        road_values = [str(record.get("road_segment_id")) for record in values if record.get("road_segment_id") not in (None, "")]
        if road_values:
            compact = " → ".join(dict.fromkeys(road_values))
            parts.append(f'<text x="{left + chart_w + 4}" y="{y + 18}" font-size="9" fill="#5e6b76">{escape(compact[:34])}</text>')
    legend_y = height - 22
    legend = (("#2563eb", "changed"), ("#f6c344", "applied/no-op"), ("#a0aec0", "skipped"), ("#edf2f7", "unknown"))
    legend_x = left
    for color, label in legend:
        parts.append(f'<rect x="{legend_x}" y="{legend_y - 10}" width="12" height="12" fill="{color}"/><text x="{legend_x + 17}" y="{legend_y}" font-size="10" fill="#5e6b76">{escape(label)}</text>')
        legend_x += 130
    parts.append("</svg>")
    return "".join(parts)


def _svg_bars(rows: Sequence[Mapping[str, Any]], *, title: str = "Action change rate") -> str:
    values = []
    for row in rows:
        # Plot only changed-input comparisons.  P00/no-op/skipped rows remain
        # visible in tables and detail files but are not ranked as importance.
        if _number(row.get("offline_changed_count")) in (None, 0):
            continue
        if str(row.get("pattern_id", "")).upper() in {"P00", "P0", "BASELINE", "NOOP", "NONE"}:
            continue
        if row.get("offline_assessment_status") == "評価対象あり（数値変化のみ）":
            # Exact float movement below the declared tolerance is retained
            # in raw/detail artifacts, but has no valid action-importance bar.
            continue
        value = _number(row.get("offline_actual_action_change_rate"))
        if value is None:
            continue
        identifier = str(row.get("pattern_id", ""))
        # SVG text stays ASCII-stable when a local Japanese font is absent;
        # the HTML/Markdown tables provide the Japanese label mapping.
        label = identifier
        values.append((identifier, label, value))
    if not values:
        return _svg_placeholder(title, "No finite offline values were saved")
    configured = [item for item in values if not item[0].lower().startswith("input_")]
    individual = sorted((item for item in values if item[0].lower().startswith("input_")), key=lambda item: item[0])[:20]
    panels = [("Configured groups", configured)]
    if individual:
        panels.append(("Individual inputs (ID order; first 20 shown; full values in details)", individual))
    panels = [(name, items) for name, items in panels if items]
    width, left, right = 980, 72, 24
    panel_height, top, bottom = 260, 42, 52
    height = 26 + len(panels) * (panel_height + 18)
    chart_w, chart_h = width - left - right, panel_height - top - bottom
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="{left}" y="22" font-size="16" fill="#17212b">{escape(title)}</text>']
    for panel_index, (panel_title, panel_values) in enumerate(panels):
        panel_top = 26 + panel_index * (panel_height + 18)
        chart_top = panel_top + top
        chart_bottom = chart_top + chart_h
        parts.append(f'<text x="{left}" y="{panel_top + 28}" font-size="13" fill="#17212b">{escape(panel_title)}</text>')
        parts.append(f'<line x1="{left}" y1="{chart_bottom}" x2="{left + chart_w}" y2="{chart_bottom}" stroke="#71808f"/>')
        parts.append(f'<line x1="{left}" y1="{chart_top}" x2="{left}" y2="{chart_bottom}" stroke="#71808f"/>')
        for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = chart_bottom - tick * chart_h
            parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + chart_w}" y2="{y:.2f}" stroke="#e5ebf0"/>')
            parts.append(f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" font-size="11" fill="#5e6b76">{tick * 100:.0f}%</text>')
        bar_w = chart_w / max(len(panel_values), 1)
        for index, (_identifier, label, value) in enumerate(panel_values):
            bounded = max(0.0, min(1.0, value))
            bar_h = chart_h * bounded
            x = left + index * bar_w + bar_w * 0.16
            y = chart_bottom - bar_h
            parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w * 0.68:.2f}" height="{bar_h:.2f}" fill="#3b82b6"><title>{escape(label)}: {value * 100:.2f}%</title></rect>')
            parts.append(f'<text x="{x + bar_w * 0.34:.2f}" y="{chart_bottom + 20}" text-anchor="middle" font-size="10" fill="#17212b">{escape(label.split(":", 1)[0][:12])}</text>')
        parts.append(f'<text x="12" y="{chart_top + chart_h / 2:.0f}" transform="rotate(-90 12 {chart_top + chart_h / 2:.0f})" text-anchor="middle" font-size="12" fill="#5e6b76">rate</text>')
    parts.append('</svg>')
    return "".join(parts)


def _record_series(rows: Sequence[Mapping[str, Any]], y_names: Sequence[str]) -> tuple[list[float], list[float]]:
    points: list[tuple[float, float]] = []
    for position, row in enumerate(rows):
        x = _number(_first_nested(row, "simulation_time_s", "sim_time_s", "sim_time_seconds", "time_s", "timestamp_s", "step", default=position))
        y = _metric(row, *y_names)
        if x is not None and y is not None:
            points.append((x, y))
    points.sort()
    return [point[0] for point in points], [point[1] for point in points]


def _grouped_series(
    rows: Sequence[Mapping[str, Any]],
    *,
    y_names: Sequence[str],
    time_names: Sequence[str] = ("post_time", "post_time_s", "simulation_time_s", "sim_time_s", "sim_time_seconds"),
) -> list[tuple[str, list[tuple[float, float]]]]:
    grouped: dict[str, list[tuple[float, float]]] = {}
    for _position, row in enumerate(rows):
        x = _number(_first_nested(row, *time_names, default=None))
        y = _metric(row, *y_names)
        if x is None or y is None:
            continue
        pattern = str(_first(row, "_plot_pattern_id", "pattern_id", "pattern", default="unknown"))
        episode = str(_first(row, "_plot_episode_id", "episode_id", "episode", default="episode-unknown"))
        key = f"{pattern} / {episode}"
        grouped.setdefault(key, []).append((x, y))
    return [(key, sorted(values)) for key, values in sorted(grouped.items())]


def _svg_series(
    rows: Sequence[Mapping[str, Any]],
    *,
    y_names: Sequence[str],
    time_names: Sequence[str] = ("post_time", "post_time_s", "simulation_time_s", "sim_time_s", "sim_time_seconds"),
    title: str,
    y_label: str,
    x_label: str = "time (s)",
) -> str:
    groups = _grouped_series(rows, y_names=y_names, time_names=time_names)
    if not groups:
        return _svg_placeholder(title, "No finite telemetry was saved")
    width, height, left, top, bottom = 860, 340, 74, 46, 66
    chart_w, chart_h = width - left - 28, height - top - bottom
    all_points = [point for _label, values in groups for point in values]
    x_min, x_max = min(point[0] for point in all_points), max(point[0] for point in all_points)
    y_min, y_max = min(point[1] for point in all_points), max(point[1] for point in all_points)
    if x_max == x_min:
        x_max = x_min + 1.0
    if y_max == y_min:
        margin = max(abs(y_max) * 0.1, 0.5)
        y_min -= margin
        y_max += margin
    palette = ("#d05a47", "#3b82b6", "#2f855a", "#805ad5", "#dd6b20", "#718096")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="26" font-size="16" fill="#17212b">{escape(title)}</text>',
        f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" stroke="#71808f"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#71808f"/>',
    ]
    for value, x in ((x_min, left), (x_max, left + chart_w)):
        parts.append(f'<text x="{x:.2f}" y="{top + chart_h + 20}" text-anchor="middle" font-size="11" fill="#5e6b76">{value:.4g}</text>')
    for value, y in ((y_min, top + chart_h), (y_max, top)):
        parts.append(f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" font-size="11" fill="#5e6b76">{value:.4g}</text>')
    for index, (label, values) in enumerate(groups):
        color = palette[index % len(palette)]
        points = " ".join(
            f"{left + (x - x_min) / (x_max - x_min) * chart_w:.2f},{top + (1 - (y - y_min) / (y_max - y_min)) * chart_h:.2f}"
            for x, y in values
        )
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"><title>{escape(label)}</title></polyline>')
        legend_x = left + (index % 3) * 245
        legend_y = height - 30 - (index // 3) * 16
        parts.append(f'<line x1="{legend_x}" y1="{legend_y - 4}" x2="{legend_x + 18}" y2="{legend_y - 4}" stroke="{color}" stroke-width="3"/><text x="{legend_x + 24}" y="{legend_y}" font-size="11" fill="#17212b">{escape(label[:32])}</text>')
    parts.append(f'<text x="{left + chart_w / 2:.0f}" y="{height - 8}" text-anchor="middle" font-size="12" fill="#5e6b76">{escape(x_label)}</text>')
    parts.append(f'<text x="14" y="{top + chart_h / 2:.0f}" transform="rotate(-90 14 {top + chart_h / 2:.0f})" text-anchor="middle" font-size="12" fill="#5e6b76">{escape(y_label)}</text></svg>')
    return "".join(parts)


def _svg_trajectory(rows: Sequence[Mapping[str, Any]], *, title: str = "World-coordinate trajectory") -> str:
    grouped: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        x = _metric(row, "_plot_x")
        y = _metric(row, "_plot_y")
        if x is None or y is None:
            # Retain a strict plot boundary: world coordinates must come from
            # the post-step telemetry prepared by _scan_step_records.
            continue
        if x is not None and y is not None:
            pattern = str(_first(row, "_plot_pattern_id", "pattern_id", "pattern", default="unknown"))
            episode = str(_first(row, "_plot_episode_id", "episode_id", "episode", default="episode-unknown"))
            grouped.setdefault(f"{pattern} / {episode}", []).append((x, y))
    groups = [(label, values) for label, values in sorted(grouped.items())]
    if not groups:
        return _svg_placeholder(title, "No world coordinates were saved")
    points = [point for _label, values in groups for point in values]
    width, height = 860, 340
    left, top, chart_w, chart_h = 74, 46, 758, 240
    min_x, max_x = min(x for x, _ in points), max(x for x, _ in points)
    min_y, max_y = min(y for _, y in points), max(y for _, y in points)
    if max_x == min_x:
        max_x = min_x + 1
    if max_y == min_y:
        max_y = min_y + 1
    palette = ("#3b82b6", "#d05a47", "#2f855a", "#805ad5", "#dd6b20", "#718096")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="26" font-size="16" fill="#17212b">{escape(title)}</text>',
        f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" stroke="#71808f"/><line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#71808f"/>',
        f'<text x="{left}" y="{top + chart_h + 20}" text-anchor="middle" font-size="11" fill="#5e6b76">{min_x:.4g}</text><text x="{left + chart_w}" y="{top + chart_h + 20}" text-anchor="middle" font-size="11" fill="#5e6b76">{max_x:.4g}</text>',
        f'<text x="{left - 8}" y="{top + chart_h + 4}" text-anchor="end" font-size="11" fill="#5e6b76">{min_y:.4g}</text><text x="{left - 8}" y="{top + 4}" text-anchor="end" font-size="11" fill="#5e6b76">{max_y:.4g}</text>',
    ]
    for index, (label, values) in enumerate(groups):
        color = palette[index % len(palette)]
        mapped = " ".join(f"{left + (x - min_x) / (max_x - min_x) * chart_w:.2f},{top + (1 - (y - min_y) / (max_y - min_y)) * chart_h:.2f}" for x, y in values)
        parts.append(f'<polyline points="{mapped}" fill="none" stroke="{color}" stroke-width="2"><title>{escape(label)}</title></polyline>')
        legend_x = left + (index % 3) * 245
        legend_y = height - 30 - (index // 3) * 16
        parts.append(f'<line x1="{legend_x}" y1="{legend_y - 4}" x2="{legend_x + 18}" y2="{legend_y - 4}" stroke="{color}" stroke-width="3"/><text x="{legend_x + 24}" y="{legend_y}" font-size="11" fill="#17212b">{escape(label[:32])}</text>')
    parts.append(f'<text x="{left + chart_w / 2:.0f}" y="{height - 8}" text-anchor="middle" font-size="12" fill="#5e6b76">world x (m)</text><text x="14" y="{top + chart_h / 2:.0f}" transform="rotate(-90 14 {top + chart_h / 2:.0f})" text-anchor="middle" font-size="12" fill="#5e6b76">world y (m)</text></svg>')
    return "".join(parts)


def _svg_ig(rows: Sequence[Mapping[str, Any]], *, title: str = "Integrated Gradients") -> str:
    values = []
    for row in rows:
        value = _number(row.get("ig"))
        if value is not None:
            index = row.get("index", "?")
            # Keep each evaluation separate.  A feature index alone would
            # make step 75 and step 80 bars look like one measurement.
            label = _ig_context_label(row, index)
            values.append((label, value))
    values = sorted(values, key=lambda item: abs(item[1]), reverse=True)[:20]
    if not values:
        return _svg_placeholder(title, "IG was not executed or no saved values were found")
    width, height, left, top = 1120, max(250, 36 * len(values) + 50), 430, 30
    maximum = max(max(abs(value) for _, value in values), 1e-12)
    zero = left + 260
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}"><rect width="100%" height="100%" fill="white"/><text x="20" y="22" font-size="16" fill="#17212b">{escape(title)}</text>']
    for index, (label, value) in enumerate(values):
        y = top + index * 34
        bar = 240 * abs(value) / maximum
        x = zero - bar if value < 0 else zero
        parts.append(f'<text x="{left - 8}" y="{y + 12}" text-anchor="end" font-size="11" fill="#17212b">{escape(label[:64])}</text><rect x="{x:.2f}" y="{y}" width="{bar:.2f}" height="18" fill="{"#d05a47" if value < 0 else "#3b82b6"}"><title>{escape(label)}: {value:.5g}</title></rect><text x="{zero + 252}" y="{y + 12}" font-size="11" fill="#5e6b76">{value:.5g}</text>')
    parts.append(f'<line x1="{zero}" y1="{top - 6}" x2="{zero}" y2="{height - 18}" stroke="#17212b"/></svg>')
    return "".join(parts)


def _scan_step_records(run_dir: Path, metadata: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    selected_roots = _stage_roots(run_dir, metadata or {})

    def number_from(mapping: Mapping[str, Any], names: Sequence[str]) -> float | None:
        for name in names:
            value = _number(mapping.get(name))
            if value is not None:
                return value
        return None

    def pair_from(mapping: Mapping[str, Any]) -> tuple[float | None, float | None]:
        for name in ("position_xy_m", "position_xy", "world_position"):
            value = mapping.get(name)
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                return _number(value[0]), _number(value[1])
        return number_from(mapping, ("x_m", "position_x_m", "world_x_m")), number_from(mapping, ("y_m", "position_y_m", "world_y_m"))

    def path_pattern(path: Path) -> str:
        lowered = {part.lower() for part in path.parts}
        if "00_reference" in lowered:
            return "P00"
        for part in reversed(path.parts):
            if part == "P00" or part == "LIDAR" or re.fullmatch(r"P\d+.*", part) or re.fullmatch(r"input_\d+", part):
                return part
        return "unknown"

    def prepare(record: Mapping[str, Any], path: Path, default_pattern: str) -> dict[str, Any]:
        post = record.get("post_telemetry") if isinstance(record.get("post_telemetry"), Mapping) else {}
        pre = record.get("pre_telemetry") if isinstance(record.get("pre_telemetry"), Mapping) else {}
        decoded = record.get("decoded_action") if isinstance(record.get("decoded_action"), Mapping) else {}
        pattern = str(_first(record, "pattern_id", "pattern", default=default_pattern))
        episode = str(_first(record, "episode_id", "episode", default="episode-unknown"))
        step = str(_first(record, "step", "observation_index", default=len(records)))
        row = dict(record)
        row["_plot_pattern_id"] = pattern
        row["_plot_episode_id"] = episode
        row["_plot_post_time"] = number_from(record, ("post_time",))
        if row["_plot_post_time"] is None:
            row["_plot_post_time"] = number_from(post, ("sim_time_s", "sim_time_seconds", "simulation_time"))
        row["_plot_pre_time"] = number_from(record, ("pre_time",))
        if row["_plot_pre_time"] is None:
            row["_plot_pre_time"] = number_from(pre, ("sim_time_s", "sim_time_seconds", "simulation_time"))
        row["_plot_lateral"] = number_from(post, ("target_lane_lateral_error_m", "target_lane_offset_m", "target_lane_error_m", "signed_target_lane_offset_m"))
        row["_plot_speed"] = number_from(post, ("speed_mps", "speed_m_s", "vehicle_speed_mps"))
        row["_plot_steering"] = number_from(decoded, ("steering", "steering_command", "steer_command"))
        row["_plot_x"], row["_plot_y"] = pair_from(post)
        if row["_plot_x"] is None or row["_plot_y"] is None:
            row["_plot_x"], row["_plot_y"] = pair_from(record)
        row["_plot_source"] = _safe_relative(path, run_dir) or path.name
        return row

    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name in {"report.jsonl", "report.html", "report.md", "report_details.json", "report_details.md", "offline_details.csv", "closed_loop_details.csv", "summary.csv"} or path.name.startswith("report_"):
            continue
        kind = _classify(path)
        if kind != "closed_loop" and "reference" not in {part.lower() for part in path.parts}:
            continue
        selected_root = selected_roots.get("closed_loop")
        if kind == "closed_loop" and selected_root is False:
            continue
        if kind == "closed_loop" and isinstance(selected_root, Path) and not _inside(path, selected_root):
            continue
        default_pattern = path_pattern(path)
        source_rows: list[Mapping[str, Any]] = []
        if path.suffix.lower() == ".jsonl":
            source_rows = _read_jsonl(path)
        elif path.suffix.lower() == ".json":
            payload = _read_json(path)
            if isinstance(payload, Mapping) and isinstance(payload.get("records"), list):
                default_pattern = str(_first(payload, "pattern_id", "pattern", default=default_pattern))
                source_rows = _as_rows(payload.get("records"))
        for source_index, source in enumerate(source_rows):
            row = prepare(source, path, default_pattern)
            key = (str(row.get("_plot_pattern_id")), str(row.get("_plot_episode_id")), str(_first(row, "step", "observation_index", default=f"row-{source_index}")))
            if key in seen:
                continue
            seen.add(key)
            records.append(row)
    return records


def _media_links(run_dir: Path, artifacts: Sequence[Path], metadata: Mapping[str, Any] | None = None, link_base: Path | None = None) -> list[dict[str, str]]:
    links: list[dict[str, str]] = []
    seen: set[str] = set()
    link_base = link_base or run_dir
    metadata = metadata or {}
    explicit: dict[Path, str] = {}
    for key in ("media", "media_links", "videos", "visualizations"):
        value = metadata.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, Mapping):
                raw_path = _first(item, "path", "relative_path", "file", "filename")
                timing = _first(item, "time", "time_mapping", "timing", "simulation_time", default="saved media metadata")
            else:
                raw_path, timing = item, "saved media metadata"
            if raw_path is None:
                continue
            candidate = Path(str(raw_path))
            if not candidate.is_absolute():
                candidate = run_dir / candidate
            candidate = candidate.resolve()
            if _safe_relative(candidate, run_dir) is not None and candidate.is_file():
                explicit[candidate] = str(timing)

    media_suffixes = {".gif", ".mp4", ".webm", ".png", ".jpg", ".jpeg"}
    media_paths = {path.resolve() for path in artifacts if path.suffix.lower() in media_suffixes and path.is_file()}
    media_paths.update(explicit)
    # A collection-level ``00_reference/video.json`` describes media below
    # ``00_reference/`` (the GIF is commonly in ``video/`` while frames are
    # in ``video/episode-N/``).  Closed-loop maps are usually closer to the
    # clip.  Keep all maps and choose the nearest ancestor for each media
    # path, rather than requiring map and clip to share a directory.
    video_maps: dict[Path, tuple[Path, Mapping[str, Any]]] = {}
    for path in artifacts:
        if path.name != "video.json" or not path.is_file():
            continue
        payload = _read_json(path)
        if not isinstance(payload, Mapping):
            continue
        video_dir = path.parent.resolve()
        video_maps[video_dir] = (path.resolve(), payload)
        raw_gif = payload.get("gif")
        if raw_gif:
            gif = Path(str(raw_gif))
            if not gif.is_absolute():
                gif = path.parent / gif
            gif = gif.resolve()
            if gif.is_file() and _safe_relative(gif, run_dir) is not None:
                media_paths.add(gif)

    def map_for(candidate: Path) -> tuple[Path, Mapping[str, Any]] | None:
        resolved = candidate.resolve()
        matches = [
            (parent, entry)
            for parent, entry in video_maps.items()
            if _inside(resolved, parent)
        ]
        if not matches:
            return None
        _parent, entry = max(matches, key=lambda item: len(item[0].parts))
        return entry

    def add_link(candidate: Path, label: str, timing: str) -> None:
        relative = _safe_relative(candidate, run_dir)
        if relative is None or relative in seen or not candidate.is_file():
            return
        seen.add(relative)
        links.append({"path": Path(os.path.relpath(candidate, link_base)).as_posix(), "label": label, "time": timing})

    def video_timing(payload: Mapping[str, Any]) -> str:
        fields = []
        if payload.get("fps") is not None:
            fields.append(f"fps={payload['fps']}")
        if payload.get("frame_count") is not None:
            fields.append(f"frames={payload['frame_count']}")
        return "; ".join(fields) if fields else "saved frame timing map"

    clips = sorted(path for path in media_paths if path.suffix.lower() in {".gif", ".mp4", ".webm"})
    for clip in clips:
        relative = _safe_relative(clip, run_dir) or clip.name
        map_entry = map_for(clip)
        timing = explicit.get(clip, video_timing(map_entry[1]) if map_entry else "saved media metadata")
        add_link(clip, relative, timing)
        if map_entry:
            map_path, payload = map_entry
            map_relative = _safe_relative(map_path, run_dir) or map_path.name
            add_link(map_path, f"{map_relative} (frame map)", video_timing(payload))

    # A frame directory without a successfully encoded clip remains useful,
    # but do not dump hundreds of PNG links when a clip already represents it.
    def represented_by_clip(frame: Path) -> bool:
        frame_map = map_for(frame)
        for clip in clips:
            # A clip in the same video directory represents nested episode
            # frame directories as well.  If maps are available, also treat
            # all media covered by one map as one encoded sequence.
            if _inside(frame, clip.parent.resolve()):
                return True
            clip_map = map_for(clip)
            if frame_map and clip_map and frame_map[0] == clip_map[0]:
                return True
        return False

    frame_candidates = sorted(
        path
        for path in media_paths
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"} and not represented_by_clip(path)
    )
    by_parent: dict[Path, list[Path]] = {}
    for frame in frame_candidates:
        by_parent.setdefault(frame.parent.resolve(), []).append(frame)
    for parent, frames in sorted(by_parent.items(), key=lambda item: str(item[0])):
        frame = frames[0]
        relative = _safe_relative(frame, run_dir) or frame.name
        add_link(frame, f"{relative} (representative frame)", explicit.get(frame, "saved frame timing metadata"))
        map_entry = map_for(frame)
        if map_entry:
            map_path, payload = map_entry
            map_relative = _safe_relative(map_path, run_dir) or map_path.name
            add_link(map_path, f"{map_relative} (frame map)", video_timing(payload))
    return links


_VIDEO_REASON_LABELS = {
    "video_disabled": "動画無効",
    "disabled_by_video_patterns": "video.patternsで無効",
    "pattern_not_selected_by_video_patterns": "video対象pattern外",
    "frame_unavailable": "フレーム未取得",
    "video_generation_failed": "動画生成失敗",
    "video_partial": "動画生成が部分的",
    "video_status_not_recorded": "動画状態未記録",
    "video_artifact_missing": "動画artifact未記録",
    "video_pending": "動画状態pending",
    "video_not_generated_reason_unrecorded": "動画未生成（理由未記録）",
}


def _video_reason_label(reason: str) -> str:
    clean = str(reason).strip()
    normalized = clean.casefold()
    label = _VIDEO_REASON_LABELS.get(normalized)
    if label is None:
        return clean
    return f"{label} [{clean}]"


def _video_config(metadata: Mapping[str, Any]) -> tuple[Mapping[str, Any], bool]:
    """Return explicitly saved video config without guessing defaults."""

    candidates: list[Mapping[str, Any]] = [metadata]
    for key in ("resolved_config", "config"):
        value = metadata.get(key)
        if isinstance(value, Mapping):
            candidates.append(value)
    stages = metadata.get("stages")
    if isinstance(stages, Mapping):
        closed_stage = stages.get("closed_loop")
        if isinstance(closed_stage, Mapping):
            if isinstance(closed_stage.get("video"), Mapping):
                candidates.append(closed_stage)
            elif isinstance(closed_stage.get("video"), bool):
                return {"enabled": closed_stage["video"]}, True
    for candidate in candidates:
        value = candidate.get("video")
        if isinstance(value, Mapping):
            return value, True
    return {}, False


def _video_reason_lines(
    metadata: Mapping[str, Any],
    closed_raw: Sequence[Mapping[str, Any]],
    artifacts: Sequence[Path],
    media: Sequence[Mapping[str, str]],
) -> list[str]:
    """Summarize saved video omission/failure reasons without inventing them."""

    entries: list[tuple[str, str, Mapping[str, Any]]] = []

    def walk(mapping: Mapping[str, Any], pattern_hint: str | None = None, episode_hint: str = "__summary__") -> None:
        pattern = _first(mapping, "pattern_id", "target_id", "id", default=pattern_hint)
        pattern_id = str(pattern) if pattern not in (None, "") else (pattern_hint or "__unknown__")
        episode_id = _closed_episode_id(mapping, episode_hint)
        video = mapping.get("video")
        if isinstance(video, Mapping):
            entries.append((pattern_id, episode_id, video))
        for key in ("metrics", "summary", "closed_loop", "performance", "patterns"):
            child = mapping.get(key)
            if isinstance(child, Mapping):
                if key == "patterns":
                    for identifier, item in child.items():
                        if isinstance(item, Mapping):
                            walk(item, str(identifier), episode_hint)
                else:
                    walk(child, pattern_id, episode_id)
            elif isinstance(child, list):
                for item in child:
                    if isinstance(item, Mapping):
                        walk(item, pattern_id, episode_id)

    for row in closed_raw:
        if isinstance(row, Mapping):
            walk(row)
    # Closed-loop summary.json may be represented in manifest metadata rather
    # than as a row selected by _collect.  It carries per-pattern video
    # reason counts, so retain it as a compact fallback.
    for key in ("closed_loop", "summary"):
        value = metadata.get(key)
        if isinstance(value, Mapping):
            walk(value)

    # A saved video.json is authoritative evidence even when its clip was
    # removed later.  Derive only a small pattern/episode hint from its path.
    for path in artifacts:
        if path.name != "video.json" or not path.is_file():
            continue
        payload = _read_json(path)
        if not isinstance(payload, Mapping):
            continue
        parts = list(path.parts)
        pattern_hint = next((part for part in reversed(parts[:-2]) if part.startswith("P") or part.startswith("input_")), None)
        episode_hint = next((part for part in reversed(parts[:-1]) if part.startswith("episode-")), "__summary__")
        if any(key in payload for key in ("status", "reason", "frame_count", "gif")):
            entries.append((pattern_hint or "__unknown__", episode_hint, payload))

    reason_counts: dict[str, int] = {}
    seen_direct: set[tuple[str, str, str]] = set()
    seen_generated: set[tuple[str, str]] = set()
    direct_patterns: set[str] = set()
    summary_entries: list[tuple[str, Mapping[str, Any]]] = []
    generated_count = 0

    def add_reason(reason: str, count: int = 1) -> None:
        clean = str(reason).strip()
        if clean:
            reason_counts[clean] = reason_counts.get(clean, 0) + max(1, int(count))

    for pattern_id, episode_id, video in entries:
        not_generated = video.get("not_generated_reasons")
        if isinstance(not_generated, Mapping):
            summary_entries.append((pattern_id, not_generated))
            continue
        direct_patterns.add(pattern_id)
        status = str(video.get("status", "")).strip().casefold()
        enabled = _bool_value(video.get("enabled"))
        requested = _bool_value(video.get("requested"))
        frame_count = _number(video.get("frame_count"))
        reason = video.get("reason")
        reason_text = str(reason).strip() if reason not in (None, "") else ""
        if status in {"generated", "saved", "frames_saved_without_store"} and enabled is not False:
            generated_key = (pattern_id, episode_id)
            if generated_key not in seen_generated:
                seen_generated.add(generated_key)
                generated_count += 1
            continue
        if not reason_text:
            if enabled is False or requested is False:
                reason_text = "video_disabled"
            elif status in {"partial"} and frame_count in (None, 0):
                reason_text = "frame_unavailable"
            elif status in {"failed", "error"}:
                reason_text = "video_generation_failed"
            elif status in {"partial"}:
                reason_text = "video_partial"
            elif status in {"pending"}:
                reason_text = "video_pending"
            elif status in {"not_generated", "disabled"}:
                reason_text = "video_not_generated_reason_unrecorded"
            elif not status:
                reason_text = "video_status_not_recorded"
        if reason_text:
            key = (pattern_id, episode_id, reason_text)
            if key not in seen_direct:
                seen_direct.add(key)
                add_reason(reason_text)

    # Summary reason counts are episode-level; ignore them when the same
    # pattern already supplied explicit episode records to avoid step/summary
    # duplication in the headline.
    for pattern_id, reasons in summary_entries:
        if pattern_id in direct_patterns:
            continue
        for reason, count in reasons.items():
            add_reason(str(reason), int(_number(count) or 1))

    config, config_present = _video_config(metadata)
    if config_present:
        enabled = _bool_value(config.get("enabled"))
        patterns = config.get("patterns", _MISSING)
        if enabled is False:
            reason_counts.pop("video_not_generated_reason_unrecorded", None)
            reason_counts.pop("video_status_not_recorded", None)
        if enabled is False and not any("video_disabled" in reason for reason in reason_counts):
            add_reason("video_disabled (video.enabled=false)")
        elif patterns is False and not any("disabled_by_video_patterns" in reason for reason in reason_counts):
            add_reason("disabled_by_video_patterns (video.patterns=false)")
        elif isinstance(patterns, (list, tuple)) and not reason_counts:
            configured = {str(item).strip() for item in patterns if str(item).strip()}
            observed_patterns = {str(row.get("pattern_id", "")) for row in closed_raw if row.get("pattern_id") not in (None, "")}
            if observed_patterns - configured:
                add_reason("pattern_not_selected_by_video_patterns", len(observed_patterns - configured))

    if generated_count and not media and not reason_counts:
        add_reason("video_artifact_missing", generated_count)
    if not entries and not config_present and not media:
        add_reason("video_status_not_recorded")
    if not entries and config_present and not reason_counts and not media:
        add_reason("video_status_not_recorded")

    return [
        f"動画未生成理由: {_video_reason_label(reason)}（{count}件）"
        for reason, count in sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _execution_failure_kind(value: Any) -> str | None:
    """Classify a saved episode's execution state for the report headline.

    ``execution_status`` and ``assessment_status`` are deliberately separate
    fields.  A natural terminal outcome (collision, road departure, arrival,
    or horizon) is therefore not classified from its terminal reason alone.
    Only an explicit runtime error or intervention abort contributes here.
    """

    token = str(value or "").strip().casefold().replace("-", "_")
    if not token:
        return None
    if token in {
        "failed",
        "failure",
        "error",
        "runtime_error",
        "runtime_failure",
        "runtime_failed",
        "not_evaluable_runtime_error",
        "実行失敗",
    }:
        return "failed"
    if token in {
        "aborted",
        "abort",
        "interrupted",
        "intervention_abort",
        "not_evaluable_intervention_abort",
        "実行失敗／中断",
        "中断",
    }:
        return "aborted"
    return None


def _closed_execution_failure_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count explicit failed/aborted episodes without treating outcomes as errors."""

    failed = 0
    aborted = 0
    completed = 0
    for row in rows:
        episode_rows = row.get("closed_loop_episode_rows")
        candidates = [item for item in episode_rows if isinstance(item, Mapping)] if isinstance(episode_rows, list) else [row]
        if not candidates:
            candidates = [row]
        for episode in candidates:
            kind = None
            for key in (
                "closed_loop_execution_status",
                "execution_status",
                "closed_loop_runtime_assessment_status",
                "assessment_status",
            ):
                kind = _execution_failure_kind(episode.get(key))
                if kind:
                    break
            if kind is None:
                terminal = str(
                    episode.get("closed_loop_termination", episode.get("terminal_reason", "")) or ""
                ).strip().casefold().replace("-", "_")
                if terminal == "runtime_error":
                    kind = "failed"
                elif terminal == "intervention_abort":
                    kind = "aborted"
            if kind == "failed":
                failed += 1
            elif kind == "aborted":
                aborted += 1
            elif str(episode.get("closed_loop_execution_status", episode.get("execution_status", ""))).casefold() == "completed":
                completed += 1
    return {"failed": failed, "aborted": aborted, "completed": completed}


def _metadata_execution_failure_counts(metadata: Mapping[str, Any]) -> dict[str, int]:
    """Read optional status counts when a report has no episode rows."""

    failed = 0
    aborted = 0
    completed = 0
    stages = metadata.get("stages")
    stage = stages.get("closed_loop") if isinstance(stages, Mapping) else None
    if not isinstance(stage, Mapping):
        return {"failed": 0, "aborted": 0, "completed": 0}
    source = stage.get("counts") if isinstance(stage.get("counts"), Mapping) else stage
    for key in ("failed_episode_count", "failed_count"):
        number = _number(source.get(key))
        if number is not None:
            failed = int(number)
            break
    for key in ("aborted_episode_count", "aborted_count"):
        number = _number(source.get(key))
        if number is not None:
            aborted = int(number)
            break
    number = _number(source.get("completed_episode_count"))
    if number is not None:
        completed = int(number)
    return {"failed": failed, "aborted": aborted, "completed": completed}


def _metadata_has_execution_failure_counts(metadata: Mapping[str, Any]) -> bool:
    """Whether the current closed-loop stage declares authoritative counts."""

    stages = metadata.get("stages")
    stage = stages.get("closed_loop") if isinstance(stages, Mapping) else None
    if not isinstance(stage, Mapping):
        return False
    source = stage.get("counts") if isinstance(stage.get("counts"), Mapping) else stage
    return isinstance(source, Mapping) and any(
        key in source
        for key in (
            "completed_episode_count",
            "failed_episode_count",
            "aborted_episode_count",
            "runtime_failure_count",
            "execution_failure_count",
        )
    )


def _report_execution_counts(
    metadata: Mapping[str, Any],
    closed: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, int], bool]:
    """Return one count set for the report header and whether it is known."""

    if _metadata_has_execution_failure_counts(metadata):
        return _metadata_execution_failure_counts(metadata), True
    if closed:
        return _closed_execution_failure_counts(
            [row for row in closed if isinstance(row, Mapping)]
        ), True
    return {"failed": 0, "aborted": 0, "completed": 0}, False


def _status_text(metadata: Mapping[str, Any], *, offline: Sequence[Any], closed: Sequence[Any], ig: Sequence[Any]) -> str:
    # The current closed-loop stage stores one authoritative count set.  Use
    # raw normalized episode rows only for old runs that predate that contract;
    # summing both would double-count the current stage and could mix an old
    # successful analysis directory into a failed run's headline.
    execution_failures, _known = _report_execution_counts(
        metadata,
        [row for row in closed if isinstance(row, Mapping)],
    )
    if execution_failures["failed"] or execution_failures["aborted"]:
        details = [
            f"completed={execution_failures['completed']}",
            f"failed={execution_failures['failed']}",
            f"aborted={execution_failures['aborted']}",
        ]
        prefix = "一部失敗" if execution_failures["completed"] else "失敗"
        return prefix + "（" + "、".join(details) + "）"
    direct = metadata.get("status")
    if isinstance(direct, Mapping):
        direct = direct.get("overall", direct.get("status"))
    stages = metadata.get("stages")
    if isinstance(stages, Mapping):
        # IG is optional; a failed/skipped IG stage must not turn the primary
        # collection/offline/closed-loop report into an overall failure.
        states = [
            str(value.get("state"))
            for name, value in stages.items()
            if str(name).lower() not in {"ig", "integrated_gradients"}
            and isinstance(value, Mapping)
            and value.get("state")
        ]
        if any(state in {"failed", "error"} for state in states):
            return "失敗（stageの一部が失敗）"
        if any(state in {"running", "pending"} for state in states):
            return "未完了（stage実行中）"
        optional_ig_failed = any(
            str(name).lower() in {"ig", "integrated_gradients"}
            and isinstance(value, Mapping)
            and str(value.get("state", "")).lower() in {"failed", "error"}
            for name, value in stages.items()
        )
        if optional_ig_failed and str(direct).lower() in {"failed", "error"} and not any(state in {"failed", "error"} for state in states):
            direct = None
    if direct:
        direct_text = str(direct)
        if direct_text.strip().casefold() in {"partial_failure", "partial_failed", "degraded"}:
            return "一部失敗"
        if direct_text.strip().casefold() in {"failed", "failure", "error"}:
            return "失敗"
        return direct_text
    if not offline and not closed:
        return "未完了／結果未記録"
    return "完了（③ IGは任意）"


def _row_classification(row: Mapping[str, Any]) -> str:
    """Return a saved comparison class without inferring scientific meaning."""

    def compact(value: Any) -> str | None:
        if isinstance(value, Mapping):
            values = [str(item).strip() for item in value.values() if str(item).strip()]
            if not values:
                return None
            unique = list(dict.fromkeys(values))
            if len(unique) == 1:
                return f"{unique[0]}（{len(values)}入力）"
            return f"複数分類（{len(values)}入力）"
        if isinstance(value, (list, tuple)):
            values = [str(item).strip() for item in value if str(item).strip()]
            if not values:
                return None
            unique = list(dict.fromkeys(values))
            return f"{unique[0]}（{len(values)}入力）" if len(unique) == 1 else f"複数分類（{len(values)}入力）"
        return str(value) if value not in (None, "") else None

    raw = row.get("raw_offline")
    candidates: list[Mapping[str, Any]] = [row]
    if isinstance(raw, Mapping):
        candidates.append(raw)
        for key in ("pattern", "metadata", "context"):
            value = raw.get(key)
            if isinstance(value, Mapping):
                candidates.append(value)
    for candidate in candidates:
        value = _first(
            candidate,
            "variant_classification",
            "classification",
            "comparison_class",
            "pattern_class",
            "analysis_class",
            "family",
            "input_class",
            "category",
            "scope_class",
            default=None,
        )
        if value not in (None, ""):
            compact_value = compact(value)
            if compact_value:
                return compact_value
    identifier = str(row.get("pattern_id", "")).lower()
    if identifier.startswith("input_"):
        return "individual"
    if identifier.startswith("group_"):
        return "group"
    if identifier.upper() in {"P00", "P0", "BASELINE", "NOOP", "NONE"}:
        return "baseline"
    return "configured"


def _findings(rows: Sequence[Mapping[str, Any]], *, closed: Sequence[Mapping[str, Any]], ig: Sequence[Mapping[str, Any]]) -> list[str]:
    findings: list[str] = []
    rates = [(_number(row.get("offline_actual_action_change_rate")), row) for row in rows]
    rates = [
        (value, row)
        for value, row in rates
        if value is not None
        and _number(row.get("offline_changed_count")) not in (None, 0)
        and str(row.get("pattern_id", "")).upper() not in {"P00", "P0", "BASELINE", "NOOP", "NONE"}
    ]
    if rates:
        by_class: dict[str, list[tuple[float, Mapping[str, Any]]]] = {}
        for value, row in rates:
            by_class.setdefault(_row_classification(row), []).append((value, row))
        for classification, values in sorted(by_class.items()):
            values_only = [value for value, _row in values]
            minimum, maximum = min(values_only), max(values_only)
            findings.append(
                f"①-A（{classification}）は実変更{len(values)}行、行動変更割合の範囲 {_pct(minimum)}〜{_pct(maximum)} です。分類をまたぐ順位付けはしていません。"
            )
        tiny_js_rows = [
            row
            for value, row in rates
            if value == 0
            and _number(row.get("offline_actual_js_divergence")) is not None
            and abs(_number(row.get("offline_actual_js_divergence"))) <= 1e-12
        ]
        if tiny_js_rows:
            precision_keys = (
                "precision",
                "tolerance",
                "meaningful_tolerance",
                "numeric_tolerance",
                "js_tolerance",
                "probability_tolerance",
                "action_probability_tolerance",
            )
            with_precision = [
                row
                for row in tiny_js_rows
                if _first_nested(row, *precision_keys, default=None) not in (None, "")
            ]
            without_precision = len(tiny_js_rows) - len(with_precision)
            if without_precision:
                precision_note = f"精度・許容幅が未記録の{without_precision}行を含むため"
                if with_precision:
                    precision_note += f"（保存済みは{len(with_precision)}行）"
            else:
                precision_note = "精度・許容幅は保存されていますが"
            findings.append(
                f"①-Aの{len(tiny_js_rows)}行はargmax変更がなくJSが微小値です。{precision_note}、これを意味のある依存とは断定しません。"
            )
    def lane_is_comparable(row: Mapping[str, Any]) -> bool:
        if row.get("closed_loop_assessment_status") in {"評価不能：変更なし", "評価不能：適用なし", "適用件数未記録", "実行失敗／中断", "実行失敗"}:
            return False
        if row.get("closed_loop_measurement_status") in {"一部未計測", "未計測"}:
            return False
        # A small lateral RMS during a stop or an interrupted/collided run is
        # an execution outcome, not evidence that the intervention preserved
        # lane keeping.  Keep the raw value in the tables/details while
        # excluding it from the compact comparative finding.
        termination = str(row.get("closed_loop_termination", "")).strip().lower()
        excluded_terms = (
            "abort",
            "中断",
            "stop",
            "stopped",
            "collision",
            "crash",
            "out_of_road",
            "road_out",
            "timeout",
        )
        if any(term in termination for term in excluded_terms):
            return False
        if _bool_value(row.get("closed_loop_crash")) is True or _bool_value(row.get("closed_loop_road_out")) is True:
            return False
        # Only an explicit successful arrival in a natural episode is a lane
        # comparison.  A missing arrival flag, a censored run, or a mixed
        # natural/aborted aggregate stays out of this compact finding.
        if _bool_value(row.get("closed_loop_arrived")) is not True:
            return False
        if (_number(row.get("closed_loop_interrupted_episode_count")) or 0.0) > 0:
            return False
        termination = str(row.get("closed_loop_termination", "")).strip().lower()
        if not termination or termination in {"未記録", "unknown", "n/a", "none"}:
            return False
        return True

    lane = [(_number(row.get("closed_loop_lane_rms_m")), row) for row in rows]
    lane = [
        (value, row)
        for value, row in lane
        if value is not None
        and str(row.get("pattern_id", "")).upper() not in {"P00", "P0", "BASELINE", "NOOP", "NONE"}
        and lane_is_comparable(row)
    ]
    if lane:
        lane_values = [value for value, _row in lane]
        findings.append(
            f"①-Bは到達済みで比較可能な{len(lane_values)}走行の横ずれRMS範囲が {_fmt(min(lane_values))}〜{_fmt(max(lane_values))} m です。停止・衝突・中断・未到達はこの比較から除外し、終了状態を別に示しています。"
        )
    if not rates:
        findings.append("①-Aの実変更時に比較できる行動変更割合はありません。変更なし・未実行・欠測を重要度0とは解釈していません。")
    if not closed:
        findings.append("①-Bの走り直し結果は保存されていません。レーン維持の結論は記載できません。")
    reason_groups: dict[str, list[str]] = {}
    for row in rows:
        reasons = [
            str(row.get(key)).strip()
            for key in ("offline_reason", "closed_loop_reason")
            if row.get(key) not in (None, "", "未記録")
        ]
        for reason in dict.fromkeys(reasons):
            reason_groups.setdefault(reason, []).append(str(row.get("pattern_id", "未記録")))
    for reason, identifiers in list(reason_groups.items())[:16]:
        shown = identifiers[0] if len(identifiers) == 1 else f"{identifiers[0]} 等（{len(identifiers)}パターン）"
        findings.append(f"{shown} スキップ・欠測理由: {reason}")
    if len(reason_groups) > 16:
        findings.append(f"その他 {len(reason_groups) - 16}種類の理由は詳細データを参照してください。")
    if ig:
        quality = _ig_quality(ig)
        findings.append(f"③ IG は補足として保存され、①-Aの順位や①-Bの性能へ合算していません（完全性: {quality['status']}）。")
        if quality["warnings"]:
            findings.append("③ IG 警告: " + "; ".join(quality["warnings"]))
    return findings


def _markdown(
    *,
    run_dir: Path,
    metadata: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    offline: Sequence[Mapping[str, Any]],
    closed: Sequence[Mapping[str, Any]],
    ig: Sequence[Mapping[str, Any]],
    media: Sequence[Mapping[str, str]],
    status: str,
    offline_details: Sequence[Mapping[str, Any]] = (),
    closed_details: Sequence[Mapping[str, Any]] = (),
    video_reasons: Sequence[str] = (),
) -> str:
    scope = _report_scope(
        metadata=metadata,
        rows=rows,
        offline=offline,
        closed=closed,
        offline_details=offline_details,
        closed_details=closed_details,
    )
    scope_line = _scope_text(scope)
    conclusion_lines = _conclusion_lines(rows=rows, offline=offline, closed=closed)
    p00_sentence = _p00_reference_sentence(rows)
    display_status = status if ig or "③ IG" not in status else status.split("（③ IG", 1)[0]
    execution_counts, execution_counts_known = _report_execution_counts(metadata, closed)
    intro = (
        "このレポートは保存済み観測・方策比較・走行テレメトリから生成しました。①-A は同じ観測の対象入力だけを置換した判断比較、①-B は置換後の方策で走り直した paired closed-loop 比較です。"
        + (" ③ Integrated Gradients は任意の補足です。" if ig else "")
    )
    method_state = f"- ①-A状態: {_method_status(metadata, 'offline', bool(offline))} / ①-B状態: {_method_status(metadata, 'closed_loop', bool(closed), closed)}"
    if ig:
        method_state += f" / ③状態: {_method_status(metadata, 'integrated_gradients', bool(ig))}"
    model = metadata.get("model")
    if isinstance(model, Mapping):
        model = _first(model, "name", "path", default=None)
    dimension = _first(metadata, "schema_dimension", "input_dimension", "observation_dimension", default=None)
    if dimension is None and isinstance(metadata.get("schema"), Mapping):
        dimension = _first(metadata["schema"], "dimension", "input_dimension", default=None)
    scenario = _first(metadata, "scenario", "scenario_name", "experiment", "run_id", default=None)
    package_version = _first(metadata, "package_version", default=None)
    offline_main = _summary_table(rows if rows else (), _OFFLINE_MAIN_COLUMNS, kind="offline")
    closed_main = _summary_table(rows if rows else (), _CLOSED_MAIN_COLUMNS, kind="closed")
    lines = [
        "# 入力依存度分析レポート",
        "",
        intro,
        "",
        f"- 実行ディレクトリ: `{run_dir}`",
        f"- 状態: **{display_status}**",
        *(
            [
                f"- 実行件数: completed={execution_counts['completed']} / failed={execution_counts['failed']} / aborted={execution_counts['aborted']}",
            ]
            if execution_counts_known
            else []
        ),
        method_state,
        f"- 対象: model={model or _NA} / 実観測次元={dimension or _NA} / scenario={scenario or _NA} / {scope_line} / version={package_version or _NA}",
        "- 解釈: 数値はこのモデル・対象シナリオ・指定置換条件における観測結果です。因果的な必要性や一般化を自動断定しません。",
        "",
        "## 結論と成立範囲",
        "",
        "実変更がある試験、変更なし、部分適用、適用不能、未実行を分けて表示しています。実変更0件の行は入力重要度の順位候補にしていません。到達・レーン維持・衝突・停止・中断も別状態です。",
        *conclusion_lines,
        "- 件数の軸: 一部適用は実変更あり・変更なしの分類と重複するため、各件数の合計をパターン総数とは解釈しません。",
        "",
        "## P00通常走行の基準（①-A表の前提）",
        "",
        p00_sentence,
        "",
        "## ①-A 主要比較（最大8列）",
        "",
        offline_main if rows else "結果行はありません。未実行・欠測を0として補完していません。",
        "",
        "適用件数/全対象と実変更件数/適用を分母付きで示しています。変更なしの確率差・JSのraw値は詳細へ残し、実変更時の列はN/Aにしています。JS divergence の単位は自然対数の nats です。",
        "",
        "## ①-B 主要比較（最大8列）",
        "",
        closed_main if rows else "結果行はありません。未実行・欠測を0として補完していません。",
        "",
        "①-Bの適用件数はそのB走行の介入記録から算出しています。target-lane telemetryのvalid率を介入率には使っていません。複数episodeの連続値はepisode平均、到達はepisode単位の件数/率です。P00差は検証済みepisodeのみです。",
        "",
        "## 数値から読める範囲",
        "",
    ]
    lines.extend(f"- {finding}" for finding in _findings(rows, closed=closed, ig=ig))
    lines.extend([
        "",
        "## 詳細データ",
        "",
        "[report_details.md](report_details.md)、[report_details.json](report_details.json)、[offline_details.csv](offline_details.csv)、[closed_loop_details.csv](closed_loop_details.csv) に全パターン・全episode・全stepのraw値、元値/変更値、donor/reference/context、skip理由、道路区間、action/probability、環境テレメトリを保存しています。個別LiDARなど主表から省略した行も詳細にあります。",
        "",
        "## coverage診断図",
        "",
        "coverage図は入力介入の適用/実変更/no-op/skip区間を確認するためのものです。道路IDは保存値を表示し、カーブ前後などの未承認フェーズを推定していません。",
        "",
        "- ![①-A coverage](report_offline_coverage.svg)",
        "- ![①-B coverage](report_closed_loop_coverage.svg)",
    ])
    ig_quality = _ig_quality(ig)
    if ig:
        lines.extend([
            "",
            "## ③ IG 完全性",
            "",
            f"- completeness status: **{ig_quality['status']}**",
            f"- 許容誤差: {_fmt(ig_quality['tolerance'])} / retry回数: {_fmt(ig_quality['retry_count'], digits=0)} / delta: {_fmt(ig_quality['delta'])} / absolute error: {_fmt(ig_quality['absolute_error'])}",
        ])
        if ig_quality["warnings"]:
            lines.extend(f"- 警告: {warning}" for warning in ig_quality["warnings"])
        lines.extend([
            "",
            "### IG 実行 context（analysis / episode / step 単位）",
            "",
            "各行は一つの IG 評価を表します。F_x は観測、F_b は baseline の選択行動スコアです。異なる step/action の attribution を合算していません。",
            "",
            _ig_context_markdown(ig),
        ])
    else:
        reason = _method_reason(metadata, "integrated_gradients")
        ig_line = "## ③ IG は未実行です。①-A/①-Bの主結果には影響しません。"
        if reason:
            ig_line += f"（理由: {reason}）"
        lines.extend(["", ig_line])
    lines.extend([
        "",
        "## 図",
        "",
        "- ![①-A 行動変更割合](report_offline_bars.svg)",
        "- ![①-B 横ずれ時系列](report_closed_loop_lateral.svg)",
        "- ![①-B 速度時系列](report_closed_loop_speed.svg)",
        "- ![①-B 行動前操舵時系列](report_closed_loop_steering.svg)",
        "- ![①-B 軌跡](report_closed_loop_trajectory.svg)",
    ])
    if ig:
        lines.append("- ![③ IG](report_ig.svg)")
    lines.extend([
        "",
        "## 未実行・欠測・媒体",
        "",
        "未実行、適用不能、依存関係不足、テレメトリ欠測は N/A または理由付きで表示します。映像のフレーム時刻は行動決定前観測と同一とは限らないため、保存された対応表を参照してください。",
        "",
    ])
    if media:
        lines.extend(f"- [{item['label']}]({item['path']})（{item['time']}）" for item in media)
    else:
        lines.append("媒体リンクはありません。")
    lines.extend(f"- {reason}" for reason in video_reasons)
    lines.extend([
        "",
        "## 参考手法との関係",
        "",
        "Greydanus et al. (ICML 2018) の入力摂動・方策/価値の変化を見る着想を参照しました。原論文の画像局所ぼかしと pre-softmax logit 二乗距離を、そのまま今回の指標とはしていません。今回の①-Aは保存されたベクトル観測の意味単位を置換し、確率分布の JS divergence と選択行動の確率差を比較します。",
    ])
    if ig:
        lines.extend([
            "",
            "Sundararajan et al. (ICML 2017) の Integrated Gradients を③に用います。選択行動の対数オッズを対象にすること、基準観測とカテゴリ/有効フラグの互換条件を固定することは今回の適用設計です。",
        ])
    lines.extend([
        "",
        "Atrey et al. の counterfactual 分析は仮説検証の必要性を示す参考であり、今回のセンサー観測コピー介入手順そのものではありません。",
        "",
        "この文書は保存結果から再生成できます。",
        "",
        "## 再生成",
        "",
        "```bash",
        f"python -m input_attribution report --run-dir {run_dir}",
        "```",
        "",
    ])
    return "\n".join(lines)


def _detect_font_family() -> str:
    """Choose a known local Japanese family without importing a font library."""

    roots = (Path("/usr/share/fonts"), Path("/usr/local/share/fonts"), Path.home() / ".fonts")
    known = ("NotoSansCJK", "NotoSansJP", "IPAexGothic", "IPAGothic", "YuGothic")
    for root in roots:
        if not root.is_dir():
            continue
        try:
            names = [path.name.lower() for path in root.rglob("*") if path.suffix.lower() in {".ttf", ".otf", ".ttc"}]
        except OSError:
            names = []
        if any(any(token.lower() in name for token in known) for name in names):
            return '"Noto Sans JP", "Yu Gothic", system-ui, sans-serif'
    return "system-ui, -apple-system, sans-serif"


def _html(*, markdown: str, rows: Sequence[Mapping[str, Any]], status: str, svgs: Mapping[str, str], media: Sequence[Mapping[str, str]], closed: Sequence[Mapping[str, Any]] = (), ig: Sequence[Mapping[str, Any]] = (), metadata: Mapping[str, Any] | None = None, font_family: str | None = None, offline: Sequence[Mapping[str, Any]] = (), offline_details: Sequence[Mapping[str, Any]] = (), closed_details: Sequence[Mapping[str, Any]] = (), video_reasons: Sequence[str] = ()) -> str:
    findings = _findings(rows, closed=closed, ig=ig)
    ig_quality = _ig_quality(ig)
    ig_warning_html = "".join(f"<li>警告: {escape(warning)}</li>" for warning in ig_quality["warnings"]) if ig else ""
    ig_reason = _method_reason(metadata or {}, "integrated_gradients")
    media_html = "".join(f'<li><a href="{quote(item["path"])}">{escape(item["label"])}</a>（{escape(item["time"])}）</li>' for item in media)
    if not media_html:
        media_html = "<li>媒体リンクはありません。</li>"
    media_html += "".join(f"<li>{escape(reason)}</li>" for reason in video_reasons)
    offline_html = _summary_table_html(rows, _OFFLINE_MAIN_COLUMNS, kind="offline")
    closed_html = _summary_table_html(rows, _CLOSED_MAIN_COLUMNS, kind="closed")
    finding_html = "".join(f"<li>{escape(finding)}</li>" for finding in findings)
    scope = _report_scope(
        metadata=metadata or {},
        rows=rows,
        offline=offline,
        closed=closed,
        offline_details=offline_details,
        closed_details=closed_details,
    )
    scope_line = escape(_scope_text(scope))
    conclusion_html = "".join(
        f"<li>{escape(line[2:] if line.startswith('- ') else line)}</li>"
        for line in _conclusion_lines(rows=rows, offline=offline, closed=closed)
    )
    p00_sentence = escape(_p00_reference_sentence(rows))
    display_status = status if ig or "③ IG" not in status else status.split("（③ IG", 1)[0]
    execution_counts, execution_counts_known = _report_execution_counts(metadata or {}, closed)
    execution_html = (
        f"<p>実行件数: completed={execution_counts['completed']} / failed={execution_counts['failed']} / aborted={execution_counts['aborted']}</p>"
        if execution_counts_known
        else ""
    )
    intro = "①-A は同一観測の入力置換による判断比較、①-B はその入力で走り直した性能比較です。" + (" ③ Integrated Gradients は任意の補足です。" if ig else "")
    coverage_links = (
        '<a href="report_offline_coverage.svg">①-A SVG</a> / '
        '<a href="report_closed_loop_coverage.svg">①-B SVG</a>'
    )
    if ig:
        ig_section = (
            f'<h2>③ IG 完全性</h2><p>status: <strong>{escape(str(ig_quality["status"]))}</strong> / '
            f'許容誤差: {escape(_fmt(ig_quality["tolerance"]))} / retry回数: '
            f'{escape(_fmt(ig_quality["retry_count"], digits=0))} / delta: '
            f'{escape(_fmt(ig_quality["delta"]))} / absolute error: '
            f'{escape(_fmt(ig_quality["absolute_error"]))}</p>'
            f'{f"<p>IG理由: {escape(ig_reason)}</p>" if ig_reason else ""}'
            f'<ul>{ig_warning_html}</ul>'
            '<h3>IG execution context（analysis / episode / step）</h3>'
            '<p>F_x は観測、F_b は baseline の選択行動スコアです。各step/actionを別行で示しています。</p>'
            f'{_ig_context_html(ig)}<figure><figcaption>③ IG 補足</figcaption>{svgs.get("ig", "")}</figure>'
        )
    else:
        ig_line = "③ IG は未実行です。①-A/①-Bの主結果には影響しません。"
        if ig_reason:
            ig_line += f"（理由: {ig_reason}）"
        ig_section = f"<h2>{escape(ig_line)}</h2>"
    return f'''<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>入力依存度分析レポート</title><style>
body{{font-family:{font_family or _detect_font_family()};line-height:1.6;color:#17212b;background:#fff;max-width:1280px;margin:0 auto;padding:1.5rem}}
table{{border-collapse:collapse;width:100%;font-size:.9rem;margin:.5rem 0 1.2rem}}th,td{{border:1px solid #ccd6e0;padding:.35rem .45rem;text-align:left;vertical-align:top}}th{{background:#eef3f7}}
.status{{padding:.6rem .8rem;border-left:5px solid #3b82b6;background:#f4f8fb}}.figures{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:1rem}}figure{{margin:0;border:1px solid #ccd6e0;padding:.5rem;background:#fafafa}}figure svg{{width:100%;height:auto}}small{{color:#5e6b76}}
</style></head><body><h1>入力依存度分析レポート</h1>
<p class="status">状態: <strong>{escape(display_status)}</strong></p>{execution_html}
<p>{escape(intro)} N/A は未実行・欠測を表し、0 とは解釈していません。</p>
<p>対象範囲: {scope_line}</p>
<h2>結論と成立範囲</h2><p>実変更、変更なし、部分適用、適用不能、未実行を分けています。実変更0件は入力重要度の順位候補にしていません。到達・レーン維持・衝突・停止・中断も別状態です。</p><ul>{conclusion_html}<li>件数の軸: 一部適用は実変更あり・変更なしの分類と重複するため、各件数の合計をパターン総数とは解釈しません。</li></ul>
<h2>P00通常走行の基準（①-A表の前提）</h2><p>{p00_sentence}</p>
<h2>①-A 主要比較（最大8列）</h2>{offline_html}<p><small>適用件数/全対象、実変更件数/適用、実変更時の選択確率差とJSを表示しています。全適用時刻のraw値やskip理由は詳細へ保存しています。</small></p>
<h2>①-B 主要比較（最大8列）</h2>{closed_html}<p><small>Bの母数はB走行の介入記録です。target-lane valid率は介入率ではありません。P00差は検証済みepisodeだけです。</small></p>
<h2>数値から読める範囲</h2><ul>{finding_html}</ul>
<p><a href="report_details.md">詳細Markdown</a> / <a href="report_details.json">詳細JSON</a> / <a href="offline_details.csv">①-A全step</a> / <a href="closed_loop_details.csv">①-B全step</a></p>
<h2>coverage診断</h2><p>適用/実変更/no-op/skip区間を確認する診断図です。道路IDは保存値を表示し、未承認のフェーズ分類を追加していません。</p>
<div class="figures"><figure><figcaption>①-A coverage（{coverage_links.split(" / ")[0]}）</figcaption>{svgs.get("offline_coverage", "")}</figure><figure><figcaption>①-B coverage（skip区間、{coverage_links.split(" / ")[1]}）</figcaption>{svgs.get("closed_loop_coverage", "")}</figure></div>
<h2>その他の図</h2><div class="figures"><figure><figcaption>①-A 行動変更割合</figcaption>{svgs.get("offline_bars", "")}</figure><figure><figcaption>①-B 横ずれ時系列（post_time）</figcaption>{svgs.get("lateral", "")}</figure><figure><figcaption>①-B 速度時系列（post_time）</figcaption>{svgs.get("speed", "")}</figure><figure><figcaption>①-B 行動前操舵時系列（pre_time）</figcaption>{svgs.get("steering", "")}</figure><figure><figcaption>①-B 世界座標軌跡</figcaption>{svgs.get("trajectory", "")}</figure></div>
{ig_section}
<h2>媒体リンクと時刻</h2><ul>{media_html}</ul>
<h2>参考手法との関係</h2><p>Greydanus et al. (ICML 2018) の入力摂動の着想を参照しました。今回の数値は保存されたベクトル観測の意味単位置換と確率比較であり、一般的重要度を自動断定しません。</p>{('<p>Sundararajan et al. (ICML 2017) の Integrated Gradients を③に用います。</p>' if ig else '')}
<p><small>保存結果のみから再生成できます。</small></p></body></html>'''


def _write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding=encoding, newline="\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_SUMMARY_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            values = {}
            for field in _SUMMARY_FIELDS:
                value = row.get(field)
                if isinstance(value, (Mapping, list, tuple)):
                    value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                values[field] = "" if value is None else value
            writer.writerow(values)


def _report_destination(root: Path, output_dir: str | os.PathLike[str] | None) -> Path:
    """Keep the first report at run root and version later regenerations."""

    if output_dir is not None:
        return Path(output_dir).expanduser()
    if not ((root / "report.md").exists() or (root / "report.html").exists()):
        return root
    reports = root / "reports"
    stamp = datetime.now(timezone.utc).strftime("report-%Y%m%dT%H%M%SZ")
    candidate = reports / stamp
    suffix = 2
    while candidate.exists():
        candidate = reports / f"{stamp}-{suffix}"
        suffix += 1
    return candidate


def generate_report(run_dir: str | os.PathLike[str], *, output_dir: str | os.PathLike[str] | None = None) -> ReportResult:
    """保存済み結果を読み、日本語 Markdown/HTML/CSV/SVG を生成する。"""

    root = Path(run_dir).expanduser()
    if not root.is_dir():
        raise ReportingError(f"run-dir がディレクトリではありません: {root}")
    metadata, offline_raw, closed_raw, ig_raw, artifacts = _collect(root)
    destination = _report_destination(root, output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    labels = metadata.get("_pattern_labels")
    labels = labels if isinstance(labels, Mapping) else None
    offline = _normalise_offline(offline_raw, labels)
    closed = _normalise_closed(closed_raw, labels)
    ig = _normalise_ig(ig_raw)
    rows = _add_p00_deltas(_join_rows(offline, closed))
    status = _status_text(metadata, offline=offline, closed=closed, ig=ig)
    media = _media_links(root, artifacts, metadata, destination)
    video_reasons = _video_reason_lines(metadata, closed_raw, artifacts, media)
    step_records = _scan_step_records(root, metadata)
    offline_details = _offline_detail_records(offline_raw)
    closed_details = _closed_detail_records(step_records)
    svgs = {
        "offline_bars": _svg_bars(rows),
        "offline_coverage": _svg_coverage(
            _coverage_records(offline_details),
            title="Offline intervention coverage (applied / changed / skipped)",
            mode="offline",
        ),
        "closed_loop_coverage": _svg_coverage(
            _coverage_records(closed_details),
            title="Closed-loop intervention coverage (applied / changed / skipped)",
            mode="closed_loop",
        ),
        "lateral": _svg_series(
            step_records,
            y_names=("_plot_lateral",),
            time_names=("_plot_post_time",),
            title="Closed-loop lateral error",
            y_label="lateral error (m)",
        ),
        "speed": _svg_series(
            step_records,
            y_names=("_plot_speed",),
            time_names=("_plot_post_time",),
            title="Closed-loop speed",
            y_label="speed (m/s)",
        ),
        "steering": _svg_series(
            step_records,
            y_names=("_plot_steering",),
            time_names=("_plot_pre_time",),
            title="Pre-action steering",
            y_label="steering",
        ),
        "trajectory": _svg_trajectory(step_records),
        "ig": _svg_ig(ig),
    }
    markdown = _markdown(
        run_dir=root,
        metadata=metadata,
        rows=rows,
        offline=offline,
        closed=closed,
        ig=ig,
        media=media,
        status=status,
        offline_details=offline_details,
        closed_details=closed_details,
        video_reasons=video_reasons,
    )
    html = _html(
        markdown=markdown,
        rows=rows,
        status=status,
        svgs=svgs,
        media=media,
        closed=closed,
        ig=ig,
        metadata=metadata,
        offline=offline,
        offline_details=offline_details,
        closed_details=closed_details,
        video_reasons=video_reasons,
    )
    _write_text(destination / "report.md", markdown)
    _write_text(destination / "report.html", html)
    _write_csv(destination / "summary.csv", rows)
    _write_detail_csv(destination / "offline_details.csv", offline_details)
    _write_detail_csv(destination / "closed_loop_details.csv", closed_details)
    details_payload = {
        "source_run_dir": str(root),
        "metadata": _plain(metadata),
        "summary_rows": _without_detail_raw(rows),
        "offline_rows": _without_detail_raw(offline),
        "closed_loop_rows": _without_detail_raw(closed),
        "integrated_gradients_rows": _plain(ig),
        "offline_details": _plain(offline_details),
        "closed_loop_details": _plain(closed_details),
        # ``offline_details`` above keeps every source/step.  This top-level
        # raw section is reduced to authoritative pattern containers so
        # diagnostic NPZ/CSV expansions do not make the JSON quadratic.
        "raw_offline": _plain(_compact_offline_containers(offline_raw)),
        # ``_collect`` expands trajectory containers into one row per step;
        # the step rows above retain that complete information, so keep one
        # raw container per source/pattern here to avoid quadratic duplication.
        "raw_closed_loop": _plain(_compact_raw_containers(closed_raw)),
        "raw_integrated_gradients": _plain(ig_raw),
    }
    _write_text(destination / "report_details.json", json.dumps(details_payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n")
    _write_text(destination / "report_details.md", _details_markdown(rows=rows, offline_details=offline_details, closed_details=closed_details, ig=ig))
    generated: list[Path] = [
        destination / "report.md",
        destination / "report.html",
        destination / "summary.csv",
        destination / "report_details.md",
        destination / "report_details.json",
        destination / "offline_details.csv",
        destination / "closed_loop_details.csv",
    ]
    for name, content in (("report_offline_bars.svg", svgs["offline_bars"]), ("report_offline_coverage.svg", svgs["offline_coverage"]), ("report_closed_loop_coverage.svg", svgs["closed_loop_coverage"]), ("report_closed_loop_lateral.svg", svgs["lateral"]), ("report_closed_loop_speed.svg", svgs["speed"]), ("report_closed_loop_steering.svg", svgs["steering"]), ("report_closed_loop_trajectory.svg", svgs["trajectory"])):
        _write_text(destination / name, content)
        generated.append(destination / name)
    if ig:
        _write_text(destination / "report_ig.svg", svgs["ig"])
        generated.append(destination / "report_ig.svg")
    report_manifest = {
        "status": status,
        "source_run_dir": str(root),
        "sections": {
            "offline": {"status": _method_status(metadata, "offline", bool(offline)), "rows": len(offline)},
            "closed_loop": {"status": _method_status(metadata, "closed_loop", bool(closed), closed), "rows": len(closed)},
            "integrated_gradients": {"status": _method_status(metadata, "integrated_gradients", bool(ig)), "rows": len(ig), "quality": _ig_quality(ig), "contexts": _ig_context_rows(ig)},
        },
        "media": media,
        "generated_files": [path.name for path in generated],
    }
    _write_text(destination / "report_manifest.json", json.dumps(report_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    generated.append(destination / "report_manifest.json")
    return ReportResult(root, destination, status, tuple(rows), tuple(generated))


# Names used by early CLI drafts remain aliases so report regeneration is a
# stable boundary while the runtime package evolves.
build_report = generate_report
render_report = generate_report
write_report = generate_report
create_report = generate_report


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="保存済み input attribution 結果から日本語レポートを生成")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    result = generate_report(args.run_dir, output_dir=args.output_dir)
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
