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
    "closed_loop_lane_rms_m",
    "closed_loop_lane_max_abs_m",
    "closed_loop_departure_count",
    "closed_loop_departure_time_s",
    "closed_loop_first_departure_time_s",
    "closed_loop_ever_departed",
    "closed_loop_arrived",
    "closed_loop_arrival_rate",
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


def _method_status(payload: Mapping[str, Any], method: str, has_rows: bool) -> str:
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
    """Resolve the latest successful analysis directory from status.json.

    ``None`` means the old flat/canonical layout has no pointer and may be
    scanned.  ``False`` means a stage explicitly failed/skipped and stale
    analysis directories must not be mixed into this report.
    """

    stages = metadata.get("stages")
    if not isinstance(stages, Mapping):
        return {"offline": None, "closed_loop": None, "ig": None}
    result: dict[str, Path | None | bool] = {"offline": None, "closed_loop": None, "ig": None}
    for kind, stage_name in (("offline", "offline"), ("closed_loop", "closed_loop"), ("ig", "ig")):
        stage = stages.get(stage_name)
        if not isinstance(stage, Mapping):
            continue
        state = str(stage.get("state", ""))
        if state in {"failed", "skipped", "pending", "running"}:
            result[kind] = False
            continue
        relative = stage.get("relative_dir")
        if relative is None:
            analysis_id = stage.get("analysis_id")
            if analysis_id:
                relative = {"offline": "01_offline", "closed_loop": "02_closed_loop", "ig": "03_ig"}[kind] + "/" + str(analysis_id)
        if relative is None:
            # A successful legacy stage with no pointer is still readable from
            # its canonical directory.
            continue
        candidate = (run_dir / str(relative)).resolve()
        try:
            candidate.relative_to(run_dir.resolve())
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
                selected_delta = raw * 100.0 if abs(raw) <= 1.0 else raw
        js = _metric(row, "js_divergence", "js", "mean_js_divergence", "js_mean")
        replacement = _first(row, "replacement_description", "description", "replacement", "replacement_value", "method", default="未記録")
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
            "offline_reason": _first(row, "reason", "skip_reason", "message", default=""),
            "raw_offline": row,
        }
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
        applied = _summary_top_metric(row, "applied_row_count", "applied_count", "n_applied", "sample_count", "applicable_count")
        if applied is None:
            applied = _summary_section_metric(row, "all_timestamps", "count")
        if applied is None:
            applied = _metric(row, "all_timestamps_count", "count", "steps")
        changed = _summary_top_metric(row, "actual_changed_row_count", "changed_count", "n_changed", "actual_change_count", "modified_count", "change_count")
        noop = _summary_top_metric(row, "no_op_row_count", "noop_count", "no_op_count", "n_noop", "unchanged_count")
        skipped_count = _summary_top_metric(row, "skipped_row_count", "skipped_count", "n_skipped")
        valid_count = _summary_top_metric(row, "valid_row_count", "valid_count")
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
                selected_delta = raw * 100.0 if abs(raw) <= 1.0 else raw
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
                selected_delta = raw * 100.0 if abs(raw) <= 1.0 else raw
        js = None if is_skipped else _summary_section_metric(row, "all_timestamps", "js_divergence", "js", "mean_js_divergence", "js_mean")
        if js is None and not is_skipped:
            js = _metric(row, "js_divergence", "js", "mean_js_divergence", "js_mean")

        episode_rate = None if is_skipped else _episode_mean_metric(row, "action_change_rate")
        if episode_rate is None and not is_skipped:
            episode_rate = _metric(row, "episode_mean_action_change_rate", "episode_action_change_rate")
        step_rate = None if is_skipped else _summary_section_metric(row, "all_timestamps", "action_change_rate")
        if step_rate is None and not is_skipped:
            step_rate = _metric(row, "step_weighted_action_change_rate", "step_action_change_rate")

        replacement = _first(row, "replacement_description", "description", "replacement", "replacement_value", "method", default="未記録")
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
            "offline_reason": _format_offline_reason(
                _offline_reason_parts(row),
                applied=applied,
                skipped=skipped_count,
                valid=valid_count,
            ),
            "raw_offline": row,
        }
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
                    if value is not None and abs(value) <= 1.0:
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
        value = _metric(row, *names)
        if value is not None:
            values.append(value)
    return values


def _aggregate_closed_steps(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Derive only directly measured aggregates from per-step telemetry.

    A closed-loop trajectory stores pre/post telemetry beside each action.  The
    report may calculate RMS and maxima from those physical values, but it does
    not invent a departure threshold or replace missing target-lane telemetry
    with current-lane values.
    """

    lateral: list[float] = []
    for row in rows:
        target_valid = _bool_metric(row, "target_lane_valid", "lane_reference_valid", "target_lane_reference_valid")
        if target_valid is False:
            continue
        value = _metric(row, "target_lane_lateral_error_m", "target_lane_offset_m", "target_lane_error_m", "signed_target_lane_offset_m")
        if value is not None:
            lateral.append(value)
    speed = _closed_step_values(rows, "speed_mps", "speed_m_s", "vehicle_speed_mps", "speed")
    progress = _closed_step_values(rows, "route_progress_m", "progress_m", "route_progress", "progress")
    times = _closed_step_values(rows, "simulation_time_s", "sim_time_s", "sim_time_seconds", "time_s", "elapsed_seconds")
    rewards = _closed_step_values(rows, "reward")
    result: dict[str, Any] = {}
    if lateral:
        result["closed_loop_lane_rms_m"] = math.sqrt(sum(value * value for value in lateral) / len(lateral))
        result["closed_loop_lane_max_abs_m"] = max(abs(value) for value in lateral)
        result["closed_loop_lateral_valid_count"] = len(lateral)
        result["closed_loop_valid_count"] = len(lateral)
    result["closed_loop_poststep_count"] = len(rows)
    if lateral and rows:
        result["closed_loop_valid_rate"] = len(lateral) / len(rows)
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
    departure_flags = [_bool_metric(row, "target_lane_departure", "lane_departure", "departed_target_lane") for row in rows]
    if any(value is not None for value in departure_flags):
        result["closed_loop_departure_count"] = sum(1 for value in departure_flags if value is True)
        result["closed_loop_ever_departed"] = any(value is True for value in departure_flags)
    arrivals = [_bool_metric(row, "arrived", "success", "reached_goal", "arrival", "arrive_dest") for row in rows]
    if any(value is not None for value in arrivals):
        result["closed_loop_arrived"] = any(value is True for value in arrivals)
    road_flags = [_bool_metric(row, "road_out", "out_of_road", "out_of_drivable") for row in rows]
    crash_flags = [_bool_metric(row, "crash", "crashed", "collision") for row in rows]
    if any(value is not None for value in road_flags):
        result["closed_loop_road_out"] = any(value is True for value in road_flags)
    if any(value is not None for value in crash_flags):
        result["closed_loop_crash"] = any(value is True for value in crash_flags)
    wrong_lane = [_bool_metric(row, "wrong_lane_arrival", "wrong_lane_goal", "arrive_wrong_lane", "goal_wrong_lane") for row in rows]
    start_lane = [_bool_metric(row, "start_lane_departure", "departed_start_lane", "start_lane_departed") for row in rows]
    if any(value is not None for value in wrong_lane):
        result["closed_loop_wrong_lane_arrival"] = any(value is True for value in wrong_lane)
    if any(value is not None for value in start_lane):
        result["closed_loop_start_lane_departure"] = any(value is True for value in start_lane)
    terminations = [_first(row, "termination_reason", "terminal_reason", "end_reason", "termination") for row in rows]
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


def _closed_metric_rows(row: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any], int]]:
    """Extract explicit episode metrics from summary/trajectory containers."""

    metrics = row.get("metrics")
    if isinstance(metrics, list):
        result: list[tuple[str, Mapping[str, Any], int]] = []
        for index, value in enumerate(metrics):
            if not isinstance(value, Mapping):
                continue
            episode = _closed_episode_id(value, _closed_episode_id(row, f"episode-{index}"))
            result.append((episode, {**dict(row), **dict(value)}, index))
        return result
    if isinstance(metrics, Mapping):
        episode = _closed_episode_id(metrics, _closed_episode_id(row))
        return [(episode, {**dict(row), **dict(metrics)}, 0)]
    if not _closed_is_step(row):
        return [(_closed_episode_id(row), row, 0)]
    return []


def _closed_item_from_metric(
    identifier: str,
    source: Mapping[str, Any],
    step_rows: Sequence[Mapping[str, Any]],
    labels: Mapping[str, str] | None,
) -> dict[str, Any]:
    """Build one episode row, filling only missing fields from telemetry."""

    nested = _nested(source, "closed_loop", "performance", "summary")
    merged = {**nested, **dict(source)}
    status = _status(merged, "完了")
    reason = _reason(merged)
    valid_count = _metric(merged, "valid_count")
    if not reason and status.strip().lower() in {"not_available", "unavailable", "na", "n/a"}:
        reason = "target-lane telemetry is unavailable"
        if valid_count == 0:
            reason += " (valid_count=0)"
    p00 = _first_nested(merged, "paired_p00", "p00", default={})
    p00 = p00 if isinstance(p00, Mapping) else {}
    paired_verified = _bool_metric(merged, "paired_p00_verified", "p00_pair_verified", "paired_p00_initial_comparison_verified")
    if paired_verified is None:
        paired_status = _first_nested(merged, "paired_p00_status", "p00_pair_status", default=None)
        if paired_status is None and isinstance(p00, Mapping):
            paired_status = _first(p00, "status", "alignment_status", default=None)
        if paired_status not in (None, ""):
            paired_verified = str(paired_status).strip().lower() in {"matched", "verified", "success", "ok", "passed"}
    item: dict[str, Any] = {
        "pattern_id": identifier,
        "episode_id": _closed_episode_id(merged),
        "pattern_name": _label(merged, labels),
        "closed_loop_status": status,
        "closed_loop_lane_rms_m": _metric(merged, "lane_rms_m", "lateral_error_rms_m", "target_lane_rms_m", "lateral_rms_m"),
        "closed_loop_lane_max_abs_m": _metric(merged, "lane_max_abs_m", "max_abs_lateral_error_m", "target_lane_max_abs_m", "lateral_max_abs_m"),
        "closed_loop_departure_count": _metric(merged, "departure_count", "lane_departure_count", "target_lane_departure_count"),
        "closed_loop_departure_time_s": _metric(merged, "departure_time_s"),
        "closed_loop_first_departure_time_s": _metric(merged, "first_departure_time_s"),
        "closed_loop_ever_departed": _first(merged, "ever_departed", "ever_departed_target_lane", default=None),
        "closed_loop_arrived": _first(merged, "arrived", "success", "reached_goal", "arrival", "arrive_dest", default=None),
        "closed_loop_wrong_lane_arrival": _first(merged, "wrong_lane_arrival", "wrong_lane_goal", "arrive_wrong_lane", "goal_wrong_lane", default=None),
        "closed_loop_start_lane_departure": _first(merged, "start_lane_departure", "departed_start_lane", "start_lane_departed", default=None),
        "closed_loop_road_out": _first(merged, "road_out", "out_of_road", "out_of_drivable", default=None),
        "closed_loop_crash": _first(merged, "crash", "crashed", "collision", default=None),
        "closed_loop_progress": _metric(merged, "progress", "route_progress", "progress_m", "route_progress_m"),
        "closed_loop_speed_mean_mps": _metric(merged, "speed_mean_mps", "speed_mean_m_s", "mean_speed_mps", "average_speed_mps", "speed_mean", "speed_m_s"),
        "closed_loop_low_speed_duration_s": _metric(merged, "low_speed_duration_s"),
        "closed_loop_duration_s": _metric(merged, "duration_s", "simulation_time_s", "sim_time_seconds", "elapsed_seconds", "time_s"),
        "closed_loop_termination": _first(merged, "termination_reason", "terminal_reason", "end_reason", "termination", "reason", default="未記録"),
        "closed_loop_reward": _metric(merged, "cumulative_reward", "return", "episode_return", "reward"),
        "closed_loop_cumulative_reward": _metric(merged, "cumulative_reward", "return", "episode_return", "reward"),
        "closed_loop_valid_count": _metric(merged, "valid_count"),
        "closed_loop_valid_time_s": _metric(merged, "valid_time_s", "valid_time_seconds", "valid_time"),
        "closed_loop_poststep_count": _metric(merged, "poststep_count"),
        "closed_loop_valid_rate": _metric(merged, "valid_rate"),
        "closed_loop_stop_fraction": _metric(merged, "stop_fraction", "low_speed_fraction", "stopped_fraction"),
        "closed_loop_action_switch_count": _metric(merged, "action_switch_count", "steering_switch_count"),
        "p00_delta_lane_rms_m": _metric(merged, "p00_delta_lane_rms_m"),
        "p00_delta_progress": _metric(merged, "p00_delta_progress", "p00_delta_progress_m"),
        "closed_loop_paired_p00_verified": paired_verified,
        "closed_loop_reason": reason,
        "raw_closed": merged,
    }
    initial_verified = _bool_metric(merged, "initial_comparison_verified")
    if initial_verified is None:
        initial_match = _first_nested(merged, "initial_match", default=None)
        if isinstance(initial_match, Mapping):
            initial_verified = _bool_value(initial_match.get("matched"))
    item["closed_loop_initial_comparison_verified"] = initial_verified
    if initial_verified is False:
        item["closed_loop_reason"] = "; ".join(value for value in (reason, "initial comparison was not verified") if value)
    if paired_verified is False:
        item["closed_loop_reason"] = "; ".join(value for value in (item.get("closed_loop_reason", ""), "P00 pairing was not verified") if value)
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
    }.items():
        if any(name in merged for name in names):
            authoritative.add(output_key)
    for key, value in aggregate.items():
        if key in authoritative:
            continue
        if item.get(key) in (None, "", "未記録"):
            item[key] = value
    if item.get("p00_delta_lane_rms_m") is None:
        item["p00_delta_lane_rms_m"] = _metric(p00, "p00_delta_lane_rms_m")
    if item.get("p00_delta_progress") is None:
        item["p00_delta_progress"] = _metric(p00, "p00_delta_progress", "p00_delta_progress_m")
    return item


def _mean_numbers(values: Iterable[Any]) -> float | None:
    numbers = [value for value in (_number(item) for item in values) if value is not None]
    return sum(numbers) / len(numbers) if numbers else None


def _sum_numbers(values: Iterable[Any]) -> float | None:
    numbers = [value for value in (_number(item) for item in values) if value is not None]
    return sum(numbers) if numbers else None


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
            source = max(candidates, key=lambda item: sum(value is not None for value in item.values())) if candidates else {"pattern_id": identifier}
            episode_items.append(_closed_item_from_metric(identifier, source, step_groups.get(episode, ()), labels))

        statuses = list(dict.fromkeys(str(item.get("closed_loop_status", "未記録")) for item in episode_items))
        reasons = list(dict.fromkeys(str(item.get("closed_loop_reason", "")).strip() for item in episode_items if str(item.get("closed_loop_reason", "")).strip()))
        bool_values = {
            key: [_bool_value(item.get(key)) for item in episode_items]
            for key in ("closed_loop_arrived", "closed_loop_road_out", "closed_loop_crash", "closed_loop_ever_departed", "closed_loop_wrong_lane_arrival", "closed_loop_start_lane_departure", "closed_loop_initial_comparison_verified")
        }
        item: dict[str, Any] = {
            "pattern_id": identifier,
            "pattern_name": _label(episode_items[0], labels),
            "closed_loop_status": statuses[0] if len(statuses) == 1 else "複数episode",
            "closed_loop_reason": "; ".join(reasons),
            "closed_loop_episode_count": len(episode_items),
            "closed_loop_lane_rms_m": _mean_numbers(item.get("closed_loop_lane_rms_m") for item in episode_items),
            "closed_loop_lane_max_abs_m": max((value for value in (_number(item.get("closed_loop_lane_max_abs_m")) for item in episode_items) if value is not None), default=None),
            "closed_loop_departure_count": _sum_numbers(item.get("closed_loop_departure_count") for item in episode_items),
            "closed_loop_departure_time_s": _mean_numbers(item.get("closed_loop_departure_time_s") for item in episode_items),
            "closed_loop_first_departure_time_s": _mean_numbers(item.get("closed_loop_first_departure_time_s") for item in episode_items),
            "closed_loop_ever_departed": (any(value is True for value in bool_values["closed_loop_ever_departed"]) if any(value is not None for value in bool_values["closed_loop_ever_departed"]) else None),
            "closed_loop_arrived": all(value is True for value in bool_values["closed_loop_arrived"]) if all(value is not None for value in bool_values["closed_loop_arrived"]) else None,
            "closed_loop_arrival_rate": (sum(1.0 for value in bool_values["closed_loop_arrived"] if value is True) / sum(value is not None for value in bool_values["closed_loop_arrived"]) if any(value is not None for value in bool_values["closed_loop_arrived"]) else None),
            "closed_loop_wrong_lane_arrival": (any(value is True for value in bool_values["closed_loop_wrong_lane_arrival"]) if any(value is not None for value in bool_values["closed_loop_wrong_lane_arrival"]) else None),
            "closed_loop_start_lane_departure": (any(value is True for value in bool_values["closed_loop_start_lane_departure"]) if any(value is not None for value in bool_values["closed_loop_start_lane_departure"]) else None),
            "closed_loop_initial_comparison_verified": (all(value is True for value in bool_values["closed_loop_initial_comparison_verified"]) if all(value is not None for value in bool_values["closed_loop_initial_comparison_verified"]) else None),
            "closed_loop_paired_p00_verified": (all(value is True for value in (_bool_value(item.get("closed_loop_paired_p00_verified")) for item in episode_items)) if all(_bool_value(item.get("closed_loop_paired_p00_verified")) is not None for item in episode_items) else None),
            "closed_loop_road_out": (any(value is True for value in bool_values["closed_loop_road_out"]) if any(value is not None for value in bool_values["closed_loop_road_out"]) else None),
            "closed_loop_crash": (any(value is True for value in bool_values["closed_loop_crash"]) if any(value is not None for value in bool_values["closed_loop_crash"]) else None),
            "closed_loop_progress": _mean_numbers(item.get("closed_loop_progress") for item in episode_items),
            "closed_loop_speed_mean_mps": _mean_numbers(item.get("closed_loop_speed_mean_mps") for item in episode_items),
            "closed_loop_low_speed_duration_s": _mean_numbers(item.get("closed_loop_low_speed_duration_s") for item in episode_items),
            "closed_loop_duration_s": _mean_numbers(item.get("closed_loop_duration_s") for item in episode_items),
            "closed_loop_reward": _mean_numbers(item.get("closed_loop_reward") for item in episode_items),
            "closed_loop_cumulative_reward": _mean_numbers(item.get("closed_loop_cumulative_reward") for item in episode_items),
            "closed_loop_valid_count": _sum_numbers(item.get("closed_loop_valid_count") for item in episode_items),
            "closed_loop_valid_time_s": _sum_numbers(item.get("closed_loop_valid_time_s") for item in episode_items),
            "closed_loop_poststep_count": _sum_numbers(item.get("closed_loop_poststep_count") for item in episode_items),
            "closed_loop_valid_rate": _mean_numbers(item.get("closed_loop_valid_rate") for item in episode_items),
            "closed_loop_stop_fraction": _mean_numbers(item.get("closed_loop_stop_fraction") for item in episode_items),
            "closed_loop_action_switch_count": _mean_numbers(item.get("closed_loop_action_switch_count") for item in episode_items),
            "closed_loop_termination": (str(episode_items[0].get("closed_loop_termination")) if len(set(str(item.get("closed_loop_termination", "未記録")) for item in episode_items)) == 1 else f"複数（{len(episode_items)} episode）"),
            "p00_delta_lane_rms_m": _mean_numbers(item.get("p00_delta_lane_rms_m") for item in episode_items),
            "p00_delta_progress": _mean_numbers(item.get("p00_delta_progress") for item in episode_items),
            "closed_loop_episode_rows": episode_items,
            "raw_closed": {"episodes": [item.get("raw_closed", {}) for item in episode_items]},
        }
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


def _add_p00_deltas(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    baseline = _p00_row(rows)
    output = []
    for source in rows:
        row = dict(source)
        for key, delta_key in (
            ("closed_loop_lane_rms_m", "p00_delta_lane_rms_m"),
            ("closed_loop_progress", "p00_delta_progress"),
        ):
            if row.get("closed_loop_initial_comparison_verified") is False or row.get("closed_loop_paired_p00_verified") is False:
                row[delta_key] = None
                continue
            if _number(row.get(delta_key)) is not None:
                continue
            current = _number(row.get(key))
            reference = _number(baseline.get(key)) if baseline is not None else None
            if current is not None and reference is not None:
                row[delta_key] = current - reference
            else:
                # A missing lane/progress measurement is N/A for the
                # baseline as well.  Zero is reserved for a measured paired
                # comparison whose two values are equal.
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


def _svg_bars(rows: Sequence[Mapping[str, Any]], *, title: str = "Action change rate") -> str:
    values = []
    for row in rows:
        value = _number(row.get("offline_action_change_rate"))
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
    individual = sorted((item for item in values if item[0].lower().startswith("input_")), key=lambda item: (-item[2], item[0]))[:20]
    panels = [("Configured groups", configured)]
    if individual:
        panels.append(("Individual inputs (top 20)", individual))
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
        if not path.is_file() or path.name in {"report.jsonl", "report.html", "report.md"} or path.name.startswith("report_"):
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


def _status_text(metadata: Mapping[str, Any], *, offline: Sequence[Any], closed: Sequence[Any], ig: Sequence[Any]) -> str:
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
        return str(direct)
    if not offline and not closed:
        return "未完了／結果未記録"
    return "完了（③ IGは任意）"


def _findings(rows: Sequence[Mapping[str, Any]], *, closed: Sequence[Mapping[str, Any]], ig: Sequence[Mapping[str, Any]]) -> list[str]:
    findings: list[str] = []
    rates = [(_number(row.get("offline_action_change_rate")), row) for row in rows]
    rates = [(value, row) for value, row in rates if value is not None]
    if rates:
        value, row = max(rates, key=lambda item: item[0])
        findings.append(f"①-Aでは {row.get('pattern_id', '未記録')} の行動変更割合が {_pct(value)} でした（保存された条件の範囲）。")
    lane = [(_number(row.get("closed_loop_lane_rms_m")), row) for row in rows]
    lane = [(value, row) for value, row in lane if value is not None]
    if lane:
        value, row = min(lane, key=lambda item: item[0])
        findings.append(f"①-Bの横ずれRMS最小値は {row.get('pattern_id', '未記録')} の {_fmt(value)} m でした。進行度・速度と併読してください。")
    if not rates:
        findings.append("①-Aの有限な行動変更割合は保存されていません。未実行・欠測を 0 と解釈していません。")
    if not closed:
        findings.append("①-Bの走り直し結果は保存されていません。レーン維持の結論は記載できません。")
    if ig:
        quality = _ig_quality(ig)
        findings.append(f"③ IG は補足として保存され、①-Aの順位や①-Bの性能へ合算していません（完全性: {quality['status']}）。")
        if quality["warnings"]:
            findings.append("③ IG 警告: " + "; ".join(quality["warnings"]))
    else:
        findings.append("③ IG は未実行または保存値がありません。①-A/①-Bの主結果には影響しません。")
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
) -> str:
    lines = [
        "# 入力依存度分析レポート",
        "",
        "このレポートは保存済み観測・方策比較・走行テレメトリから生成しました。①-A は同じ観測の対象入力だけを置換した判断比較、①-B は置換後の方策で走り直した paired closed-loop 比較です。③ Integrated Gradients は任意の補足です。",
        "",
        f"- 実行ディレクトリ: `{run_dir}`",
        f"- 状態: **{status}**",
        f"- ①-A状態: {_method_status(metadata, 'offline', bool(offline))} / ①-B状態: {_method_status(metadata, 'closed_loop', bool(closed))} / ③状態: {_method_status(metadata, 'integrated_gradients', bool(ig))}",
        "- 解釈: 数値はこのモデル・対象シナリオ・指定置換条件における観測結果です。因果的な必要性や一般化を自動断定しません。",
        "",
        "## 比較結果",
        "",
        _table(rows) if rows else "結果行はありません。未実行・欠測を 0 として補完していません。",
        "",
        "①-A の適用件数・実変更件数・no-op件数を区別しています。step加重平均と episode 平均が保存されている場合も、別項目として扱います。JS divergence は自然対数の nats です。",
        "",
        "## 数値から読める範囲",
        "",
    ]
    ig_reason = _method_reason(metadata, "integrated_gradients")
    if ig_reason and not ig:
        lines.insert(7, f"- ③ IG理由: {ig_reason}")
    lines.extend(f"- {finding}" for finding in _findings(rows, closed=closed, ig=ig))
    episode_details = _episode_detail_table(rows)
    if episode_details:
        lines.extend([
            "",
            "## ①-B episode別明細",
            "",
            "主表の①-B数値は episode 平均（最大値・件数は集約値）です。以下で各 episode の値を確認できます。",
            "",
            episode_details,
    ])
    for row in rows:
        reasons = [str(row.get(key)) for key in ("offline_reason", "closed_loop_reason") if row.get(key) not in (None, "", "未記録")]
        if reasons:
            offline_status = str(row.get("offline_status", ""))
            closed_status = str(row.get("closed_loop_status", ""))
            label = (
                "スキップ・欠測理由"
                if any("スキップ" in value or "未実行" in value for value in (offline_status, closed_status))
                else "補足"
            )
            lines.append(f"- {row.get('pattern_id', '未記録')}: {label} — {'; '.join(reasons)}")
    ig_quality = _ig_quality(ig)
    lines.extend([
        "",
        "## ③ IG 完全性",
        "",
        f"- completeness status: **{ig_quality['status']}**",
        f"- 許容誤差: {_fmt(ig_quality['tolerance'])} / retry回数: {_fmt(ig_quality['retry_count'], digits=0)} / delta: {_fmt(ig_quality['delta'])} / absolute error: {_fmt(ig_quality['absolute_error'])}",
    ])
    if ig_quality["warnings"]:
        lines.extend(f"- 警告: {warning}" for warning in ig_quality["warnings"])
    elif not ig:
        lines.append("- IG は未実行です。完全性は確認していません。")
    lines.extend([
        "",
        "### IG 実行 context（analysis / episode / step 単位）",
        "",
        "各行は一つの IG 評価を表します。F_x は観測、F_b は baseline の選択行動スコアで、sum IG と output difference の差を residual として表示します。異なる step/action の attribution を合算していません。",
        "",
        _ig_context_markdown(ig),
        "",
        "## 図",
        "",
        "- ![①-A 行動変更割合](report_offline_bars.svg)",
        "- ![①-B 横ずれ時系列](report_closed_loop_lateral.svg)",
        "- ![①-B 速度時系列](report_closed_loop_speed.svg)",
        "- ![①-B 行動前操舵時系列](report_closed_loop_steering.svg)",
        "- ![①-B 軌跡](report_closed_loop_trajectory.svg)",
        "- ![③ IG](report_ig.svg)",
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
    lines.extend([
        "",
        "## 参考手法との関係",
        "",
        "Greydanus et al. (ICML 2018) の入力摂動・方策/価値の変化を見る着想を参照しました。原論文の画像局所ぼかしと pre-softmax logit 二乗距離を、そのまま今回の指標とはしていません。今回の①-Aは保存されたベクトル観測の意味単位を置換し、確率分布の JS divergence と選択行動の確率差を比較します。",
        "",
        "Sundararajan et al. (ICML 2017) の Integrated Gradients を③に用います。選択行動の対数オッズを対象にすること、基準観測とカテゴリ/有効フラグの互換条件を固定することは今回の適用設計です。Atrey et al. の counterfactual 分析は仮説検証の必要性を示す参考であり、今回のセンサー観測コピー介入手順そのものではありません。",
        "",
        "この文書は参考資料の該当節と Captum API の仕様を確認した範囲に基づき、論文全体を読了したという主張はしません。",
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


def _html(*, markdown: str, rows: Sequence[Mapping[str, Any]], status: str, svgs: Mapping[str, str], media: Sequence[Mapping[str, str]], closed: Sequence[Mapping[str, Any]] = (), ig: Sequence[Mapping[str, Any]] = (), metadata: Mapping[str, Any] | None = None, font_family: str | None = None) -> str:
    body_table = _table(rows).replace("|", "</td><td>") if rows else "結果行はありません。"
    # Render the markdown table independently so HTML has actual semantic rows.
    headers = [label for _, label in [
        ("pattern_id", "ID"), ("pattern_name", "対象入力"), ("offline_status", "①-A状態"), ("offline_reason", "①-A理由"), ("offline_action_change_rate", "①-A行動変更割合"),
        ("offline_episode_mean_action_change_rate", "episode平均"), ("offline_step_weighted_action_change_rate", "step加重平均"),
        ("offline_selected_probability_delta_pp", "選択行動確率差(pp)"), ("offline_js_divergence", "JS(nats)"),
        ("closed_loop_status", "①-B状態"), ("closed_loop_reason", "①-B理由"), ("closed_loop_episode_count", "episode数"), ("closed_loop_lane_rms_m", "①-B横ずれRMS(episode平均,m)"), ("closed_loop_lane_max_abs_m", "横ずれ最大(m)"), ("closed_loop_departure_count", "逸脱回数"), ("closed_loop_departure_time_s", "逸脱時間(episode平均,s)"), ("closed_loop_first_departure_time_s", "初回逸脱時刻(episode平均,s)"), ("closed_loop_ever_departed", "逸脱有無"), ("closed_loop_arrived", "到達"), ("closed_loop_arrival_rate", "到達率(episode)"), ("closed_loop_wrong_lane_arrival", "誤レーン到達"), ("closed_loop_start_lane_departure", "開始レーン逸脱"), ("closed_loop_initial_comparison_verified", "初期対照確認"), ("closed_loop_paired_p00_verified", "P00対応確認"), ("closed_loop_road_out", "道路外"), ("closed_loop_crash", "衝突"), ("closed_loop_progress", "進行度(episode平均)"), ("closed_loop_speed_mean_mps", "平均速度(episode平均,m/s)"), ("closed_loop_low_speed_duration_s", "低速時間(episode平均,s)"), ("closed_loop_stop_fraction", "停車割合(episode平均)"), ("closed_loop_duration_s", "走行時間(episode平均,s)"), ("closed_loop_cumulative_reward", "累積報酬(episode平均)"), ("closed_loop_action_switch_count", "Action切替回数(episode平均)"), ("closed_loop_valid_count", "target-lane有効件数"), ("closed_loop_valid_time_s", "target-lane有効時間(s)"), ("closed_loop_valid_rate", "target-lane有効率(episode平均)"), ("closed_loop_termination", "終了理由"), ("p00_delta_lane_rms_m", "P00差 横ずれRMS"), ("p00_delta_progress", "P00差 進行度"),
    ]]
    html_rows = []
    for row in rows:
        cells = [
            escape(str(row.get("pattern_id", _NA))), escape(str(row.get("pattern_name", _NA))), escape(str(row.get("offline_status", _NA))), escape(str(row.get("offline_reason", _NA) or _NA)), _pct(row.get("offline_action_change_rate")),
            _pct(row.get("offline_episode_mean_action_change_rate")), _pct(row.get("offline_step_weighted_action_change_rate")),
            _fmt(row.get("offline_selected_probability_delta_pp"), suffix=" pp"), _fmt(row.get("offline_js_divergence")),
            escape(str(row.get("closed_loop_status", _NA))), escape(str(row.get("closed_loop_reason", _NA) or _NA)), _fmt(row.get("closed_loop_episode_count")), _fmt(row.get("closed_loop_lane_rms_m")), _fmt(row.get("closed_loop_lane_max_abs_m")), _fmt(row.get("closed_loop_departure_count")), _fmt(row.get("closed_loop_departure_time_s")), _fmt(row.get("closed_loop_first_departure_time_s")), _fmt(row.get("closed_loop_ever_departed")), _fmt(row.get("closed_loop_arrived")), _pct(row.get("closed_loop_arrival_rate")), _fmt(row.get("closed_loop_wrong_lane_arrival")), _fmt(row.get("closed_loop_start_lane_departure")), _fmt(row.get("closed_loop_initial_comparison_verified")), _fmt(row.get("closed_loop_paired_p00_verified")), _fmt(row.get("closed_loop_road_out")), _fmt(row.get("closed_loop_crash")), _fmt(row.get("closed_loop_progress")), _fmt(row.get("closed_loop_speed_mean_mps")), _fmt(row.get("closed_loop_low_speed_duration_s")), _pct(row.get("closed_loop_stop_fraction")), _fmt(row.get("closed_loop_duration_s")), _fmt(row.get("closed_loop_cumulative_reward")), _fmt(row.get("closed_loop_action_switch_count")), _fmt(row.get("closed_loop_valid_count")), _fmt(row.get("closed_loop_valid_time_s")), _pct(row.get("closed_loop_valid_rate")), escape(str(row.get("closed_loop_termination", _NA))), _fmt(row.get("p00_delta_lane_rms_m")), _fmt(row.get("p00_delta_progress")),
        ]
        html_rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    media_html = "".join(f'<li><a href="{quote(item["path"])}">{escape(item["label"])}</a>（{escape(item["time"])}）</li>' for item in media) or "<li>媒体リンクはありません。</li>"
    findings = _findings(rows, closed=closed, ig=ig)
    ig_quality = _ig_quality(ig)
    ig_warning_html = "".join(f"<li>警告: {escape(warning)}</li>" for warning in ig_quality["warnings"])
    if not ig_warning_html and not ig:
        ig_warning_html = "<li>IG は未実行です。完全性は確認していません。</li>"
    ig_reason = _method_reason(metadata or {}, "integrated_gradients")
    return f'''<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>入力依存度分析レポート</title><style>
body{{font-family:{font_family or _detect_font_family()};line-height:1.6;color:#17212b;background:#fff;max-width:1200px;margin:0 auto;padding:1.5rem}}
table{{border-collapse:collapse;width:100%;font-size:.9rem}}th,td{{border:1px solid #ccd6e0;padding:.35rem .45rem;text-align:left;vertical-align:top}}th{{background:#eef3f7}}
.status{{padding:.6rem .8rem;border-left:5px solid #3b82b6;background:#f4f8fb}}.figures{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:1rem}}figure{{margin:0;border:1px solid #ccd6e0;padding:.5rem;background:#fafafa}}figure svg{{width:100%;height:auto}}small{{color:#5e6b76}}
</style></head><body><h1>入力依存度分析レポート</h1>
<p class="status">状態: <strong>{escape(status)}</strong></p>
<p>①-A は同一観測の入力置換による判断比較、①-B はその入力で走り直した性能比較、③は任意 IG の補足です。N/A は未実行・欠測を表し、0 とは解釈していません。</p>
<h2>比較結果</h2><table><thead><tr>{''.join(f'<th>{escape(header)}</th>' for header in headers)}</tr></thead><tbody>{''.join(html_rows) or f'<tr><td colspan="{len(headers)}">結果行はありません。</td></tr>'}</tbody></table>
<p><small>適用件数・実変更件数・no-op件数、step加重平均と episode 平均は混同しません。JS divergence の単位は nats です。</small></p>
<h2>数値から読める範囲</h2><ul>{''.join(f'<li>{escape(finding)}</li>' for finding in findings)}</ul>
<h2>③ IG 完全性</h2><p>status: <strong>{escape(str(ig_quality["status"]))}</strong> / 許容誤差: {escape(_fmt(ig_quality["tolerance"]))} / retry回数: {escape(_fmt(ig_quality["retry_count"], digits=0))} / delta: {escape(_fmt(ig_quality["delta"]))} / absolute error: {escape(_fmt(ig_quality["absolute_error"]))}</p>{f'<p>IG理由: {escape(ig_reason)}</p>' if ig_reason and not ig else ''}<ul>{ig_warning_html}</ul>
<h3>IG execution context（analysis / episode / step）</h3><p>F_x は観測、F_b は baseline の選択行動スコアです。各 step/action を別行で示し、attribution を合算していません。</p>{_ig_context_html(ig)}
<div class="figures"><figure><figcaption>①-A 行動変更割合</figcaption>{svgs.get("offline_bars", "")}</figure><figure><figcaption>①-B 横ずれ時系列（post_time）</figcaption>{svgs.get("lateral", "")}</figure><figure><figcaption>①-B 速度時系列（post_time）</figcaption>{svgs.get("speed", "")}</figure><figure><figcaption>①-B 行動前操舵時系列（pre_time）</figcaption>{svgs.get("steering", "")}</figure><figure><figcaption>①-B 世界座標軌跡</figcaption>{svgs.get("trajectory", "")}</figure><figure><figcaption>③ IG 補足</figcaption>{svgs.get("ig", "")}</figure></div>
<h2>媒体リンクと時刻</h2><ul>{media_html}</ul>
<h2>参考手法との関係</h2><p>Greydanus et al. (ICML 2018) の入力摂動の着想と、Sundararajan et al. (ICML 2017) の IG を参照しました。原論文の画像局所ぼかし・logit二乗距離を今回の JS 指標とはせず、ベクトル観測の意味単位置換と確率比較を採用しています。Atrey et al. は仮説を走行で確かめる参考であり、今回の観測コピー介入の手順そのものではありません。</p>
<p><small>保存結果のみから再生成できます。論文全体を読了したという主張は含みません。</small></p>
</body></html>'''


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
                if isinstance(value, list):
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
    step_records = _scan_step_records(root, metadata)
    svgs = {
        "offline_bars": _svg_bars(rows),
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
    markdown = _markdown(run_dir=root, metadata=metadata, rows=rows, offline=offline, closed=closed, ig=ig, media=media, status=status)
    html = _html(markdown=markdown, rows=rows, status=status, svgs=svgs, media=media, closed=closed, ig=ig, metadata=metadata)
    _write_text(destination / "report.md", markdown)
    _write_text(destination / "report.html", html)
    _write_csv(destination / "summary.csv", rows)
    generated: list[Path] = [destination / "report.md", destination / "report.html", destination / "summary.csv"]
    for name, content in (("report_offline_bars.svg", svgs["offline_bars"]), ("report_closed_loop_lateral.svg", svgs["lateral"]), ("report_closed_loop_speed.svg", svgs["speed"]), ("report_closed_loop_steering.svg", svgs["steering"]), ("report_closed_loop_trajectory.svg", svgs["trajectory"]), ("report_ig.svg", svgs["ig"])):
        _write_text(destination / name, content)
        generated.append(destination / name)
    report_manifest = {
        "status": status,
        "source_run_dir": str(root),
        "sections": {
            "offline": {"status": _method_status(metadata, "offline", bool(offline)), "rows": len(offline)},
            "closed_loop": {"status": _method_status(metadata, "closed_loop", bool(closed)), "rows": len(closed)},
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
