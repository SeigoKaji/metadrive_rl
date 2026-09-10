"""Re-render saved input-attribution runs into PNG, GIF, CSV, and HTML.

This module is intentionally a read-only consumer of the run data contract.
It never creates an environment, invokes a policy, or calls a reward
function.  In particular, importing it does not import MetaDrive, PyTorch,
Stable-Baselines3, or NumPy.  That separation is what makes ``report`` useful
on a small analysis machine after a run has been copied there.

The supported raw layout is::

    run/
      data/manifest.json
      data/P00_baseline.json
      data/P01.json
      data/P01_offline.json
      data/frames/P01/0.png

The loader is deliberately tolerant about optional metadata, while missing
numeric values remain missing.  It does not pad a shorter closed-loop run to
the length of the baseline and it does not turn unavailable reward terms or
JS values into zero.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import html
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable, Mapping, Sequence

from .visuals import (
    LN2,
    VisualResult,
    render_policy_change_plot,
    render_rewards_plot,
    render_rollout_gif,
)


class ReportingError(ValueError):
    """The saved run cannot be interpreted as a report."""


SUMMARY_COLUMNS: tuple[str, ...] = (
    "変更対象",
    "置換前→後",
    "baseline累積報酬",
    "変更後累積報酬",
    "報酬差",
    "終了step・理由",
)


@dataclass(frozen=True, slots=True)
class ReportResult:
    """Machine-readable result returned to the CLI runner."""

    run_dir: Path
    output_dir: Path
    status: str
    summary_rows: tuple[dict[str, Any], ...]
    files: tuple[Path, ...]
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status == "success" and not self.errors

    @property
    def success(self) -> bool:
        """Alias used by small downstream runners."""

        return self.ok

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "output_dir": str(self.output_dir),
            "status": self.status,
            "summary_rows": [_plain(row) for row in self.summary_rows],
            "files": [str(path) for path in self.files],
            "errors": list(self.errors),
        }


@dataclass(frozen=True, slots=True)
class _Rollout:
    identifier: str
    name: str
    pattern: Mapping[str, Any]
    payload: Mapping[str, Any]
    records: tuple[Mapping[str, Any], ...]
    status: str
    path: Path | None


@dataclass(frozen=True, slots=True)
class _ArtifactPaths:
    directory: Path
    gif: Path
    rewards: Path
    policy_change: Path | None = None
    media_metadata: Path | None = None


_BASELINE_IDS = frozenset({"P00", "P0", "P00_BASELINE", "BASELINE", "NOOP", "NONE", "IDENTITY"})
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_INTEGER_RE = re.compile(r"-?\d+")


def _plain(value: Any) -> Any:
    """Convert ordinary scalar/container values to JSON-safe values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    # This handles JSON-compatible scalar wrappers without importing numpy.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _plain(item())
        except Exception:
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _plain(tolist())
        except Exception:
            pass
    return str(value)


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _fmt_number(value: Any, *, digits: int = 6) -> str:
    number = _number(value)
    if number is None:
        return "N/A"
    if number == 0.0:
        return "0"
    text = f"{number:.{digits}g}"
    return text


def _read_json(path: Path) -> tuple[Any | None, str | None]:
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream), None
    except FileNotFoundError:
        return None, None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"cannot read {path}: {type(exc).__name__}: {exc}"


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            return [dict(row) for row in csv.DictReader(stream)]
    except (OSError, UnicodeError, csv.Error):
        return []


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: Any) -> None:
    _write_text(
        path,
        json.dumps(_plain(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _write_summary(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=list(SUMMARY_COLUMNS),
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(
                {column: row.get(column, "") for column in SUMMARY_COLUMNS}
                for row in rows
            )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_identifier(value: Any, *, fallback: str = "pattern") -> str:
    """Make a filesystem component from an ID without allowing traversal."""

    text = str(value or "").strip()
    if _IDENTIFIER_RE.fullmatch(text) and text not in {".", ".."}:
        return text
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return text or fallback


def _strict_identifier(value: Any) -> str | None:
    """Return an ID only when it is already a safe path component.

    Report input IDs come from the saved manifest and are used to select files
    below ``data/``.  Sanitizing an unsafe value would both permit an attacker
    to influence which file is read and make two IDs silently share one output
    directory, so discovery rejects such IDs instead.
    """

    text = str(value or "").strip()
    if _IDENTIFIER_RE.fullmatch(text) and text not in {".", ".."}:
        return text
    return None


def _safe_relative(path: Path, root: Path) -> str | None:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    if any(part in {"", ".", ".."} for part in relative.parts):
        return None
    return PurePosixPath(*relative.parts).as_posix()


def _is_baseline(identifier: Any) -> bool:
    return str(identifier or "").strip().upper() in _BASELINE_IDS


def _first(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None and str(value).strip() != "":
            return value
    return default


def _parse_indices(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        for key in ("indices", "requested_indices", "target_indices"):
            if key in value:
                return _parse_indices(value[key])
        return []
    if isinstance(value, (list, tuple)):
        result: list[int] = []
        for item in value:
            try:
                number = int(item)
            except (TypeError, ValueError, OverflowError):
                continue
            if number not in result:
                result.append(number)
        return result
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed = None
    if isinstance(parsed, list):
        return _parse_indices(parsed)
    return [int(item) for item in _INTEGER_RE.findall(text)]


def _mapping_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        rows_value = value.get("rows", value.get("records"))
        if isinstance(rows_value, list):
            return [dict(item) for item in rows_value if isinstance(item, Mapping)]
        patterns = value.get("patterns")
        if isinstance(patterns, Mapping):
            result: list[dict[str, Any]] = []
            for identifier, item in patterns.items():
                if isinstance(item, Mapping):
                    result.append({"id": identifier, **dict(item)})
            return result
        if any(key in value for key in ("id", "pattern_id", "target_id")):
            return [dict(value)]
        result = []
        for identifier, item in value.items():
            if isinstance(item, Mapping):
                result.append({"id": identifier, **dict(item)})
        return result
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    return []


def _pattern_id(value: Mapping[str, Any], *, fallback: str = "unknown") -> str:
    raw = _first(value, "id", "pattern_id", "target_id", "identifier", default=fallback)
    if isinstance(raw, Mapping):
        raw = _first(raw, "id", "pattern_id", "target_id", default=fallback)
    return str(raw)


def _pattern_metadata(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    return _mapping_rows(manifest.get("patterns", []))


def _find_json_file(data_dir: Path, candidates: Sequence[str]) -> Path | None:
    for name in candidates:
        path = data_dir / name
        if path.is_file():
            return path
    return None


def _load_rollout(path: Path | None, *, fallback_id: str, fallback_name: str = "") -> tuple[_Rollout, str | None]:
    if path is None:
        payload: Mapping[str, Any] = {
            "id": fallback_id,
            "name": fallback_name or fallback_id,
            "status": "not_run",
            "records": [],
        }
        return _Rollout(fallback_id, fallback_name or fallback_id, {}, payload, (), "not_run", None), None
    raw, error = _read_json(path)
    if error:
        payload = {"id": fallback_id, "name": fallback_name or fallback_id, "status": "failed", "records": []}
        return _Rollout(fallback_id, fallback_name or fallback_id, {}, payload, (), "failed", path), error
    if not isinstance(raw, Mapping):
        payload = {"id": fallback_id, "name": fallback_name or fallback_id, "status": "failed", "records": []}
        return _Rollout(fallback_id, fallback_name or fallback_id, {}, payload, (), "failed", path), f"{path} is not a JSON object"
    identifier = str(_first(raw, "id", "pattern_id", default=fallback_id))
    if _strict_identifier(identifier) is None:
        payload = {"id": fallback_id, "name": fallback_name or fallback_id, "status": "failed", "records": []}
        return (
            _Rollout(fallback_id, fallback_name or fallback_id, {}, payload, (), "failed", path),
            f"unsafe rollout identifier rejected before report rendering: {identifier!r}",
        )
    name = str(_first(raw, "name", "pattern_name", default=fallback_name or identifier))
    pattern = raw.get("pattern")
    pattern_mapping = dict(pattern) if isinstance(pattern, Mapping) else {}
    records_value = raw.get("records", [])
    records: list[Mapping[str, Any]] = []
    records_error: str | None = None
    if isinstance(records_value, list):
        records = [dict(item) for item in records_value if isinstance(item, Mapping)]
        if len(records) != len(records_value):
            records_error = f"{path}: one or more records are not JSON objects"
    elif records_value not in (None, ""):
        records_error = f"{path}: records must be a JSON array"
    status = str(_first(raw, "status", "state", default="unknown"))
    return _Rollout(identifier, name, pattern_mapping, raw, tuple(records), status, path), records_error


def _load_offline(data_dir: Path, identifier: str) -> tuple[Mapping[str, Any] | None, str | None, Path | None]:
    safe_id = _strict_identifier(identifier)
    if safe_id is None:
        return (
            None,
            f"unsafe pattern identifier rejected before offline load: {identifier!r}",
            None,
        )
    path = data_dir / f"{safe_id}_offline.json"
    if not path.is_file():
        path = None
    if path is None:
        return None, None, None
    raw, error = _read_json(path)
    if error:
        return None, error, path
    if not isinstance(raw, Mapping):
        return None, f"{path} is not a JSON object", path
    return raw, None, path


def _merge_pattern_metadata(rollout: _Rollout, metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if metadata:
        merged.update(metadata)
    merged.update({key: value for key, value in rollout.payload.items() if key in {"id", "name", "pattern", "indices", "fixed_value"}})
    pattern = rollout.payload.get("pattern")
    if isinstance(pattern, Mapping):
        merged.update(pattern)
    merged.update(rollout.pattern)
    merged.setdefault("id", rollout.identifier)
    merged.setdefault("name", rollout.name)
    return merged


def _discover(
    root: Path,
) -> tuple[Mapping[str, Any], _Rollout, list[tuple[_Rollout, dict[str, Any], Mapping[str, Any] | None, Path | None]], list[str]]:
    errors: list[str] = []
    data_dir = root / "data"
    manifest_path = data_dir / "manifest.json"
    manifest: Mapping[str, Any] = {}
    raw_manifest, error = _read_json(manifest_path)
    if error:
        errors.append(error)
    elif isinstance(raw_manifest, Mapping):
        manifest = raw_manifest
    else:
        errors.append(f"missing or invalid required manifest: {manifest_path}")

    baseline_path = data_dir / "P00_baseline.json"
    baseline, error = _load_rollout(baseline_path, fallback_id="P00", fallback_name="baseline")
    if error:
        errors.append(error)
    elif baseline.path is None:
        errors.append(f"missing required baseline rollout: {baseline_path}")
    # The report contract always labels the common run P00, even when an
    # adapter chose the human name "baseline" in its JSON object.
    if _is_baseline(baseline.identifier):
        baseline = _Rollout("P00", baseline.name, baseline.pattern, baseline.payload, baseline.records, baseline.status, baseline.path)

    metadata_rows = _pattern_metadata(manifest)
    metadata_by_id = {_pattern_id(row): row for row in metadata_rows}
    ordered_ids: list[str] = []
    safe_ids: dict[str, str] = {}
    for row in metadata_rows:
        identifier = _pattern_id(row)
        safe_identifier = _strict_identifier(identifier)
        if safe_identifier is None:
            errors.append(f"unsafe pattern identifier rejected before data load: {identifier!r}")
            continue
        if _is_baseline(identifier):
            continue
        if safe_identifier in {"baseline", "data"}:
            errors.append(f"reserved pattern identifier rejected: {identifier!r}")
            continue
        if identifier in ordered_ids:
            errors.append(f"duplicate pattern identifier rejected: {identifier!r}")
            continue
        previous = safe_ids.get(safe_identifier)
        if previous is not None:
            errors.append(
                "pattern identifier collision rejected: "
                f"{previous!r} and {identifier!r} map to {safe_identifier!r}"
            )
            continue
        safe_ids[safe_identifier] = identifier
        ordered_ids.append(identifier)
    pattern_rows: list[tuple[_Rollout, dict[str, Any], Mapping[str, Any] | None, Path | None]] = []
    for identifier in ordered_ids:
        metadata = metadata_by_id.get(identifier)
        name = str(_first(metadata or {}, "name", "pattern_name", default=identifier))
        # IDs were validated while building ``ordered_ids``.  Keep this strict
        # check local as a defense against future changes to discovery.
        safe_id = _strict_identifier(identifier)
        if safe_id is None:  # pragma: no cover - guarded above
            errors.append(f"unsafe pattern identifier rejected before data load: {identifier!r}")
            continue
        path = data_dir / f"{safe_id}.json"
        if not path.is_file():
            path = None
        rollout, error = _load_rollout(path, fallback_id=identifier, fallback_name=name)
        if error:
            errors.append(error)
        elif rollout.path is None:
            errors.append(f"missing pattern rollout: {data_dir / (safe_id + '.json')}")
        merged = _merge_pattern_metadata(rollout, metadata)
        offline, offline_error, offline_path = _load_offline(data_dir, identifier)
        if offline_error:
            errors.append(offline_error)
        pattern_rows.append((rollout, merged, offline, offline_path))
    return manifest, baseline, pattern_rows, errors


def _records(rollout: _Rollout) -> list[Mapping[str, Any]]:
    return list(rollout.records)


def _executed(record: Mapping[str, Any]) -> bool:
    value = record.get("executed", False)
    if isinstance(value, str):
        return value.strip().casefold() not in {"false", "0", "no", "failed"}
    return bool(value)


def _step(record: Mapping[str, Any], fallback: int) -> int:
    try:
        return int(record.get("step", fallback))
    except (TypeError, ValueError, OverflowError):
        return fallback


def _executed_records(rollout: _Rollout) -> list[Mapping[str, Any]]:
    return [record for record in rollout.records if _executed(record)]


def _reward_sum(rollout: _Rollout) -> tuple[float | None, int, bool, list[int]]:
    values: list[float] = []
    missing: list[int] = []
    for fallback, record in enumerate(rollout.records):
        if not _executed(record):
            continue
        value = _number(record.get("reward"))
        if value is None:
            missing.append(_step(record, fallback))
        else:
            values.append(value)
    if not values and missing:
        return None, 0, False, missing
    if missing:
        # A partial sum is useful for debugging and is clearly labelled as
        # partial by the row's status.  It is never used for reward difference.
        return math.fsum(values), len(values), False, missing
    return (math.fsum(values) if values else None), len(values), bool(values), missing


def _raw_status(value: Any) -> str:
    return str(value or "unknown").strip().casefold()


def _comparable(rollout: _Rollout, baseline: _Rollout, *, is_baseline: bool) -> bool:
    # A failed/partial rollout can still contain numeric records and even a
    # stale ``comparable: true`` field.  Never let those records produce an
    # A/B reward delta.  The runner's complete status is part of the pairing
    # decision, not merely a display label.
    if _raw_status(rollout.status) not in {"complete", "success"}:
        return False
    if is_baseline:
        return bool(rollout.payload.get("comparable") is True)
    if _raw_status(baseline.status) not in {"complete", "success"}:
        return False
    if rollout.payload.get("comparable") is not True:
        return False
    # Current runners persist the detailed pairing result as well as the
    # top-level boolean.  Require both fields to confirm the validated pairing;
    # an absent detail record leaves the conditions unconfirmed.
    comparison = rollout.payload.get("comparison")
    if not isinstance(comparison, Mapping):
        return False
    return (
        comparison.get("comparable") is True
        and _raw_status(comparison.get("status")) == "comparable"
    )


def _failure_text(payload: Mapping[str, Any]) -> str | None:
    failure = payload.get("failure")
    if isinstance(failure, Mapping):
        stage = _first(failure, "stage", default=None)
        step = _first(failure, "step", default=None)
        error = _first(failure, "error", "message", default=None)
        parts = []
        if stage not in (None, ""):
            parts.append(str(stage))
        if step not in (None, ""):
            parts.append(f"step {step}")
        if error not in (None, ""):
            parts.append(str(error))
        if parts:
            return ": ".join((parts[0], " ".join(parts[1:]))) if len(parts) > 1 else parts[0]
    for key in ("error", "failure_reason", "reason"):
        value = payload.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _termination(rollout: _Rollout) -> tuple[int | None, int, str]:
    executed = _executed_records(rollout)
    count = len(executed)
    if not executed:
        reason = _failure_text(rollout.payload) or (_raw_status(rollout.status) if rollout.status else "unknown")
        return None, 0, reason
    final = executed[-1]
    last_step = _step(final, count - 1)
    reason: str | None = None
    for key in ("termination_reason", "termination", "end_reason"):
        value = final.get(key)
        if value not in (None, ""):
            reason = str(value)
            break
    info = final.get("info")
    if reason is None and isinstance(info, Mapping):
        value = _first(info, "termination_reason", "termination", "end_reason", default=None)
        if value not in (None, ""):
            reason = str(value)
    if reason is None:
        saved_termination = rollout.payload.get("termination")
        if isinstance(saved_termination, Mapping):
            value = _first(saved_termination, "reason", "termination_reason", default=None)
            if value not in (None, ""):
                reason = str(value)
    if reason is None and isinstance(info, Mapping):
        for key in ("arrive_dest", "out_of_road", "crash_vehicle", "crash_object", "crash"):
            if bool(info.get(key, False)):
                reason = key
                break
    if reason is None and bool(final.get("terminated", False)):
        reason = "terminated"
    if reason is None and bool(final.get("truncated", False)):
        reason = "truncated"
    if reason is None:
        reason = _failure_text(rollout.payload)
    status = _raw_status(rollout.status)
    if status in {"failed", "error", "aborted", "interrupted"}:
        reason = f"実行失敗: {reason or status}"
    elif reason is None:
        reason = status if status not in {"complete", "unknown", ""} else "unknown"
    return last_step, count, reason


def _termination_text(rollout: _Rollout) -> str:
    last_step, count, reason = _termination(rollout)
    if last_step is None:
        return f"N/A: {reason}"
    return f"step {last_step}（{count} step）: {reason}"


def _value_at_index(value: Any, index: int, indices: Sequence[int]) -> Any:
    if isinstance(value, Mapping):
        for key in (index, str(index), f"input_{index}"):
            if key in value:
                return value[key]
        # A nested model-input object is common in adapter snapshots.
        for key in ("values", "data", "observation", "input"):
            child = value.get(key)
            if child is not None:
                found = _value_at_index(child, index, indices)
                if found is not None:
                    return found
        return None
    if isinstance(value, (list, tuple)):
        if 0 <= index < len(value):
            return value[index]
        if index in indices:
            position = list(indices).index(index)
            if 0 <= position < len(value):
                return value[position]
    return None


def _finite_leaves(value: Any) -> list[float]:
    number = _number(value)
    if number is not None:
        return [number]
    if isinstance(value, Mapping):
        result: list[float] = []
        for child in value.values():
            result.extend(_finite_leaves(child))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for child in value:
            result.extend(_finite_leaves(child))
        return result
    return []


def _input_ranges(rollout: _Rollout, indices: Sequence[int]) -> dict[int, tuple[float, float]]:
    values: dict[int, list[float]] = {int(index): [] for index in indices}
    for record in rollout.records:
        if not _executed(record):
            continue
        # ``input`` is deliberately the only source here.  Raw observations
        # can have a different semantic boundary and must not be mixed in.
        source = record.get("input")
        for index in indices:
            candidate = _value_at_index(source, int(index), indices)
            values[int(index)].extend(_finite_leaves(candidate))
    return {
        index: (min(items), max(items))
        for index, items in values.items()
        if items
    }


def _fixed_value(pattern: Mapping[str, Any], index: int, indices: Sequence[int]) -> Any:
    for key in ("fixed_value", "replacement", "value", "level"):
        if key not in pattern:
            continue
        value = pattern[key]
        if isinstance(value, Mapping):
            found = _value_at_index(value, index, indices)
            if found is not None:
                return found
        elif isinstance(value, (list, tuple)) and len(value) == len(indices):
            try:
                return value[list(indices).index(index)]
            except (ValueError, IndexError):
                pass
        else:
            return value
    return None


def _replacement_text(rollout: _Rollout, pattern: Mapping[str, Any], *, is_baseline: bool) -> str:
    indices = _parse_indices(_first(pattern, "indices", "requested_indices", "target_indices", default=None))
    if is_baseline or not indices:
        return "変更なし"
    ranges = _input_ranges(rollout, indices)
    # The six-column table is a run-level summary.  A LiDAR replacement may
    # cover dozens of dimensions, so listing one row per index makes the main
    # report unusable.  Aggregate all finite observed values across the target
    # dimensions and steps; the unabridged per-index values remain in
    # ``report_input_details.csv`` via ``_replacement_details``.
    observed = [bound for bound in ranges.values() if bound is not None]
    before_min = min((bound[0] for bound in observed), default=None)
    before_max = max((bound[1] for bound in observed), default=None)
    fixed_values = [_fixed_value(pattern, index, indices) for index in indices]
    fixed_numbers = [number for number in (_number(value) for value in fixed_values) if number is not None]
    if fixed_numbers and len(fixed_numbers) == len(fixed_values) and all(
        math.isclose(number, fixed_numbers[0], rel_tol=0.0, abs_tol=0.0)
        for number in fixed_numbers
    ):
        fixed_text = _fmt_number(fixed_numbers[0])
    elif fixed_numbers and len(fixed_numbers) == len(fixed_values):
        fixed_text = f"[{_fmt_number(min(fixed_numbers))},{_fmt_number(max(fixed_numbers))}]"
    else:
        labels = [
            _fmt_number(value) if _number(value) is not None else (str(value) if value is not None else "N/A")
            for value in fixed_values
        ]
        unique_labels = list(dict.fromkeys(labels))
        fixed_text = unique_labels[0] if len(unique_labels) == 1 else f"{unique_labels[0]}等{len(unique_labels)}種"
    range_text = (
        f"[{_fmt_number(before_min)},{_fmt_number(before_max)}]"
        if before_min is not None and before_max is not None
        else "N/A"
    )
    return (
        f"各step実測値→{fixed_text} / 書換え前範囲{range_text}"
        f"（対象{_index_text(indices)}、{len(indices)}次元）"
    )


def _replacement_details(
    rollout: _Rollout,
    pattern: Mapping[str, Any],
    *,
    is_baseline: bool,
) -> list[dict[str, Any]]:
    """Retain one before-range/fixed-value row for every requested index."""

    if is_baseline:
        return []
    indices = _parse_indices(_first(pattern, "indices", "requested_indices", "target_indices", default=None))
    ranges = _input_ranges(rollout, indices)
    return [
        {
            "index": index,
            "fixed_value": _fixed_value(pattern, index, indices),
            "before_min": ranges[index][0] if index in ranges else None,
            "before_max": ranges[index][1] if index in ranges else None,
        }
        for index in indices
    ]


def _target_label(rollout: _Rollout, pattern: Mapping[str, Any], *, is_baseline: bool) -> str:
    if is_baseline:
        return "P00 baseline"
    name = str(_first(pattern, "name", "pattern_name", "label", default=rollout.name or rollout.identifier))
    identifier = rollout.identifier
    return f"{identifier} {name}" if name and name != identifier else identifier


def _count_value(rollout: _Rollout, key: str) -> int | None:
    """Read a saved run count without treating a missing count as zero."""

    value = rollout.payload.get(key)
    number = _number(value)
    if number is None or number < 0.0 or not number.is_integer():
        return None
    return int(number)


def _pattern_stats(rollout: _Rollout) -> str:
    """Return the compact per-pattern execution/change counts for the card."""

    executed = len(_executed_records(rollout))
    applied = _count_value(rollout, "applied_count")
    changed = _count_value(rollout, "changed_count")
    unchanged = _count_value(rollout, "unchanged_count")
    return (
        f"applied={applied if applied is not None else 'N/A'}"
        f"/executed={executed}, "
        f"changed={changed if changed is not None else 'N/A'}, "
        f"unchanged={unchanged if unchanged is not None else 'N/A'}"
    )


def _index_text(indices: Sequence[int]) -> str:
    """Render sorted indices compactly (``9-18`` instead of 60 labels)."""

    values = sorted(dict.fromkeys(int(index) for index in indices))
    if not values:
        return "-"
    ranges: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _overlay_callback(
    rollout: _Rollout,
    pattern: Mapping[str, Any],
    *,
    is_baseline: bool,
):
    """Build a pure frame annotation callback for one saved rollout."""

    totals: dict[int, float | None] = {}
    cumulative: float | None = 0.0
    for record in rollout.records:
        if not _executed(record):
            continue
        reward = _number(record.get("reward"))
        if cumulative is None or reward is None:
            cumulative = None
        else:
            cumulative += reward
        totals[id(record)] = cumulative
    indices = _parse_indices(_first(pattern, "indices", "requested_indices", "target_indices", default=None))
    if is_baseline:
        target = "P00 baseline"
        fixed = "-"
    else:
        target_name = str(_first(pattern, "name", "pattern_name", default=rollout.name)).strip()
        # Keep the per-frame header legible when a pattern has a descriptive
        # name; the full name and every index are available in the HTML/CSV.
        if len(target_name) > 24:
            target_name = target_name[:21].rstrip() + "..."
        target = f"{rollout.identifier} {target_name}".strip()
        target += f" idx={_index_text(indices)}"
        fixed_values = [_fixed_value(pattern, index, indices) for index in indices]
        fixed = ",".join(
            _fmt_number(value) if _number(value) is not None else (str(value) if value is not None else "N/A")
            for value in fixed_values
        ) or "N/A"

    def callback(record: Mapping[str, Any], step: int) -> list[str]:
        sim_time = _number(record.get("sim_time", record.get("simulation_time")))
        action = _first(record, "action", "executed_action", default="N/A")
        reward = _fmt_number(record.get("reward"))
        total = _fmt_number(totals.get(id(record)))
        reason = _first(record, "termination_reason", "termination", "end_reason", default=None)
        info = record.get("info")
        if reason is None and isinstance(info, Mapping):
            reason = _first(info, "termination_reason", default=None)
        if reason is None and bool(record.get("terminated", False)):
            reason = "terminated"
        if reason is None and bool(record.get("truncated", False)):
            reason = "truncated"
        lines = [
            f"target={target}",
            f"fixed={fixed}",
            f"step={step} time={_fmt_number(sim_time)}s action={action}",
            f"reward={reward} sum={total}",
            "map=topdown RGB",
        ]
        if reason not in (None, ""):
            lines.append(f"termination={reason}")
        return lines

    return callback


def _pattern_chart_title(rollout: _Rollout, pattern: Mapping[str, Any]) -> str:
    indices = _parse_indices(_first(pattern, "indices", "requested_indices", "target_indices", default=None))
    fixed_values = [_fixed_value(pattern, index, indices) for index in indices]
    fixed_text = ",".join(
        _fmt_number(value) if _number(value) is not None else (str(value) if value is not None else "N/A")
        for value in fixed_values
    ) or "N/A"
    return f"{rollout.name} fixed={fixed_text} idx={_index_text(indices)}"


def _summary_row(
    rollout: _Rollout,
    pattern: Mapping[str, Any],
    baseline: _Rollout,
    *,
    is_baseline: bool,
) -> dict[str, Any]:
    baseline_sum, baseline_count, baseline_complete, baseline_missing = _reward_sum(baseline)
    changed_sum, changed_count, changed_complete, changed_missing = _reward_sum(rollout)
    comparable = _comparable(rollout, baseline, is_baseline=is_baseline)
    if is_baseline:
        after = baseline_sum
        difference: float | None = 0.0 if baseline_sum is not None and baseline_complete else None
    else:
        after = changed_sum
        difference = (
            changed_sum - baseline_sum
            if comparable and baseline_sum is not None and changed_sum is not None and baseline_complete and changed_complete
            else None
        )
    status = _raw_status(rollout.status)
    if not is_baseline and status not in {"complete", "success"}:
        assessment = "実行失敗／部分結果"
    elif not is_baseline and not comparable:
        assessment = "比較不能"
    elif (baseline_missing or (not is_baseline and changed_missing)):
        assessment = "報酬欠測"
    else:
        assessment = "比較可能" if not is_baseline else "対照"
    last_step, step_count, reason = _termination(rollout)
    return {
        "pattern_id": rollout.identifier,
        "pattern_name": rollout.name,
        "status": rollout.status,
        "comparable": comparable,
        "assessment": assessment,
        "baseline_cumulative_reward": baseline_sum,
        "changed_cumulative_reward": after,
        "reward_difference": difference,
        "baseline_step_count": len(_executed_records(baseline)),
        "changed_step_count": step_count,
        "baseline_missing_reward_steps": baseline_missing,
        "changed_missing_reward_steps": changed_missing,
        "last_step": last_step,
        "termination_reason": reason,
        "target_indices": _parse_indices(_first(pattern, "indices", "requested_indices", "target_indices", default=None)),
        "replacement_details": _replacement_details(rollout, pattern, is_baseline=is_baseline),
        "target": _target_label(rollout, pattern, is_baseline=is_baseline),
        "replacement": _replacement_text(rollout, pattern, is_baseline=is_baseline),
        SUMMARY_COLUMNS[0]: _target_label(rollout, pattern, is_baseline=is_baseline),
        SUMMARY_COLUMNS[1]: _replacement_text(rollout, pattern, is_baseline=is_baseline),
        SUMMARY_COLUMNS[2]: _fmt_number(baseline_sum) if baseline_sum is not None and baseline_complete else "N/A",
        SUMMARY_COLUMNS[3]: _fmt_number(after) if after is not None and (is_baseline or changed_complete) else _fmt_number(after),
        SUMMARY_COLUMNS[4]: _fmt_number(difference),
        SUMMARY_COLUMNS[5]: _termination_text(rollout),
    }


def _report_conditions(manifest: Mapping[str, Any]) -> dict[str, str]:
    """Read the concrete manifest paths used by the current runner.

    The report should identify the actual model and environment even when the
    manifest deliberately keeps them nested.  These are explicit v1 paths;
    values are left as ``N/A`` when the adapter did not record them rather
    than guessed from an unrelated top-level alias.
    """

    config = manifest.get("config") if isinstance(manifest.get("config"), Mapping) else {}
    adapter = manifest.get("adapter") if isinstance(manifest.get("adapter"), Mapping) else {}
    model = adapter.get("model") if isinstance(adapter.get("model"), Mapping) else {}
    environment = adapter.get("environment") if isinstance(adapter.get("environment"), Mapping) else {}
    environment_config = environment.get("config") if isinstance(environment.get("config"), Mapping) else {}

    model_path = model.get("path")
    model_hash = model.get("sha256")
    model_kind = model.get("kind")
    model_weights_hash = model.get("weights_sha256")
    if model_path not in (None, ""):
        model_text = str(model_path)
        model_parts = []
        if model_hash not in (None, ""):
            model_parts.append(f"sha256={model_hash}")
        model_detail = " (" + ", ".join(model_parts) + ")" if model_parts else ""
    elif model_kind not in (None, ""):
        # Synthetic adapters intentionally have no model archive.  Their
        # stable kind and weight hash identify the policy in copied reports.
        model_text = f"kind={model_kind}"
        model_parts = []
        if model_weights_hash not in (None, ""):
            model_parts.append(f"weights_sha256={model_weights_hash}")
        model_detail = " (" + ", ".join(model_parts) + ")" if model_parts else ""
    else:
        model_text = "N/A"
        model_detail = ""

    map_name = environment_config.get("map")
    scenario_seed = config.get("scenario_seed")
    policy_seed = config.get("policy_seed")
    environment_file = config.get("environment_config")

    manifest_patterns = _pattern_metadata(manifest)
    fixed_values = [
        row.get("fixed_value")
        for row in manifest_patterns
        if isinstance(row, Mapping) and row.get("fixed_value") is not None
    ]
    fixed_text = "N/A"
    if fixed_values:
        rendered = [
            _fmt_number(value) if _number(value) is not None else str(value)
            for value in fixed_values
        ]
        fixed_text = rendered[0] if all(value == rendered[0] for value in rendered) else ", ".join(rendered)

    backend_value = manifest.get("backend")
    backend = str(backend_value).strip() if backend_value not in (None, "") else "N/A"
    if backend.casefold() == "synthetic":
        backend = "synthetic — 合成デモ（実MetaDrive走行ではありません）"

    scenario_parts = []
    if map_name not in (None, ""):
        scenario_parts.append(f"map={map_name}")
    if scenario_seed not in (None, ""):
        scenario_parts.append(f"scenario_seed={scenario_seed}")
    if environment_file not in (None, ""):
        scenario_parts.append(f"environment_config={environment_file}")
    return {
        "model": model_text,
        "model_detail": model_detail,
        "model_hash": str(model_hash) if model_hash not in (None, "") else "N/A",
        "backend": backend,
        "scenario": ", ".join(scenario_parts) if scenario_parts else "N/A",
        "policy_seed": str(policy_seed) if policy_seed not in (None, "") else "N/A",
        "fixed": fixed_text,
        "base_main_sha": str(manifest.get("base_main_sha")) if manifest.get("base_main_sha") not in (None, "") else "N/A",
    }


def _relative_href(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    relative = _safe_relative(path, root)
    return relative


def _image_link(path: Path | None, root: Path, *, alt: str) -> str:
    href = _relative_href(path, root)
    if href is None or not path.is_file():
        return f'<div class="missing">{html.escape(alt)} unavailable</div>'
    escaped = html.escape(href, quote=True)
    return (
        f'<a href="{escaped}" class="image-link"><img src="{escaped}" '
        f'alt="{html.escape(alt, quote=True)}" loading="lazy"></a>'
    )


def _media_grid(
    entries: Sequence[tuple[Path | None, str]],
    root: Path,
) -> str:
    figures: list[str] = []
    for path, caption in entries:
        figures.append(
            '<figure>'
            + _image_link(path, root, alt=caption)
            + f'<figcaption>{html.escape(caption)}</figcaption></figure>'
        )
    return '<div class="media-grid">' + "".join(figures) + "</div>"


def _html_report(
    *,
    manifest: Mapping[str, Any],
    baseline: _Rollout,
    patterns: Sequence[tuple[_Rollout, Mapping[str, Any], Mapping[str, Any] | None, _ArtifactPaths]],
    summary_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    schema_path: Path | None,
    gif_enabled: bool,
    report_status: str,
    errors: Sequence[str],
) -> str:
    conditions = _report_conditions(manifest)
    backend = conditions["backend"]
    model = conditions["model"]
    model_detail = conditions["model_detail"]
    scenario = conditions["scenario"]
    policy_seed = conditions["policy_seed"]
    fixed = conditions["fixed"]
    base_sha = conditions["base_main_sha"]
    schema_href = _relative_href(schema_path, output_dir)
    schema_link = (
        f'<a href="{html.escape(schema_href, quote=True)}">data/input_schema.csv</a>'
        if schema_href is not None and schema_path is not None and schema_path.is_file()
        else "N/A"
    )
    condition_html = (
        '<section id="conditions"><h2>条件</h2><dl>'
        f"<dt>backend</dt><dd>{html.escape(backend)}</dd>"
        f"<dt>model</dt><dd>{html.escape(model + model_detail)}</dd>"
        f"<dt>scenario</dt><dd>{html.escape(scenario)}</dd>"
        f"<dt>policy_seed</dt><dd>{html.escape(policy_seed)}</dd>"
        f"<dt>default fixed value</dt><dd>{html.escape(fixed)}</dd>"
        f"<dt>base main SHA</dt><dd><code>{html.escape(base_sha)}</code></dd>"
        f"<dt>input schema</dt><dd>{schema_link}</dd>"
        '<dt>analysis</dt><dd>①-B closed-loop and ①-A saved-observation comparison</dd>'
        '</dl></section>'
    )
    if errors:
        error_html = '<div class="notice error"><strong>report status: ' + html.escape(report_status) + '</strong><ul>'
        error_html += "".join(f"<li>{html.escape(error)}</li>" for error in errors)
        error_html += "</ul></div>"
    else:
        error_html = f'<div class="notice">report status: {html.escape(report_status)}</div>'

    table_head = "".join(f"<th>{html.escape(column)}</th>" for column in SUMMARY_COLUMNS)
    table_rows = []
    for row in summary_rows:
        cells = "".join(f"<td>{html.escape(str(row.get(column, "")))}</td>" for column in SUMMARY_COLUMNS)
        table_rows.append(f"<tr>{cells}</tr>")
    table_html = (
        '<section id="comparison"><h2>①-B 比較表</h2>'
        '<p class="note">累積報酬は各走行で実際に返った報酬の割引なし合計です。変更後−baselineの差は、対応条件を確認できた走行だけに表示します。対応するstep番号でも走行位置は同じとは限りません。</p>'
        f'<div class="table-wrap"><table><thead><tr>{table_head}</tr></thead><tbody>{"".join(table_rows)}</tbody></table></div></section>'
    )

    baseline_artifact = output_dir / "baseline"
    baseline_gif = baseline_artifact / "rollout.gif"
    baseline_rewards = baseline_artifact / "rewards.png"
    baseline_gif_html = (
        _image_link(baseline_gif, output_dir, alt="baseline rollout.gif")
        if gif_enabled
        else '<div class="missing">GIF recording disabled for this run.</div>'
    )
    baseline_html = (
        '<section id="baseline"><h2>共通baseline</h2>'
        + '<div class="media-grid"><figure>'
        + baseline_gif_html
        + '<figcaption>baseline rollout.gif</figcaption></figure><figure>'
        + _image_link(baseline_rewards, output_dir, alt="baseline rewards.png")
        + '<figcaption>baseline rewards.png</figcaption></figure></div>'
        + "</section>"
    )

    pattern_sections: list[str] = []
    offline_sections: list[str] = []
    for rollout, pattern, offline, artifact in patterns:
        anchor_id = _safe_identifier(rollout.identifier)
        label = _target_label(rollout, pattern, is_baseline=False)
        pattern_sections.append(
            f'<article class="pattern-card" id="pattern-{html.escape(anchor_id, quote=True)}">'
            f'<h3>{html.escape(label)}</h3>'
            f'<p class="pattern-stats"><strong>counts:</strong> {html.escape(_pattern_stats(rollout))}</p>'
            f'<p><a href="#offline-{html.escape(anchor_id, quote=True)}">①-A policy change</a></p>'
            + '<div class="media-grid"><figure>'
            + (
                _image_link(artifact.gif, output_dir, alt="closed-loop rollout.gif")
                if gif_enabled
                else '<div class="missing">GIF recording disabled for this run.</div>'
            )
            + '<figcaption>closed-loop rollout.gif</figcaption></figure><figure>'
            + _image_link(artifact.rewards, output_dir, alt="closed-loop rewards.png")
            + '<figcaption>closed-loop rewards.png</figcaption></figure></div>'
            + "</article>"
        )
        offline_path = artifact.policy_change
        if offline is None:
            body = '<div class="missing">offline data unavailable; no saved-observation comparison was generated.</div>'
        else:
            body = _image_link(offline_path, output_dir, alt="policy change PNG")
        offline_sections.append(
            f'<article class="offline-card" id="offline-{html.escape(anchor_id, quote=True)}">'
            f'<h3>{html.escape(label)}</h3>'
            f'<p><a href="#pattern-{html.escape(anchor_id, quote=True)}">①-B closed-loop card</a></p>'
            + body
            + "</article>"
        )
    patterns_html = '<section id="closed-loop"><h2>①-B 全pattern</h2>' + "".join(pattern_sections) + "</section>"
    offline_html = '<section id="offline"><h2>①-A 保存観測のJS</h2><p class="note">JSは同じ時刻のp/q分布です。マーカーは隣接stepの変化ではなく、同時刻のargmax差です。未比較stepは0に補完していません。</p>' + "".join(offline_sections) + "</section>"
    title = "Input attribution report"
    return f'''<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ color-scheme: light; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #263238; background: #fafafa; }}
body {{ max-width: 1180px; margin: 0 auto; padding: 1.2rem; line-height: 1.45; }}
h1, h2, h3 {{ line-height: 1.2; }}
h1 {{ font-size: 1.55rem; }} h2 {{ border-bottom: 2px solid #d8dee9; padding-bottom: .35rem; margin-top: 2rem; }} h3 {{ margin: .35rem 0; font-size: 1.05rem; }}
dl {{ display: grid; grid-template-columns: minmax(10rem, 15rem) 1fr; gap: .25rem .8rem; }} dt {{ font-weight: 650; }} dd {{ margin: 0; overflow-wrap: anywhere; }}
.notice {{ background: #eef2f7; padding: .65rem .8rem; border-radius: .35rem; }} .notice.error {{ background: #fff1f0; color: #7d2222; }}
.note {{ color: #4c566a; font-size: .92rem; }} .table-wrap {{ overflow-x: auto; }} table {{ border-collapse: collapse; width: 100%; min-width: 760px; background: white; }} th, td {{ border: 1px solid #d8dee9; text-align: left; vertical-align: top; padding: .42rem .5rem; }} th {{ background: #e5e9f0; white-space: nowrap; }}
.media-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; align-items: start; }} figure {{ margin: 0; min-width: 0; background: white; border: 1px solid #d8dee9; border-radius: .35rem; padding: .45rem; }} figure img, .offline-card img {{ width: 100%; height: auto; display: block; }} figcaption {{ font-size: .86rem; padding-top: .3rem; overflow-wrap: anywhere; }}
.pattern-stats {{ margin: .25rem 0 .45rem; color: #4c566a; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: .88rem; overflow-wrap: anywhere; }}
.pattern-card, .offline-card {{ background: #fff; border: 1px solid #d8dee9; border-radius: .4rem; padding: .75rem; margin: 1rem 0; scroll-margin-top: 1rem; }}
.missing {{ min-height: 5rem; display: grid; place-items: center; background: #f0f0f0; color: #6b7280; padding: 1rem; text-align: center; }} code {{ overflow-wrap: anywhere; }} a {{ color: #245b9b; }}
@media (max-width: 760px) {{ body {{ padding: .75rem; }} .media-grid {{ grid-template-columns: 1fr; }} dl {{ grid-template-columns: 1fr; gap: .1rem; }} dt {{ margin-top: .35rem; }} }}
</style>
</head>
<body>
<h1>Input attribution report</h1>
{error_html}
{condition_html}
{table_html}
{baseline_html}
{patterns_html}
{offline_html}
</body>
</html>
'''


def _detail_rows(summary_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "pattern_id": row.get("pattern_id"),
            "target_indices": json.dumps(row.get("target_indices", []), ensure_ascii=False),
            "replacement_details": json.dumps(row.get("replacement_details", []), ensure_ascii=False),
            "status": row.get("status"),
            "assessment": row.get("assessment"),
            "comparable": row.get("comparable"),
            "baseline_cumulative_reward": row.get("baseline_cumulative_reward"),
            "changed_cumulative_reward": row.get("changed_cumulative_reward"),
            "reward_difference": row.get("reward_difference"),
            "baseline_step_count": row.get("baseline_step_count"),
            "changed_step_count": row.get("changed_step_count"),
            "termination_reason": row.get("termination_reason"),
            "last_step": row.get("last_step"),
        }
        for row in summary_rows
    ]


def _write_details(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    details = _detail_rows(rows)
    if not details:
        _write_text(path, "")
        return
    fields = list(details[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(details)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_index_details(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write the unabridged per-index replacement ranges beside the main table."""

    fields = ("pattern_id", "index", "fixed_value", "before_min", "before_max")
    flattened: list[dict[str, Any]] = []
    for row in rows:
        for detail in row.get("replacement_details", []):
            if isinstance(detail, Mapping):
                flattened.append(
                    {field: detail.get(field) for field in fields if field != "pattern_id"}
                    | {"pattern_id": row.get("pattern_id")}
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
            writer.writeheader()
            writer.writerows(flattened)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _source_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _raw_data_hashes(root: Path) -> dict[str, str | None]:
    """Hash every regular file below ``data/`` using relative safe paths."""

    data_dir = root / "data"
    if not data_dir.is_dir():
        return {}
    hashes: dict[str, str | None] = {}
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = _safe_relative(path, root)
        if relative is None:
            # A symlink or another path escaping the copied run is not part of
            # the self-contained raw dataset and must never be hashed as if it
            # were local data.
            continue
        hashes[relative] = _source_hash(path)
    return hashes


def _artifact_paths(output_dir: Path, identifier: str, used: set[str]) -> _ArtifactPaths:
    safe = _strict_identifier(identifier)
    if safe is None:
        raise ReportingError(f"unsafe pattern identifier rejected before artifact write: {identifier!r}")
    if safe in used or safe in {"baseline", "data"}:
        raise ReportingError(f"pattern artifact identifier collision: {identifier!r}")
    used.add(safe)
    directory = output_dir / "patterns" / safe
    return _ArtifactPaths(
        directory=directory,
        gif=directory / "rollout.gif",
        rewards=directory / "rewards.png",
        policy_change=directory / "policy_change.png",
        media_metadata=directory / "media.json",
    )


def _gif_enabled(manifest: Mapping[str, Any], payload: Mapping[str, Any]) -> bool:
    """Read the explicit recording switch, defaulting to the new preset."""

    for mapping in (payload, manifest, manifest.get("config") if isinstance(manifest.get("config"), Mapping) else {}):
        if not isinstance(mapping, Mapping):
            continue
        for key in ("record_gif", "record_frames", "recording"):
            if key in mapping:
                value = mapping[key]
                if isinstance(value, str):
                    return value.strip().casefold() not in {"false", "0", "no", "off", "disabled"}
                return bool(value)
    return True


def _rollout_not_run(rollout: _Rollout) -> bool:
    return _raw_status(rollout.status) in {"not_run", "not-run", "not run", "pending"}


def _disabled_media_metadata(path: Path, *, reason: str) -> None:
    _write_json(path, {"status": "disabled", "reason": reason, "path": None})


def _append_visual_result(
    result: VisualResult,
    *,
    label: str,
    files: list[Path],
    errors: list[str],
) -> None:
    if result.path is not None and result.path.is_file():
        files.append(result.path)
    if result.errors:
        errors.extend(f"{label}: {error}" for error in result.errors)


def _visual_call(
    producer: Any,
    *args: Any,
    label: str,
    files: list[Path],
    errors: list[str],
    **kwargs: Any,
) -> VisualResult | None:
    """Contain a backend exception so raw results still receive a report."""

    try:
        result = producer(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - backend-specific failures
        errors.append(f"{label}: {type(exc).__name__}: {exc}")
        return None
    if not isinstance(result, VisualResult):
        errors.append(f"{label}: visual producer returned an invalid result")
        return None
    _append_visual_result(result, label=label, files=files, errors=errors)
    return result


def generate_report(
    run_dir: str | os.PathLike[str],
    *,
    output_dir: str | os.PathLike[str] | None = None,
) -> ReportResult:
    """Generate report artifacts from saved JSON and frames only.

    The returned ``status`` is ``success`` when all requested visualizations
    were encoded and the saved runs are complete, ``partial`` when raw runs
    contain a failed/not-run pattern but report generation itself succeeded,
    and ``failed`` when a required visualization or input read failed.  The
    CLI can use ``result.errors`` to choose a non-zero exit code without
    losing the partial HTML and data-derived plots.
    """

    root = Path(run_dir).resolve()
    destination = Path(output_dir).resolve() if output_dir is not None else root
    destination.mkdir(parents=True, exist_ok=True)
    manifest, baseline, pattern_entries, errors = _discover(root)

    files: list[Path] = []
    baseline_records = _records(baseline)
    baseline_dir = destination / "baseline"
    baseline_gif = baseline_dir / "rollout.gif"
    baseline_rewards = baseline_dir / "rewards.png"
    baseline_media = baseline_dir / "media.json"
    gif_enabled = _gif_enabled(manifest, baseline.payload)
    if gif_enabled and not _rollout_not_run(baseline):
        _visual_call(
            render_rollout_gif,
            baseline_records,
            root,
            baseline_gif,
            _first(baseline.payload, "action_dt", "action_dt_seconds", default=None),
            label="baseline GIF",
            files=files,
            errors=errors,
            metadata_path=baseline_media,
            overlay_lines=_overlay_callback(baseline, {"id": "P00", "name": "baseline"}, is_baseline=True),
        )
    else:
        _disabled_media_metadata(
            baseline_media,
            reason="record_gif=false" if not gif_enabled else "baseline status is not_run",
        )
    if baseline_media.is_file():
        files.append(baseline_media)
    _visual_call(
        render_rewards_plot,
        baseline_records,
        baseline_rewards,
        label="baseline rewards",
        files=files,
        errors=errors,
        title="baseline rewards",
    )

    summary_rows: list[dict[str, Any]] = [
        _summary_row(baseline, {"id": "P00", "name": baseline.name}, baseline, is_baseline=True)
    ]
    pattern_artifacts: list[tuple[_Rollout, Mapping[str, Any], Mapping[str, Any] | None, _ArtifactPaths]] = []
    used_components: set[str] = set()
    for rollout, pattern, offline, offline_path in pattern_entries:
        artifact = _artifact_paths(destination, rollout.identifier, used_components)
        pattern_artifacts.append((rollout, pattern, offline, artifact))
        summary_rows.append(_summary_row(rollout, pattern, baseline, is_baseline=False))
        records = _records(rollout)
        if gif_enabled and not _rollout_not_run(rollout):
            _visual_call(
                render_rollout_gif,
                records,
                root,
                artifact.gif,
                _first(rollout.payload, "action_dt", "action_dt_seconds", default=_first(baseline.payload, "action_dt", "action_dt_seconds", default=None)),
                label=f"{rollout.identifier} GIF",
                files=files,
                errors=errors,
                metadata_path=artifact.media_metadata,
                overlay_lines=_overlay_callback(rollout, pattern, is_baseline=False),
            )
        elif artifact.media_metadata is not None:
            _disabled_media_metadata(
                artifact.media_metadata,
                reason="record_gif=false" if not gif_enabled else "rollout status is not_run",
            )
        if artifact.media_metadata is not None and artifact.media_metadata.is_file():
            files.append(artifact.media_metadata)
        _visual_call(
            render_rewards_plot,
            baseline_records,
            artifact.rewards,
            label=f"{rollout.identifier} rewards",
            files=files,
            errors=errors,
            changed_records=records,
            title=f"{rollout.name} rewards",
        )

        if offline is not None:
            offline_records = offline.get("records", [])
            if not isinstance(offline_records, list):
                errors.append(f"{rollout.identifier} offline records must be a JSON array")
            else:
                _visual_call(
                    render_policy_change_plot,
                    [record for record in offline_records if isinstance(record, Mapping)],
                    artifact.policy_change,
                    label=f"{rollout.identifier} policy change",
                    files=files,
                    errors=errors,
                    title=_pattern_chart_title(rollout, pattern),
                    ylim=(0.0, LN2),
                    xlim=(
                        0.0,
                        float(max(0, len(_executed_records(baseline)) - 1)),
                    ),
                )

    # Keep the public six-column CSV exact.  Machine-readable extra detail is
    # separate and does not inflate the main table.
    _write_summary(destination / "summary.csv", summary_rows)
    files.append(destination / "summary.csv")
    _write_details(destination / "report_details.csv", summary_rows)
    files.append(destination / "report_details.csv")
    _write_index_details(destination / "report_input_details.csv", summary_rows)
    files.append(destination / "report_input_details.csv")

    schema_path = _find_json_file(root / "data", ("input_schema.csv",))
    if schema_path is None:
        candidate = root / "input_schema.csv"
        schema_path = candidate if candidate.is_file() else None
    # If output_dir is the run root the schema is already available by a
    # relative link.  For a separate output directory no raw file is copied.
    html = _html_report(
        manifest=manifest,
        baseline=baseline,
        patterns=pattern_artifacts,
        summary_rows=summary_rows,
        output_dir=destination,
        schema_path=schema_path,
        gif_enabled=gif_enabled,
        report_status="failed" if errors else "pending",
        errors=errors,
    )
    _write_text(destination / "report.html", html)
    files.append(destination / "report.html")

    raw_statuses = [_raw_status(baseline.status)] + [_raw_status(item[0].status) for item in pattern_entries]
    if errors:
        status = "failed"
    elif any(state not in {"complete", "success"} for state in raw_statuses):
        status = "partial"
    else:
        status = "success"
    if status != "failed":
        # The first HTML contains a provisional status only when errors were
        # already known.  Regenerate it once with the final status to make the
        # page self-consistent; this still reads no raw data a second time.
        html = _html_report(
            manifest=manifest,
            baseline=baseline,
            patterns=pattern_artifacts,
            summary_rows=summary_rows,
            output_dir=destination,
            schema_path=schema_path,
            gif_enabled=gif_enabled,
            report_status=status,
            errors=errors,
        )
        _write_text(destination / "report.html", html)
    report_metadata = {
        "status": status,
        "source_run_dir": str(root),
        "raw_data_sha256": _raw_data_hashes(root),
        "errors": list(errors),
        "generated_files": [str(path.relative_to(destination)) for path in files if _safe_relative(path, destination) is not None],
    }
    _write_json(destination / "report_metadata.json", report_metadata)
    files.append(destination / "report_metadata.json")
    if errors:
        _write_json(destination / "report_errors.json", {"status": status, "errors": list(errors)})
        files.append(destination / "report_errors.json")
    return ReportResult(root, destination, status, tuple(summary_rows), tuple(dict.fromkeys(files)), tuple(dict.fromkeys(errors)))


# Small aliases make report regeneration scripts stable across early package
# drafts while keeping one implementation and one raw-data boundary.
build_report = generate_report
render_report = generate_report
write_report = generate_report
create_report = generate_report


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Render a saved input-attribution run")
    parser.add_argument("run_dir", nargs="?", type=Path)
    parser.add_argument("--run-dir", dest="run_dir_option", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    run_dir = args.run_dir_option or args.run_dir
    if run_dir is None:
        parser.error("run_dir or --run-dir is required")
    result = generate_report(run_dir, output_dir=args.output_dir)
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SUMMARY_COLUMNS",
    "ReportResult",
    "ReportingError",
    "build_report",
    "create_report",
    "generate_report",
    "main",
    "render_report",
    "write_report",
]
