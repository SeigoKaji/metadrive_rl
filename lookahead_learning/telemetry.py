"""Telemetry, common metrics, and conservative mode comparisons.

The forward preview wrappers use this module as a deliberately small logging
boundary.  A transition is a mapping whose fields live below the
``lookahead_learning`` namespace when it is written to JSONL.  The logger never calls
an environment method and never fills an unknown measurement with zero.

The public API is stdlib-only so that the ``lookahead_learning`` directory can be
copied to another checkout:

``JsonlWorkerWriter``
    Append one JSON-safe record per worker.  Non-finite numbers become JSON
    ``null`` and each worker has its own file.

``summarize_rows`` / ``summarize_jsonl``
    Aggregate transition rows, retaining per-episode, learning-seed, and
    scenario-seed summaries.  Time and distance denominators are exposed with
    every metric; a metric with no valid denominator is ``None``.

``compare_summaries`` / ``evaluate_improvement_gate``
    Compare only ``baseline -> lookahead_obs`` and
    ``lookahead_obs -> lookahead_obs_pp_reward``.  Comparisons check
    seeds, observation/action metadata, episode cases, incomplete episodes,
    and matched road-progress/speed bins before claiming an improvement.

The row names below are intentionally permissive because the adapter records
some values from MetaDrive ``info`` and some from the wrapper snapshot.  The
canonical names emitted by this module are stable and documented in
``CANONICAL_FIELDS``.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO


TELEMETRY_SCHEMA_VERSION = "lookahead_learning.telemetry.v1"
SCHEMA_VERSION = TELEMETRY_SCHEMA_VERSION
PACKAGE_NAME = "lookahead_learning"
LEGACY_NAMESPACE = "preview_ab"
CANONICAL_MODES = ("baseline", "lookahead_obs", "lookahead_obs_pp_reward")
MODE_ALIASES = {"obs": "lookahead_obs", "obs_pp": "lookahead_obs_pp_reward"}
DEFAULT_STEERING_DEADBAND = 0.02
DEFAULT_LATERAL_DEADBAND_M = 0.1
DEFAULT_OPPOSITE_HOLD_S = 0.2
DEFAULT_WARNING_RELATIVE = 0.05

# Canonical fields are kept as a tuple rather than a dataclass so callers can
# add fields without a dependency or a migration.  A writer preserves all
# input fields and only adds the fields that are missing.
CANONICAL_FIELDS = (
    "run",
    "worker",
    "episode",
    "decision",
    "learning_seed",
    "scenario_seed",
    "evaluation_seed",
    "deterministic",
    "schema_version",
    "observation_schema",
    "action_space",
    "wrapper_order",
    "control_dt",
    "decision_dt",
    "vehicle_config",
    "scenario_config",
    "normalization_config",
    "learning_budget",
    "t",
    "t_next",
    "dt",
    "position",
    "heading",
    "lateral_error_m",
    "heading_error_rad",
    "speed_mps",
    "progress_m",
    "progress_delta_m",
    "action_env",
    "action_applied",
    "u_applied",
    "u_previous",
    "du",
    "steering_rate",
    "policy_action_raw",
    "policy_action_stage",
    "yaw_rate",
    "history_valid",
    "q",
    "x_g",
    "y_g",
    "preview_x_norm",
    "preview_y_norm",
    "preview_valid",
    "preview_invalid_reason",
    "S_proj",
    "S_goal",
    "projection_lane",
    "target_lane",
    "reference_lane_ids",
    "navigation_reference_ids",
    "x_rear",
    "y_rear",
    "kappa_pp",
    "delta_pp_rad",
    "u_pp_unclipped",
    "u_pp",
    "pp_valid",
    "pp_saturated",
    "e_pp",
    "r_base",
    "r_pp",
    "r_total",
    "success",
    "arrive_dest",
    "start_lane_maintained",
    "in_target_lane",
    "lane_departure",
    "start_lane_departure",
    "terminated",
    "truncated",
    "episode_end",
)


_MISSING = object()


def _canonical_mode(value: Any) -> Any:
    """Map legacy mode aliases while preserving unknown metadata values."""

    if isinstance(value, str):
        return MODE_ALIASES.get(value, value)
    return value


def _canonical_wrapper_order(value: Any) -> Any:
    """Map the historical wrapper namespace in metadata comparisons."""

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            PACKAGE_NAME if str(item) == LEGACY_NAMESPACE else item
            for item in value
        ]
    return value


def sanitize_json(value: Any) -> Any:
    """Return *value* recursively safe for ``json.dumps(..., allow_nan=False)``.

    JSON has no representation for NaN or infinity.  Measurements that are
    not finite are therefore represented by ``None``.  This is intentionally
    different from replacing a missing metric with zero: the aggregator keeps
    the corresponding denominator and missing count visible.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return sanitize_json(tolist())
        except Exception:
            pass
    # Decimal, NumPy scalars, and similar scalar objects generally expose
    # ``item``.  Avoid importing NumPy solely for telemetry portability.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return sanitize_json(item())
        except Exception:
            pass
    if isinstance(value, Mapping):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [sanitize_json(item) for item in sorted(value, key=repr)]
    if isinstance(value, Path):
        return str(value)
    # ``float`` catches Decimal and most scalar numeric types while preserving
    # a useful string for opaque objects when conversion is not possible.
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return repr(value)
    if math.isfinite(numeric):
        return numeric
    return None


def _json_dump(value: Any) -> str:
    return json.dumps(
        sanitize_json(value),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _safe_component(value: Any) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return text or "unknown"


def _copy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): item for key, item in value.items()}


class JsonlWorkerWriter:
    """Write one JSONL file for one run/worker pair.

    Parameters are intentionally path-like and do not refer to project
    globals.  The output directory is created on demand.  A caller can pass a
    complete envelope containing ``lookahead_learning`` or a flat transition mapping;
    flat mappings are wrapped under that namespace.  Existing input fields
    are preserved and never overwritten.
    """

    def __init__(
        self,
        output_dir: str | os.PathLike[str],
        run: str | int | None = None,
        worker: str | int | None = None,
        *,
        run_id: str | int | None = None,
        worker_id: str | int | None = None,
        filename: str | os.PathLike[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        flush_every: int = 1,
        resume: bool = False,
    ) -> None:
        if run is None:
            run = run_id
        if worker is None:
            worker = worker_id
        if run is None or worker is None:
            raise TypeError("run and worker (or run_id and worker_id) are required")
        if int(flush_every) <= 0:
            raise ValueError("flush_every must be positive")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_root = self.output_dir.resolve()
        self.run = run
        self.worker = worker
        self.metadata = dict(metadata or {})
        self.flush_every = int(flush_every)
        self._writes = 0
        if filename is None:
            filename = (
                f"run-{_safe_component(run)}"
                f"-worker-{_safe_component(worker)}.jsonl"
            )
        file_path = Path(filename)
        if not file_path.is_absolute():
            file_path = self.output_dir / file_path
        resolved_path = file_path.resolve()
        try:
            resolved_path.relative_to(output_root)
        except ValueError as exc:
            raise ValueError(
                "JSONL filename must remain inside output_dir: "
                f"{file_path} is outside {self.output_dir}"
            ) from exc
        self.path = resolved_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # A fresh run must never append to a prior run by accident.  Resume is
        # an explicit opt-in for callers that intentionally continue one file.
        open_mode = "a" if resume else "x"
        self._file: TextIO = self.path.open(open_mode, encoding="utf-8")
        self.closed = False

    def _envelope(self, row: Mapping[str, Any]) -> dict[str, Any]:
        source = _copy_mapping(row)
        namespace = source.get(PACKAGE_NAME)
        if not isinstance(namespace, Mapping):
            # A caller may replay a historical row through a new writer.  It
            # is copied into the canonical namespace; historical files are
            # never opened or rewritten by this class.
            namespace = source.get(LEGACY_NAMESPACE)
        if isinstance(namespace, Mapping):
            payload = _copy_mapping(namespace)
            envelope = dict(source)
        else:
            payload = dict(source)
            envelope = {}
        # Metadata is a default only.  A row-specific value takes precedence.
        for key, value in self.metadata.items():
            if key == "mode":
                value = _canonical_mode(value)
            payload.setdefault(str(key), value)
        if "mode" in payload:
            payload["mode"] = _canonical_mode(payload["mode"])
        payload.setdefault("schema_version", TELEMETRY_SCHEMA_VERSION)
        payload.setdefault("run", self.run)
        payload.setdefault("worker", self.worker)
        envelope.pop(LEGACY_NAMESPACE, None)
        envelope[PACKAGE_NAME] = payload
        # These envelope aliases make simple line-oriented tools convenient;
        # they are copies of namespace values, never replacements of a caller's
        # unrelated top-level keys.
        for key in ("schema_version", "run", "worker", "episode", "decision"):
            if key not in envelope and key in payload:
                envelope[key] = payload[key]
        return envelope

    def write(self, row: Mapping[str, Any]) -> Path:
        """Write one row and return the worker file path."""

        if self.closed:
            raise ValueError("cannot write to a closed JsonlWorkerWriter")
        if not isinstance(row, Mapping):
            raise TypeError("telemetry row must be a mapping")
        self._file.write(_json_dump(self._envelope(row)) + "\n")
        self._writes += 1
        if self._writes % self.flush_every == 0:
            self._file.flush()
        return self.path

    write_row = write

    def flush(self) -> None:
        if not self.closed:
            self._file.flush()

    def close(self) -> None:
        if not self.closed:
            self._file.flush()
            self._file.close()
            self.closed = True

    def __enter__(self) -> "JsonlWorkerWriter":
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()


# A short alias is useful to callers that already call their output object a
# telemetry writer.
TelemetryWriter = JsonlWorkerWriter
WorkerJsonlWriter = JsonlWorkerWriter


def iter_jsonl(path: str | os.PathLike[str]) -> Iterator[dict[str, Any]]:
    """Yield decoded JSON object rows from *path*.

    Malformed lines raise ``ValueError`` with the path and line number so a
    partial log cannot silently become an apparently successful experiment.
    """

    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSONL at {file_path}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(value, Mapping):
                raise ValueError(
                    f"JSONL row at {file_path}:{line_number} is not an object"
                )
            yield dict(value)


def read_jsonl(paths: str | os.PathLike[str] | Iterable[str | os.PathLike[str]]) -> list[dict[str, Any]]:
    """Read one or more JSONL files, preserving file and row order."""

    if isinstance(paths, (str, os.PathLike)):
        path_list: list[str | os.PathLike[str]] = [paths]
    else:
        path_list = list(paths)
    rows: list[dict[str, Any]] = []
    for path in path_list:
        rows.extend(iter_jsonl(path))
    return rows


def _namespace(row: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = row.get(PACKAGE_NAME)
    if isinstance(nested, Mapping):
        return nested
    legacy = row.get(LEGACY_NAMESPACE)
    return legacy if isinstance(legacy, Mapping) else row


def _lookup(row: Mapping[str, Any], *names: str, default: Any = _MISSING) -> Any:
    """Look up exact or dotted names in the preview namespace and envelope."""

    namespaces: list[Mapping[str, Any]] = []
    nested = row.get(PACKAGE_NAME)
    if isinstance(nested, Mapping):
        namespaces.append(nested)
    legacy = row.get(LEGACY_NAMESPACE)
    if isinstance(legacy, Mapping) and legacy is not nested:
        namespaces.append(legacy)
    namespaces.append(row)
    for name in names:
        for source in namespaces:
            if name in source:
                return source[name]
            current: Any = source
            found = True
            for part in name.split("."):
                if not isinstance(current, Mapping) or part not in current:
                    found = False
                    break
                current = current[part]
            if found:
                return current
    if default is not _MISSING:
        return default
    return None


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _bool(value: Any) -> bool | None:
    item = getattr(value, "item", None)
    if callable(item):
        try:
            value = item()
        except Exception:
            pass
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        if float(value) in (0.0, 1.0):
            return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "on", "1"}:
            return True
        if lowered in {"false", "no", "off", "0"}:
            return False
    return None


def _first_number(row: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        value = _number(_lookup(row, name))
        if value is not None:
            return value
    return None


def _first_bool(row: Mapping[str, Any], *names: str) -> bool | None:
    for name in names:
        value = _bool(_lookup(row, name))
        if value is not None:
            return value
    return None


def _explicit_bool(
    row: Mapping[str, Any], *names: str
) -> tuple[bool | None, bool]:
    """Read the first explicitly present boolean and whether it was present.

    A present but malformed value is different from an absent field: callers
    must retain the unknown state instead of silently falling back to an
    unrelated validity flag.
    """

    absent = object()
    for name in names:
        value = _lookup(row, name, default=absent)
        if value is absent:
            continue
        return _bool(value), True
    return None, False


def _row_id(row: Mapping[str, Any], index: int) -> tuple[Any, Any, Any, Any, Any]:
    run = _lookup(row, "run", "run_id")
    worker = _lookup(row, "worker", "worker_id")
    learning = _lookup(row, "learning_seed", "train_seed", "seed")
    scenario = _lookup(row, "scenario_seed", "scenario", "scenario_id")
    episode = _lookup(row, "episode", "episode_id")
    # Missing identifiers remain explicit.  The row index is not used as an
    # episode identifier: all unidentified rows belong to one unknown case.
    # Run names are included for storage grouping.  Comparison case keys below
    # intentionally omit run so two independently named runs can be matched.
    return (
        _stable_id(learning),
        _stable_id(scenario),
        _stable_id(episode),
        _stable_id(run),
        _stable_id(worker),
    )


def _stable_id(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, bool)):
        return value
    number = _number(value)
    if number is not None:
        return int(number) if number.is_integer() else number
    return str(value)


def _key_text(key: Sequence[Any]) -> str:
    return "|".join("null" if item is None else str(item) for item in key)


def _case_key(key: Sequence[Any]) -> tuple[Any, ...]:
    """Return a cross-run case identity (worker retained, run omitted)."""

    if len(key) >= 5:
        return (key[0], key[1], key[2], key[4])
    return tuple(key)


def _decision(row: Mapping[str, Any], index: int) -> tuple[float, int]:
    value = _first_number(row, "decision", "step", "timestep")
    return (value if value is not None else float(index), index)


def _group_rows(rows: Sequence[Mapping[str, Any]]) -> OrderedDict[tuple[Any, Any, Any], list[Mapping[str, Any]]]:
    groups: OrderedDict[tuple[Any, Any, Any], list[Mapping[str, Any]]] = OrderedDict()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError("telemetry rows must contain mappings")
        groups.setdefault(_row_id(row, index), []).append(row)
    for group in groups.values():
        group.sort(key=lambda row: _decision(row, rows.index(row) if row in rows else 0))
    return groups


def _group_rows_indexed(rows: Sequence[Mapping[str, Any]]) -> OrderedDict[tuple[Any, Any, Any], list[tuple[int, Mapping[str, Any]]]]:
    groups: OrderedDict[tuple[Any, Any, Any], list[tuple[int, Mapping[str, Any]]]] = OrderedDict()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError("telemetry rows must contain mappings")
        groups.setdefault(_row_id(row, index), []).append((index, row))
    for group in groups.values():
        group.sort(key=lambda pair: _decision(pair[1], pair[0]))
    return groups


def _dt(row: Mapping[str, Any]) -> float | None:
    direct = _first_number(
        row,
        "dt",
        "dt_s",
        "simulation_dt_s",
        "dt_seconds",
        "simulation_dt_seconds",
        "decision_dt_s",
        "decision_dt_seconds",
    )
    if direct is not None and direct > 0.0:
        return direct
    t = _first_number(row, "t", "time_s")
    t_next = _first_number(row, "t_next", "time_next_s", "t_after")
    if t is not None and t_next is not None and t_next > t:
        return t_next - t
    return None


def _time_value(row: Mapping[str, Any], *, after: bool = False) -> float | None:
    names = ("t_next", "time_next_s", "t_after") if after else ("t", "time_s")
    return _first_number(row, *names)


def _value(row: Mapping[str, Any], *names: str) -> Any:
    return _lookup(row, *names)


def _lateral(row: Mapping[str, Any]) -> float | None:
    return _first_number(
        row,
        "lateral_error_m",
        "e_y",
        "start_lane_offset_m",
        "target_lane_offset_m",
        "lane_offset_m",
    )


def _lateral_valid(row: Mapping[str, Any]) -> bool | None:
    """Return validity for the host lateral measurement independently of preview.

    ``target_lane_valid`` describes the host's start/target-lane reference,
    while ``preview_valid`` describes the separate six-metre lookahead point.
    An invalid lookahead therefore does not invalidate a host lateral value
    when the host explicitly marks that value valid.  If no host validity is
    available, a valid preview is retained as the legacy evidence for rows
    whose lateral value comes from the same snapshot; every other case stays
    unknown and is treated as missing by the metric aggregator.
    """

    explicit, present = _explicit_bool(row, "lateral_valid", "target_lane_valid")
    if present:
        return explicit
    preview_valid = _first_bool(row, "preview_valid")
    if preview_valid is True:
        return True
    return None


def _steering_valid(row: Mapping[str, Any]) -> bool | None:
    """Return validity for an applied-steering measurement."""

    explicit, present = _explicit_bool(
        row,
        "steering_valid",
        "u_applied_valid",
        "action_applied_valid",
    )
    if present:
        return explicit
    return _steering(row) is not None


def _speed(row: Mapping[str, Any]) -> float | None:
    return _first_number(row, "speed_mps", "speed", "velocity_mps")


def _steering(row: Mapping[str, Any]) -> float | None:
    return _first_number(
        row,
        "u_applied",
        "steering_applied",
        "applied_steering",
        "action_applied.steering",
        "action_applied.u",
    )


def _progress(row: Mapping[str, Any]) -> float | None:
    return _first_number(
        row,
        "progress_m",
        "S_proj",
        "s_proj",
        "route_progress_m",
        "navigation_progress_m",
    )


def _progress_delta(row: Mapping[str, Any], previous_progress: float | None) -> float | None:
    explicit = _first_number(
        row,
        "progress_delta_m",
        "delta_progress_m",
        "progress_increment_m",
    )
    if explicit is not None:
        return explicit
    before = _first_number(row, "progress_before_m", "S_proj_before")
    after = _first_number(row, "progress_after_m", "S_proj_after")
    if before is not None and after is not None:
        return after - before
    current = _progress(row)
    if current is not None and previous_progress is not None:
        return current - previous_progress
    return None


def _position_xy(value: Any) -> tuple[float, float] | None:
    """Read a world-position pair for the labeled displacement fallback."""

    if isinstance(value, Mapping):
        x = _number(value.get("x"))
        y = _number(value.get("y"))
        if x is not None and y is not None:
            return x, y
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) < 2:
            return None
        x = _number(value[0])
        y = _number(value[1])
        if x is not None and y is not None:
            return x, y
    return None


def _distance_measurement(
    row: Mapping[str, Any],
) -> tuple[float | None, str]:
    """Return physical distance and its source/status.

    ``progress_m``/``S_proj`` is deliberately not a distance fallback: route
    progress can increase while the vehicle laterally travels a longer path.
    Position displacement is labeled as an approximation because it is the
    chord between decision endpoints rather than an integrated wheel path.
    """

    explicit_names = (
        "distance_delta_m",
        "physical_distance_delta_m",
        "travel_distance_delta_m",
        "distance_increment_m",
    )
    for name in explicit_names:
        raw = _lookup(row, name, default=_MISSING)
        if raw is _MISSING or raw is None:
            continue
        number = _number(raw)
        if number is None or number < 0.0:
            return None, "invalid"
        return number, "explicit"

    before = _lookup(
        row,
        "position_before",
        "position_t",
        "p_before",
        "position_start",
        default=None,
    )
    after = _lookup(
        row,
        "position_after",
        "position_t_next",
        "p_after",
        "position_end",
        default=None,
    )
    before_xy = _position_xy(before)
    after_xy = _position_xy(after)
    if before_xy is not None and after_xy is not None:
        return math.hypot(after_xy[0] - before_xy[0], after_xy[1] - before_xy[1]), "position_displacement"
    has_explicit = any(
        (
            _lookup(row, name, default=_MISSING) is not _MISSING
            and _lookup(row, name, default=None) is not None
        )
        for name in explicit_names
    )
    if has_explicit:
        return None, "invalid"
    return None, "missing"


def _distance_delta(row: Mapping[str, Any], progress_delta: float | None = None) -> float | None:
    """Return physical distance only; route progress is never substituted."""

    return _distance_measurement(row)[0]


def _interval_label(value: float | None, intervals: Any) -> str | None:
    if value is None:
        return None
    if intervals is None:
        return None
    if isinstance(intervals, Mapping):
        items = intervals.items()
    else:
        items = ((None, item) for item in intervals)
    try:
        item_count = len(intervals)
    except TypeError:
        item_count = None
    for index, (name, bounds) in enumerate(items):
        try:
            start, end = bounds
            start_number = float(start)
            end_number = float(end)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(start_number) and math.isfinite(end_number)):
            continue
        if end_number < start_number:
            start_number, end_number = end_number, start_number
        is_last = item_count is not None and index == item_count - 1
        if start_number <= value < end_number or (is_last and value == end_number):
            return str(name if name is not None else f"[{start_number},{end_number})")
    return None


def _speed_label(value: float | None, bins: Any) -> str | None:
    return _interval_label(value, bins)


def _configured_labels(config: Any) -> list[str]:
    if isinstance(config, Mapping):
        return [str(key) for key in config]
    labels: list[str] = []
    if config is None:
        return labels
    for index, bounds in enumerate(config):
        try:
            start, end = bounds
            start_number = float(start)
            end_number = float(end)
            labels.append(f"[{start_number},{end_number})")
        except (TypeError, ValueError):
            labels.append(str(index))
    return labels


def _weighted_mean(values: Sequence[tuple[float, float]]) -> float | None:
    denominator = sum(weight for _, weight in values if weight > 0.0 and math.isfinite(weight))
    if denominator <= 0.0:
        return None
    numerator = sum(value * weight for value, weight in values if weight > 0.0)
    return numerator / denominator


def _weighted_quantile(values: Sequence[tuple[float, float]], quantile: float) -> float | None:
    usable = sorted(
        (value, weight)
        for value, weight in values
        if math.isfinite(value) and math.isfinite(weight) and weight > 0.0
    )
    total = sum(weight for _, weight in usable)
    if total <= 0.0:
        return None
    target = min(max(float(quantile), 0.0), 1.0) * total
    cumulative = 0.0
    for value, weight in usable:
        cumulative += weight
        if cumulative >= target:
            return value
    return usable[-1][0]


def _fraction(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0 or not (math.isfinite(numerator) and math.isfinite(denominator)):
        return None
    return numerator / denominator


def _relative_change(before: float | None, after: float | None) -> float | None:
    if before is None or after is None or not (math.isfinite(before) and math.isfinite(after)):
        return None
    scale = abs(before)
    if scale <= 1e-15:
        if abs(after) <= 1e-15:
            return 0.0
        return math.inf if after > before else -math.inf
    return (after - before) / scale


def _metric_values(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate one set of rows.  This function does not group episodes."""

    total_rows = len(rows)
    total_time = 0.0
    missing_dt = 0
    lateral_values: list[tuple[float, float]] = []
    speed_values: list[tuple[float, float]] = []
    steering_du_values: list[tuple[float, float]] = []
    steering_rate_values: list[tuple[float, float]] = []
    yaw_rate_values: list[tuple[float, float]] = []
    progress_total = 0.0
    progress_time = 0.0
    distance_total = 0.0
    distance_missing = 0
    distance_invalid = 0
    distance_approximation = 0
    distance_zero = 0
    tv_total = 0.0
    tv_time = 0.0
    lateral_sq_time = 0.0
    lateral_abs_time = 0.0
    speed_sum = 0.0
    speed_time = 0.0
    stop_time = 0.0
    steering_saturated_time = 0.0
    pp_saturated_time = 0.0
    steering_saturated_known_time = 0.0
    pp_saturated_known_time = 0.0
    preview_valid_time = 0.0
    preview_invalid_time = 0.0
    preview_known_time = 0.0
    pp_valid_time = 0.0
    pp_invalid_time = 0.0
    pp_known_time = 0.0
    lateral_missing = 0
    speed_missing = 0
    progress_missing = 0
    steering_missing = 0
    invalid_reasons: dict[str, float] = {}
    reward_sums = {"r_base": 0.0, "r_pp": 0.0, "r_total": 0.0}
    reward_seen = {key: False for key in reward_sums}
    progress_previous: float | None = None
    steering_previous: float | None = None
    last_time: float | None = None
    elapsed_from_clock = 0.0

    for row in rows:
        duration = _dt(row)
        if duration is None:
            missing_dt += 1
        else:
            total_time += duration
        t = _time_value(row)
        t_next = _time_value(row, after=True)
        if t is not None and t_next is not None and t_next > t:
            elapsed_from_clock += t_next - t
        elif t is not None and last_time is not None and t > last_time:
            elapsed_from_clock += t - last_time
        if t is not None:
            last_time = t
        weight = duration if duration is not None else 0.0

        valid_preview = _first_bool(row, "preview_valid")
        if valid_preview is not None and weight > 0.0:
            preview_known_time += weight
            if valid_preview:
                preview_valid_time += weight
            else:
                preview_invalid_time += weight
        reason = _lookup(
            row,
            "preview_invalid_reason",
            "invalid_reason",
            default=None,
        )
        if valid_preview is False:
            reason_text = str(reason) if reason not in (None, "") else "unknown"
            invalid_reasons[reason_text] = invalid_reasons.get(reason_text, 0.0) + weight
        valid_pp = _first_bool(row, "pp_valid")
        if valid_pp is not None and weight > 0.0:
            pp_known_time += weight
            if valid_pp:
                pp_valid_time += weight
            else:
                pp_invalid_time += weight

        lateral = _lateral(row)
        # Host target-lane validity and the six-metre preview validity are
        # separate measurements.  A route end can invalidate the lookahead
        # while the host still supplies a valid lateral offset, so never use
        # ``preview_valid`` as a proxy for lateral validity here.
        if _lateral_valid(row) is not True:
            lateral = None
        if lateral is None:
            lateral_missing += 1
        elif weight > 0.0:
            lateral_values.append((abs(lateral), weight))
            lateral_sq_time += lateral * lateral * weight
            lateral_abs_time += abs(lateral) * weight

        speed = _speed(row)
        if speed is None:
            speed_missing += 1
        elif weight > 0.0:
            speed_values.append((speed, weight))
            speed_sum += speed * weight
            speed_time += weight
            if speed < 0.5:
                stop_time += weight

        progress_increment = _progress_delta(row, progress_previous)
        current_progress = _progress(row)
        if current_progress is not None:
            progress_previous = current_progress
        if progress_increment is None:
            progress_missing += 1
        else:
            progress_total += progress_increment
            if weight > 0.0:
                progress_time += weight
        distance, distance_status = _distance_measurement(row)
        if distance_status == "invalid":
            distance_invalid += 1
        elif distance is None:
            distance_missing += 1
        elif distance_status == "position_displacement":
            distance_approximation += 1
            if distance == 0.0:
                distance_zero += 1
        elif distance == 0.0:
            distance_zero += 1
        if weight > 0.0:
            if distance is not None:
                distance_total += distance

        steering = _steering(row)
        if steering is None:
            steering_missing += 1
        explicit_du = _first_number(row, "du", "steering_delta", "delta_u")
        previous_from_row = _first_number(
            row,
            "u_previous",
            "previous_steering",
            "steering_previous",
        )
        history_valid = _first_bool(row, "history_valid")
        du = explicit_du
        if du is None and steering is not None:
            if previous_from_row is not None:
                du = steering - previous_from_row
            elif steering_previous is not None:
                du = steering - steering_previous
        if history_valid is False:
            du = None
        if du is not None and weight > 0.0:
            steering_du_values.append((du, weight))
            tv_total += abs(du)
            tv_time += weight
            steering_rate_values.append((du / weight, weight))
        if steering is not None:
            steering_previous = steering

        yaw_rate = _first_number(row, "yaw_rate", "yaw_rate_after")
        if yaw_rate is not None and weight > 0.0:
            yaw_rate_values.append((yaw_rate * yaw_rate, weight))

        for reward_name in reward_sums:
            reward = _first_number(row, reward_name)
            if reward is not None:
                reward_sums[reward_name] += reward
                reward_seen[reward_name] = True

        steering_saturated = _first_bool(
            row,
            "steering_saturated",
            "u_applied_saturated",
            "action_saturated",
        )
        pp_saturated = _first_bool(
            row,
            "pp_saturated",
            "u_pp_saturated",
            "u_pp_clipped",
        )
        if weight > 0.0 and steering_saturated is not None:
            steering_saturated_known_time += weight
            steering_saturated_time += weight if steering_saturated else 0.0
        if weight > 0.0 and pp_saturated is not None:
            pp_saturated_known_time += weight
            pp_saturated_time += weight if pp_saturated else 0.0

    denominator_time = total_time
    metrics: dict[str, Any] = {
        "duration_s": total_time if total_time > 0.0 else (elapsed_from_clock or None),
        "dt_missing_count": missing_dt,
        "decision_count": total_rows,
        "lateral_rms_m": (
            math.sqrt(lateral_sq_time / sum(weight for _, weight in lateral_values))
            if lateral_values and lateral_sq_time >= 0.0
            else None
        ),
        "lateral_abs_p95_m": _weighted_quantile(lateral_values, 0.95),
        "lateral_time_s": sum(weight for _, weight in lateral_values),
        "lateral_missing_count": lateral_missing,
        "lateral_missing_rate": _fraction(float(lateral_missing), float(total_rows))
        if total_rows
        else None,
        "steering_tv": tv_total if tv_time > 0.0 else None,
        "steering_tv_per_s": _fraction(tv_total, tv_time),
        "steering_rate_rms": (
            math.sqrt(
                sum(rate * rate * weight for rate, weight in steering_rate_values)
                / tv_time
            )
            if tv_time > 0.0 and steering_rate_values
            else None
        ),
        "steering_tv_per_m": _fraction(tv_total, distance_total),
        "steering_change_time_s": tv_time,
        "steering_missing_count": steering_missing,
        "steering_missing_rate": _fraction(float(steering_missing), float(total_rows))
        if total_rows
        else None,
        "distance_m": distance_total if distance_total > 0.0 else None,
        "physical_distance_missing_count": distance_missing,
        "physical_distance_invalid_count": distance_invalid,
        "physical_distance_approximation_count": distance_approximation,
        "physical_distance_known_count": total_rows - distance_missing - distance_invalid,
        "physical_distance_zero_count": distance_zero,
        "physical_distance_missing_rate": _fraction(
            float(distance_missing), float(total_rows)
        )
        if total_rows
        else None,
        "physical_distance_invalid_rate": _fraction(
            float(distance_invalid), float(total_rows)
        )
        if total_rows
        else None,
        "physical_distance_missing_condition": (
            total_rows == 0 or distance_missing > 0 or distance_invalid > 0
        ),
        "progress_m": progress_total if progress_time > 0.0 else None,
        "progress_per_s": _fraction(progress_total, progress_time),
        "progress_time_s": progress_time,
        "progress_missing_count": progress_missing,
        "speed_mean_mps": _fraction(speed_sum, speed_time),
        "speed_p05_mps": _weighted_quantile(
            speed_values, 0.05
        ),
        "speed_p50_mps": _weighted_quantile(speed_values, 0.50),
        "speed_p95_mps": _weighted_quantile(speed_values, 0.95),
        "speed_time_s": speed_time,
        "speed_missing_count": speed_missing,
        "stop_time_fraction": _fraction(stop_time, speed_time),
        "preview_valid_rate": _fraction(preview_valid_time, preview_known_time),
        "preview_valid_time_s": preview_valid_time,
        "preview_known_time_s": preview_known_time,
        "preview_missing_count": total_rows
        - sum(1 for row in rows if _first_bool(row, "preview_valid") is not None),
        "pp_valid_rate": _fraction(pp_valid_time, pp_known_time),
        "pp_valid_time_s": pp_valid_time,
        "pp_known_time_s": pp_known_time,
        "pp_missing_count": total_rows
        - sum(1 for row in rows if _first_bool(row, "pp_valid") is not None),
        "steering_saturation_rate": _fraction(
            steering_saturated_time, steering_saturated_known_time
        ),
        "pp_saturation_rate": _fraction(pp_saturated_time, pp_saturated_known_time),
        "invalid_reason_time_s": invalid_reasons,
        "invalid_reason_rates": {
            reason: _fraction(value, preview_known_time)
            for reason, value in invalid_reasons.items()
        },
        "yaw_rate_rms": (
            math.sqrt(
                sum(square * weight for square, weight in yaw_rate_values)
                / sum(weight for _, weight in yaw_rate_values)
            )
            if yaw_rate_values
            else None
        ),
    }
    for reward_name, value in reward_sums.items():
        metrics[reward_name] = value if reward_seen[reward_name] else None
    # A few explicit denominator aliases make reports easier to consume while
    # keeping the formula visible to callers.
    metrics["time_denominator_s"] = denominator_time if denominator_time > 0.0 else None
    metrics["distance_denominator_m"] = distance_total if distance_total > 0.0 else None
    return metrics


def _episode_status(rows: Sequence[Mapping[str, Any]], metrics: Mapping[str, Any]) -> dict[str, Any]:
    final = rows[-1] if rows else {}
    success_values = [
        _first_bool(row, "success", "arrive_dest", "completed") for row in rows
    ]
    if any(value is True for value in success_values):
        success: bool | None = True
    elif any(value is False for value in success_values):
        success = False
    else:
        success = None
    terminated = any(_first_bool(row, "terminated") is True for row in rows)
    truncated = any(_first_bool(row, "truncated") is True for row in rows)
    episode_end = any(
        _first_bool(row, "episode_end", "done") is True for row in rows
    ) or terminated or truncated
    departed = any(
        _first_bool(row, "lane_departure", "start_lane_departure") is True
        for row in rows
    )
    maintained = _first_bool(
        final,
        "start_lane_maintained",
        "lane_maintained",
        "maintains_start_lane",
    )
    if maintained is None:
        explicit_in_lane = [
            _first_bool(row, "in_target_lane", "start_lane_valid") for row in rows
        ]
        known = [value for value in explicit_in_lane if value is not None]
        if known and len(known) == len(explicit_in_lane):
            maintained = all(known)
        elif departed:
            maintained = False
    final_progress = _progress(final)
    route_length = _first_number(
        final,
        "route_length_m",
        "reference_route_length_m",
        "episode_route_length_m",
    )
    unreached = None
    if route_length is not None and final_progress is not None:
        unreached = max(0.0, route_length - final_progress)
    return {
        "success": success,
        "terminated": terminated,
        "truncated": truncated,
        "episode_end": episode_end,
        # A truncation/failure is an observed episode end but remains an
        # incomplete task episode because the destination was not reached.
        "incomplete": success is not True,
        "success_known": success is not None,
        "start_lane_maintained_known": maintained is not None,
        "start_lane_maintained": maintained,
        "lane_departure": departed,
        "duration_s": metrics.get("duration_s"),
        "completion_time_s": metrics.get("duration_s") if success is True else None,
        "final_progress_m": final_progress,
        "route_length_m": route_length,
        "unreached_progress_m": unreached,
    }


def _debounced_crossings(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_getter: Any,
    deadband: float,
    hold_s: float,
    valid_getter: Any | None = None,
    require_preview_valid: bool = True,
) -> dict[str, Any]:
    """Count sign changes only after the opposite side persists for ``hold_s``."""

    if deadband < 0.0 or hold_s <= 0.0:
        raise ValueError("deadband must be non-negative and hold_s must be positive")
    stable: int | None = None
    candidate: int | None = None
    candidate_time = 0.0
    count = 0
    valid_time = 0.0
    for row in rows:
        duration = _dt(row)
        value = value_getter(row)
        usable = duration is not None and value is not None
        if callable(valid_getter):
            # Each crossing series supplies its own validity: steering uses
            # the applied-steering measurement, while lateral uses the host
            # target-lane measurement.  Unknown validity breaks continuity.
            usable = usable and valid_getter(row) is True
        elif require_preview_valid:
            preview_valid = _first_bool(row, "preview_valid")
            usable = usable and preview_valid is not False
        if not usable:
            # A missing/invalid interval breaks temporal continuity.  Do not
            # join signs on the two sides of an unknown measurement.
            stable = None
            candidate = None
            candidate_time = 0.0
            continue
        valid_time += duration
        if value > deadband:
            side = 1
        elif value < -deadband:
            side = -1
        else:
            # Near-zero values are noise/deadband and do not erase a stable
            # side; they also cannot advance an opposite-side hold.
            candidate = None
            candidate_time = 0.0
            continue
        if stable is None:
            stable = side
            candidate = None
            candidate_time = 0.0
        elif side == stable:
            candidate = None
            candidate_time = 0.0
        elif candidate == side:
            candidate_time += duration
            if candidate_time >= hold_s:
                count += 1
                stable = side
                candidate = None
                candidate_time = 0.0
        else:
            candidate = side
            candidate_time = duration
            if candidate_time >= hold_s:
                count += 1
                stable = side
                candidate = None
                candidate_time = 0.0
    return {"count": count, "valid_time_s": valid_time}


def _rate_per_minute(count: int, duration: float | None) -> float | None:
    if duration is None or duration <= 0.0:
        return None
    return float(count) * 60.0 / duration


def _rate_per_km(count: int, distance: float | None) -> float | None:
    if distance is None or distance <= 0.0:
        return None
    return float(count) * 1000.0 / distance


def _episode_summary(
    key: Sequence[Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    steering_deadband: float,
    lateral_deadband_m: float,
    opposite_hold_s: float,
) -> dict[str, Any]:
    metrics = _metric_values(rows)
    status = _episode_status(rows, metrics)
    steering_crossings = _debounced_crossings(
        rows,
        value_getter=_steering,
        deadband=steering_deadband,
        hold_s=opposite_hold_s,
        valid_getter=_steering_valid,
    )
    lateral_crossings = _debounced_crossings(
        rows,
        value_getter=_lateral,
        deadband=lateral_deadband_m,
        hold_s=opposite_hold_s,
        valid_getter=_lateral_valid,
    )
    distance = metrics.get("distance_m")
    duration = metrics.get("duration_s")
    metrics.update(
        {
            "steering_sign_reversal_count": steering_crossings["count"],
            "steering_sign_reversal_per_min": _rate_per_minute(
                steering_crossings["count"], duration
            ),
            "steering_sign_reversal_per_km": _rate_per_km(
                steering_crossings["count"], distance
            ),
            "lateral_center_crossing_count": lateral_crossings["count"],
            "lateral_center_crossing_per_min": _rate_per_minute(
                lateral_crossings["count"], duration
            ),
            "lateral_center_crossing_per_km": _rate_per_km(
                lateral_crossings["count"], distance
            ),
            "debounce_steering_deadband": steering_deadband,
            "debounce_lateral_deadband_m": lateral_deadband_m,
            "debounce_opposite_hold_s": opposite_hold_s,
        }
    )
    result: dict[str, Any] = {
        "episode_key": _key_text(key),
        "episode_case_key": _key_text(_case_key(key)),
        "learning_seed": key[0],
        "scenario_seed": key[1],
        "episode": key[2],
        "run": key[3] if len(key) >= 5 else None,
        "worker": key[4] if len(key) >= 5 else None,
        "metrics": metrics,
    }
    result.update(status)
    # Flat metric aliases are useful for row-oriented callers and preserve the
    # nested ``metrics`` object for machine consumers.
    result.update(metrics)
    return result


def _group_summary(
    groups: Iterable[tuple[Sequence[Any], Sequence[Mapping[str, Any]]]],
    *,
    steering_deadband: float,
    lateral_deadband_m: float,
    opposite_hold_s: float,
) -> dict[str, Any]:
    all_rows: list[Mapping[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    for key, group in groups:
        all_rows.extend(group)
        episodes.append(
            _episode_summary(
                key,
                group,
                steering_deadband=steering_deadband,
                lateral_deadband_m=lateral_deadband_m,
                opposite_hold_s=opposite_hold_s,
            )
        )
    aggregate = _metric_values(all_rows)
    episode_count = len(episodes)

    def count_known(name: str, expected: bool) -> int:
        return sum(item.get(name) is expected for item in episodes)

    successes = count_known("success", True)
    failures = sum(item.get("success") is False for item in episodes)
    success_known = sum(item.get("success_known") is True for item in episodes)
    maintained = count_known("start_lane_maintained", True)
    lane_known = sum(item.get("start_lane_maintained") is not None for item in episodes)
    departed = sum(item.get("lane_departure") is True for item in episodes)
    incomplete = sum(item.get("incomplete") is True for item in episodes)
    truncated = sum(item.get("truncated") is True for item in episodes)
    terminated = sum(item.get("terminated") is True for item in episodes)
    complete_times = [
        item["completion_time_s"]
        for item in episodes
        if _number(item.get("completion_time_s")) is not None
    ]
    unreached = [
        item["unreached_progress_m"]
        for item in episodes
        if _number(item.get("unreached_progress_m")) is not None
    ]
    aggregate.update(
        {
            "episode_count": episode_count,
            "success_count": successes,
            "failure_count": failures,
            "success_known_count": success_known,
            "success_missing_count": episode_count - success_known,
            "incomplete_episode_count": incomplete,
            "truncated_episode_count": truncated,
            "terminated_episode_count": terminated,
            "completion_rate": _fraction(float(successes), float(episode_count)),
            "start_lane_maintained_count": maintained,
            "start_lane_maintained_known_count": lane_known,
            "start_lane_maintained_rate": _fraction(float(maintained), float(lane_known)),
            "lane_departure_episode_count": departed,
            "completion_time_s_mean": (
                sum(complete_times) / len(complete_times) if complete_times else None
            ),
            "completion_time_s_p95": (
                sorted(complete_times)[max(0, math.ceil(len(complete_times) * 0.95) - 1)]
                if complete_times
                else None
            ),
            "unreached_progress_m_mean": (
                sum(unreached) / len(unreached) if unreached else None
            ),
            # Case identity excludes the run name so independently named
            # baseline/lookahead logs can be compared.  Worker remains in the key to
            # prevent two workers' episode 0 records from being merged.
            "episode_cases": [item["episode_case_key"] for item in episodes],
            "episode_keys": [item["episode_key"] for item in episodes],
            "episodes": episodes,
        }
    )
    aggregate["steering_sign_reversal_count"] = sum(
        int(item.get("steering_sign_reversal_count", 0)) for item in episodes
    )
    aggregate["lateral_center_crossing_count"] = sum(
        int(item.get("lateral_center_crossing_count", 0)) for item in episodes
    )
    duration = aggregate.get("duration_s")
    distance = aggregate.get("distance_m")
    aggregate["steering_sign_reversal_per_min"] = _rate_per_minute(
        aggregate["steering_sign_reversal_count"], duration
    )
    aggregate["steering_sign_reversal_per_km"] = _rate_per_km(
        aggregate["steering_sign_reversal_count"], distance
    )
    aggregate["lateral_center_crossing_per_min"] = _rate_per_minute(
        aggregate["lateral_center_crossing_count"], duration
    )
    aggregate["lateral_center_crossing_per_km"] = _rate_per_km(
        aggregate["lateral_center_crossing_count"], distance
    )
    return aggregate


def _metadata_from_rows(
    rows: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any] | None
) -> dict[str, Any]:
    output: dict[str, Any] = dict(metadata or {})
    if "mode" in output:
        output["mode"] = _canonical_mode(output["mode"])
    if "wrapper_order" in output:
        output["wrapper_order"] = _canonical_wrapper_order(output["wrapper_order"])
    # A nested metadata mapping is common in worker headers.  It supplies
    # defaults and is then refined by explicit function metadata.
    for row in rows:
        nested = _lookup(row, "metadata", default=None)
        if isinstance(nested, Mapping):
            for key, value in nested.items():
                output.setdefault(str(key), value)
    if "mode" in output:
        output["mode"] = _canonical_mode(output["mode"])
    if "wrapper_order" in output:
        output["wrapper_order"] = _canonical_wrapper_order(output["wrapper_order"])
    fields = (
        "run",
        "mode",
        "learning_seed",
        "scenario_seed",
        "evaluation_seed",
        "deterministic",
        "schema_version",
        "observation_schema",
        "action_space",
        "wrapper_order",
        "dt",
        "control_dt",
        "decision_dt",
        "vehicle",
        "vehicle_config",
        "scenario_config",
        "config_hash",
        "normalization",
        "normalization_config",
        "learning_timesteps",
        "train_timesteps",
        "learning_budget",
        "road_progress_intervals",
        "speed_bins",
    )
    for field in fields:
        values: list[Any] = []
        for row in rows:
            value = _lookup(row, field, default=_MISSING)
            if value is not _MISSING and value is not None:
                if field == "mode":
                    value = _canonical_mode(value)
                elif field == "wrapper_order":
                    value = _canonical_wrapper_order(value)
                values.append(value)
        if not values:
            continue
        safe_values = [sanitize_json(value) for value in values]
        unique = {_json_dump(value): value for value in safe_values}
        if len(unique) == 1:
            output.setdefault(field, safe_values[0])
        else:
            output.setdefault(f"{field}s", list(unique.values()))
    # Canonical seed sets are always explicit and therefore comparable without
    # relying on an incidental first row.
    for field in ("learning_seed", "scenario_seed", "evaluation_seed"):
        values = []
        for row in rows:
            value = _lookup(row, field, default=_MISSING)
            if value is not _MISSING and value is not None:
                values.append(_stable_id(value))
        existing = output.get(f"{field}s")
        if existing is None and values:
            output[f"{field}s"] = sorted(set(values), key=lambda item: str(item))
        elif existing is not None:
            output[f"{field}s"] = sorted(
                {_stable_id(item) for item in existing}, key=lambda item: str(item)
            )
    output.setdefault("telemetry_schema_version", TELEMETRY_SCHEMA_VERSION)
    return sanitize_json(output)


def _matched_bins(
    rows: Sequence[Mapping[str, Any]],
    *,
    road_progress_intervals: Any,
    speed_bins: Any,
    steering_deadband: float,
    lateral_deadband_m: float,
    opposite_hold_s: float,
) -> dict[str, Any] | None:
    if road_progress_intervals is None or speed_bins is None:
        return None
    grouped: OrderedDict[tuple[str, str, str], list[Mapping[str, Any]]] = OrderedDict()
    observed_road: set[str] = set()
    observed_speed: set[str] = set()
    observed_cases: set[str] = set()
    for index, row in enumerate(rows):
        key = _row_id(row, index)
        case_text = _key_text(_case_key(key))
        progress = _progress(row)
        speed = _speed(row)
        road_label = _interval_label(progress, road_progress_intervals)
        speed_label = _speed_label(speed, speed_bins)
        if road_label is not None:
            observed_road.add(road_label)
        if speed_label is not None:
            observed_speed.add(speed_label)
        if road_label is None or speed_label is None:
            continue
        observed_cases.add(case_text)
        grouped.setdefault(
            (case_text, road_label, speed_label), []
        ).append(row)
    bins: dict[str, Any] = {}
    for (episode_key, road_label, speed_label), bin_rows in grouped.items():
        # Bin metrics are transition-level and therefore need no invented
        # samples.  We use a stable key that includes the episode case.
        aggregate = _metric_values(bin_rows)
        bins[f"{episode_key}|road={road_label}|speed={speed_label}"] = {
            "episode_key": episode_key,
            "road_interval": road_label,
            "speed_bin": speed_label,
            "metrics": aggregate,
            "decision_count": len(bin_rows),
        }
    expected_road = _configured_labels(road_progress_intervals)
    expected_speed = _configured_labels(speed_bins)
    expected_bin_keys = {
        f"{case}|road={road}|speed={speed}"
        for case in observed_cases
        for road in expected_road
        for speed in expected_speed
    }
    observed_bin_keys = set(bins)
    return {
        "road_progress_intervals": sanitize_json(road_progress_intervals),
        "speed_bins": sanitize_json(speed_bins),
        "bins": bins,
        "bin_count": len(bins),
        "observed_road_intervals": sorted(observed_road),
        "missing_road_intervals": sorted(set(expected_road) - observed_road),
        "observed_speed_bins": sorted(observed_speed),
        "missing_speed_bins": sorted(set(expected_speed) - observed_speed),
        "expected_bin_count": len(expected_bin_keys),
        "missing_bins": sorted(expected_bin_keys - observed_bin_keys),
    }


def summarize_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    metadata: Mapping[str, Any] | None = None,
    expected_episode_cases: Iterable[Any] | None = None,
    road_progress_intervals: Any = None,
    speed_bins: Any = None,
    steering_deadband: float = DEFAULT_STEERING_DEADBAND,
    lateral_deadband_m: float = DEFAULT_LATERAL_DEADBAND_M,
    opposite_hold_s: float = DEFAULT_OPPOSITE_HOLD_S,
) -> dict[str, Any]:
    """Summarize transition rows without hiding missing or incomplete data.

    ``road_progress_intervals`` and ``speed_bins`` are explicit lists or
    mappings of ``label -> (low, high)``.  When both are supplied, each
    summary contains matched-bin diagnostics keyed by episode, interval, and
    speed band.  Values outside the configured bins are retained in overall
    metrics but are not silently assigned to a bin.
    """

    if steering_deadband < 0.0 or lateral_deadband_m < 0.0 or opposite_hold_s <= 0.0:
        raise ValueError("debounce thresholds are invalid")
    row_list = [dict(row) for row in rows]
    indexed_groups = _group_rows_indexed(row_list)
    groups = [(key, [row for _, row in entries]) for key, entries in indexed_groups.items()]
    aggregate = _group_summary(
        groups,
        steering_deadband=steering_deadband,
        lateral_deadband_m=lateral_deadband_m,
        opposite_hold_s=opposite_hold_s,
    )
    meta = _metadata_from_rows(row_list, metadata)
    if road_progress_intervals is None:
        road_progress_intervals = meta.get("road_progress_intervals")
    if speed_bins is None:
        speed_bins = meta.get("speed_bins")
    matched = _matched_bins(
        row_list,
        road_progress_intervals=road_progress_intervals,
        speed_bins=speed_bins,
        steering_deadband=steering_deadband,
        lateral_deadband_m=lateral_deadband_m,
        opposite_hold_s=opposite_hold_s,
    )
    expected = None
    if expected_episode_cases is not None:
        expected = sorted({str(case) for case in expected_episode_cases})
        observed = set(aggregate.get("episode_cases", []))
        aggregate["expected_episode_cases"] = expected
        aggregate["missing_episode_cases"] = sorted(set(expected) - observed)
        aggregate["unexpected_episode_cases"] = sorted(observed - set(expected))
        aggregate["episode_cases_complete"] = not (
            aggregate["missing_episode_cases"] or aggregate["unexpected_episode_cases"]
        )
    else:
        aggregate["episode_cases_complete"] = None
    result: dict[str, Any] = {
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "metrics": aggregate,
        "metadata": meta,
        "episodes": aggregate.get("episodes", []),
        "episode_count": aggregate.get("episode_count", 0),
    }
    # Top-level aliases avoid forcing CLI/report clients to know whether they
    # received a pre-existing summary or this canonical envelope.
    result.update(aggregate)
    result["metadata"]["episode_cases"] = aggregate.get("episode_cases", [])
    result["metadata"]["episode_cases_complete"] = aggregate.get(
        "episode_cases_complete"
    )
    if matched is not None:
        result["matched_bins"] = matched
        result["metrics"]["matched_bin_count"] = matched["bin_count"]
        result["matched_bin_count"] = matched["bin_count"]
    result["by_learning_seed"] = {}
    result["by_scenario_seed"] = {}
    # Reuse the already sorted per-episode list for seed-indexed reporting.
    for seed_name, position in (("learning_seed", 0), ("scenario_seed", 1)):
        seed_groups: OrderedDict[Any, list[tuple[Any, Sequence[Mapping[str, Any]]]]] = OrderedDict()
        for key, group in groups:
            seed_groups.setdefault(key[position], []).append((key, group))
        target = result["by_learning_seed" if position == 0 else "by_scenario_seed"]
        for seed, seed_group in seed_groups.items():
            seed_metrics = _group_summary(
                seed_group,
                steering_deadband=steering_deadband,
                lateral_deadband_m=lateral_deadband_m,
                opposite_hold_s=opposite_hold_s,
            )
            target["null" if seed is None else str(seed)] = seed_metrics
    return sanitize_json(result)


def summarize_jsonl(
    paths: str | os.PathLike[str] | Iterable[str | os.PathLike[str]],
    **kwargs: Any,
) -> dict[str, Any]:
    """Read JSONL worker files and call :func:`summarize_rows`."""

    return summarize_rows(read_jsonl(paths), **kwargs)


def _summary_metrics(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = summary.get("metrics")
    return metrics if isinstance(metrics, Mapping) else summary


def _metadata_value(summary: Mapping[str, Any], key: str) -> Any:
    metadata = summary.get("metadata")
    if isinstance(metadata, Mapping) and key in metadata:
        value = metadata[key]
    else:
        value = summary.get(key)
    if key == "mode":
        return _canonical_mode(value)
    if key == "wrapper_order":
        return _canonical_wrapper_order(value)
    return value


def _seed_set(summary: Mapping[str, Any], name: str) -> set[Any] | None:
    value = _metadata_value(summary, name)
    if value is None:
        value = _metadata_value(summary, f"{name}s")
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        values = [value]
    elif isinstance(value, Iterable):
        values = list(value)
    else:
        values = [value]
    return {_stable_id(item) for item in values}


def _episode_case_set(summary: Mapping[str, Any]) -> set[str] | None:
    value = _metadata_value(summary, "episode_cases")
    if value is None:
        value = summary.get("episode_cases")
    if value is None:
        episodes = summary.get("episodes")
        if isinstance(episodes, Sequence):
            values = [
                item.get("episode_case_key", item.get("episode_key"))
                for item in episodes
                if isinstance(item, Mapping)
            ]
        else:
            return None
    else:
        values = list(value) if isinstance(value, Iterable) and not isinstance(value, (str, bytes)) else [value]
    return {str(item) for item in values if item is not None}


def _same_or_unknown(
    before: Mapping[str, Any], after: Mapping[str, Any], key: str
) -> bool | None:
    first = _metadata_value(before, key)
    second = _metadata_value(after, key)
    if first is None or second is None:
        return None
    return sanitize_json(first) == sanitize_json(second)


def _same_alternatives(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    keys: Sequence[str],
) -> bool | None:
    """Compare the first common metadata spelling, retaining unknown state."""

    for key in keys:
        first = _metadata_value(before, key)
        second = _metadata_value(after, key)
        if first is not None and second is not None:
            return sanitize_json(first) == sanitize_json(second)
    # If each side uses a different spelling, compare the corresponding
    # values by order only when exactly one value exists on each side.
    first_values = [
        _metadata_value(before, key) for key in keys if _metadata_value(before, key) is not None
    ]
    second_values = [
        _metadata_value(after, key) for key in keys if _metadata_value(after, key) is not None
    ]
    if len(first_values) == len(second_values) == 1:
        return sanitize_json(first_values[0]) == sanitize_json(second_values[0])
    return None


def _schema_mapping(summary: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = _metadata_value(summary, "observation_schema")
    if isinstance(value, Mapping):
        return value
    return None


def _schema_shape(schema: Mapping[str, Any] | None) -> tuple[int, ...] | None:
    if schema is None:
        return None
    value = schema.get("shape", schema.get("obs_shape"))
    if value is None:
        return None
    if isinstance(value, int):
        return (int(value),)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        try:
            return tuple(int(item) for item in value)
        except (TypeError, ValueError):
            return None
    return None


def _schema_prefix(schema: Mapping[str, Any] | None) -> Any:
    if schema is None:
        return None
    for key in (
        "prefix",
        "base",
        "base_schema",
        "existing",
        "existing_schema",
        "prefix_schema",
    ):
        value = schema.get(key)
        if value is not None:
            return sanitize_json(value)
    # Flat schema metadata is also accepted.  Remove mode-specific fields so
    # the common 262-dimensional contract can be compared independently.
    fields = (
        "base_shape",
        "base_dtype",
        "base_low",
        "base_high",
        "base_fields",
        "prefix_shape",
        "prefix_dtype",
        "prefix_low",
        "prefix_high",
        "prefix_fields",
    )
    selected = {key: schema[key] for key in fields if key in schema}
    return sanitize_json(selected) if selected else None


def _schema_extra_fields(schema: Mapping[str, Any] | None) -> list[Any] | None:
    if schema is None:
        return None
    for key in ("extra_fields", "added_fields", "preview_fields", "suffix_fields"):
        value = schema.get(key)
        if value is not None:
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return [sanitize_json(item) for item in value]
            return [sanitize_json(value)]
    fields = schema.get("fields")
    shape = _schema_shape(schema)
    if isinstance(fields, Sequence) and not isinstance(fields, (str, bytes)):
        if shape and len(shape) == 1 and len(fields) >= 265:
            return [sanitize_json(item) for item in fields[262:]]
    return None


def _observation_schema_compatible(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    before_mode: str | None,
    after_mode: str | None,
) -> tuple[bool | None, bool | None, dict[str, Any]]:
    """Compare the base observation contract and mode-specific suffix.

    Baseline intentionally has 262 values while lookahead modes have 265.
    Comparing the complete shape would reject the intended baseline→lookahead
    experiment, so this
    helper checks the common prefix separately and checks the exact 3-value
    suffix for the lookahead modes.  Missing semantic metadata remains unknown.
    """

    before_mode = _canonical_mode(before_mode)
    after_mode = _canonical_mode(after_mode)
    first = _schema_mapping(before)
    second = _schema_mapping(after)
    first_shape = _schema_shape(first)
    second_shape = _schema_shape(second)
    first_prefix = _schema_prefix(first)
    second_prefix = _schema_prefix(second)
    first_extra = _schema_extra_fields(first)
    second_extra = _schema_extra_fields(second)
    detail: dict[str, Any] = {
        "before_shape": first_shape,
        "after_shape": second_shape,
        "prefix_equal": None,
        "extra_fields_equal": None,
        "expected_suffix": ["preview_x_norm", "preview_y_norm", "preview_valid"],
    }
    if first_prefix is not None and second_prefix is not None:
        detail["prefix_equal"] = first_prefix == second_prefix
    if first_extra is not None and second_extra is not None:
        detail["extra_fields_equal"] = first_extra == second_extra
    exact_equal: bool | None = None
    compatible: bool | None = None
    if first_shape is not None and second_shape is not None:
        exact_equal = first_shape == second_shape
    if before_mode == "baseline" and after_mode == "lookahead_obs":
        expected_suffix = detail["expected_suffix"]
        shape_ok = first_shape == (262,) and second_shape == (265,)
        prefix_ok = detail["prefix_equal"] is True
        suffix_ok = (
            second_extra == expected_suffix
            if second_extra is not None
            else False
        )
        compatible = shape_ok and prefix_ok and suffix_ok
        if first_prefix is None or second_prefix is None or second_extra is None:
            compatible = None
    elif before_mode == "lookahead_obs" and after_mode == "lookahead_obs_pp_reward":
        compatible = (
            exact_equal
            and detail["prefix_equal"] is True
            and detail["extra_fields_equal"] is True
        )
        if first_prefix is None or second_prefix is None or first_extra is None or second_extra is None:
            compatible = None
    return exact_equal, compatible, detail


def _wrapper_order_compatible(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    before_mode: str | None,
    after_mode: str | None,
) -> bool | None:
    before_mode = _canonical_mode(before_mode)
    after_mode = _canonical_mode(after_mode)
    first = _metadata_value(before, "wrapper_order")
    second = _metadata_value(after, "wrapper_order")
    if first is None or second is None:
        return None
    if before_mode == "baseline" and after_mode == "lookahead_obs":
        if not isinstance(first, Sequence) or isinstance(first, (str, bytes)):
            return None
        if not isinstance(second, Sequence) or isinstance(second, (str, bytes)):
            return None
        # Some runs place a common diagnostic ``lookahead_learning`` wrapper around
        # every mode, including baseline.  When both orders are literally the
        # same, the observation schema gate separately verifies baseline's
        # 262-value prefix and lookahead_obs's 265-value suffix; do not reject that
        # otherwise fair arrangement merely because both lists contain the
        # wrapper name.
        if sanitize_json(first) == sanitize_json(second):
            return True
        first_without_preview = [item for item in first if str(item) != "lookahead_learning"]
        second_without_preview = [item for item in second if str(item) != "lookahead_learning"]
        return (
            list(first_without_preview) == list(second_without_preview)
            and sum(str(item) == "lookahead_learning" for item in second) == 1
            and sum(str(item) == "lookahead_learning" for item in first) == 0
        )
    if before_mode == "lookahead_obs" and after_mode == "lookahead_obs_pp_reward":
        return sanitize_json(first) == sanitize_json(second)
    return sanitize_json(first) == sanitize_json(second)


def _metadata_comparison(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    before_mode: str | None,
    after_mode: str | None,
) -> dict[str, Any]:
    before_mode = _canonical_mode(before_mode)
    after_mode = _canonical_mode(after_mode)
    first_mode = before_mode or _metadata_value(before, "mode")
    second_mode = after_mode or _metadata_value(after, "mode")
    pair = (first_mode, second_mode)
    allowed = pair in (
        ("baseline", "lookahead_obs"),
        ("lookahead_obs", "lookahead_obs_pp_reward"),
    )
    checks: dict[str, Any] = {
        "allowed_pair": allowed,
        "mode_pair": [first_mode, second_mode],
        "learning_seeds_equal": None,
        "scenario_seeds_equal": None,
        "evaluation_seeds_equal": None,
        "observation_schema_equal": _same_or_unknown(
            before, after, "observation_schema"
        ),
        "observation_schema_compatible": None,
        "action_space_equal": _same_or_unknown(before, after, "action_space"),
        "wrapper_order_equal": _same_or_unknown(before, after, "wrapper_order"),
        "wrapper_order_compatible": None,
        "deterministic_equal": _same_or_unknown(before, after, "deterministic"),
        "dt_equal": _same_alternatives(
            before, after, ("dt", "decision_dt", "control_dt")
        ),
        "control_dt_equal": _same_or_unknown(before, after, "control_dt"),
        "decision_dt_equal": _same_or_unknown(before, after, "decision_dt"),
        "vehicle_equal": _same_or_unknown(before, after, "vehicle"),
        "vehicle_config_equal": _same_alternatives(
            before, after, ("vehicle_config", "vehicle")
        ),
        "scenario_config_equal": _same_alternatives(
            before, after, ("scenario_config", "config_hash")
        ),
        "config_hash_equal": _same_or_unknown(before, after, "config_hash"),
        "normalization_equal": _same_or_unknown(
            before, after, "normalization"
        ),
        "normalization_config_equal": _same_alternatives(
            before, after, ("normalization_config", "normalization")
        ),
        "learning_timesteps_equal": _same_or_unknown(
            before, after, "learning_timesteps"
        ),
        "train_timesteps_equal": _same_or_unknown(
            before, after, "train_timesteps"
        ),
        "learning_budget_equal": _same_alternatives(
            before,
            after,
            ("learning_budget", "train_timesteps", "learning_timesteps"),
        ),
        "episode_cases_equal": None,
    }
    schema_equal, schema_compatible, schema_detail = _observation_schema_compatible(
        before,
        after,
        before_mode=first_mode,
        after_mode=second_mode,
    )
    # ``observation_schema_equal`` retains its literal meaning for reports;
    # compatibility is the gate used for the intended 262→265 comparison.
    if schema_equal is not None:
        checks["observation_schema_equal"] = schema_equal
    checks["observation_schema_compatible"] = schema_compatible
    checks["observation_schema_detail"] = schema_detail
    checks["wrapper_order_compatible"] = _wrapper_order_compatible(
        before,
        after,
        before_mode=first_mode,
        after_mode=second_mode,
    )
    for label, name in (
        ("learning_seeds_equal", "learning_seed"),
        ("scenario_seeds_equal", "scenario_seed"),
        ("evaluation_seeds_equal", "evaluation_seed"),
    ):
        first = _seed_set(before, name)
        second = _seed_set(after, name)
        if first is not None and second is not None:
            checks[label] = first == second
            checks[f"{name}s_before"] = sorted(first, key=str)
            checks[f"{name}s_after"] = sorted(second, key=str)
    first_cases = _episode_case_set(before)
    second_cases = _episode_case_set(after)
    if first_cases is not None and second_cases is not None:
        checks["episode_cases_equal"] = first_cases == second_cases
        checks["episode_cases_before_only"] = sorted(first_cases - second_cases)
        checks["episode_cases_after_only"] = sorted(second_cases - first_cases)
        checks["common_episode_cases"] = sorted(first_cases & second_cases)
    return checks


def _metric_delta(
    before_metrics: Mapping[str, Any], after_metrics: Mapping[str, Any], name: str
) -> dict[str, Any]:
    before = _number(before_metrics.get(name))
    after = _number(after_metrics.get(name))
    return {
        "before": before,
        "after": after,
        "absolute": after - before if before is not None and after is not None else None,
        "relative": _relative_change(before, after),
    }


def _matched_comparison(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any] | None:
    first = before.get("matched_bins")
    second = after.get("matched_bins")
    if not isinstance(first, Mapping) or not isinstance(second, Mapping):
        return None
    first_bins = first.get("bins")
    second_bins = second.get("bins")
    if not isinstance(first_bins, Mapping) or not isinstance(second_bins, Mapping):
        return {"available": False, "reason": "invalid_bin_data", "bins": {}}
    common = sorted(set(first_bins) & set(second_bins))
    output_bins: dict[str, Any] = {}
    for key in common:
        left = first_bins[key]
        right = second_bins[key]
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            continue
        left_metrics = left.get("metrics")
        right_metrics = right.get("metrics")
        if not isinstance(left_metrics, Mapping) or not isinstance(right_metrics, Mapping):
            continue
        output_bins[str(key)] = {
            "road_interval": left.get("road_interval"),
            "speed_bin": left.get("speed_bin"),
            "lateral_rms_m": _metric_delta(left_metrics, right_metrics, "lateral_rms_m"),
            "lateral_abs_p95_m": _metric_delta(
                left_metrics, right_metrics, "lateral_abs_p95_m"
            ),
            "steering_tv_per_s": _metric_delta(
                left_metrics, right_metrics, "steering_tv_per_s"
            ),
            "steering_rate_rms": _metric_delta(
                left_metrics, right_metrics, "steering_rate_rms"
            ),
            "speed_mean_mps": _metric_delta(left_metrics, right_metrics, "speed_mean_mps"),
        }
    return {
        "available": True,
        "common_bin_count": len(output_bins),
        "before_bin_count": len(first_bins),
        "after_bin_count": len(second_bins),
        "before_missing_road_intervals": list(
            first.get("missing_road_intervals", [])
        ),
        "after_missing_road_intervals": list(
            second.get("missing_road_intervals", [])
        ),
        "before_missing_speed_bins": list(first.get("missing_speed_bins", [])),
        "after_missing_speed_bins": list(second.get("missing_speed_bins", [])),
        "before_missing_bins": list(first.get("missing_bins", [])),
        "after_missing_bins": list(second.get("missing_bins", [])),
        "unreached_intervals_equal": (
            list(first.get("missing_road_intervals", []))
            == list(second.get("missing_road_intervals", []))
        ),
        "bins": output_bins,
    }


def evaluate_improvement_gate(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    pair: tuple[str, str] | None = None,
    strict_metadata: bool = True,
    require_matched: bool = True,
    warning_relative: float = DEFAULT_WARNING_RELATIVE,
) -> dict[str, Any]:
    """Evaluate the pre-registered conservative improvement conditions.

    ``None`` is unknown evidence.  Unknown or mismatched required evidence
    never becomes a passing improvement.  ``strict_metadata=False`` is useful
    for exploratory reports, but the result still records every unknown check.
    """

    if warning_relative < 0.0:
        raise ValueError("warning_relative must be non-negative")
    before_metrics = _summary_metrics(before)
    after_metrics = _summary_metrics(after)
    before_mode = _canonical_mode(pair[0]) if pair else _metadata_value(before, "mode")
    after_mode = _canonical_mode(pair[1]) if pair else _metadata_value(after, "mode")
    metadata = _metadata_comparison(
        before,
        after,
        before_mode=before_mode,
        after_mode=after_mode,
    )
    failed: list[str] = []
    unknown: list[str] = []
    checks: dict[str, Any] = {}

    if not metadata["allowed_pair"]:
        failed.append(
            "only baseline->lookahead_obs and "
            "lookahead_obs->lookahead_obs_pp_reward comparisons are allowed"
        )
    required_metadata = (
        "learning_seeds_equal",
        "scenario_seeds_equal",
        "episode_cases_equal",
        "evaluation_seeds_equal",
        "observation_schema_compatible",
        "action_space_equal",
        "wrapper_order_compatible",
        "deterministic_equal",
        "dt_equal",
        "vehicle_config_equal",
        "scenario_config_equal",
        "normalization_config_equal",
        "learning_budget_equal",
    )
    for name in required_metadata:
        value = metadata.get(name)
        checks[name] = value
        if value is False:
            failed.append(name)
        elif value is None:
            unknown.append(name)
    def compare_no_worse(name: str) -> tuple[float | None, float | None, bool | None]:
        delta = _metric_delta(before_metrics, after_metrics, name)
        checks[name] = delta
        if delta["before"] is None or delta["after"] is None:
            unknown.append(name)
            return delta["before"], delta["after"], None
        passed = delta["after"] <= delta["before"] + 1e-12
        if not passed:
            failed.append(name)
        return delta["before"], delta["after"], passed

    lateral_results = [
        compare_no_worse("lateral_rms_m"),
        compare_no_worse("lateral_abs_p95_m"),
    ]
    lateral_improved = any(
        before_value is not None
        and after_value is not None
        and after_value < before_value - 1e-12
        for before_value, after_value, _ in lateral_results
    )
    checks["lateral_at_least_one_improved"] = lateral_improved
    if not lateral_improved:
        failed.append("lateral metric did not improve")

    steering_results = [
        compare_no_worse("steering_tv_per_s"),
        compare_no_worse("steering_rate_rms"),
        compare_no_worse("steering_tv_per_m"),
    ]
    steering_improved = any(
        before_value is not None
        and after_value is not None
        and after_value < before_value - 1e-12
        for before_value, after_value, _ in steering_results
    )
    checks["steering_change_at_least_one_improved"] = steering_improved
    if not steering_improved:
        failed.append("steering change metric did not improve")

    def compare_no_lower(name: str) -> tuple[float | None, float | None, bool | None]:
        delta = _metric_delta(before_metrics, after_metrics, name)
        checks[name] = delta
        if delta["before"] is None or delta["after"] is None:
            unknown.append(name)
            return delta["before"], delta["after"], None
        passed = delta["after"] >= delta["before"] - 1e-12
        if not passed:
            failed.append(name)
        return delta["before"], delta["after"], passed

    # Completion, lane maintenance, and preview validity are safety gates:
    # missing after-values cannot be interpreted as a good result.  Stopping
    # is lower-is-better and is handled by the other direction below.
    for name in ("completion_rate", "start_lane_maintained_rate", "preview_valid_rate"):
        compare_no_lower(name)
    compare_no_worse("stop_time_fraction")

    warnings: list[str] = []
    for name, text in (
        ("speed_mean_mps", "mean speed decreased by more than 5%"),
        ("progress_per_s", "progress per second decreased by more than 5%"),
    ):
        delta = _metric_delta(before_metrics, after_metrics, name)
        checks[name] = delta
        relative = delta["relative"]
        if relative is None:
            unknown.append(name)
        elif relative < -warning_relative:
            warnings.append(text)
    time_delta = _metric_delta(
        before_metrics, after_metrics, "completion_time_s_mean"
    )
    checks["completion_time_s_mean"] = time_delta
    if time_delta["relative"] is None:
        unknown.append("completion_time_s_mean")
    elif time_delta["relative"] > warning_relative:
        warnings.append("completion time increased by more than 5%")
    # A large speed/progress loss or longer completion time can make lateral
    # metrics look better by driving less.  It is therefore a failed gate as
    # well as a visible management warning.
    for warning in warnings:
        failed.append(f"comparison condition warning: {warning}")

    matched = _matched_comparison(before, after)
    if require_matched:
        if matched is None:
            unknown.append("matched_road_progress_speed_bins")
        elif matched.get("common_bin_count", 0) <= 0:
            unknown.append("matched_road_progress_speed_bins")
        else:
            missing_interval_fields = (
                "before_missing_road_intervals",
                "after_missing_road_intervals",
                "before_missing_speed_bins",
                "after_missing_speed_bins",
                "before_missing_bins",
                "after_missing_bins",
            )
            checks["matched_missing_intervals"] = {
                field: list(matched.get(field, [])) for field in missing_interval_fields
            }
            if any(matched.get(field) for field in missing_interval_fields):
                # A run that never reached an interval cannot support a
                # claim about that interval.  Keep the missing list visible
                # and make the gate inconclusive rather than intersecting it
                # away.
                unknown.append("unreached_road_progress_or_speed_interval")
            matched_lateral_improvement = False
            matched_steering_improvement = False
            matched_bad = False
            for item in matched.get("bins", {}).values():
                lateral_item = item["lateral_rms_m"]
                p95_item = item["lateral_abs_p95_m"]
                steering_item = item["steering_tv_per_s"]
                for metric_item in (lateral_item, p95_item, steering_item):
                    if metric_item["before"] is None or metric_item["after"] is None:
                        unknown.append("matched_bin_metric")
                    elif metric_item["after"] > metric_item["before"] + 1e-12:
                        matched_bad = True
                if (
                    lateral_item["before"] is not None
                    and lateral_item["after"] is not None
                    and lateral_item["after"] < lateral_item["before"] - 1e-12
                ) or (
                    p95_item["before"] is not None
                    and p95_item["after"] is not None
                    and p95_item["after"] < p95_item["before"] - 1e-12
                ):
                    matched_lateral_improvement = True
                if (
                    steering_item["before"] is not None
                    and steering_item["after"] is not None
                    and steering_item["after"] < steering_item["before"] - 1e-12
                ):
                    matched_steering_improvement = True
            checks["matched_lateral_improved"] = matched_lateral_improvement
            checks["matched_steering_improved"] = matched_steering_improvement
            if matched_bad:
                failed.append("matched road/speed metric worsened")
            if not matched_lateral_improvement:
                failed.append("matched lateral metric did not improve")
            if not matched_steering_improvement:
                failed.append("matched steering metric did not improve")

    # A PP error can improve while the driving task remains unchanged.  No PP
    # agreement field appears in the positive conditions above by design.
    if (
        _number(before_metrics.get("pp_error_mean")) is not None
        and _number(after_metrics.get("pp_error_mean")) is not None
        and _number(after_metrics.get("pp_error_mean"))
        < _number(before_metrics.get("pp_error_mean"))
        and not lateral_improved
    ):
        failed.append("PP agreement alone is not task improvement")

    if strict_metadata and unknown:
        failed.extend(f"missing evidence: {name}" for name in sorted(set(unknown)))
    # Duplicate failure reasons make CLI output noisy; retain deterministic
    # ordering for machine and human reports.
    failed = list(dict.fromkeys(failed))
    unknown = list(dict.fromkeys(unknown))
    improved = not failed and not unknown
    return {
        "improved": improved,
        "status": "improvement" if improved else ("inconclusive" if unknown and not failed else "no_improvement"),
        "failed_conditions": failed,
        "unknown_conditions": unknown,
        "warnings": warnings,
        "checks": checks,
        "metadata": metadata,
        "require_matched": require_matched,
        "warning_relative": warning_relative,
    }


def compare_summaries(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    before_mode: str | None = None,
    after_mode: str | None = None,
    strict_metadata: bool = True,
    require_matched: bool = True,
    warning_relative: float = DEFAULT_WARNING_RELATIVE,
) -> dict[str, Any]:
    """Return deltas and an honest improvement decision for two summaries."""

    before_mode = _canonical_mode(before_mode)
    after_mode = _canonical_mode(after_mode)
    before_metrics = _summary_metrics(before)
    after_metrics = _summary_metrics(after)
    metric_names = sorted(
        set(before_metrics.keys()) & set(after_metrics.keys())
    )
    deltas = {
        name: _metric_delta(before_metrics, after_metrics, name)
        for name in metric_names
        if isinstance(name, str)
        and name
        not in {
            "episodes",
            "invalid_reason_time_s",
            "episode_cases",
            "expected_episode_cases",
            "missing_episode_cases",
            "unexpected_episode_cases",
        }
    }
    metadata = _metadata_comparison(
        before,
        after,
        before_mode=before_mode,
        after_mode=after_mode,
    )
    gate = evaluate_improvement_gate(
        before,
        after,
        pair=(before_mode, after_mode)
        if before_mode is not None and after_mode is not None
        else None,
        strict_metadata=strict_metadata,
        require_matched=require_matched,
        warning_relative=warning_relative,
    )
    return sanitize_json(
        {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "comparison": [
                before_mode or _metadata_value(before, "mode"),
                after_mode or _metadata_value(after, "mode"),
            ],
            "metadata": metadata,
            "metric_deltas": deltas,
            "matched": _matched_comparison(before, after),
            "warnings": gate["warnings"],
            "improvement_gate": gate,
            "improved": gate["improved"],
        }
    )


def compare_runs(
    before_rows: Iterable[Mapping[str, Any]] | Mapping[str, Any],
    after_rows: Iterable[Mapping[str, Any]] | Mapping[str, Any],
    *,
    before_mode: str | None = None,
    after_mode: str | None = None,
    summarize_kwargs: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Summarize raw rows if necessary and compare the resulting runs."""

    options = dict(summarize_kwargs or {})
    if isinstance(before_rows, Mapping):
        before = before_rows
    else:
        before = summarize_rows(before_rows, **options)
    if isinstance(after_rows, Mapping):
        after = after_rows
    else:
        after = summarize_rows(after_rows, **options)
    return compare_summaries(
        before,
        after,
        before_mode=before_mode,
        after_mode=after_mode,
        **kwargs,
    )


# Readable aliases for callers that describe the operation as aggregation.
aggregate_metrics = summarize_rows
summarize = summarize_rows
compute_metrics = summarize_rows
compare = compare_summaries
compare_metrics = compare_summaries


__all__ = [
    "CANONICAL_FIELDS",
    "CANONICAL_MODES",
    "DEFAULT_LATERAL_DEADBAND_M",
    "DEFAULT_OPPOSITE_HOLD_S",
    "DEFAULT_STEERING_DEADBAND",
    "JsonlWorkerWriter",
    "LEGACY_NAMESPACE",
    "MODE_ALIASES",
    "PACKAGE_NAME",
    "SCHEMA_VERSION",
    "TELEMETRY_SCHEMA_VERSION",
    "TelemetryWriter",
    "WorkerJsonlWriter",
    "aggregate_metrics",
    "compare",
    "compare_metrics",
    "compare_runs",
    "compare_summaries",
    "evaluate_improvement_gate",
    "compute_metrics",
    "iter_jsonl",
    "read_jsonl",
    "sanitize_json",
    "summarize",
    "summarize_jsonl",
    "summarize_rows",
]
