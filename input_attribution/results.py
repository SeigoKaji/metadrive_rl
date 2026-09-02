"""Safe, reproducible artifact writing for input-attribution runs.

The analysis commands deliberately keep all files for a run in a sibling
staging directory.  A partially completed analysis therefore cannot overwrite
an earlier successful result directory.  This module has no MetaDrive or SB3
dependency so it is also usable by the offline ``analyze`` command.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


class ArtifactError(ValueError):
    """Raised when an artifact destination is unsafe or cannot be published."""


def validate_basename(value: str, *, label: str = "output prefix") -> str:
    """Return a safe single path component, rejecting traversal on all hosts."""

    windows_path = PureWindowsPath(value) if isinstance(value, str) else None
    has_windows_invalid_character = isinstance(value, str) and any(
        character in '<>:"|?*' or ord(character) < 32 for character in value
    )
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "\x00" in value
        or "/" in value
        or "\\" in value
        or Path(value).name != value
        or windows_path is None
        or windows_path.name != value
        or windows_path.is_reserved()
        or value.endswith((" ", "."))
        or has_windows_invalid_character
    ):
        raise ArtifactError(
            f"{label} must be a non-empty basename without directory components: "
            f"{value!r}"
        )
    return value


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a regular input file."""

    if path.is_symlink() or not path.is_file():
        raise ArtifactError(f"expected a regular file for hashing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while chunk := file_obj.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now_iso() -> str:
    """Return an explicit, timezone-aware timestamp for metadata."""

    return datetime.now(timezone.utc).isoformat()


def json_value(value: Any) -> Any:
    """Convert normal analysis values into JSON-safe values without guessing."""

    if is_dataclass(value) and not isinstance(value, type):
        return json_value(asdict(value))
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _require_regular_parent(path: Path) -> None:
    """Create ``path`` while refusing to descend through symlink components."""

    # ``Path.mkdir`` follows an existing symlink.  Artifact paths can be used
    # in automated jobs, so make that choice explicit rather than silently
    # escaping the configured OUTPUT_DIR.
    pending: list[Path] = []
    current = path
    while not current.exists():
        pending.append(current)
        if current.parent == current:
            raise ArtifactError(f"cannot create artifact root: {path}")
        current = current.parent
    if current.is_symlink() or not current.is_dir():
        raise ArtifactError(f"artifact parent is not a regular directory: {current}")
    for directory in reversed(pending):
        directory.mkdir()
        if directory.is_symlink() or not directory.is_dir():
            raise ArtifactError(f"unsafe artifact directory: {directory}")


def safe_child(root: Path, relative: str | Path) -> Path:
    """Resolve a relative artifact name without allowing traversal or symlinks."""

    candidate = Path(relative)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ArtifactError(f"artifact path must be a relative non-traversing path: {relative!r}")
    _require_regular_parent(root)
    root_resolved = root.resolve(strict=True)
    target = root / candidate
    parent = target.parent
    _require_regular_parent(parent)
    # Every existing component below the root must be a real directory.  The
    # final target itself may be absent or a regular file, never a symlink.
    current = root
    for part in candidate.parts[:-1]:
        current = current / part
        if current.is_symlink() or not current.is_dir():
            raise ArtifactError(f"unsafe artifact directory component: {current}")
    try:
        target.parent.resolve(strict=True).relative_to(root_resolved)
    except ValueError as error:
        raise ArtifactError(f"artifact path escapes root: {relative!r}") from error
    if target.is_symlink():
        raise ArtifactError(f"refusing to replace symlink artifact: {target}")
    return target


def _temporary_sibling(path: Path) -> Path:
    _require_regular_parent(path.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    return Path(temporary_name)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes through a sibling temporary and atomically replace ``path``."""

    if path.is_symlink():
        raise ArtifactError(f"refusing to replace symlink artifact: {path}")
    temporary = _temporary_sibling(path)
    try:
        with temporary.open("wb") as file_obj:
            file_obj.write(data)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write readable deterministic JSON using the standard safe writer."""

    text = json.dumps(
        json_value(payload),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    atomic_write_text(path, text)


def atomic_write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    """Write JSON Lines atomically and return the number of records."""

    lines: list[str] = []
    for record in records:
        lines.append(
            json.dumps(
                json_value(record), ensure_ascii=False, sort_keys=True, allow_nan=False
            )
        )
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def atomic_write_npz(path: Path, **arrays: Any) -> None:
    """Write a compressed NumPy archive atomically without suffix surprises."""

    if path.suffix != ".npz":
        raise ArtifactError(f"NPZ artifact must use a .npz suffix: {path}")
    if path.is_symlink():
        raise ArtifactError(f"refusing to replace symlink artifact: {path}")
    temporary = _temporary_sibling(path)
    try:
        with temporary.open("wb") as file_obj:
            np.savez_compressed(file_obj, **arrays)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _row_mapping(row: Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return {str(key): json_value(value) for key, value in row.items()}
    if hasattr(row, "to_dict"):
        converted = row.to_dict()
        if isinstance(converted, Mapping):
            return {str(key): json_value(value) for key, value in converted.items()}
    raise TypeError(f"CSV row must be mapping-like, got {type(row)!r}")


def atomic_write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]] | Any,
    *,
    fieldnames: Sequence[str] | None = None,
) -> int:
    """Write mapping rows (or a pandas DataFrame) to CSV atomically."""

    if hasattr(rows, "to_dict") and hasattr(rows, "columns"):
        rows = rows.to_dict(orient="records")
    materialized = [_row_mapping(row) for row in rows]
    if fieldnames is None:
        fieldnames = list(
            dict.fromkeys(key for row in materialized for key in row)
        )
    names = [str(name) for name in fieldnames]
    if path.is_symlink():
        raise ArtifactError(f"refusing to replace symlink artifact: {path}")
    temporary = _temporary_sibling(path)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=names, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(materialized)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return len(materialized)


class StagedRunDirectory:
    """A staging directory that publishes only after a successful analysis.

    ``publish`` deliberately keeps an old completed directory intact until the
    new staging directory has been made visible.  It rejects symlink targets so
    an untrusted output prefix cannot redirect artifact writes elsewhere.
    """

    def __init__(self, target: Path) -> None:
        self.target = target
        self.staging: Path | None = None
        self._published = False

    def __enter__(self) -> Path:
        validate_basename(self.target.name, label="run directory name")
        _require_regular_parent(self.target.parent)
        if self.target.is_symlink():
            raise ArtifactError(f"refusing to replace symlink run directory: {self.target}")
        if self.target.exists() and not self.target.is_dir():
            raise ArtifactError(
                f"run directory target exists but is not a regular directory: {self.target}"
            )
        self.staging = Path(
            tempfile.mkdtemp(prefix=f".{self.target.name}.staging-", dir=self.target.parent)
        )
        return self.staging

    def publish(self) -> Path:
        if self.staging is None:
            raise RuntimeError("staged run directory has not been entered")
        if self._published:
            return self.target
        if self.target.is_symlink():
            raise ArtifactError(f"refusing to replace symlink run directory: {self.target}")
        if self.target.exists() and not self.target.is_dir():
            raise ArtifactError(
                f"run directory target exists but is not a regular directory: {self.target}"
            )

        backup: Path | None = None
        try:
            if self.target.exists():
                backup = Path(
                    tempfile.mkdtemp(prefix=f".{self.target.name}.previous-", dir=self.target.parent)
                )
                # ``mkdtemp`` creates the name, while rename needs a vacant
                # destination.  It is a known, local directory under the same
                # parent and therefore safe to remove non-recursively.
                backup.rmdir()
                os.replace(self.target, backup)
            os.replace(self.staging, self.target)
        except BaseException:
            if backup is not None and backup.exists() and not self.target.exists():
                os.replace(backup, self.target)
            raise
        else:
            self._published = True
            self.staging = None
            if backup is not None:
                shutil.rmtree(backup)
            return self.target

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.staging is not None and self.staging.exists():
            shutil.rmtree(self.staging)


def expanded_schema_rows(schema: Any) -> list[dict[str, Any]]:
    """Return schema rows from the public schema object with light duck typing.

    The core schema module owns validation and canonical feature expansion.  A
    small normalizer here keeps artifact/report code independent of its exact
    dataclass representation and is intentionally read-only.
    """

    candidates: Any = getattr(schema, "expanded_rows", None)
    if callable(candidates):
        candidates = candidates()
    if candidates is None:
        candidates = getattr(schema, "expanded_features", None)
        if callable(candidates):
            candidates = candidates()
    if candidates is None:
        for name in ("expand", "expanded", "features"):
            candidate = getattr(schema, name, None)
            if callable(candidate):
                candidates = candidate()
                break
            if candidate is not None:
                candidates = candidate
                break
    if candidates is None:
        raise ArtifactError("schema does not expose expanded feature rows")

    # ``ObservationSchema.groups`` is the authoritative membership mapping.
    # Parsed schemas usually repeat it on every Feature, but programmatic
    # schemas and lightweight duck-typed schemas need the mapping filled back
    # into their artifact rows as well (including overlapping groups).
    groups_by_index: dict[int, list[str]] = {}
    schema_groups = getattr(schema, "groups", None)
    if isinstance(schema_groups, Mapping):
        for group_name, indices in schema_groups.items():
            if not isinstance(group_name, str):
                continue
            try:
                members = tuple(indices)
            except TypeError:
                continue
            for member in members:
                if isinstance(member, (int, np.integer)) and not isinstance(member, bool):
                    groups_by_index.setdefault(int(member), []).append(group_name)

    rows: list[dict[str, Any]] = []
    for position, feature in enumerate(candidates):
        if isinstance(feature, Mapping):
            row = dict(feature)
        elif is_dataclass(feature):
            row = asdict(feature)
        else:
            row = {
                key: getattr(feature, key)
                for key in (
                    "index",
                    "name",
                    "feature_name",
                    "block",
                    "group",
                    "kind",
                    "angle_deg",
                    "resolved",
                    "description",
                    "constant",
                )
                if hasattr(feature, key)
            }
        index = row.get("index", position)
        name = row.get("name", row.get("feature_name", f"feature_{index}"))
        numeric_index = int(index)
        raw_groups = row.get("groups")
        if isinstance(raw_groups, str):
            feature_groups = [group for group in raw_groups.split(",") if group]
        elif isinstance(raw_groups, (list, tuple)):
            feature_groups = [str(group) for group in raw_groups if str(group)]
        else:
            feature_groups = []
        explicit_group = row.get("group")
        if explicit_group is not None and str(explicit_group) not in feature_groups:
            feature_groups.insert(0, str(explicit_group))
        for group in groups_by_index.get(numeric_index, []):
            if group not in feature_groups:
                feature_groups.append(group)
        group_value = str(explicit_group) if explicit_group is not None else (
            feature_groups[0] if feature_groups else None
        )
        rows.append(
            {
                "index": numeric_index,
                "name": str(name),
                "block": row.get("block"),
                "group": group_value,
                "groups": ",".join(feature_groups),
                "kind": row.get("kind"),
                "angle_deg": row.get("angle_deg"),
                "resolved": row.get("resolved", True),
                "description": row.get("description"),
                "constant": row.get("constant"),
            }
        )
    return sorted(rows, key=lambda row: int(row["index"]))


def write_expanded_schema_csv(path: Path, schema: Any) -> list[dict[str, Any]]:
    """Write the schema expansion that gives every result table its meaning."""

    rows = expanded_schema_rows(schema)
    atomic_write_csv(
        path,
        rows,
        fieldnames=(
            "index",
            "name",
            "block",
            "group",
            "groups",
            "kind",
            "angle_deg",
            "resolved",
            "description",
            "constant",
        ),
    )
    return rows


def write_report(path: Path, report_markdown: str) -> None:
    """Write a complete UTF-8 report as an atomic artifact."""

    atomic_write_text(path, report_markdown.rstrip() + "\n")
