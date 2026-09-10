"""Portable JSON/frame storage for attribution runs.

Raw rollout JSON is intentionally boring and inspectable.  Report generation
can be repeated from a copied run directory without importing a simulator or
changing these files.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np


FORMAT_VERSION = "input-attribution.v1"
BASE_MAIN_SHA = "7849aad80ac353fd616c1a1398c11dd3497eed05"


class StorageError(ValueError):
    """Run data cannot be represented or safely addressed."""


def jsonable(value: object) -> object:
    """Convert common numpy values to strict JSON-compatible values.

    Non-finite numbers raise instead of becoming JSON ``NaN`` or being
    replaced by zero.  Missing measurements must remain ``None`` at the
    caller's explicit boundary.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise StorageError("cannot serialize NaN or Inf")
        return value
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): jsonable(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    # Preserve small third-party scalar values where possible.  Do not use
    # repr for arbitrary model/environment objects because it can hide state
    # and produce unstable manifests.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return jsonable(item())
        except Exception:
            pass
    raise StorageError(f"value of type {type(value).__name__} is not JSONable")


def canonical_json(value: object) -> str:
    return json.dumps(
        jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _safe_component(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StorageError(f"{name} must be a non-empty string")
    if value in {".", ".."} or "/" in value or "\\" in value or "\x00" in value:
        raise StorageError(f"unsafe {name}: {value!r}")
    return value


def ensure_run_layout(run_dir: str | Path) -> Path:
    path = Path(run_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    (path / "data").mkdir(parents=True, exist_ok=True)
    (path / "data" / "frames").mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, value: object) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(jsonable(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    # A temporary sibling prevents a report process from observing a partial
    # document while keeping the operation local to the run directory.
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def read_json(path: str | Path) -> Any:
    source = Path(path).expanduser().resolve()
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StorageError(f"cannot read JSON: {source}") from error


def rollout_file(run_dir: str | Path, rollout_id: str, *, offline: bool = False) -> Path:
    run_path = ensure_run_layout(run_dir)
    safe_id = _safe_component(rollout_id, name="rollout id")
    if not offline and safe_id.upper() in {"P00", "P00_BASELINE", "BASELINE"}:
        filename = "P00_baseline.json"
    else:
        suffix = "_offline" if offline else ""
        filename = f"{safe_id}{suffix}.json"
    return run_path / "data" / filename


def write_rollout(run_dir: str | Path, rollout: Mapping[str, object], *, offline: bool = False) -> Path:
    if "id" not in rollout:
        raise StorageError("rollout is missing id")
    return write_json(rollout_file(run_dir, str(rollout["id"]), offline=offline), rollout)


def read_rollout(run_dir: str | Path, rollout_id: str, *, offline: bool = False) -> dict[str, object]:
    value = read_json(rollout_file(run_dir, rollout_id, offline=offline))
    if not isinstance(value, dict):
        raise StorageError("rollout JSON must contain an object")
    return value


def write_manifest(run_dir: str | Path, manifest: Mapping[str, object]) -> Path:
    run_path = ensure_run_layout(run_dir)
    return write_json(run_path / "data" / "manifest.json", manifest)


def read_manifest(run_dir: str | Path) -> dict[str, object]:
    value = read_json(ensure_run_layout(run_dir) / "data" / "manifest.json")
    if not isinstance(value, dict):
        raise StorageError("manifest JSON must contain an object")
    return value


def _frame_array(frame: object) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] not in {3, 4}:
        raise StorageError(f"frame must be HxWx3/RGB or HxWx4/RGBA, found {array.shape}")
    if np.issubdtype(array.dtype, np.complexfloating) or not np.all(np.isfinite(array)):
        raise StorageError("frame contains non-finite or complex values")
    if array.dtype != np.uint8:
        if np.any(array < 0) or np.any(array > 255):
            raise StorageError("frame values must lie in [0, 255]")
        array = np.asarray(np.rint(array), dtype=np.uint8)
    return np.ascontiguousarray(array)


def save_frame(run_dir: str | Path, rollout_id: str, step: int, frame: object) -> str:
    """Save an RGB frame and return its run-relative JSON reference."""

    run_path = ensure_run_layout(run_dir)
    safe_id = _safe_component(rollout_id, name="rollout id")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise StorageError("frame step must be a non-negative integer")
    array = _frame_array(frame)
    relative = Path("data") / "frames" / safe_id / f"{step:06d}.png"
    destination = run_path / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
    except ModuleNotFoundError as error:  # pragma: no cover - requirements include Pillow
        raise StorageError("Pillow is required to save PNG frames") from error
    Image.fromarray(array[..., :3], mode="RGB").save(destination, format="PNG")
    return relative.as_posix()


def frame_paths(run_dir: str | Path, rollout_id: str) -> list[Path]:
    run_path = ensure_run_layout(run_dir)
    safe_id = _safe_component(rollout_id, name="rollout id")
    directory = run_path / "data" / "frames" / safe_id
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.png"))


def _git_output(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def implementation_identity(repo_root: str | Path | None = None) -> dict[str, object]:
    """Record git context plus a content hash for uncommitted package code."""

    root = Path(repo_root or Path(__file__).resolve().parents[1]).expanduser().resolve()
    package = root / "input_attribution"
    digest = hashlib.sha256()
    files: list[str] = []
    if package.is_dir():
        for path in sorted(package.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(root).as_posix()
            files.append(relative)
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
    status = _git_output(root, "status", "--porcelain", "--untracked-files=all")
    return {
        "git_head": _git_output(root, "rev-parse", "HEAD"),
        "dirty": bool(status),
        "git_status": status or "",
        "source_sha256": digest.hexdigest(),
        "source_files": files,
    }


__all__ = [
    "BASE_MAIN_SHA",
    "FORMAT_VERSION",
    "StorageError",
    "canonical_json",
    "ensure_run_layout",
    "frame_paths",
    "implementation_identity",
    "jsonable",
    "read_json",
    "read_manifest",
    "read_rollout",
    "rollout_file",
    "save_frame",
    "write_json",
    "write_manifest",
    "write_rollout",
]
