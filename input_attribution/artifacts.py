"""再現可能な解析成果物の保存とmanifest整合性。

数値処理本体からファイル名・hash・runライフサイクルを切り離す。各runは
必ず新しいIDで作成し、途中失敗も ``status.json`` とmanifestへ残すため、成功
レポートを後から偽装しにくい構造にしている。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import tempfile
from typing import Any


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        while chunk := file_obj.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_object(value: object) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_paths(paths: Iterable[str | Path]) -> str:
    """Hash file names and bytes in stable order.

    Missing paths are represented explicitly. This is useful in a manifest where
    an absent adapter source must be visible rather than silently omitted.
    """

    digest = hashlib.sha256()
    for item in sorted((str(Path(path)) for path in paths)):
        path = Path(item)
        digest.update(item.encode("utf-8"))
        digest.update(b"\0")
        if not path.is_file():
            digest.update(b"<missing>\0")
            continue
        with path.open("rb") as file_obj:
            while chunk := file_obj.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _safe_name(value: str, label: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"{label} must be a basename: {value!r}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=_json_default) + "\n"
    # A temporary sibling makes interrupted writes leave the previous metadata
    # readable. Replacement is only inside the current run directory.
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file_obj:
            file_obj.write(payload)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        Path(temporary_name).replace(path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # pragma: no cover - exotic array scalar
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:  # pragma: no cover - exotic tensor
            pass
    return str(value)


@dataclass(frozen=True)
class RunArtifacts:
    """Paths belonging to one immutable experiment run."""

    run_dir: Path
    experiment: str
    model: str
    run_id: str
    analysis_stage: str | None = None
    analysis_id: str | None = None

    def for_analysis(self, stage: str) -> "RunArtifacts":
        directory = {"offline": "01_offline", "closed_loop": "02_closed_loop", "ig": "03_ig"}[stage]
        identifier = _default_run_id()
        (self.run_dir / directory / identifier).mkdir(parents=True, exist_ok=False)
        return replace(self, analysis_stage=stage, analysis_id=identifier)

    def _stage_path(self, stage: str, directory: str) -> Path:
        root = self.run_dir / directory
        return root / self.analysis_id if self.analysis_stage == stage and self.analysis_id else root

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def resolved_config_path(self) -> Path:
        return self.run_dir / "resolved_config.json"

    @property
    def status_path(self) -> Path:
        return self.run_dir / "status.json"

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / "logs"

    @property
    def reference_dir(self) -> Path:
        return self.run_dir / "00_reference"

    @property
    def offline_dir(self) -> Path:
        return self._stage_path("offline", "01_offline")

    @property
    def closed_loop_dir(self) -> Path:
        return self._stage_path("closed_loop", "02_closed_loop")

    @property
    def ig_dir(self) -> Path:
        return self._stage_path("ig", "03_ig")

    def ensure_layout(self) -> "RunArtifacts":
        for path in (self.logs_dir, self.reference_dir, self.offline_dir, self.closed_loop_dir, self.ig_dir):
            path.mkdir(parents=True, exist_ok=True)
        return self

    def write_json(self, relative: str | Path, value: object) -> Path:
        relative = Path(relative)
        if self.analysis_stage and self.analysis_id:
            prefix = {"offline": "01_offline", "closed_loop": "02_closed_loop", "ig": "03_ig"}[self.analysis_stage]
            if relative.parts and relative.parts[0] == prefix and (len(relative.parts) < 2 or relative.parts[1] != self.analysis_id):
                relative = Path(prefix) / self.analysis_id / Path(*relative.parts[1:])
        target = self.run_dir / relative
        if self.run_dir.resolve() not in target.resolve().parents and target.resolve() != self.run_dir.resolve():
            raise ValueError("artifact path escapes run directory")
        _write_json(target, value)
        return target

    def read_json(self, relative: str | Path) -> Any:
        path = self.run_dir / Path(relative)
        with path.open("r", encoding="utf-8") as file_obj:
            return json.load(file_obj)

    def update_status(self, stage: str, state: str, *, error: str | None = None, **details: Any) -> dict[str, Any]:
        if state not in {"pending", "running", "success", "failed", "skipped"}:
            raise ValueError(f"unsupported artifact state: {state}")
        try:
            current = self.read_json("status.json")
        except FileNotFoundError:
            current = {"run_id": self.run_id, "stages": {}}
        entry: dict[str, Any] = {
            "state": state,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if error:
            entry["error"] = str(error)
        if self.analysis_stage == stage and self.analysis_id:
            directory = {"offline": "01_offline", "closed_loop": "02_closed_loop", "ig": "03_ig"}[stage]
            entry.update(analysis_id=self.analysis_id, relative_dir=f"{directory}/{self.analysis_id}")
        entry.update(details)
        current.setdefault("stages", {})[stage] = entry
        current["updated_at"] = entry["updated_at"]
        _write_json(self.status_path, current)
        return current

    def save_resolved_config(self, config: object) -> Path:
        value = config.to_dict() if hasattr(config, "to_dict") else config
        return self.write_json("resolved_config.json", value)

    def save_manifest(self, manifest: Mapping[str, Any]) -> Path:
        return self.write_json("manifest.json", dict(manifest))

    def load_manifest(self) -> dict[str, Any]:
        return self.read_json("manifest.json")


def _default_run_id() -> str:
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{now}-{secrets.token_hex(4)}"


def create_run(
    root: str | Path,
    experiment: str,
    model: str,
    *,
    run_id: str | None = None,
    exist_ok: bool = False,
) -> RunArtifacts:
    """Create a unique output directory without overwriting prior runs."""

    experiment = _safe_name(str(experiment), "experiment")
    model = _safe_name(str(model), "model")
    parent = Path(root).expanduser().resolve() / experiment / model
    parent.mkdir(parents=True, exist_ok=True)
    requested = _safe_name(str(run_id), "run_id") if run_id else None
    candidate = requested or _default_run_id()
    for _ in range(100):
        destination = parent / candidate
        if not destination.exists():
            destination.mkdir(parents=True)
            result = RunArtifacts(destination, experiment, model, candidate)
            result.ensure_layout()
            result.update_status("run", "pending")
            return result
        if requested and not exist_ok:
            raise FileExistsError(f"run directory already exists: {destination}")
        candidate = f"{_default_run_id()}-{secrets.token_hex(2)}"
    raise FileExistsError(f"could not allocate a unique run directory under {parent}")


def open_run(run_dir: str | Path) -> RunArtifacts:
    path = Path(run_dir).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"run directory not found: {path}")
    # run_id/model/experiment are informative only; keep paths from disk intact.
    return RunArtifacts(path, path.parent.parent.name, path.parent.name, path.name)


def build_manifest(
    *,
    config: object,
    command: str | None = None,
    model_path: str | Path | None = None,
    schema_path: str | Path | None = None,
    patterns: object | None = None,
    preprocess: object | None = None,
    observations_path: str | Path | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a manifest with content hashes for every reproducibility input."""

    config_dict = config.to_dict() if hasattr(config, "to_dict") else config
    git: dict[str, Any] = {"head": None, "dirty": None}
    try:
        repo = Path(__file__).resolve().parent.parent
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        git = {"head": head.stdout.strip(), "dirty": bool(dirty.stdout.strip())}
    except (OSError, subprocess.CalledProcessError):
        git = {"head": None, "dirty": None, "error": "git metadata unavailable"}

    manifest: dict[str, Any] = {
        "manifest_version": 1,
        "base_commit": "0184eb26509eb33997229d0aa99c0b8939ec6a1e",
        "package_version": __import__("input_attribution").__version__,
        "argv": list(__import__("sys").argv),
        "seeds": (config_dict or {}).get("seeds", {}),
        "input_files": input_file_hashes(config_dict or {}),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "git": git,
        "python": {"version": __import__("platform").python_version(), "executable": __import__("sys").executable},
        "config_sha256": sha256_object(config_dict),
        "patterns_sha256": sha256_object(patterns if patterns is not None else (config_dict or {}).get("patterns", [])),
        "preprocess_sha256": sha256_object(preprocess if preprocess is not None else (config_dict or {}).get("preprocess", {})),
        "model": {"path": str(model_path) if model_path is not None else None, "sha256": None},
        "schema": {"path": str(schema_path) if schema_path is not None else None, "sha256": None},
        "observations": {"path": str(observations_path) if observations_path is not None else None, "sha256": None},
        "code_sha256": code_fingerprint(),
    }
    for key, path in (("model", model_path), ("schema", schema_path), ("observations", observations_path)):
        if path is not None and Path(path).is_file():
            manifest[key]["sha256"] = sha256_file(path)
    if extra:
        manifest.update(dict(extra))
    return manifest


def code_fingerprint(package_dir: str | Path | None = None) -> str:
    package = Path(package_dir) if package_dir is not None else Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        if "tests" in path.relative_to(package).parts:
            continue
        digest.update(path.relative_to(package).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def input_file_hashes(config: Mapping[str, Any]) -> dict[str, str | None]:
    paths = []
    for section, keys in (("environment", ("path", "config")), ("preprocess", ("stats_path", "stats")), ("adapter", ("path",))):
        for key in keys:
            value = (config.get(section) or {}).get(key)
            if isinstance(value, str):
                paths.append(Path(value))
    return {str(path.resolve()): sha256_file(path) if path.is_file() else None for path in paths}


def seal_reference(store: RunArtifacts) -> None:
    manifest = store.load_manifest()
    files = {str(path.relative_to(store.run_dir)): sha256_file(path)
             for path in sorted(store.reference_dir.rglob("*")) if path.is_file()}
    manifest["reference_files"] = files
    manifest["data_id"] = sha256_object(files)
    manifest["observations"] = {"path": str(store.reference_dir / "observations.npy"),
                                "sha256": files["00_reference/observations.npy"]}
    store.save_manifest(manifest)


def verify_reference(store: RunArtifacts) -> None:
    manifest = store.load_manifest()
    expected = manifest.get("reference_files")
    if not expected:
        raise ValueError("reference data hashes are missing; collection is incomplete")
    for relative, digest in expected.items():
        path = store.run_dir / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"saved reference hash mismatch: {relative}")
    if manifest.get("data_id") != sha256_object(expected):
        raise ValueError("reference data ID mismatch")


def verify_manifest_compatibility(
    manifest: Mapping[str, Any],
    *,
    config: object | None = None,
    model_path: str | Path | None = None,
    schema_path: str | Path | None = None,
    patterns: object | None = None,
    preprocess: object | None = None,
    observations_path: str | Path | None = None,
) -> tuple[bool, list[str]]:
    """Check hashes before a partial command appends to an existing run."""

    mismatches: list[str] = []
    config_dict = config.to_dict() if hasattr(config, "to_dict") else config
    checks = {
        "config_sha256": sha256_object(config_dict) if config is not None else None,
        "patterns_sha256": sha256_object(patterns) if patterns is not None else None,
        "preprocess_sha256": sha256_object(preprocess) if preprocess is not None else None,
    }
    for key, actual in checks.items():
        if actual is not None and manifest.get(key) != actual:
            mismatches.append(key)
    for key, path in (("model", model_path), ("schema", schema_path), ("observations", observations_path)):
        if path is None:
            continue
        expected = (manifest.get(key) or {}).get("sha256")
        actual = sha256_file(path) if Path(path).is_file() else None
        if expected != actual:
            mismatches.append(f"{key}.sha256")
    if manifest.get("code_sha256") != code_fingerprint():
        mismatches.append("code_sha256")
    if config is not None and manifest.get("input_files") != input_file_hashes(config_dict):
        mismatches.append("input_files")
    for path, expected in (manifest.get("adapter_sources") or {}).items():
        if not Path(path).is_file() or sha256_file(path) != expected:
            mismatches.append(f"adapter_source:{path}")
    return (not mismatches, mismatches)


def jsonl_write(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, default=_json_default))
            file_obj.write("\n")
    return target


def jsonl_read(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_observations(
    store: RunArtifacts,
    observations: Any,
    records: Iterable[Mapping[str, Any]],
    *,
    stage: str = "00_reference",
) -> dict[str, str | None]:
    """Write model-input arrays and JSONL metadata without mutating source obs."""

    directory = store.run_dir / stage
    directory.mkdir(parents=True, exist_ok=True)
    array_path = directory / "observations.npy"
    import numpy as np
    copied = np.array(observations, copy=True)
    if copied.ndim != 2 or not np.issubdtype(copied.dtype, np.number):
        raise ValueError("saved observations must be a rectangular numeric vector batch")
    np.save(array_path, copied, allow_pickle=False)
    records_path = jsonl_write(directory / "records.jsonl", records)
    return {
        "observations": str(array_path),
        "records": str(records_path),
        "observations_sha256": sha256_file(array_path) if array_path.is_file() else None,
    }


__all__ = [
    "RunArtifacts",
    "build_manifest",
    "canonical_json",
    "code_fingerprint",
    "create_run",
    "jsonl_read",
    "jsonl_write",
    "open_run",
    "save_observations",
    "sha256_file",
    "sha256_object",
    "sha256_paths",
    "verify_manifest_compatibility",
]
