"""Read-only diagnostics used by :mod:`lookahead_learning.runner`.

The diagnostic layer deliberately keeps imports to the Python standard
library.  In particular, importing the command line package must not import
MetaDrive, Panda3D, Stable-Baselines3, or Matplotlib.  Runtime inspection is
performed only by :func:`probe_raw_environment` after the caller has selected
``doctor --probe``.

The current checkout is known to expose MetaDrive's upstream 259-wide raw
observation.  The requested experiment requires a separately verified
262-wide host schema.  Diagnostics report this distinction and never pad,
truncate, or otherwise reinterpret a raw vector.
"""

from __future__ import annotations

from dataclasses import dataclass
import base64
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import pickletools
import re
import subprocess
import sys
import zipfile
from typing import Any, Iterable, Mapping


DIAGNOSTIC_SCHEMA_VERSION = "lookahead_learning.diagnostics.v1"


def _json_safe(value: Any) -> Any:
    """Return a JSON-safe representation without importing NumPy."""

    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not value == value:
            return None
        if isinstance(value, float) and value in (float("inf"), float("-inf")):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    return repr(value)


def write_json(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> Path:
    """Write one diagnostic JSON file, refusing an existing destination."""

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing diagnostic file: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return destination


def read_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read a JSON object and reject a different top-level type."""

    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {source}")
    return value


def _distribution_version(*names: str) -> str | None:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _module_spec(name: str) -> dict[str, Any]:
    """Inspect an import spec without executing the module."""

    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ModuleNotFoundError, AttributeError, ValueError) as error:
        return {"name": name, "available": False, "error": f"{type(error).__name__}: {error}"}
    if spec is None:
        return {"name": name, "available": False, "origin": None}
    locations = None if spec.submodule_search_locations is None else [str(item) for item in spec.submodule_search_locations]
    return {
        "name": name,
        "available": True,
        "origin": None if spec.origin is None else str(spec.origin),
        "search_locations": locations,
        "loader": None if spec.loader is None else type(spec.loader).__name__,
    }


def _run_git(root: Path, *arguments: str) -> tuple[str | None, str | None]:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return None, f"{type(error).__name__}: {error}"
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        return None, f"git exit {result.returncode}: {message}"
    return result.stdout.strip(), None


def git_state(root: str | os.PathLike[str]) -> dict[str, Any]:
    """Return commit and dirty state for one checkout without changing it."""

    path = Path(root).expanduser().resolve()
    commit, commit_error = _run_git(path, "rev-parse", "HEAD")
    status, status_error = _run_git(path, "status", "--porcelain=v1", "--untracked-files=all")
    return {
        "path": str(path),
        "exists": path.is_dir(),
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
        "status": status,
        "errors": [item for item in (commit_error, status_error) if item is not None],
    }


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _metadrive_location() -> dict[str, Any]:
    """Resolve MetaDrive from the interpreter's import spec, without import.

    A runner can be copied to a different project checkout while its editable
    MetaDrive dependency remains installed elsewhere.  Looking beside the
    project in that case reports the wrong assets (or invents a missing asset
    failure), so all package facts start with the interpreter's actual spec.
    """

    spec = _module_spec("metadrive")
    package_dir: Path | None = None
    locations = spec.get("search_locations")
    if isinstance(locations, list) and locations:
        candidate = Path(str(locations[0])).expanduser().resolve()
        if candidate.is_dir():
            package_dir = candidate
    if package_dir is None:
        origin = spec.get("origin")
        if isinstance(origin, str) and origin not in {"", "built-in", "frozen"}:
            candidate = Path(origin).expanduser().resolve().parent
            if candidate.is_dir():
                package_dir = candidate
    source_root: Path | None = None
    if package_dir is not None:
        current = package_dir
        for candidate in (current, *current.parents):
            if (candidate / ".git").exists():
                source_root = candidate
                break
        if source_root is None:
            # For a wheel/site-packages install there is no checkout.  The
            # package's parent is still the precise source location to report.
            source_root = package_dir.parent
    return {
        "spec": spec,
        "package_dir": None if package_dir is None else str(package_dir),
        "source_root": None if source_root is None else str(source_root),
    }


def _find_source_version(_project_root: Path | None = None) -> str | None:
    """Read MetaDrive's installed package version without importing it."""

    location = _metadrive_location()
    package_dir_value = location.get("package_dir")
    if not isinstance(package_dir_value, str):
        return None
    package_dir = Path(package_dir_value)
    candidates = (package_dir / "version.py", package_dir / "VERSION")
    pattern = re.compile(r"^VERSION\s*=\s*['\"]([^'\"]+)['\"]", re.MULTILINE)
    for candidate in candidates:
        try:
            match = pattern.search(candidate.read_text(encoding="utf-8"))
        except OSError:
            continue
        if match:
            return match.group(1)
    return None


def asset_audit(project_root: str | os.PathLike[str]) -> dict[str, Any]:
    """Describe MetaDrive asset state and the normal pull decision."""

    root = Path(project_root).expanduser().resolve()
    location = _metadrive_location()
    package_dir_value = location.get("package_dir")
    package_dir = None if not isinstance(package_dir_value, str) else Path(package_dir_value)
    source_root_value = location.get("source_root")
    source_root = None if not isinstance(source_root_value, str) else Path(source_root_value)
    asset_candidate = None if package_dir is None else package_dir / "assets"
    asset_root = asset_candidate.resolve() if asset_candidate is not None and asset_candidate.is_dir() else None
    version_file = None if asset_root is None else asset_root / "version.txt"
    grass_file = None if asset_root is None else asset_root / "textures" / "grass1" / "GroundGrassGreen002_COL_1K.jpg"
    source_version = _find_source_version(root)
    try:
        asset_version = None if version_file is None else version_file.read_text(encoding="utf-8").strip()
    except OSError:
        asset_version = None
    grass_exists = bool(grass_file and grass_file.is_file())
    assets_zip = [] if source_root is None else [source_root / "assets.zip"]
    assets_lock = [] if source_root is None else [source_root / "assets.lock"]
    zip_path = next((item.resolve() for item in assets_zip if item.exists()), None)
    lock_path = next((item.resolve() for item in assets_lock if item.exists()), None)
    should_update = source_version is None or asset_version is None or asset_version != source_version or not grass_exists
    return {
        "asset_root": None if asset_root is None else str(asset_root),
        "package_dir": None if package_dir is None else str(package_dir.resolve()),
        "source_root": None if source_root is None else str(source_root.resolve()),
        "package_spec": location.get("spec"),
        "project_root_used_for_context": str(root),
        "source_version": source_version,
        "asset_version": asset_version,
        "version_match": source_version is not None and source_version == asset_version,
        "grass_texture": None if grass_file is None else {
            "path": str(grass_file),
            "exists": grass_exists,
            "size_bytes": grass_file.stat().st_size if grass_exists else None,
        },
        "assets_zip": None if zip_path is None else str(zip_path),
        "assets_lock": None if lock_path is None else str(lock_path),
        "normal_engine_try_pull_asset_would_update": should_update,
        "normal_engine_try_pull_asset_note": (
            "BaseEngine.try_pull_asset calls pull_asset only when the asset directory is missing "
            "or AssetLoader.should_update_asset() is true."
        ),
        "map_store_is_disk_cache": False,
        "map_store_note": "PGMapManager store_map retains generated maps in memory.",
    }


def cache_environment() -> dict[str, str | None]:
    """Return cache variables that a probe will isolate before host imports."""

    return {
        name: os.environ.get(name)
        for name in ("MPLCONFIGDIR", "XDG_CACHE_HOME", "TMPDIR", "TORCH_HOME")
    }


def static_source_evidence(project_root: str | os.PathLike[str]) -> dict[str, Any]:
    """List local source facts used by the runner's contract checks."""

    root = Path(project_root).expanduser().resolve()
    location = _metadrive_location()
    source_root_value = location.get("source_root")
    package_dir_value = location.get("package_dir")
    metadrive_root = (
        Path(source_root_value).resolve()
        if isinstance(source_root_value, str)
        else Path(package_dir_value).resolve().parent
        if isinstance(package_dir_value, str)
        else root
    )
    package_dir = (
        Path(package_dir_value).resolve()
        if isinstance(package_dir_value, str)
        else metadrive_root / "metadrive"
    )
    evidence = [
        {
            "path": str(package_dir / "envs" / "base_env.py"),
            "symbols": ["BaseEnv.reset", "BaseEnv.step", "BaseEnv._step_simulator", "BaseEnv._get_step_return"],
            "lines": "435-473, 512-643",
            "fact": "reset is lazy and step performs one engine decision with configured decision_repeat physics steps.",
        },
        {
            "path": str(package_dir / "engine" / "base_engine.py"),
            "symbols": ["BaseEngine.before_step", "BaseEngine.step", "BaseEngine.try_pull_asset", "BaseEngine.seed"],
            "lines": "416-460, 562-583, 768-783",
            "fact": "external action is stored before manager actuation; physics_world_step_size is repeated; assets pull only when stale.",
        },
        {
            "path": str(package_dir / "component" / "vehicle" / "base_vehicle.py"),
            "symbols": ["BaseVehicle.before_step", "BaseVehicle._set_action", "BaseVehicle.current_action", "BaseVehicle.last_action"],
            "lines": "210-231, 472-520, 974-979",
            "fact": "the clipped applied pair is appended to last_current_action, then normalized steering is mapped to Bullet max_steering.",
        },
        {
            "path": str(package_dir / "policy" / "env_input_policy.py"),
            "symbols": ["EnvInputPolicy.act", "EnvInputPolicy.convert_to_continuous_action"],
            "lines": "26-48",
            "fact": "Discrete action IDs decode steering by modulo and throttle by floor division, then clip to [-1,1].",
        },
        {
            "path": str(package_dir / "component" / "vehicle" / "vehicle_type.py"),
            "symbols": ["DefaultVehicle", "XLVehicle", "LVehicle", "MVehicle", "SVehicle"],
            "lines": "12-185",
            "fact": "front/rear axle offsets are vehicle-model constants and must be read from the actual class.",
        },
        {
            "path": str(package_dir / "component" / "lane" / "abs_lane.py"),
            "symbols": ["AbstractLane.position", "AbstractLane.local_coordinates", "AbstractLane.heading_theta_at"],
            "lines": "12-78",
            "fact": "lane protocol uses finite longitudinal/lateral coordinates, world position, and heading in radians.",
        },
        {
            "path": str(package_dir / "obs" / "state_obs.py"),
            "symbols": ["StateObservation.vehicle_state", "LidarStateObservation.observe", "LidarStateObservation._add_noise_to_cloud_points"],
            "lines": "37-160, 167-235",
            "fact": "the current raw state prefix is 9 values, navigation contributes 10 values, optional nearby-vehicle fields contribute 16, and the configured lidar contributes 240; observation code refreshes lidar state as part of observe.",
        },
        {
            "path": str(package_dir / "component" / "sensors" / "lidar.py"),
            "symbols": ["Lidar.perceive"],
            "lines": "1-220",
            "fact": "lidar perception is a simulator observation side effect; diagnostics never call observe a second time or re-sample lidar noise.",
        },
    ]
    return {
        "host_project_root": str(root),
        "metadrive_package": location,
        "metadrive_source": git_state(metadrive_root),
        "files": evidence,
    }


def dependency_audit(project_root: str | os.PathLike[str]) -> dict[str, Any]:
    """Collect interpreter, package spec, and version evidence without imports."""

    root = Path(project_root).expanduser().resolve()
    metadrive_location = _metadrive_location()
    source_root_value = metadrive_location.get("source_root")
    metadrive_source = (
        git_state(source_root_value)
        if isinstance(source_root_value, str)
        else None
    )
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "packages": {
            "metadrive": {
                "distribution_version": _distribution_version("metadrive", "metadrive-simulator"),
                "source_version": _find_source_version(root),
                "spec": metadrive_location.get("spec"),
                "source_checkout": metadrive_source,
            },
            "stable_baselines3": {
                "version": _distribution_version("stable-baselines3"),
                "spec": _module_spec("stable_baselines3"),
            },
            "gymnasium": {"version": _distribution_version("gymnasium"), "spec": _module_spec("gymnasium")},
            "numpy": {"version": _distribution_version("numpy"), "spec": _module_spec("numpy")},
            "panda3d": {"version": _distribution_version("Panda3D", "panda3d"), "spec": _module_spec("panda3d")},
            "torch": {"version": _distribution_version("torch"), "spec": _module_spec("torch")},
        },
    }


def checkpoint_metadata(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read SB3 ZIP metadata without loading policy tensors or creating an env."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {source}")
    if source.stat().st_size <= 0:
        raise OSError(f"checkpoint is empty: {source}")
    result: dict[str, Any] = {
        "path": str(source),
        "size_bytes": source.stat().st_size,
        "sha256": sha256_file(source),
        "format": "zip",
        "entries": [],
        "sb3_version": None,
        "data": {},
        "observation_shape": None,
        "action_space": None,
        "diagnostic_label": "legacy_raw_diagnostic",
    }
    try:
        archive = zipfile.ZipFile(source)
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError(f"checkpoint is not a readable SB3 zip: {source}: {error}") from error
    with archive:
        result["entries"] = sorted(archive.namelist())
        try:
            result["sb3_version"] = archive.read("_stable_baselines3_version").decode("utf-8").strip()
        except KeyError:
            pass
        try:
            data_raw = json.loads(archive.read("data").decode("utf-8"))
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError):
            data_raw = {}
        if isinstance(data_raw, dict):
            # Keep scalar metadata and type/shape hints, but avoid copying the
            # serialized last observation or model internals into a report.
            for key in ("num_timesteps", "_total_timesteps", "n_steps", "batch_size", "seed", "verbose"):
                if key in data_raw:
                    result["data"][key] = _json_safe(data_raw[key])
            observation = data_raw.get("observation_space")
            action = data_raw.get("action_space")
            result["observation_space_serialized"] = isinstance(observation, Mapping)
            result["action_space_serialized"] = isinstance(action, Mapping)
            # Some SB3 versions serialize useful repr fields directly.  Do not
            # claim a shape from an opaque cloudpickle payload.
            result["observation_shape"] = _shape_hint(observation)
            result["action_space"] = _action_hint(action)
    return result


def _shape_hint(value: Any) -> list[int] | None:
    if not isinstance(value, Mapping):
        return None
    shape = value.get("shape")
    if isinstance(shape, (list, tuple)) and all(isinstance(item, int) for item in shape):
        return list(shape)
    serialized = value.get(":serialized:")
    if not isinstance(serialized, str):
        return None
    try:
        operations = list(pickletools.genops(base64.b64decode(serialized, validate=True)))
    except (ValueError, TypeError, base64.binascii.Error):
        return None
    for index, (operation, argument, _position) in enumerate(operations):
        if operation.name not in {"SHORT_BINUNICODE", "BINUNICODE"} or argument != "_shape":
            continue
        for next_operation, next_argument, _next_position in operations[index + 1 : index + 5]:
            if next_operation.name in {"BININT1", "BININT2", "BININT", "LONG1", "LONG4"}:
                try:
                    return [int(next_argument)]
                except (TypeError, ValueError):
                    return None
    return None


def _action_hint(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    direct = value.get("n")
    if isinstance(direct, int) and not isinstance(direct, bool):
        return f"Discrete({direct})"
    serialized = value.get(":serialized:")
    if not isinstance(serialized, str):
        return None
    try:
        operations = list(pickletools.genops(base64.b64decode(serialized, validate=True)))
    except (ValueError, TypeError, base64.binascii.Error):
        return None
    found_n = False
    for operation, argument, _position in operations:
        if operation.name in {"SHORT_BINUNICODE", "BINUNICODE"} and argument == "n":
            found_n = True
            continue
        if found_n and operation.name in {"SHORT_BINBYTES", "BINBYTES", "BINBYTES8"}:
            if isinstance(argument, (bytes, bytearray)) and len(argument) in {1, 2, 4, 8}:
                value_int = int.from_bytes(argument, byteorder="little", signed=True)
                if value_int > 0:
                    return f"Discrete({value_int})"
            return None
    return None


def build_static_report(
    project_root: str | os.PathLike[str],
    *,
    profile: str | None = None,
    config_path: str | os.PathLike[str] | None = None,
    prefix_evidence: Any = None,
) -> dict[str, Any]:
    """Build the part of a doctor report that needs no environment creation."""

    root = Path(project_root).expanduser().resolve()
    report: dict[str, Any] = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "status": "static",
        "project_root": str(root),
        "project_git": git_state(root),
        "dependencies": dependency_audit(root),
        "assets": asset_audit(root),
        "cache_environment_before_probe": cache_environment(),
        "source_evidence": static_source_evidence(root),
        "raw_observation_layout": {
            "source": "metadrive/obs/state_obs.py:37-235",
            "segments": [
                {"indices": "0..1", "meaning": "left/right road-border distances or side detector values"},
                {"indices": "2..6", "meaning": "current reference heading/speed/steering/last normalized steering+throttle"},
                {"indices": "7", "meaning": "unsigned yaw-rate proxy from heading dot product"},
                {"indices": "8", "meaning": "current-lane signed lateral offset encoding"},
                {"indices": "9..18", "meaning": "two navigation checkpoints x five values"},
                {"indices": "19..258", "meaning": "240 configured lidar points"},
            ],
            "observation_side_effect": "LidarStateObservation.observe refreshes current_observation and lidar detections; gaussian/dropout noise uses NumPy random when configured.",
            "requested_prefix": "indices 259..261 are absent from the current raw host and are never synthesized here",
        },
        "configuration": {
            "profile": profile,
            "config_path": None if config_path is None else str(Path(config_path).expanduser().resolve()),
        },
        "prefix_feature_evidence_supplied": prefix_evidence is not None,
        "notes": [
            "Raw shape alone does not establish the requested 262-dimensional start-lane schema.",
            "Lookahead modes refuse a 259-wide host and a 262-wide host without semantic evidence for indices 259..261.",
            "No model, simulator, or environment is initialized by this static report.",
        ],
    }
    return report


__all__ = [
    "DIAGNOSTIC_SCHEMA_VERSION",
    "asset_audit",
    "build_static_report",
    "cache_environment",
    "checkpoint_metadata",
    "dependency_audit",
    "git_state",
    "read_json",
    "sha256_file",
    "static_source_evidence",
    "write_json",
]
