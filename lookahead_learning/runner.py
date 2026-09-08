"""Lazy command line runner for the forward-preview experiment.

The module is deliberately split into a standard-library command layer and
explicit runtime functions.  ``build_parser`` and ``main --help`` therefore
work on a machine without Panda3D, MetaDrive, Gymnasium, NumPy, or SB3.  A
runtime command creates a fresh output directory, redirects writable caches,
and only then imports the host adapter.

The checked-in host in this repository exposes raw ``(259,)`` observations.
The requested experiment requires a host supplied, semantically verified
``(262,)`` float32 observation.  ``train``, ``evaluate``, and wrapped
``compare`` refuse until that contract exists; the raw probe is intentionally
limited to ``doctor --probe`` and is labelled as a diagnostic.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import functools
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import types
import unittest
from typing import Any, Mapping, Sequence


PACKAGE_NAME = "lookahead_learning"
DEFAULT_LOOKAHEAD_M = 6.0
DEFAULT_PP_WEIGHT = 0.1
DEFAULT_PROBE_STEPS = 3
_CANONICAL_MODE_CHOICES = ("baseline", "lookahead_obs", "lookahead_obs_pp_reward")
_MODE_ALIASES = {"obs": "lookahead_obs", "obs_pp": "lookahead_obs_pp_reward"}
_MODE_CHOICES = _CANONICAL_MODE_CHOICES + tuple(_MODE_ALIASES)


def normalize_mode(value: str) -> str:
    """Normalize canonical and legacy mode spellings at the CLI boundary."""

    if not isinstance(value, str):
        raise ValueError(f"mode must be a string, found {type(value).__name__}")
    canonical = _MODE_ALIASES.get(value, value)
    if canonical not in _CANONICAL_MODE_CHOICES:
        choices = ", ".join(_CANONICAL_MODE_CHOICES)
        raise ValueError(f"unknown mode {value!r}; choose one of {choices}")
    return canonical

# The current host is raw 259 and therefore has no registered requested
# schema.  A future 262 host can be added here with complete source-backed
# evidence in this new module.  Self-reported environment attributes and CLI
# strings are intentionally never accepted as semantic proof.
_VERIFIED_HOST_PREFIX_EVIDENCE: dict[str, Any] = {}
# Shared finite bins keep comparisons aligned across seeds and modes.  Values
# outside the declared ranges remain visible in overall metrics and are
# reported as unmatched instead of silently placed in an edge bin.
ROAD_PROGRESS_INTERVALS = ((0.0, 25.0), (25.0, 50.0), (50.0, 100.0), (100.0, 200.0), (200.0, 500.0), (500.0, 1000.0))
SPEED_BINS = ((0.0, 1.0), (1.0, 3.0), (3.0, 6.0), (6.0, 12.0), (12.0, 30.0))

# These are the project files reached through the normal host import path and
# the add-on modules that can affect a saved model's construction, wrapper,
# reward, geometry, compatibility, or telemetry contract.  The package
# docstring-only ``__init__`` and all tests are deliberately excluded.
_HOST_RUNTIME_SOURCE_FILES = (
    "env_factory.py",
    "start_lane_env.py",
    "project_paths.py",
    "configs/experiment_config.py",
)
_ADDON_RUNTIME_SOURCE_FILES = (
    "lookahead_learning/__main__.py",
    "lookahead_learning/adapter.py",
    "lookahead_learning/checkpoint.py",
    "lookahead_learning/diagnostics.py",
    "lookahead_learning/env.py",
    "lookahead_learning/geometry.py",
    "lookahead_learning/portability.py",
    "lookahead_learning/runner.py",
    "lookahead_learning/telemetry.py",
)
_SOURCE_IDENTITIES_SCHEMA_VERSION = "lookahead_learning.source-identities.v1"


class RunnerError(RuntimeError):
    """Expected command error rendered without a traceback by the CLI."""


class ContractRefusal(RunnerError):
    """The selected host/checkpoint cannot satisfy the requested semantics."""


def _safe(value: Any) -> Any:
    """Convert common third-party scalars to strict JSON values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_safe(v) for v in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _safe(tolist())
        except Exception:
            pass
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _safe(item())
        except Exception:
            pass
    return repr(value)


def _json_dump(value: Any) -> str:
    return json.dumps(_safe(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _config_get(config: object, key: str, default: object = None) -> object:
    """Read a value from either a TOML mapping or MetaDrive ``Config``.

    MetaDrive 0.4.3's ``Config`` implements the mapping protocol incompletely:
    it supports subscription but does not register as ``collections.abc.Mapping``.
    Runtime audit code therefore must not use ``isinstance(config, Mapping)`` as
    the only way to read a configuration value.
    """

    if isinstance(config, Mapping):
        return config.get(key, default)
    try:
        return config[key]  # type: ignore[index]
    except (KeyError, IndexError, TypeError, AttributeError):
        return default


def _config_snapshot(config: object, *, keys: Sequence[str] | None = None) -> dict[str, Any]:
    """Copy selected config values without depending on MetaDrive Config ABCs."""

    if keys is None:
        if isinstance(config, Mapping):
            keys = tuple(str(key) for key in config)
        else:
            key_method = getattr(config, "keys", None)
            if not callable(key_method):
                return {}
            try:
                keys = tuple(str(key) for key in key_method())
            except (TypeError, AttributeError):
                return {}
    return {str(key): _safe(_config_get(config, str(key))) for key in keys}


def _config_hash(config: object) -> str | None:
    snapshot = _config_snapshot(config)
    if not snapshot:
        return None
    return hashlib.sha256(_json_dump(snapshot).encode("utf-8")).hexdigest()


def _required_source_sha256(root: Path, relative_path: str, *, label: str) -> str:
    """Hash one required runtime source while retaining a portable key."""

    root = root.expanduser().resolve()
    source = (root / relative_path).resolve()
    try:
        source.relative_to(root)
    except ValueError as error:
        raise ContractRefusal(
            f"required {label} source escapes project root: {relative_path}"
        ) from error
    if not source.is_file():
        raise ContractRefusal(f"required {label} source is missing: {relative_path}")
    digest = hashlib.sha256()
    try:
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ContractRefusal(
            f"required {label} source cannot be read: {relative_path}"
        ) from error
    return digest.hexdigest()


def _source_identities(root: Path) -> dict[str, Any]:
    """Build portable, strict identities for checkpoint compatibility.

    Checkpoint validation must reject a same-shape run when the imported host
    factory, task reward, or add-on wrapper implementation changed.  Only
    Only relative file names and hashes enter this mapping: checkout paths and
    dirty flags are useful audit context but cannot decide whether a copied
    experiment is semantically equivalent.
    """

    root = _resolve_root(root)
    host = {
        relative_path: _required_source_sha256(
            root,
            relative_path,
            label="host runtime",
        )
        for relative_path in _HOST_RUNTIME_SOURCE_FILES
    }
    addon = {
        relative_path: _required_source_sha256(
            root,
            relative_path,
            label="lookahead_learning runtime",
        )
        for relative_path in _ADDON_RUNTIME_SOURCE_FILES
    }

    return {
        "schema_version": _SOURCE_IDENTITIES_SCHEMA_VERSION,
        "host": host,
        "addon": addon,
    }


def _resolve_root(value: str | os.PathLike[str] | None) -> Path:
    root = Path(value or Path(__file__).resolve().parents[1]).expanduser().resolve()
    if not root.is_dir():
        raise RunnerError(f"project root is not a directory: {root}")
    return root


def _resolve_config_path(root: Path, profile: str | None, config: str | None) -> Path:
    if profile and config:
        raise RunnerError("--profile and --config cannot be used together")
    if config:
        candidate = Path(config).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
    else:
        candidate = root / "configs" / f"{profile or 'official'}.toml"
    candidate = candidate.resolve()
    if candidate.suffix != ".toml" or not candidate.is_file():
        raise RunnerError(f"configuration TOML does not exist: {candidate}")
    return candidate


def _fresh_run_dir(root: Path, requested: str | None, command: str) -> Path:
    """Return a new output directory and refuse all existing destinations."""

    if requested:
        path = Path(requested).expanduser()
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if path.exists():
            raise RunnerError(f"refusing to reuse existing output directory: {path}")
    else:
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        base = root / "outputs" / "lookahead_learning" / f"{command}-{stamp}-{os.getpid()}"
        path = base
        suffix = 0
        while path.exists():
            suffix += 1
            path = Path(f"{base}-{suffix}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def _prepare_runtime_cache(run_dir: Path, worker: int | None = None) -> dict[str, str]:
    """Redirect libraries that may write during import or simulator startup."""

    cache_root = run_dir / ".runtime_cache"
    if worker is not None:
        cache_root = cache_root / f"worker-{worker}"
    cache_root.mkdir(parents=True, exist_ok=True)
    values = {
        "MPLCONFIGDIR": str(cache_root / "matplotlib"),
        "XDG_CACHE_HOME": str(cache_root / "xdg"),
        "TMPDIR": str(cache_root / "tmp"),
        "TORCH_HOME": str(cache_root / "torch"),
    }
    for value in values.values():
        Path(value).mkdir(parents=True, exist_ok=True)
    os.environ.update(values)
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    return values


def _prepare_panda_cache(run_dir: Path, worker: int | None = None) -> dict[str, Any]:
    """Disable Panda model caches and redirect its writable cache directory.

    Panda3D is imported only by explicit simulator commands.  Static help and
    static doctor never call this function.  ``loadPrcFileData`` is Panda's
    public configuration API and is applied before any MetaDrive engine is
    constructed.
    """

    cache_root = run_dir / ".runtime_cache" / (f"worker-{worker}" if worker is not None else "parent") / "panda-model-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    try:
        from panda3d.core import loadPrcFileData

        loadPrcFileData("lookahead_learning", f"model-cache-dir {cache_root}")
        loadPrcFileData("lookahead_learning", "model-cache-models 0")
        loadPrcFileData("lookahead_learning", "model-cache-textures 0")
    except ImportError as error:
        raise RunnerError(f"Panda3D is required for simulator startup: {error}") from error
    return {
        "model_cache_dir": str(cache_root),
        "model_cache_models": False,
        "model_cache_textures": False,
        "configuration_api": "panda3d.core.loadPrcFileData",
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise RunnerError(f"refusing to overwrite output file: {path}")
    path.write_text(_json_dump(value), encoding="utf-8")


def _load_selection(root: Path, args: argparse.Namespace):
    """Import the local TOML helper only after a command selected runtime."""

    from .adapter import import_host_module

    module = import_host_module("configs.experiment_config", project_root=root)
    selector = getattr(module, "select_experiment", None)
    if not callable(selector):
        raise RunnerError("configs.experiment_config.select_experiment is unavailable")
    config_path = _resolve_config_path(root, getattr(args, "profile", None), getattr(args, "config", None))
    try:
        selection = selector(config_path=str(config_path))
    except Exception as error:
        raise RunnerError(f"failed to load experiment config {config_path}: {error}") from error
    return selection


def _config_for(selection: Any, stage: str) -> Mapping[str, object]:
    profile = selection.profile
    value = getattr(profile, f"{stage}_env_config", None)
    if not isinstance(value, Mapping):
        raise RunnerError(f"experiment selection has no {stage} environment config")
    return value


def _training_config(selection: Any) -> Mapping[str, object]:
    value = getattr(selection.profile, "training_config", None)
    if not isinstance(value, Mapping):
        raise RunnerError("experiment selection has no training config")
    return value


def _collect_feature_evidence(root: Path, raw_env: object | None = None) -> Any:
    """Return only source-reviewed evidence from the added-module registry."""

    del root
    if raw_env is None:
        return None
    identity = f"{type(raw_env).__module__}.{type(raw_env).__qualname__}"
    return _VERIFIED_HOST_PREFIX_EVIDENCE.get(identity)


def _host_metadata(root: Path) -> dict[str, Any]:
    from .diagnostics import asset_audit, dependency_audit, git_state, static_source_evidence

    return {
        "project_git": git_state(root),
        "dependencies": dependency_audit(root),
        "assets": asset_audit(root),
        "source_evidence": static_source_evidence(root),
    }


def _close_env(env: object | None) -> None:
    if env is None:
        return
    close = getattr(env, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _legacy_evaluate_raw(
    raw: object,
    checkpoint: Path,
    *,
    seed: int,
    max_steps: int,
) -> dict[str, Any]:
    """Run the explicitly labelled legacy 259 diagnostic on the raw env.

    This path is intentionally separate from operational evaluation.  It
    records the current selected TOML as context, but never claims that the
    old checkpoint was trained with that configuration.
    """

    if max_steps <= 0 or max_steps > 500:
        raise RunnerError("legacy diagnostic --max-steps must be between 1 and 500")
    from .diagnostics import checkpoint_metadata
    from .adapter import get_single_agent, read_applied_action, read_applied_steering, read_vehicle_state, simulation_dt_seconds

    report = checkpoint_metadata(checkpoint)
    report.update(
        {
            "diagnostic_label": "legacy_raw259_diagnostic",
            "training_config_provenance": "unverified",
            "evaluation_semantics": "current_selected_toml_raw_host",
            "deterministic": True,
            "scenario_seed": int(seed),
            "max_steps": int(max_steps),
            "steps": [],
        }
    )
    from stable_baselines3 import PPO

    model = PPO.load(str(checkpoint), device="cpu")
    model_shape = tuple(getattr(getattr(model, "observation_space", None), "shape", ()) or ())
    raw_shape = tuple(getattr(getattr(raw, "observation_space", None), "shape", ()) or ())
    if model_shape != raw_shape:
        raise ContractRefusal(
            f"legacy checkpoint observation shape {model_shape} does not match raw host {raw_shape}"
        )
    model_n = getattr(getattr(model, "action_space", None), "n", None)
    raw_n = getattr(getattr(raw, "action_space", None), "n", None)
    if model_n is not None and raw_n is not None and int(model_n) != int(raw_n):
        raise ContractRefusal("legacy checkpoint action space does not match raw host action space")
    observation, _info = raw.reset(seed=int(seed))
    terminated = truncated = False
    total_reward = 0.0
    dt = simulation_dt_seconds(raw)
    final_info: Mapping[str, Any] = {}
    for decision in range(1, int(max_steps) + 1):
        if terminated or truncated:
            break
        action_out, _state = model.predict(observation, deterministic=True)
        item = getattr(action_out, "item", None)
        action = item() if callable(item) else action_out
        before = read_vehicle_state(raw)
        observation, reward, terminated, truncated, info = raw.step(action)
        after = read_vehicle_state(raw)
        applied = read_applied_steering(raw)
        vehicle = get_single_agent(raw)
        current_action = getattr(vehicle, "current_action", None)
        # Keep the complete host info envelope, including reward breakdowns
        # and task-specific termination fields.  Values are copied at the
        # transition boundary so no later MetaDrive mutation changes the log.
        info_values = _safe(dict(info)) if isinstance(info, Mapping) else {}
        applied_pair = _safe(read_applied_action(raw))
        final_info = info_values
        total_reward += float(reward)
        report["steps"].append(
            {
                "decision": decision,
                "action_env": action,
                "action_applied_steering": applied,
                "action_applied_pair": applied_pair,
                "dt": dt,
                "t": (decision - 1) * dt,
                "t_next": decision * dt,
                "reward": float(reward),
                "info": info_values,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "before": before,
                "after": after,
                "info_keys": sorted(str(k) for k in info) if isinstance(info, Mapping) else None,
            }
        )
    report.update(
        {
            "observation_shape": list(raw_shape),
            "action_space": f"Discrete({raw_n})" if raw_n is not None else None,
            "steps_executed": len(report["steps"]),
            "total_reward": total_reward,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "incomplete": not (terminated or truncated),
            "final_info": final_info,
            "final_flags": {
                key: final_info.get(key)
                for key in (
                    "success",
                    "arrive_dest",
                    "in_target_lane",
                    "target_lane_valid",
                    "lane_departure",
                    "start_lane_departure",
                    "terminated",
                    "truncated",
                )
                if isinstance(final_info, Mapping) and key in final_info
            },
            "final_state": read_vehicle_state(raw),
        }
    )
    return report


def _audit_host(root: Path, config: Mapping[str, object], *, require_semantic: bool = True):
    """Create one raw env, audit it, and optionally return a strict contract."""

    from .adapter import HostAdapter, ObservationContract

    raw = None
    try:
        preliminary = HostAdapter.for_project(root)
        raw = preliminary.make_raw_env(config)
        audit = preliminary.audit_env(raw)
        evidence = _collect_feature_evidence(root, raw)
        if evidence is None:
            if require_semantic:
                raise ContractRefusal(
                    "operational modes require raw (262,) float32 plus explicit "
                    "host evidence for indices 259..261; found "
                    f"raw_shape={audit.raw_shape!r}, raw_dtype={audit.raw_dtype!r}; "
                    "no registered verified 262 implementation is available"
                )
            return raw, audit, None, None, None
        adapter = HostAdapter.for_project(root, semantic_evidence=evidence)
        contract, action, dt = adapter.inspect_env(raw)
        return raw, audit, adapter, (contract, action, dt), evidence
    except Exception:
        if raw is not None:
            _close_env(raw)
        raise


def _runtime_audit_details(raw: object) -> dict[str, Any]:
    """Collect bounded runtime facts without observing or stepping the env."""

    from .adapter import get_single_agent, simulation_dt_seconds

    config = getattr(raw, "config", {})
    relevant_keys = (
        "physics_world_step_size",
        "decision_repeat",
        "horizon",
        "num_agents",
        "discrete_action",
        "discrete_steering_dim",
        "discrete_throttle_dim",
        "use_multi_discrete",
        "is_multi_agent",
        "image_observation",
        "vehicle_config",
        "show_interface",
        "environment_num",
    )
    selected_config = {
        key: _safe(_config_get(config, key))
        for key in relevant_keys
        if _config_get(config, key, None) is not None
    }
    space = getattr(raw, "observation_space", None)
    action_space = getattr(raw, "action_space", None)
    try:
        import numpy as np

        low = np.asarray(getattr(space, "low"))
        high = np.asarray(getattr(space, "high"))
        bounds = {
            "shape": list(getattr(space, "shape", ()) or ()),
            "dtype": str(getattr(space, "dtype", None)),
            "low_min": float(np.min(low)) if low.size else None,
            "low_max": float(np.max(low)) if low.size else None,
            "high_min": float(np.min(high)) if high.size else None,
            "high_max": float(np.max(high)) if high.size else None,
        }
    except Exception as error:
        bounds = {"error": f"{type(error).__name__}: {error}"}
    vehicle = None
    try:
        vehicle = get_single_agent(raw)
    except Exception:
        pass
    details: dict[str, Any] = {
        "environment_config_relevant": selected_config,
        "observation_bounds_summary": bounds,
        "action_space": {
            "type": type(action_space).__name__,
            "n": getattr(action_space, "n", None),
            "shape": list(getattr(action_space, "shape", ()) or ()),
        },
        "simulation": {
            "dt_seconds": simulation_dt_seconds(raw),
            "physics_world_step_size": _safe(_config_get(config, "physics_world_step_size")),
            "decision_repeat": _safe(_config_get(config, "decision_repeat")),
        },
        "task": {
            "type": None if getattr(raw, "task", None) is None else f"{type(raw.task).__module__}.{type(raw.task).__qualname__}",
            "horizon": _safe(_config_get(config, "horizon")),
        },
        "vehicle": None,
        "policy": None,
        "action_path": {
            "external_actions_storage": "MetaDrive BaseEngine.external_actions -> VehicleAgentManager.try_actuate_agent",
            "applied_reader": "vehicle.current_action property (BaseVehicle source-backed)",
            "observation_action_history": "StateObservation.vehicle_state reads last_current_action[1]",
        },
    }
    if vehicle is not None:
        policy = None
        try:
            policy = _read_direct_policy(raw, vehicle)
        except ContractRefusal:
            # The bounded audit reports policy absence; Pure Pursuit's strict geometry
            # reader raises the same fact when the mode actually requires it.
            policy = None
        details["vehicle"] = {
            "class": f"{type(vehicle).__module__}.{type(vehicle).__qualname__}",
            "front_wheelbase_m": _safe(getattr(vehicle, "FRONT_WHEELBASE", None)),
            "rear_wheelbase_m": _safe(getattr(vehicle, "REAR_WHEELBASE", None)),
            "max_steering_deg": _safe(getattr(vehicle, "max_steering", None)),
        }
        details["policy"] = None if policy is None else f"{type(policy).__module__}.{type(policy).__qualname__}"
    return details


def _require_operational_contract(root: Path, config: Mapping[str, object], mode: str):
    from .adapter import UnsupportedHostError

    mode = normalize_mode(mode)
    if mode == "lookahead_obs_pp_reward":
        # Validate at the CLI boundary too, so accidental omission cannot turn
        # a requested PP run into a zero-weight observation run.
        pass
    raw, audit, adapter, contract_info, evidence = _audit_host(root, config, require_semantic=True)
    contract, action, dt = contract_info
    if tuple(contract.shape) != (262,) or str(contract.dtype) != "float32":
        _close_env(raw)
        raise UnsupportedHostError(
            f"requested {mode} host contract requires raw (262,) float32; "
            f"found {contract.shape} {contract.dtype}"
        )
    return raw, audit, adapter, contract, action, dt, evidence


def _read_direct_policy(env: object, vehicle: object) -> object:
    """Read the policy registered for the active vehicle through the engine."""

    engine = getattr(env, "engine", None)
    getter = getattr(engine, "get_policy", None)
    name = getattr(vehicle, "name", None)
    if not callable(getter) or name is None:
        raise ContractRefusal(
            "Pure Pursuit mode requires the direct engine policy lookup for the active vehicle"
        )
    policy = getter(name)
    if policy is None:
        raise ContractRefusal(f"Pure Pursuit mode found no engine policy for active vehicle {name!r}")
    return policy


def _read_vehicle_geometry(
    env: object,
    *,
    verify_direct_policy: bool = True,
) -> dict[str, float]:
    """Read source-audited axle geometry and, for Pure Pursuit, its direct policy path.

    The observation modes may report a vehicle whose PP geometry is unavailable.
    Pure Pursuit reward mode
    is deliberately narrower: only the audited ``DefaultVehicle`` and the
    exact ``EnvInputPolicy`` engine registration are accepted.  ``max_steering``
    is a degree value in this source path; units are never guessed by size.
    """

    from .adapter import get_single_agent

    vehicle = get_single_agent(env)
    module_name = type(vehicle).__module__
    qualname = type(vehicle).__qualname__
    if not module_name.startswith("metadrive."):
        raise ContractRefusal(
            "vehicle geometry reader is verified only for MetaDrive vehicle classes; "
            f"found {module_name}.{qualname}"
        )
    if verify_direct_policy and (
        module_name != "metadrive.component.vehicle.vehicle_type"
        or qualname != "DefaultVehicle"
    ):
        raise ContractRefusal(
            "Pure Pursuit vehicle geometry is source-audited only for "
            "metadrive.component.vehicle.vehicle_type.DefaultVehicle; "
            f"found {module_name}.{qualname}"
        )

    def number(name: str, *fallback: str) -> float:
        value = getattr(vehicle, name, None)
        if value is None:
            config = getattr(vehicle, "config", None)
            value = _config_get(config, name)
        if value is None:
            for alternate in fallback:
                value = getattr(vehicle, alternate, None)
                if value is not None:
                    break
        try:
            value = float(value)
        except (TypeError, ValueError) as error:
            raise ContractRefusal(f"vehicle geometry {name} is unavailable") from error
        if not math.isfinite(value) or value <= 0:
            raise ContractRefusal(f"vehicle geometry {name} must be positive")
        return value

    front = number("FRONT_WHEELBASE")
    rear = number("REAR_WHEELBASE")
    # MetaDrive's VehicleParameterSpace and BaseVehicle._set_action source
    # define ``max_steering`` in degrees; normalized steering is multiplied by
    # this value before Panda/Bullet receives it.
    max_deg = number("max_steering")
    if verify_direct_policy:
        policy = _read_direct_policy(env, vehicle)
        policy_path = f"{type(policy).__module__}.{type(policy).__qualname__}"
        if policy_path != "metadrive.policy.env_input_policy.EnvInputPolicy":
            raise ContractRefusal(
                "Pure Pursuit mode requires metadrive.policy.env_input_policy.EnvInputPolicy "
                f"through engine.get_policy; found {policy_path}"
            )
    return {
        "front_wheelbase_m": front,
        "rear_wheelbase_m": rear,
        "wheelbase_m": front + rear,
        "max_steering_deg": max_deg,
    }


def _target_ordinal_after_reset(env: object, vehicle: object) -> tuple[int, bool]:
    """Resolve the reset target ordinal and whether the host retained it."""

    from .adapter import _saved_start_lane_ordinal

    saved = _saved_start_lane_ordinal(env, vehicle)
    if saved is not None:
        return int(saved), True
    lane_index = getattr(vehicle, "lane_index", None)
    if lane_index is None:
        raise ContractRefusal("vehicle lane_index is unavailable after reset")
    try:
        value = lane_index[-1]
    except (IndexError, KeyError, TypeError) as error:
        raise ContractRefusal("vehicle lane_index has no reset target ordinal") from error
    if isinstance(value, bool):
        raise ContractRefusal("vehicle lane ordinal must be an integer")
    try:
        return int(value), False
    except (TypeError, ValueError) as error:
        raise ContractRefusal("vehicle lane ordinal must be an integer") from error


def _target_lane_state(
    env: object,
    vehicle: object,
    target_ordinal: int,
    *,
    project_root: str | Path | None = None,
) -> object | None:
    """Use the host target-lane resolver, with an explicit raw fallback."""

    del vehicle, target_ordinal
    from .adapter import HostImportError, HostContractError, UnsupportedHostError, read_target_lane_state

    try:
        return read_target_lane_state(env, project_root=project_root)
    except UnsupportedHostError:
        # A plain upstream 259 host has no reset-target map.  Its exact
        # current-ref ordinal is retained as a labelled diagnostic fallback;
        # no neighbouring lane is selected.
        return None
    except HostImportError:
        # A host checkout without the optional project target resolver can
        # still run the raw route diagnostic.  Malformed resolver code is not
        # caught here (HostContractError propagates).
        return None
    except HostContractError:
        raise


def _target_reference_present(vehicle: object, target_ordinal: int) -> bool:
    """Fallback validity for a plain upstream host without target-lane code."""

    navigation = getattr(vehicle, "navigation", None)
    references = getattr(navigation, "current_ref_lanes", None)
    if references is None:
        return False
    for lane in references:
        index = getattr(lane, "index", None)
        if index is None:
            continue
        try:
            ordinal = index[-1]
        except (IndexError, KeyError, TypeError):
            continue
        if isinstance(ordinal, bool):
            continue
        try:
            if int(ordinal) == target_ordinal:
                return True
        except (TypeError, ValueError):
            continue
    return False


class MetaDrivePreviewProvider:
    """Build and cache one fixed directed Navigation route per episode.

    ``project_root`` is retained for the target-lane reader only.  Supplying
    it makes a relocated runner verify that ``start_lane_env`` comes from the
    selected host checkout instead of accepting an already-imported module
    from another checkout.
    """

    def __init__(
        self,
        *,
        lookahead_m: float = DEFAULT_LOOKAHEAD_M,
        project_root: str | Path | None = None,
    ) -> None:
        self.lookahead_m = float(lookahead_m)
        self._project_root = (
            None
            if project_root is None
            else Path(project_root).expanduser().resolve()
        )
        self._route = None
        self._route_error: str | None = None
        self._episode_key: object = None
        self._target_ordinal: int | None = None
        self._target_ordinal_persistent = False

    def reset(self, env: object) -> None:
        self._route = None
        self._route_error = None
        self._episode_key = None
        self._target_ordinal = None
        self._target_ordinal_persistent = False
        from .adapter import UnsupportedHostError, build_fixed_navigation_route, get_single_agent, read_vehicle_state

        vehicle = get_single_agent(env)
        target_ordinal, persistent = _target_ordinal_after_reset(env, vehicle)
        # The route builder and the provider must use the same reset-time
        # ordinal.  It is a candidate key only; current lane identity is never
        # used to replace the fixed route after a checkpoint crossing.
        self._target_ordinal = target_ordinal
        self._target_ordinal_persistent = persistent
        self._route = build_fixed_navigation_route(env)
        state = read_vehicle_state(env)
        self._episode_key = tuple(state.get("checkpoints", ()))
        if not _target_reference_present(vehicle, target_ordinal):
            raise UnsupportedHostError(
                "reset target lane ordinal is absent from current Navigation references"
            )

    def __call__(self, env: object) -> Mapping[str, object]:
        from .adapter import get_single_agent, read_vehicle_state
        from .geometry import compute_preview

        state = read_vehicle_state(env)
        point = state.get("position_xy")
        psi = state.get("heading_theta")
        vehicle = get_single_agent(env)
        velocity = getattr(vehicle, "velocity", None)
        speed = state.get("speed_m_s")
        if velocity is not None and psi is not None:
            try:
                speed = float(velocity[0]) * math.cos(float(psi)) + float(velocity[1]) * math.sin(float(psi))
            except (TypeError, ValueError, IndexError):
                speed = state.get("speed_m_s")
        current_checkpoints = tuple(state.get("checkpoints", ()))
        if self._route is None:
            return {
                "preview_valid": False,
                "x_g_m": 0.0,
                "y_g_m": 0.0,
                "invalid_reason": self._route_error or "route_unavailable",
                "details": {"route_error": self._route_error},
            }
        if self._episode_key is not None and current_checkpoints != self._episode_key:
            return {
                "preview_valid": False,
                "x_g_m": 0.0,
                "y_g_m": 0.0,
                "invalid_reason": "navigation_route_changed",
                "details": {"reset_checkpoints": self._episode_key, "current_checkpoints": current_checkpoints},
            }
        assert self._target_ordinal is not None
        if self._target_ordinal_persistent:
            observed_target, observed_persistent = _target_ordinal_after_reset(env, vehicle)
            if not observed_persistent or observed_target != self._target_ordinal:
                return {
                    "preview_valid": False,
                    "x_g_m": 0.0,
                    "y_g_m": 0.0,
                    "invalid_reason": "start_lane_reference_changed",
                    "details": {
                        "reset_target_ordinal": self._target_ordinal,
                        "observed_target_ordinal": observed_target,
                    },
                }
        target_state = _target_lane_state(
            env,
            vehicle,
            self._target_ordinal,
            project_root=self._project_root,
        )
        if target_state is None:
            start_lane_valid = _target_reference_present(vehicle, self._target_ordinal)
            target_state_fields: dict[str, object] = {
                "target_lane_state_source": "navigation.current_ref_lanes.ordinal_presence",
                "target_lane_valid": start_lane_valid,
            }
        else:
            start_lane_valid = bool(getattr(target_state, "valid"))
            target_state_fields = {
                "target_lane_state_source": "start_lane_env.resolve_target_lane_state",
                "target_lane_valid": start_lane_valid,
                "target_lane_ordinal": getattr(target_state, "target_ordinal", None),
                "current_lane_ordinal": getattr(target_state, "current_ordinal", None),
                "target_lane_offset_m": getattr(target_state, "target_lane_offset_m", None),
                "target_lane_normalized_error": getattr(target_state, "normalized_error", None),
                "in_target_lane": getattr(target_state, "in_target_lane", None),
                "target_lane_departed": getattr(target_state, "departed", None),
            }
        if point is None or psi is None:
            return {
                "preview_valid": False,
                "x_g_m": 0.0,
                "y_g_m": 0.0,
                "invalid_reason": "vehicle_geometry_unavailable",
            }
        result = compute_preview(
            self._route.path,
            point,
            psi,
            lookahead_m=self.lookahead_m,
            forward_speed_mps=speed,
            start_lane_valid=start_lane_valid,
        )
        details: dict[str, object] = {
            "p_xy": point,
            "psi_rad": psi,
            "route": self._route.diagnostics.as_dict(),
            "fixed_route_lane_ids": [item.lane_id for item in self._route.path.metadata],
            "fixed_route_lane_types": [item.lane_type for item in self._route.path.metadata],
            "projection": None if result.projection is None else result.projection.as_dict(),
            "heading_error_rad": result.heading_error_rad,
            "distance_to_goal_m": result.distance_to_goal_m,
        }
        details.update(target_state_fields)
        return {
            "preview_valid": bool(result.valid),
            "x_g_m": result.x_g,
            "y_g_m": result.y_g,
            "q_xy": result.q,
            "s_proj_m": result.s_proj,
            "s_goal_m": result.s_goal,
            "projected_lane_id": result.projected_lane_index,
            "goal_lane_id": result.goal_lane_index,
            "invalid_reason": result.reason,
            "x_clipped": result.x_clipped,
            "y_clipped": result.y_clipped,
            "details": details,
        }


class MetaDrivePPProvider:
    """Compute PP from the same preview point and rear axle geometry."""

    def __init__(self, *, steering_sign: float) -> None:
        if float(steering_sign) not in (-1.0, 1.0):
            raise ValueError("steering_sign must be +1 or -1")
        self.steering_sign = float(steering_sign)
        self.geometry: dict[str, float] | None = None

    def reset(self, env: object) -> None:
        self.geometry = _read_vehicle_geometry(env)

    def __call__(self, env: object, preview: object) -> Mapping[str, object]:
        from .adapter import get_single_agent, read_vehicle_state
        from .geometry import pure_pursuit_from_rear_coordinates

        if not getattr(preview, "preview_valid", False) or getattr(preview, "q_xy", None) is None:
            return {"pp_valid": False, "u_pp": None, "invalid_reason": "preview_invalid"}
        if self.geometry is None:
            self.geometry = _read_vehicle_geometry(env)
        details = dict(getattr(preview, "details", ()))
        point = details.get("p_xy")
        psi = details.get("psi_rad")
        q = getattr(preview, "q_xy", None)
        if point is None or psi is None or q is None:
            return {"pp_valid": False, "u_pp": None, "invalid_reason": "vehicle_geometry_unavailable"}
        forward = (math.cos(float(psi)), math.sin(float(psi)))
        rear = (
            float(point[0]) - self.geometry["rear_wheelbase_m"] * forward[0],
            float(point[1]) - self.geometry["rear_wheelbase_m"] * forward[1],
        )
        left = (-math.sin(float(psi)), math.cos(float(psi)))
        delta = (float(q[0]) - rear[0], float(q[1]) - rear[1])
        x_rear = delta[0] * forward[0] + delta[1] * forward[1]
        y_rear = delta[0] * left[0] + delta[1] * left[1]
        result = pure_pursuit_from_rear_coordinates(
            x_rear,
            y_rear,
            wheelbase_m=self.geometry["wheelbase_m"],
            max_steering_deg=self.geometry["max_steering_deg"],
            steering_sign=self.steering_sign,
            q=q,
            rear_position=rear,
            rear_wheelbase_m=self.geometry["rear_wheelbase_m"],
        )
        return {
            "pp_valid": bool(result.valid),
            "u_pp": result.u_pp,
            "u_pp_unclipped": result.u_pp_unclipped,
            "x_rear_m": result.x_rear,
            "y_rear_m": result.y_rear,
            "kappa_pp": result.kappa_pp,
            "delta_pp_rad": result.delta_pp_rad,
            "saturated": result.saturated,
            "invalid_reason": result.reason,
            "details": {
                "wheelbase_m": self.geometry["wheelbase_m"],
                "rear_wheelbase_m": self.geometry["rear_wheelbase_m"],
                "max_steering_deg": self.geometry["max_steering_deg"],
                "steering_sign": self.steering_sign,
            },
        }


def _state_reader(env: object) -> Mapping[str, object]:
    from .adapter import read_vehicle_state

    return read_vehicle_state(env)


def _make_wrapped_env(
    raw: object,
    *,
    project_root: str | Path,
    mode: str,
    contract: object,
    pp_weight: float | None,
    steering_sign: float | None,
    run_id: str,
    monitor_path: Path | None = None,
) -> object:
    from .adapter import read_applied_action, read_applied_steering, simulation_dt_seconds
    from .env import make_preview_env

    mode = normalize_mode(mode)
    # All modes use the same fixed-route provider.  LookaheadEnv preserves the
    # raw 262-wide array and host reward in baseline while retaining this
    # shared geometry in telemetry.
    provider = MetaDrivePreviewProvider(project_root=project_root)
    pp = None
    if mode == "lookahead_obs_pp_reward":
        pp = MetaDrivePPProvider(steering_sign=1.0 if steering_sign is None else steering_sign)
    wrapped = make_preview_env(
        raw,
        mode=mode,
        contract=contract,
        preview_provider=provider,
        pp_provider=pp,
        pp_weight=pp_weight,
        applied_action_reader=read_applied_action,
        applied_steering_reader=read_applied_steering,
        dt_reader=simulation_dt_seconds,
        state_reader=_state_reader,
        run_id=run_id,
    )
    if monitor_path is not None:
        try:
            from stable_baselines3.common.monitor import Monitor

            monitor_path.parent.mkdir(parents=True, exist_ok=True)
            wrapped = Monitor(wrapped, filename=str(monitor_path))
        except Exception:
            _close_env(wrapped)
            raise
    return wrapped


@dataclass(frozen=True)
class WorkerSpec:
    project_root: str
    env_config: dict[str, object]
    mode: str
    evidence: Any
    pp_weight: float
    steering_sign: float | None
    run_id: str
    output_dir: str
    worker: int
    learning_seed: int = 0


def _worker_entry(spec: WorkerSpec) -> object:
    """Top-level spawn target; every worker gets an isolated cache path."""

    _prepare_runtime_cache(Path(spec.output_dir), spec.worker)
    _prepare_panda_cache(Path(spec.output_dir), spec.worker)
    from .adapter import HostAdapter

    adapter = HostAdapter.for_project(spec.project_root, semantic_evidence=spec.evidence)
    raw = None
    try:
        raw = adapter.make_raw_env(spec.env_config)
        contract, _action, _dt = adapter.inspect_env(raw)
        wrapped = _make_wrapped_env(
            raw,
            project_root=spec.project_root,
            mode=spec.mode,
            contract=contract,
            pp_weight=spec.pp_weight,
            steering_sign=spec.steering_sign,
            run_id=spec.run_id,
            monitor_path=Path(spec.output_dir) / "monitor" / f"worker-{spec.worker}.monitor.csv",
        )
        worker_seed = int(spec.learning_seed) + int(spec.worker)
        wrapped.action_space.seed(worker_seed)
        wrapped.observation_space.seed(worker_seed)
        return wrapped
    except Exception:
        _close_env(raw)
        raise


def make_training_worker(spec: WorkerSpec):
    """Return a picklable spawn factory for ``SubprocVecEnv``."""

    return functools.partial(_worker_entry, spec)


def make_transition_logging_callback(
    *,
    base_callback_class: type,
    output_dir: Path,
    worker_count: int,
    metadata: Mapping[str, Any],
    learning_seed: int,
):
    """Build an opt-in SB3 callback without importing SB3 at module import.

    SB3 supplies already sampled rollout actions and info mappings in
    ``self.locals``.  The callback only copies those values to one JSONL file
    per worker; it never calls ``predict``, changes actions, or touches the
    rollout buffer.
    """

    from .telemetry import JsonlWorkerWriter

    class TransitionLoggingCallback(base_callback_class):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.writers = [
                JsonlWorkerWriter(
                    output_dir / "training_telemetry",
                    metadata.get("run_id", "run"),
                    worker,
                    metadata={**dict(metadata), "worker": worker},
                )
                for worker in range(worker_count)
            ]
            self.episode_indices = [0] * worker_count
            self.decision_indices = [0] * worker_count

        @staticmethod
        def _worker_action(actions: Any, worker: int) -> Any:
            try:
                return actions[worker]
            except (IndexError, KeyError, TypeError):
                return actions

        @staticmethod
        def _scenario_seed(info: Mapping[str, Any]) -> Any:
            namespace = info.get("lookahead_learning")
            if not isinstance(namespace, Mapping):
                return None
            state = namespace.get("state_after")
            if isinstance(state, Mapping):
                return state.get("scenario_seed")
            return None

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", ())
            actions = self.locals.get("actions")
            dones = self.locals.get("dones", ())
            if not isinstance(infos, (list, tuple)):
                infos = (infos,)
            if not isinstance(dones, (list, tuple)):
                try:
                    dones = tuple(dones)
                except TypeError:
                    dones = (dones,)
            for worker, info in enumerate(infos[:worker_count]):
                if not isinstance(info, Mapping):
                    continue
                action = self._worker_action(actions, worker)
                namespace = info.get("lookahead_learning")
                row = dict(info)
                if isinstance(namespace, Mapping):
                    row.update(namespace)
                self.decision_indices[worker] += 1
                row.update(
                    {
                        "run": metadata.get("run_id", "run"),
                        "worker": worker,
                        "episode": self.episode_indices[worker],
                        "decision": self.decision_indices[worker],
                        "learning_seed": learning_seed,
                        "scenario_seed": self._scenario_seed(info),
                        "policy_action_raw": _safe(action),
                        "policy_action_stage": "sb3_rollout_action_before_environment_clipping",
                    }
                )
                self.writers[worker].write(row)
                # SB3 may autoreset a VecEnv before the callback sees the
                # info mapping.  ``dones`` is the authoritative rollout end
                # flag; explicit namespace flags remain a useful fallback for
                # fake/test environments that omit self.locals.dones.
                done_from_vec = False
                if worker < len(dones):
                    done_from_vec = bool(dones[worker])
                namespace_done = isinstance(namespace, Mapping) and bool(
                    namespace.get("terminated", False)
                    or namespace.get("truncated", False)
                    or namespace.get("episode_end", False)
                )
                if done_from_vec or namespace_done:
                    self.episode_indices[worker] += 1
                    self.decision_indices[worker] = 0
            return True

        def close_writers(self) -> None:
            for writer in self.writers:
                writer.close()

        def _on_training_end(self) -> None:
            self.close_writers()

    return TransitionLoggingCallback()


def _preflight_asset_update(root: Path) -> dict[str, Any]:
    from .diagnostics import asset_audit

    assets = asset_audit(root)
    if assets.get("normal_engine_try_pull_asset_would_update"):
        raise RunnerError(
            "refusing simulator startup because the installed MetaDrive asset "
            "directory is missing/stale; doctor reports the normal pull decision "
            "would update assets"
        )
    return assets


def _runtime_vehicle_metadata(
    raw: object,
    *,
    require_pp: bool,
) -> tuple[dict[str, float] | None, dict[str, Any]]:
    """Describe the reset vehicle; only Pure Pursuit reward requires PP geometry."""

    from .adapter import get_single_agent

    vehicle = get_single_agent(raw)
    vehicle_class = f"{type(vehicle).__module__}.{type(vehicle).__qualname__}"
    geometry: dict[str, float] | None = None
    geometry_error: str | None = None
    try:
        geometry = _read_vehicle_geometry(raw, verify_direct_policy=require_pp)
    except ContractRefusal as error:
        if require_pp:
            raise
        geometry_error = str(error)
    policy_class = None
    policy_error = None
    try:
        policy = _read_direct_policy(raw, vehicle)
        policy_class = f"{type(policy).__module__}.{type(policy).__qualname__}"
    except ContractRefusal as error:
        policy_error = str(error)
    description: dict[str, Any] = {
        "class": vehicle_class,
        "policy_class": policy_class,
        "policy_path_verified": policy_class == "metadrive.policy.env_input_policy.EnvInputPolicy",
        "pp_geometry_verified": geometry is not None and not bool(geometry_error),
    }
    if geometry is not None:
        description.update(geometry)
    if geometry_error is not None:
        description["pp_geometry_error"] = geometry_error
    if policy_error is not None:
        description["policy_error"] = policy_error
    # Keep a small, stable subset of the actual vehicle Config for metadata;
    # opaque Panda objects and process-specific handles are deliberately omitted.
    vehicle_config = getattr(vehicle, "config", None)
    selected = _config_snapshot(
        vehicle_config,
        keys=(
            "vehicle_model",
            "max_steering",
        ),
    )
    if selected:
        description["config"] = selected
    return geometry, description


def _observation_schema(contract: Any, *, mode: str) -> dict[str, Any]:
    """Build the semantic 262 prefix plus the exact lookahead suffix contract."""

    mode = normalize_mode(mode)
    raw_shape = list(getattr(contract, "shape", ()))
    raw_dtype = str(getattr(contract, "dtype", None))
    prefix: dict[str, Any] = {
        "shape": raw_shape,
        "dtype": raw_dtype,
        "low": _safe(getattr(contract, "low", None)),
        "high": _safe(getattr(contract, "high", None)),
        "index_range": [0, 261],
        "semantic_source": "verified host observation contract",
    }
    extra = [] if mode == "baseline" else [
        "preview_x_norm",
        "preview_y_norm",
        "preview_valid",
    ]
    return {
        "shape": raw_shape if mode == "baseline" else [265],
        "dtype": raw_dtype,
        "prefix": prefix,
        "extra_fields": extra,
    }


def _runtime_control_dt(raw: object, decision_dt: float) -> float | None:
    """Derive the physics control step from the instantiated host config."""

    value = _config_get(getattr(raw, "config", None), "physics_world_step_size")
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RunnerError("physics_world_step_size must be numeric") from error
    if not math.isfinite(result) or result <= 0.0:
        raise RunnerError("physics_world_step_size must be positive")
    return result


def _evaluation_scenario_seeds(
    config: Mapping[str, object],
    explicit: Sequence[int] | None,
) -> list[int]:
    """Resolve explicit scenario IDs or the configured contiguous range."""

    if explicit:
        return [int(value) for value in explicit]
    try:
        start = int(_config_get(config, "start_seed"))
        count = int(_config_get(config, "num_scenarios"))
    except (TypeError, ValueError) as error:
        raise RunnerError("evaluation config must provide start_seed and num_scenarios") from error
    if count <= 0:
        raise RunnerError("evaluation config num_scenarios must be positive")
    return [start + index for index in range(count)]


def _metadata_base(
    root: Path,
    selection: Any,
    *,
    mode: str,
    pp_weight: float,
    steering_sign: float | None,
    contract: Any,
    action: Any,
    dt: float,
    evidence: Any,
    command: str,
    run_dir: Path,
    env_config: Mapping[str, object] | None = None,
    vehicle_geometry: Mapping[str, Any] | None = None,
    vehicle_config: Mapping[str, Any] | None = None,
    learning_seed: int | None = None,
    evaluation_seed: int | None = None,
    scenario_seeds: Sequence[int] | None = None,
    learning_budget: Mapping[str, Any] | None = None,
    control_dt: float | None = None,
) -> dict[str, Any]:
    mode = normalize_mode(mode)
    metadata = _host_metadata(root)
    source_identities = _source_identities(root)
    env_config_value = dict(env_config or {})
    schema = _observation_schema(contract, mode=mode)
    action_metadata = {
        "type": "Discrete",
        "n": int(action.n),
        "steering_dim": int(action.steering_dim),
        "throttle_dim": int(action.throttle_dim),
        "use_multi_discrete": bool(action.use_multi_discrete),
        "source": action.source,
    }
    budget = dict(learning_budget or {})
    if "requested_timesteps" not in budget:
        budget["requested_timesteps"] = None
    if "actual_timesteps" not in budget:
        budget["actual_timesteps"] = None
    if "num_envs" not in budget:
        budget["num_envs"] = None
    if "n_steps" not in budget:
        budget["n_steps"] = None
    normalization = {
        "preview_distance_m": 10.0,
        "invalid_preview": [0.5, 0.5, 0.0],
        "raw_prefix_unchanged": True,
    }
    # ``wrapper_order`` describes the canonical training/evaluation contract;
    # ``runtime_wrapper_order`` records whether this particular command also
    # instantiated a VecEnv.  Keeping the logical contract stable lets a
    # checkpoint trained through VecEnv be evaluated in a single env.
    wrapper_order = ["raw_host", "lookahead_learning", "Monitor", "VecEnv"]
    runtime_wrapper_order = (
        list(wrapper_order)
        if command == "train"
        else ["raw_host", "lookahead_learning", "Monitor"]
    )
    metadata.update(
        {
            "schema_version": "lookahead_learning.runner.v1",
            "command": command,
            "run_id": run_dir.name,
            "mode": mode,
            "pp_weight": pp_weight,
            "steering_sign": steering_sign,
            "raw_observation_shape": list(contract.shape),
            "raw_observation_dtype": str(contract.dtype),
            "observation_shape": list(contract.shape) if mode == "baseline" else [265],
            "observation_schema": schema,
            "action": action_metadata,
            "action_space": action_metadata,
            "dt_seconds": dt,
            "dt": dt,
            "decision_dt": dt,
            "control_dt": control_dt,
            "prefix_feature_evidence": _safe(evidence),
            "road_progress_intervals": ROAD_PROGRESS_INTERVALS,
            "speed_bins": SPEED_BINS,
            "normalization": normalization,
            "normalization_config": normalization,
            "wrapper_order": wrapper_order,
            "runtime_wrapper_order": runtime_wrapper_order,
            "config": selection.source_metadata(),
            "scenario_config": _safe(env_config_value),
            "effective_env_config": _safe(env_config_value),
            "config_hash": _config_hash(env_config_value),
            "source_identities": source_identities,
            "vehicle_config": _safe(vehicle_config),
            "vehicle": _safe(vehicle_config),
            "vehicle_geometry": _safe(vehicle_geometry),
            "learning_budget": _safe(budget),
            "learning_timesteps": budget.get("actual_timesteps", budget.get("requested_timesteps")),
            "train_timesteps": budget.get("actual_timesteps", budget.get("requested_timesteps")),
            "learning_seed": learning_seed,
            "evaluation_seed": evaluation_seed,
            "scenario_seeds": list(scenario_seeds or []),
            "cache_environment": _prepare_runtime_cache(run_dir),
        }
    )
    return metadata


def command_doctor(args: argparse.Namespace) -> int:
    root = _resolve_root(args.project_root)
    runtime_requested = bool(args.output or args.probe or args.pulse or args.checkpoint or getattr(args, "legacy_evaluate", False))
    run_dir = _fresh_run_dir(root, args.output, "doctor") if runtime_requested else None
    if run_dir is not None:
        _prepare_runtime_cache(run_dir)
    config_path = None
    try:
        config_path = _resolve_config_path(root, args.profile, args.config)
    except RunnerError:
        if args.probe or args.pulse or getattr(args, "legacy_evaluate", False):
            raise
    from .diagnostics import build_static_report, checkpoint_metadata, write_json

    report = build_static_report(root, profile=args.profile, config_path=config_path)
    if args.checkpoint:
        checkpoint = Path(args.checkpoint).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = root / checkpoint
        report["legacy_checkpoint"] = checkpoint_metadata(checkpoint)
        report["legacy_checkpoint"]["training_config_provenance"] = "unverified"
        report["legacy_checkpoint"]["diagnostic_label"] = "legacy_raw259_diagnostic"
    if args.probe or args.pulse or getattr(args, "legacy_evaluate", False):
        _preflight_asset_update(root)
        _prepare_runtime_cache(run_dir or root / "outputs" / "lookahead_learning")
        panda_cache_policy = _prepare_panda_cache(run_dir or root / "outputs" / "lookahead_learning")
        selection = _load_selection(root, args)
        config = _config_for(selection, "evaluation")
        from .adapter import HostAdapter, read_applied_steering, read_vehicle_state

        raw = None
        try:
            adapter = HostAdapter.for_project(root)
            raw = adapter.make_raw_env(config)
            audit = adapter.audit_env(raw)
            report["runtime_audit"] = audit.as_dict()
            report["cache_policy"] = panda_cache_policy
            report["runtime_audit"]["semantic_evidence"] = bool(_collect_feature_evidence(root, raw))
            report["runtime_audit"]["operational_modes_supported"] = bool(
                audit.supported_baseline_schema and report["runtime_audit"]["semantic_evidence"]
            )
            if args.probe or args.pulse:
                seed = int(args.seed)
                obs, info = raw.reset(seed=seed)
                report["runtime_audit"].update(_runtime_audit_details(raw))
                preview_provider = MetaDrivePreviewProvider(project_root=root)
                preview_provider.reset(raw)
                preview_value = preview_provider(raw)
                from .env import PPReference, PreviewState

                preview_state = PreviewState.from_object(preview_value)
                pp_value = None
                if preview_state.preview_valid:
                    pp_provider = MetaDrivePPProvider(steering_sign=1.0)
                    pp_value = PPReference.from_object(
                        pp_provider(raw, preview_state),
                        preview=preview_state,
                    )
                report["probe"] = {
                    "diagnostic_label": "legacy_raw259_diagnostic" if tuple(getattr(obs, "shape", ())) == (259,) else "raw_host_probe",
                    "seed": seed,
                    "reset_observation_shape": list(getattr(obs, "shape", ())),
                    "reset_observation_dtype": str(getattr(obs, "dtype", None)),
                    "reset_info_keys": sorted(str(k) for k in info) if isinstance(info, Mapping) else None,
                    "geometry_preview": preview_state.as_dict(),
                    "pp_reference": None if pp_value is None else pp_value.as_dict(),
                    "steps": [],
                }
                action = 4
                for index in range(max(1, min(int(args.steps), 10))):
                    before = read_vehicle_state(raw)
                    pre_value = preview_provider(raw)
                    next_obs, reward, terminated, truncated, step_info = raw.step(action)
                    after = read_vehicle_state(raw)
                    post_value = preview_provider(raw)
                    report["probe"]["steps"].append(
                        {
                            "decision": index + 1,
                            "action_env": action,
                            "action_applied_steering": read_applied_steering(raw),
                            "reward": float(reward),
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                            "observation_shape": list(getattr(next_obs, "shape", ())),
                            "before": before,
                            "after": after,
                            "preview_before": PreviewState.from_object(pre_value).as_dict(),
                            "preview_after": PreviewState.from_object(post_value).as_dict(),
                            "info_keys": sorted(str(k) for k in step_info) if isinstance(step_info, Mapping) else None,
                        }
                    )
                    if terminated or truncated:
                        break
                if args.pulse:
                    from .adapter import ActionContract

                    action_contract = ActionContract.from_space(raw.action_space, raw.config)
                    pulses = []
                    # Canonical 3x3 IDs are selected from the verified action
                    # contract; no raw action values are guessed for other n.
                    if action_contract.steering_dim >= 3 and action_contract.throttle_dim >= 3:
                        straight = action_contract.steering_dim // 2
                        throttle = action_contract.throttle_dim - 1
                        for steer_name, steer_index in (("left", action_contract.steering_dim - 1), ("right", 0)):
                            pulse_action = throttle * action_contract.steering_dim + steer_index
                            raw.reset(seed=seed)
                            for _ in range(5):
                                raw.step(throttle * action_contract.steering_dim + straight)
                            before = read_vehicle_state(raw)
                            pulse_preview_before = preview_provider(raw)
                            raw.step(pulse_action)
                            after = read_vehicle_state(raw)
                            pulse_preview_after = preview_provider(raw)
                            heading_before = before.get("heading_theta")
                            heading_after = after.get("heading_theta")
                            if heading_before is None or heading_after is None:
                                yaw_delta = None
                            else:
                                from .geometry import wrap_to_pi

                                yaw_delta = wrap_to_pi(float(heading_after) - float(heading_before))
                            pulses.append({
                                "name": "positive_command" if steer_index == action_contract.steering_dim - 1 else "negative_command",
                                "action_env": pulse_action,
                                "decoded": action_contract.decode(pulse_action),
                                "applied_steering": read_applied_steering(raw),
                                "yaw_delta_rad": yaw_delta,
                                "speed_before_m_s": before.get("speed_m_s"),
                                "speed_after_m_s": after.get("speed_m_s"),
                                "before": before,
                                "after": after,
                                "preview_before": PreviewState.from_object(pulse_preview_before).as_dict(),
                                "preview_after": PreviewState.from_object(pulse_preview_after).as_dict(),
                            })
                    report["probe"]["pulses"] = pulses
                    nonzero = [item for item in pulses if isinstance(item.get("yaw_delta_rad"), (int, float)) and abs(float(item["yaw_delta_rad"])) > 1.0e-9]
                    if nonzero:
                        positive = next((item for item in pulses if item["name"] == "positive_command"), None)
                        if positive and isinstance(positive.get("yaw_delta_rad"), (int, float)):
                            report["probe"]["steering_sign_measured"] = 1 if float(positive["yaw_delta_rad"]) > 0.0 else -1
            if getattr(args, "legacy_evaluate", False):
                if not args.checkpoint:
                    raise RunnerError("--legacy-evaluate requires --checkpoint")
                checkpoint = Path(args.checkpoint).expanduser()
                if not checkpoint.is_absolute():
                    checkpoint = root / checkpoint
                report["legacy_evaluation"] = _legacy_evaluate_raw(
                    raw,
                    checkpoint.resolve(),
                    seed=int(args.seed),
                    max_steps=int(getattr(args, "max_steps", 500)),
                )
        finally:
            _close_env(raw)
    if run_dir is None:
        print(json.dumps(_safe(report), ensure_ascii=False, indent=2, sort_keys=True))
    else:
        write_json(run_dir / "doctor.json", report)
        print(json.dumps({"status": "ok", "output": str(run_dir / "doctor.json")}, ensure_ascii=False))
    return 0


def _validate_mode_args(args: argparse.Namespace) -> tuple[float, float | None]:
    mode = normalize_mode(args.mode)
    args.mode = mode
    weight = getattr(args, "pp_weight", None)
    if mode == "lookahead_obs_pp_reward":
        if weight is None:
            raise RunnerError("lookahead_obs_pp_reward requires --pp-weight, including zero")
        weight = float(weight)
        if not math.isfinite(weight) or weight < 0:
            raise RunnerError("--pp-weight must be finite and non-negative")
        sign = getattr(args, "steering_sign", None)
        # The audited EnvInputPolicy -> DefaultVehicle path and the isolated
        # pulse evidence establish +1.  Keep the CLI optional and reject a
        # contrary override until another source-backed transform is audited.
        sign = 1.0 if sign is None else float(sign)
        if sign != 1.0:
            raise RunnerError(
                "the audited MetaDrive steering transform has sign +1; "
                "a -1 override requires separate source evidence"
            )
        return weight, sign
    if weight is not None:
        weight = float(weight)
        if not math.isfinite(weight) or weight < 0:
            raise RunnerError("--pp-weight must be finite and non-negative")
    return 0.0, getattr(args, "steering_sign", None)


def command_train(args: argparse.Namespace) -> int:
    root = _resolve_root(args.project_root)
    weight, sign = _validate_mode_args(args)
    run_dir = _fresh_run_dir(root, args.output, "train")
    _prepare_runtime_cache(run_dir)
    _preflight_asset_update(root)
    panda_cache_policy = _prepare_panda_cache(run_dir)
    selection = _load_selection(root, args)
    config = dict(_config_for(selection, "train"))
    training = dict(_training_config(selection))
    # ``0`` is an invalid training budget, so an explicit zero must not be
    # mistaken for an omitted CLI value through ``or`` fallback.
    timesteps = int(
        training["total_timesteps"] if args.timesteps is None else args.timesteps
    )
    n_envs = int(training["num_envs"] if args.num_envs is None else args.num_envs)
    n_steps = int(training["n_steps"] if args.n_steps is None else args.n_steps)
    if timesteps <= 0 or n_envs <= 0 or n_steps <= 0:
        raise RunnerError("--timesteps, --num-envs, and --n-steps must be positive")
    learning_seed = int(args.seed if args.seed is not None else training["seed"])
    scenario_seed = int(_config_get(config, "start_seed", learning_seed))
    scenario_seeds = [scenario_seed + index for index in range(int(_config_get(config, "num_scenarios", 1)))]
    raw, _audit, adapter, contract, action, dt, evidence = _require_operational_contract(root, config, args.mode)
    try:
        # The raw factory exposes spaces before reset, while vehicle/policy
        # objects are created by MetaDrive only during reset.  Audit geometry
        # in that order and close this parent probe before spawn workers.
        raw.reset(seed=scenario_seed)
        vehicle_geometry, vehicle_config = _runtime_vehicle_metadata(
            raw,
            require_pp=args.mode == "lookahead_obs_pp_reward",
        )
        control_dt = _runtime_control_dt(raw, dt)
    finally:
        _close_env(raw)
    metadata = _metadata_base(
        root,
        selection,
        mode=args.mode,
        pp_weight=weight,
        steering_sign=sign,
        contract=contract,
        action=action,
        dt=dt,
        evidence=evidence,
        command="train",
        run_dir=run_dir,
        env_config=config,
        vehicle_geometry=vehicle_geometry,
        vehicle_config=vehicle_config,
        learning_seed=learning_seed,
        scenario_seeds=scenario_seeds,
        learning_budget={
            "requested_timesteps": timesteps,
            "actual_timesteps": None,
            "num_envs": n_envs,
            "n_steps": n_steps,
        },
        control_dt=control_dt,
    )
    metadata["panda_cache_policy"] = panda_cache_policy
    metadata["resolved_ppo_config"] = {
        key: _safe(training.get(key))
        for key in ("policy", "learning_rate", "batch_size", "n_epochs", "gamma", "gae_lambda", "clip_range", "normalize_advantage", "ent_coef", "vf_coef", "max_grad_norm", "device", "log_interval")
        if key in training
    }
    try:
        from .adapter import import_host_module

        config_module = import_host_module("configs.experiment_config", project_root=root)
        normalize_model_name = getattr(config_module, "normalize_model_name")
        model_name = str(normalize_model_name(args.model_name or training.get("model_name") or selection.profile.default_model_name, "--model-name"))
    except Exception as error:
        raise RunnerError(f"invalid model name: {error}") from error
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.utils import set_random_seed
        from stable_baselines3.common.vec_env import SubprocVecEnv
        from stable_baselines3.common.callbacks import BaseCallback

        seed = learning_seed
        set_random_seed(seed)
        specs = [WorkerSpec(str(root), dict(config), args.mode, evidence, weight, sign, run_dir.name, str(run_dir), rank, seed) for rank in range(n_envs)]
        vec = SubprocVecEnv([make_training_worker(spec) for spec in specs], start_method="spawn")
        transition_callback = None
        try:
            common = {key: training[key] for key in ("learning_rate", "batch_size", "n_epochs", "gamma", "gae_lambda", "clip_range", "normalize_advantage", "ent_coef", "vf_coef", "max_grad_norm") if key in training}
            policy = args.policy or str(training.get("policy", "MlpPolicy"))
            # RL seed is set explicitly through SB3/space seeding above; it is
            # deliberately not forwarded to PPO's constructor as a second
            # seed source.
            model = PPO(policy, vec, n_steps=n_steps, verbose=1, device=args.device or training.get("device", "auto"), **common)
            if args.log_transitions:
                transition_callback = make_transition_logging_callback(
                    base_callback_class=BaseCallback,
                    output_dir=run_dir,
                    worker_count=n_envs,
                    metadata={**metadata, "transition_logging": True},
                    learning_seed=seed,
                )
            model.learn(
                total_timesteps=timesteps,
                log_interval=int(training.get("log_interval", 1)),
                callback=transition_callback,
            )
            destination = run_dir / f"{model_name}.zip"
            model.save(str(destination.with_suffix("")))
            reloaded = PPO.load(str(destination), env=vec, device=args.device or training.get("device", "auto"))
            if tuple(getattr(reloaded.observation_space, "shape", ()) or ()) != tuple(getattr(vec.observation_space, "shape", ()) or ()):
                raise RunnerError("saved PPO checkpoint failed observation-space reload verification")
            if int(getattr(reloaded.action_space, "n", -1)) != int(action.n):
                raise RunnerError("saved PPO checkpoint failed action-space reload verification")
            metadata.update({"requested_timesteps": timesteps, "actual_timesteps": int(model.num_timesteps), "seed": seed, "num_envs": n_envs, "n_steps": n_steps, "transition_logging": bool(args.log_transitions), "model_path": str(destination), "model_sha256": hashlib.sha256(destination.read_bytes()).hexdigest() if destination.exists() else None})
            metadata["learning_budget"] = {
                **dict(metadata.get("learning_budget", {})),
                "requested_timesteps": timesteps,
                "actual_timesteps": int(model.num_timesteps),
                "num_envs": n_envs,
                "n_steps": n_steps,
            }
            metadata["learning_timesteps"] = int(model.num_timesteps)
            metadata["train_timesteps"] = int(model.num_timesteps)
            _write_json(run_dir / "metadata.json", metadata)
        finally:
            if transition_callback is not None:
                transition_callback.close_writers()
            vec.close()
    except Exception:
        _close_env(raw)
        raise
    print(json.dumps({"status": "ok", "output": str(run_dir), "model": str(run_dir / f"{model_name}.zip")}, ensure_ascii=False))
    return 0


def _checkpoint_path(root: Path, args: argparse.Namespace) -> Path:
    value = getattr(args, "checkpoint", None) or getattr(args, "model", None)
    if not value:
        raise RunnerError("evaluate requires --checkpoint")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file():
        raise RunnerError(f"checkpoint does not exist: {path}")
    return path


def _saved_checkpoint_metadata(
    checkpoint: Path,
    *,
    root: Path,
    selection: Any,
    mode: str,
    contract: Any,
    action: Any,
    dt: float,
    vehicle_geometry: Mapping[str, Any] | None,
    vehicle_config: Mapping[str, Any],
    pp_weight: float,
    steering_sign: float | None,
    evidence: Any,
    control_dt: float | None,
) -> dict[str, Any]:
    """Require the sidecar contract written by this runner before PPO.load."""

    mode = normalize_mode(mode)
    sidecar = checkpoint.parent / "metadata.json"
    if not sidecar.is_file():
        raise ContractRefusal(
            f"checkpoint has no verified metadata sidecar: {sidecar}; "
            "legacy/raw259 ZIPs are accepted only by doctor --legacy-evaluate"
        )
    try:
        value = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractRefusal(f"checkpoint metadata is unreadable: {sidecar}") from error
    if not isinstance(value, Mapping):
        raise ContractRefusal(f"checkpoint metadata must be an object: {sidecar}")
    saved_mode = value.get("mode")
    if isinstance(saved_mode, str):
        try:
            saved_mode = normalize_mode(saved_mode)
        except ValueError:
            pass
    if saved_mode != mode:
        raise ContractRefusal(f"checkpoint metadata mode={saved_mode!r} does not match requested mode={mode!r}")
    budget = value.get("learning_budget")
    if not isinstance(budget, Mapping):
        raise ContractRefusal("checkpoint metadata has no learning budget")
    actual_budget = budget.get("actual_timesteps", value.get("actual_timesteps"))
    requested_budget = budget.get("requested_timesteps", value.get("requested_timesteps"))
    try:
        if int(actual_budget) <= 0 or int(requested_budget) <= 0:
            raise ContractRefusal("checkpoint metadata learning budget must be positive")
        int(value["seed"])
    except (KeyError, TypeError, ValueError):
        raise ContractRefusal("checkpoint metadata has no valid learning seed/budget") from None

    action_metadata = {
        "type": "Discrete",
        "n": int(action.n),
        "steering_dim": int(action.steering_dim),
        "throttle_dim": int(action.throttle_dim),
        "use_multi_discrete": bool(action.use_multi_discrete),
        "source": action.source,
    }
    expected: dict[str, Any] = {
        "mode": mode,
        "raw_observation_shape": list(contract.shape),
        "raw_observation_dtype": str(contract.dtype),
        "observation_shape": list(contract.shape) if mode == "baseline" else [265],
        "observation_schema": _observation_schema(contract, mode=mode),
        "action": action_metadata,
        "action_space": action_metadata,
        "dt_seconds": float(dt),
        "dt": float(dt),
        "decision_dt": float(dt),
        "control_dt": control_dt,
        "wrapper_order": ["raw_host", "lookahead_learning", "Monitor", "VecEnv"],
        "normalization": {
            "preview_distance_m": 10.0,
            "invalid_preview": [0.5, 0.5, 0.0],
            "raw_prefix_unchanged": True,
        },
        "normalization_config": {
            "preview_distance_m": 10.0,
            "invalid_preview": [0.5, 0.5, 0.0],
            "raw_prefix_unchanged": True,
        },
        "vehicle_config": _safe(vehicle_config),
        "vehicle_geometry": _safe(vehicle_geometry),
        "config": {"sha256": getattr(selection, "source_sha256", None)},
        "source_identities": _source_identities(root),
        "pp_weight": float(pp_weight),
        "steering_sign": steering_sign,
    }
    if evidence is not None:
        expected["prefix_feature_evidence"] = _safe(evidence)
    try:
        from .checkpoint import CheckpointContractError, validate_checkpoint_metadata

        validate_checkpoint_metadata(checkpoint, dict(value), expected)
    except CheckpointContractError as error:
        raise ContractRefusal(f"checkpoint semantic contract failed: {error}") from error
    return dict(value)


def _write_transition(writer: Any, *, run: str, worker: int, episode: int, decision: int, learning_seed: int, scenario_seed: int, evaluation_seed: int | None, deterministic: bool, action: Any, info: Mapping[str, Any], reward: float, terminated: bool, truncated: bool) -> None:
    namespace = info.get("lookahead_learning") if isinstance(info.get("lookahead_learning"), Mapping) else {}
    # Keep the complete host info envelope.  The namespaced wrapper fields are
    # flattened only as convenient aliases; no host reward/status key is lost.
    row = dict(info)
    row.update(namespace)
    row.update({"run": run, "worker": worker, "episode": episode, "decision": decision, "learning_seed": learning_seed, "scenario_seed": scenario_seed, "evaluation_seed": evaluation_seed, "deterministic": deterministic, "action_env": action, "r_total": reward, "terminated": terminated, "truncated": truncated, "episode_end": terminated or truncated})
    writer.write(row)


def command_evaluate(args: argparse.Namespace) -> int:
    root = _resolve_root(args.project_root)
    weight, sign = _validate_mode_args(args)
    run_dir = _fresh_run_dir(root, args.output, "evaluate")
    _prepare_runtime_cache(run_dir)
    _preflight_asset_update(root)
    panda_cache_policy = _prepare_panda_cache(run_dir)
    selection = _load_selection(root, args)
    config = dict(_config_for(selection, "evaluation"))
    checkpoint = _checkpoint_path(root, args)
    scenarios = _evaluation_scenario_seeds(config, args.scenario_seeds)
    evaluation_defaults = getattr(selection.profile, "evaluation_defaults", {})
    configured_evaluation_seed = (
        evaluation_defaults.get("seed", 0)
        if isinstance(evaluation_defaults, Mapping)
        else 0
    )
    evaluation_seed = int(
        args.seed if args.seed is not None else configured_evaluation_seed
    )
    if int(args.max_steps) <= 0:
        raise RunnerError("--max-steps must be positive")
    raw, _audit, adapter, contract, action, dt, evidence = _require_operational_contract(root, config, args.mode)
    from .diagnostics import checkpoint_metadata
    checkpoint_report = checkpoint_metadata(checkpoint)
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.utils import set_random_seed

        # Evaluation RNG and scenario selection are separate from the learning
        # seed stored in the checkpoint.  Seed spaces before the audit reset so
        # any policy/space sampling is reproducible without changing scenario IDs.
        set_random_seed(evaluation_seed)
        raw.action_space.seed(evaluation_seed)
        raw.observation_space.seed(evaluation_seed)
        raw.reset(seed=scenarios[0])
        vehicle_geometry, vehicle_config = _runtime_vehicle_metadata(
            raw,
            require_pp=args.mode == "lookahead_obs_pp_reward",
        )
        control_dt = _runtime_control_dt(raw, dt)
        saved_metadata = _saved_checkpoint_metadata(
            checkpoint,
            root=root,
            selection=selection,
            mode=args.mode,
            contract=contract,
            action=action,
            dt=dt,
            vehicle_geometry=vehicle_geometry,
            vehicle_config=vehicle_config,
            pp_weight=weight,
            steering_sign=sign,
            evidence=evidence,
            control_dt=control_dt,
        )
        try:
            learning_seed = int(saved_metadata["seed"])
        except (KeyError, TypeError, ValueError):
            raise ContractRefusal("checkpoint metadata has no valid learning seed") from None
        model = PPO.load(str(checkpoint), device=args.device or "cpu")
        expected = 262 if args.mode == "baseline" else 265
        actual_shape = tuple(getattr(getattr(model, "observation_space", None), "shape", ()) or ())
        if actual_shape != (expected,):
            raise ContractRefusal(f"checkpoint observation shape {actual_shape} is incompatible with mode={args.mode} expected ({expected},); raw259 legacy checkpoints are diagnostic only")
        if getattr(model, "action_space", None) is not None and int(getattr(model.action_space, "n", -1)) != action.n:
            raise ContractRefusal("checkpoint action space does not match selected host action contract")
        wrapped = _make_wrapped_env(
            raw,
            project_root=root,
            mode=args.mode,
            contract=contract,
            pp_weight=weight,
            steering_sign=sign,
            run_id=run_dir.name,
            monitor_path=run_dir / "monitor" / "evaluation.monitor.csv",
        )
        from .telemetry import JsonlWorkerWriter, summarize_jsonl
        saved_budget = saved_metadata.get("learning_budget")
        metadata = _metadata_base(
            root,
            selection,
            mode=args.mode,
            pp_weight=weight,
            steering_sign=sign,
            contract=contract,
            action=action,
            dt=dt,
            evidence=evidence,
            command="evaluate",
            run_dir=run_dir,
            env_config=config,
            vehicle_geometry=vehicle_geometry,
            vehicle_config=vehicle_config,
            learning_seed=learning_seed,
            evaluation_seed=evaluation_seed,
            scenario_seeds=scenarios,
            learning_budget=saved_budget if isinstance(saved_budget, Mapping) else None,
            control_dt=control_dt,
        )
        metadata.update({"checkpoint": checkpoint_report, "saved_checkpoint_metadata": saved_metadata, "deterministic": bool(args.deterministic), "checkpoint_training_config_provenance": "verified_sidecar"})
        metadata["panda_cache_policy"] = panda_cache_policy
        metadata["evaluation_seed"] = evaluation_seed
        metadata["scenario_seeds"] = list(scenarios)
        writer = JsonlWorkerWriter(run_dir, run_dir.name, 0, metadata=metadata)
        episodes = []
        try:
            for episode, scenario_seed in enumerate(scenarios):
                observation, reset_info = wrapped.reset(seed=scenario_seed)
                terminated = truncated = False
                decision = 0
                while not (terminated or truncated) and decision < int(args.max_steps):
                    action_out, _state = model.predict(observation, deterministic=bool(args.deterministic))
                    action_value = int(getattr(action_out, "item", lambda: action_out)())
                    observation, reward, terminated, truncated, info = wrapped.step(action_value)
                    _write_transition(writer, run=run_dir.name, worker=0, episode=episode, decision=decision + 1, learning_seed=learning_seed, scenario_seed=scenario_seed, evaluation_seed=evaluation_seed, deterministic=bool(args.deterministic), action=action_value, info=info, reward=float(reward), terminated=bool(terminated), truncated=bool(truncated))
                    decision += 1
                episodes.append({"episode": episode, "scenario_seed": scenario_seed, "decisions": decision, "terminated": bool(terminated), "truncated": bool(truncated), "totals": getattr(wrapped, "episode_totals", {})})
        finally:
            writer.close()
            _close_env(wrapped)
        summary = summarize_jsonl(
            writer.path,
            metadata={
                **metadata,
                "episode_cases": [f"{learning_seed}:{scenario}" for scenario in scenarios],
                "road_progress_intervals": ROAD_PROGRESS_INTERVALS,
                "speed_bins": SPEED_BINS,
            },
            road_progress_intervals=ROAD_PROGRESS_INTERVALS,
            speed_bins=SPEED_BINS,
        )
        summary["metadata"] = {**(summary.get("metadata", {}) if isinstance(summary.get("metadata"), Mapping) else {}), **metadata}
        summary["evaluation_episodes"] = episodes
        _write_json(run_dir / "summary.json", summary)
        print(json.dumps({"status": "ok", "output": str(run_dir), "summary": str(run_dir / "summary.json")}, ensure_ascii=False))
        return 0
    finally:
        _close_env(raw)


def _load_input(path: Path) -> Mapping[str, Any] | list[dict[str, Any]]:
    if path.is_dir():
        json_candidates = sorted(path.glob("summary*.json"))
        if json_candidates:
            from .diagnostics import read_json
            return read_json(json_candidates[0])
        jsonl = sorted(path.glob("*.jsonl"))
        if not jsonl:
            raise RunnerError(f"no summary.json or JSONL files in {path}")
        from .telemetry import read_jsonl
        return read_jsonl(jsonl)
    if path.suffix == ".jsonl":
        from .telemetry import read_jsonl
        return read_jsonl(path)
    from .diagnostics import read_json
    return read_json(path)


def command_compare(args: argparse.Namespace) -> int:
    root = _resolve_root(args.project_root)
    before = Path(args.before).expanduser()
    after = Path(args.after).expanduser()
    if not before.is_absolute():
        before = root / before
    if not after.is_absolute():
        after = root / after
    before_data = _load_input(before.resolve())
    after_data = _load_input(after.resolve())
    from .telemetry import compare_runs
    result = compare_runs(before_data, after_data, before_mode=args.before_mode, after_mode=args.after_mode, strict_metadata=bool(args.strict_metadata), require_matched=not bool(args.allow_unmatched))
    if args.output:
        out = _fresh_run_dir(root, args.output, "compare")
        _write_json(out / "comparison.json", result)
        print(json.dumps({"status": "ok", "output": str(out / "comparison.json")}, ensure_ascii=False))
    else:
        print(json.dumps(_safe(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def command_test(args: argparse.Namespace) -> int:
    root = _resolve_root(args.project_root)
    if args.integration:
        # Integration is an explicit doctor probe and remains raw diagnostic;
        # it does not pretend that the current host is a 262 baseline.
        doctor_args = argparse.Namespace(**vars(args))
        doctor_args.probe = True
        doctor_args.pulse = False
        doctor_args.checkpoint = None
        doctor_args.output = args.output
        return command_doctor(doctor_args)
    if args.portability:
        run_dir = _fresh_run_dir(root, args.output, "test")
        _prepare_runtime_cache(run_dir)
        from .portability import run_portability

        config_path = _resolve_config_path(root, args.profile, args.config)
        report = run_portability(
            root,
            run_dir / "portability",
            config_path,
            probe=False,
        )
        _write_json(run_dir / "portability_result.json", report)
        print(json.dumps({"status": "ok" if report.get("ok") else "failed", "output": str(run_dir), "report": str(run_dir / "portability_result.json")}, ensure_ascii=False))
        return 0 if report.get("ok") else 1
    test_run_dir = None
    if args.output:
        test_run_dir = _fresh_run_dir(root, args.output, "test")
        _prepare_runtime_cache(test_run_dir)
    suite = unittest.defaultTestLoader.loadTestsFromName("lookahead_learning.tests")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def _add_root_config_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", default=None, help="host checkout root")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--profile", choices=("official", "generalization"), default=None)
    group.add_argument("--config", "--base-config", dest="config", default=None, help="explicit experiment TOML")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lookahead_learning", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="static dependency audit and opt-in raw diagnostic")
    _add_root_config_options(doctor)
    doctor.add_argument("--output", default=None)
    doctor.add_argument("--probe", action="store_true", help="explicitly reset/step the raw host")
    doctor.add_argument("--pulse", action="store_true", help="include short steering sign pulses")
    doctor.add_argument("--checkpoint", "--legacy-checkpoint", default=None, help="read legacy SB3 ZIP metadata only")
    doctor.add_argument("--legacy-evaluate", action="store_true", help="explicit deterministic raw-checkpoint diagnostic (259 may be reported)")
    doctor.add_argument("--seed", type=int, default=5)
    doctor.add_argument("--steps", type=int, default=DEFAULT_PROBE_STEPS)
    doctor.add_argument("--max-steps", type=int, default=500)
    doctor.set_defaults(handler=command_doctor)

    for name in ("train", "evaluate"):
        command = sub.add_parser(name, help=f"{name} an explicitly verified 262-wide host")
        _add_root_config_options(command)
        command.add_argument("--output", default=None)
        command.add_argument(
            "--mode",
            choices=_MODE_CHOICES,
            type=normalize_mode,
            default="baseline",
            help=(
                "実験モード / experiment mode: baseline (既存観測), "
                "lookahead_obs (前方注視点入力追加), or "
                "lookahead_obs_pp_reward (入力追加 + Pure Pursuit 操舵不一致ペナルティ). "
                "Legacy aliases: obs -> lookahead_obs, obs_pp -> lookahead_obs_pp_reward."
            ),
        )
        command.add_argument("--pp-weight", type=float, default=None)
        command.add_argument("--steering-sign", type=float, choices=(-1.0, 1.0), default=None)
        command.add_argument("--seed", type=int, default=None)
        command.add_argument("--device", default=None)
        if name == "train":
            command.add_argument("--timesteps", type=int, default=None)
            command.add_argument("--num-envs", type=int, default=None)
            command.add_argument("--n-steps", type=int, default=None)
            command.add_argument("--policy", default=None)
            command.add_argument("--model-name", default=None)
            command.add_argument("--log-transitions", action="store_true", help="write opt-in per-worker rollout JSONL diagnostics")
        else:
            command.add_argument("--checkpoint", "--model", default=None)
            command.add_argument("--scenario-seeds", type=int, nargs="+")
            command.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
            command.add_argument("--max-steps", type=int, default=500)
        command.set_defaults(handler=command_train if name == "train" else command_evaluate)

    compare = sub.add_parser(
        "compare",
        help="compare telemetry summaries with strict metadata gates",
    )
    compare.add_argument("before")
    compare.add_argument("after")
    compare.add_argument("--project-root", default=None)
    compare.add_argument(
        "--before-mode",
        choices=_MODE_CHOICES,
        type=normalize_mode,
        default=None,
        help="canonical mode for the before summary; obs and obs_pp are legacy aliases",
    )
    compare.add_argument(
        "--after-mode",
        choices=_MODE_CHOICES,
        type=normalize_mode,
        default=None,
        help="canonical mode for the after summary; obs and obs_pp are legacy aliases",
    )
    compare.add_argument("--strict-metadata", action=argparse.BooleanOptionalAction, default=True)
    compare.add_argument("--allow-unmatched", action="store_true")
    compare.add_argument("--output", default=None)
    compare.set_defaults(handler=command_compare)

    test = sub.add_parser("test", help="run pure unit tests; integration is opt-in")
    test.add_argument("--project-root", default=None)
    test.add_argument("--integration", action="store_true")
    test.add_argument("--portability", action="store_true")
    test.add_argument("--output", default=None)
    test.add_argument("--profile", default=None)
    test.add_argument("--config", "--base-config", dest="config", default=None)
    test.add_argument("--seed", type=int, default=5)
    test.add_argument("--steps", type=int, default=1)
    test.set_defaults(handler=command_test)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (RunnerError, ValueError, FileNotFoundError, OSError) as error:
        print(f"lookahead_learning: {error}", file=sys.stderr)
        return 2


__all__ = [
    "ContractRefusal",
    "MetaDrivePPProvider",
    "MetaDrivePreviewProvider",
    "RunnerError",
    "WorkerSpec",
    "build_parser",
    "main",
    "make_training_worker",
]
