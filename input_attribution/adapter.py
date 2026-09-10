"""Small, explicit adapter between the host MetaDrive project and analysis.

The analysis package works at the policy-input boundary.  This module keeps
the host-specific pieces in one place and deliberately does not import
MetaDrive, Stable-Baselines3, Torch, or Pillow at module import time.  The
official host currently supplies a 259-wide ``float32`` Box observation and a
single ``Discrete(9)`` action space.  The dimension is discovered from the
model (and optionally checked against a schema) rather than hard-coded, so a
verified port can use another dimension without padding or truncation.

The supported path is intentionally narrow:

* one-dimensional ``gymnasium.spaces.Box`` observations;
* an identity raw-observation to policy-input boundary;
* SB3's MLP policy with a ``FlattenExtractor``;
* one-agent ``gymnasium.spaces.Discrete`` actions with ``start=0``.

Interventions are applied by the caller to a copy returned by
:meth:`prepare`.  ``predict`` never clips or repairs values, including the
explicit stress values ``-1`` and ``-100``.  A read-only forward pre-hook on
the policy's MLP verifies that the exact array reaches the model.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
import copy
import hashlib
import importlib
import importlib.util
from importlib import metadata as importlib_metadata
import inspect
import math
import os
import random
from pathlib import Path
import platform
import sys
from types import ModuleType
from typing import Any, Iterator, Final

import numpy as np


DEFAULT_POLICY_SEED: Final[int] = 0
DEFAULT_RENDER_SIZE: Final[tuple[int, int]] = (600, 600)
SUPPORTED_PREPROCESSING: Final[str] = "identity"


class AdapterContractError(ValueError):
    """The host, model, schema, or policy input contract is invalid."""


class UnsupportedAdapterError(AdapterContractError):
    """The requested model/environment path is outside the verified scope."""


class ModelInputBoundaryError(AdapterContractError):
    """The read-only MLP probe observed an input different from the request."""


class AdapterDependencyError(AdapterContractError):
    """A lazily required third-party dependency is unavailable."""


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise AdapterContractError(f"{name} must be a finite number, not bool")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise AdapterContractError(f"{name} must be a finite number") from error
    if not math.isfinite(number):
        raise AdapterContractError(f"{name} must be finite")
    return number


def _safe_attr(value: object, name: str) -> object | None:
    """Read a property without allowing one unavailable metric to abort a run."""

    try:
        return getattr(value, name)
    except Exception:
        return None


def _mapping_view(value: object) -> Mapping[str, object] | None:
    """Expose plain mappings and MetaDrive's read-only ``Config`` alike."""

    if isinstance(value, Mapping):
        return value
    get_dict = getattr(value, "get_dict", None)
    if callable(get_dict):
        try:
            converted = get_dict()
        except Exception:
            return None
        if isinstance(converted, Mapping):
            return converted
    return None


def _jsonable(value: object) -> object:
    """Convert bounded state/metadata values to JSON-compatible values.

    State readers should retain missing values as ``None``.  Unknown host
    objects are represented by their qualified type name instead of invoking
    arbitrary methods or serializing mutable internals.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return f"<{type(value).__module__}.{type(value).__qualname__}>"


def _module_path(module_name: str) -> str | None:
    module = sys.modules.get(module_name)
    if module is not None:
        origin = getattr(module, "__file__", None)
        if origin:
            return str(Path(origin).resolve())
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError, ValueError):
        spec = None
    origin = None if spec is None else spec.origin
    if origin in {None, "built-in", "frozen"}:
        return None
    try:
        return str(Path(origin).resolve())
    except (OSError, TypeError, ValueError):
        return str(origin)


def _distribution_info(distribution: str, module_name: str) -> dict[str, str | None]:
    try:
        version = importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        version = None
    except Exception as error:  # pragma: no cover - defensive metadata path
        version = f"metadata_error:{type(error).__name__}"
    return {"version": version, "path": _module_path(module_name)}


def _numpy_rng_snapshot() -> tuple[object, ...]:
    state = np.random.get_state()
    return (
        state[0],
        np.array(state[1], dtype=np.uint32, copy=True),
        int(state[2]),
        int(state[3]),
        float(state[4]),
    )


def _torch_rng_snapshot() -> bytes | None:
    """Read Torch's CPU RNG state without importing Torch for frame-only use."""

    torch = sys.modules.get("torch")
    if torch is None:
        return None
    get_rng_state = getattr(torch, "get_rng_state", None)
    if not callable(get_rng_state):
        return None
    try:
        state = get_rng_state()
        detach = getattr(state, "detach", None)
        if callable(detach):
            state = detach()
        cpu = getattr(state, "cpu", None)
        if callable(cpu):
            state = cpu()
        to_bytes = getattr(state, "numpy", None)
        if not callable(to_bytes):
            return None
        return bytes(np.asarray(to_bytes(), dtype=np.uint8).tobytes())
    except Exception:  # pragma: no cover - defensive optional Torch state
        return None


def _rng_snapshot() -> tuple[object, tuple[object, ...], bytes | None]:
    return random.getstate(), _numpy_rng_snapshot(), _torch_rng_snapshot()


def _rng_unchanged(
    before: tuple[object, tuple[object, ...], bytes | None],
    after: tuple[object, tuple[object, ...], bytes | None],
) -> bool:
    before_py, before_numpy, before_torch = before
    after_py, after_numpy, after_torch = after
    if before_py != after_py or before_torch != after_torch:
        return False
    if before_numpy[0] != after_numpy[0]:
        return False
    if not np.array_equal(before_numpy[1], after_numpy[1]):
        return False
    return before_numpy[2:] == after_numpy[2:]


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as file_obj:
        while chunk := file_obj.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _space_shape(space: object) -> tuple[int, ...]:
    value = getattr(space, "shape", None)
    if value is None:
        return ()
    try:
        return tuple(int(item) for item in value)
    except (TypeError, ValueError) as error:
        raise AdapterContractError(f"space shape is invalid: {value!r}") from error


def _schema_dimension(schema: object | None) -> int | None:
    """Read an explicit schema dimension without importing the schema module."""

    if schema is None:
        return None
    candidates: list[object] = []
    if isinstance(schema, Mapping):
        for key in ("dimension", "observation_dim", "feature_count", "D"):
            if key in schema:
                candidates.append(schema[key])
        nested = schema.get("observation")
        if isinstance(nested, Mapping):
            for key in ("dimension", "observation_dim", "feature_count", "D"):
                if key in nested:
                    candidates.append(nested[key])
    elif isinstance(schema, Sequence) and not isinstance(schema, (str, bytes)):
        # ``schema.load_schema`` returns the validated per-index rows rather
        # than its document wrapper.  A complete ordered row list is an
        # explicit schema: its length is the declared width, while an
        # arbitrary list is rejected below instead of being treated as D.
        rows = list(schema)
        if not rows or not all(isinstance(row, Mapping) for row in rows):
            raise AdapterContractError(
                "schema rows must be a non-empty sequence of mappings"
            )
        if not all("index" in row for row in rows):
            raise AdapterContractError(
                "schema rows must expose an explicit zero-based index"
            )
        indices = [row["index"] for row in rows]
        if indices != list(range(len(rows))):
            raise AdapterContractError(
                "schema rows must cover indices 0..dimension-1 in order"
            )
        candidates.append(len(rows))
    else:
        for key in ("dimension", "observation_dim", "feature_count", "D"):
            value = _safe_attr(schema, key)
            if value is not None:
                candidates.append(value)
    if not candidates:
        raise AdapterContractError(
            "schema must expose an explicit dimension (dimension, "
            "observation_dim, or feature_count)"
        )
    value = candidates[0]
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise AdapterContractError(f"schema dimension must be an integer: {value!r}")
    dimension = int(value)
    if dimension <= 0:
        raise AdapterContractError(f"schema dimension must be positive: {dimension}")
    if any(
        item != value
        for item in candidates[1:]
        if isinstance(item, (int, np.integer)) and not isinstance(item, bool)
    ):
        raise AdapterContractError("schema exposes conflicting observation dimensions")
    return dimension


def _schema_rows(schema: object | None) -> list[Mapping[str, object]]:
    """Return complete schema rows when the caller supplied them."""

    if isinstance(schema, Mapping):
        rows = schema.get("features")
    elif isinstance(schema, Sequence) and not isinstance(schema, (str, bytes)):
        rows = schema
    else:
        rows = None
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        converted = list(rows)
        if all(isinstance(row, Mapping) for row in converted):
            return [row for row in converted if isinstance(row, Mapping)]
    return []


def _schema_summary(schema: object | None) -> dict[str, object]:
    """Summarize schema provenance without importing the schema package."""

    summary: dict[str, object] = {
        "verified": None,
        "schema_id": None,
        "source_identity": None,
        "source_file_sha256": {},
        "source_commits": [],
        "source_versions": [],
    }
    if isinstance(schema, Mapping):
        summary["schema_id"] = schema.get("schema_id")
        summary["source_identity"] = _jsonable(schema.get("source_identity"))
        if isinstance(schema.get("verified"), bool):
            summary["verified"] = schema["verified"]
    rows = _schema_rows(schema)
    if rows:
        statuses = {row.get("status") for row in rows}
        summary["verified"] = statuses == {"verified"}
        commits = {str(row["source_commit"]) for row in rows if row.get("source_commit")}
        versions = {str(row["source_version"]) for row in rows if row.get("source_version")}
        hashes: dict[str, str] = {}
        conflicting_hashes: set[str] = set()
        for row in rows:
            value = row.get("source_file_sha256")
            if not isinstance(value, Mapping):
                continue
            for path, digest in value.items():
                path_text = str(path)
                digest_text = str(digest)
                previous = hashes.get(path_text)
                if previous is not None and previous != digest_text:
                    conflicting_hashes.add(path_text)
                hashes[path_text] = digest_text
        if conflicting_hashes:
            summary["source_file_sha256_conflicts"] = sorted(conflicting_hashes)
        summary["source_file_sha256"] = hashes
        summary["source_commits"] = sorted(commits)
        summary["source_versions"] = sorted(versions)
    return summary


def _is_official_schema(schema: object | None, *, dimension: int) -> bool:
    """Identify the source-backed 259 schema without treating D as evidence."""

    if dimension != 259:
        return False
    rows = _schema_rows(schema)
    summary = _schema_summary(schema)
    if summary.get("verified") is not True:
        return False
    source_identity = summary.get("source_identity")
    if isinstance(source_identity, Mapping):
        distribution = str(source_identity.get("distribution", "")).lower()
        if distribution == "metadrive-simulator":
            return True
    hashes = summary.get("source_file_sha256")
    if isinstance(hashes, Mapping) and {
        "metadrive/obs/state_obs.py",
        "metadrive/component/navigation_module/node_network_navigation.py",
        "metadrive/component/sensors/distance_detector.py",
    }.issubset(str(path) for path in hashes):
        return bool(rows)
    # A document can carry only schema_id/source_identity metadata after a
    # round trip through JSON.  Keep that explicit standard identifier useful
    # while still requiring the document to say it is verified.
    if str(summary.get("schema_id", "")).lower() == "metadrive_state_lidar_259":
        return True
    return False


def _verified_real_schema(schema: object | None, *, dimension: int) -> bool:
    """Require per-index verification before using a real host adapter.

    The synthetic backend has its own adapter and schema path.  A real model
    cannot use the all-unknown host migration template, or a document whose
    top-level ``verified`` flag disagrees with its actual feature rows.  When
    rows are present they are the source of truth; metadata-only documents do
    not establish an input ordering for a real environment.
    """

    rows = _schema_rows(schema)
    if len(rows) != dimension:
        return False
    return all(row.get("status") == "verified" for row in rows)


def _source_file_candidates(
    relative_path: str,
    *,
    project_root: Path | None,
) -> tuple[Path, ...]:
    candidates: list[Path] = []
    if project_root is not None:
        candidates.extend(
            (
                project_root / relative_path,
                project_root / "metadrive" / relative_path,
            )
        )
    module_file = _module_path("metadrive")
    if module_file is not None:
        package = Path(module_file).resolve()
        package_dir = package.parent
        candidates.extend((package_dir.parent / relative_path, package_dir / relative_path))
    # Preserve order while avoiding duplicate filesystem checks.
    return tuple(dict.fromkeys(candidates))


def _verify_source_hashes(
    expected: Mapping[str, object],
    *,
    project_root: Path | None,
) -> dict[str, object]:
    """Compare source-backed schema hashes with the installed MetaDrive tree."""

    checked: dict[str, str] = {}
    missing: list[str] = []
    mismatched: dict[str, dict[str, str]] = {}
    for relative, digest in expected.items():
        relative_text = str(relative)
        expected_text = str(digest)
        path = next(
            (candidate for candidate in _source_file_candidates(relative_text, project_root=project_root) if candidate.is_file()),
            None,
        )
        if path is None:
            missing.append(relative_text)
            continue
        actual, _size = _sha256_file(path)
        checked[relative_text] = str(path)
        if actual != expected_text:
            mismatched[relative_text] = {
                "expected": expected_text,
                "actual": actual,
                "path": str(path),
            }
    return {
        "status": "verified" if not missing and not mismatched else "failed",
        "checked_paths": checked,
        "missing": sorted(missing),
        "mismatched": mismatched,
    }


def _copy_config(config: Mapping[str, object] | None) -> dict[str, object]:
    if config is None:
        return {}
    if not isinstance(config, Mapping):
        raise AdapterContractError("environment config must be a mapping")
    try:
        return copy.deepcopy(dict(config))
    except Exception as error:
        raise AdapterContractError("environment config is not deepcopy-able") from error


def _extract_environment_config(config: Mapping[str, object]) -> dict[str, object]:
    """Accept either resolved stage config or a small profile-like mapping."""

    for key in ("evaluation_env_config", "environment_config"):
        nested = config.get(key)
        if isinstance(nested, Mapping):
            return _copy_config(nested)
    profile = config.get("profile")
    if profile is not None:
        nested = _safe_attr(profile, "evaluation_env_config")
        if isinstance(nested, Mapping):
            return _copy_config(nested)
    environment = config.get("environment")
    if isinstance(environment, Mapping):
        evaluation = environment.get("evaluation")
        common = environment.get("common")
        if isinstance(evaluation, Mapping):
            result = _copy_config(common if isinstance(common, Mapping) else None)
            result.update(_copy_config(evaluation))
            return result
    return _copy_config(config)


def _preprocessing_value(config: Mapping[str, object]) -> object | None:
    for key in ("observation_preprocessing", "preprocessing", "input_boundary"):
        if key in config:
            return config[key]
    return None


def _validate_identity_preprocessing(config: Mapping[str, object]) -> str:
    """Reject normalization paths whose exact model boundary cannot be proven."""

    value = _preprocessing_value(config)
    if isinstance(value, Mapping):
        stage = value.get("stage", value.get("kind", value.get("name")))
        if stage is None:
            raise UnsupportedAdapterError(
                "preprocessing mapping must explicitly declare identity; "
                "normalization is not inferred"
            )
        value = stage
    if value is not None:
        if not isinstance(value, str) or value.strip().lower() not in {
            "identity",
            "raw_to_model_identity",
            "model_input",
            "post_preprocess_identity",
        }:
            raise UnsupportedAdapterError(
                "only identity raw->policy preprocessing is supported; "
                f"found {value!r}"
            )
    for key in ("vec_normalize", "normalize_observation", "observation_normalized"):
        if key in config and config[key] not in (False, None):
            raise UnsupportedAdapterError(
                "unsupported observation normalization path (VecNormalize or "
                f"equivalent): {key}={config[key]!r}"
            )
    return SUPPORTED_PREPROCESSING


def _require_box_space(space: object, *, name: str) -> tuple[int, ...]:
    try:
        from gymnasium.spaces import Box
    except (ImportError, ModuleNotFoundError) as error:
        raise AdapterDependencyError(
            "gymnasium is required to inspect the observation space"
        ) from error
    if not isinstance(space, Box):
        raise UnsupportedAdapterError(
            f"{name} must be a gymnasium.spaces.Box, found {type(space).__name__}"
        )
    shape = _space_shape(space)
    if len(shape) != 1 or shape[0] <= 0:
        raise UnsupportedAdapterError(
            f"{name} must be a one-dimensional Box, found shape={shape}"
        )
    dtype = np.dtype(getattr(space, "dtype", object))
    if dtype != np.dtype(np.float32):
        raise UnsupportedAdapterError(
            f"{name} must use float32 for the verified MLP path, found {dtype}"
        )
    try:
        low = np.asarray(getattr(space, "low", None))
        high = np.asarray(getattr(space, "high", None))
    except (TypeError, ValueError) as error:
        raise AdapterContractError(f"{name} bounds are not array-like") from error
    if low.shape != shape or high.shape != shape:
        raise AdapterContractError(
            f"{name} bounds must have shape {shape}; low={low.shape}, high={high.shape}"
        )
    try:
        finite_bounds = np.isfinite(low).all() and np.isfinite(high).all()
    except TypeError as error:
        raise AdapterContractError(f"{name} bounds must be numeric") from error
    if not finite_bounds:
        raise AdapterContractError(f"{name} bounds must be finite")
    if np.any(low > high):
        raise AdapterContractError(f"{name} low bound exceeds high bound")
    return shape


def _require_discrete_space(space: object, *, name: str) -> int:
    try:
        from gymnasium.spaces import Discrete
    except (ImportError, ModuleNotFoundError) as error:
        raise AdapterDependencyError(
            "gymnasium is required to inspect the action space"
        ) from error
    if not isinstance(space, Discrete):
        raise UnsupportedAdapterError(
            f"{name} must be a single gymnasium.spaces.Discrete, "
            f"found {type(space).__name__}"
        )
    start = int(getattr(space, "start", 0))
    if start != 0:
        raise UnsupportedAdapterError(
            f"{name} must use Discrete(start=0), found start={start}"
        )
    count = int(space.n)
    if count <= 0:
        raise AdapterContractError(f"{name} must have a positive action count")
    return count


def _library_metadata() -> dict[str, object]:
    libraries = {
        "numpy": _distribution_info("numpy", "numpy"),
        "gymnasium": _distribution_info("gymnasium", "gymnasium"),
        "stable_baselines3": _distribution_info(
            "stable-baselines3", "stable_baselines3"
        ),
        "torch": _distribution_info("torch", "torch"),
        "metadrive": _distribution_info("metadrive-simulator", "metadrive"),
        "pillow": _distribution_info("pillow", "PIL"),
    }
    return {
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "libraries": libraries,
    }


def _space_summary(space: object) -> dict[str, object]:
    shape = _space_shape(space)
    result: dict[str, object] = {
        "type": f"{type(space).__module__}.{type(space).__qualname__}",
        "shape": list(shape),
        "dtype": str(getattr(space, "dtype", None)),
    }
    for key in ("n", "start"):
        value = _safe_attr(space, key)
        if value is not None:
            result[key] = _jsonable(value)
    if hasattr(space, "low") and hasattr(space, "high"):
        try:
            low = np.asarray(getattr(space, "low"))
            high = np.asarray(getattr(space, "high"))
            result.update(
                {
                    "low_min": float(np.min(low)),
                    "low_max": float(np.max(low)),
                    "high_min": float(np.min(high)),
                    "high_max": float(np.max(high)),
                }
            )
        except (TypeError, ValueError):
            result["bounds_summary_error"] = True
    return result


def _qualified_type(value: object) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _static_attribute_present(value: object, name: str) -> bool:
    """Check an attribute without invoking a wrapper's dynamic properties."""

    try:
        inspect.getattr_static(value, name)
    except (AttributeError, TypeError):
        return False
    return True


def _environment_chain(env: object) -> Iterator[tuple[str, object]]:
    """Walk common Gym/SB3 wrapper links without probing arbitrary methods."""

    pending: list[tuple[str, object]] = [("environment", env)]
    seen: set[int] = set()
    while pending:
        path, current = pending.pop(0)
        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)
        yield path, current
        for relation in ("env", "venv", "wrapped_env", "unwrapped"):
            child = _safe_attr(current, relation)
            if child is None or child is current or id(child) in seen:
                continue
            pending.append((f"{path}.{relation}", child))


def _unsupported_environment_wrapper(env: object) -> str | None:
    """Return a reason when an env wrapper can alter the policy observation.

    Shape equality alone cannot prove an identity raw-to-policy boundary: a
    normalizer can preserve the Box shape while changing every value.  The
    adapter therefore rejects explicit normalization statistics and Gym
    observation-transform wrappers anywhere in the returned wrapper chain.
    Common episode/time-limit wrappers have none of these markers and remain
    valid.
    """

    for path, current in _environment_chain(env):
        type_name = _qualified_type(current)
        lowered = type_name.lower()
        if "vecnormalize" in lowered:
            return f"{path} is a VecNormalize wrapper ({type_name})"
        if any(
            token in lowered
            for token in ("normalizeobservation", "normalize_observation", "transformobservation")
        ):
            return f"{path} is an observation normalization/transform wrapper ({type_name})"
        for attribute in (
            "obs_rms",
            "ret_rms",
            "normalize_obs",
            "normalize_reward",
            "norm_obs",
            "norm_reward",
        ):
            if _static_attribute_present(current, attribute):
                return (
                    f"{path} exposes normalization state {attribute!r} "
                    f"({type_name})"
                )
        observation_transform = _safe_attr(current, "observation")
        if callable(observation_transform):
            return f"{path} defines a custom observation transform ({type_name})"
    return None


class InputAttributionAdapter:
    """Connect a PPO MLP model to the host environment through a strict API.

    ``model`` may be an already-loaded SB3 model, a model path, or ``None``.
    Loading is lazy so report-only/help paths do not import SB3 or MetaDrive.
    The resolved environment config is copied before the host factory receives
    it.  ``env_factory`` can accept one mapping argument or the project's
    ``make_evaluation_env(seed=..., env_config=...)`` signature.
    """

    def __init__(
        self,
        model: object | str | Path | None = None,
        *,
        model_path: str | Path | None = None,
        config: Mapping[str, object] | None = None,
        env_config: Mapping[str, object] | None = None,
        schema: object | None = None,
        observation_dim: int | None = None,
        expected_observation_dim: int | None = None,
        project_root: str | Path | None = None,
        env_factory: Callable[..., object] | None = None,
        device: str = "cpu",
        deterministic: bool = True,
        verify_mlp_input: bool = True,
        frame_reader: Callable[[object], object] | None = None,
        snapshot_reader: Callable[[object], Mapping[str, object]] | None = None,
    ) -> None:
        if isinstance(model, (str, Path)) and model_path is None:
            model_path = model
            model = None
        if model is not None and model_path is not None:
            raise AdapterContractError("provide model or model_path, not both")
        if observation_dim is not None and expected_observation_dim is not None:
            if int(observation_dim) != int(expected_observation_dim):
                raise AdapterContractError(
                    "observation_dim and expected_observation_dim disagree"
                )
        explicit_dim = (
            expected_observation_dim
            if expected_observation_dim is not None
            else observation_dim
        )
        if explicit_dim is not None:
            if isinstance(explicit_dim, bool) or not isinstance(
                explicit_dim, (int, np.integer)
            ):
                raise AdapterContractError("observation dimension must be an integer")
            if int(explicit_dim) <= 0:
                raise AdapterContractError("observation dimension must be positive")
            explicit_dim = int(explicit_dim)
        schema_dim = _schema_dimension(schema)
        if explicit_dim is not None and schema_dim is not None and explicit_dim != schema_dim:
            raise AdapterContractError(
                f"schema/model expected dimensions disagree: schema={schema_dim}, "
                f"requested={explicit_dim}"
            )
        self._schema = schema
        self._schema_dim = schema_dim
        self._expected_dim = explicit_dim if explicit_dim is not None else schema_dim
        self._model = model
        self.model_path = None if model_path is None else Path(model_path).expanduser()
        self.env_config = _extract_environment_config(config) if config is not None else _copy_config(env_config)
        if config is not None and env_config is not None:
            config_env = _extract_environment_config(config)
            if config_env != _copy_config(env_config):
                raise AdapterContractError("config and env_config disagree")
        self.project_root = (
            None if project_root is None else Path(project_root).expanduser().resolve()
        )
        self._env_factory = env_factory
        self.device = str(device)
        self.deterministic = bool(deterministic)
        self.verify_mlp_input = bool(verify_mlp_input)
        self._frame_reader = frame_reader
        self._snapshot_reader = snapshot_reader
        self._policy_seed: int | None = None
        self._last_mlp_inputs: tuple[np.ndarray, ...] = ()
        self._model_observation_space: object | None = None
        self._model_action_space: object | None = None
        self._official_schema = False
        self._schema_hash_status: dict[str, object] | None = None
        self._preprocessing = _validate_identity_preprocessing(self.env_config)
        self.metadata: dict[str, object] = {
            "adapter": {
                "class": _qualified_type(self),
                "contract": "single-agent PPO MLP / identity Box input / Discrete action",
            },
            "preprocessing": {
                "raw_to_model": self._preprocessing,
                "boundary": "after host observation generation, before SB3 MLP",
                "normalization_applied_by_adapter": False,
                "clip_applied_by_adapter": False,
                "mlp_pre_hook": "pending_model_validation",
            },
            "environment": {
                "factory": None,
                "config": _jsonable(self.env_config),
                "actual_observation_shape": None,
                "actual_action_count": None,
            },
            "schema": {
                "dimension": self._schema_dim,
                "provided": schema is not None,
                "source": _qualified_type(schema) if schema is not None else None,
                "official_contract": False,
                "source_hash_check": None,
            },
            "model": {"path": None, "resolved_path": None},
            "runtime": _library_metadata(),
        }
        schema_metadata = self.metadata["schema"]
        if isinstance(schema_metadata, dict):
            schema_metadata.update(_schema_summary(schema))
        if self.model_path is not None:
            self.metadata["model"] = {
                "path": str(self.model_path),
                "resolved_path": None,
                "sha256": None,
            }
        if self._model is not None:
            self._validate_model_contract()

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
        *,
        model: object | None = None,
        model_path: str | Path | None = None,
        schema: object | None = None,
        env_factory: Callable[..., object] | None = None,
        **kwargs: object,
    ) -> "InputAttributionAdapter":
        """Build an adapter from a resolved profile or direct stage mapping."""

        if model_path is None:
            candidate = config.get("model_path")
            if isinstance(candidate, (str, Path)):
                model_path = candidate
        return cls(
            model=model,
            model_path=model_path,
            config=config,
            schema=schema,
            env_factory=env_factory,
            **kwargs,
        )

    @property
    def model(self) -> object:
        return self.ensure_loaded()

    @property
    def observation_dim(self) -> int:
        self.ensure_loaded()
        assert self._expected_dim is not None
        return self._expected_dim

    @property
    def dimension(self) -> int:
        """Canonical width consumed by the shared A/B comparison code."""

        return self.observation_dim

    @property
    def object_dimension(self) -> int:
        """Alias used by the experiment core for the policy input width."""

        return self.observation_dim

    @property
    def action_count(self) -> int:
        self.ensure_loaded()
        assert self._model_action_space is not None
        return int(self._model_action_space.n)

    @property
    def last_mlp_inputs(self) -> tuple[np.ndarray, ...]:
        return tuple(item.copy() for item in self._last_mlp_inputs)

    def ensure_loaded(self) -> object:
        if self._model is None:
            return self.load()
        if self._model_observation_space is None:
            self._validate_model_contract()
        return self._model

    def load(self) -> object:
        """Load the configured PPO zip and validate its full input contract."""

        if self._model is not None:
            self._validate_model_contract()
            return self._model
        if self.model_path is None:
            raise AdapterContractError("no model or model_path was provided")
        path = self.model_path
        if not path.is_absolute():
            if self.project_root is not None:
                path = self.project_root / path
            else:
                path = Path.cwd() / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"model file not found: {path}")
        digest, size = _sha256_file(path)
        try:
            from stable_baselines3 import PPO
        except (ImportError, ModuleNotFoundError) as error:
            raise AdapterDependencyError(
                "stable-baselines3 is required to load a PPO model"
            ) from error
        try:
            self._model = PPO.load(str(path), device=self.device)
        except Exception as error:
            raise AdapterContractError(
                f"failed to load PPO model {path}: {type(error).__name__}: {error}"
            ) from error
        self.model_path = path
        self.metadata["model"] = {
            "path": str(path),
            "resolved_path": str(path),
            "sha256": digest,
            "size_bytes": size,
            "requested_device": self.device,
            "actual_device": str(_safe_attr(_safe_attr(self._model, "device"), "type") or _safe_attr(self._model, "device") or "unknown"),
        }
        self._validate_model_contract()
        return self._model

    def _validate_model_contract(self) -> None:
        model = self._model
        if model is None:
            return
        observation_space = _safe_attr(model, "observation_space")
        action_space = _safe_attr(model, "action_space")
        if observation_space is None or action_space is None:
            raise AdapterContractError("model must expose observation_space and action_space")
        shape = _require_box_space(observation_space, name="model.observation_space")
        model_dim = shape[0]
        if self._schema_dim is not None and model_dim != self._schema_dim:
            raise AdapterContractError(
                f"model/schema observation dimension mismatch: model={model_dim}, "
                f"schema={self._schema_dim}"
            )
        if self._expected_dim is not None and model_dim != self._expected_dim:
            raise AdapterContractError(
                f"model observation dimension mismatch: model={model_dim}, "
                f"expected={self._expected_dim}"
            )
        if self._schema is not None and not _verified_real_schema(
            self._schema,
            dimension=model_dim,
        ):
            raise UnsupportedAdapterError(
                f"real model dimension {model_dim} requires a complete verified "
                "per-index input schema; unverified or metadata-only schemas "
                "cannot establish the host feature order"
            )
        if model_dim >= 259 and self._schema is None:
            raise UnsupportedAdapterError(
                f"a {model_dim}-wide real model requires an explicit input schema "
                "that is complete and verified; dimension alone does not establish "
                "host feature semantics"
            )
        if model_dim == 259:
            schema_rows = _schema_rows(self._schema)
            schema_summary = _schema_summary(self._schema)
            self._official_schema = _is_official_schema(self._schema, dimension=model_dim)
            if self._official_schema:
                expected_hashes = schema_summary.get("source_file_sha256")
                if not isinstance(expected_hashes, Mapping) or not expected_hashes:
                    raise UnsupportedAdapterError(
                        "the verified MetaDrive 259 schema has no source hashes"
                    )
                self._schema_hash_status = _verify_source_hashes(
                    expected_hashes,
                    project_root=self.project_root,
                )
                if self._schema_hash_status.get("status") != "verified":
                    raise UnsupportedAdapterError(
                        "installed MetaDrive source does not match the verified "
                        f"259 schema: {self._schema_hash_status}"
                    )
            schema_metadata = self.metadata.get("schema")
            if isinstance(schema_metadata, dict):
                schema_metadata["official_contract"] = self._official_schema
                schema_metadata["source_hash_check"] = copy.deepcopy(
                    self._schema_hash_status
                )
        self._expected_dim = model_dim
        action_count = _require_discrete_space(action_space, name="model.action_space")
        policy = _safe_attr(model, "policy")
        if policy is None:
            raise UnsupportedAdapterError("PPO model has no policy object")
        extractor = _safe_attr(policy, "features_extractor")
        extractor_name = type(extractor).__name__ if extractor is not None else None
        if extractor_name != "FlattenExtractor":
            raise UnsupportedAdapterError(
                "only SB3 MLP/FlattenExtractor policies are supported; "
                f"found features_extractor={extractor_name!r}"
            )
        mlp_extractor = _safe_attr(policy, "mlp_extractor")
        # SB3 ActorCriticPolicy.get_distribution() calls
        # ``mlp_extractor.forward_actor(features)``.  In SB3 2.x that method
        # invokes ``policy_net(features)`` directly, bypassing a hook on the
        # MlpExtractor container.  Hook the actor branch that is actually
        # executed so a zero-capture probe cannot be mistaken for evidence.
        mlp = _safe_attr(mlp_extractor, "policy_net")
        if mlp is None or not callable(getattr(mlp, "register_forward_pre_hook", None)):
            raise UnsupportedAdapterError(
                "policy does not expose a hookable actor MLP policy_net"
            )
        normalize_images = _safe_attr(policy, "normalize_images")
        if normalize_images not in (None, True, False):
            raise AdapterContractError("policy.normalize_images is invalid")
        try:
            from stable_baselines3.common.preprocessing import is_image_space

            image_space = bool(is_image_space(observation_space))
        except (ImportError, ModuleNotFoundError) as error:
            raise AdapterDependencyError(
                "stable-baselines3 preprocessing helpers are unavailable"
            ) from error
        if image_space:
            raise UnsupportedAdapterError(
                "image observation preprocessing is outside the verified MLP path"
            )
        get_env = getattr(model, "get_env", None)
        attached_env = get_env() if callable(get_env) else None
        if attached_env is not None:
            wrapper_issue = _unsupported_environment_wrapper(attached_env)
            if wrapper_issue is not None:
                raise UnsupportedAdapterError(
                    "unsupported VecNormalize/observation wrapper attached to "
                    f"model: {wrapper_issue}"
                )
        self._model_observation_space = observation_space
        self._model_action_space = action_space
        environment = self.metadata.setdefault("environment", {})
        if isinstance(environment, dict):
            environment["model_observation_shape"] = list(shape)
            environment["model_action_count"] = action_count
        self.metadata["observation"] = _space_summary(observation_space)
        self.metadata["action"] = _space_summary(action_space)
        preprocessing = self.metadata.setdefault("preprocessing", {})
        if isinstance(preprocessing, dict):
            preprocessing.update(
                {
                    "sb3_observation_space": _qualified_type(observation_space),
                    "sb3_policy": _qualified_type(policy),
                    "features_extractor": _qualified_type(extractor),
                    "policy_normalize_images": normalize_images,
                    "image_space": image_space,
                    "mlp_pre_hook": "policy.mlp_extractor.policy_net.forward_pre_hook",
                    "verified": True,
                }
            )
        model_meta = self.metadata.setdefault("model", {})
        if isinstance(model_meta, dict):
            model_meta.update(
                {
                    "policy_class": _qualified_type(policy),
                    "features_extractor_class": _qualified_type(extractor),
                    "action_count": action_count,
                    "observation_dim": model_dim,
                }
            )

    def prepare(self, raw: object) -> np.ndarray:
        """Validate one host observation and return an independent float32 copy.

        Bounds are intentionally not checked here: this same method is used
        for the model-input copy after a caller applies an explicit stress
        intervention.  Shape, dtype, finite values, and dimension remain
        strict; no flattening, padding, truncation, normalization, or clip is
        performed.
        """

        dimension = self.observation_dim
        array = np.asarray(raw)
        if array.shape != (dimension,):
            raise AdapterContractError(
                f"observation shape must be ({dimension},), found {array.shape}"
            )
        if array.dtype != np.dtype(np.float32):
            raise AdapterContractError(
                f"observation dtype must be float32, found {array.dtype}"
            )
        if not np.isfinite(array).all():
            raise AdapterContractError("observation contains NaN or Inf")
        return np.array(array, dtype=np.float32, copy=True)

    def _validate_batch(self, value: object) -> tuple[np.ndarray, bool]:
        array = np.asarray(value)
        single = array.ndim == 1
        if single:
            # Keep one canonical representation for the policy and the hook:
            # SB3 receives a 1-D value for ordinary ``predict`` while its
            # tensor path always presents the MLP with a leading batch axis.
            return self.prepare(array).reshape(1, -1), True
        if array.ndim != 2 or array.shape[1] != self.observation_dim:
            raise AdapterContractError(
                f"policy input must be ({self.observation_dim},) or "
                f"(batch, {self.observation_dim}), found {array.shape}"
            )
        if array.dtype != np.dtype(np.float32):
            raise AdapterContractError(
                f"policy batch dtype must be float32, found {array.dtype}"
            )
        if array.shape[0] <= 0:
            raise AdapterContractError("policy input batch must not be empty")
        if not np.isfinite(array).all():
            raise AdapterContractError("policy input contains NaN or Inf")
        return np.array(array, dtype=np.float32, copy=True), False

    def _mlp_module(self) -> object:
        model = self.ensure_loaded()
        policy = getattr(model, "policy", None)
        extractor = None if policy is None else getattr(policy, "mlp_extractor", None)
        mlp = None if extractor is None else getattr(extractor, "policy_net", None)
        if mlp is None or not callable(getattr(mlp, "register_forward_pre_hook", None)):
            raise UnsupportedAdapterError(
                "model policy has no hookable actor MLP policy_net"
            )
        return mlp

    @contextmanager
    def _probe_mlp_inputs(self, expected: np.ndarray) -> Iterator[list[np.ndarray]]:
        """Capture MLP inputs without changing the module or tensor values."""

        module = self._mlp_module()
        observed: list[np.ndarray] = []

        def hook(_module: object, args: tuple[object, ...]) -> None:
            if not args:
                raise ModelInputBoundaryError("MLP pre-hook received no input tensor")
            tensor = args[0]
            detach = getattr(tensor, "detach", None)
            cpu = getattr(detach(), "cpu", None) if callable(detach) else None
            numpy = getattr(cpu(), "numpy", None) if callable(cpu) else None
            if not callable(numpy):
                raise ModelInputBoundaryError(
                    "MLP pre-hook input is not a tensor convertible to NumPy"
                )
            value = np.asarray(numpy(), dtype=np.float32).copy()
            observed.append(value)
            if value.shape != expected.shape or not np.array_equal(value, expected):
                raise ModelInputBoundaryError(
                    "MLP input differs from requested policy input; "
                    f"expected shape/value {expected.shape}, observed {value.shape}"
                )

        handle = module.register_forward_pre_hook(hook)
        try:
            yield observed
            if not observed:
                raise ModelInputBoundaryError(
                    "MLP pre-hook captured no inputs; the verified actor path "
                    "was not executed"
                )
        finally:
            remove = getattr(handle, "remove", None)
            if callable(remove):
                remove()

    @staticmethod
    def _categorical_probabilities(distribution: object) -> np.ndarray:
        raw_distribution = getattr(distribution, "distribution", distribution)
        probs = getattr(raw_distribution, "probs", None)
        if probs is None:
            raise UnsupportedAdapterError(
                "policy distribution does not expose categorical probabilities"
            )
        detach = getattr(probs, "detach", None)
        cpu = getattr(detach(), "cpu", None) if callable(detach) else None
        numpy = getattr(cpu(), "numpy", None) if callable(cpu) else None
        if not callable(numpy):
            raise AdapterContractError("categorical probabilities are not tensor-like")
        array = np.asarray(numpy(), dtype=np.float64)
        if array.ndim != 2 or array.shape[1] <= 0:
            raise AdapterContractError(
                f"categorical probabilities must have shape (batch, K), found {array.shape}"
            )
        if not np.isfinite(array).all() or np.any(array < 0.0):
            raise AdapterContractError("policy probabilities contain invalid values")
        sums = np.sum(array, axis=1)
        if not np.allclose(sums, 1.0, rtol=1e-6, atol=1e-7):
            raise AdapterContractError(
                f"policy probabilities do not sum to one: range={sums.min()}..{sums.max()}"
            )
        return array

    def _policy_probabilities(self, batch: np.ndarray) -> np.ndarray:
        model = self.ensure_loaded()
        policy = getattr(model, "policy", None)
        get_distribution = getattr(policy, "get_distribution", None)
        obs_to_tensor = getattr(policy, "obs_to_tensor", None)
        if not callable(get_distribution) or not callable(obs_to_tensor):
            raise UnsupportedAdapterError(
                "SB3 policy must expose obs_to_tensor and get_distribution"
            )
        obs_tensor, _vectorized = obs_to_tensor(batch)
        distribution = get_distribution(obs_tensor)
        return self._categorical_probabilities(distribution)

    @staticmethod
    def _normalise_actions(raw_action: object, batch_size: int, action_count: int) -> np.ndarray:
        actions = np.asarray(raw_action)
        if actions.size != batch_size:
            raise AdapterContractError(
                f"model.predict returned {actions.shape} for batch size {batch_size}"
            )
        actions = actions.reshape(batch_size)
        if not np.issubdtype(actions.dtype, np.integer):
            try:
                if not np.isfinite(actions.astype(float)).all():
                    raise AdapterContractError("model.predict returned a non-finite action")
            except (TypeError, ValueError) as error:
                raise AdapterContractError("model.predict returned non-numeric actions") from error
            if not np.all(actions.astype(float) == np.floor(actions.astype(float))):
                raise AdapterContractError("model.predict returned non-integral actions")
        integer_actions = actions.astype(np.int64)
        if np.any(integer_actions < 0) or np.any(integer_actions >= action_count):
            raise AdapterContractError(
                f"model.predict action outside Discrete({action_count})"
            )
        return integer_actions

    def ordinary_predict(
        self,
        value: object,
        *,
        deterministic: bool | None = None,
    ) -> object:
        """Call the ordinary SB3 ``model.predict`` path on validated input."""

        batch, single = self._validate_batch(value)
        model = self.ensure_loaded()
        predict = getattr(model, "predict", None)
        if not callable(predict):
            raise AdapterContractError("model does not expose predict")
        deterministic_value = self.deterministic if deterministic is None else bool(deterministic)
        if self.verify_mlp_input:
            with self._probe_mlp_inputs(batch) as probe_records:
                actions, state = predict(
                    batch[0] if single else batch,
                    deterministic=deterministic_value,
                )
            self._last_mlp_inputs = tuple(item.copy() for item in probe_records)
        else:
            actions, state = predict(
                batch[0] if single else batch,
                deterministic=deterministic_value,
            )
        del state
        normalised = self._normalise_actions(actions, 1 if single else batch.shape[0], self.action_count)
        return int(normalised[0]) if single else normalised

    def predict(
        self,
        value: object,
        *,
        deterministic: bool | None = None,
    ) -> tuple[np.ndarray, int | np.ndarray]:
        """Return categorical probabilities and ordinary SB3-selected actions.

        For a single 1-D input the return is ``(K,)`` probabilities and a
        Python ``int`` action.  For a batch the return is ``(B,K)`` and
        ``(B,)``.  Actions come from ``model.predict`` itself; probabilities
        come from the same policy distribution.  With deterministic inference
        the two paths must agree with the probability argmax.
        """

        batch, single = self._validate_batch(value)
        model = self.ensure_loaded()
        predict = getattr(model, "predict", None)
        if not callable(predict):
            raise AdapterContractError("model does not expose predict")
        deterministic_value = self.deterministic if deterministic is None else bool(deterministic)
        expected = batch
        probe_records: list[np.ndarray]
        with self._probe_mlp_inputs(expected) if self.verify_mlp_input else _null_probe() as probe_records:
            raw_actions, _state = predict(batch[0] if single else batch, deterministic=deterministic_value)
            # ``get_distribution`` uses the exact same policy object and its
            # normal SB3 preprocessing.  Keep it inside the hook so both
            # ordinary action and probability calls are checked.
            probabilities = self._policy_probabilities(batch)
        self._last_mlp_inputs = tuple(item.copy() for item in probe_records)
        batch_actions = self._normalise_actions(
            raw_actions,
            1 if single else batch.shape[0],
            self.action_count,
        )
        if deterministic_value:
            argmax_actions = np.argmax(probabilities, axis=1).astype(np.int64)
            if not np.array_equal(batch_actions, argmax_actions):
                raise ModelInputBoundaryError(
                    "ordinary deterministic model.predict action differs from "
                    "categorical probability argmax"
                )
        if single:
            return probabilities[0], int(batch_actions[0])
        return probabilities, batch_actions

    def seed_policy(self, seed: int) -> int:
        """Seed SB3/global policy RNGs and model spaces without resetting an env."""

        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise AdapterContractError("policy seed must be an integer")
        seed_int = int(seed)
        if not 0 <= seed_int <= 2**32 - 1:
            raise AdapterContractError("policy seed must be in [0, 2**32-1]")
        try:
            from stable_baselines3.common.utils import set_random_seed
        except (ImportError, ModuleNotFoundError):
            np.random.seed(seed_int)
        else:
            set_random_seed(seed_int)
        model = self._model
        if model is not None:
            for space in (
                _safe_attr(model, "action_space"),
                _safe_attr(model, "observation_space"),
            ):
                seed_method = getattr(space, "seed", None)
                if callable(seed_method):
                    seed_method(seed_int)
        self._policy_seed = seed_int
        self.metadata["policy_seed"] = seed_int
        return seed_int

    def make_env(self) -> object:
        """Construct one validated raw evaluation environment on explicit request."""

        config = _copy_config(self.env_config)
        factory = self._env_factory
        if factory is None:
            try:
                from env_factory import make_evaluation_env
            except (ImportError, ModuleNotFoundError) as error:
                raise AdapterDependencyError(
                    "the host env_factory.make_evaluation_env is unavailable"
                ) from error
            factory = make_evaluation_env
        seed = DEFAULT_POLICY_SEED if self._policy_seed is None else self._policy_seed
        try:
            parameters = inspect.signature(factory).parameters
        except (TypeError, ValueError):
            parameters = {}
        if "env_config" in parameters:
            env = factory(seed=seed, env_config=config)
        elif "config" in parameters:
            env = factory(config=config)
        else:
            env = factory(config)
        try:
            self._validate_environment(env)
        except Exception:
            close = getattr(env, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            raise
        self.metadata["environment"]["factory"] = _qualified_type(factory)
        return env

    def _validate_environment(self, env: object) -> None:
        wrapper_issue = _unsupported_environment_wrapper(env)
        if wrapper_issue is not None:
            raise UnsupportedAdapterError(
                "unsupported VecNormalize/observation wrapper in returned "
                f"environment: {wrapper_issue}"
            )
        space = _safe_attr(env, "observation_space")
        action_space = _safe_attr(env, "action_space")
        if space is None or action_space is None:
            raise AdapterContractError("environment must expose observation_space and action_space")
        shape = _require_box_space(space, name="environment.observation_space")
        if shape[0] != self.observation_dim:
            raise AdapterContractError(
                f"model/schema/environment observation dimensions disagree: "
                f"model={self.observation_dim}, environment={shape[0]}"
            )
        count = _require_discrete_space(action_space, name="environment.action_space")
        if count != self.action_count:
            raise AdapterContractError(
                f"model/environment action counts disagree: model={self.action_count}, "
                f"environment={count}"
            )
        self.metadata["environment"]["actual_observation_shape"] = list(shape)
        self.metadata["environment"]["actual_action_count"] = count
        self.metadata["environment"]["class"] = _qualified_type(env)
        if self._official_schema:
            self._validate_official_environment_config(env)

    def _validate_official_environment_config(self, env: object) -> None:
        """Check the merged MetaDrive options behind the verified 259 schema."""

        config = _mapping_view(_safe_attr(env, "config"))
        if config is None:
            raise UnsupportedAdapterError(
                "verified MetaDrive 259 schema requires the merged environment "
                "config for lidar/detector validation"
            )

        scalar_expectations: dict[str, object] = {
            "discrete_action": True,
            "discrete_throttle_dim": 3,
            "discrete_steering_dim": 3,
            "num_agents": 1,
            "traffic_density": 0.0,
            "random_spawn_lane_index": False,
            "random_lane_width": False,
            "random_lane_num": False,
            "accident_prob": 0.0,
        }
        observed: dict[str, object] = {}
        for key, expected in scalar_expectations.items():
            if key not in config:
                raise UnsupportedAdapterError(
                    f"official MetaDrive config lacks required key {key!r}"
                )
            actual = config[key]
            observed[key] = _jsonable(actual)
            if actual != expected:
                raise UnsupportedAdapterError(
                    f"official 259 observation contract requires {key}={expected!r}; "
                    f"found {actual!r}"
                )

        vehicle_config = _mapping_view(config.get("vehicle_config"))
        if vehicle_config is None:
            raise UnsupportedAdapterError(
                "official MetaDrive config lacks vehicle_config for detector validation"
            )
        expected_detectors: dict[str, dict[str, object]] = {
            "lidar": {"num_lasers": 240, "distance": 50.0, "num_others": 0},
            "side_detector": {"num_lasers": 0, "distance": 50.0},
            "lane_line_detector": {"num_lasers": 0, "distance": 20.0},
        }
        detector_observed: dict[str, object] = {}
        for detector_name, expectations in expected_detectors.items():
            detector = _mapping_view(vehicle_config.get(detector_name))
            if detector is None:
                raise UnsupportedAdapterError(
                    f"official MetaDrive config lacks vehicle_config.{detector_name}"
                )
            detector_values: dict[str, object] = {}
            for key, expected in expectations.items():
                if key not in detector:
                    raise UnsupportedAdapterError(
                        f"official MetaDrive config lacks "
                        f"vehicle_config.{detector_name}.{key}"
                    )
                actual = detector[key]
                detector_values[key] = _jsonable(actual)
                if actual != expected:
                    raise UnsupportedAdapterError(
                        f"official 259 observation contract requires "
                        f"vehicle_config.{detector_name}.{key}={expected!r}; "
                        f"found {actual!r}"
                    )
            detector_observed[detector_name] = detector_values
        environment_metadata = self.metadata.get("environment")
        if isinstance(environment_metadata, dict):
            environment_metadata["verified_observation_config"] = {
                "scalar": observed,
                "vehicle_detectors": detector_observed,
                "standard_dimension": 259,
                "source_hash_check": copy.deepcopy(self._schema_hash_status),
            }

    def snapshot(self, env: object) -> dict[str, object]:
        """Read a small JSON-safe vehicle/environment state snapshot."""

        if self._snapshot_reader is not None:
            value = self._snapshot_reader(env)
            if not isinstance(value, Mapping):
                raise AdapterContractError("snapshot_reader must return a mapping")
            return {str(key): _jsonable(item) for key, item in value.items()}
        result: dict[str, object] = {
            "environment_class": _qualified_type(env),
            "scenario_seed": _jsonable(_safe_attr(env, "current_seed")),
            "episode_step": _jsonable(_safe_attr(env, "episode_step")),
        }
        agent: object | None = None
        agents = _safe_attr(env, "agents")
        if isinstance(agents, Mapping):
            if len(agents) == 1:
                agent = next(iter(agents.values()))
        if agent is None:
            agent = _safe_attr(env, "agent")
        if agent is None:
            result["agent"] = None
            return result
        agent_state: dict[str, object] = {}
        for key in (
            "position",
            "heading",
            "heading_theta",
            "velocity",
            "speed",
            "speed_km_h",
            "steering",
            "throttle_brake",
            "current_action",
            "last_current_action",
            "lane_index",
        ):
            value = _safe_attr(agent, key)
            if value is not None:
                agent_state[key] = _jsonable(value)
        navigation = _safe_attr(agent, "navigation")
        if navigation is not None:
            for key in ("travelled_length", "checkpoints", "current_ref_lanes"):
                value = _safe_attr(navigation, key)
                if value is not None:
                    agent_state[f"navigation_{key}"] = _jsonable(value)
        result["agent"] = agent_state
        return {str(key): _jsonable(value) for key, value in result.items()}

    def frame(self, env: object) -> np.ndarray:
        """Render one top-down RGB image while preserving recorder frames."""

        rng_before = _rng_snapshot()
        if self._frame_reader is not None:
            raw = self._frame_reader(env)
        else:
            render = getattr(env, "render", None)
            if not callable(render):
                raise AdapterContractError("environment has no render method")
            renderer = _safe_attr(env, "top_down_renderer")
            frames = _safe_attr(renderer, "_screen_frames") if renderer is not None else None
            saved_frames = list(frames) if isinstance(frames, list) else None
            try:
                raw = render(
                    mode="topdown",
                    screen_record=False,
                    window=False,
                    screen_size=DEFAULT_RENDER_SIZE,
                    camera_position=None,
                )
            finally:
                # Existing recorders may have screen_record=True.  MetaDrive
                # 0.4.3 appends despite a later screen_record=False kwarg when
                # the renderer already exists, so restore its frame list.
                if saved_frames is not None and isinstance(frames, list):
                    frames[:] = saved_frames
        if not _rng_unchanged(rng_before, _rng_snapshot()):
            raise AdapterContractError(
                "top-down rendering changed Python, NumPy, or Torch RNG state"
            )
        rendering = self.metadata.setdefault("rendering", {})
        if isinstance(rendering, dict):
            rendering.update(
                {
                    "api": "env.render(mode='topdown')",
                    "screen_record": False,
                    "camera_position": None,
                    "rng_invariance_verified": True,
                }
            )
        array = np.asarray(raw)
        if array.ndim != 3 or array.shape[2] != 3:
            raise AdapterContractError(
                f"top-down render must return HxWx3 RGB, found {array.shape}"
            )
        if not np.issubdtype(array.dtype, np.integer):
            if not np.isfinite(array).all():
                raise AdapterContractError("rendered frame contains non-finite values")
            array = np.rint(array)
        if np.any(array < 0) or np.any(array > 255):
            raise AdapterContractError("rendered RGB values must be in [0, 255]")
        return np.array(array, dtype=np.uint8, copy=True)

    def action_dt(self, env: object) -> float:
        """Return the host decision duration through ``derive_timing``."""

        try:
            from evaluation_visualization import derive_timing
        except (ImportError, ModuleNotFoundError) as error:
            raise AdapterDependencyError(
                "evaluation_visualization.derive_timing is unavailable"
            ) from error
        config = _mapping_view(_safe_attr(env, "config"))
        if config is None:
            raise AdapterContractError("environment has no config for timing derivation")
        try:
            timing = derive_timing(config)
        except Exception as error:
            raise AdapterContractError(
                f"could not derive action duration from environment config: "
                f"{type(error).__name__}: {error}"
            ) from error
        return _finite_float(timing.action_duration_seconds, name="action duration")

    def model_metadata(self) -> dict[str, object]:
        """Return a detached metadata snapshot suitable for manifest JSON."""

        if self._model is not None:
            self._validate_model_contract()
        return copy.deepcopy(_jsonable(self.metadata))  # type: ignore[return-value]


@contextmanager
def _null_probe() -> Iterator[list[np.ndarray]]:
    observed: list[np.ndarray] = []
    yield observed


# Short aliases keep the adapter easy to discover from the parent core while
# retaining one implementation and one contract.
MetaDriveInputAdapter = InputAttributionAdapter
Adapter = InputAttributionAdapter


__all__ = [
    "Adapter",
    "AdapterContractError",
    "AdapterDependencyError",
    "InputAttributionAdapter",
    "MetaDriveInputAdapter",
    "ModelInputBoundaryError",
    "UnsupportedAdapterError",
]
