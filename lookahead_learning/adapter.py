"""Read-only connection and contract checks for the host MetaDrive project.

The host checkout used by this repository currently exposes the upstream
259-wide state observation.  The requested extension is defined against a
*verified* 262-wide host observation which already contains the three
start-lane fields.  This module deliberately refuses to manufacture those
fields when the host does not provide them.

No MetaDrive module is imported at module import time.  A caller can therefore
use :mod:`lookahead_learning.adapter` for ``--help`` and static checks on a machine
where the simulator dependencies are absent.  Environment construction is
explicit and is never performed by an import.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
import copy
from dataclasses import dataclass, field, replace
import importlib
import math
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Final, Literal, TypeAlias, cast

import numpy as np


BASELINE_OBS_DIM: Final[int] = 262
"""Required width of the already modified host observation."""

PREVIEW_FEATURE_DIM: Final[int] = 3
AUGMENTED_OBS_DIM: Final[int] = BASELINE_OBS_DIM + PREVIEW_FEATURE_DIM
NORMALIZATION_DISTANCE_M: Final[float] = 10.0
PREFIX_FEATURE_INDICES: Final[tuple[int, int, int]] = (259, 260, 261)
PREFIX_FEATURE_NAMES: Final[tuple[str, str, str]] = (
    "start_lane_lateral_offset",
    "start_lane_heading_error",
    "start_lane_reference_valid",
)

Mode: TypeAlias = Literal["baseline", "lookahead_obs", "lookahead_obs_pp_reward"]

# The descriptive names are the values carried through runtime metadata and
# telemetry.  The short names remain accepted at the input boundary so old
# launch scripts can be rerun while producing canonical records.
CANONICAL_MODES: Final[tuple[str, str, str]] = (
    "baseline",
    "lookahead_obs",
    "lookahead_obs_pp_reward",
)
MODE_ALIASES: Final[dict[str, str]] = {
    "obs": "lookahead_obs",
    "obs_pp": "lookahead_obs_pp_reward",
}
MODE_CHOICES: Final[tuple[str, ...]] = CANONICAL_MODES + tuple(MODE_ALIASES)


def normalize_mode(value: str) -> Mode:
    """Normalize a CLI or API mode to its descriptive canonical value."""

    if not isinstance(value, str):
        raise ValueError(f"mode must be a string, found {type(value).__name__}")
    canonical = MODE_ALIASES.get(value, value)
    if canonical not in CANONICAL_MODES:
        choices = ", ".join(CANONICAL_MODES)
        raise ValueError(f"unknown mode {value!r}; choose one of {choices}")
    return cast(Mode, canonical)


class HostContractError(RuntimeError):
    """The host environment does not satisfy a contract used by this module."""


class UnsupportedHostError(HostContractError):
    """A requested mode is outside the evidence available from the host."""


class HostImportError(HostContractError):
    """A host module could not be imported from the requested project root."""


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise HostContractError(f"{name} must be numeric, not bool")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise HostContractError(f"{name} must be numeric") from error
    if not math.isfinite(number):
        raise HostContractError(f"{name} must be finite")
    return number


def _normalise_root(project_root: str | Path) -> Path:
    root = Path(project_root).expanduser().resolve()
    if not root.is_dir():
        raise HostImportError(f"project root is not a directory: {root}")
    return root


def _host_config_mapping(value: object, *, name: str = "config") -> Mapping[str, object]:
    """Read a host config through the verified MetaDrive ``Config`` API.

    MetaDrive 0.4.3's ``Config`` deliberately provides mapping-like methods but
    does not inherit ``collections.abc.Mapping``.  The source-backed
    ``get_dict`` API is the supported, non-mutating conversion for adapter
    inspection; arbitrary objects exposing a coincidental ``get`` method are
    not accepted.
    """

    if isinstance(value, Mapping):
        return value
    try:
        from metadrive.utils import Config as MetaDriveConfig
    except (ImportError, ModuleNotFoundError) as error:
        raise HostContractError(
            f"{name} is not a mapping and MetaDrive Config is unavailable"
        ) from error
    if not isinstance(value, MetaDriveConfig):
        raise HostContractError(
            f"{name} must be a mapping or metadrive.utils.Config, found "
            f"{type(value).__module__}.{type(value).__qualname__}"
        )
    result = value.get_dict()
    if not isinstance(result, Mapping):
        raise HostContractError(
            f"{name}.get_dict() did not return a mapping"
        )
    return result


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


@contextmanager
def _temporary_import_root(project_root: Path):
    """Temporarily make a flat host checkout importable.

    The path is restored immediately after import.  This is a process-local
    import operation, not a PYTHONPATH edit or a persistent environment change.
    """

    path_string = str(project_root)
    old_path = list(sys.path)
    # Put the requested checkout first even when it was already present later
    # in sys.path.  This avoids importing a same-named module from another
    # checkout during a relocated run.
    sys.path[:] = [path_string] + [entry for entry in old_path if entry != path_string]
    try:
        yield
    finally:
        sys.path[:] = old_path


def _module_path(module: ModuleType, module_name: str) -> Path:
    origin = getattr(module, "__file__", None)
    if origin is None:
        raise HostImportError(f"host module has no file origin: {module_name}")
    return Path(origin).resolve()


def import_host_module(
    module_name: str,
    *,
    project_root: str | Path,
) -> ModuleType:
    """Import one flat host module and verify its file origin.

    ``env_factory`` imports sibling modules by their historical top-level
    names, so a normal import is retained.  A pre-existing module from another
    checkout is rejected instead of silently reusing it.
    """

    root = _normalise_root(project_root)
    # Check before import as well as after it: a wrong-root parent package
    # (for example ``configs``) can otherwise satisfy importlib while the
    # requested child is loaded lazily from the wrong checkout.
    _check_known_host_origins(root)
    existing = sys.modules.get(module_name)
    if existing is not None:
        try:
            existing_path = _module_path(existing, module_name)
        except HostImportError:
            raise
        if not _inside(existing_path, root):
            raise HostImportError(
                f"{module_name} is already imported from another root: "
                f"{existing_path}; requested root={root}"
            )
        return existing

    try:
        with _temporary_import_root(root):
            module = importlib.import_module(module_name)
    except HostContractError:
        raise
    except Exception as error:
        raise HostImportError(
            f"failed to import {module_name!r} from {root}: "
            f"{type(error).__name__}: {error}"
        ) from error

    origin = _module_path(module, module_name)
    if not _inside(origin, root):
        raise HostImportError(
            f"{module_name} resolved outside requested project root: "
            f"{origin} (root={root})"
        )
    _check_known_host_origins(root)
    return module


# These are flat top-level modules used by this checkout's factory and config
# path.  A process may have imported a same-named module from another clone
# before the preview runner starts; accepting that module would make the
# requested project root advisory rather than authoritative.  Third-party
# modules (including the separately checked-out MetaDrive dependency) are
# intentionally outside this list.
_KNOWN_HOST_MODULES: Final[tuple[str, ...]] = (
    "env_factory",
    "project_paths",
    "start_lane_env",
    "configs",
    "configs.experiment_config",
)


def _check_known_host_origins(project_root: Path) -> None:
    """Reject loaded project modules whose origins are outside ``project_root``."""

    for module_name in _KNOWN_HOST_MODULES:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        origin = _module_path(module, module_name)
        if not _inside(origin, project_root):
            raise HostImportError(
                f"{module_name} is loaded from another project root: {origin}; "
                f"requested root={project_root}"
            )


@dataclass(frozen=True, slots=True)
class PrefixFeatureEvidence:
    """Evidence for one of the three host-provided start-lane fields.

    Shape alone is intentionally insufficient.  ``meaning``, ``encoding``
    and ``source`` are required so a caller cannot accidentally reinterpret an
    unrelated 262-wide vector as the requested schema.
    """

    index: int
    name: str
    meaning: str
    encoding: str
    source: str

    def __post_init__(self) -> None:
        if self.index not in PREFIX_FEATURE_INDICES:
            raise HostContractError(
                f"host prefix feature index must be one of {PREFIX_FEATURE_INDICES}: "
                f"{self.index!r}"
            )
        for field_name in ("name", "meaning", "encoding", "source"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise HostContractError(
                    f"prefix feature {field_name} must be a non-empty string"
                )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        expected_index: int | None = None,
    ) -> "PrefixFeatureEvidence":
        try:
            index_value = value["index"]
            name = value["name"]
            meaning = value["meaning"]
            encoding = value["encoding"]
            source = value["source"]
        except KeyError as error:
            raise HostContractError(
                f"prefix feature evidence missing key: {error.args[0]}"
            ) from error
        if isinstance(index_value, bool) or not isinstance(index_value, int):
            raise HostContractError("prefix feature evidence index must be int")
        if expected_index is not None and index_value != expected_index:
            raise HostContractError(
                f"prefix feature evidence index {index_value} does not match "
                f"expected {expected_index}"
            )
        return cls(
            index=index_value,
            name=str(name),
            meaning=str(meaning),
            encoding=str(encoding),
            source=str(source),
        )


def _coerce_feature_evidence(
    evidence: Mapping[int, PrefixFeatureEvidence | Mapping[str, object]]
    | Sequence[PrefixFeatureEvidence | Mapping[str, object]]
    | None,
) -> tuple[PrefixFeatureEvidence, ...]:
    if evidence is None:
        return ()
    values: list[PrefixFeatureEvidence] = []
    if isinstance(evidence, Mapping):
        for expected_index in PREFIX_FEATURE_INDICES:
            if expected_index not in evidence:
                raise HostContractError(
                    f"prefix feature evidence missing index {expected_index}"
                )
            item = evidence[expected_index]
            if isinstance(item, PrefixFeatureEvidence):
                feature = item
                if feature.index != expected_index:
                    raise HostContractError(
                        f"prefix feature evidence key/index mismatch: "
                        f"key={expected_index}, index={feature.index}"
                    )
            elif isinstance(item, Mapping):
                feature = PrefixFeatureEvidence.from_mapping(
                    item,
                    expected_index=expected_index,
                )
            else:
                raise HostContractError(
                    f"prefix feature evidence at {expected_index} must be mapping"
                )
            values.append(feature)
    else:
        if len(evidence) == 0:
            return ()
        if len(evidence) != len(PREFIX_FEATURE_INDICES):
            raise HostContractError(
                "prefix feature evidence must contain exactly three entries"
            )
        for expected_index, item in zip(PREFIX_FEATURE_INDICES, evidence):
            if isinstance(item, PrefixFeatureEvidence):
                feature = item
                if feature.index != expected_index:
                    raise HostContractError(
                        f"prefix feature evidence order/index mismatch: "
                        f"expected={expected_index}, actual={feature.index}"
                    )
            elif isinstance(item, Mapping):
                feature = PrefixFeatureEvidence.from_mapping(
                    item,
                    expected_index=expected_index,
                )
            else:
                raise HostContractError(
                    f"prefix feature evidence at {expected_index} must be mapping"
                )
            values.append(feature)

    by_index = {feature.index: feature for feature in values}
    if set(by_index) != set(PREFIX_FEATURE_INDICES):
        raise HostContractError("prefix feature evidence must cover indices 259..261")
    for index, expected_name in zip(PREFIX_FEATURE_INDICES, PREFIX_FEATURE_NAMES):
        if by_index[index].name != expected_name:
            raise HostContractError(
                f"prefix feature {index} has name {by_index[index].name!r}; "
                f"expected {expected_name!r}"
            )
    return tuple(by_index[index] for index in PREFIX_FEATURE_INDICES)


@dataclass(frozen=True, slots=True)
class ObservationContract:
    """Verified raw observation schema and the immutable Box bounds."""

    shape: tuple[int, ...]
    dtype: np.dtype
    low: np.ndarray = field(repr=False)
    high: np.ndarray = field(repr=False)
    prefix_features: tuple[PrefixFeatureEvidence, ...] = ()
    source: str = ""
    semantic_verified: bool = False

    def __post_init__(self) -> None:
        shape = tuple(int(value) for value in self.shape)
        object.__setattr__(self, "shape", shape)
        dtype = np.dtype(self.dtype)
        object.__setattr__(self, "dtype", dtype)
        low = np.array(self.low, dtype=dtype, copy=True)
        high = np.array(self.high, dtype=dtype, copy=True)
        low.setflags(write=False)
        high.setflags(write=False)
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)
        if shape != (BASELINE_OBS_DIM,):
            raise UnsupportedHostError(
                f"requested lookahead_learning host raw observation must have shape "
                f"({BASELINE_OBS_DIM},), found {shape}"
            )
        if dtype != np.dtype(np.float32):
            raise UnsupportedHostError(
                f"requested lookahead_learning host raw observation must use float32, "
                f"found {dtype}"
            )
        if low.shape != shape or high.shape != shape:
            raise HostContractError(
                f"observation Box bounds must have shape {shape}; "
                f"low={low.shape}, high={high.shape}"
            )
        if not np.isfinite(low).all() or not np.isfinite(high).all():
            raise HostContractError("observation Box bounds must be finite")
        if np.any(low > high):
            raise HostContractError("observation Box low must not exceed high")
        if self.semantic_verified:
            if len(self.prefix_features) != 3:
                raise UnsupportedHostError(
                    "semantic_verified raw schema requires evidence for indices 259..261"
                )
            _coerce_feature_evidence(self.prefix_features)

    @classmethod
    def from_space(
        cls,
        space: object,
        *,
        feature_evidence: Mapping[int, PrefixFeatureEvidence | Mapping[str, object]]
        | Sequence[PrefixFeatureEvidence | Mapping[str, object]]
        | None = None,
        source: str = "",
    ) -> "ObservationContract":
        from gymnasium.spaces import Box

        if not isinstance(space, Box):
            raise UnsupportedHostError(
                "lookahead_learning requires a one-dimensional gymnasium.spaces.Box "
                f"raw observation space, found {type(space).__name__}"
            )
        shape = tuple(getattr(space, "shape", ()) or ())
        dtype = np.dtype(getattr(space, "dtype", object))
        low = np.asarray(getattr(space, "low", np.array([], dtype=dtype)))
        high = np.asarray(getattr(space, "high", np.array([], dtype=dtype)))
        prefix = _coerce_feature_evidence(feature_evidence)
        if not prefix:
            raise UnsupportedHostError(
                "raw shape alone is insufficient: provide verified semantic "
                "evidence for existing start-lane features at indices 259..261"
            )
        return cls(
            shape=shape,
            dtype=dtype,
            low=low,
            high=high,
            prefix_features=prefix,
            source=source,
            semantic_verified=bool(prefix),
        )

    def validate_raw(self, observation: object) -> np.ndarray:
        """Validate without padding, truncation, flattening, or re-normalizing."""

        array = np.asarray(observation)
        if tuple(array.shape) != self.shape:
            raise UnsupportedHostError(
                f"raw observation shape changed from {self.shape} to {array.shape}"
            )
        if array.dtype != self.dtype:
            raise UnsupportedHostError(
                f"raw observation dtype changed from {self.dtype} to {array.dtype}"
            )
        if not np.isfinite(array).all():
            raise HostContractError("raw observation contains NaN or Inf")
        if np.any(array < self.low) or np.any(array > self.high):
            raise HostContractError("raw observation is outside the host Box bounds")
        return array

    def augmented_space(self):
        """Build the augmented observation Box while preserving raw bounds."""

        from gymnasium.spaces import Box

        low = np.concatenate(
            (self.low, np.zeros(PREVIEW_FEATURE_DIM, dtype=self.dtype))
        )
        high = np.concatenate(
            (self.high, np.ones(PREVIEW_FEATURE_DIM, dtype=self.dtype))
        )
        return Box(low=low, high=high, dtype=self.dtype)

    def append_preview(self, raw_observation: object, preview_values: Sequence[float]) -> np.ndarray:
        raw = self.validate_raw(raw_observation)
        values = np.asarray(preview_values, dtype=np.float32)
        if values.shape != (PREVIEW_FEATURE_DIM,):
            raise HostContractError(
                f"preview values must have shape ({PREVIEW_FEATURE_DIM},), "
                f"found {values.shape}"
            )
        if not np.isfinite(values).all():
            raise HostContractError("preview values contain NaN or Inf")
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise HostContractError("preview values must be in [0, 1]")
        augmented = np.concatenate((raw, values)).astype(np.float32, copy=False)
        if augmented.shape != (AUGMENTED_OBS_DIM,):
            raise AssertionError("unexpected augmented observation shape")
        return augmented


@dataclass(frozen=True, slots=True)
class ActionContract:
    """Single-agent action-space and discrete decoding contract."""

    n: int
    steering_dim: int
    throttle_dim: int
    use_multi_discrete: bool = False
    source: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("n", self.n),
            ("steering_dim", self.steering_dim),
            ("throttle_dim", self.throttle_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise HostContractError(f"{name} must be a positive integer")
        if self.use_multi_discrete:
            raise UnsupportedHostError(
                "lookahead_learning currently requires the single Discrete action path"
            )
        if self.n != self.steering_dim * self.throttle_dim:
            raise HostContractError(
                "Discrete action count does not match configured steering/throttle "
                f"dimensions: n={self.n}, product={self.steering_dim * self.throttle_dim}"
            )
        if self.steering_dim < 2 or self.throttle_dim < 2:
            raise UnsupportedHostError(
                "lookahead_learning requires at least two steering and throttle levels"
            )

    @classmethod
    def from_space(
        cls,
        space: object,
        config: object,
        *,
        source: str = "",
    ) -> "ActionContract":
        from gymnasium.spaces import Discrete

        if not isinstance(space, Discrete):
            raise UnsupportedHostError(
                f"lookahead_learning requires a single gymnasium.spaces.Discrete action "
                f"space, found {type(space).__name__}"
            )
        # Gymnasium's non-zero ``start`` changes the integer values delivered
        # to the policy.  EnvInputPolicy's verified decoder is defined for
        # the canonical 0..n-1 action IDs only.
        if int(getattr(space, "start", 0)) != 0:
            raise UnsupportedHostError(
                "lookahead_learning requires Discrete(start=0) for the verified "
                "EnvInputPolicy decoding"
            )
        config_mapping = _host_config_mapping(config, name="environment config")
        steering_dim = config_mapping.get("discrete_steering_dim")
        throttle_dim = config_mapping.get("discrete_throttle_dim")
        if isinstance(steering_dim, bool) or not isinstance(steering_dim, int):
            raise HostContractError("discrete_steering_dim must be an int")
        if isinstance(throttle_dim, bool) or not isinstance(throttle_dim, int):
            raise HostContractError("discrete_throttle_dim must be an int")
        return cls(
            n=int(space.n),
            steering_dim=steering_dim,
            throttle_dim=throttle_dim,
            use_multi_discrete=bool(config_mapping.get("use_multi_discrete", False)),
            source=source,
        )

    def decode(self, action: object) -> tuple[float, float]:
        """Decode host EnvInputPolicy's row-major discrete action to normalized values."""

        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise HostContractError(f"Discrete action must be an integer: {action!r}")
        action_int = int(action)
        if not 0 <= action_int < self.n:
            raise HostContractError(
                f"Discrete action {action_int} outside [0, {self.n})"
            )
        steering = action_int % self.steering_dim
        throttle = action_int // self.steering_dim
        steering_value = 2.0 * steering / (self.steering_dim - 1) - 1.0
        throttle_value = 2.0 * throttle / (self.throttle_dim - 1) - 1.0
        return float(steering_value), float(throttle_value)


def get_single_agent(env: object) -> object:
    """Return the one active agent without importing a MetaDrive class."""

    agents = getattr(env, "agents", None)
    if isinstance(agents, Mapping):
        if len(agents) != 1:
            raise UnsupportedHostError(
                f"lookahead_learning requires one active agent, found {len(agents)}"
            )
        return next(iter(agents.values()))
    agent = getattr(env, "agent", None)
    if agent is None:
        raise HostContractError("environment exposes neither agents nor agent")
    return agent


def _read_applied_action_pair(env: object) -> tuple[float, float]:
    """Read the normalized action pair stored by the verified host vehicle."""

    vehicle = get_single_agent(env)
    current = getattr(vehicle, "current_action", None)
    if current is None:
        raise UnsupportedHostError(
            "cannot identify the normalized applied action; provide an explicit "
            "applied action reader"
        )
    try:
        steering_value = current[0]
        throttle_value = current[1]
    except (IndexError, KeyError, TypeError):
        raise UnsupportedHostError(
            "host current_action must expose steering and throttle components"
        ) from None
    steering = _finite_number(steering_value, name="applied steering")
    throttle = _finite_number(throttle_value, name="applied throttle")
    if not -1.0 <= steering <= 1.0:
        raise HostContractError(
            f"applied normalized steering must be in [-1, 1], found {steering}"
        )
    if not -1.0 <= throttle <= 1.0:
        raise HostContractError(
            f"applied normalized throttle must be in [-1, 1], found {throttle}"
        )
    return steering, throttle


def read_applied_action(env: object) -> tuple[float, float]:
    """Read both normalized commands stored by the host policy.

    In the verified MetaDrive path this is ``vehicle.current_action``:
    ``EnvInputPolicy.act`` decodes/clips the discrete action once and
    ``BaseVehicle.before_step`` stores that normalized pair before the repeated
    physics steps.  A host with a different action path must provide a custom
    reader instead of being silently interpreted here.
    """

    return _read_applied_action_pair(env)


def read_applied_steering(env: object) -> float:
    """Read the normalized steering command stored by the host policy.

    In the verified MetaDrive path this is ``vehicle.current_action[0]``:
    ``EnvInputPolicy.act`` decodes/clips the action once and
    ``BaseVehicle.before_step`` stores that normalized pair before the repeated
    physics steps.  A host with a different action path must provide a custom
    reader instead of being silently interpreted here.
    """

    return _read_applied_action_pair(env)[0]


def simulation_dt_seconds(env: object) -> float:
    """Return fixed decision duration from the host's physics configuration."""

    config = _host_config_mapping(
        getattr(env, "config", None),
        name="env.config",
    )
    physics_step = _finite_number(
        config.get("physics_world_step_size"),
        name="physics_world_step_size",
    )
    repeat = config.get("decision_repeat")
    if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat <= 0:
        raise HostContractError("decision_repeat must be a positive integer")
    dt = physics_step * repeat
    if dt <= 0.0 or not math.isfinite(dt):
        raise HostContractError("computed simulation decision dt must be positive")
    return dt


def read_vehicle_state(env: object) -> dict[str, object]:
    """Read common physical state scalars without stepping or observing.

    This reader is deliberately small and uses only properties exposed by the
    verified MetaDrive vehicle.  Optional values are omitted when a host does
    not expose them; callers that need a project-specific route metric can
    supply a stricter ``state_reader`` to :class:`lookahead_learning.env.LookaheadEnv`.
    """

    vehicle = get_single_agent(env)
    state: dict[str, object] = {}
    position = getattr(vehicle, "position", None)
    if position is not None:
        try:
            if len(position) >= 2:
                px = _finite_number(position[0], name="vehicle.position[0]")
                py = _finite_number(position[1], name="vehicle.position[1]")
                state["position_xy"] = (px, py)
        except (TypeError, IndexError):
            pass
    heading_theta = getattr(vehicle, "heading_theta", None)
    if heading_theta is not None:
        state["heading_theta"] = _finite_number(
            heading_theta,
            name="vehicle.heading_theta",
        )
    else:
        heading = getattr(vehicle, "heading", None)
        try:
            if heading is not None and len(heading) >= 2:
                hx = _finite_number(heading[0], name="vehicle.heading[0]")
                hy = _finite_number(heading[1], name="vehicle.heading[1]")
                state["heading_theta"] = math.atan2(hy, hx)
        except (TypeError, IndexError):
            pass
    speed = getattr(vehicle, "speed", None)
    if speed is None:
        velocity = getattr(vehicle, "velocity", None)
        try:
            if velocity is not None and len(velocity) >= 2:
                vx = _finite_number(velocity[0], name="vehicle.velocity[0]")
                vy = _finite_number(velocity[1], name="vehicle.velocity[1]")
                speed = math.hypot(vx, vy)
        except (TypeError, IndexError):
            speed = None
    if speed is not None:
        state["speed_m_s"] = _finite_number(speed, name="vehicle.speed")
    navigation = getattr(vehicle, "navigation", None)
    if navigation is not None:
        travelled = getattr(navigation, "travelled_length", None)
        if travelled is not None:
            state["travelled_length_m"] = _finite_number(
                travelled,
                name="navigation.travelled_length",
            )
        current_refs = getattr(navigation, "current_ref_lanes", None)
        if current_refs is not None:
            state["current_ref_lane_ids"] = tuple(
                str(getattr(lane, "index", lane)) for lane in current_refs
            )
        checkpoints = getattr(navigation, "checkpoints", None)
        if checkpoints is not None:
            state["checkpoints"] = tuple(str(item) for item in checkpoints)
    lane_index = getattr(vehicle, "lane_index", None)
    if lane_index is not None:
        state["lane_index"] = _freeze_for_state(lane_index)
    try:
        state["scenario_seed"] = int(getattr(env, "current_seed"))
    except (AttributeError, TypeError, ValueError, RuntimeError):
        pass
    return state


def _freeze_for_state(value: object) -> object:
    """Local immutable conversion for adapter state values."""

    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, Mapping):
        return tuple(
            (str(key), _freeze_for_state(item))
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_for_state(item) for item in value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (str, bool)) or value is None:
        return value
    return str(value)


def _vehicle_from_env_or_vehicle(value: object) -> object:
    """Resolve either a raw environment or its already selected vehicle."""

    if getattr(value, "navigation", None) is not None:
        return value
    return get_single_agent(value)


def _lane_ordinal(lane: object, *, context: str) -> int:
    """Read the final lane-index component used only for route candidates."""

    lane_index = getattr(lane, "index", None)
    if lane_index is None:
        raise UnsupportedHostError(f"{context} lane has no index tuple")
    try:
        ordinal = lane_index[-1]
    except (IndexError, KeyError, TypeError):
        raise UnsupportedHostError(f"{context} lane index has no ordinal") from None
    if isinstance(ordinal, bool) or not isinstance(ordinal, (int, np.integer)):
        raise UnsupportedHostError(f"{context} lane ordinal must be an integer")
    return int(ordinal)


def _navigation_edge_lanes(graph: object, start: object, end: object) -> tuple[object, ...]:
    """Read one directed NodeRoadNetwork edge without searching alternatives."""

    try:
        first = graph[start]  # type: ignore[index]
        lanes = first[end]  # type: ignore[index]
    except (KeyError, IndexError, TypeError, AttributeError) as error:
        raise UnsupportedHostError(
            f"navigation route has no directed edge {start!r}->{end!r}"
        ) from error
    if isinstance(lanes, (str, bytes)):
        raise UnsupportedHostError(
            f"navigation edge {start!r}->{end!r} does not expose lane objects"
        )
    try:
        lane_tuple = tuple(lanes)
    except TypeError as error:
        raise UnsupportedHostError(
            f"navigation edge {start!r}->{end!r} does not expose a lane sequence"
        ) from error
    if not lane_tuple:
        raise UnsupportedHostError(
            f"navigation edge {start!r}->{end!r} has no lanes"
        )
    return lane_tuple


def _host_containers(value: object) -> tuple[object, ...]:
    """Return raw/wrapper containers that may retain start-lane evidence."""

    values = [value]
    unwrapped = getattr(value, "unwrapped", None)
    if unwrapped is not None and unwrapped is not value:
        values.append(unwrapped)
    return tuple(values)


def _coerce_ordinal(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise UnsupportedHostError(f"{name} must be an integer")
    return int(value)


def _saved_start_lane_ordinal(env_or_vehicle: object, vehicle: object) -> int | None:
    """Read reset-time ordinal evidence retained by the start-lane host.

    ``StartLaneMetaDriveEnv`` stores one ordinal per agent in
    ``_target_lane_ordinals``.  That value has priority over the vehicle's
    current ``lane_index`` because Navigation can move to a later road before
    a provider is asked to rebuild diagnostics.  A plain host without this
    project extension may expose ``start_lane_ordinal`` directly; otherwise
    callers are expected to invoke the builder immediately after reset.
    """

    containers = _host_containers(env_or_vehicle)
    if getattr(env_or_vehicle, "navigation", None) is not None:
        containers = (env_or_vehicle,)
    for container in containers:
        target_map = getattr(container, "_target_lane_ordinals", None)
        if isinstance(target_map, Mapping) and target_map:
            keys: list[object] = []
            agents = getattr(container, "agents", None)
            if isinstance(agents, Mapping):
                keys.extend(key for key, item in agents.items() if item is vehicle)
            for attr_name in ("id", "agent_id", "name"):
                item = getattr(vehicle, attr_name, None)
                if item is not None:
                    keys.append(item)
            keys.extend(("agent0", "default", None))
            for key in keys:
                if key in target_map:
                    value = target_map[key]
                    if value is None:
                        raise UnsupportedHostError(
                            "host retained no reset start-lane ordinal for the active agent"
                        )
                    return _coerce_ordinal(value, name="saved start-lane ordinal")
            if len(target_map) == 1:
                value = next(iter(target_map.values()))
                if value is None:
                    raise UnsupportedHostError(
                        "host retained no reset start-lane ordinal"
                    )
                return _coerce_ordinal(value, name="saved start-lane ordinal")
            raise UnsupportedHostError(
                "host start-lane ordinal map has no entry for the active agent"
            )
        for attr_name in (
            "start_lane_ordinal",
            "target_lane_ordinal",
            "_start_lane_ordinal",
        ):
            value = getattr(container, attr_name, None)
            if value is not None:
                return _coerce_ordinal(value, name=attr_name)
        for attr_name in ("start_lane_ordinal", "target_lane_ordinal"):
            value = getattr(vehicle, attr_name, None)
            if value is not None:
                return _coerce_ordinal(value, name=attr_name)
    return None


def read_target_lane_state(
    env: object,
    *,
    project_root: str | Path | None = None,
    tolerance_ratio: float | None = None,
) -> object:
    """Read the audited host start-lane state without stepping or re-observing.

    The current project stores the reset ordinal in
    ``StartLaneMetaDriveEnv._target_lane_ordinals`` and exposes the pure
    ``resolve_target_lane_state`` function in ``start_lane_env.py``.  This
    helper imports that function normally at call time, passes the preserved
    reset ordinal, and returns the host ``TargetLaneState`` object.  It never
    calls ``observe``, ``step``, ``reward_function``, or mutates the host.

    ``project_root`` should be supplied by a relocated runner so the flat
    project module's origin is checked before use.  Omitting it retains the
    ordinary import behavior for an already configured host process.
    """

    vehicle = get_single_agent(env)
    target_ordinal = _saved_start_lane_ordinal(env, vehicle)
    if target_ordinal is None:
        raise UnsupportedHostError(
            "host start-lane reset ordinal is unavailable; refusing to infer "
            "target geometry from the current lane"
        )
    if tolerance_ratio is None:
        config = _host_config_mapping(
            getattr(env, "config", None),
            name="env.config",
        )
        tolerance_ratio = config.get("start_lane_tolerance_ratio")
    tolerance = _finite_number(
        tolerance_ratio,
        name="start_lane_tolerance_ratio",
    )
    if tolerance < 0.0:
        raise HostContractError("start_lane_tolerance_ratio must be non-negative")

    if project_root is None:
        try:
            module = importlib.import_module("start_lane_env")
        except Exception as error:
            raise HostImportError(
                "failed to import start_lane_env for target-lane diagnostics: "
                f"{type(error).__name__}: {error}"
            ) from error
    else:
        module = import_host_module("start_lane_env", project_root=project_root)
    resolver = getattr(module, "resolve_target_lane_state", None)
    if not callable(resolver):
        raise HostImportError(
            "start_lane_env.resolve_target_lane_state is unavailable in the "
            "requested host"
        )
    try:
        return resolver(
            vehicle,
            target_ordinal=target_ordinal,
            tolerance_ratio=tolerance,
        )
    except (HostContractError, UnsupportedHostError):
        raise
    except Exception as error:
        # Resolver failures are host/program errors, not a valid per-step
        # invalid reference.  Keep the original type and message visible.
        raise HostContractError(
            "start-lane state resolver failed: "
            f"{type(error).__name__}: {error}"
        ) from error


def _lane_width_at(lane: object, longitudinal: float, *, context: str) -> float:
    width_at = getattr(lane, "width_at", None)
    if callable(width_at):
        value = width_at(longitudinal)
    else:
        value = getattr(lane, "width", getattr(lane, "width_m", None))
    width = _finite_number(value, name=f"{context} width")
    if width <= 0.0:
        raise UnsupportedHostError(f"{context} width must be positive")
    return width


def _lane_descriptor(
    lane: object,
    lane_count: int,
    *,
    edge_index: int,
    width_tolerance_m: float,
) -> dict[str, object]:
    """Build strict geometry metadata for one selected/candidate lane."""

    lane_type = type(lane).__name__
    if lane_type not in {"StraightLane", "CircularLane"}:
        raise UnsupportedHostError(
            f"edge {edge_index} uses unsupported lane type {lane_type!r}; "
            "only StraightLane and CircularLane are verified"
        )
    length_value = getattr(lane, "length", None)
    if callable(length_value):
        length_value = length_value()
    length = _finite_number(length_value, name=f"edge {edge_index} lane length")
    if length <= 0.0:
        raise UnsupportedHostError(f"edge {edge_index} lane length must be positive")
    width_start = _lane_width_at(lane, 0.0, context=f"edge {edge_index} lane")
    width_end = _lane_width_at(lane, length, context=f"edge {edge_index} lane")
    if abs(width_end - width_start) > width_tolerance_m:
        raise UnsupportedHostError(
            f"edge {edge_index} lane width changes by {abs(width_end - width_start)} m"
        )
    return {
        "lane_type": lane_type,
        "lane_count": int(lane_count),
        "width_m": width_start,
        "lane_id": str(getattr(lane, "index", edge_index)),
        "route_key": str(edge_index),
    }


def _navigation_error_reason(error: Exception) -> str:
    """Classify a route collection failure without hiding its message."""

    text = str(error).lower()
    if "unsupported lane type" in text:
        return "unsupported_lane_type"
    if "width changes" in text or "width mismatch" in text:
        return "lane_width_mismatch"
    if "length" in text:
        return "invalid_lane_geometry"
    return "navigation_lane_unavailable"


def _navigation_context(
    env_or_vehicle: object,
    *,
    allow_partial: bool,
    endpoint_tolerance_m: float,
    width_tolerance_m: float,
    tangent_tolerance_rad: float,
) -> tuple[
    object,
    tuple[object, ...],
    int,
    tuple[object, ...],
    dict[int, dict[str, object]],
    tuple[str, int, str] | None,
]:
    """Collect a planned lane prefix and preserve the first unresolved edge."""

    vehicle = _vehicle_from_env_or_vehicle(env_or_vehicle)
    navigation = getattr(vehicle, "navigation", None)
    if navigation is None:
        raise UnsupportedHostError("vehicle exposes no Navigation module")
    checkpoints_value = getattr(navigation, "checkpoints", None)
    if checkpoints_value is None or isinstance(checkpoints_value, (str, bytes)):
        raise UnsupportedHostError("Navigation.checkpoints is unavailable after reset")
    try:
        checkpoints = tuple(checkpoints_value)
    except TypeError as error:
        raise UnsupportedHostError("Navigation.checkpoints is not a sequence") from error
    if len(checkpoints) < 2:
        raise UnsupportedHostError(
            "Navigation.checkpoints must contain at least one directed road"
        )
    target_ordinal = _saved_start_lane_ordinal(env_or_vehicle, vehicle)
    if target_ordinal is None:
        lane_index = getattr(vehicle, "lane_index", None)
        if lane_index is None:
            raise UnsupportedHostError(
                "vehicle.lane_index is unavailable; start-lane ordinal cannot be verified"
            )
        try:
            target_ordinal = _coerce_ordinal(lane_index[-1], name="vehicle lane ordinal")
        except (IndexError, KeyError, TypeError):
            raise UnsupportedHostError(
                "vehicle.lane_index has no final lane ordinal"
            ) from None
    map_object = getattr(navigation, "map", None)
    road_network = getattr(map_object, "road_network", None)
    graph = getattr(road_network, "graph", None)
    if graph is None:
        raise UnsupportedHostError(
            "Navigation map does not expose a directed road-network graph"
        )

    selected: list[object] = []
    metadata_by_identity: dict[int, dict[str, object]] = {}
    boundary: tuple[str, int, str] | None = None
    for edge_index, (start, end) in enumerate(zip(checkpoints[:-1], checkpoints[1:])):
        try:
            candidates = _navigation_edge_lanes(graph, start, end)
        except UnsupportedHostError as error:
            boundary = ("navigation_edge_unavailable", edge_index, str(error))
            break
        matching: list[object] = []
        candidate_index_error: UnsupportedHostError | None = None
        for candidate in candidates:
            try:
                if _lane_ordinal(candidate, context=f"edge {edge_index}") == target_ordinal:
                    matching.append(candidate)
            except UnsupportedHostError as error:
                candidate_index_error = error
                break
        if candidate_index_error is not None:
            boundary = ("navigation_lane_index_unavailable", edge_index, str(candidate_index_error))
            break
        if len(matching) != 1:
            boundary = (
                "ambiguous_start_lane_ordinal"
                if len(matching) > 1
                else "missing_start_lane_ordinal",
                edge_index,
                f"navigation edge {start!r}->{end!r} has {len(matching)} "
                f"lanes with start ordinal {target_ordinal}; expected exactly one",
            )
            break
        selected_lane = matching[0]
        try:
            descriptor = _lane_descriptor(
                selected_lane,
                len(candidates),
                edge_index=edge_index,
                width_tolerance_m=width_tolerance_m,
            )
        except UnsupportedHostError as error:
            boundary = (_navigation_error_reason(error), edge_index, str(error))
            break

        if selected:
            # A lane ordinal is only a candidate lookup key.  The directed
            # geometry must identify one continuation among *all* lanes on the
            # next road, including lanes with another ordinal.
            from .geometry import LaneMetadata, connection_measurement

            previous = selected[-1]
            previous_descriptor = metadata_by_identity[id(previous)]
            previous_metadata = LaneMetadata(
                str(previous_descriptor["lane_type"]),
                int(previous_descriptor["lane_count"]),
                float(previous_descriptor["width_m"]),
                str(previous_descriptor["lane_id"]),
                str(previous_descriptor["route_key"]),
            )
            connected: list[object] = []
            for candidate_index, candidate in enumerate(candidates):
                try:
                    candidate_descriptor = _lane_descriptor(
                        candidate,
                        len(candidates),
                        edge_index=edge_index,
                        width_tolerance_m=width_tolerance_m,
                    )
                    candidate_metadata = LaneMetadata(
                        str(candidate_descriptor["lane_type"]),
                        int(candidate_descriptor["lane_count"]),
                        float(candidate_descriptor["width_m"]),
                        str(candidate_descriptor["lane_id"]),
                        str(candidate_descriptor["route_key"]),
                    )
                    measurement = connection_measurement(
                        previous,
                        candidate,
                        previous_metadata=previous_metadata,
                        next_metadata=candidate_metadata,
                        width_tolerance_m=width_tolerance_m,
                    )
                except UnsupportedHostError as error:
                    # A lane type/width that cannot be inspected on a
                    # directed successor is an unsupported boundary.  Treat
                    # it conservatively even when its ordinal differs; using
                    # the chosen ordinal alone would hide a branch/merge.
                    boundary = (
                        "unsupported_successor_type",
                        edge_index,
                        str(error),
                    )
                    break
                except (TypeError, ValueError, AttributeError):
                    continue
                if (
                    measurement.endpoint_distance_m <= endpoint_tolerance_m
                    and measurement.tangent_error_rad <= tangent_tolerance_rad
                    and measurement.lane_count_match is not False
                    and measurement.width_match is not False
                ):
                    connected.append(candidate)
            if boundary is not None:
                break
            if len(connected) > 1:
                boundary = (
                    "ambiguous_successor",
                    edge_index,
                    f"{len(connected)} directed lane continuations satisfy "
                    "endpoint/tangent/count/width checks",
                )
                break
            if connected and connected[0] is not selected_lane:
                boundary = (
                    "start_lane_ordinal_not_continuous",
                    edge_index,
                    "the selected start-lane ordinal is not the unique geometric "
                    "continuation of the planned directed route",
                )
                break

        selected.append(selected_lane)
        metadata_by_identity[id(selected_lane)] = descriptor

    if boundary is not None and not allow_partial:
        raise UnsupportedHostError(boundary[2])
    return (
        vehicle,
        tuple(selected),
        target_ordinal,
        checkpoints,
        metadata_by_identity,
        boundary,
    )


def navigation_lane_sequence(
    env_or_vehicle: object,
) -> tuple[tuple[object, ...], int, tuple[object, ...]]:
    """Select one fixed lane per planned Navigation road after reset.

    The function is strict and raises on an unresolved edge.  Use
    :func:`build_fixed_navigation_route` when a diagnostic route prefix should
    be retained instead of discarded at the first unresolved boundary.
    """

    _, lanes, target_ordinal, checkpoints, _, _ = _navigation_context(
        env_or_vehicle,
        allow_partial=False,
        endpoint_tolerance_m=1.0e-3,
        width_tolerance_m=1.0e-3,
        tangent_tolerance_rad=1.0e-3,
    )
    return lanes, target_ordinal, checkpoints


def build_fixed_navigation_route(
    env_or_vehicle: object,
    *,
    endpoint_tolerance_m: float = 1.0e-3,
    width_tolerance_m: float = 1.0e-3,
    tangent_tolerance_rad: float = 1.0e-3,
) -> object:
    """Build the fixed directed route from the post-reset Navigation state.

    The returned object is :class:`lookahead_learning.geometry.RouteBuildResult`.  A
    validated prefix and a diagnostic boundary are retained when a later
    lane is unsupported or disconnected; geometry then marks lookahead that
    crosses the boundary invalid.  This function imports the pure geometry
    module lazily so importing or displaying CLI help never imports a simulator.
    """

    (
        _vehicle,
        lanes,
        _target_ordinal,
        _checkpoints,
        edge_metadata,
        boundary,
    ) = _navigation_context(
        env_or_vehicle,
        allow_partial=True,
        endpoint_tolerance_m=endpoint_tolerance_m,
        width_tolerance_m=width_tolerance_m,
        tangent_tolerance_rad=tangent_tolerance_rad,
    )

    from .geometry import build_reference_route, RouteIssue

    def metadata(lane: object) -> Mapping[str, object]:
        descriptor = edge_metadata.get(id(lane))
        if descriptor is None:
            raise UnsupportedHostError(
                "fixed navigation route encountered a lane outside its planned "
                "directed edge sequence"
            )
        return descriptor

    result = build_reference_route(
        lanes,
        metadata=metadata,
        endpoint_tolerance_m=endpoint_tolerance_m,
        width_tolerance_m=width_tolerance_m,
        tangent_tolerance_rad=tangent_tolerance_rad,
    )
    if boundary is None or result.diagnostics.boundary_reason not in {
        "route_end",
        "empty_route",
    }:
        return result
    reason, boundary_index, message = boundary
    diagnostics = replace(
        result.diagnostics,
        total_lanes=len(_checkpoints) - 1,
        boundary_reason=reason,
        boundary_lane_index=boundary_index,
        issues=tuple(
            [
                *result.diagnostics.issues,
                RouteIssue(reason, boundary_index, message),
            ]
        ),
    )
    path = replace(result.path, diagnostics=diagnostics)
    return replace(result, path=path, diagnostics=diagnostics)


@dataclass(frozen=True, slots=True)
class HostAudit:
    """Read-only audit result suitable for doctor output."""

    project_root: Path
    imported_module_origins: Mapping[str, str] = field(default_factory=dict)
    raw_shape: tuple[int, ...] | None = None
    raw_dtype: str | None = None
    action_space: str | None = None
    dt_seconds: float | None = None
    supported_baseline_schema: bool = False
    semantic_evidence: bool = False
    issues: tuple[str, ...] = ()

    @property
    def supported(self) -> bool:
        return self.supported_baseline_schema and not self.issues

    def as_dict(self) -> dict[str, object]:
        return {
            "project_root": str(self.project_root),
            "imported_module_origins": dict(self.imported_module_origins),
            "raw_shape": list(self.raw_shape) if self.raw_shape is not None else None,
            "raw_dtype": self.raw_dtype,
            "action_space": self.action_space,
            "dt_seconds": self.dt_seconds,
            "supported_baseline_schema": self.supported_baseline_schema,
            "semantic_evidence": self.semantic_evidence,
            "supported": self.supported,
            "issues": list(self.issues),
        }


@dataclass(slots=True)
class HostAdapter:
    """Explicit host connection used by the portable runner.

    The adapter owns no environment instance.  ``make_raw_env`` constructs one
    only when called by the runner, which keeps module imports and doctor static
    checks free of simulator side effects.
    """

    project_root: Path
    semantic_evidence: tuple[PrefixFeatureEvidence, ...] = ()
    _factory: Callable[[Mapping[str, object]], object] | None = field(
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.project_root = _normalise_root(self.project_root)
        self.semantic_evidence = _coerce_feature_evidence(self.semantic_evidence)

    @classmethod
    def for_project(
        cls,
        project_root: str | Path,
        *,
        semantic_evidence: Mapping[int, PrefixFeatureEvidence | Mapping[str, object]]
        | Sequence[PrefixFeatureEvidence | Mapping[str, object]]
        | None = None,
    ) -> "HostAdapter":
        return cls(
            project_root=_normalise_root(project_root),
            semantic_evidence=_coerce_feature_evidence(semantic_evidence),
        )

    def load_factory(self) -> Callable[[Mapping[str, object]], object]:
        if self._factory is None:
            module = import_host_module(
                "env_factory",
                project_root=self.project_root,
            )
            _check_known_host_origins(self.project_root)
            factory = getattr(module, "make_env", None)
            if not callable(factory):
                raise HostImportError(
                    f"env_factory.make_env is not callable in {self.project_root}"
                )
            self._factory = factory
        return self._factory

    def make_raw_env(self, env_config: Mapping[str, object]) -> object:
        """Construct exactly one raw host environment on explicit request."""

        factory = self.load_factory()
        # MetaDrive merges nested configuration tables in place.  Give the
        # host factory a deep copy so a run cannot mutate a shared TOML object.
        with _temporary_import_root(self.project_root):
            env = factory(copy.deepcopy(dict(env_config)))
        # ``env_factory`` may import ``start_lane_env`` lazily when the
        # configuration selects that subclass.  Check after construction too,
        # while retaining the temporary import-root context for the factory
        # call above.
        _check_known_host_origins(self.project_root)
        return env

    def inspect_env(self, env: object) -> tuple[ObservationContract, ActionContract, float]:
        """Inspect an existing raw env without reset/step calls."""

        source = f"{type(env).__module__}.{type(env).__qualname__}"
        observation_contract = ObservationContract.from_space(
            getattr(env, "observation_space", None),
            feature_evidence=self.semantic_evidence or None,
            source=source,
        )
        config = _host_config_mapping(
            getattr(env, "config", None),
            name="raw environment config",
        )
        action_contract = ActionContract.from_space(
            getattr(env, "action_space", None),
            config,
            source=source,
        )
        dt = simulation_dt_seconds(env)
        return observation_contract, action_contract, dt

    def audit_env(self, env: object) -> HostAudit:
        """Create a non-throwing doctor report for an already constructed env."""

        issues: list[str] = []
        shape: tuple[int, ...] | None = None
        dtype: str | None = None
        action_space_name: str | None = None
        dt: float | None = None
        schema_ok = False
        semantic_ok = bool(self.semantic_evidence)
        try:
            space = getattr(env, "observation_space")
            shape = tuple(getattr(space, "shape", ()) or ())
            dtype = str(getattr(space, "dtype", None))
            action_space_name = type(getattr(env, "action_space")).__name__
        except Exception as error:
            issues.append(f"space inspection failed: {type(error).__name__}: {error}")
        if shape != (BASELINE_OBS_DIM,):
            issues.append(
                "raw observation schema mismatch: "
                f"expected ({BASELINE_OBS_DIM},), found {shape}"
            )
        elif dtype != str(np.dtype(np.float32)):
            issues.append(
                f"raw observation dtype mismatch: expected float32, found {dtype}"
            )
        else:
            schema_ok = True
        if not self.semantic_evidence:
            semantic_ok = False
            issues.append(
                "indices 259..261 semantic evidence is absent; shape-only support "
                "is refused for operational modes"
            )
        try:
            dt = simulation_dt_seconds(env)
        except Exception as error:
            issues.append(f"simulation dt unavailable: {type(error).__name__}: {error}")
        try:
            ActionContract.from_space(
                getattr(env, "action_space"),
                getattr(env, "config"),
            )
        except Exception as error:
            issues.append(f"action contract invalid: {type(error).__name__}: {error}")
        try:
            _check_known_host_origins(self.project_root)
            origins = {
                name: str(_module_path(sys.modules[name], name))
                for name in _KNOWN_HOST_MODULES
                if name in sys.modules
            }
        except HostImportError as error:
            origins = {}
            issues.append(str(error))
        return HostAudit(
            project_root=self.project_root,
            imported_module_origins=origins,
            raw_shape=shape,
            raw_dtype=dtype,
            action_space=action_space_name,
            dt_seconds=dt,
            supported_baseline_schema=schema_ok,
            semantic_evidence=semantic_ok,
            issues=tuple(issues),
        )


def normalize_preview_coordinate(value_m: float) -> float:
    """Encode a finite metric in metres to the requested [0, 1] value."""

    value = _finite_number(value_m, name="preview coordinate")
    clipped = min(max(value / NORMALIZATION_DISTANCE_M, -1.0), 1.0)
    return (clipped + 1.0) / 2.0


def invalid_preview_values() -> tuple[float, float, float]:
    """The specified finite observation encoding for an invalid reference."""

    return 0.5, 0.5, 0.0


__all__ = [
    "ActionContract",
    "AUGMENTED_OBS_DIM",
    "BASELINE_OBS_DIM",
    "HostAdapter",
    "HostAudit",
    "HostContractError",
    "HostImportError",
    "CANONICAL_MODES",
    "MODE_ALIASES",
    "MODE_CHOICES",
    "Mode",
    "NORMALIZATION_DISTANCE_M",
    "ObservationContract",
    "PREVIEW_FEATURE_DIM",
    "PREFIX_FEATURE_INDICES",
    "PREFIX_FEATURE_NAMES",
    "PrefixFeatureEvidence",
    "UnsupportedHostError",
    "get_single_agent",
    "build_fixed_navigation_route",
    "navigation_lane_sequence",
    "import_host_module",
    "invalid_preview_values",
    "normalize_preview_coordinate",
    "normalize_mode",
    "read_applied_steering",
    "read_applied_action",
    "read_target_lane_state",
    "read_vehicle_state",
    "simulation_dt_seconds",
]
