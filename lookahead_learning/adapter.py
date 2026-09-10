"""Host boundary and runtime helpers for lookahead learning.

The lookahead core treats the host observation as an opaque one-dimensional
``float32`` vector.  The host adapter validates its concrete shape, dtype and
Box bounds, then appends the three preview values without interpreting or
padding the host prefix.  Route, vehicle and action meaning is checked at the
host boundary where those objects are available.

No MetaDrive module is imported at module import time.  The adapter can
therefore be imported on a machine where simulator dependencies are absent;
environment construction is explicit and is never performed by an import.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
import importlib
import math
from typing import TYPE_CHECKING, Final, Literal, TypeAlias, cast

import numpy as np

if TYPE_CHECKING:
    from .env import LookaheadEnv


PREVIEW_FEATURE_DIM: Final[int] = 3
NORMALIZATION_DISTANCE_M: Final[float] = 10.0

Mode: TypeAlias = Literal["baseline", "lookahead_obs", "lookahead_obs_pp_reward"]

# The descriptive names are the values carried through runtime metadata.  The
# short names remain accepted at the input boundary for existing configurations
# while producing canonical records.
CANONICAL_MODES: Final[tuple[str, str, str]] = (
    "baseline",
    "lookahead_obs",
    "lookahead_obs_pp_reward",
)
MODE_ALIASES: Final[dict[str, str]] = {
    "obs": "lookahead_obs",
    "obs_pp": "lookahead_obs_pp_reward",
}
def normalize_mode(value: str) -> Mode:
    """Normalize a configuration or API mode to its canonical value."""

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


@dataclass(frozen=True, slots=True)
class ObservationContract:
    """Verified raw observation schema and immutable Box bounds.

    The host width is data driven.  ``shape[0]`` is the only baseline
    dimension used by the wrapper, so both 259-wide and 262-wide host vectors
    follow the same path without padding or truncation.
    """

    shape: tuple[int, ...]
    dtype: np.dtype
    low: np.ndarray = field(repr=False)
    high: np.ndarray = field(repr=False)
    source: str = ""

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
        if len(shape) != 1 or shape[0] <= 0:
            raise UnsupportedHostError(
                "lookahead_learning requires a positive one-dimensional raw "
                f"observation, found {shape}"
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
    @classmethod
    def from_space(
        cls,
        space: object,
        *,
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
        return cls(
            shape=shape,
            dtype=dtype,
            low=low,
            high=high,
            source=source,
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
        expected_shape = (self.shape[0] + PREVIEW_FEATURE_DIM,)
        if augmented.shape != expected_shape:
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
    tolerance_ratio: float | None = None,
) -> object:
    """Read the audited host start-lane state without stepping or re-observing.

    The current project stores the reset ordinal in
    ``StartLaneMetaDriveEnv._target_lane_ordinals`` and exposes the pure
    ``resolve_target_lane_state`` function in ``start_lane_env.py``.  This
    helper imports that function normally at call time, passes the preserved
    reset ordinal, and returns the host ``TargetLaneState`` object.  It never
    calls ``observe``, ``step``, ``reward_function``, or mutates the host.

    The resolver is imported lazily so a host without the optional
    ``start_lane_env`` module can continue with the fixed Navigation route.
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

    try:
        module = importlib.import_module("start_lane_env")
    except Exception as error:
        raise UnsupportedHostError(
            "optional start_lane_env target resolver is unavailable: "
            f"{type(error).__name__}: {error}"
        ) from error
    resolver = getattr(module, "resolve_target_lane_state", None)
    if not callable(resolver):
        raise UnsupportedHostError(
            "start_lane_env.resolve_target_lane_state is unavailable"
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
    module lazily so importing the adapter itself never imports a simulator.
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


def _target_ordinal_after_reset(env: object, vehicle: object) -> tuple[int, bool]:
    """Resolve the reset target lane ordinal without re-observing the host."""

    saved = _saved_start_lane_ordinal(env, vehicle)
    if saved is not None:
        return int(saved), True
    lane_index = getattr(vehicle, "lane_index", None)
    if lane_index is None:
        raise UnsupportedHostError("vehicle lane_index is unavailable after reset")
    try:
        value = lane_index[-1]
    except (IndexError, KeyError, TypeError) as error:
        raise UnsupportedHostError(
            "vehicle lane_index has no reset target ordinal"
        ) from error
    if isinstance(value, bool):
        raise UnsupportedHostError("vehicle lane ordinal must be an integer")
    try:
        return int(value), False
    except (TypeError, ValueError) as error:
        raise UnsupportedHostError("vehicle lane ordinal must be an integer") from error


def _target_lane_state(
    env: object,
) -> object | None:
    """Read optional host target-lane state, preserving raw route fallback."""

    try:
        return read_target_lane_state(env)
    except UnsupportedHostError:
        # A plain host may have no target-lane resolver or retained ordinal.
        # The fixed Navigation route remains the source of preview geometry.
        return None


def _target_reference_present(vehicle: object, target_ordinal: int) -> bool | None:
    """Return target-lane presence, or ``None`` when the host exposes no refs."""

    navigation = getattr(vehicle, "navigation", None)
    references = getattr(navigation, "current_ref_lanes", None)
    if references is None:
        return None
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


def _read_direct_policy(env: object, vehicle: object) -> object:
    """Read the policy registered for the active vehicle through the engine."""

    engine = getattr(env, "engine", None)
    getter = getattr(engine, "get_policy", None)
    name = getattr(vehicle, "name", None)
    if not callable(getter) or name is None:
        raise UnsupportedHostError(
            "Pure Pursuit mode requires the direct engine policy lookup for "
            "the active vehicle"
        )
    policy = getter(name)
    if policy is None:
        raise UnsupportedHostError(
            f"Pure Pursuit mode found no engine policy for active vehicle {name!r}"
        )
    return policy


def _read_vehicle_geometry(
    env: object,
    *,
    verify_direct_policy: bool = True,
) -> dict[str, float]:
    """Read source-audited axle geometry used by the PP provider.

    MetaDrive exposes ``FRONT_WHEELBASE`` and ``REAR_WHEELBASE`` in metres and
    ``max_steering`` in degrees.  The provider passes the rear axle position
    explicitly to the pure geometry function; no body-centre approximation or
    unit inference is performed here.
    """

    vehicle = get_single_agent(env)
    module_name = type(vehicle).__module__
    qualname = type(vehicle).__qualname__
    if not module_name.startswith("metadrive."):
        raise UnsupportedHostError(
            "vehicle geometry is verified only for MetaDrive vehicle classes; "
            f"found {module_name}.{qualname}"
        )
    if verify_direct_policy and (
        module_name != "metadrive.component.vehicle.vehicle_type"
        or qualname != "DefaultVehicle"
    ):
        raise UnsupportedHostError(
            "Pure Pursuit vehicle geometry is source-audited only for "
            "metadrive.component.vehicle.vehicle_type.DefaultVehicle; "
            f"found {module_name}.{qualname}"
        )

    def number(name: str, *fallback: str) -> float:
        value = getattr(vehicle, name, None)
        if value is None:
            config = getattr(vehicle, "config", None)
            try:
                mapping = _host_config_mapping(config, name="vehicle.config")
            except HostContractError:
                mapping = {}
            value = mapping.get(name)
        if value is None:
            for alternate in fallback:
                value = getattr(vehicle, alternate, None)
                if value is not None:
                    break
        try:
            number_value = float(value)
        except (TypeError, ValueError) as error:
            raise UnsupportedHostError(
                f"vehicle geometry {name} is unavailable"
            ) from error
        if not math.isfinite(number_value) or number_value <= 0.0:
            raise UnsupportedHostError(
                f"vehicle geometry {name} must be positive"
            )
        return number_value

    front = number("FRONT_WHEELBASE")
    rear = number("REAR_WHEELBASE")
    max_deg = number("max_steering")
    if verify_direct_policy:
        policy = _read_direct_policy(env, vehicle)
        policy_path = f"{type(policy).__module__}.{type(policy).__qualname__}"
        if policy_path != "metadrive.policy.env_input_policy.EnvInputPolicy":
            raise UnsupportedHostError(
                "Pure Pursuit mode requires metadrive.policy.env_input_policy."
                "EnvInputPolicy through engine.get_policy; "
                f"found {policy_path}"
            )
    return {
        "front_wheelbase_m": front,
        "rear_wheelbase_m": rear,
        "wheelbase_m": front + rear,
        "max_steering_deg": max_deg,
    }


class MetaDrivePreviewProvider:
    """Build one fixed Navigation route and compute preview features per state.

    This provider is part of the host adapter used by both ordinary
    ``train.py`` and ``evaluate.py`` paths.  The wrapper remains independent
    of MetaDrive imports until an episode actually starts.
    """

    def __init__(
        self,
        *,
        lookahead_m: float = 6.0,
    ) -> None:
        value = _finite_number(lookahead_m, name="lookahead_m")
        if value < 0.0:
            raise ValueError("lookahead_m must be non-negative")
        self.lookahead_m = value
        self._route: object | None = None
        self._route_error: str | None = None
        self._episode_key: object = None
        self._target_ordinal: int | None = None
        self._target_ordinal_persistent = False

    def reset(self, env: object) -> None:
        """Rebuild route state exactly once after the raw environment reset."""

        self._route = None
        self._route_error = None
        self._episode_key = None
        self._target_ordinal = None
        self._target_ordinal_persistent = False
        vehicle = get_single_agent(env)
        target_ordinal, persistent = _target_ordinal_after_reset(env, vehicle)
        self._target_ordinal = target_ordinal
        self._target_ordinal_persistent = persistent
        self._route = build_fixed_navigation_route(env)
        state = read_vehicle_state(env)
        self._episode_key = tuple(state.get("checkpoints", ()))

    @staticmethod
    def _invalid(reason: str, **details: object) -> Mapping[str, object]:
        return {
            "preview_valid": False,
            "x_g_m": 0.0,
            "y_g_m": 0.0,
            "invalid_reason": reason,
            "details": details,
        }

    def __call__(self, env: object) -> Mapping[str, object]:
        from .geometry import compute_preview

        state = read_vehicle_state(env)
        point = state.get("position_xy")
        psi = state.get("heading_theta")
        vehicle = get_single_agent(env)
        velocity = getattr(vehicle, "velocity", None)
        speed = state.get("speed_m_s")
        if velocity is not None and psi is not None:
            try:
                speed = (
                    float(velocity[0]) * math.cos(float(psi))
                    + float(velocity[1]) * math.sin(float(psi))
                )
            except (TypeError, ValueError, IndexError):
                speed = state.get("speed_m_s")
        current_checkpoints = tuple(state.get("checkpoints", ()))
        if self._route is None:
            return self._invalid(self._route_error or "route_unavailable")
        if self._episode_key is not None and current_checkpoints != self._episode_key:
            return self._invalid(
                "navigation_route_changed",
                reset_checkpoints=self._episode_key,
                current_checkpoints=current_checkpoints,
            )
        assert self._target_ordinal is not None
        if self._target_ordinal_persistent:
            observed_target, observed_persistent = _target_ordinal_after_reset(
                env, vehicle
            )
            if not observed_persistent or observed_target != self._target_ordinal:
                return self._invalid(
                    "start_lane_reference_changed",
                    reset_target_ordinal=self._target_ordinal,
                    observed_target_ordinal=observed_target,
                )
        target_state = _target_lane_state(env)
        target_fields: dict[str, object]
        if target_state is None:
            present = _target_reference_present(vehicle, self._target_ordinal)
            # ``None`` means this host does not expose current reference lane
            # objects.  The fixed route and reset lane ordinal still provide
            # the route contract, so do not make the optional resolver a hard
            # dependency.
            start_lane_valid = True if present is None else bool(present)
            target_fields = {
                "target_lane_state_source": (
                    "navigation.current_ref_lanes.ordinal_presence"
                    if present is not None
                    else "fixed_navigation_route"
                ),
                "target_lane_valid": start_lane_valid,
            }
        else:
            start_lane_valid = bool(getattr(target_state, "valid", False))
            target_fields = {
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
            return self._invalid("vehicle_geometry_unavailable")
        route = self._route
        result = compute_preview(
            route.path,
            point,
            psi,
            lookahead_m=self.lookahead_m,
            forward_speed_mps=speed,
            start_lane_valid=start_lane_valid,
        )
        details: dict[str, object] = {
            "p_xy": point,
            "psi_rad": psi,
            "route": route.diagnostics.as_dict(),
            "fixed_route_lane_ids": [
                item.lane_id for item in route.path.metadata
            ],
            "fixed_route_lane_types": [
                item.lane_type for item in route.path.metadata
            ],
            "projection": (
                None
                if result.projection is None
                else result.projection.as_dict()
            ),
            "heading_error_rad": result.heading_error_rad,
            "distance_to_goal_m": result.distance_to_goal_m,
        }
        details.update(target_fields)
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
    """Calculate a PP reference from the same pre-action preview point."""

    def __init__(self, *, steering_sign: float = 1.0) -> None:
        sign = _finite_number(steering_sign, name="steering_sign")
        if sign not in (-1.0, 1.0):
            raise ValueError("steering_sign must be +1 or -1")
        self.steering_sign = sign
        self.geometry: dict[str, float] | None = None

    def reset(self, env: object) -> None:
        self.geometry = _read_vehicle_geometry(env)

    def __call__(self, env: object, preview: object) -> Mapping[str, object]:
        from .geometry import pure_pursuit_from_rear_coordinates

        if not getattr(preview, "preview_valid", False) or getattr(
            preview, "q_xy", None
        ) is None:
            return {
                "pp_valid": False,
                "u_pp": None,
                "invalid_reason": "preview_invalid",
            }
        if self.geometry is None:
            self.geometry = _read_vehicle_geometry(env)
        details = dict(getattr(preview, "details", ()))
        point = details.get("p_xy")
        psi = details.get("psi_rad")
        q = getattr(preview, "q_xy", None)
        if point is None or psi is None or q is None:
            return {
                "pp_valid": False,
                "u_pp": None,
                "invalid_reason": "vehicle_geometry_unavailable",
            }
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


def wrap_lookahead_env(
    raw_env: object,
    *,
    lookahead_m: float = 6.0,
    pp_weight: float = 0.0,
) -> LookaheadEnv:
    """Wrap a raw host environment using ordinary TOML-resolved parameters.

    ``pp_weight == 0`` selects observation-only mode.  A positive weight adds
    the PP penalty while retaining the same augmented observation.  The raw
    environment is never stepped or reset during construction; all host
    access happens through :class:`lookahead_learning.env.LookaheadEnv`.
    """

    lookahead = _finite_number(lookahead_m, name="lookahead_m")
    if lookahead <= 0.0:
        raise ValueError("lookahead_m must be finite and positive")
    weight = _finite_number(pp_weight, name="pp_weight")
    if weight < 0.0:
        raise ValueError("pp_weight must be finite and non-negative")
    from .env import LookaheadEnv

    contract = ObservationContract.from_space(
        getattr(raw_env, "observation_space", None),
        source=f"{type(raw_env).__module__}.{type(raw_env).__qualname__}",
    )
    provider = MetaDrivePreviewProvider(lookahead_m=lookahead)
    if weight == 0.0:
        return LookaheadEnv(
            raw_env,
            mode="lookahead_obs",
            contract=contract,
            preview_provider=provider,
        )
    pp_provider = MetaDrivePPProvider()
    return LookaheadEnv(
        raw_env,
        mode="lookahead_obs_pp_reward",
        contract=contract,
        preview_provider=provider,
        pp_provider=pp_provider,
        pp_weight=weight,
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
    "HostContractError",
    "MetaDrivePPProvider",
    "MetaDrivePreviewProvider",
    "CANONICAL_MODES",
    "MODE_ALIASES",
    "Mode",
    "NORMALIZATION_DISTANCE_M",
    "ObservationContract",
    "PREVIEW_FEATURE_DIM",
    "UnsupportedHostError",
    "get_single_agent",
    "build_fixed_navigation_route",
    "navigation_lane_sequence",
    "invalid_preview_values",
    "normalize_preview_coordinate",
    "normalize_mode",
    "read_applied_steering",
    "read_applied_action",
    "read_target_lane_state",
    "read_vehicle_state",
    "simulation_dt_seconds",
    "wrap_lookahead_env",
]
