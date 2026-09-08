"""Observation and reward wrapper for the forward-preview learning modes.

``LookaheadEnv`` is intentionally an ordinary Gymnasium wrapper around the
existing raw environment.  It never calls ``unwrapped.step`` and never
recomputes the host reward.  A geometry provider reads the post-reset or
post-step state and returns a :class:`PreviewState`; an optional PP provider
returns the reference used by mode ``lookahead_obs_pp_reward``.  Both providers are expected to
be pure reads of the host state.  They are called once per state and are never
used to alter the action.

The wrapper requires an :class:`~lookahead_learning.adapter.ObservationContract` that
was verified against the host.  Consequently a 259-wide upstream MetaDrive
observation cannot be padded into the requested 262-wide schema.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any, Protocol, TypeAlias

import gymnasium as gym
import numpy as np

from .adapter import (
    AUGMENTED_OBS_DIM,
    BASELINE_OBS_DIM,
    HostContractError,
    Mode,
    ObservationContract,
    UnsupportedHostError,
    get_single_agent,
    invalid_preview_values,
    normalize_preview_coordinate,
    read_applied_action,
    read_applied_steering,
    read_vehicle_state,
    normalize_mode,
    simulation_dt_seconds,
)


class PreviewProvider(Protocol):
    """Read one state and return a :class:`PreviewState` or mapping."""

    def __call__(self, env: object) -> object: ...


class PPProvider(Protocol):
    """Read one state and return a :class:`PPReference` or mapping."""

    def __call__(self, env: object, preview: "PreviewState") -> object: ...


StateReader: TypeAlias = Callable[[object], Mapping[str, object]]
AppliedSteeringReader: TypeAlias = Callable[[object], float]
AppliedActionReader: TypeAlias = Callable[[object], Sequence[float]]
DtReader: TypeAlias = Callable[[object], float]


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise HostContractError(f"{name} must be numeric, not bool")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise HostContractError(f"{name} must be numeric") from error
    if not math.isfinite(number):
        raise HostContractError(f"{name} must be finite")
    return number


def _strict_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise HostContractError(f"{name} must be bool")
    return bool(value)


def _optional_xy(value: object, *, name: str) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise HostContractError(f"{name} must be a 2D sequence")
    if len(value) < 2:
        raise HostContractError(f"{name} must contain at least two values")
    return (
        _finite(value[0], name=f"{name}[0]"),
        _finite(value[1], name=f"{name}[1]"),
    )


def _field(value: object, *names: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
        return default
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _freeze_value(value: object) -> object:
    """Make provider metadata safe from later external mutation."""

    if isinstance(value, Mapping):
        return tuple(
            (str(key), _freeze_value(item))
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        )
    if isinstance(value, np.ndarray):
        return tuple(_freeze_value(item) for item in value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (str, bool)) or value is None:
        return value
    return str(value)


def _details_tuple(value: object) -> tuple[tuple[str, object], ...]:
    """Normalize optional provider diagnostics to immutable key/value pairs."""

    if value is None:
        return ()
    if isinstance(value, Mapping):
        return tuple((str(key), item) for key, item in value.items())
    if isinstance(value, (str, bytes)):
        raise HostContractError("provider details must be a mapping or pair sequence")
    try:
        pairs = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise HostContractError(
            "provider details must be a mapping or pair sequence"
        ) from error
    result: list[tuple[str, object]] = []
    for pair in pairs:
        if (
            not isinstance(pair, Sequence)
            or isinstance(pair, (str, bytes))
            or len(pair) != 2
        ):
            raise HostContractError("provider details entries must be (key, value) pairs")
        result.append((str(pair[0]), pair[1]))
    return tuple(result)


def _mapping_from_frozen(value: object) -> Mapping[str, object] | None:
    """Read a mapping that may already have passed through ``_freeze_value``."""

    if isinstance(value, Mapping):
        return value
    if isinstance(value, tuple):
        result: dict[str, object] = {}
        for item in value:
            if not isinstance(item, tuple) or len(item) != 2:
                return None
            result[str(item[0])] = item[1]
        return result
    return None


@dataclass(frozen=True, slots=True)
class PreviewState:
    """One immutable geometric preview result.

    ``x_g_m`` and ``y_g_m`` are world-coordinate-derived vehicle-frame metres:
    forward is positive x and left is positive y.  The provider owns route
    validity checks (finite lane connection, projection, lookahead boundary,
    forward speed, and so on); this class only enforces the representation
    needed by the wrapper.
    """

    x_g_m: float | None
    y_g_m: float | None
    preview_valid: bool
    q_xy: tuple[float, float] | None = None
    s_proj_m: float | None = None
    s_goal_m: float | None = None
    projected_lane_id: object = None
    goal_lane_id: object = None
    invalid_reason: str | None = None
    x_clipped: bool = False
    y_clipped: bool = False
    details: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        valid = _strict_bool(self.preview_valid, name="preview_valid")
        object.__setattr__(self, "preview_valid", valid)
        if self.x_g_m is not None:
            object.__setattr__(
                self,
                "x_g_m",
                _finite(self.x_g_m, name="x_g_m"),
            )
        if self.y_g_m is not None:
            object.__setattr__(
                self,
                "y_g_m",
                _finite(self.y_g_m, name="y_g_m"),
            )
        if valid and (self.x_g_m is None or self.y_g_m is None):
            raise HostContractError(
                "valid preview requires finite x_g_m and y_g_m values"
            )
        object.__setattr__(self, "q_xy", _optional_xy(self.q_xy, name="q_xy"))
        for field_name in ("s_proj_m", "s_goal_m"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _finite(value, name=field_name),
                )
        if self.invalid_reason is not None and not isinstance(
            self.invalid_reason, str
        ):
            raise HostContractError("invalid_reason must be a string or None")
        object.__setattr__(self, "x_clipped", bool(self.x_clipped))
        object.__setattr__(self, "y_clipped", bool(self.y_clipped))
        object.__setattr__(
            self,
            "details",
            tuple(
                (str(key), _freeze_value(value))
                for key, value in _details_tuple(self.details)
            ),
        )

    @classmethod
    def from_object(cls, value: object) -> "PreviewState":
        if isinstance(value, cls):
            return value
        valid_value = _field(value, "preview_valid", "valid", default=None)
        if valid_value is None:
            raise HostContractError(
                "preview provider must expose preview_valid explicitly"
            )
        valid = _strict_bool(valid_value, name="preview_valid")
        x_value = _field(value, "x_g_m", "x_g", default=None)
        y_value = _field(value, "y_g_m", "y_g", default=None)
        x = None if x_value is None else _finite(x_value, name="x_g_m")
        y = None if y_value is None else _finite(y_value, name="y_g_m")
        reason = _field(value, "invalid_reason", "reason", default=None)
        details: list[tuple[str, object]] = []
        if isinstance(value, Mapping):
            details.extend(_details_tuple(value.get("details")))
            reserved = {
                "preview_valid",
                "valid",
                "x_g_m",
                "x_g",
                "y_g_m",
                "y_g",
                "q_xy",
                "q",
                "s_proj_m",
                "s_proj",
                "s_goal_m",
                "s_goal",
                "projected_lane_id",
                "goal_lane_id",
                "invalid_reason",
                "reason",
                "x_clipped",
                "y_clipped",
                "details",
            }
            details.extend(
                (str(key), item)
                for key, item in value.items()
                if key not in reserved
            )
        else:
            details.extend(_details_tuple(_field(value, "details", default=None)))
        return cls(
            x_g_m=x,
            y_g_m=y,
            preview_valid=valid,
            q_xy=_field(value, "q_xy", "q", default=None),
            s_proj_m=_field(value, "s_proj_m", "s_proj", default=None),
            s_goal_m=_field(value, "s_goal_m", "s_goal", default=None),
            projected_lane_id=_field(value, "projected_lane_id", default=None),
            goal_lane_id=_field(value, "goal_lane_id", default=None),
            invalid_reason=None if reason is None else str(reason),
            x_clipped=bool(_field(value, "x_clipped", default=False)),
            y_clipped=bool(_field(value, "y_clipped", default=False)),
            details=tuple(details),
        )

    @property
    def observation_values(self) -> tuple[float, float, float]:
        if not self.preview_valid:
            return invalid_preview_values()
        assert self.x_g_m is not None and self.y_g_m is not None
        return (
            normalize_preview_coordinate(self.x_g_m),
            normalize_preview_coordinate(self.y_g_m),
            1.0,
        )

    def as_dict(self) -> dict[str, object]:
        values = self.observation_values
        return {
            "x_g_m": self.x_g_m,
            "y_g_m": self.y_g_m,
            "q_xy": list(self.q_xy) if self.q_xy is not None else None,
            "s_proj_m": self.s_proj_m,
            "s_goal_m": self.s_goal_m,
            "projected_lane_id": _freeze_value(self.projected_lane_id),
            "goal_lane_id": _freeze_value(self.goal_lane_id),
            "preview_valid": self.preview_valid,
            "invalid_reason": self.invalid_reason,
            "x_clipped": self.x_clipped,
            "y_clipped": self.y_clipped,
            "normalized_x_g": values[0],
            "normalized_y_g": values[1],
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class PPReference:
    """One PP steering reference calculated at the pre-action state."""

    pp_valid: bool
    u_pp: float | None
    u_pp_unclipped: float | None = None
    x_rear_m: float | None = None
    y_rear_m: float | None = None
    kappa_pp: float | None = None
    delta_pp_rad: float | None = None
    saturated: bool = False
    discrete_nearest_error: float | None = None
    invalid_reason: str | None = None
    details: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        valid = _strict_bool(self.pp_valid, name="pp_valid")
        object.__setattr__(self, "pp_valid", valid)
        if valid and self.u_pp is None:
            raise HostContractError("pp_valid requires a finite u_pp")
        for field_name in (
            "u_pp",
            "u_pp_unclipped",
            "x_rear_m",
            "y_rear_m",
            "kappa_pp",
            "delta_pp_rad",
            "discrete_nearest_error",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _finite(value, name=field_name),
                )
        if self.u_pp is not None and not -1.0 <= self.u_pp <= 1.0:
            raise HostContractError("clipped u_pp must be in [-1, 1]")
        if self.invalid_reason is not None and not isinstance(
            self.invalid_reason, str
        ):
            raise HostContractError("invalid_reason must be a string or None")
        object.__setattr__(self, "saturated", bool(self.saturated))
        object.__setattr__(
            self,
            "details",
            tuple(
                (str(key), _freeze_value(value))
                for key, value in _details_tuple(self.details)
            ),
        )

    @classmethod
    def from_object(
        cls,
        value: object,
        *,
        preview: PreviewState,
    ) -> "PPReference":
        if isinstance(value, cls):
            result = value
        else:
            valid_value = _field(value, "pp_valid", "valid", default=None)
            if valid_value is None:
                raise HostContractError(
                    "PP provider must expose pp_valid explicitly"
                )
            valid = _strict_bool(valid_value, name="pp_valid")
            raw_u = _field(value, "u_pp", "steering", default=None)
            u_pp = None if raw_u is None else _finite(raw_u, name="u_pp")
            details: list[tuple[str, object]] = []
            if isinstance(value, Mapping):
                details.extend(_details_tuple(value.get("details")))
                reserved = {
                    "pp_valid",
                    "valid",
                    "u_pp",
                    "steering",
                    "u_pp_unclipped",
                    "x_rear_m",
                    "y_rear_m",
                    "kappa_pp",
                    "delta_pp_rad",
                    "saturated",
                    "discrete_nearest_error",
                    "invalid_reason",
                    "reason",
                    "details",
                }
                details.extend(
                    (str(key), item)
                    for key, item in value.items()
                    if key not in reserved
                )
            else:
                details.extend(_details_tuple(_field(value, "details", default=None)))
            result = cls(
                pp_valid=valid,
                u_pp=u_pp,
                u_pp_unclipped=_field(value, "u_pp_unclipped", default=None),
                x_rear_m=_field(value, "x_rear_m", default=None),
                y_rear_m=_field(value, "y_rear_m", default=None),
                kappa_pp=_field(value, "kappa_pp", default=None),
                delta_pp_rad=_field(value, "delta_pp_rad", default=None),
                saturated=bool(_field(value, "saturated", default=False)),
                discrete_nearest_error=_field(
                    value,
                    "discrete_nearest_error",
                    default=None,
                ),
                invalid_reason=(
                    None
                    if _field(value, "invalid_reason", "reason", default=None)
                    is None
                    else str(_field(value, "invalid_reason", "reason"))
                ),
                details=tuple(details),
            )
        if result.pp_valid and not preview.preview_valid:
            raise HostContractError("pp_valid cannot be true when preview is invalid")
        return result

    def as_dict(self) -> dict[str, object]:
        return {
            "pp_valid": self.pp_valid,
            "u_pp": self.u_pp,
            "u_pp_unclipped": self.u_pp_unclipped,
            "x_rear_m": self.x_rear_m,
            "y_rear_m": self.y_rear_m,
            "kappa_pp": self.kappa_pp,
            "delta_pp_rad": self.delta_pp_rad,
            "saturated": self.saturated,
            "discrete_nearest_error": self.discrete_nearest_error,
            "invalid_reason": self.invalid_reason,
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class PreviewSnapshot:
    """Pre/post state snapshot used to preserve action-time alignment."""

    decision: int
    t_seconds: float
    preview: PreviewState | None
    pp: PPReference | None
    state: tuple[tuple[str, object], ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "decision": self.decision,
            "t_seconds": self.t_seconds,
            "preview": None if self.preview is None else self.preview.as_dict(),
            "pp": None if self.pp is None else self.pp.as_dict(),
            "state": dict(self.state),
        }


def _snapshot_state(value: Mapping[str, object] | None) -> tuple[tuple[str, object], ...]:
    if value is None:
        return ()
    return tuple(
        (str(key), _freeze_value(item))
        for key, item in sorted(value.items(), key=lambda item: str(item[0]))
    )


def _state_value(snapshot: PreviewSnapshot | None, *names: str) -> float | None:
    if snapshot is None:
        return None
    state = dict(snapshot.state)
    for name in names:
        value = state.get(name)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            number = float(value)
            if math.isfinite(number):
                return number
    return None


def _position_pair(value: object) -> tuple[float, float] | None:
    """Read a finite world-position pair for transition distance telemetry."""

    if value is None or isinstance(value, (str, bytes)):
        return None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    try:
        if not isinstance(value, Sequence) or len(value) < 2:
            return None
        x = float(value[0])
        y = float(value[1])
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return x, y


def _position_distance_m(
    before: object,
    after: object,
) -> float | None:
    before_xy = _position_pair(before)
    after_xy = _position_pair(after)
    if before_xy is None or after_xy is None:
        return None
    return math.hypot(after_xy[0] - before_xy[0], after_xy[1] - before_xy[1])


def _wrap_to_pi(value: float) -> float:
    wrapped = (value + math.pi) % (2.0 * math.pi) - math.pi
    # Keep +pi stable when the input is exactly +pi.
    return math.pi if wrapped == -math.pi and value > 0.0 else wrapped


def _yaw_rate_after(
    before: PreviewSnapshot | None,
    after: PreviewSnapshot,
    dt: float | None,
) -> float | None:
    if before is None or dt is None or dt <= 0.0:
        return None
    before_heading = _state_value(before, "heading_theta", "heading_rad", "heading")
    after_heading = _state_value(after, "heading_theta", "heading_rad", "heading")
    if before_heading is None or after_heading is None:
        return None
    return _wrap_to_pi(after_heading - before_heading) / dt


def _action_record(action: object) -> object:
    if isinstance(action, np.ndarray):
        return _freeze_value(action)
    if isinstance(action, (np.integer, int)) and not isinstance(action, bool):
        return int(action)
    if isinstance(action, (np.floating, float)):
        return float(action) if math.isfinite(float(action)) else None
    if isinstance(action, (list, tuple, Mapping)):
        return _freeze_value(action)
    return str(action)


def _snapshot_flat_fields(
    snapshot: PreviewSnapshot | None,
) -> dict[str, object]:
    """Expose canonical telemetry fields from one immutable snapshot.

    ``lookahead_learning.pre_step``/``post_step`` remain the lossless diagnostic
    representation.  These aliases are intentionally derived here so the
    stdlib telemetry aggregator can consume rows without knowing wrapper
    implementation details or making a second host-state read.
    """

    if snapshot is None:
        return {
            "position": None,
            "heading": None,
            "speed_mps": None,
            "progress_m": None,
            "reference_lane_ids": None,
            "navigation_reference_ids": None,
            "preview_valid": None,
            "preview_invalid_reason": None,
            "q": None,
            "x_g": None,
            "y_g": None,
            "preview_x_norm": None,
            "preview_y_norm": None,
            "S_proj": None,
            "S_goal": None,
            "projection_lane": None,
            "target_lane": None,
            "heading_error_rad": None,
            "lateral_error_m": None,
            "preview_x_clipped": None,
            "preview_y_clipped": None,
            "preview_route_boundary": None,
        }
    state = dict(snapshot.state)
    position = state.get("position_xy")
    heading = state.get("heading_theta", state.get("heading_rad"))
    speed = state.get("speed_m_s", state.get("speed_mps"))
    progress = state.get("travelled_length_m", state.get("progress_m"))
    preview = snapshot.preview
    detail_map = {} if preview is None else dict(preview.details)
    fixed_reference_ids = state.get("fixed_reference_lane_ids")
    if fixed_reference_ids is None:
        fixed_reference_ids = state.get("fixed_route_lane_ids")
    if fixed_reference_ids is None:
        for detail_key in (
            "reference_lane_ids",
            "fixed_reference_lane_ids",
            "fixed_route_lane_ids",
        ):
            fixed_reference_ids = detail_map.get(detail_key)
            if fixed_reference_ids is not None:
                break
    result: dict[str, object] = {
        "position": position,
        "heading": heading,
        "speed_mps": speed,
        "progress_m": progress,
        # This is the immutable route chosen at reset when the provider has
        # recorded it.  Current Navigation IDs remain a separate field below;
        # they may change as the vehicle crosses a checkpoint.
        "reference_lane_ids": fixed_reference_ids,
        "navigation_reference_ids": state.get("current_ref_lane_ids"),
        "preview_valid": None if preview is None else preview.preview_valid,
        "preview_invalid_reason": None if preview is None else preview.invalid_reason,
        "q": None if preview is None or preview.q_xy is None else preview.q_xy,
        "x_g": None if preview is None else preview.x_g_m,
        "y_g": None if preview is None else preview.y_g_m,
        "preview_x_norm": (
            None if preview is None else preview.observation_values[0]
        ),
        "preview_y_norm": (
            None if preview is None else preview.observation_values[1]
        ),
        "S_proj": None if preview is None else preview.s_proj_m,
        "S_goal": None if preview is None else preview.s_goal_m,
        "projection_lane": None if preview is None else preview.projected_lane_id,
        "target_lane": None if preview is None else preview.goal_lane_id,
        "heading_error_rad": None,
        "lateral_error_m": None,
        "preview_x_clipped": None if preview is None else preview.x_clipped,
        "preview_y_clipped": None if preview is None else preview.y_clipped,
        "preview_route_boundary": detail_map.get("route_boundary_reason"),
    }
    if preview is not None:
        for field_name in ("heading_error_rad", "heading_error_m", "lateral_error_m"):
            if field_name in detail_map:
                result[field_name] = detail_map[field_name]
        if result["lateral_error_m"] is None:
            projection = _mapping_from_frozen(detail_map.get("projection"))
            if projection is not None:
                result["lateral_error_m"] = projection.get("lateral_m")
    return result


class LookaheadEnv(gym.Wrapper):
    """Apply the requested observation mode and optional PP reward term.

    Args:
        env: Existing raw 262-wide single-agent Gymnasium environment.
        mode: ``baseline``, ``lookahead_obs``, or ``lookahead_obs_pp_reward``.  The
            legacy ``obs`` and ``obs_pp`` spellings are accepted and normalized.
        contract: Host evidence for the raw observation schema.  It must be
            semantic-verified; this prevents accidental 259/262 conflation.
        preview_provider: Required for ``lookahead_obs`` and
            ``lookahead_obs_pp_reward``.  It must return
            a mapping accepted by :meth:`PreviewState.from_object`.
        pp_provider: Required for ``lookahead_obs_pp_reward``.  It receives the same pre-action
            preview object and must return an explicit ``pp_valid`` result.
        pp_weight: Non-negative PP coefficient.  It is required for ``lookahead_obs_pp_reward``
            even when zero, so a zero-weight equivalence test is explicit.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        mode: Mode,
        contract: ObservationContract,
        preview_provider: PreviewProvider | None = None,
        pp_provider: PPProvider | None = None,
        pp_weight: float | None = None,
        applied_action_reader: AppliedActionReader | None = None,
        applied_steering_reader: AppliedSteeringReader | None = None,
        dt_reader: DtReader | None = None,
        state_reader: StateReader | None = None,
        run_id: str = "",
    ) -> None:
        mode = normalize_mode(mode)
        if not isinstance(contract, ObservationContract):
            raise UnsupportedHostError(
                "LookaheadEnv requires an ObservationContract built from host evidence"
            )
        if not contract.semantic_verified:
            raise UnsupportedHostError(
                "LookaheadEnv refuses a shape-only raw observation contract"
            )
        super().__init__(env)
        actual_space = getattr(env, "observation_space", None)
        actual_shape = tuple(getattr(actual_space, "shape", ()) or ())
        actual_dtype = np.dtype(getattr(actual_space, "dtype", object))
        if actual_shape != contract.shape or actual_dtype != contract.dtype:
            raise UnsupportedHostError(
                "raw environment observation space differs from the audited "
                f"contract: expected={contract.shape}/{contract.dtype}, "
                f"found={actual_shape}/{actual_dtype}"
            )
        if not np.array_equal(np.asarray(actual_space.low), contract.low) or not np.array_equal(
            np.asarray(actual_space.high), contract.high
        ):
            raise HostContractError("raw environment Box bounds differ from audited contract")
        if mode in ("lookahead_obs", "lookahead_obs_pp_reward") and not callable(preview_provider):
            raise UnsupportedHostError(
                f"mode={mode} requires an explicit geometry preview_provider"
            )
        if mode == "lookahead_obs_pp_reward":
            if not callable(pp_provider):
                raise UnsupportedHostError("mode=lookahead_obs_pp_reward requires an explicit pp_provider")
            if pp_weight is None:
                raise ValueError("mode=lookahead_obs_pp_reward requires --pp-weight, including zero")
            if isinstance(pp_weight, bool):
                raise TypeError("pp_weight must be numeric, not bool")
            try:
                pp_weight_value = float(pp_weight)
            except (TypeError, ValueError) as error:
                raise TypeError("pp_weight must be numeric") from error
            if not math.isfinite(pp_weight_value) or pp_weight_value < 0.0:
                raise ValueError("pp_weight must be finite and non-negative")
        else:
            pp_weight_value = 0.0
        if not callable(applied_action_reader or applied_steering_reader or read_applied_action):
            raise UnsupportedHostError(
                "all modes require an applied normalized action reader"
            )
        if not callable(dt_reader or simulation_dt_seconds):
            raise UnsupportedHostError(
                "all modes require a simulation dt reader"
            )
        self._mode: Mode = mode
        self._contract = contract
        self._preview_provider = preview_provider
        self._pp_provider = pp_provider
        self._pp_weight = pp_weight_value
        self._read_applied_action = applied_action_reader
        self._read_applied_steering = applied_steering_reader or read_applied_steering
        self._read_dt = dt_reader or simulation_dt_seconds
        self._read_state = state_reader or read_vehicle_state
        self._run_id = str(run_id)
        self._raw_observation_space = actual_space
        self._augmented_observation_space = (
            contract.augmented_space() if mode != "baseline" else actual_space
        )
        self._snapshot: PreviewSnapshot | None = None
        self._terminal_snapshot: PreviewSnapshot | None = None
        self._needs_reset = True
        self._decision = 0
        self._time_seconds = 0.0
        self._dt_seconds: float | None = None
        self._episode_r_base = 0.0
        self._episode_r_pp = 0.0
        self._episode_r_total = 0.0
        self._previous_applied_steering: float | None = None
        self._previous_applied_action: tuple[float, float] | None = None

    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def pp_weight(self) -> float:
        return self._pp_weight

    @property
    def observation_space(self):
        return self._augmented_observation_space

    @property
    def contract(self) -> ObservationContract:
        return self._contract

    @property
    def last_snapshot(self) -> PreviewSnapshot | None:
        """Return the immutable snapshot for the next action, if available."""

        return self._snapshot

    @property
    def terminal_snapshot(self) -> PreviewSnapshot | None:
        return self._terminal_snapshot

    @property
    def episode_totals(self) -> dict[str, float]:
        return {
            "r_base": self._episode_r_base,
            "r_pp": self._episode_r_pp,
            "r_total": self._episode_r_total,
        }

    def _read_state_snapshot(self) -> tuple[tuple[str, object], ...]:
        if self._read_state is None:
            return ()
        value = self._read_state(self.env)
        if not isinstance(value, Mapping):
            raise HostContractError("state_reader must return a mapping")
        return _snapshot_state(value)

    def _read_dt_checked(self) -> float:
        value = self._read_dt(self.env)
        dt = _finite(value, name="simulation dt")
        if dt <= 0.0:
            raise HostContractError("simulation dt must be positive")
        if self._dt_seconds is None:
            self._dt_seconds = dt
        elif not math.isclose(dt, self._dt_seconds, rel_tol=0.0, abs_tol=1e-12):
            raise HostContractError(
                f"simulation dt changed within an episode: "
                f"initial={self._dt_seconds}, current={dt}"
            )
        return dt

    def _make_snapshot(self, *, decision: int, time_seconds: float) -> PreviewSnapshot:
        preview: PreviewState | None = None
        pp: PPReference | None = None
        # A baseline run may receive a provider solely for common geometry
        # diagnostics.  It still returns the raw 262-vector and host reward;
        # the provider is only read once to produce the shared snapshot.
        if self._preview_provider is not None:
            assert self._preview_provider is not None
            preview = PreviewState.from_object(self._preview_provider(self.env))
            if self._mode == "lookahead_obs_pp_reward":
                assert self._pp_provider is not None
                pp = PPReference.from_object(
                    self._pp_provider(self.env, preview),
                    preview=preview,
                )
        return PreviewSnapshot(
            decision=decision,
            t_seconds=float(time_seconds),
            preview=preview,
            pp=pp,
            state=self._read_state_snapshot(),
        )

    def _format_observation(
        self,
        raw_observation: object,
        snapshot: PreviewSnapshot,
    ) -> np.ndarray:
        raw = self._contract.validate_raw(raw_observation)
        if self._mode == "baseline":
            # Preserve the host array and values exactly for baseline.
            return raw
        assert snapshot.preview is not None
        return self._contract.append_preview(
            raw,
            snapshot.preview.observation_values,
        )

    def _namespace_info(
        self,
        info: object,
        *,
        action: object | None,
        pre_snapshot: PreviewSnapshot | None,
        post_snapshot: PreviewSnapshot,
        r_base: float,
        r_pp: float,
        r_total: float,
        terminated: bool,
        truncated: bool,
        u_applied: float | None,
        u_previous: float | None,
        e_pp: float | None,
        applied_throttle: float | None,
        previous_applied_action: tuple[float, float] | None,
    ) -> dict[str, object]:
        if not isinstance(info, Mapping):
            raise HostContractError("host reset/step info must be a mapping")
        result = dict(info)
        if "lookahead_learning" in result:
            raise HostContractError(
                "host info already contains lookahead_learning; refusing to overwrite it"
            )
        dt = self._dt_seconds
        flat_before = _snapshot_flat_fields(pre_snapshot)
        flat_after = _snapshot_flat_fields(post_snapshot)
        # Transition metrics are aligned to s_t for preview/PP and to s_{t+1}
        # for physical state.  The complete snapshots below retain both sides
        # for auditability; these aliases are the canonical telemetry surface.
        transition_fields = dict(flat_before if pre_snapshot is not None else flat_after)
        transition_fields.update(
            {
                "position": flat_after["position"],
                "heading": flat_after["heading"],
                "speed_mps": flat_after["speed_mps"],
                "progress_m": flat_after["progress_m"],
                "reference_lane_ids": flat_after["reference_lane_ids"],
                "navigation_reference_ids": flat_after["navigation_reference_ids"],
            }
        )
        before_progress = flat_before.get("progress_m")
        after_progress = flat_after.get("progress_m")
        if isinstance(before_progress, (int, float)) and isinstance(
            after_progress, (int, float)
        ):
            transition_fields["progress_delta_m"] = float(after_progress) - float(
                before_progress
            )
        else:
            transition_fields["progress_delta_m"] = None
        before_position = (
            None if pre_snapshot is None else flat_before.get("position")
        )
        after_position = flat_after.get("position")
        physical_distance = _position_distance_m(before_position, after_position)
        before_s_number = (
            None
            if pre_snapshot is None or pre_snapshot.preview is None
            else pre_snapshot.preview.s_proj_m
        )
        after_s_number = (
            None
            if post_snapshot.preview is None
            else post_snapshot.preview.s_proj_m
        )
        state_after = dict(post_snapshot.state)
        scenario_seed = state_after.get("scenario_seed")
        metric_pp_snapshot = (
            pre_snapshot.pp
            if pre_snapshot is not None and pre_snapshot.pp is not None
            else post_snapshot.pp
        )
        namespace: dict[str, object] = {
            "run": self._run_id,
            "mode": self._mode,
            "raw_observation_dim": BASELINE_OBS_DIM,
            "observation_dim": (
                BASELINE_OBS_DIM if self._mode == "baseline" else AUGMENTED_OBS_DIM
            ),
            "decision": post_snapshot.decision,
            "t_seconds": post_snapshot.t_seconds,
            "t": None if pre_snapshot is None else pre_snapshot.t_seconds,
            "t_next": post_snapshot.t_seconds,
            "dt_seconds": dt,
            "action_env": None if action is None else _action_record(action),
            "scenario_seed": scenario_seed,
            "pre_step": None if pre_snapshot is None else pre_snapshot.as_dict(),
            "post_step": post_snapshot.as_dict(),
            "position_before": before_position,
            "position_after": after_position,
            "physical_distance_m": physical_distance,
            "S_proj_before": before_s_number,
            "S_proj_after": after_s_number,
            "S_proj_delta_m": (
                None
                if before_s_number is None or after_s_number is None
                else after_s_number - before_s_number
            ),
            "progress_before_m": (
                None if pre_snapshot is None else flat_before.get("progress_m")
            ),
            "progress_after_m": flat_after.get("progress_m"),
            "u_applied": u_applied,
            "u_previous": u_previous,
            "du": (
                None if u_applied is None or u_previous is None else u_applied - u_previous
            ),
            "history_valid": (
                u_applied is not None and u_previous is not None
            ),
            "steering_rate": (
                None
                if u_applied is None
                or u_previous is None
                or dt is None
                else (u_applied - u_previous) / dt
            ),
            "yaw_rate_after": _yaw_rate_after(pre_snapshot, post_snapshot, dt),
            "u_pp": (
                None
                if metric_pp_snapshot is None
                else metric_pp_snapshot.u_pp
            ),
            "pp_valid": (
                None
                if metric_pp_snapshot is None
                else metric_pp_snapshot.pp_valid
            ),
            "e_pp": e_pp,
            "r_base": r_base,
            "r_pp": r_pp,
            "r_total": r_total,
            "episode_r_base": self._episode_r_base,
            "episode_r_pp": self._episode_r_pp,
            "episode_r_total": self._episode_r_total,
            "state_before": (
                None if pre_snapshot is None else dict(pre_snapshot.state)
            ),
            "state_after": dict(post_snapshot.state),
            "terminated": terminated,
            "truncated": truncated,
        }
        namespace.update(transition_fields)
        namespace.update(
            {
                # ``dt`` is the canonical telemetry spelling; the explicit
                # seconds alias remains for callers that use the diagnostic
                # schema directly.
                "dt": dt,
                "action_applied": (
                    None
                    if u_applied is None
                    else {
                        "steering": u_applied,
                        "throttle": applied_throttle,
                    }
                ),
                "u_throttle_applied": applied_throttle,
                "action_previous": (
                    None
                    if previous_applied_action is None
                    else {
                        "steering": previous_applied_action[0],
                        "throttle": previous_applied_action[1],
                    }
                ),
                "policy_action_raw": None if action is None else _action_record(action),
                "policy_action_stage": "env" if action is not None else None,
                "yaw_rate": _yaw_rate_after(pre_snapshot, post_snapshot, dt),
                "episode_end": bool(terminated or truncated),
                "lookahead_learning_schema_version": "lookahead_learning.env.v1",
            }
        )
        # Preserve host-defined task telemetry exactly.  In particular,
        # ``target_lane_offset_m`` is the existing start-lane metric and must
        # not be replaced by the preview projection's lateral coordinate.
        host_metrics = dict(info)
        for key in (
            "target_lane_valid",
            "target_lane_offset_m",
            "normalized_target_lane_error",
            "in_target_lane",
            "start_lane_maintained",
            "start_lane_departure",
            "target_lane_heading_error_rad",
            "start_lane_heading_error_rad",
            "start_lane_heading_error",
        ):
            if key not in host_metrics and key in state_after:
                host_metrics[key] = state_after[key]
        for key in (
            "success",
            "arrive_dest",
            "start_lane_maintained",
            "in_target_lane",
            "lane_departure",
            "start_lane_departure",
            "target_lane_valid",
            "target_lane_offset_m",
            "normalized_target_lane_error",
            "wrong_lane_arrival",
            "lane_departure_count",
            "ever_departed_target_lane",
            "off_target_duration_seconds",
            "time_in_target_lane_ratio",
            "target_lane_ordinal",
            "current_lane_ordinal",
            "target_lane_cost",
            "low_speed_penalty",
            "timeout_penalty",
            "target_lane_forward_distance_m",
            "target_lane_progress_reward",
            "target_lane_heading_error_rad",
            "start_lane_heading_error_rad",
            "start_lane_heading_error",
        ):
            if key in host_metrics:
                namespace.setdefault(key, host_metrics[key])
        # Preserve a host-provided start-lane heading metric when available.
        # Preview geometry remains the fallback for hosts that expose no such
        # task field; no heading value is inferred from the observation vector.
        for key in (
            "target_lane_heading_error_rad",
            "start_lane_heading_error_rad",
            "start_lane_heading_error",
            "heading_error_rad",
        ):
            if key not in host_metrics:
                continue
            try:
                heading_error = float(host_metrics[key])
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(heading_error):
                namespace["heading_error_rad"] = heading_error
                break
        if namespace.get("scenario_seed") is None and "scenario_seed" in info:
            namespace["scenario_seed"] = info["scenario_seed"]
        if "start_lane_maintained" not in namespace:
            maintained = host_metrics.get("in_target_lane")
            if isinstance(maintained, (bool, np.bool_)):
                namespace["start_lane_maintained"] = bool(maintained)
        host_target_metric_present = any(
            key in host_metrics
            for key in ("target_lane_valid", "target_lane_offset_m")
        )
        if host_target_metric_present:
            # The project task's target-lane metric is the common lateral
            # error.  When the host says that reference is unavailable, keep
            # the value missing instead of substituting preview projection
            # onto a later route lane.
            target_valid = host_metrics.get("target_lane_valid")
            target_offset = host_metrics.get("target_lane_offset_m")
            target_number: float | None = None
            if target_offset is not None:
                try:
                    candidate = float(target_offset)
                except (TypeError, ValueError, OverflowError):
                    candidate = math.nan
                if math.isfinite(candidate):
                    target_number = candidate
            if isinstance(target_valid, (bool, np.bool_)) and not bool(target_valid):
                namespace["lateral_error_m"] = None
            else:
                namespace["lateral_error_m"] = target_number
        else:
            for key in ("target_lane_offset_m", "lateral_error_m"):
                value = namespace.get(key)
                if value is None:
                    continue
                try:
                    number = float(value)
                except (TypeError, ValueError, OverflowError):
                    continue
                if math.isfinite(number):
                    namespace["lateral_error_m"] = number
                    break
        pp = metric_pp_snapshot
        namespace.update(
            {
                "x_rear": None if pp is None else pp.x_rear_m,
                "y_rear": None if pp is None else pp.y_rear_m,
                "kappa_pp": None if pp is None else pp.kappa_pp,
                "delta_pp_rad": None if pp is None else pp.delta_pp_rad,
                "u_pp_unclipped": None if pp is None else pp.u_pp_unclipped,
                "pp_saturated": None if pp is None else pp.saturated,
                "preview_metrics_available": post_snapshot.preview is not None,
                "pp_metrics_available": metric_pp_snapshot is not None,
                "policy_action_raw": None,
                "policy_action_unavailable_reason": (
                    "wrapper receives the environment action after policy output "
                    "processing; raw policy output is not observed"
                ),
                "policy_action_stage": None,
            }
        )
        result["lookahead_learning"] = namespace
        return result

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """Reset the raw env once, then create the s_0 snapshot."""

        if options is None:
            result = self.env.reset(seed=seed)
        else:
            result = self.env.reset(seed=seed, options=options)
        if not isinstance(result, tuple) or len(result) != 2:
            raise HostContractError("host reset must return (observation, info)")
        raw_observation, info = result
        self._contract.validate_raw(raw_observation)
        self._snapshot = None
        self._terminal_snapshot = None
        self._needs_reset = False
        self._decision = 0
        self._time_seconds = 0.0
        self._dt_seconds = None
        self._episode_r_base = 0.0
        self._episode_r_pp = 0.0
        self._episode_r_total = 0.0
        self._previous_applied_steering = None
        self._previous_applied_action = None
        # All modes share the same timing and steering diagnostics.  The
        # duration comes from the host's physics step and decision repeat.
        self._read_dt_checked()
        # Navigation/checkpoints and vehicle geometry become available only
        # after raw reset.  Rebuild each state provider exactly once per
        # episode; the wrapper owns this lifecycle so a reused PPO worker
        # cannot retain a route or vehicle geometry from its prior episode.
        reset_providers: list[object] = []
        for provider in (self._preview_provider, self._pp_provider):
            if provider is None or any(provider is item for item in reset_providers):
                continue
            reset_hook = getattr(provider, "reset", None)
            if callable(reset_hook):
                reset_hook(self.env)
            reset_providers.append(provider)
        snapshot = self._make_snapshot(decision=0, time_seconds=0.0)
        self._snapshot = snapshot
        observation = self._format_observation(raw_observation, snapshot)
        reset_info = self._namespace_info(
            info,
            action=None,
            pre_snapshot=None,
            post_snapshot=snapshot,
            r_base=0.0,
            r_pp=0.0,
            r_total=0.0,
            terminated=False,
            truncated=False,
            u_applied=None,
            u_previous=None,
            e_pp=None,
            applied_throttle=None,
            previous_applied_action=None,
        )
        return observation, reset_info

    def step(self, action: object):
        """Apply one host decision and calculate the PP reward from s_t."""

        if self._needs_reset or self._snapshot is None:
            raise RuntimeError("LookaheadEnv.step() called before reset or after done")
        pre_snapshot = self._snapshot
        previous_applied = self._previous_applied_steering
        previous_applied_action = self._previous_applied_action
        # The only simulator transition in this method is this single call.
        result = self.env.step(action)
        if not isinstance(result, tuple) or len(result) != 5:
            raise HostContractError(
                "host step must return (observation, reward, terminated, truncated, info)"
            )
        raw_observation, reward, terminated_raw, truncated_raw, info = result
        terminated = _strict_bool(terminated_raw, name="terminated")
        truncated = _strict_bool(truncated_raw, name="truncated")
        self._contract.validate_raw(raw_observation)
        r_base = _finite(reward, name="base reward")

        u_applied: float | None = None
        applied_throttle: float | None = None
        e_pp: float | None = None
        r_pp = 0.0
        dt = self._read_dt_checked()
        if self._read_applied_action is not None:
            raw_applied_action = self._read_applied_action(self.env)
            if isinstance(raw_applied_action, (str, bytes)):
                raise HostContractError(
                    "applied action reader must return steering/throttle values"
                )
            try:
                if len(raw_applied_action) < 2:
                    raise HostContractError(
                        "applied action reader must return two values"
                    )
                raw_steering = raw_applied_action[0]
                raw_throttle = raw_applied_action[1]
            except (TypeError, IndexError, KeyError) as error:
                raise HostContractError(
                    "applied action reader must return two values"
                ) from error
            u_applied = _finite(raw_steering, name="applied normalized steering")
            applied_throttle = _finite(
                raw_throttle,
                name="applied normalized throttle",
            )
        else:
            u_applied = _finite(
                self._read_applied_steering(self.env),
                name="applied normalized steering",
            )
        if not -1.0 <= u_applied <= 1.0:
            raise HostContractError(
                f"applied normalized steering must be in [-1, 1], found {u_applied}"
            )
        if self._mode == "lookahead_obs_pp_reward":
            assert pre_snapshot.pp is not None
            if pre_snapshot.pp.pp_valid and not (terminated or truncated):
                assert pre_snapshot.pp.u_pp is not None
                e_pp = abs(u_applied - pre_snapshot.pp.u_pp) / 2.0
                r_pp = -self._pp_weight * dt * e_pp
            else:
                e_pp = None
        if self._mode == "lookahead_obs_pp_reward":
            r_total = r_base + r_pp
        else:
            r_total = r_base

        next_decision = pre_snapshot.decision + 1
        assert self._dt_seconds is not None
        next_time = pre_snapshot.t_seconds + self._dt_seconds
        post_snapshot = self._make_snapshot(
            decision=next_decision,
            time_seconds=next_time,
        )
        observation = self._format_observation(raw_observation, post_snapshot)
        self._decision = next_decision
        self._time_seconds = next_time
        self._episode_r_base += r_base
        self._episode_r_pp += r_pp
        self._episode_r_total += r_total
        self._previous_applied_steering = u_applied
        self._previous_applied_action = (
            u_applied,
            0.0 if applied_throttle is None else applied_throttle,
        )
        if terminated or truncated:
            self._terminal_snapshot = post_snapshot
            self._needs_reset = True
        else:
            self._snapshot = post_snapshot
        step_info = self._namespace_info(
            info,
            action=action,
            pre_snapshot=pre_snapshot,
            post_snapshot=post_snapshot,
            r_base=r_base,
            r_pp=r_pp,
            r_total=r_total,
            terminated=terminated,
            truncated=truncated,
            u_applied=u_applied,
            u_previous=previous_applied,
            e_pp=e_pp,
            applied_throttle=applied_throttle,
            previous_applied_action=previous_applied_action,
        )
        return observation, (r_total if self._mode == "lookahead_obs_pp_reward" else reward), terminated, truncated, step_info

    def close(self):
        return self.env.close()


def make_preview_env(
    env: gym.Env,
    *,
    mode: Mode,
    contract: ObservationContract,
    preview_provider: PreviewProvider | None = None,
    pp_provider: PPProvider | None = None,
    pp_weight: float | None = None,
    applied_action_reader: AppliedActionReader | None = None,
    applied_steering_reader: AppliedSteeringReader | None = None,
    dt_reader: DtReader | None = None,
    state_reader: StateReader | None = None,
    run_id: str = "",
) -> LookaheadEnv:
    """Convenience factory kept separate from the host's raw ``make_env``."""

    return LookaheadEnv(
        env,
        mode=mode,
        contract=contract,
        preview_provider=preview_provider,
        pp_provider=pp_provider,
        pp_weight=pp_weight,
        applied_action_reader=applied_action_reader,
        applied_steering_reader=applied_steering_reader,
        dt_reader=dt_reader,
        state_reader=state_reader,
        run_id=run_id,
    )


__all__ = [
    "AppliedActionReader",
    "AppliedSteeringReader",
    "DtReader",
    "PPProvider",
    "PPReference",
    "LookaheadEnv",
    "PreviewProvider",
    "PreviewSnapshot",
    "PreviewState",
    "StateReader",
    "make_preview_env",
]
