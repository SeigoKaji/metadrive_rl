"""Pure forward-preview and pure-pursuit geometry.

The extension deliberately does not import MetaDrive.  A host adapter supplies
objects implementing the small lane protocol documented by :class:`LaneLike`
and, when the host type is not literally named ``StraightLane`` or
``CircularLane``, supplies :class:`LaneMetadata`.  This keeps route validation,
projection, preview features, and the PP reward term deterministic and easy to
test without creating a simulator.

All distances are metres and all angles passed to or returned by this module
are radians unless a field explicitly ends in ``_deg``.  ``x_g`` is forward
positive and ``y_g`` is left positive in the vehicle frame.  The route is a
fixed, ordered prefix selected after reset; this module never substitutes a
nearby lane, extrapolates beyond a lane, or clamps a lookahead point at the
end of a route.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import math
from typing import Any, Callable, Iterable, Mapping, NamedTuple, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable


Point2 = Tuple[float, float]
MetadataSource = Union[
    Mapping[Any, Any],
    Sequence[Any],
    Callable[[Any], Any],
    None,
]


class GeometryError(ValueError):
    """A lane protocol or finite-geometry error raised during configuration."""


class NonFiniteGeometryError(GeometryError):
    """A lane callback returned a NaN/Inf geometry value."""

DEFAULT_ENDPOINT_TOLERANCE_M = 1.0e-3
DEFAULT_WIDTH_TOLERANCE_M = 1.0e-3
DEFAULT_TANGENT_TOLERANCE_RAD = 1.0e-3
DEFAULT_AMBIGUITY_TOLERANCE_M = 1.0e-3
DEFAULT_LOOKAHEAD_M = 6.0
NORMALIZATION_DISTANCE_M = 10.0


@runtime_checkable
class LaneLike(Protocol):
    """Minimal host supplied lane protocol.

    ``length`` may be a finite numeric property or a zero-argument method.
    The remaining methods are called with the exact signatures shown here.
    ``position`` and ``local_coordinates`` may return any finite 2-D sequence,
    including a NumPy/Panda vector supplied by the host.
    """

    length: float

    def position(self, s: float, lateral: float) -> Sequence[float]:
        ...

    def local_coordinates(self, point: Sequence[float]) -> Sequence[float]:
        ...

    def heading_theta_at(self, s: float) -> float:
        ...


@dataclass(frozen=True)
class LaneMetadata:
    """Adapter supplied facts used to validate a lane in a fixed route.

    ``lane_type`` must identify one of the supported planar lane geometries.
    ``lane_count`` and ``width_m`` are optional on this descriptor so a single
    lane can be inspected in isolation.  ``build_reference_route`` requires
    both values before accepting a lane into a verified route and compares
    them at each connection.  ``lane_id`` is diagnostic only and is never
    used as a connection decision.  ``route_key`` can hold an adapter's
    directed-road identifier for diagnostics.
    """

    lane_type: str
    lane_count: Optional[int] = None
    width_m: Optional[float] = None
    lane_id: Optional[str] = None
    route_key: Optional[str] = None

    def __post_init__(self) -> None:
        kind = _normalise_lane_type(self.lane_type)
        if kind is None:
            raise ValueError(
                "lane_type must identify a supported StraightLane or CircularLane"
            )
        object.__setattr__(self, "lane_type", kind)
        if self.lane_count is not None:
            if isinstance(self.lane_count, bool) or not isinstance(
                self.lane_count, int
            ) or self.lane_count <= 0:
                raise ValueError("lane_count must be a positive integer or None")
        if self.width_m is not None:
            width = _finite_float(self.width_m, "width_m")
            if width <= 0.0:
                raise ValueError("width_m must be positive or None")
            object.__setattr__(self, "width_m", width)
        if self.lane_id is not None:
            object.__setattr__(self, "lane_id", str(self.lane_id))
        if self.route_key is not None:
            object.__setattr__(self, "route_key", str(self.route_key))

    def as_dict(self) -> dict[str, Any]:
        return {
            "lane_type": self.lane_type,
            "lane_count": self.lane_count,
            "width_m": self.width_m,
            "lane_id": self.lane_id,
            "route_key": self.route_key,
        }


@dataclass(frozen=True)
class RouteIssue:
    """One route validation issue, retained at the first unsupported boundary."""

    code: str
    lane_index: Optional[int]
    message: str
    details: Tuple[Tuple[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "lane_index": self.lane_index,
            "message": self.message,
            "details": {key: _to_dict(value) for key, value in self.details},
        }


@dataclass(frozen=True)
class RouteDiagnostics:
    """Immutable diagnostics for a fixed route and its validated prefix."""

    total_lanes: int
    validated_lanes: int
    boundary_reason: str
    boundary_lane_index: Optional[int]
    boundary_s: float
    issues: Tuple[RouteIssue, ...] = ()

    @property
    def route_end(self) -> bool:
        return self.boundary_reason == "route_end"

    @property
    def validated_prefix_length_m(self) -> float:
        return self.boundary_s

    @property
    def valid_prefix_length_m(self) -> float:
        return self.boundary_s

    @property
    def first_issue(self) -> Optional[RouteIssue]:
        return self.issues[0] if self.issues else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_lanes": self.total_lanes,
            "validated_lanes": self.validated_lanes,
            "boundary_reason": self.boundary_reason,
            "boundary_lane_index": self.boundary_lane_index,
            "boundary_s": self.boundary_s,
            "route_end": self.route_end,
            "issues": [issue.as_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class RouteBuildResult:
    """Result returned by :func:`build_reference_route`.

    ``path`` contains only the connected, supported prefix.  A caller may use
    that prefix for diagnostics, but a preview whose goal crosses its boundary
    is invalid and must not be shortened to the boundary.
    """

    path: "ReferencePath"
    diagnostics: RouteDiagnostics

    @property
    def valid(self) -> bool:
        return self.diagnostics.validated_lanes > 0

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path.as_dict(), "diagnostics": self.diagnostics.as_dict()}


class RouteConnection(NamedTuple):
    """Diagnostic connection measurements between two lane objects."""

    endpoint_distance_m: float
    tangent_error_rad: float
    lane_count_match: Optional[bool]
    width_match: Optional[bool]


@dataclass(frozen=True)
class _Segment:
    lane: Any
    metadata: LaneMetadata
    start_s: float
    end_s: float
    lane_index: int


@dataclass(frozen=True)
class ReferencePath:
    """Finite, ordered, immutable lane prefix parameterised by cumulative ``S``."""

    _segments: Tuple[_Segment, ...]
    diagnostics: RouteDiagnostics

    @property
    def lanes(self) -> Tuple[Any, ...]:
        return tuple(segment.lane for segment in self._segments)

    @property
    def metadata(self) -> Tuple[LaneMetadata, ...]:
        return tuple(segment.metadata for segment in self._segments)

    @property
    def lane_starts(self) -> Tuple[float, ...]:
        return tuple(segment.start_s for segment in self._segments)

    @property
    def lane_ends(self) -> Tuple[float, ...]:
        return tuple(segment.end_s for segment in self._segments)

    @property
    def lengths(self) -> Tuple[float, ...]:
        return tuple(segment.end_s - segment.start_s for segment in self._segments)

    @property
    def total_length(self) -> float:
        return self.diagnostics.boundary_s

    @property
    def valid_length(self) -> float:
        return self.total_length

    @property
    def boundary_reason(self) -> str:
        return self.diagnostics.boundary_reason

    @property
    def boundary_s(self) -> float:
        return self.diagnostics.boundary_s

    def _segment_for_s(self, s: float) -> Tuple[_Segment, float]:
        value = _finite_float(s, "s")
        if value < 0.0 or value > self.total_length:
            raise ValueError(
                f"s={value!r} is outside the finite validated route [0, {self.total_length}]"
            )
        if not self._segments:
            raise ValueError("reference route has no validated lane")
        # bisect_right gives the previous segment at an interior shared
        # endpoint; at the route end the final segment is selected.
        starts = self.lane_starts
        index = bisect_right(starts, value) - 1
        if index < 0:
            index = 0
        if index >= len(self._segments):
            index = len(self._segments) - 1
        segment = self._segments[index]
        local_s = value - segment.start_s
        if local_s < 0.0 or local_s > segment.end_s - segment.start_s:
            # This is only reachable at a floating-point boundary.  A route
            # position must never be extrapolated, so reject rather than clamp.
            raise ValueError("cumulative distance selected no finite lane interval")
        return segment, local_s

    def lane_at_s(self, s: float) -> Any:
        return self._segment_for_s(s)[0].lane

    def lane_index_at_s(self, s: float) -> int:
        return self._segment_for_s(s)[0].lane_index

    def position_at(self, s: float) -> Point2:
        segment, local_s = self._segment_for_s(s)
        return _lane_position(segment.lane, local_s, 0.0)

    def heading_at(self, s: float) -> float:
        segment, local_s = self._segment_for_s(s)
        return _lane_heading(segment.lane, local_s)

    # Names used by adapters that treat the path as P(S).
    position = position_at
    heading_theta_at = heading_at

    def as_dict(self) -> dict[str, Any]:
        return {
            "lanes": [
                {
                    "lane_index": segment.lane_index,
                    "start_s": segment.start_s,
                    "end_s": segment.end_s,
                    "length_m": segment.end_s - segment.start_s,
                    "metadata": _to_dict(segment.metadata),
                }
                for segment in self._segments
            ],
            "total_length_m": self.total_length,
            "boundary_reason": self.boundary_reason,
            "diagnostics": self.diagnostics.as_dict(),
        }


@dataclass(frozen=True)
class ProjectionCandidate:
    """Finite candidate returned while projecting a point onto one lane."""

    lane_index: int
    s_local: float
    s_global: float
    lateral_m: float
    distance_m: float
    is_endpoint: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "lane_index": self.lane_index,
            "s_local": self.s_local,
            "s_global": self.s_global,
            "lateral_m": self.lateral_m,
            "distance_m": self.distance_m,
            "is_endpoint": self.is_endpoint,
        }


@dataclass(frozen=True)
class ProjectionResult:
    """Finite projection result, including ambiguity diagnostics."""

    valid: bool
    s_proj: Optional[float]
    lane_index: Optional[int]
    lateral_m: Optional[float]
    distance_m: Optional[float]
    reason: Optional[str]
    candidates: Tuple[ProjectionCandidate, ...] = ()
    lane_id: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "s_proj": self.s_proj,
            "lane_index": self.lane_index,
            "lateral_m": self.lateral_m,
            "distance_m": self.distance_m,
            "reason": self.reason,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "lane_id": self.lane_id,
        }


@dataclass(frozen=True)
class PreviewResult:
    """Shared preview geometry and the exact three observation values."""

    valid: bool
    observation: Tuple[float, float, float]
    q: Optional[Point2]
    p: Optional[Point2]
    psi_rad: Optional[float]
    x_g: Optional[float]
    y_g: Optional[float]
    x_normalized: float
    y_normalized: float
    preview_valid: int
    s_proj: Optional[float]
    s_goal: Optional[float]
    projected_lane_index: Optional[int]
    goal_lane_index: Optional[int]
    heading_error_rad: Optional[float]
    distance_to_goal_m: Optional[float]
    x_clipped: bool
    y_clipped: bool
    reason: Optional[str]
    projection: Optional[ProjectionResult] = None
    forward_speed_mps: Optional[float] = None

    @property
    def features(self) -> Tuple[float, float, float]:
        return self.observation

    @property
    def normalized(self) -> Tuple[float, float, float]:
        return self.observation

    @property
    def preview_point(self) -> Optional[Point2]:
        return self.q

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "observation": list(self.observation),
            "q": list(self.q) if self.q is not None else None,
            "p": list(self.p) if self.p is not None else None,
            "psi_rad": self.psi_rad,
            "x_g": self.x_g,
            "y_g": self.y_g,
            "x_normalized": self.x_normalized,
            "y_normalized": self.y_normalized,
            "preview_valid": self.preview_valid,
            "s_proj": self.s_proj,
            "s_goal": self.s_goal,
            "projected_lane_index": self.projected_lane_index,
            "goal_lane_index": self.goal_lane_index,
            "heading_error_rad": self.heading_error_rad,
            "distance_to_goal_m": self.distance_to_goal_m,
            "x_clipped": self.x_clipped,
            "y_clipped": self.y_clipped,
            "reason": self.reason,
            "projection": self.projection.as_dict() if self.projection else None,
            "forward_speed_mps": self.forward_speed_mps,
        }


@dataclass(frozen=True)
class PurePursuitResult:
    """Pure-pursuit reference computed from the shared preview point."""

    valid: bool
    pp_valid: bool
    q: Optional[Point2]
    rear_position: Optional[Point2]
    x_rear: Optional[float]
    y_rear: Optional[float]
    wheelbase_m: Optional[float]
    rear_wheelbase_m: Optional[float]
    kappa_pp: Optional[float]
    delta_pp_rad: Optional[float]
    delta_pp_deg: Optional[float]
    u_pp_unclipped: Optional[float]
    u_pp: Optional[float]
    steering_sign: Optional[float]
    max_steering_deg: Optional[float]
    saturated: bool
    reason: Optional[str]

    @property
    def observation_preview_valid(self) -> Optional[int]:
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "pp_valid": self.pp_valid,
            "q": list(self.q) if self.q is not None else None,
            "rear_position": list(self.rear_position)
            if self.rear_position is not None
            else None,
            "x_rear": self.x_rear,
            "y_rear": self.y_rear,
            "wheelbase_m": self.wheelbase_m,
            "rear_wheelbase_m": self.rear_wheelbase_m,
            "kappa_pp": self.kappa_pp,
            "delta_pp_rad": self.delta_pp_rad,
            "delta_pp_deg": self.delta_pp_deg,
            "u_pp_unclipped": self.u_pp_unclipped,
            "u_pp": self.u_pp,
            "steering_sign": self.steering_sign,
            "max_steering_deg": self.max_steering_deg,
            "saturated": self.saturated,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PPPenaltyResult:
    """Pure PP reward term and the values used to produce it."""

    r_pp: float
    e_pp: Optional[float]
    mask: bool
    u_applied: Optional[float]
    u_pp: Optional[float]
    dt_s: float
    pp_weight: float
    terminated: bool
    truncated: bool

    @property
    def reward(self) -> float:
        return self.r_pp

    def as_dict(self) -> dict[str, Any]:
        return {
            "r_pp": self.r_pp,
            "e_pp": self.e_pp,
            "mask": self.mask,
            "u_applied": self.u_applied,
            "u_pp": self.u_pp,
            "dt_s": self.dt_s,
            "pp_weight": self.pp_weight,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_optional_float(value: Any, name: str) -> Optional[float]:
    if value is None:
        return None
    return _finite_float(value, name)


def _point2(value: Any, name: str = "point") -> Point2:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a finite 2-D sequence")
    try:
        values = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a finite 2-D sequence") from exc
    if len(values) < 2:
        raise ValueError(f"{name} must contain at least two coordinates")
    return (_finite_float(values[0], f"{name}[0]"), _finite_float(values[1], f"{name}[1]"))


def _point2_state(value: Any, name: str = "point") -> Tuple[Optional[Point2], bool]:
    """Parse state geometry while distinguishing malformed shape from NaN/Inf."""

    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a finite 2-D sequence")
    try:
        values = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a finite 2-D sequence") from exc
    if len(values) < 2:
        raise ValueError(f"{name} must contain at least two coordinates")
    parsed: list[float] = []
    for index in range(2):
        if isinstance(values[index], bool):
            raise TypeError(f"{name}[{index}] must be a real number")
        try:
            number = float(values[index])
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(f"{name}[{index}] must be a real number") from exc
        if not math.isfinite(number):
            return None, True
        parsed.append(number)
    return (parsed[0], parsed[1]), False


def _scalar_state(value: Any, name: str) -> Tuple[Optional[float], bool]:
    """Parse a state scalar, returning ``(None, True)`` for NaN/Inf."""

    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not math.isfinite(number):
        return None, True
    return number, False


def _callback_float(value: Any, name: str) -> float:
    """Parse a callback return value, distinguishing non-finite geometry."""

    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not math.isfinite(number):
        raise NonFiniteGeometryError(f"{name} returned non-finite geometry")
    return number


def _normalise_lane_type(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).rsplit(".", 1)[-1].strip().lower()
    text = text.replace("_", "").replace("-", "")
    if text.endswith("lane"):
        text = text[:-4]
    if text in {"straight", "straightlane"}:
        return "StraightLane"
    if text in {"circular", "circularlane", "circle"}:
        return "CircularLane"
    return None


def _lookup_mapping(source: Mapping[Any, Any], lane: Any, index: int) -> Any:
    for key in (lane, id(lane), index):
        try:
            if key in source:
                return source[key]
        except (TypeError, KeyError):
            continue
    for attr in ("lane_id", "id", "index", "ordinal"):
        try:
            key = getattr(lane, attr)
        except Exception:  # pragma: no cover - hostile host object
            continue
        try:
            if key in source:
                return source[key]
        except (TypeError, KeyError):
            continue
    return None


def _metadata_for_lane(source: MetadataSource, lane: Any, index: int) -> Any:
    if source is None:
        return None
    if isinstance(source, LaneMetadata):
        return source
    if callable(source):
        return source(lane)
    if isinstance(source, Mapping):
        # A direct descriptor is convenient for a one-lane route.  A mapping
        # without these descriptor keys is interpreted as lane->metadata or
        # index->metadata below.
        if any(key in source for key in ("lane_type", "type", "kind")):
            return source
        return _lookup_mapping(source, lane, index)
    try:
        return source[index]  # type: ignore[index]
    except (IndexError, KeyError, TypeError):
        return None


def _coerce_metadata(lane: Any, source: MetadataSource, index: int) -> LaneMetadata:
    raw = _metadata_for_lane(source, lane, index)
    if raw is None:
        raw = lane

    if isinstance(raw, LaneMetadata):
        return raw
    if isinstance(raw, Mapping):
        kind = raw.get("lane_type", raw.get("type", raw.get("kind")))
        lane_count = raw.get("lane_count", raw.get("lane_num", raw.get("lane_number")))
        width = raw.get("width_m", raw.get("width"))
        lane_id = raw.get("lane_id", raw.get("id"))
        route_key = raw.get("route_key", raw.get("road_id", raw.get("road")))
    else:
        kind = getattr(raw, "lane_type", getattr(raw, "type", None))
        lane_count = getattr(
            raw,
            "lane_count",
            getattr(raw, "lane_num", getattr(raw, "lane_number", None)),
        )
        width = getattr(raw, "width_m", getattr(raw, "width", None))
        lane_id = getattr(raw, "lane_id", getattr(raw, "id", None))
        route_key = getattr(raw, "route_key", getattr(raw, "road_id", None))
        if kind is None:
            kind = type(lane).__name__

    normalised = _normalise_lane_type(kind)
    if normalised is None:
        raise ValueError(
            f"unsupported_lane_type at lane {index}: {kind!r}; "
            "adapter metadata must identify StraightLane or CircularLane"
        )
    parsed_count: Optional[int]
    if lane_count is None:
        parsed_count = None
    else:
        if isinstance(lane_count, bool):
            raise ValueError(f"invalid lane_count at lane {index}")
        try:
            parsed_count = int(lane_count)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"invalid lane_count at lane {index}") from exc
        if parsed_count <= 0 or float(parsed_count) != float(lane_count):
            raise ValueError(f"invalid lane_count at lane {index}")
    parsed_width = None if width is None else _finite_float(width, f"width_m at lane {index}")
    if parsed_width is not None and parsed_width <= 0.0:
        raise ValueError(f"width_m at lane {index} must be positive")
    return LaneMetadata(normalised, parsed_count, parsed_width, lane_id, route_key)


def _lane_length(lane: Any) -> float:
    value = getattr(lane, "length")
    if callable(value):
        value = value()
    result = _callback_float(value, "lane.length")
    if result <= 0.0:
        raise ValueError("lane.length must be positive")
    return result


def _lane_position(lane: Any, s: float, lateral: float) -> Point2:
    # Keep the callback outside parse handling: TypeError/ValueError raised by
    # a host callback are programming/configuration errors and must propagate.
    value = lane.position(s, lateral)
    point, nonfinite = _point2_state(value, "lane.position result")
    if nonfinite:
        raise NonFiniteGeometryError("lane.position returned non-finite geometry")
    assert point is not None
    return point


def _lane_local_coordinates(lane: Any, point: Point2) -> Tuple[float, float]:
    # Callback exceptions intentionally remain visible to the caller.
    value = tuple(lane.local_coordinates(point))
    if len(value) < 2:
        raise TypeError("lane.local_coordinates must return (longitudinal, lateral)")
    longitudinal = _callback_float(
        value[0], "lane.local_coordinates longitudinal"
    )
    lateral = _callback_float(value[1], "lane.local_coordinates lateral")
    return longitudinal, lateral


def _lane_heading(lane: Any, s: float) -> float:
    # Callback exceptions intentionally remain visible to the caller.
    value = lane.heading_theta_at(s)
    return _callback_float(value, "lane heading")


def wrap_to_pi(angle_rad: float) -> float:
    """Wrap a finite angle to ``[-pi, pi)`` without NumPy dependencies."""

    value = _finite_float(angle_rad, "angle_rad")
    wrapped = (value + math.pi) % (2.0 * math.pi) - math.pi
    # Avoid returning +pi due to a platform-specific modulo edge.
    return -math.pi if wrapped >= math.pi else wrapped


def angle_difference(a_rad: float, b_rad: float) -> float:
    """Return the signed wrapped difference ``a_rad - b_rad``."""

    return wrap_to_pi(_finite_float(a_rad, "a_rad") - _finite_float(b_rad, "b_rad"))


def normalize_preview_value(value_m: float, distance_m: float = NORMALIZATION_DISTANCE_M) -> float:
    """Map a signed metre value to the observation encoding ``[0, 1]``."""

    value = _finite_float(value_m, "value_m")
    scale = _finite_float(distance_m, "distance_m")
    if scale <= 0.0:
        raise ValueError("distance_m must be positive")
    clipped = max(-1.0, min(1.0, value / scale))
    return (clipped + 1.0) / 2.0


def validate_lane(lane: Any, *, metadata: MetadataSource = None, index: int = 0) -> LaneMetadata:
    """Validate one lane and return normalized adapter metadata.

    Validation is intentionally eager.  A malformed lane is a configuration
    error and must not be turned into a per-step ``preview_valid=0`` value.
    """

    lane_length = _lane_length(lane)
    md = _coerce_metadata(lane, metadata, index)
    _validate_lane_protocol(lane, lane_length)
    return md


def _validate_lane_protocol(lane: Any, lane_length: float) -> None:
    """Validate lane callbacks after metadata has been resolved."""

    # Check required protocol methods and endpoint values without changing
    # state.  These calls are pure for real MetaDrive lanes and fake test lanes.
    start_position = _lane_position(lane, 0.0, 0.0)
    _lane_position(lane, lane_length, 0.0)
    _lane_heading(lane, 0.0)
    _lane_heading(lane, lane_length)
    _lane_local_coordinates(lane, start_position)


def connection_measurement(
    previous_lane: Any,
    next_lane: Any,
    *,
    previous_metadata: Optional[LaneMetadata] = None,
    next_metadata: Optional[LaneMetadata] = None,
    width_tolerance_m: float = DEFAULT_WIDTH_TOLERANCE_M,
) -> RouteConnection:
    """Measure the endpoint/tangent and optional lane-attribute agreement."""

    previous_length = _lane_length(previous_lane)
    previous_end = _lane_position(previous_lane, previous_length, 0.0)
    next_start = _lane_position(next_lane, 0.0, 0.0)
    endpoint_distance = math.hypot(
        previous_end[0] - next_start[0], previous_end[1] - next_start[1]
    )
    tangent_error = abs(
        angle_difference(
            _lane_heading(next_lane, 0.0),
            _lane_heading(previous_lane, previous_length),
        )
    )
    lane_count_match: Optional[bool] = None
    if previous_metadata is not None and next_metadata is not None:
        if previous_metadata.lane_count is not None and next_metadata.lane_count is not None:
            lane_count_match = previous_metadata.lane_count == next_metadata.lane_count
    width_match: Optional[bool] = None
    width_tol = _finite_float(width_tolerance_m, "width_tolerance_m")
    if width_tol < 0.0:
        raise ValueError("width_tolerance_m must be non-negative")
    if previous_metadata is not None and next_metadata is not None:
        if previous_metadata.width_m is not None and next_metadata.width_m is not None:
            width_match = math.isclose(
                previous_metadata.width_m,
                next_metadata.width_m,
                rel_tol=0.0,
                abs_tol=width_tol,
            )
    return RouteConnection(endpoint_distance, tangent_error, lane_count_match, width_match)


def _issue(code: str, index: Optional[int], message: str, **details: Any) -> RouteIssue:
    return RouteIssue(code, index, message, tuple(details.items()))


def build_reference_route(
    lanes: Iterable[Any],
    *,
    metadata: MetadataSource = None,
    endpoint_tolerance_m: float = DEFAULT_ENDPOINT_TOLERANCE_M,
    width_tolerance_m: float = DEFAULT_WIDTH_TOLERANCE_M,
    tangent_tolerance_rad: float = DEFAULT_TANGENT_TOLERANCE_RAD,
) -> RouteBuildResult:
    """Build a fixed connected prefix from an ordered lane sequence.

    The input sequence is the adapter's already planned directed route.  This
    function only verifies that order geometrically continues; it does not
    search neighboring lanes or infer a route from lane numbers.  If lane ``i``
    is unsupported or fails validation, lanes before ``i`` remain as the
    validated prefix and ``boundary_reason`` identifies why no later lane can
    be used.
    """

    endpoint_tol = _finite_float(endpoint_tolerance_m, "endpoint_tolerance_m")
    width_tol = _finite_float(width_tolerance_m, "width_tolerance_m")
    tangent_tol = _finite_float(tangent_tolerance_rad, "tangent_tolerance_rad")
    if endpoint_tol < 0.0 or width_tol < 0.0 or tangent_tol < 0.0:
        raise ValueError("route tolerances must be non-negative")
    lane_tuple = tuple(lanes)
    segments: list[_Segment] = []
    issues: list[RouteIssue] = []
    cumulative = 0.0
    boundary_reason = "route_end"
    boundary_lane_index: Optional[int] = None
    previous_lane: Any = None
    previous_md: Optional[LaneMetadata] = None

    for index, lane in enumerate(lane_tuple):
        try:
            # Metadata errors are route configuration diagnostics.  Callback
            # exceptions are deliberately handled in the separate protocol
            # block below so a host programming error is never swallowed.
            md = _coerce_metadata(lane, metadata, index)
        except (TypeError, ValueError) as exc:
            boundary_reason = _classify_validation_issue(str(exc))
            boundary_lane_index = index
            issues.append(_issue(boundary_reason, index, str(exc)))
            break
        try:
            length = _lane_length(lane)
            _validate_lane_protocol(lane, length)
        except GeometryError as exc:
            boundary_reason = _classify_validation_issue(str(exc))
            boundary_lane_index = index
            issues.append(_issue(boundary_reason, index, str(exc)))
            break

        if md.lane_count is None or md.width_m is None:
            boundary_reason = "missing_lane_metadata"
            boundary_lane_index = index
            missing = []
            if md.lane_count is None:
                missing.append("lane_count")
            if md.width_m is None:
                missing.append("width_m")
            issues.append(
                _issue(
                    boundary_reason,
                    index,
                    "adapter metadata must provide lane count and width",
                    missing=tuple(missing),
                )
            )
            break

        if previous_lane is not None and previous_md is not None:
            try:
                connection = connection_measurement(
                    previous_lane,
                    lane,
                    previous_metadata=previous_md,
                    next_metadata=md,
                    width_tolerance_m=width_tol,
                )
            except GeometryError as exc:
                boundary_reason = "connection_geometry_error"
                boundary_lane_index = index
                issues.append(_issue(boundary_reason, index, str(exc)))
                break
            if connection.endpoint_distance_m > endpoint_tol:
                boundary_reason = "discontinuous_endpoint"
                boundary_lane_index = index
                issues.append(
                    _issue(
                        boundary_reason,
                        index,
                        "lane endpoints are not coincident",
                        endpoint_distance_m=connection.endpoint_distance_m,
                        tolerance_m=endpoint_tol,
                    )
                )
                break
            if connection.tangent_error_rad > tangent_tol:
                boundary_reason = "discontinuous_tangent"
                boundary_lane_index = index
                issues.append(
                    _issue(
                        boundary_reason,
                        index,
                        "lane tangents do not agree",
                        tangent_error_rad=connection.tangent_error_rad,
                        tolerance_rad=tangent_tol,
                    )
                )
                break
            if connection.lane_count_match is False:
                boundary_reason = "lane_count_mismatch"
                boundary_lane_index = index
                issues.append(
                    _issue(
                        boundary_reason,
                        index,
                        "lane count changes at connection",
                        previous_lane_count=previous_md.lane_count,
                        next_lane_count=md.lane_count,
                    )
                )
                break
            if connection.width_match is False:
                boundary_reason = "lane_width_mismatch"
                boundary_lane_index = index
                issues.append(
                    _issue(
                        boundary_reason,
                        index,
                        "lane width changes at connection",
                        previous_width_m=previous_md.width_m,
                        next_width_m=md.width_m,
                    )
                )
                break

        segments.append(_Segment(lane, md, cumulative, cumulative + length, index))
        cumulative += length
        previous_lane, previous_md = lane, md

    if not lane_tuple:
        boundary_reason = "empty_route"
        boundary_lane_index = None
    elif boundary_lane_index is None:
        boundary_reason = "route_end"
        boundary_lane_index = None

    diagnostics = RouteDiagnostics(
        total_lanes=len(lane_tuple),
        validated_lanes=len(segments),
        boundary_reason=boundary_reason,
        boundary_lane_index=boundary_lane_index,
        boundary_s=cumulative,
        issues=tuple(issues),
    )
    path = ReferencePath(tuple(segments), diagnostics)
    return RouteBuildResult(path, diagnostics)


def build_reference_path(
    lanes: Iterable[Any],
    *,
    metadata: MetadataSource = None,
    endpoint_tolerance_m: float = DEFAULT_ENDPOINT_TOLERANCE_M,
    width_tolerance_m: float = DEFAULT_WIDTH_TOLERANCE_M,
    tangent_tolerance_rad: float = DEFAULT_TANGENT_TOLERANCE_RAD,
) -> ReferencePath:
    """Build a fixed route and return its immutable path.

    For callers that need the diagnostic wrapper, use
    :func:`build_reference_route`; the returned path retains the same
    ``RouteDiagnostics`` object.
    """

    return build_reference_route(
        lanes,
        metadata=metadata,
        endpoint_tolerance_m=endpoint_tolerance_m,
        width_tolerance_m=width_tolerance_m,
        tangent_tolerance_rad=tangent_tolerance_rad,
    ).path


def _classify_validation_issue(message: str) -> str:
    text = message.lower()
    if "unsupported_lane_type" in text:
        return "unsupported_lane_type"
    if "length" in text:
        return "invalid_lane_length"
    if "position" in text or "heading" in text or "local_coordinates" in text:
        return "lane_protocol_error"
    if "finite" in text:
        return "nonfinite_lane_geometry"
    return "invalid_lane"


def connected_successor_candidates(
    previous_lane: Any,
    candidates: Iterable[Any],
    *,
    previous_metadata: Optional[LaneMetadata] = None,
    metadata: MetadataSource = None,
    endpoint_tolerance_m: float = DEFAULT_ENDPOINT_TOLERANCE_M,
    width_tolerance_m: float = DEFAULT_WIDTH_TOLERANCE_M,
    tangent_tolerance_rad: float = DEFAULT_TANGENT_TOLERANCE_RAD,
) -> Tuple[Tuple[Any, ...], RouteDiagnostics]:
    """Filter successors by directed geometry and return ambiguity diagnostics.

    Lane number or object identity may be used by an adapter to form the
    ``candidates`` list, but this function makes the final decision from
    endpoint, tangent, lane count, and width checks.  More than one connected
    candidate is rejected as ``ambiguous_successor``.
    """

    candidate_tuple = tuple(candidates)
    if previous_metadata is None:
        previous_metadata = validate_lane(previous_lane)
    if previous_metadata.lane_count is None or previous_metadata.width_m is None:
        raise ValueError(
            "previous lane metadata must provide lane_count and width_m"
        )
    accepted: list[Any] = []
    issues: list[RouteIssue] = []
    for index, candidate in enumerate(candidate_tuple):
        try:
            next_md = validate_lane(candidate, metadata=metadata, index=index)
            if next_md.lane_count is None or next_md.width_m is None:
                raise ValueError(
                    "successor metadata must provide lane_count and width_m"
                )
            connection = connection_measurement(
                previous_lane,
                candidate,
                previous_metadata=previous_metadata,
                next_metadata=next_md,
                width_tolerance_m=width_tolerance_m,
            )
        except GeometryError as exc:
            issues.append(_issue("invalid_successor", index, str(exc)))
            continue
        if connection.endpoint_distance_m > _finite_float(
            endpoint_tolerance_m, "endpoint_tolerance_m"
        ):
            continue
        if connection.tangent_error_rad > _finite_float(
            tangent_tolerance_rad, "tangent_tolerance_rad"
        ):
            continue
        if connection.lane_count_match is False or connection.width_match is False:
            continue
        accepted.append(candidate)

    if len(accepted) > 1:
        issues.append(
            _issue(
                "ambiguous_successor",
                None,
                "more than one successor satisfies connection checks",
                candidate_count=len(accepted),
            )
        )
        accepted = []
        reason = "ambiguous_successor"
    elif accepted:
        reason = "connected"
    else:
        reason = "no_connected_successor"
    diagnostics = RouteDiagnostics(
        total_lanes=len(candidate_tuple),
        validated_lanes=len(accepted),
        boundary_reason=reason,
        boundary_lane_index=None,
        boundary_s=_lane_length(previous_lane) if accepted else 0.0,
        issues=tuple(issues),
    )
    return tuple(accepted), diagnostics


def _distance(a: Point2, b: Point2) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def project_to_path(
    path: ReferencePath,
    point: Sequence[float],
    *,
    endpoint_tolerance_m: float = DEFAULT_ENDPOINT_TOLERANCE_M,
    ambiguity_tolerance_m: float = DEFAULT_AMBIGUITY_TOLERANCE_M,
) -> ProjectionResult:
    """Project a point onto finite lane intervals of ``path``.

    Candidates are obtained from every lane's ``local_coordinates`` and are
    retained only when the longitudinal coordinate lies in that lane's finite
    interval (with a small numerical endpoint tolerance).  The nearest
    candidate wins.  A tie at a shared adjacent endpoint is resolved to the
    earlier cumulative interval; a tie involving a non-adjacent interval is
    rejected as ambiguous.
    """

    if not isinstance(path, ReferencePath):
        raise TypeError("path must be a ReferencePath")
    point2, nonfinite = _point2_state(point)
    if nonfinite:
        return ProjectionResult(False, None, None, None, None, "nonfinite_point")
    assert point2 is not None
    endpoint_tol = _finite_float(endpoint_tolerance_m, "endpoint_tolerance_m")
    ambiguity_tol = _finite_float(ambiguity_tolerance_m, "ambiguity_tolerance_m")
    if endpoint_tol < 0.0 or ambiguity_tol < 0.0:
        raise ValueError("projection tolerances must be non-negative")
    candidates: list[ProjectionCandidate] = []
    for segment in path._segments:
        try:
            s_local, lateral = _lane_local_coordinates(segment.lane, point2)
            length = segment.end_s - segment.start_s

            # The finite projection of a point before or after a lane is its
            # nearest endpoint.  The out-of-range local coordinate itself is
            # never used as an extrapolated path point.  Always compare both
            # endpoints, then add an interior candidate when local_coordinates
            # reports one inside the finite interval.
            endpoint_values = [0.0, length]
            if 0.0 <= s_local <= length:
                # An in-range value is already a finite path coordinate, even
                # when it lies within the numerical endpoint tolerance.
                endpoint_values.append(s_local)
            elif -endpoint_tol <= s_local < 0.0:
                # Only an actually out-of-range value may snap to an endpoint.
                endpoint_values.append(0.0)
            elif length < s_local <= length + endpoint_tol:
                endpoint_values.append(length)
            emitted: set[float] = set()
            lane_candidates: list[ProjectionCandidate] = []
            for finite_s_local in endpoint_values:
                if finite_s_local in emitted:
                    continue
                emitted.add(finite_s_local)
                center = _lane_position(segment.lane, finite_s_local, 0.0)
                distance = _distance(point2, center)
                lane_candidates.append(
                    ProjectionCandidate(
                        lane_index=segment.lane_index,
                        s_local=finite_s_local,
                        s_global=segment.start_s + finite_s_local,
                        lateral_m=lateral,
                        distance_m=distance,
                        is_endpoint=(finite_s_local == 0.0 or finite_s_local == length),
                    )
                )
            # Endpoint candidates are needed when local_coordinates lies
            # outside the interval, but a same-lane endpoint that is merely
            # close to the interior projection is not a second route branch.
            # Keep that lane's exact nearest finite candidate for the global
            # ambiguity check.
            if lane_candidates:
                candidates.append(
                    min(lane_candidates, key=lambda candidate: candidate.distance_m)
                )
        except GeometryError as exc:
            # A malformed lane should have been rejected while constructing the
            # path.  A runtime finite-geometry failure invalidates the fixed
            # route; silently trying another lane would be an undocumented
            # fallback to a different reference.
            reason = "nonfinite_projection_geometry" if "finite" in str(exc) else "projection_geometry_error"
            return ProjectionResult(
                False,
                None,
                None,
                None,
                None,
                reason,
                tuple(candidates),
            )

    if not candidates:
        return ProjectionResult(False, None, None, None, None, "projection_unavailable")
    candidates.sort(key=lambda candidate: (candidate.distance_m, candidate.s_global))
    best = candidates[0]
    tied = [
        candidate
        for candidate in candidates[1:]
        if abs(candidate.distance_m - best.distance_m) <= ambiguity_tol
    ]
    if tied:
        # A shared endpoint is represented by consecutive lane indices and the
        # same global S.  Selecting the earlier interval is deterministic and
        # preserves the fixed route identity.
        all_tied = [best, *tied]
        same_endpoint = all(
            candidate.is_endpoint
            and best.is_endpoint
            and abs(candidate.s_global - best.s_global) <= endpoint_tol
            and abs(candidate.distance_m - best.distance_m) <= ambiguity_tol
            for candidate in all_tied
        )
        adjacent = all(
            abs(candidate.lane_index - best.lane_index) == 1 for candidate in tied
        )
        any_nonadjacent = any(
            abs(candidate.lane_index - best.lane_index) > 1 for candidate in tied
        )
        if any_nonadjacent:
            return ProjectionResult(
                False,
                None,
                None,
                None,
                None,
                "ambiguous_projection",
                tuple(candidates),
            )
        if same_endpoint and adjacent:
            best = min(all_tied, key=lambda candidate: candidate.lane_index)
    best_lane_id = None
    try:
        best_lane_id = path.metadata[best.lane_index].lane_id
    except (IndexError, TypeError):
        pass
    return ProjectionResult(
        True,
        best.s_global,
        best.lane_index,
        best.lateral_m,
        best.distance_m,
        None,
        tuple(candidates),
        best_lane_id,
    )


def _invalid_preview(
    *,
    p: Optional[Point2],
    psi_rad: Optional[float],
    reason: str,
    projection: Optional[ProjectionResult] = None,
    forward_speed_mps: Optional[float] = None,
    s_proj: Optional[float] = None,
    s_goal: Optional[float] = None,
    x_g: Optional[float] = None,
    y_g: Optional[float] = None,
    q: Optional[Point2] = None,
    projected_lane_index: Optional[int] = None,
    goal_lane_index: Optional[int] = None,
    heading_error_rad: Optional[float] = None,
    distance_to_goal_m: Optional[float] = None,
    x_clipped: bool = False,
    y_clipped: bool = False,
) -> PreviewResult:
    return PreviewResult(
        valid=False,
        observation=(0.5, 0.5, 0.0),
        q=q,
        p=p,
        psi_rad=psi_rad,
        x_g=x_g,
        y_g=y_g,
        x_normalized=0.5,
        y_normalized=0.5,
        preview_valid=0,
        s_proj=s_proj,
        s_goal=s_goal,
        projected_lane_index=projected_lane_index,
        goal_lane_index=goal_lane_index,
        heading_error_rad=heading_error_rad,
        distance_to_goal_m=distance_to_goal_m,
        x_clipped=x_clipped,
        y_clipped=y_clipped,
        reason=reason,
        projection=projection,
        forward_speed_mps=forward_speed_mps,
    )


def compute_preview(
    path: ReferencePath,
    point: Sequence[float],
    psi_rad: float,
    *,
    lookahead_m: float = DEFAULT_LOOKAHEAD_M,
    forward_speed_mps: Optional[float] = None,
    start_lane_valid: bool = True,
    endpoint_tolerance_m: float = DEFAULT_ENDPOINT_TOLERANCE_M,
    ambiguity_tolerance_m: float = DEFAULT_AMBIGUITY_TOLERANCE_M,
    normalization_distance_m: float = NORMALIZATION_DISTANCE_M,
) -> PreviewResult:
    """Compute the common preview point and ``(x_g, y_g, valid)`` values."""

    if not isinstance(path, ReferencePath):
        raise TypeError("path must be a ReferencePath")
    point2, nonfinite_point = _point2_state(point)
    if nonfinite_point:
        return _invalid_preview(
            p=None,
            psi_rad=None,
            reason="nonfinite_vehicle_geometry",
        )
    assert point2 is not None
    psi, nonfinite_heading = _scalar_state(psi_rad, "psi_rad")
    if nonfinite_heading:
        return _invalid_preview(
            p=point2,
            psi_rad=None,
            reason="nonfinite_vehicle_geometry",
        )
    assert psi is not None
    lookahead = _finite_float(lookahead_m, "lookahead_m")
    if lookahead < 0.0:
        raise ValueError("lookahead_m must be non-negative")
    normalization_distance = _finite_float(
        normalization_distance_m, "normalization_distance_m"
    )
    if normalization_distance <= 0.0:
        raise ValueError("normalization_distance_m must be positive")
    if forward_speed_mps is None:
        speed = None
    else:
        speed, nonfinite_speed = _scalar_state(forward_speed_mps, "forward_speed_mps")
        if nonfinite_speed:
            return _invalid_preview(
                p=point2,
                psi_rad=psi,
                reason="nonfinite_vehicle_geometry",
            )
    if speed is not None and speed < -0.1:
        return _invalid_preview(
            p=point2,
            psi_rad=psi,
            reason="reverse_motion",
            forward_speed_mps=speed,
        )
    if not isinstance(start_lane_valid, bool):
        raise TypeError("start_lane_valid must be bool")
    if not start_lane_valid:
        return _invalid_preview(
            p=point2,
            psi_rad=psi,
            reason="start_lane_unavailable",
            forward_speed_mps=speed,
        )
    if not path._segments:
        return _invalid_preview(
            p=point2,
            psi_rad=psi,
            reason="empty_route",
            forward_speed_mps=speed,
        )

    projection = project_to_path(
        path,
        point2,
        endpoint_tolerance_m=endpoint_tolerance_m,
        ambiguity_tolerance_m=ambiguity_tolerance_m,
    )
    if not projection.valid or projection.s_proj is None:
        return _invalid_preview(
            p=point2,
            psi_rad=psi,
            reason=projection.reason or "projection_unavailable",
            projection=projection,
            forward_speed_mps=speed,
        )
    s_proj = projection.s_proj
    s_goal = s_proj + lookahead
    # Lookahead crossing a boundary is a semantic invalidity.  It is not a
    # near-endpoint numerical connection, so do not apply endpoint tolerance.
    if s_goal > path.total_length:
        reason = (
            "lookahead_past_route_end"
            if path.boundary_reason == "route_end"
            else "lookahead_past_unvalidated_boundary"
        )
        return _invalid_preview(
            p=point2,
            psi_rad=psi,
            reason=reason,
            projection=projection,
            forward_speed_mps=speed,
            s_proj=s_proj,
            s_goal=s_goal,
            projected_lane_index=projection.lane_index,
        )
    try:
        q = path.position_at(s_goal)
        goal_heading = path.heading_at(s_goal)
    except GeometryError:
        return _invalid_preview(
            p=point2,
            psi_rad=psi,
            reason="goal_geometry_unavailable",
            projection=projection,
            forward_speed_mps=speed,
            s_proj=s_proj,
            s_goal=s_goal,
            projected_lane_index=projection.lane_index,
        )
    forward = (math.cos(psi), math.sin(psi))
    left = (-math.sin(psi), math.cos(psi))
    difference = (q[0] - point2[0], q[1] - point2[1])
    x_g = difference[0] * forward[0] + difference[1] * forward[1]
    y_g = difference[0] * left[0] + difference[1] * left[1]
    x_g = _finite_float(x_g, "x_g")
    y_g = _finite_float(y_g, "y_g")
    distance_to_goal = math.hypot(difference[0], difference[1])
    if x_g <= 1.0e-3:
        return _invalid_preview(
            p=point2,
            psi_rad=psi,
            reason="goal_not_in_front",
            projection=projection,
            forward_speed_mps=speed,
            s_proj=s_proj,
            s_goal=s_goal,
            x_g=x_g,
            y_g=y_g,
            q=q,
            projected_lane_index=projection.lane_index,
            goal_lane_index=path.lane_index_at_s(s_goal),
            heading_error_rad=angle_difference(goal_heading, psi),
            distance_to_goal_m=distance_to_goal,
            x_clipped=abs(x_g / normalization_distance) > 1.0,
            y_clipped=abs(y_g / normalization_distance) > 1.0,
        )
    if distance_to_goal <= 1.0e-3:
        return _invalid_preview(
            p=point2,
            psi_rad=psi,
            reason="goal_distance_too_small",
            projection=projection,
            forward_speed_mps=speed,
            s_proj=s_proj,
            s_goal=s_goal,
            x_g=x_g,
            y_g=y_g,
            q=q,
            projected_lane_index=projection.lane_index,
            goal_lane_index=path.lane_index_at_s(s_goal),
            heading_error_rad=angle_difference(goal_heading, psi),
            distance_to_goal_m=distance_to_goal,
            x_clipped=abs(x_g / normalization_distance) > 1.0,
            y_clipped=abs(y_g / normalization_distance) > 1.0,
        )
    x_norm = normalize_preview_value(x_g, normalization_distance)
    y_norm = normalize_preview_value(y_g, normalization_distance)
    return PreviewResult(
        valid=True,
        observation=(float(x_norm), float(y_norm), 1.0),
        q=q,
        p=point2,
        psi_rad=psi,
        x_g=x_g,
        y_g=y_g,
        x_normalized=float(x_norm),
        y_normalized=float(y_norm),
        preview_valid=1,
        s_proj=s_proj,
        s_goal=s_goal,
        projected_lane_index=projection.lane_index,
        goal_lane_index=path.lane_index_at_s(s_goal),
        heading_error_rad=angle_difference(goal_heading, psi),
        distance_to_goal_m=distance_to_goal,
        x_clipped=abs(x_g / normalization_distance) > 1.0,
        y_clipped=abs(y_g / normalization_distance) > 1.0,
        reason=None,
        projection=projection,
        forward_speed_mps=speed,
    )


# Descriptive alias used by adapters that call the common feature vector a
# "forward preview" rather than a generic preview.
compute_forward_preview = compute_preview


def pure_pursuit_from_rear_coordinates(
    x_rear: float,
    y_rear: float,
    *,
    wheelbase_m: float,
    max_steering_deg: float,
    steering_sign: float,
    q: Optional[Sequence[float]] = None,
    rear_position: Optional[Sequence[float]] = None,
    rear_wheelbase_m: Optional[float] = None,
) -> PurePursuitResult:
    """Compute PP from rear-axle vehicle coordinates.

    ``q`` and ``rear_position`` are diagnostic fields; the calculation itself
    uses the supplied rear-frame coordinates.  ``rear_wheelbase_m`` is
    retained to make the confirmed vehicle geometry visible in logs.
    """

    x = _finite_float(x_rear, "x_rear")
    y = _finite_float(y_rear, "y_rear")
    wheelbase = _finite_float(wheelbase_m, "wheelbase_m")
    max_angle = _finite_float(max_steering_deg, "max_steering_deg")
    sign = _finite_float(steering_sign, "steering_sign")
    if wheelbase <= 0.0 or max_angle <= 0.0:
        raise ValueError("wheelbase_m and max_steering_deg must be positive")
    if abs(abs(sign) - 1.0) > 1.0e-12:
        raise ValueError("steering_sign must be +1 or -1")
    rear_wheelbase = _finite_optional_float(rear_wheelbase_m, "rear_wheelbase_m")
    if rear_wheelbase is not None and rear_wheelbase < 0.0:
        raise ValueError("rear_wheelbase_m must be non-negative")
    q2 = None if q is None else _point2(q, "q")
    rear2 = None if rear_position is None else _point2(rear_position, "rear_position")
    denominator = x * x + y * y
    if denominator <= 1.0e-6:
        return PurePursuitResult(
            valid=False,
            pp_valid=False,
            q=q2,
            rear_position=rear2,
            x_rear=x,
            y_rear=y,
            wheelbase_m=wheelbase,
            rear_wheelbase_m=rear_wheelbase,
            kappa_pp=None,
            delta_pp_rad=None,
            delta_pp_deg=None,
            u_pp_unclipped=None,
            u_pp=None,
            steering_sign=sign,
            max_steering_deg=max_angle,
            saturated=False,
            reason="rear_goal_distance_too_small",
        )
    kappa = 2.0 * y / denominator
    delta = math.atan(wheelbase * kappa)
    delta_deg = math.degrees(delta)
    u_unclipped = sign * delta_deg / max_angle
    u = max(-1.0, min(1.0, u_unclipped))
    return PurePursuitResult(
        valid=True,
        pp_valid=True,
        q=q2,
        rear_position=rear2,
        x_rear=x,
        y_rear=y,
        wheelbase_m=wheelbase,
        rear_wheelbase_m=rear_wheelbase,
        kappa_pp=kappa,
        delta_pp_rad=delta,
        delta_pp_deg=delta_deg,
        u_pp_unclipped=u_unclipped,
        u_pp=u,
        steering_sign=sign,
        max_steering_deg=max_angle,
        saturated=abs(u_unclipped) > 1.0,
        reason=None,
    )


def compute_pure_pursuit(
    preview: PreviewResult,
    point: Sequence[float],
    psi_rad: float,
    *,
    wheelbase_m: float,
    max_steering_deg: float,
    steering_sign: float,
    rear_wheelbase_m: Optional[float] = None,
    rear_position: Optional[Sequence[float]] = None,
) -> PurePursuitResult:
    """Compute PP from a shared :class:`PreviewResult`.

    A rear axle location must be explicit: either ``rear_position`` measured by
    the adapter or a confirmed ``rear_wheelbase_m`` behind the observation
    point.  Omitting both is an unsupported vehicle-geometry configuration,
    rather than a per-step invalid reference.
    """

    if not isinstance(preview, PreviewResult):
        raise TypeError("preview must be a PreviewResult")
    # Vehicle geometry parameters are persistent configuration and must be
    # validated even when a particular state later turns out to be non-finite.
    rear_offset = _finite_optional_float(rear_wheelbase_m, "rear_wheelbase_m")
    if rear_position is None and rear_offset is None:
        raise ValueError(
            "rear axle geometry is required: provide rear_position or rear_wheelbase_m"
        )
    if rear_offset is not None and rear_offset < 0.0:
        raise ValueError("rear_wheelbase_m must be non-negative")
    wheelbase = _finite_float(wheelbase_m, "wheelbase_m")
    max_steering = _finite_float(max_steering_deg, "max_steering_deg")
    sign = _finite_float(steering_sign, "steering_sign")
    if wheelbase <= 0.0 or max_steering <= 0.0:
        raise ValueError("wheelbase_m and max_steering_deg must be positive")
    if abs(abs(sign) - 1.0) > 1.0e-12:
        raise ValueError("steering_sign must be +1 or -1")

    point2, nonfinite_point = _point2_state(point)
    if nonfinite_point:
        return PurePursuitResult(
            valid=False,
            pp_valid=False,
            q=preview.q,
            rear_position=None,
            x_rear=None,
            y_rear=None,
            wheelbase_m=wheelbase,
            rear_wheelbase_m=rear_offset,
            kappa_pp=None,
            delta_pp_rad=None,
            delta_pp_deg=None,
            u_pp_unclipped=None,
            u_pp=None,
            steering_sign=sign,
            max_steering_deg=max_steering,
            saturated=False,
            reason="nonfinite_vehicle_geometry",
        )
    assert point2 is not None
    psi, nonfinite_heading = _scalar_state(psi_rad, "psi_rad")
    if nonfinite_heading:
        return PurePursuitResult(
            valid=False,
            pp_valid=False,
            q=preview.q,
            rear_position=None,
            x_rear=None,
            y_rear=None,
            wheelbase_m=wheelbase,
            rear_wheelbase_m=rear_offset,
            kappa_pp=None,
            delta_pp_rad=None,
            delta_pp_deg=None,
            u_pp_unclipped=None,
            u_pp=None,
            steering_sign=sign,
            max_steering_deg=max_steering,
            saturated=False,
            reason="nonfinite_vehicle_geometry",
        )
    assert psi is not None

    rear2: Optional[Point2] = None
    if rear_position is not None:
        rear2, nonfinite_rear = _point2_state(rear_position, "rear_position")
        if nonfinite_rear:
            return PurePursuitResult(
                valid=False,
                pp_valid=False,
                q=preview.q,
                rear_position=None,
                x_rear=None,
                y_rear=None,
                wheelbase_m=wheelbase,
                rear_wheelbase_m=rear_offset,
                kappa_pp=None,
                delta_pp_rad=None,
                delta_pp_deg=None,
                u_pp_unclipped=None,
                u_pp=None,
                steering_sign=sign,
                max_steering_deg=max_steering,
                saturated=False,
                reason="nonfinite_vehicle_geometry",
            )
        assert rear2 is not None
    if not preview.valid or preview.q is None:
        return PurePursuitResult(
            valid=False,
            pp_valid=False,
            q=preview.q,
            rear_position=None,
            x_rear=None,
            y_rear=None,
            wheelbase_m=wheelbase,
            rear_wheelbase_m=rear_offset,
            kappa_pp=None,
            delta_pp_rad=None,
            delta_pp_deg=None,
            u_pp_unclipped=None,
            u_pp=None,
            steering_sign=sign,
            max_steering_deg=max_steering,
            saturated=False,
            reason="preview_invalid",
        )
    forward = (math.cos(psi), math.sin(psi))
    if rear2 is None:
        assert rear_offset is not None
        rear2 = (
            point2[0] - rear_offset * forward[0],
            point2[1] - rear_offset * forward[1],
        )
    else:
        if rear_offset is not None:
            expected = (
                point2[0] - rear_offset * forward[0],
                point2[1] - rear_offset * forward[1],
            )
            if _distance(rear2, expected) > DEFAULT_ENDPOINT_TOLERANCE_M:
                return PurePursuitResult(
                    valid=False,
                    pp_valid=False,
                    q=preview.q,
                    rear_position=rear2,
                    x_rear=None,
                    y_rear=None,
                    wheelbase_m=wheelbase,
                    rear_wheelbase_m=rear_offset,
                    kappa_pp=None,
                    delta_pp_rad=None,
                    delta_pp_deg=None,
                    u_pp_unclipped=None,
                    u_pp=None,
                    steering_sign=sign,
                    max_steering_deg=max_steering,
                    saturated=False,
                    reason="rear_axle_geometry_mismatch",
                )
    difference = (preview.q[0] - rear2[0], preview.q[1] - rear2[1])
    left = (-math.sin(psi), math.cos(psi))
    x_rear = difference[0] * forward[0] + difference[1] * forward[1]
    y_rear = difference[0] * left[0] + difference[1] * left[1]
    result = pure_pursuit_from_rear_coordinates(
        x_rear,
        y_rear,
        wheelbase_m=wheelbase,
        max_steering_deg=max_steering,
        steering_sign=sign,
        q=preview.q,
        rear_position=rear2,
        rear_wheelbase_m=rear_offset,
    )
    return result


# Short alias for adapter code.
compute_pp_reference = compute_pure_pursuit


def pp_penalty_result(
    u_applied: Optional[float],
    u_pp: Optional[float],
    *,
    dt_s: float,
    pp_weight: float,
    pp_valid: bool,
    terminated: bool = False,
    truncated: bool = False,
) -> PPPenaltyResult:
    """Return the pure PP reward term and its mask diagnostics.

    The terminal transition is always masked out, preserving the base
    environment's terminal reward.  If the mask is false, control values may
    be ``None`` because no PP reference exists for that transition.
    """

    dt = _finite_float(dt_s, "dt_s")
    weight = _finite_float(pp_weight, "pp_weight")
    if dt < 0.0:
        raise ValueError("dt_s must be non-negative")
    if weight < 0.0:
        raise ValueError("pp_weight must be non-negative")
    if not isinstance(pp_valid, bool):
        raise TypeError("pp_valid must be bool")
    if not isinstance(terminated, bool) or not isinstance(truncated, bool):
        raise TypeError("terminated and truncated must be bool")
    applied = None if u_applied is None else _finite_float(u_applied, "u_applied")
    reference = None if u_pp is None else _finite_float(u_pp, "u_pp")
    if applied is not None and not -1.0 <= applied <= 1.0:
        raise ValueError("u_applied must be in [-1, 1]")
    if reference is not None and not -1.0 <= reference <= 1.0:
        raise ValueError("u_pp must be in [-1, 1]")
    mask = pp_valid and not terminated and not truncated
    if not mask:
        return PPPenaltyResult(
            r_pp=0.0,
            e_pp=None,
            mask=False,
            u_applied=applied,
            u_pp=reference,
            dt_s=dt,
            pp_weight=weight,
            terminated=terminated,
            truncated=truncated,
        )
    if u_applied is None or u_pp is None:
        raise ValueError("u_applied and u_pp are required when pp_valid is true")
    assert applied is not None and reference is not None
    error = abs(applied - reference) / 2.0
    return PPPenaltyResult(
        r_pp=-weight * dt * error,
        e_pp=error,
        mask=True,
        u_applied=applied,
        u_pp=reference,
        dt_s=dt,
        pp_weight=weight,
        terminated=terminated,
        truncated=truncated,
    )


def compute_pp_penalty(
    u_applied: Optional[float],
    u_pp: Optional[float],
    *,
    dt_s: float,
    pp_weight: float,
    pp_valid: bool,
    terminated: bool = False,
    truncated: bool = False,
) -> float:
    """Return ``r_pp = -weight * dt * mask * abs(u-u_pp)/2`` as a scalar."""

    return pp_penalty_result(
        u_applied,
        u_pp,
        dt_s=dt_s,
        pp_weight=pp_weight,
        pp_valid=pp_valid,
        terminated=terminated,
        truncated=truncated,
    ).r_pp


def compute_pp_reward(*args: Any, **kwargs: Any) -> float:
    """Alias for :func:`compute_pp_penalty` used by reward wrappers."""

    return compute_pp_penalty(*args, **kwargs)


def pp_penalty(*args: Any, **kwargs: Any) -> float:
    """Alias for :func:`compute_pp_penalty`."""

    return compute_pp_penalty(*args, **kwargs)


def _to_dict(value: Any) -> Any:
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return value.as_dict()
    if isinstance(value, Mapping):
        return {str(key): _to_dict(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_dict(item) for item in value]
    if isinstance(value, list):
        return [_to_dict(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


__all__ = [
    "GeometryError",
    "LaneLike",
    "LaneMetadata",
    "RouteIssue",
    "RouteDiagnostics",
    "RouteConnection",
    "RouteBuildResult",
    "ReferencePath",
    "ProjectionCandidate",
    "ProjectionResult",
    "PreviewResult",
    "PurePursuitResult",
    "PPPenaltyResult",
    "DEFAULT_ENDPOINT_TOLERANCE_M",
    "DEFAULT_WIDTH_TOLERANCE_M",
    "DEFAULT_TANGENT_TOLERANCE_RAD",
    "DEFAULT_AMBIGUITY_TOLERANCE_M",
    "DEFAULT_LOOKAHEAD_M",
    "NORMALIZATION_DISTANCE_M",
    "wrap_to_pi",
    "angle_difference",
    "normalize_preview_value",
    "validate_lane",
    "connection_measurement",
    "build_reference_route",
    "build_reference_path",
    "connected_successor_candidates",
    "project_to_path",
    "compute_preview",
    "compute_forward_preview",
    "pure_pursuit_from_rear_coordinates",
    "compute_pure_pursuit",
    "compute_pp_reference",
    "pp_penalty_result",
    "compute_pp_penalty",
    "compute_pp_reward",
    "pp_penalty",
]
