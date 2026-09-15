"""Portable, simulator-free contracts for reference-route lateral demand.

This is a geometric demand evaluated at a post-action planar speed, not
measured lateral acceleration or a future-speed prediction.  No host imports.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Real


def finite_real(value: object, name: str) -> float:
    """Reject bool, strings, NaN/Inf and overflow at the adapter boundary."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number, not bool/string")
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def planar_speed_mps(value: object, *, unit: str, signed: bool = False) -> float:
    """Convert an explicitly audited planar speed contract to a magnitude.

    A signed scalar is supported only when the host guarantees that its
    absolute value equals planar speed; longitudinal velocity alone does not
    satisfy this contract when lateral motion is possible.
    """

    speed = finite_real(value, "planar speed")
    if unit not in ("m/s", "km/h"):
        raise ValueError("planar speed unit must be explicitly 'm/s' or 'km/h'")
    if type(signed) is not bool:
        raise ValueError("signed speed contract must be bool")
    if signed:
        speed = abs(speed)
    elif speed < 0.0:
        raise ValueError("planar speed magnitude must be non-negative")
    return speed / 3.6 if unit == "km/h" else speed


@dataclass(frozen=True, slots=True)
class LateralReference:
    """The interval and its maximum absolute curvature at one state."""

    valid: bool
    s_proj_m: float | None = None
    s_goal_m: float | None = None
    kappa_abs_max_inv_m: float | None = None
    invalid_reason: str | None = None

    def __post_init__(self) -> None:
        if type(self.valid) is not bool:
            raise ValueError("lateral reference valid must be bool")
        for name in ("s_proj_m", "s_goal_m", "kappa_abs_max_inv_m"):
            value = getattr(self, name)
            if value is not None:
                number = finite_real(value, name)
                if number < 0.0:
                    raise ValueError(f"{name} must be non-negative")
                object.__setattr__(self, name, number)
        if self.valid:
            if any(getattr(self, name) is None for name in (
                "s_proj_m", "s_goal_m", "kappa_abs_max_inv_m"
            )):
                raise ValueError("valid lateral reference requires interval and curvature")
            if self.s_goal_m < self.s_proj_m:
                raise ValueError("lateral interval must be ordered")
            if self.invalid_reason is not None:
                raise ValueError("valid lateral reference cannot have an invalid reason")
        elif not isinstance(self.invalid_reason, str) or not self.invalid_reason:
            raise ValueError("invalid lateral reference requires a reason")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LateralAccelPenalty:
    required_lateral_accel_mps2: float
    exceedance_ratio: float
    curve_speed_limit_mps: float | None
    curve_speed_unlimited: bool
    reward: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def lateral_accel_penalty(
    *,
    kappa_abs_max_inv_m: float,
    speed_mps: float,
    max_lateral_accel: float,
    weight: float,
    dt_seconds: float,
) -> LateralAccelPenalty:
    """Compute -weight * dt * max(0, v**2 * K / A_max - 1)**2."""

    curvature = finite_real(kappa_abs_max_inv_m, "route curvature")
    speed = planar_speed_mps(speed_mps, unit="m/s")
    limit = finite_real(max_lateral_accel, "max_lateral_accel")
    coefficient = finite_real(weight, "lateral_accel_weight")
    dt = finite_real(dt_seconds, "decision dt")
    if curvature < 0.0 or coefficient < 0.0 or limit <= 0.0 or dt <= 0.0:
        raise ValueError("curvature/weight must be >= 0; acceleration limit/dt must be > 0")
    if curvature == 0.0:
        return LateralAccelPenalty(0.0, 0.0, None, True, 0.0)
    required = finite_real(speed * speed * curvature, "required lateral acceleration")
    excess = finite_real(max(0.0, required / limit - 1.0), "lateral exceedance ratio")
    # Multiplication keeps overflow visible as a checked non-finite result.
    reward = finite_real(-coefficient * dt * excess * excess, "lateral reward")
    curve_speed = finite_real(math.sqrt(limit) / math.sqrt(curvature), "curve speed limit")
    return LateralAccelPenalty(required, excess, curve_speed, False, reward)


@dataclass(slots=True)
class LateralEpisodeMetrics:
    """Decision-time metrics; terminal/reset/disabled steps are excluded."""

    reference_seconds: float = 0.0
    reference_invalid_seconds: float = 0.0
    evaluated_seconds: float = 0.0
    exceeded_seconds: float = 0.0
    required_lateral_accel_max_mps2: float | None = None

    def observe(self, diagnostic: dict[str, object], dt: float) -> None:
        if diagnostic["skip_reason"] in ("disabled", "zero_weight", "reset", "episode_end"):
            return
        self.reference_seconds += dt
        if not diagnostic["reference_valid"]:
            self.reference_invalid_seconds += dt
        if diagnostic["active"]:
            self.evaluated_seconds += dt
            required = float(diagnostic["required_lateral_accel_mps2"])
            self.required_lateral_accel_max_mps2 = max(
                required, self.required_lateral_accel_max_mps2 or 0.0
            )
            if float(diagnostic["exceedance_ratio"]) > 0.0:
                self.exceeded_seconds += dt

    def as_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "exceedance_time_ratio": (
                self.exceeded_seconds / self.evaluated_seconds
                if self.evaluated_seconds else None
            ),
            "reference_invalid_time_ratio": (
                self.reference_invalid_seconds / self.reference_seconds
                if self.reference_seconds else None
            ),
        }
