"""Constant-speed/curvature prediction in metres, radians and seconds.

Pure Python: no simulator, learner, host project or future transitions. The
observed speeds estimate only the past distance; no acceleration is used.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
import math

from .lateral_acceleration import finite_real

MIN_PREDICTION_SPEED_MPS = 0.1
MIN_HISTORY_DISTANCE_M = 1e-4


def _xy(value: object, name: str) -> tuple[float, float]:
    try:
        if isinstance(value, (str, bytes)) or len(value) != 2:
            raise ValueError(f"{name} requires exactly two metre coordinates")
        return (finite_real(value[0], name), finite_real(value[1], name))
    except (TypeError, IndexError, KeyError) as error:
        raise ValueError(f"{name} requires two metre coordinates") from error


@dataclass(frozen=True)
class MotionState:
    """Vehicle centre, heading, planar magnitude and signed forward speed."""

    position_xy: tuple[float, float]
    heading_rad: float
    speed_mps: float
    forward_speed_mps: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_xy", _xy(self.position_xy, "position_xy"))
        for name in ("heading_rad", "speed_mps", "forward_speed_mps"):
            object.__setattr__(self, name, finite_real(getattr(self, name), name))
        if self.speed_mps < 0:
            raise ValueError("speed_mps must be a planar magnitude >= 0")

    @classmethod
    def from_mapping(cls, state: Mapping[str, object]) -> MotionState:
        if state.get("speed_unit") != "m/s" or state.get("speed_meaning") != "planar magnitude":
            raise ValueError("motion state requires audited speed_unit=m/s and speed_meaning=planar magnitude")
        try:
            return cls(state["position_xy"], state["heading_theta"],
                       state["speed_m_s"], state["forward_speed_mps"])
        except KeyError as error:
            raise ValueError(f"motion state missing required field: {error.args[0]}") from error


def time_lookahead_distance(speed_mps: float, time_s: float) -> float:
    speed = finite_real(speed_mps, "planar speed [m/s]")
    time = finite_real(time_s, "lookahead_time_s")
    if speed < 0 or time <= 0:
        raise ValueError("speed must be non-negative and time must be positive")
    return finite_real(speed * time, "time lookahead distance [m]")


def sinc_cosc(alpha: float) -> tuple[float, float]:
    """sin(a)/a and (1-cos(a))/a, without cancellation or NumPy's pi."""
    a = finite_real(alpha, "alpha")
    if abs(a) < 1e-4:
        a2 = a * a
        return (1 - a2 / 6 + a2 * a2 / 120,
                a * (0.5 - a2 / 24 + a2 * a2 / 720))
    return math.sin(a) / a, 2 * math.sin(a / 2) ** 2 / a


def predict_position(post: MotionState, *, distance_m: float, curvature_inv_m: float) -> tuple[float, float]:
    distance = finite_real(distance_m, "prediction distance")
    curvature = finite_real(curvature_inv_m, "curvature")
    if distance < 0:
        raise ValueError("prediction distance must be non-negative")
    sinc, cosc = sinc_cosc(curvature * distance)
    x, y = distance * sinc, distance * cosc
    c, s = math.cos(post.heading_rad), math.sin(post.heading_rad)
    return _xy((post.position_xy[0] + c * x - s * y,
                post.position_xy[1] + s * x + c * y), "predicted position")


@dataclass(frozen=True)
class PredictionResult:
    valid: bool = False
    skip_reason: str | None = None
    dpsi_rad: float | None = None
    ds_hist_m: float | None = None
    kappa_hat_inv_m: float | None = None
    distance_m: float | None = None
    alpha_rad: float | None = None
    predicted_xy: tuple[float, float] | None = None
    error_m: float | None = None
    reward: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def prediction_penalty(
    pre: MotionState | None, post: MotionState, goal_xy: tuple[float, float] | None,
    *, preview_valid: bool, time_s: float, dt_seconds: float, weight: float,
    error_scale_m: float, terminated: bool = False, truncated: bool = False,
) -> PredictionResult:
    """Predict from post, using pre only for the observed interval curvature."""
    dt = finite_real(dt_seconds, "decision dt")
    coefficient = finite_real(weight, "prediction weight")
    scale = finite_real(error_scale_m, "prediction error scale")
    distance = time_lookahead_distance(post.speed_mps, time_s)
    if dt <= 0 or coefficient < 0 or scale <= 0:
        raise ValueError("dt/scale must be positive and weight non-negative")
    if any(type(flag) is not bool for flag in (preview_valid, terminated, truncated)):
        raise ValueError("preview_valid/terminated/truncated must be bool")
    goal = None if goal_xy is None else _xy(goal_xy, "prediction goal")
    if preview_valid and goal is None:
        raise ValueError("valid preview requires an unnormalized world goal")
    if terminated or truncated:
        return PredictionResult(skip_reason="episode_end", distance_m=distance)
    if pre is None:
        return PredictionResult(skip_reason="history_unavailable", distance_m=distance)
    if not preview_valid:
        return PredictionResult(skip_reason="preview_invalid", distance_m=distance)
    if pre.forward_speed_mps < 0 or post.forward_speed_mps < 0:
        return PredictionResult(skip_reason="reverse_motion", distance_m=distance)
    if post.speed_mps <= MIN_PREDICTION_SPEED_MPS:
        return PredictionResult(skip_reason="low_speed", distance_m=distance)
    dpsi = (post.heading_rad - pre.heading_rad + math.pi) % (2 * math.pi) - math.pi
    dpsi = finite_real(dpsi, "wrapped heading difference")
    ds = finite_real((0.5 * pre.speed_mps + 0.5 * post.speed_mps) * dt, "history distance")
    if ds <= MIN_HISTORY_DISTANCE_M:
        return PredictionResult(skip_reason="insufficient_history_distance", distance_m=distance,
                                dpsi_rad=dpsi, ds_hist_m=ds)
    curvature = finite_real(dpsi / ds, "estimated curvature")
    predicted = predict_position(post, distance_m=distance, curvature_inv_m=curvature)
    error = finite_real(math.dist(predicted, goal), "prediction error")
    reward = finite_real(-coefficient * dt * error / scale, "prediction reward")
    return PredictionResult(True, None, dpsi, ds, curvature, distance,
                            curvature * distance, predicted, error, reward)


@dataclass
class PredictionEpisodeMetrics:
    """Decision durations and returned-observation saturation, reset included."""
    evaluated_seconds: float = 0.0
    invalid_seconds: float = 0.0
    skipped_seconds: float = 0.0
    reason_seconds: dict[str, float] = field(default_factory=dict)
    observation_samples: int = 0
    valid_observation_samples: int = 0
    saturated_observation_samples: int = 0

    def observe_reward(self, diagnostic: Mapping[str, object], dt: float) -> None:
        if diagnostic["valid"]:
            self.evaluated_seconds += dt
        else:
            reason = str(diagnostic["skip_reason"])
            self.reason_seconds[reason] = self.reason_seconds.get(reason, 0.0) + dt
            if reason in ("disabled", "zero_weight", "episode_end"):
                self.skipped_seconds += dt
            else:
                self.invalid_seconds += dt

    def observe_preview(self, *, valid: bool, saturated: bool) -> None:
        self.observation_samples += 1
        if valid:
            self.valid_observation_samples += 1
            self.saturated_observation_samples += int(saturated)

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        eligible = self.evaluated_seconds + self.invalid_seconds
        result["valid_time_ratio"] = self.evaluated_seconds / eligible if eligible else None
        result["observation_saturation_rate"] = (
            self.saturated_observation_samples / self.valid_observation_samples
            if self.valid_observation_samples else None
        )
        return result
