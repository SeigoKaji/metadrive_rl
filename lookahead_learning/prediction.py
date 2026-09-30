"""Post-origin constant-speed or constant-acceleration/curvature prediction.

Pure Python in metres, radians and seconds: no simulator, learner or host.
Curvature always uses the adjacent past interval; acceleration is opt-in.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
import math

from .lateral_acceleration import finite_real
from .checkpoint import PREDICTION_MOTION_MODELS

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


def estimate_acceleration(pre_speed_mps: float, post_speed_mps: float, dt_seconds: float) -> float:
    """Adjacent decision-step change of planar speed, in m/s² (not v²K)."""
    pre = finite_real(pre_speed_mps, "pre speed [m/s]")
    post = finite_real(post_speed_mps, "post speed [m/s]")
    dt = finite_real(dt_seconds, "decision dt")
    if pre < 0 or post < 0 or dt <= 0:
        raise ValueError("speeds must be non-negative and dt positive")
    return finite_real((post - pre) / dt, "estimated acceleration [m/s²]")


@dataclass(frozen=True)
class PredictionTravel:
    distance_m: float
    end_speed_mps: float
    moving_time_s: float
    stop_time_s: float | None
    stopped: bool


def constant_acceleration_travel(speed_mps: float, acceleration_mps2: float, time_s: float) -> PredictionTravel:
    """Advance at constant tangential acceleration; freeze after reaching zero.

    Average endpoint speeds avoid cancellation and squaring a large speed.
    Non-representable results (including stop time) raise, never clip to finite.
    """
    speed = finite_real(speed_mps, "post speed [m/s]")
    acceleration = finite_real(acceleration_mps2, "acceleration [m/s²]")
    time = finite_real(time_s, "lookahead_time_s")
    if speed < 0 or time <= 0:
        raise ValueError("speed must be non-negative and time positive")
    stop_time = None
    if acceleration < 0:
        stop_time = finite_real(speed / -acceleration, "predicted stop time [s]")
    stopped = stop_time is not None and time >= stop_time
    tau = stop_time if stopped else time
    # Do not evaluate a*T after stopping: it may overflow despite finite travel.
    end_speed = 0.0 if stopped else max(0.0, finite_real(
        speed + acceleration * time, "predicted end speed [m/s]"
    ))
    distance = (speed * time if acceleration == 0 else
                (0.5 * speed + 0.5 * end_speed) * tau)
    return PredictionTravel(finite_real(distance, "prediction distance [m]"),
                            end_speed, tau, stop_time, stopped)


@dataclass(frozen=True)
class PredictionReference:
    """Read-only post-origin goal on the fixed route, at the requested distance."""
    valid: bool
    goal_xy: tuple[float, float] | None = None
    s_proj_m: float | None = None
    s_goal_m: float | None = None
    invalid_reason: str | None = None

    def __post_init__(self) -> None:
        if type(self.valid) is not bool:
            raise ValueError("prediction reference valid must be bool")
        if self.goal_xy is not None:
            object.__setattr__(self, "goal_xy", _xy(self.goal_xy, "prediction reference goal"))
        for name in ("s_proj_m", "s_goal_m"):
            if getattr(self, name) is not None:
                value = finite_real(getattr(self, name), name)
                if value < 0:
                    raise ValueError(f"{name} must be non-negative")
                object.__setattr__(self, name, value)
        if self.s_proj_m is not None and self.s_goal_m is not None and self.s_goal_m < self.s_proj_m:
            raise ValueError("prediction reference goal must follow the projection")
        if self.valid:
            if self.goal_xy is None or self.s_proj_m is None or self.s_goal_m is None or self.invalid_reason is not None:
                raise ValueError("valid prediction reference requires goal/S_proj/S_goal and no invalid reason")
        elif not isinstance(self.invalid_reason, str) or not self.invalid_reason:
            raise ValueError("invalid prediction reference requires a reason")


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
    acceleration_mps2: float | None = None
    end_speed_mps: float | None = None
    moving_time_s: float | None = None
    stop_time_s: float | None = None
    stopped: bool | None = None
    goal_xy: tuple[float, float] | None = None
    s_proj_m: float | None = None
    s_goal_m: float | None = None
    reference_valid: bool | None = None
    reference_invalid_reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def prediction_penalty(
    pre: MotionState | None, post: MotionState, goal_xy: tuple[float, float] | None,
    *, preview_valid: bool, time_s: float, dt_seconds: float, weight: float,
    error_scale_m: float, terminated: bool = False, truncated: bool = False,
    motion_model: str = "constant_speed",
    reference_at_distance: Callable[[float], PredictionReference] | None = None,
) -> PredictionResult:
    """One post-origin penalty. Acceleration mode reads a goal at the same D.

    The optional callback is only required by constant_acceleration. It must
    read the saved fixed route and this post state without another host step
    or ordinary preview call. Existing constant-speed callers are unchanged.
    """
    dt = finite_real(dt_seconds, "decision dt")
    coefficient = finite_real(weight, "prediction weight")
    scale = finite_real(error_scale_m, "prediction error scale")
    time = finite_real(time_s, "lookahead_time_s")
    if dt <= 0 or coefficient < 0 or scale <= 0 or time <= 0:
        raise ValueError("dt/time/scale must be positive and weight non-negative")
    if not isinstance(motion_model, str) or motion_model not in PREDICTION_MOTION_MODELS:
        raise ValueError("unknown prediction motion model")
    accelerated = motion_model == "constant_acceleration"
    if accelerated and not callable(reference_at_distance):
        raise ValueError("constant_acceleration requires reference_at_distance")
    if any(type(flag) is not bool for flag in (preview_valid, terminated, truncated)):
        raise ValueError("preview_valid/terminated/truncated must be bool")
    goal = None if goal_xy is None else _xy(goal_xy, "prediction goal")
    if not accelerated and preview_valid and goal is None:
        raise ValueError("valid preview requires an unnormalized world goal")
    values = {"distance_m": None if accelerated else time_lookahead_distance(post.speed_mps, time)}
    if terminated or truncated:
        return PredictionResult(skip_reason="episode_end", **values)
    if pre is None:
        return PredictionResult(skip_reason="history_unavailable", **values)
    if not accelerated and not preview_valid:
        return PredictionResult(skip_reason="preview_invalid", **values)
    if pre.forward_speed_mps < 0 or post.forward_speed_mps < 0:
        return PredictionResult(skip_reason="reverse_motion", **values)
    if post.speed_mps <= MIN_PREDICTION_SPEED_MPS:
        return PredictionResult(skip_reason="low_speed", **values)
    dpsi = (post.heading_rad - pre.heading_rad + math.pi) % (2 * math.pi) - math.pi
    dpsi = finite_real(dpsi, "wrapped heading difference")
    ds = finite_real((0.5 * pre.speed_mps + 0.5 * post.speed_mps) * dt, "history distance")
    values.update(dpsi_rad=dpsi, ds_hist_m=ds)
    if ds <= MIN_HISTORY_DISTANCE_M:
        return PredictionResult(skip_reason="insufficient_history_distance", **values)
    curvature = finite_real(dpsi / ds, "estimated curvature")
    values["kappa_hat_inv_m"] = curvature
    if accelerated:
        acceleration = estimate_acceleration(pre.speed_mps, post.speed_mps, dt)
        travel = constant_acceleration_travel(post.speed_mps, acceleration, time)
        values.update(asdict(travel), acceleration_mps2=acceleration)
        reference = reference_at_distance(travel.distance_m)
        if not isinstance(reference, PredictionReference):
            raise ValueError("reference_at_distance must return PredictionReference")
        values.update(goal_xy=reference.goal_xy, s_proj_m=reference.s_proj_m,
                      s_goal_m=reference.s_goal_m, reference_valid=reference.valid,
                      reference_invalid_reason=reference.invalid_reason)
        if reference.s_proj_m is not None and reference.s_goal_m is not None:
            expected = finite_real(reference.s_proj_m + travel.distance_m, "reward S_goal")
            if reference.s_goal_m != expected:
                raise ValueError("prediction reference must use S_proj + the requested distance")
        if not reference.valid:
            return PredictionResult(skip_reason="reward_reference_invalid", **values)
        goal = reference.goal_xy
    else:
        values.update(end_speed_mps=post.speed_mps, moving_time_s=time, stopped=False)
    distance = values["distance_m"]
    predicted = predict_position(post, distance_m=distance, curvature_inv_m=curvature)
    error = finite_real(math.dist(predicted, goal), "prediction error")
    reward = finite_real(-coefficient * dt * error / scale, "prediction reward")
    return PredictionResult(valid=True, predicted_xy=predicted, error_m=error,
                            alpha_rad=curvature * distance, reward=reward, **values)


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
            if reason == "reward_reference_invalid":
                reason += ":" + str(diagnostic["reference_invalid_reason"])
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
