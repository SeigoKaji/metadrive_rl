"""Project-local MetaDrive environment for preserving the reset lane.

This module intentionally subclasses :class:`metadrive.envs.MetaDriveEnv`
instead of modifying the vendored MetaDrive source tree.  The built-in
``use_lateral_reward`` follows whichever lane MetaDrive currently localizes;
this environment remembers the reset lane ordinal and evaluates lateral error
against the lane with that ordinal on every current road segment.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Any, Literal

from metadrive.constants import TerminationState
from metadrive.envs import MetaDriveEnv
from metadrive.utils import Config


StartLaneObjective = Literal["off", "return", "strict"]


START_LANE_DEFAULT_CONFIG: dict[str, object] = {
    # ``off`` retains the upstream environment behaviour.  The project factory
    # selects this subclass whenever the objective key is explicitly present.
    "start_lane_objective": "off",
    # Both coefficients multiply the maximum ordinary positive reward that can
    # be earned in one control interval.  They therefore remain comparable
    # when lane width, decision repeat, or the base reward scale changes.
    "start_lane_center_coef": 0.25,
    "start_lane_wrong_coef": 0.10,
    # A small geometry margin avoids treating a one-frame localization jitter
    # at the painted boundary as a lane departure.
    "start_lane_tolerance_ratio": 0.05,
    "start_lane_violation_hold_steps": 2,
    "start_lane_terminal_penalty": 50.0,
    # A duration-normalized low-speed cost can be enabled for the idle-policy
    # ablation without changing the existing start-lane objective by default.
    "low_speed_threshold_km_h": 10.0,
    "low_speed_penalty_rate": 0.0,
    # Applied only to a pure horizon truncation, never to another terminal.
    "timeout_penalty": 0.0,
    # Duckietown-style fallback: replace the ordinary dense scalar with only
    # forward distance travelled while remaining in the reset target lane.
    "target_lane_progress_only": False,
}


@dataclass(frozen=True, slots=True)
class TargetLaneState:
    """Target-lane geometry at one simulation instant.

    ``valid=False`` means that the reset lane ordinal cannot be resolved in
    ``navigation.current_ref_lanes``.  Callers must preserve that uncertainty
    rather than silently selecting a neighbouring lane.
    """

    valid: bool
    target_ordinal: int | None
    current_ordinal: int | None
    target_lane_offset_m: float | None
    lane_width_m: float | None
    normalized_error: float | None
    in_target_lane: bool | None
    departed: bool


def _lane_ordinal(value: Any) -> int | None:
    """Return a lane's ordinal component without assuming a full lane tuple."""

    try:
        # A tuple/list is already a lane index.  Checking ``.index`` first
        # would accidentally select tuple.index (the builtin method) instead.
        lane_index = value if isinstance(value, (tuple, list)) else value.index
        ordinal = lane_index[-1]
    except (AttributeError, IndexError, KeyError, TypeError):
        return None
    if isinstance(ordinal, bool):
        return None
    try:
        return int(ordinal)
    except (TypeError, ValueError):
        return None


def resolve_target_lane_state(
    vehicle: Any,
    *,
    target_ordinal: int | None,
    tolerance_ratio: float,
) -> TargetLaneState:
    """Resolve reset-lane geometry from the vehicle's *current* road segment.

    PG-map lane-index node names change as a vehicle traverses road segments,
    so only the ordinal is retained across the episode.  No closest-lane or
    ordinal fallback is used when the ordinal is absent from the current
    reference lanes.
    """

    current_ordinal = _lane_ordinal(getattr(vehicle, "lane_index", None))
    if target_ordinal is None:
        return TargetLaneState(
            valid=False,
            target_ordinal=None,
            current_ordinal=current_ordinal,
            target_lane_offset_m=None,
            lane_width_m=None,
            normalized_error=None,
            in_target_lane=None,
            departed=False,
        )

    try:
        reference_lanes = vehicle.navigation.current_ref_lanes
    except (AttributeError, TypeError):
        reference_lanes = None
    target_lane = next(
        (
            lane
            for lane in (reference_lanes or ())
            if _lane_ordinal(lane) == target_ordinal
        ),
        None,
    )
    if target_lane is None:
        return TargetLaneState(
            valid=False,
            target_ordinal=target_ordinal,
            current_ordinal=current_ordinal,
            target_lane_offset_m=None,
            lane_width_m=None,
            normalized_error=None,
            in_target_lane=None,
            departed=False,
        )

    try:
        longitude, lateral = target_lane.local_coordinates(vehicle.position)
        longitude = float(longitude)
        lane_length = float(target_lane.length)
        lateral = float(lateral)
        if (
            not math.isfinite(longitude)
            or not math.isfinite(lane_length)
            or lane_length <= 0
            or not math.isfinite(lateral)
        ):
            raise ValueError("invalid target-lane local coordinates")
        safe_longitude = min(max(longitude, 0.0), lane_length)
        lane_width = float(target_lane.width_at(safe_longitude))
    except (AttributeError, TypeError, ValueError):
        return TargetLaneState(
            valid=False,
            target_ordinal=target_ordinal,
            current_ordinal=current_ordinal,
            target_lane_offset_m=None,
            lane_width_m=None,
            normalized_error=None,
            in_target_lane=None,
            departed=False,
        )
    if (
        not math.isfinite(lane_width)
        or lane_width <= 0
    ):
        return TargetLaneState(
            valid=False,
            target_ordinal=target_ordinal,
            current_ordinal=current_ordinal,
            target_lane_offset_m=None,
            lane_width_m=None,
            normalized_error=None,
            in_target_lane=None,
            departed=False,
        )

    normalized_error = abs(lateral) / (lane_width / 2.0)
    departed = normalized_error > 1.0 + tolerance_ratio
    return TargetLaneState(
        valid=True,
        target_ordinal=target_ordinal,
        current_ordinal=current_ordinal,
        target_lane_offset_m=lateral,
        lane_width_m=lane_width,
        normalized_error=normalized_error,
        in_target_lane=not departed,
        departed=departed,
    )


def huber_loss(error: float) -> float:
    """Return unit-threshold Huber loss for a non-negative normalized error."""

    if error < 0 or not math.isfinite(error):
        raise ValueError(f"error must be finite and non-negative: {error!r}")
    return 0.5 * error * error if error <= 1.0 else error - 0.5


def target_lane_cost(
    *,
    normalized_error: float,
    departed: bool,
    action_duration_seconds: float,
    driving_reward: float,
    speed_reward: float,
    max_speed_m_s: float,
    center_coef: float,
    wrong_coef: float,
) -> float:
    """Return a shaped penalty normalized to the base one-action reward scale.

    ``normalized_error`` already divides by half the target-lane width.  The
    remaining scale uses the active control interval and the maximum positive
    upstream driving/speed rewards, so a TOML experiment need not retune a
    metre-valued coefficient when those runtime values change.
    """

    values = (
        action_duration_seconds,
        driving_reward,
        speed_reward,
        max_speed_m_s,
        center_coef,
        wrong_coef,
    )
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("target-lane cost inputs must be finite")
    if action_duration_seconds <= 0 or max_speed_m_s < 0:
        raise ValueError("invalid action duration or maximum speed")
    if center_coef < 0 or wrong_coef < 0:
        raise ValueError("target-lane coefficients must be non-negative")

    # MetaDrive's speed reward is specified once per control decision while
    # driving progress is distance-per-decision.  Include both positive terms
    # without altering their original rewards.
    reward_scale = (
        max(0.0, driving_reward) * max_speed_m_s * action_duration_seconds
        + max(0.0, speed_reward)
    )
    return reward_scale * (
        center_coef * huber_loss(normalized_error)
        + wrong_coef * float(departed)
    )


def low_speed_penalty(
    *,
    speed_km_h: float,
    threshold_km_h: float,
    penalty_rate: float,
    action_duration_seconds: float,
) -> float:
    """Return the duration-normalized penalty for speed below ``threshold``.

    Negative measured speeds are treated as stationary.  The rate is expressed
    in reward per second, so the same experiment has the same real-time cost
    when MetaDrive's physics rate or decision repeat changes.
    """

    values = (
        speed_km_h,
        threshold_km_h,
        penalty_rate,
        action_duration_seconds,
    )
    if any(isinstance(value, bool) for value in values):
        raise TypeError("low-speed penalty inputs must be numeric, not bool")
    try:
        numeric_values = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise TypeError("low-speed penalty inputs must be numeric") from error
    if any(not math.isfinite(value) for value in numeric_values):
        raise ValueError("low-speed penalty inputs must be finite")

    speed, threshold, rate, action_duration = numeric_values
    if threshold <= 0:
        raise ValueError("low_speed_threshold_km_h must be positive")
    if action_duration <= 0:
        raise ValueError("action_duration_seconds must be positive")
    if rate < 0:
        raise ValueError("low_speed_penalty_rate must be non-negative")
    return rate * action_duration * max(
        0.0,
        1.0 - max(speed, 0.0) / threshold,
    )


def _finite_float(value: Any, *, name: str) -> float:
    """Convert one runtime scalar while rejecting booleans and non-finite values."""

    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric, not bool")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be numeric") from error
    if not math.isfinite(numeric_value):
        raise ValueError(f"{name} must be finite")
    return numeric_value


def target_lane_forward_distance(
    *,
    previous_travelled_length_m: float | None,
    travelled_length_m: float,
    previous_in_target_lane: bool,
    state: TargetLaneState,
) -> float:
    """Return one Duckietown-style, target-lane-gated forward distance.

    ``navigation.travelled_length`` is route-cumulative, unlike a lane's local
    longitudinal coordinate, so it stays monotonic at PG-map segment
    boundaries.  The caller still updates its previous values on every actual
    decision: a target-lane re-entry must not retroactively earn distance that
    was driven outside the target lane.
    """

    current_length = _finite_float(
        travelled_length_m,
        name="navigation.travelled_length",
    )
    if previous_travelled_length_m is None:
        return 0.0
    previous_length = _finite_float(
        previous_travelled_length_m,
        name="previous navigation.travelled_length",
    )
    if (
        not state.valid
        or state.in_target_lane is not True
        or previous_in_target_lane is not True
    ):
        return 0.0

    distance = max(0.0, current_length - previous_length)
    if not math.isfinite(distance):
        raise ValueError("target-lane forward distance must be finite")
    return distance


class StartLaneMetaDriveEnv(MetaDriveEnv):
    """MetaDrive environment with return/strict reset-lane objectives.

    ``return`` adds a continuous target-lane penalty but permits a crossing so
    a policy can recover.  ``strict`` additionally terminates after a target
    geometry departure has persisted for the configured number of policy
    decisions.  Both modes turn an arrival in a non-target lane into a failed
    terminal transition.
    """

    @classmethod
    def default_config(cls) -> Config:
        config = super().default_config()
        config.update(START_LANE_DEFAULT_CONFIG)
        return config

    def __init__(self, config: dict[str, object] | None = None):
        # These are initialized before BaseEnv can call any overridable method.
        self._target_lane_ordinals: dict[str, int | None] = {}
        self._ever_departed_target_lane: dict[str, bool] = defaultdict(bool)
        self._lane_departure_counts: dict[str, int] = defaultdict(int)
        self._off_target_seconds: dict[str, float] = defaultdict(float)
        self._in_target_steps: dict[str, int] = defaultdict(int)
        self._was_departed: dict[str, bool] = defaultdict(bool)
        self._violation_steps: dict[str, int] = defaultdict(int)
        self._wrong_lane_arrivals: dict[str, bool] = defaultdict(bool)
        self._start_lane_departures: dict[str, bool] = defaultdict(bool)
        self._target_lane_costs: dict[str, float] = defaultdict(float)
        self._low_speed_penalties: dict[str, float] = defaultdict(float)
        self._timeout_penalties: dict[str, float] = defaultdict(float)
        self._target_lane_previous_travelled_lengths: dict[str, float] = {}
        self._target_lane_previous_in_target_lane: dict[str, bool] = {}
        self._target_lane_forward_distances: dict[str, float] = defaultdict(float)
        self._target_lane_progress_rewards: dict[str, float] = defaultdict(float)
        super().__init__(config)

    def _post_process_config(self, config: Config) -> Config:
        config = super()._post_process_config(config)
        self._validate_start_lane_config(config)
        return config

    @staticmethod
    def _validate_start_lane_config(config: Config) -> None:
        objective = config["start_lane_objective"]
        if objective not in {"off", "return", "strict"}:
            raise ValueError(
                "start_lane_objective must be one of 'off', 'return', or 'strict': "
                f"{objective!r}"
            )
        if not isinstance(config["target_lane_progress_only"], bool):
            raise TypeError("target_lane_progress_only must be a bool")
        for key in (
            "start_lane_center_coef",
            "start_lane_wrong_coef",
            "start_lane_tolerance_ratio",
            "start_lane_terminal_penalty",
            "low_speed_threshold_km_h",
            "low_speed_penalty_rate",
            "timeout_penalty",
        ):
            value = config[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{key} must be a finite numeric value")
            if not math.isfinite(float(value)):
                raise ValueError(f"{key} must be finite")
        for key in (
            "start_lane_center_coef",
            "start_lane_wrong_coef",
            "start_lane_tolerance_ratio",
            "low_speed_penalty_rate",
            "timeout_penalty",
        ):
            if float(config[key]) < 0:
                raise ValueError(f"{key} must be non-negative")
        if float(config["low_speed_threshold_km_h"]) <= 0:
            raise ValueError("low_speed_threshold_km_h must be positive")
        if float(config["start_lane_terminal_penalty"]) <= 0:
            raise ValueError("start_lane_terminal_penalty must be positive")
        hold_steps = config["start_lane_violation_hold_steps"]
        if isinstance(hold_steps, bool) or not isinstance(hold_steps, int):
            raise TypeError("start_lane_violation_hold_steps must be an integer")
        if hold_steps <= 0:
            raise ValueError("start_lane_violation_hold_steps must be positive")

    @property
    def _start_lane_objective(self) -> StartLaneObjective:
        # Validation in _post_process_config establishes this narrowed type.
        return self.config["start_lane_objective"]  # type: ignore[return-value]

    @property
    def _target_lane_progress_only(self) -> bool:
        # Validation in _post_process_config rejects truthy non-bool values.
        return self.config["target_lane_progress_only"]  # type: ignore[return-value]

    def _get_reset_return(self, reset_info: dict[str, Any]):
        """Capture reset ordinals before BaseEnv invokes reward/done functions."""

        if self._start_lane_objective != "off":
            self._reset_target_lane_tracking()
            for vehicle_id, vehicle in self.agents.items():
                self._target_lane_ordinals[vehicle_id] = _lane_ordinal(
                    getattr(vehicle, "lane_index", None)
                )
                self._assert_target_lane_progress_supported(vehicle_id)
        return super()._get_reset_return(reset_info)

    def _reset_target_lane_tracking(self) -> None:
        self._target_lane_ordinals.clear()
        self._ever_departed_target_lane.clear()
        self._lane_departure_counts.clear()
        self._off_target_seconds.clear()
        self._in_target_steps.clear()
        self._was_departed.clear()
        self._violation_steps.clear()
        self._wrong_lane_arrivals.clear()
        self._start_lane_departures.clear()
        self._target_lane_costs.clear()
        self._low_speed_penalties.clear()
        self._timeout_penalties.clear()
        self._target_lane_previous_travelled_lengths.clear()
        self._target_lane_previous_in_target_lane.clear()
        self._target_lane_forward_distances.clear()
        self._target_lane_progress_rewards.clear()

    def _target_lane_state(self, vehicle_id: str) -> TargetLaneState:
        return resolve_target_lane_state(
            self.agents[vehicle_id],
            target_ordinal=self._target_lane_ordinals.get(vehicle_id),
            tolerance_ratio=float(self.config["start_lane_tolerance_ratio"]),
        )

    def _assert_target_lane_progress_supported(self, vehicle_id: str) -> None:
        """Fail closed outside the official bundle's reference-lane-0 scope."""

        if not self._target_lane_progress_only:
            return
        target_ordinal = self._target_lane_ordinals.get(vehicle_id)
        if target_ordinal != 0:
            raise NotImplementedError(
                "target_lane_progress_only currently supports only reset target "
                "lane ordinal 0, because navigation.travelled_length follows "
                "the route reference lane 0; got "
                f"{target_ordinal!r}"
            )

    def _is_post_step(self, vehicle_id: str) -> bool:
        """Exclude BaseEnv's reset-time reward/done probes from accounting."""

        return int(self.episode_lengths[vehicle_id]) > 0

    def _action_duration_seconds(self) -> float:
        return float(self.config["physics_world_step_size"]) * int(
            self.config["decision_repeat"]
        )

    def _target_lane_cost(self, vehicle_id: str, state: TargetLaneState) -> float:
        if not state.valid or state.normalized_error is None:
            return 0.0
        vehicle = self.agents[vehicle_id]
        return target_lane_cost(
            normalized_error=state.normalized_error,
            departed=state.departed,
            action_duration_seconds=self._action_duration_seconds(),
            driving_reward=float(self.config["driving_reward"]),
            speed_reward=float(self.config["speed_reward"]),
            max_speed_m_s=float(vehicle.max_speed_km_h) / 3.6,
            center_coef=float(self.config["start_lane_center_coef"]),
            wrong_coef=float(self.config["start_lane_wrong_coef"]),
        )

    def _low_speed_penalty(self, vehicle_id: str) -> float:
        """Return the configured post-step low-speed penalty, if enabled."""

        penalty_rate = float(self.config["low_speed_penalty_rate"])
        if penalty_rate == 0.0:
            return 0.0
        vehicle = self.agents[vehicle_id]
        return low_speed_penalty(
            speed_km_h=float(vehicle.speed_km_h),
            threshold_km_h=float(self.config["low_speed_threshold_km_h"]),
            penalty_rate=penalty_rate,
            action_duration_seconds=self._action_duration_seconds(),
        )

    def _navigation_travelled_length(self, vehicle_id: str) -> float:
        """Read the route-cumulative progress used by the narrow fallback."""

        try:
            value = self.agents[vehicle_id].navigation.travelled_length
        except (AttributeError, KeyError, TypeError) as error:
            raise ValueError(
                "target_lane_progress_only requires "
                "vehicle.navigation.travelled_length"
            ) from error
        return _finite_float(value, name="navigation.travelled_length")

    def _clear_target_lane_progress_telemetry(self, vehicle_id: str) -> None:
        """Record that no target-lane progress scalar survived this transition."""

        self._target_lane_forward_distances[vehicle_id] = 0.0
        self._target_lane_progress_rewards[vehicle_id] = 0.0

    def _initialize_target_lane_progress_tracking(
        self,
        vehicle_id: str,
        state: TargetLaneState,
    ) -> None:
        """Save the reset probe as the first non-rewarded progress endpoint."""

        self._assert_target_lane_progress_supported(vehicle_id)
        self._target_lane_previous_travelled_lengths[vehicle_id] = (
            self._navigation_travelled_length(vehicle_id)
        )
        self._target_lane_previous_in_target_lane[vehicle_id] = bool(
            state.valid and state.in_target_lane is True
        )
        self._clear_target_lane_progress_telemetry(vehicle_id)

    def _update_target_lane_progress(
        self,
        vehicle_id: str,
        state: TargetLaneState,
    ) -> float:
        """Update one target-lane progress interval and return its reward.

        This is deliberately limited to target ordinal 0.  For the official
        map-C bundle that ordinal matches MetaDrive's route reference lane,
        whose ``travelled_length`` remains cumulative over segment boundaries.
        Other target ordinals can have different lane lengths and must not use
        this value as a silent proxy.
        """

        self._assert_target_lane_progress_supported(vehicle_id)
        travelled_length = self._navigation_travelled_length(vehicle_id)
        forward_distance = target_lane_forward_distance(
            previous_travelled_length_m=(
                self._target_lane_previous_travelled_lengths.get(vehicle_id)
            ),
            travelled_length_m=travelled_length,
            previous_in_target_lane=(
                self._target_lane_previous_in_target_lane.get(vehicle_id, False)
            ),
            state=state,
        )
        # Update even for invalid/off-target/reverse transitions.  Otherwise a
        # later re-entry could accidentally claim distance driven outside the
        # reset lane.
        self._target_lane_previous_travelled_lengths[vehicle_id] = travelled_length
        self._target_lane_previous_in_target_lane[vehicle_id] = bool(
            state.valid and state.in_target_lane is True
        )
        progress_reward = _finite_float(
            self.config["driving_reward"],
            name="driving_reward",
        ) * forward_distance
        if not math.isfinite(progress_reward):
            raise ValueError("target-lane progress reward must be finite")
        self._target_lane_forward_distances[vehicle_id] = forward_distance
        self._target_lane_progress_rewards[vehicle_id] = progress_reward
        return progress_reward

    def _update_target_lane_tracking(
        self,
        vehicle_id: str,
        state: TargetLaneState,
    ) -> None:
        """Update episode counters only for actual policy decisions."""

        if not self._is_post_step(vehicle_id):
            return
        if not state.valid:
            # Missing geometry is surfaced through telemetry, not converted into
            # a guessed departure or a strict-mode false positive.
            self._was_departed[vehicle_id] = False
            self._violation_steps[vehicle_id] = 0
            return
        if state.departed:
            if not self._was_departed[vehicle_id]:
                self._lane_departure_counts[vehicle_id] += 1
            self._ever_departed_target_lane[vehicle_id] = True
            self._off_target_seconds[vehicle_id] += self._action_duration_seconds()
            self._violation_steps[vehicle_id] += 1
        else:
            self._in_target_steps[vehicle_id] += 1
            self._violation_steps[vehicle_id] = 0
        self._was_departed[vehicle_id] = state.departed

    def _strict_departure(self, vehicle_id: str) -> bool:
        return (
            self._start_lane_objective == "strict"
            and self._violation_steps[vehicle_id]
            >= int(self.config["start_lane_violation_hold_steps"])
        )

    def _is_pure_max_step(
        self,
        vehicle_id: str,
        *,
        arrive_destination: bool,
        upstream_failure: bool,
        strict_departure: bool,
        wrong_lane_arrival: bool,
    ) -> bool:
        """Whether this transition is a horizon truncation with no terminal peer.

        ``reward_function`` runs before MetaDrive gathers done info, so mirror
        the stock max-step predicate here.  A timeout cost is intentionally not
        charged when arrival, upstream failure, or a project-local terminal
        condition shares the same final transition.
        """

        horizon = self.config["horizon"]
        return bool(
            self._is_post_step(vehicle_id)
            and horizon is not None
            and self.episode_lengths[vehicle_id] >= int(horizon)
            and not arrive_destination
            and not upstream_failure
            and not strict_departure
            and not wrong_lane_arrival
        )

    @staticmethod
    def _wrong_lane_arrival(
        *,
        arrive_destination: bool,
        state: TargetLaneState,
    ) -> bool:
        """Arrival is successful only when target-lane geometry is known/in-bounds."""

        return arrive_destination and (not state.valid or state.in_target_lane is not True)

    def _target_lane_info(
        self,
        vehicle_id: str,
        state: TargetLaneState,
    ) -> dict[str, object]:
        policy_steps = int(self.episode_lengths[vehicle_id])
        time_in_target_lane_ratio = (
            1.0
            if policy_steps == 0
            else self._in_target_steps[vehicle_id] / policy_steps
        )
        return {
            "target_lane_valid": state.valid,
            "target_lane_ordinal": state.target_ordinal,
            "current_lane_ordinal": state.current_ordinal,
            "target_lane_offset_m": state.target_lane_offset_m,
            "normalized_target_lane_error": state.normalized_error,
            "in_target_lane": state.in_target_lane,
            "ever_departed_target_lane": self._ever_departed_target_lane[vehicle_id],
            "lane_departure_count": self._lane_departure_counts[vehicle_id],
            "off_target_duration_seconds": self._off_target_seconds[vehicle_id],
            "time_in_target_lane_ratio": time_in_target_lane_ratio,
            "wrong_lane_arrival": self._wrong_lane_arrivals[vehicle_id],
            "start_lane_departure": self._start_lane_departures[vehicle_id],
            "target_lane_cost": self._target_lane_costs[vehicle_id],
            "low_speed_penalty": self._low_speed_penalties[vehicle_id],
            "timeout_penalty": self._timeout_penalties[vehicle_id],
            "target_lane_forward_distance_m": self._target_lane_forward_distances[
                vehicle_id
            ],
            "target_lane_progress_reward": self._target_lane_progress_rewards[
                vehicle_id
            ],
        }

    def _has_upstream_terminal_failure(self, vehicle: Any) -> bool:
        """Match MetaDrive's configurable upstream terminal conditions exactly."""

        return bool(
            (
                self._is_out_of_road(vehicle)
                and self.config["out_of_road_done"]
            )
            or (
                getattr(vehicle, "crash_vehicle", False)
                and self.config["crash_vehicle_done"]
            )
            or (
                getattr(vehicle, "crash_object", False)
                and self.config["crash_object_done"]
            )
            or getattr(vehicle, "crash_building", False)
            or (
                getattr(vehicle, "crash_human", False)
                and self.config["crash_human_done"]
            )
        )

    def _upstream_terminal_reward(self, vehicle: Any) -> float | None:
        """Mirror upstream failure rewards while giving failure precedence over success.

        MetaDrive's stock reward checks success first.  A custom objective must
        not turn a simultaneous collision/out-of-road terminal state into a
        target-lane penalty or a success reward, so replay the documented
        upstream failure priorities here.  Building/human crashes have no
        standalone stock reward value; callers clamp a simultaneous upstream
        success reward to zero rather than inventing a different penalty.
        """

        if self._is_out_of_road(vehicle) and self.config["out_of_road_done"]:
            return -float(self.config["out_of_road_penalty"])
        if (
            getattr(vehicle, "crash_vehicle", False)
            and self.config["crash_vehicle_done"]
        ):
            return -float(self.config["crash_vehicle_penalty"])
        if (
            getattr(vehicle, "crash_object", False)
            and self.config["crash_object_done"]
        ):
            return -float(self.config["crash_object_penalty"])
        return None

    def reward_function(self, vehicle_id: str):
        reward, step_info = super().reward_function(vehicle_id)
        if self._start_lane_objective == "off":
            return reward, step_info

        state = self._target_lane_state(vehicle_id)
        progress_only = self._target_lane_progress_only
        upstream_step_reward = float(reward) if progress_only else 0.0
        if self._is_post_step(vehicle_id):
            self._timeout_penalties[vehicle_id] = 0.0
            self._update_target_lane_tracking(vehicle_id, state)
            if progress_only:
                # Duckietown's lane-distance alternative does not retain
                # MetaDrive's current-lane progress/speed terms or this
                # environment's lane/low-speed shaping on ordinary steps.
                self._target_lane_costs[vehicle_id] = 0.0
                self._low_speed_penalties[vehicle_id] = 0.0
                reward = self._update_target_lane_progress(vehicle_id, state)
                low_speed_cost = 0.0
            else:
                lane_cost = self._target_lane_cost(vehicle_id, state)
                low_speed_cost = self._low_speed_penalty(vehicle_id)
                self._target_lane_costs[vehicle_id] = lane_cost
                self._low_speed_penalties[vehicle_id] = low_speed_cost
                self._clear_target_lane_progress_telemetry(vehicle_id)
                reward -= lane_cost
                reward -= low_speed_cost
        else:
            # Reset calls reward_function to build info but does not create an
            # environment transition or an episode reward.  In particular, do
            # not reinterpret a reset-time arrival/failure probe as a custom
            # terminal transition or alter the upstream scalar reward.
            self._target_lane_costs[vehicle_id] = 0.0
            self._low_speed_penalties[vehicle_id] = 0.0
            self._timeout_penalties[vehicle_id] = 0.0
            if progress_only:
                self._initialize_target_lane_progress_tracking(vehicle_id, state)
            else:
                self._clear_target_lane_progress_telemetry(vehicle_id)
            step_info.update(self._target_lane_info(vehicle_id, state))
            return reward, step_info

        vehicle = self.agents[vehicle_id]
        upstream_failure = self._has_upstream_terminal_failure(vehicle)
        strict_departure = self._strict_departure(vehicle_id)
        arrive_destination = self._is_arrive_destination(vehicle)
        wrong_lane_arrival = self._wrong_lane_arrival(
            arrive_destination=arrive_destination,
            state=state,
        )
        if upstream_failure:
            # Collision/out-of-road owns the terminal scalar.  In particular,
            # do not replace its base penalty with the custom terminal penalty.
            upstream_reward = self._upstream_terminal_reward(vehicle)
            if upstream_reward is not None:
                reward = upstream_reward
            else:
                # MetaDrive has no dedicated building/human crash reward.  Do
                # not map it to an arbitrary penalty, but never retain a
                # simultaneous +success_reward for an upstream failure.
                reward = min(
                    upstream_step_reward if progress_only else reward + low_speed_cost,
                    0.0,
                )
            self._low_speed_penalties[vehicle_id] = 0.0
            self._clear_target_lane_progress_telemetry(vehicle_id)
        elif strict_departure:
            self._start_lane_departures[vehicle_id] = True
            # Strict departure outranks a simultaneous wrong-lane arrival.
            reward = -float(self.config["start_lane_terminal_penalty"])
            self._low_speed_penalties[vehicle_id] = 0.0
            self._clear_target_lane_progress_telemetry(vehicle_id)
        elif wrong_lane_arrival:
            self._wrong_lane_arrivals[vehicle_id] = True
            # MetaDrive replaces dense reward with +success_reward at arrival;
            # undo that escape route for an off-target finish.
            reward = -float(self.config["start_lane_terminal_penalty"])
            self._low_speed_penalties[vehicle_id] = 0.0
            self._clear_target_lane_progress_telemetry(vehicle_id)
        elif arrive_destination:
            # A valid target-lane arrival retains the existing target-lane
            # shaping, but the upstream success scalar takes priority over the
            # independent low-speed shaping term.
            if progress_only:
                reward = upstream_step_reward
                self._clear_target_lane_progress_telemetry(vehicle_id)
            else:
                reward += low_speed_cost
            self._low_speed_penalties[vehicle_id] = 0.0
        elif self._is_pure_max_step(
            vehicle_id,
            arrive_destination=arrive_destination,
            upstream_failure=upstream_failure,
            strict_departure=strict_departure,
            wrong_lane_arrival=wrong_lane_arrival,
        ):
            timeout_cost = float(self.config["timeout_penalty"])
            reward -= timeout_cost
            self._timeout_penalties[vehicle_id] = timeout_cost

        step_info.update(self._target_lane_info(vehicle_id, state))
        return reward, step_info

    def done_function(self, vehicle_id: str):
        done, done_info = super().done_function(vehicle_id)
        if self._start_lane_objective == "off":
            return done, done_info

        state = self._target_lane_state(vehicle_id)
        if not self._is_post_step(vehicle_id):
            # BaseEnv probes done_function during reset.  Surface target
            # telemetry, but preserve every upstream reset flag verbatim.
            done_info.update(self._target_lane_info(vehicle_id, state))
            return done, done_info

        upstream_failure = self._has_upstream_terminal_failure(
            self.agents[vehicle_id]
        )
        wrong_lane_arrival = self._wrong_lane_arrival(
            arrive_destination=bool(done_info.get(TerminationState.SUCCESS, False)),
            state=state,
        )
        strict_departure = self._strict_departure(vehicle_id)
        if upstream_failure:
            # Keep failure flags authoritative if MetaDrive simultaneously
            # reports arrival.  This gives reward, done info, and evaluator
            # classification the same collision/out-of-road priority.
            done = True
            done_info[TerminationState.SUCCESS] = False
        elif strict_departure:
            self._start_lane_departures[vehicle_id] = True
            done = True
            done_info[TerminationState.SUCCESS] = False
        elif wrong_lane_arrival:
            self._wrong_lane_arrivals[vehicle_id] = True
            # Retain the terminal transition but do not report it as a task
            # success merely because MetaDrive accepts any final-road lane.
            done = True
            done_info[TerminationState.SUCCESS] = False

        done_info.update(self._target_lane_info(vehicle_id, state))
        return done, done_info
