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
        for key in (
            "start_lane_center_coef",
            "start_lane_wrong_coef",
            "start_lane_tolerance_ratio",
            "start_lane_terminal_penalty",
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
        ):
            if float(config[key]) < 0:
                raise ValueError(f"{key} must be non-negative")
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

    def _get_reset_return(self, reset_info: dict[str, Any]):
        """Capture reset ordinals before BaseEnv invokes reward/done functions."""

        if self._start_lane_objective != "off":
            self._reset_target_lane_tracking()
            for vehicle_id, vehicle in self.agents.items():
                self._target_lane_ordinals[vehicle_id] = _lane_ordinal(
                    getattr(vehicle, "lane_index", None)
                )
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

    def _target_lane_state(self, vehicle_id: str) -> TargetLaneState:
        return resolve_target_lane_state(
            self.agents[vehicle_id],
            target_ordinal=self._target_lane_ordinals.get(vehicle_id),
            tolerance_ratio=float(self.config["start_lane_tolerance_ratio"]),
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
        if self._is_post_step(vehicle_id):
            lane_cost = self._target_lane_cost(vehicle_id, state)
            self._target_lane_costs[vehicle_id] = lane_cost
            self._update_target_lane_tracking(vehicle_id, state)
            reward -= lane_cost
        else:
            # Reset calls reward_function to build info but does not create an
            # environment transition or an episode reward.  In particular, do
            # not reinterpret a reset-time arrival/failure probe as a custom
            # terminal transition or alter the upstream scalar reward.
            self._target_lane_costs[vehicle_id] = 0.0
            step_info.update(self._target_lane_info(vehicle_id, state))
            return reward, step_info

        vehicle = self.agents[vehicle_id]
        upstream_failure = self._has_upstream_terminal_failure(vehicle)
        strict_departure = self._strict_departure(vehicle_id)
        wrong_lane_arrival = self._wrong_lane_arrival(
            arrive_destination=self._is_arrive_destination(vehicle),
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
                reward = min(float(reward), 0.0)
        elif strict_departure:
            self._start_lane_departures[vehicle_id] = True
            # Strict departure outranks a simultaneous wrong-lane arrival.
            reward = -float(self.config["start_lane_terminal_penalty"])
        elif wrong_lane_arrival:
            self._wrong_lane_arrivals[vehicle_id] = True
            # MetaDrive replaces dense reward with +success_reward at arrival;
            # undo that escape route for an off-target finish.
            reward = -float(self.config["start_lane_terminal_penalty"])

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
