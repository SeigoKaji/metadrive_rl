"""Regression tests for the project-local reset-lane objective environment."""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import pytest

from configs.experiment_config import load_experiment_config, select_experiment
from env_factory import make_env
from evaluate import _aggregate_target_lane_metrics, _final_target_lane_metrics
from evaluation_results import _termination_reason
from evaluation_visualization import (
    STEP_TELEMETRY_FIELDS,
    RuntimeRoadMetrics,
    decode_discrete_action,
    derive_timing,
    make_step_telemetry,
)
from metadrive.constants import TerminationState
from metadrive.envs import MetaDriveEnv
from project_paths import PROJECT_ROOT
from start_lane_env import (
    StartLaneMetaDriveEnv,
    TargetLaneState,
    huber_loss,
    resolve_target_lane_state,
    target_lane_cost,
)


_RETURN_SELECTION = load_experiment_config(
    PROJECT_ROOT / "configs/official_start_lane_return.toml"
)
_RETURN_ENV_CONFIG = _RETURN_SELECTION.profile.evaluation_env_config
_OFFICIAL_ENV_CONFIG = select_experiment(
    profile_name="official"
).profile.evaluation_env_config

_ACTION_CONFIG = {
    "discrete_action": True,
    "use_multi_discrete": False,
    "discrete_steering_dim": 3,
    "discrete_throttle_dim": 3,
    "vehicle_config": {"enable_reverse": False},
}


class _FakeLane:
    def __init__(
        self,
        index: tuple[str, str, int],
        *,
        longitude: float = 4.0,
        lateral: float = 0.0,
        width: float = 3.5,
        length: float = 10.0,
    ) -> None:
        self.index = index
        self.longitude = longitude
        self.lateral = lateral
        self.width = width
        self.length = length
        self.width_queries: list[float] = []

    def local_coordinates(self, _position: object) -> tuple[float, float]:
        return self.longitude, self.lateral

    def width_at(self, longitude: float) -> float:
        self.width_queries.append(longitude)
        return self.width


def _fake_vehicle(
    *,
    target_lane: _FakeLane,
    current_ordinal: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        navigation=SimpleNamespace(current_ref_lanes=[target_lane]),
        lane_index=("current_start", "current_end", current_ordinal),
        position=(0.0, 0.0),
        max_speed_km_h=72.0,
        crash_vehicle=False,
        crash_object=False,
        crash_building=False,
        crash_human=False,
        crash_sidewalk=False,
    )


def _bare_start_lane_env(
    vehicle: SimpleNamespace,
    *,
    objective: str = "return",
    episode_length: int = 0,
) -> StartLaneMetaDriveEnv:
    """Build an uninitialized harness for side-effect-free unit tests."""

    env = object.__new__(StartLaneMetaDriveEnv)
    env.config = {
        "start_lane_objective": objective,
        "start_lane_center_coef": 0.25,
        "start_lane_wrong_coef": 0.10,
        "start_lane_tolerance_ratio": 0.05,
        "start_lane_violation_hold_steps": 2,
        "start_lane_terminal_penalty": 50.0,
        "physics_world_step_size": 0.02,
        "decision_repeat": 5,
        "driving_reward": 1.0,
        "speed_reward": 0.1,
        "out_of_road_penalty": 5.0,
        "crash_vehicle_penalty": 5.0,
        "crash_object_penalty": 5.0,
        "crash_sidewalk_penalty": 0.0,
        # Keep the harness aligned with MetaDriveEnv's default termination
        # switches.  Individual tests override these to verify disabled paths.
        "out_of_road_done": True,
        "crash_vehicle_done": True,
        "crash_object_done": True,
        "crash_human_done": True,
    }
    env.agent_manager = SimpleNamespace(active_agents={"ego": vehicle})
    env.episode_lengths = defaultdict(int, {"ego": episode_length})
    env._target_lane_ordinals = {"ego": 0}
    env._ever_departed_target_lane = defaultdict(bool)
    env._lane_departure_counts = defaultdict(int)
    env._off_target_seconds = defaultdict(float)
    env._in_target_steps = defaultdict(int)
    env._was_departed = defaultdict(bool)
    env._violation_steps = defaultdict(int)
    env._wrong_lane_arrivals = defaultdict(bool)
    env._start_lane_departures = defaultdict(bool)
    env._target_lane_costs = defaultdict(float)
    return env


def test_return_config_uses_project_local_objective_without_broken_line_done() -> None:
    """The dedicated TOML selects return mode and preserves upstream rewards."""

    assert _RETURN_SELECTION.name == "official_start_lane_return"
    assert _RETURN_SELECTION.profile.default_model_name == "official_start_lane_return"
    for config in (
        _RETURN_SELECTION.profile.train_env_config,
        _RETURN_SELECTION.profile.evaluation_env_config,
    ):
        assert config["start_lane_objective"] == "return"
        assert config["use_lateral_reward"] is False
        assert config["on_broken_line_done"] is False
        assert config["out_of_road_done"] is True
        assert config["out_of_road_penalty"] == 200.0
        assert config["start_lane_terminal_penalty"] == 200.0
        # The bundle does not shrink the official positive rewards; MetaDrive
        # supplies its unmodified defaults at runtime.
        assert "driving_reward" not in config
        assert "speed_reward" not in config


def test_factory_keeps_official_raw_and_selects_start_lane_subclass() -> None:
    """Only a non-off objective selects the project-local subclass."""

    official_env = make_env(_OFFICIAL_ENV_CONFIG)
    custom_env = make_env(_RETURN_ENV_CONFIG)
    try:
        assert type(official_env) is MetaDriveEnv
        assert isinstance(custom_env, StartLaneMetaDriveEnv)
        assert custom_env.observation_space.shape == official_env.observation_space.shape
        assert custom_env.action_space == official_env.action_space
    finally:
        custom_env.close()
        official_env.close()


def test_factory_accepts_an_explicit_off_start_lane_objective() -> None:
    """Explicit project-local off mode remains schema-valid but behavior-neutral."""

    env = make_env({**_OFFICIAL_ENV_CONFIG, "start_lane_objective": "off"})
    try:
        assert isinstance(env, StartLaneMetaDriveEnv)
        assert env.config["start_lane_objective"] == "off"
    finally:
        env.close()


def test_start_lane_state_resolves_ordinal_across_segments_and_clips_longitude() -> None:
    """The ordinal persists across node changes and width lookup stays in bounds."""

    target_lane = _FakeLane(
        ("new_segment_start", "new_segment_end", 0),
        longitude=25.0,
        lateral=3.5,
        width=3.5,
        length=10.0,
    )
    vehicle = _fake_vehicle(target_lane=target_lane, current_ordinal=1)

    state = resolve_target_lane_state(
        vehicle,
        target_ordinal=0,
        tolerance_ratio=0.05,
    )

    assert target_lane.width_queries == [10.0]
    assert state.valid is True
    assert state.target_ordinal == 0
    assert state.current_ordinal == 1
    assert state.target_lane_offset_m == pytest.approx(3.5)
    assert state.normalized_error == pytest.approx(2.0)
    assert state.in_target_lane is False
    assert state.departed is True


def test_start_lane_state_surfaces_missing_target_without_neighbor_fallback() -> None:
    """A vanished ordinal remains explicit rather than becoming a nearby lane."""

    neighbour_lane = _FakeLane(("segment", "next", 1))
    vehicle = _fake_vehicle(target_lane=neighbour_lane, current_ordinal=1)

    state = resolve_target_lane_state(
        vehicle,
        target_ordinal=0,
        tolerance_ratio=0.05,
    )

    assert state.valid is False
    assert state.target_ordinal == 0
    assert state.current_ordinal == 1
    assert state.target_lane_offset_m is None
    assert state.in_target_lane is None
    assert neighbour_lane.width_queries == []


def test_target_lane_cost_is_huber_shaped_and_normalized_to_control_reward() -> None:
    assert huber_loss(0.5) == pytest.approx(0.125)
    assert huber_loss(2.0) == pytest.approx(1.5)
    cost = target_lane_cost(
        normalized_error=2.0,
        departed=True,
        action_duration_seconds=0.1,
        driving_reward=1.0,
        speed_reward=0.1,
        max_speed_m_s=20.0,
        center_coef=0.25,
        wrong_coef=0.10,
    )
    # (1.0 * 20.0 * 0.1 + 0.1) * (0.25 * 1.5 + 0.10)
    assert cost == pytest.approx(0.9975)


def test_reset_probe_does_not_count_as_departure_or_strict_hold_step() -> None:
    """BaseEnv's reset-time reward/done probes must not mutate episode counters."""

    target_lane = _FakeLane(("segment", "next", 0), lateral=3.5)
    env = _bare_start_lane_env(
        _fake_vehicle(target_lane=target_lane),
        objective="strict",
        episode_length=0,
    )
    departed = TargetLaneState(
        valid=True,
        target_ordinal=0,
        current_ordinal=1,
        target_lane_offset_m=3.5,
        lane_width_m=3.5,
        normalized_error=2.0,
        in_target_lane=False,
        departed=True,
    )
    in_target = TargetLaneState(
        valid=True,
        target_ordinal=0,
        current_ordinal=0,
        target_lane_offset_m=0.0,
        lane_width_m=3.5,
        normalized_error=0.0,
        in_target_lane=True,
        departed=False,
    )

    env._update_target_lane_tracking("ego", departed)
    assert env._lane_departure_counts["ego"] == 0
    assert env._off_target_seconds["ego"] == 0.0
    assert env._violation_steps["ego"] == 0
    assert env._strict_departure("ego") is False

    env.episode_lengths["ego"] = 1
    env._update_target_lane_tracking("ego", departed)
    assert env._lane_departure_counts["ego"] == 1
    assert env._off_target_seconds["ego"] == pytest.approx(0.1)
    assert env._violation_steps["ego"] == 1
    assert env._strict_departure("ego") is False

    env.episode_lengths["ego"] = 2
    env._update_target_lane_tracking("ego", departed)
    assert env._lane_departure_counts["ego"] == 1
    assert env._off_target_seconds["ego"] == pytest.approx(0.2)
    assert env._violation_steps["ego"] == 2
    assert env._strict_departure("ego") is True

    env.episode_lengths["ego"] = 3
    env._update_target_lane_tracking("ego", in_target)
    assert env._violation_steps["ego"] == 0
    assert env._in_target_steps["ego"] == 1
    assert env._target_lane_info("ego", in_target)[
        "time_in_target_lane_ratio"
    ] == pytest.approx(1 / 3)


def test_reset_pseudo_step_preserves_upstream_success_and_never_sets_custom_terminal_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reset-time reward/done probes expose telemetry without custom terminal effects."""

    # The target ordinal 0 cannot be resolved: the reset location is both
    # target-invalid and (for a real policy step) a wrong-lane arrival.
    neighbour_lane = _FakeLane(("segment", "next", 1), lateral=3.5)
    env = _bare_start_lane_env(
        _fake_vehicle(target_lane=neighbour_lane),
        objective="strict",
        episode_length=0,
    )
    # Prove the post-step guard even if a stale threshold were present.
    env._violation_steps["ego"] = 2
    monkeypatch.setattr(
        MetaDriveEnv,
        "reward_function",
        lambda _env, _vehicle_id: (10.0, {"route_completion": 1.0}),
    )
    monkeypatch.setattr(
        MetaDriveEnv,
        "done_function",
        lambda _env, _vehicle_id: (True, {TerminationState.SUCCESS: True}),
    )
    monkeypatch.setattr(
        StartLaneMetaDriveEnv,
        "_is_arrive_destination",
        staticmethod(lambda _vehicle: True),
    )

    reward, reward_info = env.reward_function("ego")
    done, done_info = env.done_function("ego")

    assert reward == 10.0
    assert reward_info["target_lane_valid"] is False
    assert reward_info["target_lane_cost"] == 0.0
    assert reward_info["wrong_lane_arrival"] is False
    assert reward_info["start_lane_departure"] is False
    assert done is True
    assert done_info[TerminationState.SUCCESS] is True
    assert done_info["target_lane_valid"] is False
    assert done_info["wrong_lane_arrival"] is False
    assert done_info["start_lane_departure"] is False


def test_wrong_lane_arrival_and_strict_departure_replace_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Custom terminal states remove MetaDrive's generic any-lane success."""

    target_lane = _FakeLane(("segment", "next", 0), lateral=3.5)

    def base_reward(_env: object, _vehicle_id: str) -> tuple[float, dict[str, object]]:
        return 10.0, {"route_completion": 1.0}

    def base_done(_env: object, _vehicle_id: str) -> tuple[bool, dict[str, object]]:
        return True, {TerminationState.SUCCESS: True}

    monkeypatch.setattr(MetaDriveEnv, "reward_function", base_reward)
    monkeypatch.setattr(MetaDriveEnv, "done_function", base_done)
    monkeypatch.setattr(
        StartLaneMetaDriveEnv,
        "_is_arrive_destination",
        staticmethod(lambda _vehicle: True),
    )
    monkeypatch.setattr(
        StartLaneMetaDriveEnv,
        "_is_out_of_road",
        lambda _env, _vehicle: False,
    )

    return_env = _bare_start_lane_env(
        _fake_vehicle(target_lane=target_lane),
        objective="return",
        episode_length=1,
    )
    reward, reward_info = return_env.reward_function("ego")
    done, done_info = return_env.done_function("ego")
    assert reward == -50.0
    assert reward_info["wrong_lane_arrival"] is True
    assert reward_info["start_lane_departure"] is False
    assert done is True
    assert done_info[TerminationState.SUCCESS] is False
    assert done_info["wrong_lane_arrival"] is True

    strict_env = _bare_start_lane_env(
        _fake_vehicle(target_lane=target_lane),
        objective="strict",
        episode_length=2,
    )
    strict_env._violation_steps["ego"] = 2
    strict_reward, strict_reward_info = strict_env.reward_function("ego")
    strict_done, strict_done_info = strict_env.done_function("ego")
    assert strict_reward == -50.0
    assert strict_reward_info["start_lane_departure"] is True
    assert strict_reward_info["wrong_lane_arrival"] is False
    assert strict_done is True
    assert strict_done_info[TerminationState.SUCCESS] is False
    assert strict_done_info["start_lane_departure"] is True


def test_collision_out_of_road_wins_over_custom_terminal_penalty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An upstream failure keeps its base penalty even beside a wrong arrival."""

    target_lane = _FakeLane(("segment", "next", 0), lateral=3.5)
    env = _bare_start_lane_env(
        _fake_vehicle(target_lane=target_lane),
        episode_length=1,
    )
    monkeypatch.setattr(
        MetaDriveEnv,
        "reward_function",
        lambda _env, _vehicle_id: (10.0, {"route_completion": 1.0}),
    )
    monkeypatch.setattr(
        MetaDriveEnv,
        "done_function",
        lambda _env, _vehicle_id: (
            True,
            {
                TerminationState.SUCCESS: True,
                TerminationState.OUT_OF_ROAD: True,
                TerminationState.CRASH: False,
            },
        ),
    )
    monkeypatch.setattr(
        StartLaneMetaDriveEnv,
        "_is_arrive_destination",
        staticmethod(lambda _vehicle: True),
    )
    monkeypatch.setattr(
        StartLaneMetaDriveEnv,
        "_is_out_of_road",
        lambda _env, _vehicle: True,
    )

    reward, reward_info = env.reward_function("ego")
    done, done_info = env.done_function("ego")
    assert reward == -5.0
    assert reward_info["wrong_lane_arrival"] is False
    assert reward_info["start_lane_departure"] is False
    assert done is True
    assert done_info[TerminationState.SUCCESS] is False
    assert done_info[TerminationState.OUT_OF_ROAD] is True


def test_building_crash_does_not_retain_a_simultaneous_success_reward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No arbitrary penalty is invented for building/human crash terminals."""

    target_lane = _FakeLane(("segment", "next", 0), lateral=3.5)
    vehicle = _fake_vehicle(target_lane=target_lane)
    vehicle.crash_building = True
    env = _bare_start_lane_env(vehicle, episode_length=1)
    monkeypatch.setattr(
        MetaDriveEnv,
        "reward_function",
        lambda _env, _vehicle_id: (10.0, {"route_completion": 1.0}),
    )
    monkeypatch.setattr(
        MetaDriveEnv,
        "done_function",
        lambda _env, _vehicle_id: (
            True,
            {
                TerminationState.SUCCESS: True,
                TerminationState.CRASH: True,
                TerminationState.CRASH_BUILDING: True,
            },
        ),
    )
    monkeypatch.setattr(
        StartLaneMetaDriveEnv,
        "_is_arrive_destination",
        staticmethod(lambda _vehicle: True),
    )
    monkeypatch.setattr(
        StartLaneMetaDriveEnv,
        "_is_out_of_road",
        lambda _env, _vehicle: False,
    )

    reward, reward_info = env.reward_function("ego")
    done, done_info = env.done_function("ego")
    assert reward == 0.0
    assert reward_info["wrong_lane_arrival"] is False
    assert reward_info["start_lane_departure"] is False
    assert done is True
    assert done_info[TerminationState.SUCCESS] is False
    assert done_info[TerminationState.CRASH_BUILDING] is True


@pytest.mark.parametrize(
    ("done_config_key", "crash_attribute", "failure_flag", "is_out_of_road"),
    [
        ("out_of_road_done", None, TerminationState.OUT_OF_ROAD, True),
        (
            "crash_vehicle_done",
            "crash_vehicle",
            TerminationState.CRASH_VEHICLE,
            False,
        ),
        (
            "crash_object_done",
            "crash_object",
            TerminationState.CRASH_OBJECT,
            False,
        ),
        (
            "crash_human_done",
            "crash_human",
            TerminationState.CRASH_HUMAN,
            False,
        ),
    ],
)
def test_disabled_upstream_terminal_conditions_preserve_stock_arrival_success(
    monkeypatch: pytest.MonkeyPatch,
    done_config_key: str,
    crash_attribute: str | None,
    failure_flag: str,
    is_out_of_road: bool,
) -> None:
    """Disabled stock done switches must not erase a simultaneous success."""

    target_lane = _FakeLane(("segment", "next", 0), lateral=0.0)
    vehicle = _fake_vehicle(target_lane=target_lane, current_ordinal=0)
    if crash_attribute is not None:
        setattr(vehicle, crash_attribute, True)
    env = _bare_start_lane_env(vehicle, episode_length=1)
    env.config[done_config_key] = False
    monkeypatch.setattr(
        MetaDriveEnv,
        "reward_function",
        lambda _env, _vehicle_id: (10.0, {"route_completion": 1.0}),
    )
    monkeypatch.setattr(
        MetaDriveEnv,
        "done_function",
        lambda _env, _vehicle_id: (
            True,
            {
                TerminationState.SUCCESS: True,
                failure_flag: True,
                # The compatibility aggregate must not by itself make the
                # custom objective treat a disabled crash as terminal.
                TerminationState.CRASH: crash_attribute is not None,
            },
        ),
    )
    monkeypatch.setattr(
        StartLaneMetaDriveEnv,
        "_is_arrive_destination",
        staticmethod(lambda _vehicle: True),
    )
    monkeypatch.setattr(
        StartLaneMetaDriveEnv,
        "_is_out_of_road",
        lambda _env, _vehicle: is_out_of_road,
    )

    reward, reward_info = env.reward_function("ego")
    done, done_info = env.done_function("ego")

    assert reward == 10.0
    assert reward_info["wrong_lane_arrival"] is False
    assert reward_info["start_lane_departure"] is False
    assert done is True
    assert done_info[TerminationState.SUCCESS] is True
    assert done_info[failure_flag] is True
    assert done_info["wrong_lane_arrival"] is False
    assert done_info["start_lane_departure"] is False


@pytest.mark.parametrize(
    ("crash_attribute", "done_config_key", "failure_flag", "expected_reward"),
    [
        (
            "crash_vehicle",
            "crash_vehicle_done",
            TerminationState.CRASH_VEHICLE,
            -5.0,
        ),
        (
            "crash_object",
            "crash_object_done",
            TerminationState.CRASH_OBJECT,
            -5.0,
        ),
        (
            "crash_human",
            "crash_human_done",
            TerminationState.CRASH_HUMAN,
            0.0,
        ),
    ],
)
def test_enabled_upstream_terminal_conditions_override_simultaneous_success(
    monkeypatch: pytest.MonkeyPatch,
    crash_attribute: str,
    done_config_key: str,
    failure_flag: str,
    expected_reward: float,
) -> None:
    """Enabled MetaDrive failure switches take precedence over target arrival."""

    target_lane = _FakeLane(("segment", "next", 0), lateral=0.0)
    vehicle = _fake_vehicle(target_lane=target_lane, current_ordinal=0)
    setattr(vehicle, crash_attribute, True)
    env = _bare_start_lane_env(vehicle, episode_length=1)
    assert env.config[done_config_key] is True
    monkeypatch.setattr(
        MetaDriveEnv,
        "reward_function",
        lambda _env, _vehicle_id: (10.0, {"route_completion": 1.0}),
    )
    monkeypatch.setattr(
        MetaDriveEnv,
        "done_function",
        lambda _env, _vehicle_id: (
            True,
            {
                TerminationState.SUCCESS: True,
                failure_flag: True,
                TerminationState.CRASH: True,
            },
        ),
    )
    monkeypatch.setattr(
        StartLaneMetaDriveEnv,
        "_is_arrive_destination",
        staticmethod(lambda _vehicle: True),
    )
    monkeypatch.setattr(
        StartLaneMetaDriveEnv,
        "_is_out_of_road",
        lambda _env, _vehicle: False,
    )

    reward, reward_info = env.reward_function("ego")
    done, done_info = env.done_function("ego")

    assert reward == expected_reward
    assert reward_info["wrong_lane_arrival"] is False
    assert reward_info["start_lane_departure"] is False
    assert done is True
    assert done_info[TerminationState.SUCCESS] is False
    assert done_info[failure_flag] is True


@pytest.mark.parametrize(
    ("out_of_road_done", "expected_reward", "expected_success"),
    [(False, 10.0, True), (True, -5.0, False)],
)
def test_sidewalk_crash_uses_only_the_out_of_road_done_path(
    monkeypatch: pytest.MonkeyPatch,
    out_of_road_done: bool,
    expected_reward: float,
    expected_success: bool,
) -> None:
    """A sidewalk crash is terminal only through `_is_out_of_road`'s toggle."""

    target_lane = _FakeLane(("segment", "next", 0), lateral=0.0)
    vehicle = _fake_vehicle(target_lane=target_lane, current_ordinal=0)
    vehicle.crash_sidewalk = True
    env = _bare_start_lane_env(vehicle, episode_length=1)
    env.config["out_of_road_done"] = out_of_road_done
    monkeypatch.setattr(
        MetaDriveEnv,
        "reward_function",
        lambda _env, _vehicle_id: (10.0, {"route_completion": 1.0}),
    )
    monkeypatch.setattr(
        MetaDriveEnv,
        "done_function",
        lambda _env, _vehicle_id: (
            True,
            {
                TerminationState.SUCCESS: True,
                TerminationState.OUT_OF_ROAD: True,
                TerminationState.CRASH_SIDEWALK: True,
                TerminationState.CRASH: True,
            },
        ),
    )
    monkeypatch.setattr(
        StartLaneMetaDriveEnv,
        "_is_arrive_destination",
        staticmethod(lambda _vehicle: True),
    )
    monkeypatch.setattr(
        StartLaneMetaDriveEnv,
        "_is_out_of_road",
        lambda _env, _vehicle: True,
    )

    reward, reward_info = env.reward_function("ego")
    done, done_info = env.done_function("ego")

    assert reward == expected_reward
    assert reward_info["wrong_lane_arrival"] is False
    assert done is True
    assert done_info[TerminationState.SUCCESS] is expected_success
    assert done_info[TerminationState.CRASH_SIDEWALK] is True


def test_sidewalk_crash_without_out_of_road_does_not_use_crash_aggregate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The compatibility crash aggregate is never an independent done cause."""

    target_lane = _FakeLane(("segment", "next", 0), lateral=0.0)
    vehicle = _fake_vehicle(target_lane=target_lane, current_ordinal=0)
    vehicle.crash_sidewalk = True
    env = _bare_start_lane_env(vehicle, episode_length=1)
    monkeypatch.setattr(
        MetaDriveEnv,
        "reward_function",
        lambda _env, _vehicle_id: (10.0, {"route_completion": 1.0}),
    )
    monkeypatch.setattr(
        MetaDriveEnv,
        "done_function",
        lambda _env, _vehicle_id: (
            True,
            {
                TerminationState.SUCCESS: True,
                TerminationState.CRASH_SIDEWALK: True,
                TerminationState.CRASH: True,
            },
        ),
    )
    monkeypatch.setattr(
        StartLaneMetaDriveEnv,
        "_is_arrive_destination",
        staticmethod(lambda _vehicle: True),
    )
    monkeypatch.setattr(
        StartLaneMetaDriveEnv,
        "_is_out_of_road",
        lambda _env, _vehicle: False,
    )

    reward, _reward_info = env.reward_function("ego")
    done, done_info = env.done_function("ego")

    assert reward == 10.0
    assert done is True
    assert done_info[TerminationState.SUCCESS] is True
    assert done_info[TerminationState.CRASH] is True


def test_final_target_lane_metrics_and_aggregate_skip_legacy_zeroes() -> None:
    """Only emitted and valid target-lane episodes participate in means/rates."""

    assert _final_target_lane_metrics({"velocity": 1.0}) is None

    valid = _final_target_lane_metrics(
        {
            "target_lane_valid": True,
            "ever_departed_target_lane": True,
            "lane_departure_count": 2,
            "off_target_duration_seconds": 0.4,
            "time_in_target_lane_ratio": 0.6,
        }
    )
    invalid = _final_target_lane_metrics(
        {
            "target_lane_valid": False,
            "ever_departed_target_lane": False,
            "lane_departure_count": 999,
            "off_target_duration_seconds": 999.0,
            "time_in_target_lane_ratio": 0.0,
        }
    )
    assert valid == {
        "target_lane_valid": True,
        "ever_departed_target_lane": True,
        "lane_departure_count": 2,
        "off_target_duration_seconds": 0.4,
        "time_in_target_lane_ratio": 0.6,
    }
    assert invalid is not None

    aggregate = _aggregate_target_lane_metrics(
        [
            {"target_lane": valid},
            {"target_lane": invalid},
            {},
        ]
    )
    assert aggregate["status"] == "available"
    assert aggregate["tracked_episode_count"] == 2
    assert aggregate["tracked_episode_rate"] == pytest.approx(2 / 3)
    assert aggregate["valid_episode_count"] == 1
    assert aggregate["valid_episode_rate_among_tracked"] == pytest.approx(0.5)
    assert aggregate["ever_departed_episode_count"] == 1
    assert aggregate["ever_departed_episode_rate_among_valid"] == pytest.approx(1.0)
    assert aggregate["mean_lane_departure_count_among_valid"] == pytest.approx(2.0)
    assert aggregate["mean_off_target_duration_seconds_among_valid"] == pytest.approx(0.4)
    assert aggregate["mean_time_in_target_lane_ratio_among_valid"] == pytest.approx(0.6)

    legacy_aggregate = _aggregate_target_lane_metrics([{}])
    assert legacy_aggregate["status"] == "not_available"
    assert legacy_aggregate["tracked_episode_count"] == 0
    assert legacy_aggregate["ever_departed_episode_count"] is None
    assert legacy_aggregate["mean_lane_departure_count_among_valid"] is None


def test_target_telemetry_schema_and_termination_priority() -> None:
    timing = derive_timing(
        {"physics_world_step_size": 0.02, "decision_repeat": 5}
    )
    telemetry = make_step_telemetry(
        episode=1,
        scenario_seed=5,
        step=2,
        horizon=500,
        timing=timing,
        decoded_action=decode_discrete_action(7, _ACTION_CONFIG),
        info={
            "velocity": 2.0,
            "route_completion": 0.1,
            "target_lane_valid": True,
            "target_lane_ordinal": 0,
            "current_lane_ordinal": 1,
            "target_lane_offset_m": 3.5,
            "normalized_target_lane_error": 2.0,
            "in_target_lane": False,
            "ever_departed_target_lane": True,
            "lane_departure_count": 1,
            "off_target_duration_seconds": 0.1,
            "time_in_target_lane_ratio": 0.5,
            "target_lane_cost": 0.9975,
            "wrong_lane_arrival": False,
            "start_lane_departure": True,
        },
        reward=-50.0,
        cumulative_reward=-49.0,
        terminated=True,
        truncated=False,
        road=RuntimeRoadMetrics(None, None, None, None, None, None, None),
        action_switch_count=0,
        action_switches_per_second=0.0,
    )

    assert tuple(telemetry) == STEP_TELEMETRY_FIELDS
    assert telemetry["target_lane_ordinal"] == 0
    assert telemetry["current_lane_ordinal"] == 1
    assert telemetry["target_lane_offset_m"] == pytest.approx(3.5)
    assert telemetry["target_lane_cost"] == pytest.approx(0.9975)
    assert telemetry["status"] == "START_LANE_DEPARTURE"
    assert _termination_reason(
        terminated=True,
        truncated=False,
        flags={"start_lane_departure": True, "arrive_dest": False},
    ) == "start_lane_departure"
    assert _termination_reason(
        terminated=True,
        truncated=False,
        flags={"wrong_lane_arrival": True, "arrive_dest": False},
    ) == "wrong_lane_arrival"
    assert _termination_reason(
        terminated=True,
        truncated=False,
        flags={
            "out_of_road": True,
            "start_lane_departure": True,
            "wrong_lane_arrival": True,
        },
    ) == "out_of_road"


def test_start_lane_return_environment_reset_and_one_step() -> None:
    """The custom environment preserves the stock observation/action contract."""

    env = make_env(_RETURN_ENV_CONFIG)
    try:
        observation, reset_info = env.reset(seed=5)
        assert isinstance(env, StartLaneMetaDriveEnv)
        assert env.action_space.n == 9
        # MetaDrive 0.4.3's official lidar observation is 259-wide here; the
        # project-local task must preserve that exact existing space.
        assert env.observation_space.shape == (259,)
        assert env.observation_space.contains(observation)
        assert reset_info["target_lane_valid"] is True
        assert isinstance(reset_info["target_lane_ordinal"], int)
        assert reset_info["lane_departure_count"] == 0
        assert reset_info["off_target_duration_seconds"] == 0.0
        assert reset_info["time_in_target_lane_ratio"] == 1.0
        assert reset_info["target_lane_cost"] == 0.0

        next_observation, reward, terminated, truncated, step_info = env.step(7)
        assert env.observation_space.contains(next_observation)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert step_info["target_lane_valid"] is True
        assert step_info["lane_departure_count"] == 0
        assert step_info["off_target_duration_seconds"] == 0.0
    finally:
        env.close()


def test_official_environment_remains_raw_and_can_reset_and_step() -> None:
    """The canonical profile keeps MetaDrive's original class and space contract."""

    env = make_env(_OFFICIAL_ENV_CONFIG)
    try:
        assert type(env) is MetaDriveEnv
        observation, info = env.reset(seed=5)
        assert env.observation_space.contains(observation)
        assert "target_lane_valid" not in info
        next_observation, _reward, terminated, truncated, _step_info = env.step(7)
        assert env.observation_space.contains(next_observation)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    finally:
        env.close()
