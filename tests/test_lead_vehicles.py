"""Seed isolation and physical contracts for the optional lead vehicles."""

from copy import deepcopy
import pickle
import random

import numpy as np
import pytest
import torch

from configs.experiment_config import load_experiment_config
from env_factory import make_env
from lead_vehicle_env import LeadVehicleMetaDriveEnv, sample_lead_vehicles
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.engine.engine_utils import engine_initialized
from project_paths import PROJECT_ROOT


BASE = {
    "map_config": {"type": "block_sequence", "config": "C", "lane_num": 2},
    "start_seed": 0, "num_scenarios": 1, "random_spawn_lane_index": False,
    "discrete_action": True, "traffic_density": 0, "accident_prob": 0, "log_level": 50,
}
LEAD = {
    "count": [1, 1], "distance_m": [60, 60], "gap_m": [20, 20],
    "speed_kmh": [12, 12], "lane": "ego",
}


def _global_rng_bytes():
    return (pickle.dumps((random.getstate(), np.random.get_state())),
            torch.get_rng_state().numpy().tobytes())


def _world_state(env):
    traffic = sorted((type(v).__name__, v.lane_index, tuple(v.position))
                     for v in env.engine.traffic_manager.traffic_vehicles)
    manager_rng = {name: pickle.dumps(manager.np_random.get_state())
                   for name, manager in env.engine.managers.items()
                   if name != "lead_vehicle_manager"}
    return (tuple(env.agent.navigation.checkpoints),
            env.current_map.road_network.get_bounding_box(), traffic,
            pickle.dumps(env.engine.np_random.get_state()), manager_rng)


def test_sampling_is_repeatable_inclusive_and_does_not_consume_global_rng():
    settings = {**LEAD, "count": [1, 2], "distance_m": [30, 60],
                "gap_m": [20, 40], "speed_kmh": [10, 20], "lane": "random"}
    original = deepcopy(settings)
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    try:
        before = _global_rng_bytes()
        samples = [sample_lead_vehicles(settings, seed, 2, 0) for seed in range(32)]
        assert samples == [sample_lead_vehicles(settings, seed, 2, 0) for seed in range(32)]
        assert _global_rng_bytes() == before
        assert {len(rows) for rows in samples} == {1, 2}
        assert {row["lane"] for rows in samples for row in rows} == {0, 1}
        for rows in samples:
            assert 30 <= rows[0]["distance_m"] <= 60
            assert all(10 <= row["speed_kmh"] <= 20 for row in rows)
            if len(rows) == 2:
                assert 20 <= rows[1]["distance_m"] - rows[0]["distance_m"] <= 40
        assert sample_lead_vehicles(LEAD, 0, 2, 1)[0]["lane"] == 1
        assert settings == original
    finally:
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])


@pytest.mark.parametrize("override", [
    {"count": [1.5, 2]}, {"distance_m": [2, 1]},
    {"speed_kmh": [10, float("inf")]}, {"lane": "opposite"},
])
def test_invalid_lead_ranges_fail_before_spawning(override):
    with pytest.raises(ValueError):
        sample_lead_vehicles({**LEAD, **override}, 0, 2, 0)


def test_spawning_preserves_existing_map_traffic_and_all_rng_streams():
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    snapshots = []
    try:
        for extra in ({}, {"lead_vehicles": LEAD}):
            random.seed(17)
            np.random.seed(17)
            torch.set_rng_state(state[2])
            env = make_env({**BASE, "traffic_density": 0.1,
                            "traffic_mode": "respawn", **extra})
            try:
                env.reset(seed=0)
                snapshots.append((_world_state(env), _global_rng_bytes()))
            finally:
                env.close()
        assert snapshots[0] == snapshots[1]
    finally:
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])


def test_route_spawn_curved_constant_speed_and_reset_cleanup():
    env = make_env({**BASE, "lead_vehicles": LEAD})
    try:
        env.reset(seed=0)
        manager = env.engine.lead_vehicle_manager
        records = deepcopy(manager.records)
        lead = manager.vehicles[0]
        assert lead.lane_index[:2] != env.agent.lane_index[:2]
        assert manager.route_progress(lead) - manager.route_progress(env.agent) == pytest.approx(60)
        assert lead.speed_km_h == pytest.approx(12, abs=0.1)
        progress = manager.route_progress(lead)
        heading = lead.heading_theta
        dt = env.config["physics_world_step_size"] * env.config["decision_repeat"]
        for _ in range(20):
            _, _, terminated, truncated, _ = env.step(4)
            assert not (terminated or truncated)
            assert lead.speed_km_h == pytest.approx(12, abs=0.1)
            assert lead.lane_index[2] == records[0]["lane_index"][2]
            assert abs(lead.lane.local_coordinates(lead.position)[1]) < 0.25
        assert manager.route_progress(lead) - progress == pytest.approx(20 * dt * 12 / 3.6, abs=0.1)
        assert abs(lead.heading_theta - heading) > 0.01
        env.reset(seed=0)
        assert manager.records == records
        assert len(manager.vehicles) == 1
        assert len(env.engine.get_objects(filter=lambda obj: isinstance(obj, BaseVehicle))) == 2
    finally:
        env.close()
    assert not engine_initialized()


def test_overtaken_count_uses_current_centers_and_leads_are_collidable():
    env = make_env({**BASE, "lead_vehicles": {**LEAD, "distance_m": [30, 30]}})
    try:
        env.reset(seed=0)
        manager = env.engine.lead_vehicle_manager
        lead = manager.vehicles[0]
        assert manager.overtaken_count == 0
        longitude = lead.lane.local_coordinates(lead.position)[0]
        env.agent.set_position(lead.lane.position(longitude + 6, 0))
        assert manager.overtaken_count == 1
        env.agent.set_position(lead.position)
        env.agent.set_heading_theta(lead.heading_theta)
        _, _, terminated, _, info = env.step(4)
        assert terminated and info["crash_vehicle"]
        assert manager.overtaken_count == 0
    finally:
        env.close()


def test_short_route_error_still_closes_partially_spawned_leads():
    env = make_env({**BASE, "lead_vehicles": {
        **LEAD, "count": [2, 2], "distance_m": [30, 30], "gap_m": [10_000, 10_000],
    }})
    try:
        with pytest.raises(ValueError, match="route is too short"):
            env.reset(seed=0)
    finally:
        env.close()
    assert not engine_initialized()


def test_config_loader_start_lane_and_lookahead_share_one_real_step(tmp_path):
    source = (PROJECT_ROOT / "configs/official_start_lane_return_lookahead.toml").read_text()
    inline = ('lead_vehicles = { count = [1, 1], distance_m = [60, 60], '
              'gap_m = [20, 20], speed_kmh = [12, 12], lane = "ego" }')
    path = tmp_path / "lead_lookahead.toml"
    path.write_text(source.replace("[environment.common]", "[environment.common]\n" + inline))
    profile = load_experiment_config(path).profile
    assert profile.train_env_config["lead_vehicles"] == profile.evaluation_env_config["lead_vehicles"] == LEAD
    env = make_env(profile.evaluation_env_config, lookahead_config=profile.lookahead_config)
    try:
        observation, info = env.reset(seed=5)
        raw = env.unwrapped
        assert isinstance(raw, LeadVehicleMetaDriveEnv)
        assert raw.config["start_lane_objective"] == "return"
        assert info["target_lane_valid"]
        assert env.observation_space.contains(observation)
        assert len(observation) == raw.observation_space.shape[0] + 3
        manager = raw.engine.lead_vehicle_manager
        progress = manager.route_progress(manager.vehicles[0])
        step = raw.engine.episode_step
        observation, _, terminated, truncated, info = env.step(4)
        assert not (terminated or truncated)
        assert env.observation_space.contains(observation) and info["target_lane_valid"]
        assert raw.engine.episode_step == step + 1
        dt = raw.config["physics_world_step_size"] * raw.config["decision_repeat"]
        assert manager.route_progress(manager.vehicles[0]) - progress == pytest.approx(dt * 12 / 3.6, abs=0.05)
    finally:
        env.close()
    assert not engine_initialized()
