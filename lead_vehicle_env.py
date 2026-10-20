"""Seeded, constant-speed lead vehicles without changing upstream MetaDrive."""

from collections.abc import Mapping
import math

import numpy as np

from metadrive.component.vehicle.vehicle_type import TrafficDefaultVehicle
from metadrive.manager.base_manager import BaseManager
from start_lane_env import StartLaneMetaDriveEnv


def sample_lead_vehicles(settings, seed, lane_count, ego_lane):
    """Sample only with a private RNG; distances are relative to the ego spawn."""
    keys = {"count", "distance_m", "gap_m", "speed_kmh", "lane"}
    if not isinstance(settings, Mapping) or set(settings) != keys:
        raise ValueError(f"lead_vehicles requires exactly {sorted(keys)}")
    ranges = {}
    for key in keys - {"lane"}:
        bounds = settings[key]
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"lead_vehicles.{key} must be [minimum, maximum]")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in bounds):
            raise ValueError(f"lead_vehicles.{key} must contain finite numbers")
        if bounds[0] < 0 or bounds[0] > bounds[1]:
            raise ValueError(f"invalid lead_vehicles.{key} range")
        if key == "count" and any(not isinstance(v, int) for v in bounds):
            raise ValueError("lead_vehicles.count must contain integers")
        ranges[key] = bounds
    if settings["lane"] not in {"ego", "random"}:
        raise ValueError("lead_vehicles.lane must be 'ego' or 'random'")
    if lane_count <= 0 or not 0 <= ego_lane < lane_count:
        raise ValueError("invalid lead lane geometry")
    rng = np.random.default_rng(seed)
    count = int(rng.integers(ranges["count"][0], ranges["count"][1] + 1))
    distance = float(rng.uniform(*ranges["distance_m"]))
    records = []
    for index in range(count):
        if index:
            distance += float(rng.uniform(*ranges["gap_m"]))
        records.append({
            "distance_m": distance,
            "speed_kmh": float(rng.uniform(*ranges["speed_kmh"])),
            "lane": ego_lane if settings["lane"] == "ego" else int(rng.integers(lane_count)),
            "random_seed": int(rng.integers(2**31)),
        })
    return records


class LeadVehicleManager(BaseManager):
    """Move physical vehicles along lane centerlines at every physics tick."""

    PRIORITY = 20

    def before_reset(self):
        super().before_reset()
        self.records = []
        self._motion = []
        self.ego = None
        self.reference_lanes = []

    @property
    def vehicles(self):
        return list(self.spawned_objects.values())

    def after_reset(self):
        agents = self.engine.agent_manager.active_agents
        if len(agents) != 1:
            raise ValueError("lead_vehicles requires exactly one ego agent")
        self.ego = next(iter(agents.values()))
        navigation = self.ego.navigation
        graph = navigation.map.road_network.graph
        roads = list(zip(navigation.checkpoints[:-1], navigation.checkpoints[1:]))
        self.reference_lanes = [graph[a][b][0] for a, b in roads]
        first_lanes = graph[roads[0][0]][roads[0][1]]
        samples = sample_lead_vehicles(
            dict(self.engine.global_config["lead_vehicles"]), self.engine.global_random_seed,
            len(first_lanes), self.ego.lane_index[2],
        )
        for index, sample in enumerate(samples):
            ordinal = sample["lane"]
            if any(ordinal >= len(graph[a][b]) for a, b in roads):
                raise ValueError("lead vehicle lane disappears along the route")
            lanes = [graph[a][b][ordinal] for a, b in roads]
            progress = lanes[0].local_coordinates(self.ego.position)[0] + sample["distance_m"]
            end = sum(lane.length for lane in lanes) - TrafficDefaultVehicle.DEFAULT_LENGTH / 2
            if progress > end:
                raise ValueError("route is too short for the requested lead vehicle distance")
            remaining = progress
            for lane in lanes:
                if remaining <= lane.length:
                    break
                remaining -= lane.length
            vehicle = self.spawn_object(
                TrafficDefaultVehicle, name=f"lead_vehicle_{index}",
                random_seed=sample["random_seed"], vehicle_config={
                    "spawn_lane_index": lane.index, "spawn_longitude": float(remaining),
                    "destination": navigation.checkpoints[-1], "enable_reverse": False,
                    "use_special_color": False,
                },
            )
            vehicle.set_static(True)
            motion = {"vehicle": vehicle, "lanes": lanes, "progress_m": progress,
                      "end_m": end, "speed_m_s": sample["speed_kmh"] / 3.6}
            self._motion.append(motion)
            self._place(motion)
            self.records.append({
                "vehicle_id": vehicle.id, "distance_m": sample["distance_m"],
                "speed_kmh": sample["speed_kmh"], "lane_index": lane.index,
                "spawn_longitude": float(remaining),
            })

    @staticmethod
    def _place(motion):
        remaining = motion["progress_m"]
        for lane in motion["lanes"]:
            if remaining <= lane.length:
                break
            remaining -= lane.length
        vehicle = motion["vehicle"]
        heading = lane.heading_theta_at(remaining)
        speed = motion["speed_m_s"] if motion["progress_m"] < motion["end_m"] else 0.0
        vehicle.set_position(lane.position(remaining, 0))
        vehicle.set_heading_theta(heading)
        vehicle.set_velocity([math.cos(heading) * speed, math.sin(heading) * speed])
        vehicle.set_angular_velocity(0)

    def before_step(self):
        for vehicle in self.vehicles:
            vehicle.before_step([0, 0])
        return {}

    def step(self):
        dt = self.engine.global_config["physics_world_step_size"]
        for motion in self._motion:
            motion["progress_m"] = min(motion["end_m"], motion["progress_m"] + motion["speed_m_s"] * dt)
            self._place(motion)

    def after_step(self):
        for motion in self._motion:
            self._place(motion)
            motion["vehicle"].after_step()
        return {}

    def route_progress(self, vehicle):
        offset = 0.0
        for lane in self.reference_lanes:
            if lane.index[:2] == vehicle.lane_index[:2]:
                return offset + lane.local_coordinates(vehicle.position)[0]
            offset += lane.length
        return None

    @property
    def overtaken_count(self):
        ego_progress = self.route_progress(self.ego)
        if ego_progress is None:
            return 0
        return int(sum(
            progress is not None and progress < ego_progress
            for progress in (self.route_progress(vehicle) for vehicle in self.vehicles)
        ))


class LeadVehicleMetaDriveEnv(StartLaneMetaDriveEnv):
    """Retain the optional reset-lane objective and add the lead manager."""

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({"lead_vehicles": {
            "count": [0, 0], "distance_m": [30, 60], "gap_m": [20, 40],
            "speed_kmh": [10, 20], "lane": "ego",
        }})
        return config

    def setup_engine(self):
        super().setup_engine()
        self.engine.register_manager("lead_vehicle_manager", LeadVehicleManager())
