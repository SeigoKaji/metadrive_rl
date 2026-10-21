"""選択されたタスク設定からMetaDrive環境を生成するfactory。"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import random
from typing import TYPE_CHECKING

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

from configs.experiment_config import SCENE_GROUP_SWITCH_KEYS, validate_scenario_seeds

if TYPE_CHECKING:
    from metadrive.envs import MetaDriveEnv


class _ScenarioSeedSampler(gym.Wrapper):
    """Sample explicit scenarios with a worker-local RNG and count actual resets."""

    def __init__(self, env: gym.Env, scenario_seeds: list[int], worker_seed: int) -> None:
        super().__init__(env)
        self.scenario_seeds = tuple(scenario_seeds)
        self.scenario_reset_counts = dict.fromkeys(self.scenario_seeds, 0)
        self._scenario_rng = random.Random(worker_seed)

    def reset(self, *, seed: int | None = None, **kwargs):
        if seed is None:
            seed = self._scenario_rng.choice(self.scenario_seeds)
        if isinstance(seed, bool) or seed not in self.scenario_reset_counts:
            raise ValueError(f"scenario seed must be one of {self.scenario_seeds}: {seed}")
        result = self.env.reset(seed=seed, **kwargs)
        actual_seed = int(self.unwrapped.current_seed)
        if actual_seed != seed:
            raise RuntimeError(f"requested scenario seed {seed}, got {actual_seed}")
        self.scenario_reset_counts[actual_seed] += 1
        return result


class _SceneGroupEnv(gym.Wrapper):
    """Switch a single simulator between groups; keep each group's maps separate."""

    def __init__(self, env: gym.Env, groups: list[dict]) -> None:
        super().__init__(env)
        from metadrive.component.map.pg_map import parse_map_config
        from metadrive.utils.config import Config

        self.scene_groups = deepcopy(groups)
        self._groups = {group["name"]: group for group in self.scene_groups}
        self._scene_pairs = [(group["name"], seed) for group in groups for seed in group["seeds"]]
        self._scene_rng = random.Random(0)
        self.scene_group_reset_counts = {
            group["name"]: dict.fromkeys(group["seeds"], 0) for group in groups
        }
        self.current_scene_group = groups[0]["name"]
        raw = self.unwrapped
        self._agent_configs = deepcopy(raw.config["agent_configs"])
        self._group_configs, self._group_maps = {}, {}
        defaults = raw.default_config_copy
        for group in groups:
            values = Config({key: defaults[key] for key in SCENE_GROUP_SWITCH_KEYS if key in defaults})
            values.update(group["env_config"])
            values["map_config"] = parse_map_config(values["map"], values["map_config"], defaults)
            self._group_configs[group["name"]] = values
            self._group_maps[group["name"]] = dict.fromkeys(group["seeds"])

    def reset(self, *, seed=None, options=None, **kwargs):
        options = dict(options or {})
        name = options.pop("scene_group", None)
        if options:
            raise ValueError(f"unsupported reset options: {sorted(options)}")
        if name is None:
            if seed is None:
                name, seed = self._scene_rng.choice(self._scene_pairs)
            else:
                name = self.current_scene_group
        if name not in self._groups:
            raise ValueError(f"unknown scene group: {name}")
        seeds = self._groups[name]["seeds"]
        seed = seeds[0] if seed is None else seed
        if isinstance(seed, bool) or seed not in seeds:
            raise ValueError(f"scenario seed must be one of {seeds}: {seed}")
        raw = self.unwrapped
        for key, value in self._group_configs[name].items():
            raw.config[key] = deepcopy(value)
        # The upstream random-spawn manager mutates this nested configuration.
        raw.config["agent_configs"] = deepcopy(self._agent_configs)
        raw.lazy_init()
        engine = raw.engine
        needs_objects = any(abs(values["accident_prob"]) > 1e-2 for values in self._group_configs.values())
        if needs_objects and not hasattr(engine, "object_manager"):
            from metadrive.manager.object_manager import TrafficObjectManager

            engine.register_manager("object_manager", TrafficObjectManager())
        engine.traffic_manager.mode = raw.config["traffic_mode"]
        engine.traffic_manager.random_traffic = False
        engine.map_manager.maps = self._group_maps[name]
        probability = raw.config["accident_prob"]
        # A fresh env has no object manager at <= .01; its reset uses < .01.
        # Keep the always-registered manager equivalent at that boundary.
        if abs(probability) <= 1e-2:
            raw.config["accident_prob"] = 0.0
        try:
            result = self.env.reset(seed=seed, **kwargs)
        finally:
            raw.config["accident_prob"] = probability
        if int(raw.current_seed) != seed:
            raise RuntimeError(f"requested scenario seed {seed}, got {raw.current_seed}")
        self.current_scene_group = name
        self.scene_group_reset_counts[name][seed] += 1
        return result

    def close(self):
        raw = self.unwrapped
        if raw.engine is not None:
            manager = raw.engine.map_manager
            maps = {id(m): m for cache in self._group_maps.values() for m in cache.values() if m is not None}
            if manager.current_map is not None:
                maps[id(manager.current_map)] = manager.current_map
            # Upstream clear_stored_maps detaches and destroys each map once.
            manager.maps, manager.current_map = maps, None
        try:
            return self.env.close()
        finally:
            self._group_maps.clear()


def make_env(
    env_config: Mapping[str, object],
    *,
    lookahead_config: Mapping[str, object] | None = None,
) -> gym.Env:
    """指定設定を適用したraw MetaDrive環境、またはlookahead wrapperを返す。

    呼出元は解決済みTOMLのstage設定を明示して渡す。
    """

    # Importを実際の生成時まで遅らせ、未導入時にもinspect_env側が先に
    # outputs/inspect_env/<experiment>/<stage>.logを開いてImportError全文を
    # 記録できるようにする。
    # MetaDriveは受け取った設定を内部でmergeする。呼出元のconstantの
    # 偶発的な変更を防ぐため、環境ごとに浅いcopyを渡す。
    config = dict(env_config)
    groups = config.pop("scene_groups", None)
    if groups and any("lead_vehicles" in group["env_config"] for group in groups):
        config["lead_vehicles"] = {"count": [0, 0]}
    if "scenario_seeds" in config:
        seeds = validate_scenario_seeds(config.pop("scenario_seeds"))
        config.update(start_seed=min(seeds), num_scenarios=max(seeds) - min(seeds) + 1)
    # A missing key means the canonical upstream environment.  An explicitly
    # supplied ``off`` still needs this subclass so MetaDrive's closed config
    # schema recognizes the project-local key while retaining upstream runtime
    # reward/done behavior.
    if "lead_vehicles" in config:
        from lead_vehicle_env import LeadVehicleMetaDriveEnv

        raw_env = LeadVehicleMetaDriveEnv(config)
    elif "start_lane_objective" in config:
        # The custom task stays project-local so the installed/vendored
        # MetaDrive implementation remains byte-for-byte untouched.
        from start_lane_env import StartLaneMetaDriveEnv

        raw_env = StartLaneMetaDriveEnv(config)
    else:
        from metadrive.envs import MetaDriveEnv

        raw_env = MetaDriveEnv(config)
    if lookahead_config is None and groups is None:
        return raw_env

    # Keep the host factory responsible only for constructing the raw
    # environment.  The optional package owns its adapter/wrapper boundary and
    # receives the already resolved TOML values without a second mode switch.
    try:
        if lookahead_config is not None:
            from lookahead_learning.adapter import wrap_lookahead_env

            raw_env = wrap_lookahead_env(raw_env, **dict(lookahead_config))
        return _SceneGroupEnv(raw_env, groups) if groups is not None else raw_env
    except Exception:
        # A failed optional-wrapper import or construction must not leave the
        # just-created simulator alive in a worker process.
        raw_env.close()
        raise


def make_training_env(
    rank: int,
    seed: int,
    env_config: Mapping[str, object],
    lookahead_config: Mapping[str, object] | None = None,
) -> gym.Env:
    """CSVを出力せず、エピソード統計を集計する学習用環境を生成する。

    Args:
        rank: SubprocVecEnv内のworker番号。
        seed: Action/Observation spaceの乱数seedの基準値。
        env_config: MetaDriveへ渡す解決済みTOMLの環境設定。
        lookahead_config: Optional resolved ``[lookahead]`` settings.  When
            present, the lookahead wrapper is inserted before ``Monitor``.

    Returns:
        TensorBoard用のエピソード統計を集計するSB3 ``Monitor`` で包んだ環境。

    Notes:
        MetaDrive 0.4.3の ``reset(seed=...)`` はscenario indexを意味する。
        RL seedとの混同を避けるため、ここではspaceだけをseedする。
        scenario_seeds指定時だけ、worker固有の乱数で一覧からscenarioを選ぶ。
    """

    if rank < 0:
        raise ValueError(f"rank must be non-negative: {rank}")

    env = make_env(env_config, lookahead_config=lookahead_config)
    try:
        worker_seed = seed + rank
        env.action_space.seed(worker_seed)
        env.observation_space.seed(worker_seed)
        if "scene_groups" in env_config:
            env._scene_rng = random.Random(worker_seed)
        elif "scenario_seeds" in env_config:
            env = _ScenarioSeedSampler(env, env_config["scenario_seeds"], worker_seed)
        # Keep episode statistics for SB3/TensorBoard without writing CSV files.
        return Monitor(env)
    except Exception:
        # construction途中で失敗した環境はSubprocVecEnv側へ返らず、callerが
        # closeできないため、この場で確実に解放してから元の例外を伝える。
        env.close()
        raise


def make_evaluation_env(
    seed: int,
    env_config: Mapping[str, object],
    lookahead_config: Mapping[str, object] | None = None,
) -> gym.Env:
    """評価用の単一環境を生成する。

    MetaDrive 0.4.3のtop-down記録はconstruction時のconfigではなく
    ``env.render(..., screen_record=True)`` で開始する。録画の有無は評価側の
    recorderが扱い、このfactoryには渡さない。
    """

    env = make_env(env_config, lookahead_config=lookahead_config)
    try:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    except Exception:
        env.close()
        raise
