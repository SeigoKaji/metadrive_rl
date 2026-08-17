"""複数の道路シナリオで学習・評価する汎化実験の設定。"""

from typing import Final


# 学習と評価で同じ行動・観測仕様およびシナリオ生成条件を使う。
# scenario seedの範囲だけは、未知シナリオで評価できるように下で分離する。
COMMON_GENERALIZATION_ENV_CONFIG: Final[dict[str, object]] = {
    "map": 3,
    "discrete_action": True,
    "discrete_throttle_dim": 3,
    "discrete_steering_dim": 3,
    "horizon": 1000,
    "random_spawn_lane_index": True,
    "random_lane_width": True,
    "random_lane_num": True,
    # 車種情報が標準state observationへ加わるため、Phase 0 modelとはspaceが異なる。
    "random_agent_model": True,
    "traffic_density": 0.1,
    "random_traffic": False,
    "accident_prob": 0.0,
    "store_map": False,
    "log_level": 50,
}

GENERALIZATION_TRAIN_ENV_CONFIG: Final[dict[str, object]] = {
    **COMMON_GENERALIZATION_ENV_CONFIG,
    "start_seed": 1000,
    "num_scenarios": 1000,
}

GENERALIZATION_EVALUATION_ENV_CONFIG: Final[dict[str, object]] = {
    **COMMON_GENERALIZATION_ENV_CONFIG,
    "start_seed": 0,
    "num_scenarios": 200,
}

GENERALIZATION_TRAINING_CONFIG: Final[dict[str, object]] = {
    "seed": 0,
    "num_envs": 4,
    "n_steps": 4096,
    "total_timesteps": 1_000_000,
    "log_interval": 4,
    "policy": "MlpPolicy",
}

GENERALIZATION_DEFAULT_MODEL_NAME: Final[str] = "generalization"
GENERALIZATION_EVALUATION_EPISODES: Final[int] = 200
