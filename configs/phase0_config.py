"""MetaDrive公式SB3ミニ例のPhase 0設定を一元管理する。"""

from pathlib import Path
from typing import Final


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

# 公式ドキュメントのcreate_env()にある設定だけを保持する。
# 描画や検査の都合でタスク設定をここへ追加してはならない。
OFFICIAL_ENV_CONFIG: Final[dict[str, object]] = {
    "map": "C",
    "discrete_action": True,
    "discrete_throttle_dim": 3,
    "discrete_steering_dim": 3,
    "horizon": 500,
    "random_spawn_lane_index": False,
    "num_scenarios": 1,
    "start_seed": 5,
    "traffic_density": 0,
    "accident_prob": 0,
    "log_level": 50,
}

# PPOの未記載パラメータはStable-Baselines3のデフォルトに委ねる。
OFFICIAL_TRAINING_CONFIG: Final[dict[str, object]] = {
    "seed": 0,
    "num_envs": 4,
    "n_steps": 4096,
    "total_timesteps": 300_000,
    "log_interval": 4,
    "policy": "MlpPolicy",
}

# scenario seedはMetaDriveが道路シナリオを選ぶ値、RL seedはPPO等の乱数seed。
# 両者を混同してenv.reset(seed=RL_SEED)を呼ばないこと。
SCENARIO_SEED: Final[int] = int(OFFICIAL_ENV_CONFIG["start_seed"])
RL_SEED: Final[int] = int(OFFICIAL_TRAINING_CONFIG["seed"])
NUM_ENVS: Final[int] = int(OFFICIAL_TRAINING_CONFIG["num_envs"])
N_STEPS: Final[int] = int(OFFICIAL_TRAINING_CONFIG["n_steps"])
TOTAL_TIMESTEPS: Final[int] = int(OFFICIAL_TRAINING_CONFIG["total_timesteps"])

# 生成物はすべてこのPhase 0ディレクトリ内へ保存する。
MODEL_DIR: Final[Path] = PROJECT_ROOT / "models"
LOG_DIR: Final[Path] = PROJECT_ROOT / "logs"
MONITOR_LOG_DIR: Final[Path] = LOG_DIR / "monitor"
TENSORBOARD_LOG_DIR: Final[Path] = LOG_DIR / "tensorboard"
OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "outputs"

DEFAULT_MODEL_NAME: Final[str] = "phase0_official"
MODEL_SAVE_PATH: Final[Path] = MODEL_DIR / DEFAULT_MODEL_NAME
OFFICIAL_TRAINING_OUTPUT_DIR: Final[Path] = (
    OUTPUT_DIR / "official" / "training" / DEFAULT_MODEL_NAME
)
OFFICIAL_EVALUATION_OUTPUT_DIR: Final[Path] = (
    OUTPUT_DIR / "official" / "evaluation" / DEFAULT_MODEL_NAME
)
EVALUATION_RESULTS_DIR: Final[Path] = OFFICIAL_EVALUATION_OUTPUT_DIR
EVALUATION_RESULTS_PATH: Final[Path] = (
    EVALUATION_RESULTS_DIR / "evaluation.json"
)
