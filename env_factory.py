"""選択されたタスク設定からMetaDrive環境を生成するfactory。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

from configs.phase0_config import MONITOR_LOG_DIR, OFFICIAL_ENV_CONFIG, RL_SEED

if TYPE_CHECKING:
    from metadrive.envs import MetaDriveEnv


def make_env(env_config: Mapping[str, object] | None = None) -> MetaDriveEnv:
    """指定設定を適用した、wrapperなしのMetaDrive環境を返す。

    ``env_config`` を省略した場合は、Phase 0の公式設定を維持する。
    """

    # Importを実際の生成時まで遅らせ、未導入時にもinspect_env側が先に
    # outputs/inspect_env.logを開いてImportError全文を記録できるようにする。
    from metadrive.envs import MetaDriveEnv

    # MetaDriveは受け取った設定を内部でmergeする。呼出元のconstantの
    # 偶発的な変更を防ぐため、環境ごとに浅いcopyを渡す。
    selected_config = OFFICIAL_ENV_CONFIG if env_config is None else env_config
    return MetaDriveEnv(dict(selected_config))


def make_training_env(
    rank: int,
    seed: int = RL_SEED,
    monitor_dir: Path | str = MONITOR_LOG_DIR,
    env_config: Mapping[str, object] | None = None,
) -> gym.Env:
    """rank固有のMonitorログを持つ学習用環境を生成する。

    Args:
        rank: SubprocVecEnv内のworker番号。
        seed: Action/Observation spaceの乱数seedの基準値。
        monitor_dir: ``*.monitor.csv`` の保存先。
        env_config: MetaDriveへ渡す環境設定。省略時はPhase 0公式設定。

    Returns:
        記録専用のSB3 ``Monitor`` で包んだMetaDrive環境。

    Notes:
        MetaDrive 0.4.3の ``reset(seed=...)`` はscenario indexを意味する。
        RL seedとの混同を避けるため、ここではspaceだけをseedする。
    """

    if rank < 0:
        raise ValueError(f"rank must be non-negative: {rank}")

    destination = Path(monitor_dir)
    destination.mkdir(parents=True, exist_ok=True)

    env = make_env(env_config)
    try:
        worker_seed = seed + rank
        env.action_space.seed(worker_seed)
        env.observation_space.seed(worker_seed)
        monitor_file = destination / f"env_{rank}.monitor.csv"
        return Monitor(env, filename=str(monitor_file))
    except Exception:
        # construction途中で失敗した環境はSubprocVecEnv側へ返らず、callerが
        # closeできないため、この場で確実に解放してから元の例外を伝える。
        env.close()
        raise


def make_evaluation_env(
    seed: int = RL_SEED,
    record_gif: bool = False,
    env_config: Mapping[str, object] | None = None,
) -> MetaDriveEnv:
    """評価用の単一raw環境を生成する。

    ``record_gif`` は呼出側の意図を明示する引数である。MetaDrive 0.4.3は
    construction時のconfigではなく ``env.render(..., screen_record=True)`` で
    top-down記録を開始するため、ここでは選択された環境設定を変更しない。
    """

    if not isinstance(record_gif, bool):
        raise TypeError("record_gif must be bool")

    env = make_env(env_config)
    try:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    except Exception:
        env.close()
        raise
