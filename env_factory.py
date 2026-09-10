"""選択されたタスク設定からMetaDrive環境を生成するfactory。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

if TYPE_CHECKING:
    from metadrive.envs import MetaDriveEnv


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
    # A missing key means the canonical upstream environment.  An explicitly
    # supplied ``off`` still needs this subclass so MetaDrive's closed config
    # schema recognizes the project-local key while retaining upstream runtime
    # reward/done behavior.
    if "start_lane_objective" in config:
        # The custom task stays project-local so the installed/vendored
        # MetaDrive implementation remains byte-for-byte untouched.
        from start_lane_env import StartLaneMetaDriveEnv

        raw_env = StartLaneMetaDriveEnv(config)
    else:
        from metadrive.envs import MetaDriveEnv

        raw_env = MetaDriveEnv(config)
    if lookahead_config is None:
        return raw_env

    # Keep the host factory responsible only for constructing the raw
    # environment.  The optional package owns its adapter/wrapper boundary and
    # receives the already resolved TOML values without a second mode switch.
    try:
        from lookahead_learning.adapter import wrap_lookahead_env

        return wrap_lookahead_env(raw_env, **dict(lookahead_config))
    except Exception:
        # A failed optional-wrapper import or construction must not leave the
        # just-created simulator alive in a worker process.
        raw_env.close()
        raise


def make_training_env(
    rank: int,
    seed: int,
    monitor_dir: Path | str,
    env_config: Mapping[str, object],
    lookahead_config: Mapping[str, object] | None = None,
) -> gym.Env:
    """rank固有のMonitorログを持つ学習用環境を生成する。

    Args:
        rank: SubprocVecEnv内のworker番号。
        seed: Action/Observation spaceの乱数seedの基準値。
        monitor_dir: ``*.monitor.csv`` の保存先。
        env_config: MetaDriveへ渡す解決済みTOMLの環境設定。
        lookahead_config: Optional resolved ``[lookahead]`` settings.  When
            present, the lookahead wrapper is inserted before ``Monitor``.

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

    env = make_env(env_config, lookahead_config=lookahead_config)
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
