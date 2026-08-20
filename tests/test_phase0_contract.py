"""Phase 0が公式MetaDriveタスクを変更していないことを検証する。"""

import hashlib
import json
from numbers import Real

import gymnasium as gym
from stable_baselines3.common.env_checker import check_env

from env_factory import make_env
from inspect_env import _CheckEnvFixedScenarioAdapter
from configs.phase0_config import OFFICIAL_ENV_CONFIG, SCENARIO_SEED


# 公式dictを別ファイルへ複製せず、canonical JSONのdigestでkey/valueを厳密固定する。
OFFICIAL_ENV_CONFIG_SHA256 = "0cd930ac933937c9bf2bd8313eb66deb88a7864bcbc12808fcb5a669f61e5e37"


def test_official_environment_config_is_exact() -> None:
    """公式例のkey/value以外が追加・変更されていない。"""

    canonical_config = json.dumps(
        OFFICIAL_ENV_CONFIG,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    assert hashlib.sha256(canonical_config).hexdigest() == OFFICIAL_ENV_CONFIG_SHA256
    assert "use_lateral_reward" not in OFFICIAL_ENV_CONFIG
    assert "truncate_as_terminate" not in OFFICIAL_ENV_CONFIG


def test_raw_environment_contract_and_close() -> None:
    """生成、reset、1 step、closeまでGymnasium契約を満たす。"""

    env = make_env()
    try:
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == 9

        reset_result = env.reset()
        assert isinstance(reset_result, tuple)
        assert len(reset_result) == 2
        observation, info = reset_result
        assert isinstance(info, dict)
        assert env.observation_space.contains(observation)

        step_result = env.step(env.action_space.sample())
        assert isinstance(step_result, tuple)
        assert len(step_result) == 5
        next_observation, reward, terminated, truncated, step_info = step_result
        assert env.observation_space.contains(next_observation)
        assert isinstance(reward, Real) and not isinstance(reward, bool)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(step_info, dict)
    finally:
        env.close()


def test_check_env_adapter_preserves_official_scenario() -> None:
    """SB3 checkerのseed=0を受理しても公式scenario 5を変更しない。"""

    env = make_env()
    try:
        checker_env = _CheckEnvFixedScenarioAdapter(env)
        assert checker_env.unwrapped is env

        check_env(checker_env, warn=True)

        assert env.current_seed == SCENARIO_SEED
        actual_config = {key: env.config[key] for key in OFFICIAL_ENV_CONFIG}
        assert actual_config == OFFICIAL_ENV_CONFIG
    finally:
        env.close()
