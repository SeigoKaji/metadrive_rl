"""元の公式例を基準にしたofficial baselineの現在のprofile契約を検証する。"""

import hashlib
import json
from numbers import Real

import gymnasium as gym
from stable_baselines3.common.env_checker import check_env

from env_factory import make_env
from inspect_env import (
    _CheckEnvFixedScenarioAdapter,
    _is_known_check_env_seed_conflict,
    _scenario_seed_bounds,
)
from configs.experiment_config import select_experiment


_OFFICIAL_SELECTION = select_experiment(profile_name="official")
OFFICIAL_TRAIN_ENV_CONFIG = _OFFICIAL_SELECTION.profile.train_env_config
OFFICIAL_EVALUATION_ENV_CONFIG = _OFFICIAL_SELECTION.profile.evaluation_env_config
SCENARIO_SEED = int(OFFICIAL_TRAIN_ENV_CONFIG["start_seed"])


# 現在のofficial profile契約を別ファイルへ複製せず、canonical JSONのdigestで固定する。
# 学習・評価とも、公式例と同じseed 5の単一scenarioを使う。
OFFICIAL_ENV_CONFIG_SHA256 = "0cd930ac933937c9bf2bd8313eb66deb88a7864bcbc12808fcb5a669f61e5e37"


def _canonical_config_sha256(env_config: dict[str, object]) -> str:
    """設定のcanonical JSON digestを返す。"""

    canonical_config = json.dumps(
        env_config,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(canonical_config).hexdigest()


def test_official_environment_configs_match_current_contract() -> None:
    """学習・評価とも公式例の単一scenario契約を検出する。"""

    assert OFFICIAL_TRAIN_ENV_CONFIG == OFFICIAL_EVALUATION_ENV_CONFIG
    for env_config in (
        OFFICIAL_TRAIN_ENV_CONFIG,
        OFFICIAL_EVALUATION_ENV_CONFIG,
    ):
        assert _canonical_config_sha256(env_config) == OFFICIAL_ENV_CONFIG_SHA256
        assert (env_config["start_seed"], env_config["num_scenarios"]) == (5, 1)
        assert "use_lateral_reward" not in env_config
        assert "truncate_as_terminate" not in env_config


def test_check_env_conflict_ranges_use_current_official_configs() -> None:
    """学習・評価とも診断対象のscenario範囲を現在のconfigから導出する。"""

    for env_config in (
        OFFICIAL_TRAIN_ENV_CONFIG,
        OFFICIAL_EVALUATION_ENV_CONFIG,
    ):
        scenario_start, scenario_stop = _scenario_seed_bounds(env_config)
        assert (scenario_start, scenario_stop) == (5, 6)
        assert _is_known_check_env_seed_conflict(
            AssertionError(
                f"scenario_index (seed) should be in [{scenario_start}:{scenario_stop})"
            ),
            env_config,
        )


def test_raw_environment_contract_and_close() -> None:
    """生成、reset、1 step、closeまでGymnasium契約を満たす。"""

    env = make_env(OFFICIAL_TRAIN_ENV_CONFIG)
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
    """SB3 checkerのseed=0を受理しても公式profileの開始scenarioを変更しない。"""

    env = make_env(OFFICIAL_TRAIN_ENV_CONFIG)
    try:
        checker_env = _CheckEnvFixedScenarioAdapter(env, scenario_seed=SCENARIO_SEED)
        assert checker_env.unwrapped is env

        check_env(checker_env, warn=True)

        assert env.current_seed == SCENARIO_SEED
        actual_config = {key: env.config[key] for key in OFFICIAL_TRAIN_ENV_CONFIG}
        assert actual_config == OFFICIAL_TRAIN_ENV_CONFIG
    finally:
        env.close()
