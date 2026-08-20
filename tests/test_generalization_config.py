"""汎化設定、profile選択、MetaDriveへの接続を検証する。"""

import pytest

from env_factory import make_env
from evaluate import (
    _evaluation_output_directory,
    parse_args as parse_evaluation_args,
)
from configs.experiment_profiles import PROFILE_NAMES, get_experiment_profile
from configs.generalization_config import (
    COMMON_GENERALIZATION_ENV_CONFIG,
    GENERALIZATION_EVALUATION_ENV_CONFIG,
    GENERALIZATION_EVALUATION_EPISODES,
    GENERALIZATION_TRAIN_ENV_CONFIG,
    GENERALIZATION_TRAINING_CONFIG,
)
from configs.phase0_config import OFFICIAL_ENV_CONFIG, OFFICIAL_TRAINING_CONFIG, OUTPUT_DIR
from train import _training_output_directory, parse_args as parse_training_args


EXPECTED_COMMON_CONFIG = {
    "map": 3,
    "discrete_action": True,
    "discrete_throttle_dim": 3,
    "discrete_steering_dim": 3,
    "horizon": 1000,
    "random_spawn_lane_index": True,
    "random_lane_width": True,
    "random_lane_num": True,
    "random_agent_model": True,
    "traffic_density": 0.1,
    "random_traffic": False,
    "accident_prob": 0.0,
    "store_map": False,
    "log_level": 50,
}


def test_generalization_environment_configs_only_split_scenario_ranges() -> None:
    """学習・評価は共通仕様を保ち、scenario範囲だけを変更する。"""

    assert COMMON_GENERALIZATION_ENV_CONFIG == EXPECTED_COMMON_CONFIG

    differing_keys = {
        key
        for key in GENERALIZATION_TRAIN_ENV_CONFIG.keys()
        | GENERALIZATION_EVALUATION_ENV_CONFIG.keys()
        if GENERALIZATION_TRAIN_ENV_CONFIG.get(key)
        != GENERALIZATION_EVALUATION_ENV_CONFIG.get(key)
    }
    assert differing_keys == {"start_seed", "num_scenarios"}

    for key, expected_value in EXPECTED_COMMON_CONFIG.items():
        assert GENERALIZATION_TRAIN_ENV_CONFIG[key] == expected_value
        assert GENERALIZATION_EVALUATION_ENV_CONFIG[key] == expected_value

    # 9通りの離散行動と同じ既定観測方式をtrain/evaluationで共有する。
    assert GENERALIZATION_TRAIN_ENV_CONFIG["discrete_action"] is True
    assert GENERALIZATION_TRAIN_ENV_CONFIG["discrete_throttle_dim"] == 3
    assert GENERALIZATION_TRAIN_ENV_CONFIG["discrete_steering_dim"] == 3
    assert "image_observation" not in GENERALIZATION_TRAIN_ENV_CONFIG
    assert "image_observation" not in GENERALIZATION_EVALUATION_ENV_CONFIG


def test_generalization_scenario_ranges_are_disjoint() -> None:
    """評価scenarioは学習中に一度も選ばれない。"""

    train_start = int(GENERALIZATION_TRAIN_ENV_CONFIG["start_seed"])
    train_count = int(GENERALIZATION_TRAIN_ENV_CONFIG["num_scenarios"])
    evaluation_start = int(GENERALIZATION_EVALUATION_ENV_CONFIG["start_seed"])
    evaluation_count = int(
        GENERALIZATION_EVALUATION_ENV_CONFIG["num_scenarios"]
    )

    train_seeds = range(train_start, train_start + train_count)
    evaluation_seeds = range(evaluation_start, evaluation_start + evaluation_count)

    assert (train_start, train_count) == (1000, 1000)
    assert (evaluation_start, evaluation_count) == (0, 200)
    assert set(train_seeds).isdisjoint(evaluation_seeds)


def test_generalization_training_and_evaluation_defaults() -> None:
    """汎化実験のPPO予算と評価episode数を固定する。"""

    assert GENERALIZATION_TRAINING_CONFIG == {
        "seed": 0,
        "num_envs": 4,
        "n_steps": 4096,
        "total_timesteps": 1_000_000,
        "log_interval": 4,
        "policy": "MlpPolicy",
    }
    assert GENERALIZATION_EVALUATION_EPISODES == 200


def test_profile_selector_returns_typed_config_bundles() -> None:
    """profile名から公式・汎化それぞれの設定一式を選べる。"""

    assert PROFILE_NAMES == ("official", "generalization")

    official = get_experiment_profile("official")
    assert official.train_env_config is OFFICIAL_ENV_CONFIG
    assert official.evaluation_env_config is OFFICIAL_ENV_CONFIG
    assert official.training_config is OFFICIAL_TRAINING_CONFIG
    assert official.default_model_name == "phase0_official"
    assert official.evaluation_episodes == 1

    generalization = get_experiment_profile("generalization")
    assert generalization.train_env_config is GENERALIZATION_TRAIN_ENV_CONFIG
    assert (
        generalization.evaluation_env_config
        is GENERALIZATION_EVALUATION_ENV_CONFIG
    )
    assert generalization.training_config is GENERALIZATION_TRAINING_CONFIG
    assert generalization.default_model_name == "generalization"
    assert generalization.evaluation_episodes == 200


def test_profile_selector_rejects_unknown_name() -> None:
    """typoを暗黙に公式profileへfallbackさせない。"""

    with pytest.raises(ValueError, match="unknown experiment profile"):
        get_experiment_profile("unknown")


def test_generalization_profile_is_connected_to_both_clis() -> None:
    """profile指定が学習・評価CLIの既定値をまとめて切り替える。"""

    official_training_args = parse_training_args([])
    assert official_training_args.profile == "official"
    assert official_training_args.timesteps == 300_000
    assert official_training_args.model_name == "phase0_official"

    official_evaluation_args = parse_evaluation_args([])
    assert official_evaluation_args.episodes == 1
    assert official_evaluation_args.model.name == "phase0_official.zip"

    training_args = parse_training_args(["--profile", "generalization"])
    assert training_args.timesteps == 1_000_000
    assert training_args.num_envs == 4
    assert training_args.n_steps == 4096
    assert training_args.model_name == "generalization"

    evaluation_args = parse_evaluation_args(["--profile", "generalization"])
    assert evaluation_args.episodes == 200
    assert evaluation_args.model.name == "generalization.zip"
    assert evaluation_args.output_prefix == "generalization"


def test_artifact_directories_are_separated_by_profile_and_stage() -> None:
    """同じrun名でもprofileと学習・評価の組み合わせで衝突しない。"""

    paths = {
        _training_output_directory("official", "shared"),
        _training_output_directory("generalization", "shared"),
        _evaluation_output_directory("official", "shared"),
        _evaluation_output_directory("generalization", "shared"),
    }

    assert len(paths) == 4
    assert _training_output_directory("official", "shared") == (
        OUTPUT_DIR / "official" / "training" / "shared"
    )
    assert _evaluation_output_directory("generalization", "shared") == (
        OUTPUT_DIR / "generalization" / "evaluation" / "shared"
    )


def test_generalization_environment_can_reset_and_step() -> None:
    """新configをMetaDriveへ渡して未見scenarioの1 stepを実行できる。"""

    env = make_env(GENERALIZATION_EVALUATION_ENV_CONFIG)
    try:
        observation, reset_info = env.reset(seed=0)
        assert env.current_seed == 0
        assert env.action_space.n == 9
        assert env.observation_space.shape == (261,)
        assert env.observation_space.contains(observation)
        assert isinstance(reset_info, dict)

        next_observation, _reward, terminated, truncated, step_info = env.step(
            env.action_space.sample()
        )
        assert env.observation_space.contains(next_observation)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(step_info, dict)
    finally:
        env.close()
