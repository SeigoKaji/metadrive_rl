"""外部TOML実験bundleの検証とCLI接続を確認する。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from configs.experiment_config import (
    ExperimentConfigError,
    experiment_selection_from_args,
    load_experiment_config,
    select_experiment,
)
from configs.phase0_config import PROJECT_ROOT
from evaluate import parse_args as parse_evaluation_args
from train import parse_args as parse_training_args


VALID_TOML = """\
schema_version = 1
name = "custom_bundle"
algorithm = "ppo"
default_model_name = "custom_model"

[training]
policy = "MlpPolicy"
seed = 11
num_envs = 2
n_steps = 64
total_timesteps = 512
log_interval = 2
device = "cpu"
model_name = "custom_training.zip"
log_file = "logs/custom_train.log"

[evaluation]
episodes = 2
model_path = "models/custom_model.zip"
record_gif = false
output_prefix = "custom_evaluation"
seed = 12
device = "cpu"
log_file = "logs/custom_evaluate.log"
deterministic = false

[environment.common]
map = "C"
discrete_action = true
traffic_density = 0.0

[environment.common.vehicle_config]
enable_reverse = false
max_speed = 20

[environment.train]
start_seed = 100
num_scenarios = 10

[environment.train.vehicle_config]
max_speed = 30

[environment.evaluation]
start_seed = 0
num_scenarios = 2
"""


def _write_config(tmp_path: Path, text: str = VALID_TOML) -> Path:
    path = tmp_path / "experiment.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_example_bundle_drives_both_cli_default_sets() -> None:
    """同じTOMLだけで学習・評価の既定値とsourceを解決できる。"""

    relative_path = "configs/example_experiment.toml"
    expected_path = (PROJECT_ROOT / relative_path).resolve()
    expected_sha256 = hashlib.sha256(expected_path.read_bytes()).hexdigest()

    training_args = parse_training_args(["--config", relative_path])
    evaluation_args = parse_evaluation_args(["--config", relative_path])

    assert training_args.profile == "example_generalization"
    assert training_args.config == expected_path
    assert training_args.timesteps == 2_000
    assert training_args.num_envs == 1
    assert training_args.n_steps == 256
    assert training_args.seed == 0
    assert training_args.log_interval == 1
    assert training_args.model_name == "example_generalization"
    assert training_args.experiment.source_sha256 == expected_sha256

    assert evaluation_args.profile == "example_generalization"
    assert evaluation_args.config == expected_path
    assert evaluation_args.model == Path("models/example_generalization.zip")
    assert evaluation_args.episodes == 5
    assert evaluation_args.record_gif is False
    assert evaluation_args.output_prefix == "example_generalization"
    assert evaluation_args.seed == 0
    assert evaluation_args.device == "cpu"
    assert evaluation_args.deterministic is True
    assert evaluation_args.experiment.source_sha256 == expected_sha256
    assert training_args.experiment.profile.train_env_config["num_scenarios"] == 1000
    assert evaluation_args.experiment.profile.evaluation_env_config["num_scenarios"] == 5


def test_cli_options_override_toml_defaults() -> None:
    """二段parse後も明示したCLI値がTOMLの既定値に勝つ。"""

    config_path = "configs/example_experiment.toml"
    training_args = parse_training_args(
        [
            "--config",
            config_path,
            "--timesteps",
            "123",
            "--num-envs",
            "2",
            "--n-steps",
            "32",
            "--seed",
            "9",
            "--device",
            "auto",
            "--model-name",
            "override.zip",
            "--log-interval",
            "3",
            "--log-file",
            "logs/override_train.log",
        ]
    )
    assert training_args.timesteps == 123
    assert training_args.num_envs == 2
    assert training_args.n_steps == 32
    assert training_args.seed == 9
    assert training_args.device == "auto"
    assert training_args.model_name == "override"
    assert training_args.log_interval == 3
    assert training_args.log_file == Path("logs/override_train.log")

    evaluation_args = parse_evaluation_args(
        [
            "--config",
            config_path,
            "--model",
            "models/override.zip",
            "--episodes",
            "3",
            "--record-gif",
            "--output-prefix",
            "override_evaluation",
            "--seed",
            "10",
            "--device",
            "auto",
            "--log-file",
            "logs/override_evaluate.log",
            "--no-deterministic",
        ]
    )
    assert evaluation_args.model == Path("models/override.zip")
    assert evaluation_args.episodes == 3
    assert evaluation_args.record_gif is True
    assert evaluation_args.output_prefix == "override_evaluation"
    assert evaluation_args.seed == 10
    assert evaluation_args.device == "auto"
    assert evaluation_args.log_file == Path("logs/override_evaluate.log")
    assert evaluation_args.deterministic is False


def test_training_model_name_is_the_default_evaluation_model_and_prefix(
    tmp_path: Path,
) -> None:
    """evaluationの両keyを省略しても学習済みmodelと同じ名前を使う。"""

    path = _write_config(
        tmp_path,
        VALID_TOML.replace(
            "model_name = \"custom_training.zip\"",
            "model_name = \"linked_training_model\"",
        )
        .replace("model_path = \"models/custom_model.zip\"\n", "")
        .replace("output_prefix = \"custom_evaluation\"\n", ""),
    )

    training_args = parse_training_args(["--config", str(path)])
    evaluation_args = parse_evaluation_args(["--config", str(path)])

    assert training_args.experiment.profile.default_model_name == "custom_model"
    assert training_args.model_name == "linked_training_model"
    assert evaluation_args.model == Path("models/linked_training_model.zip")
    assert evaluation_args.output_prefix == "linked_training_model"


def test_loader_deep_merges_common_environment_and_records_source(
    tmp_path: Path,
) -> None:
    """commonのnested tableはstage側の値だけを再帰的に上書きする。"""

    path = _write_config(tmp_path)
    selection = load_experiment_config(path)

    assert selection.name == "custom_bundle"
    assert selection.source_kind == "toml"
    assert selection.source_path == path.resolve()
    assert selection.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert selection.source_metadata() == {
        "kind": "toml",
        "name": "custom_bundle",
        "profile": "custom_bundle",
        "path": str(path.resolve()),
        "sha256": selection.source_sha256,
    }
    assert selection.profile.train_env_config["vehicle_config"] == {
        "enable_reverse": False,
        "max_speed": 30,
    }
    assert selection.profile.evaluation_env_config["vehicle_config"] == {
        "enable_reverse": False,
        "max_speed": 20,
    }
    assert selection.profile.training_config["model_name"] == "custom_training"
    assert selection.profile.evaluation_defaults == {
        "episodes": 2,
        "model_path": "models/custom_model.zip",
        "record_gif": False,
        "output_prefix": "custom_evaluation",
        "seed": 12,
        "device": "cpu",
        "log_file": "logs/custom_evaluate.log",
        "deterministic": False,
    }


@pytest.mark.parametrize(
    ("path_name", "contents", "match"),
    [
        ("missing.toml", None, "解決できません"),
        ("invalid.toml", "schema_version = ", "TOMLの構文"),
        ("wrong_suffix.json", VALID_TOML, ".toml"),
    ],
)
def test_loader_rejects_missing_invalid_and_non_toml_sources(
    tmp_path: Path,
    path_name: str,
    contents: str | None,
    match: str,
) -> None:
    """外部sourceは存在する有効なTOMLだけを受け入れる。"""

    path = tmp_path / path_name
    if contents is not None:
        path.write_text(contents, encoding="utf-8")

    with pytest.raises(ExperimentConfigError, match=match):
        load_experiment_config(path)


def test_loader_wraps_invalid_utf8_as_a_config_error(tmp_path: Path) -> None:
    """TOMLとして読めないbytesをUnicodeDecodeErrorのまま漏らさない。"""

    path = tmp_path / "invalid_utf8.toml"
    path.write_bytes(b"\xff")

    with pytest.raises(ExperimentConfigError, match="TOMLの構文"):
        load_experiment_config(path)


@pytest.mark.parametrize(
    ("before", "after", "match"),
    [
        ("policy = \"MlpPolicy\"\n", "", "training"),
        (
            "[environment.evaluation]\nstart_seed = 0\nnum_scenarios = 2\n",
            "",
            "environment.evaluation",
        ),
    ],
)
def test_loader_rejects_missing_required_tables_and_keys(
    tmp_path: Path,
    before: str,
    after: str,
    match: str,
) -> None:
    """実行に必要なtraining keyとstage tableを省略できない。"""

    path = _write_config(tmp_path, VALID_TOML.replace(before, after, 1))

    with pytest.raises(ExperimentConfigError, match=match):
        load_experiment_config(path)


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("algorithm = \"ppo\"", "algorithm = \"sac\""),
        ("algorithm = \"ppo\"", "algorithm = \"ppo\"\nextra = true"),
        ("log_interval = 2", "log_interval = 2\nlearning_rate = 0.001"),
        (
            "[environment.train]",
            "[environment.wrapper]\nunknown = true\n\n[environment.train]",
        ),
    ],
)
def test_loader_rejects_unsupported_algorithm_and_unknown_keys(
    tmp_path: Path,
    before: str,
    after: str,
) -> None:
    """algorithmやschemaで黙って無視されるkeyを作らない。"""

    path = _write_config(tmp_path, VALID_TOML.replace(before, after, 1))

    with pytest.raises(ExperimentConfigError):
        load_experiment_config(path)


@pytest.mark.parametrize(
    ("before", "after", "match"),
    [
        ("name = \"custom_bundle\"", "name = \"../outside\"", "name"),
        (
            "default_model_name = \"custom_model\"",
            "default_model_name = \"../model\"",
            "default_model_name",
        ),
        (
            "model_name = \"custom_training.zip\"",
            "model_name = \"../model\"",
            "training.model_name",
        ),
        (
            "output_prefix = \"custom_evaluation\"",
            "output_prefix = \"../output\"",
            "evaluation.output_prefix",
        ),
    ],
)
def test_loader_rejects_path_traversal_in_output_basenames(
    tmp_path: Path,
    before: str,
    after: str,
    match: str,
) -> None:
    """成果物dirやmodel名に親directoryを混入させない。"""

    path = _write_config(tmp_path, VALID_TOML.replace(before, after, 1))

    with pytest.raises(ExperimentConfigError, match=match):
        load_experiment_config(path)


@pytest.mark.parametrize(
    ("before", "after", "match"),
    [
        ("seed = 11", "seed = true", "training.seed"),
        ("seed = 11", "seed = -1", "training.seed"),
        ("seed = 11", "seed = 4294967296", "training.seed"),
        ("seed = 12", "seed = -1", "evaluation.seed"),
        ("seed = 12", "seed = 4294967296", "evaluation.seed"),
        ("num_scenarios = 10", "num_scenarios = 0", "environment.train"),
        ("traffic_density = 0.0", "traffic_density = nan", "environment.common"),
        (
            "traffic_density = 0.0",
            "traffic_density = 0.0\ncreated_on = 2026-08-25",
            "environment.common",
        ),
    ],
)
def test_loader_rejects_invalid_scalar_and_environment_values(
    tmp_path: Path,
    before: str,
    after: str,
    match: str,
) -> None:
    """bool seed、非有限float、日時型などをruntimeへ流さない。"""

    path = _write_config(tmp_path, VALID_TOML.replace(before, after, 1))

    with pytest.raises(ExperimentConfigError, match=match):
        load_experiment_config(path)


@pytest.mark.parametrize(
    ("before", "after", "match"),
    [
        ("discrete_action = true", "discrete_action = false", "discrete_action"),
        (
            "discrete_action = true",
            "discrete_action = \"true\"",
            "discrete_action",
        ),
        (
            "traffic_density = 0.0",
            "traffic_density = 0.0\nuse_multi_discrete = true",
            "use_multi_discrete",
        ),
        (
            "traffic_density = 0.0",
            "traffic_density = 0.0\nuse_multi_discrete = \"false\"",
            "use_multi_discrete",
        ),
        (
            "traffic_density = 0.0",
            "traffic_density = 0.0\nis_multi_agent = true",
            "is_multi_agent",
        ),
        (
            "traffic_density = 0.0",
            "traffic_density = 0.0\nis_multi_agent = \"false\"",
            "is_multi_agent",
        ),
        (
            "traffic_density = 0.0",
            "traffic_density = 0.0\ndiscrete_steering_dim = 1",
            "discrete_steering_dim",
        ),
        (
            "traffic_density = 0.0",
            "traffic_density = 0.0\ndiscrete_throttle_dim = true",
            "discrete_throttle_dim",
        ),
    ],
)
def test_loader_rejects_action_settings_unsupported_by_evaluation(
    tmp_path: Path,
    before: str,
    after: str,
    match: str,
) -> None:
    """連続、多次元、多agent、無効な離散dimはload時に拒否する。"""

    path = _write_config(tmp_path, VALID_TOML.replace(before, after, 1))

    with pytest.raises(ExperimentConfigError, match=match):
        load_experiment_config(path)


@pytest.mark.parametrize(
    ("before", "after", "match"),
    [
        (
            "num_scenarios = 10\n\n[environment.train.vehicle_config]",
            "num_scenarios = 10\nuse_multi_discrete = true"
            "\n\n[environment.train.vehicle_config]",
            "environment.train.use_multi_discrete",
        ),
        (
            "num_scenarios = 2\n",
            "num_scenarios = 2\nis_multi_agent = true\n",
            "environment.evaluation.is_multi_agent",
        ),
        (
            "num_scenarios = 10\n\n[environment.train.vehicle_config]",
            "num_scenarios = 10\nnum_agents = 2"
            "\n\n[environment.train.vehicle_config]",
            "environment.train.num_agents",
        ),
        (
            "num_scenarios = 2\n",
            "num_scenarios = 2\nnum_agents = 2\n",
            "environment.evaluation.num_agents",
        ),
        (
            "num_scenarios = 2\n",
            "num_scenarios = 2\nnum_agents = true\n",
            "environment.evaluation.num_agents",
        ),
    ],
)
def test_loader_validates_action_compatibility_for_each_stage(
    tmp_path: Path,
    before: str,
    after: str,
    match: str,
) -> None:
    """commonだけでなく各stage固有のAction・agent設定も検査する。"""

    path = _write_config(tmp_path, VALID_TOML.replace(before, after, 1))

    with pytest.raises(ExperimentConfigError, match=match):
        load_experiment_config(path)


def test_loader_rejects_toml_episode_default_larger_than_scenario_range(
    tmp_path: Path,
) -> None:
    """複数scenario評価の重複をTOML既定値の解決時に防ぐ。"""

    path = _write_config(tmp_path, VALID_TOML.replace("episodes = 2", "episodes = 3"))

    with pytest.raises(ExperimentConfigError, match="evaluation.episodes"):
        load_experiment_config(path)


def test_loader_rejects_single_item_ppo_rollout_batch(tmp_path: Path) -> None:
    """SB3 PPOが後段で必ず失敗する1件rolloutをload時に止める。"""

    path = _write_config(
        tmp_path,
        VALID_TOML.replace("num_envs = 2", "num_envs = 1").replace(
            "n_steps = 64", "n_steps = 1"
        ),
    )

    with pytest.raises(ExperimentConfigError, match="training"):
        load_experiment_config(path)


def test_loader_allows_repeated_episodes_for_a_single_scenario(tmp_path: Path) -> None:
    """単一scenario profileの従来どおりの複数episode反復を維持する。"""

    path = _write_config(
        tmp_path,
        VALID_TOML.replace("episodes = 2", "episodes = 3").replace(
            "num_scenarios = 2", "num_scenarios = 1", 1
        ),
    )

    assert load_experiment_config(path).profile.evaluation_episodes == 3


def test_profile_and_config_are_mutually_exclusive_for_both_clis() -> None:
    """選択sourceを二重指定するとbootstrap parserで止める。"""

    argv = [
        "--profile",
        "official",
        "--config",
        "configs/example_experiment.toml",
    ]
    with pytest.raises(SystemExit):
        parse_training_args(argv)
    with pytest.raises(SystemExit):
        parse_evaluation_args(argv)


def test_builtin_selection_and_legacy_namespace_keep_working() -> None:
    """既存profileとconfig属性を持たない評価test用Namespaceを維持する。"""

    selection = select_experiment(profile_name="official")
    legacy_selection = experiment_selection_from_args(
        SimpleNamespace(profile="official")
    )

    assert selection.source_kind == "builtin_profile"
    assert selection.source_path is None
    assert selection.source_sha256 is None
    assert selection.profile is legacy_selection.profile
    assert selection.source_metadata()["profile"] == "official"


def test_cli_does_not_abbreviate_config_option() -> None:
    """二段parseを含め、誤った短縮optionを受理しない。"""

    with pytest.raises(SystemExit):
        parse_training_args(["--conf", "configs/example_experiment.toml"])
    with pytest.raises(SystemExit):
        parse_evaluation_args(["--conf", "configs/example_experiment.toml"])
