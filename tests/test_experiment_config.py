"""外部TOML実験bundleの検証とCLI接続を確認する。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import configs.experiment_config as experiment_config_module
from configs.experiment_config import (
    ExperimentConfigError,
    PPO_COMMON_SCALAR_DEFAULTS,
    PPO_COMMON_SCALAR_KEYS,
    PROFILE_NAMES,
    canonical_config_path,
    experiment_selection_from_args,
    load_experiment_config,
    select_experiment,
)
from evaluate import parse_args as parse_evaluation_args
from project_paths import PROJECT_ROOT
from train import parse_args as parse_training_args


VALID_TOML = """\
schema_version = 2
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


@pytest.mark.parametrize("profile_name", PROFILE_NAMES)
def test_profile_aliases_resolve_to_canonical_toml_with_source_hash(
    profile_name: str,
) -> None:
    """互換aliasも実行時metadataでは実TOMLのpathとhashを残す。"""

    selection = select_experiment(profile_name=profile_name)
    expected_path = canonical_config_path(profile_name).resolve()

    assert selection.source_kind == "toml"
    assert selection.source_path == expected_path
    assert selection.source_sha256 == hashlib.sha256(expected_path.read_bytes()).hexdigest()


def test_default_cli_selection_is_the_official_canonical_toml() -> None:
    """引数なし学習・評価はbuiltin Pythonではなくofficial.tomlを使う。"""

    expected_path = canonical_config_path("official").resolve()
    expected_sha256 = hashlib.sha256(expected_path.read_bytes()).hexdigest()

    training_args = parse_training_args([])
    evaluation_args = parse_evaluation_args([])

    for args in (training_args, evaluation_args):
        assert args.config == expected_path
        assert args.experiment.source_path == expected_path
        assert args.experiment.source_sha256 == expected_sha256
        assert args.experiment.source_metadata()["path"] == str(expected_path)


def test_profile_alias_rejects_a_canonical_toml_with_a_different_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """alias先の誤編集で成果物が別名directoryへ流れることを防ぐ。"""

    mismatched = _write_config(
        tmp_path,
        VALID_TOML.replace('name = "custom_bundle"', 'name = "other_name"'),
    )
    monkeypatch.setitem(
        experiment_config_module._PROFILE_FILENAMES,
        "official",
        str(mismatched),
    )

    with pytest.raises(ExperimentConfigError, match="nameはaliasと一致"):
        select_experiment(profile_name="official")


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
    assert evaluation_args.record_gif is True
    assert evaluation_args.output_prefix == "override_evaluation"
    assert evaluation_args.seed == 10
    assert evaluation_args.device == "auto"
    assert evaluation_args.log_file == Path("logs/override_evaluate.log")
    assert evaluation_args.deterministic is False


def test_training_cli_rejects_single_item_rollout_after_overrides() -> None:
    """TOMLは有効でもCLIのnum-envs/n-steps上書き後に再検証する。"""

    with pytest.raises(SystemExit):
        parse_training_args(
            [
                "--config",
                "configs/example_experiment.toml",
                "--num-envs",
                "1",
                "--n-steps",
                "1",
            ]
        )


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
        "model_path": "models/custom_model.zip",
        "record_gif": False,
        "output_prefix": "custom_evaluation",
        "seed": 12,
        "device": "cpu",
        "log_file": "logs/custom_evaluate.log",
        "deterministic": False,
    }


def test_loader_resolves_optional_ppo_scalars_to_sb3_defaults(tmp_path: Path) -> None:
    """既存schema v2 bundleでも全PPO scalarを明示した解決値として受け取る。"""

    selection = load_experiment_config(_write_config(tmp_path))

    assert tuple(PPO_COMMON_SCALAR_DEFAULTS) == PPO_COMMON_SCALAR_KEYS
    assert {
        key: selection.profile.training_config[key]
        for key in PPO_COMMON_SCALAR_KEYS
    } == PPO_COMMON_SCALAR_DEFAULTS


@pytest.mark.parametrize(
    ("key", "toml_value", "expected"),
    [
        ("learning_rate", "0.001", 0.001),
        ("batch_size", "32", 32),
        ("n_epochs", "3", 3),
        ("gamma", "0.9", 0.9),
        ("gae_lambda", "0.8", 0.8),
        ("clip_range", "0.1", 0.1),
        ("normalize_advantage", "false", False),
        ("ent_coef", "0.02", 0.02),
        ("vf_coef", "0.7", 0.7),
        ("max_grad_norm", "0.3", 0.3),
    ],
)
def test_loader_accepts_each_ppo_scalar(
    tmp_path: Path,
    key: str,
    toml_value: str,
    expected: object,
) -> None:
    """ユーザー向けTOMLはPPOの10個のcommon scalarを個別に設定できる。"""

    path = _write_config(
        tmp_path,
        VALID_TOML.replace(
            "log_interval = 2",
            f"log_interval = 2\n{key} = {toml_value}",
        ),
    )

    assert load_experiment_config(path).profile.training_config[key] == expected


@pytest.mark.parametrize(
    ("key", "toml_value", "match"),
    [
        ("learning_rate", "0", "training.learning_rate"),
        ("batch_size", "0", "training.batch_size"),
        ("n_epochs", "0", "training.n_epochs"),
        ("gamma", "1.01", "training.gamma"),
        ("gae_lambda", "-0.01", "training.gae_lambda"),
        ("clip_range", "0", "training.clip_range"),
        ("normalize_advantage", "\"true\"", "training.normalize_advantage"),
        ("ent_coef", "-0.01", "training.ent_coef"),
        ("vf_coef", "-0.01", "training.vf_coef"),
        ("max_grad_norm", "-0.01", "training.max_grad_norm"),
    ],
)
def test_loader_rejects_invalid_ppo_scalar_ranges(
    tmp_path: Path,
    key: str,
    toml_value: str,
    match: str,
) -> None:
    """不正なPPO scalarをSB3 construction前にTOML loaderで止める。"""

    path = _write_config(
        tmp_path,
        VALID_TOML.replace(
            "log_interval = 2",
            f"log_interval = 2\n{key} = {toml_value}",
        ),
    )

    with pytest.raises(ExperimentConfigError, match=match):
        load_experiment_config(path)


def test_loader_checks_normalize_advantage_batch_and_rollout_sizes(
    tmp_path: Path,
) -> None:
    """normalize_advantageが有効ならSB3が拒否する1件batchを防ぐ。"""

    single_batch = _write_config(
        tmp_path,
        VALID_TOML.replace("log_interval = 2", "log_interval = 2\nbatch_size = 1"),
    )
    with pytest.raises(ExperimentConfigError, match="training.batch_size"):
        load_experiment_config(single_batch)

    single_rollout = _write_config(
        tmp_path,
        VALID_TOML.replace("num_envs = 2", "num_envs = 1").replace(
            "n_steps = 64", "n_steps = 1"
        ),
    )
    with pytest.raises(ExperimentConfigError, match=r"num_envs \* n_steps"):
        load_experiment_config(single_rollout)

    no_normalization = _write_config(
        tmp_path,
        VALID_TOML.replace("num_envs = 2", "num_envs = 1")
        .replace("n_steps = 64", "n_steps = 1")
        .replace("log_interval = 2", "log_interval = 2\nbatch_size = 1\nnormalize_advantage = false"),
    )
    assert load_experiment_config(no_normalization).profile.training_config[
        "normalize_advantage"
    ] is False


def test_loader_allows_an_empty_required_evaluation_table(tmp_path: Path) -> None:
    """[evaluation]自体は必須だが、各評価既定値は省略できる。"""

    populated_evaluation = (
        "[evaluation]\n"
        "model_path = \"models/custom_model.zip\"\n"
        "record_gif = false\n"
        "output_prefix = \"custom_evaluation\"\n"
        "seed = 12\n"
        "device = \"cpu\"\n"
        "log_file = \"logs/custom_evaluate.log\"\n"
        "deterministic = false\n\n"
    )
    path = _write_config(
        tmp_path,
        VALID_TOML.replace(populated_evaluation, "[evaluation]\n\n"),
    )

    selection = load_experiment_config(path)

    assert selection.profile.evaluation_defaults == {
        "model_path": "models/custom_training.zip",
        "record_gif": True,
        "output_prefix": "custom_training",
        "seed": 11,
        "device": "cpu",
        "deterministic": True,
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
        (
            "[evaluation]\n"
            "model_path = \"models/custom_model.zip\"\n"
            "record_gif = false\n"
            "output_prefix = \"custom_evaluation\"\n"
            "seed = 12\n"
            "device = \"cpu\"\n"
            "log_file = \"logs/custom_evaluate.log\"\n"
            "deterministic = false\n\n",
            "",
            "evaluation",
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
        ("log_interval = 2", "log_interval = 2\nunsupported_ppo_scalar = 0.001"),
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


def test_loader_rejects_schema_version_1(tmp_path: Path) -> None:
    """評価全件走査を導入した外部bundle schemaはv2だけを受け入れる。"""

    path = _write_config(
        tmp_path,
        VALID_TOML.replace("schema_version = 2", "schema_version = 1"),
    )

    with pytest.raises(ExperimentConfigError, match="versionは2だけです"):
        load_experiment_config(path)


def test_loader_rejects_removed_evaluation_count_key(tmp_path: Path) -> None:
    """v2では旧評価回数keyを黙って受理・無視しない。"""

    path = _write_config(
        tmp_path,
        VALID_TOML.replace(
            "[evaluation]\n",
            "[evaluation]\nepisodes = 2\n",
            1,
        ),
    )

    with pytest.raises(
        ExperimentConfigError,
        match="evaluation: 未対応のkeyがあります: episodes",
    ):
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


def test_evaluation_cli_rejects_removed_count_option() -> None:
    """評価件数は環境設定だけで決まり、旧CLI optionは受理しない。"""

    with pytest.raises(SystemExit):
        parse_evaluation_args(["--episodes", "1"])


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


def test_profile_alias_and_legacy_namespace_resolve_the_same_toml() -> None:
    """profile aliasも古いNamespace fallbackも実TOML sourceを返す。"""

    selection = select_experiment(profile_name="official")
    legacy_selection = experiment_selection_from_args(
        SimpleNamespace(profile="official")
    )

    assert selection.source_kind == "toml"
    assert selection.source_path == canonical_config_path("official").resolve()
    assert selection.source_sha256 == hashlib.sha256(
        selection.source_path.read_bytes()
    ).hexdigest()
    assert selection.profile.train_env_config == legacy_selection.profile.train_env_config
    assert selection.source_metadata()["profile"] == "official"


def test_cli_does_not_abbreviate_config_option() -> None:
    """二段parseを含め、誤った短縮optionを受理しない。"""

    with pytest.raises(SystemExit):
        parse_training_args(["--conf", "configs/example_experiment.toml"])
    with pytest.raises(SystemExit):
        parse_evaluation_args(["--conf", "configs/example_experiment.toml"])
