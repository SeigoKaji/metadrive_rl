"""TOML実験bundleの読み込み、検証、profile選択を共通化する。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import io
import math
from pathlib import Path, PureWindowsPath
import tomllib
from typing import Literal, TypeAlias

from .experiment_profiles import ExperimentProfile, get_experiment_profile
from .phase0_config import PROJECT_ROOT


ConfigSourceKind: TypeAlias = Literal["builtin_profile", "toml"]
PlainData: TypeAlias = (
    str | int | float | bool | list["PlainData"] | dict[str, "PlainData"]
)


class ExperimentConfigError(ValueError):
    """ユーザー指定TOMLが実験bundleとして有効でないことを表す。"""


@dataclass(frozen=True, slots=True)
class ExperimentSelection:
    """CLIが使う、設定sourceを含む解決済み実験bundle。"""

    name: str
    profile: ExperimentProfile
    source_kind: ConfigSourceKind
    source_path: Path | None
    source_sha256: str | None

    @property
    def profile_name(self) -> str:
        """成果物ディレクトリに使う互換性のためのprofile名。"""

        return self.name

    def source_metadata(self) -> dict[str, str | None]:
        """JSON metadataへそのまま記録できる設定source情報。"""

        return {
            "kind": self.source_kind,
            "name": self.name,
            "profile": self.name,
            "path": None if self.source_path is None else str(self.source_path),
            "sha256": self.source_sha256,
        }


_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "name",
        "algorithm",
        "default_model_name",
        "training",
        "evaluation",
        "environment",
    }
)
_TRAINING_REQUIRED_KEYS = frozenset(
    {
        "policy",
        "seed",
        "num_envs",
        "n_steps",
        "total_timesteps",
        "log_interval",
    }
)
_TRAINING_OPTIONAL_KEYS = frozenset({"device", "model_name", "log_file"})
_EVALUATION_REQUIRED_KEYS = frozenset({"episodes"})
_EVALUATION_OPTIONAL_KEYS = frozenset(
    {
        "model_path",
        "record_gif",
        "output_prefix",
        "seed",
        "device",
        "log_file",
        "deterministic",
    }
)
_ENVIRONMENT_KEYS = frozenset({"common", "train", "evaluation"})
_MAX_RL_SEED = 2**32 - 1


def _error(location: str, message: str) -> ExperimentConfigError:
    return ExperimentConfigError(f"{location}: {message}")


def _table(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise _error(location, "TOML tableで指定してください")
    if not all(isinstance(key, str) for key in value):
        raise _error(location, "table keyは文字列で指定してください")
    return value


def _required_table(
    root: Mapping[str, object],
    key: str,
    location: str,
) -> dict[str, object]:
    if key not in root:
        raise _error(location, "必須tableがありません")
    return _table(root[key], location)


def _reject_unknown_keys(
    table: Mapping[str, object],
    *,
    allowed: frozenset[str],
    location: str,
) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise _error(location, f"未対応のkeyがあります: {', '.join(unknown)}")


def _require_string(value: object, location: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str):
        raise _error(location, "文字列で指定してください")
    if nonempty and not value.strip():
        raise _error(location, "空でない文字列で指定してください")
    if "\x00" in value:
        raise _error(location, "NUL文字は指定できません")
    return value


def _require_int(
    value: object,
    location: str,
    *,
    positive: bool = False,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _error(location, "boolではない整数で指定してください")
    if positive and value <= 0:
        raise _error(location, "0より大きい整数で指定してください")
    return value


def _require_bool(value: object, location: str) -> bool:
    if not isinstance(value, bool):
        raise _error(location, "boolで指定してください")
    return value


def _require_rl_seed(value: object, location: str) -> int:
    """Stable-Baselines3/NumPyへ安全に渡せるseed範囲を検証する。"""

    seed = _require_int(value, location)
    if not 0 <= seed <= _MAX_RL_SEED:
        raise _error(
            location,
            f"0以上{_MAX_RL_SEED}以下の整数を指定してください",
        )
    return seed


def _safe_basename(value: object, location: str) -> str:
    """出力配下のbasenameに使える名前だけを受け入れる。"""

    name = _require_string(value, location)
    if (
        name in {".", ".."}
        or "/" in name
        or "\\" in name
        or Path(name).name != name
        or PureWindowsPath(name).name != name
    ):
        raise _error(location, "ディレクトリを含まないbasenameを指定してください")
    return name


def normalize_model_name(value: object, location: str) -> str:
    """model basenameを検査し、任意の ``.zip`` suffixを除く。"""

    name = _safe_basename(value, location)
    stem = name[:-4] if name.endswith(".zip") else name
    if stem in {"", ".", ".."}:
        raise _error(location, "有効なmodel名を指定してください")
    return stem


def _path_string(value: object, location: str) -> str:
    return _require_string(value, location)


def _plain_data(value: object, location: str) -> PlainData:
    """MetaDriveへ渡せるTOML由来のプレーンデータだけを残す。"""

    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _error(location, "有限のfloatで指定してください")
        return value
    if isinstance(value, str):
        if "\x00" in value:
            raise _error(location, "NUL文字は指定できません")
        return value
    if isinstance(value, list):
        return [
            _plain_data(item, f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        result: dict[str, PlainData] = {}
        for key, nested_value in value.items():
            if not isinstance(key, str):
                raise _error(location, "table keyは文字列で指定してください")
            result[key] = _plain_data(nested_value, f"{location}.{key}")
        return result
    raise _error(
        location,
        "bool/int/有限float/str/list/tableだけを指定してください",
    )


def _plain_table(value: object, location: str) -> dict[str, PlainData]:
    plain = _plain_data(_table(value, location), location)
    if not isinstance(plain, dict):  # ``_table`` succeeded, for type checkers.
        raise AssertionError("a TOML table must remain a dict")
    return plain


def _clone_plain(value: PlainData) -> PlainData:
    if isinstance(value, list):
        return [_clone_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_plain(item) for key, item in value.items()}
    return value


def _deep_merge(
    common: Mapping[str, PlainData],
    stage: Mapping[str, PlainData],
) -> dict[str, PlainData]:
    """Nested MetaDrive tablesを保ったままstage側を優先してmergeする。"""

    merged = {key: _clone_plain(value) for key, value in common.items()}
    for key, stage_value in stage.items():
        common_value = merged.get(key)
        if isinstance(common_value, dict) and isinstance(stage_value, dict):
            merged[key] = _deep_merge(common_value, stage_value)
        else:
            merged[key] = _clone_plain(stage_value)
    return merged


def _validate_environment(
    value: Mapping[str, PlainData],
    location: str,
) -> dict[str, object]:
    start_seed = _require_int(value.get("start_seed"), f"{location}.start_seed")
    num_scenarios = _require_int(
        value.get("num_scenarios"),
        f"{location}.num_scenarios",
        positive=True,
    )
    validated = {key: _clone_plain(item) for key, item in value.items()}
    # The values above are intentionally assigned back as ordinary ``int`` for
    # callers that inspect only the required scenario controls.
    validated["start_seed"] = start_seed
    validated["num_scenarios"] = num_scenarios
    return validated


def _validate_evaluation_action_compatibility(
    environment: Mapping[str, object],
    location: str,
) -> None:
    """現行PPO評価loopで確実に扱えるAction設定だけを許可する。"""

    discrete_action = environment.get("discrete_action")
    if not isinstance(discrete_action, bool):
        raise _error(f"{location}.discrete_action", "bool trueで指定してください")
    if not discrete_action:
        raise _error(
            f"{location}.discrete_action",
            "現行の評価経路はdiscrete_action=trueだけをサポートします",
        )

    for key in ("use_multi_discrete", "is_multi_agent"):
        if key not in environment:
            continue
        value = environment[key]
        if not isinstance(value, bool):
            raise _error(f"{location}.{key}", "boolで指定してください")
        if value:
            raise _error(
                f"{location}.{key}",
                f"現行の評価経路は{key}=falseだけをサポートします",
            )

    if "num_agents" in environment:
        num_agents = _require_int(environment["num_agents"], f"{location}.num_agents")
        if num_agents != 1:
            raise _error(
                f"{location}.num_agents",
                "現行のsingle-agent評価では1だけを指定してください",
            )

    for key in ("discrete_steering_dim", "discrete_throttle_dim"):
        if key not in environment:
            continue
        dimension = _require_int(environment[key], f"{location}.{key}")
        if dimension < 2:
            raise _error(
                f"{location}.{key}",
                "現行の離散Actionでは2以上の整数を指定してください",
            )


def _validate_training(value: object) -> dict[str, object]:
    table = _table(value, "training")
    _reject_unknown_keys(
        table,
        allowed=_TRAINING_REQUIRED_KEYS | _TRAINING_OPTIONAL_KEYS,
        location="training",
    )
    missing = sorted(_TRAINING_REQUIRED_KEYS - set(table))
    if missing:
        raise _error("training", f"必須keyがありません: {', '.join(missing)}")

    training: dict[str, object] = {
        "policy": _require_string(table["policy"], "training.policy"),
        "seed": _require_rl_seed(table["seed"], "training.seed"),
        "num_envs": _require_int(
            table["num_envs"], "training.num_envs", positive=True
        ),
        "n_steps": _require_int(table["n_steps"], "training.n_steps", positive=True),
        "total_timesteps": _require_int(
            table["total_timesteps"],
            "training.total_timesteps",
            positive=True,
        ),
        "log_interval": _require_int(
            table["log_interval"], "training.log_interval", positive=True
        ),
    }
    if "device" in table:
        training["device"] = _require_string(table["device"], "training.device")
    if "model_name" in table:
        training["model_name"] = normalize_model_name(
            table["model_name"], "training.model_name"
        )
    if "log_file" in table:
        training["log_file"] = _path_string(table["log_file"], "training.log_file")
    return training


def _validate_evaluation(
    value: object,
    *,
    default_training_model_name: str,
    training_seed: int,
) -> dict[str, object]:
    table = _table(value, "evaluation")
    _reject_unknown_keys(
        table,
        allowed=_EVALUATION_REQUIRED_KEYS | _EVALUATION_OPTIONAL_KEYS,
        location="evaluation",
    )
    missing = sorted(_EVALUATION_REQUIRED_KEYS - set(table))
    if missing:
        raise _error("evaluation", f"必須keyがありません: {', '.join(missing)}")

    evaluation: dict[str, object] = {
        "episodes": _require_int(table["episodes"], "evaluation.episodes", positive=True),
        "model_path": _path_string(
            table.get("model_path", f"models/{default_training_model_name}.zip"),
            "evaluation.model_path",
        ),
        "record_gif": _require_bool(
            table.get("record_gif", True), "evaluation.record_gif"
        ),
        "output_prefix": _safe_basename(
            table.get("output_prefix", default_training_model_name),
            "evaluation.output_prefix",
        ),
        "seed": _require_rl_seed(
            table.get("seed", training_seed), "evaluation.seed"
        ),
        "device": _require_string(table.get("device", "cpu"), "evaluation.device"),
        "deterministic": _require_bool(
            table.get("deterministic", True), "evaluation.deterministic"
        ),
    }
    if "log_file" in table:
        evaluation["log_file"] = _path_string(
            table["log_file"], "evaluation.log_file"
        )
    return evaluation


def _profile_from_toml(raw: object) -> tuple[str, ExperimentProfile]:
    root = _table(raw, "root")
    _reject_unknown_keys(root, allowed=_ROOT_KEYS, location="root")

    schema_version = _require_int(root.get("schema_version"), "schema_version")
    if schema_version != 1:
        raise _error("schema_version", "対応しているversionは1だけです")
    name = _safe_basename(root.get("name"), "name")
    algorithm = _require_string(root.get("algorithm"), "algorithm")
    if algorithm != "ppo":
        raise _error("algorithm", "現在対応しているalgorithmはppoだけです")

    default_model_name = normalize_model_name(
        root.get("default_model_name", name), "default_model_name"
    )
    training = _validate_training(_required_table(root, "training", "training"))
    if "model_name" not in training:
        training["model_name"] = default_model_name
    rollout_batch_size = int(training["num_envs"]) * int(training["n_steps"])
    if rollout_batch_size <= 1:
        raise _error(
            "training",
            "PPOの既定normalize_advantageではnum_envs * n_stepsを2以上にしてください",
        )

    evaluation = _validate_evaluation(
        _required_table(root, "evaluation", "evaluation"),
        default_training_model_name=str(training["model_name"]),
        training_seed=int(training["seed"]),
    )

    environment = _required_table(root, "environment", "environment")
    _reject_unknown_keys(
        environment,
        allowed=_ENVIRONMENT_KEYS,
        location="environment",
    )
    common = _plain_table(environment.get("common", {}), "environment.common")
    train = _plain_table(
        _required_table(environment, "train", "environment.train"),
        "environment.train",
    )
    evaluation_environment = _plain_table(
        _required_table(environment, "evaluation", "environment.evaluation"),
        "environment.evaluation",
    )
    train_env_config = _validate_environment(
        _deep_merge(common, train), "environment.train"
    )
    evaluation_env_config = _validate_environment(
        _deep_merge(common, evaluation_environment), "environment.evaluation"
    )
    _validate_evaluation_action_compatibility(train_env_config, "environment.train")
    _validate_evaluation_action_compatibility(
        evaluation_env_config,
        "environment.evaluation",
    )
    evaluation_scenario_count = int(evaluation_env_config["num_scenarios"])
    if (
        evaluation_scenario_count > 1
        and int(evaluation["episodes"]) > evaluation_scenario_count
    ):
        raise _error(
            "evaluation.episodes",
            "評価scenario数が複数の場合はnum_scenarios以下にしてください: "
            f"episodes={evaluation['episodes']}, "
            f"num_scenarios={evaluation_scenario_count}",
        )

    return name, ExperimentProfile(
        train_env_config=train_env_config,
        evaluation_env_config=evaluation_env_config,
        training_config=training,
        default_model_name=default_model_name,
        evaluation_episodes=int(evaluation["episodes"]),
        evaluation_defaults=evaluation,
    )


def _resolve_config_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ExperimentConfigError(
            f"config fileを解決できません: {candidate}: {error}"
        ) from error
    if resolved.suffix != ".toml":
        raise ExperimentConfigError(
            f"config fileは.tomlを指定してください: {resolved}"
        )
    if not resolved.is_file():
        raise ExperimentConfigError(f"config fileが見つかりません: {resolved}")
    return resolved


def load_experiment_config(path: str | Path) -> ExperimentSelection:
    """TOML実験bundleを読み込み、完全に検証したselectionを返す。"""

    source_path = _resolve_config_path(path)
    try:
        source_bytes = source_path.read_bytes()
    except OSError as error:
        raise ExperimentConfigError(
            f"config fileを読めません: {source_path}: {error}"
        ) from error
    try:
        raw = tomllib.load(io.BytesIO(source_bytes))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
        raise ExperimentConfigError(
            f"TOMLの構文が不正です: {source_path}: {error}"
        ) from error

    name, profile = _profile_from_toml(raw)
    return ExperimentSelection(
        name=name,
        profile=profile,
        source_kind="toml",
        source_path=source_path,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
    )


def select_experiment(
    *,
    profile_name: str | None = None,
    config_path: str | Path | None = None,
) -> ExperimentSelection:
    """組み込みprofileまたは外部TOMLを同じ形へ解決する。"""

    if profile_name is not None and config_path is not None:
        raise ExperimentConfigError("--profileと--configは同時に指定できません")
    if config_path is not None:
        return load_experiment_config(config_path)

    name = "official" if profile_name is None else profile_name
    return ExperimentSelection(
        name=name,
        profile=get_experiment_profile(name),
        source_kind="builtin_profile",
        source_path=None,
        source_sha256=None,
    )


def experiment_selection_from_args(args: object) -> ExperimentSelection:
    """parse済みargsを優先し、旧テスト用Namespaceではprofileから復元する。"""

    for attribute in ("experiment", "experiment_selection"):
        selected = getattr(args, attribute, None)
        if isinstance(selected, ExperimentSelection):
            return selected

    config_path = getattr(args, "config", None)
    if config_path is not None:
        return select_experiment(config_path=config_path)
    return select_experiment(profile_name=getattr(args, "profile", "official"))
