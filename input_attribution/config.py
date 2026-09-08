"""Strict reader for input-attribution analysis TOML files.

The experiment TOML and the attribution TOML deliberately have separate
schemas.  Keeping this module independent makes an analysis bundle portable
without changing the training/evaluation configuration parser.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import io
import math
from pathlib import Path
import tomllib
from typing import Final, Literal, TypeAlias


BaselineStrategy: TypeAlias = Literal[
    "episode_start",
    "specified_steps",
    "sampled_observations",
    "external_npz",
]
IGTarget: TypeAlias = Literal[
    "selected_log_probability",
    "selected_vs_runner_up_margin",
    "critic_value",
]
ReplacementStrategy: TypeAlias = Literal[
    "episode_start_constant",
    "specified_reference_constant",
    "dataset_median_constant",
    "schema_constant",
]


class AnalysisConfigError(ValueError):
    """Raised when an attribution analysis TOML is invalid."""


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Settings which affect policy execution but not model weights."""

    deterministic: bool
    device: str
    record_visualization: bool


@dataclass(frozen=True, slots=True)
class CollectionConfig:
    """Settings for the unperturbed rollout collector."""

    save_observations: bool


@dataclass(frozen=True, slots=True)
class BaselineReference:
    """One observation selected by its rollout episode and decision step."""

    episode_id: int
    step: int


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    """How reference observations for replacement/IG are selected."""

    strategy: BaselineStrategy
    count: int
    seed: int
    specified_steps: tuple[BaselineReference, ...] = ()
    external_npz_path: Path | None = None
    external_npz_key: str | None = None


@dataclass(frozen=True, slots=True)
class PerturbationConfig:
    """Settings for offline baseline-replacement perturbations."""

    enabled: bool
    analyze_features: bool
    analyze_groups: bool
    batch_size: int
    lidar_sector_degrees: float


@dataclass(frozen=True, slots=True)
class IntegratedGradientsConfig:
    """Settings for the autograd Integrated Gradients calculation."""

    enabled: bool
    targets: tuple[IGTarget, ...]
    steps: int
    batch_size: int


FeatureSelector: TypeAlias = str | int


@dataclass(frozen=True, slots=True)
class ClosedLoopConfig:
    """Settings for optional counterfactual closed-loop validation."""

    enabled: bool
    top_k_features: int
    top_k_groups: int
    replacement_strategy: ReplacementStrategy
    explicit_features: tuple[FeatureSelector, ...] = ()
    explicit_groups: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """Fully validated attribution configuration with source provenance."""

    schema_version: int
    name: str
    run: RunConfig
    collection: CollectionConfig
    baseline: BaselineConfig
    perturbation: PerturbationConfig
    integrated_gradients: IntegratedGradientsConfig
    closed_loop: ClosedLoopConfig
    source_path: Path | None = None
    source_sha256: str | None = None


_ROOT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "name",
        "run",
        "collection",
        "baseline",
        "perturbation",
        "integrated_gradients",
        "closed_loop",
    }
)
_RUN_KEYS: Final[frozenset[str]] = frozenset(
    {"deterministic", "device", "record_visualization"}
)
_COLLECTION_KEYS: Final[frozenset[str]] = frozenset({"save_observations"})
_BASELINE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "strategy",
        "count",
        "seed",
        "specified_steps",
        "path",
        "key",
        "external_npz",
        "external_npz_path",
        "external_npz_key",
    }
)
_PERTURBATION_KEYS: Final[frozenset[str]] = frozenset(
    {
        "enabled",
        "analyze_features",
        "analyze_groups",
        "batch_size",
        "lidar_sector_degrees",
    }
)
_IG_KEYS: Final[frozenset[str]] = frozenset(
    {"enabled", "targets", "steps", "batch_size"}
)
_CLOSED_LOOP_KEYS: Final[frozenset[str]] = frozenset(
    {
        "enabled",
        "top_k_features",
        "top_k_groups",
        "replacement_strategy",
        "explicit_features",
        "explicit_groups",
        # Short aliases are intentionally accepted because they map to the
        # same unambiguous canonical fields above.
        "features",
        "groups",
    }
)
_BASELINE_STRATEGIES: Final[frozenset[str]] = frozenset(
    {"episode_start", "specified_steps", "sampled_observations", "external_npz"}
)
_IG_TARGETS: Final[frozenset[str]] = frozenset(
    {
        "selected_log_probability",
        "selected_vs_runner_up_margin",
        "critic_value",
    }
)
_REPLACEMENT_STRATEGIES: Final[frozenset[str]] = frozenset(
    {
        "episode_start_constant",
        "specified_reference_constant",
        "dataset_median_constant",
        "schema_constant",
    }
)


def _error(location: str, message: str) -> AnalysisConfigError:
    return AnalysisConfigError(f"{location}: {message}")


def _table(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise _error(location, "TOML tableで指定してください")
    return value


def _required_table(root: Mapping[str, object], key: str) -> dict[str, object]:
    if key not in root:
        raise _error(key, "必須tableがありません")
    return _table(root[key], key)


def _reject_unknown(
    table: Mapping[str, object], allowed: frozenset[str], location: str
) -> None:
    unknown = sorted(set(table).difference(allowed))
    if unknown:
        raise _error(location, f"未対応のkeyがあります: {', '.join(unknown)}")


def _require_key(table: Mapping[str, object], key: str, location: str) -> object:
    if key not in table:
        raise _error(location, f"必須key '{key}' がありません")
    return table[key]


def _require_bool(value: object, location: str) -> bool:
    if not isinstance(value, bool):
        raise _error(location, "boolで指定してください")
    return value


def _require_string(value: object, location: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str):
        raise _error(location, "文字列で指定してください")
    if nonempty and not value.strip():
        raise _error(location, "空でない文字列で指定してください")
    if "\x00" in value:
        raise _error(location, "NUL文字は指定できません")
    return value


def _require_int(value: object, location: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _error(location, "boolではない整数で指定してください")
    if minimum is not None and value < minimum:
        raise _error(location, f"{minimum} 以上の整数で指定してください")
    return value


def _require_real(
    value: object,
    location: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error(location, "boolではない有限の数値で指定してください")
    result = float(value)
    if not math.isfinite(result):
        raise _error(location, "有限の数値で指定してください")
    if minimum is not None and (result <= minimum if strict_minimum else result < minimum):
        comparator = "より大きい" if strict_minimum else "以上"
        raise _error(location, f"{minimum} {comparator}の数値で指定してください")
    if maximum is not None and result > maximum:
        raise _error(location, f"{maximum} 以下の数値で指定してください")
    return result


def _require_list(value: object, location: str) -> list[object]:
    if not isinstance(value, list):
        raise _error(location, "arrayで指定してください")
    return value


def _parse_run(table: Mapping[str, object]) -> RunConfig:
    _reject_unknown(table, _RUN_KEYS, "run")
    record_visualization = _require_bool(
        _require_key(table, "record_visualization", "run"),
        "run.record_visualization",
    )
    if record_visualization:
        raise _error(
            "run.record_visualization",
            "初版ではsimulator recordingをサポートしません。falseを指定してください（解析plotは常に保存します）",
        )
    return RunConfig(
        deterministic=_require_bool(_require_key(table, "deterministic", "run"), "run.deterministic"),
        device=_require_string(_require_key(table, "device", "run"), "run.device"),
        record_visualization=record_visualization,
    )


def _parse_collection(table: Mapping[str, object]) -> CollectionConfig:
    _reject_unknown(table, _COLLECTION_KEYS, "collection")
    save_observations = _require_bool(
        _require_key(table, "save_observations", "collection"),
        "collection.save_observations",
    )
    if not save_observations:
        raise _error(
            "collection.save_observations",
            "offline perturbation/IG再解析に必要なため初版ではtrueを指定してください",
        )
    return CollectionConfig(save_observations=save_observations)


def _parse_baseline_references(value: object) -> tuple[BaselineReference, ...]:
    references: list[BaselineReference] = []
    for number, item in enumerate(_require_list(value, "baseline.specified_steps")):
        location = f"baseline.specified_steps[{number}]"
        entry = _table(item, location)
        _reject_unknown(entry, frozenset({"episode_id", "episode", "step"}), location)
        if "episode_id" in entry and "episode" in entry:
            raise _error(location, "episode_idとepisodeを同時に指定できません")
        episode_key = "episode_id" if "episode_id" in entry else "episode"
        if episode_key not in entry:
            raise _error(location, "episode_id（またはepisode）がありません")
        references.append(
            BaselineReference(
                episode_id=_require_int(entry[episode_key], f"{location}.{episode_key}", minimum=0),
                step=_require_int(_require_key(entry, "step", location), f"{location}.step", minimum=0),
            )
        )
    if not references:
        raise _error("baseline.specified_steps", "少なくとも1件指定してください")
    if len({(item.episode_id, item.step) for item in references}) != len(references):
        raise _error("baseline.specified_steps", "同じepisode_id/stepを重複指定できません")
    return tuple(references)


def _parse_external_npz(
    table: Mapping[str, object], source_path: Path
) -> tuple[Path | None, str | None]:
    """Read canonical keys and one nested TOML form without ambiguity."""

    path_value: object | None = None
    key_value: object | None = None
    used_path_keys = [key for key in ("path", "external_npz_path") if key in table]
    used_key_keys = [key for key in ("key", "external_npz_key") if key in table]
    if len(used_path_keys) > 1 or len(used_key_keys) > 1:
        raise _error("baseline", "external NPZのpath/keyを重複指定できません")
    if used_path_keys:
        path_value = table[used_path_keys[0]]
    if used_key_keys:
        key_value = table[used_key_keys[0]]

    if "external_npz" in table:
        nested = table["external_npz"]
        if isinstance(nested, str):
            if path_value is not None:
                raise _error("baseline", "external_npzのpathを重複指定できません")
            path_value = nested
        else:
            nested_table = _table(nested, "baseline.external_npz")
            _reject_unknown(nested_table, frozenset({"path", "key"}), "baseline.external_npz")
            if path_value is not None or key_value is not None:
                raise _error("baseline", "external_npzとpath/keyを同時に指定できません")
            path_value = _require_key(nested_table, "path", "baseline.external_npz")
            key_value = nested_table.get("key")

    if path_value is None:
        return None, None
    text_path = _require_string(path_value, "baseline.external_npz.path")
    path = Path(text_path)
    if not path.is_absolute():
        path = (source_path.parent / path).resolve()
    key = "observations" if key_value is None else _require_string(
        key_value, "baseline.external_npz.key"
    )
    return path, key


def _parse_baseline(table: Mapping[str, object], source_path: Path) -> BaselineConfig:
    _reject_unknown(table, _BASELINE_KEYS, "baseline")
    strategy = _require_string(_require_key(table, "strategy", "baseline"), "baseline.strategy")
    if strategy not in _BASELINE_STRATEGIES:
        expected = ", ".join(sorted(_BASELINE_STRATEGIES))
        raise _error("baseline.strategy", f"{expected} のいずれかで指定してください")
    count = _require_int(_require_key(table, "count", "baseline"), "baseline.count", minimum=1)
    seed = _require_int(_require_key(table, "seed", "baseline"), "baseline.seed", minimum=0)
    specified_steps = (
        _parse_baseline_references(table["specified_steps"])
        if "specified_steps" in table
        else ()
    )
    external_path, external_key = _parse_external_npz(table, source_path)

    if strategy == "specified_steps":
        if not specified_steps:
            raise _error("baseline", "strategy=specified_stepsにはspecified_stepsが必要です")
        if external_path is not None:
            raise _error("baseline", "specified_stepsとexternal_npzを同時に指定できません")
    elif strategy == "external_npz":
        if external_path is None:
            raise _error("baseline", "strategy=external_npzにはpathとkeyが必要です")
        if specified_steps:
            raise _error("baseline", "external_npzとspecified_stepsを同時に指定できません")
    elif specified_steps or external_path is not None:
        raise _error("baseline", "このstrategyではspecified_steps/external_npzを指定できません")

    return BaselineConfig(
        strategy=strategy,  # type: ignore[arg-type]
        count=count,
        seed=seed,
        specified_steps=specified_steps,
        external_npz_path=external_path,
        external_npz_key=external_key,
    )


def _parse_perturbation(table: Mapping[str, object]) -> PerturbationConfig:
    _reject_unknown(table, _PERTURBATION_KEYS, "perturbation")
    return PerturbationConfig(
        enabled=_require_bool(_require_key(table, "enabled", "perturbation"), "perturbation.enabled"),
        analyze_features=_require_bool(
            _require_key(table, "analyze_features", "perturbation"),
            "perturbation.analyze_features",
        ),
        analyze_groups=_require_bool(
            _require_key(table, "analyze_groups", "perturbation"),
            "perturbation.analyze_groups",
        ),
        batch_size=_require_int(
            _require_key(table, "batch_size", "perturbation"),
            "perturbation.batch_size",
            minimum=1,
        ),
        lidar_sector_degrees=_require_real(
            _require_key(table, "lidar_sector_degrees", "perturbation"),
            "perturbation.lidar_sector_degrees",
            minimum=0.0,
            maximum=360.0,
            strict_minimum=True,
        ),
    )


def _parse_ig(table: Mapping[str, object]) -> IntegratedGradientsConfig:
    _reject_unknown(table, _IG_KEYS, "integrated_gradients")
    targets: list[IGTarget] = []
    for index, value in enumerate(
        _require_list(_require_key(table, "targets", "integrated_gradients"), "integrated_gradients.targets")
    ):
        target = _require_string(value, f"integrated_gradients.targets[{index}]")
        if target not in _IG_TARGETS:
            raise _error(f"integrated_gradients.targets[{index}]", "未対応のtargetです")
        targets.append(target)  # type: ignore[arg-type]
    if not targets:
        raise _error("integrated_gradients.targets", "少なくとも1件指定してください")
    if len(set(targets)) != len(targets):
        raise _error("integrated_gradients.targets", "同じtargetを重複指定できません")
    return IntegratedGradientsConfig(
        enabled=_require_bool(
            _require_key(table, "enabled", "integrated_gradients"),
            "integrated_gradients.enabled",
        ),
        targets=tuple(targets),
        steps=_require_int(
            _require_key(table, "steps", "integrated_gradients"),
            "integrated_gradients.steps",
            minimum=2,
        ),
        batch_size=_require_int(
            _require_key(table, "batch_size", "integrated_gradients"),
            "integrated_gradients.batch_size",
            minimum=1,
        ),
    )


def _parse_feature_selectors(value: object, location: str) -> tuple[FeatureSelector, ...]:
    selectors: list[FeatureSelector] = []
    for index, item in enumerate(_require_list(value, location)):
        if isinstance(item, str):
            selectors.append(_require_string(item, f"{location}[{index}]"))
        elif not isinstance(item, bool) and isinstance(item, int) and item >= 0:
            selectors.append(item)
        else:
            raise _error(f"{location}[{index}]", "非負整数またはfeature名で指定してください")
    if len(set(selectors)) != len(selectors):
        raise _error(location, "同じfeatureを重複指定できません")
    return tuple(selectors)


def _parse_closed_loop(table: Mapping[str, object]) -> ClosedLoopConfig:
    _reject_unknown(table, _CLOSED_LOOP_KEYS, "closed_loop")
    for canonical, alias in (("explicit_features", "features"), ("explicit_groups", "groups")):
        if canonical in table and alias in table:
            raise _error("closed_loop", f"{canonical}と{alias}を同時に指定できません")
    strategy = _require_string(
        _require_key(table, "replacement_strategy", "closed_loop"),
        "closed_loop.replacement_strategy",
    )
    if strategy not in _REPLACEMENT_STRATEGIES:
        raise _error("closed_loop.replacement_strategy", "未対応のreplacement_strategyです")
    feature_value = table.get("explicit_features", table.get("features", []))
    group_value = table.get("explicit_groups", table.get("groups", []))
    explicit_features = _parse_feature_selectors(feature_value, "closed_loop.explicit_features")
    explicit_groups = tuple(
        _require_string(item, f"closed_loop.explicit_groups[{index}]")
        for index, item in enumerate(_require_list(group_value, "closed_loop.explicit_groups"))
    )
    if len(set(explicit_groups)) != len(explicit_groups):
        raise _error("closed_loop.explicit_groups", "同じgroupを重複指定できません")
    return ClosedLoopConfig(
        enabled=_require_bool(_require_key(table, "enabled", "closed_loop"), "closed_loop.enabled"),
        top_k_features=_require_int(
            _require_key(table, "top_k_features", "closed_loop"),
            "closed_loop.top_k_features",
            minimum=0,
        ),
        top_k_groups=_require_int(
            _require_key(table, "top_k_groups", "closed_loop"),
            "closed_loop.top_k_groups",
            minimum=0,
        ),
        replacement_strategy=strategy,  # type: ignore[arg-type]
        explicit_features=explicit_features,
        explicit_groups=explicit_groups,
    )


def load_analysis_config(path: str | Path) -> AnalysisConfig:
    """Load a strict, self-contained attribution analysis TOML.

    Relative external baseline paths are resolved against this TOML's parent.
    This function only parses configuration; it intentionally does not read a
    model, environment, or external NPZ data.
    """

    source_path = Path(path).expanduser().resolve()
    if source_path.suffix.lower() != ".toml":
        raise AnalysisConfigError("analysis config: .tomlファイルを指定してください")
    try:
        source = source_path.read_bytes()
    except OSError as error:
        raise AnalysisConfigError(f"analysis configを読めません: {source_path}") from error
    try:
        raw = tomllib.load(io.BytesIO(source))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
        raise AnalysisConfigError(f"analysis configのTOML形式が不正です: {source_path}") from error
    root = _table(raw, "root")
    _reject_unknown(root, _ROOT_KEYS, "root")
    schema_version = _require_int(
        _require_key(root, "schema_version", "root"), "schema_version", minimum=1
    )
    if schema_version != 1:
        raise _error("schema_version", "現在対応しているversionは1です")
    name = _require_string(_require_key(root, "name", "root"), "name")
    return AnalysisConfig(
        schema_version=schema_version,
        name=name,
        run=_parse_run(_required_table(root, "run")),
        collection=_parse_collection(_required_table(root, "collection")),
        baseline=_parse_baseline(_required_table(root, "baseline"), source_path),
        perturbation=_parse_perturbation(_required_table(root, "perturbation")),
        integrated_gradients=_parse_ig(_required_table(root, "integrated_gradients")),
        closed_loop=_parse_closed_loop(_required_table(root, "closed_loop")),
        source_path=source_path,
        source_sha256=hashlib.sha256(source).hexdigest(),
    )
