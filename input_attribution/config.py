"""解析用TOMLの読み込みと厳密な既定値解決。

このモジュールは学習用 ``configs/*.toml`` とは独立した解析bundleだけを
扱う。MetaDriveやStable-Baselines3をimportしないため、環境のないPCでも
設定検査・保存結果の再解析に利用できる。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import tomllib
from typing import Any


class ConfigError(ValueError):
    """解析設定が不正、未対応、または未解決である。"""


_TOP_LEVEL_KEYS = frozenset(
    {
        "analysis",
        "adapter",
        "closed_loop",
        "environment",
        "ig",
        "model",
        "output",
        "patterns",
        "preprocess",
        "scenario",
        "schema",
        "seeds",
        "video",
    }
)
_SECTION_KEYS: dict[str, frozenset[str]] = {
    "analysis": frozenset({"name", "experiment", "description", "deterministic", "individual_inputs", "reference_id"}),
    "model": frozenset({"path", "name", "format", "device", "deterministic"}),
    "adapter": frozenset({"module", "factory", "class", "path", "config"}),
    "schema": frozenset({"path", "name", "dimension", "template"}),
    "environment": frozenset({"config", "path", "module", "factory", "settings"}),
    "scenario": frozenset(
        {"map", "seed", "scenario_seed", "rl_seed", "horizon", "traffic_density", "episodes"}
    ),
    "seeds": frozenset({"scenario", "scenario_seed", "rl", "rl_seed", "intervention"}),
    "preprocess": frozenset(
        {"name", "module", "factory", "stats", "stats_path", "steps", "external_normalization"}
    ),
    "output": frozenset({"root", "experiment", "model", "run_id", "overwrite"}),
    "closed_loop": frozenset({"patterns", "episodes", "max_steps", "enabled", "seed", "departure_tolerance_ratio", "departure_consecutive_steps", "low_speed_m_s"}),
    "video": frozenset({"enabled", "format", "fps", "directory", "patterns"}),
    "ig": frozenset({"enabled", "steps", "baseline", "n_steps", "method", "completeness_tolerance", "max_retries", "compatibility_keys"}),
}

_PATTERN_METHODS = frozenset(
    {"identity", "fixed", "reference", "neutral", "reflection", "fixed_level", "fixed-value", "fixed_value"}
)
_PATTERN_SCOPES = frozenset({"full_episode", "explicitly_conditional"})
_PATTERN_INAPPLICABLE = frozenset({"abort_pattern", "continue_unmodified_with_warning"})


def _mapping(value: object, location: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{location} はtableで指定してください")
    return dict(value)


def _plain(value: object, location: str = "config") -> Any:
    """JSON/TOMLに保存可能な値へコピーする。

    arbitrary Python objectを設定へ混ぜるとmanifestのhashが実行ごとに変わる
    ため、ここで明示的に拒否する。
    """

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _plain(v, f"{location}.{k}") for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v, f"{location}[]") for v in value]
    raise ConfigError(f"{location} に未対応の値型があります: {type(value).__name__}")


def _basename(value: str, location: str) -> str:
    value = str(value)
    if not value.strip() or value in {".", ".."} or "/" in value or "\\" in value:
        raise ConfigError(f"{location} は単純な名前で指定してください")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise ConfigError(f"{location} に使用できない文字があります: {value!r}")
    return value


def _relative_path(value: object, base: Path, location: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{location} は空でないパス文字列で指定してください")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    return str(path)


def _section(raw: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    section = _mapping(value, name)
    allowed = _SECTION_KEYS[name]
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ConfigError(f"{name}: 未対応のkeyがあります: {', '.join(unknown)}")
    return section


def _check_patterns(raw: object) -> list[dict[str, Any]]:
    if raw is None:
        return [
            {
                "id": "P00",
                "description": "変更なし",
                "indices": [],
                "method": "identity",
            }
        ]
    if not isinstance(raw, list):
        raise ConfigError("patterns はarray of tablesで指定してください")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        pattern = _mapping(item, f"patterns[{index}]")
        if "id" not in pattern:
            raise ConfigError(f"patterns[{index}].id がありません")
        identifier = _basename(str(pattern["id"]), f"patterns[{index}].id")
        if identifier in seen:
            raise ConfigError(f"patterns に重複idがあります: {identifier}")
        seen.add(identifier)
        # Intervention details are validated again by schema/interventions;
        # retain their fields here so new input groups can travel through the
        # CLI while checking the execution-scope contract at config load time.
        method = pattern.get("method", pattern.get("operation", pattern.get("kind", "fixed")))
        if not isinstance(method, str) or method.strip().casefold() not in _PATTERN_METHODS:
            raise ConfigError(
                f"patterns[{index}].method はidentity/fixed/reference/neutral/reflection/fixed_levelのいずれかで指定してください"
            )
        if "indices" in pattern:
            indices = pattern["indices"]
            if not isinstance(indices, list) or any(
                isinstance(v, bool) or not isinstance(v, (int, str)) or (isinstance(v, int) and v < 0) or (isinstance(v, str) and not v.strip()) for v in indices
            ):
                raise ConfigError(f"patterns[{index}].indices は0以上の整数または意味IDのarrayで指定してください")
        pattern = _plain(pattern, f"patterns[{index}]")
        pattern["id"] = identifier
        normalized_method = str(
            pattern.get("method", pattern.get("operation", pattern.get("kind", "fixed")))
        ).strip().casefold().replace("-", "_")
        if normalized_method == "fixed_value":
            normalized_method = "fixed_level"
        if normalized_method == "fixed-value":
            normalized_method = "fixed"
        if "scope" not in pattern:
            # Preserve the old reference-bank experiment as explicitly
            # conditional.  Local typed operations are full-episode by
            # default, so a missing donor/context cannot silently turn them
            # into the old 19/127 experiment.
            pattern["scope"] = "explicitly_conditional" if normalized_method == "reference" else "full_episode"
        scope = str(pattern["scope"]).strip().casefold().replace("-", "_")
        if scope not in _PATTERN_SCOPES:
            raise ConfigError(
                f"patterns[{index}].scope はfull_episodeまたはexplicitly_conditionalで指定してください"
            )
        pattern["scope"] = scope
        if "on_inapplicable" not in pattern:
            pattern["on_inapplicable"] = (
                "continue_unmodified_with_warning"
                if scope == "explicitly_conditional"
                else "abort_pattern"
            )
        action = str(pattern["on_inapplicable"]).strip().casefold().replace("-", "_")
        if action not in _PATTERN_INAPPLICABLE:
            raise ConfigError(
                f"patterns[{index}].on_inapplicable はabort_patternまたはcontinue_unmodified_with_warningで指定してください"
            )
        pattern["on_inapplicable"] = action
        if normalized_method == "reference" and scope == "explicitly_conditional":
            # Compatibility keys are a property of the legacy reference
            # pattern, not a global rule for every future reference adapter.
            metadata = pattern.setdefault("metadata", {})
            if not isinstance(metadata, dict):
                raise ConfigError(f"patterns[{index}].metadata はtableで指定してください")
            metadata.setdefault("compatibility_keys", ["road_segment_id", "target_lane_ordinal"])
        result.append(pattern)
    if not any(item["id"] == "P00" for item in result):
        raise ConfigError("patterns には変更なしのP00を明示してください")
    return result


def _check_video_patterns(raw: object, pattern_ids: set[str]) -> list[str]:
    """Validate an optional, explicit subset used for video capture.

    Video selection is an output policy.  It must refer to the same declared
    intervention IDs as the experiment instead of silently creating a second
    pattern namespace or falling back to an unrelated row.
    """

    if not isinstance(raw, list):
        raise ConfigError("video.patterns はpattern idの文字列arrayで指定してください")
    result: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(raw):
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(
                f"video.patterns[{index}] は空でないpattern id文字列で指定してください"
            )
        identifier = _basename(value.strip(), f"video.patterns[{index}]")
        if identifier in seen:
            raise ConfigError(f"video.patterns に重複idがあります: {identifier}")
        seen.add(identifier)
        result.append(identifier)
    missing = sorted(seen - pattern_ids)
    if missing:
        raise ConfigError(
            f"video.patterns に未定義idがあります: {', '.join(missing)}"
        )
    return result


@dataclass(frozen=True)
class AnalysisConfig(Mapping[str, Any]):
    """解決済み解析設定。

    ``Mapping`` としても使えるため、アダプターが従来のdict契約を要求する
    場合でも ``config.raw`` を意識せず渡せる。パスはconfigファイル基準の
    absolute pathへ解決し、保存時は ``to_dict`` でJSON-safeに戻す。
    """

    source_path: Path
    raw: dict[str, Any]
    analysis: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    adapter: dict[str, Any] = field(default_factory=dict)
    schema: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    scenario: dict[str, Any] = field(default_factory=dict)
    seeds: dict[str, Any] = field(default_factory=dict)
    preprocess: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    closed_loop: dict[str, Any] = field(default_factory=dict)
    video: dict[str, Any] = field(default_factory=dict)
    ig: dict[str, Any] = field(default_factory=dict)
    patterns: tuple[dict[str, Any], ...] = ()

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self):
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    @property
    def deterministic(self) -> bool:
        return bool(self.analysis.get("deterministic", True))

    @property
    def experiment_name(self) -> str:
        return str(self.output.get("experiment") or self.analysis.get("experiment") or self.analysis.get("name") or "default")

    @property
    def model_name(self) -> str:
        value = self.output.get("model") or self.model.get("name")
        if value:
            return _basename(str(value), "model.name")
        path = self.model.get("path")
        return _basename(Path(str(path)).stem if path else "model", "model.name")

    @property
    def model_path(self) -> Path | None:
        value = self.model.get("path")
        return Path(value) if value else None

    @property
    def schema_path(self) -> Path | None:
        value = self.schema.get("path")
        return Path(value) if value else None

    @property
    def output_root(self) -> Path:
        value = self.output.get("root") or "outputs/input_attribution"
        path = Path(str(value))
        return path if path.is_absolute() else (self.source_path.parent / path).resolve()

    @property
    def scenario_seed(self) -> int | None:
        value = self.seeds.get("scenario_seed", self.seeds.get("scenario", self.scenario.get("scenario_seed", self.scenario.get("seed"))))
        return int(value) if value is not None else None

    @property
    def rl_seed(self) -> int | None:
        value = self.seeds.get("rl_seed", self.seeds.get("rl", self.scenario.get("rl_seed")))
        return int(value) if value is not None else None

    def to_dict(self) -> dict[str, Any]:
        return _plain(self.raw)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def config_from_mapping(raw: Mapping[str, Any], *, source_path: str | Path = "<mapping>") -> AnalysisConfig:
    """Resolve a mapping using the same checks as a TOML file.

    This function is useful for fake adapters and for ``resolved_config.json``
    used by partial commands. It never mutates the caller's mapping.
    """

    source = Path(source_path).expanduser().resolve() if str(source_path) != "<mapping>" else Path.cwd() / "<mapping>"
    root = _mapping(raw, "root")
    unknown = sorted(set(root) - _TOP_LEVEL_KEYS)
    if unknown:
        raise ConfigError(f"root: 未対応のkeyがあります: {', '.join(unknown)}")

    sections: dict[str, dict[str, Any]] = {}
    for name in _SECTION_KEYS:
        sections[name] = _section(root, name)

    analysis = sections["analysis"]
    analysis.setdefault("name", "input_attribution")
    analysis.setdefault("experiment", analysis.get("name", "default"))
    analysis.setdefault("deterministic", True)
    if not isinstance(analysis["deterministic"], bool):
        raise ConfigError("analysis.deterministic はboolで指定してください")

    if not analysis["deterministic"]:
        raise ConfigError("この版はdeterministic=trueの評価のみ対応します。評価モードを自動変更しません")
    analysis.setdefault("individual_inputs", True)
    if not isinstance(analysis["individual_inputs"], bool):
        raise ConfigError("analysis.individual_inputs はboolで指定してください")
    analysis.setdefault("reference_id", "episode-0:0")

    model = sections["model"]
    if "path" in model:
        model["path"] = _relative_path(model["path"], source.parent, "model.path")
    if "name" in model:
        model["name"] = _basename(str(model["name"]), "model.name")
    if "deterministic" in model and model["deterministic"] != analysis["deterministic"]:
        raise ConfigError("model.deterministic and analysis.deterministic conflict")
    adapter = sections["adapter"]
    schema = sections["schema"]
    if "path" in schema:
        schema["path"] = _relative_path(schema["path"], source.parent, "schema.path")
    environment = sections["environment"]
    for key in ("path", "config"):
        if isinstance(environment.get(key), str):
            environment[key] = _relative_path(environment[key], source.parent, f"environment.{key}")
    preprocess = sections["preprocess"]
    for key in ("stats_path", "stats"):
        if isinstance(preprocess.get(key), str):
            preprocess[key] = _relative_path(preprocess[key], source.parent, f"preprocess.{key}")
    output = sections["output"]
    if "root" in output:
        output["root"] = _relative_path(output["root"], source.parent, "output.root")
    if "experiment" in output:
        output["experiment"] = _basename(str(output["experiment"]), "output.experiment")
    if "model" in output:
        output["model"] = _basename(str(output["model"]), "output.model")
    if output.get("overwrite") is True:
        raise ConfigError("output.overwrite=true は成果物保護のため未対応です")

    scenario = sections["scenario"]
    seeds = sections["seeds"]
    # Explicit seed aliases are accepted but conflicting values are rejected.
    scenario_seed_values = [
        value
        for value in (seeds.get("scenario_seed"), seeds.get("scenario"), scenario.get("scenario_seed"), scenario.get("seed"))
        if value is not None
    ]
    if len({int(value) for value in scenario_seed_values}) > 1:
        raise ConfigError("scenario seedが複数指定され一致しません")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in scenario_seed_values):
        raise ConfigError("scenario seeds must be nonnegative integers")
    if scenario_seed_values:
        seeds["scenario_seed"] = int(scenario_seed_values[0])
    rl_values = [value for value in (seeds.get("rl_seed"), seeds.get("rl"), scenario.get("rl_seed")) if value is not None]
    if len({int(value) for value in rl_values}) > 1:
        raise ConfigError("RL seedが複数指定され一致しません")
    if any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < 2**32 for v in rl_values):
        raise ConfigError("RL seeds must be integers in [0, 2**32)")
    if rl_values:
        seeds["rl_seed"] = int(rl_values[0])

    closed_loop = sections["closed_loop"]
    if "patterns" in closed_loop:
        if not isinstance(closed_loop["patterns"], list) or any(not isinstance(v, str) for v in closed_loop["patterns"]):
            raise ConfigError("closed_loop.patterns はpattern idのarrayで指定してください")
    else:
        closed_loop["patterns"] = ["P00"]
    closed_loop.setdefault("episodes", int(scenario.get("episodes", 1)))
    closed_loop.setdefault("enabled", True)
    closed_loop.setdefault("departure_tolerance_ratio", 0.05)
    closed_loop.setdefault("departure_consecutive_steps", 1)
    closed_loop.setdefault("low_speed_m_s", 0.5)
    if not isinstance(closed_loop["enabled"], bool):
        raise ConfigError("closed_loop.enabled はboolで指定してください")

    for section, key in ((scenario, "episodes"), (closed_loop, "episodes"), (closed_loop, "departure_consecutive_steps")):
        value = section.get(key, 1)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ConfigError(f"{key} must be a positive integer")
    if "max_steps" in closed_loop and (isinstance(closed_loop["max_steps"], bool) or not isinstance(closed_loop["max_steps"], int) or closed_loop["max_steps"] < 1):
        raise ConfigError("closed_loop.max_steps must be a positive integer")
    video = sections["video"]
    video.setdefault("enabled", False)
    ig = sections["ig"]
    ig.setdefault("enabled", False)
    if "steps" in ig:
        if "n_steps" in ig and ig["n_steps"] != ig["steps"]:
            raise ConfigError("ig.steps and ig.n_steps conflict")
        ig.setdefault("n_steps", ig.pop("steps"))
    ig.setdefault("n_steps", 64)
    ig.setdefault("method", "gausslegendre")
    ig.setdefault("completeness_tolerance", 1e-4)
    ig.setdefault("max_retries", 2)
    ig.setdefault("compatibility_keys", ["road_segment_id", "target_lane_ordinal"])
    if not isinstance(video["enabled"], bool) or not isinstance(ig["enabled"], bool):
        raise ConfigError("video.enabled and ig.enabled must be boolean")
    for key, minimum in (("n_steps", 2), ("max_retries", 0)):
        if isinstance(ig[key], bool) or not isinstance(ig[key], int) or ig[key] < minimum:
            raise ConfigError(f"ig.{key} must be an integer >= {minimum}")
    if not isinstance(ig["compatibility_keys"], list) or any(not isinstance(v, str) or not v for v in ig["compatibility_keys"]):
        raise ConfigError("ig.compatibility_keys must be a list of context keys")
    if ig["completeness_tolerance"] <= 0:
        raise ConfigError("ig.completeness_tolerance must be positive")
    patterns = _check_patterns(root.get("patterns"))
    pattern_ids = {entry["id"] for entry in patterns}
    missing = sorted(set(closed_loop["patterns"]) - pattern_ids)
    if missing:
        raise ConfigError(f"closed_loop.patterns に未定義idがあります: {', '.join(missing)}")
    if "patterns" in video:
        video["patterns"] = _check_video_patterns(video["patterns"], pattern_ids)

    resolved: dict[str, Any] = {
        "analysis": analysis,
        "model": model,
        "adapter": adapter,
        "schema": schema,
        "environment": environment,
        "scenario": scenario,
        "seeds": seeds,
        "preprocess": preprocess,
        "output": output,
        "closed_loop": closed_loop,
        "video": video,
        "ig": ig,
        "patterns": patterns,
    }
    return AnalysisConfig(
        source_path=source,
        raw=_plain(resolved),
        analysis=analysis,
        model=model,
        adapter=adapter,
        schema=schema,
        environment=environment,
        scenario=scenario,
        seeds=seeds,
        preprocess=preprocess,
        output=output,
        closed_loop=closed_loop,
        video=video,
        ig=ig,
        patterns=tuple(patterns),
    )


def load_config(path: str | Path) -> AnalysisConfig:
    """TOMLまたは保存済みresolved JSONを読み込む。"""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ConfigError(f"設定ファイルがありません: {source}")
    try:
        if source.suffix.lower() == ".json":
            raw = json.loads(source.read_text(encoding="utf-8"))
        else:
            with source.open("rb") as file_obj:
                raw = tomllib.load(file_obj)
    except (OSError, UnicodeError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"設定を読み込めません: {source}: {exc}") from exc
    return config_from_mapping(raw, source_path=source)


load_analysis_config = load_config
resolve_config = config_from_mapping


__all__ = [
    "AnalysisConfig",
    "ConfigError",
    "config_from_mapping",
    "load_analysis_config",
    "load_config",
    "resolve_config",
]
