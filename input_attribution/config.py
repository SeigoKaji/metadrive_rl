"""Configuration loading for the input attribution experiment.

The file format is intentionally a small TOML mapping.  A config path is the
anchor for relative paths; callers can provide ``repo_root`` when a config is
copied elsewhere.  No simulator or model dependency is imported here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib  # type: ignore[no-redef]

from .interventions import InterventionError, coerce_intervention


class ConfigError(ValueError):
    """A config cannot describe a safe attribution run."""


DEFAULT_BASELINE_SHA = "7849aad80ac353fd616c1a1398c11dd3497eed05"
DEFAULT_HORIZON = 127
DEFAULT_FIXED_VALUE = -1.0


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConfigError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigError(f"{name} must be finite")
    return number


def _positive_int(value: object, *, name: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ConfigError(f"{name} must be an integer")
    result = int(value)
    if result < 0 or (result == 0 and not allow_zero):
        comparator = "non-negative" if allow_zero else "positive"
        raise ConfigError(f"{name} must be {comparator}")
    return result


def _seed(value: object, *, name: str) -> int:
    result = _positive_int(value, name=name, allow_zero=True)
    if result > 2**32 - 1:
        raise ConfigError(f"{name} must be at most 2**32-1")
    return result


def _string(value: object, *, name: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ConfigError(f"{name} must be a non-empty string")
    if "\x00" in value:
        raise ConfigError(f"{name} contains NUL")
    return value


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a TOML table/mapping")
    result: dict[str, object] = {}
    for key, nested in value.items():
        if not isinstance(key, str):
            raise ConfigError(f"{name} keys must be strings")
        result[key] = nested
    return result


def _safe_relative_name(value: object, *, name: str) -> str:
    result = _string(value, name=name)
    path = Path(result)
    if path.is_absolute() or result in {".", ".."}:
        raise ConfigError(f"{name} must be a simple relative name")
    if "/" in result or "\\" in result:
        raise ConfigError(f"{name} must not contain a directory")
    return result


def default_patterns(dimension: int = 259, *, fixed_value: float = DEFAULT_FIXED_VALUE) -> tuple[dict[str, object], ...]:
    """Return groups from the verified schema module.

    A width alone never selects the 259-dimensional preset.  The schema owns
    the source-backed indices and rejects wider/unknown hosts; callers for a
    synthetic or ported dimension must provide explicit patterns.
    """

    dimension = _positive_int(dimension, name="dimension")
    fixed_value = _finite_number(fixed_value, name="fixed_value")
    try:
        from .schema import default_patterns as schema_default_patterns
        source = schema_default_patterns(dimension=dimension, fixed_value=fixed_value)
    except Exception as error:
        raise ConfigError(
            "verified schema patterns are unavailable for this dimension; "
            "provide explicit patterns"
        ) from error
    return tuple(
        {
            "id": str(item["id"]),
            "name": str(item["name"]),
            "indices": list(item["indices"]),
            "fixed_value": float(item["fixed_value"]),
        }
        for item in source
    )


def synthetic_default_patterns(
    dimension: int = 259,
    *,
    fixed_value: float = DEFAULT_FIXED_VALUE,
) -> tuple[dict[str, object], ...]:
    """Return explicit, schema-free groups for the dependency-free backend."""

    dimension = _positive_int(dimension, name="dimension")
    fixed_value = _finite_number(fixed_value, name="fixed_value")
    candidate_groups = (
        ("P01", "synthetic feature 0", (0,)),
        ("P02", "synthetic feature 1", (1,)),
        ("P03", "synthetic feature 2", (2,)),
        ("P04", "synthetic feature 3", (3,)),
        ("P05", "synthetic feature 4", (4,)),
        ("P06", "synthetic feature 5", (5,)),
        ("P07", "synthetic feature 6", (6,)),
        ("P08", "synthetic feature 7", (7,)),
        ("P09", "synthetic feature 8", (8,)),
        ("P10", "synthetic feature 9", (9,)),
    )
    return tuple(
        {
            "id": pattern_id,
            "name": name,
            "indices": list(indices),
            "fixed_value": fixed_value,
        }
        for pattern_id, name, indices in candidate_groups
        if max(indices) < dimension
    )


def _verified_standard_schema_source(
    source: str | dict[str, object] | None,
    *,
    root: Path | None,
) -> bool:
    """Return whether ``source`` resolves to the verified MetaDrive 259 rows.

    A real-host config may omit ``patterns`` only when a source-backed schema
    proves the standard contract.  Looking at a dimension (or a synthetic
    schema with the same width) is insufficient evidence for importing the
    official group positions.
    """

    if source is None:
        return False
    resolved_source: object = source
    if isinstance(source, str):
        candidate = Path(source).expanduser()
        if not candidate.is_absolute() and root is not None:
            anchored = (root / candidate).resolve()
            if anchored.is_file():
                resolved_source = anchored
    try:
        from .schema import load_schema

        rows = load_schema(resolved_source, dimension=259, require_verified=True)
    except Exception:
        return False
    if not rows or len(rows) != 259:
        return False
    # Every generated official row carries the MetaDrive source path.  This
    # excludes synthetic_259_contract, even though it is deliberately marked
    # verified for dependency-light tests.
    return all(
        row.get("status") == "verified"
        and str(row.get("source_path", "")).startswith("metadrive/")
        for row in rows
    )
@dataclass(frozen=True, slots=True)
class AttributionConfig:
    """Validated settings shared by baseline, offline A, and closed-loop B."""

    backend: str = "synthetic"
    model_path: str | None = None
    environment_config: str | dict[str, object] | None = None
    schema: str | dict[str, object] | None = None
    output_dir: str = "outputs/input_attribution"
    scenario_seed: int = 5
    policy_seed: int = 5
    horizon: int = DEFAULT_HORIZON
    record_gif: bool = True
    patterns: tuple[dict[str, object], ...] = field(default_factory=lambda: synthetic_default_patterns())
    synthetic_dimension: int = 259
    synthetic_steps: int = DEFAULT_HORIZON
    synthetic_natural_end_step: int | None = None
    synthetic_action_count: int = 9
    deterministic: bool = True
    reward_atol: float = 1e-6
    reward_rtol: float = 1e-6
    strict_reward_terms: bool = False
    repo_root: str | None = None
    baseline_main_sha: str = DEFAULT_BASELINE_SHA

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        config_path: str | Path | None = None,
    ) -> "AttributionConfig":
        root = _infer_root(config_path)
        source = _mapping(value, name="config")
        backend = _string(source.get("backend", "synthetic"), name="backend").lower()
        if backend not in {"synthetic", "metadrive"}:
            raise ConfigError("backend must be 'synthetic' or 'metadrive'")

        model_path_value = source.get("model_path")
        model_path: str | None
        if model_path_value is None:
            model_path = None
        else:
            model_path = _string(model_path_value, name="model_path")

        environment_value = source.get("environment_config")
        environment: str | dict[str, object] | None
        if environment_value is None or isinstance(environment_value, str):
            environment = environment_value
        else:
            environment = _mapping(environment_value, name="environment_config")

        schema_value = source.get("schema")
        schema: str | dict[str, object] | None
        if schema_value is None or isinstance(schema_value, str):
            schema = schema_value
        else:
            schema = _mapping(schema_value, name="schema")

        output_dir = _string(
            source.get("output_dir", "outputs/input_attribution"),
            name="output_dir",
        )
        scenario_seed = _seed(source.get("scenario_seed", 5), name="scenario_seed")
        policy_seed = _seed(source.get("policy_seed", 5), name="policy_seed")
        horizon = _positive_int(
            source.get("horizon", DEFAULT_HORIZON),
            name="horizon",
            allow_zero=True,
        )
        record_gif = source.get("record_gif", True)
        if not isinstance(record_gif, bool):
            raise ConfigError("record_gif must be a boolean")
        deterministic = source.get("deterministic", True)
        if not isinstance(deterministic, bool):
            raise ConfigError("deterministic must be a boolean")
        synthetic_dimension = _positive_int(
            source.get("synthetic_dimension", source.get("dimension", 259)),
            name="synthetic_dimension",
        )
        synthetic_steps = _positive_int(
            source.get("synthetic_steps", horizon or DEFAULT_HORIZON),
            name="synthetic_steps",
            allow_zero=True,
        )
        natural_end_value = source.get("synthetic_natural_end_step")
        if natural_end_value is None:
            synthetic_natural_end_step = None
        else:
            synthetic_natural_end_step = _positive_int(
                natural_end_value,
                name="synthetic_natural_end_step",
                allow_zero=False,
            )
        synthetic_action_count = _positive_int(
            source.get("synthetic_action_count", source.get("action_count", 9)),
            name="synthetic_action_count",
        )
        reward_atol = _finite_number(source.get("reward_atol", 1e-6), name="reward_atol")
        reward_rtol = _finite_number(source.get("reward_rtol", 1e-6), name="reward_rtol")
        if reward_atol < 0 or reward_rtol < 0:
            raise ConfigError("reward tolerances must be non-negative")
        strict_reward_terms = source.get(
            "strict_reward_terms",
            source.get("reward_terms_strict", False),
        )
        if not isinstance(strict_reward_terms, bool):
            raise ConfigError("strict_reward_terms must be a boolean")

        fixed_value = _finite_number(source.get("fixed_value", DEFAULT_FIXED_VALUE), name="fixed_value")
        patterns_value = source.get("patterns")
        if patterns_value is None:
            if backend == "synthetic":
                patterns = synthetic_default_patterns(synthetic_dimension, fixed_value=fixed_value)
            else:
                if not _verified_standard_schema_source(schema, root=root):
                    raise ConfigError(
                        "metadrive configs require explicit patterns unless a "
                        "verified standard 259 schema is provided"
                    )
                patterns = default_patterns(259, fixed_value=fixed_value)
        else:
            if isinstance(patterns_value, Mapping):
                # A table of named patterns is accepted as a convenience, but
                # order remains the TOML insertion order.
                patterns_value = [
                    dict(_mapping(item, name=f"patterns.{key}"), id=str(key))
                    for key, item in patterns_value.items()
                ]
            if isinstance(patterns_value, (str, bytes)) or not isinstance(patterns_value, Sequence):
                raise ConfigError("patterns must be an array of tables")
            parsed: list[dict[str, object]] = []
            seen_ids: set[str] = set()
            for offset, item in enumerate(patterns_value):
                try:
                    pattern_mapping = _mapping(item, name=f"patterns[{offset}]")
                    if "fixed_value" not in pattern_mapping:
                        pattern_mapping["fixed_value"] = fixed_value
                    pattern = coerce_intervention(
                        pattern_mapping,
                        dimension=synthetic_dimension if backend == "synthetic" else None,
                    )
                except (ConfigError, InterventionError, TypeError, ValueError) as error:
                    raise ConfigError(f"patterns[{offset}] is invalid: {error}") from error
                if pattern.id in seen_ids:
                    raise ConfigError(f"duplicate pattern id: {pattern.id}")
                seen_ids.add(pattern.id)
                parsed.append(
                    {
                        "id": pattern.id,
                        "name": pattern.name,
                        "indices": list(pattern.indices),
                        "fixed_value": pattern.fixed_value,
                    }
                )
            if not parsed:
                raise ConfigError("patterns must contain at least one pattern")
            patterns = tuple(parsed)

        repo_root_value = source.get("repo_root")
        if repo_root_value is None:
            repo_root = str(root) if root is not None else None
        else:
            repo_root = _string(repo_root_value, name="repo_root")

        baseline_main_sha = _string(
            source.get("base_main_sha", DEFAULT_BASELINE_SHA),
            name="base_main_sha",
        )
        return cls(
            backend=backend,
            model_path=model_path,
            environment_config=environment,
            schema=schema,
            output_dir=output_dir,
            scenario_seed=scenario_seed,
            policy_seed=policy_seed,
            horizon=horizon,
            record_gif=record_gif,
            patterns=patterns,
            synthetic_dimension=synthetic_dimension,
            synthetic_steps=synthetic_steps,
            synthetic_natural_end_step=synthetic_natural_end_step,
            synthetic_action_count=synthetic_action_count,
            deterministic=deterministic,
            reward_atol=reward_atol,
            reward_rtol=reward_rtol,
            strict_reward_terms=strict_reward_terms,
            repo_root=repo_root,
            baseline_main_sha=baseline_main_sha,
        )

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["patterns"] = [dict(pattern) for pattern in self.patterns]
        return result


def _infer_root(config_path: str | Path | None) -> Path | None:
    if config_path is None:
        return None
    path = Path(config_path).expanduser().resolve()
    if path.is_file():
        # Package configs are conventionally input_attribution/configs/*.toml;
        # use the repository directory as the anchor in that case.
        if path.parent.name == "configs" and path.parent.parent.name == "input_attribution":
            return path.parent.parent.parent
        return path.parent
    return path.parent


def load_config(path: str | Path) -> AttributionConfig:
    """Load and validate a TOML attribution config."""

    config_path = Path(path).expanduser().resolve()
    if config_path.suffix.lower() != ".toml":
        raise ConfigError(f"config must be a TOML file: {config_path}")
    try:
        with config_path.open("rb") as stream:
            mapping = tomllib.load(stream)
    except OSError as error:
        raise ConfigError(f"cannot read config: {config_path}") from error
    except Exception as error:
        raise ConfigError(f"invalid TOML config: {config_path}: {error}") from error
    return AttributionConfig.from_mapping(mapping, config_path=config_path)


def parse_config(value: Mapping[str, object] | str | Path) -> AttributionConfig:
    """Parse either a mapping or a path, convenient for tests and adapters."""

    if isinstance(value, Mapping):
        return AttributionConfig.from_mapping(value)
    return load_config(value)


def resolve_path(value: str | Path, *, config: AttributionConfig | None = None, base: str | Path | None = None) -> Path:
    """Resolve a config path using explicit base, then ``repo_root``."""

    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    anchor = Path(base).expanduser() if base is not None else None
    if anchor is None and config is not None and config.repo_root:
        anchor = Path(config.repo_root).expanduser()
    if anchor is None:
        anchor = Path.cwd()
    return (anchor / candidate).resolve()


__all__ = [
    "AttributionConfig",
    "ConfigError",
    "DEFAULT_BASELINE_SHA",
    "DEFAULT_FIXED_VALUE",
    "DEFAULT_HORIZON",
    "default_patterns",
    "load_config",
    "parse_config",
    "resolve_path",
]
