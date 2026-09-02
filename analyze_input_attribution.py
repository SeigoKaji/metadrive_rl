"""Independent CLI for PPO vector-input dependence and attribution analysis.

``--help`` and ``schema-template`` intentionally avoid importing PPO or
creating MetaDrive.  Runtime-only imports live inside command handlers so a
portable schema can be prepared on a machine before its simulator assets are
installed.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np

from configs.experiment_config import ExperimentConfigError, select_experiment
from input_attribution.closed_loop import (
    ClosedLoopResult,
    ClosedLoopTarget,
    normalize_targets,
    run_paired_closed_loop,
    save_closed_loop,
    schema_targets,
)
from input_attribution.results import (
    ArtifactError,
    StagedRunDirectory,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_npz,
    atomic_write_text,
    expanded_schema_rows,
    json_value,
    safe_child,
    sha256_file,
    utc_now_iso,
    validate_basename,
    write_expanded_schema_csv,
    write_report,
)
from input_attribution.rollout import (
    RolloutData,
    RolloutError,
    collect_rollout,
    load_rollout,
    save_rollout,
    validate_runtime_contract,
)
from project_paths import MODEL_DIR, OUTPUT_DIR, PROJECT_ROOT


class AttributionCLIError(ValueError):
    """A user-facing CLI error with no partial result publication."""


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("positive integer is required") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("positive integer is required")
    return parsed


def _output_prefix(value: str) -> str:
    try:
        return validate_basename(value)
    except ArtifactError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _resolve_project_path(value: str | Path) -> Path:
    """Resolve command paths from the project root, not the current shell."""

    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _safe_experiment_name(value: str) -> str:
    try:
        return validate_basename(value, label="experiment name")
    except ArtifactError as error:
        raise AttributionCLIError(str(error)) from error


def _add_runtime_arguments(
    parser: argparse.ArgumentParser,
    *,
    require_analysis_config: bool,
    require_output_prefix: bool,
) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="experiment TOML (project-relative paths are allowed)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="PPO .zip; defaults to models/<experiment default_model_name>.zip",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        required=True,
        help="observation schema TOML",
    )
    if require_analysis_config:
        parser.add_argument(
            "--analysis-config",
            type=Path,
            required=True,
            help="strict attribution analysis TOML",
        )
    if require_output_prefix:
        parser.add_argument(
            "--output-prefix",
            type=_output_prefix,
            required=True,
            help="safe basename below outputs/<experiment>/attribution",
        )
    parser.add_argument(
        "--device",
        default=None,
        help="override analysis TOML device for PPO.load(), e.g. cpu or cuda",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="override RL-space/evaluation seed without changing scenario seed range",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser without starting MetaDrive or importing PPO."""

    parser = argparse.ArgumentParser(
        description="Analyze PPO vector-input dependence and Integrated Gradients",
        allow_abbrev=False,
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    run = subcommands.add_parser(
        "run", help="collect, analyze, plot, and optionally validate closed loop", allow_abbrev=False
    )
    _add_runtime_arguments(run, require_analysis_config=True, require_output_prefix=True)
    run.add_argument(
        "--closed-loop",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override [closed_loop].enabled for this complete run",
    )

    collect = subcommands.add_parser(
        "collect", help="save ordinary pre-action rollout data only", allow_abbrev=False
    )
    _add_runtime_arguments(collect, require_analysis_config=True, require_output_prefix=True)

    analyze = subcommands.add_parser(
        "analyze", help="analyze a saved rollout without starting MetaDrive", allow_abbrev=False
    )
    _add_runtime_arguments(analyze, require_analysis_config=True, require_output_prefix=True)
    analyze.add_argument(
        "--rollout",
        type=Path,
        required=True,
        help="directory containing rollout_arrays.npz and rollout metadata",
    )
    analyze.add_argument(
        "--closed-loop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="reserved for a future explicit env run; offline analyze never starts MetaDrive",
    )

    closed_loop = subcommands.add_parser(
        "closed-loop", help="run explicitly selected paired interventions", allow_abbrev=False
    )
    _add_runtime_arguments(closed_loop, require_analysis_config=True, require_output_prefix=True)
    closed_loop.add_argument(
        "--feature",
        action="append",
        default=[],
        help="feature index or schema feature name; may be repeated",
    )
    closed_loop.add_argument(
        "--group",
        action="append",
        default=[],
        help="schema group name; may be repeated",
    )
    closed_loop.add_argument(
        "--rollout",
        type=Path,
        default=None,
        help="optional saved rollout, required by dataset_median_constant",
    )

    validate = subcommands.add_parser(
        "validate-schema", help="check model, env, reset, and schema dimensions", allow_abbrev=False
    )
    _add_runtime_arguments(validate, require_analysis_config=False, require_output_prefix=False)

    template = subcommands.add_parser(
        "schema-template", help="write an unresolved generic schema template", allow_abbrev=False
    )
    template.add_argument("--dim", type=_positive_int, required=True, help="flat observation dimension")
    template.add_argument("--output", type=Path, required=True, help="schema TOML path")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Public parser entry point used by tests and external scripts."""

    return build_parser().parse_args(argv)


def _load_runtime(args: argparse.Namespace, *, require_analysis: bool) -> dict[str, Any]:
    """Load configuration, schema, PPO, and the narrow policy adapter lazily."""

    from input_attribution.config import load_analysis_config
    from input_attribution.sb3_adapter import SB3PolicyAdapter
    from input_attribution.schema import load_observation_schema
    from stable_baselines3 import PPO
    from stable_baselines3.common.utils import set_random_seed

    config_path = _resolve_project_path(args.config)
    try:
        experiment = select_experiment(config_path=config_path)
    except ExperimentConfigError as error:
        raise AttributionCLIError(str(error)) from error
    schema_path = _resolve_project_path(args.schema)
    schema = load_observation_schema(schema_path)
    analysis = None
    if require_analysis:
        analysis_path = _resolve_project_path(args.analysis_config)
        analysis = load_analysis_config(analysis_path)
    device = (
        str(args.device)
        if args.device is not None
        else (str(analysis.run.device) if analysis is not None else "cpu")
    )
    seed = (
        int(args.seed)
        if args.seed is not None
        else int(experiment.profile.evaluation_defaults.get("seed", experiment.profile.training_config["seed"]))
    )
    model_path = (
        _resolve_project_path(args.model)
        if args.model is not None
        else (MODEL_DIR / f"{experiment.profile.default_model_name}.zip").resolve()
    )
    if model_path.is_symlink() or not model_path.is_file() or model_path.stat().st_size <= 0:
        raise AttributionCLIError(
            "PPO model is missing or empty: "
            f"{model_path}. Train the configured model first or pass --model."
        )
    set_random_seed(seed)
    model = PPO.load(str(model_path), device=device)
    adapter = SB3PolicyAdapter(model, device=device)
    return {
        "experiment": experiment,
        "schema": schema,
        "analysis": analysis,
        "model": model,
        "adapter": adapter,
        "model_path": model_path,
        "seed": seed,
        "device": str(getattr(model, "device", device)),
    }


def _versions() -> dict[str, str | None]:
    packages = ("numpy", "pandas", "torch", "stable-baselines3", "metadrive", "matplotlib")
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            if package == "metadrive":
                try:
                    versions[package] = str(importlib.import_module("metadrive.version").VERSION)
                except (AttributeError, ImportError):
                    versions[package] = None
            else:
                versions[package] = None
    return versions


def _source_record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def _metadata(
    *,
    args: argparse.Namespace,
    runtime: Mapping[str, Any],
    rollout: RolloutData | None,
    started_at: str,
    finished_at: str | None = None,
    runtime_contract: Mapping[str, Any] | None = None,
    baseline_metadata: Mapping[str, Any] | None = None,
    closed_loop_targets: Sequence[str] = (),
    closed_loop_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    experiment = runtime["experiment"]
    schema = runtime["schema"]
    analysis = runtime.get("analysis")
    environment = experiment.profile.evaluation_env_config
    result: dict[str, Any] = {
        "analysis_format_version": 1,
        "command": [sys.executable, "analyze_input_attribution.py", *sys.argv[1:]],
        "started_at": started_at,
        "finished_at": finished_at,
        "python": sys.version,
        "platform": platform.platform(),
        "versions": _versions(),
        "experiment": experiment.source_metadata(),
        "model": _source_record(runtime["model_path"]),
        "schema": {
            "name": schema.name,
            "observation_dim": schema.observation_dim,
            **_source_record(schema.source_path),
        },
        "analysis_config": (
            None
            if analysis is None
            else {
                "name": analysis.name,
                "path": str(analysis.source_path),
                "sha256": analysis.source_sha256,
            }
        ),
        "device": runtime["device"],
        "deterministic": True if analysis is None else analysis.run.deterministic,
        "rl_seed": runtime["seed"],
        "scenario_seed_range": {
            "start": int(environment["start_seed"]),
            "count": int(environment["num_scenarios"]),
        },
        "model_observation_dim": int(getattr(runtime["adapter"], "observation_dim")),
        "action_dim": int(getattr(runtime["adapter"], "action_count")),
        "runtime_contract": dict(runtime_contract or {}),
        "rollout": (
            None
            if rollout is None
            else {
                "row_count": rollout.row_count,
                "observation_dim": rollout.observation_dim,
                "scenario_seeds": sorted({int(seed) for seed in rollout.scenario_seeds}),
            }
        ),
        "baseline": None if baseline_metadata is None else json_value(baseline_metadata),
        "closed_loop_targets": list(closed_loop_targets),
        "closed_loop_provenance": (
            None if closed_loop_provenance is None else json_value(closed_loop_provenance)
        ),
    }
    if analysis is not None:
        result.update(
            {
                "integrated_gradients": {
                    "targets": list(analysis.integrated_gradients.targets),
                    "steps": analysis.integrated_gradients.steps,
                    "batch_size": analysis.integrated_gradients.batch_size,
                },
                "perturbation": {
                    "batch_size": analysis.perturbation.batch_size,
                    "lidar_sector_degrees": analysis.perturbation.lidar_sector_degrees,
                    "definition": "baseline replacement; JS divergence is the primary actor metric",
                },
                "phases": [json_value(asdict(phase)) for phase in analysis.phases],
            }
        )
    return result


def _runtime_source_sha256(runtime: Mapping[str, Any]) -> str:
    """Return the selected experiment TOML digest with a clear invariant error."""

    value = getattr(runtime["experiment"], "source_sha256", None)
    if not isinstance(value, str) or not value:
        raise AttributionCLIError("current experiment configuration has no source SHA-256")
    return value


def _require_provenance_mapping(
    metadata: Mapping[str, Any],
    key: str,
    *,
    missing: list[str],
) -> Mapping[str, Any] | None:
    value = metadata.get(key)
    if not isinstance(value, Mapping):
        missing.append(key)
        return None
    return value


def _require_provenance_string(
    metadata: Mapping[str, Any] | None,
    key: str,
    *,
    label: str,
    missing: list[str],
) -> str | None:
    if metadata is None:
        return None
    value = metadata.get(key)
    if not isinstance(value, str) or not value:
        missing.append(label)
        return None
    return value


def _require_provenance_integer(
    metadata: Mapping[str, Any],
    key: str,
    *,
    missing: list[str],
) -> int | None:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        missing.append(key)
        return None
    return int(value)


def _assert_offline_rollout_provenance(
    *,
    runtime: Mapping[str, Any],
    rollout: RolloutData,
) -> None:
    """Fail closed when a saved rollout was not made by this exact runtime.

    Matching only vector dimensions would allow a different same-shape PPO or
    a differently interpreted same-shape schema to be analyzed accidentally.
    Hashes make those provenance boundaries explicit before any output staging
    directory is opened.
    """

    metadata = rollout.metadata
    if not isinstance(metadata, Mapping):
        raise AttributionCLIError(
            "offline rollout provenance cannot be verified: rollout metadata is not an object"
        )
    # Gather all missing fields before raising, rather than obscuring the next
    # action behind a sequence of one-field-at-a-time errors.
    missing: list[str] = []
    model_metadata = _require_provenance_mapping(metadata, "model", missing=missing)
    experiment_metadata = _require_provenance_mapping(
        metadata, "experiment", missing=missing
    )
    schema_metadata = _require_provenance_mapping(metadata, "schema", missing=missing)
    saved_model_sha = _require_provenance_string(
        model_metadata, "sha256", label="model.sha256", missing=missing
    )
    saved_config_sha = _require_provenance_string(
        experiment_metadata, "sha256", label="experiment.sha256", missing=missing
    )
    saved_schema_sha = _require_provenance_string(
        schema_metadata, "sha256", label="schema.sha256", missing=missing
    )
    saved_schema_dim = _require_provenance_integer(
        schema_metadata or {}, "observation_dim", missing=missing
    )
    saved_action_dim = _require_provenance_integer(
        metadata, "action_dim", missing=missing
    )
    saved_model_observation_dim = _require_provenance_integer(
        metadata, "model_observation_dim", missing=missing
    )
    saved_deterministic = metadata.get("deterministic")
    if not isinstance(saved_deterministic, bool):
        missing.append("deterministic")
    if missing:
        raise AttributionCLIError(
            "offline rollout provenance cannot be verified; saved metadata is missing "
            + ", ".join(sorted(set(missing)))
        )

    schema = runtime["schema"]
    adapter = runtime["adapter"]
    schema_path = getattr(schema, "source_path", None)
    if not isinstance(schema_path, Path):
        raise AttributionCLIError("current observation schema has no source path for provenance validation")
    expected_model_sha = sha256_file(runtime["model_path"])
    expected_config_sha = _runtime_source_sha256(runtime)
    expected_schema_sha = sha256_file(schema_path)
    expected_action_dim = int(getattr(adapter, "action_count"))
    expected_model_observation_dim = int(getattr(adapter, "observation_dim"))
    expected_schema_dim = int(getattr(schema, "observation_dim"))
    expected_deterministic = bool(runtime["analysis"].run.deterministic)

    mismatches: list[str] = []
    if saved_model_sha != expected_model_sha:
        mismatches.append("model.sha256")
    if saved_config_sha != expected_config_sha:
        mismatches.append("experiment.sha256")
    if saved_schema_sha != expected_schema_sha:
        mismatches.append("schema.sha256")
    if saved_action_dim != expected_action_dim:
        mismatches.append(
            f"action_dim (saved={saved_action_dim}, current={expected_action_dim})"
        )
    if saved_model_observation_dim != expected_model_observation_dim:
        mismatches.append(
            "model_observation_dim "
            f"(saved={saved_model_observation_dim}, current={expected_model_observation_dim})"
        )
    if saved_schema_dim != expected_schema_dim:
        mismatches.append(
            "schema.observation_dim "
            f"(saved={saved_schema_dim}, current={expected_schema_dim})"
        )
    if rollout.observation_dim != expected_model_observation_dim:
        mismatches.append(
            "rollout observation_dim "
            f"(saved={rollout.observation_dim}, current={expected_model_observation_dim})"
        )
    if rollout.logits.ndim != 2 or rollout.logits.shape[1] != expected_action_dim:
        mismatches.append(
            "rollout logits action_dim "
            f"(saved={rollout.logits.shape}, current_action_dim={expected_action_dim})"
        )
    if bool(saved_deterministic) != expected_deterministic:
        mismatches.append(
            "deterministic "
            f"(saved={bool(saved_deterministic)}, current={expected_deterministic})"
        )
    if mismatches:
        raise AttributionCLIError(
            "offline rollout provenance mismatch: " + "; ".join(mismatches)
        )


def _source_rollout_record(path: Path, rollout: RolloutData) -> dict[str, Any]:
    """Record the exact saved rollout metadata used as a closed-loop reference."""

    metadata_path = path / "rollout_metadata.json"
    return {
        "path": str(path),
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "metadata": json_value(dict(rollout.metadata)),
        "row_count": rollout.row_count,
        "observation_dim": rollout.observation_dim,
    }


def _closed_loop_replacement_provenance(
    *,
    strategy: str,
    rollout: RolloutData | None,
    resolved_baselines: Any | None = None,
) -> dict[str, Any]:
    """Describe exactly how a paired closed-loop replacement is sourced."""

    result: dict[str, Any] = {
        "strategy": strategy,
        "reference_source": (
            "per_intervention_episode_reset"
            if strategy == "episode_start_constant"
            else "schema_declared_constants"
            if strategy == "schema_constant"
            else "saved_or_current_rollout"
        ),
    }
    if rollout is not None:
        result["rollout"] = {
            "row_count": rollout.row_count,
            "observation_dim": rollout.observation_dim,
        }
    if strategy == "specified_reference_constant":
        if resolved_baselines is None:
            raise AttributionCLIError(
                "specified_reference_constant provenance requires resolved baselines"
            )
        result["resolved_baseline"] = json_value(resolved_baselines.metadata)
        result["specified_reference_baseline_id"] = str(
            resolved_baselines.baseline_ids[0, 0]
        )
    elif strategy == "dataset_median_constant" and rollout is not None:
        result["dataset_median"] = {
            "source_row_count": rollout.row_count,
            "source_observation_dim": rollout.observation_dim,
        }
    return result


def _validate_adapter_contract(adapter: Any, contract: Mapping[str, Any]) -> None:
    """Check that the differentiable adapter exposes the validated env contract."""

    try:
        adapter_observation_dim = int(getattr(adapter, "observation_dim"))
        adapter_action_dim = int(getattr(adapter, "action_count"))
    except (TypeError, ValueError) as error:
        raise AttributionCLIError(
            "policy adapter must expose integer observation_dim and action_count"
        ) from error
    if adapter_observation_dim != int(contract["env_observation_dim"]):
        raise AttributionCLIError(
            "adapter/environment dimension mismatch: "
            f"adapter={adapter_observation_dim}, env={contract['env_observation_dim']}"
        )
    if adapter_action_dim != int(contract["env_action_dim"]):
        raise AttributionCLIError(
            "adapter/environment action mismatch: "
            f"adapter={adapter_action_dim}, env={contract['env_action_dim']}"
        )


def _validate_live_runtime(runtime: Mapping[str, Any]) -> dict[str, int]:
    """Reset one throwaway env and validate env/model/schema/adapter agreement."""

    from env_factory import make_evaluation_env

    environment = runtime["experiment"].profile.evaluation_env_config
    scenario_seed = int(environment["start_seed"])
    env = make_evaluation_env(seed=runtime["seed"], env_config=environment)
    try:
        reset = env.reset(seed=scenario_seed)
        if isinstance(reset, tuple):
            if len(reset) != 2:
                raise AttributionCLIError(
                    "validation env.reset must return observation or (observation, info)"
                )
            observation = reset[0]
        else:
            observation = reset
        contract = validate_runtime_contract(
            env=env,
            model=runtime["model"],
            schema=runtime["schema"],
            reset_observation=observation,
        )
        _validate_adapter_contract(runtime["adapter"], contract)
        return contract
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def _run_directory(experiment_name: str, prefix: str) -> Path:
    return OUTPUT_DIR / _safe_experiment_name(experiment_name) / "attribution" / _output_prefix(prefix)


def _run_in_staging(
    runtime: Mapping[str, Any],
    prefix: str,
    callback: Callable[[Path], Any],
) -> tuple[Path, Any]:
    target = _run_directory(runtime["experiment"].name, prefix)
    holder = StagedRunDirectory(target)
    with holder as staging:
        result = callback(staging)
        published = holder.publish()
    return published, result


def _feature_index(schema: Any, selector: str | int) -> int:
    if isinstance(selector, int):
        return int(selector)
    text = str(selector)
    if text.isdecimal():
        return int(text)
    feature = schema.feature_named(text)
    return int(feature.index)


def _stats(values: np.ndarray) -> dict[str, float | int | None]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(flat.size),
        "mean": float(np.mean(flat)),
        "median": float(np.median(flat)),
        "std": float(np.std(flat)),
        "p90": float(np.percentile(flat, 90)),
        "p95": float(np.percentile(flat, 95)),
        "max": float(np.max(flat)),
    }


def _summarize_perturbation(result: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create portable summary rows even when pandas aggregation is unavailable."""

    feature_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    metric_names = tuple(getattr(result, "metric_names"))
    for target_index, target in enumerate(result.targets):
        row: dict[str, Any] = {
            "target_name": target.name,
            "target_kind": target.kind,
            "target_indices": ";".join(str(index) for index in target.indices),
            "feature_index": target.indices[0] if len(target.indices) == 1 else None,
        }
        for metric in metric_names:
            values = np.asarray(getattr(result, metric))[:, target_index, :]
            stats = _stats(values)
            row.update({f"{key}_{metric}": value for key, value in stats.items()})
        action_changed = np.asarray(result.action_changed)[:, target_index, :]
        row["action_flip_rate"] = float(np.mean(action_changed))
        if target.kind == "feature":
            feature_rows.append(row)
        else:
            group_rows.append(row)
    return feature_rows, group_rows


def _summarize_ig(
    result: Any,
    schema: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    feature_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    completeness_rows: list[dict[str, Any]] = []
    attributions = np.asarray(result.attributions)
    for target_index, target_name in enumerate(result.targets):
        target_values = attributions[:, target_index, :, :]
        for feature in schema.features:
            signed = target_values[:, :, feature.index]
            absolute = np.abs(signed)
            signed_stats = _stats(signed)
            absolute_stats = _stats(absolute)
            feature_rows.append(
                {
                    "ig_target": target_name,
                    "feature_index": feature.index,
                    "feature_name": feature.name,
                    "mean_signed_ig": signed_stats["mean"],
                    "median_signed_ig": signed_stats["median"],
                    "mean_absolute_ig": absolute_stats["mean"],
                    "median_absolute_ig": absolute_stats["median"],
                    "p90_absolute_ig": absolute_stats["p90"],
                    "p95_absolute_ig": absolute_stats["p95"],
                    "positive_rate": float(np.mean(signed > 0)),
                    "negative_rate": float(np.mean(signed < 0)),
                    "zero_rate": float(np.mean(signed == 0)),
                    "count": signed_stats["count"],
                }
            )
        for group_name, indices in schema.groups.items():
            signed = target_values[:, :, list(indices)].sum(axis=-1)
            absolute_mass = np.abs(target_values[:, :, list(indices)]).sum(axis=-1)
            signed_stats = _stats(signed)
            mass_stats = _stats(absolute_mass)
            group_rows.append(
                {
                    "ig_target": target_name,
                    "group_name": group_name,
                    "group_indices": ";".join(str(index) for index in indices),
                    "group_signed_ig": signed_stats["mean"],
                    "group_absolute_mass": mass_stats["mean"],
                    "median_group_signed_ig": signed_stats["median"],
                    "median_group_absolute_mass": mass_stats["median"],
                    "p90_group_absolute_mass": mass_stats["p90"],
                    "p95_group_absolute_mass": mass_stats["p95"],
                    "count": signed_stats["count"],
                }
            )
        for sample_index in range(result.sample_count):
            for baseline_index in range(result.baseline_count):
                completeness_rows.append(
                    {
                        "sample_index": sample_index,
                        "ig_target": target_name,
                        "baseline_id": str(result.baseline_ids[sample_index, baseline_index]),
                        "target_action": int(result.target_actions[sample_index]),
                        "runner_up_action": int(result.runner_up_actions[sample_index]),
                        "F_x": float(result.input_outputs[sample_index, target_index, baseline_index]),
                        "F_x0": float(result.baseline_outputs[sample_index, target_index, baseline_index]),
                        "sum_ig": float(result.attribution_sums[sample_index, target_index, baseline_index]),
                        "completeness_residual": float(
                            result.completeness_residuals[sample_index, target_index, baseline_index]
                        ),
                        "absolute_completeness_residual": float(
                            result.absolute_completeness_residuals[sample_index, target_index, baseline_index]
                        ),
                        "relative_completeness_error": float(
                            result.relative_completeness_errors[sample_index, target_index, baseline_index]
                        ),
                    }
                )
    return feature_rows, group_rows, completeness_rows


def _array_fields(value: Any) -> dict[str, np.ndarray]:
    """Extract non-object NumPy fields from a result dataclass safely."""

    source: Mapping[str, Any]
    if is_dataclass(value) and not isinstance(value, type):
        source = {field: getattr(value, field) for field in value.__dataclass_fields__}
    elif isinstance(value, Mapping):
        source = value
    else:
        source = vars(value)
    arrays: dict[str, np.ndarray] = {}
    for name, item in source.items():
        if isinstance(item, np.ndarray) and item.dtype != object:
            arrays[str(name)] = np.asarray(item)
    return arrays


def _write_empty_csv(path: Path, fields: Sequence[str]) -> None:
    atomic_write_csv(path, [], fieldnames=fields)


def _table_rows(table: Any) -> list[dict[str, Any]]:
    if hasattr(table, "to_dict") and hasattr(table, "columns"):
        return [dict(row) for row in table.to_dict(orient="records")]
    if isinstance(table, Sequence):
        return [dict(row) for row in table if isinstance(row, Mapping)]
    return []


def _full_episode_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Pick the global baseline-mean view for rankings/overview plots."""

    selected = [
        dict(row)
        for row in rows
        if row.get("scope") == "full_episode"
        and row.get("baseline_scope", "mean_over_baselines") == "mean_over_baselines"
    ]
    return selected or [dict(row) for row in rows]


def _ig_rows_for_target(
    rows: Sequence[Mapping[str, Any]], target: str | None
) -> list[dict[str, Any]]:
    """Keep an IG overview target-pure instead of ranking mixed functions."""

    full_episode = _full_episode_rows(rows)
    if target is None:
        return full_episode
    return [row for row in full_episode if row.get("target") == target]


def _write_perturbation_artifacts(
    directory: Path,
    result: Any | None,
    *,
    rollout: RolloutData,
    analysis: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray | None]:
    steps_path = safe_child(directory, "perturbation_feature_steps.npz")
    feature_path = safe_child(directory, "perturbation_feature_summary.csv")
    group_path = safe_child(directory, "perturbation_group_summary.csv")
    if result is None:
        atomic_write_npz(steps_path, empty=np.empty((0,), dtype=np.float32))
        _write_empty_csv(feature_path, ("target_name", "target_kind", "mean_js_divergence"))
        _write_empty_csv(group_path, ("target_name", "target_kind", "mean_js_divergence"))
        return [], [], None
    arrays = _array_fields(result)
    arrays["target_names"] = np.asarray([target.name for target in result.targets], dtype=str)
    arrays["target_kinds"] = np.asarray([target.kind for target in result.targets], dtype=str)
    arrays["target_indices_json"] = np.asarray(
        [json.dumps(list(target.indices)) for target in result.targets], dtype=str
    )
    atomic_write_npz(steps_path, **arrays)
    from input_attribution.aggregation import (
        perturbation_feature_summary,
        perturbation_group_summary,
        summarize_perturbation,
    )

    summary = summarize_perturbation(
        result,
        episode_ids=rollout.episode_ids,
        steps=rollout.steps,
        progress_bins=analysis.aggregation.progress_bins,
        phases=analysis.phases,
        include_baseline_rows=False,
    )
    feature_rows = _table_rows(perturbation_feature_summary(summary))
    group_rows = _table_rows(perturbation_group_summary(summary))
    atomic_write_csv(feature_path, feature_rows)
    atomic_write_csv(group_path, group_rows)
    feature_matrix: np.ndarray | None = None
    feature_targets = [
        (index, target)
        for index, target in enumerate(result.targets)
        if target.kind == "feature" and len(target.indices) == 1
    ]
    if feature_targets:
        dimension = max(target.indices[0] for _index, target in feature_targets) + 1
        feature_matrix = np.full((result.sample_count, dimension), np.nan, dtype=np.float64)
        means = np.asarray(result.js_divergence).mean(axis=2)
        for target_index, target in feature_targets:
            feature_matrix[:, target.indices[0]] = means[:, target_index]
    return feature_rows, group_rows, feature_matrix


def _write_ig_artifacts(
    directory: Path,
    result: Any | None,
    schema: Any,
    *,
    rollout: RolloutData,
    analysis: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], np.ndarray | None]:
    archive_path = safe_child(directory, "ig_attributions.npz")
    feature_path = safe_child(directory, "ig_feature_summary.csv")
    group_path = safe_child(directory, "ig_group_summary.csv")
    completeness_path = safe_child(directory, "ig_completeness.csv")
    if result is None:
        atomic_write_npz(archive_path, empty=np.empty((0,), dtype=np.float32))
        _write_empty_csv(feature_path, ("ig_target", "feature_index", "mean_absolute_ig"))
        _write_empty_csv(group_path, ("ig_target", "group_name", "group_absolute_mass"))
        _write_empty_csv(completeness_path, ("ig_target", "completeness_residual"))
        return [], [], [], None
    arrays = _array_fields(result)
    arrays["targets"] = np.asarray(result.targets, dtype=str)
    # Preserve both signed IG (the canonical attribution) and explicit
    # per-observation absolute IG so downstream users do not have to infer a
    # different aggregation from a signed archive.
    arrays["absolute_attributions"] = np.abs(np.asarray(result.attributions))
    atomic_write_npz(archive_path, **arrays)
    from input_attribution.aggregation import summarize_integrated_gradients

    summaries = summarize_integrated_gradients(
        result,
        schema,
        episode_ids=rollout.episode_ids,
        steps=rollout.steps,
        progress_bins=analysis.aggregation.progress_bins,
        phases=analysis.phases,
        include_baseline_rows=False,
        lidar_sector_degrees=analysis.perturbation.lidar_sector_degrees,
    )
    feature_rows = _table_rows(summaries.feature_summary)
    group_rows = _table_rows(summaries.group_summary)
    # Make the two semantically distinct group columns self-explanatory in
    # exports and plot callers while retaining the aggregation table's normal
    # `mean_*` statistical naming.
    for row in group_rows:
        row["group_signed_ig"] = row.get("mean_signed_ig")
        row["group_absolute_mass"] = row.get("mean_absolute_mass")
    completeness_rows = _table_rows(summaries.completeness_summary)
    atomic_write_csv(feature_path, feature_rows)
    atomic_write_csv(group_path, group_rows)
    atomic_write_csv(completeness_path, completeness_rows)
    # The first configured target is sufficient for the overview time/angle
    # plot.  The NPZ and tables preserve every target separately.
    matrix = np.asarray(result.attributions)[:, 0, :, :].mean(axis=1)
    return feature_rows, group_rows, completeness_rows, matrix


def _top_names(rows: Sequence[Mapping[str, Any]], metric: str, *, limit: int = 10) -> list[str]:
    candidates: list[tuple[float, str]] = []
    for row in rows:
        value = row.get(metric)
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            continue
        if not np.isfinite(value):
            continue
        candidates.append((abs(float(value)), str(row.get("feature_name", row.get("target_name", row.get("group_name", "unknown"))))))
    return [name for _value, name in sorted(candidates, reverse=True)[:limit]]


def _numeric_metric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _row_label(row: Mapping[str, Any], names: Sequence[str]) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value):
            return str(value)
    return "unknown"


def _rank_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    names: Sequence[str],
    limit: int = 10,
    annotate_kind: bool = False,
) -> list[tuple[str, float]]:
    ranked: list[tuple[str, float]] = []
    for row in rows:
        value = _numeric_metric(row.get(metric))
        if value is None:
            continue
        label = _row_label(row, names)
        if annotate_kind:
            kind = row.get("group_kind", row.get("target_kind"))
            if kind is not None and str(kind):
                label = f"{label} [{kind}]"
        ranked.append((label, value))
    return sorted(ranked, key=lambda item: abs(item[1]), reverse=True)[:limit]


def _format_ranked_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    names: Sequence[str],
    limit: int = 10,
    annotate_kind: bool = False,
    unavailable: str = "該当する有限値の結果はありません。",
) -> str:
    ranked = _rank_rows(
        rows,
        metric=metric,
        names=names,
        limit=limit,
        annotate_kind=annotate_kind,
    )
    if not ranked:
        return unavailable
    return "\n".join(
        f"- `{name}`: `{metric}={value:.6g}`" for name, value in ranked
    )


def _target_values(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    values: list[str] = []
    for row in rows:
        value = row.get("target")
        if value is None:
            continue
        text = str(value)
        if text and text not in values:
            values.append(text)
    return values


def _ig_target_function_lines(value: Any) -> str:
    """Render the exact scalar function used for each configured IG target."""

    if isinstance(value, str):
        targets = (value,)
    elif isinstance(value, Sequence):
        targets = tuple(str(target) for target in value)
    else:
        targets = ()
    definitions = {
        "selected_log_probability": (
            "`selected_log_probability`: `F(z) = log π(a* | z)` "
            "（`a*` は元の observation `x` で固定）"
        ),
        "selected_vs_runner_up_margin": (
            "`selected_vs_runner_up_margin`: `F(z) = l_{a*}(z) − l_{a2}(z)` "
            "（raw logits、`a*` と runner-up `a2` は元の `x` で固定）"
        ),
        "critic_value": "`critic_value`: `F(z) = V(z)`",
    }
    lines = [definitions.get(target, f"`{target}`: 未知のIG target") for target in targets]
    return "\n".join(f"- {line}" for line in lines) if lines else "- IG target は設定されていません。"


def _report_markdown(
    *,
    metadata: Mapping[str, Any],
    schema: Any,
    perturbation_features: Sequence[Mapping[str, Any]],
    perturbation_groups: Sequence[Mapping[str, Any]],
    ig_features: Sequence[Mapping[str, Any]],
    ig_groups: Sequence[Mapping[str, Any]],
    closed_loop: ClosedLoopResult | None,
) -> str:
    """Produce a Japanese report with target-pure numerical rankings."""

    model = metadata.get("model", {})
    experiment = metadata.get("experiment", {})
    baseline = metadata.get("baseline")
    ig = metadata.get("integrated_gradients", {})
    perturbation = metadata.get("perturbation", {})
    perturbation_feature_text = _format_ranked_rows(
        perturbation_features,
        metric="mean_js_divergence",
        names=("feature_name", "target_name", "name"),
        unavailable="摂動 feature の full_episode / mean_over_baselines 結果はありません。",
    )
    perturbation_group_text = _format_ranked_rows(
        perturbation_groups,
        metric="mean_js_divergence",
        names=("group_name", "target_name", "name"),
        annotate_kind=True,
        unavailable="摂動 group / LiDAR sector の結果はありません。",
    )

    ig_targets = _target_values([*ig_features, *ig_groups])
    ig_sections: list[str] = []
    for target in ig_targets:
        target_features = [row for row in ig_features if row.get("target") == target]
        target_groups = [row for row in ig_groups if row.get("target") == target]
        ig_sections.append(
            "\n".join(
                (
                    f"### target: `{target}`",
                    "feature（`mean_absolute_ig`）:",
                    _format_ranked_rows(
                        target_features,
                        metric="mean_absolute_ig",
                        names=("feature_name", "target_name", "name"),
                    ),
                    "group / LiDAR sector（`group_absolute_mass`）:",
                    _format_ranked_rows(
                        target_groups,
                        metric="group_absolute_mass",
                        names=("group_name", "target_name", "name"),
                        annotate_kind=True,
                    ),
                )
            )
        )
    ig_text = "\n\n".join(ig_sections) if ig_sections else "IG は無効、または集計対象の結果がありません。"

    perturbation_lidar = [
        row for row in perturbation_groups if row.get("target_kind") == "lidar_sector"
    ]
    ig_lidar = [row for row in ig_groups if row.get("group_kind") == "lidar_sector"]
    ig_lidar_sections = [
        "\n".join(
            (
                f"### IG target: `{target}`",
                _format_ranked_rows(
                    [row for row in ig_lidar if row.get("target") == target],
                    metric="group_absolute_mass",
                    names=("group_name", "target_name", "name"),
                    annotate_kind=True,
                ),
            )
        )
        for target in _target_values(ig_lidar)
    ]
    lidar_text = "\n\n".join(
        (
            "摂動 LiDAR sector（`mean_js_divergence`）:",
            _format_ranked_rows(
                perturbation_lidar,
                metric="mean_js_divergence",
                names=("group_name", "target_name", "name"),
                annotate_kind=True,
                unavailable="摂動 LiDAR sector は設定されていないか、結果がありません。",
            ),
            "IG LiDAR sector（`group_absolute_mass`、target ごとに独立）:",
            "\n\n".join(ig_lidar_sections)
            if ig_lidar_sections
            else "IG LiDAR sector は設定されていないか、結果がありません。",
            "`lidar_ig_heatmap.png` と `lidar_perturbation_heatmap.png` は schema の index/angle metadata を使い、固定 index や固定 ray 数を仮定しません。",
        )
    )

    custom_names = {
        str(feature.name)
        for feature in getattr(schema, "features", ())
        if str(getattr(feature, "kind", "")) == "custom"
    }
    if not custom_names:
        custom_text = "schema に `kind = \"custom\"` の解決済み feature はありません。"
    else:
        custom_perturbation = [
            row
            for row in perturbation_features
            if _row_label(row, ("feature_name", "target_name", "name")) in custom_names
        ]
        custom_ig = [
            row
            for row in ig_features
            if _row_label(row, ("feature_name", "target_name", "name")) in custom_names
        ]
        custom_ig_sections = [
            "\n".join(
                (
                    f"### IG target: `{target}`",
                    _format_ranked_rows(
                        [row for row in custom_ig if row.get("target") == target],
                        metric="mean_absolute_ig",
                        names=("feature_name", "target_name", "name"),
                    ),
                )
            )
            for target in _target_values(custom_ig)
        ]
        custom_text = "\n\n".join(
            (
                "摂動 custom feature:",
                _format_ranked_rows(
                    custom_perturbation,
                    metric="mean_js_divergence",
                    names=("feature_name", "target_name", "name"),
                ),
                "IG custom feature（target ごとに独立）:",
                "\n\n".join(custom_ig_sections)
                if custom_ig_sections
                else "IG custom feature の結果はありません。",
            )
        )

    if closed_loop is None:
        closed_loop_text = "closed-loop は無効、またはこの offline 再解析では実行していません。"
    elif not closed_loop.summary_rows:
        closed_loop_text = "closed-loop は実行されましたが、集計可能な target 行がありません。"
    else:
        closed_rows: list[str] = []
        for row in closed_loop.summary_rows:
            parts = [
                f"- `{row.get('target_name')}` ({row.get('target_kind')}, indices={row.get('target_indices')}):",
                f"Δ total reward={_numeric_metric(row.get('mean_delta_total_reward'))!r}",
                f"Δ success={_numeric_metric(row.get('mean_delta_success'))!r}",
                f"Δ out_of_road={_numeric_metric(row.get('mean_delta_out_of_road'))!r}",
                f"Δ crash={_numeric_metric(row.get('mean_delta_crash'))!r}",
                f"Δ route_completion={_numeric_metric(row.get('mean_delta_route_completion'))!r}",
            ]
            closed_rows.append(" ".join(parts))
        closed_loop_text = "\n".join(closed_rows)

    configured_ig_targets = ig.get("targets", ())
    if isinstance(configured_ig_targets, str):
        configured_ig_targets = (configured_ig_targets,)
    elif not isinstance(configured_ig_targets, Sequence):
        configured_ig_targets = ()
    actor_target = "selected_log_probability"
    actor_target_configured = actor_target in configured_ig_targets
    actor_ig_rows = (
        [row for row in ig_features if row.get("target") == actor_target]
        if actor_target_configured
        else []
    )
    perturbation_top = {
        name
        for name, _value in _rank_rows(
            perturbation_features,
            metric="mean_js_divergence",
            names=("feature_name", "target_name", "name"),
            limit=10,
        )
    }
    ig_top = {
        name
        for name, _value in _rank_rows(
            actor_ig_rows,
            metric="mean_absolute_ig",
            names=("feature_name", "target_name", "name"),
            limit=10,
        )
    }
    if not actor_target_configured:
        agreement_text = (
            "IG に actor target `selected_log_probability` が設定されていないため、"
            "critic value 等を actor 摂動ランキングと比較しません。"
        )
    elif not perturbation_top or not ig_top:
        agreement_text = "片方または両方の feature ranking がないため、一致/不一致を評価できません。"
    else:
        agreement_text = "\n".join(
            (
                f"- 共通: {', '.join(sorted(perturbation_top & ig_top)) or 'なし'}",
                f"- 摂動のみ: {', '.join(sorted(perturbation_top - ig_top)) or 'なし'}",
                f"- IGのみ（`{actor_target}`）: {', '.join(sorted(ig_top - perturbation_top)) or 'なし'}",
            )
        )

    return f"""# PPO入力寄与・入力依存度レポート

## 1. 実行対象

- model: `{model.get('path')}`
- experiment: `{experiment.get('name')}`、scenario seeds: `{metadata.get('scenario_seed_range')}`
- observation dimension: `{schema.observation_dim}`、action dimension: `{metadata.get('action_dim')}`
- schema: `{schema.name}` (`{schema.source_path}`)。`feature_schema_expanded.csv` が index-to-meaning の正本です。

## 2. Baseline と方法

baseline:

`{json.dumps(json_value(baseline), ensure_ascii=False)}`

perturbation:

{perturbation.get('definition', 'baseline replacement perturbation は無効です。')}

IG target: `{ig.get('targets', [])}`。台形則の両端を含む `{ig.get('steps')}` 点です。actor target の `a*` は元 observation で一度だけ選び、各補間点で同じ action index の出力を評価します。これは環境 action を強制する意味ではありません。

IG target function:

{_ig_target_function_lines(ig.get('targets', []))}

## 3. 摂動の上位 feature

{perturbation_feature_text}

## 4. 摂動の上位 group / LiDAR sector

{perturbation_group_text}

## 5. IG の target 別ランキング

{ig_text}

## 6. LiDAR angle / sector の結果

{lidar_text}

## 7. Custom feature の結果

{custom_text}

## 8. Closed-loop の target 別 paired 差分

{closed_loop_text}

closed-loop replacement provenance:

`{json.dumps(json_value(metadata.get('closed_loop_provenance')), ensure_ascii=False)}`

## 9. 摂動とIGの一致・不一致

{agreement_text}

摂動依存度とIG寄与度は別の問いに答えるため、単一 score に合成しません。一致は補助的な示唆であり、不一致は baseline、局所的非線形性、feature 相関で生じ得ます。

## 10. 解釈上の注意

- 摂動は置換 baseline、IG は baseline と積分 path に依存します。
- IG は baseline から現在 observation までの出力差の帰属であり、現在点の局所微分感度そのものではありません。
- 個別 feature の摂動結果は相関・冗長性のため加算できず、hybrid observation は学習分布外になり得ます。
- closed-loop は固定済み policy の入力置換への依存を調べる paired intervention であり、一般的必要性や Remove-and-Retrain の因果証明ではありません。
- 1 scenario / seed の結果を他の道路、traffic、seed、policy へ一般化しません。
"""


def _select_closed_loop_targets(
    *,
    schema: Any,
    analysis: Any,
    feature_rows: Sequence[Mapping[str, Any]],
    group_rows: Sequence[Mapping[str, Any]],
    perturbation_result: Any | None,
) -> tuple[Any, ...]:
    """Choose explicit targets or perturbation-ranked targets for validation.

    The ranking is deliberately based only on offline perturbation's
    ``full_episode`` / ``mean_over_baselines`` ``mean_js_divergence`` rows.
    IG is a separate attribution method and must not silently decide which
    closed-loop intervention is run.  Dynamic LiDAR sectors are reconstructed
    from the perturbation target's exact indices, not from schema groups.
    """

    explicit_features = tuple(
        _feature_index(schema, selector)
        for selector in analysis.closed_loop.explicit_features
    )
    explicit_groups = tuple(str(group) for group in analysis.closed_loop.explicit_groups)
    explicit_targets = schema_targets(
        schema,
        feature_indices=explicit_features,
        group_names=explicit_groups,
    ) if explicit_features or explicit_groups else ()

    if perturbation_result is None:
        if not explicit_targets:
            raise AttributionCLIError(
                "closed-loop is enabled but perturbation is disabled; enable "
                "[perturbation] or provide explicit_features/explicit_groups"
            )
        return explicit_targets

    target_by_id: dict[str, Any] = {}
    for target in perturbation_result.targets:
        target_id = getattr(target, "target_id", None)
        if not isinstance(target_id, str) or not target_id:
            target_id = f"{getattr(target, 'kind', 'unknown')}:{getattr(target, 'name', '')}"
        target_by_id[target_id] = target

    def ranked_targets(
        rows: Sequence[Mapping[str, Any]],
        *,
        target_kinds: set[str],
        limit: int,
    ) -> list[ClosedLoopTarget]:
        if limit <= 0:
            return []
        ranked: list[tuple[float, int, ClosedLoopTarget]] = []
        for position, row in enumerate(rows):
            if (
                row.get("scope") != "full_episode"
                or row.get("baseline_scope") != "mean_over_baselines"
                or str(row.get("target_kind")) not in target_kinds
            ):
                continue
            score = row.get("mean_js_divergence")
            if isinstance(score, bool) or not isinstance(score, (int, float, np.number)):
                continue
            numeric_score = float(score)
            if not np.isfinite(numeric_score):
                continue
            target_id = row.get("target_id")
            target = target_by_id.get(str(target_id))
            if target is None:
                raise AttributionCLIError(
                    "perturbation ranking row cannot be mapped to its original target: "
                    f"{target_id!r}"
                )
            kind = str(getattr(target, "kind", ""))
            if kind not in target_kinds:
                raise AttributionCLIError(
                    "perturbation ranking target kind disagrees with its summary row: "
                    f"{target_id!r}"
                )
            ranked.append(
                (
                    numeric_score,
                    position,
                    ClosedLoopTarget(
                        name=str(getattr(target, "name")),
                        indices=tuple(int(index) for index in getattr(target, "indices")),
                        kind=kind,
                    ),
                )
            )
        return [
            target
            for _score, _position, target in sorted(
                ranked, key=lambda item: (-item[0], item[1])
            )[:limit]
        ]

    selected: list[ClosedLoopTarget] = list(explicit_targets)
    if not explicit_features:
        selected.extend(
            ranked_targets(
                feature_rows,
                target_kinds={"feature"},
                limit=int(analysis.closed_loop.top_k_features),
            )
        )
    if not explicit_groups:
        selected.extend(
            ranked_targets(
                group_rows,
                target_kinds={"group", "lidar_sector"},
                limit=int(analysis.closed_loop.top_k_groups),
            )
        )

    # Explicit targets stay first and win a rare name collision with an
    # automatically generated target.  ``normalize_targets`` then validates
    # all index bounds against the schema's full dimension.
    unique: list[ClosedLoopTarget] = []
    names: set[str] = set()
    for target in selected:
        if target.name in names:
            continue
        unique.append(target)
        names.add(target.name)
    if not unique:
        raise AttributionCLIError(
            "closed-loop is enabled but no target was selected; provide explicit "
            "selectors or set a positive perturbation top_k"
        )
    return normalize_targets(unique, observation_dim=int(schema.observation_dim))


def _write_disabled_closed_loop(directory: Path) -> None:
    atomic_write_jsonl(safe_child(directory, "closed_loop_runs.jsonl"), [])
    _write_empty_csv(
        safe_child(directory, "closed_loop_summary.csv"),
        ("target_name", "replacement_strategy", "mean_delta_total_reward"),
    )


def _analyze_rollout(
    *,
    directory: Path,
    runtime: Mapping[str, Any],
    rollout: RolloutData,
    allow_closed_loop: bool,
) -> dict[str, Any]:
    """Run offline analyses, serialize numeric artifacts, then make plots/report."""

    from input_attribution.baselines import BaselineProvider
    from input_attribution.integrated_gradients import run_integrated_gradients
    from input_attribution.perturbation import build_perturbation_targets, run_perturbation
    from input_attribution.visualization import generate_standard_plots

    analysis = runtime["analysis"]
    schema = runtime["schema"]
    adapter = runtime["adapter"]
    if rollout.observation_dim != schema.observation_dim:
        raise AttributionCLIError(
            f"saved rollout/schema dimension mismatch: rollout={rollout.observation_dim}, "
            f"schema={schema.observation_dim}"
        )
    if rollout.observation_dim != int(adapter.observation_dim):
        raise AttributionCLIError(
            f"saved rollout/model dimension mismatch: rollout={rollout.observation_dim}, "
            f"model={adapter.observation_dim}"
        )
    baselines = BaselineProvider(
        rollout.observations, rollout.episode_ids, rollout.steps
    ).resolve(analysis.baseline)
    perturbation_result = None
    if analysis.perturbation.enabled:
        targets = build_perturbation_targets(
            schema,
            analyze_features=analysis.perturbation.analyze_features,
            analyze_groups=analysis.perturbation.analyze_groups,
            lidar_sector_degrees=analysis.perturbation.lidar_sector_degrees,
        )
        perturbation_result = run_perturbation(
            adapter,
            rollout.observations,
            baselines,
            targets,
            batch_size=analysis.perturbation.batch_size,
        )
    ig_result = None
    if analysis.integrated_gradients.enabled:
        ig_result = run_integrated_gradients(
            adapter,
            rollout.observations,
            baselines,
            targets=analysis.integrated_gradients.targets,
            steps=analysis.integrated_gradients.steps,
            batch_size=analysis.integrated_gradients.batch_size,
        )
    perturbation_features, perturbation_groups, perturbation_matrix = _write_perturbation_artifacts(
        directory,
        perturbation_result,
        rollout=rollout,
        analysis=analysis,
    )
    ig_features, ig_groups, completeness, ig_matrix = _write_ig_artifacts(
        directory,
        ig_result,
        schema,
        rollout=rollout,
        analysis=analysis,
    )

    closed_loop_result: ClosedLoopResult | None = None
    closed_loop_provenance: dict[str, Any] | None = None
    should_run_closed_loop = bool(analysis.closed_loop.enabled and allow_closed_loop)
    if should_run_closed_loop:
        selected = _select_closed_loop_targets(
            schema=schema,
            analysis=analysis,
            feature_rows=perturbation_features,
            group_rows=perturbation_groups,
            perturbation_result=perturbation_result,
        )
        from env_factory import make_evaluation_env

        experiment = runtime["experiment"]
        scenario_start = int(experiment.profile.evaluation_env_config["start_seed"])
        scenario_count = int(experiment.profile.evaluation_env_config["num_scenarios"])
        specified_reference = (
            baselines.values[0, 0]
            if analysis.closed_loop.replacement_strategy == "specified_reference_constant"
            else None
        )
        closed_loop_provenance = _closed_loop_replacement_provenance(
            strategy=analysis.closed_loop.replacement_strategy,
            rollout=rollout,
            resolved_baselines=(
                baselines
                if analysis.closed_loop.replacement_strategy == "specified_reference_constant"
                else None
            ),
        )
        closed_loop_result = run_paired_closed_loop(
            env_factory=lambda: make_evaluation_env(
                seed=runtime["seed"],
                env_config=experiment.profile.evaluation_env_config,
            ),
            adapter=adapter,
            scenario_seeds=range(scenario_start, scenario_start + scenario_count),
            targets=selected,
            replacement_strategy=analysis.closed_loop.replacement_strategy,
            specified_reference=specified_reference,
            dataset_observations=rollout.observations,
            schema=schema,
            rl_seed=runtime["seed"],
        )
        save_closed_loop(directory, closed_loop_result)
    else:
        _write_disabled_closed_loop(directory)

    closed_loop_summary = (
        None if closed_loop_result is None else closed_loop_result.summary_rows
    )
    overview_perturbation_features = _full_episode_rows(perturbation_features)
    overview_perturbation_groups = _full_episode_rows(perturbation_groups)
    report_ig_features = _full_episode_rows(ig_features)
    report_ig_groups = _full_episode_rows(ig_groups)
    overview_ig_target = (
        str(analysis.integrated_gradients.targets[0])
        if ig_result is not None and analysis.integrated_gradients.targets
        else None
    )
    overview_ig_features = _ig_rows_for_target(report_ig_features, overview_ig_target)
    overview_ig_groups = _ig_rows_for_target(report_ig_groups, overview_ig_target)
    generate_standard_plots(
        directory,
        schema=schema,
        perturbation_feature_summary=overview_perturbation_features,
        perturbation_group_summary=overview_perturbation_groups,
        ig_feature_summary=overview_ig_features,
        ig_group_summary=overview_ig_groups,
        ig_attributions={"attributions": ig_matrix} if ig_matrix is not None else None,
        perturbation_step_values=(
            {"values": perturbation_matrix} if perturbation_matrix is not None else None
        ),
        steps=rollout.steps,
        closed_loop_summary=closed_loop_summary,
        ig_target_name=overview_ig_target,
    )
    return {
        "baselines": baselines,
        "perturbation": perturbation_result,
        "ig": ig_result,
        "perturbation_features": perturbation_features,
        "perturbation_groups": perturbation_groups,
        "ig_features": ig_features,
        "ig_groups": ig_groups,
        "overview_perturbation_features": overview_perturbation_features,
        "overview_perturbation_groups": overview_perturbation_groups,
        "overview_ig_features": overview_ig_features,
        "overview_ig_groups": overview_ig_groups,
        "report_ig_features": report_ig_features,
        "report_ig_groups": report_ig_groups,
        "overview_ig_target": overview_ig_target,
        "completeness": completeness,
        "closed_loop": closed_loop_result,
        "closed_loop_provenance": closed_loop_provenance,
    }


def _collect_command(args: argparse.Namespace) -> Path:
    runtime = _load_runtime(args, require_analysis=True)
    analysis = runtime["analysis"]
    if not analysis.run.deterministic:
        raise AttributionCLIError("initial collector supports deterministic=true only")
    from env_factory import make_evaluation_env

    started_at = utc_now_iso()

    def build(directory: Path) -> None:
        env = make_evaluation_env(
            seed=runtime["seed"],
            env_config=runtime["experiment"].profile.evaluation_env_config,
        )
        try:
            rollout = collect_rollout(
                env=env,
                adapter=runtime["adapter"],
                model=runtime["model"],
                schema=runtime["schema"],
                scenario_start=int(runtime["experiment"].profile.evaluation_env_config["start_seed"]),
                scenario_count=int(runtime["experiment"].profile.evaluation_env_config["num_scenarios"]),
                deterministic=True,
                rl_seed=runtime["seed"],
            )
        finally:
            env.close()
        save_rollout(
            directory,
            rollout,
            metadata=_metadata(
                args=args,
                runtime=runtime,
                rollout=rollout,
                started_at=started_at,
                finished_at=utc_now_iso(),
                runtime_contract=rollout.metadata.get("runtime_contract", {}),
            ),
        )
        write_expanded_schema_csv(safe_child(directory, "feature_schema_expanded.csv"), runtime["schema"])

    published, _result = _run_in_staging(runtime, args.output_prefix, build)
    return published


def _run_command(args: argparse.Namespace) -> Path:
    runtime = _load_runtime(args, require_analysis=True)
    analysis = runtime["analysis"]
    if not analysis.run.deterministic:
        raise AttributionCLIError("initial run supports deterministic=true only")
    from env_factory import make_evaluation_env

    started_at = utc_now_iso()

    def build(directory: Path) -> None:
        env = make_evaluation_env(
            seed=runtime["seed"],
            env_config=runtime["experiment"].profile.evaluation_env_config,
        )
        try:
            rollout = collect_rollout(
                env=env,
                adapter=runtime["adapter"],
                model=runtime["model"],
                schema=runtime["schema"],
                scenario_start=int(runtime["experiment"].profile.evaluation_env_config["start_seed"]),
                scenario_count=int(runtime["experiment"].profile.evaluation_env_config["num_scenarios"]),
                deterministic=True,
                rl_seed=runtime["seed"],
            )
        finally:
            env.close()
        write_expanded_schema_csv(safe_child(directory, "feature_schema_expanded.csv"), runtime["schema"])
        save_rollout(directory, rollout)
        artifacts = _analyze_rollout(
            directory=directory,
            runtime=runtime,
            rollout=rollout,
            allow_closed_loop=(
                bool(args.closed_loop)
                if args.closed_loop is not None
                else bool(analysis.closed_loop.enabled)
            ),
        )
        metadata = _metadata(
            args=args,
            runtime=runtime,
            rollout=rollout,
            started_at=started_at,
            finished_at=utc_now_iso(),
            runtime_contract=rollout.metadata.get("runtime_contract", {}),
            baseline_metadata=artifacts["baselines"].metadata,
            closed_loop_targets=(
                []
                if artifacts["closed_loop"] is None
                else [row["target_name"] for row in artifacts["closed_loop"].summary_rows]
            ),
            closed_loop_provenance=artifacts["closed_loop_provenance"],
        )
        atomic_write_json(safe_child(directory, "analysis_metadata.json"), metadata)
        # Re-write rollout metadata with run provenance after all analysis
        # succeeds.  Existing arrays/JSONL remain atomically written.
        save_rollout(directory, rollout, metadata=metadata)
        write_report(
            safe_child(directory, "report.md"),
            _report_markdown(
                metadata=metadata,
                schema=runtime["schema"],
                perturbation_features=artifacts["overview_perturbation_features"],
                perturbation_groups=artifacts["overview_perturbation_groups"],
                ig_features=artifacts["report_ig_features"],
                ig_groups=artifacts["report_ig_groups"],
                closed_loop=artifacts["closed_loop"],
            ),
        )

    published, _result = _run_in_staging(runtime, args.output_prefix, build)
    return published


def _analyze_command(args: argparse.Namespace) -> Path:
    """Offline command: no env factory import or MetaDrive construction here."""

    runtime = _load_runtime(args, require_analysis=True)
    if bool(args.closed_loop):
        raise AttributionCLIError("analyze is offline only; use closed-loop or run for environment validation")
    rollout_directory = _resolve_project_path(args.rollout)
    rollout = load_rollout(rollout_directory)
    _assert_offline_rollout_provenance(runtime=runtime, rollout=rollout)
    started_at = utc_now_iso()

    def build(directory: Path) -> None:
        write_expanded_schema_csv(safe_child(directory, "feature_schema_expanded.csv"), runtime["schema"])
        save_rollout(directory, rollout)
        artifacts = _analyze_rollout(
            directory=directory,
            runtime=runtime,
            rollout=rollout,
            allow_closed_loop=False,
        )
        metadata = _metadata(
            args=args,
            runtime=runtime,
            rollout=rollout,
            started_at=started_at,
            finished_at=utc_now_iso(),
            runtime_contract=rollout.metadata.get("runtime_contract", {}),
            baseline_metadata=artifacts["baselines"].metadata,
        )
        metadata["offline_source_rollout"] = str(rollout_directory)
        metadata["offline_source_rollout_metadata"] = json_value(dict(rollout.metadata))
        atomic_write_json(safe_child(directory, "analysis_metadata.json"), metadata)
        save_rollout(directory, rollout, metadata=metadata)
        write_report(
            safe_child(directory, "report.md"),
            _report_markdown(
                metadata=metadata,
                schema=runtime["schema"],
                perturbation_features=artifacts["overview_perturbation_features"],
                perturbation_groups=artifacts["overview_perturbation_groups"],
                ig_features=artifacts["report_ig_features"],
                ig_groups=artifacts["report_ig_groups"],
                closed_loop=None,
            ),
        )

    published, _result = _run_in_staging(runtime, args.output_prefix, build)
    return published


def _closed_loop_command(args: argparse.Namespace) -> Path:
    from input_attribution.visualization import generate_standard_plots

    runtime = _load_runtime(args, require_analysis=True)
    analysis = runtime["analysis"]
    schema = runtime["schema"]
    selectors: list[int] = []
    for selector in args.feature:
        selectors.append(_feature_index(schema, selector))
    groups = list(args.group)
    if not selectors and not groups:
        selectors.extend(_feature_index(schema, value) for value in analysis.closed_loop.explicit_features)
        groups.extend(analysis.closed_loop.explicit_groups)
    if not selectors and not groups:
        raise AttributionCLIError(
            "closed-loop requires --feature/--group or explicit selectors in [closed_loop]"
        )
    targets = schema_targets(schema, feature_indices=selectors, group_names=groups)
    rollout_directory = None if args.rollout is None else _resolve_project_path(args.rollout)
    rollout = None if rollout_directory is None else load_rollout(rollout_directory)
    if rollout is not None:
        _assert_offline_rollout_provenance(runtime=runtime, rollout=rollout)
    strategy = analysis.closed_loop.replacement_strategy
    if strategy in {"dataset_median_constant", "specified_reference_constant"} and rollout is None:
        raise AttributionCLIError(f"{strategy} requires --rollout")
    specified_reference = None
    replacement_provenance: dict[str, Any] = {
        "strategy": strategy,
        "reference_source": (
            "per_intervention_episode_reset"
            if strategy == "episode_start_constant"
            else "schema_declared_constants"
            if strategy == "schema_constant"
            else "saved_rollout"
        ),
    }
    if rollout is not None:
        assert rollout_directory is not None
        replacement_provenance["source_rollout"] = _source_rollout_record(
            rollout_directory, rollout
        )
    if strategy == "specified_reference_constant":
        assert rollout is not None
        from input_attribution.baselines import BaselineProvider

        # The analysis TOML's validated baseline strategy is the explicit
        # source of this constant.  Keep the first resolved reference visible
        # in metadata rather than inventing a zero/reference vector here.
        resolved = BaselineProvider(
            rollout.observations, rollout.episode_ids, rollout.steps
        ).resolve(analysis.baseline)
        specified_reference = resolved.values[0, 0]
        replacement_provenance["resolved_baseline"] = json_value(resolved.metadata)
        replacement_provenance["specified_reference_baseline_id"] = str(
            resolved.baseline_ids[0, 0]
        )
    elif strategy == "dataset_median_constant":
        assert rollout is not None
        replacement_provenance["dataset_median"] = {
            "source_row_count": rollout.row_count,
            "source_observation_dim": rollout.observation_dim,
        }
    runtime_contract = _validate_live_runtime(runtime)
    from env_factory import make_evaluation_env

    started_at = utc_now_iso()

    def build(directory: Path) -> None:
        result = run_paired_closed_loop(
            env_factory=lambda: make_evaluation_env(
                seed=runtime["seed"],
                env_config=runtime["experiment"].profile.evaluation_env_config,
            ),
            adapter=runtime["adapter"],
            scenario_seeds=range(
                int(runtime["experiment"].profile.evaluation_env_config["start_seed"]),
                int(runtime["experiment"].profile.evaluation_env_config["start_seed"])
                + int(runtime["experiment"].profile.evaluation_env_config["num_scenarios"]),
            ),
            targets=targets,
            replacement_strategy=strategy,
            specified_reference=specified_reference,
            dataset_observations=None if rollout is None else rollout.observations,
            schema=schema,
            rl_seed=runtime["seed"],
        )
        save_closed_loop(directory, result)
        write_expanded_schema_csv(safe_child(directory, "feature_schema_expanded.csv"), schema)
        generate_standard_plots(
            directory,
            schema=schema,
            closed_loop_summary=result.summary_rows,
        )
        metadata = _metadata(
            args=args,
            runtime=runtime,
            rollout=rollout,
            started_at=started_at,
            finished_at=utc_now_iso(),
            runtime_contract=runtime_contract,
            baseline_metadata=replacement_provenance,
            closed_loop_targets=[target.name for target in targets],
            closed_loop_provenance=replacement_provenance,
        )
        atomic_write_json(safe_child(directory, "analysis_metadata.json"), metadata)
        write_report(
            safe_child(directory, "report.md"),
            _report_markdown(
                metadata=metadata,
                schema=schema,
                perturbation_features=[],
                perturbation_groups=[],
                ig_features=[],
                ig_groups=[],
                closed_loop=result,
            ),
        )

    published, _result = _run_in_staging(runtime, args.output_prefix, build)
    return published


def _validate_schema_command(args: argparse.Namespace) -> int:
    runtime = _load_runtime(args, require_analysis=False)
    contract = _validate_live_runtime(runtime)
    print(json.dumps(json_value({"status": "valid", **contract}), ensure_ascii=False, sort_keys=True))
    return 0


def _generic_schema_template(dimension: int) -> str:
    return f'''# Generic unresolved template. Replace every placeholder after source verification.
schema_version = 1
name = "custom_{dimension}_template"
observation_dim = {dimension}

[[ranges]]
name = "unresolved_features"
start = 0
count = {dimension}
feature_name_template = "UNRESOLVED_feature_{{index:03d}}"
kind = "custom"
group = "unresolved_features"
resolved = false
description = "Fill in verified feature names, meanings, groups, and ranges before analysis."
'''


def _schema_template_command(args: argparse.Namespace) -> Path:
    output = _resolve_project_path(args.output)
    if output.suffix.lower() != ".toml":
        raise AttributionCLIError("schema template output must have a .toml suffix")
    if output.is_symlink():
        raise AttributionCLIError(f"refusing to replace symlink schema template: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output, _generic_schema_template(args.dim))
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "schema-template":
            print(_schema_template_command(args))
            return 0
        if args.command == "validate-schema":
            return _validate_schema_command(args)
        if args.command == "collect":
            print(_collect_command(args))
            return 0
        if args.command == "run":
            print(_run_command(args))
            return 0
        if args.command == "analyze":
            print(_analyze_command(args))
            return 0
        if args.command == "closed-loop":
            print(_closed_loop_command(args))
            return 0
        raise AssertionError(f"unhandled command: {args.command}")
    except (AttributionCLIError, ArtifactError, RolloutError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
