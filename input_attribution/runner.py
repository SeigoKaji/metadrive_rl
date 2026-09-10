"""Small orchestration layer for check, run, offline, and report commands."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from numbers import Integral
import tempfile
from typing import Any, Mapping, Sequence

from .collection import collect_baseline, collect_closed_loop
from .config import AttributionConfig, ConfigError, load_config, resolve_path
from .policy_comparison import compare_saved_baseline, policy_identity
from .interventions import coerce_intervention, InterventionError
from .storage import (
    BASE_MAIN_SHA,
    FORMAT_VERSION,
    ensure_run_layout,
    implementation_identity,
    read_manifest,
    read_rollout,
    write_manifest,
    write_rollout,
)
from .synthetic import SyntheticAdapter


class RunnerError(RuntimeError):
    """A requested command cannot be completed safely."""


@dataclass(frozen=True, slots=True)
class RunResult:
    run_dir: Path
    status: str
    baseline: Mapping[str, object]
    patterns: tuple[Mapping[str, object], ...]
    offline: tuple[Mapping[str, object], ...]
    report: Mapping[str, object] | None = None
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status == "complete" and not self.errors

    def as_dict(self) -> dict[str, object]:
        return {
            "run_dir": str(self.run_dir),
            "status": self.status,
            "baseline": dict(self.baseline),
            "patterns": [dict(item) for item in self.patterns],
            "offline": [dict(item) for item in self.offline],
            "report": dict(self.report) if isinstance(self.report, Mapping) else self.report,
            "errors": list(self.errors),
        }


def _mapping(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return {str(key): nested for key, nested in value.items()}
    return {}


def _environment_mapping(config: AttributionConfig) -> dict[str, object] | None:
    value = config.environment_config
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    path = resolve_path(value, config=config)
    if path.suffix.lower() != ".toml" or not path.is_file():
        raise RunnerError(f"environment_config file does not exist: {path}")
    # The host repository's TOML is an experiment bundle.  Resolve it through
    # the existing loader so ``environment.common`` and the evaluation stage
    # are merged exactly once, retaining the canonical seed=5/one-scenario
    # profile semantics.
    try:
        from configs.experiment_config import select_experiment

        selected = select_experiment(config_path=path)
        return dict(selected.profile.evaluation_env_config)
    except Exception as error:
        if path.parent.name == "configs":
            raise RunnerError(f"cannot resolve host experiment config {path}: {error}") from error
    try:
        import tomllib

        with path.open("rb") as stream:
            loaded = tomllib.load(stream)
    except Exception as error:
        raise RunnerError(f"cannot read environment_config {path}: {error}") from error
    if not isinstance(loaded, Mapping):
        raise RunnerError("environment_config TOML must contain a table")
    # Existing host configs commonly wrap stage values under ``environment``;
    # pass the full mapping to the host adapter, which owns stage selection.
    return dict(loaded)


def _schema_object(config: AttributionConfig, *, expected_dimension: int | None = None) -> object | None:
    if config.schema is None:
        return None
    try:
        from .schema import load_schema

        source: object = config.schema
        # Schema aliases (for example ``standard_259``) are resolved by the
        # schema module.  A file path, however, follows the same repo-root
        # rule as model/environment paths when the TOML config is elsewhere.
        if isinstance(source, str):
            candidate = Path(source).expanduser()
            if not candidate.is_absolute():
                anchored = resolve_path(source, config=config)
                if anchored.is_file():
                    source = anchored
        return load_schema(source, dimension=expected_dimension)
    except Exception as error:
        raise RunnerError(f"cannot load input schema: {error}") from error


def _adapter_dimension(adapter: object) -> int | None:
    for name in ("dimension", "observation_dim", "object_dimension"):
        try:
            value = getattr(adapter, name)
        except Exception:
            continue
        if isinstance(value, Integral) and not isinstance(value, bool):
            return int(value)
    return None


def build_adapter(config: AttributionConfig) -> object:
    """Construct exactly one backend adapter without eager simulator imports."""

    if config.backend == "synthetic":
        # An explicit synthetic schema still guards D and gives the run a
        # portable feature definition; a schema-free synthetic demo remains
        # useful for arbitrary dimensions with explicitly listed patterns.
        _schema_object(config, expected_dimension=config.synthetic_dimension)
        return SyntheticAdapter(
            dimension=config.synthetic_dimension,
            action_count=config.synthetic_action_count,
            steps=config.synthetic_steps,
            natural_end_step=config.synthetic_natural_end_step,
        )
    if config.backend != "metadrive":  # validated config normally prevents this
        raise RunnerError(f"unsupported backend: {config.backend}")
    try:
        from .adapter import InputAttributionAdapter
    except (ImportError, ModuleNotFoundError) as error:
        raise RunnerError(f"MetaDrive adapter is unavailable: {error}") from error
    model_path = None if config.model_path is None else resolve_path(config.model_path, config=config)
    environment = _environment_mapping(config)
    # A schema path/name is resolved by schema.load_schema; model dimension is
    # checked by InputAttributionAdapter after lazy PPO loading.
    schema_rows = _schema_object(config)
    if isinstance(schema_rows, Sequence) and not isinstance(schema_rows, (str, bytes, Mapping)):
        schema: object = {"dimension": len(schema_rows), "features": list(schema_rows)}
    else:
        schema = schema_rows
    kwargs: dict[str, object] = {
        "model_path": model_path,
        "env_config": environment,
        "schema": schema,
        "project_root": config.repo_root,
        "deterministic": config.deterministic,
    }
    adapter = InputAttributionAdapter(**kwargs)
    dimension = _adapter_dimension(adapter)
    if dimension is None:
        raise RunnerError("real adapter did not expose a verified observation dimension")
    try:
        for pattern in config.patterns:
            coerce_intervention(pattern, dimension=dimension)
    except InterventionError as error:
        raise RunnerError(f"pattern does not match verified adapter dimension {dimension}: {error}") from error
    return adapter


def _pattern_mapping(pattern: object) -> dict[str, object]:
    if isinstance(pattern, Mapping):
        return {str(key): value for key, value in pattern.items()}
    return {
        "id": getattr(pattern, "id", "unknown"),
        "name": getattr(pattern, "name", "unknown"),
        "indices": list(getattr(pattern, "indices", ())),
        "fixed_value": getattr(pattern, "fixed_value", -1.0),
    }


def _new_run_dir(config: AttributionConfig, requested: str | Path | None = None) -> Path:
    base = resolve_path(requested or config.output_dir, config=config)
    if requested is not None:
        if base.exists():
            raise RunnerError(f"refusing to reuse existing run directory: {base}")
        base.mkdir(parents=True, exist_ok=False)
        return ensure_run_layout(base)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = base / f"run-{stamp}-{os.getpid()}"
    suffix = 0
    candidate = root
    while candidate.exists():
        suffix += 1
        candidate = base / f"run-{stamp}-{os.getpid()}-{suffix}"
    return ensure_run_layout(candidate)


def _manifest(config: AttributionConfig, adapter: object, run_dir: Path) -> dict[str, object]:
    implementation = implementation_identity(config.repo_root)
    identity = policy_identity(adapter)
    metadata_method = getattr(adapter, "model_metadata", None)
    adapter_metadata = metadata_method() if callable(metadata_method) else getattr(adapter, "metadata", identity)
    if not isinstance(adapter_metadata, Mapping):
        adapter_metadata = identity
    return {
        "format_version": FORMAT_VERSION,
        # The run's provenance must not be spoofed by a user config value.
        # Keep the requested value in ``config`` for diagnosis, while this
        # top-level field always records the package's immutable base.
        "base_main_sha": BASE_MAIN_SHA,
        "implementation_sha": implementation.get("git_head"),
        "implementation_source_sha256": implementation.get("source_sha256"),
        "dirty": implementation.get("dirty", True),
        "implementation": implementation,
        "backend": config.backend,
        "config": config.to_dict(),
        "adapter": dict(adapter_metadata),
        "adapter_identity": identity,
        "patterns": [_pattern_mapping(pattern) for pattern in config.patterns],
        "status": "running",
        "run_dir": str(run_dir),
    }


def _report(run_dir: Path) -> dict[str, object]:
    """Generate derived artifacts lazily from saved data."""

    try:
        from .reporting import generate_report

        value = generate_report(run_dir)
    except Exception as error:
        return {"status": "failed", "errors": [f"{type(error).__name__}: {error}"]}
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return value.as_dict()
    if isinstance(value, Mapping):
        return dict(value)
    return {"status": "success", "result": str(value)}


def _compact_rollout(value: Mapping[str, object] | None) -> dict[str, object] | None:
    """Keep CLI output small while raw JSON retains every observation/probability."""

    if value is None:
        return None
    records = value.get("records")
    return {
        "id": value.get("id"),
        "name": value.get("name"),
        "status": value.get("status"),
        "comparable": value.get("comparable"),
        "steps": len(records) if isinstance(records, list) else 0,
        "applied_count": value.get("applied_count", 0),
        "changed_count": value.get("changed_count", 0),
        "unchanged_count": value.get("unchanged_count", 0),
        "termination": value.get("termination"),
        "failure": value.get("failure"),
    }


def _compact_run(result: RunResult) -> dict[str, object]:
    report_path = None
    if isinstance(result.report, Mapping):
        report_path = result.report.get("report_html")
        if report_path is None:
            output_dir = result.report.get("output_dir")
            if output_dir:
                report_path = str(Path(str(output_dir)) / "report.html")
    return {
        "run_dir": str(result.run_dir),
        "status": result.status,
        "baseline": _compact_rollout(result.baseline),
        "patterns": [_compact_rollout(item) for item in result.patterns],
        "offline": [_compact_rollout(item) for item in result.offline],
        "report": (
            {
                "status": result.report.get("status"),
                "report_path": report_path,
                "errors": result.report.get("errors", []),
            }
            if isinstance(result.report, Mapping)
            else None
        ),
        "errors": list(result.errors),
    }


def _compact_report(value: Mapping[str, object]) -> dict[str, object]:
    """Keep report CLI output to status/path/counts; details stay on disk."""

    files = value.get("files")
    report_path = value.get("report_html")
    if report_path is None and value.get("output_dir"):
        report_path = str(Path(str(value["output_dir"])) / "report.html")
    return {
        "status": value.get("status"),
        "run_dir": value.get("run_dir"),
        "report_path": report_path,
        "files_count": len(files) if isinstance(files, Sequence) and not isinstance(files, (str, bytes)) else None,
        "errors": value.get("errors", []),
    }


def _pair_rollouts(
    baseline: Mapping[str, object],
    changed: Mapping[str, object],
) -> dict[str, object]:
    """Compare initial conditions and model/reward boundaries for A/B pairing."""

    reasons: list[str] = []
    if baseline.get("status") != "complete":
        reasons.append("baseline is not complete")
    if changed.get("status") != "complete":
        reasons.append("closed-loop run is not complete")
    baseline_initial = baseline.get("initial")
    changed_initial = changed.get("initial")
    if not isinstance(baseline_initial, Mapping) or not isinstance(changed_initial, Mapping):
        reasons.append("initial condition metadata is missing")
    else:
        if not baseline_initial.get("seed_verified") or not changed_initial.get("seed_verified"):
            reasons.append("scenario seed was not verified")
        if baseline_initial.get("seed") != changed_initial.get("seed"):
            reasons.append("actual scenario seeds differ")
        if baseline_initial.get("observation_hash") != changed_initial.get("observation_hash"):
            reasons.append("initial observations differ")
        if baseline_initial.get("state") != changed_initial.get("state"):
            reasons.append("available initial states differ")
    baseline_identity = baseline.get("adapter_identity")
    changed_identity = changed.get("adapter_identity")
    if not isinstance(baseline_identity, Mapping) or not isinstance(changed_identity, Mapping):
        reasons.append("model/preprocessing identity is missing")
    elif dict(baseline_identity) != dict(changed_identity):
        reasons.append("model/preprocessing identities differ")
    baseline_dt = baseline.get("action_dt")
    changed_dt = changed.get("action_dt")
    if baseline_dt is None or changed_dt is None:
        reasons.append("action duration is unavailable")
    else:
        try:
            if not math.isclose(float(baseline_dt), float(changed_dt), rel_tol=1e-9, abs_tol=1e-12):
                reasons.append("action durations differ")
        except (TypeError, ValueError):
            reasons.append("action duration is invalid")
    # Reward-term availability (verified/unavailable/mismatch) is a warning
    # about telemetry, not evidence that the two reward boundaries differ.
    # Pair only on the explicit stage at which env.step returned the reward.
    baseline_stage = baseline.get("reward_stage")
    changed_stage = changed.get("reward_stage")
    if not isinstance(baseline_stage, str) or not baseline_stage.strip():
        reasons.append("baseline reward stage is unavailable")
    if not isinstance(changed_stage, str) or not changed_stage.strip():
        reasons.append("closed-loop reward stage is unavailable")
    if (
        isinstance(baseline_stage, str)
        and isinstance(changed_stage, str)
        and baseline_stage != changed_stage
    ):
        reasons.append("reward stages differ")
    return {
        "status": "comparable" if not reasons else "incomparable",
        "comparable": not reasons,
        "reasons": reasons,
    }


def _offline_failure(
    pattern: object,
    *,
    stage: str,
    error: str,
    status: str = "failed",
) -> dict[str, object]:
    """Build a visible per-pattern ①-A failure artifact."""

    metadata = _pattern_mapping(pattern)
    return {
        "id": str(metadata.get("id", "unknown")),
        "name": str(metadata.get("name", "unknown")),
        "pattern": metadata,
        "status": status,
        "records": [],
        "failure": {"stage": stage, "step": 0, "error": error},
        "comparable": False,
        "applied_count": 0,
        "changed_count": 0,
        "unchanged_count": 0,
    }


def run_experiment(
    config: AttributionConfig | Mapping[str, object] | str | Path,
    *,
    run_dir: str | Path | None = None,
    report: bool = True,
) -> RunResult:
    """Run baseline, saved-input A, and independent closed-loop B episodes."""

    if not isinstance(config, AttributionConfig):
        if isinstance(config, Mapping):
            config = AttributionConfig.from_mapping(config)
        else:
            config = load_config(config)
    adapter = build_adapter(config)
    destination = _new_run_dir(config, run_dir)
    manifest = _manifest(config, adapter, destination)
    write_manifest(destination, manifest)
    if config.schema is not None:
        try:
            from .schema import write_schema_csv

            schema_rows = _schema_object(config, expected_dimension=_adapter_dimension(adapter))
            if schema_rows is not None:
                write_schema_csv(destination / "data" / "input_schema.csv", schema_rows)
        except Exception as error:
            errors = [f"schema export failed: {type(error).__name__}: {error}"]
        else:
            errors = []
    else:
        errors = []

    baseline = collect_baseline(
        adapter,
        scenario_seed=config.scenario_seed,
        policy_seed=config.policy_seed,
        horizon=config.horizon,
        run_dir=str(destination),
        record_gif=config.record_gif,
        deterministic=config.deterministic,
        reward_atol=config.reward_atol,
        reward_rtol=config.reward_rtol,
        strict_reward_terms=config.strict_reward_terms,
    )
    write_rollout(destination, baseline)
    if baseline.get("status") != "complete":
        errors.append("baseline collection failed")

    offline_values: list[Mapping[str, object]] = []
    if baseline.get("status") == "complete":
        seed_policy = getattr(adapter, "seed_policy", None)
        if callable(seed_policy):
            try:
                seed_policy(config.policy_seed)
            except Exception as error:
                errors.append(f"offline policy seed failed: {type(error).__name__}: {error}")
        try:
            offline_values = compare_saved_baseline(
                adapter,
                baseline,
                config.patterns,
                deterministic=config.deterministic,
            )
        except Exception as error:
            errors.append(
                f"offline comparison failed: {type(error).__name__}: {error}"
            )
            offline_values = [
                _offline_failure(
                    pattern,
                    stage="offline_validation",
                    error=f"{type(error).__name__}: {error}",
                )
                for pattern in config.patterns
            ]
        for value in offline_values:
            write_rollout(destination, value, offline=True)
            if value.get("status") != "complete":
                errors.append(f"offline {value.get('id', 'pattern')} failed")
    else:
        # Preserve an explicit not_run file for every requested pattern so a
        # partial experiment cannot be mistaken for an empty successful run.
        for pattern in config.patterns:
            value = _offline_failure(
                pattern,
                stage="baseline",
                error="baseline failed",
                status="not_run",
            )
            offline_values.append(value)
            write_rollout(destination, value, offline=True)

    pattern_values: list[Mapping[str, object]] = []
    for pattern in config.patterns:
        value = collect_closed_loop(
            adapter,
            pattern,
            scenario_seed=config.scenario_seed,
            policy_seed=config.policy_seed,
            horizon=config.horizon,
            run_dir=str(destination),
            record_gif=config.record_gif,
            deterministic=config.deterministic,
            reward_atol=config.reward_atol,
            reward_rtol=config.reward_rtol,
            strict_reward_terms=config.strict_reward_terms,
        )
        # A/B comparison is meaningful only after matching the actual initial
        # state and the adapter/model/reward boundaries.
        if isinstance(value, dict):
            value["comparison"] = _pair_rollouts(baseline, value)
            value["comparable"] = bool(
                value.get("comparable") and value["comparison"].get("comparable")
            )
        pattern_values.append(value)
        write_rollout(destination, value)
        if value.get("status") != "complete":
            errors.append(f"pattern {value.get('id', 'unknown')} failed")

    raw_complete = baseline.get("status") == "complete" and all(
        value.get("status") == "complete" for value in pattern_values
    ) and all(value.get("status") == "complete" for value in offline_values)
    manifest["baseline_result"] = _compact_rollout(baseline)
    manifest["pattern_results"] = [_compact_rollout(value) for value in pattern_values]
    # Refresh after model loading and the environment probes.  Keep a detached
    # metadata snapshot so the manifest retains actual spaces/factory details.
    metadata_method = getattr(adapter, "model_metadata", None)
    if callable(metadata_method):
        try:
            metadata = metadata_method()
            if isinstance(metadata, Mapping):
                manifest["adapter"] = dict(metadata)
        except Exception as error:
            errors.append(f"adapter metadata refresh failed: {type(error).__name__}: {error}")
    manifest["adapter_identity"] = policy_identity(adapter)
    manifest["status"] = "complete" if raw_complete and not errors else "partial" if pattern_values or offline_values else "failed"
    manifest["errors"] = errors
    write_manifest(destination, manifest)

    report_result: dict[str, object] | None = None
    if report:
        report_result = _report(destination)
        report_status = str(report_result.get("status", "failed"))
        if report_status not in {"success", "complete"}:
            report_errors = report_result.get("errors")
            if isinstance(report_errors, Sequence) and not isinstance(report_errors, (str, bytes)) and report_errors:
                errors.extend(f"report: {error}" for error in report_errors)
            else:
                errors.append(f"report generation {report_status}")

    status = "complete" if raw_complete and not errors else "partial" if pattern_values or offline_values else "failed"
    manifest["status"] = status
    manifest["errors"] = errors
    write_manifest(destination, manifest)
    return RunResult(destination, status, baseline, tuple(pattern_values), tuple(offline_values), report_result, tuple(errors))


def check_config(config: AttributionConfig | Mapping[str, object] | str | Path) -> dict[str, object]:
    """Perform a one-step contract smoke check without a full experiment."""

    if not isinstance(config, AttributionConfig):
        config = AttributionConfig.from_mapping(config) if isinstance(config, Mapping) else load_config(config)
    try:
        adapter = build_adapter(config)
        dimension = _adapter_dimension(adapter)
        action_count = getattr(adapter, "action_count", None)
        if dimension is None or not isinstance(action_count, Integral) or isinstance(action_count, bool):
            raise RunnerError("adapter did not expose dimension/action_count")
        action_count = int(action_count)
        # Run one environment/frame probe in a private temporary directory.
        # This checks render availability when recording is requested without
        # leaving check artifacts in the configured experiment output.
        temporary_parent = Path(config.repo_root or Path.cwd()).expanduser().resolve()
        probe_horizon = max(1, min(2, config.horizon))
        with tempfile.TemporaryDirectory(
            dir=temporary_parent,
            prefix=".input_attribution_check-",
        ) as temporary_run:
            probe = collect_baseline(
                adapter,
                scenario_seed=config.scenario_seed,
                policy_seed=config.policy_seed,
                horizon=probe_horizon,
                run_dir=temporary_run,
                record_gif=config.record_gif,
                deterministic=config.deterministic,
                reward_atol=config.reward_atol,
                reward_rtol=config.reward_rtol,
                strict_reward_terms=config.strict_reward_terms,
            )
            intervention_probe = None
            if config.patterns:
                intervention_probe = collect_closed_loop(
                    adapter,
                    config.patterns[0],
                    scenario_seed=config.scenario_seed,
                    policy_seed=config.policy_seed,
                    horizon=probe_horizon,
                    run_dir=temporary_run,
                    record_gif=config.record_gif,
                    deterministic=config.deterministic,
                    reward_atol=config.reward_atol,
                    reward_rtol=config.reward_rtol,
                    strict_reward_terms=config.strict_reward_terms,
                )
        status = "complete" if probe.get("status") == "complete" and (intervention_probe is None or intervention_probe.get("status") == "complete") else "failed"
        def _reward_connection(value: Mapping[str, object] | None) -> dict[str, object]:
            if value is None:
                return {"statuses": [], "verified_steps": 0}
            records = value.get("records")
            statuses: set[str] = set()
            verified_steps = 0
            if isinstance(records, list):
                for record in records:
                    if not isinstance(record, Mapping):
                        continue
                    terms = record.get("reward_terms")
                    if isinstance(terms, Mapping):
                        current = str(terms.get("status", "unknown"))
                        statuses.add(current)
                        if current == "verified":
                            verified_steps += 1
            return {"statuses": sorted(statuses), "verified_steps": verified_steps}

        mlp_probe = getattr(adapter, "last_mlp_inputs", None)
        mlp_capture_count = len(mlp_probe) if isinstance(mlp_probe, Sequence) else None
        return {
            "status": status,
            "backend": config.backend,
            "dimension": dimension,
            "action_count": action_count,
            "adapter": policy_identity(adapter),
            "probe": probe,
            "intervention_probe": intervention_probe,
            "reward_connection": {
                "strict": config.strict_reward_terms,
                "baseline": _reward_connection(probe),
                "intervention": _reward_connection(intervention_probe),
            },
            "input_boundary": {
                "mlp_probe_capture_count": mlp_capture_count,
                "render_probe_requested": config.record_gif,
            },
            "errors": [] if status == "complete" else [probe.get("failure"), intervention_probe.get("failure") if isinstance(intervention_probe, Mapping) else None],
        }
    except Exception as error:
        return {"status": "failed", "errors": [f"{type(error).__name__}: {error}"]}


def offline_experiment(run_dir: str | Path) -> dict[str, object]:
    """Re-run ①-A for an existing synthetic/configured run."""

    root = Path(run_dir).expanduser().resolve()
    manifest = read_manifest(root)
    config_value = manifest.get("config")
    if not isinstance(config_value, Mapping):
        raise RunnerError("manifest has no embedded config for offline rerun")
    config = AttributionConfig.from_mapping(config_value)
    adapter = build_adapter(config)
    seed_policy = getattr(adapter, "seed_policy", None)
    if callable(seed_policy):
        seed_policy(config.policy_seed)
    baseline = read_rollout(root, "P00")
    values = compare_saved_baseline(adapter, baseline, config.patterns, deterministic=config.deterministic)
    for value in values:
        write_rollout(root, value, offline=True)
    return {"status": "complete" if all(value.get("status") == "complete" for value in values) else "partial", "patterns": values}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m input_attribution", description="Fixed-value input attribution")
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check", help="validate adapter/config with a one- or two-step probe")
    check.add_argument("--config", required=True, type=Path)
    run = subparsers.add_parser("run", help="collect baseline, ①-A, and ①-B artifacts")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--run-dir", type=Path)
    run.add_argument("--no-report", action="store_true")
    report = subparsers.add_parser("report", help="regenerate report from saved run data only")
    report.add_argument("--run-dir", required=True, type=Path)
    offline = subparsers.add_parser("offline", help="rerun saved-observation ①-A only")
    offline.add_argument("--run-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            value = check_config(args.config)
            compact = dict(value)
            compact["probe"] = _compact_rollout(value.get("probe") if isinstance(value.get("probe"), Mapping) else None)
            compact["intervention_probe"] = _compact_rollout(value.get("intervention_probe") if isinstance(value.get("intervention_probe"), Mapping) else None)
            print(json.dumps(compact, ensure_ascii=False, indent=2, default=str))
            return 0 if value.get("status") == "complete" else 1
        if args.command == "run":
            value = run_experiment(args.config, run_dir=args.run_dir, report=not args.no_report)
            print(json.dumps(_compact_run(value), ensure_ascii=False, indent=2, default=str))
            return 0 if value.ok else 1
        if args.command == "offline":
            value = offline_experiment(args.run_dir)
            compact = dict(value)
            compact["patterns"] = [
                _compact_rollout(item) if isinstance(item, Mapping) else item
                for item in value.get("patterns", [])
            ]
            print(json.dumps(compact, ensure_ascii=False, indent=2, default=str))
            return 0 if value.get("status") == "complete" else 1
        if args.command == "report":
            value = _report(args.run_dir)
            print(json.dumps(_compact_report(value), ensure_ascii=False, indent=2, default=str))
            return 0 if value.get("status") in {"success", "complete"} else 1
    except (ConfigError, RunnerError, OSError, ValueError) as error:
        print(f"input_attribution: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    return 2


__all__ = [
    "RunResult",
    "RunnerError",
    "build_adapter",
    "build_parser",
    "check_config",
    "main",
    "offline_experiment",
    "run_experiment",
]
