"""初心者向けのcheck/runと、保存結果からの部分実行CLI。"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import importlib
import inspect
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from .artifacts import (
    build_manifest,
    copy_reference,
    create_run,
    jsonl_read,
    open_run,
    schema_semantics_hash,
    schema_semantics_hash_from_value,
    verify_manifest_compatibility,
    seal_reference, verify_reference, sha256_file, sha256_object,
)
from .checks import CheckResult, format_check, run_check, resolve_pattern_listing
from .closed_loop import run_closed_loop
from .collection import collect_reference
from .config import AnalysisConfig, ConfigError, load_config


class CLIError(RuntimeError):
    """CLI実行を継続できない設定・依存エラー。"""


def _load_attr(spec: str) -> Any:
    if ":" in spec:
        module_name, attribute = spec.split(":", 1)
    else:
        module_name, separator, attribute = spec.rpartition(".")
        if not separator:
            raise CLIError(f"module:attribute形式が必要です: {spec!r}")
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise CLIError(f"component moduleをimportできません: {module_name}: {exc}") from exc
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise CLIError(f"component attributeがありません: {spec}") from exc


def _construct(factory: Any, config: AnalysisConfig) -> Any:
    candidates = []
    for name in ("from_analysis_config", "from_config"):
        hook = getattr(factory, name, None)
        if callable(hook):
            candidates.append((hook, ((config,), {})))
    if callable(factory):
        component_kwargs = config.adapter.get("config", {})
        if isinstance(component_kwargs, Mapping):
            candidates.append((factory, ((), dict(component_kwargs))))
        candidates.extend(
            [
                (factory, ((), {"config": config})),
                (factory, ((config,), {})),
                (factory, ((), {})),
            ]
        )
    if not candidates:
        return factory
    import inspect

    for hook, (args, kwargs) in candidates:
        try:
            inspect.signature(hook).bind(*args, **kwargs)
        except (TypeError, ValueError):
            continue
        return hook(*args, **kwargs)
    raise CLIError(f"component factoryがAnalysisConfigを受け取れません: {factory!r}")


def load_components(config: AnalysisConfig) -> tuple[Any, Any, Any]:
    from .schema import load_schema
    if config.schema_path is None:
        raise CLIError("schema.path is required; input order is never inferred")
    schema = load_schema(config.schema_path)
    spec = config.adapter.get("factory") or config.adapter.get("class") or config.adapter.get("module")
    if not spec:
        raise CLIError("adapter.factory must identify a small connection adapter")
    adapter = _construct(_load_attr(str(spec)), config)
    if not callable(getattr(adapter, "load_policy", None)):
        raise CLIError("adapter must implement load_policy(config) without creating an environment")
    policy = adapter.load_policy(config)
    return adapter, policy, schema


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()} for row in rows)


def _all_patterns(config: AnalysisConfig, schema: Any) -> list[Any]:
    from .interventions import generate_individual_patterns, patterns_from_config
    patterns = list(patterns_from_config(config.patterns))
    if config.analysis.get("individual_inputs", True):
        patterns.extend(generate_individual_patterns(schema, reference_id=config.analysis["reference_id"]))
    ids = [p.pattern_id for p in patterns]
    if len(ids) != len(set(ids)):
        raise CLIError("configured and individual pattern IDs overlap")
    return patterns


def _pattern_index_value(
    value: Any,
    schema: Any,
    index: int,
    indices: Sequence[int],
) -> Any:
    """Resolve a scalar, target-ordered sequence, or input-keyed mapping."""

    if isinstance(value, Mapping):
        spec = schema.spec(index)
        for key in (index, str(index), getattr(spec, "id", None)):
            if key is not None and key in value:
                return value[key]
        if len(value) == 1:
            return next(iter(value.values()))
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
        position = list(indices).index(index)
        return values[position] if position < len(values) else None
    return value


def _pattern_variant_metadata(pattern: Any, schema: Any) -> dict[str, Any]:
    """Persist typed variant values/evidence beside the offline pattern row.

    The execution layer resolves these values again for every observation.  A
    compact resolution record here makes ``patterns.json`` and the report
    auditable without pretending that a reflection has one fixed replacement
    value: its value is ``2 * center - source_value`` at each step.
    """

    indices = tuple(pattern.resolve_indices(schema))
    if not indices:
        return {}
    records: dict[str, Mapping[str, Any]] = {}
    for index in indices:
        record: Mapping[str, Any] | None = None
        if isinstance(getattr(pattern, "variant", None), Mapping):
            record = pattern.variant
        else:
            variant_id = getattr(pattern, "variant_id", None)
            if variant_id is None and isinstance(getattr(pattern, "metadata", None), Mapping):
                variant_id = pattern.metadata.get("variant_id")
            if variant_id:
                record = schema.variant(index, str(variant_id))
        if record is not None:
            records[str(index)] = dict(record)

    metadata: dict[str, Any] = {}
    if not records:
        return metadata
    metadata["variant_ids"] = {
        key: record.get("id", getattr(pattern, "variant_id", None))
        for key, record in records.items()
    }
    metadata["variant_evidence"] = {
        key: record.get(
            "evidence",
            record.get("source", record.get("reference", record.get("provenance"))),
        )
        for key, record in records.items()
    }
    metadata["variant_classification"] = {
        key: record.get(
            "classification",
            record.get("class", record.get("replacement_kind", record.get("operation"))),
        )
        for key, record in records.items()
    }
    metadata["variant_records"] = records

    method = str(getattr(pattern, "method", "")).casefold()
    resolved: dict[str, Any] = {}
    if method == "reflection":
        for key, record in records.items():
            center = record.get("centers", record.get("center"))
            index = int(key)
            resolved[key] = {
                "center": _pattern_index_value(center, schema, index, indices),
                "expression": "2 * center - source_value",
            }
        metadata["resolved_expression"] = "2 * center - source_value"
    elif method in {"neutral", "fixed_level", "fixed"}:
        for key, record in records.items():
            raw = record.get("values", record.get("levels"))
            if raw is None:
                raw = record.get("level", record.get("value"))
            index = int(key)
            if raw is not None:
                resolved[key] = _pattern_index_value(raw, schema, index, indices)
    if not resolved and getattr(pattern, "values", None) is not None:
        for index in indices:
            resolved[str(index)] = _pattern_index_value(pattern.values, schema, index, indices)
    if resolved:
        metadata["resolved_values"] = resolved
    return metadata


def _new_store(config: AnalysisConfig, command: str, adapter: Any, policy: Any, schema: Any, check: CheckResult):
    store = create_run(
        config.output_root,
        config.experiment_name,
        config.model_name,
        run_id=config.output.get("run_id"),
    )
    store.save_resolved_config(config)
    store.save_manifest(
        build_manifest(
            config=config,
            command=command,
            model_path=config.model_path,
            schema_path=config.schema_path,
            patterns=config.patterns,
            preprocess=config.preprocess,
        )
    )
    manifest = store.load_manifest()
    source = Path(inspect.getfile(type(adapter))).resolve()
    sources = [source]
    hook = getattr(adapter, "source_paths", None)
    if callable(hook):
        sources.extend(Path(path).resolve() for path in hook())
    manifest.update(adapter_sources={str(path): sha256_file(path) for path in sources},
        policy_fingerprint=policy.fingerprint(), runtime=check.checks.get("packages", {}),
        environment_resolved=getattr(adapter, "env_config", None),
        experiment=config.experiment_name, model_name=config.model_name,
        run_id=store.run_id, optional_ig="not executed by run; use the ig command explicitly")
    store.save_manifest(manifest)
    store.write_json("input_schema.json", schema.to_dict())
    _write_csv(store.run_dir / "input_schema.csv", schema.to_dict()["inputs"])
    listing = []
    for pattern in _all_patterns(config, schema):
        row = pattern.to_dict() | {
            "indices": list(pattern.resolve_indices(schema)),
            "names_ja": [schema.spec(i).name_ja for i in pattern.resolve_indices(schema)],
        }
        row.update(_pattern_variant_metadata(pattern, schema))
        listing.append(row)
    store.write_json("patterns.json", listing)
    _write_csv(store.run_dir / "patterns.csv", listing)
    store.write_json("check.json", check.as_dict())
    manifest = store.load_manifest()
    manifest["snapshot_files"] = {name: sha256_file(store.run_dir / name) for name in ("input_schema.json", "patterns.json", "check.json")}
    store.save_manifest(manifest)
    store.update_status("ig", "skipped", reason="任意実行。igコマンドで対象時刻と実観測基準を指定してください")
    return store


def _load_observations(store: Any) -> tuple[np.ndarray, list[dict[str, Any]]]:
    path = store.reference_dir / "observations.npy"
    if not path.is_file():
        raise CLIError(f"reference observations are missing: {path}")
    value = np.load(path, allow_pickle=False)
    observations = np.array(value, copy=True)
    records = jsonl_read(store.reference_dir / "records.jsonl")
    if observations.ndim != 2 or observations.shape[0] != len(records):
        raise CLIError("reference observations and records have inconsistent lengths")
    return observations, records


def _run_offline(store: Any, config: AnalysisConfig, policy: Any, schema: Any) -> Any:
    from .offline import run_offline
    verify_reference(store)
    analysis = store.for_analysis("offline")
    analysis.update_status("offline", "running")
    try:
        observations, records = _load_observations(store)
        saved = np.asarray([row["probabilities"] for row in records])
        replay = policy.probabilities(observations)
        if not np.allclose(saved, replay, atol=1e-7, rtol=1e-6):
            raise CLIError("saved observation probabilities do not reproduce collection")
        before = policy.fingerprint()
        references, contexts = _references(store)
        patterns = _all_patterns(config, schema)
        resolved_patterns = []
        for pattern in patterns:
            indices = pattern.resolve_indices(schema)
            row = pattern.to_dict() | {"indices": list(indices), "names_ja": [schema.spec(i).name_ja for i in indices]}
            row.update(_pattern_variant_metadata(pattern, schema))
            if pattern.method == "reference":
                if pattern.reference_id not in references:
                    row["resolution_status"] = "unavailable"
                    row["skip_reason"] = f"saved reference does not exist: {pattern.reference_id}"
                else:
                    row["resolved_values"] = {str(i): float(references[pattern.reference_id][i]) for i in indices}
            elif pattern.method == "fixed" and "resolved_values" not in row:
                row["resolved_values"] = pattern.values if pattern.values is not None else {str(i): schema.spec(i).replacement.get("value") for i in indices}
            resolved_patterns.append(row)
        analysis.write_json("01_offline/patterns.json", resolved_patterns)
        _write_csv(analysis.offline_dir / "patterns.csv", resolved_patterns)
        _write_csv(analysis.offline_dir / "input_statistics.csv", [{"index": spec.index, "id": spec.id, "name_ja": spec.name_ja,
            "min": float(observations[:,spec.index].min()), "max": float(observations[:,spec.index].max()),
            "mean": float(observations[:,spec.index].mean()), "std": float(observations[:,spec.index].std()),
            "constant": bool(np.all(observations[:,spec.index] == observations[0,spec.index]))} for spec in schema.inputs])
        check_payload = store.read_json("check.json")
        action_rows = (check_payload.get("checks") or {}).get("action_mapping", [])
        if not isinstance(action_rows, list):
            action_rows = []
        action_mapping = {key: [str(row[key]) for row in action_rows] for key in ("steering", "throttle_brake")
                          if all(isinstance(row, dict) and key in row for row in action_rows)}
        result = run_offline(observations, policy, schema, patterns,
            actions=np.asarray([row["action"] for row in records]),
            episode_ids=np.asarray([row["episode"] for row in records]),
            steps=np.asarray([row["step"] for row in records]),
            reference_observations=references, reference_contexts=contexts,
            observation_contexts=[row["pre_telemetry"] for row in records], strict=False,
            action_mapping=action_mapping,
            metadata={"source": "00_reference", "run_id": store.run_id,
                      "analysis_id": analysis.analysis_id, "saved_probabilities_reproduced": True})
        if before != policy.fingerprint():
            raise CLIError("model weights changed during offline analysis")
        payload = result.to_dict(include_arrays=False)
        payload["input_schema"] = schema.to_dict()
        for item, resolved in zip(payload["patterns"], resolved_patterns, strict=True):
            item["pattern"].update(
                names_ja=resolved["names_ja"],
                **{
                    key: resolved[key]
                    for key in (
                        "resolved_values",
                        "resolved_expression",
                        "variant_ids",
                        "variant_id",
                        "variant_evidence",
                        "variant_classification",
                        "variant_records",
                    )
                    if key in resolved
                },
            )
        for pattern, item in zip(result.patterns, payload["patterns"], strict=True):
            identifier = pattern.pattern.pattern_id
            target = analysis.offline_dir / identifier
            target.mkdir(exist_ok=False)
            np.savez_compressed(target / "arrays.npz", modified_observations=pattern.altered_observations,
                probabilities=pattern.changed_probabilities, changed_input_mask=pattern.changed_input_mask)
            item["summary"]["per_input"] = [row for row in item["summary"]["per_input"] if row["requested"]]
            _write_csv(target / "steps.csv", list(pattern.rows))
        analysis.write_json("01_offline/result.json", payload)
        analysis.update_status("offline", "success", pattern_count=len(result.patterns),
                               saved_probabilities_reproduced=True, policy_unchanged=True)
        return result
    except Exception as exc:
        analysis.update_status("offline", "failed", error=f"{type(exc).__name__}: {exc}")
        raise


def _pattern_selection(config: AnalysisConfig, requested: str | None) -> list[Mapping[str, Any]]:
    if not requested:
        ids = list(config.closed_loop.get("patterns", ["P00"]))
    else:
        ids = [item.strip() for item in requested.split(",") if item.strip()]
    by_id = {str(item["id"]): item for item in config.patterns}
    missing = [identifier for identifier in ids if identifier not in by_id]
    if missing:
        raise CLIError(f"未定義pattern id: {', '.join(missing)}")
    return [by_id[identifier] for identifier in ids]


def _references(store: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    observations, records = _load_observations(store)
    refs, contexts = {}, {}
    for index, row in enumerate(records):
        for identifier in (f"{row['episode_id']}:{row['step']}", f"{row['episode']}:{row['step']}"):
            refs[identifier] = observations[index].copy()
            contexts[identifier] = row["pre_telemetry"]
    return refs, contexts


def _reference_map(store: Any) -> dict[str, np.ndarray]:
    return _references(store)[0]


def _runtime_failure_kind(value: Any) -> str | None:
    """Classify explicit closed-loop execution failures for CLI assessment."""

    token = str(value or "").strip().casefold().replace("-", "_")
    if token in {
        "failed",
        "failure",
        "error",
        "runtime_error",
        "runtime_failure",
        "runtime_failed",
        "not_evaluable_runtime_error",
        "実行失敗",
    }:
        return "failed"
    if token in {
        "aborted",
        "abort",
        "interrupted",
        "intervention_abort",
        "not_evaluable_intervention_abort",
        "実行失敗／中断",
        "中断",
    }:
        return "aborted"
    return None


def _closed_loop_runtime_diagnostics(result: Any) -> dict[str, Any]:
    """Return execution/assessment counts without judging natural outcomes.

    The current worker exposes authoritative counts from ``as_dict``.  The
    raw ``patterns`` episode list is retained only for a small legacy test
    double; summary lists of unique states are deliberately not expanded into
    invented episodes.
    Natural terminal reasons (arrival, collision, road departure, horizon)
    are never failures unless the saved execution/assessment status says so.
    """

    episodes: list[Mapping[str, Any]] = []
    candidate = result.as_dict() if callable(getattr(result, "as_dict", None)) else result
    payload: Mapping[str, Any] | None = candidate if isinstance(candidate, Mapping) else None
    declared_counts = payload.get("counts") if isinstance(payload, Mapping) else None
    if isinstance(declared_counts, Mapping) and any(
        key in declared_counts
        for key in ("failed_episode_count", "aborted_episode_count", "runtime_failure_count")
    ):
        def count(name: str) -> int:
            try:
                return max(0, int(declared_counts.get(name, 0) or 0))
            except (TypeError, ValueError):
                return 0

        failed = count("failed_episode_count")
        aborted = count("aborted_episode_count")
        completed = count("completed_episode_count")
        return {
            "episode_count": count("episode_count"),
            "completed_episode_count": completed,
            "failed_episode_count": failed,
            "aborted_episode_count": aborted,
            "runtime_failure_count": failed + aborted,
            "execution_failure_count": count("execution_failure_count") or failed + aborted,
            "unexpected_interruption_count": count("unexpected_interruption_count") or aborted,
            "execution_status_counts": dict(payload.get("execution_status_counts", {}))
            if isinstance(payload.get("execution_status_counts"), Mapping)
            else {},
            "assessment_status_counts": dict(payload.get("assessment_status_counts", {}))
            if isinstance(payload.get("assessment_status_counts"), Mapping)
            else {},
            "assessment_status": "runtime_failure" if failed else "aborted" if aborted else "completed",
            "counts": dict(declared_counts),
        }
    raw_patterns = getattr(result, "patterns", None)
    if isinstance(raw_patterns, Mapping):
        for values in raw_patterns.values():
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes, Mapping)):
                episodes.extend(item for item in values if isinstance(item, Mapping))

    failed = 0
    aborted = 0
    completed = 0
    status_counts: dict[str, int] = {}
    assessment_counts: dict[str, int] = {}
    for episode in episodes:
        execution = str(episode.get("execution_status") or "unknown")
        assessment = str(episode.get("assessment_status") or "unknown")
        status_counts[execution] = status_counts.get(execution, 0) + 1
        assessment_counts[assessment] = assessment_counts.get(assessment, 0) + 1
        kind = _runtime_failure_kind(episode.get("execution_status"))
        if kind is None:
            kind = _runtime_failure_kind(episode.get("assessment_status"))
        if kind is None:
            terminal = str(episode.get("terminal_reason") or "").strip().casefold().replace("-", "_")
            if terminal == "runtime_error":
                kind = "failed"
            elif terminal == "intervention_abort":
                kind = "aborted"
        if kind == "failed":
            failed += 1
        elif kind == "aborted":
            aborted += 1
        elif str(episode.get("execution_status") or "").casefold() == "completed":
            completed += 1

    return {
        "episode_count": len(episodes),
        "completed_episode_count": completed,
        "failed_episode_count": failed,
        "aborted_episode_count": aborted,
        "runtime_failure_count": failed + aborted,
        "execution_status_counts": status_counts,
        "assessment_status_counts": assessment_counts,
        "assessment_status": "runtime_failure" if failed else "aborted" if aborted else "completed",
        "counts": {
            "episode_count": len(episodes),
            "completed_episode_count": completed,
            "failed_episode_count": failed,
            "aborted_episode_count": aborted,
            "runtime_failure_count": failed + aborted,
            "required_experiment_failed": bool(failed + aborted),
        },
    }


def _closed_loop_counts_line(diagnostics: Mapping[str, Any]) -> str:
    """Render one stable machine-readable count line before a run path."""

    return (
        "closed-loop counts: "
        f"completed={int(diagnostics.get('completed_episode_count', 0) or 0)}, "
        f"failed={int(diagnostics.get('failed_episode_count', 0) or 0)}, "
        f"aborted={int(diagnostics.get('aborted_episode_count', 0) or 0)}"
    )


def _run_closed_loop(
    store: Any,
    config: AnalysisConfig,
    adapter: Any,
    policy: Any,
    schema: Any,
    requested_patterns: str | None,
) -> Any:
    verify_reference(store)
    selected = _pattern_selection(config, requested_patterns)
    if not any(p["id"] == "P00" for p in selected):
        selected.insert(0, next(p for p in config.patterns if p["id"] == "P00"))
    refs, contexts = _references(store)
    _, records = _load_observations(store)
    analysis = store.for_analysis("closed_loop")
    result = run_closed_loop(
        policy,
        adapter,
        config,
        patterns=selected,
        store=analysis,
        episodes=int(config.closed_loop.get("episodes", 1)),
        max_steps=config.closed_loop.get("max_steps"),
        scenario_seed=config.scenario_seed,
        rl_seed=config.rl_seed,
        deterministic=config.deterministic,
        schema=schema,
        reference_observations=refs,
        reference_contexts=contexts,
        reference_records=records,
    )
    diagnostics = _closed_loop_runtime_diagnostics(result)
    if diagnostics["runtime_failure_count"]:
        failure_kind = "runtime failure" if diagnostics["failed_episode_count"] else "aborted execution"
        message = (
            f"closed-loop {failure_kind}: "
            f"failed={diagnostics['failed_episode_count']}, "
            f"aborted={diagnostics['aborted_episode_count']}, "
            f"completed={diagnostics['completed_episode_count']}"
        )
        # Keep the saved analysis pointer and diagnostics, while separating
        # execution state (failed) from the assessment label (runtime_failure
        # or aborted).  A newer runtime may already have written a rich
        # failed stage entry; preserve its counts/pattern pointer instead of
        # replacing that evidence with a smaller CLI-shaped entry.
        try:
            current_status = analysis.read_json("status.json")
        except (FileNotFoundError, OSError, ValueError):
            current_status = {}
        current_stage = current_status.get("stages", {}).get("closed_loop", {}) if isinstance(current_status, Mapping) else {}
        if not isinstance(current_stage, Mapping) or str(current_stage.get("state", "")).casefold() != "failed":
            result_payload = result.as_dict() if callable(getattr(result, "as_dict", None)) else {}
            detail_counts = result_payload.get("counts") if isinstance(result_payload, Mapping) else None
            status_details = dict(diagnostics)
            status_details.update(
                status=(result_payload.get("status") if isinstance(result_payload, Mapping) else None) or "partial_failure",
                patterns=list(getattr(result, "patterns", {}) or {}),
                counts=detail_counts if isinstance(detail_counts, Mapping) else diagnostics,
            )
            analysis.update_status(
                "closed_loop",
                "failed",
                error=message,
                execution_status="failed" if diagnostics["failed_episode_count"] else "aborted",
                **status_details,
            )
        failure = CLIError(message)
        failure.diagnostics = diagnostics
        raise failure
    return result


def _write_check(path: Path, result: CheckResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def command_check(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    adapter, policy, schema = load_components(config)
    result = run_check(config, policy=policy, adapter=adapter, schema=schema, probe=args.probe, probe_steps=args.probe_steps)
    print(format_check(result))
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, default=str))
    return 0 if result.ok else 1


def command_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    adapter, policy, schema = load_components(config)
    check_result = run_check(config, policy=policy, adapter=adapter, schema=schema, probe=args.probe, probe_steps=args.probe_steps)
    if not check_result.ok:
        print(format_check(check_result), file=sys.stderr)
        return 1
    store = _new_store(config, "run", adapter, policy, schema, check_result)
    store.write_json("check.json", check_result.as_dict())
    closed_loop_result: Any | None = None
    try:
        collect_reference(
            policy,
            adapter,
            config,
            store=store,
            schema=schema,
            episodes=int(config.scenario.get("episodes", 1)),
            scenario_seed=config.scenario_seed,
            rl_seed=config.rl_seed,
            deterministic=config.deterministic,
        )
        seal_reference(store)
        _run_offline(store, config, policy, schema)
        if config.closed_loop.get("enabled", True):
            closed_loop_result = _run_closed_loop(store, config, adapter, policy, schema, args.patterns)
        from .reporting import generate_report

        run_details: dict[str, Any] = {}
        if closed_loop_result is not None:
            closed_diagnostics = _closed_loop_runtime_diagnostics(closed_loop_result)
            run_details.update(
                status="success",
                counts=closed_diagnostics.get("counts", closed_diagnostics),
                closed_loop_assessment_status=closed_diagnostics.get("assessment_status"),
            )
        store.update_status("run", "success", **run_details)
        report = generate_report(store.run_dir)
        store.update_status("report", "success", files=[str(path) for path in report.files])
        if closed_loop_result is not None:
            print(_closed_loop_counts_line(_closed_loop_runtime_diagnostics(closed_loop_result)))
        print(store.run_dir)
        return 0
    except Exception as exc:
        failure_details: dict[str, Any] = {}
        diagnostics = getattr(exc, "diagnostics", None)
        if isinstance(diagnostics, Mapping):
            failure_details.update(
                counts=diagnostics.get("counts", diagnostics),
                closed_loop_assessment_status=diagnostics.get("assessment_status"),
            )
        store.update_status("run", "failed", error=f"{type(exc).__name__}: {exc}", **failure_details)
        try:
            from .reporting import generate_report
            report = generate_report(store.run_dir)
            store.update_status("report", "success", files=[str(path) for path in report.files], partial=True)
        except Exception as report_exc:
            store.update_status("report", "failed", error=f"{type(report_exc).__name__}: {report_exc}")
        if isinstance(diagnostics, Mapping):
            print(_closed_loop_counts_line(diagnostics), file=sys.stderr)
        print(f"run failed: {type(exc).__name__}: {exc}; saved results: {store.run_dir}", file=sys.stderr)
        return 1


def _load_existing(args: argparse.Namespace) -> tuple[Any, AnalysisConfig, Any, Any, Any]:
    store = open_run(args.run_dir)
    config = load_config(store.resolved_config_path)
    manifest = store.load_manifest() if store.manifest_path.is_file() else {}
    compatible, mismatches = verify_manifest_compatibility(
        manifest,
        config=config,
        model_path=config.model_path,
        schema_path=config.schema_path,
        patterns=config.patterns,
        preprocess=config.preprocess,
    )
    if not compatible:
        raise CLIError(f"existing run provenance mismatch: {', '.join(mismatches)}")
    verify_reference(store)
    for relative, digest in manifest.get("snapshot_files", {}).items():
        if sha256_file(store.run_dir / relative) != digest:
            raise CLIError(f"saved schema/pattern snapshot mismatch: {relative}")
    adapter, policy, schema = load_components(config)
    if policy.fingerprint() != manifest.get("policy_fingerprint"):
        raise CLIError("loaded policy fingerprint mismatch")
    return store, config, adapter, policy, schema


def _reuse_provenance_mismatches(
    parent: Any,
    parent_config: AnalysisConfig,
    config: AnalysisConfig,
) -> list[str]:
    """Check the immutable contract needed to reuse saved observations.

    A new analysis may change explicitly declared intervention variants, so a
    full resolved-config/pattern hash comparison would reject the intended
    operation.  The model bytes, schema semantics/order, and preprocessing
    contract remain strict and are checked independently here.
    """

    manifest = parent.load_manifest()
    mismatches: list[str] = []
    expected_model = (manifest.get("model") or {}).get("sha256")
    actual_model = sha256_file(config.model_path) if config.model_path and config.model_path.is_file() else None
    if expected_model != actual_model:
        mismatches.append("model.sha256")

    expected_semantics, _semantics_source = _parent_input_semantics(parent)
    schema_meta = manifest.get("schema") or {}
    actual_semantics = schema_semantics_hash(config.schema_path) if config.schema_path else None
    if expected_semantics is not None:
        if expected_semantics != actual_semantics:
            mismatches.append("input_semantics_sha256")
    else:
        expected_schema = schema_meta.get("sha256")
        actual_schema = sha256_file(config.schema_path) if config.schema_path and config.schema_path.is_file() else None
        if expected_schema != actual_schema:
            mismatches.append("schema.sha256")

    expected_preprocess = manifest.get("preprocess_sha256")
    actual_preprocess = sha256_object(config.preprocess)
    if expected_preprocess != actual_preprocess:
        mismatches.append("preprocess_sha256")

    # Saved observations carry the parent collection's scenario and adapter
    # provenance.  A new A variant may change analysis/pattern sections, but it
    # must not be labelled as if it came from a different world or producer.
    parent_mapping = parent_config.to_dict()
    new_mapping = config.to_dict()
    for section in ("scenario", "seeds", "environment", "adapter"):
        if parent_mapping.get(section, {}) != new_mapping.get(section, {}):
            mismatches.append(f"parent.{section}")

    # The saved config is resolved from the parent run.  A config that points
    # to a different observation dimension is rejected even when an old
    # manifest predates the semantic hash field.
    try:
        if config.schema_path is None or parent_config.schema_path is None:
            mismatches.append("schema.path")
        else:
            from .schema import load_schema

            if load_schema(config.schema_path).dimension != load_schema(parent_config.schema_path).dimension:
                mismatches.append("schema.dimension")
    except Exception:
        mismatches.append("schema")
    return list(dict.fromkeys(mismatches))


def _parent_input_semantics(parent: Any) -> tuple[str | None, str | None]:
    """Return the parent input-meaning hash and its immutable evidence path."""

    manifest = parent.load_manifest()
    value = manifest.get("input_semantics_sha256") or (manifest.get("schema") or {}).get("semantics_sha256")
    if value is not None:
        return str(value), "manifest.input_semantics_sha256"
    snapshot_path = parent.run_dir / "input_schema.json"
    if snapshot_path.is_file():
        try:
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            return schema_semantics_hash_from_value(snapshot), "input_schema.json"
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass
    return None, None


def _verify_parent_integrity(parent: Any) -> list[str]:
    """Verify immutable metadata before creating a reuse child run.

    The package code hash is deliberately excluded: a report/reuse fix may be
    made after the original run.  Saved config, snapshots, model, adapter
    source, preprocessing/environment files, and the sealed reference remain
    strict provenance inputs.
    """

    manifest = parent.load_manifest()
    mismatches: list[str] = []
    resolved_path = parent.resolved_config_path
    if not resolved_path.is_file():
        mismatches.append("resolved_config.json")
    else:
        try:
            saved_config = json.loads(resolved_path.read_text(encoding="utf-8"))
            expected = manifest.get("config_sha256")
            if expected is not None and expected != sha256_object(saved_config):
                mismatches.append("config_sha256")
            expected_preprocess = manifest.get("preprocess_sha256")
            if expected_preprocess is not None and expected_preprocess != sha256_object(saved_config.get("preprocess", {})):
                mismatches.append("parent.preprocess_sha256")
        except (OSError, UnicodeError, json.JSONDecodeError):
            mismatches.append("resolved_config.json")

    for relative, digest in (manifest.get("snapshot_files") or {}).items():
        path = (parent.run_dir / str(relative)).resolve()
        try:
            path.relative_to(parent.run_dir.resolve())
        except ValueError:
            mismatches.append(f"snapshot_path:{relative}")
            continue
        if not path.is_file() or sha256_file(path) != digest:
            mismatches.append(f"snapshot:{relative}")

    for path_text, expected in (manifest.get("input_files") or {}).items():
        path = Path(path_text)
        if not path.is_file() or expected != sha256_file(path):
            mismatches.append(f"input_file:{path_text}")
    for path_text, expected in (manifest.get("adapter_sources") or {}).items():
        path = Path(path_text)
        if not path.is_file() or expected != sha256_file(path):
            mismatches.append(f"adapter_source:{path_text}")
    model_meta = manifest.get("model") or {}
    model_path = Path(model_meta["path"]) if model_meta.get("path") else None
    if model_meta.get("sha256") is not None and (
        model_path is None or not model_path.is_file() or sha256_file(model_path) != model_meta["sha256"]
    ):
        mismatches.append("parent.model.sha256")
    return list(dict.fromkeys(mismatches))


def _reuse_boundary_check(config: AnalysisConfig, adapter: Any, policy: Any, schema: Any) -> CheckResult:
    """Validate a new A-only config without constructing an environment."""

    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {"mode": "saved_reference_reuse", "probe_isolated": True}
    try:
        schema.validate_for_execution()
        declared_dimension = config.schema.get("dimension", schema.dimension)
        if declared_dimension != schema.dimension:
            raise CLIError(f"configured schema dimension mismatch: {declared_dimension} != {schema.dimension}")
        checks["patterns"] = resolve_pattern_listing(schema, config.patterns)
        checks["schema_validation"] = {
            "dimension": schema.dimension,
            "entry_count": len(schema.inputs),
            "indices": sorted(item.index for item in schema.inputs),
        }
        checks["adapter_contract"] = adapter.assert_contract(expected_dimension=schema.dimension)
        checks["adapter_schema_contract"] = adapter.verify_schema_contract(
            schema, expected_dimension=schema.dimension
        )
        if checks["adapter_schema_contract"].get("verified") is not True:
            raise CLIError("observation ordering/source contract has not been verified")
        checks["policy_fingerprint"] = policy.fingerprint()
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    return CheckResult(not errors, errors, warnings, checks)


def _new_reuse_store(
    parent: Any,
    parent_config: AnalysisConfig,
    config: AnalysisConfig,
    adapter: Any,
    policy: Any,
    schema: Any,
    check: CheckResult,
) -> Any:
    """Create a child analysis run and copy the parent's sealed reference."""

    parent_manifest = parent.load_manifest()
    parent_semantics, parent_semantics_source = _parent_input_semantics(parent)
    parent_mapping = parent_config.to_dict()
    collection_provenance = {
        section: parent_mapping.get(section, {})
        for section in ("scenario", "seeds", "environment", "adapter")
    }
    store = create_run(config.output_root, config.experiment_name, config.model_name)
    store.save_resolved_config(config)
    store.save_manifest(
        build_manifest(
            config=config,
            command="offline --run-dir --config (reuse saved reference)",
            model_path=config.model_path,
            schema_path=config.schema_path,
            patterns=config.patterns,
            preprocess=config.preprocess,
            observations_path=store.reference_dir / "observations.npy",
            extra={
                "reuse": {
                    "mode": "saved_reference_child_run",
                    "parent_run_id": parent.run_id,
                    "parent_run_dir": str(parent.run_dir),
                    "parent_data_id": parent_manifest.get("data_id"),
                    "parent_reference_files": parent_manifest.get("reference_files", {}),
                    "parent_model_sha256": (parent_manifest.get("model") or {}).get("sha256"),
                    "parent_input_semantics_sha256": parent_semantics,
                    "parent_input_semantics_source": parent_semantics_source,
                    "parent_preprocess_sha256": parent_manifest.get("preprocess_sha256"),
                    "parent_collection_provenance": collection_provenance,
                    "parent_collection_provenance_sha256": sha256_object(collection_provenance),
                },
                "parent_run_id": parent.run_id,
                "parent_data_id": parent_manifest.get("data_id"),
                "parent_reference_files": parent_manifest.get("reference_files", {}),
                "parent_model_sha256": (parent_manifest.get("model") or {}).get("sha256"),
                "parent_input_semantics_sha256": parent_semantics,
                "parent_input_semantics_source": parent_semantics_source,
                "parent_preprocess_sha256": parent_manifest.get("preprocess_sha256"),
                "reference_reused": True,
            },
        )
    )
    manifest = store.load_manifest()
    source = Path(inspect.getfile(type(adapter))).resolve()
    sources = [source]
    hook = getattr(adapter, "source_paths", None)
    if callable(hook):
        sources.extend(Path(path).resolve() for path in hook())
    manifest.update(
        adapter_sources={str(path): sha256_file(path) for path in sources},
        policy_fingerprint=policy.fingerprint(),
        runtime=check.checks.get("packages", {}),
        environment_resolved=getattr(adapter, "env_config", None),
        experiment=config.experiment_name,
        model_name=config.model_name,
        run_id=store.run_id,
        optional_ig="not executed by run; use the ig command explicitly",
    )
    store.save_manifest(manifest)
    copy_reference(parent, store)
    seal_reference(store)
    store.write_json("input_schema.json", schema.to_dict())
    _write_csv(store.run_dir / "input_schema.csv", schema.to_dict()["inputs"])
    listing = []
    for pattern in _all_patterns(config, schema):
        row = pattern.to_dict() | {
            "indices": list(pattern.resolve_indices(schema)),
            "names_ja": [schema.spec(i).name_ja for i in pattern.resolve_indices(schema)],
        }
        row.update(_pattern_variant_metadata(pattern, schema))
        listing.append(row)
    store.write_json("patterns.json", listing)
    _write_csv(store.run_dir / "patterns.csv", listing)
    store.write_json("check.json", check.as_dict())
    manifest = store.load_manifest()
    manifest["snapshot_files"] = {
        name: sha256_file(store.run_dir / name)
        for name in ("input_schema.json", "patterns.json", "check.json")
    }
    store.save_manifest(manifest)
    store.update_status("ig", "skipped", reason="任意実行。igコマンドで対象時刻と実観測基準を指定してください")
    store.update_status(
        "collect",
        "skipped",
        reason="保存通常観測を親runから再利用。新しい走行収集は行っていません",
    )
    store.update_status(
        "closed_loop",
        "skipped",
        reason="A-only reuse child。新しい条件のBはclosed-loopを別途実行してください",
    )
    return store


def command_collect(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    adapter, policy, schema = load_components(config)
    check = run_check(config, adapter=adapter, policy=policy, schema=schema)
    if not check.ok:
        print(format_check(check), file=sys.stderr)
        return 1
    store = _new_store(config, "collect", adapter, policy, schema, check)
    try:
        collect_reference(policy, adapter, config, store=store, schema=schema, episodes=int(config.scenario.get("episodes", 1)), scenario_seed=config.scenario_seed, rl_seed=config.rl_seed, deterministic=config.deterministic)
        seal_reference(store)
        store.update_status("run", "success")
        print(store.run_dir)
        return 0
    except Exception as exc:
        store.update_status("run", "failed", error=f"{type(exc).__name__}: {exc}")
        print(f"collect failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def command_offline(args: argparse.Namespace) -> int:
    if args.config is not None:
        try:
            parent = open_run(args.run_dir)
            verify_reference(parent)
            parent_integrity = _verify_parent_integrity(parent)
            if parent_integrity:
                raise CLIError(
                    "parent run provenance mismatch: " + ", ".join(parent_integrity)
                )
            parent_config = load_config(parent.resolved_config_path)
            config = load_config(args.config)
            mismatches = _reuse_provenance_mismatches(parent, parent_config, config)
            if mismatches:
                raise CLIError(
                    "saved-reference reuse provenance mismatch: " + ", ".join(mismatches)
                )
            adapter, policy, schema = load_components(config)
            parent_manifest = parent.load_manifest()
            expected_policy = parent_manifest.get("policy_fingerprint")
            if expected_policy is not None and policy.fingerprint() != expected_policy:
                raise CLIError("saved-reference reuse policy fingerprint mismatch")
            # The reuse path is intentionally environment-free.  A full
            # boundary check would reset an environment even with probe=False;
            # the saved-observation replay below is the strict runtime check.
            check = _reuse_boundary_check(config, adapter, policy, schema)
            try:
                parent_check = parent.read_json("check.json")
                parent_action_mapping = (parent_check.get("checks") or {}).get("action_mapping")
                if isinstance(parent_action_mapping, list):
                    check.checks["action_mapping"] = parent_action_mapping
            except FileNotFoundError:
                pass
            if not check.ok:
                raise CLIError("new analysis config failed check: " + "; ".join(check.errors))
            child = _new_reuse_store(parent, parent_config, config, adapter, policy, schema, check)
            try:
                _run_offline(child, config, policy, schema)
                from .reporting import generate_report

                child.update_status("run", "success", mode="offline_reuse")
                report = generate_report(child.run_dir)
                child.update_status("report", "success", files=[str(path) for path in report.files])
                print(child.run_dir)
                return 0
            except Exception as exc:
                child.update_status("run", "failed", error=f"{type(exc).__name__}: {exc}", mode="offline_reuse")
                try:
                    from .reporting import generate_report

                    report = generate_report(child.run_dir)
                    child.update_status("report", "success", files=[str(path) for path in report.files], partial=True)
                except Exception as report_exc:
                    child.update_status("report", "failed", error=f"{type(report_exc).__name__}: {report_exc}")
                raise
        except Exception as exc:
            print(f"offline reuse failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
    try:
        store, config, _adapter, policy, schema = _load_existing(args)
        _run_offline(store, config, policy, schema)
        print(store.run_dir)
        return 0
    except Exception as exc:
        print(f"offline failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def command_closed_loop(args: argparse.Namespace) -> int:
    try:
        store, config, adapter, policy, schema = _load_existing(args)
        result = _run_closed_loop(store, config, adapter, policy, schema, args.patterns)
        print(_closed_loop_counts_line(_closed_loop_runtime_diagnostics(result)))
        print(store.run_dir)
        return 0
    except Exception as exc:
        diagnostics = getattr(exc, "diagnostics", None)
        if isinstance(diagnostics, Mapping):
            print(_closed_loop_counts_line(diagnostics), file=sys.stderr)
        print(f"closed-loop failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def command_report(args: argparse.Namespace) -> int:
    try:
        store = open_run(args.run_dir)
        from .reporting import generate_report

        result = generate_report(store.run_dir, output_dir=args.output_dir)
        # Report regeneration is a read-only view of the saved run even when
        # the destination is the versioned ``reports/<id>`` directory chosen
        # by the renderer.  The original status is evidence for this report
        # and must not be rewritten as a side effect of reading it.
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2, default=str))
        return 0
    except Exception as exc:
        print(f"report failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def command_ig(args: argparse.Namespace) -> int:
    try:
        store, config, _adapter, policy, schema = _load_existing(args)
        from .integrated_gradients import compute_integrated_gradients

        observations, records = _load_observations(store)
        selected_steps = [int(value) for value in args.steps.split(",") if value.strip()]
        selected = []
        for index, row in enumerate(records):
            if str(args.episode) in {str(row.get("episode")), str(row.get("episode_id"))} and int(row.get("step", -1)) in selected_steps:
                selected.append((index, row))
        missing_steps = sorted(set(selected_steps) - {int(row["step"]) for _, row in selected})
        if not selected or missing_steps:
            raise CLIError(f"指定episode/stepsの保存観測がありません: episode={args.episode}, missing_steps={missing_steps}")
        baseline_map = _reference_map(store)
        baseline = baseline_map.get(args.baseline)
        if baseline is None:
            raise CLIError(f"baseline referenceがありません: {args.baseline}")
        analysis = store.for_analysis("ig")
        analysis.update_status("ig", "running")
        results = []
        for index, row in selected:
            current_probabilities = policy.probabilities(observations[index:index + 1])
            action = int(np.argmax(current_probabilities[0]))
            if action != int(row["action"]) or not np.allclose(current_probabilities[0], row["probabilities"], atol=1e-7, rtol=1e-6):
                raise CLIError("IG source action/probabilities do not reproduce the saved original observation")
            value = compute_integrated_gradients(
                observations[index],
                baseline,
                policy,
                schema=schema,
                target_action=action,
                baseline_id=args.baseline,
                backend="captum",
                n_steps=config.ig["n_steps"], method=config.ig["method"],
                completeness_tolerance=config.ig["completeness_tolerance"], max_retries=config.ig["max_retries"],
                observation_context=row["pre_telemetry"], baseline_context=_references(store)[1][args.baseline],
                compatibility_keys=config.ig["compatibility_keys"],
            )
            results.append(value.to_dict() | {"episode": row["episode"], "step": row["step"]})
        analysis.write_json("03_ig/result.json", {"analysis_id": analysis.analysis_id, "results": results})
        analysis.update_status("ig", "success")
        print(analysis.ig_dir)
        return 0
    except Exception as exc:
        if "analysis" in locals():
            analysis.update_status("ig", "failed", error=f"{type(exc).__name__}: {exc}")
        print(f"ig failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MetaDrive PPOの入力依存度分析")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="モデル・schema・環境境界を検証")
    check.add_argument("--config", type=Path, required=True)
    check.add_argument("--probe", action="store_true", help="reset後に代表数stepも確認")
    check.add_argument("--probe-steps", type=int, default=3)
    check.set_defaults(function=command_check)

    run = subparsers.add_parser("run", help="collect→offline→closed-loop→report")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--patterns", default=None, help="closed-loop対象pattern idをカンマ区切り")
    run.add_argument("--probe", action="store_true")
    run.add_argument("--probe-steps", type=int, default=3)
    run.set_defaults(function=command_run)

    collect = subparsers.add_parser("collect", help="変更なし通常走行を保存")
    collect.add_argument("--config", type=Path, required=True)
    collect.set_defaults(function=command_collect)

    offline = subparsers.add_parser("offline", help="保存観測だけで①-Aを実行")
    offline.add_argument("--run-dir", type=Path, required=True)
    offline.add_argument(
        "--config",
        type=Path,
        default=None,
        help="新しい解析設定で保存通常観測を再利用し、別の子runへ①-Aを保存",
    )
    offline.set_defaults(function=command_offline)

    closed = subparsers.add_parser("closed-loop", help="指定patternを逐次環境で実行")
    closed.add_argument("--run-dir", type=Path, required=True)
    closed.add_argument("--patterns", default=None)
    closed.set_defaults(function=command_closed_loop)

    ig = subparsers.add_parser("ig", help="指定時刻だけ任意のIntegrated Gradients")
    ig.add_argument("--run-dir", type=Path, required=True)
    ig.add_argument("--episode", required=True)
    ig.add_argument("--steps", required=True, help="stepをカンマ区切り")
    ig.add_argument("--baseline", required=True, help="保存reference id")
    ig.set_defaults(function=command_ig)

    report = subparsers.add_parser("report", help="保存結果から日本語レポート再生成")
    report.add_argument("--run-dir", type=Path, required=True)
    report.add_argument("--output-dir", type=Path, default=None, help="旧run外へレポートだけを書き出す")
    report.set_defaults(function=command_report)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.function(args))
    except Exception as exc:
        if args.command == "check":
            failed = CheckResult(False, [f"{type(exc).__name__}: {exc}"], [], {"config": str(args.config)})
            print(format_check(failed))
            print(json.dumps(failed.as_dict(), ensure_ascii=False, indent=2))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 2


__all__ = ["CLIError", "build_parser", "load_components", "main"]
