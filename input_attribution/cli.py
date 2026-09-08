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
    create_run,
    jsonl_read,
    open_run,
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
    listing = [p.to_dict() | {"indices": list(p.resolve_indices(schema)),
               "names_ja": [schema.spec(i).name_ja for i in p.resolve_indices(schema)]} for p in _all_patterns(config, schema)]
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
            if pattern.method == "reference":
                if pattern.reference_id not in references:
                    row["resolution_status"] = "unavailable"
                    row["skip_reason"] = f"saved reference does not exist: {pattern.reference_id}"
                else:
                    row["resolved_values"] = {str(i): float(references[pattern.reference_id][i]) for i in indices}
            elif pattern.method == "fixed":
                row["resolved_values"] = pattern.values if pattern.values is not None else {str(i): schema.spec(i).replacement.get("value") for i in indices}
            resolved_patterns.append(row)
        analysis.write_json("01_offline/patterns.json", resolved_patterns)
        _write_csv(analysis.offline_dir / "patterns.csv", resolved_patterns)
        _write_csv(analysis.offline_dir / "input_statistics.csv", [{"index": spec.index, "id": spec.id, "name_ja": spec.name_ja,
            "min": float(observations[:,spec.index].min()), "max": float(observations[:,spec.index].max()),
            "mean": float(observations[:,spec.index].mean()), "std": float(observations[:,spec.index].std()),
            "constant": bool(np.all(observations[:,spec.index] == observations[0,spec.index]))} for spec in schema.inputs])
        action_rows = store.read_json("check.json")["checks"]["action_mapping"]
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
            item["pattern"].update(names_ja=resolved["names_ja"], resolved_values=resolved.get("resolved_values"))
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
            _run_closed_loop(store, config, adapter, policy, schema, args.patterns)
        from .reporting import generate_report

        store.update_status("run", "success")
        report = generate_report(store.run_dir)
        store.update_status("report", "success", files=[str(path) for path in report.files])
        store.update_status("run", "success")
        print(store.run_dir)
        return 0
    except Exception as exc:
        store.update_status("run", "failed", error=f"{type(exc).__name__}: {exc}")
        try:
            from .reporting import generate_report
            report = generate_report(store.run_dir)
            store.update_status("report", "success", files=[str(path) for path in report.files], partial=True)
        except Exception as report_exc:
            store.update_status("report", "failed", error=f"{type(report_exc).__name__}: {report_exc}")
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
        _run_closed_loop(store, config, adapter, policy, schema, args.patterns)
        print(store.run_dir)
        return 0
    except Exception as exc:
        print(f"closed-loop failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def command_report(args: argparse.Namespace) -> int:
    try:
        store = open_run(args.run_dir)
        from .reporting import generate_report

        result = generate_report(store.run_dir)
        store.update_status("report", "success", files=[str(path) for path in result.files])
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
