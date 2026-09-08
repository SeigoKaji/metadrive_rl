"""Explicit, isolated boundary checks; no probe samples enter the experiment."""
from __future__ import annotations
from dataclasses import dataclass
import importlib.metadata
import importlib.util
import platform
import sys
from typing import Any, Mapping
import numpy as np
from .collection import (close_environment, decode_action, make_environment, model_input,
                         policy_predict, policy_probabilities, reset_environment,
                         step_environment, telemetry)
from .interventions import InterventionPattern, apply_intervention
from .artifacts import input_file_hashes

class CheckError(ValueError):
    pass

@dataclass
class CheckResult:
    ok: bool
    errors: list[str]
    warnings: list[str]
    checks: dict[str, Any]
    probe: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings,
                "checks": self.checks, "probe": self.probe}


def package_versions() -> dict[str, Any]:
    result = {}
    for module, distribution in (("numpy", "numpy"), ("torch", "torch"),
            ("stable_baselines3", "stable-baselines3"), ("metadrive", "metadrive-simulator"),
            ("gymnasium", "gymnasium"), ("PIL", "Pillow")):
        try:
            spec = importlib.util.find_spec(module)
            try:
                version = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                version = None
            result[module] = {"available": spec is not None, "version": version,
                              "origin": spec.origin if spec else None}
        except (ImportError, ValueError) as exc:
            result[module] = {"available": False, "error": str(exc)}
    return result


def resolve_pattern_listing(schema: Any, patterns: Any) -> list[dict[str, Any]]:
    result = []
    for raw in patterns:
        pattern = raw if isinstance(raw, InterventionPattern) else InterventionPattern.from_dict(raw)
        indices = pattern.resolve_indices(schema)
        if pattern.pattern_id == "P00" and (indices or pattern.method != "identity"):
            raise CheckError("P00 must be identity with no target indices")
        if pattern.method == "identity" and indices:
            raise CheckError("identity patterns cannot declare replacement indices")
        for index in indices:
            item = schema.spec(index)
            if not item.independently_replaceable and not set(item.coupled_indices).issubset(indices):
                raise CheckError(f"{pattern.pattern_id}: inconsistent coupled input {index}")
        row = pattern.to_dict()
        row.update(indices=list(indices), names_ja=[schema.spec(i).name_ja for i in indices],
                   dimension_count=len(indices))
        result.append(row)
    for row in result:
        row["overlaps"] = [r["pattern_id"] for r in result
            if r is not row and set(row["indices"]) & set(r["indices"])]
    return result


def run_check(config: Any, *, policy: Any = None, adapter: Any = None,
              schema: Any = None, probe: bool = False, probe_steps: int = 3) -> CheckResult:
    errors, warnings = [], []
    checks = {"python": sys.executable, "python_version": platform.python_version(),
              "packages": package_versions(), "deterministic": config.deterministic,
              "probe_isolated": True, "probe_mode": "explicit" if probe else "boundary"}
    observed = None
    try:
        if schema is None or adapter is None or policy is None:
            raise CheckError("adapter, loaded policy, and explicit schema are required")
        schema.validate_for_execution()
        declared_dimension = config.schema.get("dimension", schema.dimension)
        if isinstance(declared_dimension, bool) or not isinstance(declared_dimension, int) or declared_dimension != schema.dimension:
            raise CheckError(f"configured schema dimension mismatch: {declared_dimension} != {schema.dimension}")
        schema_external = schema.preprocessing.get("external_normalization")
        config_external = config.preprocess.get("external_normalization", False)
        if not isinstance(schema_external, bool) or schema_external != config_external:
            raise CheckError("schema/config preprocessing external_normalization mismatch or unresolved")
        checks["schema_config_contract"] = {"configured_dimension": declared_dimension,
            "loaded_dimension": schema.dimension, "schema_external_normalization": schema_external,
            "config_external_normalization": config_external}
        if probe_steps < 1 or probe_steps > 100:
            raise CheckError("probe_steps must be within 1..100")
        for label, path in (("model", config.model_path), ("schema", config.schema_path)):
            checks[label] = {"path": str(path) if path else None,
                             "exists": path.is_file() if path else False}
            if path is not None and not path.is_file():
                raise CheckError(f"{label} file missing: {path}")
        if not config.deterministic:
            raise CheckError("stochastic evaluation is unsupported; mode is never changed automatically")
        declared = config.preprocess.get("external_normalization", False)
        if not isinstance(declared, bool):
            raise CheckError("external_normalization is unresolved; confirm training preprocessing")
        checks["patterns"] = resolve_pattern_listing(schema, config.patterns)
        if config.analysis.get("individual_inputs", True):
            from .interventions import generate_individual_patterns
            individual = generate_individual_patterns(schema, reference_id=config.analysis["reference_id"])
            checks["individual_patterns"] = [p.to_dict() | {"indices": list(p.resolve_indices(schema))} for p in individual]
        checks["schema_validation"] = {"dimension": schema.dimension,
            "entry_count": len(schema.inputs), "indices": sorted(s.index for s in schema.inputs)}
        hashes_before = input_file_hashes(config.to_dict())
        if any(value is None for value in hashes_before.values()):
            raise CheckError("environment/preprocessing/adapter input file missing")
        fingerprint_before = policy.fingerprint()
        observed = _probe(config, adapter, policy, schema=schema,
                          steps=probe_steps if probe else 0)
        checks["adapter_contract"] = observed.pop("adapter_contract")
        checks["adapter_schema_contract"] = observed.pop("schema_contract")
        checks["preprocess"] = observed["records"][0]["preprocess"]
        checks["action_mapping"] = observed.pop("action_mapping")
        checks["weights_unchanged"] = fingerprint_before == policy.fingerprint()
        checks["preprocessing_files_unchanged"] = hashes_before == input_file_hashes(config.to_dict())
        if not checks["weights_unchanged"] or not checks["preprocessing_files_unchanged"]:
            raise CheckError("model weights or fixed preprocessing statistics changed during check")
        if not observed["records"][0]["telemetry"].get("target_lane_valid", False):
            warnings.append("開始時目標レーン参照が欠測です。横ずれ指標は理由付きN/Aになります")
        checks["policy_fingerprint"] = fingerprint_before
        if config.ig.get("enabled", False):
            try:
                from captum.attr import IntegratedGradients
                checks["ig_dependency"] = {"available": True, "class": IntegratedGradients.__name__}
            except ImportError as exc:
                warnings.append(f"任意IG依存なし: {exc}。①-A/①-Bは実行できます")
                checks["ig_dependency"] = {"available": False, "error": str(exc)}
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    return CheckResult(not errors, errors, warnings, checks, observed)


def _probe(config: Any, adapter: Any, policy: Any, *, schema: Any, steps: int) -> dict[str, Any]:
    env = make_environment(adapter, config)
    records = []
    try:
        raw, info = reset_environment(adapter, env, config.scenario_seed)
        actual_seed = getattr(env, "current_seed", info.get("scenario_seed"))
        if actual_seed is None or int(actual_seed) != config.scenario_seed:
            raise CheckError(f"actual scenario seed mismatch: {actual_seed} != {config.scenario_seed}")
        contract = adapter.assert_contract(expected_dimension=schema.dimension)
        source_contract = adapter.verify_schema_contract(schema, expected_dimension=schema.dimension)
        if source_contract.get("verified") is not True:
            raise CheckError("observation ordering/source contract has not been verified")
        declared_external = config.preprocess.get("external_normalization", False)
        if contract.get("external_normalization") is not declared_external:
            raise CheckError("adapter/config external normalization mismatch")
        if declared_external:
            if contract.get("fixed_statistics_verified") is not True or contract.get("statistics_updates") is not False:
                raise CheckError("external preprocessing needs verified frozen training statistics")
        action_space = env.action_space
        if not hasattr(action_space, "n") or hasattr(action_space, "nvec") or int(getattr(action_space, "start", 0)) != 0:
            raise CheckError("only zero-based single Discrete action spaces are supported")
        count = int(action_space.n)
        if count < 2:
            raise CheckError("at least two categorical actions are required")
        mapping = [decode_action(adapter, action, config=config) for action in range(count)]
        p00 = next(InterventionPattern.from_dict(p) for p in config.patterns if p["id"] == "P00")
        for step in range(steps + 1):
            pre = telemetry(adapter, env, info, phase="pre", step=step)
            raw_copy = np.array(raw, copy=True)
            obs, preprocess = model_input(adapter, raw, info)
            schema.validate_observation(obs)
            if not np.array_equal(raw_copy, raw):
                raise CheckError("adapter preprocessing mutated raw observation")
            if not declared_external:
                if raw_copy.dtype != obs.dtype or not np.array_equal(raw_copy, obs):
                    raise CheckError("declared identity preprocessing changed input values or dtype")
            expected_dtype = getattr(env.observation_space, "dtype", None)
            if not declared_external and expected_dtype is not None and obs.dtype != np.dtype(expected_dtype):
                raise CheckError(f"observation dtype mismatch: {obs.dtype} != {expected_dtype}")
            probabilities = policy_probabilities(policy, obs)
            if probabilities.shape != (count,):
                raise CheckError("environment/model action count mismatch")
            action = policy_predict(policy, obs, deterministic=True)
            action_array = np.asarray(action)
            if action_array.size != 1 or float(action_array.item()) != int(action_array.item()):
                raise CheckError("predict must return exactly one integer action")
            action = int(action_array.item())
            if action != int(np.argmax(probabilities)):
                raise CheckError("deterministic predict differs from distribution argmax")
            identity = apply_intervention(obs, p00, schema).observation
            replay = policy_probabilities(policy, identity)
            if not np.array_equal(identity, obs) or not np.allclose(probabilities, replay, rtol=1e-7, atol=1e-7):
                raise CheckError("P00 copy/reinference path is inconsistent")
            for pattern in config.patterns:
                if pattern.get("method", pattern.get("operation")) in {
                    "fixed",
                    "fixed_level",
                    "neutral",
                    "reflection",
                }:
                    outcome = apply_intervention(obs, pattern, schema, strict=False)
                    if outcome.skipped and not any(term in (outcome.skip_reason or "") for term in ("precondition", "context")):
                        raise CheckError(f"invalid fixed replacement: {outcome.skip_reason}")
            records.append({"step": step, "actual_scenario_seed": int(actual_seed),
                "input_shape": list(obs.shape), "input_dtype": str(obs.dtype),
                "input_min": float(obs.min()), "input_max": float(obs.max()),
                "model_input": obs.tolist(), "probabilities": probabilities.tolist(),
                "predict_action": action, "argmax_action": action, "predict_argmax_match": True,
                "p00_reinference_match": True, "preprocess": preprocess,
                "telemetry": pre, "telemetry_keys": sorted(pre), "info_keys": sorted(info)})
            if step == steps:
                break
            raw, reward, terminated, truncated, info = step_environment(env, action)
            if terminated or truncated:
                break
        return {"ok": True, "steps_applied": min(steps, len(records)),
                "records": records, "actual_scenario_seed": int(actual_seed),
                "env_closed_after_return": True, "adapter_contract": contract,
                "schema_contract": source_contract, "action_mapping": mapping}
    finally:
        close_environment(adapter, env)


def format_check(result: CheckResult | Mapping[str, Any]) -> str:
    value = result.as_dict() if isinstance(result, CheckResult) else dict(result)
    lines = ["input_attribution check", f"status: {'OK' if value.get('ok') else 'FAILED'}"]
    lines.extend(f"error: {item}" for item in value.get("errors", []))
    lines.extend(f"warning: {item}" for item in value.get("warnings", []))
    for pattern in value.get("checks", {}).get("patterns", []):
        lines.append(f"{pattern['pattern_id']}: indices={pattern['indices']} method={pattern['method']} values={pattern['values']} reference={pattern['reference_id']}")
    return "\n".join(lines)

check_config = run_check
