"""Saved-observation policy comparison (experiment ①-A).

This module accepts prepared policy inputs saved by the baseline run.  It
never constructs an environment or calls ``reset``/``step``; each modified
input is a fresh copy of the corresponding baseline input.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from typing import Any

import numpy as np

from .interventions import Intervention, apply_intervention, coerce_intervention


class PolicyComparisonError(ValueError):
    """Saved baseline data or policy output cannot be compared."""


def _probability_vector(value: object, *, action_count: int | None = None, name: str = "probabilities") -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise PolicyComparisonError(f"{name} must be a non-empty 1-D vector")
    if action_count is not None and array.size != action_count:
        raise PolicyComparisonError(
            f"{name} has {array.size} actions; expected {action_count}"
        )
    if not np.all(np.isfinite(array)):
        raise PolicyComparisonError(f"{name} contains NaN or Inf")
    if np.any(array < 0):
        raise PolicyComparisonError(f"{name} contains a negative probability")
    total = float(np.sum(array, dtype=np.float64))
    if not math.isfinite(total) or not math.isclose(total, 1.0, abs_tol=1e-6, rel_tol=1e-6):
        raise PolicyComparisonError(f"{name} does not sum to 1 (sum={total!r})")
    return array


def js_divergence(
    p: object,
    q: object,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> float:
    """Return Jensen-Shannon divergence in nats for two probability vectors."""

    p_array = _probability_vector(p, name="p")
    q_array = _probability_vector(q, name="q")
    if p_array.shape != q_array.shape:
        raise PolicyComparisonError(
            f"p and q must have the same shape, found {p_array.shape} and {q_array.shape}"
        )
    # The explicit checks above allow only tiny floating-point summation error;
    # do not normalize a malformed distribution into a different experiment.
    midpoint = 0.5 * (p_array + q_array)

    def _relative_entropy(values: np.ndarray) -> float:
        mask = values > 0
        if not np.any(mask):
            return 0.0
        return float(np.sum(values[mask] * np.log(values[mask] / midpoint[mask])))

    result = 0.5 * _relative_entropy(p_array) + 0.5 * _relative_entropy(q_array)
    if not math.isfinite(result) or result < -1e-12 or result > math.log(2.0) + 1e-9:
        raise PolicyComparisonError(f"invalid JS divergence: {result!r}")
    return max(0.0, float(result))


def policy_identity(adapter: object) -> dict[str, object]:
    """Extract stable model/preprocessing identity from an adapter."""

    # Resolve dimensions/action count first.  For a lazy SB3 adapter these
    # properties force model validation/loading; taking metadata before that
    # point would preserve a pending model path and silently omit its weight
    # hash from the saved identity.
    dimension: object | None = None
    for name in ("dimension", "observation_dim", "object_dimension"):
        try:
            candidate = getattr(adapter, name)
        except Exception:
            continue
        if isinstance(candidate, (int, np.integer)) and not isinstance(candidate, bool):
            dimension = int(candidate)
            break
    action_count: object | None = None
    try:
        candidate_action_count = getattr(adapter, "action_count")
    except Exception:
        candidate_action_count = None
    if isinstance(candidate_action_count, (int, np.integer)) and not isinstance(candidate_action_count, bool):
        action_count = int(candidate_action_count)

    identity_method = getattr(adapter, "model_identity", None)
    model = identity_method() if callable(identity_method) else None
    metadata_method = getattr(adapter, "model_metadata", None)
    metadata = metadata_method() if callable(metadata_method) else getattr(adapter, "metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    if model is None and isinstance(metadata.get("model"), Mapping):
        model = metadata.get("model")
    preprocessing = metadata.get("preprocessing", {})
    # Some adapters expose a detached complete metadata snapshot from
    # ``model_metadata``; retain only the model/preprocessing boundary in the
    # comparison identity so runtime counters and policy seeds cannot make a
    # matching baseline look different.
    if isinstance(metadata.get("identity"), Mapping):
        identity_metadata = metadata["identity"]
        if model is None and isinstance(identity_metadata, Mapping):
            model = identity_metadata.get("model")
        if isinstance(identity_metadata, Mapping) and "preprocessing" in identity_metadata:
            preprocessing = identity_metadata["preprocessing"]
    if dimension is None:
        dimension = metadata.get("dimension")
    if action_count is None:
        action_count = metadata.get("action_count")
    action_count = getattr(adapter, "action_count", metadata.get("action_count"))
    return {
        "model": _jsonable_identity(model),
        "preprocessing": _jsonable_identity(preprocessing),
        "dimension": int(dimension) if isinstance(dimension, (int, np.integer)) else dimension,
        "action_count": int(action_count) if isinstance(action_count, (int, np.integer)) else action_count,
    }


def _jsonable_identity(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return _jsonable_identity(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable_identity(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_identity(item) for item in value]
    return repr(value)


def identity_digest(identity: Mapping[str, object]) -> str:
    payload = json.dumps(_jsonable_identity(identity), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _baseline_identity(baseline: Mapping[str, object]) -> Mapping[str, object] | None:
    candidate = baseline.get("adapter_identity")
    if isinstance(candidate, Mapping):
        return candidate
    adapter = baseline.get("adapter")
    if isinstance(adapter, Mapping):
        candidate = adapter.get("identity", adapter)
        if isinstance(candidate, Mapping):
            return candidate
    return None


def _verify_identity(adapter: object, baseline: Mapping[str, object]) -> None:
    expected = _baseline_identity(baseline)
    if expected is None:
        raise PolicyComparisonError(
            "saved baseline has no model/preprocessing identity; refusing offline comparison"
        )
    expected_model = expected.get("model")
    expected_preprocessing = expected.get("preprocessing")
    if not isinstance(expected_model, Mapping) or not expected_model:
        raise PolicyComparisonError(
            "saved baseline model identity is incomplete; a model/weights hash is required"
        )
    model_hash = next(
        (
            expected_model.get(key)
            for key in ("sha256", "weights_sha256", "model_sha256", "hash", "digest")
            if expected_model.get(key) not in (None, "")
        ),
        None,
    )
    if not isinstance(model_hash, str) or not model_hash.strip():
        raise PolicyComparisonError(
            "saved baseline model identity has no model/weights hash"
        )
    if not isinstance(expected_preprocessing, Mapping) or not expected_preprocessing:
        raise PolicyComparisonError(
            "saved baseline preprocessing identity is incomplete"
        )
    actual = policy_identity(adapter)
    if identity_digest(expected) != identity_digest(actual):
        raise PolicyComparisonError(
            "saved baseline model/preprocessing identity does not match adapter"
        )


def _records(baseline: Mapping[str, object]) -> list[Mapping[str, object]]:
    value = baseline.get("records")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise PolicyComparisonError("baseline.records must be an array")
    result: list[Mapping[str, object]] = []
    for step, record in enumerate(value):
        if not isinstance(record, Mapping):
            raise PolicyComparisonError(f"baseline.records[{step}] must be an object")
        result.append(record)
    return result


def _validate_saved_baseline_for_offline(
    baseline: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
) -> None:
    """Reject partial raw rows before reporting an empty/zero A comparison."""

    if baseline.get("status") != "complete":
        raise PolicyComparisonError(
            f"saved baseline status is {baseline.get('status')!r}; "
            "offline comparison requires a complete baseline"
        )
    if not records:
        raise PolicyComparisonError(
            "saved baseline has no records; refusing an empty offline comparison"
        )
    for step, record in enumerate(records):
        if "input" not in record or record.get("input") is None:
            raise PolicyComparisonError(
                f"saved baseline record {step} has no prepared input"
            )
        if "probabilities" not in record and "p" not in record:
            raise PolicyComparisonError(
                f"saved baseline record {step} has no probabilities"
            )
        if record.get("input") is None or record.get("probabilities", record.get("p")) is None:
            raise PolicyComparisonError(
                f"saved baseline record {step} has a partial input/probability row"
            )
        if "executed" in record and record.get("executed") is not True:
            raise PolicyComparisonError(
                f"saved baseline record {step} is not an executed transition"
            )


def _baseline_input(record: Mapping[str, object], step: int, dimension: int) -> np.ndarray:
    value = record.get("input", record.get("policy_input"))
    if value is None:
        raise PolicyComparisonError(f"baseline record {step} has no prepared input")
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.shape[0] != dimension:
        raise PolicyComparisonError(
            f"baseline record {step} input shape {array.shape} does not match ({dimension},)"
        )
    if not np.all(np.isfinite(array)):
        raise PolicyComparisonError(f"baseline record {step} input contains NaN or Inf")
    return array


def _baseline_probabilities(record: Mapping[str, object], step: int, action_count: int) -> np.ndarray:
    value = record.get("probabilities", record.get("p"))
    if value is None:
        raise PolicyComparisonError(f"baseline record {step} has no probabilities")
    return _probability_vector(value, action_count=action_count, name=f"baseline p[{step}]")


def _unpack_prediction(adapter: object, values: object, *, deterministic: bool) -> tuple[np.ndarray, np.ndarray | int]:
    predict = getattr(adapter, "predict", None)
    if not callable(predict):
        raise PolicyComparisonError("adapter does not provide predict(inputs)")
    output = predict(np.asarray(values, dtype=np.float32), deterministic=deterministic)
    if not isinstance(output, tuple) or len(output) != 2:
        raise PolicyComparisonError("adapter.predict must return (probabilities, actions)")
    probabilities, actions = output
    return np.asarray(probabilities, dtype=np.float64), actions


def _action_for_row(actions: object, row: int, batch: int) -> int:
    values = np.asarray(actions)
    if batch == 1 and values.ndim == 0:
        return int(values)
    if values.ndim == 0 or values.shape[0] != batch:
        raise PolicyComparisonError("adapter action output has an unexpected batch shape")
    return int(values[row])


def compare_saved_baseline(
    adapter: object,
    baseline: Mapping[str, object],
    patterns: Sequence[Intervention | Mapping[str, object]],
    *,
    deterministic: bool = True,
    batch: bool = False,
) -> list[dict[str, object]]:
    """Run ①-A for every pattern using saved baseline inputs only."""

    _verify_identity(adapter, baseline)
    dimension = getattr(adapter, "dimension", None)
    action_count = getattr(adapter, "action_count", None)
    if not isinstance(dimension, (int, np.integer)) or not isinstance(action_count, (int, np.integer)):
        raise PolicyComparisonError("adapter must expose integer dimension and action_count")
    baseline_records = _records(baseline)
    _validate_saved_baseline_for_offline(baseline, baseline_records)
    prepared = [_baseline_input(record, step, int(dimension)) for step, record in enumerate(baseline_records)]
    baseline_probabilities = [
        _baseline_probabilities(record, step, int(action_count))
        for step, record in enumerate(baseline_records)
    ]
    results: list[dict[str, object]] = []
    for raw_pattern in patterns:
        pattern = coerce_intervention(raw_pattern, dimension=int(dimension))
        result: dict[str, object] = {
            "id": pattern.id,
            "name": pattern.name,
            "status": "complete",
            "records": [],
            "applied_count": 0,
            "changed_count": 0,
            "unchanged_count": 0,
            "failure": None,
            "comparable": True,
        }
        output_records: list[dict[str, object]] = []
        try:
            modifications = [
                apply_intervention(values, pattern, dimension=int(dimension))
                for values in prepared
            ]
            result["applied_count"] = len(modifications)
            result["changed_count"] = sum(item.changed for item in modifications)
            result["unchanged_count"] = sum(not item.changed for item in modifications)
            if batch and modifications:
                modified_values = np.stack([item.modified for item in modifications])
                probabilities, actions = _unpack_prediction(
                    adapter, modified_values, deterministic=deterministic
                )
                probabilities = np.asarray(probabilities, dtype=np.float64)
                if probabilities.shape != (len(modifications), int(action_count)):
                    raise PolicyComparisonError(
                        f"batch probabilities have shape {probabilities.shape}; expected "
                        f"({len(modifications)}, {action_count})"
                    )
                for step, (record, baseline_p, modification) in enumerate(
                    zip(baseline_records, baseline_probabilities, modifications)
                ):
                    q = _probability_vector(probabilities[step], action_count=int(action_count), name=f"q[{step}]")
                    action = _action_for_row(actions, step, len(modifications))
                    output_records.append(
                        _offline_record(step, baseline_p, q, action, modification)
                    )
            else:
                for step, (record, baseline_p, modification) in enumerate(
                    zip(baseline_records, baseline_probabilities, modifications)
                ):
                    probabilities, actions = _unpack_prediction(
                        adapter, modification.modified, deterministic=deterministic
                    )
                    probability_array = _probability_vector(
                        probabilities,
                        action_count=int(action_count),
                        name=f"q[{step}]",
                    )
                    action = _action_for_row(actions, 0, 1)
                    output_records.append(
                        _offline_record(step, baseline_p, probability_array, action, modification)
                    )
        except Exception as error:
            result["status"] = "failed"
            result["comparable"] = False
            result["failure"] = {
                "stage": "offline_predict",
                "step": len(output_records),
                "error": f"{type(error).__name__}: {error}",
            }
        result["records"] = output_records
        results.append(result)
    return results


def _offline_record(
    step: int,
    p: np.ndarray,
    q: np.ndarray,
    action: int,
    modification: Any,
) -> dict[str, object]:
    baseline_action = int(np.argmax(p))
    # ``argmax_changed`` is a distribution comparison at the same saved
    # timestep.  It must not depend on a sampled/non-deterministic action
    # returned by the adapter.
    q_action = int(np.argmax(q))
    changed = bool(baseline_action != q_action)
    return {
        "step": int(step),
        "js": js_divergence(p, q),
        "argmax_changed": changed,
        "p": p.tolist(),
        "q": q.tolist(),
        "applied": bool(modification.applied),
        "changed": bool(modification.changed),
        "changed_dimensions": int(modification.changed_dimensions),
        "baseline_action": baseline_action,
        "action": int(action),
    }


# Explicit aliases keep the API easy to discover during porting.
run_offline_interventions = compare_saved_baseline
compare_policy_inputs = compare_saved_baseline
jensen_shannon_divergence = js_divergence


__all__ = [
    "PolicyComparisonError",
    "compare_policy_inputs",
    "compare_saved_baseline",
    "identity_digest",
    "jensen_shannon_divergence",
    "js_divergence",
    "policy_identity",
    "run_offline_interventions",
]
