"""Environment-free ①-A analysis over saved model observations."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .interventions import (
    InterventionPattern,
    InterventionResult,
    apply_intervention,
    generate_individual_patterns,
    patterns_from_config,
)
from .metrics import (
    DistributionComparison,
    compare_distributions,
    marginal_probability_comparison,
    summarize_distribution_comparison,
)
from .policy import PolicyLike, as_policy_adapter
from .schema import InputSchema, SchemaError


class OfflineAnalysisError(ValueError):
    """Raised when saved observations cannot be analyzed safely."""


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _validate_batch(observations: Any, schema: InputSchema) -> np.ndarray:
    array = np.asarray(observations)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise OfflineAnalysisError(
            f"observations must have shape (time, dimension), got {array.shape}"
        )
    if array.shape[0] == 0:
        raise OfflineAnalysisError("at least one observation is required")
    rows: list[np.ndarray] = []
    for row in array:
        try:
            rows.append(schema.validate_observation(row, copy=True))
        except SchemaError as exc:
            raise OfflineAnalysisError(str(exc)) from exc
    return np.stack(rows, axis=0)


def _reject_unresolved_schema(schema: InputSchema) -> None:
    unresolved = [
        item.index
        for item in schema.inputs
        if item.id.startswith("UNRESOLVED_")
        or (isinstance(item.replacement, Mapping) and item.replacement.get("kind") in {"unresolved", "unknown", "unset"})
    ]
    if unresolved:
        raise OfflineAnalysisError(
            "schema contains unresolved input meanings; verify source before offline analysis: "
            f"indices={unresolved}"
        )


def _index_stats(
    observations: np.ndarray,
    pattern: InterventionPattern,
    changed_rows: np.ndarray,
    schema: InputSchema,
    valid_mask: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    requested = set(pattern.resolve_indices(schema))
    if valid_mask is None:
        valid = np.ones(observations.shape[0], dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
    result: list[dict[str, Any]] = []
    for spec in sorted(schema.inputs, key=lambda item: item.index):
        values = observations[:, spec.index].astype(np.float64, copy=False)
        targeted = spec.index in requested
        actual_changes = changed_rows[:, spec.index] & valid if targeted else np.zeros(values.shape[0], dtype=bool)
        finite = values[np.isfinite(values)]
        result.append(
            {
                "index": spec.index,
                "id": spec.id,
                "name_ja": spec.name_ja,
                "group": spec.group,
                "requested": targeted,
                "applied_count": int(np.sum(valid)) if targeted else 0,
                "actual_change_count": int(np.sum(actual_changes)),
                "no_op_count": int(np.sum(valid) - np.sum(actual_changes)) if targeted else 0,
                "original_min": float(np.min(finite)) if finite.size else None,
                "original_max": float(np.max(finite)) if finite.size else None,
                "original_mean": float(np.mean(finite)) if finite.size else None,
                "original_std": float(np.std(finite)) if finite.size else None,
                "constant": bool(finite.size > 0 and np.all(values == values[0])),
                "skip_reason": (
                    None
                    if targeted and np.any(valid)
                    else "all applications skipped" if targeted else "not targeted by this pattern"
                ),
            }
        )
    return result


def _episode_summaries(
    episodes: np.ndarray,
    comparison: DistributionComparison,
    changed_mask: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for episode in dict.fromkeys(episodes.tolist()):
        mask = episodes == episode
        result[str(episode)] = summarize_distribution_comparison(
            DistributionComparison(
                original_argmax=comparison.original_argmax[mask],
                changed_argmax=comparison.changed_argmax[mask],
                selected_actions=comparison.selected_actions[mask],
                selected_probability_original=comparison.selected_probability_original[mask],
                selected_probability_changed=comparison.selected_probability_changed[mask],
                selected_probability_delta=comparison.selected_probability_delta[mask],
                js=comparison.js[mask],
            ),
            changed_input_mask=changed_mask[mask],
            valid_mask=valid_mask[mask],
        )
    return result


@dataclass(frozen=True, slots=True)
class OfflinePatternResult:
    """Results for one intervention pattern."""

    pattern: InterventionPattern
    rows: tuple[dict[str, Any], ...]
    summary: Mapping[str, Any]
    altered_observations: np.ndarray
    changed_input_mask: np.ndarray
    changed_indices_by_row: tuple[tuple[int, ...], ...]
    changed_probabilities: np.ndarray

    def to_dict(self, *, include_arrays: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "pattern": self.pattern.to_dict(),
            "rows": [_json_value(row) for row in self.rows],
            "summary": _json_value(self.summary),
        }
        if include_arrays:
            result.update(
                {
                    "altered_observations": self.altered_observations.tolist(),
                    "changed_input_mask": self.changed_input_mask.tolist(),
                    "changed_indices_by_row": [list(item) for item in self.changed_indices_by_row],
                    "changed_probabilities": self.changed_probabilities.tolist(),
                }
            )
        return result


@dataclass(frozen=True, slots=True)
class OfflineAnalysisResult:
    """Serializable result of a complete ①-A run."""

    schema_dimension: int
    policy_fingerprint: str
    observations: np.ndarray
    original_probabilities: np.ndarray
    original_actions: np.ndarray
    episodes: np.ndarray
    steps: np.ndarray
    patterns: tuple[OfflinePatternResult, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_arrays: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_dimension": self.schema_dimension,
            "policy_fingerprint": self.policy_fingerprint,
            "metadata": _json_value(self.metadata),
            "patterns": [pattern.to_dict(include_arrays=include_arrays) for pattern in self.patterns],
        }
        if include_arrays:
            result.update(
                {
                    "observations": self.observations.tolist(),
                    "original_probabilities": self.original_probabilities.tolist(),
                    "original_actions": self.original_actions.tolist(),
                    "episodes": self.episodes.tolist(),
                    "steps": self.steps.tolist(),
                }
            )
        return result

    def save_json(self, path: str | Path, *, include_arrays: bool = True) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as stream:
            json.dump(self.to_dict(include_arrays=include_arrays), stream, ensure_ascii=False, indent=2)
            stream.write("\n")


def _build_rows(
    *,
    pattern: InterventionPattern,
    episodes: np.ndarray,
    steps: np.ndarray,
    comparison: DistributionComparison,
    interventions: Sequence[InterventionResult],
    marginals: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for index, intervention in enumerate(interventions):
        rows.append(
            {
                "pattern_id": pattern.pattern_id,
                "episode": _json_value(episodes[index]),
                "step": _json_value(steps[index]),
                "requested_indices": list(intervention.requested_indices),
                "changed_indices": list(intervention.changed_indices),
                "no_op_indices": list(intervention.no_op_indices),
                "actual_input_changed": intervention.changed,
                "skipped": intervention.skipped,
                "skip_reason": intervention.skip_reason,
                "original_argmax": int(comparison.original_argmax[index]),
                "changed_argmax": int(comparison.changed_argmax[index]),
                "action_changed": bool(comparison.action_changed[index]),
                "selected_action": int(comparison.selected_actions[index]),
                "selected_probability_original": float(comparison.selected_probability_original[index]),
                "selected_probability_changed": float(comparison.selected_probability_changed[index]),
                "selected_probability_delta": float(comparison.selected_probability_delta[index]),
                "selected_probability_delta_pp": float(comparison.selected_probability_delta_pp[index]),
                "js_divergence": float(comparison.js[index]),
            }
        )
        if marginals:
            rows[-1]["action_marginal_probability_delta"] = {
                dimension: {
                    category: float(values["delta"][category][index])
                    for category in values["categories"]
                }
                for dimension, values in marginals.items()
            }
            rows[-1]["action_marginal_probability_delta_pp"] = {
                dimension: {
                    category: float(values["delta_pp"][category][index])
                    for category in values["categories"]
                }
                for dimension, values in marginals.items()
            }
    return tuple(rows)


def analyze_offline(
    observations: Any,
    policy: Any,
    schema: InputSchema,
    patterns: Iterable[InterventionPattern | Mapping[str, Any]] | None = None,
    *,
    actions: Sequence[int] | np.ndarray | None = None,
    episode_ids: Sequence[Any] | np.ndarray | None = None,
    steps: Sequence[int] | np.ndarray | None = None,
    reference_observations: Mapping[str, Any] | None = None,
    reference_contexts: Mapping[str, Mapping[str, Any]] | None = None,
    reference_id: str | None = None,
    observation_contexts: Sequence[Mapping[str, Any] | None] | None = None,
    strict: bool = True,
    action_mapping: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> OfflineAnalysisResult:
    """Run ①-A independently from each saved original observation.

    No environment object is accepted or called.  Each pattern is applied to a
    fresh copy of every original row, so an intervention at time ``t`` cannot
    leak into time ``t+1``.
    """

    try:
        schema.validate_for_execution()
    except SchemaError as exc:
        raise OfflineAnalysisError(f"schema is not ready for offline execution: {exc}") from exc
    _reject_unresolved_schema(schema)
    original_observations = _validate_batch(observations, schema)
    time_count = original_observations.shape[0]
    adapter = as_policy_adapter(policy)
    fingerprint_fn = getattr(adapter, "fingerprint", None)
    fingerprint_before: str | None = None
    if callable(fingerprint_fn):
        try:
            fingerprint_before = str(fingerprint_fn())
        except Exception as exc:
            raise OfflineAnalysisError(f"failed to fingerprint policy before analysis: {exc}") from exc
    set_eval = getattr(adapter, "set_eval", None)
    if callable(set_eval):
        try:
            set_eval()
        except Exception as exc:
            raise OfflineAnalysisError(f"failed to set policy to evaluation mode: {exc}") from exc
    try:
        original_probabilities = adapter.probabilities(original_observations)
    except Exception as exc:
        if isinstance(exc, OfflineAnalysisError):
            raise
        raise OfflineAnalysisError(f"failed to evaluate original observations: {exc}") from exc
    if original_probabilities.shape[0] != time_count:
        raise OfflineAnalysisError("policy returned an unexpected number of original distributions")
    action_count = original_probabilities.shape[1]
    if actions is None:
        original_actions = np.argmax(original_probabilities, axis=1).astype(np.int64)
    else:
        original_actions = np.asarray(actions)
        if original_actions.ndim != 1 or original_actions.shape[0] != time_count:
            raise OfflineAnalysisError("actions must contain one action per saved observation")
        if not np.all(np.equal(original_actions, np.floor(original_actions))):
            raise OfflineAnalysisError("actions must be integer-valued")
        original_actions = original_actions.astype(np.int64)
        if np.any(original_actions < 0) or np.any(original_actions >= action_count):
            raise OfflineAnalysisError("actions contain an out-of-range value")
    if episode_ids is None:
        episodes = np.zeros(time_count, dtype=np.int64)
    else:
        episodes = np.asarray(episode_ids)
        if episodes.ndim != 1 or episodes.shape[0] != time_count:
            raise OfflineAnalysisError("episode_ids must contain one id per observation")
    if steps is None:
        step_values = np.arange(time_count, dtype=np.int64)
    else:
        step_values = np.asarray(steps)
        if step_values.ndim != 1 or step_values.shape[0] != time_count:
            raise OfflineAnalysisError("steps must contain one value per observation")
    if observation_contexts is None:
        contexts: list[Mapping[str, Any] | None] = [None] * time_count
    else:
        contexts = list(observation_contexts)
        if len(contexts) != time_count:
            raise OfflineAnalysisError("observation_contexts must contain one mapping per observation")
        if any(context is not None and not isinstance(context, Mapping) for context in contexts):
            raise OfflineAnalysisError("observation_contexts entries must be mappings or None")

    resolved_patterns = (
        (
            InterventionPattern("P00", method="identity", description="unchanged reference"),
            *generate_individual_patterns(schema, reference_id=reference_id),
        )
        if patterns is None
        else patterns_from_config(patterns)
    )
    if not resolved_patterns:
        raise OfflineAnalysisError("at least one intervention pattern is required")
    outputs: list[OfflinePatternResult] = []
    for pattern in resolved_patterns:
        interventions: list[InterventionResult] = []
        altered_rows: list[np.ndarray] = []
        changed_mask = np.zeros((time_count, schema.dimension), dtype=bool)
        changed_row_indices: list[tuple[int, ...]] = []
        for row_index, row in enumerate(original_observations):
            try:
                intervention = apply_intervention(
                    row,
                    pattern,
                    schema,
                    reference_observations=reference_observations,
                    reference_contexts=reference_contexts,
                    observation_context=contexts[row_index],
                    strict=strict,
                )
            except Exception as exc:
                if isinstance(exc, OfflineAnalysisError):
                    raise
                raise OfflineAnalysisError(
                    f"pattern {pattern.pattern_id} failed: {exc}"
                ) from exc
            interventions.append(intervention)
            altered_rows.append(intervention.observation)
            changed_mask[len(interventions) - 1, list(intervention.changed_indices)] = True
            changed_row_indices.append(intervention.changed_indices)
        altered_observations = np.stack(altered_rows, axis=0)
        try:
            changed_probabilities = adapter.probabilities(altered_observations)
        except Exception as exc:
            raise OfflineAnalysisError(
                f"policy evaluation failed for pattern {pattern.pattern_id}: {exc}"
            ) from exc
        if changed_probabilities.shape != original_probabilities.shape:
            raise OfflineAnalysisError(
                f"pattern {pattern.pattern_id} returned probability shape {changed_probabilities.shape}, expected {original_probabilities.shape}"
            )
        comparison = compare_distributions(
            original_probabilities,
            changed_probabilities,
            selected_actions=original_actions,
        )
        marginals = (
            marginal_probability_comparison(
                original_probabilities,
                changed_probabilities,
                action_mapping,
            )
            if action_mapping
            else None
        )
        rows = _build_rows(
            pattern=pattern,
            episodes=episodes,
            steps=step_values,
            comparison=comparison,
            interventions=interventions,
            marginals=marginals,
        )
        actual_row_mask = np.any(changed_mask, axis=1)
        valid_row_mask = np.asarray([not item.skipped for item in interventions], dtype=bool)
        summary = summarize_distribution_comparison(
            comparison,
            changed_input_mask=actual_row_mask,
            episodes=episodes,
            valid_mask=valid_row_mask,
        )
        valid_count = int(np.sum(valid_row_mask))
        actual_valid_count = int(np.sum(actual_row_mask & valid_row_mask))
        no_op_valid_count = valid_count - actual_valid_count
        action_marginal_summary: dict[str, Any] = {}
        if marginals:
            for dimension, values in marginals.items():
                action_marginal_summary[dimension] = {
                    "actions": values["actions"],
                    "delta_pp_mean": {
                        category: float(np.mean(values["delta_pp"][category][valid_row_mask]))
                        if valid_count
                        else None
                        for category in values["categories"]
                    },
                }
        summary = {
            **summary,
            "pattern_id": pattern.pattern_id,
            "requested_indices": list(pattern.resolve_indices(schema)),
            "requested_dimension_count": len(pattern.resolve_indices(schema)),
            "actual_changed_row_count": actual_valid_count,
            "no_op_row_count": no_op_valid_count,
            "applied_row_count": valid_count,
            "skipped_row_count": int(sum(item.skipped for item in interventions)),
            "valid_row_count": valid_count,
            "skip_reasons": sorted({item.skip_reason for item in interventions if item.skip_reason}),
            "action_count": action_count,
            "per_input": _index_stats(original_observations, pattern, changed_mask, schema, valid_row_mask),
            "episode_summaries": _episode_summaries(episodes, comparison, actual_row_mask, valid_row_mask),
            "action_marginals": action_marginal_summary,
            "group_note": "per-input and group results are diagnostic comparisons; no composite importance score is computed",
        }
        outputs.append(
            OfflinePatternResult(
                pattern=pattern,
                rows=rows,
                summary=summary,
                altered_observations=altered_observations,
                changed_input_mask=changed_mask,
                changed_indices_by_row=tuple(changed_row_indices),
                changed_probabilities=changed_probabilities,
            )
        )
    fingerprint_after: str | None = None
    if callable(fingerprint_fn):
        try:
            fingerprint_after = str(fingerprint_fn())
        except Exception as exc:
            raise OfflineAnalysisError(f"failed to fingerprint policy after analysis: {exc}") from exc
    policy_unchanged: bool | None = None
    if fingerprint_before is not None and fingerprint_after is not None:
        policy_unchanged = fingerprint_before == fingerprint_after
        if not policy_unchanged:
            raise OfflineAnalysisError(
                "policy parameters/buffers changed during offline analysis"
            )
    result_metadata = dict(metadata or {})
    result_metadata.update(
        {
            "policy_fingerprint_before": fingerprint_before,
            "policy_fingerprint_after": fingerprint_after,
            "policy_unchanged": policy_unchanged,
        }
    )
    return OfflineAnalysisResult(
        schema_dimension=schema.dimension,
        policy_fingerprint=fingerprint_after or fingerprint_before or "",
        observations=np.array(original_observations, copy=True),
        original_probabilities=np.array(original_probabilities, copy=True),
        original_actions=np.array(original_actions, copy=True),
        episodes=np.array(episodes, copy=True),
        steps=np.array(step_values, copy=True),
        patterns=tuple(outputs),
        metadata=result_metadata,
    )


def run_offline(*args: Any, **kwargs: Any) -> OfflineAnalysisResult:
    """Alias for the public ①-A entry point."""

    return analyze_offline(*args, **kwargs)


def load_offline_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)
