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
    intervention_counts,
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


def _telemetry_time_s(context: Mapping[str, Any] | None) -> float | None:
    """Return a declared pre-action simulator time from an observation context.

    Offline rows also carry an observation ``step`` index, but that index is
    not a clock.  Keep the time unavailable when the saved telemetry has no
    explicit time field instead of manufacturing seconds from the index.
    """

    if not isinstance(context, Mapping):
        return None
    for key in (
        "simulation_time_s",
        "simulation_time_seconds",
        "simulation_time",
        "sim_time_seconds",
        "sim_time_s",
        "sim_time",
        "time_s",
        "time",
    ):
        value = context.get(key)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if np.isfinite(numeric):
            return numeric
    return None


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
    intervention_counts: Mapping[str, Any] = field(default_factory=dict)
    step_masks: Mapping[str, np.ndarray] = field(default_factory=dict)
    road_segment_ids: tuple[Any, ...] = ()

    def to_dict(self, *, include_arrays: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "pattern": self.pattern.to_dict(),
            "rows": [_json_value(row) for row in self.rows],
            "summary": _json_value(self.summary),
            "intervention_counts": _json_value(self.intervention_counts),
            "road_segment_ids": _json_value(self.road_segment_ids),
        }
        if include_arrays:
            result.update(
                {
                    "altered_observations": self.altered_observations.tolist(),
                    "changed_input_mask": self.changed_input_mask.tolist(),
                    "changed_indices_by_row": [list(item) for item in self.changed_indices_by_row],
                    "changed_probabilities": self.changed_probabilities.tolist(),
                    "step_masks": {
                        str(name): np.asarray(values, dtype=bool).tolist()
                        for name, values in self.step_masks.items()
                    },
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
    observations: np.ndarray,
    changed_indices_by_row: Sequence[tuple[int, ...]],
    schema: InputSchema,
    observation_contexts: Sequence[Mapping[str, Any] | None] | None = None,
    marginals: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for index, intervention in enumerate(interventions):
        changed_indices = tuple(changed_indices_by_row[index])
        requested_indices = tuple(
            int(value)
            for value in getattr(intervention, "requested_indices", pattern.resolve_indices(schema))
        )
        original_values = {
            str(input_index): _json_value(observations[index, input_index])
            for input_index in requested_indices
        }
        modified_values = {
            str(input_index): _json_value(intervention.observation[input_index])
            for input_index in requested_indices
        }
        delta_values = {
            input_index: _json_value(
                np.asarray(intervention.observation[input_index], dtype=np.float64)
                - np.asarray(observations[index, input_index], dtype=np.float64)
            )
            for input_index in requested_indices
        }
        delta_abs_values = {
            input_index: _json_value(abs(float(delta_values[input_index])))
            for input_index in requested_indices
            if isinstance(delta_values[input_index], (int, float, np.integer, np.floating))
        }
        context = observation_contexts[index] if observation_contexts is not None else None
        road_segment_id = context.get("road_segment_id") if isinstance(context, Mapping) else None
        time_s = _telemetry_time_s(context)
        eligible = getattr(intervention, "eligible", None)
        if eligible is None:
            eligible = not bool(getattr(intervention, "skipped", False))
        reason_text = str(getattr(intervention, "skip_reason", "") or "").casefold()
        if getattr(pattern, "scope", None) == "explicitly_conditional" and any(
            token in reason_text
            for token in ("compatibility context", "reference context", "reference observation")
        ):
            eligible = False
        applied = getattr(intervention, "applied", None)
        if applied is None:
            applied = bool(eligible) and not bool(getattr(intervention, "skipped", False))
        skipped = bool(getattr(intervention, "skipped", False))
        rows.append(
            {
                "pattern_id": pattern.pattern_id,
                "episode": _json_value(episodes[index]),
                "step": _json_value(steps[index]),
                "requested_indices": list(requested_indices),
                "changed_indices": list(changed_indices),
                "no_op_indices": (
                    [item for item in requested_indices if item not in changed_indices]
                    if applied and not skipped
                    else []
                ),
                "actual_input_changed": bool(changed_indices),
                "changed_exact": bool(changed_indices),
                "eligible": bool(eligible),
                "applied": bool(applied),
                "skipped": skipped,
                "skip_reason": getattr(intervention, "skip_reason", None),
                "skip_category": getattr(intervention, "skip_category", None),
                "road_segment_id": _json_value(road_segment_id),
                "time_s": time_s,
                "original_values": original_values,
                "modified_values": modified_values,
                "delta_values": delta_values,
                "delta_abs_values": delta_abs_values,
                "meaningful_changed_indices": list(
                    getattr(intervention, "meaningful_changed_indices", ()) or ()
                ),
                "meaningful_changed_count": int(
                    getattr(intervention, "meaningful_changed_count", 0) or 0
                ),
                "tolerance": _json_value(getattr(intervention, "tolerance", {})),
                "clipped_indices": list(getattr(intervention, "clipped_indices", ()) or ()),
                "clip_count": int(getattr(intervention, "clip_count", 0) or 0),
                "original_argmax": int(comparison.original_argmax[index]),
                "changed_argmax": int(comparison.changed_argmax[index]),
                "action_changed": bool(comparison.action_changed[index]),
                "selected_action": int(comparison.selected_actions[index]),
                "selected_probability_original": float(comparison.selected_probability_original[index]),
                "selected_probability_changed": float(comparison.selected_probability_changed[index]),
                "selected_probability_delta": float(comparison.selected_probability_delta[index]),
                "selected_probability_delta_pp": float(comparison.selected_probability_delta_pp[index]),
                "selected_probability_abs_delta": float(comparison.selected_probability_abs_delta[index]),
                "selected_probability_abs_delta_pp": float(comparison.selected_probability_abs_delta_pp[index]),
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
        eligible_mask = np.zeros(time_count, dtype=bool)
        applied_mask = np.zeros(time_count, dtype=bool)
        meaningful_mask = np.zeros(time_count, dtype=bool)
        road_segment_ids: list[Any] = []
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
            altered = np.asarray(intervention.observation).copy()
            altered_rows.append(altered)
            requested_indices = tuple(
                int(value)
                for value in getattr(intervention, "requested_indices", pattern.resolve_indices(schema))
            )
            # Recompute exact changes from the final dtype passed to the policy.
            # This deliberately keeps tiny representational differences in the
            # exact count; meaningful thresholds are a separate core field.
            exact_changed = tuple(
                index
                for index in requested_indices
                if not np.array_equal(row[index], altered[index], equal_nan=False)
            )
            changed_mask[row_index, list(exact_changed)] = True
            changed_row_indices.append(exact_changed)
            skipped = bool(getattr(intervention, "skipped", False))
            skip_reason_text = str(getattr(intervention, "skip_reason", "") or "").casefold()
            declared_out_of_scope = bool(
                getattr(pattern, "scope", None) == "explicitly_conditional"
                and any(
                    token in skip_reason_text
                    for token in ("compatibility context", "reference context", "reference observation")
                )
            )
            eligible = getattr(intervention, "eligible", None)
            if eligible is None:
                eligible = not skipped
            if declared_out_of_scope:
                eligible = False
            applied = getattr(intervention, "applied", None)
            if applied is None:
                applied = bool(eligible) and not skipped
            eligible_mask[row_index] = bool(eligible)
            applied_mask[row_index] = bool(applied) and not skipped
            meaningful = getattr(intervention, "meaningful", None)
            if meaningful is None:
                meaningful = getattr(intervention, "meaningful_changed", None)
            if meaningful is None:
                meaningful_indices = getattr(intervention, "meaningful_changed_indices", ()) or ()
                tolerance_values = getattr(intervention, "tolerance", {}) or {}
                if meaningful_indices or tolerance_values:
                    meaningful = getattr(intervention, "meaningful_changed_count", None)
            if meaningful is not None:
                meaningful_mask[row_index] = bool(meaningful)
            context = contexts[row_index]
            road_segment_ids.append(
                _json_value(context.get("road_segment_id")) if isinstance(context, Mapping) else None
            )
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
            observations=original_observations,
            changed_indices_by_row=changed_row_indices,
            schema=schema,
            observation_contexts=contexts,
            marginals=marginals,
        )
        actual_row_mask = np.any(changed_mask, axis=1)
        valid_row_mask = np.array(applied_mask, copy=True)
        summary = summarize_distribution_comparison(
            comparison,
            changed_input_mask=actual_row_mask,
            episodes=episodes,
            valid_mask=valid_row_mask,
        )
        valid_count = int(np.sum(valid_row_mask))
        actual_valid_count = int(np.sum(actual_row_mask & valid_row_mask))
        no_op_valid_count = valid_count - actual_valid_count
        counts = intervention_counts(
            interventions,
            target_step_count=time_count,
            declared_out_of_scope=[
                bool(
                    getattr(pattern, "scope", None) == "explicitly_conditional"
                    and any(
                        token in str(getattr(item, "skip_reason", "") or "").casefold()
                        for token in ("compatibility context", "reference context", "reference observation")
                    )
                )
                for item in interventions
            ],
        )
        counts.update(
            {
                "eligible_count": int(np.sum(eligible_mask)),
                "applied_count": int(np.sum(applied_mask)),
                "changed_count_exact": actual_valid_count,
                "changed_count": actual_valid_count,
                "noop_count": no_op_valid_count,
                "meaningful_changed_count": int(np.sum(meaningful_mask))
                if any(
                    getattr(item, "meaningful", None) is not None
                    or getattr(item, "meaningful_changed", None) is not None
                    or bool(getattr(item, "meaningful_changed_indices", ()) or ())
                    or bool(getattr(item, "tolerance", {}) or {})
                    for item in interventions
                )
                else None,
            }
        )
        # Keep denominator/rates consistent with the exact dtype comparison.
        counts["applied_rate_over_target"] = float(np.sum(applied_mask) / time_count) if time_count else None
        counts["eligible_rate_over_target"] = float(np.sum(eligible_mask) / time_count) if time_count else None
        counts["changed_rate_over_applied"] = float(actual_valid_count / valid_count) if valid_count else None
        counts["noop_rate_over_applied"] = float(no_op_valid_count / valid_count) if valid_count else None
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
        per_input_summary = _index_stats(
            original_observations,
            pattern,
            changed_mask,
            schema,
            valid_row_mask,
        )
        for input_summary in per_input_summary:
            delta_summary = counts.get("per_input_delta", {}).get(str(input_summary["index"]))
            if isinstance(delta_summary, Mapping):
                input_summary.update(delta_summary)
        summary = {
            **summary,
            "pattern_id": pattern.pattern_id,
            "requested_indices": list(pattern.resolve_indices(schema)),
            "requested_dimension_count": len(pattern.resolve_indices(schema)),
            "actual_changed_row_count": actual_valid_count,
            "no_op_row_count": no_op_valid_count,
            "applied_row_count": valid_count,
            "eligible_row_count": int(np.sum(eligible_mask)),
            "skipped_row_count": int(sum(bool(getattr(item, "skipped", False)) for item in interventions)),
            "changed_count_exact": actual_valid_count,
            "noop_count": no_op_valid_count,
            "changed_element_count_exact": counts.get("changed_element_count_exact"),
            "applied_element_count": counts.get("applied_element_count"),
            "noop_element_count": counts.get("noop_element_count"),
            "target_step_count": time_count,
            "valid_row_count": valid_count,
            "skip_reasons": sorted({item.skip_reason for item in interventions if item.skip_reason}),
            "intervention_counts": counts,
            "step_masks": {
                "target": np.ones(time_count, dtype=bool).tolist(),
                "eligible": eligible_mask.tolist(),
                "applied": applied_mask.tolist(),
                "changed_exact": np.any(changed_mask, axis=1).tolist(),
                "meaningful_changed": meaningful_mask.tolist(),
                "skipped": (~applied_mask).tolist(),
            },
            "road_segment_ids": road_segment_ids,
            "action_count": action_count,
            "per_input": per_input_summary,
            "episode_summaries": _episode_summaries(episodes, comparison, actual_row_mask, valid_row_mask),
            "action_marginals": action_marginal_summary,
            "execution_status": "completed",
            "assessment_status": (
                "control_baseline"
                if pattern.pattern_id == "P00"
                else "not_evaluable_no_application"
                if valid_count == 0
                else "conditional_or_partial_application"
                if valid_count < time_count
                else "numerical_only"
                if actual_valid_count > 0 and counts.get("meaningful_changed_count") == 0
                else "no_exact_input_change"
                if actual_valid_count == 0
                else "evaluated"
            ),
            "scope": getattr(pattern, "scope", None),
            "on_inapplicable": getattr(pattern, "on_inapplicable", None),
            "variant_id": getattr(pattern, "variant_id", None),
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
                intervention_counts=counts,
                step_masks={
                    "target": np.ones(time_count, dtype=bool),
                    "eligible": eligible_mask,
                    "applied": applied_mask,
                    "changed_exact": np.any(changed_mask, axis=1),
                    "meaningful_changed": meaningful_mask,
                    "skipped": ~applied_mask,
                },
                road_segment_ids=tuple(road_segment_ids),
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
