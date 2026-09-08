"""Probability based comparison metrics for offline interventions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


class MetricError(ValueError):
    """Raised for invalid probability distributions or aggregation inputs."""


def _probability_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] < 2:
        raise MetricError(f"{name} must have shape (batch, actions>=2)")
    if not np.all(np.isfinite(array)):
        raise MetricError(f"{name} contains non-finite probabilities")
    if np.any(array < 0.0):
        raise MetricError(f"{name} contains negative probabilities")
    sums = array.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-7, rtol=1e-7):
        raise MetricError(f"{name} rows must sum to 1 (got {sums.tolist()})")
    return array


def _safe_xlogy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    result = np.zeros_like(x, dtype=np.float64)
    mask = x > 0.0
    result[mask] = x[mask] * np.log(y[mask])
    return result


def js_divergence(
    p: Any,
    q: Any,
    *,
    axis: int = -1,
) -> np.ndarray | float:
    """Return Jensen-Shannon divergence using natural logarithms.

    The result is in ``[0, ln(2)]`` for probability distributions.  Zero
    probability terms contribute zero and never produce a NaN.
    """

    p_array = _probability_array(p, name="p")
    q_array = _probability_array(q, name="q")
    if p_array.shape != q_array.shape:
        raise MetricError(f"p and q shapes differ: {p_array.shape} vs {q_array.shape}")
    if axis not in {-1, 1}:
        raise MetricError("js_divergence currently supports the action axis only")
    midpoint = 0.5 * (p_array + q_array)
    # m is positive wherever p or q is positive.  The helper masks x == 0,
    # implementing the mathematical convention 0 log(0/m) == 0.
    p_term = _safe_xlogy(p_array, np.divide(p_array, midpoint, out=np.ones_like(p_array), where=midpoint > 0))
    q_term = _safe_xlogy(q_array, np.divide(q_array, midpoint, out=np.ones_like(q_array), where=midpoint > 0))
    value = 0.5 * (p_term.sum(axis=1) + q_term.sum(axis=1))
    value = np.clip(value, 0.0, math.log(2.0))
    return float(value[0]) if np.asarray(p).ndim == 1 else value


def kl_divergence(p: Any, q: Any) -> np.ndarray | float:
    """Return KL(p || q) with the standard zero-term convention."""

    p_array = _probability_array(p, name="p")
    q_array = _probability_array(q, name="q")
    if p_array.shape != q_array.shape:
        raise MetricError("p and q shapes differ")
    if np.any((p_array > 0.0) & (q_array == 0.0)):
        result = np.full((p_array.shape[0],), np.inf, dtype=np.float64)
    else:
        ratio = np.divide(p_array, q_array, out=np.ones_like(p_array), where=q_array > 0)
        result = _safe_xlogy(p_array, ratio).sum(axis=1)
    return float(result[0]) if np.asarray(p).ndim == 1 else result


def _finite_summary(values: np.ndarray) -> dict[str, float | int | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "count": int(values.size),
            "finite_count": 0,
            "mean": None,
            "median": None,
            "p95": None,
        }
    return {
        "count": int(values.size),
        "finite_count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95)),
    }


@dataclass(frozen=True, slots=True)
class DistributionComparison:
    """Per-observation policy distribution comparison."""

    original_argmax: np.ndarray
    changed_argmax: np.ndarray
    selected_actions: np.ndarray
    selected_probability_original: np.ndarray
    selected_probability_changed: np.ndarray
    selected_probability_delta: np.ndarray
    js: np.ndarray

    @property
    def action_changed(self) -> np.ndarray:
        return self.original_argmax != self.changed_argmax

    @property
    def selected_probability_delta_pp(self) -> np.ndarray:
        """Selected-action probability change in percentage points."""

        return self.selected_probability_delta * 100.0

    @property
    def selected_probability_abs_delta(self) -> np.ndarray:
        """Absolute probability change for the original policy's action."""

        return np.abs(self.selected_probability_delta)

    @property
    def selected_probability_abs_delta_pp(self) -> np.ndarray:
        """Absolute selected-action probability change in percentage points."""

        return self.selected_probability_abs_delta * 100.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_argmax": self.original_argmax.tolist(),
            "changed_argmax": self.changed_argmax.tolist(),
            "selected_actions": self.selected_actions.tolist(),
            "selected_probability_original": self.selected_probability_original.tolist(),
            "selected_probability_changed": self.selected_probability_changed.tolist(),
            "selected_probability_delta": self.selected_probability_delta.tolist(),
            "selected_probability_delta_pp": self.selected_probability_delta_pp.tolist(),
            "selected_probability_abs_delta": self.selected_probability_abs_delta.tolist(),
            "selected_probability_abs_delta_pp": self.selected_probability_abs_delta_pp.tolist(),
            "js": self.js.tolist(),
            "action_changed": self.action_changed.tolist(),
        }


def compare_distributions(
    original: Any,
    changed: Any,
    *,
    selected_actions: Sequence[int] | np.ndarray | None = None,
) -> DistributionComparison:
    """Compare p and q for each saved observation.

    When ``selected_actions`` is omitted, the original policy argmax is used as
    the action whose probability change is reported.  This matches the
    offline experiment definition and avoids treating the replacement action as
    the selected action.
    """

    p = _probability_array(original, name="original")
    q = _probability_array(changed, name="changed")
    if p.shape != q.shape:
        raise MetricError(f"original and changed shapes differ: {p.shape} vs {q.shape}")
    original_argmax = np.argmax(p, axis=1).astype(np.int64)
    changed_argmax = np.argmax(q, axis=1).astype(np.int64)
    if selected_actions is None:
        actions = original_argmax
    else:
        actions = np.asarray(selected_actions)
        if actions.ndim != 1 or actions.shape[0] != p.shape[0]:
            raise MetricError("selected_actions must have one action per observation")
        if np.issubdtype(actions.dtype, np.floating) and not np.all(np.equal(actions, np.floor(actions))):
            raise MetricError("selected_actions must be integer-valued")
        actions = actions.astype(np.int64)
        if np.any(actions < 0) or np.any(actions >= p.shape[1]):
            raise MetricError("selected_actions contains an out-of-range action")
    rows = np.arange(p.shape[0])
    p_selected = p[rows, actions]
    q_selected = q[rows, actions]
    js = np.asarray(js_divergence(p, q), dtype=np.float64)
    return DistributionComparison(
        original_argmax=original_argmax,
        changed_argmax=changed_argmax,
        selected_actions=actions,
        selected_probability_original=p_selected,
        selected_probability_changed=q_selected,
        selected_probability_delta=q_selected - p_selected,
        js=js,
    )


def compare_policy_outputs(*args: Any, **kwargs: Any) -> DistributionComparison:
    """Alias used by offline callers."""

    return compare_distributions(*args, **kwargs)


def _resolve_action_groups(
    specification: Any,
    action_count: int,
) -> dict[str, tuple[int, ...]]:
    """Normalize confirmed action mapping forms to category -> action ids."""

    if isinstance(specification, Mapping):
        # Common config form: {"left": [0, 3, 6], "right": [2, 5, 8]}.
        if all(isinstance(value, (list, tuple, set, np.ndarray)) for value in specification.values()):
            groups = {
                str(category): tuple(int(action) for action in values)
                for category, values in specification.items()
            }
        else:
            # Alternate form: {"0": "left", "1": "left", ...}.
            groups = {}
            for action, category in specification.items():
                try:
                    action_id = int(action)
                except (TypeError, ValueError) as exc:
                    raise MetricError("action mapping keys must be actions or category names") from exc
                groups.setdefault(str(category), tuple())
                groups[str(category)] = (*groups[str(category)], action_id)
    elif isinstance(specification, (list, tuple, np.ndarray)):
        labels = list(specification)
        if len(labels) != action_count:
            raise MetricError("action mapping label sequence must have one label per action")
        groups = {}
        for action_id, category in enumerate(labels):
            groups.setdefault(str(category), tuple())
            groups[str(category)] = (*groups[str(category)], action_id)
    else:
        raise MetricError("action mapping must be a category mapping or action label sequence")
    normalized: dict[str, tuple[int, ...]] = {}
    seen: set[int] = set()
    for category, values in groups.items():
        if not values:
            raise MetricError(f"action category {category!r} has no actions")
        action_ids = tuple(dict.fromkeys(values))
        if any(action < 0 or action >= action_count for action in action_ids):
            raise MetricError(f"action category {category!r} contains an out-of-range action")
        if seen.intersection(action_ids):
            raise MetricError("action categories overlap; steering/throttle mapping must be disjoint")
        seen.update(action_ids)
        normalized[category] = action_ids
    all_actions = set(range(action_count))
    if seen != all_actions:
        missing = sorted(all_actions - seen)
        raise MetricError(
            "action mapping must cover every action exactly once; "
            f"missing actions={missing}"
        )
    return normalized


def marginal_probability_comparison(
    original: Any,
    changed: Any,
    action_mapping: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Compare confirmed steering/throttle/action marginals in each row.

    ``action_mapping`` maps a dimension name (for example ``steering``) to
    either category -> action ids or an action-label sequence.  Raw
    probabilities and percentage-point deltas are both retained.
    """

    p = _probability_array(original, name="original")
    q = _probability_array(changed, name="changed")
    if p.shape != q.shape:
        raise MetricError("original and changed shapes differ")
    if not isinstance(action_mapping, Mapping):
        raise MetricError("action_mapping must be a mapping of dimensions")
    result: dict[str, dict[str, Any]] = {}
    for dimension, specification in action_mapping.items():
        groups = _resolve_action_groups(specification, p.shape[1])
        original_values: dict[str, np.ndarray] = {}
        changed_values: dict[str, np.ndarray] = {}
        delta_values: dict[str, np.ndarray] = {}
        for category, action_ids in groups.items():
            original_value = p[:, action_ids].sum(axis=1)
            changed_value = q[:, action_ids].sum(axis=1)
            original_values[category] = original_value
            changed_values[category] = changed_value
            delta_values[category] = changed_value - original_value
        result[str(dimension)] = {
            "categories": list(groups),
            "actions": {category: list(actions) for category, actions in groups.items()},
            "original": original_values,
            "changed": changed_values,
            "delta": delta_values,
            "delta_pp": {category: value * 100.0 for category, value in delta_values.items()},
        }
    return result


def summarize_distribution_comparison(
    comparison: DistributionComparison,
    *,
    changed_input_mask: np.ndarray | Sequence[bool] | None = None,
    episodes: Sequence[Any] | np.ndarray | None = None,
    valid_mask: np.ndarray | Sequence[bool] | None = None,
) -> dict[str, Any]:
    """Summarize all timestamps and, separately, timestamps that changed input."""

    changed_actions = comparison.action_changed
    selected_delta = comparison.selected_probability_delta
    js = comparison.js
    if changed_input_mask is None:
        changed_mask = np.ones(changed_actions.shape[0], dtype=bool)
    else:
        changed_mask = np.asarray(changed_input_mask, dtype=bool)
        if changed_mask.shape != changed_actions.shape:
            raise MetricError("changed_input_mask must have one value per observation")
    if valid_mask is None:
        valid = np.ones(changed_actions.shape[0], dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != changed_actions.shape:
            raise MetricError("valid_mask must have one value per observation")

    def section(mask: np.ndarray) -> dict[str, Any]:
        if not np.any(mask):
            return {
                "count": 0,
                "action_change_count": 0,
                "action_change_rate": None,
                "selected_probability_delta": _finite_summary(np.asarray([], dtype=np.float64)),
                "selected_probability_abs_delta": _finite_summary(np.asarray([], dtype=np.float64)),
                "selected_probability_delta_pp": _finite_summary(np.asarray([], dtype=np.float64)),
                "selected_probability_delta_abs_pp": _finite_summary(np.asarray([], dtype=np.float64)),
                "selected_probability_abs_delta_pp": _finite_summary(np.asarray([], dtype=np.float64)),
                "js": _finite_summary(np.asarray([], dtype=np.float64)),
            }
        return {
            "count": int(np.sum(mask)),
            "action_change_count": int(np.sum(changed_actions[mask])),
            "action_change_rate": float(np.mean(changed_actions[mask])),
            "selected_probability_delta": _finite_summary(selected_delta[mask]),
            "selected_probability_abs_delta": _finite_summary(np.abs(selected_delta[mask])),
            "selected_probability_delta_pp": _finite_summary(selected_delta[mask] * 100.0),
            "selected_probability_delta_abs_pp": _finite_summary(np.abs(selected_delta[mask]) * 100.0),
            "selected_probability_abs_delta_pp": _finite_summary(np.abs(selected_delta[mask]) * 100.0),
            "js": _finite_summary(js[mask]),
        }

    result: dict[str, Any] = {
        "all_timestamps": section(valid),
        "actual_input_changes_only": section(changed_mask & valid),
    }
    if episodes is not None:
        episode_array = np.asarray(episodes)
        if episode_array.ndim != 1 or episode_array.shape[0] != changed_actions.shape[0]:
            raise MetricError("episodes must have one id per observation")
        episode_sections: dict[str, Any] = {}
        for episode in dict.fromkeys(episode_array.tolist()):
            mask = episode_array == episode
            episode_sections[str(episode)] = {
                "all_timestamps": section(mask & valid),
                "actual_input_changes_only": section(mask & changed_mask & valid),
            }
        result["episodes"] = episode_sections
        # Keep episode-mean and step-weighted views explicit.  The former gives
        # every episode equal weight; the latter pools timestamps in the same
        # mask used by ``all_timestamps``/``actual_input_changes_only``.
        for name, mask in (
            ("all_timestamps", valid),
            ("actual_input_changes_only", changed_mask & valid),
        ):
            means: list[float] = []
            counts: list[int] = []
            for episode in dict.fromkeys(episode_array.tolist()):
                episode_mask = (episode_array == episode) & mask
                if np.any(episode_mask):
                    means.append(float(np.mean(selected_delta[episode_mask])))
                    counts.append(int(np.sum(episode_mask)))
            result.setdefault("episode_mean", {})[name] = float(np.mean(means)) if means else None
            result.setdefault("step_weighted", {})[name] = (
                float(np.average(np.asarray(means), weights=np.asarray(counts)))
                if means and sum(counts)
                else None
            )
            abs_means: list[float] = []
            for episode in dict.fromkeys(episode_array.tolist()):
                episode_mask = (episode_array == episode) & mask
                if np.any(episode_mask):
                    abs_means.append(float(np.mean(np.abs(selected_delta[episode_mask]))))
            result.setdefault("episode_mean_abs", {})[name] = float(np.mean(abs_means)) if abs_means else None
            abs_values = np.abs(selected_delta[mask])
            result.setdefault("step_weighted_abs", {})[name] = (
                float(np.mean(abs_values)) if abs_values.size else None
            )
        # A compact, metric-complete view for consumers that need episode and
        # step weighting for JS as well as signed/absolute probability deltas.
        episode_mean_metrics: dict[str, Any] = {}
        step_weighted_metrics: dict[str, Any] = {}
        for name, mask in (
            ("all_timestamps", valid),
            ("actual_input_changes_only", changed_mask & valid),
        ):
            per_episode: dict[str, list[float]] = {
                "selected_probability_delta": [],
                "selected_probability_abs_delta": [],
                "js": [],
            }
            weighted_values: dict[str, list[tuple[float, int]]] = {
                key: [] for key in per_episode
            }
            for episode in dict.fromkeys(episode_array.tolist()):
                episode_mask = (episode_array == episode) & mask
                count = int(np.sum(episode_mask))
                if not count:
                    continue
                values_by_name = {
                    "selected_probability_delta": selected_delta[episode_mask],
                    "selected_probability_abs_delta": np.abs(selected_delta[episode_mask]),
                    "js": js[episode_mask],
                }
                for metric_name, values in values_by_name.items():
                    mean_value = float(np.mean(values))
                    per_episode[metric_name].append(mean_value)
                    weighted_values[metric_name].append((mean_value, count))
            episode_mean_metrics[name] = {
                metric_name: float(np.mean(values)) if values else None
                for metric_name, values in per_episode.items()
            }
            step_weighted_metrics[name] = {
                metric_name: (
                    float(sum(value * weight for value, weight in values) / sum(weight for _, weight in values))
                    if values and sum(weight for _, weight in values)
                    else None
                )
                for metric_name, values in weighted_values.items()
            }
        result["episode_mean_metrics"] = episode_mean_metrics
        result["step_weighted_metrics"] = step_weighted_metrics
    return result


def _intervention_bool(item: Any, name: str, *, default: bool = False) -> bool:
    """Read a bool-like intervention field across old and new result objects."""

    value = item.get(name, default) if isinstance(item, Mapping) else getattr(item, name, default)
    if value is None:
        return bool(default)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return bool(value)


def _intervention_reason(item: Any) -> str | None:
    if isinstance(item, Mapping):
        value = item.get("skip_reason")
    else:
        value = getattr(item, "skip_reason", None)
    return None if value in (None, "") else str(value)


def _reason_category(reason: str | None, *, declared_out_of_scope: bool = False) -> str | None:
    """Classify a skipped application without treating every skip as a no-op."""

    if declared_out_of_scope:
        return "out_of_scope"
    if reason is None:
        return None
    lowered = reason.casefold()
    if any(
        token in lowered
        for token in ("out_of_scope", "out-of-scope", "outside scope", "declared inapplicable")
    ):
        return "out_of_scope"
    return "failed"


def summarize_intervention_records(
    interventions: Sequence[Any] | None = None,
    *,
    records: Sequence[Mapping[str, Any]] | None = None,
    target_step_count: int | None = None,
    declared_out_of_scope: Sequence[bool] | None = None,
) -> dict[str, Any]:
    """Count intervention applicability independently of policy/trajectory metrics.

    Counts are step-based: a pattern applied to several indices at one timestamp
    contributes one applied step. ``applied = changed_exact + noop`` is kept as
    an invariant, and skipped steps never enter the no-op denominator. Both
    live ``InterventionResult`` objects and serialized mappings are accepted so
    A and B can report their own observations.
    """

    if interventions is None:
        items: list[Any] = []
        if records is not None:
            items = [
                row.get("intervention") if "intervention" in row else None
                for row in records
                if isinstance(row, Mapping)
            ]
    else:
        items = list(interventions)
    total = len(items) if target_step_count is None else max(0, int(target_step_count))
    if len(items) > total:
        total = len(items)
    scope_flags = list(declared_out_of_scope or ())
    if scope_flags and len(scope_flags) != len(items):
        raise MetricError("declared_out_of_scope must match intervention count")
    if not scope_flags:
        scope_flags = [False] * len(items)

    eligible = applied = changed = noop = skipped = out_of_scope = failed = 0
    applied_element_count = changed_element_count = noop_element_count = 0
    reasons: dict[str, int] = {}
    meaningful_observed = False
    meaningful = 0
    unknown_records = 0
    per_input_accumulators: dict[str, dict[str, Any]] = {}

    def field_value(item: Any, name: str, default: Any = None) -> Any:
        return item.get(name, default) if isinstance(item, Mapping) else getattr(item, name, default)

    def indexed_value(values: Any, index: int) -> Any:
        if not isinstance(values, Mapping):
            return None
        if index in values:
            return values[index]
        return values.get(str(index))

    def finite_scalar(value: Any) -> float | None:
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if array.ndim != 0:
            return None
        number = float(array)
        return number if np.isfinite(number) else None

    for index, item in enumerate(items):
        if item is None or (
            isinstance(item, Mapping)
            and not set(item).intersection(
                {
                    "skipped",
                    "applied",
                    "eligible",
                    "requested_indices",
                    "changed_indices",
                    "no_op_indices",
                    "changed",
                }
            )
        ) or (
            not isinstance(item, Mapping) and not hasattr(item, "skipped")
        ):
            unknown_records += 1
            continue
        reason = _intervention_reason(item)
        skipped_item = _intervention_bool(item, "skipped", default=bool(reason))
        explicit_eligible = item.get("eligible") if isinstance(item, Mapping) else getattr(item, "eligible", None)
        explicit_applied = item.get("applied") if isinstance(item, Mapping) else getattr(item, "applied", None)
        is_eligible = (not skipped_item) if explicit_eligible is None else bool(explicit_eligible)
        if scope_flags[index]:
            is_eligible = False
        is_applied = (not skipped_item) if explicit_applied is None else bool(explicit_applied)
        category = _reason_category(reason, declared_out_of_scope=scope_flags[index])
        raw_category = item.get("skip_category") if isinstance(item, Mapping) else getattr(item, "skip_category", None)
        if raw_category is not None and str(raw_category).casefold().replace("-", "_") in {
            "out_of_scope",
            "declared_out_of_scope",
            "inapplicable",
        }:
            category = "out_of_scope"
            is_eligible = False
        if is_eligible:
            eligible += 1
        if is_applied and not skipped_item:
            applied += 1
            requested = field_value(item, "requested_indices", ())
            changed_indices = field_value(item, "changed_indices", ())
            no_op_indices = field_value(item, "no_op_indices", ())
            try:
                applied_element_count += len(requested)
                changed_element_count += len(changed_indices)
                noop_element_count += len(no_op_indices)
            except TypeError:
                pass
            try:
                requested_indices = tuple(int(value) for value in requested)
            except (TypeError, ValueError):
                requested_indices = ()
            delta_values = field_value(item, "delta_values", {})
            delta_abs_values = field_value(item, "delta_abs_values", {})
            tolerance_values = field_value(item, "tolerance", {})
            clipped_indices = field_value(item, "clipped_indices", ())
            try:
                clipped_set = {int(value) for value in clipped_indices}
            except (TypeError, ValueError):
                clipped_set = set()
            # Only applied target indices enter this table. Missing delta or
            # tolerance fields remain unmeasured instead of becoming zero.
            for input_index in requested_indices:
                key = str(input_index)
                accumulator = per_input_accumulators.setdefault(
                    key,
                    {
                        "applied_count": 0,
                        "delta_abs_values": [],
                        "tolerance_values": [],
                        "clip_count": 0,
                    },
                )
                accumulator["applied_count"] += 1
                delta_abs = finite_scalar(indexed_value(delta_abs_values, input_index))
                if delta_abs is None:
                    delta = finite_scalar(indexed_value(delta_values, input_index))
                    if delta is not None:
                        delta_abs = abs(delta)
                if delta_abs is not None:
                    accumulator["delta_abs_values"].append(abs(delta_abs))
                tolerance = finite_scalar(indexed_value(tolerance_values, input_index))
                if tolerance is not None:
                    accumulator["tolerance_values"].append(tolerance)
                if input_index in clipped_set:
                    accumulator["clip_count"] += 1
            changed_item = _intervention_bool(item, "changed", default=False)
            if isinstance(item, Mapping) and "changed" not in item:
                changed_item = bool(item.get("changed_indices", ()))
            if changed_item:
                changed += 1
            else:
                noop += 1
            meaningful_value = item.get("meaningful", None) if isinstance(item, Mapping) else getattr(item, "meaningful", None)
            if meaningful_value is None:
                meaningful_value = item.get("meaningful_changed", None) if isinstance(item, Mapping) else getattr(item, "meaningful_changed", None)
            if meaningful_value is None:
                meaningful_indices = item.get("meaningful_changed_indices", None) if isinstance(item, Mapping) else getattr(item, "meaningful_changed_indices", None)
                tolerance = item.get("tolerance", None) if isinstance(item, Mapping) else getattr(item, "tolerance", None)
                # ``meaningful_changed_count=0`` is emitted by newer cores for
                # every result, including patterns without a declared
                # tolerance.  It becomes an observed diagnostic only when the
                # core also records indices or a non-empty tolerance mapping.
                if meaningful_indices or tolerance:
                    meaningful_value = item.get("meaningful_changed_count", None) if isinstance(item, Mapping) else getattr(item, "meaningful_changed_count", None)
            if meaningful_value is not None:
                meaningful_observed = True
                meaningful += int(bool(meaningful_value))
        else:
            skipped += 1
            if category == "out_of_scope":
                out_of_scope += 1
            else:
                failed += 1
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1

    unobserved = max(0, total - len(items))
    if applied != changed + noop:
        raise MetricError("intervention count invariant violated: applied != changed_exact + noop")
    per_input_delta: dict[str, dict[str, Any]] = {}
    for input_index in sorted(per_input_accumulators, key=lambda value: int(value)):
        accumulator = per_input_accumulators[input_index]
        delta_abs_values = np.asarray(accumulator["delta_abs_values"], dtype=np.float64)
        tolerance_values = accumulator["tolerance_values"]
        unique_tolerances = sorted({float(value) for value in tolerance_values})
        per_input_delta[input_index] = {
            "applied_count": int(accumulator["applied_count"]),
            "delta_abs_count": int(delta_abs_values.size),
            "delta_abs_mean": float(np.mean(delta_abs_values)) if delta_abs_values.size else None,
            "delta_abs_max": float(np.max(delta_abs_values)) if delta_abs_values.size else None,
            "tolerance_observed_count": len(tolerance_values),
            "tolerance_values": unique_tolerances,
            "tolerance": unique_tolerances[0] if len(unique_tolerances) == 1 else None,
            "clip_count": int(accumulator["clip_count"]),
        }

    result: dict[str, Any] = {
        "target_step_count": int(total),
        "observed_step_count": int(len(items)),
        "unobserved_step_count": int(unobserved),
        "eligible_count": int(eligible),
        "applied_count": int(applied),
        "changed_count_exact": int(changed),
        "changed_count": int(changed),
        "applied_element_count": int(applied_element_count),
        "changed_element_count_exact": int(changed_element_count),
        "noop_element_count": int(noop_element_count),
        "meaningful_changed_count": int(meaningful) if meaningful_observed else None,
        "noop_count": int(noop),
        "skipped_count": int(skipped),
        "out_of_scope_count": int(out_of_scope),
        "failed_count": int(failed),
        "unknown_record_count": int(unknown_records),
        "recorded_count": int(len(items) - unknown_records),
        "status": "unavailable" if unknown_records else "available",
        "skip_reasons": reasons,
        "skip_reason_list": sorted(reasons),
        "skipped_reasons": sorted(reasons),
        "skipped_reason_counts": reasons,
        "applied_rate_over_target": float(applied / total) if total else None,
        "eligible_rate_over_target": float(eligible / total) if total else None,
        "changed_rate_over_applied": float(changed / applied) if applied else None,
        "noop_rate_over_applied": float(noop / applied) if applied else None,
        "measurement": "observed_intervention_records",
        "per_input_delta": per_input_delta,
    }
    return result


# Short aliases keep the helper convenient for downstream analysis scripts.
summarize_interventions = summarize_intervention_records
intervention_counts = summarize_intervention_records


def summarize_values(values: Iterable[float | int | None]) -> dict[str, Any]:
    """Public small helper for closed-loop/offline summary code."""

    numeric = np.asarray([float(v) for v in values if v is not None], dtype=np.float64)
    return _finite_summary(numeric)
