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

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_argmax": self.original_argmax.tolist(),
            "changed_argmax": self.changed_argmax.tolist(),
            "selected_actions": self.selected_actions.tolist(),
            "selected_probability_original": self.selected_probability_original.tolist(),
            "selected_probability_changed": self.selected_probability_changed.tolist(),
            "selected_probability_delta": self.selected_probability_delta.tolist(),
            "selected_probability_delta_pp": self.selected_probability_delta_pp.tolist(),
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
                "js": _finite_summary(np.asarray([], dtype=np.float64)),
            }
        return {
            "count": int(np.sum(mask)),
            "action_change_count": int(np.sum(changed_actions[mask])),
            "action_change_rate": float(np.mean(changed_actions[mask])),
            "selected_probability_delta": _finite_summary(selected_delta[mask]),
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
    return result


def summarize_values(values: Iterable[float | int | None]) -> dict[str, Any]:
    """Public small helper for closed-loop/offline summary code."""

    numeric = np.asarray([float(v) for v in values if v is not None], dtype=np.float64)
    return _finite_summary(numeric)
