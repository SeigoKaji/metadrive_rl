"""Offline baseline-replacement perturbation analysis for vector policies."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from .baselines import ResolvedBaselines
from .schema import ObservationSchema


class PerturbationError(ValueError):
    """Raised when perturbation inputs or policy outputs are inconsistent."""


TargetKind = Literal["feature", "group", "lidar_sector"]


@dataclass(frozen=True, slots=True)
class PerturbationTarget:
    """A feature, semantic group, or LiDAR sector to replace together."""

    name: str
    kind: TargetKind | str
    indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise PerturbationError("target nameは空にできません")
        if self.kind not in {"feature", "group", "lidar_sector"}:
            raise PerturbationError(f"未対応のtarget kindです: {self.kind}")
        if not self.indices:
            raise PerturbationError(f"target {self.name}には少なくとも1つのindexが必要です")
        normalized: list[int] = []
        for index in self.indices:
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise PerturbationError(f"target {self.name}のindexは非負整数で指定してください")
            if index not in normalized:
                normalized.append(index)
        object.__setattr__(self, "indices", tuple(sorted(normalized)))

    @property
    def target_id(self) -> str:
        """Stable identifier which remains unique across target kinds."""

        return f"{self.kind}:{self.name}"


@runtime_checkable
class _PolicyEvaluator(Protocol):
    def evaluate(self, observations: np.ndarray) -> object:
        """Return batch logits/probabilities/actions/values."""


@dataclass(frozen=True, slots=True)
class PerturbationResult:
    """Per-step, per-target, per-baseline offline perturbation measurements.

    Every metric has shape ``(sample_count, target_count, baseline_count)``.
    ``value_delta`` is defined as ``V(perturbed) - V(original)``.  Centered
    logit L2 is ``|| (l' - mean(l')) - (l - mean(l)) ||_2`` and therefore is
    invariant to a common additive logit shift.
    """

    targets: tuple[PerturbationTarget, ...]
    original_logits: np.ndarray
    original_probabilities: np.ndarray
    original_actions: np.ndarray
    original_values: np.ndarray
    baseline_ids: np.ndarray
    js_divergence: np.ndarray
    action_changed: np.ndarray
    selected_action_probability_drop: np.ndarray
    absolute_selected_action_probability_drop: np.ndarray
    centered_logit_l2: np.ndarray
    value_delta: np.ndarray
    absolute_value_delta: np.ndarray
    squared_value_delta: np.ndarray
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        target_count = len(self.targets)
        if target_count == 0:
            raise PerturbationError("少なくとも1つのperturbation targetが必要です")
        if len({target.target_id for target in self.targets}) != target_count:
            raise PerturbationError("同じkind/nameのperturbation targetを重複指定できません")
        probabilities = np.asarray(self.original_probabilities)
        logits = np.asarray(self.original_logits)
        actions = np.asarray(self.original_actions)
        values = np.asarray(self.original_values)
        if probabilities.ndim != 2 or probabilities.shape[0] == 0 or probabilities.shape[1] == 0:
            raise PerturbationError("original_probabilitiesは非空の(N, A)配列で指定してください")
        sample_count, action_count = probabilities.shape
        if logits.shape != probabilities.shape:
            raise PerturbationError("original_logitsとoriginal_probabilitiesのshapeが一致しません")
        if actions.shape not in {(sample_count,), (sample_count, 1)}:
            raise PerturbationError("original_actionsは(N,)配列で指定してください")
        if values.shape not in {(sample_count,), (sample_count, 1)}:
            raise PerturbationError("original_valuesは(N,)配列で指定してください")
        expected_metric_shape: tuple[int, int, int] | None = None
        ids = np.asarray(self.baseline_ids)
        if ids.ndim != 2 or ids.shape[0] != sample_count or ids.shape[1] == 0:
            raise PerturbationError("baseline_idsは(N, B)配列で指定してください")
        expected_metric_shape = (sample_count, target_count, ids.shape[1])
        metric_names = (
            "js_divergence",
            "action_changed",
            "selected_action_probability_drop",
            "absolute_selected_action_probability_drop",
            "centered_logit_l2",
            "value_delta",
            "absolute_value_delta",
            "squared_value_delta",
        )
        for name in metric_names:
            value = np.asarray(getattr(self, name))
            if value.shape != expected_metric_shape:
                raise PerturbationError(
                    f"{name}は(N, target_count, baseline_count)形状で指定してください: "
                    f"expected {expected_metric_shape}, got {value.shape}"
                )
        if not np.all(np.isfinite(probabilities)) or not np.all(np.isfinite(logits)) or not np.all(np.isfinite(values)):
            raise PerturbationError("original policy outputsは有限値で指定してください")
        normalized_arrays: dict[str, np.ndarray] = {
            "original_logits": np.array(logits, dtype=np.float64, copy=True),
            "original_probabilities": np.array(probabilities, dtype=np.float64, copy=True),
            "original_actions": np.array(actions, dtype=np.int64, copy=True).reshape(sample_count),
            "original_values": np.array(values, dtype=np.float64, copy=True).reshape(sample_count),
            "baseline_ids": np.array(ids, dtype=str, copy=True),
            "js_divergence": np.array(self.js_divergence, dtype=np.float64, copy=True),
            "action_changed": np.array(self.action_changed, dtype=bool, copy=True),
            "selected_action_probability_drop": np.array(
                self.selected_action_probability_drop, dtype=np.float64, copy=True
            ),
            "absolute_selected_action_probability_drop": np.array(
                self.absolute_selected_action_probability_drop, dtype=np.float64, copy=True
            ),
            "centered_logit_l2": np.array(self.centered_logit_l2, dtype=np.float64, copy=True),
            "value_delta": np.array(self.value_delta, dtype=np.float64, copy=True),
            "absolute_value_delta": np.array(self.absolute_value_delta, dtype=np.float64, copy=True),
            "squared_value_delta": np.array(self.squared_value_delta, dtype=np.float64, copy=True),
        }
        for name, value in normalized_arrays.items():
            if name != "action_changed" and name != "baseline_ids" and not np.all(np.isfinite(value)):
                raise PerturbationError(f"{name}は有限値で指定してください")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def sample_count(self) -> int:
        """Number of unperturbed rollout observations."""

        return int(self.original_probabilities.shape[0])

    @property
    def target_count(self) -> int:
        """Number of features/groups/sectors analysed."""

        return len(self.targets)

    @property
    def baseline_count(self) -> int:
        """Number of baseline replacements preserved per sample/target."""

        return int(self.baseline_ids.shape[1])

    @property
    def metric_names(self) -> tuple[str, ...]:
        """Names whose arrays have the common ``(N, T, B)`` layout."""

        return (
            "js_divergence",
            "action_changed",
            "selected_action_probability_drop",
            "absolute_selected_action_probability_drop",
            "centered_logit_l2",
            "value_delta",
            "absolute_value_delta",
            "squared_value_delta",
        )

    def metric(self, name: str, *, average_baselines: bool = False) -> np.ndarray:
        """Return one metric, optionally averaged over the baseline axis."""

        if name not in self.metric_names:
            raise PerturbationError(f"未定義のperturbation metricです: {name}")
        values = np.asarray(getattr(self, name))
        return values.mean(axis=2) if average_baselines else values

    def baseline_mean_metrics(self) -> dict[str, np.ndarray]:
        """Return each metric averaged over baselines with shape ``(N, T)``."""

        return {name: self.metric(name, average_baselines=True) for name in self.metric_names}

    @property
    def mean_js_divergence(self) -> np.ndarray:
        """Jensen-Shannon divergence averaged across baselines, ``(N, T)``."""

        return self.js_divergence.mean(axis=2)


def _as_finite_observations(observations: np.ndarray, *, name: str = "observations") -> np.ndarray:
    result = np.asarray(observations)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] == 0:
        raise PerturbationError(f"{name}は非空の(N, D)配列で指定してください")
    if not np.issubdtype(result.dtype, np.number) or not np.all(np.isfinite(result)):
        raise PerturbationError(f"{name}は有限の数値配列で指定してください")
    return np.array(result, dtype=np.float32, copy=True)


def _normalise_baselines(
    baselines: ResolvedBaselines | np.ndarray,
    *,
    sample_count: int,
    observation_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(baselines, ResolvedBaselines):
        values = baselines.values
        ids = cast_baseline_ids(baselines.baseline_ids)
        if values.shape != (sample_count, values.shape[1], observation_dim):
            raise PerturbationError(
                "ResolvedBaselinesのsample_count/observation_dimがobservationsと一致しません: "
                f"baselines={values.shape}, observations={(sample_count, observation_dim)}"
            )
        return np.array(values, dtype=np.float32, copy=True), np.array(ids, dtype=str, copy=True)
    raw = np.asarray(baselines)
    if not np.issubdtype(raw.dtype, np.number) or not np.all(np.isfinite(raw)):
        raise PerturbationError("baselinesは有限の数値配列で指定してください")
    if raw.ndim == 1:
        if raw.shape[0] != observation_dim:
            raise PerturbationError("baseline observation_dimが一致しません")
        values = np.broadcast_to(raw[np.newaxis, np.newaxis, :], (sample_count, 1, observation_dim)).copy()
    elif raw.ndim == 2:
        if raw.shape[1] != observation_dim:
            raise PerturbationError("baseline observation_dimが一致しません")
        if raw.shape[0] == sample_count:
            # (N, D) denotes one baseline tailored to each sample.  A global
            # set of N baselines can always be made unambiguous as (1, N, D)
            # or a ResolvedBaselines instance.
            values = raw[:, np.newaxis, :].copy()
        else:
            values = np.broadcast_to(raw[np.newaxis, :, :], (sample_count, raw.shape[0], observation_dim)).copy()
    elif raw.ndim == 3:
        if raw.shape[0] != sample_count or raw.shape[2] != observation_dim or raw.shape[1] == 0:
            raise PerturbationError(
                "baseline配列は(N, B, D)でobservationsと同じN/Dを持つ必要があります"
            )
        values = raw.copy()
    else:
        raise PerturbationError("baseline配列は(D,), (B, D), (N, D), または(N, B, D)で指定してください")
    ids = np.asarray(
        [[f"baseline_{baseline_index}" for baseline_index in range(values.shape[1])] for _ in range(sample_count)],
        dtype=str,
    )
    # NumPy 2 correctly refuses ``np.array(float64, dtype=float32,
    # copy=False)`` because conversion necessarily allocates.  Baselines are
    # normalized at this boundary, so a cast copy is both expected and safe.
    return np.asarray(values, dtype=np.float32), ids


def cast_baseline_ids(value: np.ndarray | None) -> np.ndarray:
    """Narrow optional IDs after :class:`ResolvedBaselines` validation."""

    if value is None:  # defensive; __post_init__ always fills this.
        raise PerturbationError("ResolvedBaselinesにbaseline_idsがありません")
    return np.asarray(value)


def replace_with_baseline(
    observations: np.ndarray,
    baseline: np.ndarray,
    indices: Sequence[int],
) -> np.ndarray:
    """Copy observations and replace exactly ``indices`` from a baseline.

    The returned array is independent: neither ``observations`` nor
    ``baseline`` is ever modified.  Inputs may be ``(D,)`` or matching
    ``(N, D)`` arrays; a one-row baseline broadcasts over an observation batch.
    """

    source = np.asarray(observations)
    reference = np.asarray(baseline)
    was_vector = source.ndim == 1
    if was_vector:
        source = source[np.newaxis, :]
    if source.ndim != 2 or source.shape[0] == 0 or source.shape[1] == 0:
        raise PerturbationError("observationsは(D,)または非空の(N, D)配列で指定してください")
    if reference.ndim == 1:
        reference = reference[np.newaxis, :]
    if reference.ndim != 2 or reference.shape[1] != source.shape[1] or reference.shape[0] not in {1, source.shape[0]}:
        raise PerturbationError("baselineは(D,), (1, D), またはobservationsと同じ(N, D)形状で指定してください")
    normalized_indices: list[int] = []
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise PerturbationError("replacement indexは整数で指定してください")
        integer = int(index)
        if integer < 0 or integer >= source.shape[1]:
            raise PerturbationError(
                f"replacement indexが範囲外です: {integer} (D={source.shape[1]})"
            )
        if integer not in normalized_indices:
            normalized_indices.append(integer)
    if not normalized_indices:
        raise PerturbationError("少なくとも1つのreplacement indexを指定してください")
    result = np.array(source, copy=True)
    result[:, normalized_indices] = reference[:, normalized_indices]
    return result[0] if was_vector else result


# Descriptive compatibility alias.
replace_indices_with_baseline = replace_with_baseline


def jensen_shannon_divergence(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray | float:
    """Compute natural-log Jensen-Shannon divergence without SciPy.

    Inputs may be one probability vector or a batch.  They are normalized
    defensively, so tiny floating-point drift in an otherwise valid policy
    distribution does not change the metric.  The result is symmetric and is
    exactly zero (up to normal floating arithmetic) for identical inputs.
    """

    p = np.asarray(first, dtype=np.float64)
    q = np.asarray(second, dtype=np.float64)
    if p.shape != q.shape or p.ndim < 1 or p.shape[-1] == 0:
        raise PerturbationError("JS divergenceには同じshapeの確率ベクトルを指定してください")
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)) or np.any(p < 0.0) or np.any(q < 0.0):
        raise PerturbationError("JS divergenceの確率は有限かつ非負で指定してください")
    p_total = p.sum(axis=-1, keepdims=True)
    q_total = q.sum(axis=-1, keepdims=True)
    if np.any(p_total <= 0.0) or np.any(q_total <= 0.0):
        raise PerturbationError("JS divergenceの確率和は正である必要があります")
    p = p / p_total
    q = q / q_total
    midpoint = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_p = np.where(p > 0.0, p * (np.log(p) - np.log(midpoint)), 0.0).sum(axis=-1)
        kl_q = np.where(q > 0.0, q * (np.log(q) - np.log(midpoint)), 0.0).sum(axis=-1)
    result = np.maximum(0.0, 0.5 * (kl_p + kl_q))
    return float(result) if result.ndim == 0 else result


def centered_logit_l2(first: np.ndarray, second: np.ndarray) -> np.ndarray | float:
    """Return L2 after centering each score vector along its action axis."""

    original = np.asarray(first, dtype=np.float64)
    perturbed = np.asarray(second, dtype=np.float64)
    if original.shape != perturbed.shape or original.ndim < 1 or original.shape[-1] == 0:
        raise PerturbationError("centered logit L2には同じshapeのlogit配列を指定してください")
    if not np.all(np.isfinite(original)) or not np.all(np.isfinite(perturbed)):
        raise PerturbationError("logitsは有限値で指定してください")
    difference = (
        perturbed - perturbed.mean(axis=-1, keepdims=True)
        - (original - original.mean(axis=-1, keepdims=True))
    )
    result = np.sqrt(np.sum(difference * difference, axis=-1))
    return float(result) if result.ndim == 0 else result


def build_perturbation_targets(
    schema: ObservationSchema,
    *,
    analyze_features: bool = True,
    analyze_groups: bool = True,
    lidar_sector_degrees: float | None = None,
) -> tuple[PerturbationTarget, ...]:
    """Build deterministic targets from external feature/group definitions."""

    targets: list[PerturbationTarget] = []
    if analyze_features:
        targets.extend(
            PerturbationTarget(name=feature.name, kind="feature", indices=(feature.index,))
            for feature in schema.features
        )
    if analyze_groups:
        targets.extend(
            PerturbationTarget(name=name, kind="group", indices=tuple(indices))
            for name, indices in schema.groups.items()
        )
    if lidar_sector_degrees is not None:
        targets.extend(
            PerturbationTarget(name=name, kind="lidar_sector", indices=tuple(indices))
            for name, indices in schema.lidar_sector_groups(lidar_sector_degrees).items()
        )
    if not targets:
        raise PerturbationError("feature/group/LiDAR sectorの少なくとも1つを解析対象にしてください")
    return tuple(targets)


def _extract_output(output: object, *, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def attribute(*names: str) -> object:
        for name in names:
            if hasattr(output, name):
                return getattr(output, name)
        raise PerturbationError(f"policy evaluationに必要な出力がありません: {'/'.join(names)}")

    logits = np.asarray(attribute("logits"), dtype=np.float64)
    probabilities = np.asarray(attribute("probabilities"), dtype=np.float64)
    actions = np.asarray(attribute("deterministic_actions", "actions"), dtype=np.int64)
    values = np.asarray(attribute("values"), dtype=np.float64)
    if logits.ndim != 2 or logits.shape[0] != batch_size or logits.shape[1] == 0:
        raise PerturbationError("policy logitsは(B, A)形状で返す必要があります")
    if probabilities.shape != logits.shape:
        raise PerturbationError("policy probabilitiesとlogitsのshapeが一致しません")
    if actions.shape not in {(batch_size,), (batch_size, 1)}:
        raise PerturbationError("policy actionは(B,)形状で返す必要があります")
    if values.shape not in {(batch_size,), (batch_size, 1)}:
        raise PerturbationError("policy valueは(B,)形状で返す必要があります")
    if not np.all(np.isfinite(logits)) or not np.all(np.isfinite(probabilities)) or not np.all(np.isfinite(values)):
        raise PerturbationError("policy outputに非有限値があります")
    if np.any(probabilities < -1e-12) or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6):
        raise PerturbationError("policy probabilitiesは各rowで非負かつ和1である必要があります")
    normalized_actions = actions.reshape(batch_size)
    if np.any(normalized_actions < 0) or np.any(normalized_actions >= logits.shape[1]):
        raise PerturbationError("policy actionがaction range外です")
    return logits, probabilities, normalized_actions, values.reshape(batch_size)


def _evaluate_in_batches(
    adapter: _PolicyEvaluator,
    observations: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise PerturbationError("batch_sizeは1以上の整数で指定してください")
    logits_chunks: list[np.ndarray] = []
    probability_chunks: list[np.ndarray] = []
    action_chunks: list[np.ndarray] = []
    value_chunks: list[np.ndarray] = []
    action_count: int | None = None
    for start in range(0, observations.shape[0], batch_size):
        current = observations[start : start + batch_size]
        logits, probabilities, actions, values = _extract_output(
            adapter.evaluate(current), batch_size=current.shape[0]
        )
        if action_count is None:
            action_count = logits.shape[1]
        elif logits.shape[1] != action_count:
            raise PerturbationError("policy action数がbatch間で変化しました")
        logits_chunks.append(logits)
        probability_chunks.append(probabilities)
        action_chunks.append(actions)
        value_chunks.append(values)
    return (
        np.concatenate(logits_chunks, axis=0),
        np.concatenate(probability_chunks, axis=0),
        np.concatenate(action_chunks, axis=0),
        np.concatenate(value_chunks, axis=0),
    )


def run_perturbation(
    adapter: _PolicyEvaluator,
    observations: np.ndarray,
    baselines: ResolvedBaselines | np.ndarray,
    targets: Sequence[PerturbationTarget],
    *,
    batch_size: int = 1024,
) -> PerturbationResult:
    """Evaluate baseline replacements for every sample/target/baseline.

    The source observation and every baseline are copied before replacement;
    only target indices change.  The temporary perturbed array is limited to
    one ``batch_size`` sample slice, rather than allocating ``(N, D)`` once
    per target/baseline.  Outputs retain the individual baseline axis rather
    than hiding it in an early average.
    """

    if not isinstance(adapter, _PolicyEvaluator):
        raise PerturbationError("adapterはevaluate(observations)を実装する必要があります")
    source = _as_finite_observations(observations)
    selected_targets = tuple(targets)
    if not selected_targets:
        raise PerturbationError("少なくとも1つのperturbation targetを指定してください")
    if len({target.target_id for target in selected_targets}) != len(selected_targets):
        raise PerturbationError("同じkind/nameのtargetを重複指定できません")
    for target in selected_targets:
        invalid = [index for index in target.indices if index >= source.shape[1]]
        if invalid:
            raise PerturbationError(
                f"target {target.target_id}に観測次元外indexがあります: {invalid}"
            )
    baseline_values, baseline_ids = _normalise_baselines(
        baselines,
        sample_count=source.shape[0],
        observation_dim=source.shape[1],
    )
    original_logits, original_probabilities, original_actions, original_values = _evaluate_in_batches(
        adapter, source, batch_size=batch_size
    )
    sample_count = source.shape[0]
    target_count = len(selected_targets)
    baseline_count = baseline_values.shape[1]
    metric_shape = (sample_count, target_count, baseline_count)
    js = np.empty(metric_shape, dtype=np.float64)
    changed = np.empty(metric_shape, dtype=bool)
    selected_drop = np.empty(metric_shape, dtype=np.float64)
    centered_l2 = np.empty(metric_shape, dtype=np.float64)
    value_delta = np.empty(metric_shape, dtype=np.float64)
    for target_index, target in enumerate(selected_targets):
        for baseline_index in range(baseline_count):
            for start in range(0, sample_count, batch_size):
                end = min(start + batch_size, sample_count)
                source_slice = source[start:end]
                perturbed_slice = replace_with_baseline(
                    source_slice,
                    baseline_values[start:end, baseline_index, :],
                    target.indices,
                )
                p_logits, p_probabilities, p_actions, p_values = _extract_output(
                    adapter.evaluate(perturbed_slice), batch_size=end - start
                )
                original_logits_slice = original_logits[start:end]
                original_probabilities_slice = original_probabilities[start:end]
                original_actions_slice = original_actions[start:end]
                original_values_slice = original_values[start:end]
                if p_logits.shape[1] != original_logits_slice.shape[1]:
                    raise PerturbationError(
                        "policy action数がoriginal/perturbed batch間で変化しました"
                    )
                js[start:end, target_index, baseline_index] = np.asarray(
                    jensen_shannon_divergence(
                        original_probabilities_slice, p_probabilities
                    ),
                    dtype=np.float64,
                )
                changed[start:end, target_index, baseline_index] = (
                    original_actions_slice != p_actions
                )
                local_indices = np.arange(end - start)
                selected_drop[start:end, target_index, baseline_index] = (
                    original_probabilities_slice[
                        local_indices, original_actions_slice
                    ]
                    - p_probabilities[local_indices, original_actions_slice]
                )
                centered_l2[start:end, target_index, baseline_index] = np.asarray(
                    centered_logit_l2(original_logits_slice, p_logits), dtype=np.float64
                )
                value_delta[start:end, target_index, baseline_index] = (
                    p_values - original_values_slice
                )
    return PerturbationResult(
        targets=selected_targets,
        original_logits=original_logits,
        original_probabilities=original_probabilities,
        original_actions=original_actions,
        original_values=original_values,
        baseline_ids=baseline_ids,
        js_divergence=js,
        action_changed=changed,
        selected_action_probability_drop=selected_drop,
        absolute_selected_action_probability_drop=np.abs(selected_drop),
        centered_logit_l2=centered_l2,
        value_delta=value_delta,
        absolute_value_delta=np.abs(value_delta),
        squared_value_delta=np.square(value_delta),
        metadata={
            "metric_definitions": {
                "value_delta": "V(perturbed) - V(original)",
                "centered_logit_l2": "L2 of logits after each action vector is mean-centered",
                "selected_action_probability_drop": "p_original(a*) - p_perturbed(a*)",
                "js_divergence": "natural-log Jensen-Shannon divergence",
            },
            "batch_size": batch_size,
        },
    )


# Verb aliases make older notebooks and an intuitive plural spelling work
# without maintaining multiple implementations.
analyze_perturbations = run_perturbation
analyze_perturbation = run_perturbation
