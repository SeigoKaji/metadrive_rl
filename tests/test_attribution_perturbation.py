"""Offline perturbation metrics run without MetaDrive or SB3."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from input_attribution.baselines import BaselineProvider
from input_attribution.perturbation import (
    PerturbationTarget,
    build_perturbation_targets,
    centered_logit_l2,
    jensen_shannon_divergence,
    replace_with_baseline,
    run_perturbation,
)
from input_attribution.schema import Feature, ObservationSchema


class _ToyPolicy:
    """A deterministic two-action policy with an easily checked critic."""

    def evaluate(self, observations: np.ndarray) -> object:
        values = np.asarray(observations, dtype=np.float64)
        logits = np.column_stack((values[:, 0], values[:, 1]))
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return SimpleNamespace(
            logits=logits,
            probabilities=probabilities,
            deterministic_actions=np.argmax(logits, axis=1),
            values=values[:, 0] - 2.0 * values[:, 1],
        )


class _RecordingToyPolicy(_ToyPolicy):
    """Records every policy batch to verify perturbation workspace bounds."""

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def evaluate(self, observations: np.ndarray) -> object:
        self.batch_sizes.append(int(np.asarray(observations).shape[0]))
        return super().evaluate(observations)


def test_replacement_changes_only_requested_indices_and_never_mutates_sources() -> None:
    observations = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    baseline = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)
    original = observations.copy()
    original_baseline = baseline.copy()

    replaced = replace_with_baseline(observations, baseline, (1,))

    np.testing.assert_array_equal(replaced[:, 0], original[:, 0])
    np.testing.assert_array_equal(replaced[:, 2], original[:, 2])
    np.testing.assert_array_equal(replaced[:, 1], np.array([20.0, 20.0]))
    np.testing.assert_array_equal(observations, original)
    np.testing.assert_array_equal(baseline, original_baseline)


def test_js_is_symmetric_and_centered_logits_ignore_common_offset() -> None:
    first = np.array([0.8, 0.2])
    second = np.array([0.3, 0.7])

    assert jensen_shannon_divergence(first, first) == 0.0
    assert jensen_shannon_divergence(first, second) == jensen_shannon_divergence(second, first)
    assert centered_logit_l2(np.array([1.0, -1.0]), np.array([11.0, 9.0])) == 0.0


def test_group_targets_and_multiple_baseline_metrics_are_preserved() -> None:
    schema = ObservationSchema(
        schema_version=1,
        name="tiny",
        observation_dim=3,
        features=(
            Feature(0, "first", "state"),
            Feature(1, "second", "state"),
            Feature(2, "third", "state"),
        ),
        groups={"pair": (0, 2)},
    )
    targets = build_perturbation_targets(
        schema, analyze_features=False, analyze_groups=True
    )
    assert targets == (PerturbationTarget("pair", "group", (0, 2)),)

    observations = np.array([[2.0, 0.0, 7.0], [2.0, 0.0, 8.0]], dtype=np.float32)
    # First baseline leaves feature 0 unchanged; second flips the selected
    # action.  Index 2 is also replaced as part of the semantic group.
    baselines = np.array(
        [
            [[2.0, 99.0, -7.0], [-2.0, 99.0, -8.0]],
            [[2.0, 99.0, -7.0], [-2.0, 99.0, -8.0]],
        ],
        dtype=np.float32,
    )
    result = run_perturbation(
        _ToyPolicy(), observations, baselines, targets, batch_size=1
    )

    assert result.js_divergence.shape == (2, 1, 2)
    assert result.action_changed[:, 0, 0].tolist() == [False, False]
    assert result.action_changed[:, 0, 1].tolist() == [True, True]
    np.testing.assert_allclose(result.metric("action_changed", average_baselines=True), 0.5)
    assert np.all(result.selected_action_probability_drop[:, 0, 1] > 0.0)
    # The stored source was never touched, including non-target index 1.
    np.testing.assert_array_equal(observations[:, 1], [0.0, 0.0])


def test_episode_start_baseline_ids_keep_complete_unicode_episode_labels() -> None:
    provider = BaselineProvider(
        np.arange(12, dtype=np.float32).reshape(4, 3),
        episode_ids=np.array([10, 10, 11, 11]),
        steps=np.array([0, 1, 0, 1]),
    )

    resolved = provider.resolve_episode_start()

    assert resolved.baseline_ids is not None
    assert resolved.baseline_ids.dtype.kind == "U"
    assert resolved.baseline_ids[:, 0].tolist() == [
        "episode_start:episode=10:step=0",
        "episode_start:episode=10:step=0",
        "episode_start:episode=11:step=0",
        "episode_start:episode=11:step=0",
    ]


def test_raw_float64_baseline_array_is_normalized_for_perturbation() -> None:
    result = run_perturbation(
        _ToyPolicy(),
        np.array([[1.0, 0.0], [2.0, -1.0]], dtype=np.float32),
        np.array([[-1.0, 0.0]], dtype=np.float64),
        (PerturbationTarget("first", "feature", (0,)),),
    )

    assert result.baseline_count == 1
    assert result.js_divergence.shape == (2, 1, 1)


def test_perturbation_never_passes_more_than_configured_batch_to_adapter() -> None:
    adapter = _RecordingToyPolicy()
    observations = np.column_stack(
        (np.arange(11, dtype=np.float32), np.zeros(11, dtype=np.float32))
    )
    original = observations.copy()
    baseline = np.array([[-5.0, 1.0]], dtype=np.float32)

    result = run_perturbation(
        adapter,
        observations,
        baseline,
        (
            PerturbationTarget("first", "feature", (0,)),
            PerturbationTarget("second", "feature", (1,)),
        ),
        batch_size=3,
    )

    assert result.js_divergence.shape == (11, 2, 1)
    assert adapter.batch_sizes
    assert max(adapter.batch_sizes) <= 3
    np.testing.assert_array_equal(observations, original)
