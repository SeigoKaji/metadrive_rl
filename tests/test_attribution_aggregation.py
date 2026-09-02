"""DataFrame aggregation tests for temporal perturbation and IG summaries."""

from __future__ import annotations

import numpy as np
import pytest

from input_attribution.aggregation import (
    AggregationError,
    summarize_ig_groups,
    summarize_integrated_gradients,
    summarize_perturbation,
)
from input_attribution.config import PhaseConfig
from input_attribution.integrated_gradients import IntegratedGradientsResult
from input_attribution.perturbation import PerturbationResult, PerturbationTarget
from input_attribution.schema import Feature, ObservationSchema


def _schema() -> ObservationSchema:
    return ObservationSchema(
        schema_version=1,
        name="aggregation_toy",
        observation_dim=2,
        features=(Feature(0, "a", "state"), Feature(1, "b", "state")),
        groups={"both": (0, 1)},
    )


def _perturbation_result() -> PerturbationResult:
    shape = (4, 2, 2)
    values = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) / 10.0
    actions = np.array(
        [
            [[False, True], [False, False]],
            [[True, True], [False, False]],
            [[False, False], [True, False]],
            [[True, False], [False, True]],
        ]
    )
    return PerturbationResult(
        targets=(
            PerturbationTarget("a", "feature", (0,)),
            PerturbationTarget("both", "group", (0, 1)),
        ),
        original_logits=np.zeros((4, 2)),
        original_probabilities=np.full((4, 2), 0.5),
        original_actions=np.zeros(4, dtype=np.int64),
        original_values=np.zeros(4),
        baseline_ids=np.array([["b0", "b1"]] * 4),
        js_divergence=values,
        action_changed=actions,
        selected_action_probability_drop=values,
        absolute_selected_action_probability_drop=np.abs(values),
        centered_logit_l2=values,
        value_delta=values - 0.5,
        absolute_value_delta=np.abs(values - 0.5),
        squared_value_delta=(values - 0.5) ** 2,
    )


def _ig_result() -> IntegratedGradientsResult:
    attributions = np.array([[[[1.0, -2.0]]], [[[3.0, 4.0]]]])
    scalar_shape = (2, 1, 1)
    return IntegratedGradientsResult(
        targets=("critic_value",),
        attributions=attributions,
        baseline_ids=np.array([["b0"], ["b0"]]),
        target_actions=np.array([0, 0]),
        runner_up_actions=np.array([1, 1]),
        input_outputs=np.ones(scalar_shape),
        baseline_outputs=np.zeros(scalar_shape),
        attribution_sums=np.ones(scalar_shape),
        completeness_residuals=np.zeros(scalar_shape),
        absolute_completeness_residuals=np.zeros(scalar_shape),
        relative_completeness_errors=np.zeros(scalar_shape),
    )


def _lidar_schema() -> ObservationSchema:
    return ObservationSchema(
        schema_version=1,
        name="lidar_240",
        observation_dim=240,
        features=tuple(
            Feature(
                index=index,
                name=f"ray_{index:03d}",
                block="lidar",
                kind="lidar",
                angle_deg=index * 1.5,
            )
            for index in range(240)
        ),
        groups={"lidar_all": tuple(range(240))},
    )


def _lidar_ig_result() -> IntegratedGradientsResult:
    attributions = np.ones((1, 1, 1, 240), dtype=np.float64)
    scalar_shape = (1, 1, 1)
    return IntegratedGradientsResult(
        targets=("critic_value",),
        attributions=attributions,
        baseline_ids=np.array([["b0"]]),
        target_actions=np.array([0]),
        runner_up_actions=np.array([1]),
        input_outputs=np.ones(scalar_shape),
        baseline_outputs=np.zeros(scalar_shape),
        attribution_sums=np.ones(scalar_shape),
        completeness_residuals=np.zeros(scalar_shape),
        absolute_completeness_residuals=np.zeros(scalar_shape),
        relative_completeness_errors=np.zeros(scalar_shape),
    )


def test_perturbation_summary_includes_full_episode_progress_and_phase_statistics() -> None:
    summary = summarize_perturbation(
        _perturbation_result(),
        episode_ids=np.array([0, 0, 1, 1]),
        steps=np.array([0, 1, 0, 1]),
        progress_bins=2,
        phases=(PhaseConfig("second_step", start_step=1, end_step=2),),
    )

    assert {"full_episode", "episode", "progress_bin", "phase"}.issubset(
        set(summary["scope"])
    )
    full_feature = summary.loc[
        (summary["scope"] == "full_episode")
        & (summary["target_name"] == "a")
        & (summary["baseline_scope"] == "mean_over_baselines")
    ].iloc[0]
    assert full_feature["count"] == 4
    assert full_feature["action_flip_rate"] == 0.5
    assert "p95_js_divergence" in summary.columns
    assert "mean_absolute_value_delta" in summary.columns


def test_normalized_progress_phase_ending_at_one_includes_final_decision() -> None:
    summary = summarize_perturbation(
        _perturbation_result(),
        episode_ids=np.array([0, 0, 1, 1]),
        steps=np.array([0, 1, 0, 1]),
        progress_bins=2,
        phases=(PhaseConfig("whole", start_progress=0.0, end_progress=1.0),),
    )

    whole_feature = summary.loc[
        (summary["scope"] == "phase")
        & (summary["phase"] == "whole")
        & (summary["target_name"] == "a")
    ].iloc[0]
    assert whole_feature["count"] == 4


def test_ig_feature_and_group_signed_sum_are_distinct_from_absolute_mass() -> None:
    result = _ig_result()
    summaries = summarize_integrated_gradients(
        result,
        _schema(),
        episode_ids=np.array([0, 0]),
        steps=np.array([0, 1]),
        progress_bins=2,
    )
    features = summaries.feature_summary
    groups = summaries.group_summary
    feature_a = features.loc[
        (features["scope"] == "full_episode") & (features["feature_name"] == "a")
    ].iloc[0]
    both = groups.loc[
        (groups["scope"] == "full_episode") & (groups["group_name"] == "both")
    ].iloc[0]

    assert feature_a["mean_signed_ig"] == 2.0
    assert feature_a["mean_absolute_ig"] == 2.0
    # Per-sample group signed sums are -1 and +7, whereas absolute masses are
    # 3 and 7.  They must remain different tables/columns.
    assert both["mean_signed_ig"] == 3.0
    assert both["mean_absolute_mass"] == 5.0
    assert both["positive_rate"] == 0.5
    assert both["negative_rate"] == 0.5
    assert summaries.completeness_summary["mean_completeness_residual"].eq(0.0).all()
    assert summarize_ig_groups(result, _schema()).shape[0] > 0


def test_lidar_sector_groups_cover_all_rays_and_preserve_absolute_mass() -> None:
    summary = summarize_ig_groups(
        _lidar_ig_result(),
        _lidar_schema(),
        lidar_sector_degrees=15,
    )
    sectors = summary.loc[
        (summary["scope"] == "full_episode")
        & (summary["group_kind"] == "lidar_sector")
        & (summary["baseline_scope"] == "mean_over_baselines")
    ]

    assert len(sectors) == 24
    sector_indices = [tuple(indices) for indices in sectors["group_indices"]]
    assert set().union(*map(set, sector_indices)) == set(range(240))
    assert sum(len(indices) for indices in sector_indices) == 240
    np.testing.assert_allclose(sectors["group_signed_ig"], 10.0)
    np.testing.assert_allclose(sectors["group_absolute_mass"], 10.0)
    assert sectors["group_absolute_mass"].sum() == 240.0
    assert sectors["mean_normalized_absolute_mass"].sum() == pytest.approx(1.0)
    assert set(sectors["normalization_basis"]) == {
        "per-sample total LiDAR absolute IG mass"
    }


def test_lidar_sector_name_collision_with_schema_group_is_rejected() -> None:
    schema = _lidar_schema()
    conflicting = ObservationSchema(
        schema_version=schema.schema_version,
        name="conflicting_lidar",
        observation_dim=schema.observation_dim,
        features=schema.features,
        groups={"lidar_sector_000_015_deg": (0,)},
    )

    with pytest.raises(AggregationError, match="名前が衝突"):
        summarize_ig_groups(
            _lidar_ig_result(), conflicting, lidar_sector_degrees=15
        )
