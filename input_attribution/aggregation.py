"""Pandas summaries for offline perturbation and Integrated Gradients results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .integrated_gradients import IntegratedGradientsResult
from .perturbation import PerturbationResult
from .schema import ObservationSchema, ObservationSchemaError


class AggregationError(ValueError):
    """Raised when attribution result data cannot be summarized."""


@dataclass(frozen=True, slots=True)
class IntegratedGradientsSummaries:
    """Separate tables so signed IG and absolute mass cannot be conflated."""

    feature_summary: pd.DataFrame
    group_summary: pd.DataFrame
    completeness_summary: pd.DataFrame


def _statistics(values: np.ndarray) -> dict[str, float | int]:
    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    if flattened.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(flattened.size),
        "mean": float(np.mean(flattened)),
        "median": float(np.median(flattened)),
        "std": float(np.std(flattened, ddof=0)),
        "p90": float(np.percentile(flattened, 90)),
        "p95": float(np.percentile(flattened, 95)),
        "max": float(np.max(flattened)),
    }


def _add_statistics(row: dict[str, object], prefix: str, values: np.ndarray, *, count_key: str | None = None) -> None:
    stats = _statistics(values)
    if count_key is not None:
        row[count_key] = stats.pop("count")
    for name, value in stats.items():
        row[f"{name}_{prefix}"] = value


def _summary_row_base(*, baseline_scope: str, baseline_index: int | None) -> dict[str, object]:
    return {
        # Stable identifiers for the summary over all rollout samples.
        "scope": "full_episode",
        "slice_name": "all",
        "baseline_scope": baseline_scope,
        "baseline_index": baseline_index,
    }


def summarize_perturbation(
    result: PerturbationResult,
    *,
    include_baseline_rows: bool = False,
) -> pd.DataFrame:
    """Summarize each perturbation target across all rollout samples.

    A ``mean_over_baselines`` row averages baseline measurements per sample
    before summary statistics. Optional baseline rows retain each individual
    reference's distribution for users who need it.
    """

    if not isinstance(result, PerturbationResult):
        raise AggregationError("resultはPerturbationResultで指定してください")
    metric_data: dict[str, np.ndarray] = {
        name: result.metric(name, average_baselines=False) for name in result.metric_names
    }
    rows: list[dict[str, object]] = []
    baseline_modes: list[tuple[str, int | None]] = [("mean_over_baselines", None)]
    if include_baseline_rows:
        baseline_modes.extend(("individual_baseline", index) for index in range(result.baseline_count))
    for baseline_scope, baseline_index in baseline_modes:
        for target_index, target in enumerate(result.targets):
            row = _summary_row_base(
                baseline_scope=baseline_scope,
                baseline_index=baseline_index,
            )
            row.update(
                {
                    "target_id": target.target_id,
                    "target_name": target.name,
                    "target_kind": target.kind,
                    "target_size": len(target.indices),
                }
            )
            for metric_name, values in metric_data.items():
                if baseline_index is None:
                    selected = values[:, target_index, :].mean(axis=1)
                else:
                    selected = values[:, target_index, baseline_index]
                _add_statistics(
                    row,
                    metric_name,
                    selected,
                    count_key="count" if metric_name == "js_divergence" else None,
                )
            if baseline_index is None:
                flips = result.action_changed[:, target_index, :].mean(axis=1)
            else:
                flips = result.action_changed[:, target_index, baseline_index]
            row["action_flip_rate"] = float(np.mean(flips)) if flips.size else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def perturbation_feature_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Filter a perturbation summary to individual features only."""

    return summary.loc[summary["target_kind"] == "feature"].reset_index(drop=True)


def perturbation_group_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Filter a perturbation summary to semantic groups and LiDAR sectors."""

    return summary.loc[summary["target_kind"].isin(["group", "lidar_sector"])].reset_index(drop=True)


def _ig_baseline_values(
    result: IntegratedGradientsResult,
    *,
    baseline_index: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return signed and absolute-per-baseline-average arrays as (N,T,D)."""

    if baseline_index is None:
        return result.attributions.mean(axis=2), np.abs(result.attributions).mean(axis=2)
    return result.attributions[:, :, baseline_index, :], np.abs(
        result.attributions[:, :, baseline_index, :]
    )


def _ig_group_entities(
    schema: ObservationSchema,
    *,
    lidar_sector_degrees: float | None,
) -> list[tuple[str, str, tuple[int, ...]]]:
    """Return schema groups plus optional disjoint LiDAR angular sectors.

    Sector labels are generated by :meth:`ObservationSchema.lidar_sector_groups`
    and intentionally remain a separate ``group_kind``.  A name collision is
    rejected rather than silently extending a user-authored semantic group.
    """

    entities = [
        (name, "group", tuple(indices)) for name, indices in schema.groups.items()
    ]
    if lidar_sector_degrees is None:
        return entities
    try:
        sectors = schema.lidar_sector_groups(lidar_sector_degrees)
    except ObservationSchemaError as error:
        raise AggregationError(
            f"LiDAR sector groupを作れません: {error}"
        ) from error
    collisions = sorted(set(schema.groups).intersection(sectors))
    if collisions:
        raise AggregationError(
            "schema groupと生成LiDAR sector groupの名前が衝突しています: "
            + ", ".join(collisions)
        )
    entities.extend(
        (name, "lidar_sector", tuple(indices)) for name, indices in sectors.items()
    )
    return entities


def _ig_summary_rows(
    result: IntegratedGradientsResult,
    schema: ObservationSchema,
    *,
    include_baseline_rows: bool,
    groups: bool,
    lidar_sector_degrees: float | None = None,
) -> list[dict[str, object]]:
    if result.observation_dim != schema.observation_dim:
        raise AggregationError(
            "IG observation_dimとschemaが一致しません: "
            f"ig={result.observation_dim}, schema={schema.observation_dim}"
        )
    baseline_modes: list[tuple[str, int | None]] = [("mean_over_baselines", None)]
    if include_baseline_rows:
        baseline_modes.extend(("individual_baseline", index) for index in range(result.baseline_count))
    rows: list[dict[str, object]] = []
    entities: list[tuple[str, str, tuple[int, ...]]] = (
        [(feature.name, "feature", (feature.index,)) for feature in schema.features]
        if not groups
        else _ig_group_entities(
            schema, lidar_sector_degrees=lidar_sector_degrees
        )
    )
    lidar_indices = tuple(feature.index for feature in schema.lidar_features)
    for baseline_scope, baseline_index in baseline_modes:
        signed, absolute = _ig_baseline_values(result, baseline_index=baseline_index)
        for target_index, target in enumerate(result.targets):
            for entity_name, entity_kind, indices in entities:
                # Group signed attribution is the signed sum, whereas
                # group absolute mass is sum(abs(IG)); do not take abs
                # after summing because that would hide cancellation.
                signed_values = signed[:, target_index, :][:, indices].sum(axis=1)
                absolute_values = absolute[:, target_index, :][:, indices].sum(axis=1)
                row = _summary_row_base(
                    baseline_scope=baseline_scope,
                    baseline_index=baseline_index,
                )
                if groups:
                    row.update(
                        {
                            "group_name": entity_name,
                            "group_kind": entity_kind,
                            "group_size": len(indices),
                            "group_indices": indices,
                        }
                    )
                else:
                    feature = schema.feature_at(indices[0])
                    row.update(
                        {
                            "feature_index": feature.index,
                            "feature_name": feature.name,
                            "block": feature.block,
                            "kind": feature.kind,
                            "angle_deg": feature.angle_deg,
                        }
                    )
                row["target"] = target
                _add_statistics(row, "signed_ig", signed_values, count_key="count")
                # Feature-level standard names match the research report;
                # group aliases below additionally spell out "mass".
                absolute_stats = _statistics(absolute_values)
                row["mean_absolute_ig"] = absolute_stats["mean"]
                row["median_absolute_ig"] = absolute_stats["median"]
                row["p90_absolute_ig"] = absolute_stats["p90"]
                row["p95_absolute_ig"] = absolute_stats["p95"]
                row["max_absolute_ig"] = absolute_stats["max"]
                row["mean_absolute_mass"] = absolute_stats["mean"]
                row["median_absolute_mass"] = absolute_stats["median"]
                row["p90_absolute_mass"] = absolute_stats["p90"]
                row["p95_absolute_mass"] = absolute_stats["p95"]
                row["max_absolute_mass"] = absolute_stats["max"]
                if groups:
                    row["group_signed_ig"] = row["mean_signed_ig"]
                    row["group_absolute_mass"] = row["mean_absolute_mass"]
                    if entity_kind == "lidar_sector":
                        if not lidar_indices:  # Defensive: sector creation requires them.
                            raise AggregationError("LiDAR sectorに対応するLiDAR featureがありません")
                        lidar_absolute_mass = absolute[:, target_index, :][:, lidar_indices].sum(axis=1)
                        normalized_mass = np.divide(
                            absolute_values,
                            lidar_absolute_mass,
                            out=np.zeros_like(absolute_values, dtype=np.float64),
                            where=lidar_absolute_mass > 0.0,
                        )
                        normalized_stats = _statistics(normalized_mass)
                        row["normalization_basis"] = "per-sample total LiDAR absolute IG mass"
                        row["mean_normalized_absolute_mass"] = normalized_stats["mean"]
                        row["median_normalized_absolute_mass"] = normalized_stats["median"]
                        row["p90_normalized_absolute_mass"] = normalized_stats["p90"]
                        row["p95_normalized_absolute_mass"] = normalized_stats["p95"]
                        row["max_normalized_absolute_mass"] = normalized_stats["max"]
                        # Compact aliases ease CSV/report consumers while
                        # preserving the summary statistic in the name.
                        row["normalized_absolute_mass"] = normalized_stats["mean"]
                    else:
                        row["normalization_basis"] = None
                        row["mean_normalized_absolute_mass"] = float("nan")
                        row["median_normalized_absolute_mass"] = float("nan")
                        row["p90_normalized_absolute_mass"] = float("nan")
                        row["p95_normalized_absolute_mass"] = float("nan")
                        row["max_normalized_absolute_mass"] = float("nan")
                        row["normalized_absolute_mass"] = float("nan")
                if signed_values.size:
                    row["positive_rate"] = float(np.mean(signed_values > 0.0))
                    row["negative_rate"] = float(np.mean(signed_values < 0.0))
                    row["zero_rate"] = float(np.mean(signed_values == 0.0))
                else:
                    row["positive_rate"] = float("nan")
                    row["negative_rate"] = float("nan")
                    row["zero_rate"] = float("nan")
                rows.append(row)
    return rows


def summarize_ig_features(
    result: IntegratedGradientsResult,
    schema: ObservationSchema,
    *,
    include_baseline_rows: bool = False,
) -> pd.DataFrame:
    """Return the all-samples feature IG summary with signed and absolute values separate."""

    return pd.DataFrame(
        _ig_summary_rows(
            result,
            schema,
            include_baseline_rows=include_baseline_rows,
            groups=False,
        )
    )


def summarize_ig_groups(
    result: IntegratedGradientsResult,
    schema: ObservationSchema,
    *,
    include_baseline_rows: bool = False,
    lidar_sector_degrees: float | None = None,
) -> pd.DataFrame:
    """Return all-samples group/sector IG summaries with signed and absolute mass separate.

    When ``lidar_sector_degrees`` is supplied, the schema's angle metadata is
    expanded into non-overlapping LiDAR sector rows.  Sector absolute mass is
    also normalized by the per-sample total LiDAR absolute IG mass.
    """

    return pd.DataFrame(
        _ig_summary_rows(
            result,
            schema,
            include_baseline_rows=include_baseline_rows,
            groups=True,
            lidar_sector_degrees=lidar_sector_degrees,
        )
    )


def summarize_ig_completeness(
    result: IntegratedGradientsResult,
    *,
    include_baseline_rows: bool = False,
) -> pd.DataFrame:
    """Summarize all-samples IG completeness residuals without absolute IG sums."""

    rows: list[dict[str, object]] = []
    baseline_modes: list[tuple[str, int | None]] = [("mean_over_baselines", None)]
    if include_baseline_rows:
        baseline_modes.extend(("individual_baseline", index) for index in range(result.baseline_count))
    for baseline_scope, baseline_index in baseline_modes:
        for target_index, target in enumerate(result.targets):
            row = _summary_row_base(
                baseline_scope=baseline_scope,
                baseline_index=baseline_index,
            )
            row["target"] = target
            for name, values in (
                ("completeness_residual", result.completeness_residuals),
                ("absolute_completeness_residual", result.absolute_completeness_residuals),
                ("relative_completeness_error", result.relative_completeness_errors),
            ):
                if baseline_index is None:
                    selected = values[:, target_index, :].mean(axis=1)
                else:
                    selected = values[:, target_index, baseline_index]
                _add_statistics(
                    row,
                    name,
                    selected,
                    count_key="count" if name == "completeness_residual" else None,
                )
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_integrated_gradients(
    result: IntegratedGradientsResult,
    schema: ObservationSchema,
    *,
    include_baseline_rows: bool = False,
    lidar_sector_degrees: float | None = None,
) -> IntegratedGradientsSummaries:
    """Return the three all-samples DataFrames written as IG CSV artifacts."""

    shared = {
        "include_baseline_rows": include_baseline_rows,
    }
    return IntegratedGradientsSummaries(
        feature_summary=summarize_ig_features(result, schema, **shared),
        group_summary=summarize_ig_groups(
            result,
            schema,
            lidar_sector_degrees=lidar_sector_degrees,
            **shared,
        ),
        completeness_summary=summarize_ig_completeness(result, **shared),
    )


# Explicit aliases make command-layer code read naturally.
aggregate_perturbation = summarize_perturbation
aggregate_integrated_gradients = summarize_integrated_gradients
