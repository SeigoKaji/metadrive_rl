"""Headless matplotlib visualizations for attribution artifacts.

Plot labels are deliberately English so saved files do not depend on an OS
Japanese font.  Semantic names, LiDAR indices, and angles always come from the
external observation schema rather than from fixed observation dimensions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path
import tempfile
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt
import numpy as np

from .results import expanded_schema_rows, safe_child


class VisualizationError(ValueError):
    """Raised for malformed numeric data passed to a plot helper."""


def _save_figure(figure: Any, path: Path) -> Path:
    """Save a PNG through a sibling temporary, preserving old plot files."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        figure.savefig(temporary, format="png", dpi=160, bbox_inches="tight")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(figure)
    return path


def placeholder_figure(path: Path, *, title: str, reason: str) -> Path:
    """Create a consistently named explanatory plot for unavailable data."""

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.set_title(title)
    axis.text(
        0.5,
        0.52,
        reason,
        ha="center",
        va="center",
        wrap=True,
        transform=axis.transAxes,
    )
    axis.set_axis_off()
    return _save_figure(figure, path)


def _records(data: Any) -> list[dict[str, Any]]:
    if data is None:
        return []
    if hasattr(data, "to_dict") and hasattr(data, "columns"):
        result = data.to_dict(orient="records")
        return [dict(row) for row in result]
    if isinstance(data, Mapping):
        # A mapping of equal-length columns is common for compact core
        # results.  A scalar mapping represents exactly one row.
        values = list(data.values())
        sequences = [
            value
            for value in values
            if isinstance(value, (list, tuple, np.ndarray)) and np.asarray(value).ndim == 1
        ]
        if sequences and all(len(value) == len(sequences[0]) for value in sequences):
            rows: list[dict[str, Any]] = []
            for index in range(len(sequences[0])):
                rows.append(
                    {
                        str(key): (
                            value[index]
                            if isinstance(value, (list, tuple, np.ndarray))
                            and np.asarray(value).ndim == 1
                            else value
                        )
                        for key, value in data.items()
                    }
                )
            return rows
        return [dict(data)]
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
        rows: list[dict[str, Any]] = []
        for item in data:
            if isinstance(item, Mapping):
                rows.append(dict(item))
            elif hasattr(item, "_asdict"):
                rows.append(dict(item._asdict()))
            elif hasattr(item, "__dict__"):
                rows.append(dict(item.__dict__))
        return rows
    return []


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _first_column(rows: Sequence[Mapping[str, Any]], names: Sequence[str]) -> str | None:
    for name in names:
        if any(_numeric(row.get(name)) is not None for row in rows):
            return name
    return None


def _schema_maps(schema: Any) -> tuple[dict[int, str], list[dict[str, Any]]]:
    rows = expanded_schema_rows(schema)
    names = {int(row["index"]): str(row["name"]) for row in rows}
    return names, rows


def _feature_labels(rows: Sequence[Mapping[str, Any]], schema: Any) -> list[str]:
    names, _schema_rows = _schema_maps(schema)
    labels: list[str] = []
    for position, row in enumerate(rows):
        value = row.get("feature_name", row.get("name"))
        if value is not None:
            labels.append(str(value))
            continue
        index = row.get("feature_index", row.get("index", position))
        try:
            labels.append(names[int(index)])
        except (KeyError, TypeError, ValueError):
            labels.append(f"feature_{index}")
    return labels


def plot_top_features(
    path: Path,
    *,
    title: str,
    summary: Any,
    schema: Any,
    metric_candidates: Sequence[str],
    top_k: int = 20,
    x_label: str,
) -> Path:
    """Plot the largest finite feature score rows, or a placeholder."""

    rows = _records(summary)
    metric = _first_column(rows, metric_candidates)
    if metric is None:
        return placeholder_figure(
            path,
            title=title,
            reason="No finite feature-level analysis data was available.",
        )
    ranked = [
        row for row in rows if _numeric(row.get(metric)) is not None
    ]
    if not ranked:
        return placeholder_figure(path, title=title, reason="No finite scores were available.")
    ranked = sorted(ranked, key=lambda row: abs(float(row[metric])), reverse=True)[:top_k]
    labels = _feature_labels(ranked, schema)
    values = [float(row[metric]) for row in ranked]
    figure, axis = plt.subplots(figsize=(10, max(4.5, 0.35 * len(ranked) + 1.5)))
    positions = np.arange(len(ranked))
    axis.barh(positions, values, color="#3b82b6")
    axis.set_yticks(positions, labels=labels)
    axis.invert_yaxis()
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.grid(axis="x", alpha=0.25)
    return _save_figure(figure, path)


def plot_top_groups(
    path: Path,
    *,
    title: str,
    summary: Any,
    metric_candidates: Sequence[str],
    top_k: int = 20,
    x_label: str,
    signed: bool = False,
) -> Path:
    """Plot group summaries, preserving the signed/absolute distinction."""

    rows = _records(summary)
    metric = _first_column(rows, metric_candidates)
    if metric is None:
        return placeholder_figure(
            path,
            title=title,
            reason="No finite group-level analysis data was available.",
        )
    ranked = [row for row in rows if _numeric(row.get(metric)) is not None]
    if not ranked:
        return placeholder_figure(path, title=title, reason="No finite scores were available.")
    ranked = sorted(ranked, key=lambda row: abs(float(row[metric])), reverse=True)[:top_k]
    labels = [str(row.get("group_name", row.get("name", "unnamed_group"))) for row in ranked]
    values = [float(row[metric]) for row in ranked]
    colors = ["#c44e52" if value < 0 else "#3b82b6" for value in values] if signed else "#3b82b6"
    figure, axis = plt.subplots(figsize=(10, max(4.5, 0.35 * len(ranked) + 1.5)))
    positions = np.arange(len(ranked))
    axis.barh(positions, values, color=colors)
    axis.set_yticks(positions, labels=labels)
    axis.invert_yaxis()
    if signed:
        axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.grid(axis="x", alpha=0.25)
    return _save_figure(figure, path)


def _matrix_from(value: Any, names: Sequence[str]) -> np.ndarray | None:
    for name in names:
        candidate = value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)
        if candidate is None:
            continue
        matrix = np.asarray(candidate, dtype=np.float32)
        if matrix.ndim == 2 and matrix.size:
            return matrix
        if matrix.ndim == 3 and matrix.size:
            # Baseline/target axes are intentionally averaged only for an
            # overview heatmap; numeric artifacts retain the full data.
            return np.nanmean(matrix, axis=tuple(range(1, matrix.ndim - 1)))
    return None


def _time_axis(
    steps: Any | None,
    size: int,
) -> tuple[np.ndarray, str, tuple[int, ...]]:
    """Choose an honest x-axis and expose concatenated episode boundaries.

    Per-episode decision steps restart at one.  Treating a concatenated
    ``[1..T, 1..U]`` vector as one numeric x-axis would draw a backwards line
    and make the heatmap ticks ambiguous.  In that case use a monotonic global
    decision index, split line segments, and mark every reset boundary.
    """

    global_index = np.arange(size)
    if steps is None:
        return global_index, "Global policy decision index", ()
    candidate = np.asarray(steps).reshape(-1)
    if candidate.size != size or size == 0:
        return global_index, "Global policy decision index", ()
    resets = tuple(int(index) for index in np.flatnonzero(np.diff(candidate) <= 0) + 1)
    if resets:
        return (
            global_index,
            "Global policy decision index (episode boundaries marked)",
            resets,
        )
    return candidate, "Policy decision step", ()


def plot_importance_over_time(
    path: Path,
    *,
    title: str,
    values: Any,
    steps: Any | None,
    y_label: str,
) -> Path:
    """Plot one mean absolute importance value per recorded decision."""

    matrix = _matrix_from(values, ("signed_attributions", "attributions", "ig", "values", "scores"))
    if matrix is None:
        return placeholder_figure(path, title=title, reason="No step-wise data was available.")
    series = np.nanmean(np.abs(matrix), axis=1)
    x, x_label, boundaries = _time_axis(steps, series.size)
    figure, axis = plt.subplots(figsize=(10, 4.5))
    starts = (0, *boundaries)
    ends = (*boundaries, series.size)
    for start, end in zip(starts, ends, strict=True):
        axis.plot(x[start:end], series[start:end], color="#3b82b6", linewidth=1.5)
    for boundary in boundaries:
        axis.axvline(boundary - 0.5, color="black", linestyle="--", linewidth=0.7, alpha=0.5)
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(alpha=0.25)
    return _save_figure(figure, path)


def plot_lidar_heatmap(
    path: Path,
    *,
    title: str,
    values: Any,
    schema: Any,
    steps: Any | None,
    colorbar_label: str,
) -> Path:
    """Render step × schema-defined LiDAR angle heatmap without fixed indices."""

    matrix = _matrix_from(values, ("signed_attributions", "attributions", "ig", "values", "scores"))
    _names, rows = _schema_maps(schema)
    lidar = [
        row
        for row in rows
        if str(row.get("kind")) == "lidar" and _numeric(row.get("angle_deg")) is not None
    ]
    if matrix is None or not lidar:
        reason = "No LiDAR features are defined by this schema." if not lidar else "No LiDAR step data was available."
        return placeholder_figure(path, title=title, reason=reason)
    indices = [int(row["index"]) for row in lidar]
    if matrix.ndim != 2 or matrix.shape[1] <= max(indices):
        return placeholder_figure(
            path,
            title=title,
            reason="LiDAR indices did not match the available analysis matrix.",
        )
    ordered = sorted(lidar, key=lambda row: float(row["angle_deg"]))
    indices = [int(row["index"]) for row in ordered]
    angles = [float(row["angle_deg"]) for row in ordered]
    heatmap = np.abs(matrix[:, indices]).T
    x_values, x_label, boundaries = _time_axis(steps, matrix.shape[0])
    figure, axis = plt.subplots(figsize=(11, 5.5))
    image = axis.imshow(heatmap, aspect="auto", interpolation="nearest", origin="lower")
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel("LiDAR angle (degrees; schema direction)")
    tick_count = min(8, len(angles))
    if tick_count:
        tick_positions = np.linspace(0, len(angles) - 1, tick_count, dtype=int)
        axis.set_yticks(tick_positions, labels=[f"{angles[index]:g}" for index in tick_positions])
    if x_values.size:
        tick_count = min(8, x_values.size)
        tick_positions = np.linspace(0, x_values.size - 1, tick_count, dtype=int)
        axis.set_xticks(tick_positions, labels=[str(x_values[index]) for index in tick_positions])
    for boundary in boundaries:
        axis.axvline(boundary - 0.5, color="white", linestyle="--", linewidth=0.8, alpha=0.8)
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label(colorbar_label)
    return _save_figure(figure, path)


def plot_custom_feature_timeseries(
    path: Path,
    *,
    values: Any,
    schema: Any,
    steps: Any | None,
    ig_target_name: str | None = None,
) -> Path:
    """Plot custom features for one explicitly named IG target, when available."""

    suffix = "" if ig_target_name is None else f" ({ig_target_name})"
    title = f"Custom feature attribution over time{suffix}"
    matrix = _matrix_from(values, ("signed_attributions", "attributions", "ig", "values", "scores"))
    _names, rows = _schema_maps(schema)
    custom = [row for row in rows if str(row.get("kind")) == "custom"]
    if matrix is None or not custom:
        reason = "No custom features are defined by this schema." if not custom else "No custom feature data was available."
        return placeholder_figure(
            path,
            title=title,
            reason=reason,
        )
    indices = [int(row["index"]) for row in custom]
    if matrix.shape[1] <= max(indices):
        return placeholder_figure(
            path,
            title=title,
            reason="Custom feature indices did not match the available matrix.",
        )
    x, x_label, boundaries = _time_axis(steps, matrix.shape[0])
    figure, axis = plt.subplots(figsize=(10, 4.5))
    for row, index in zip(custom, indices, strict=True):
        starts = (0, *boundaries)
        ends = (*boundaries, matrix.shape[0])
        for segment, (start, end) in enumerate(zip(starts, ends, strict=True)):
            axis.plot(
                x[start:end],
                matrix[start:end, index],
                linewidth=1.3,
                label=str(row["name"]) if segment == 0 else None,
            )
    for boundary in boundaries:
        axis.axvline(boundary - 0.5, color="black", linestyle="--", linewidth=0.7, alpha=0.5)
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel("Signed attribution")
    axis.legend(loc="best", fontsize="small")
    axis.grid(alpha=0.25)
    return _save_figure(figure, path)


def plot_closed_loop_performance(path: Path, *, summary: Any) -> Path:
    """Plot paired reward change for each intervention target."""

    rows = _records(summary)
    metric = _first_column(rows, ("mean_delta_total_reward", "delta_total_reward"))
    if metric is None:
        return placeholder_figure(
            path,
            title="Closed-loop paired performance",
            reason="Closed-loop validation was disabled or produced no finite paired rewards.",
        )
    rows = [row for row in rows if _numeric(row.get(metric)) is not None]
    if not rows:
        return placeholder_figure(
            path,
            title="Closed-loop paired performance",
            reason="No finite paired reward deltas were available.",
        )
    labels = [str(row.get("target_name", row.get("name", "target"))) for row in rows]
    values = [float(row[metric]) for row in rows]
    figure, axis = plt.subplots(figsize=(10, max(4.5, 0.35 * len(rows) + 1.5)))
    positions = np.arange(len(rows))
    axis.barh(positions, values, color=["#c44e52" if value < 0 else "#3b82b6" for value in values])
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_yticks(positions, labels=labels)
    axis.invert_yaxis()
    axis.set_title("Closed-loop paired performance")
    axis.set_xlabel("Mean intervention − baseline total reward")
    axis.grid(axis="x", alpha=0.25)
    return _save_figure(figure, path)


def generate_standard_plots(
    output_directory: Path,
    *,
    schema: Any,
    perturbation_feature_summary: Any | None = None,
    perturbation_group_summary: Any | None = None,
    ig_feature_summary: Any | None = None,
    ig_group_summary: Any | None = None,
    ig_attributions: Any | None = None,
    perturbation_step_values: Any | None = None,
    steps: Any | None = None,
    closed_loop_summary: Any | None = None,
    top_k: int = 20,
    ig_target_name: str | None = None,
) -> dict[str, Path]:
    """Create all standard plot names, including reason-bearing placeholders."""

    ig_suffix = "" if ig_target_name is None else f" ({ig_target_name})"
    plots = safe_child(output_directory, "plots")
    plots.mkdir(parents=True, exist_ok=True)
    paths = {
        "perturbation_feature_top": plots / "perturbation_feature_top.png",
        "perturbation_group_top": plots / "perturbation_group_top.png",
        "ig_feature_top_absolute": plots / "ig_feature_top_absolute.png",
        "ig_group_top_absolute": plots / "ig_group_top_absolute.png",
        "ig_group_signed": plots / "ig_group_signed.png",
        "attribution_over_time": plots / "attribution_over_time.png",
        "perturbation_over_time": plots / "perturbation_over_time.png",
        "lidar_ig_heatmap": plots / "lidar_ig_heatmap.png",
        "lidar_perturbation_heatmap": plots / "lidar_perturbation_heatmap.png",
        "closed_loop_performance": plots / "closed_loop_performance.png",
        "custom_feature_attribution_over_time": plots / "custom_feature_attribution_over_time.png",
    }
    plot_top_features(
        paths["perturbation_feature_top"],
        title="Top input dependence by perturbation",
        summary=perturbation_feature_summary,
        schema=schema,
        metric_candidates=("mean_js_divergence", "mean_js", "mean_absolute_value_delta"),
        top_k=top_k,
        x_label="Mean perturbation score (Jensen-Shannon divergence where available)",
    )
    plot_top_groups(
        paths["perturbation_group_top"],
        title="Top input groups by perturbation",
        summary=perturbation_group_summary,
        metric_candidates=("mean_js_divergence", "mean_js", "mean_absolute_value_delta"),
        top_k=top_k,
        x_label="Mean perturbation score (Jensen-Shannon divergence where available)",
    )
    plot_top_features(
        paths["ig_feature_top_absolute"],
        title=f"Top features by mean absolute Integrated Gradients{ig_suffix}",
        summary=ig_feature_summary,
        schema=schema,
        metric_candidates=("mean_absolute_ig", "mean_abs_ig", "absolute_mean"),
        top_k=top_k,
        x_label="Mean absolute Integrated Gradients",
    )
    plot_top_groups(
        paths["ig_group_top_absolute"],
        title=f"Top groups by Integrated Gradients absolute mass{ig_suffix}",
        summary=ig_group_summary,
        metric_candidates=("group_absolute_mass", "mean_absolute_ig", "mean_abs_ig"),
        top_k=top_k,
        x_label="Integrated Gradients absolute mass",
    )
    plot_top_groups(
        paths["ig_group_signed"],
        title=f"Signed Integrated Gradients by group{ig_suffix}",
        summary=ig_group_summary,
        metric_candidates=("group_signed_ig", "mean_signed_ig", "mean_ig"),
        top_k=top_k,
        x_label="Signed Integrated Gradients",
        signed=True,
    )
    plot_importance_over_time(
        paths["attribution_over_time"],
        title=f"Integrated Gradients magnitude over time{ig_suffix}",
        values=ig_attributions,
        steps=steps,
        y_label="Mean absolute Integrated Gradients",
    )
    plot_importance_over_time(
        paths["perturbation_over_time"],
        title="Perturbation dependence over time",
        values=perturbation_step_values,
        steps=steps,
        y_label="Mean absolute perturbation score",
    )
    plot_lidar_heatmap(
        paths["lidar_ig_heatmap"],
        title=f"LiDAR Integrated Gradients heatmap{ig_suffix}",
        values=ig_attributions,
        schema=schema,
        steps=steps,
        colorbar_label="Absolute Integrated Gradients",
    )
    plot_lidar_heatmap(
        paths["lidar_perturbation_heatmap"],
        title="LiDAR perturbation heatmap",
        values=perturbation_step_values,
        schema=schema,
        steps=steps,
        colorbar_label="Absolute perturbation score",
    )
    plot_closed_loop_performance(paths["closed_loop_performance"], summary=closed_loop_summary)
    plot_custom_feature_timeseries(
        paths["custom_feature_attribution_over_time"],
        values=ig_attributions,
        schema=schema,
        steps=steps,
        ig_target_name=ig_target_name,
    )
    return paths
