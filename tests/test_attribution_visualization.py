"""Time-axis semantics for multi-episode attribution plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from input_attribution import visualization
from input_attribution.schema import Feature, ObservationSchema
from input_attribution.visualization import _time_axis


def test_time_axis_uses_episode_steps_for_one_monotonic_episode() -> None:
    x, label, boundaries = _time_axis(np.asarray([1, 2, 3]), 3)

    np.testing.assert_array_equal(x, [1, 2, 3])
    assert label == "Policy decision step"
    assert boundaries == ()


def test_time_axis_uses_global_index_and_marks_episode_resets() -> None:
    x, label, boundaries = _time_axis(np.asarray([1, 2, 3, 1, 2]), 5)

    np.testing.assert_array_equal(x, np.arange(5))
    assert "Global policy decision index" in label
    assert boundaries == (3,)


def test_custom_feature_timeseries_title_names_the_ig_target(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    schema = ObservationSchema(
        schema_version=1,
        name="custom_plot",
        observation_dim=1,
        features=(Feature(0, "custom_signal", "custom", kind="custom"),),
        groups={"custom": (0,)},
    )
    captured: dict[str, str] = {}

    def capture(figure: object, path: Path) -> Path:
        captured["title"] = figure.axes[0].get_title()
        visualization.plt.close(figure)
        return path

    monkeypatch.setattr(visualization, "_save_figure", capture)
    output = visualization.plot_custom_feature_timeseries(
        tmp_path / "custom.png",
        values={"attributions": np.asarray([[0.1], [0.2]], dtype=np.float32)},
        schema=schema,
        steps=np.asarray([1, 2]),
        ig_target_name="selected_log_probability",
    )

    assert output == tmp_path / "custom.png"
    assert captured["title"] == "Custom feature attribution over time (selected_log_probability)"
