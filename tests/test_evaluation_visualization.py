"""評価telemetryとGIF/MP4/PNG共通パネルを検証する。"""

from __future__ import annotations

import json
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import evaluate as evaluate_module
from env_factory import make_evaluation_env
from evaluation_visualization import (
    STEP_TELEMETRY_FIELDS,
    ActionSwitchTracker,
    RuntimeRoadMetrics,
    compose_telemetry_panel,
    decode_discrete_action,
    derive_timing,
    make_step_telemetry,
    read_runtime_road_metrics,
    replace_latest_recorded_frame,
)
from configs.phase0_config import OFFICIAL_ENV_CONFIG, SCENARIO_SEED


ACTION_CONFIG = {
    "discrete_action": True,
    "use_multi_discrete": False,
    "discrete_steering_dim": 3,
    "discrete_throttle_dim": 3,
    "vehicle_config": {"enable_reverse": False},
}


def test_runtime_timing_is_derived_from_effective_config(tmp_path: Path) -> None:
    timing = derive_timing(
        {"physics_world_step_size": 0.02, "decision_repeat": 5}
    )

    assert timing.physics_hz == pytest.approx(50.0)
    assert timing.control_hz == pytest.approx(10.0)
    assert timing.action_duration_seconds == pytest.approx(0.1)
    assert timing.gif_frame_duration_ms == 100
    assert timing.gif_playback_vs_simulation == pytest.approx(1.0)
    frame_count = 10
    assert frame_count * timing.gif_frame_duration_ms / 1000 == pytest.approx(1.0)
    assert frame_count / timing.control_hz == pytest.approx(1.0)

    gif_path = tmp_path / "ten_frames.gif"
    gif_frames = [
        Image.new("RGB", (2, 2), (frame_index * 20, 0, 0))
        for frame_index in range(frame_count)
    ]
    gif_frames[0].save(
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=timing.gif_frame_duration_ms,
        loop=0,
    )
    measured_count, measured_durations = evaluate_module._inspect_gif(gif_path)
    assert measured_count == frame_count
    assert measured_durations == [100] * frame_count
    assert sum(measured_durations) / 1000 == pytest.approx(1.0)


def test_gif_duration_must_be_exactly_representable() -> None:
    with pytest.raises(ValueError, match="cannot be represented exactly"):
        derive_timing({"physics_world_step_size": 0.003, "decision_repeat": 1})


def test_disabled_episode_recorder_creates_no_visualization_artifacts(
    tmp_path: Path,
) -> None:
    recorder = evaluate_module._EpisodeVisualizationRecorder(
        requested=False,
        run_dir=tmp_path / "evaluation",
        episode_number=1,
        scenario_seed=5,
    )
    recorder.prepare()
    result = recorder.finalize(
        env=None,
        timing=derive_timing(
            {"physics_world_step_size": 0.02, "decision_repeat": 5}
        ),
        expected_frame_count=3,
    )
    recorder.cleanup()

    assert result["artifact_directory"] is None
    assert result["gif"]["status"] == "not_requested"
    assert result["mp4"]["status"] == "not_requested"
    assert result["frames"]["status"] == "not_requested"
    assert not (tmp_path / "evaluation/episodes").exists()


def test_mp4_cleanup_failure_is_reported_without_aborting_evaluation(
    tmp_path: Path,
) -> None:
    recorder = evaluate_module._EpisodeVisualizationRecorder(
        requested=True,
        run_dir=tmp_path / "evaluation",
        episode_number=1,
        scenario_seed=5,
    )
    recorder.prepare()
    recorder.mp4_exception_trace = "injected MP4 failure"
    recorder.mp4_writer_temporary_path = recorder.paths["mp4_temporary"]
    recorder.mp4_writer_temporary_path.mkdir()

    result = recorder.finalize(
        env=SimpleNamespace(top_down_renderer=None),
        timing=derive_timing(
            {"physics_world_step_size": 0.02, "decision_repeat": 5}
        ),
        expected_frame_count=0,
    )
    recorder.cleanup()

    assert result["mp4"]["status"] == "failed"
    assert result["mp4"]["error"] == "injected MP4 failure"
    assert "IsADirectoryError" in result["mp4"]["cleanup_error"]


def test_discrete_action_and_switch_rate_are_explicit() -> None:
    left_throttle = decode_discrete_action(8, ACTION_CONFIG)
    right_brake = decode_discrete_action(np.asarray([0]), ACTION_CONFIG)

    assert (left_throttle.steering, left_throttle.throttle_brake) == (1.0, 1.0)
    assert left_throttle.label == "LEFT + THROTTLE"
    assert (right_brake.steering, right_brake.throttle_brake) == (-1.0, -1.0)
    assert right_brake.label == "RIGHT + BRAKE"

    tracker = ActionSwitchTracker(history_length=3)
    observed = [
        tracker.observe(action_id, sim_time_seconds=step * 0.1)
        for step, action_id in enumerate((7, 7, 6, 7), start=1)
    ]

    assert observed == [(0, 0.0), (0, 0.0), (1, pytest.approx(10 / 3)), (2, 5.0)]
    assert list(tracker.history) == [7, 6, 7]


def test_post_step_speed_and_runtime_road_values_are_kept_in_each_row() -> None:
    timing = derive_timing(
        {"physics_world_step_size": 0.02, "decision_repeat": 5}
    )
    road = RuntimeRoadMetrics(
        lane_width_m=3.5,
        lane_count_one_way=3,
        current_segment_drivable_width_m=11.2,
        current_segment_width_source="navigation.get_current_lateral_range",
        center_to_left_boundary_m=4.0,
        center_to_right_boundary_m=7.2,
        lane_center_offset_m=-0.3,
    )
    common = {
        "episode": 1,
        "scenario_seed": 5,
        "horizon": 1000,
        "timing": timing,
        "decoded_action": decode_discrete_action(7, ACTION_CONFIG),
        "reward": 0.25,
        "cumulative_reward": 0.25,
        "terminated": False,
        "truncated": False,
        "road": road,
        "action_switch_count": 0,
        "action_switches_per_second": 0.0,
    }
    first = make_step_telemetry(
        step=1,
        info={"velocity": 2.0, "route_completion": 0.1},
        **common,
    )
    second = make_step_telemetry(
        step=2,
        info={"velocity": 3.0, "route_completion": 0.2},
        **common,
    )

    assert tuple(first) == STEP_TELEMETRY_FIELDS
    assert first["speed_m_s"] == 2.0
    assert first["speed_km_h"] == pytest.approx(7.2)
    assert second["speed_m_s"] == 3.0
    assert second["sim_time_seconds"] == pytest.approx(0.2)
    assert second["current_segment_drivable_width_m"] == 11.2


def test_runtime_road_reader_uses_current_navigation_segment() -> None:
    class Navigation:
        def get_current_lane_width(self) -> float:
            return 3.5

        def get_current_lane_num(self) -> int:
            return 3

        def get_current_lateral_range(self, position: object, engine: object) -> float:
            assert position == (1.0, 2.0)
            assert engine == "engine"
            return 11.2

    class Lane:
        def local_coordinates(self, position: object) -> tuple[float, float]:
            assert position == (1.0, 2.0)
            return 8.0, -0.3

    vehicle = SimpleNamespace(
        navigation=Navigation(),
        position=(1.0, 2.0),
        engine="engine",
        lane=Lane(),
        dist_to_left_side=4.0,
        dist_to_right_side=7.2,
    )

    road = read_runtime_road_metrics(vehicle)

    assert road.current_segment_drivable_width_m == 11.2
    assert road.current_segment_width_source == "navigation.get_current_lateral_range"
    assert road.lane_width_m == 3.5
    assert road.lane_count_one_way == 3
    assert road.lane_center_offset_m == -0.3


def test_unavailable_runtime_road_values_fall_back_to_na() -> None:
    class BrokenVehicle:
        @property
        def navigation(self) -> object:
            raise RuntimeError("navigation unavailable")

        @property
        def lane(self) -> object:
            raise RuntimeError("lane unavailable")

        @property
        def dist_to_left_side(self) -> float:
            raise RuntimeError("boundary unavailable")

        @property
        def dist_to_right_side(self) -> float:
            raise RuntimeError("boundary unavailable")

    road = read_runtime_road_metrics(BrokenVehicle())

    assert road == RuntimeRoadMetrics(None, None, None, None, None, None, None)


def test_panel_appends_right_side_without_covering_map() -> None:
    frame = np.full((600, 600, 3), 127, dtype=np.uint8)
    timing = derive_timing(
        {"physics_world_step_size": 0.02, "decision_repeat": 5}
    )
    action = decode_discrete_action(8, ACTION_CONFIG)
    telemetry = make_step_telemetry(
        episode=1,
        scenario_seed=5,
        step=12,
        horizon=1000,
        timing=timing,
        decoded_action=action,
        info={
            "velocity": 12.0,
            "route_completion": 0.25,
            "steering": 1.0,
            "acceleration": 1.0,
        },
        reward=0.5,
        cumulative_reward=4.0,
        terminated=False,
        truncated=False,
        road=RuntimeRoadMetrics(3.5, 3, 11.2, "runtime", 4.0, 7.2, -0.3),
        action_switch_count=3,
        action_switches_per_second=2.5,
    )

    composite = compose_telemetry_panel(frame, telemetry, [action] * 20)

    assert composite.shape == (600, 920, 3)
    np.testing.assert_array_equal(composite[:, :600], frame)
    assert np.unique(composite[:, 600:].reshape(-1, 3), axis=0).shape[0] > 10
    # Timing remains in JSON telemetry for synchronization, but the derived
    # action-duration row is intentionally not drawn in the panel.
    panel_source = inspect.getsource(compose_telemetry_panel)
    assert "GIF PLAYBACK" not in panel_source
    assert "ACTION DT" not in panel_source
    for heading in (
        "EVALUATION TELEMETRY",
        "SIMULATION TIMING (runtime config)",
        "VEHICLE / APPLIED ACTION",
        "CURRENT SEGMENT (runtime)",
        "TASK",
    ):
        assert heading in panel_source

    renderer = SimpleNamespace(_screen_frames=[frame.copy()])
    replace_latest_recorded_frame(renderer, composite)
    assert renderer._screen_frames[-1].shape == (600, 920, 3)


def test_one_step_connects_to_metadrive_043_recorded_frame(
    tmp_path: Path,
) -> None:
    """実環境のruntime値と記録frameをローカルpanelへ接続できる。"""

    env = make_evaluation_env(
        seed=0,
        record_gif=True,
        env_config=OFFICIAL_ENV_CONFIG,
    )
    try:
        env.reset(seed=SCENARIO_SEED)
        action = 7
        _obs, reward, terminated, truncated, info = env.step(action)
        timing = derive_timing(env.config)
        decoded = decode_discrete_action(action, env.config)
        tracker = ActionSwitchTracker(history_length=20)
        switch_count, switch_rate = tracker.observe(
            decoded.action_id,
            sim_time_seconds=timing.action_duration_seconds,
        )
        telemetry = make_step_telemetry(
            episode=1,
            scenario_seed=SCENARIO_SEED,
            step=1,
            horizon=int(env.config["horizon"]),
            timing=timing,
            decoded_action=decoded,
            info=info,
            reward=float(reward),
            cumulative_reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            road=read_runtime_road_metrics(env.agent),
            action_switch_count=switch_count,
            action_switches_per_second=switch_rate,
        )
        frame = env.render(
            mode="topdown",
            screen_record=True,
            window=False,
            screen_size=(600, 600),
            camera_position=(50, 50),
        )
        composite = compose_telemetry_panel(frame, telemetry, [decoded])
        renderer = env.top_down_renderer
        assert renderer is not None
        replace_latest_recorded_frame(renderer, composite)

        assert telemetry["physics_hz"] == pytest.approx(50.0)
        assert telemetry["control_hz"] == pytest.approx(10.0)
        assert telemetry["speed_m_s"] == float(info["velocity"])
        assert renderer.screen_frames[-1].shape == (600, 920, 3)

        gif_path = tmp_path / "panel.gif"
        renderer.generate_gif(gif_name=str(gif_path), duration=100)
        assert gif_path.stat().st_size > 0
        with Image.open(gif_path) as gif_image:
            assert gif_image.size == (920, 600)
            assert gif_image.info["duration"] == 100
    finally:
        env.close()


def test_evaluate_keeps_all_steps_when_gif_and_trace_writes_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GIF系の二重failureでも評価JSONと全step JSONLを確定する。"""

    class Navigation:
        def get_current_lane_width(self) -> float:
            return 3.5

        def get_current_lane_num(self) -> int:
            return 3

        def get_current_lateral_range(self, _position: object, _engine: object) -> float:
            return 10.5

    class Lane:
        def local_coordinates(self, _position: object) -> tuple[float, float]:
            return 0.0, 0.0

    class FakeEnv:
        def __init__(self) -> None:
            self.config = {
                **OFFICIAL_ENV_CONFIG,
                "physics_world_step_size": 0.02,
                "decision_repeat": 5,
                "use_multi_discrete": False,
                "vehicle_config": {"enable_reverse": False},
            }
            self.observation_space = object()
            self.action_space = object()
            self.current_seed = SCENARIO_SEED
            self.top_down_renderer = None
            self.step_number = 0
            self.closed = False
            self.agent = SimpleNamespace(
                navigation=Navigation(),
                position=(0.0, 0.0),
                engine="engine",
                lane=Lane(),
                dist_to_left_side=5.25,
                dist_to_right_side=5.25,
            )

        def reset(self, *, seed: int) -> tuple[np.ndarray, dict[str, object]]:
            self.current_seed = seed
            self.step_number = 0
            return np.zeros(1, dtype=np.float32), {}

        def step(
            self,
            _action: object,
        ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
            self.step_number += 1
            terminated = self.step_number == 2
            info: dict[str, object] = {
                "velocity": float(self.step_number),
                "steering": 0.0 if self.step_number == 1 else 1.0,
                "acceleration": 1.0,
                "route_completion": self.step_number / 2,
                "arrive_dest": np.bool_(terminated),
                "out_of_road": False,
                "crash": False,
                "crash_vehicle": False,
                "crash_object": False,
                "max_step": False,
            }
            return (
                np.zeros(1, dtype=np.float32),
                1.0,
                terminated,
                False,
                info,
            )

        def render(self, **_kwargs: object) -> np.ndarray:
            raise RuntimeError("injected GIF render failure")

        def close(self) -> None:
            self.closed = True

    class FakeModel:
        def __init__(self) -> None:
            self.observation_space = object()
            self.action_space = object()
            self.device = "cpu"
            self.index = 0

        def predict(
            self,
            _observation: object,
            *,
            deterministic: bool,
        ) -> tuple[np.ndarray, None]:
            assert deterministic is True
            action_id = (7, 8)[self.index]
            self.index += 1
            return np.asarray([action_id]), None

    output_dir = tmp_path / "outputs"
    log_dir = tmp_path / "logs"
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"fake-model")
    fake_env = FakeEnv()
    fake_model = FakeModel()

    monkeypatch.setattr(evaluate_module, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(evaluate_module, "LOG_DIR", log_dir)
    monkeypatch.setattr(
        evaluate_module.PPO,
        "load",
        lambda *_args, **_kwargs: fake_model,
    )
    monkeypatch.setattr(
        evaluate_module,
        "make_evaluation_env",
        lambda **_kwargs: fake_env,
    )
    monkeypatch.setattr(
        evaluate_module,
        "check_for_correct_spaces",
        lambda *_args, **_kwargs: None,
    )

    def fail_trace_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected GIF trace write failure")

    monkeypatch.setattr(evaluate_module, "_write_gif_trace", fail_trace_write)
    args = SimpleNamespace(
        profile="official",
        model=model_path,
        episodes=1,
        record_gif=True,
        output_prefix="failure_case",
        seed=0,
        device="cpu",
    )

    result_path = evaluate_module._evaluate(args, log_dir / "evaluate.log")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    step_path = Path(result["step_telemetry"]["path"])
    step_rows = [
        json.loads(line)
        for line in step_path.read_text(encoding="utf-8").splitlines()
    ]

    assert fake_env.closed is True
    assert result["evaluation_status"] == "success"
    assert result["gif"]["status"] == "failed"
    assert result["gif"]["traceback_path"] is None
    assert "injected GIF trace write failure" in result["gif"][
        "traceback_write_error"
    ]
    assert not Path(result["gif"]["output_path"]).exists()
    assert result["mp4"]["status"] == "failed"
    assert not Path(result["mp4"]["output_path"]).exists()
    assert not Path(result["mp4"]["output_path"]).with_name(
        "evaluation.tmp.mp4"
    ).exists()
    assert result["episodes"][0]["visualization"]["artifact_directory"]
    assert result["visualizations"] == [result["episodes"][0]["visualization"]]
    assert result["step_telemetry"]["row_count"] == 2
    assert [row["speed_m_s"] for row in step_rows] == [1.0, 2.0]
    assert step_rows[-1]["action_switch_count"] == 1
    assert step_rows[-1]["action_switches_per_second"] == pytest.approx(5.0)
    assert result["episodes"][0]["termination_reason"] == "success"


def test_evaluate_finalizes_png_gif_mp4_with_simulation_timing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A short fake episode exercises all three successful recording paths."""

    pytest.importorskip("cv2")

    class FakeRenderer:
        def __init__(self) -> None:
            self._screen_frames: list[np.ndarray] = []
            self.gif_duration: int | None = None

        @property
        def screen_frames(self) -> list[np.ndarray]:
            return [frame.copy() for frame in self._screen_frames]

        def generate_gif(self, *, gif_name: str, duration: int) -> None:
            self.gif_duration = duration
            images = [Image.fromarray(frame) for frame in self._screen_frames]
            if not images:
                raise RuntimeError("no frames")
            images[0].save(
                gif_name,
                save_all=True,
                append_images=images[1:],
                duration=duration,
                loop=0,
            )

    class FakeEnv:
        def __init__(self) -> None:
            self.config = {
                **OFFICIAL_ENV_CONFIG,
                "physics_world_step_size": 0.02,
                "decision_repeat": 5,
                "use_multi_discrete": False,
                "vehicle_config": {"enable_reverse": False},
                "horizon": 3,
            }
            self.observation_space = object()
            self.action_space = object()
            self.current_seed = SCENARIO_SEED
            self.top_down_renderer = FakeRenderer()
            self.step_number = 0
            self.closed = False
            self.agent = SimpleNamespace(
                navigation=None,
                lane=None,
                dist_to_left_side=None,
                dist_to_right_side=None,
                position=(0.0, 0.0),
                engine=None,
            )

        def reset(self, *, seed: int) -> tuple[np.ndarray, dict[str, object]]:
            self.current_seed = seed
            self.step_number = 0
            self.top_down_renderer._screen_frames.clear()
            return np.zeros(1, dtype=np.float32), {}

        def step(
            self,
            _action: object,
        ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
            self.step_number += 1
            terminated = self.step_number == 3
            info: dict[str, object] = {
                "velocity": float(self.step_number),
                "steering": 0.0,
                "acceleration": 1.0,
                "route_completion": self.step_number / 3.0,
                "arrive_dest": terminated,
                "out_of_road": False,
                "crash": False,
                "crash_vehicle": False,
                "crash_object": False,
                "max_step": False,
            }
            return (
                np.zeros(1, dtype=np.float32),
                1.0,
                terminated,
                False,
                info,
            )

        def render(self, **_kwargs: object) -> np.ndarray:
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            frame[:, :, 0] = self.step_number * 40
            self.top_down_renderer._screen_frames.append(frame)
            return frame

        def close(self) -> None:
            self.closed = True

    class FakeModel:
        def __init__(self) -> None:
            self.observation_space = object()
            self.action_space = object()
            self.device = "cpu"
            self.index = 0

        def predict(
            self,
            _observation: object,
            *,
            deterministic: bool,
        ) -> tuple[np.ndarray, None]:
            assert deterministic is True
            action_id = (7, 7, 8, 7, 7, 8)[self.index]
            self.index += 1
            return np.asarray([action_id]), None

    output_dir = tmp_path / "outputs"
    log_dir = tmp_path / "logs"
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"fake-model")
    stale_episode_dir = (
        output_dir
        / "official/evaluation/success_case/episodes/episode_9999_scenario_999999"
    )
    stale_episode_dir.mkdir(parents=True)
    (stale_episode_dir / "stale.txt").write_text("old run", encoding="utf-8")
    fake_env = FakeEnv()
    fake_model = FakeModel()

    monkeypatch.setattr(evaluate_module, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(evaluate_module, "LOG_DIR", log_dir)
    monkeypatch.setattr(
        evaluate_module.PPO,
        "load",
        lambda *_args, **_kwargs: fake_model,
    )
    monkeypatch.setattr(
        evaluate_module,
        "make_evaluation_env",
        lambda **_kwargs: fake_env,
    )
    monkeypatch.setattr(
        evaluate_module,
        "check_for_correct_spaces",
        lambda *_args, **_kwargs: None,
    )
    args = SimpleNamespace(
        profile="official",
        model=model_path,
        episodes=2,
        record_gif=True,
        output_prefix="success_case",
        seed=0,
        device="cpu",
    )

    result_path = evaluate_module._evaluate(args, log_dir / "evaluate.log")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert len(result["episodes"]) == 2
    assert len(result["visualizations"]) == 2
    assert result["visualizations"] == [
        episode["visualization"] for episode in result["episodes"]
    ]
    assert result["gif"] == result["episodes"][-1]["visualization"]["gif"]
    assert result["mp4"] == result["episodes"][-1]["visualization"]["mp4"]
    assert result["frames"] == result["episodes"][-1]["visualization"]["frames"]
    gif = result["gif"]
    mp4 = result["mp4"]
    frames = result["frames"]

    assert fake_env.closed is True
    assert fake_env.top_down_renderer.gif_duration == 100
    assert gif["status"] == "success"
    assert mp4["status"] == "success"
    assert frames["status"] == "success"
    assert gif["frame_count"] == mp4["frame_count"] == frames["frame_count"] == 3
    assert gif["frame_duration_ms"] == 100
    assert mp4["fps"] == pytest.approx(10.0)
    assert mp4["measured_fps"] == pytest.approx(10.0, rel=1e-3)
    assert gif["playback_duration_seconds"] == pytest.approx(0.3)
    assert mp4["playback_duration_seconds"] == pytest.approx(0.3)
    assert gif["expected_playback_duration_seconds"] == pytest.approx(0.3)
    assert mp4["expected_playback_duration_seconds"] == pytest.approx(0.3)
    assert gif["playback_duration_seconds"] == pytest.approx(
        gif["expected_playback_duration_seconds"]
    )
    assert mp4["playback_duration_seconds"] == pytest.approx(
        mp4["expected_playback_duration_seconds"]
    )
    for episode in result["episodes"]:
        visualization = episode["visualization"]
        assert visualization["episode"] == episode["episode"]
        assert visualization["scenario_seed"] == episode["scenario_seed"]
        assert visualization["requested"] is True
        assert episode["episode_length"] == 3
        assert episode["simulation_seconds"] == pytest.approx(0.3)
        assert visualization["gif"]["status"] == "success"
        assert visualization["mp4"]["status"] == "success"
        assert visualization["frames"]["status"] == "success"
        assert visualization["gif"]["frame_count"] == 3
        assert visualization["mp4"]["frame_count"] == 3
        assert visualization["frames"]["frame_count"] == 3
        assert visualization["gif"]["playback_duration_seconds"] == pytest.approx(0.3)
        assert visualization["mp4"]["playback_duration_seconds"] == pytest.approx(0.3)

    artifact_directories = [
        Path(episode["visualization"]["artifact_directory"])
        for episode in result["episodes"]
    ]
    assert artifact_directories[0] != artifact_directories[1]
    assert artifact_directories[0].name == "episode_0001_scenario_000005"
    assert artifact_directories[1].name == "episode_0002_scenario_000005"
    assert not stale_episode_dir.exists()

    for episode in result["episodes"]:
        visualization = episode["visualization"]
        for artifact_path in (
            Path(visualization["gif"]["output_path"]),
            Path(visualization["mp4"]["output_path"]),
        ):
            assert artifact_path.is_file()
            assert artifact_path.stat().st_size > 0
        frame_paths = sorted(
            Path(visualization["frames"]["output_path"]).glob("frame_*.png")
        )
        assert len(frame_paths) == 3
        assert all(path.stat().st_size > 0 for path in frame_paths)
        assert not Path(visualization["mp4"]["output_path"]).with_name(
            "evaluation.tmp.mp4"
        ).exists()


def test_numpy_nonfinite_values_are_strict_json_safe() -> None:
    converted = evaluate_module._json_value(
        {"scalar": np.float64(np.nan), "array": np.asarray([np.inf])}
    )

    assert converted == {"scalar": "nan", "array": ["inf"]}
    assert json.loads(json.dumps(converted, allow_nan=False)) == converted


def test_mp4_writer_round_trip_preserves_control_fps_and_frame_count(
    tmp_path: Path,
) -> None:
    """OpenCV writer uses CFR control timing and finalizes atomically."""

    pytest.importorskip("cv2")
    output_path = tmp_path / "panel.mp4"
    writer, temporary_path = evaluate_module._open_mp4_writer(
        output_path,
        fps=10.0,
        frame_size=(8, 6),
    )
    try:
        for frame_index in range(10):
            value = frame_index * 25
            rgb_frame = np.full((6, 8, 3), value, dtype=np.uint8)
            writer.write(rgb_frame[:, :, ::-1].copy())
    finally:
        assert evaluate_module._release_mp4_writer(writer) is None
    temporary_path.replace(output_path)

    frame_count, measured_fps = evaluate_module._inspect_mp4(
        output_path,
        expected_fps=10.0,
    )
    assert frame_count == 10
    assert measured_fps == pytest.approx(10.0, rel=1e-3)
    assert frame_count / measured_fps == pytest.approx(1.0)
