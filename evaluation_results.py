"""Evaluation result serialization and per-episode artifact lifecycle helpers."""

from __future__ import annotations

import json
import math
import shutil
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any, Mapping, TextIO

import numpy as np
from PIL import Image

from evaluation_visualization import (
    ACTION_HISTORY_SECONDS,
    TELEMETRY_PANEL_WIDTH,
    ActionSwitchTracker,
    SimulationTiming,
    compose_telemetry_panel,
    decode_discrete_action,
    replace_latest_recorded_frame,
)


GIF_RENDER_API = (
    'env.render(mode="topdown", screen_record=True, window=False, '
    "screen_size=(600, 600), camera_position=(50, 50))"
)
GIF_GENERATE_API = (
    "env.top_down_renderer.generate_gif("
    "gif_name=<output>, duration=<runtime action dt in ms>)"
)
MP4_CODEC = "mp4v"


def _json_value(value: Any) -> Any:
    """Convert NumPy and container values to JSON-safe Python values."""

    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _optional_info_value(info: Mapping[str, Any], key: str) -> Any:
    """Read optional MetaDrive info without manufacturing a missing value."""

    if key not in info:
        return None
    return _json_value(info.get(key))


def _termination_reason(
    *,
    terminated: bool,
    truncated: bool,
    flags: Mapping[str, Any],
) -> str:
    """Classify the end while retaining every original flag separately."""

    if bool(flags.get("arrive_dest", False)):
        return "success"
    if bool(flags.get("out_of_road", False)):
        return "out_of_road"
    if bool(flags.get("crash_vehicle", False)):
        return "crash_vehicle"
    if bool(flags.get("crash_object", False)):
        return "crash_object"
    if terminated:
        return "other_termination"
    if truncated:
        return "max_step_truncation"
    return "unknown"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write UTF-8 JSON through a sibling temporary file."""

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(
            _json_value(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _write_jsonl_record(file_obj: TextIO, payload: Mapping[str, Any]) -> None:
    """Append one complete telemetry row and make it visible immediately."""

    file_obj.write(
        json.dumps(
            _json_value(payload),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    file_obj.flush()


def _write_gif_trace(
    path: Path,
    exception_trace: str,
    attempted_apis: list[str],
) -> None:
    """Persist a complete GIF failure traceback independently of evaluation."""

    attempted = "\n".join(f"- {api}" for api in attempted_apis) or "- none"
    path.write_text(
        "MetaDrive 0.4.3 GIF recording failed.\n"
        f"Attempted APIs:\n{attempted}\n\n"
        f"{exception_trace}",
        encoding="utf-8",
    )


def _try_write_gif_trace(
    path: Path,
    exception_trace: str,
    attempted_apis: list[str],
    *,
    write_gif_trace: Callable[[Path, str, list[str]], None] | None = None,
) -> tuple[str | None, str | None]:
    """Persist a GIF traceback without letting log I/O fail evaluation."""

    try:
        (write_gif_trace or _write_gif_trace)(
            path,
            exception_trace,
            attempted_apis,
        )
    except Exception:
        return None, traceback.format_exc()
    return str(path.resolve()), None


def _try_unlink(path: Path) -> str | None:
    """Remove one artifact without allowing cleanup to mask evaluation state."""

    try:
        path.unlink(missing_ok=True)
    except Exception:
        return traceback.format_exc().strip()
    return None


def _temporary_mp4_path(path: Path) -> Path:
    """Return a sibling temporary path that retains the ``.mp4`` suffix."""

    return path.with_name(f"{path.stem}.tmp{path.suffix}")


def _open_mp4_writer(
    path: Path,
    *,
    fps: float,
    frame_size: tuple[int, int],
) -> tuple[Any, Path]:
    """Open an OpenCV MP4 writer on a sibling temporary file.

    Importing OpenCV lazily keeps GIF/PNG evaluation usable when an optional
    video backend is unavailable.  The caller owns both the writer and the
    temporary path and must release/clean them on every exit path.
    """

    import cv2

    temporary_path = _temporary_mp4_path(path)
    temporary_path.unlink(missing_ok=True)
    width, height = (int(frame_size[0]), int(frame_size[1]))
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid MP4 frame size: {(width, height)}")
    writer = cv2.VideoWriter(
        str(temporary_path),
        cv2.VideoWriter_fourcc(*MP4_CODEC),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        writer.release()
        temporary_path.unlink(missing_ok=True)
        raise OSError(
            "OpenCV MP4 VideoWriter could not be opened "
            f"(codec={MP4_CODEC}, path={temporary_path})"
        )
    return writer, temporary_path


def _release_mp4_writer(writer: Any) -> str | None:
    """Release a writer without masking evaluation/other-output failures."""

    try:
        writer.release()
    except Exception:
        return traceback.format_exc()
    return None


def _inspect_gif(path: Path) -> tuple[int, list[int]]:
    """Return GIF frame count and per-frame durations in milliseconds."""

    durations: list[int] = []
    with Image.open(path) as image:
        frame_count = int(getattr(image, "n_frames", 1))
        for frame_index in range(frame_count):
            image.seek(frame_index)
            durations.append(int(image.info.get("duration", 0)))
    return frame_count, durations


def _inspect_mp4(path: Path, *, expected_fps: float) -> tuple[int, float]:
    """Read MP4 metadata and verify its frame rate using OpenCV."""

    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"生成されたMP4を開けません: {path}")
    try:
        measured_fps = float(capture.get(cv2.CAP_PROP_FPS))
        measured_frame_count = int(
            round(float(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        )
    finally:
        capture.release()
    if measured_frame_count <= 0:
        raise OSError(f"MP4にframeがありません: {path}")
    if not math.isclose(
        measured_fps,
        float(expected_fps),
        rel_tol=1e-3,
        abs_tol=1e-6,
    ):
        raise RuntimeError(
            "MP4 fps does not match runtime control_hz: "
            f"measured={measured_fps}, expected={expected_fps}"
        )
    return measured_frame_count, measured_fps


def _episode_visualization_paths(
    run_dir: Path,
    *,
    episode_number: int,
    scenario_seed: int,
) -> dict[str, Path]:
    """Return collision-free artifact paths for one evaluated episode."""

    episode_dir = run_dir / "episodes" / (
        f"episode_{episode_number:04d}_scenario_{scenario_seed:06d}"
    )
    frame_dir = episode_dir / "frames"
    return {
        "directory": episode_dir,
        "gif": episode_dir / "evaluation.gif",
        "mp4": episode_dir / "evaluation.mp4",
        "mp4_temporary": _temporary_mp4_path(episode_dir / "evaluation.mp4"),
        "frames": frame_dir,
        "gif_trace": episode_dir / "gif_error.log",
    }


def _prepare_evaluation_output_directory(run_dir: Path) -> None:
    """Prepare one run and remove episode artifacts left by an older rerun."""

    if run_dir.is_symlink():
        raise OSError(f"評価出力先にsymlinkは使用できません: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    episode_root = run_dir / "episodes"
    if episode_root.is_symlink():
        raise OSError(f"episode出力先にsymlinkは使用できません: {episode_root}")
    if episode_root.exists():
        # A shorter rerun must not leave higher-numbered episodes that look as
        # though they were produced by the current invocation.
        shutil.rmtree(episode_root)


class _EpisodeVisualizationRecorder:
    """Own all recording state and finalization for one evaluation episode."""

    def __init__(
        self,
        *,
        requested: bool,
        run_dir: Path,
        episode_number: int,
        scenario_seed: int,
        write_gif_trace: Callable[[Path, str, list[str]], None] | None = None,
    ) -> None:
        self.requested = bool(requested)
        self.episode_number = int(episode_number)
        self.scenario_seed = int(scenario_seed)
        self._write_gif_trace = write_gif_trace or _write_gif_trace
        self.paths = _episode_visualization_paths(
            run_dir,
            episode_number=self.episode_number,
            scenario_seed=self.scenario_seed,
        )

        self.gif_result: dict[str, Any] = {
            "requested": self.requested,
            "status": "not_requested",
            "output_path": (
                str(self.paths["gif"].resolve()) if self.requested else None
            ),
            "recorded_episode": self.episode_number if self.requested else None,
            "scenario_seed": self.scenario_seed if self.requested else None,
            "render_api": GIF_RENDER_API if self.requested else None,
            "generate_api": GIF_GENERATE_API if self.requested else None,
            "verified_before_env_close": False,
            "size_bytes": None,
            "frame_count": 0,
            "frame_duration_ms": None,
            "expected_playback_duration_seconds": None,
            "playback_duration_seconds": None,
            "error": None,
            "cleanup_error": None,
            "traceback_path": None,
            "traceback_write_error": None,
            "attempted_apis": [],
            "telemetry_panel": {
                "enabled": self.requested,
                "width_pixels": TELEMETRY_PANEL_WIDTH if self.requested else None,
                "action_history_seconds": (
                    ACTION_HISTORY_SECONDS if self.requested else None
                ),
                "road_value_annotation": (
                    "CURRENT SEGMENT (runtime)" if self.requested else None
                ),
            },
        }
        self.mp4_result: dict[str, Any] = {
            "requested": self.requested,
            "status": "not_requested",
            "output_path": (
                str(self.paths["mp4"].resolve()) if self.requested else None
            ),
            "recorded_episode": self.episode_number if self.requested else None,
            "scenario_seed": self.scenario_seed if self.requested else None,
            "codec": MP4_CODEC if self.requested else None,
            "fps": None,
            "measured_fps": None,
            "frame_count": 0,
            "measured_frame_count": 0,
            "expected_playback_duration_seconds": None,
            "playback_duration_seconds": None,
            "verified_before_env_close": False,
            "size_bytes": None,
            "error": None,
            "cleanup_error": None,
            "traceback_write_error": None,
        }
        self.frame_result: dict[str, Any] = {
            "requested": self.requested,
            "status": "not_requested",
            "output_path": (
                str(self.paths["frames"].resolve()) if self.requested else None
            ),
            "recorded_episode": self.episode_number if self.requested else None,
            "scenario_seed": self.scenario_seed if self.requested else None,
            "frame_count": 0,
            "error": None,
        }

        self.gif_attempted_apis: list[str] = []
        self.gif_exception_trace: str | None = None
        self.render_exception_trace: str | None = None
        self.frame_dump_exception_trace: str | None = None
        self.mp4_exception_trace: str | None = None
        self.mp4_writer: Any | None = None
        self.mp4_writer_temporary_path: Path | None = None
        self.renderer: Any | None = None
        self.mp4_frame_count = 0
        self.frame_count = 0
        self.png_frame_count = 0

    def prepare(self) -> None:
        """Prepare only this episode's files and remove its stale artifacts."""

        if not self.requested:
            return

        try:
            self.paths["directory"].mkdir(parents=True, exist_ok=True)
            self.paths["gif"].unlink(missing_ok=True)
            self.paths["gif_trace"].unlink(missing_ok=True)
            self.gif_result["status"] = "recording"
        except Exception:
            self.gif_exception_trace = traceback.format_exc()
            self.gif_result["status"] = "failed"

        try:
            self.paths["mp4"].unlink(missing_ok=True)
            self.paths["mp4_temporary"].unlink(missing_ok=True)
            self.mp4_result["status"] = "recording"
        except Exception:
            self.mp4_exception_trace = traceback.format_exc()
            self.mp4_result["status"] = "failed"

        try:
            frame_output_dir = self.paths["frames"]
            if frame_output_dir.is_symlink():
                raise OSError(
                    "フレーム出力先にsymlinkは使用できません: "
                    f"{frame_output_dir}"
                )
            frame_output_dir.mkdir(parents=True, exist_ok=True)
            for stale_frame_path in frame_output_dir.glob("frame_*.png"):
                stale_frame_path.unlink()
            for stale_temporary_path in frame_output_dir.glob("frame_*.png.tmp"):
                stale_temporary_path.unlink()
            self.frame_result["status"] = "recording"
        except Exception:
            self.frame_dump_exception_trace = traceback.format_exc()

    def record_frame(
        self,
        *,
        env: Any,
        telemetry: Mapping[str, Any],
        switch_tracker: ActionSwitchTracker,
    ) -> None:
        """Render and stream one post-step panel without aborting evaluation."""

        if not self.requested or self.render_exception_trace is not None:
            return

        try:
            if GIF_RENDER_API not in self.gif_attempted_apis:
                self.gif_attempted_apis.append(GIF_RENDER_API)
            rendered_frame = env.render(
                mode="topdown",
                screen_record=True,
                window=False,
                screen_size=(600, 600),
                camera_position=(50, 50),
            )
            renderer = env.top_down_renderer
            if renderer is None:
                raise RuntimeError("top_down_renderer was not created")
            self.renderer = renderer
            action_history = [
                decode_discrete_action(action_id, env.config)
                for action_id in switch_tracker.history
            ]
            panel_frame = compose_telemetry_panel(
                rendered_frame,
                telemetry,
                action_history,
            )
            replace_latest_recorded_frame(renderer, panel_frame)
        except Exception:
            # Rendering/composition is the common prerequisite for all three
            # artifacts, but policy evaluation continues and later episodes
            # get a fresh recorder.
            self.render_exception_trace = traceback.format_exc()
            return

        self.frame_count += 1
        panel_array = np.asarray(panel_frame, dtype=np.uint8)

        if self.frame_dump_exception_trace is None:
            try:
                frame_path = self.paths["frames"] / (
                    f"frame_{self.frame_count:06d}.png"
                )
                temporary_frame_path = frame_path.with_suffix(
                    frame_path.suffix + ".tmp"
                )
                try:
                    Image.fromarray(panel_array).save(
                        temporary_frame_path,
                        format="PNG",
                    )
                    temporary_frame_path.replace(frame_path)
                except Exception:
                    temporary_frame_path.unlink(missing_ok=True)
                    raise
                self.png_frame_count += 1
            except Exception:
                self.frame_dump_exception_trace = traceback.format_exc()

        if self.mp4_exception_trace is None:
            try:
                if self.mp4_writer is None:
                    self.mp4_writer, self.mp4_writer_temporary_path = (
                        _open_mp4_writer(
                            self.paths["mp4"],
                            fps=self._control_hz,
                            frame_size=(panel_array.shape[1], panel_array.shape[0]),
                        )
                    )
                self.mp4_writer.write(panel_array[:, :, ::-1].copy())
                self.mp4_frame_count += 1
            except Exception:
                self.mp4_exception_trace = traceback.format_exc()
                if self.mp4_writer is not None:
                    release_error = _release_mp4_writer(self.mp4_writer)
                    if release_error is not None:
                        self.mp4_exception_trace += release_error
                    self.mp4_writer = None
                if self.mp4_writer_temporary_path is not None:
                    cleanup_error = _try_unlink(self.mp4_writer_temporary_path)
                    if cleanup_error is not None:
                        self.mp4_result["cleanup_error"] = cleanup_error

    @property
    def _control_hz(self) -> float:
        """The timing value is set by ``finalize`` before the first frame."""

        if not hasattr(self, "control_hz"):
            raise RuntimeError("episode recorder timing is not initialized")
        return float(self.control_hz)

    def set_timing(self, timing: SimulationTiming) -> None:
        self.control_hz = float(timing.control_hz)
        self.gif_result["frame_duration_ms"] = timing.gif_frame_duration_ms
        self.gif_result["generate_api"] = (
            "env.top_down_renderer.generate_gif("
            f"gif_name=<output>, duration={timing.gif_frame_duration_ms})"
        )
        self.mp4_result["fps"] = timing.control_hz

    def finalize(
        self,
        *,
        env: Any,
        timing: SimulationTiming,
        expected_frame_count: int,
    ) -> dict[str, Any]:
        """Finalize all current-episode outputs before the next reset."""

        if not self.requested:
            return self.to_dict()

        self.set_timing(timing)
        if (
            self.render_exception_trace is None
            and self.frame_count != int(expected_frame_count)
        ):
            self.render_exception_trace = (
                "rendered post-step frame count does not match the recorded "
                f"episode length: rendered={self.frame_count}, "
                f"expected={expected_frame_count}"
            )
        if self.render_exception_trace is not None:
            if self.gif_exception_trace is None:
                self.gif_exception_trace = self.render_exception_trace
            if self.mp4_exception_trace is None:
                self.mp4_exception_trace = self.render_exception_trace
            if self.frame_dump_exception_trace is None:
                self.frame_dump_exception_trace = self.render_exception_trace

        self._finalize_gif(env=env, timing=timing)
        self._finalize_mp4(timing=timing)
        self.frame_result["frame_count"] = self.png_frame_count
        if self.frame_dump_exception_trace is not None:
            self.frame_result.update(
                {
                    "status": "failed",
                    "error": self.frame_dump_exception_trace.strip().splitlines()[-1],
                }
            )
            print(
                "frame_dump_failed=True error="
                f"{self.frame_result['error']}",
                file=sys.stderr,
            )
        else:
            self.frame_result["status"] = "success"
            print(
                f"frames_saved={self.paths['frames'].resolve()} "
                f"count={self.png_frame_count}"
            )
        return self.to_dict()

    def _finalize_gif(self, *, env: Any, timing: SimulationTiming) -> None:
        if self.gif_exception_trace is None:
            try:
                renderer = self.renderer or env.top_down_renderer
                if renderer is None:
                    raise RuntimeError("top_down_renderer was not created")
                if self.frame_count <= 0:
                    raise RuntimeError("no post-step frames were rendered")
                gif_generate_api = (
                    "env.top_down_renderer.generate_gif("
                    f"gif_name=<output>, duration={timing.gif_frame_duration_ms})"
                )
                self.gif_attempted_apis.append(gif_generate_api)
                renderer.generate_gif(
                    gif_name=str(self.paths["gif"]),
                    duration=timing.gif_frame_duration_ms,
                )
                if not self.paths["gif"].is_file():
                    raise FileNotFoundError(
                        f"GIFが生成されませんでした: {self.paths['gif']}"
                    )
                gif_size = self.paths["gif"].stat().st_size
                if gif_size <= 0:
                    raise OSError(f"生成されたGIFが空です: {self.paths['gif']}")
                gif_frame_count, gif_durations = _inspect_gif(self.paths["gif"])
                if gif_frame_count != self.frame_count:
                    raise RuntimeError(
                        "GIF frame count does not match rendered post-step frames: "
                        f"gif={gif_frame_count}, rendered={self.frame_count}"
                    )
                if any(
                    duration != timing.gif_frame_duration_ms
                    for duration in gif_durations
                ):
                    raise RuntimeError(
                        "GIF frame duration does not match runtime action dt: "
                        f"durations={sorted(set(gif_durations))}, "
                        f"expected={timing.gif_frame_duration_ms}ms"
                    )
                expected_playback = (
                    self.frame_count * timing.action_duration_seconds
                )
                playback_duration = sum(gif_durations) / 1000.0
                if not math.isclose(
                    playback_duration,
                    expected_playback,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ):
                    raise RuntimeError(
                        "GIF playback duration does not match simulation time: "
                        f"playback={playback_duration}, simulation={expected_playback}"
                    )
                self.gif_result.update(
                    {
                        "status": "success",
                        "verified_before_env_close": True,
                        "size_bytes": gif_size,
                        "frame_count": gif_frame_count,
                        "expected_playback_duration_seconds": expected_playback,
                        "playback_duration_seconds": playback_duration,
                    }
                )
            except Exception:
                self.gif_exception_trace = traceback.format_exc()

        if self.gif_exception_trace is not None:
            gif_cleanup_error = None
            try:
                self.paths["gif"].unlink(missing_ok=True)
            except Exception:
                gif_cleanup_error = traceback.format_exc().strip()
            traceback_path, traceback_write_error = _try_write_gif_trace(
                self.paths["gif_trace"],
                self.gif_exception_trace,
                self.gif_attempted_apis,
                write_gif_trace=self._write_gif_trace,
            )
            self.gif_result.update(
                {
                    "status": "failed",
                    "error": self.gif_exception_trace.strip().splitlines()[-1],
                    "cleanup_error": gif_cleanup_error,
                    "traceback_path": traceback_path,
                    "traceback_write_error": (
                        None
                        if traceback_write_error is None
                        else traceback_write_error.strip()
                    ),
                    "attempted_apis": self.gif_attempted_apis,
                }
            )
            if traceback_write_error is not None:
                print(
                    "gif_trace_write_failed=True error="
                    f"{traceback_write_error.strip().splitlines()[-1]}",
                    file=sys.stderr,
                )
            print(
                "gif_failed=True traceback="
                f"{traceback_path or 'unavailable'}",
                file=sys.stderr,
            )
        else:
            self.gif_result["attempted_apis"] = self.gif_attempted_apis
            print(
                f"gif_saved={self.paths['gif'].resolve()} "
                f"size_bytes={self.gif_result['size_bytes']} "
                f"duration={self.gif_result['playback_duration_seconds']:.3f}s"
            )

    def _finalize_mp4(self, *, timing: SimulationTiming) -> None:
        if self.mp4_exception_trace is None:
            try:
                if (
                    self.mp4_writer is None
                    or self.mp4_writer_temporary_path is None
                ):
                    raise RuntimeError("no MP4 writer was opened")
                release_error = _release_mp4_writer(self.mp4_writer)
                self.mp4_writer = None
                if release_error is not None:
                    raise OSError(
                        "OpenCV MP4 VideoWriter release failed:\n"
                        f"{release_error}"
                    )
                temporary_path = self.mp4_writer_temporary_path
                if not temporary_path.is_file():
                    raise FileNotFoundError(
                        f"MP4が生成されませんでした: {temporary_path}"
                    )
                mp4_size = temporary_path.stat().st_size
                if mp4_size <= 0:
                    raise OSError(f"生成されたMP4が空です: {temporary_path}")
                temporary_path.replace(self.paths["mp4"])
                measured_count, measured_fps = _inspect_mp4(
                    self.paths["mp4"], expected_fps=timing.control_hz
                )
                if measured_count != self.mp4_frame_count:
                    raise RuntimeError(
                        "MP4 frame count does not match written panel frames: "
                        f"mp4={measured_count}, written={self.mp4_frame_count}"
                    )
                expected_playback = (
                    self.mp4_frame_count * timing.action_duration_seconds
                )
                playback_duration = measured_count / measured_fps
                if not math.isclose(
                    playback_duration,
                    expected_playback,
                    rel_tol=1e-3,
                    abs_tol=1e-6,
                ):
                    raise RuntimeError(
                        "MP4 playback duration does not match simulation time: "
                        f"playback={playback_duration}, simulation={expected_playback}"
                    )
                self.mp4_result.update(
                    {
                        "status": "success",
                        "verified_before_env_close": True,
                        "size_bytes": mp4_size,
                        "frame_count": measured_count,
                        "measured_frame_count": measured_count,
                        "measured_fps": measured_fps,
                        "expected_playback_duration_seconds": expected_playback,
                        "playback_duration_seconds": playback_duration,
                    }
                )
            except Exception:
                self.mp4_exception_trace = traceback.format_exc()

        if self.mp4_exception_trace is not None:
            cleanup_errors: list[str] = []
            if self.mp4_writer is not None:
                release_error = _release_mp4_writer(self.mp4_writer)
                self.mp4_writer = None
                if release_error is not None:
                    self.mp4_exception_trace += release_error
            if self.mp4_writer_temporary_path is not None:
                cleanup_error = _try_unlink(self.mp4_writer_temporary_path)
                if cleanup_error is not None:
                    cleanup_errors.append(cleanup_error)
            cleanup_error = _try_unlink(self.paths["mp4"])
            if cleanup_error is not None:
                cleanup_errors.append(cleanup_error)
            previous_cleanup_error = self.mp4_result.get("cleanup_error")
            all_cleanup_errors = (
                ([str(previous_cleanup_error)] if previous_cleanup_error else [])
                + cleanup_errors
            )
            self.mp4_result.update(
                {
                    "status": "failed",
                    "error": self.mp4_exception_trace.strip().splitlines()[-1],
                    "frame_count": self.mp4_frame_count,
                    "cleanup_error": "\n".join(all_cleanup_errors) or None,
                }
            )
            print(
                "mp4_failed=True error="
                f"{self.mp4_result['error']}",
                file=sys.stderr,
            )
        else:
            print(
                f"mp4_saved={self.paths['mp4'].resolve()} "
                f"size_bytes={self.mp4_result['size_bytes']} "
                f"duration={self.mp4_result['playback_duration_seconds']:.3f}s"
            )

    def cleanup(self) -> None:
        """Release a live writer and remove only this episode's temporary file."""

        cleanup_errors: list[str] = []
        if self.mp4_writer is not None:
            release_error = _release_mp4_writer(self.mp4_writer)
            self.mp4_writer = None
            if release_error is not None:
                cleanup_errors.append(release_error.strip())
        if self.mp4_writer_temporary_path is not None:
            cleanup_error = _try_unlink(self.mp4_writer_temporary_path)
            if cleanup_error is not None:
                cleanup_errors.append(cleanup_error)
        if cleanup_errors:
            previous_error = self.mp4_result.get("cleanup_error")
            errors = ([str(previous_error)] if previous_error else []) + cleanup_errors
            self.mp4_result["cleanup_error"] = "\n".join(errors)
            print(
                "mp4_cleanup_failed=True error="
                f"{cleanup_errors[-1].splitlines()[-1]}",
                file=sys.stderr,
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode": self.episode_number,
            "scenario_seed": self.scenario_seed,
            "requested": self.requested,
            "artifact_directory": (
                str(self.paths["directory"].resolve()) if self.requested else None
            ),
            "gif": self.gif_result,
            "mp4": self.mp4_result,
            "frames": self.frame_result,
        }
