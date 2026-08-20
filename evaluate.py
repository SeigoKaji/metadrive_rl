"""保存済みPPOを選択profileのシナリオで評価し、結果を保存する。"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import math
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, TextIO

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.utils import check_for_correct_spaces

from env_factory import make_evaluation_env
from evaluation_visualization import (
    ACTION_HISTORY_SECONDS,
    STEP_TELEMETRY_FIELDS,
    ActionSwitchTracker,
    SimulationTiming,
    decode_discrete_action,
    derive_timing,
    make_step_telemetry,
    read_runtime_road_metrics,
)
from configs.experiment_profiles import PROFILE_NAMES, get_experiment_profile
from configs.phase0_config import (
    LOG_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
    PROJECT_ROOT,
)

# Keep the former evaluate-module helper surface while the implementations
# live with the result/artifact lifecycle they own.
from evaluation_results import (
    GIF_GENERATE_API,
    GIF_RENDER_API,
    MP4_CODEC,
    _EpisodeVisualizationRecorder,
    _episode_visualization_paths,
    _inspect_gif,
    _inspect_mp4,
    _json_value,
    _optional_info_value,
    _open_mp4_writer,
    _prepare_evaluation_output_directory,
    _release_mp4_writer,
    _temporary_mp4_path,
    _termination_reason,
    _try_unlink,
    _try_write_gif_trace,
    _write_gif_trace,
    _write_json,
    _write_jsonl_record,
)


class _Tee:
    """Write evaluation output to both the terminal and a persistent log."""

    def __init__(self, terminal: TextIO, log_file: TextIO) -> None:
        self._terminal = terminal
        self._log_file = log_file

    def write(self, text: str) -> int:
        terminal_count = self._terminal.write(text)
        self._log_file.write(text)
        return terminal_count

    def flush(self) -> None:
        self._terminal.flush()
        self._log_file.flush()

    def isatty(self) -> bool:
        return self._terminal.isatty()

    @property
    def encoding(self) -> str | None:
        return self._terminal.encoding


def _positive_int(value: str) -> int:
    """Parse a strictly positive CLI integer."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("0より大きい整数を指定してください")
    return parsed


def _output_prefix(value: str) -> str:
    """Accept a basename only so generated artifacts remain under outputs/."""

    if not value or Path(value).name != value or value in {".", ".."}:
        raise argparse.ArgumentTypeError("--output-prefixはディレクトリを含まない名前にしてください")
    return value


def _resolve_project_path(path: Path) -> Path:
    """Resolve CLI paths consistently even when launched outside the project."""

    return path if path.is_absolute() else PROJECT_ROOT / path


def _default_evaluation_log(output_prefix: str) -> Path:
    """Use the requested official/smoke names and a predictable custom fallback."""

    suffix = output_prefix.removeprefix("phase0_")
    return LOG_DIR / f"evaluate_{suffix}.log"


def _resolve_log_path(path: Path | None, output_prefix: str) -> Path:
    """Resolve an optional log path relative to this project."""

    if path is None:
        return _default_evaluation_log(output_prefix)
    return _resolve_project_path(path)


def _sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for the exact evaluated model."""

    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while chunk := file_obj.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluation_output_directory(profile_name: str, output_prefix: str) -> Path:
    """Return the run directory shared by evaluation JSON and artifacts."""

    return OUTPUT_DIR / profile_name / "evaluation" / output_prefix


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse deterministic evaluation options for the selected profile."""

    profile_parser = argparse.ArgumentParser(add_help=False)
    profile_parser.add_argument(
        "--profile",
        choices=PROFILE_NAMES,
        default="official",
    )
    selected, _unknown = profile_parser.parse_known_args(argv)
    profile = get_experiment_profile(selected.profile)

    parser = argparse.ArgumentParser(
        description="MetaDriveの保存済みPPOを評価",
    )
    parser.add_argument(
        "--profile",
        choices=PROFILE_NAMES,
        default=selected.profile,
        help="評価環境profile（既定: official）",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_DIR / f"{profile.default_model_name}.zip",
        help="PPO .zipモデル（相対パスはproject直下基準）",
    )
    parser.add_argument(
        "--episodes",
        type=_positive_int,
        default=profile.evaluation_episodes,
        help="評価episode数",
    )
    parser.add_argument(
        "--record-gif",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "全評価episodeをtop-down GIF/MP4とフレーム別PNGで記録"
            "（既定: 有効、--no-record-gifで全て無効）"
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=_output_prefix,
        default=profile.default_model_name,
        help="outputs/とlogs/で使うベース名",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(profile.training_config["seed"]),
        help="評価過程のRL乱数seed（scenario seed範囲とは別）",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="PPO.load()に指定するdevice",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="標準出力/標準エラーの複製先（相対パスはPhase 0直下基準）",
    )
    return parser.parse_args(argv)


def _evaluate(args: argparse.Namespace, log_path: Path) -> Path:
    """Run deterministic evaluation and optionally record every episode."""

    profile = get_experiment_profile(args.profile)
    environment_config = profile.evaluation_env_config
    scenario_start = int(environment_config["start_seed"])
    scenario_count = int(environment_config["num_scenarios"])
    if scenario_count > 1 and args.episodes > scenario_count:
        raise ValueError(
            "評価episode数はprofileのscenario数以下にしてください: "
            f"episodes={args.episodes}, num_scenarios={scenario_count}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    model_path = _resolve_project_path(args.model).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"評価モデルが見つかりません: {model_path}")
    model_size = model_path.stat().st_size
    if model_size <= 0:
        raise OSError(f"評価モデルが空です: {model_path}")
    model_sha256 = _sha256_file(model_path)

    set_random_seed(args.seed)
    model = PPO.load(str(model_path), device=args.device)
    actual_device = str(model.device)
    run_dir = _evaluation_output_directory(args.profile, args.output_prefix)
    _prepare_evaluation_output_directory(run_dir)
    result_path = run_dir / "evaluation.json"
    step_trace_path = run_dir / "evaluation_steps.jsonl"
    step_trace_temporary_path = step_trace_path.with_suffix(
        step_trace_path.suffix + ".tmp"
    )
    step_trace_result: dict[str, Any] = {
        "status": "writing",
        "path": str(step_trace_path.resolve()),
        "format": "jsonl",
        "record_semantics": "post-step; one row per policy decision",
        "switch_rate_basis": (
            "cumulative action-ID changes / elapsed simulation seconds"
        ),
        "row_count": 0,
        "fields": list(STEP_TELEMETRY_FIELDS),
    }

    env = None
    active_recorder: _EpisodeVisualizationRecorder | None = None
    step_trace_file: TextIO | None = None
    step_trace_row_count = 0
    simulation_timing: SimulationTiming | None = None
    episodes: list[dict[str, Any]] = []
    evaluation_started_at = datetime.now(timezone.utc)
    evaluation_start_time = time.perf_counter()
    try:
        # A sibling temporary keeps a previous successful trace intact if this
        # run aborts. Each row is still flushed immediately during evaluation.
        step_trace_temporary_path.unlink(missing_ok=True)
        step_trace_file = step_trace_temporary_path.open(
            "w", encoding="utf-8", buffering=1
        )
        try:
            env = make_evaluation_env(
                seed=args.seed,
                record_gif=args.record_gif,
                env_config=environment_config,
            )
            check_for_correct_spaces(env, model.observation_space, model.action_space)
            simulation_timing = derive_timing(
                env.config,
            )
            action_history_length = max(
                1,
                math.ceil(ACTION_HISTORY_SECONDS * simulation_timing.control_hz),
            )
            horizon_value = env.config.get("horizon")
            horizon = None if horizon_value is None else int(horizon_value)

            print(
                "simulation_timing="
                f"physics_hz={simulation_timing.physics_hz:.3f} "
                f"control_hz={simulation_timing.control_hz:.3f} "
                f"action_dt={simulation_timing.action_duration_seconds:.6f}s "
                f"gif_duration={simulation_timing.gif_frame_duration_ms}ms "
                f"mp4_fps={simulation_timing.control_hz:.3f}"
            )

            for episode_number in range(1, args.episodes + 1):
                episode_start_time = time.perf_counter()
                # MetaDriveのseedはRL乱数ではなくscenario indexである。複数scenario
                # profileでは先頭から一度ずつ走査し、ランダム抽選による重複を避ける。
                scenario_seed = (
                    scenario_start
                    if scenario_count == 1
                    else scenario_start + episode_number - 1
                )
                obs, reset_info = env.reset(seed=scenario_seed)
                actual_scenario_seed = int(env.current_seed)
                if actual_scenario_seed != scenario_seed:
                    raise RuntimeError(
                        "要求したscenarioと実際のscenarioが一致しません: "
                        f"requested={scenario_seed}, actual={actual_scenario_seed}"
                    )
                total_reward = 0.0
                episode_length = 0
                terminated_flag = False
                truncated_flag = False
                final_info: Mapping[str, Any] = {}
                speed_samples_m_s: list[float] = []
                switch_tracker = ActionSwitchTracker(
                    history_length=action_history_length
                )
                active_recorder = _EpisodeVisualizationRecorder(
                    requested=args.record_gif,
                    run_dir=run_dir,
                    episode_number=episode_number,
                    scenario_seed=actual_scenario_seed,
                    write_gif_trace=_write_gif_trace,
                )
                active_recorder.prepare()
                if simulation_timing is not None:
                    active_recorder.set_timing(simulation_timing)

                while True:
                    action, _state = model.predict(obs, deterministic=True)
                    decoded_action = decode_discrete_action(action, env.config)
                    obs, reward, terminated, truncated, info = env.step(action)
                    step_reward = float(reward)
                    total_reward += step_reward
                    episode_length += 1
                    terminated_flag = bool(terminated)
                    truncated_flag = bool(truncated)
                    final_info = info

                    sim_time_seconds = (
                        episode_length * simulation_timing.action_duration_seconds
                    )
                    switch_count, switches_per_second = switch_tracker.observe(
                        decoded_action.action_id,
                        sim_time_seconds=sim_time_seconds,
                    )
                    telemetry = make_step_telemetry(
                        episode=episode_number,
                        scenario_seed=actual_scenario_seed,
                        step=episode_length,
                        horizon=horizon,
                        timing=simulation_timing,
                        decoded_action=decoded_action,
                        info=info,
                        reward=step_reward,
                        cumulative_reward=total_reward,
                        terminated=terminated_flag,
                        truncated=truncated_flag,
                        road=read_runtime_road_metrics(env.agent),
                        action_switch_count=switch_count,
                        action_switches_per_second=switches_per_second,
                    )
                    _write_jsonl_record(step_trace_file, telemetry)
                    step_trace_row_count += 1
                    speed_samples_m_s.append(float(telemetry["speed_m_s"]))

                    active_recorder.record_frame(
                        env=env,
                        telemetry=telemetry,
                        switch_tracker=switch_tracker,
                    )

                    if terminated_flag or truncated_flag:
                        break

                flags = {
                    key: _optional_info_value(final_info, key)
                    for key in (
                        "arrive_dest",
                        "out_of_road",
                        "crash",
                        "crash_vehicle",
                        "crash_object",
                        "max_step",
                        "route_completion",
                    )
                }
                episode_sim_time = (
                    episode_length * simulation_timing.action_duration_seconds
                )
                mean_speed_m_s = statistics.fmean(speed_samples_m_s)
                max_speed_m_s = max(speed_samples_m_s)
                episode_result = {
                    "episode": episode_number,
                    "scenario_seed": actual_scenario_seed,
                    "total_reward": total_reward,
                    "episode_length": episode_length,
                    "simulation_seconds": episode_sim_time,
                    "mean_speed_m_s": mean_speed_m_s,
                    "mean_speed_km_h": mean_speed_m_s * 3.6,
                    "max_speed_m_s": max_speed_m_s,
                    "max_speed_km_h": max_speed_m_s * 3.6,
                    "action_switch_count": switch_tracker.switch_count,
                    "action_switches_per_second": (
                        switch_tracker.switch_count / episode_sim_time
                    ),
                    "terminated": terminated_flag,
                    "truncated": truncated_flag,
                    **flags,
                    "termination_reason": _termination_reason(
                        terminated=terminated_flag,
                        truncated=truncated_flag,
                        flags=flags,
                    ),
                    "reset_info_keys": sorted(
                        str(key) for key in reset_info.keys()
                    ),
                    "final_info_keys": sorted(
                        str(key) for key in final_info.keys()
                    ),
                    "execution_seconds": time.perf_counter() - episode_start_time,
                }
                episode_result["visualization"] = active_recorder.finalize(
                    env=env,
                    timing=simulation_timing,
                    expected_frame_count=episode_length,
                )
                active_recorder.cleanup()
                active_recorder = None
                episodes.append(episode_result)
                print(
                    f"episode={episode_number} scenario_seed={actual_scenario_seed} "
                    f"reward={total_reward:.6f} "
                    f"length={episode_length} terminated={terminated_flag} "
                    f"truncated={truncated_flag} "
                    f"reason={episode_result['termination_reason']} "
                    f"switches={switch_tracker.switch_count} "
                    f"switches_per_second="
                    f"{episode_result['action_switches_per_second']:.3f}"
                )

        finally:
            try:
                if step_trace_file is not None:
                    step_trace_file.close()
            finally:
                try:
                    if env is not None:
                        env.close()
                        print("evaluation_env_closed=True")
                finally:
                    # An evaluation exception can bypass normal artifact
                    # finalization; never leave a live writer or .tmp.mp4.
                    if active_recorder is not None:
                        active_recorder.cleanup()
    except BaseException:
        step_trace_temporary_path.unlink(missing_ok=True)
        if active_recorder is not None:
            active_recorder.cleanup()
        raise

    step_trace_temporary_path.replace(step_trace_path)
    step_trace_result.update(status="success", row_count=step_trace_row_count)
    print(
        f"step_telemetry_saved={step_trace_path.resolve()} "
        f"rows={step_trace_row_count}"
    )

    rewards = [float(episode["total_reward"]) for episode in episodes]
    lengths = [int(episode["episode_length"]) for episode in episodes]
    success_count = sum(
        episode["termination_reason"] == "success" for episode in episodes
    )
    out_of_road_count = sum(
        episode["termination_reason"] == "out_of_road" for episode in episodes
    )
    if simulation_timing is None:
        raise AssertionError("successful evaluation has no simulation timing")
    visualizations = [episode["visualization"] for episode in episodes]
    final_visualization = visualizations[-1]
    evaluation_finished_at = datetime.now(timezone.utc)
    result = {
        "evaluation_status": "success",
        "started_at_utc": evaluation_started_at.isoformat(),
        "finished_at_utc": evaluation_finished_at.isoformat(),
        "execution_seconds": time.perf_counter() - evaluation_start_time,
        "command": [sys.executable, *sys.argv],
        "model": {
            "path": str(model_path),
            "size_bytes": model_size,
            "sha256": model_sha256,
            "requested_device": args.device,
            "actual_device": actual_device,
        },
        "profile": args.profile,
        "output_directory": str(run_dir.resolve()),
        "environment_config": dict(environment_config),
        "scenario_seed_range": {
            "start": scenario_start,
            "stop_exclusive": scenario_start + scenario_count,
        },
        "rl_seed": args.seed,
        "deterministic": True,
        "simulation_timing": simulation_timing.to_dict(),
        "episode_count": args.episodes,
        "episodes": episodes,
        "aggregate": {
            "mean_reward": statistics.fmean(rewards),
            "min_reward": min(rewards),
            "max_reward": max(rewards),
            "mean_episode_length": statistics.fmean(lengths),
            "success_count": success_count,
            "success_rate": success_count / len(episodes),
            "out_of_road_count": out_of_road_count,
            "out_of_road_rate": out_of_road_count / len(episodes),
        },
        # Recording failures are deliberately independent of successful policy
        # evaluation and of each other where they do not share rendering.
        "visualizations": visualizations,
        # Keep the historical scalar fields as aliases to the final episode.
        "gif": final_visualization["gif"],
        "mp4": final_visualization["mp4"],
        "frames": final_visualization["frames"],
        "step_telemetry": step_trace_result,
        "console_log_path": str(log_path.resolve()),
    }
    if scenario_count == 1:
        # Keep the Phase 0 scalar field for existing result consumers.
        result["scenario_seed"] = scenario_start
    _write_json(result_path, result)
    print(f"evaluation_saved={result_path.resolve()}")
    return result_path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point with persistent stdout/stderr capture."""

    args = parse_args(argv)
    log_path = _resolve_log_path(args.log_file, args.output_prefix)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        tee_stdout = _Tee(sys.stdout, log_file)
        tee_stderr = _Tee(sys.stderr, log_file)
        with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(tee_stderr):
            print(f"console_log={log_path.resolve()}")
            try:
                _evaluate(args, log_path)
            except BaseException:
                print("evaluation_failed=True", file=sys.stderr)
                traceback.print_exc()
                raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
