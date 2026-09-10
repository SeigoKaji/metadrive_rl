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
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping, TextIO

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.utils import check_for_correct_spaces

from env_factory import make_evaluation_env
from lookahead_learning.checkpoint import validate_lookahead_model_metadata
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
from configs.experiment_config import (
    ExperimentConfigError,
    PROFILE_NAMES,
    experiment_selection_from_args,
    select_experiment,
)
from project_paths import (
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


def _output_prefix(value: str) -> str:
    """Accept a basename only so generated artifacts remain under outputs/."""

    if (
        not value
        or value in {".", ".."}
        or "\x00" in value
        or "/" in value
        or "\\" in value
        or Path(value).name != value
        or PureWindowsPath(value).name != value
    ):
        raise argparse.ArgumentTypeError("--output-prefixはディレクトリを含まない名前にしてください")
    return value


def _resolve_project_path(path: Path) -> Path:
    """Resolve CLI paths consistently even when launched outside the project."""

    return path if path.is_absolute() else PROJECT_ROOT / path


def _default_evaluation_log(output_prefix: str) -> Path:
    """Use the canonical official log name and a predictable custom fallback."""

    if output_prefix == "official_baseline":
        return LOG_DIR / "evaluate_official.log"
    return LOG_DIR / f"evaluate_{output_prefix}.log"


def _resolve_log_path(path: Path | None, output_prefix: str) -> Path:
    """Resolve an optional log path relative to this project."""

    if path is None:
        return _default_evaluation_log(output_prefix)
    return _resolve_project_path(path)


def _host_env(env: Any) -> Any:
    """Return the raw host below an optional Gymnasium lookahead wrapper."""

    return getattr(env, "unwrapped", env)


def _sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for the exact evaluated model."""

    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while chunk := file_obj.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


_FINAL_TARGET_LANE_FIELDS: tuple[str, ...] = (
    "target_lane_valid",
    "ever_departed_target_lane",
    "lane_departure_count",
    "off_target_duration_seconds",
    "time_in_target_lane_ratio",
)


def _final_target_lane_metrics(
    final_info: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return final target-lane metrics only when this env emitted them.

    Legacy/official environments never expose ``target_lane_valid``.  Keeping
    the nested result absent in that case avoids presenting a fabricated zero
    as a lane-keeping measurement.
    """

    if not any(key in final_info for key in _FINAL_TARGET_LANE_FIELDS):
        return None
    return {
        key: _optional_info_value(final_info, key)
        for key in _FINAL_TARGET_LANE_FIELDS
    }


def _finite_target_lane_value(
    metrics: Mapping[str, Any],
    key: str,
) -> float | None:
    """Read an optional finite numeric metric from JSON-safe episode data."""

    value = metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _mean_target_lane_metric(
    metrics: list[Mapping[str, Any]],
    key: str,
) -> float | None:
    values = [
        value
        for metric in metrics
        if (value := _finite_target_lane_value(metric, key)) is not None
    ]
    return statistics.fmean(values) if values else None


def _aggregate_target_lane_metrics(
    episodes: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate only episodes that supplied a target-lane objective metric."""

    tracked = [
        metric
        for episode in episodes
        if isinstance((metric := episode.get("target_lane")), Mapping)
        and "target_lane_valid" in metric
    ]
    if not tracked:
        return {
            "status": "not_available",
            "tracked_episode_count": 0,
            "tracked_episode_rate": None,
            "valid_episode_count": 0,
            "valid_episode_rate_among_tracked": None,
            "ever_departed_episode_count": None,
            "ever_departed_episode_rate_among_valid": None,
            "mean_lane_departure_count_among_valid": None,
            "mean_off_target_duration_seconds_among_valid": None,
            "mean_time_in_target_lane_ratio_among_valid": None,
        }

    valid = [metric for metric in tracked if metric.get("target_lane_valid") is True]
    if not valid:
        return {
            "status": "available_no_valid_final_target_lane",
            "tracked_episode_count": len(tracked),
            "tracked_episode_rate": len(tracked) / len(episodes),
            "valid_episode_count": 0,
            "valid_episode_rate_among_tracked": 0.0,
            "ever_departed_episode_count": None,
            "ever_departed_episode_rate_among_valid": None,
            "mean_lane_departure_count_among_valid": None,
            "mean_off_target_duration_seconds_among_valid": None,
            "mean_time_in_target_lane_ratio_among_valid": None,
        }

    ever_departed_count = sum(
        metric.get("ever_departed_target_lane") is True for metric in valid
    )
    return {
        "status": "available",
        "tracked_episode_count": len(tracked),
        "tracked_episode_rate": len(tracked) / len(episodes),
        "valid_episode_count": len(valid),
        "valid_episode_rate_among_tracked": len(valid) / len(tracked),
        "ever_departed_episode_count": ever_departed_count,
        "ever_departed_episode_rate_among_valid": ever_departed_count / len(valid),
        "mean_lane_departure_count_among_valid": _mean_target_lane_metric(
            valid,
            "lane_departure_count",
        ),
        "mean_off_target_duration_seconds_among_valid": _mean_target_lane_metric(
            valid,
            "off_target_duration_seconds",
        ),
        "mean_time_in_target_lane_ratio_among_valid": _mean_target_lane_metric(
            valid,
            "time_in_target_lane_ratio",
        ),
    }


def _evaluation_output_directory(profile_name: str, output_prefix: str) -> Path:
    """Return the run directory shared by evaluation JSON and artifacts."""

    return OUTPUT_DIR / profile_name / "evaluation" / output_prefix


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """canonical TOML aliasまたは外部bundleを評価CLI既定値としてparseする。"""

    profile_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selection_group = profile_parser.add_mutually_exclusive_group()
    selection_group.add_argument(
        "--profile",
        choices=PROFILE_NAMES,
        default=None,
    )
    selection_group.add_argument("--config", type=Path, default=None)
    selected, _unknown = profile_parser.parse_known_args(argv)
    try:
        experiment = select_experiment(
            profile_name=selected.profile,
            config_path=selected.config,
        )
    except ExperimentConfigError as error:
        profile_parser.error(str(error))
    profile = experiment.profile
    evaluation_defaults = profile.evaluation_defaults

    parser = argparse.ArgumentParser(
        description="MetaDriveの保存済みPPOを評価",
        allow_abbrev=False,
    )
    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument(
        "--profile",
        choices=PROFILE_NAMES,
        default=selected.profile or "official",
        help="canonical TOMLへの互換alias（既定: official.toml）",
    )
    selection_group.add_argument(
        "--config",
        type=Path,
        default=selected.config,
        help="実験bundle TOML（相対パスはproject直下基準）",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            str(
                evaluation_defaults.get(
                    "model_path",
                    MODEL_DIR / f"{profile.default_model_name}.zip",
                )
            )
        ),
        help="PPO .zipモデル（相対パスはproject直下基準）",
    )
    parser.add_argument(
        "--record-gif",
        action=argparse.BooleanOptionalAction,
        default=bool(evaluation_defaults.get("record_gif", True)),
        help=(
            "全評価episodeをtop-down GIF/MP4とフレーム別PNGで記録"
            "（既定: 有効、--no-record-gifで全て無効）"
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=_output_prefix,
        default=str(evaluation_defaults.get("output_prefix", profile.default_model_name)),
        help="outputs/とlogs/で使うベース名",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(
            evaluation_defaults.get("seed", profile.training_config["seed"])
        ),
        help="評価過程のRL乱数seed（scenario seed範囲とは別）",
    )
    parser.add_argument(
        "--device",
        default=str(evaluation_defaults.get("device", "cpu")),
        help="PPO.load()に指定するdevice",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=(
            None
            if evaluation_defaults.get("log_file") is None
            else Path(str(evaluation_defaults["log_file"]))
        ),
        help="標準出力/標準エラーの複製先（相対pathはproject直下基準）",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=bool(evaluation_defaults.get("deterministic", True)),
        help="PPOの決定論的action選択（既定: config値、--no-deterministicで無効）",
    )
    args = parser.parse_args(argv)
    args.profile = experiment.name
    args.config = experiment.source_path
    args.experiment = experiment
    args.experiment_selection = experiment
    return args


def _evaluate(args: argparse.Namespace, log_path: Path) -> Path:
    """設定されたaction選択で評価し、必要なら全episodeを記録する。"""

    experiment = experiment_selection_from_args(args)
    profile = experiment.profile
    environment_config = profile.evaluation_env_config
    lookahead_config = profile.lookahead_config
    record_gif = bool(getattr(args, "record_gif", True))
    deterministic = bool(getattr(args, "deterministic", True))
    scenario_start = int(environment_config["start_seed"])
    scenario_count = int(environment_config["num_scenarios"])

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
    validate_lookahead_model_metadata(model, lookahead_config)
    actual_device = str(model.device)
    run_dir = _evaluation_output_directory(experiment.name, args.output_prefix)
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
                env_config=environment_config,
                lookahead_config=lookahead_config,
            )
            host_env = _host_env(env)
            check_for_correct_spaces(env, model.observation_space, model.action_space)
            simulation_timing = derive_timing(
                host_env.config,
            )
            action_history_length = max(
                1,
                math.ceil(ACTION_HISTORY_SECONDS * simulation_timing.control_hz),
            )
            horizon_value = host_env.config.get("horizon")
            horizon = None if horizon_value is None else int(horizon_value)

            print(
                "simulation_timing="
                f"physics_hz={simulation_timing.physics_hz:.3f} "
                f"control_hz={simulation_timing.control_hz:.3f} "
                f"action_dt={simulation_timing.action_duration_seconds:.6f}s "
                f"gif_duration={simulation_timing.gif_frame_duration_ms}ms "
                f"mp4_fps={simulation_timing.control_hz:.3f}"
            )

            for episode_number in range(1, scenario_count + 1):
                episode_start_time = time.perf_counter()
                # MetaDriveのseedはRL乱数ではなくscenario indexである。設定された
                # scenario範囲を先頭から一度ずつ走査し、ランダム抽選による重複を避ける。
                scenario_seed = scenario_start + episode_number - 1
                obs, reset_info = env.reset(seed=scenario_seed)
                actual_scenario_seed = int(host_env.current_seed)
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
                    requested=record_gif,
                    run_dir=run_dir,
                    episode_number=episode_number,
                    scenario_seed=actual_scenario_seed,
                    write_gif_trace=_write_gif_trace,
                )
                active_recorder.prepare()
                if simulation_timing is not None:
                    active_recorder.set_timing(simulation_timing)

                while True:
                    action, _state = model.predict(obs, deterministic=deterministic)
                    decoded_action = decode_discrete_action(action, host_env.config)
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
                        road=read_runtime_road_metrics(host_env.agent),
                        action_switch_count=switch_count,
                        action_switches_per_second=switches_per_second,
                    )
                    _write_jsonl_record(step_trace_file, telemetry)
                    step_trace_row_count += 1
                    speed_samples_m_s.append(float(telemetry["speed_m_s"]))

                    active_recorder.record_frame(
                        env=host_env,
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
                        "wrong_lane_arrival",
                        "start_lane_departure",
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
                target_lane_metrics = _final_target_lane_metrics(final_info)
                if target_lane_metrics is not None:
                    episode_result["target_lane"] = target_lane_metrics
                episode_result["visualization"] = active_recorder.finalize(
                    env=host_env,
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
    wrong_lane_arrival_count = sum(
        episode["termination_reason"] == "wrong_lane_arrival"
        for episode in episodes
    )
    start_lane_departure_count = sum(
        episode["termination_reason"] == "start_lane_departure"
        for episode in episodes
    )
    target_lane_aggregate = _aggregate_target_lane_metrics(episodes)
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
        "profile": experiment.name,
        "config_source": experiment.source_metadata(),
        "output_directory": str(run_dir.resolve()),
        "environment_config": dict(environment_config),
        "lookahead": (
            None if lookahead_config is None else dict(lookahead_config)
        ),
        "scenario_seed_range": {
            "start": scenario_start,
            "stop_exclusive": scenario_start + scenario_count,
        },
        "rl_seed": args.seed,
        "deterministic": deterministic,
        "simulation_timing": simulation_timing.to_dict(),
        "episode_count": scenario_count,
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
            "wrong_lane_arrival_count": wrong_lane_arrival_count,
            "wrong_lane_arrival_rate": wrong_lane_arrival_count / len(episodes),
            "start_lane_departure_count": start_lane_departure_count,
            "start_lane_departure_rate": start_lane_departure_count
            / len(episodes),
            "target_lane": target_lane_aggregate,
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
        # Preserve the scalar scenario_seed alongside scenario_seed_range.
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
