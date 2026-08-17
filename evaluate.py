"""保存済みPPOを選択profileのシナリオで評価し、結果を保存する。"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, TextIO

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.utils import check_for_correct_spaces

from env_factory import make_evaluation_env
from experiment_profiles import PROFILE_NAMES, get_experiment_profile
from phase0_config import (
    LOG_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
    PROJECT_ROOT,
)


GIF_RENDER_API = (
    'env.render(mode="topdown", screen_record=True, window=False, '
    "screen_size=(600, 600), camera_position=(50, 50))"
)
GIF_GENERATE_API = (
    "env.top_down_renderer.generate_gif(gif_name=<output>, duration=30)"
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


def _json_value(value: Any) -> Any:
    """Convert NumPy and container values to JSON-safe Python values."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
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

    if flags.get("arrive_dest") is True:
        return "success"
    if flags.get("out_of_road") is True:
        return "out_of_road"
    if flags.get("crash_vehicle") is True:
        return "crash_vehicle"
    if flags.get("crash_object") is True:
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
        json.dumps(_json_value(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


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
        action="store_true",
        help="最後の評価episodeをMetaDrive 0.4.3のtop-down GIFで記録",
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
    """Run deterministic Gymnasium evaluation and optionally record one GIF."""

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
    result_path = OUTPUT_DIR / f"{args.output_prefix}_evaluation.json"
    gif_path = OUTPUT_DIR / f"{args.output_prefix}_evaluation.gif"
    gif_trace_path = OUTPUT_DIR / f"{args.output_prefix}_gif_error.log"
    gif_result: dict[str, Any] = {
        "requested": bool(args.record_gif),
        "status": "not_requested",
        "output_path": str(gif_path.resolve()) if args.record_gif else None,
        "recorded_episode": args.episodes if args.record_gif else None,
        "render_api": GIF_RENDER_API if args.record_gif else None,
        "generate_api": GIF_GENERATE_API if args.record_gif else None,
        "verified_before_env_close": False,
        "size_bytes": None,
        "error": None,
        "traceback_path": None,
        "attempted_apis": [],
    }

    gif_attempted_apis: list[str] = []
    gif_exception_trace: str | None = None
    if args.record_gif:
        try:
            # A stale artifact must not be mistaken for this run's output.
            gif_path.unlink(missing_ok=True)
            gif_trace_path.unlink(missing_ok=True)
            gif_result["status"] = "recording"
        except Exception:
            # Artifact preparation belongs to GIF output, not policy evaluation.
            gif_exception_trace = traceback.format_exc()
            gif_result["status"] = "failed"

    env = None
    episodes: list[dict[str, Any]] = []
    evaluation_started_at = datetime.now(timezone.utc)
    evaluation_start_time = time.perf_counter()
    try:
        env = make_evaluation_env(
            seed=args.seed,
            record_gif=args.record_gif,
            env_config=environment_config,
        )
        check_for_correct_spaces(env, model.observation_space, model.action_space)

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
            record_this_episode = args.record_gif and episode_number == args.episodes

            while True:
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                episode_length += 1
                terminated_flag = bool(terminated)
                truncated_flag = bool(truncated)
                final_info = info

                if record_this_episode and gif_exception_trace is None:
                    try:
                        # This is the exact headless top-down API in MetaDrive 0.4.3.
                        if GIF_RENDER_API not in gif_attempted_apis:
                            gif_attempted_apis.append(GIF_RENDER_API)
                        env.render(
                            mode="topdown",
                            screen_record=True,
                            window=False,
                            screen_size=(600, 600),
                            camera_position=(50, 50),
                        )
                    except Exception:
                        gif_exception_trace = traceback.format_exc()

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
            episode_result = {
                "episode": episode_number,
                "scenario_seed": actual_scenario_seed,
                "total_reward": total_reward,
                "episode_length": episode_length,
                "terminated": terminated_flag,
                "truncated": truncated_flag,
                **flags,
                "termination_reason": _termination_reason(
                    terminated=terminated_flag,
                    truncated=truncated_flag,
                    flags=flags,
                ),
                "reset_info_keys": sorted(str(key) for key in reset_info.keys()),
                "final_info_keys": sorted(str(key) for key in final_info.keys()),
                "execution_seconds": time.perf_counter() - episode_start_time,
            }
            episodes.append(episode_result)
            print(
                f"episode={episode_number} scenario_seed={actual_scenario_seed} "
                f"reward={total_reward:.6f} "
                f"length={episode_length} terminated={terminated_flag} "
                f"truncated={truncated_flag} "
                f"reason={episode_result['termination_reason']}"
            )

        if args.record_gif:
            if gif_exception_trace is None:
                try:
                    renderer = env.top_down_renderer
                    if renderer is None:
                        raise RuntimeError("top_down_renderer was not created")
                    # Signature in MetaDrive 0.4.3: generate_gif(gif_name, duration=30).
                    gif_attempted_apis.append(GIF_GENERATE_API)
                    renderer.generate_gif(gif_name=str(gif_path), duration=30)
                    if not gif_path.is_file():
                        raise FileNotFoundError(f"GIFが生成されませんでした: {gif_path}")
                    gif_size = gif_path.stat().st_size
                    if gif_size <= 0:
                        raise OSError(f"生成されたGIFが空です: {gif_path}")
                    # This validation intentionally occurs before env.close().
                    gif_result.update(
                        {
                            "status": "success",
                            "verified_before_env_close": True,
                            "size_bytes": gif_size,
                        }
                    )
                except Exception:
                    gif_exception_trace = traceback.format_exc()

            if gif_exception_trace is not None:
                _write_gif_trace(
                    gif_trace_path,
                    gif_exception_trace,
                    gif_attempted_apis,
                )
                gif_result.update(
                    {
                        "status": "failed",
                        "error": gif_exception_trace.strip().splitlines()[-1],
                        "traceback_path": str(gif_trace_path.resolve()),
                        "attempted_apis": gif_attempted_apis,
                    }
                )
                print(
                    f"gif_failed=True traceback={gif_trace_path.resolve()}",
                    file=sys.stderr,
                )
            else:
                gif_result["attempted_apis"] = gif_attempted_apis
                print(f"gif_saved={gif_path.resolve()} size_bytes={gif_result['size_bytes']}")
    finally:
        if env is not None:
            env.close()
            print("evaluation_env_closed=True")

    rewards = [float(episode["total_reward"]) for episode in episodes]
    lengths = [int(episode["episode_length"]) for episode in episodes]
    success_count = sum(
        episode["termination_reason"] == "success" for episode in episodes
    )
    out_of_road_count = sum(
        episode["termination_reason"] == "out_of_road" for episode in episodes
    )
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
        "environment_config": dict(environment_config),
        "scenario_seed_range": {
            "start": scenario_start,
            "stop_exclusive": scenario_start + scenario_count,
        },
        "rl_seed": args.seed,
        "deterministic": True,
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
        # GIF failure is deliberately independent of successful policy evaluation.
        "gif": gif_result,
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
