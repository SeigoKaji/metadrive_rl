"""MetaDrive公式SB3ミニ例のPPO学習をCLIから実行する。"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import multiprocessing
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, TextIO

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv

from env_factory import make_training_env
from phase0_config import (
    LOG_DIR,
    MODEL_DIR,
    MONITOR_LOG_DIR,
    OFFICIAL_ENV_CONFIG,
    OFFICIAL_TRAINING_CONFIG,
    OUTPUT_DIR,
    TENSORBOARD_LOG_DIR,
)


class _Tee:
    """Write SB3's console output to the terminal and a persistent log."""

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


def _model_stem(value: str) -> str:
    """Validate a model basename and normalize an optional .zip suffix."""

    if not value or Path(value).name != value:
        raise argparse.ArgumentTypeError("--model-nameはディレクトリを含まないファイル名にしてください")
    stem = value[:-4] if value.endswith(".zip") else value
    if stem in {"", ".", ".."}:
        raise argparse.ArgumentTypeError("--model-nameに有効な名前を指定してください")
    return stem


def _default_training_log(model_name: str) -> Path:
    """Keep the requested canonical log names while supporting custom runs."""

    if model_name == "phase0_smoke":
        return LOG_DIR / "smoke_train.log"
    if model_name == "phase0_official":
        return LOG_DIR / "full_train.log"
    return LOG_DIR / f"{model_name}_train.log"


def _resolve_log_path(path: Path | None, model_name: str) -> Path:
    """Resolve an optional log path relative to this Phase 0 project."""

    if path is None:
        return _default_training_log(model_name)
    if path.is_absolute():
        return path
    return LOG_DIR.parent / path


def _distribution_version(*names: str) -> str | None:
    """Return the first installed distribution version without guessing."""

    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _runtime_versions() -> dict[str, str | None]:
    """Collect versions relevant to reproducing the saved PPO model."""

    return {
        "python": platform.python_version(),
        "metadrive": _distribution_version("metadrive", "metadrive-simulator"),
        "stable_baselines3": _distribution_version("stable-baselines3"),
        "gymnasium": _distribution_version("gymnasium"),
        "torch": _distribution_version("torch"),
        "numpy": _distribution_version("numpy"),
        "panda3d": _distribution_version("panda3d"),
    }


def _sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for a model artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while chunk := file_obj.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write UTF-8 JSON through a sibling temporary file."""

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse training options; defaults exactly match the official full run."""

    parser = argparse.ArgumentParser(
        description="MetaDrive公式SB3ミニ例のPPO学習",
    )
    parser.add_argument(
        "--timesteps",
        type=_positive_int,
        default=int(OFFICIAL_TRAINING_CONFIG["total_timesteps"]),
        help="model.learn()へ渡す最小timestep数",
    )
    parser.add_argument(
        "--num-envs",
        type=_positive_int,
        default=int(OFFICIAL_TRAINING_CONFIG["num_envs"]),
        help="SubprocVecEnvで並列実行する環境数",
    )
    parser.add_argument(
        "--n-steps",
        type=_positive_int,
        default=int(OFFICIAL_TRAINING_CONFIG["n_steps"]),
        help="1環境あたりのPPO rollout長",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(OFFICIAL_TRAINING_CONFIG["seed"]),
        help="RL/PPOの乱数seed（scenario seedは環境設定の5で固定）",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="SB3 PPOに明示するdevice（例: cpu, cuda, auto）",
    )
    parser.add_argument(
        "--model-name",
        type=_model_stem,
        default="phase0_official",
        help="models/とoutputs/で使うベース名",
    )
    parser.add_argument(
        "--log-interval",
        type=_positive_int,
        default=int(OFFICIAL_TRAINING_CONFIG["log_interval"]),
        help="model.learn()のログ間隔",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="標準出力/標準エラーの複製先（相対パスはPhase 0直下基準）",
    )
    return parser.parse_args(argv)


def _run_training(args: argparse.Namespace, log_path: Path) -> Path:
    """Train, save, validate, and reload one PPO model."""

    for directory in (
        MODEL_DIR,
        MONITOR_LOG_DIR,
        TENSORBOARD_LOG_DIR,
        OUTPUT_DIR,
        LOG_DIR,
        log_path.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    set_random_seed(args.seed)
    env_factories = [
        partial(
            make_training_env,
            rank=rank,
            seed=args.seed,
            monitor_dir=MONITOR_LOG_DIR,
        )
        for rank in range(args.num_envs)
    ]

    train_env: SubprocVecEnv | None = None
    started_at = datetime.now(timezone.utc)
    start_time = time.perf_counter()
    try:
        # Each picklable partial creates exactly one MetaDrive instance in its worker.
        train_env = SubprocVecEnv(env_factories)
        model = PPO(
            str(OFFICIAL_TRAINING_CONFIG["policy"]),
            train_env,
            n_steps=args.n_steps,
            verbose=1,
            device=args.device,
            tensorboard_log=str(TENSORBOARD_LOG_DIR),
        )
        actual_device = str(model.device)
        print(
            "training_start",
            {
                "timesteps": args.timesteps,
                "num_envs": args.num_envs,
                "n_steps": args.n_steps,
                "seed": args.seed,
                "requested_device": args.device,
                "actual_device": actual_device,
            },
        )
        model.learn(total_timesteps=args.timesteps, log_interval=args.log_interval)

        model_base_path = MODEL_DIR / args.model_name
        model.save(str(model_base_path))
        model_path = Path(f"{model_base_path}.zip")
        if not model_path.is_file():
            raise FileNotFoundError(f"保存したモデルが見つかりません: {model_path}")
        model_size = model_path.stat().st_size
        if model_size <= 0:
            raise OSError(f"保存したモデルが空です: {model_path}")
        model_sha256 = _sha256_file(model_path)

        # Loading with the same VecEnv also checks the saved observation/action spaces.
        reloaded_model = PPO.load(str(model_path), env=train_env, device=args.device)
        reload_device = str(reloaded_model.device)
        del reloaded_model

        finished_at = datetime.now(timezone.utc)
        metadata_path = OUTPUT_DIR / f"{args.model_name}_training_metadata.json"
        metadata = {
            "status": "success",
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "elapsed_seconds": time.perf_counter() - start_time,
            "command": [sys.executable, *sys.argv],
            "platform": platform.platform(),
            "python_executable": sys.executable,
            "versions": _runtime_versions(),
            "official_environment_config": dict(OFFICIAL_ENV_CONFIG),
            "training": {
                "policy": str(OFFICIAL_TRAINING_CONFIG["policy"]),
                "requested_total_timesteps": args.timesteps,
                "actual_total_timesteps": int(model.num_timesteps),
                "num_envs": args.num_envs,
                "n_steps": args.n_steps,
                "rollout_batch_size": args.num_envs * args.n_steps,
                "log_interval": args.log_interval,
                "rl_seed": args.seed,
                "scenario_seed": int(OFFICIAL_ENV_CONFIG["start_seed"]),
                "requested_device": args.device,
                "actual_device": actual_device,
                "ppo_seed_argument": None,
                "seed_note": (
                    "set_random_seed() and worker space seeding are used; PPO's seed "
                    "argument is intentionally omitted because SB3 would forward it to "
                    "env.reset(seed=...), which MetaDrive 0.4.3 interprets as a scenario index"
                ),
                "ppo_unspecified_parameters": "stable-baselines3 defaults",
            },
            "artifacts": {
                "model_path": str(model_path.resolve()),
                "model_size_bytes": model_size,
                "model_sha256": model_sha256,
                "monitor_log_directory": str(MONITOR_LOG_DIR.resolve()),
                "tensorboard_log_directory": str(TENSORBOARD_LOG_DIR.resolve()),
                "console_log_path": str(log_path.resolve()),
            },
            "reload_verification": {
                "succeeded": True,
                "device": reload_device,
            },
        }
        _write_json(metadata_path, metadata)
        print(f"model_saved={model_path} size_bytes={model_size} sha256={model_sha256}")
        print(f"model_reload_verified=True metadata={metadata_path}")
        return model_path
    finally:
        if train_env is not None:
            train_env.close()
            print("train_env_closed=True")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point with persistent stdout/stderr capture."""

    args = parse_args(argv)
    log_path = _resolve_log_path(args.log_file, args.model_name)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        tee_stdout = _Tee(sys.stdout, log_file)
        tee_stderr = _Tee(sys.stderr, log_file)
        with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(tee_stderr):
            print(f"console_log={log_path.resolve()}")
            try:
                _run_training(args, log_path)
            except BaseException:
                print("training_failed=True", file=sys.stderr)
                traceback.print_exc()
                raise
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
