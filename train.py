"""選択したMetaDrive profileのPPO学習をCLIから実行する。"""

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
from collections.abc import Mapping
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, TextIO

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv

from env_factory import make_training_env
from lookahead_learning.checkpoint import (
    set_lookahead_model_metadata,
    validate_lookahead_model_metadata,
)
from configs.experiment_config import (
    ExperimentConfigError,
    PPO_COMMON_SCALAR_KEYS,
    PROFILE_NAMES,
    experiment_selection_from_args,
    normalize_model_name,
    select_experiment,
)
from project_paths import (
    LOG_DIR,
    MODEL_DIR,
    MONITOR_LOG_DIR,
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

    try:
        return normalize_model_name(value, "--model-name")
    except ExperimentConfigError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


def _default_training_log(model_name: str) -> Path:
    """Keep the requested canonical log names while supporting custom runs."""

    if model_name == "official_baseline":
        return LOG_DIR / "full_train.log"
    return LOG_DIR / f"{model_name}_train.log"


def _resolve_log_path(path: Path | None, model_name: str) -> Path:
    """Resolve an optional log path relative to this project."""

    if path is None:
        return _default_training_log(model_name)
    if path.is_absolute():
        return path
    return LOG_DIR.parent / path


def _resolved_ppo_config(
    args: argparse.Namespace,
    training_config: Mapping[str, object],
) -> dict[str, object]:
    """Return the exact scalar PPO configuration after CLI rollout overrides."""

    ppo_config = {
        "n_steps": int(args.n_steps),
        **{
            key: training_config[key]
            for key in PPO_COMMON_SCALAR_KEYS
        },
    }
    if bool(ppo_config["normalize_advantage"]):
        rollout_batch_size = int(args.num_envs) * int(ppo_config["n_steps"])
        if rollout_batch_size <= 1:
            raise ValueError(
                "normalize_advantage=trueでは--num-envs * --n-stepsを2以上にしてください"
            )
        if int(ppo_config["batch_size"]) <= 1:
            raise ValueError(
                "normalize_advantage=trueではtraining.batch_sizeを2以上にしてください"
            )
    return ppo_config


def _training_output_directory(profile_name: str, model_name: str) -> Path:
    """Return the profile- and run-specific directory for training metadata."""

    return OUTPUT_DIR / profile_name / "training" / model_name


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
    """canonical TOML aliasまたは外部bundleをCLI既定値としてparseする。"""

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
    training_config = profile.training_config

    parser = argparse.ArgumentParser(
        description="MetaDrive SB3 PPO学習",
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
        "--timesteps",
        type=_positive_int,
        default=int(training_config["total_timesteps"]),
        help="model.learn()へ渡す最小timestep数",
    )
    parser.add_argument(
        "--num-envs",
        type=_positive_int,
        default=int(training_config["num_envs"]),
        help="SubprocVecEnvで並列実行する環境数",
    )
    parser.add_argument(
        "--n-steps",
        type=_positive_int,
        default=int(training_config["n_steps"]),
        help="1環境あたりのPPO rollout長",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(training_config["seed"]),
        help="RL/PPOの乱数seed（scenario seed範囲とは別）",
    )
    parser.add_argument(
        "--device",
        default=str(training_config.get("device", "cpu")),
        help="SB3 PPOに明示するdevice（例: cpu, cuda, auto）",
    )
    parser.add_argument(
        "--model-name",
        type=_model_stem,
        default=str(training_config.get("model_name", profile.default_model_name)),
        help="models/とoutputs/<profile>/training/で使うrun名",
    )
    parser.add_argument(
        "--log-interval",
        type=_positive_int,
        default=int(training_config["log_interval"]),
        help="model.learn()のログ間隔",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=(
            None
            if training_config.get("log_file") is None
            else Path(str(training_config["log_file"]))
        ),
        help="標準出力/標準エラーの複製先（相対pathはproject直下基準）",
    )
    args = parser.parse_args(argv)
    try:
        _resolved_ppo_config(args, training_config)
    except ValueError as error:
        parser.error(str(error))
    # Do not re-read the source after defaults have been resolved.  Keeping the
    # selection makes both runtime and saved metadata use the exact same bundle.
    args.profile = experiment.name
    args.config = experiment.source_path
    args.experiment = experiment
    args.experiment_selection = experiment
    return args


def _run_training(args: argparse.Namespace, log_path: Path) -> Path:
    """Train, save, validate, and reload one PPO model."""

    experiment = experiment_selection_from_args(args)
    profile = experiment.profile
    environment_config = profile.train_env_config
    lookahead_config = profile.lookahead_config
    training_config = profile.training_config
    ppo_config = _resolved_ppo_config(args, training_config)
    scenario_start = int(environment_config["start_seed"])
    scenario_count = int(environment_config["num_scenarios"])
    training_output_dir = _training_output_directory(
        experiment.name,
        args.model_name,
    )

    for directory in (
        MODEL_DIR,
        MONITOR_LOG_DIR,
        TENSORBOARD_LOG_DIR,
        training_output_dir,
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
            env_config=environment_config,
            lookahead_config=lookahead_config,
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
            str(training_config["policy"]),
            train_env,
            **ppo_config,
            verbose=1,
            device=args.device,
            tensorboard_log=str(TENSORBOARD_LOG_DIR),
        )
        # Stable-Baselines3 serializes custom instance attributes in the PPO
        # ZIP.  Store the resolved TOML values before saving so evaluation can
        # reject same-shaped checkpoints made with different lookahead values.
        set_lookahead_model_metadata(model, lookahead_config)
        actual_device = str(model.device)
        print(
            "training_start",
            {
                "profile": args.profile,
                "timesteps": args.timesteps,
                "num_envs": args.num_envs,
                "n_steps": args.n_steps,
                "seed": args.seed,
                "scenario_seed_range": [
                    scenario_start,
                    scenario_start + scenario_count,
                ],
                "requested_device": args.device,
                "actual_device": actual_device,
                "ppo_config": ppo_config,
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
        validate_lookahead_model_metadata(reloaded_model, lookahead_config)
        reload_device = str(reloaded_model.device)
        del reloaded_model

        finished_at = datetime.now(timezone.utc)
        metadata_path = training_output_dir / "training_metadata.json"
        metadata = {
            "status": "success",
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "elapsed_seconds": time.perf_counter() - start_time,
            "command": [sys.executable, *sys.argv],
            "platform": platform.platform(),
            "python_executable": sys.executable,
            "versions": _runtime_versions(),
            "profile": experiment.name,
            "config_source": experiment.source_metadata(),
            "environment_config": dict(environment_config),
            "lookahead": (
                None if lookahead_config is None else dict(lookahead_config)
            ),
            "profile_training_config": dict(training_config),
            "training": {
                "policy": str(training_config["policy"]),
                "requested_total_timesteps": args.timesteps,
                "actual_total_timesteps": int(model.num_timesteps),
                "num_envs": args.num_envs,
                "n_steps": args.n_steps,
                "rollout_batch_size": args.num_envs * args.n_steps,
                "log_interval": args.log_interval,
                "rl_seed": args.seed,
                "scenario_seed_range": {
                    "start": scenario_start,
                    "stop_exclusive": scenario_start + scenario_count,
                },
                "requested_device": args.device,
                "actual_device": actual_device,
                "ppo_seed_argument": None,
                "seed_note": (
                    "set_random_seed() and worker space seeding are used; PPO's seed "
                    "argument is intentionally omitted because SB3 would forward it to "
                    "env.reset(seed=...), which MetaDrive 0.4.3 interprets as a scenario index"
                ),
                "scenario_sampling_note": (
                    "reset() without a seed lets MetaDrive sample inside the configured "
                    "scenario range; this sampling is separate from the RL seed"
                ),
                "scenario_sampling_reproducible_from_rl_seed": scenario_count == 1,
                "resolved_ppo_config": {
                    "policy": str(training_config["policy"]),
                    **ppo_config,
                    "device": args.device,
                    "verbose": 1,
                    "tensorboard_log": str(TENSORBOARD_LOG_DIR),
                },
                "ppo_nonconfigured_parameters": "stable-baselines3 defaults",
            },
            "artifacts": {
                "output_directory": str(training_output_dir.resolve()),
                "metadata_path": str(metadata_path.resolve()),
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
        if experiment.name == "official":
            # Keep official baseline metadata available to existing consumers.
            metadata["official_environment_config"] = dict(environment_config)
            metadata["training"]["scenario_seed"] = scenario_start
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
