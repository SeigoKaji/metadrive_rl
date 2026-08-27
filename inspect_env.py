"""学習前にversion、Gymnasium契約、離散Action変換を検査する。"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from numbers import Integral, Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

import gymnasium as gym
import numpy as np
from stable_baselines3.common.env_checker import check_env

from configs.experiment_config import (
    ExperimentConfigError,
    ExperimentSelection,
    PROFILE_NAMES,
    environment_config_for_stage,
    select_experiment,
)
from env_factory import make_env
from project_paths import OUTPUT_DIR

if TYPE_CHECKING:
    from metadrive.envs import MetaDriveEnv


PACKAGE_DISTRIBUTIONS: tuple[tuple[str, str], ...] = (
    ("MetaDrive", "metadrive-simulator"),
    ("Stable-Baselines3", "stable-baselines3"),
    ("Gymnasium", "gymnasium"),
    ("PyTorch", "torch"),
    ("NumPy", "numpy"),
    ("Panda3D", "panda3d"),
)


class _CheckEnvFixedScenarioAdapter(gym.Wrapper):
    """SB3 checkerのGym seedとMetaDriveのscenario indexを検査時だけ分離する。

    MetaDrive 0.4.3は``reset(seed=...)``をGymnasiumの乱数seedではなく
    scenario indexとして扱う。一方、SB3 2.9の``check_env``は契約確認のため
    必ず``seed=0``を渡す。このwrapperはcheckerが渡すseedをspacesへ適用し、
    underlying raw環境は選択stageの開始scenarioでresetする。stepやspaces、Reward、
    終了値には手を加えず、学習・評価にも使用しない。
    """

    def __init__(self, env: gym.Env, *, scenario_seed: int) -> None:
        super().__init__(env)
        self._scenario_seed = scenario_seed

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """checkerの乱数seedを受理しつつ開始scenarioでresetする。"""

        if options:
            raise NotImplementedError(
                "check_env adapter does not support non-empty reset options"
            )
        if seed is not None:
            self.action_space.seed(seed)
            self.observation_space.seed(seed)

        reset_result = self.env.reset(seed=self._scenario_seed)
        assert getattr(self.env, "current_seed", None) == self._scenario_seed
        return reset_result


def _scenario_seed_bounds(env_config: Mapping[str, object]) -> tuple[int, int]:
    """Return an environment config's half-open scenario seed range."""

    scenario_start = int(env_config["start_seed"])
    scenario_count = int(env_config["num_scenarios"])
    return scenario_start, scenario_start + scenario_count


def _is_known_check_env_seed_conflict(
    error: Exception,
    env_config: Mapping[str, object],
) -> bool:
    """MetaDrive 0.4.3とSB3 2.9の既知のseed意味衝突だけを識別する。"""

    scenario_start, scenario_stop = _scenario_seed_bounds(env_config)
    expected_message = (
        f"scenario_index (seed) should be in "
        f"[{scenario_start}:{scenario_stop})"
    )
    return isinstance(error, AssertionError) and expected_message in str(error)


class _Tee:
    """stdout/stderrをterminalと検査ログの両方へ複製する。"""

    def __init__(self, terminal: TextIO, log_file: TextIO) -> None:
        self._terminal = terminal
        self._log_file = log_file

    def write(self, text: str) -> int:
        """両方のstreamへ同じ文字列を書く。"""

        self._terminal.write(text)
        self._log_file.write(text)
        return len(text)

    def flush(self) -> None:
        """両方のstreamをflushする。"""

        self._terminal.flush()
        self._log_file.flush()

    def isatty(self) -> bool:
        """terminal側のTTY判定を委譲する。"""

        return self._terminal.isatty()

    @property
    def encoding(self) -> str | None:
        """terminal側のencodingを公開する。"""

        return self._terminal.encoding


def _installed_version(distribution: str) -> str:
    """distribution metadataからversionを取得する。"""

    try:
        return version(distribution)
    except PackageNotFoundError:
        return "NOT INSTALLED"


def _print_versions() -> None:
    """要求されたruntime versionを表示する。"""

    print("== Versions ==")
    print(f"Python: {platform.python_version()}")
    print(f"Python executable: {sys.executable}")
    for label, distribution in PACKAGE_DISTRIBUTIONS:
        print(f"{label}: {_installed_version(distribution)}")


def _assert_valid_observation(env: MetaDriveEnv, observation: object) -> np.ndarray:
    """Observationがspace内かつ有限値だけであることを検証する。"""

    assert env.observation_space.contains(observation), "observation is outside observation_space"
    array = np.asarray(observation)
    assert bool(np.isfinite(array).all()), "observation contains NaN or Inf"
    return array


def _print_action_conversion(env: MetaDriveEnv) -> None:
    """実環境のEnvInputPolicyを使い、設定された離散Actionを全て表示する。"""

    from metadrive.policy.env_input_policy import EnvInputPolicy

    action_space = env.action_space
    action_count = getattr(action_space, "n", None)
    assert isinstance(action_count, Integral) and action_count > 0, (
        f"expected a non-empty Discrete action space, got {action_space}"
    )
    action_count = int(action_count)
    steering_dim = int(env.config["discrete_steering_dim"])
    throttle_dim = int(env.config["discrete_throttle_dim"])
    expected_count = steering_dim * throttle_dim
    assert action_count == expected_count, (
        "discrete action space and configured dimensions disagree: "
        f"n={action_count}, steering={steering_dim}, throttle={throttle_dim}"
    )

    policy = env.engine.get_policy(env.agent.name)
    assert isinstance(policy, EnvInputPolicy), f"unexpected policy: {type(policy)!r}"

    print("== Discrete action conversion ==")
    print(
        f"action_count={action_count} "
        f"(steering={steering_dim} * throttle={throttle_dim})"
    )
    for action_id in range(action_count):
        steering, throttle_brake = policy.convert_to_continuous_action(action_id)
        print(
            f"action_id={action_id} -> steering={steering:+.1f}, "
            f"throttle_brake={throttle_brake:+.1f}"
        )


def _run_random_actions(env: MetaDriveEnv, max_steps: int = 50) -> None:
    """最大50 stepのrandom Actionでstep戻り値を検証する。"""

    if not 1 <= max_steps <= 50:
        raise ValueError(f"max_steps must be in [1, 50]: {max_steps}")

    observation, _ = env.reset()
    _assert_valid_observation(env, observation)
    resets_after_done = 0
    total_reward = 0.0

    for _step_index in range(max_steps):
        action = env.action_space.sample()
        step_result = env.step(action)
        assert isinstance(step_result, tuple) and len(step_result) == 5
        observation, reward, terminated, truncated, _info = step_result

        assert isinstance(reward, Real) and not isinstance(reward, bool), "reward must be numeric"
        assert isinstance(terminated, bool), "terminated must be bool"
        assert isinstance(truncated, bool), "truncated must be bool"
        _assert_valid_observation(env, observation)
        total_reward += float(reward)

        if terminated or truncated:
            observation, _ = env.reset()
            _assert_valid_observation(env, observation)
            resets_after_done += 1

    print("== Random action run ==")
    print(f"steps: {max_steps}")
    print(f"total_reward: {total_reward:.6f}")
    print(f"resets_after_done: {resets_after_done}")
    print("random action run: PASS")


def _run_inspection(experiment: ExperimentSelection, stage: str) -> None:
    """選択されたstageの単一raw MetaDrive環境に対して全検査を実行する。"""

    environment_config = environment_config_for_stage(experiment, stage)
    scenario_start, scenario_stop = _scenario_seed_bounds(environment_config)
    _print_versions()
    print("\n== Selected experiment ==")
    print(
        json.dumps(
            {
                "stage": stage,
                "config_source": experiment.source_metadata(),
                "scenario_seed_range": [scenario_start, scenario_stop],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("\n== Applied environment config ==")
    print(json.dumps(environment_config, ensure_ascii=False, indent=2))

    env = make_env(environment_config)
    checker_error: Exception | None = None
    try:
        observation, reset_info = env.reset()
        observation_array = _assert_valid_observation(env, observation)

        actual_config = {key: env.config[key] for key in environment_config}
        print("\n== Effective values in env.config ==")
        print(json.dumps(actual_config, ensure_ascii=False, indent=2))
        assert actual_config == environment_config

        print("\n== Spaces and reset observation ==")
        print(f"observation_space: {env.observation_space}")
        print(f"action_space: {env.action_space}")
        print(f"observation_shape: {observation_array.shape}")
        print(f"observation_dtype: {observation_array.dtype}")
        print(f"observation_min: {float(observation_array.min())}")
        print(f"observation_max: {float(observation_array.max())}")
        print(f"observation_has_nan: {bool(np.isnan(observation_array).any())}")
        print(f"observation_has_inf: {bool(np.isinf(observation_array).any())}")
        print(f"reset_info_keys: {sorted(map(str, reset_info.keys()))}")

        _print_action_conversion(env)

        probe_result = env.step(env.action_space.sample())
        assert isinstance(probe_result, tuple) and len(probe_result) == 5
        probe_observation, probe_reward, terminated, truncated, step_info = probe_result
        _assert_valid_observation(env, probe_observation)
        assert isinstance(probe_reward, Real) and not isinstance(probe_reward, bool)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        print("\n== One-step return ==")
        print(f"step_info_keys: {sorted(map(str, step_info.keys()))}")
        print(f"terminated: {terminated}")
        print(f"truncated: {truncated}")

        print("\n== Stable-Baselines3 check_env: raw environment ==")
        try:
            check_env(env, warn=True)
        except Exception as error:
            print(f"raw check_env fatal error: {type(error).__name__}: {error}")
            traceback.print_exc()
            if _is_known_check_env_seed_conflict(error, environment_config):
                print(
                    "raw check_env compatibility finding: SB3 seed=0 conflicts "
                    "with MetaDrive's configured scenario range "
                    f"[{scenario_start}:{scenario_stop}); the adapter does not "
                    "modify the task configuration"
                )
                print("\n== Stable-Baselines3 check_env: seed-only inspection adapter ==")
                checker_env = _CheckEnvFixedScenarioAdapter(
                    env,
                    scenario_seed=scenario_start,
                )
                try:
                    check_env(checker_env, warn=True)
                except Exception as adapter_error:
                    checker_error = adapter_error
                    print(
                        "adapted check_env fatal error: "
                        f"{type(adapter_error).__name__}: {adapter_error}"
                    )
                    traceback.print_exc()
                else:
                    print(
                        "adapted check_env: PASS "
                        f"(stage start scenario={scenario_start}; warnings, if any, "
                        "are shown above)"
                    )
            else:
                checker_error = error
        else:
            print("raw check_env: PASS (warnings, if any, are shown above)")

        # check_envは内部でreset/stepするため、random走行の開始状態を明示的に戻す。
        _run_random_actions(env, max_steps=50)
    finally:
        env.close()
        print("environment closed: PASS")

    if checker_error is not None:
        raise RuntimeError("check_env reported a fatal error; see the inspection log") from checker_error


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """選択TOMLと検査対象stageを解決する。"""

    selection_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selection_group = selection_parser.add_mutually_exclusive_group()
    selection_group.add_argument("--profile", choices=PROFILE_NAMES, default=None)
    selection_group.add_argument("--config", type=Path, default=None)
    selected, _unknown = selection_parser.parse_known_args(argv)
    try:
        experiment = select_experiment(
            profile_name=selected.profile,
            config_path=selected.config,
        )
    except ExperimentConfigError as error:
        selection_parser.error(str(error))

    parser = argparse.ArgumentParser(
        description="MetaDrive環境を学習・評価前に検査します",
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
        help="実験bundle TOML（相対pathはproject直下基準）",
    )
    parser.add_argument(
        "--stage",
        choices=("train", "evaluation"),
        default="train",
        help="検査する環境設定（既定: train）",
    )
    args = parser.parse_args(argv)
    args.profile = experiment.name
    args.config = experiment.source_path
    args.experiment = experiment
    args.experiment_selection = experiment
    return args


def _inspection_log_path(experiment: ExperimentSelection, stage: str) -> Path:
    """選択sourceとstageが分かる、outputs配下の検査ログpathを返す。"""

    return OUTPUT_DIR / "inspect_env" / experiment.name / f"{stage}.log"


def main(argv: list[str] | None = None) -> int:
    """terminal表示を維持しながら選択sourceの検査結果をoutputsへ保存する。"""

    args = parse_args(argv)
    log_path = _inspection_log_path(args.experiment, args.stage)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        stdout_tee = _Tee(sys.stdout, log_file)
        stderr_tee = _Tee(sys.stderr, log_file)
        with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
            print(f"inspection_log={log_path.resolve()}")
            try:
                _run_inspection(args.experiment, args.stage)
            except Exception:
                print("inspect_env: FATAL ERROR")
                traceback.print_exc()
                raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
