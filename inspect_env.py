"""学習前にversion、Gymnasium契約、離散Action変換を検査する。"""

from __future__ import annotations

import json
import platform
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from importlib.metadata import PackageNotFoundError, version
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

import gymnasium as gym
import numpy as np
from stable_baselines3.common.env_checker import check_env

from env_factory import make_env
from configs.phase0_config import OFFICIAL_ENV_CONFIG, OUTPUT_DIR, SCENARIO_SEED

if TYPE_CHECKING:
    from metadrive.envs import MetaDriveEnv


INSPECTION_LOG: Path = OUTPUT_DIR / "inspect_env.log"
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
    underlying raw環境は公式scenario 5でresetする。stepやspaces、Reward、
    終了値には手を加えず、学習・評価にも使用しない。
    """

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """checkerの乱数seedを受理しつつ公式scenarioを固定してresetする。"""

        if options:
            raise NotImplementedError(
                "Phase 0 check_env adapter does not support non-empty reset options"
            )
        if seed is not None:
            self.action_space.seed(seed)
            self.observation_space.seed(seed)

        reset_result = self.env.reset(seed=SCENARIO_SEED)
        assert getattr(self.env, "current_seed", None) == SCENARIO_SEED
        return reset_result


def _is_known_check_env_seed_conflict(error: Exception) -> bool:
    """MetaDrive 0.4.3とSB3 2.9の既知のseed意味衝突だけを識別する。"""

    scenario_count = int(OFFICIAL_ENV_CONFIG["num_scenarios"])
    expected_message = (
        f"scenario_index (seed) should be in "
        f"[{SCENARIO_SEED}:{SCENARIO_SEED + scenario_count})"
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
    """実環境のEnvInputPolicyを使い、9 Actionすべての変換値を表示する。"""

    from metadrive.policy.env_input_policy import EnvInputPolicy

    action_space = env.action_space
    assert getattr(action_space, "n", None) == 9, f"expected Discrete(9), got {action_space}"

    policy = env.engine.get_policy(env.agent.name)
    assert isinstance(policy, EnvInputPolicy), f"unexpected policy: {type(policy)!r}"

    print("== Discrete action conversion ==")
    for action_id in range(action_space.n):
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


def _run_inspection() -> None:
    """単一のraw MetaDrive環境に対して全検査を実行する。"""

    _print_versions()
    print("\n== Applied environment config ==")
    print(json.dumps(OFFICIAL_ENV_CONFIG, ensure_ascii=False, indent=2))

    env = make_env()
    checker_error: Exception | None = None
    try:
        observation, reset_info = env.reset()
        observation_array = _assert_valid_observation(env, observation)

        actual_config = {key: env.config[key] for key in OFFICIAL_ENV_CONFIG}
        print("\n== Effective values in env.config ==")
        print(json.dumps(actual_config, ensure_ascii=False, indent=2))
        assert actual_config == OFFICIAL_ENV_CONFIG

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
            if _is_known_check_env_seed_conflict(error):
                print(
                    "raw check_env compatibility finding: SB3 seed=0 conflicts "
                    "with MetaDrive's fixed scenario range [5:6); task config "
                    "remains unchanged"
                )
                print("\n== Stable-Baselines3 check_env: seed-only inspection adapter ==")
                checker_env = _CheckEnvFixedScenarioAdapter(env)
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
                        "adapted check_env: PASS (official scenario=5; "
                        "warnings, if any, are shown above)"
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
        raise RuntimeError("check_env reported a fatal error; see outputs/inspect_env.log") from checker_error


def main() -> None:
    """terminal表示を維持しながら検査結果をoutputsへ保存する。"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with INSPECTION_LOG.open("w", encoding="utf-8") as log_file:
        stdout_tee = _Tee(sys.stdout, log_file)
        stderr_tee = _Tee(sys.stderr, log_file)
        with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
            try:
                _run_inspection()
            except Exception:
                print("inspect_env: FATAL ERROR")
                traceback.print_exc()
                raise


if __name__ == "__main__":
    main()
