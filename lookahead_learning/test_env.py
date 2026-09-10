"""Contract and wrapper tests for ``python -B -m unittest lookahead_learning.test_env``.

The fakes implement only the audited Gymnasium/MetaDrive seams.  No simulator
module is imported and no existing project file is modified by these tests.
"""

from __future__ import annotations

import math
import unittest

import gymnasium as gym
import numpy as np

from .adapter import (
    ActionContract,
    MetaDrivePreviewProvider,
    ObservationContract,
    UnsupportedHostError,
    build_fixed_navigation_route,
    navigation_lane_sequence,
    wrap_lookahead_env,
)
from .env import LookaheadEnv


class FakeVehicle:
    def __init__(self) -> None:
        self.current_action = np.array([0.0, 0.0], dtype=np.float32)
        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.heading_theta = 0.0
        self.speed = 1.0
        self.lane_index = ("A", "B", 0)


class FakeRawEnv(gym.Env):
    """A raw single-agent env with the audited dt/action seams."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        observation_dim: int = 262,
        terminal_after: int | None = None,
        truncated_after: int | None = None,
        target_valid: bool = True,
    ) -> None:
        self.observation_dim = int(observation_dim)
        self.observation_space = gym.spaces.Box(
            low=np.full((self.observation_dim,), -2.0, dtype=np.float32),
            high=np.full((self.observation_dim,), 3.0, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(9)
        self.config = {
            "physics_world_step_size": 0.02,
            "decision_repeat": 5,
            "discrete_steering_dim": 3,
            "discrete_throttle_dim": 3,
            "use_multi_discrete": False,
        }
        self.vehicle = FakeVehicle()
        self.agents = {"agent0": self.vehicle}
        self.action_contract = ActionContract.from_space(
            self.action_space,
            self.config,
        )
        self.reset_calls = 0
        self.step_calls = 0
        self.close_calls = 0
        self.current_seed: int | None = None
        self.terminal_after = terminal_after
        self.truncated_after = truncated_after
        self.target_valid = bool(target_valid)

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        del options
        self.reset_calls += 1
        self.current_seed = 0 if seed is None else int(seed)
        self.step_calls = 0
        self.vehicle.current_action = np.array([0.0, 0.0], dtype=np.float32)
        self.vehicle.position[:] = (0.0, 0.0, 0.0)
        self.vehicle.heading_theta = 0.0
        observation = np.full((self.observation_dim,), 0.25, dtype=np.float32)
        return observation, {
            "raw_reset": self.reset_calls,
            "scenario_seed": self.current_seed,
            "target_lane_valid": self.target_valid,
            "target_lane_offset_m": 0.4 if self.target_valid else None,
            "in_target_lane": True if self.target_valid else None,
            "start_lane_departure": False,
            "target_lane_heading_error_rad": 0.15 if self.target_valid else None,
        }

    def step(self, action):
        self.step_calls += 1
        # SB3 emits a zero-dimensional ndarray for a scalar Discrete action;
        # the host decoder receives its scalar value while the wrapper still
        # records the original action object.
        if isinstance(action, np.ndarray) and action.ndim == 0:
            action = action.item()
        steering, throttle = self.action_contract.decode(action)
        self.vehicle.current_action = np.array(
            [steering, throttle], dtype=np.float32
        )
        self.vehicle.position[0] = float(self.step_calls)
        self.vehicle.heading_theta = 0.02 * self.step_calls
        self.vehicle.speed = 1.0 + self.step_calls
        observation = np.full((self.observation_dim,), 0.5, dtype=np.float32)
        terminated = (
            self.terminal_after is not None
            and self.step_calls >= self.terminal_after
        )
        truncated = (
            self.truncated_after is not None
            and self.step_calls >= self.truncated_after
        )
        return observation, 2.0, bool(terminated), bool(truncated), {
            "raw_step": self.step_calls,
            "scenario_seed": self.current_seed,
            "target_lane_valid": self.target_valid,
            "target_lane_offset_m": 0.4 if self.target_valid else None,
            "in_target_lane": True if self.target_valid else None,
            "start_lane_departure": False,
            "target_lane_heading_error_rad": 0.15 if self.target_valid else None,
        }

    def close(self):
        self.close_calls += 1


class PreviewProbe:
    def __init__(self, *, invalid_after: int | None = None) -> None:
        self.calls = 0
        self.reset_calls = 0
        self.invalid_after = invalid_after

    def reset(self, env):
        del env
        self.reset_calls += 1

    def __call__(self, env):
        self.calls += 1
        step = int(getattr(env, "step_calls", 0))
        valid = self.invalid_after is None or step < self.invalid_after
        if not valid:
            return {
                "x_g_m": None,
                "y_g_m": None,
                "preview_valid": False,
                "invalid_reason": "post_step_route_invalid",
            }
        return {
            "x_g_m": 6.0 + step,
            "y_g_m": 0.0,
            "preview_valid": True,
            "q_xy": (6.0 + step, 0.0),
            "s_proj_m": float(step),
            "s_goal_m": float(step) + 6.0,
            "projected_lane_id": "lane0",
            "goal_lane_id": "lane0",
            "details": {
                "source": "test provider",
                "fixed_route_lane_ids": ["lane0"],
            },
            "lateral_error_m": 0.25,
        }


class PPProbe:
    def __init__(self, u_pp: float = 0.1, valid: bool = True) -> None:
        self.calls = 0
        self.u_pp = u_pp
        self.valid = valid

    def __call__(self, env, preview):
        del env, preview
        self.calls += 1
        return {
            "pp_valid": self.valid,
            "u_pp": self.u_pp if self.valid else None,
            "u_pp_unclipped": self.u_pp if self.valid else None,
            "saturated": False,
        }


def _contract(raw: FakeRawEnv) -> ObservationContract:
    return ObservationContract.from_space(
        raw.observation_space,
        source="lookahead_learning.test_env",
    )


def _state(env: FakeRawEnv) -> dict[str, object]:
    return {
        "position_xy": tuple(float(item) for item in env.vehicle.position[:2]),
        "heading_theta": env.vehicle.heading_theta,
        "speed_m_s": env.vehicle.speed,
        "travelled_length_m": float(env.step_calls),
        "current_ref_lane_ids": ("lane0",),
    }


class ObservationWrapperTests(unittest.TestCase):
    def test_scalar_numpy_action_is_recorded_without_freeze_error(self) -> None:
        raw = FakeRawEnv()
        wrapped = LookaheadEnv(
            raw,
            mode="lookahead_obs",
            contract=_contract(raw),
            preview_provider=PreviewProbe(),
            state_reader=_state,
        )
        wrapped.reset(seed=17)
        _, reward, terminated, truncated, info = wrapped.step(np.asarray(5))
        self.assertEqual(reward, 2.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["lookahead_learning"]["action_env"], 5)

    def test_259_wide_host_is_augmented_without_padding(self) -> None:
        raw = FakeRawEnv(observation_dim=259)
        wrapped = LookaheadEnv(
            raw,
            mode="lookahead_obs",
            contract=_contract(raw),
            preview_provider=PreviewProbe(),
            state_reader=_state,
        )
        initial, _ = wrapped.reset(seed=13)
        self.assertEqual(initial.shape, (262,))
        np.testing.assert_array_equal(initial[:259], np.full(259, 0.25, np.float32))
        next_observation, reward, *_ = wrapped.step(5)
        self.assertEqual(next_observation.shape, (262,))
        np.testing.assert_array_equal(
            next_observation[:259], np.full(259, 0.5, np.float32)
        )
        self.assertEqual(reward, 2.0)

    def test_adapter_wrap_uses_toml_parameters_and_host_width(self) -> None:
        raw = FakeRawEnv()
        wrapped = wrap_lookahead_env(raw, lookahead_m=6.0, pp_weight=0.0)
        self.assertIsInstance(wrapped, LookaheadEnv)
        self.assertEqual(wrapped.mode, "lookahead_obs")
        self.assertEqual(wrapped.contract.shape, (262,))
        self.assertEqual(wrapped.observation_space.shape, (265,))
        self.assertEqual(raw.reset_calls, 0)

    def test_adapter_wrap_rejects_zero_lookahead_distance(self) -> None:
        with self.assertRaises(ValueError):
            wrap_lookahead_env(FakeRawEnv(), lookahead_m=0.0, pp_weight=0.0)

    def test_adapter_wrap_selects_pp_mode_for_positive_weight(self) -> None:
        wrapped = wrap_lookahead_env(
            FakeRawEnv(),
            lookahead_m=6.0,
            pp_weight=0.25,
        )
        self.assertEqual(wrapped.mode, "lookahead_obs_pp_reward")
        self.assertEqual(wrapped.pp_weight, 0.25)
        self.assertEqual(wrapped.observation_space.shape, (265,))

    def test_obs_preserves_raw_prefix_and_uses_one_transition(self) -> None:
        raw = FakeRawEnv()
        preview = PreviewProbe()
        wrapped = LookaheadEnv(
            raw,
            mode="obs",
            contract=_contract(raw),
            preview_provider=preview,
            state_reader=_state,
            run_id="obs-test",
        )
        self.assertEqual(wrapped.mode, "lookahead_obs")
        initial, reset_info = wrapped.reset(seed=7)
        self.assertEqual(initial.shape, (265,))
        self.assertEqual(initial.dtype, np.dtype(np.float32))
        np.testing.assert_array_equal(initial[:262], np.full(262, 0.25, np.float32))
        np.testing.assert_allclose(initial[262:], [0.8, 0.5, 1.0])
        self.assertEqual(reset_info["lookahead_learning"]["dt"], 0.1)
        self.assertEqual(reset_info["lookahead_learning"]["preview_valid"], True)
        next_observation, reward, terminated, truncated, info = wrapped.step(5)
        self.assertEqual(raw.reset_calls, 1)
        self.assertEqual(raw.step_calls, 1)
        self.assertEqual(preview.reset_calls, 1)
        self.assertEqual(preview.calls, 2)
        self.assertEqual(next_observation.shape, (265,))
        np.testing.assert_array_equal(next_observation[:262], np.full(262, 0.5, np.float32))
        self.assertEqual(reward, 2.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        namespace = info["lookahead_learning"]
        self.assertEqual(namespace["t"], 0.0)
        self.assertEqual(namespace["t_next"], 0.1)
        self.assertEqual(namespace["dt"], 0.1)
        self.assertEqual(namespace["u_applied"], 1.0)
        self.assertFalse(namespace["history_valid"])
        self.assertIsNone(namespace["du"])
        self.assertEqual(namespace["progress_delta_m"], 1.0)
        self.assertEqual(namespace["preview_invalid_reason"], None)
        self.assertEqual(namespace["r_base"], 2.0)
        self.assertEqual(namespace["r_total"], 2.0)

    def test_baseline_keeps_shape_reward_and_common_timing_metrics(self) -> None:
        raw = FakeRawEnv()
        wrapped = LookaheadEnv(
            raw,
            mode="baseline",
            contract=_contract(raw),
            state_reader=_state,
        )
        observation, _ = wrapped.reset()
        self.assertEqual(observation.shape, (262,))
        next_observation, reward, *_rest = wrapped.step(4)
        self.assertEqual(next_observation.shape, (262,))
        self.assertEqual(reward, 2.0)
        self.assertEqual(wrapped.observation_space.shape, (262,))
        self.assertEqual(raw.step_calls, 1)
        self.assertEqual(wrapped.last_snapshot.t_seconds, 0.1)

    def test_baseline_can_log_shared_preview_without_changing_raw_contract(self) -> None:
        raw = FakeRawEnv()
        preview = PreviewProbe()
        wrapped = LookaheadEnv(
            raw,
            mode="baseline",
            contract=_contract(raw),
            preview_provider=preview,
            state_reader=_state,
        )
        initial, reset_info = wrapped.reset(seed=31)
        self.assertEqual(initial.shape, (262,))
        np.testing.assert_array_equal(initial, np.full(262, 0.25, np.float32))
        self.assertEqual(preview.reset_calls, 1)
        self.assertEqual(preview.calls, 1)
        self.assertTrue(reset_info["lookahead_learning"]["preview_metrics_available"])
        self.assertEqual(reset_info["lookahead_learning"]["preview_valid"], True)
        self.assertEqual(reset_info["lookahead_learning"]["scenario_seed"], 31)
        self.assertEqual(reset_info["lookahead_learning"]["target_lane_valid"], True)
        self.assertAlmostEqual(reset_info["lookahead_learning"]["lateral_error_m"], 0.4)
        self.assertAlmostEqual(reset_info["lookahead_learning"]["heading_error_rad"], 0.15)
        self.assertTrue(reset_info["lookahead_learning"]["start_lane_maintained"])

        next_observation, reward, terminated, truncated, info = wrapped.step(4)
        namespace = info["lookahead_learning"]
        self.assertEqual(next_observation.shape, (262,))
        self.assertEqual(reward, 2.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(preview.calls, 2)
        self.assertEqual(namespace["position_before"], (0.0, 0.0))
        self.assertEqual(namespace["position_after"], (1.0, 0.0))
        self.assertAlmostEqual(namespace["physical_distance_m"], 1.0)
        self.assertAlmostEqual(namespace["S_proj_before"], 0.0)
        self.assertAlmostEqual(namespace["S_proj_after"], 1.0)
        self.assertAlmostEqual(namespace["S_proj_delta_m"], 1.0)
        self.assertEqual(namespace["scenario_seed"], 31)
        self.assertTrue(namespace["preview_metrics_available"])
        self.assertIsNone(namespace["pp_valid"])

    def test_sb3_check_env_accepts_generic_host_wrapper(self) -> None:
        from stable_baselines3.common.env_checker import check_env

        raw = FakeRawEnv()
        wrapped = LookaheadEnv(
            raw,
            mode="lookahead_obs",
            contract=_contract(raw),
            preview_provider=PreviewProbe(),
            state_reader=_state,
        )
        check_env(wrapped, warn=True, skip_render_check=True)

    def test_invalid_host_target_reference_does_not_use_preview_lateral_fallback(self) -> None:
        raw = FakeRawEnv(target_valid=False)
        wrapped = LookaheadEnv(
            raw,
            mode="baseline",
            contract=_contract(raw),
            preview_provider=PreviewProbe(),
            state_reader=_state,
        )
        wrapped.reset()
        _, _, _, _, info = wrapped.step(4)
        namespace = info["lookahead_learning"]
        self.assertFalse(namespace["target_lane_valid"])
        self.assertIsNone(namespace["target_lane_offset_m"])
        self.assertIsNone(namespace["lateral_error_m"])

    def test_b_uses_pre_action_pp_and_masks_terminal_reward(self) -> None:
        raw = FakeRawEnv()
        preview = PreviewProbe()
        pp = PPProbe(u_pp=0.1)
        wrapped = LookaheadEnv(
            raw,
            mode="lookahead_obs_pp_reward",
            contract=_contract(raw),
            preview_provider=preview,
            pp_provider=pp,
            pp_weight=1.0,
            applied_steering_reader=lambda env: 0.3,
            state_reader=_state,
        )
        wrapped.reset()
        _, reward, _, _, info = wrapped.step(4)
        namespace = info["lookahead_learning"]
        self.assertAlmostEqual(reward, 1.99)
        self.assertAlmostEqual(namespace["e_pp"], 0.1)
        self.assertAlmostEqual(namespace["r_pp"], -0.01)
        self.assertAlmostEqual(namespace["r_total"], 1.99)
        self.assertEqual(namespace["u_pp"], 0.1)
        self.assertEqual(namespace["pp_valid"], True)
        self.assertEqual(pp.calls, 2)

        terminal_raw = FakeRawEnv(terminal_after=1)
        terminal_preview = PreviewProbe()
        terminal_pp = PPProbe(u_pp=0.1)
        terminal = LookaheadEnv(
            terminal_raw,
            mode="lookahead_obs_pp_reward",
            contract=_contract(terminal_raw),
            preview_provider=terminal_preview,
            pp_provider=terminal_pp,
            pp_weight=1.0,
            applied_steering_reader=lambda env: 0.3,
            state_reader=_state,
        )
        terminal.reset()
        _, terminal_reward, terminated, truncated, terminal_info = terminal.step(4)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(terminal_reward, 2.0)
        self.assertEqual(terminal_info["lookahead_learning"]["r_pp"], 0.0)
        self.assertIsNone(terminal_info["lookahead_learning"]["e_pp"])
        with self.assertRaises(RuntimeError):
            terminal.step(4)

    def test_zero_weight_ab_has_identical_observation_reward_and_state(self) -> None:
        raw_a = FakeRawEnv()
        raw_b = FakeRawEnv()
        preview_a = PreviewProbe()
        preview_b = PreviewProbe()
        a = LookaheadEnv(
            raw_a,
            mode="lookahead_obs",
            contract=_contract(raw_a),
            preview_provider=preview_a,
            state_reader=_state,
        )
        b = LookaheadEnv(
            raw_b,
            mode="lookahead_obs_pp_reward",
            contract=_contract(raw_b),
            preview_provider=preview_b,
            pp_provider=PPProbe(u_pp=0.1),
            pp_weight=0.0,
            state_reader=_state,
        )
        observation_a, _ = a.reset(seed=19)
        observation_b, _ = b.reset(seed=19)
        np.testing.assert_array_equal(observation_a, observation_b)
        step_a = a.step(5)
        step_b = b.step(5)
        np.testing.assert_array_equal(step_a[0], step_b[0])
        self.assertEqual(step_a[1], step_b[1])
        self.assertEqual(step_a[2:4], step_b[2:4])
        info_a = step_a[4]["lookahead_learning"]
        info_b = step_b[4]["lookahead_learning"]
        for key in ("position", "heading", "speed_mps", "progress_m", "state_after"):
            self.assertEqual(info_a[key], info_b[key])
        self.assertEqual(info_b["r_pp"], 0.0)

    def test_positive_weight_changes_only_reward_for_matched_transition(self) -> None:
        raw_a = FakeRawEnv()
        raw_b = FakeRawEnv()
        a = LookaheadEnv(
            raw_a,
            mode="lookahead_obs",
            contract=_contract(raw_a),
            preview_provider=PreviewProbe(),
            applied_steering_reader=lambda env: 0.3,
            state_reader=_state,
        )
        b = LookaheadEnv(
            raw_b,
            mode="lookahead_obs_pp_reward",
            contract=_contract(raw_b),
            preview_provider=PreviewProbe(),
            pp_provider=PPProbe(u_pp=0.1),
            pp_weight=1.0,
            applied_steering_reader=lambda env: 0.3,
            state_reader=_state,
        )
        a.reset(seed=23)
        b.reset(seed=23)
        step_a = a.step(4)
        step_b = b.step(4)
        np.testing.assert_array_equal(step_a[0], step_b[0])
        self.assertAlmostEqual(step_a[1] - step_b[1], 0.01)
        info_a = step_a[4]["lookahead_learning"]
        info_b = step_b[4]["lookahead_learning"]
        self.assertEqual(info_a["state_after"], info_b["state_after"])
        self.assertEqual(info_a["position"], info_b["position"])
        self.assertAlmostEqual(info_b["r_total"], info_a["r_total"] - 0.01)

    def test_monitor_episode_return_equals_returned_rewards(self) -> None:
        from stable_baselines3.common.monitor import Monitor

        raw = FakeRawEnv(terminal_after=2)
        wrapped = LookaheadEnv(
            raw,
            mode="lookahead_obs_pp_reward",
            contract=_contract(raw),
            preview_provider=PreviewProbe(),
            pp_provider=PPProbe(u_pp=0.1),
            pp_weight=1.0,
            applied_steering_reader=lambda env: 0.3,
            state_reader=_state,
        )
        # A filename is optional; omitting it keeps this contract test
        # read-only in restricted CI workspaces.
        monitored = Monitor(wrapped)
        monitored.reset()
        returned_total = 0.0
        terminal_info = None
        for _ in range(2):
            _, reward, terminated, truncated, terminal_info = monitored.step(4)
            returned_total += reward
            if terminated or truncated:
                break
        assert terminal_info is not None
        self.assertAlmostEqual(terminal_info["episode"]["r"], returned_total)
        self.assertAlmostEqual(
            terminal_info["lookahead_learning"]["episode_r_total"], returned_total
        )

    def test_vecenv_keeps_terminal_observation_separate_from_autoreset(self) -> None:
        from stable_baselines3.common.vec_env import DummyVecEnv

        raw_holder: list[FakeRawEnv] = []
        provider_holder: list[PreviewProbe] = []

        def make():
            raw = FakeRawEnv(terminal_after=1)
            provider = PreviewProbe()
            raw_holder.append(raw)
            provider_holder.append(provider)
            return LookaheadEnv(
                raw,
                mode="lookahead_obs",
                contract=_contract(raw),
                preview_provider=provider,
                state_reader=_state,
            )

        vec = DummyVecEnv([make])
        initial = vec.reset()
        stepped, rewards, dones, infos = vec.step(np.array([4]))
        self.assertTrue(dones[0])
        self.assertEqual(infos[0]["terminal_observation"].shape, (265,))
        self.assertEqual(stepped.shape, (1, 265))
        # PreviewProbe uses x=7 after the transition and x=6 after the
        # autoreset, so terminal and reset observations are distinguishable.
        self.assertAlmostEqual(infos[0]["terminal_observation"][262], 0.85)
        self.assertAlmostEqual(stepped[0, 262], initial[0, 262])
        self.assertEqual(raw_holder[0].reset_calls, 2)
        self.assertEqual(raw_holder[0].step_calls, 0)
        self.assertEqual(provider_holder[0].calls, 3)
        vec.close()

    def test_truncated_transition_masks_pp_and_preserves_pre_snapshot(self) -> None:
        raw = FakeRawEnv(truncated_after=1)
        wrapped = LookaheadEnv(
            raw,
            mode="lookahead_obs_pp_reward",
            contract=_contract(raw),
            preview_provider=PreviewProbe(invalid_after=1),
            pp_provider=lambda env, preview: {
                "pp_valid": preview.preview_valid,
                "u_pp": 0.1 if preview.preview_valid else None,
            },
            pp_weight=1.0,
            applied_steering_reader=lambda env: 0.3,
            state_reader=_state,
        )
        wrapped.reset()
        _, reward, terminated, truncated, info = wrapped.step(4)
        namespace = info["lookahead_learning"]
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(reward, 2.0)
        self.assertTrue(namespace["pre_step"]["preview"]["preview_valid"])
        self.assertFalse(namespace["post_step"]["preview"]["preview_valid"])
        self.assertEqual(namespace["pp_valid"], True)
        self.assertIsNone(namespace["e_pp"])
        self.assertEqual(namespace["r_pp"], 0.0)

    def test_repeated_snapshot_access_does_not_resample_raw_or_provider(self) -> None:
        raw = FakeRawEnv()
        provider = PreviewProbe()
        wrapped = LookaheadEnv(
            raw,
            mode="lookahead_obs",
            contract=_contract(raw),
            preview_provider=provider,
            state_reader=_state,
        )
        wrapped.reset()
        snapshot = wrapped.last_snapshot
        self.assertEqual(provider.calls, 1)
        self.assertEqual(raw.step_calls, 0)
        self.assertIs(snapshot, wrapped.last_snapshot)
        self.assertEqual(provider.calls, 1)
        observation, *_ = wrapped.step(4)
        self.assertEqual(provider.calls, 2)
        self.assertEqual(raw.step_calls, 1)
        np.testing.assert_allclose(observation[262:], [0.85, 0.5, 1.0])

    def test_first_action_history_is_missing_then_second_is_defined(self) -> None:
        raw = FakeRawEnv()
        wrapped = LookaheadEnv(
            raw,
            mode="baseline",
            contract=_contract(raw),
            state_reader=_state,
        )
        wrapped.reset()
        _, _, _, _, first = wrapped.step(4)
        _, _, _, _, second = wrapped.step(8)
        self.assertFalse(first["lookahead_learning"]["history_valid"])
        self.assertTrue(second["lookahead_learning"]["history_valid"])
        self.assertAlmostEqual(second["lookahead_learning"]["u_previous"], 0.0)
        self.assertAlmostEqual(second["lookahead_learning"]["u_applied"], 1.0)
        self.assertAlmostEqual(second["lookahead_learning"]["du"], 1.0)
        self.assertAlmostEqual(second["lookahead_learning"]["steering_rate"], 10.0)
        self.assertAlmostEqual(second["lookahead_learning"]["yaw_rate"], 0.2)


class ContractAndNavigationTests(unittest.TestCase):
    def test_provider_passes_lookahead_distance_to_geometry(self) -> None:
        lane = StraightLane((0.0, 0.0), (20.0, 0.0), lane_index=("A", "B", 0))
        navigation = FakeNavigation(("A", "B"), {"A": {"B": [lane]}})
        vehicle = FakeVehicle()
        vehicle.navigation = navigation
        env = type("Env", (), {"agents": {"agent0": vehicle}})()

        short = MetaDrivePreviewProvider(lookahead_m=3.0)
        long = MetaDrivePreviewProvider(lookahead_m=6.0)
        short.reset(env)
        long.reset(env)
        short_result = short(env)
        long_result = long(env)
        self.assertTrue(short_result["preview_valid"])
        self.assertTrue(long_result["preview_valid"])
        self.assertAlmostEqual(short_result["s_goal_m"], 3.0)
        self.assertAlmostEqual(long_result["s_goal_m"], 6.0)
        self.assertNotEqual(short_result["q_xy"], long_result["q_xy"])

    def test_host_width_is_generic_and_prefix_is_preserved(self) -> None:
        for width in (259, 262):
            space = gym.spaces.Box(
                low=np.full(width, -2.0, dtype=np.float32),
                high=np.full(width, 3.0, dtype=np.float32),
                dtype=np.float32,
            )
            contract = ObservationContract.from_space(space)
            # Keep values inside this test Box while retaining a distinct
            # prefix value at every position.
            raw = np.linspace(-1.0, 2.0, width, dtype=np.float32)
            augmented = contract.append_preview(raw, (0.2, 0.8, 1.0))
            self.assertEqual(contract.shape, (width,))
            self.assertEqual(contract.augmented_space().shape, (width + 3,))
            np.testing.assert_array_equal(augmented[:width], raw)
            np.testing.assert_allclose(augmented[width:], (0.2, 0.8, 1.0))

    def test_non_vector_observation_is_rejected(self) -> None:
        image = gym.spaces.Box(
            low=np.zeros((2, 2), dtype=np.float32),
            high=np.ones((2, 2), dtype=np.float32),
            dtype=np.float32,
        )
        with self.assertRaises(UnsupportedHostError):
            ObservationContract.from_space(image)

    def test_nonzero_discrete_start_is_rejected(self) -> None:
        raw = FakeRawEnv()
        nonzero = gym.spaces.Discrete(9, start=1)
        with self.assertRaises(UnsupportedHostError):
            ActionContract.from_space(nonzero, raw.config)

    def test_navigation_route_uses_planned_edge_and_start_ordinal(self) -> None:
        first = StraightLane((0.0, 0.0), (3.0, 0.0), lane_index=("A", "B", 0))
        second = StraightLane((3.0, 0.0), (8.0, 0.0), lane_index=("B", "C", 0))
        first_other = StraightLane(
            (0.0, 3.5), (3.0, 3.5), lane_index=("A", "B", 1)
        )
        second_other = StraightLane(
            (3.0, 3.5), (8.0, 3.5), lane_index=("B", "C", 1)
        )
        navigation = FakeNavigation(
            ("A", "B", "C"),
            {"A": {"B": [first, first_other]}, "B": {"C": [second, second_other]}},
        )
        vehicle = FakeVehicle()
        vehicle.navigation = navigation
        vehicle.lane_index = ("A", "B", 0)
        env = type("Env", (), {"agents": {"agent0": vehicle}})()
        lanes, ordinal, checkpoints = navigation_lane_sequence(env)
        self.assertEqual(lanes, (first, second))
        self.assertEqual(ordinal, 0)
        self.assertEqual(checkpoints, ("A", "B", "C"))
        built = build_fixed_navigation_route(env)
        self.assertTrue(built.valid)
        self.assertEqual(built.path.lanes, (first, second))
        self.assertEqual(built.path.total_length, 8.0)
        self.assertEqual(built.path.metadata[0].lane_count, 2)

    def test_navigation_ambiguous_ordinal_is_rejected_without_fallback(self) -> None:
        first = StraightLane((0.0, 0.0), (3.0, 0.0), lane_index=("A", "B", 0))
        duplicate = StraightLane((0.0, 1.0), (3.0, 1.0), lane_index=("A", "B", 0))
        navigation = FakeNavigation(("A", "B"), {"A": {"B": [first, duplicate]}})
        vehicle = FakeVehicle()
        vehicle.navigation = navigation
        vehicle.lane_index = ("A", "B", 0)
        env = type("Env", (), {"agents": {"agent0": vehicle}})()
        with self.assertRaises(UnsupportedHostError):
            navigation_lane_sequence(env)

    def test_navigation_builder_keeps_verified_prefix_at_late_missing_edge(self) -> None:
        first = StraightLane((0.0, 0.0), (3.0, 0.0), lane_index=("A", "B", 0))
        navigation = FakeNavigation(("A", "B", "C"), {"A": {"B": [first]}})
        vehicle = FakeVehicle()
        vehicle.navigation = navigation
        vehicle.lane_index = ("A", "B", 0)
        env = type("Env", (), {"agents": {"agent0": vehicle}})()
        built = build_fixed_navigation_route(env)
        self.assertTrue(built.valid)
        self.assertEqual(built.path.lanes, (first,))
        self.assertEqual(built.diagnostics.boundary_reason, "navigation_edge_unavailable")
        self.assertEqual(built.diagnostics.boundary_lane_index, 1)

    def test_navigation_unsupported_first_lane_retains_boundary_reason(self) -> None:
        unsupported = PointLane((0.0, 0.0), (3.0, 0.0), lane_index=("A", "B", 0))
        navigation = FakeNavigation(("A", "B"), {"A": {"B": [unsupported]}})
        vehicle = FakeVehicle()
        vehicle.navigation = navigation
        vehicle.lane_index = ("A", "B", 0)
        env = type("Env", (), {"agents": {"agent0": vehicle}})()
        built = build_fixed_navigation_route(env)
        self.assertFalse(built.valid)
        self.assertEqual(built.diagnostics.boundary_reason, "unsupported_lane_type")
        self.assertEqual(built.diagnostics.boundary_lane_index, 0)

    def test_saved_reset_ordinal_wins_over_later_vehicle_lane_index(self) -> None:
        first = StraightLane((0.0, 0.0), (3.0, 0.0), lane_index=("A", "B", 0))
        other = StraightLane((0.0, 3.5), (3.0, 3.5), lane_index=("A", "B", 1))
        navigation = FakeNavigation(("A", "B"), {"A": {"B": [first, other]}})
        vehicle = FakeVehicle()
        vehicle.navigation = navigation
        vehicle.lane_index = ("A", "B", 1)
        env = type(
            "Env",
            (),
            {
                "agents": {"agent0": vehicle},
                "_target_lane_ordinals": {"agent0": 0},
            },
        )()
        lanes, ordinal, _ = navigation_lane_sequence(env)
        self.assertEqual(ordinal, 0)
        self.assertEqual(lanes, (first,))


class FakeNavigation:
    def __init__(self, checkpoints, graph):
        self.checkpoints = checkpoints
        self.map = type(
            "Map",
            (),
            {"road_network": type("RoadNetwork", (), {"graph": graph})()},
        )()


class StraightLane:
    def __init__(self, start, end, *, lane_index):
        self.start = start
        self.end = end
        self.index = lane_index
        self.width = 3.5
        self.length = math.dist(start, end)

    def position(self, s, lateral):
        del lateral
        ratio = float(s) / self.length
        return (
            self.start[0] + ratio * (self.end[0] - self.start[0]),
            self.start[1] + ratio * (self.end[1] - self.start[1]),
        )

    def local_coordinates(self, point):
        dx = point[0] - self.start[0]
        dy = point[1] - self.start[1]
        ux = (self.end[0] - self.start[0]) / self.length
        uy = (self.end[1] - self.start[1]) / self.length
        return (dx * ux + dy * uy, -dx * uy + dy * ux)

    def heading_theta_at(self, s):
        del s
        return math.atan2(
            self.end[1] - self.start[1],
            self.end[0] - self.start[0],
        )


class PointLane(StraightLane):
    """Unsupported lane class used to ensure type checks are conservative."""

    pass


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
