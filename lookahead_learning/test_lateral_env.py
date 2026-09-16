"""Fake-host acceptance tests: timing, masks, composition and portable wiring."""

import json
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import gymnasium as gym
import numpy as np

from .adapter import HostContractError, MetaDrivePreviewProvider, wrap_lookahead_env
from .checkpoint import resolve_lookahead_config, set_lookahead_model_metadata, validate_lookahead_model_metadata
from .env import LookaheadEnv
from .lateral_acceleration import LateralReference, planar_speed_mps
from .test_env import FakeNavigation, FakeRawEnv, PPProbe, PreviewProbe, _contract, _state
from .test_lateral_acceleration import mixed_route


class SpeedHost(FakeRawEnv):
    def __init__(self, *, speed=10.0, **kwargs):
        super().__init__(observation_dim=7, **kwargs)
        self.post_speed = speed

    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        self.vehicle.speed = 1.0
        return result

    def step(self, action):
        result = super().step(action)
        self.vehicle.speed = self.post_speed
        return result


class CurvePreview(PreviewProbe):
    def __init__(self, *, next_curvature=0.02, invalid_at=None):
        super().__init__()
        self.next_curvature = next_curvature
        self.invalid_at = invalid_at

    def __call__(self, env):
        host = env.unwrapped
        result = dict(super().__call__(host))
        valid = host.step_calls != self.invalid_at
        result["lateral_reference"] = LateralReference(
            valid, result["s_proj_m"], result["s_goal_m"],
            (0.02 if host.step_calls == 0 else self.next_curvature) if valid else None,
            None if valid else "ambiguous_projection",
        )
        return result


def make_wrapped(*, enabled=True, weight=0.1, pp_weight=0.0, provider=None, raw=None, **kwargs):
    host = raw or SpeedHost()
    return LookaheadEnv(
        host, mode="lookahead_obs_pp_reward" if pp_weight else "lookahead_obs",
        contract=_contract(host), preview_provider=provider or CurvePreview(),
        pp_provider=PPProbe() if pp_weight else None, pp_weight=pp_weight,
        state_reader=kwargs.pop("state_reader", _state),
        lateral_accel_reward_enabled=enabled, lateral_accel_weight=weight,
        **kwargs,
    )


def navigable_host():
    host = SpeedHost()
    route = mixed_route()
    graph = {}
    for index, lane in enumerate(route.path.lanes):
        lane.index = (str(index), str(index + 1), 0)
        graph[str(index)] = {str(index + 1): [lane]}
    host.vehicle.lane_index = ("0", "1", 0)
    host.vehicle.navigation = FakeNavigation(("0", "1", "2", "3"), graph)
    return host


class LateralWrapperTests(unittest.TestCase):
    def test_old_constructor_off_and_zero_weight_preserve_transition(self):
        legacy = make_wrapped(enabled=False, provider=PreviewProbe())
        off = make_wrapped(enabled=False, weight=4, provider=PreviewProbe(), max_lateral_accel=9)
        zero = make_wrapped(enabled=True, weight=0, provider=PreviewProbe())
        envs = (legacy, off, zero)
        initial = [env.reset(seed=42)[0] for env in envs]
        for observation in initial:
            np.testing.assert_array_equal(observation, initial[0])
        for _ in range(3):
            results = [env.step(5) for env in envs]
            for result in results:
                np.testing.assert_array_equal(result[0], results[0][0])
                self.assertEqual(result[1:4], results[0][1:4])
                self.assertEqual(result[4]["lookahead_learning"]["r_lateral_accel"], 0)
        self.assertEqual(zero.last_snapshot.preview.lateral_reference, None)

    def test_four_combinations_return_once_and_keep_observations(self):
        observations = []
        for pp_on in (False, True):
            for lateral_on in (False, True):
                with self.subTest(pp=pp_on, lateral=lateral_on):
                    env = make_wrapped(enabled=lateral_on, pp_weight=1.0 if pp_on else 0)
                    env.reset()
                    obs, reward, terminated, truncated, info = env.step(4)
                    data = info["lookahead_learning"]
                    expected_pp = -0.005 if pp_on else 0
                    expected_lat = -0.0225 if lateral_on else 0
                    self.assertAlmostEqual(data["r_pp"], expected_pp)
                    self.assertAlmostEqual(data["r_lateral_accel"], expected_lat)
                    self.assertAlmostEqual(reward, 2 + expected_pp + expected_lat)
                    self.assertEqual(data["r_total"], reward)
                    self.assertEqual(env.env.step_calls, 1)
                    self.assertFalse(terminated or truncated)
                    self.assertEqual(obs.shape, (10,))
                    observations.append(obs)
                    json.dumps(info, allow_nan=False)
        for obs in observations:
            np.testing.assert_array_equal(obs, observations[0])

    def test_pre_curvature_post_speed_and_one_provider_call_per_state(self):
        provider = CurvePreview(next_curvature=0)
        env = make_wrapped(provider=provider)
        env.reset()
        before = env.last_snapshot
        self.assertEqual(dict(before.state)["speed_m_s"], 1)
        _, reward, *_, info = env.step(4)
        diagnostic = info["lookahead_learning"]["lateral_accel"]
        self.assertAlmostEqual(reward, 1.9775)
        self.assertEqual(diagnostic["kappa_abs_max_inv_m"], 0.02)
        self.assertEqual(diagnostic["speed_post_mps"], 10)
        self.assertEqual(diagnostic["s_proj_m"], 0)
        self.assertEqual(env.last_snapshot.preview.lateral_reference.kappa_abs_max_inv_m, 0)
        self.assertEqual(provider.calls, 2)
        self.assertEqual(before.preview.lateral_reference.kappa_abs_max_inv_m, 0.02)
        self.assertEqual(env.step(4)[1], 2)
        self.assertEqual(provider.calls, 3)

    def test_pp_validity_and_curvature_do_not_control_lateral_term(self):
        rewards = []
        for pp_valid, kappa in ((True, -8.0), (True, 8.0), (False, None)):
            env = make_wrapped(pp_weight=1)
            def pp(_env, _preview):
                return {"pp_valid": pp_valid, "u_pp": 0 if pp_valid else None, "kappa_pp": kappa}
            env._pp_provider = pp
            env.reset()
            info = env.step(4)[4]["lookahead_learning"]
            self.assertTrue(info["lateral_accel"]["active"])
            rewards.append(info["r_lateral_accel"])
        for reward in rewards:
            self.assertAlmostEqual(reward, -0.0225)

    def test_reference_invalidity_is_independent_and_recovers_on_next_action(self):
        env = make_wrapped(provider=CurvePreview(invalid_at=0), pp_weight=1)
        env.reset()
        first = env.step(4)[4]["lookahead_learning"]
        self.assertEqual(first["r_lateral_accel"], 0)
        self.assertLess(first["r_pp"], 0)
        self.assertEqual(first["lateral_accel"]["reference_invalid_reason"], "ambiguous_projection")
        second = env.step(4)[4]["lookahead_learning"]
        self.assertLess(second["r_lateral_accel"], 0)
        metrics = second["lateral_accel_episode"]
        self.assertEqual(metrics["reference_invalid_time_ratio"], 0.5)
        self.assertEqual(metrics["exceedance_time_ratio"], 1)
        self.assertEqual(metrics["required_lateral_accel_max_mps2"], 2)

    def test_terminal_and_truncated_steps_reset_and_episode_metrics(self):
        for ending in ("terminal_after", "truncated_after"):
            env = make_wrapped(raw=SpeedHost(**{ending: 2}), pp_weight=1)
            env.reset()
            total = env.step(4)[1]
            result = env.step(4)
            total += result[1]
            self.assertEqual(result[1], 2)
            self.assertTrue(result[2] or result[3])
            info = result[4]["lookahead_learning"]
            self.assertEqual(info["lateral_accel"]["skip_reason"], "episode_end")
            self.assertEqual(info["episode_r_total"], total)
            self.assertAlmostEqual(info["episode_r_lateral_accel"], -0.0225)
            self.assertEqual(info["lateral_accel_episode"]["evaluated_seconds"], 0.1)
            self.assertIsNotNone(env.terminal_snapshot)
            with self.assertRaises(RuntimeError):
                env.step(4)
            _, reset_info = env.reset()
            self.assertEqual(env.episode_totals, {
                "r_base": 0, "r_pp": 0, "r_lateral_accel": 0, "r_total": 0
            })
            self.assertEqual(reset_info["lookahead_learning"]["lateral_accel"]["skip_reason"], "reset")
            self.assertIsNone(env.terminal_snapshot)

    def test_stop_and_temporary_speed_unavailability(self):
        for speed in (0, None):
            env = make_wrapped(raw=SpeedHost(speed=speed))
            env.reset()
            _, reward, *_, info = env.step(4)
            self.assertEqual(reward, 2)
            data = info["lookahead_learning"]["lateral_accel"]
            self.assertEqual(data["active"], speed == 0)
            self.assertEqual(data["skip_reason"], None if speed == 0 else "speed_unavailable")
        default_reader = make_wrapped(raw=SpeedHost(speed=None), state_reader=None)
        default_reader.reset()
        self.assertEqual(default_reader.step(4)[4]["lookahead_learning"]["lateral_accel"]["skip_reason"],
                         "speed_unavailable")

    def test_invalid_post_reference_does_not_mask_the_valid_pre_interval(self):
        env = make_wrapped(provider=CurvePreview(invalid_at=1))
        env.reset()
        self.assertAlmostEqual(env.step(4)[1], 1.9775)
        self.assertFalse(env.last_snapshot.preview.lateral_reference.valid)
        self.assertEqual(env.step(4)[1], 2.0)

    def test_speed_contract_rejects_missing_nan_inf_string_bool_and_negative(self):
        with self.assertRaises(HostContractError):
            make_wrapped(state_reader=lambda env: {}).reset()
        with self.assertRaises(HostContractError):
            make_wrapped(state_reader=lambda env: {**_state(env), "speed_unit": "km/h"}).reset()
        for speed in (math.nan, math.inf, -math.inf, "10", True, -1):
            env = make_wrapped(raw=SpeedHost(speed=speed))
            env.reset()
            with self.subTest(speed=speed), self.assertRaises(HostContractError):
                env.step(4)
        with self.assertRaises(HostContractError):
            make_wrapped(provider=PreviewProbe()).reset()

    def test_explicit_kmh_state_reader_matches_mps(self):
        a = make_wrapped()
        b = make_wrapped(state_reader=lambda env: {
            **_state(env), "speed_m_s": planar_speed_mps(env.vehicle.speed * 3.6, unit="km/h")
        })
        a.reset()
        b.reset()
        self.assertEqual(a.step(4)[1], b.step(4)[1])

    def test_monitor_and_multi_env_autoreset_keep_totals_and_snapshots_separate(self):
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv

        envs = [
            make_wrapped(raw=SpeedHost(speed=10, terminal_after=2)),
            make_wrapped(raw=SpeedHost(speed=5, terminal_after=3)),
        ]
        vec = DummyVecEnv([lambda env=env: Monitor(env) for env in envs])
        try:
            vec.reset()
            _, rewards, dones, _ = vec.step(np.array([4, 4]))
            self.assertAlmostEqual(float(rewards[0]), 1.9775, places=6)
            self.assertEqual(float(rewards[1]), 2)
            self.assertFalse(any(dones))
            _, _, dones, infos = vec.step(np.array([4, 4]))
            self.assertEqual(list(dones), [True, False])
            first = infos[0]["lookahead_learning"]
            self.assertAlmostEqual(infos[0]["episode"]["r"], 3.9775)
            self.assertEqual(first["episode_r_total"], infos[0]["episode"]["r"])
            self.assertEqual(envs[0].last_snapshot.decision, 0)
            self.assertEqual(envs[1].last_snapshot.decision, 2)
            self.assertEqual(envs[0].episode_totals["r_total"], 0)
        finally:
            vec.close()

    def test_host_wrapper_chain_is_used_for_reset_and_step(self):
        class HostWrapper(gym.Wrapper):
            reset_calls = 0
            step_calls = 0

            def reset(self, **kwargs):
                self.reset_calls += 1
                return self.env.reset(**kwargs)

            def step(self, action):
                self.step_calls += 1
                obs, reward, term, trunc, info = self.env.step(action)
                return obs, reward + 3, term, trunc, info

        chain = HostWrapper(SpeedHost())
        env = make_wrapped(
            raw=chain, state_reader=lambda env: _state(env.unwrapped),
            applied_action_reader=lambda env: env.unwrapped.vehicle.current_action,
            dt_reader=lambda env: 0.1,
        )
        env.reset()
        self.assertAlmostEqual(env.step(4)[1], 4.9775)
        self.assertEqual((chain.reset_calls, chain.step_calls), (1, 1))
        self.assertEqual((chain.unwrapped.reset_calls, chain.unwrapped.step_calls), (1, 1))


class HostConnectionFixtures(unittest.TestCase):
    def test_new_old_and_updated_host_connections(self):
        # These are minimal host hookup states, not a Copilot/real port claim.
        for kind in ("new", "old", "updated"):
            config = resolve_lookahead_config({
                "lateral_accel_reward_enabled": kind != "old"
            })
            old_config = {"lookahead_m": 6.0, "pp_weight": 0.0}
            passed_config = old_config if kind == "old" else config
            with self.subTest(kind=kind), patch(
                "lookahead_learning.adapter.MetaDrivePPProvider",
                side_effect=AssertionError("PP Off must not construct a PP provider"),
            ):
                raw = navigable_host()
                env = wrap_lookahead_env(raw, **passed_config)
                model = SimpleNamespace()
                if kind == "old":
                    model.lookahead_config, model.lookahead_schema_version = old_config.copy(), 1
                else:
                    set_lookahead_model_metadata(model, passed_config)
                validate_lookahead_model_metadata(model, config)
                obs, _ = env.reset()
                self.assertEqual(obs.shape, (10,))
                result = env.step(4)
                self.assertAlmostEqual(result[1], 2 if kind == "old" else 1.9775)
                self.assertEqual(raw.step_calls, 1)
                with self.assertRaisesRegex(HostContractError, "already connected"):
                    wrap_lookahead_env(env, **config)
                env.close()

    def test_factory_off_zero_weight_and_on_toggle_keep_prefix(self):
        results = []
        for enabled, weight in ((False, 0.1), (True, 0), (True, 0.1)):
            env = wrap_lookahead_env(
                navigable_host(), lateral_accel_reward_enabled=enabled,
                lateral_accel_weight=weight,
            )
            env.reset()
            results.append(env.step(4))
            env.close()
        for result in results:
            np.testing.assert_array_equal(result[0], results[0][0])
            self.assertEqual(result[2:4], results[0][2:4])
        self.assertEqual(results[0][1], results[1][1])
        self.assertAlmostEqual(results[2][1], 1.9775)

    def test_on_validates_radius_at_reset_while_off_requires_no_radius_api(self):
        # A host adapter whose circular position API works, but radius is not
        # exposed. Legacy/zero-weight calls must never request that new API.
        class CircularLane:
            def __init__(self, wrapped):
                self.wrapped = wrapped

            def __getattr__(self, name):
                if name == "radius":
                    raise AttributeError("radius API unavailable")
                return getattr(self.wrapped, name)

        for enabled, weight in ((False, 0.1), (True, 0), (True, 0.1)):
            raw = navigable_host()
            lanes = raw.vehicle.navigation.map.road_network.graph["1"]["2"]
            lanes[0] = CircularLane(lanes[0])
            env = wrap_lookahead_env(
                raw, lateral_accel_reward_enabled=enabled, lateral_accel_weight=weight
            )
            if enabled and weight > 0:
                with self.assertRaises(ValueError):
                    env.reset()
            else:
                env.reset()
                self.assertEqual(env.step(4)[1], 2)
        raw = navigable_host()
        arc = raw.vehicle.navigation.map.road_network.graph["1"]["2"][0]
        # Geometry stays valid; an explicitly wrong radius adapter fails when On.
        provider = MetaDrivePreviewProvider(
            include_lateral_accel=True, radius_reader=lambda lane: math.nan
        )
        self.assertIsNotNone(arc.radius)
        with self.assertRaises(ValueError):
            provider.reset(raw)


if __name__ == "__main__":
    unittest.main()
