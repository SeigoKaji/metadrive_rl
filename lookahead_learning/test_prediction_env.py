"""Fake-host acceptance: actual provider, wrapper, RNG, Monitor and history."""
import json
import math
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np

from .adapter import HostContractError, MetaDrivePreviewProvider, read_vehicle_state, wrap_lookahead_env
from .env import LookaheadEnv
from .test_env import FakeRawEnv, FakeNavigation, StraightLane, _contract


class MotionHost(FakeRawEnv):
    def __init__(self, *, speed=10., offset=0., headings=(0.,), velocities=None,
                 positions=None, length=1000., **kwargs):
        super().__init__(observation_dim=kwargs.pop('observation_dim', 7), **kwargs)
        self.speed = speed
        self.offset = offset
        self.headings = headings
        self.velocities = velocities
        self.positions = positions
        lane = StraightLane((0., 0.), (length, 0.), lane_index=('A', 'B', 0))
        self.vehicle.navigation = FakeNavigation(('A', 'B'), {'A': {'B': [lane]}})

    def _motion(self):
        i = self.step_calls
        psi = self.headings[min(i, len(self.headings) - 1)]
        self.vehicle.heading_theta = psi
        if self.velocities is None:
            velocity = (self.speed * math.cos(psi), self.speed * math.sin(psi))
        else:
            velocity = self.velocities[min(i, len(self.velocities) - 1)]
        self.vehicle.velocity = np.array(velocity)
        self.vehicle.speed = math.hypot(*velocity)
        pos = ((10. + i * self.speed * 0.1, self.offset) if self.positions is None
               else self.positions[min(i, len(self.positions) - 1)])
        self.vehicle.position[:2] = pos
        self.vehicle.navigation.travelled_length = float(pos[0])

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        self._motion()
        return observation, info

    def step(self, action):
        _, reward, term, trunc, info = super().step(action)
        self._motion()
        # Equality below checks RNG consumption, not only deterministic constants.
        obs = self.np_random.random(self.observation_dim).astype(np.float32)
        return obs, reward, term, trunc, info


def wrap(raw=None, **kwargs):
    return wrap_lookahead_env(raw or MotionHost(), **{
        'lookahead_time_s': 1., 'prediction_reward_enabled': True, **kwargs,
    })


class TimePreviewTests(unittest.TestCase):
    def test_variable_distance_uses_magnitude_and_time_over_distance(self):
        env = wrap(MotionHost(velocities=((6., 8.),)), lookahead_m=200)
        obs, info = env.reset()
        data = info['lookahead_learning']
        self.assertEqual(data['effective_lookahead_m'], 10)
        self.assertEqual(data['lookahead_priority'], 'time_over_distance')
        self.assertEqual(env.last_snapshot.preview.q_xy, (20, 0))
        self.assertEqual(dict(env.last_snapshot.state)['forward_speed_mps'], 6)
        self.assertEqual(dict(env.last_snapshot.state)['speed_m_s'], 10)
        self.assertEqual(obs.shape, (10,))

    def test_stop_reverse_and_past_end_reuse_existing_invalid_observation(self):
        cases = [
            (MotionHost(speed=0), 'goal_not_in_front'),
            (MotionHost(velocities=((-10., 0.),)), 'reverse_motion'),
            (MotionHost(length=15), 'lookahead_past_route_end'),
        ]
        for raw, reason in cases:
            with self.subTest(reason=reason):
                env = wrap(raw)
                obs, _ = env.reset()
                np.testing.assert_array_equal(obs[-3:], (0.5, 0.5, 0))
                self.assertEqual(env.last_snapshot.preview.invalid_reason, reason)
        # No artificial minimum distance at rest, no end clamp or extrapolation.
        env = wrap(MotionHost(speed=0))
        self.assertEqual(env.reset()[1]['lookahead_learning']['effective_lookahead_m'], 0)

    def test_distance_legacy_and_disabled_zero_need_no_new_velocity_api(self):
        envs = []
        for config in ({}, {'prediction_reward_enabled': False, 'prediction_reward_weight': 99},
                       {'prediction_reward_enabled': True, 'prediction_reward_weight': 0}):
            raw = FakeRawEnv()
            lane = StraightLane((0, 0), (100, 0), lane_index=('A', 'B', 0))
            raw.vehicle.navigation = FakeNavigation(('A', 'B'), {'A': {'B': [lane]}})
            envs.append(wrap_lookahead_env(raw, **config))
        for env in envs:
            env.reset(seed=42)
        for action in (4, 5, 3):
            results = [env.step(action) for env in envs]
            for result in results:
                np.testing.assert_array_equal(result[0], results[0][0])
                self.assertEqual(result[1:4], results[0][1:4])

    def test_contract_errors_are_distinct_from_geometric_invalidity(self):
        for bad in (math.nan, math.inf, True, '10'):
            env = wrap()
            env.reset()
            env.env.velocities = ((bad, 0),)
            with self.subTest(bad=bad), self.assertRaises(HostContractError):
                # Direct reader also audits scalar types before conversion.
                env.env.vehicle.velocity = [bad, 0]
                read_vehicle_state(env.env, require_motion=True)
        raw = MotionHost()
        raw.reset()
        del raw.vehicle.velocity
        provider = MetaDrivePreviewProvider(lookahead_time_s=1)
        provider.reset(raw)
        with self.assertRaises(HostContractError):
            provider(raw)


class PredictionWrapperTests(unittest.TestCase):
    def test_post_goal_first_step_one_addition_and_recomputable_log(self):
        env = wrap(MotionHost(offset=1))
        env.reset()
        pre_goal = env.last_snapshot.preview.q_xy
        obs, reward, term, trunc, info = env.step(4)
        data = info['lookahead_learning']
        diagnostic = data['prediction']
        self.assertEqual(pre_goal, (20, 0))
        self.assertEqual(data['q'], (20, 0))  # existing pre reference is unchanged
        self.assertEqual(diagnostic['goal_xy'], (21, 0))
        self.assertEqual(data['prediction_goal_xy'], (21, 0))
        self.assertEqual(diagnostic['predicted_xy'], (21, 1))
        self.assertEqual(diagnostic['error_m'], 1)
        self.assertAlmostEqual(diagnostic['origin_time_s'], 0.1)
        self.assertAlmostEqual(diagnostic['target_time_s'], 1.1)
        self.assertAlmostEqual(reward, 1.99)
        self.assertEqual(data['r_total'], reward)
        self.assertAlmostEqual(diagnostic['reward'], -diagnostic['weight'] * data['dt']
                               * math.dist(diagnostic['goal_xy'], diagnostic['predicted_xy'])
                               / diagnostic['error_scale_m'])
        self.assertFalse(term or trunc)
        self.assertEqual((env.env.reset_calls, env.env.step_calls), (1, 1))
        self.assertEqual(obs.shape, (10,))
        json.dumps(info, allow_nan=False)

    def test_unclipped_error_and_saturation_denominator(self):
        env = wrap(MotionHost(speed=20, offset=12))
        env.reset()
        obs, reward, _, _, info = env.step(4)
        self.assertEqual(tuple(obs[-3:]), (1, 0, 1))
        data = info['lookahead_learning']
        self.assertEqual(data['prediction']['error_m'], 12)
        self.assertAlmostEqual(reward, 1.88)
        metrics = data['prediction_episode']
        self.assertEqual(metrics['observation_samples'], 2)
        self.assertEqual(metrics['valid_observation_samples'], 2)
        self.assertEqual(metrics['saturated_observation_samples'], 2)
        self.assertEqual(metrics['observation_saturation_rate'], 1)
        self.assertEqual(data['effective_lookahead_m'], 20)

    def test_prediction_masks_do_not_change_valid_preview(self):
        for raw, reason in (
            (MotionHost(speed=0.1), 'low_speed'),
            # A small negative forward component preserves the old preview threshold.
            (MotionHost(velocities=((-0.01, 10),)), 'reverse_motion'),
        ):
            env = wrap(raw)
            env.reset()
            obs, reward, _, _, info = env.step(4)
            self.assertEqual(obs[-1], 1)
            self.assertEqual(reward, 2)
            self.assertEqual(info['lookahead_learning']['prediction']['skip_reason'], reason)
            self.assertIsNone(info['lookahead_learning']['prediction']['kappa_hat_inv_m'])
        env = wrap(MotionHost())
        env.env.config['physics_world_step_size'] = 1e-7
        env.reset()
        result = env.step(4)
        self.assertEqual(result[0][-1], 1)
        self.assertEqual(result[4]['lookahead_learning']['prediction']['skip_reason'], 'insufficient_history_distance')

    def test_invalid_post_reference_and_nonfinite_custom_state(self):
        env = wrap(MotionHost(length=20.5))
        self.assertEqual(env.reset()[0][-1], 1)
        data = env.step(4)[4]['lookahead_learning']
        self.assertEqual(data['prediction']['skip_reason'], 'preview_invalid')
        self.assertEqual(data['prediction']['preview_invalid_reason'], 'lookahead_past_route_end')
        self.assertEqual(data['prediction_episode']['invalid_seconds'], 0.1)
        for overrides in ({'speed_unit': 'km/h'}, {'speed_meaning': 'signed'},
                          {'heading_theta': math.nan}, {'speed_m_s': None},
                          {'forward_speed_mps': math.inf}):
            raw = MotionHost()
            provider = MetaDrivePreviewProvider(lookahead_time_s=1)
            env = LookaheadEnv(raw, mode='lookahead_obs', contract=_contract(raw),
                preview_provider=provider, lookahead_time_s=1,
                prediction_reward_enabled=True,
                state_reader=lambda host: {**read_vehicle_state(host, require_motion=True), **overrides})
            with self.subTest(overrides=overrides), self.assertRaises(HostContractError):
                env.reset()

    def test_off_zero_equal_transitions_and_rng_without_prediction_calls(self):
        envs = [wrap(MotionHost(), prediction_reward_enabled=enabled, prediction_reward_weight=weight)
                for enabled, weight in ((False, 0.1), (False, 4), (True, 0))]
        with patch('lookahead_learning.env.prediction_penalty', side_effect=AssertionError('Off calculated prediction')):
            initial = [env.reset(seed=321)[0] for env in envs]
            for obs in initial:
                np.testing.assert_array_equal(obs, initial[0])
            for action in (4, 5, 3):
                results = [env.step(action) for env in envs]
                for result in results:
                    np.testing.assert_array_equal(result[0], results[0][0])
                    self.assertEqual(result[1:4], results[0][1:4])
        rng = [env.env.np_random.random() for env in envs]
        self.assertEqual(rng, [rng[0]] * len(rng))

    def test_terminal_truncated_and_reset_clear_history_and_metrics(self):
        for ending in ('terminal_after', 'truncated_after'):
            env = wrap(MotionHost(offset=1, **{ending: 2}))
            self.assertEqual(env.reset()[1]['lookahead_learning']['prediction']['skip_reason'], 'reset')
            env.step(4)
            result = env.step(4)
            data = result[4]['lookahead_learning']
            self.assertEqual(result[1], 2)
            self.assertEqual(data['prediction']['skip_reason'], 'episode_end')
            self.assertAlmostEqual(data['episode_r_prediction'], -0.01)
            self.assertEqual(data['prediction_episode']['evaluated_seconds'], 0.1)
            self.assertEqual(data['prediction_episode']['skipped_seconds'], 0.1)
            with self.assertRaises(RuntimeError):
                env.step(4)
            env.env.headings = (0.4, 0.4)
            reset = env.reset()[1]['lookahead_learning']
            self.assertEqual(reset['episode_r_prediction'], 0)
            self.assertEqual(reset['prediction_episode']['evaluated_seconds'], 0)
            self.assertAlmostEqual(env.step(4)[4]['lookahead_learning']['prediction']['dpsi_rad'], 0)

    def test_monitor_vec_autoreset_no_episode_or_worker_history_leak(self):
        try:
            from stable_baselines3.common.monitor import Monitor
            from stable_baselines3.common.vec_env import DummyVecEnv
        except ImportError:
            self.skipTest('SB3 unavailable; pure/fake-host portability tests remain active')
        envs = [wrap(MotionHost(offset=offset, terminal_after=2)) for offset in (1, 2)]
        vec = DummyVecEnv([lambda env=env: Monitor(env) for env in envs])
        try:
            vec.reset()
            _, rewards, _, _ = vec.step(np.array([4, 4]))
            np.testing.assert_allclose(rewards, (1.99, 1.98))
            _, rewards, dones, infos = vec.step(np.array([4, 4]))
            self.assertTrue(all(dones))
            for index, info in enumerate(infos):
                self.assertAlmostEqual(info['episode']['r'], 4 - 0.01 * (index + 1))
                self.assertAlmostEqual(info['lookahead_learning']['episode_r_total'], info['episode']['r'])
                self.assertEqual(envs[index].episode_totals['r_prediction'], 0)
                self.assertEqual(envs[index].last_snapshot.decision, 0)
        finally:
            vec.close()

    def test_pp_lateral_prediction_terms_compose_once_with_existing_timing(self):
        from .lateral_acceleration import LateralReference
        from .test_env import PPProbe
        raw = MotionHost(offset=1)
        provider = MetaDrivePreviewProvider(lookahead_time_s=1)
        class SharedPreview:
            def reset(self, env):
                provider.reset(env)
            def __call__(self, env):
                result = dict(provider(env))
                result['lateral_reference'] = LateralReference(True, result['s_proj_m'],
                    result['s_goal_m'], 0.02 if env.step_calls == 0 else 0.)
                return result
        env = LookaheadEnv(raw, mode='lookahead_obs_pp_reward', contract=_contract(raw),
            preview_provider=SharedPreview(), pp_provider=PPProbe(), pp_weight=1,
            lookahead_time_s=1, prediction_reward_enabled=True, lateral_accel_reward_enabled=True)
        env.reset()
        data = env.step(4)[4]['lookahead_learning']
        self.assertAlmostEqual(data['r_pp'], -0.005)
        self.assertAlmostEqual(data['r_lateral_accel'], -0.0225)  # pre curvature, post speed
        self.assertAlmostEqual(data['r_prediction'], -0.01)
        self.assertAlmostEqual(data['r_total'], 2 - 0.005 - 0.0225 - 0.01)
        self.assertEqual(data['post_step']['preview']['lateral_reference']['kappa_abs_max_inv_m'], 0)

    def test_real_ppo_zip_attributes_and_legacy_validation_do_not_rewrite_zip(self):
        try:
            from stable_baselines3 import PPO
        except ImportError:
            self.skipTest('SB3 unavailable; pure/fake-host portability tests remain active')
        import hashlib
        from pathlib import Path
        import tempfile
        from .checkpoint import (set_lookahead_model_metadata, validate_lookahead_model_metadata,
                                 CheckpointContractError)
        with tempfile.TemporaryDirectory(prefix='lookahead-zip-') as directory:
            config = {'lookahead_time_s': 1., 'prediction_reward_enabled': True}
            env = wrap()
            model = PPO('MlpPolicy', env, n_steps=2, batch_size=2, seed=0,
                        policy_kwargs={'net_arch': [8]}, device='cpu')
            set_lookahead_model_metadata(model, config)
            path = Path(directory) / 'new.zip'
            model.save(path)
            loaded = PPO.load(path, device='cpu')
            validate_lookahead_model_metadata(loaded, config)
            with self.assertRaises(CheckpointContractError):
                validate_lookahead_model_metadata(loaded, {**config, 'lookahead_time_s': 2})
            env.close()
            old_env = wrap_lookahead_env(MotionHost())
            old_model = PPO('MlpPolicy', old_env, n_steps=2, batch_size=2, seed=0,
                            policy_kwargs={'net_arch': [8]}, device='cpu')
            old_model.lookahead_schema_version = 1
            old_model.lookahead_config = {'lookahead_m': 6., 'pp_weight': 0.}
            old_path = Path(directory) / 'legacy.zip'
            old_model.save(old_path)
            before = hashlib.sha256(old_path.read_bytes()).hexdigest()
            legacy = PPO.load(old_path, device='cpu')
            validate_lookahead_model_metadata(legacy, {})
            self.assertEqual(legacy.lookahead_schema_version, 1)
            self.assertEqual(hashlib.sha256(old_path.read_bytes()).hexdigest(), before)
            old_env.close()

    def test_subprocess_workers_receive_new_config_and_return_prediction_rewards(self):
        try:
            from stable_baselines3.common.vec_env import SubprocVecEnv
        except ImportError:
            self.skipTest('SB3 unavailable; pure/fake-host portability tests remain active')
        from functools import partial
        config = {'lookahead_time_s': 1.2, 'prediction_reward_enabled': True,
                  'prediction_reward_weight': 0.3, 'prediction_error_scale_m': 2.}
        # Worker processes are deterministic test runners, not AI agents.
        vec = SubprocVecEnv([partial(wrap_lookahead_env, MotionHost(offset=1), **config)
                            for _ in range(2)], start_method='spawn')
        try:
            self.assertEqual(vec.reset().shape, (2, 10))
            _, rewards, _, infos = vec.step(np.array([4, 4]))
            np.testing.assert_allclose(rewards, (1.985, 1.985))
            for info in infos:
                data = info['lookahead_learning']
                self.assertEqual(data['lookahead_time_s'], 1.2)
                self.assertEqual(data['effective_lookahead_m'], 12)
                self.assertEqual(data['prediction']['weight'], 0.3)
                self.assertEqual(data['prediction']['error_scale_m'], 2)
        finally:
            vec.close()

    def test_same_step_autoreset_below_wrapper_is_a_contract_error(self):
        class AutoResetHost(MotionHost):
            def step(self, action):
                obs, reward, term, trunc, info = super().step(action)
                info['final_observation'] = obs.copy()
                self.vehicle.position[:2] = (0., 0.)  # simulated next episode state
                return obs, reward, term, trunc, info
        env = wrap(AutoResetHost(terminal_after=1))
        env.reset()
        with self.assertRaisesRegex(HostContractError, 'autoreset'):
            env.step(4)
        self.assertEqual(env.env.step_calls, 1)

    def test_existing_host_chain_and_double_wrapper_rejection(self):
        class Chain(gym.Wrapper):
            reset_calls = step_calls = 0
            def reset(self, **kwargs):
                self.reset_calls += 1
                return self.env.reset(**kwargs)
            def step(self, action):
                self.step_calls += 1
                obs, reward, term, trunc, info = self.env.step(action)
                return obs, reward + 3, term, trunc, info
        raw = MotionHost(offset=1)
        chain = Chain(raw)
        provider = MetaDrivePreviewProvider(lookahead_time_s=1)
        class Preview:
            def reset(self, env):
                provider.reset(env.unwrapped)
            def __call__(self, env):
                return provider(env.unwrapped)
        env = LookaheadEnv(chain, mode='lookahead_obs', contract=_contract(raw),
            preview_provider=Preview(), lookahead_time_s=1, prediction_reward_enabled=True,
            state_reader=lambda e: read_vehicle_state(e.unwrapped, require_motion=True),
            applied_action_reader=lambda e: e.unwrapped.vehicle.current_action,
            dt_reader=lambda e: 0.1)
        env.reset()
        self.assertAlmostEqual(env.step(4)[1], 4.99)
        self.assertEqual((chain.reset_calls, chain.step_calls, raw.reset_calls, raw.step_calls), (1, 1, 1, 1))
        with self.assertRaisesRegex(HostContractError, 'already connected'):
            wrap(env)


if __name__ == '__main__':
    unittest.main()
