"""Standard-library numeric/contract tests; no simulator or learner."""
import math
import unittest
from dataclasses import replace
from types import SimpleNamespace

from .checkpoint import (CheckpointContractError, resolve_lookahead_config,
                         set_lookahead_model_metadata, validate_lookahead_model_metadata)
from .prediction import (MotionState, MIN_HISTORY_DISTANCE_M, MIN_PREDICTION_SPEED_MPS,
                         prediction_penalty, predict_position, sinc_cosc, time_lookahead_distance,
                         estimate_acceleration, constant_acceleration_travel, PredictionReference)


def penalty(pre, post, goal, **kwargs):
    return prediction_penalty(pre, post, goal, **{
        'preview_valid': True, 'time_s': 1.0, 'dt_seconds': 0.1,
        'weight': 0.1, 'error_scale_m': 1.0, **kwargs,
    })


class PredictionTests(unittest.TestCase):
    def test_straight_center_and_parallel_offset_one_metre(self):
        for offset in (0, 1, -1):
            post = MotionState((1, offset), 0, 10, 10)
            result = penalty(MotionState((0, offset), 0, 10, 10), post, (11, 0))
            self.assertTrue(result.valid)
            self.assertEqual(result.predicted_xy, (11, offset))
            self.assertEqual(result.error_m, abs(offset))
            self.assertAlmostEqual(result.reward, -0.01 * abs(offset))
        self.assertEqual(time_lookahead_distance(10, 1), 10)
        self.assertEqual(time_lookahead_distance(0, 1), 0)

    def test_circular_arcs_left_right_and_body_center_rotation(self):
        # Radius 20m, v=10m/s: last 0.1s rotates 0.05 rad, future 1s rotates 0.5.
        for sign in (-1, 1):
            pre = MotionState((99, 20), 0, 10, 10)
            post = MotionState((100, 20), sign * 0.05, 10, 10)
            goal = (100 + 20 * (math.sin(0.55) - math.sin(0.05)),
                    20 + sign * 20 * (math.cos(0.05) - math.cos(0.55)))
            result = penalty(pre, post, goal)
            self.assertAlmostEqual(result.kappa_hat_inv_m, sign / 20)
            self.assertAlmostEqual(result.error_m, 0, places=12)
            self.assertAlmostEqual(result.alpha_rad, sign * 0.5)
        # A rear axle shift would change this result; the input centre is the origin.
        rotated = MotionState((100, 20), math.pi / 2, 10, 10)
        xy = predict_position(rotated, distance_m=10, curvature_inv_m=0)
        self.assertAlmostEqual(xy[0], 100)
        self.assertAlmostEqual(xy[1], 30)

    def test_zero_near_zero_and_heading_wrap(self):
        self.assertEqual(sinc_cosc(0), (1, 0))
        for angle in (1e-12, -1e-12, 0.999e-4, 1.001e-4):
            sinc, cosc = sinc_cosc(angle)
            self.assertAlmostEqual(sinc, 1 - angle * angle / 6, places=14)
            self.assertAlmostEqual(cosc, angle / 2, places=12)
        pre = MotionState((0, 0), math.pi - 0.01, 10, 10)
        post = MotionState((0, 0), -math.pi + 0.01, 10, 10)
        result = penalty(pre, post, (-10, 0))
        self.assertAlmostEqual(result.dpsi_rad, 0.02)
        self.assertAlmostEqual(result.kappa_hat_inv_m, 0.02)

    def test_two_speeds_only_estimate_history_future_uses_post_and_reward_dt(self):
        pre = MotionState((0, 0), 0, 2, 2)
        post = MotionState((0.6, 0), 0.03, 10, 10)
        result = penalty(pre, post, (10, 1), time_s=2, error_scale_m=2, weight=0.3)
        self.assertAlmostEqual(result.ds_hist_m, 0.6)
        self.assertAlmostEqual(result.kappa_hat_inv_m, 0.05)
        self.assertEqual(result.distance_m, 20)
        self.assertAlmostEqual(result.reward, -0.3 * 0.1 * result.error_m / 2)

    def test_masks_preserve_unavailable_curvature(self):
        state = MotionState((0, 0), 0, 10, 10)
        cases = [
            (None, state, {}, 'history_unavailable'),
            (state, state, {'preview_valid': False}, 'preview_invalid'),
            (replace(state, forward_speed_mps=-1), state, {}, 'reverse_motion'),
            (state, replace(state, forward_speed_mps=-0.01), {}, 'reverse_motion'),
            (state, replace(state, speed_mps=MIN_PREDICTION_SPEED_MPS), {}, 'low_speed'),
            (state, state, {'terminated': True}, 'episode_end'),
            (state, state, {'truncated': True}, 'episode_end'),
            (state, state, {'dt_seconds': MIN_HISTORY_DISTANCE_M / 10}, 'insufficient_history_distance'),
        ]
        for pre, post, kw, reason in cases:
            with self.subTest(reason=reason):
                result = penalty(pre, post, (10, 0), **kw)
                self.assertFalse(result.valid)
                self.assertEqual(result.reward, 0)
                self.assertEqual(result.skip_reason, reason)
                self.assertIsNone(result.kappa_hat_inv_m)
                self.assertIsNone(result.error_m)

    def test_nonfinite_or_unknown_units_are_contract_errors_even_when_masked(self):
        state = MotionState((0, 0), 0, 10, 10)
        for bad in (True, '10', math.nan, math.inf, -math.inf):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    replace(state, speed_mps=bad)
                with self.assertRaises(ValueError):
                    penalty(state, state, (bad, 0), terminated=True)
        for bad in ({}, {'speed_unit': 'km/h', 'speed_meaning': 'planar magnitude'}):
            with self.assertRaises(ValueError):
                MotionState.from_mapping(bad)
        with self.assertRaises(ValueError):
            time_lookahead_distance(1e308, 10)
        with self.assertRaises(ValueError):
            predict_position(state, distance_m=1e308, curvature_inv_m=1e308)


class PredictionConfigTests(unittest.TestCase):
    def test_strict_config_and_effective_on_requires_time(self):
        for key, bad_values in {
            'lookahead_time_s': (True, False, 0, -1, '1', math.nan, math.inf),
            'prediction_reward_enabled': (0, 1, None, 'true', math.nan),
            'prediction_motion_model': (None, True, 1, [], {}, 'acceleration', ''),
            'prediction_reward_weight': (True, None, '0.1', -1, math.nan, math.inf),
            'prediction_error_scale_m': (True, None, '1', 0, -1, math.nan, math.inf),
        }.items():
            for bad in bad_values:
                with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                    resolve_lookahead_config({key: bad})
        with self.assertRaisesRegex(ValueError, 'lookahead_time_s'):
            resolve_lookahead_config({'prediction_reward_enabled': True})
        zero = resolve_lookahead_config({'prediction_reward_enabled': True, 'prediction_reward_weight': 0})
        self.assertIsNone(zero['lookahead_time_s'])
        with self.assertRaises(ValueError):
            resolve_lookahead_config({'prediction_weight': 1})

    def test_v1_v2_distance_off_compatibility_and_no_mutation(self):
        configs = [
            {'lookahead_m': 6., 'pp_weight': 0.},
            {'lookahead_m': 6., 'pp_weight': 0., 'lateral_accel_reward_enabled': True,
             'max_lateral_accel': 0.8, 'lateral_accel_weight': 0.1},
        ]
        for schema, config in enumerate(configs, 1):
            model = SimpleNamespace(lookahead_schema_version=schema, lookahead_config=config.copy())
            expected = {**config, 'prediction_reward_weight': 0.7, 'prediction_error_scale_m': 10}
            validate_lookahead_model_metadata(model, expected)
            validate_lookahead_model_metadata(model, {**config, 'prediction_reward_enabled': True,
                                                      'prediction_reward_weight': 0})
            with self.assertRaises(CheckpointContractError):
                validate_lookahead_model_metadata(model, {**config, 'lookahead_time_s': 1})
            self.assertEqual(model.lookahead_schema_version, schema)
            self.assertEqual(model.lookahead_config, config)
        with self.assertRaises(CheckpointContractError):
            validate_lookahead_model_metadata(SimpleNamespace(
                lookahead_schema_version=2, lookahead_config={**configs[0], 'lookahead_time_s': 1}
            ), {'lookahead_time_s': 1})

    def test_same_dimension_time_distance_T_and_effective_reward_must_match(self):
        config = {'lookahead_time_s': 1, 'prediction_reward_enabled': True}
        model = SimpleNamespace()
        set_lookahead_model_metadata(model, config)
        self.assertEqual(model.lookahead_schema_version, 4)
        validate_lookahead_model_metadata(model, {**config, 'lookahead_m': 100})
        for change in ({'lookahead_time_s': 2}, {'prediction_reward_enabled': False},
                       {'prediction_reward_weight': 0}, {'prediction_reward_weight': 0.2},
                       {'prediction_error_scale_m': 2}):
            with self.subTest(change=change), self.assertRaises(CheckpointContractError):
                validate_lookahead_model_metadata(model, {**config, **change})
        with self.assertRaises(CheckpointContractError):
            validate_lookahead_model_metadata(model, {})
        set_lookahead_model_metadata(model, {'lookahead_time_s': 1})
        validate_lookahead_model_metadata(model, {'lookahead_time_s': 1, 'lookahead_m': 2,
            'prediction_reward_enabled': True, 'prediction_reward_weight': 0, 'prediction_error_scale_m': 5})



class AccelerationTests(unittest.TestCase):
    def test_scalar_acceleration_and_distance_examples(self):
        self.assertAlmostEqual(estimate_acceleration(10.2, 10, .1), -2)
        self.assertAlmostEqual(estimate_acceleration(9.8, 10, .1), 2)
        self.assertEqual(estimate_acceleration(10, 10, .1), 0)
        for acceleration, distance, end in ((2, 11, 12), (0, 10, 10), (-2, 9, 8)):
            travel = constant_acceleration_travel(10, acceleration, 1)
            self.assertEqual((travel.distance_m, travel.end_speed_mps), (distance, end))
            self.assertFalse(travel.stopped)
            self.assertEqual(travel.moving_time_s, 1)

    def test_stop_distance_plateau_continuity_near_zero_and_zero_initial_speed(self):
        for time, distance, end in ((3, 21, 4), (5, 25, 0), (7, 25, 0)):
            travel = constant_acceleration_travel(10, -2, time)
            self.assertEqual((travel.distance_m, travel.end_speed_mps), (distance, end))
            self.assertEqual(travel.stop_time_s, 5)
            self.assertEqual(travel.stopped, time >= 5)
            self.assertEqual(travel.moving_time_s, min(time, 5))
        for time in (math.nextafter(5, 0), math.nextafter(5, math.inf)):
            self.assertAlmostEqual(constant_acceleration_travel(10, -2, time).distance_m, 25)
        for acceleration in (1e-12, -1e-12, 1e-300, -1e-300):
            travel = constant_acceleration_travel(10, acceleration, 1)
            self.assertAlmostEqual(travel.distance_m, 10, places=11)
        # Small nonzero acceleration is not replaced by zero: long T exposes it.
        self.assertEqual(constant_acceleration_travel(0, 1e-12, 1e6).distance_m, .5)
        for acceleration, distance, end in ((2, 1, 2), (0, 0, 0), (-2, 0, 0)):
            travel = constant_acceleration_travel(0, acceleration, 1)
            self.assertEqual((travel.distance_m, travel.end_speed_mps), (distance, end))

    def test_finite_contracts_and_calculation_order(self):
        for bad in (True, None, '1', math.nan, math.inf, -math.inf):
            for index in range(3):
                args = [10., 2., 1.]; args[index] = bad
                with self.subTest(args=args), self.assertRaises(ValueError):
                    constant_acceleration_travel(*args)
                args = [10., 10., .1]; args[index] = bad
                with self.subTest(args=args), self.assertRaises(ValueError):
                    estimate_acceleration(*args)
        for args in ((-1, 0, 1), (10, 2, 0), (10, 2, -1),
                     (1e308, 1e308, 2), (10, -5e-324, 1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                constant_acceleration_travel(*args)
        for args in ((-1, 1, .1), (1, -1, .1), (1, 1, 0), (0, 1e308, 1e-308)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                estimate_acceleration(*args)
        # Squaring speed or evaluating a*T after the stop would overflow.
        travel = constant_acceleration_travel(1e200, -1e200, 1e200)
        self.assertEqual(travel.distance_m, 5e199)
        self.assertEqual(travel.end_speed_mps, 0)
        self.assertEqual(constant_acceleration_travel(1e308, 1e308, 1e-308).distance_m, .9999999999999999)

    def test_shared_distance_straight_curve_wrap_and_zero_acceleration_equivalence(self):
        from unittest.mock import Mock
        for acceleration in (-2., 0., 2.):
            for curvature in (-.05, 0., 1e-12, .05):
                v_pre = 10 - acceleration * .1
                ds = .5 * (v_pre + 10) * .1
                psi_pre = math.pi - .01
                psi_post = (psi_pre + curvature * ds + math.pi) % (2 * math.pi) - math.pi
                pre = MotionState((0, 0), psi_pre, v_pre, v_pre)
                post = MotionState((1, 0), psi_post, 10, 10)
                travel = constant_acceleration_travel(10, acceleration, 1)
                # Analytic centreline arc, independently of predict_position.
                alpha = curvature * travel.distance_m
                if curvature == 0:
                    goal = (1 + travel.distance_m * math.cos(psi_post), travel.distance_m * math.sin(psi_post))
                elif abs(curvature) < 1e-10:
                    goal = (1 + travel.distance_m * math.cos(psi_post), travel.distance_m * math.sin(psi_post))
                else:
                    goal = (1 + (math.sin(psi_post + alpha) - math.sin(psi_post)) / curvature,
                            (math.cos(psi_post) - math.cos(psi_post + alpha)) / curvature)
                reader = Mock(side_effect=lambda distance: PredictionReference(True, goal, 1, 1 + distance))
                result = penalty(pre, post, None, preview_valid=False,
                                 motion_model='constant_acceleration', reference_at_distance=reader)
                self.assertTrue(result.valid)
                reader.assert_called_once_with(result.distance_m)
                self.assertAlmostEqual(result.acceleration_mps2, acceleration)
                self.assertAlmostEqual(result.kappa_hat_inv_m, curvature)
                self.assertAlmostEqual(result.alpha_rad, curvature * travel.distance_m)
                self.assertAlmostEqual(result.error_m, 0, places=8)
                if acceleration == 0:
                    legacy = penalty(pre, post, goal)
                    for field in ('distance_m', 'predicted_xy', 'error_m', 'reward', 'kappa_hat_inv_m'):
                        self.assertEqual(getattr(result, field), getattr(legacy, field))

    def test_masks_do_not_call_reference_and_reference_contracts_raise(self):
        from unittest.mock import Mock
        state = MotionState((0, 0), 0, 10, 10)
        for pre, post, kw, reason in (
            (None, state, {}, 'history_unavailable'),
            (state, state, {'terminated': True}, 'episode_end'),
            (state, state, {'truncated': True}, 'episode_end'),
            (replace(state, forward_speed_mps=-1), state, {}, 'reverse_motion'),
            (state, replace(state, speed_mps=.1), {}, 'low_speed'),
            (state, state, {'dt_seconds':1e-7}, 'insufficient_history_distance'),
        ):
            reader = Mock(side_effect=AssertionError('masked reference read'))
            result = penalty(pre, post, None, preview_valid=False, motion_model='constant_acceleration',
                             reference_at_distance=reader, **kw)
            self.assertEqual(result.skip_reason, reason)
            self.assertIsNone(result.acceleration_mps2)
            reader.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'reference_at_distance'):
            penalty(state, state, None, motion_model='constant_acceleration')
        for value in (None, {'valid':True}, PredictionReference(True, (10,0), 0, 9)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                penalty(state, state, None, motion_model='constant_acceleration', reference_at_distance=lambda d: value)
        for kwargs in ({'valid':False}, {'valid':True}, {'valid':1},
                       {'valid':True,'goal_xy':(math.nan,0),'s_proj_m':0,'s_goal_m':10}):
            with self.assertRaises(ValueError):
                PredictionReference(**kwargs)

    def test_v3_constant_speed_migration_and_effective_model_checks(self):
        old = {'lookahead_time_s':1., 'prediction_reward_enabled':True}
        model = SimpleNamespace(lookahead_schema_version=3, lookahead_config=old.copy())
        validate_lookahead_model_metadata(model, old)
        validate_lookahead_model_metadata(model, {**old,'prediction_motion_model':'constant_speed'})
        with self.assertRaises(CheckpointContractError):
            validate_lookahead_model_metadata(model, {**old,'prediction_motion_model':'constant_acceleration'})
        for schema in (1,2,3):
            for name in ('constant_speed','constant_acceleration'):
                malformed = SimpleNamespace(lookahead_schema_version=schema,
                    lookahead_config={'lookahead_m':6.,'pp_weight':0.,'prediction_motion_model':name})
                with self.assertRaises(CheckpointContractError):
                    validate_lookahead_model_metadata(malformed, {})
        self.assertEqual(model.lookahead_config, old)
        self.assertEqual(model.lookahead_schema_version, 3)
        set_lookahead_model_metadata(model, {})
        validate_lookahead_model_metadata(model, {'prediction_motion_model':'constant_acceleration'})
        validate_lookahead_model_metadata(model, {'prediction_motion_model':'constant_acceleration',
                                                'prediction_reward_enabled':True,'prediction_reward_weight':0})
        new = {**old,'prediction_motion_model':'constant_acceleration'}
        set_lookahead_model_metadata(model, new)
        validate_lookahead_model_metadata(model, new)
        with self.assertRaises(CheckpointContractError):
            validate_lookahead_model_metadata(model, old)


if __name__ == '__main__':
    unittest.main()
