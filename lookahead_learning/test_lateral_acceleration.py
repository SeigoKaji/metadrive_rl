"""Standard-library-only numerical and exact route-interval acceptance tests."""

import json
import math
import unittest

from .geometry import GeometryError, RouteCurvatureProfile, build_reference_route, compute_preview
from .lateral_acceleration import LateralReference, lateral_accel_penalty, planar_speed_mps
from .test_geometry import CircularLane, StraightLane


def mixed_route(*, radius=50.0, arc_length=0.1, direction=1):
    """A short tangent arc between two straights, reflected for right turns."""
    straight = StraightLane((0.0, 0.0), (5.0, 0.0))
    arc = CircularLane(
        (5.0, direction * radius), radius, -direction * math.pi / 2,
        arc_length / radius, direction=direction,
    )
    end = arc.position(arc.length, 0.0)
    heading = arc.heading_theta_at(arc.length)
    last = StraightLane(end, (end[0] + 20 * math.cos(heading), end[1] + 20 * math.sin(heading)))
    return build_reference_route([straight, arc, last])


class LateralNumericalTests(unittest.TestCase):
    def score(self, **changes):
        return lateral_accel_penalty(**{
            "kappa_abs_max_inv_m": 1 / 50, "speed_mps": 10,
            "max_lateral_accel": 0.8, "weight": 0.1, "dt_seconds": 0.1,
            **changes,
        })

    def test_requested_examples_threshold_stop_straight_and_monotonicity(self):
        below = self.score(speed_mps=5)
        self.assertEqual(below.required_lateral_accel_mps2, 0.5)
        self.assertEqual(below.reward, 0.0)
        self.assertEqual(self.score(speed_mps=5, max_lateral_accel=0.5).reward, 0.0)
        high = self.score()
        self.assertEqual(high.required_lateral_accel_mps2, 2.0)
        self.assertEqual(high.exceedance_ratio, 1.5)
        self.assertAlmostEqual(high.reward, -0.0225)
        self.assertAlmostEqual(high.curve_speed_limit_mps, math.sqrt(40))
        self.assertAlmostEqual(high.curve_speed_limit_mps * 3.6, 22.7683991532)
        self.assertLess(self.score(speed_mps=12).reward, high.reward)
        self.assertEqual(self.score(speed_mps=0).reward, 0.0)
        straight = self.score(kappa_abs_max_inv_m=0)
        self.assertEqual(straight.reward, 0.0)
        self.assertTrue(straight.curve_speed_unlimited)
        self.assertIsNone(straight.curve_speed_limit_mps)
        json.dumps(straight.as_dict(), allow_nan=False)

    def test_limit_weight_and_decision_duration(self):
        self.assertEqual(self.score(max_lateral_accel=2.0).reward, 0.0)
        self.assertAlmostEqual(self.score(weight=0.2).reward, 2 * self.score().reward)
        self.assertAlmostEqual(self.score(dt_seconds=0.2).reward, 2 * self.score().reward)
        self.assertEqual(self.score(weight=0).reward, 0.0)

    def test_speed_units_and_explicit_signed_contract(self):
        for value, unit, signed in ((10, "m/s", False), (36, "km/h", False), (-36, "km/h", True)):
            converted = planar_speed_mps(value, unit=unit, signed=signed)
            self.assertEqual(converted, 10)
            self.assertEqual(self.score(speed_mps=converted), self.score())
        for value, unit in ((10, "unknown"), (-1, "m/s"), ("36", "km/h"), (True, "m/s")):
            with self.subTest(value=value, unit=unit), self.assertRaises(ValueError):
                planar_speed_mps(value, unit=unit)

    def test_nonfinite_bad_types_and_overflow_raise(self):
        for key in ("kappa_abs_max_inv_m", "speed_mps", "max_lateral_accel", "weight", "dt_seconds"):
            for bad in (math.nan, math.inf, -math.inf, True, "1", None, -1):
                with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                    self.score(**{key: bad})
        for key in ("max_lateral_accel", "dt_seconds"):
            with self.assertRaises(ValueError):
                self.score(**{key: 0})
        with self.assertRaises(ValueError):
            self.score(speed_mps=1e300)
        with self.assertRaises(ValueError):
            LateralReference(True, 0, 6, math.nan)
        with self.assertRaises(ValueError):
            LateralReference(False)


class RouteCurvatureTests(unittest.TestCase):
    def test_straights_left_right_and_actual_lane_radius(self):
        straight = build_reference_route([StraightLane((0, 0), (50, 0))]).path
        self.assertEqual(RouteCurvatureProfile.from_path(straight).max_abs_curvature(0, 50), 0)
        for direction in (-1, 1):
            for radius in (25.0, 50.0):
                arc = CircularLane((0, 0), radius, 0, 0.2, direction=direction)
                route = build_reference_route([arc]).path
                profile = RouteCurvatureProfile.from_path(route)
                self.assertAlmostEqual(profile.max_abs_curvature(0, route.total_length), 1 / radius)

    def test_short_internal_curve_and_goal_on_straight(self):
        for direction in (-1, 1):
            route = mixed_route(direction=direction)
            self.assertEqual(route.diagnostics.validated_lanes, 3)
            preview = compute_preview(route.path, (0, 0), 0, lookahead_m=6)
            self.assertTrue(preview.valid)
            self.assertEqual(preview.goal_lane_index, 2)
            profile = RouteCurvatureProfile.from_path(route.path)
            self.assertAlmostEqual(profile.max_abs_curvature(preview.s_proj, preview.s_goal), 0.02)

    def test_right_continuous_lane_boundaries_and_closed_query(self):
        route = mixed_route()
        path = route.path
        profile = RouteCurvatureProfile.from_path(path)
        curve_start, curve_end = path.lane_starts[1], path.lane_ends[1]
        self.assertEqual(path.lane_index_at_s(curve_start), 1)
        self.assertEqual(profile.max_abs_curvature(0, math.nextafter(curve_start, 0)), 0)
        self.assertEqual(profile.max_abs_curvature(0, curve_start), 0.02)
        self.assertEqual(profile.max_abs_curvature(curve_start, curve_start), 0.02)
        self.assertEqual(profile.max_abs_curvature(curve_start, curve_end), 0.02)
        self.assertEqual(path.lane_index_at_s(curve_end), 2)
        self.assertEqual(profile.max_abs_curvature(curve_end, curve_end + 1), 0)
        self.assertEqual(profile.max_abs_curvature(path.total_length, path.total_length), 0)
        with self.assertRaises(GeometryError):
            profile.max_abs_curvature(0, math.nextafter(path.total_length, math.inf))
        with self.assertRaises(GeometryError):
            profile.max_abs_curvature(3, 2)

    def test_validated_prefix_end_and_discontinuous_unsupported_boundaries(self):
        arc = CircularLane((0, 50), 50, -math.pi / 2, 0.2)
        unconnected = StraightLane((100, 100), (110, 100))
        route = build_reference_route([arc, unconnected])
        self.assertEqual(route.diagnostics.validated_lanes, 1)
        profile = RouteCurvatureProfile.from_path(route.path)
        self.assertEqual(profile.max_abs_curvature(0, route.path.total_length), 0.02)
        self.assertEqual(profile.max_abs_curvature(route.path.total_length, route.path.total_length), 0.02)
        with self.assertRaises(GeometryError):
            profile.max_abs_curvature(0, route.path.total_length + 0.1)

    def test_radius_is_required_only_when_profile_requested_and_supports_adapter(self):
        arc = CircularLane((0, 50), 50, -math.pi / 2, 0.2)
        path = build_reference_route([arc]).path
        for bad in (None, math.nan, math.inf, 0, -1, True, "50"):
            arc.radius = bad
            with self.subTest(bad=bad), self.assertRaises(GeometryError):
                RouteCurvatureProfile.from_path(path)
        del arc.radius
        with self.assertRaises(GeometryError):
            RouteCurvatureProfile.from_path(path)
        calls = []
        profile = RouteCurvatureProfile.from_path(
            path, radius_reader=lambda lane: calls.append(lane) or 50.0
        )
        self.assertEqual(profile.max_abs_curvature(0, 6), 0.02)
        self.assertEqual(profile.max_abs_curvature(1, 7), 0.02)
        self.assertEqual(calls, [arc])


if __name__ == "__main__":
    unittest.main()
