"""Pure geometry tests used by ``python -B -m unittest lookahead_learning.test_geometry``.

The fake lanes below implement the adapter protocol only; importing this
module never imports MetaDrive, Gymnasium, Panda3D, NumPy, or SB3.
"""

from __future__ import annotations

import math
import unittest
from dataclasses import FrozenInstanceError

from .geometry import (
    LaneMetadata,
    PreviewResult,
    ReferencePath,
    RouteDiagnostics,
    _Segment,
    build_reference_route,
    compute_pp_penalty,
    compute_preview,
    compute_pure_pursuit,
    connected_successor_candidates,
    project_to_path,
    pure_pursuit_from_rear_coordinates,
)


class StraightLane:
    """Analytic finite straight lane used only by these tests."""

    def __init__(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        width: float = 3.5,
        lane_count: int = 1,
        lane_id: str | None = None,
    ) -> None:
        self.start = start
        self.end = end
        self.width = width
        self.lane_count = lane_count
        self.lane_id = lane_id
        self.length = math.dist(start, end)
        if self.length <= 0.0:
            raise ValueError("test lane must have positive length")
        ux = (end[0] - start[0]) / self.length
        uy = (end[1] - start[1]) / self.length
        self.forward = (ux, uy)
        self.left = (-uy, ux)

    def position(self, s: float, lateral: float) -> tuple[float, float]:
        return (
            self.start[0] + self.forward[0] * s + self.left[0] * lateral,
            self.start[1] + self.forward[1] * s + self.left[1] * lateral,
        )

    def local_coordinates(self, point: tuple[float, float]) -> tuple[float, float]:
        dx = point[0] - self.start[0]
        dy = point[1] - self.start[1]
        return (
            dx * self.forward[0] + dy * self.forward[1],
            dx * self.left[0] + dy * self.left[1],
        )

    def heading_theta_at(self, s: float) -> float:
        del s
        return math.atan2(self.forward[1], self.forward[0])


class CircularLane:
    """Analytic circular lane supporting both left and right curvature."""

    def __init__(
        self,
        center: tuple[float, float],
        radius: float,
        start_angle: float,
        arc_angle: float,
        *,
        direction: int = 1,
        width: float = 3.5,
        lane_count: int = 1,
    ) -> None:
        if direction not in (-1, 1):
            raise ValueError("direction must be -1 or +1")
        self.center = center
        self.radius = radius
        self.start_angle = start_angle
        self.direction = direction
        self.arc_angle = arc_angle
        self.length = abs(arc_angle) * radius
        self.width = width
        self.lane_count = lane_count

    def _angle(self, s: float) -> float:
        return self.start_angle + self.direction * s / self.radius

    def position(self, s: float, lateral: float) -> tuple[float, float]:
        angle = self._angle(s)
        radial_distance = self.radius - self.direction * lateral
        return (
            self.center[0] + radial_distance * math.cos(angle),
            self.center[1] + radial_distance * math.sin(angle),
        )

    def local_coordinates(self, point: tuple[float, float]) -> tuple[float, float]:
        dx = point[0] - self.center[0]
        dy = point[1] - self.center[1]
        angle = math.atan2(dy, dx)
        # Select the wrapped angular equivalent nearest this finite arc.  This
        # mirrors the adapter contract for an arc crossing the -pi/+pi cut;
        # blindly using one principal ``atan2`` delta would project its latter
        # half outside the finite lane interval.
        principal = (angle - self.start_angle + math.pi) % (2.0 * math.pi) - math.pi
        expected = self.direction * self.arc_angle * self.radius / 2.0
        choices = [
            self.direction * (principal + 2.0 * math.pi * turns) * self.radius
            for turns in range(-2, 3)
        ]
        s = min(choices, key=lambda candidate: abs(candidate - expected))
        radial = math.hypot(dx, dy)
        return s, self.direction * (self.radius - radial)

    def heading_theta_at(self, s: float) -> float:
        angle = self._angle(s)
        return math.atan2(self.direction * math.cos(angle), -self.direction * math.sin(angle))


class ProgrammerErrorLane(StraightLane):
    """Lane whose runtime failure must not be hidden as preview invalidity."""

    def local_coordinates(self, point: tuple[float, float]) -> tuple[float, float]:
        del point
        raise RuntimeError("simulated adapter bug")


def _metadata(lanes: list[object]) -> list[dict[str, object]]:
    return [
        {
            "type": type(lane).__name__,
            "width": getattr(lane, "width", 3.5),
            "lane_count": getattr(lane, "lane_count", 1),
            "lane_id": getattr(lane, "lane_id", None),
        }
        for lane in lanes
    ]


def _route(*lanes: object):
    return build_reference_route(list(lanes), metadata=_metadata(list(lanes)))


class GeometryRouteTests(unittest.TestCase):
    def test_straight_preview_has_expected_encoding_and_prefix(self) -> None:
        lane = StraightLane((0.0, 0.0), (10.0, 0.0), lane_id="s0")
        result = _route(lane)
        self.assertEqual(result.diagnostics.boundary_reason, "route_end")
        preview = compute_preview(result.path, (0.0, 0.0), 0.0)
        self.assertTrue(preview.valid)
        self.assertEqual(preview.observation, (0.8, 0.5, 1.0))
        self.assertEqual(preview.s_proj, 0.0)
        self.assertEqual(preview.s_goal, 6.0)
        self.assertEqual(preview.goal_lane_index, 0)
        self.assertEqual(preview.projection.lane_id, "s0")

    def test_rotation_translation_and_lateral_sign_use_vehicle_frame(self) -> None:
        lane = StraightLane((11.0, -7.0), (11.0, 8.0))
        result = _route(lane)
        preview = compute_preview(result.path, (11.0, -7.0), math.pi / 2.0)
        self.assertTrue(preview.valid)
        self.assertAlmostEqual(preview.x_g or 0.0, 6.0)
        self.assertAlmostEqual(preview.y_g or 0.0, 0.0)
        shifted = compute_preview(result.path, (10.5, -7.0), math.pi / 2.0)
        self.assertTrue(shifted.valid)
        self.assertAlmostEqual(shifted.x_g or 0.0, 6.0)
        self.assertAlmostEqual(shifted.y_g or 0.0, -0.5)

    def test_left_and_right_circular_lanes_have_opposite_preview_signs(self) -> None:
        left_lane = CircularLane((0.0, 0.0), 20.0, 0.0, math.pi / 2.0, direction=1)
        right_lane = CircularLane((0.0, 0.0), 20.0, 0.0, math.pi / 2.0, direction=-1)
        left_start = left_lane.position(0.0, 0.0)
        right_start = right_lane.position(0.0, 0.0)
        left_heading = left_lane.heading_theta_at(0.0)
        right_heading = right_lane.heading_theta_at(0.0)
        left_preview = compute_preview(_route(left_lane).path, left_start, left_heading)
        right_preview = compute_preview(_route(right_lane).path, right_start, right_heading)
        self.assertTrue(left_preview.valid)
        self.assertTrue(right_preview.valid)
        self.assertGreater(left_preview.y_g or 0.0, 0.0)
        self.assertLess(right_preview.y_g or 0.0, 0.0)
        self.assertGreater(left_preview.heading_error_rad or 0.0, 0.0)
        self.assertLess(right_preview.heading_error_rad or 0.0, 0.0)

    def test_circular_projection_uses_finite_arc_across_angle_wrap(self) -> None:
        lane = CircularLane((0.0, 0.0), 20.0, 3.0, 1.0, direction=1)
        path = _route(lane).path
        point = lane.position(15.0, 0.0)
        projection = project_to_path(path, point)
        self.assertTrue(projection.valid)
        self.assertAlmostEqual(projection.s_proj or 0.0, 15.0, places=7)

    def test_multiple_short_lanes_are_parameterized_by_cumulative_distance(self) -> None:
        lanes = [
            StraightLane((0.0, 0.0), (2.0, 0.0)),
            StraightLane((2.0, 0.0), (4.0, 0.0)),
            StraightLane((4.0, 0.0), (7.0, 0.0)),
        ]
        result = _route(*lanes)
        self.assertEqual(result.path.lengths, (2.0, 2.0, 3.0))
        preview = compute_preview(result.path, (0.0, 0.0), 0.0, lookahead_m=5.0)
        self.assertTrue(preview.valid)
        self.assertEqual(preview.s_goal, 5.0)
        self.assertEqual(preview.goal_lane_index, 2)
        self.assertEqual(preview.q, (5.0, 0.0))

    def test_discontinuous_lane_with_same_id_is_rejected(self) -> None:
        first = StraightLane((0.0, 0.0), (3.0, 0.0), lane_id="same")
        second = StraightLane((4.0, 0.0), (8.0, 0.0), lane_id="same")
        result = _route(first, second)
        self.assertEqual(result.path.lanes, (first,))
        self.assertEqual(result.diagnostics.boundary_reason, "discontinuous_endpoint")
        self.assertAlmostEqual(
            result.diagnostics.issues[0].details[0][1], 1.0
        )

    def test_lane_count_and_width_mismatch_stop_prefix(self) -> None:
        first = StraightLane((0.0, 0.0), (3.0, 0.0), width=3.5, lane_count=1)
        count_change = StraightLane((3.0, 0.0), (6.0, 0.0), width=3.5, lane_count=2)
        result = _route(first, count_change)
        self.assertEqual(result.diagnostics.boundary_reason, "lane_count_mismatch")
        wider = StraightLane((3.0, 0.0), (6.0, 0.0), width=3.7, lane_count=1)
        width_result = _route(first, wider)
        self.assertEqual(width_result.diagnostics.boundary_reason, "lane_width_mismatch")

    def test_missing_metadata_is_not_treated_as_verified(self) -> None:
        lane = StraightLane((0.0, 0.0), (8.0, 0.0))
        result = build_reference_route([lane], metadata=[{"type": "StraightLane"}])
        self.assertEqual(result.path.lanes, ())
        self.assertEqual(result.diagnostics.boundary_reason, "missing_lane_metadata")

    def test_unsupported_lane_type_is_boundary(self) -> None:
        lane = StraightLane((0.0, 0.0), (8.0, 0.0))
        result = build_reference_route(
            [lane],
            metadata=[{"type": "BezierLane", "width": 3.5, "lane_count": 1}],
        )
        self.assertEqual(result.diagnostics.boundary_reason, "unsupported_lane_type")

    def test_endpoint_projection_is_finite_and_shared_tie_is_deterministic(self) -> None:
        first = StraightLane((0.0, 0.0), (3.0, 0.0))
        second = StraightLane((3.0, 0.0), (8.0, 0.0))
        path = _route(first, second).path
        before = project_to_path(path, (-1.0, 0.0))
        after = project_to_path(path, (9.0, 0.0))
        joint = project_to_path(path, (3.0, 0.0))
        self.assertEqual((before.valid, before.s_proj), (True, 0.0))
        self.assertEqual((after.valid, after.s_proj), (True, 8.0))
        self.assertEqual((joint.valid, joint.s_proj, joint.lane_index), (True, 3.0, 0))
        just_after = project_to_path(path, (3.0005, 0.0))
        just_before = project_to_path(path, (2.9995, 0.0))
        self.assertEqual((just_after.valid, just_after.lane_index), (True, 1))
        self.assertAlmostEqual(just_after.s_proj or 0.0, 3.0005)
        self.assertEqual((just_before.valid, just_before.lane_index), (True, 0))
        self.assertAlmostEqual(just_before.s_proj or 0.0, 2.9995)
        near_joint = project_to_path(path, (3.01, 1.0))
        self.assertTrue(near_joint.valid)
        self.assertAlmostEqual(near_joint.s_proj or 0.0, 3.01)

    def test_nonadjacent_equal_projection_is_ambiguous(self) -> None:
        lanes = [
            StraightLane((0.0, 0.0), (2.0, 0.0)),
            StraightLane((2.0, 0.0), (2.0, 2.0)),
            StraightLane((2.0, 2.0), (0.0, 2.0)),
        ]
        # The route builder intentionally rejects those ninety-degree corners
        # under its tangent-continuity contract.  Build this diagnostic path
        # directly to exercise projection ambiguity independently: its first
        # and third finite intervals are parallel and one metre from (1, 1).
        diagnostics = RouteDiagnostics(
            total_lanes=3,
            validated_lanes=3,
            boundary_reason="route_end",
            boundary_lane_index=None,
            boundary_s=6.0,
        )
        path = ReferencePath(
            (
                _Segment(lanes[0], LaneMetadata("StraightLane", 1, 3.5), 0.0, 2.0, 0),
                _Segment(lanes[1], LaneMetadata("StraightLane", 1, 3.5), 2.0, 4.0, 1),
                _Segment(lanes[2], LaneMetadata("StraightLane", 1, 3.5), 4.0, 6.0, 2),
            ),
            diagnostics,
        )
        projection = project_to_path(path, (1.0, 1.0))
        self.assertFalse(projection.valid)
        self.assertEqual(projection.reason, "ambiguous_projection")

    def test_unvalidated_boundary_and_route_end_are_distinct(self) -> None:
        first = StraightLane((0.0, 0.0), (3.0, 0.0))
        disconnected = StraightLane((4.0, 0.0), (10.0, 0.0))
        prefix = _route(first, disconnected)
        preview = compute_preview(prefix.path, (0.0, 0.0), 0.0)
        self.assertFalse(preview.valid)
        self.assertEqual(preview.reason, "lookahead_past_unvalidated_boundary")
        end_preview = compute_preview(_route(first).path, (0.0, 0.0), 0.0)
        self.assertFalse(end_preview.valid)
        self.assertEqual(end_preview.reason, "lookahead_past_route_end")

    def test_invalid_state_values_and_motion_conditions_are_explicit(self) -> None:
        path = _route(StraightLane((0.0, 0.0), (8.0, 0.0))).path
        nan_preview = compute_preview(path, (math.nan, 0.0), 0.0)
        inf_preview = compute_preview(path, (0.0, 0.0), math.inf)
        reverse_preview = compute_preview(path, (0.0, 0.0), 0.0, forward_speed_mps=-0.11)
        stopped_preview = compute_preview(path, (0.0, 0.0), 0.0, forward_speed_mps=0.0)
        behind_preview = compute_preview(path, (0.0, 0.0), math.pi)
        self.assertEqual(nan_preview.observation, (0.5, 0.5, 0.0))
        self.assertIsNone(nan_preview.x_g)
        self.assertEqual(inf_preview.observation, (0.5, 0.5, 0.0))
        self.assertEqual(reverse_preview.reason, "reverse_motion")
        self.assertTrue(stopped_preview.valid)
        self.assertEqual(behind_preview.reason, "goal_not_in_front")
        self.assertEqual(compute_preview(path, (0.0, 0.0), 0.0, lookahead_m=0.0).reason, "goal_not_in_front")

    def test_clip_flags_report_finite_values_even_when_preview_is_invalid(self) -> None:
        path = _route(StraightLane((0.0, 0.0), (30.0, 0.0))).path
        behind = compute_preview(path, (50.0, 0.0), 0.0, lookahead_m=0.0)
        far_left = compute_preview(path, (0.0, -20.0), 0.0)
        self.assertFalse(behind.valid)
        self.assertEqual(behind.reason, "goal_not_in_front")
        self.assertAlmostEqual(behind.x_g or 0.0, -20.0)
        self.assertTrue(behind.x_clipped)
        self.assertTrue(far_left.valid)
        self.assertAlmostEqual(far_left.y_g or 0.0, 20.0)
        self.assertTrue(far_left.y_clipped)

    def test_programmer_errors_are_not_hidden_by_projection(self) -> None:
        lane = ProgrammerErrorLane((0.0, 0.0), (8.0, 0.0))
        diagnostics = RouteDiagnostics(
            total_lanes=1,
            validated_lanes=1,
            boundary_reason="route_end",
            boundary_lane_index=None,
            boundary_s=8.0,
        )
        path = ReferencePath(
            (_Segment(lane, LaneMetadata("StraightLane", 1, 3.5), 0.0, 8.0, 0),),
            diagnostics,
        )
        with self.assertRaisesRegex(RuntimeError, "simulated adapter bug"):
            project_to_path(path, (1.0, 0.0))

    def test_start_lane_invalid_does_not_fallback_to_nearest_lane(self) -> None:
        path = _route(StraightLane((0.0, 0.0), (8.0, 0.0))).path
        preview = compute_preview(path, (0.0, 0.0), 0.0, start_lane_valid=False)
        self.assertEqual(preview.observation, (0.5, 0.5, 0.0))
        self.assertEqual(preview.reason, "start_lane_unavailable")


class PurePursuitTests(unittest.TestCase):
    def _preview(self, lateral: float = 2.0) -> PreviewResult:
        path = _route(StraightLane((0.0, 0.0), (12.0, 0.0))).path
        return compute_preview(path, (0.0, -lateral), 0.0)

    def test_rear_wheelbase_and_left_right_sign(self) -> None:
        preview = self._preview(2.0)
        left = compute_pure_pursuit(
            preview,
            (0.0, -2.0),
            0.0,
            wheelbase_m=2.5,
            max_steering_deg=30.0,
            steering_sign=1.0,
            rear_wheelbase_m=1.5,
        )
        right = compute_pure_pursuit(
            preview,
            (0.0, -2.0),
            0.0,
            wheelbase_m=2.5,
            max_steering_deg=30.0,
            steering_sign=-1.0,
            rear_wheelbase_m=1.5,
        )
        self.assertTrue(left.pp_valid)
        self.assertAlmostEqual(left.x_rear or 0.0, 7.5)
        self.assertAlmostEqual(left.y_rear or 0.0, 2.0)
        self.assertGreater(left.u_pp or 0.0, 0.0)
        self.assertLess(right.u_pp or 0.0, 0.0)
        expected_kappa = 4.0 / (7.5 * 7.5 + 4.0)
        self.assertAlmostEqual(left.kappa_pp or 0.0, expected_kappa)

    def test_pure_pursuit_formula_and_saturation(self) -> None:
        result = pure_pursuit_from_rear_coordinates(
            7.0,
            2.0,
            wheelbase_m=2.5,
            max_steering_deg=30.0,
            steering_sign=1.0,
        )
        self.assertTrue(result.pp_valid)
        self.assertAlmostEqual(result.kappa_pp or 0.0, 4.0 / 53.0)
        self.assertAlmostEqual(
            result.delta_pp_rad or 0.0, math.atan(2.5 * 4.0 / 53.0)
        )
        saturated = pure_pursuit_from_rear_coordinates(
            0.1,
            20.0,
            wheelbase_m=2.5,
            max_steering_deg=1.0,
            steering_sign=1.0,
        )
        self.assertEqual(saturated.u_pp, 1.0)
        self.assertTrue(saturated.saturated)

    def test_invalid_preview_or_missing_rear_geometry_is_explicit(self) -> None:
        path = _route(StraightLane((0.0, 0.0), (12.0, 0.0))).path
        invalid = compute_preview(path, (0.0, 0.0), 0.0, start_lane_valid=False)
        with self.assertRaises(ValueError):
            compute_pure_pursuit(
                invalid,
                (0.0, 0.0),
                0.0,
                wheelbase_m=2.5,
                max_steering_deg=30.0,
                steering_sign=1.0,
            )
        result = compute_pure_pursuit(
            invalid,
            (0.0, 0.0),
            0.0,
            wheelbase_m=2.5,
            max_steering_deg=30.0,
            steering_sign=1.0,
            rear_wheelbase_m=1.5,
        )
        self.assertFalse(result.pp_valid)
        self.assertEqual(result.reason, "preview_invalid")

    def test_nonfinite_rear_position_is_invalid_but_bad_parameters_still_raise(self) -> None:
        preview = self._preview(1.0)
        result = compute_pure_pursuit(
            preview,
            (0.0, -1.0),
            0.0,
            wheelbase_m=2.5,
            max_steering_deg=30.0,
            steering_sign=1.0,
            rear_position=(math.nan, 0.0),
        )
        self.assertFalse(result.pp_valid)
        self.assertEqual(result.reason, "nonfinite_vehicle_geometry")
        self.assertIsNone(result.x_rear)
        with self.assertRaises(ValueError):
            compute_pure_pursuit(
                preview,
                (0.0, -1.0),
                0.0,
                wheelbase_m=0.0,
                max_steering_deg=30.0,
                steering_sign=1.0,
                rear_position=(math.nan, 0.0),
            )

    def test_connected_successor_rejects_ambiguous_candidates(self) -> None:
        previous = StraightLane((0.0, 0.0), (2.0, 0.0))
        candidate_a = StraightLane((2.0, 0.0), (4.0, 0.0))
        candidate_b = StraightLane((2.0, 0.0), (4.0, 0.0))
        accepted, diagnostics = connected_successor_candidates(
            previous,
            [candidate_a, candidate_b],
            previous_metadata=LaneMetadata("StraightLane", 1, 3.5),
            metadata=[LaneMetadata("StraightLane", 1, 3.5), LaneMetadata("StraightLane", 1, 3.5)],
        )
        self.assertEqual(accepted, ())
        self.assertEqual(diagnostics.boundary_reason, "ambiguous_successor")


class PenaltyTests(unittest.TestCase):
    def test_penalty_equation_and_masks(self) -> None:
        self.assertAlmostEqual(
            compute_pp_penalty(
                0.3,
                0.1,
                dt_s=0.1,
                pp_weight=1.0,
                pp_valid=True,
            ),
            -0.01,
        )
        self.assertEqual(
            compute_pp_penalty(
                None,
                None,
                dt_s=0.1,
                pp_weight=1.0,
                pp_valid=False,
            ),
            0.0,
        )
        self.assertEqual(
            compute_pp_penalty(
                0.3,
                0.1,
                dt_s=0.1,
                pp_weight=1.0,
                pp_valid=True,
                terminated=True,
            ),
            0.0,
        )
        self.assertEqual(
            compute_pp_penalty(
                0.3,
                0.1,
                dt_s=0.1,
                pp_weight=1.0,
                pp_valid=True,
                truncated=True,
            ),
            0.0,
        )

    def test_penalty_rejects_invalid_weight_dt_and_active_controls(self) -> None:
        with self.assertRaises(ValueError):
            compute_pp_penalty(0.0, 0.0, dt_s=-0.1, pp_weight=1.0, pp_valid=True)
        with self.assertRaises(ValueError):
            compute_pp_penalty(0.0, 0.0, dt_s=0.1, pp_weight=-1.0, pp_valid=True)
        with self.assertRaises(ValueError):
            compute_pp_penalty(2.0, 0.0, dt_s=0.1, pp_weight=1.0, pp_valid=True)
        with self.assertRaises(ValueError):
            compute_pp_penalty(
                2.0,
                None,
                dt_s=0.1,
                pp_weight=1.0,
                pp_valid=False,
            )
        with self.assertRaises(ValueError):
            compute_pp_penalty(
                None,
                -2.0,
                dt_s=0.1,
                pp_weight=1.0,
                pp_valid=True,
                terminated=True,
            )

    def test_result_dataclasses_are_immutable_and_serializable(self) -> None:
        lane = StraightLane((0.0, 0.0), (12.0, 0.0), lane_id="lane0")
        route = _route(lane)
        preview = compute_preview(route.path, (0.0, 0.0), 0.0)
        self.assertIn("s_goal", preview.as_dict())
        self.assertEqual(route.diagnostics.as_dict()["route_end"], True)
        with self.assertRaises(FrozenInstanceError):
            preview.valid = False  # type: ignore[misc]


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
