"""Pure stdlib checks for :mod:`lookahead_learning.telemetry`.

Run with ``PYTHONDONTWRITEBYTECODE=1 python -B -m unittest -v
lookahead_learning.test_telemetry``.  These tests intentionally use synthetic rows so
they do not create a MetaDrive engine or depend on optional packages.
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from .telemetry import (
    JsonlWorkerWriter,
    compare_summaries,
    evaluate_improvement_gate,
    read_jsonl,
    sanitize_json,
    summarize_rows,
)


def _row(
    *,
    mode: str = "baseline",
    episode: int = 0,
    decision: int = 0,
    dt: float = 0.1,
    lateral: float | None = 0.2,
    speed: float | None = 1.0,
    progress: float | None = None,
    steering: float | None = 0.0,
    history_valid: bool = False,
    preview_valid: bool | None = True,
    pp_valid: bool | None = True,
    run: str | None = None,
    worker: int | None = None,
    schema_shape: int = 265,
    **extra: object,
) -> dict[str, object]:
    row: dict[str, object] = {
        "mode": mode,
        "learning_seed": 7,
        "scenario_seed": 5,
        "evaluation_seed": 9,
        "episode": episode,
        "decision": decision,
        "dt": dt,
        "lateral_error_m": lateral,
        "speed_mps": speed,
        "u_applied": steering,
        "history_valid": history_valid,
        "preview_valid": preview_valid,
        "pp_valid": pp_valid,
        "in_target_lane": True,
        "observation_schema": {
            "shape": [schema_shape],
            "prefix": {"shape": [262], "dtype": "float32", "fields": "base262"},
            "extra_fields": (
                ["preview_x_norm", "preview_y_norm", "preview_valid"]
                if schema_shape == 265
                else []
            ),
        },
        "action_space": {"type": "discrete", "n": 3},
        "wrapper_order": ["base", "lookahead_learning", "monitor"],
    }
    if progress is not None:
        row["progress_m"] = progress
    if run is not None:
        row["run"] = run
    if worker is not None:
        row["worker"] = worker
    row.update(extra)
    return row


class TelemetryTest(unittest.TestCase):
    def test_sanitize_nonfinite_to_json_null(self) -> None:
        value = sanitize_json({"nan": float("nan"), "inf": float("inf"), "ok": 2.0})
        self.assertEqual(value, {"nan": None, "inf": None, "ok": 2.0})
        self.assertEqual(json.loads(json.dumps(value, allow_nan=False))["nan"], None)

    def test_worker_writer_namespaces_and_flushes_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with JsonlWorkerWriter(directory, "run one", 2, metadata={"mode": "lookahead_obs"}) as writer:
                writer.write({"episode": 3, "decision": 0, "bad": float("nan")})
                path = writer.path
            self.assertEqual(path.name, "run-run_one-worker-2.jsonl")
            raw = read_jsonl(path)
            self.assertEqual(raw[0]["lookahead_learning"]["mode"], "lookahead_obs")
            self.assertIsNone(raw[0]["lookahead_learning"]["bad"])
            self.assertEqual(raw[0]["worker"], 2)

    def test_worker_writer_normalizes_aliases_and_historical_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with JsonlWorkerWriter(
                directory,
                "legacy replay",
                0,
                metadata={"mode": "obs_pp"},
            ) as writer:
                writer.write({"preview_ab": {"mode": "obs_pp", "value": 1}})
                path = writer.path
            raw = read_jsonl(path)
            self.assertNotIn("preview_ab", raw[0])
            self.assertEqual(
                raw[0]["lookahead_learning"]["mode"],
                "lookahead_obs_pp_reward",
            )

    def test_summary_reads_historical_namespace_and_mode_alias(self) -> None:
        historical = {"preview_ab": _row(mode="obs")}
        summary = summarize_rows([historical])
        self.assertEqual(summary["metadata"]["mode"], "lookahead_obs")
        self.assertAlmostEqual(summary["metrics"]["lateral_rms_m"], 0.2)

    def test_worker_writer_is_exclusive_and_rejects_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            writer = JsonlWorkerWriter(directory, "run", 0)
            writer.close()
            with self.assertRaises(FileExistsError):
                JsonlWorkerWriter(directory, "run", 0)
            with self.assertRaises(ValueError):
                JsonlWorkerWriter(directory, "run", 1, filename="../outside.jsonl")
            # Explicit resume is available when continuation is intentional.
            resumed = JsonlWorkerWriter(directory, "run", 0, resume=True)
            resumed.close()

    def test_host_invalid_lateral_is_missing_and_not_a_zero(self) -> None:
        rows = [
            _row(lateral=1.0, decision=0, target_lane_valid=True),
            _row(
                lateral=99.0,
                decision=1,
                preview_valid=False,
                target_lane_valid=False,
                preview_invalid_reason="route_end",
            ),
        ]
        summary = summarize_rows(rows)
        metrics = summary["metrics"]
        self.assertAlmostEqual(metrics["lateral_rms_m"], 1.0)
        self.assertAlmostEqual(metrics["lateral_abs_p95_m"], 1.0)
        self.assertEqual(metrics["lateral_missing_count"], 1)
        self.assertAlmostEqual(metrics["preview_valid_rate"], 0.5)
        self.assertEqual(metrics["invalid_reason_time_s"], {"route_end": 0.1})

    def test_preview_invalid_keeps_explicit_host_lateral(self) -> None:
        summary = summarize_rows(
            [
                _row(
                    lateral=1.0,
                    preview_valid=False,
                    target_lane_valid=True,
                    preview_invalid_reason="route_end",
                )
            ]
        )
        metrics = summary["metrics"]
        self.assertAlmostEqual(metrics["lateral_rms_m"], 1.0)
        self.assertAlmostEqual(metrics["lateral_time_s"], 0.1)
        self.assertEqual(metrics["lateral_missing_count"], 0)
        self.assertEqual(metrics["preview_valid_rate"], 0.0)

    def test_metrics_use_actual_dt_and_positive_progress_distance(self) -> None:
        rows = [
            _row(decision=0, dt=0.1, lateral=0.0, progress=0.0, steering=0.3),
            _row(decision=1, dt=0.2, lateral=0.4, progress=0.2, steering=0.1, history_valid=True),
        ]
        summary = summarize_rows(rows)
        metrics = summary["metrics"]
        self.assertAlmostEqual(metrics["duration_s"], 0.3)
        self.assertAlmostEqual(metrics["lateral_rms_m"], math.sqrt(0.4 * 0.4 * 0.2 / 0.3))
        self.assertAlmostEqual(metrics["steering_tv_per_s"], 0.2 / 0.2)
        self.assertAlmostEqual(metrics["steering_rate_rms"], 1.0)
        # Route progress is a path coordinate, not physical driven distance;
        # TV/km stays unknown until a distance measurement is recorded.
        self.assertIsNone(metrics["steering_tv_per_m"])
        self.assertEqual(metrics["physical_distance_missing_condition"], True)
        self.assertAlmostEqual(metrics["progress_per_s"], 1.0)

    def test_physical_distance_does_not_follow_route_progress(self) -> None:
        rows = [
            _row(decision=0, progress=0.0, steering=0.2, history_valid=False),
            _row(decision=1, progress=1.0, steering=0.3, history_valid=True),
        ]
        no_distance = summarize_rows(rows)["metrics"]
        self.assertIsNone(no_distance["distance_m"])
        self.assertIsNone(no_distance["steering_tv_per_m"])
        self.assertGreater(no_distance["progress_m"], 0.0)

        with_distance = summarize_rows(
            [
                {**rows[0], "distance_delta_m": 0.6},
                {**rows[1], "distance_delta_m": 0.8},
            ]
        )["metrics"]
        self.assertAlmostEqual(with_distance["distance_m"], 1.4)
        self.assertIsNotNone(with_distance["steering_tv_per_m"])

    def test_negative_physical_distance_is_invalid_and_not_absed(self) -> None:
        summary = summarize_rows([_row(distance_delta_m=-1.0)])
        metrics = summary["metrics"]
        self.assertIsNone(metrics["distance_m"])
        self.assertEqual(metrics["physical_distance_invalid_count"], 1)
        self.assertEqual(metrics["physical_distance_missing_condition"], True)

    def test_position_displacement_is_labeled_approximation(self) -> None:
        summary = summarize_rows(
            [
                _row(
                    position_before=[0.0, 0.0],
                    position_after=[0.3, 0.4],
                )
            ]
        )
        metrics = summary["metrics"]
        self.assertAlmostEqual(metrics["distance_m"], 0.5)
        self.assertEqual(metrics["physical_distance_approximation_count"], 1)

    def test_incomplete_episode_is_retained(self) -> None:
        summary = summarize_rows([_row(decision=0, success=None)])
        metrics = summary["metrics"]
        self.assertEqual(metrics["episode_count"], 1)
        self.assertEqual(metrics["incomplete_episode_count"], 1)
        self.assertEqual(metrics["success_missing_count"], 1)
        self.assertIsNone(metrics["episodes"][0]["success"])

    def test_workers_are_separate_but_run_names_are_not_comparison_case(self) -> None:
        rows = [
            _row(run="old", worker=0, episode=0),
            _row(run="old", worker=1, episode=0),
        ]
        summary = summarize_rows(rows)
        self.assertEqual(summary["metrics"]["episode_count"], 2)
        self.assertEqual(len(summary["metadata"]["episode_cases"]), 2)

        before = summarize_rows(
            [_row(run="baseline-run", worker=0, episode=0)],
            metadata={"mode": "baseline"},
        )
        after = summarize_rows(
            [_row(run="obs-run", worker=0, episode=0)],
            metadata={"mode": "lookahead_obs"},
        )
        result = compare_summaries(before, after, require_matched=False, strict_metadata=False)
        self.assertTrue(result["metadata"]["episode_cases_equal"])

    def test_baseline_262_to_obs_265_schema_checks_common_prefix(self) -> None:
        before = summarize_rows(
            [_row(mode="baseline", schema_shape=262)],
            metadata={"mode": "baseline"},
        )
        after = summarize_rows(
            [_row(mode="lookahead_obs", schema_shape=265)],
            metadata={"mode": "lookahead_obs"},
        )
        result = compare_summaries(
            before,
            after,
            require_matched=False,
            strict_metadata=False,
        )
        self.assertTrue(result["metadata"]["observation_schema_compatible"])
        self.assertFalse(result["metadata"]["observation_schema_equal"])
        self.assertTrue(result["metadata"]["wrapper_order_compatible"])

    def test_completion_and_preview_deterioration_are_rejected(self) -> None:
        before = summarize_rows(
            [_row(mode="baseline", lateral=0.4, preview_valid=True, success=True, arrive_dest=True, episode_end=True)],
            metadata={"mode": "baseline"},
        )
        after = summarize_rows(
            [_row(mode="lookahead_obs", lateral=0.2, preview_valid=False, success=False, episode_end=True)],
            metadata={"mode": "lookahead_obs"},
        )
        gate = evaluate_improvement_gate(
            before, after, strict_metadata=False, require_matched=False
        )
        self.assertIn("completion_rate", gate["failed_conditions"])
        self.assertIn("preview_valid_rate", gate["failed_conditions"])

    def test_directional_gate_can_accept_improved_completion(self) -> None:
        common = {
            "start_lane_maintained_rate": 1.0,
            "stop_time_fraction": 0.0,
            "preview_valid_rate": 1.0,
            "speed_mean_mps": 1.0,
            "progress_per_s": 1.0,
            "completion_time_s_mean": 1.0,
        }
        before = {
            "metadata": {"mode": "baseline", "episode_cases": ["case"]},
            "metrics": {
                **common,
                "completion_rate": 0.5,
                "lateral_rms_m": 0.4,
                "lateral_abs_p95_m": 0.5,
                "steering_tv_per_s": 0.5,
                "steering_rate_rms": 0.5,
            },
        }
        after = {
            "metadata": {"mode": "lookahead_obs", "episode_cases": ["case"]},
            "metrics": {
                **common,
                "completion_rate": 1.0,
                "lateral_rms_m": 0.3,
                "lateral_abs_p95_m": 0.4,
                "steering_tv_per_s": 0.4,
                "steering_rate_rms": 0.4,
            },
        }
        gate = evaluate_improvement_gate(
            before, after, strict_metadata=False, require_matched=False
        )
        self.assertNotIn("completion_rate", gate["failed_conditions"])
        # Missing fairness metadata keeps the result inconclusive even when
        # the directional completion condition itself passes.
        self.assertEqual(gate["status"], "inconclusive")

    def test_large_speed_loss_is_warning_and_failure(self) -> None:
        common = {
            "completion_rate": 1.0,
            "start_lane_maintained_rate": 1.0,
            "stop_time_fraction": 0.0,
            "preview_valid_rate": 1.0,
            "lateral_rms_m": 0.4,
            "lateral_abs_p95_m": 0.5,
            "steering_tv_per_s": 0.5,
            "steering_rate_rms": 0.5,
            "progress_per_s": 1.0,
            "completion_time_s_mean": 1.0,
        }
        before = {"metadata": {"mode": "baseline", "episode_cases": ["case"]}, "metrics": {**common, "speed_mean_mps": 1.0}}
        after = {"metadata": {"mode": "lookahead_obs", "episode_cases": ["case"]}, "metrics": {**common, "speed_mean_mps": 0.8}}
        gate = evaluate_improvement_gate(before, after, strict_metadata=False, require_matched=False)
        self.assertTrue(gate["warnings"])
        self.assertTrue(any("condition warning" in item for item in gate["failed_conditions"]))

    def test_sign_reversal_uses_measurement_specific_validity(self) -> None:
        rows = [
            _row(
                decision=0,
                steering=0.5,
                lateral=0.5,
                target_lane_valid=True,
                history_valid=False,
            ),
            _row(
                decision=1,
                steering=-0.5,
                lateral=-0.5,
                preview_valid=False,
                target_lane_valid=True,
                history_valid=True,
            ),
            _row(
                decision=2,
                steering=-0.5,
                lateral=-0.5,
                preview_valid=False,
                target_lane_valid=True,
                history_valid=True,
            ),
        ]
        summary = summarize_rows(rows)
        self.assertEqual(summary["episodes"][0]["steering_sign_reversal_count"], 1)
        # The preview target is invalid, but host target-lane lateral values
        # remain valid and the lateral crossing is counted independently.
        self.assertEqual(summary["episodes"][0]["lateral_center_crossing_count"], 1)

        host_invalid = rows[:1] + [
            _row(
                decision=1,
                steering=-0.5,
                lateral=-0.5,
                preview_valid=False,
                target_lane_valid=False,
                history_valid=True,
            )
        ] + rows[2:]
        host_invalid_summary = summarize_rows(host_invalid)
        self.assertEqual(
            host_invalid_summary["episodes"][0]["steering_sign_reversal_count"],
            1,
        )
        self.assertEqual(
            host_invalid_summary["episodes"][0]["lateral_center_crossing_count"],
            0,
        )

    def test_matched_road_and_speed_bins_are_explicit(self) -> None:
        intervals = {"near": (0.0, 1.0), "far": (1.0, 2.0)}
        speeds = {"slow": (0.0, 0.5), "moving": (0.5, 2.0)}
        before = summarize_rows(
            [_row(progress=0.2, speed_mps=1.0, lateral=0.4)],
            metadata={"mode": "baseline"},
            road_progress_intervals=intervals,
            speed_bins=speeds,
        )
        after = summarize_rows(
            [_row(mode="lookahead_obs", progress=0.2, speed_mps=1.0, lateral=0.2)],
            metadata={"mode": "lookahead_obs"},
            road_progress_intervals=intervals,
            speed_bins=speeds,
        )
        result = compare_summaries(before, after, require_matched=False)
        self.assertEqual(result["matched"]["common_bin_count"], 1)
        self.assertTrue(result["matched"]["bins"])

    def test_comparison_pair_and_metadata_are_checked(self) -> None:
        before_rows = [_row(lateral=0.4, mode="baseline", success=True, arrive_dest=True, episode_end=True)]
        after_rows = [_row(lateral=0.2, mode="lookahead_obs", success=True, arrive_dest=True, episode_end=True)]
        before = summarize_rows(before_rows, metadata={"mode": "baseline"})
        after = summarize_rows(after_rows, metadata={"mode": "lookahead_obs"})
        gate = evaluate_improvement_gate(before, after, require_matched=False)
        self.assertFalse(gate["improved"])
        self.assertIn("steering change metric did not improve", gate["failed_conditions"])

        invalid = compare_summaries(
            before,
            after,
            before_mode="baseline",
            after_mode="lookahead_obs_pp_reward",
            require_matched=False,
        )
        self.assertFalse(invalid["metadata"]["allowed_pair"])
        self.assertFalse(invalid["improved"])


if __name__ == "__main__":
    unittest.main()
