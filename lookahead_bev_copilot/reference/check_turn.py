"""Verify the recorded BEV markers on a short MetaDrive left turn."""

from contextlib import ExitStack
import os
from pathlib import Path
import tempfile
import tomllib


SCENARIO_SEED = 0
STEERING_LEVELS = 81
SPEED_MPS = 10.0


def record_turn(output_dir: Path) -> int:
    # Import the runtime after main has placed its cache under the temp directory.
    import numpy as np
    from PIL import Image

    from env_factory import make_evaluation_env
    from evaluation_results import _EpisodeVisualizationRecorder
    from evaluation_visualization import (
        ACTION_HISTORY_SECONDS,
        _DECISION_LOOKAHEAD_COLOR,
        _REFERENCE_ROUTE_COLOR,
        ActionSwitchTracker,
        decode_discrete_action,
        derive_timing,
        make_step_telemetry,
        read_runtime_road_metrics,
    )

    source = Path("configs/official_start_lane_return_lookahead.toml")
    settings = tomllib.loads(source.read_text(encoding="utf-8"))
    env_config = dict(
        settings["environment"]["common"],
        map="SXS",
        start_seed=SCENARIO_SEED,
        num_scenarios=1,
        discrete_steering_dim=STEERING_LEVELS,
        vehicle_config={
            "spawn_longitude": 35.0,
            "spawn_velocity": (SPEED_MPS, 0.0),
            "spawn_velocity_car_frame": True,
        },
    )
    # Enable the PP reference for the demo driver. No model is loaded or trained.
    lookahead = dict(settings["lookahead"], lookahead_m=6.0, pp_weight=0.1)

    with ExitStack() as resources:
        env = make_evaluation_env(0, env_config, lookahead)
        resources.callback(env.close)
        host = env.unwrapped
        _, info = env.reset(seed=SCENARIO_SEED)
        path = env.get_wrapper_attr("reference_path")
        assert path is not None and path.diagnostics.route_end, "Expected a connected SXS route"
        curve_index = next(i for i, lane in enumerate(path.lanes) if type(lane).__name__ == "CircularLane")
        turn_start = path.lane_starts[curve_index]
        timing = derive_timing(host.config)
        tracker = ActionSwitchTracker(history_length=round(ACTION_HISTORY_SECONDS * timing.control_hz))
        recorder = _EpisodeVisualizationRecorder(
            requested=True,
            run_dir=output_dir,
            episode_number=1,
            scenario_seed=SCENARIO_SEED,
            reference_path=path,
        )
        resources.callback(recorder.cleanup)
        recorder.prepare()
        recorder.set_timing(timing)

        cumulative_reward = 0.0
        goal_distances = []
        marker_centers = set()
        for step in range(1, 201):
            state = info["lookahead_learning"]["post_step"]
            pp = state["pp"]
            assert pp["pp_valid"], (step, pp)
            steering_index = round((pp["u_pp"] + 1.0) * (STEERING_LEVELS - 1) / 2.0)
            throttle_index = 2 if host.agent.speed < SPEED_MPS else 1
            action = throttle_index * STEERING_LEVELS + steering_index
            _, reward, terminated, truncated, info = env.step(action)
            preview = info["lookahead_learning"]["pre_step"]["preview"]
            assert preview == state["preview"], "The displayed goal must be the decision input"
            assert preview["preview_valid"], (step, preview["invalid_reason"])
            if preview["s_proj_m"] > turn_start + 18.0:
                break

            cumulative_reward += float(reward)
            switch_count, switch_rate = tracker.observe(
                action, sim_time_seconds=step * timing.action_duration_seconds,
            )
            if preview["s_proj_m"] >= turn_start - 10.0:
                telemetry = make_step_telemetry(
                    episode=1,
                    scenario_seed=SCENARIO_SEED,
                    step=step,
                    horizon=int(host.config["horizon"]),
                    timing=timing,
                    decoded_action=decode_discrete_action(action, host.config),
                    info=info,
                    reward=float(reward),
                    cumulative_reward=cumulative_reward,
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    road=read_runtime_road_metrics(host.agent),
                    action_switch_count=switch_count,
                    action_switches_per_second=switch_rate,
                )
                recorder.record_frame(env=host, telemetry=telemetry, switch_tracker=tracker)
                assert recorder.render_exception_trace is None, recorder.render_exception_trace
                goal_distances.append(preview["s_goal_m"])

                renderer = host.top_down_renderer
                width, height = renderer.screen_canvas.get_size()
                png = recorder.paths["frames"] / f"frame_{len(goal_distances):06d}.png"
                with Image.open(png) as image:
                    bev = np.asarray(image)[:height, :width]
                assert np.any(np.all(bev == _REFERENCE_ROUTE_COLOR, axis=2)), (step, "no route")
                rows, cols = np.nonzero(np.all(bev == _DECISION_LOOKAHEAD_COLOR, axis=2))
                assert rows.size, (step, "no marker")
                observed = ((cols.min() + cols.max()) / 2, (rows.min() + rows.max()) / 2)
                # Use the renderer's actual map-to-film transform and blit crop.
                pos2pix = renderer._frame_canvas.pos2pix
                camera_x, camera_y = pos2pix(*renderer.position)
                goal_x, goal_y = pos2pix(*preview["q_xy"])
                expected = (goal_x - int(camera_x - width / 2), goal_y - int(camera_y - height / 2))
                assert observed == expected, (step, observed, expected)
                marker_centers.add(observed)
            assert not (terminated or truncated), (step, "ended before recording the turn")

        count = len(goal_distances)
        assert count > 1 and goal_distances[0] < turn_start < goal_distances[-1], "Goal did not cross the turn entrance"
        assert len(marker_centers) == count, "Marker did not move on each recorded frame"
        result = recorder.finalize(env=host, timing=timing, expected_frame_count=count)
        for kind in ("frames", "gif", "mp4"):
            assert result[kind]["status"] == "success", result[kind]
        return count


def main() -> None:
    output_dir = Path(tempfile.mkdtemp(prefix="metadrive-lookahead-bev-"))
    print(f"SXS seed {SCENARIO_SEED} output: {output_dir}", flush=True)
    os.environ["MPLCONFIGDIR"] = str(output_dir / "matplotlib")
    count = record_turn(output_dir)
    print(f"Verified decision-time marker pixels for {count} frames: {output_dir}")


if __name__ == "__main__":
    main()
