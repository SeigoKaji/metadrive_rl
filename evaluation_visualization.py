"""Evaluation telemetry collection and recorded-frame rendering helpers."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Pillow stores GIF frame durations in 10 ms units.  Keep this value as the
# project-default for callers that still import the constant, while
# ``derive_timing`` derives the effective duration from the runtime action dt.
# The official runtime (0.02 s physics step x 5 repeats) therefore resolves to
# exactly 100 ms per recorded post-step frame.
GIF_FRAME_DURATION_MS = 100
TELEMETRY_PANEL_WIDTH = 320
ACTION_HISTORY_SECONDS = 2.0


STEP_TELEMETRY_FIELDS: tuple[str, ...] = (
    "episode",
    "scenario_seed",
    "step",
    "horizon",
    "interval_start_seconds",
    "sim_time_seconds",
    "physics_hz",
    "control_hz",
    "action_duration_seconds",
    "gif_playback_vs_simulation",
    "action_id",
    "action_label",
    "decoded_steering",
    "decoded_throttle_brake",
    "applied_steering",
    "applied_throttle_brake",
    "speed_m_s",
    "speed_km_h",
    "lane_width_m",
    "lane_count_one_way",
    "current_segment_drivable_width_m",
    "current_segment_width_source",
    "center_to_left_boundary_m",
    "center_to_right_boundary_m",
    "lane_center_offset_m",
    "route_completion",
    "step_reward",
    "cumulative_reward",
    "terminated",
    "truncated",
    "arrive_dest",
    "out_of_road",
    "crash",
    "crash_vehicle",
    "crash_object",
    "max_step",
    "status",
    "action_switch_count",
    "action_switches_per_second",
)


@dataclass(frozen=True, slots=True)
class SimulationTiming:
    """Simulation-time relationship between physics, decisions, and recordings."""

    physics_step_seconds: float
    decision_repeat: int
    action_duration_seconds: float
    physics_hz: float
    control_hz: float
    gif_frame_duration_ms: int
    gif_playback_vs_simulation: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DecodedAction:
    """Raw discrete action and its normalized steering/longitudinal command."""

    action_id: int
    steering: float
    throttle_brake: float
    steering_label: str
    longitudinal_label: str

    @property
    def label(self) -> str:
        return f"{self.steering_label} + {self.longitudinal_label}"


@dataclass(frozen=True, slots=True)
class RuntimeRoadMetrics:
    """Geometry observed at the vehicle's current segment at runtime."""

    lane_width_m: float | None
    lane_count_one_way: int | None
    current_segment_drivable_width_m: float | None
    current_segment_width_source: str | None
    center_to_left_boundary_m: float | None
    center_to_right_boundary_m: float | None
    lane_center_offset_m: float | None


class ActionSwitchTracker:
    """Track cumulative raw-action changes within one episode.

    A switch is counted only when the current raw action ID differs from the
    preceding decision. ``switches_per_second`` is the cumulative switch count
    divided by elapsed simulation time, so the first decision always reports 0.
    """

    def __init__(self, *, history_length: int) -> None:
        if history_length <= 0:
            raise ValueError("history_length must be positive")
        self.previous_action_id: int | None = None
        self.switch_count = 0
        self.history: deque[int] = deque(maxlen=history_length)

    def observe(self, action_id: int, *, sim_time_seconds: float) -> tuple[int, float]:
        if sim_time_seconds <= 0:
            raise ValueError("sim_time_seconds must be positive for a post-step decision")
        if self.previous_action_id is not None and action_id != self.previous_action_id:
            self.switch_count += 1
        self.previous_action_id = action_id
        self.history.append(action_id)
        return self.switch_count, self.switch_count / sim_time_seconds


def derive_timing(
    config: Mapping[str, Any],
    *,
    gif_frame_duration_ms: int | None = None,
) -> SimulationTiming:
    """Derive effective frequencies from the merged runtime environment config.

    A recorded frame represents one post-step policy decision, so its playback
    duration is derived from the effective action duration.  GIF encoders
    quantize durations to 10 ms. The official 100 ms action interval is exact;
    custom intervals that cannot be represented exactly are rejected instead
    of being rounded into playback drift.
    """

    physics_step_seconds = float(config["physics_world_step_size"])
    decision_repeat = int(config["decision_repeat"])
    if physics_step_seconds <= 0:
        raise ValueError("physics_world_step_size must be positive")
    if decision_repeat <= 0:
        raise ValueError("decision_repeat must be positive")
    action_duration_seconds = physics_step_seconds * decision_repeat
    if gif_frame_duration_ms is None:
        exact_duration_ms = action_duration_seconds * 1000.0
        gif_frame_duration_ms = int(round(exact_duration_ms / 10.0) * 10)
        if gif_frame_duration_ms <= 0 or not math.isclose(
            gif_frame_duration_ms / 1000.0,
            action_duration_seconds,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "action duration cannot be represented exactly by GIF's 10 ms "
                f"duration unit: action_dt={action_duration_seconds:.9f}s"
            )
    if gif_frame_duration_ms <= 0:
        raise ValueError("gif_frame_duration_ms must be positive")
    if gif_frame_duration_ms % 10 != 0:
        raise ValueError("gif_frame_duration_ms must be a multiple of 10 ms")
    if not math.isclose(
        gif_frame_duration_ms / 1000.0,
        action_duration_seconds,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "gif_frame_duration_ms must equal the runtime action duration "
            f"({action_duration_seconds:.9f}s)"
        )

    return SimulationTiming(
        physics_step_seconds=physics_step_seconds,
        decision_repeat=decision_repeat,
        action_duration_seconds=action_duration_seconds,
        physics_hz=1.0 / physics_step_seconds,
        control_hz=1.0 / action_duration_seconds,
        gif_frame_duration_ms=gif_frame_duration_ms,
        gif_playback_vs_simulation=(
            action_duration_seconds / (gif_frame_duration_ms / 1000.0)
        ),
    )


def _single_action_id(action: Any) -> int:
    array = np.asarray(action)
    if array.size != 1:
        raise ValueError(f"expected one discrete action ID, got shape={array.shape}")
    scalar = array.reshape(-1)[0]
    action_id = int(scalar)
    if float(scalar) != float(action_id):
        raise ValueError(f"action ID must be integral: {scalar!r}")
    return action_id


def decode_discrete_action(action: Any, config: Mapping[str, Any]) -> DecodedAction:
    """Decode MetaDrive's single Discrete steering/throttle product action."""

    if not bool(config.get("discrete_action", False)):
        raise ValueError("evaluation visualization currently requires discrete_action=True")
    if bool(config.get("use_multi_discrete", False)):
        raise ValueError("a single action ID is unavailable when use_multi_discrete=True")

    steering_dim = int(config["discrete_steering_dim"])
    throttle_dim = int(config["discrete_throttle_dim"])
    if steering_dim < 2 or throttle_dim < 2:
        raise ValueError("discrete action dimensions must both be at least 2")

    action_id = _single_action_id(action)
    action_count = steering_dim * throttle_dim
    if not 0 <= action_id < action_count:
        raise ValueError(f"action ID {action_id} is outside Discrete({action_count})")

    steering = (action_id % steering_dim) * (2.0 / (steering_dim - 1)) - 1.0
    throttle_brake = (action_id // steering_dim) * (2.0 / (throttle_dim - 1)) - 1.0
    epsilon = 1e-9
    steering_label = (
        "LEFT"
        if steering > epsilon
        else ("RIGHT" if steering < -epsilon else "STRAIGHT")
    )
    vehicle_config = config.get("vehicle_config", {})
    enable_reverse = bool(
        config.get(
            "enable_reverse",
            vehicle_config.get("enable_reverse", False),
        )
    )
    if throttle_brake > epsilon:
        longitudinal_label = "THROTTLE"
    elif throttle_brake < -epsilon:
        longitudinal_label = "BRAKE" if not enable_reverse else "REVERSE/BRAKE"
    else:
        longitudinal_label = "NEUTRAL"

    return DecodedAction(
        action_id=action_id,
        steering=float(steering),
        throttle_brake=float(throttle_brake),
        steering_label=steering_label,
        longitudinal_label=longitudinal_label,
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def read_runtime_road_metrics(vehicle: Any) -> RuntimeRoadMetrics:
    """Best-effort current-segment values; unavailable metrics remain ``None``."""

    try:
        navigation = vehicle.navigation
    except Exception:
        navigation = None

    lane_width = None
    lane_count = None
    if navigation is not None:
        try:
            lane_width = _optional_float(navigation.get_current_lane_width())
        except Exception:
            lane_width = None
        try:
            raw_lane_count = navigation.get_current_lane_num()
            lane_count = None if raw_lane_count is None else int(raw_lane_count)
        except Exception:
            lane_count = None

    segment_width = None
    width_source = None
    if navigation is not None:
        try:
            segment_width = _optional_float(
                navigation.get_current_lateral_range(
                    vehicle.position,
                    vehicle.engine,
                )
            )
            if segment_width is not None:
                width_source = "navigation.get_current_lateral_range"
        except Exception:
            segment_width = None
    if segment_width is None and lane_width is not None and lane_count is not None:
        segment_width = lane_width * lane_count
        width_source = "lane_width_x_lane_count_fallback"

    lane_center_offset = None
    try:
        lane = vehicle.lane
        if lane is not None:
            _longitude, lateral = lane.local_coordinates(vehicle.position)
            lane_center_offset = float(lateral)
    except Exception:
        lane_center_offset = None

    try:
        center_to_left_boundary = _optional_float(vehicle.dist_to_left_side)
    except Exception:
        center_to_left_boundary = None
    try:
        center_to_right_boundary = _optional_float(vehicle.dist_to_right_side)
    except Exception:
        center_to_right_boundary = None

    return RuntimeRoadMetrics(
        lane_width_m=lane_width,
        lane_count_one_way=lane_count,
        current_segment_drivable_width_m=segment_width,
        current_segment_width_source=width_source,
        center_to_left_boundary_m=center_to_left_boundary,
        center_to_right_boundary_m=center_to_right_boundary,
        lane_center_offset_m=lane_center_offset,
    )


def step_status(*, terminated: bool, truncated: bool, info: Mapping[str, Any]) -> str:
    """Return a short panel/trace status with terminal causes taking precedence."""

    if bool(info.get("arrive_dest", False)):
        return "SUCCESS"
    if bool(info.get("out_of_road", False)):
        return "OUT_OF_ROAD"
    if bool(info.get("crash_vehicle", False)):
        return "CRASH_VEHICLE"
    if bool(info.get("crash_object", False)):
        return "CRASH_OBJECT"
    if bool(info.get("crash", False)):
        return "CRASH"
    if truncated or bool(info.get("max_step", False)):
        return "MAX_STEP"
    if terminated:
        return "TERMINATED"
    return "RUNNING"


def make_step_telemetry(
    *,
    episode: int,
    scenario_seed: int,
    step: int,
    horizon: int | None,
    timing: SimulationTiming,
    decoded_action: DecodedAction,
    info: Mapping[str, Any],
    reward: float,
    cumulative_reward: float,
    terminated: bool,
    truncated: bool,
    road: RuntimeRoadMetrics,
    action_switch_count: int,
    action_switches_per_second: float,
) -> dict[str, Any]:
    """Build one post-step JSON-safe telemetry record."""

    if step <= 0:
        raise ValueError("step must be one-based and positive")
    if "velocity" not in info:
        raise KeyError("MetaDrive step info is missing velocity")

    speed_m_s = float(info["velocity"])
    sim_time_seconds = step * timing.action_duration_seconds
    telemetry: dict[str, Any] = {
        "episode": int(episode),
        "scenario_seed": int(scenario_seed),
        "step": int(step),
        "horizon": None if horizon is None else int(horizon),
        "interval_start_seconds": sim_time_seconds - timing.action_duration_seconds,
        "sim_time_seconds": sim_time_seconds,
        "physics_hz": timing.physics_hz,
        "control_hz": timing.control_hz,
        "action_duration_seconds": timing.action_duration_seconds,
        "gif_playback_vs_simulation": timing.gif_playback_vs_simulation,
        "action_id": decoded_action.action_id,
        "action_label": decoded_action.label,
        "decoded_steering": decoded_action.steering,
        "decoded_throttle_brake": decoded_action.throttle_brake,
        "applied_steering": float(info.get("steering", decoded_action.steering)),
        # MetaDrive calls this info key "acceleration", but it is the normalized
        # throttle/brake command, not physical acceleration in m/s^2.
        "applied_throttle_brake": float(
            info.get("acceleration", decoded_action.throttle_brake)
        ),
        "speed_m_s": speed_m_s,
        "speed_km_h": speed_m_s * 3.6,
        "lane_width_m": road.lane_width_m,
        "lane_count_one_way": road.lane_count_one_way,
        "current_segment_drivable_width_m": road.current_segment_drivable_width_m,
        "current_segment_width_source": road.current_segment_width_source,
        "center_to_left_boundary_m": road.center_to_left_boundary_m,
        "center_to_right_boundary_m": road.center_to_right_boundary_m,
        "lane_center_offset_m": road.lane_center_offset_m,
        "route_completion": _optional_float(info.get("route_completion")),
        "step_reward": float(reward),
        "cumulative_reward": float(cumulative_reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "arrive_dest": info.get("arrive_dest"),
        "out_of_road": info.get("out_of_road"),
        "crash": info.get("crash"),
        "crash_vehicle": info.get("crash_vehicle"),
        "crash_object": info.get("crash_object"),
        "max_step": info.get("max_step"),
        "status": step_status(terminated=terminated, truncated=truncated, info=info),
        "action_switch_count": int(action_switch_count),
        "action_switches_per_second": float(action_switches_per_second),
    }
    if tuple(telemetry) != STEP_TELEMETRY_FIELDS:
        raise AssertionError("step telemetry schema and field declaration diverged")
    return telemetry


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    font_name = "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"
    try:
        return ImageFont.truetype(font_name, size=size)
    except OSError:
        return ImageFont.load_default()


def _number(value: Any, *, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}{suffix}"


def _history_colors(action: DecodedAction) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    steering_color = (
        (60, 150, 245)
        if action.steering > 0
        else ((245, 165, 65) if action.steering < 0 else (115, 125, 140))
    )
    throttle_color = (
        (55, 185, 105)
        if action.throttle_brake > 0
        else ((225, 75, 75) if action.throttle_brake < 0 else (115, 125, 140))
    )
    return steering_color, throttle_color


def compose_telemetry_panel(
    frame: np.ndarray,
    telemetry: Mapping[str, Any],
    action_history: Sequence[DecodedAction],
    *,
    panel_width: int = TELEMETRY_PANEL_WIDTH,
) -> np.ndarray:
    """Append a high-contrast telemetry panel without obscuring the map frame."""

    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected HxWx3 frame, got shape={array.shape}")
    if panel_width < 280:
        raise ValueError("panel_width must be at least 280 pixels")

    map_image = Image.fromarray(array.astype(np.uint8, copy=False))
    output = Image.new("RGB", (map_image.width + panel_width, map_image.height), (20, 24, 31))
    output.paste(map_image, (0, 0))
    draw = ImageDraw.Draw(output)
    panel_x = map_image.width
    x = panel_x + 14
    width = panel_width - 28
    y = 10
    body_font = _font(13)
    small_font = _font(11)
    heading_font = _font(14, bold=True)
    title_font = _font(16, bold=True)
    light = (232, 236, 242)
    muted = (165, 175, 188)
    cyan = (90, 205, 235)

    def line(
        text: str,
        *,
        color: tuple[int, int, int] = light,
        font: ImageFont.ImageFont = body_font,
        gap: int = 18,
    ) -> None:
        nonlocal y
        draw.text((x, y), text, fill=color, font=font)
        y += gap

    def divider() -> None:
        nonlocal y
        draw.line((x, y, x + width, y), fill=(65, 72, 84), width=1)
        y += 8

    line("EVALUATION TELEMETRY", color=cyan, font=title_font, gap=21)
    line(
        f"POST-STEP {telemetry['step']}/{telemetry.get('horizon') or '-'}  "
        f"t={telemetry['sim_time_seconds']:.2f}s",
        font=heading_font,
        gap=19,
    )
    line(
        f"ACTION APPLIED [{telemetry['interval_start_seconds']:.2f}, "
        f"{telemetry['sim_time_seconds']:.2f}]s",
        color=muted,
        font=small_font,
        gap=18,
    )
    divider()

    line("SIMULATION TIMING (runtime config)", font=heading_font)
    line(f"PHYSICS       {telemetry['physics_hz']:.1f} Hz")
    line(f"CONTROL       {telemetry['control_hz']:.1f} Hz")
    divider()

    line("VEHICLE / APPLIED ACTION", font=heading_font)
    line(
        f"SPEED         {telemetry['speed_km_h']:.1f} km/h  "
        f"({telemetry['speed_m_s']:.2f} m/s)"
    )
    line(f"ACTION {telemetry['action_id']}     {telemetry['action_label']}")
    line(f"STEERING      {telemetry['applied_steering']:+.2f}")
    line(f"THROTTLE/BRK  {telemetry['applied_throttle_brake']:+.2f}")
    line(f"SWITCH COUNT  {telemetry['action_switch_count']} (cumulative)")
    line(
        f"SWITCH RATE   {telemetry['action_switches_per_second']:.2f}/s "
        "(sim avg)"
    )
    divider()

    line("CURRENT SEGMENT (runtime)", color=cyan, font=heading_font)
    line(f"LANE WIDTH    {_number(telemetry.get('lane_width_m'), suffix=' m')}")
    lane_count = telemetry.get("lane_count_one_way")
    line(f"LANES         {lane_count if lane_count is not None else 'N/A'} one-way")
    line(
        "DRIVABLE      "
        + _number(telemetry.get("current_segment_drivable_width_m"), suffix=" m")
    )
    line(
        "CENTER->EDGE  L "
        + _number(telemetry.get("center_to_left_boundary_m"), suffix=" m")
        + "  R "
        + _number(telemetry.get("center_to_right_boundary_m"), suffix=" m")
    )
    divider()

    line("TASK", font=heading_font)
    route = telemetry.get("route_completion")
    line(f"ROUTE         {'N/A' if route is None else f'{route * 100:.1f}%'}")
    line(
        f"REWARD        {telemetry['step_reward']:+.3f}  "
        f"total {telemetry['cumulative_reward']:+.3f}"
    )
    status = str(telemetry["status"])
    status_color = (
        (70, 205, 115)
        if status in {"RUNNING", "SUCCESS"}
        else ((240, 190, 70) if status == "MAX_STEP" else (235, 80, 80))
    )
    line(f"STATUS        {status}", color=status_color, font=heading_font, gap=20)

    line(
        f"ACTION HISTORY (last {ACTION_HISTORY_SECONDS:.0f}s)",
        font=small_font,
        color=muted,
        gap=15,
    )
    history = list(action_history)
    if history:
        label_width = 34
        cell_gap = 1
        cell_width = max(3, (width - label_width - cell_gap * (len(history) - 1)) // len(history))
        cell_width = min(cell_width, 12)
        start_x = x + label_width
        draw.text((x, y - 1), "S", fill=muted, font=small_font)
        draw.text((x, y + 10), "T/B", fill=muted, font=small_font)
        for index, action in enumerate(history):
            cell_x = start_x + index * (cell_width + cell_gap)
            steering_color, throttle_color = _history_colors(action)
            draw.rectangle((cell_x, y, cell_x + cell_width - 1, y + 7), fill=steering_color)
            draw.rectangle((cell_x, y + 11, cell_x + cell_width - 1, y + 18), fill=throttle_color)
        draw.text(
            (x, y + 23),
            "S: blue L / orange R   T/B: green / red",
            fill=muted,
            font=small_font,
        )

    return np.asarray(output)


def replace_latest_recorded_frame(renderer: Any, frame: np.ndarray) -> None:
    """Replace MetaDrive 0.4.3's latest recorded frame with a local composite.

    MetaDrive exposes ``screen_frames`` as a deep-copy-only property. The
    version-pinned evaluator therefore contains this single private-API touch
    here instead of modifying the upstream renderer or spreading the dependency.
    """

    frames = getattr(renderer, "_screen_frames", None)
    if not isinstance(frames, list) or not frames:
        raise RuntimeError("MetaDrive 0.4.3 renderer has no recorded frame to replace")
    frames[-1] = np.asarray(frame)
