"""Source backed observation schema and fixed stress presets.

The official MetaDrive state observation used by this project is a flat
``float32`` vector of 259 values.  This module keeps the mapping in one place
so that an intervention can name a group without guessing from the vector
width.  The 240 LiDAR rows are generated from the upstream ray formula; they
are not abbreviated into one ``"lidar"`` row.

The module deliberately has no MetaDrive, Gymnasium, NumPy, or Stable
Baselines imports.  It can therefore be used by the report-only command on a
machine that only has the saved data and the Python standard library.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import copy
import csv
import math
from pathlib import Path
import tomllib
from typing import Any


SCHEMA_VERSION = "input_attribution.schema.v1"
METADRIVE_VERSION = "0.4.3"
METADRIVE_COMMIT = "85e5dadc6c7436d324348f6e3d8f8e680c06b4db"
STANDARD_DIMENSION = 259
TEMPLATE_DIMENSION = 262
LIDAR_START_INDEX = 19
LIDAR_COUNT = 240
LIDAR_END_INDEX = LIDAR_START_INDEX + LIDAR_COUNT - 1
LIDAR_ANGLE_STEP_DEG = 360.0 / LIDAR_COUNT


class SchemaError(ValueError):
    """Raised when an input schema or intervention pattern is ambiguous."""


_STATE_SOURCE = {
    "source": (
        "MetaDrive 0.4.3: metadrive/obs/state_obs.py "
        "(StateObservation.vehicle_state)"
    ),
    "source_version": METADRIVE_VERSION,
    "source_commit": METADRIVE_COMMIT,
    "source_path": "metadrive/obs/state_obs.py",
    "source_import": "metadrive.obs.state_obs.StateObservation",
    "source_dirty": False,
}

_NAV_SOURCE = {
    "source": (
        "MetaDrive 0.4.3: "
        "metadrive/component/navigation_module/node_network_navigation.py "
        "(NodeNetworkNavigation._get_info_for_checkpoint)"
    ),
    "source_version": METADRIVE_VERSION,
    "source_commit": METADRIVE_COMMIT,
    "source_path": "metadrive/component/navigation_module/node_network_navigation.py",
    "source_import": (
        "metadrive.component.navigation_module.node_network_navigation."
        "NodeNetworkNavigation"
    ),
    "source_dirty": False,
}

_LIDAR_SOURCE = {
    "source": (
        "MetaDrive 0.4.3: "
        "metadrive/component/sensors/distance_detector.py and "
        "metadrive/utils/math.py"
    ),
    "source_version": METADRIVE_VERSION,
    "source_commit": METADRIVE_COMMIT,
    "source_path": "metadrive/component/sensors/distance_detector.py",
    "source_import": "metadrive.component.sensors.distance_detector.DistanceDetector",
    "source_dirty": False,
}

# This source describes the deterministic test generator only.  It is kept
# separate from the MetaDrive provenance above so a synthetic 262-wide smoke
# test can never be mistaken for evidence about a real host observation.
_SYNTHETIC_SOURCE = {
    "source": (
        "input_attribution synthetic contract: SyntheticEnv._observation "
        "(test-only; no MetaDrive semantics)"
    ),
    "source_version": SCHEMA_VERSION,
    "source_commit": None,
    "source_path": "input_attribution/synthetic.py",
    "source_import": "input_attribution.synthetic.SyntheticEnv",
    "source_dirty": None,
}

_SOURCE_FILE_SHA256 = {
    "metadrive/obs/state_obs.py": "2bf137e9f19388faa12069b93537b4d4f69cd1eca3d123c014637c8954c05c82",
    "metadrive/component/navigation_module/node_network_navigation.py": (
        "a423fdc5238d95aa4d1e01c6d614eff231d9e40fc24d40e7828c6874da05388f"
    ),
    "metadrive/component/sensors/distance_detector.py": (
        "ab12f0b9fd9ed9e5262d42c99294e007e20b361499767eee6dc527fb4eacfdf0"
    ),
    "metadrive/utils/math.py": "0f96c6f59e9141cde0e6c3e4acfee8656a6062c9821056eceaf2d79a182679ef",
    "metadrive/component/sensors/lidar.py": (
        "a48e02d5ab49053b005fa4829db343af42e6e637f6c87e7ea84ce1cc7ef9683c"
    ),
    "metadrive/component/map/base_map.py": (
        "2f1b60cd529aa495e871de8a79490abbc26e7e032b9c7683a773bbb98682ec91"
    ),
    "metadrive/component/vehicle/base_vehicle.py": (
        "1660147eddb47faa8c2496bbef37ce369f6f82406a563cd34db7d9d9e4256f5a"
    ),
    "metadrive/component/pg_space.py": (
        "b87325023e7612913a24c447e8d4612f438595e7b8a357455db962fa4085134d"
    ),
    "metadrive/base_class/base_object.py": (
        "edfc48d63acdefafe5cff202374a6cf8ed06ffb327f01b55052352b0ddb74b50"
    ),
    "metadrive/component/lane/straight_lane.py": (
        "b27f2def9aee09fd2fa8465e7b51c9c549e2d40ff34e6892f898fa70d7b7552f"
    ),
    "metadrive/component/lane/circular_lane.py": (
        "45ddd24b4746313e4e5fd6d49548f041a2c891df8761624c8c185a5955c5be51"
    ),
}


def _source_details(source: Mapping[str, Any], *, lines: str, extra_paths: Sequence[str] = ()) -> dict[str, Any]:
    """Return immutable provenance fields copied into every feature row."""

    result = dict(source)
    result["source_lines"] = lines
    refs = [str(result["source_path"]), *[str(path) for path in extra_paths]]
    result["source_refs"] = refs
    result["source_file_sha256"] = {
        path: _SOURCE_FILE_SHA256[path]
        for path in refs
        if path in _SOURCE_FILE_SHA256
    }
    return result


def _feature(
    index: int,
    name: str,
    group: str,
    normalization: str,
    zero_meaning: str,
    one_meaning: str,
    *,
    normal_range: Sequence[float] | None = (0.0, 1.0),
    unit: str | None = None,
    source: Mapping[str, Any] = _STATE_SOURCE,
    source_lines: str = "67-160",
    extra_source_paths: Sequence[str] = (),
    status: str = "verified",
    notes: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build one normalized feature row with explicit provenance."""

    row: dict[str, Any] = {
        "index": int(index),
        "name": str(name),
        "group": str(group),
        "dtype": "float32",
        "normalization": str(normalization),
        "normal_range": None if normal_range is None else [float(v) for v in normal_range],
        "space_range": [0.0, 1.0],
        "zero_meaning": str(zero_meaning),
        "one_meaning": str(one_meaning),
        "status": str(status),
        "unit": unit,
    }
    row.update(_source_details(source, lines=source_lines, extra_paths=extra_source_paths))
    if notes is not None:
        row["notes"] = notes
    row.update(extra)
    return row


def source_identity() -> dict[str, Any]:
    """Return the checked-out MetaDrive identity used by the standard schema."""

    return {
        "distribution": "metadrive-simulator",
        "version": METADRIVE_VERSION,
        "commit": METADRIVE_COMMIT,
        "dirty": False,
        "source_files": dict(_SOURCE_FILE_SHA256),
    }


# The angle in MetaDrive is ``heading_theta + i * 2*pi/num_lasers``.  With
# ``heading_theta = 0`` index 0 points along +x and index 60 points along +y.
# BaseVehicle.convert_to_local_coordinates maps that +90-degree ray to local
# side +1, which NodeNetworkNavigation calls the right-hand-side component.
# The source comment says "clockwise", but the executable formula increments
# mathematical angle; these sets follow the formula and the local-side mapping.
LIDAR_GROUP_INDICES: dict[str, tuple[int, ...]] = {
    "lidar_front": tuple(range(0, 30)) + tuple(range(210, 240)),
    "lidar_left": tuple(range(150, 210)),
    "lidar_rear": tuple(range(90, 150)),
    "lidar_right": tuple(range(30, 90)),
}

LIDAR_GROUP_ANGLE_RANGES: dict[str, str] = {
    "lidar_front": "[315,360) deg union [0,45) deg",
    "lidar_left": "[225,315) deg",
    "lidar_rear": "[135,225) deg",
    "lidar_right": "[45,135) deg",
}

STANDARD_GROUPS: tuple[dict[str, Any], ...] = (
    {
        "id": "road_edges",
        "name": "road edge distances",
        "indices": (0, 1),
    },
    {
        "id": "lane_heading",
        "name": "lane heading side component",
        "indices": (2,),
    },
    {
        "id": "speed",
        "name": "speed",
        "indices": (3,),
    },
    {
        "id": "controls_history_yaw",
        "name": "steering, past actions, and yaw rate",
        "indices": (4, 5, 6, 7),
    },
    {
        "id": "lane_lateral",
        "name": "current lane lateral position",
        "indices": (8,),
    },
    {
        "id": "navigation",
        "name": "two checkpoint navigation",
        "indices": tuple(range(9, 19)),
    },
    {
        "id": "lidar_front",
        "name": "LiDAR front 90 degrees",
        "indices": tuple(LIDAR_START_INDEX + beam for beam in LIDAR_GROUP_INDICES["lidar_front"]),
    },
    {
        "id": "lidar_left",
        "name": "LiDAR left 90 degrees",
        "indices": tuple(LIDAR_START_INDEX + beam for beam in LIDAR_GROUP_INDICES["lidar_left"]),
    },
    {
        "id": "lidar_rear",
        "name": "LiDAR rear 90 degrees",
        "indices": tuple(LIDAR_START_INDEX + beam for beam in LIDAR_GROUP_INDICES["lidar_rear"]),
    },
    {
        "id": "lidar_right",
        "name": "LiDAR right 90 degrees",
        "indices": tuple(LIDAR_START_INDEX + beam for beam in LIDAR_GROUP_INDICES["lidar_right"]),
    },
)


def standard_groups() -> list[dict[str, Any]]:
    """Return the ten ordered groups used by the default fixed preset."""

    return [
        {"id": group["id"], "name": group["name"], "indices": list(group["indices"])}
        for group in STANDARD_GROUPS
    ]


def lidar_partition() -> dict[str, list[int]]:
    """Return a copy of the four disjoint 60-beam direction sets."""

    return {name: list(indices) for name, indices in LIDAR_GROUP_INDICES.items()}


def _base_features() -> list[dict[str, Any]]:
    """Build indices 0--18 from the actual MetaDrive observation code."""

    features = [
        _feature(
            0,
            "road_edge_left_distance_norm",
            "road_edges",
            "clip(dist_to_left_side / ((current_map.MAX_LANE_NUM + 1) * current_map.MAX_LANE_WIDTH), 0, 1)",
            "the labeled left edge distance is zero or below the clip boundary",
            "the labeled left edge distance reaches the map normalization width; values above it also clip to 1",
            unit="m before normalization",
            source_lines="67-101",
            extra_source_paths=("metadrive/component/vehicle/base_vehicle.py", "metadrive/component/map/base_map.py"),
            notes="The denominator is runtime map constants; BaseMap defaults are MAX_LANE_NUM=3 and MAX_LANE_WIDTH=4.5 m (18 m total).",
        ),
        _feature(
            1,
            "road_edge_right_distance_norm",
            "road_edges",
            "clip(dist_to_right_side / ((current_map.MAX_LANE_NUM + 1) * current_map.MAX_LANE_WIDTH), 0, 1)",
            "the labeled right edge distance is zero or below the clip boundary",
            "the labeled right edge distance reaches the map normalization width; values above it also clip to 1",
            unit="m before normalization",
            source_lines="67-101",
            extra_source_paths=("metadrive/component/vehicle/base_vehicle.py", "metadrive/component/map/base_map.py"),
            notes="The denominator is runtime map constants; BaseMap defaults are MAX_LANE_NUM=3 and MAX_LANE_WIDTH=4.5 m (18 m total).",
        ),
        _feature(
            2,
            "lane_heading_side_component",
            "lane_heading",
            "clip(dot(vehicle.heading, get_vertical_vector(lane.end - lane.start)[1]), -1, 1) / 2 + 0.5",
            "vehicle heading points opposite the lane right-side normal (dot=-1)",
            "vehicle heading points along the lane right-side normal (dot=+1)",
            source_lines="103-130",
            extra_source_paths=("metadrive/component/vehicle/base_vehicle.py", "metadrive/utils/math.py"),
            notes="The implementation calls this heading_diff. A vehicle heading parallel to the lane itself encodes 0.5; this follows the executable dot-product formula rather than the older comment.",
        ),
        _feature(
            3,
            "speed_km_h_norm",
            "speed",
            "clip((vehicle.speed_km_h + 1) / (vehicle.max_speed_km_h + 1), 0, 1)",
            "raw speed <= -1 km/h after clipping (a physical non-negative speed does not encode exact 0)",
            "raw speed >= vehicle.max_speed_km_h; the default vehicle max is 80 km/h",
            unit="km/h before normalization",
            source_lines="103-115",
            extra_source_paths=("metadrive/base_class/base_object.py", "metadrive/component/vehicle/base_vehicle.py"),
        ),
        _feature(
            4,
            "steering_state_norm",
            "controls_history_yaw",
            "clip((vehicle.steering / vehicle.MAX_STEERING + 1) / 2, 0, 1)",
            "vehicle.steering <= -BaseVehicle.MAX_STEERING",
            "vehicle.steering >= BaseVehicle.MAX_STEERING",
            unit="normalized steering state before encoding",
            source_lines="113-118",
            extra_source_paths=("metadrive/component/vehicle/base_vehicle.py",),
            notes="The source divides by the class constant MAX_STEERING=60, while _set_action applies the runtime max_steering multiplier. This row records the source formula verbatim.",
        ),
        _feature(
            5,
            "last_action_steering_norm",
            "controls_history_yaw",
            "clip((vehicle.last_current_action[1][0] + 1) / 2, 0, 1)",
            "latest stored steering action is -1 or below",
            "latest stored steering action is +1 or above",
            source_lines="119-122",
            extra_source_paths=("metadrive/component/vehicle/base_vehicle.py",),
            notes="The [1] deque entry is the latest clipped action pair. The code order is steering then throttle/brake.",
        ),
        _feature(
            6,
            "last_action_throttle_brake_norm",
            "controls_history_yaw",
            "clip((vehicle.last_current_action[1][1] + 1) / 2, 0, 1)",
            "latest stored throttle/brake action is -1 or below",
            "latest stored throttle/brake action is +1 or above",
            source_lines="119-122",
            extra_source_paths=("metadrive/component/vehicle/base_vehicle.py",),
            notes="The [1] deque entry is the latest clipped action pair. A negative value is braking unless reverse is enabled.",
        ),
        _feature(
            7,
            "yaw_rate_norm",
            "controls_history_yaw",
            "clip(arccos(clip(dot(heading_now, heading_last) / (norm(now) * norm(last)), 0, 1)) / 0.1, 0, 1)",
            "no measured heading change (encoded yaw rate is 0)",
            "the clipped heading change is at least 0.1 rad per 0.1 s control interval",
            unit="rad/s before clip",
            source_lines="124-130",
            extra_source_paths=("metadrive/utils/math.py", "metadrive/component/vehicle/base_vehicle.py"),
            notes="The source clips cosine to [0,1], removes turn sign, and uses a fixed 0.1 s denominator.",
        ),
        _feature(
            8,
            "lane_lateral_position_norm",
            "lane_lateral",
            "clip((lateral * 2 / navigation.map.MAX_LANE_WIDTH + 1) / 2, 0, 1)",
            "lateral <= -MAX_LANE_WIDTH / 2 after clipping",
            "lateral >= +MAX_LANE_WIDTH / 2 after clipping",
            unit="m before normalization",
            source_lines="145-153",
            extra_source_paths=("metadrive/component/lane/straight_lane.py",),
            notes="The lane local-coordinate lateral sign follows the active lane implementation. Center is encoded 0.5.",
        ),
    ]

    checkpoint_names = ("next_checkpoint", "next_next_checkpoint")
    for checkpoint_number, checkpoint_name in enumerate(checkpoint_names):
        offset = 9 + checkpoint_number * 5
        features.extend(
            [
                _feature(
                    offset,
                    f"navigation_{checkpoint_name}_forward_norm",
                    "navigation",
                    "clip((checkpoint_in_heading / 50 + 1) / 2, 0, 1)",
                    "checkpoint projection is at or behind -50 m in the vehicle heading coordinate",
                    "checkpoint projection is at or beyond +50 m in the vehicle heading coordinate",
                    unit="m before normalization",
                    source=_NAV_SOURCE,
                    source_lines="288-317",
                    extra_source_paths=("metadrive/component/vehicle/base_vehicle.py",),
                ),
                _feature(
                    offset + 1,
                    f"navigation_{checkpoint_name}_right_norm",
                    "navigation",
                    "clip((checkpoint_in_rhs / 50 + 1) / 2, 0, 1)",
                    "checkpoint projection is at or beyond -50 m in the local right-side coordinate",
                    "checkpoint projection is at or beyond +50 m in the local right-side coordinate",
                    unit="m before normalization",
                    source=_NAV_SOURCE,
                    source_lines="288-317",
                    extra_source_paths=("metadrive/component/vehicle/base_vehicle.py",),
                ),
                _feature(
                    offset + 2,
                    f"navigation_{checkpoint_name}_curve_radius_norm",
                    "navigation",
                    "clip(ref_lane.radius / (curve_radius_max + lane_count * lane_width), 0, 1) for CircularLane; 0 otherwise",
                    "straight lane (the source initializes bendradius to 0)",
                    "the normalized radius reaches the configured curve denominator",
                    unit="dimensionless after normalization",
                    source=_NAV_SOURCE,
                    source_lines="319-337",
                    extra_source_paths=("metadrive/component/pg_space.py",),
                ),
                _feature(
                    offset + 3,
                    f"navigation_{checkpoint_name}_curve_direction_norm",
                    "navigation",
                    "clip((dir + 1) / 2, 0, 1), where dir=0 for straight, +1 clockwise, -1 anticlockwise",
                    "anticlockwise curve (dir=-1)",
                    "clockwise curve (dir=+1)",
                    source=_NAV_SOURCE,
                    source_lines="319-340",
                    notes="For a straight lane dir is initialized to 0, so the executable formula returns 0.5; the old comment saying 0 is inaccurate.",
                ),
                _feature(
                    offset + 4,
                    f"navigation_{checkpoint_name}_curve_angle_norm",
                    "navigation",
                    "clip((degrees(ref_lane.angle) / 135 + 1) / 2, 0, 1), angle=0 for straight",
                    "source angle is 0 degrees (straight lane encodes 0.5)",
                    "source angle reaches 135 degrees or more",
                    unit="degrees before normalization",
                    source=_NAV_SOURCE,
                    source_lines="319-345",
                    notes="For a straight lane angle is initialized to 0, so the executable formula returns 0.5.",
                ),
            ]
        )
    return features


def _lidar_features() -> list[dict[str, Any]]:
    """Generate one explicit row for each of the 240 configured beams."""

    group_by_index = {
        index: group_name
        for group_name, indices in LIDAR_GROUP_INDICES.items()
        for index in indices
    }
    rows: list[dict[str, Any]] = []
    for beam_index in range(LIDAR_COUNT):
        group = group_by_index[beam_index]
        angle = beam_index * LIDAR_ANGLE_STEP_DEG
        rows.append(
            _feature(
                LIDAR_START_INDEX + beam_index,
                f"lidar_beam_{beam_index:03d}",
                group,
                "cloud_points[i] = rayTestClosest(...).getHitFraction(); no hit remains 1.0",
                "a ray hit at the sensor origin (zero clearance)",
                "no hit within the configured 50 m distance, or a hit at the ray endpoint",
                unit="hit fraction of configured distance",
                source=_LIDAR_SOURCE,
                source_lines="27-85,117-179",
                extra_source_paths=("metadrive/utils/math.py", "metadrive/obs/state_obs.py", "metadrive/component/sensors/lidar.py"),
                relative_angle_deg=angle,
                angle_step_deg=LIDAR_ANGLE_STEP_DEG,
                direction_angle_range=LIDAR_GROUP_ANGLE_RANGES[group],
                beam_index=beam_index,
                lidar_distance_m=50.0,
                notes=(
                    "World ray angle is vehicle.heading_theta + beam_index * 1.5 degrees. "
                    "The 50 m value is the official BaseEnv default; verify the runtime config if changed."
                ),
            )
        )
    return rows


def standard_259_schema() -> list[dict[str, Any]]:
    """Return the complete source-backed 259-row MetaDrive schema."""

    rows = _base_features() + _lidar_features()
    return validate_schema(rows, dimension=STANDARD_DIMENSION, require_verified=True)


def _unverified_host_feature(index: int) -> dict[str, Any]:
    """Build one blank row for the host supplied 262-dimensional contract.

    Every position is intentionally blank.  In particular, this helper does
    not call :func:`standard_259_schema`: a dimension of 262 gives no evidence
    that the first 259 positions use MetaDrive's order, nor that the extra
    positions are trailing fields.
    """

    return _feature(
        index,
        f"host_input_{index:03d}_unverified",
        "host_input_unverified",
        "UNVERIFIED: obtain the target host's preprocessing and normalization formula",
        "UNVERIFIED: confirm the target host's encoded zero meaning",
        "UNVERIFIED: confirm the target host's encoded one meaning",
        normal_range=None,
        source={
            "source": "UNVERIFIED: target host observation source required",
            "source_version": None,
            "source_commit": None,
            "source_path": "<target-host-source-required>",
            "source_import": None,
            "source_dirty": None,
        },
        source_lines="",
        status="unverified",
        notes=(
            "Verify this position's location/order, meaning, normalization, "
            "reference lane, and valid flag in the target host. Do not infer "
            "it from D=262, append it to a 259-vector, pad, or truncate."
        ),
        host_verification_fields=(
            "position",
            "meaning",
            "normalization",
            "reference_lane",
            "valid_flag",
        ),
    )


def template_262_schema() -> list[dict[str, Any]]:
    """Return a fully unverified 262-position host migration template.

    This is a checklist-shaped schema, not a claim about any real 262-wide
    environment.  All 262 rows are unassigned until the porting host supplies
    position, meaning, normalization, reference-lane, and validity evidence.
    """

    rows = [_unverified_host_feature(index) for index in range(TEMPLATE_DIMENSION)]
    return validate_schema(rows, dimension=TEMPLATE_DIMENSION, require_verified=False)


def _synthetic_schema(dimension: int) -> list[dict[str, Any]]:
    """Build an artificial schema for a dependency-light synthetic width."""

    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise SchemaError("synthetic schema dimension must be a positive integer")
    rows: list[dict[str, Any]] = []
    for index in range(dimension):
        rows.append(
            _feature(
                index,
                f"synthetic_feature_{index:03d}",
                "synthetic_features",
                "SyntheticEnv._observation(index): clip(0.5 + deterministic test waveform, 0.02, 0.98)",
                "synthetic test value at or below the declared [0, 1] lower bound",
                "synthetic test value at or above the declared [0, 1] upper bound",
                normal_range=(0.0, 1.0),
                source=_SYNTHETIC_SOURCE,
                source_lines="145-153",
                status="verified",
                notes=(
                    "Artificial test feature only. It carries no real MetaDrive "
                    "meaning and must not be used to infer a host schema."
                ),
                synthetic_index=index,
            )
        )
    return validate_schema(rows, dimension=dimension, require_verified=True)


def synthetic_259_schema() -> list[dict[str, Any]]:
    """Return an artificial 259-wide schema for synthetic smoke tests."""

    return _synthetic_schema(STANDARD_DIMENSION)


def synthetic_262_schema() -> list[dict[str, Any]]:
    """Return an explicit, artificial 262-wide schema for T13 tests.

    These rows describe only the deterministic synthetic adapter's contract.
    They intentionally make no claim that a real MetaDrive or ported host has
    the same positions, names, ranges, or ordering.
    """

    return _synthetic_schema(TEMPLATE_DIMENSION)


def synthetic_262_schema_document() -> dict[str, Any]:
    """Return a manifest document for the artificial 262-wide test schema."""

    features = synthetic_262_schema()
    return {
        "schema_version": SCHEMA_VERSION,
        "schema_id": "synthetic_262_contract",
        "dimension": TEMPLATE_DIMENSION,
        "dtype": "float32",
        "verified": True,
        "real_host_verified": False,
        "source_identity": {
            "kind": "synthetic_test_only",
            "source": "input_attribution.synthetic.SyntheticEnv",
            "schema_version": SCHEMA_VERSION,
        },
        "groups": [
            {
                "id": "synthetic_features",
                "name": "artificial synthetic test features",
                "indices": list(range(TEMPLATE_DIMENSION)),
            }
        ],
        "features": features,
    }


def synthetic_259_schema_document() -> dict[str, Any]:
    """Return a manifest document for the artificial 259-wide test schema."""

    features = synthetic_259_schema()
    return {
        "schema_version": SCHEMA_VERSION,
        "schema_id": "synthetic_259_contract",
        "dimension": STANDARD_DIMENSION,
        "dtype": "float32",
        "verified": True,
        "real_host_verified": False,
        "source_identity": {
            "kind": "synthetic_test_only",
            "source": "input_attribution.synthetic.SyntheticEnv",
            "schema_version": SCHEMA_VERSION,
        },
        "groups": [
            {
                "id": "synthetic_features",
                "name": "artificial synthetic test features",
                "indices": list(range(STANDARD_DIMENSION)),
            }
        ],
        "features": features,
    }


def schema_document(dimension: int = STANDARD_DIMENSION) -> dict[str, Any]:
    """Return a serializable schema document for manifests and schema CSVs."""

    if dimension == STANDARD_DIMENSION:
        features = standard_259_schema()
        schema_id = "metadrive_state_lidar_259"
        verified = True
    elif dimension == TEMPLATE_DIMENSION:
        features = template_262_schema()
        schema_id = "host_262_migration_template"
        verified = False
    else:
        raise SchemaError(
            f"unsupported generated schema dimension {dimension}; provide explicit features for a custom host"
        )
    provenance: dict[str, Any]
    if verified:
        provenance = source_identity()
    else:
        provenance = {
            "kind": "unverified_host_template",
            "real_host_verified": False,
            "note": "No MetaDrive row order is inherited; target host provenance is required.",
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "schema_id": schema_id,
        "dimension": dimension,
        "dtype": "float32",
        "verified": verified,
        "real_host_verified": verified,
        "source_identity": provenance,
        "groups": standard_groups()
        if dimension == STANDARD_DIMENSION
        else [
            {
                "id": "host_input_unverified",
                "name": "target host positions requiring verification",
                "indices": list(range(TEMPLATE_DIMENSION)),
            }
        ],
        "features": features,
    }


def validate_schema(
    features: Sequence[Mapping[str, Any]],
    *,
    dimension: int | None = None,
    require_verified: bool = False,
) -> list[dict[str, Any]]:
    """Validate and copy a complete per-index schema.

    A dimension by itself is never enough to identify a schema.  Callers that
    provide custom rows must provide all metadata fields for every index,
    including an explicit ``status`` and provenance.  ``unverified`` rows may
    omit a normal range and source version only when ``require_verified`` is
    false; this is how the 262 migration template makes unknown fields visible.
    """

    if isinstance(features, (str, bytes)) or not isinstance(features, Sequence):
        raise SchemaError("schema features must be a sequence of per-index mappings")
    if dimension is None:
        dimension = len(features)
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise SchemaError("schema dimension must be a positive integer")
    if len(features) != dimension:
        raise SchemaError(f"schema dimension is {dimension}, but {len(features)} rows were provided")

    required = (
        "index",
        "name",
        "group",
        "dtype",
        "normalization",
        "normal_range",
        "zero_meaning",
        "one_meaning",
        "source",
        "source_version",
        "source_path",
        "source_lines",
        "status",
    )
    copied: list[dict[str, Any]] = []
    seen: set[int] = set()
    for position, original in enumerate(features):
        if not isinstance(original, Mapping):
            raise SchemaError(f"schema row {position} must be a mapping")
        missing = [key for key in required if key not in original]
        if missing:
            raise SchemaError(f"schema row {position} is missing metadata: {', '.join(missing)}")
        row = copy.deepcopy(dict(original))
        index = row["index"]
        if isinstance(index, bool) or not isinstance(index, int):
            raise SchemaError(f"schema row {position} index must be an integer")
        if index != position:
            raise SchemaError(f"schema row {position} has index {index}; expected {position}")
        if index in seen:
            raise SchemaError(f"schema has duplicate index {index}")
        seen.add(index)
        for key in ("name", "group", "normalization", "zero_meaning", "one_meaning", "status"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise SchemaError(f"schema row {index} field {key} must be a non-empty string")
        if row["dtype"] != "float32":
            raise SchemaError(f"schema row {index} dtype must be float32")
        status = row["status"]
        if status not in {"verified", "unverified"}:
            raise SchemaError(f"schema row {index} status must be verified or unverified")
        normal_range = row["normal_range"]
        if normal_range is None:
            if status == "verified" or require_verified:
                raise SchemaError(f"schema row {index} lacks a verified normal_range")
        else:
            if isinstance(normal_range, (str, bytes)) or not isinstance(normal_range, Sequence) or len(normal_range) != 2:
                raise SchemaError(f"schema row {index} normal_range must be [low, high] or null")
            try:
                low, high = float(normal_range[0]), float(normal_range[1])
            except (TypeError, ValueError) as error:
                raise SchemaError(f"schema row {index} normal_range must be numeric") from error
            if not math.isfinite(low) or not math.isfinite(high) or low > high:
                raise SchemaError(f"schema row {index} normal_range is not finite and ordered")
            row["normal_range"] = [low, high]
        if row["source_version"] is None:
            if status == "verified" or require_verified:
                raise SchemaError(f"schema row {index} lacks a verified source_version")
        elif not isinstance(row["source_version"], str) or not row["source_version"].strip():
            raise SchemaError(f"schema row {index} source_version must be a string or null")
        if not isinstance(row["source"], str) or not row["source"].strip():
            raise SchemaError(f"schema row {index} source must be a non-empty string")
        if not isinstance(row["source_path"], str) or not row["source_path"].strip():
            raise SchemaError(f"schema row {index} source_path must be a non-empty string")
        if not isinstance(row["source_lines"], str):
            raise SchemaError(f"schema row {index} source_lines must be a string")
        if "space_range" in row:
            space_range = row["space_range"]
            if not isinstance(space_range, Sequence) or len(space_range) != 2:
                raise SchemaError(f"schema row {index} space_range must have two values")
        copied.append(row)
    if seen != set(range(dimension)):
        raise SchemaError("schema indices must cover every index from zero to dimension-1")
    return copied


def _schema_alias(value: str) -> int | None:
    aliases = {
        "standard259": STANDARD_DIMENSION,
        "standard_259": STANDARD_DIMENSION,
        "metadrive_259": STANDARD_DIMENSION,
        "metadrive_state_lidar_259": STANDARD_DIMENSION,
        "template262": TEMPLATE_DIMENSION,
        "template_262": TEMPLATE_DIMENSION,
        # Kept as a migration-template compatibility alias.  It resolves to
        # the all-unknown host template; the artificial T13 schema has the
        # explicit ``synthetic_262_contract`` name below.
        "synthetic_262_template": TEMPLATE_DIMENSION,
        "host_262_migration_template": TEMPLATE_DIMENSION,
    }
    return aliases.get(value.lower())


_SYNTHETIC_SCHEMA_DIMENSIONS = {
    "synthetic259": STANDARD_DIMENSION,
    "synthetic_259": STANDARD_DIMENSION,
    "synthetic_259_contract": STANDARD_DIMENSION,
    "synthetic_259_schema": STANDARD_DIMENSION,
    "synthetic262": TEMPLATE_DIMENSION,
    "synthetic_262": TEMPLATE_DIMENSION,
    "synthetic_262_contract": TEMPLATE_DIMENSION,
    "synthetic_262_schema": TEMPLATE_DIMENSION,
}


def _resolve_schema_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_file():
        return path.resolve()
    candidates = [
        Path.cwd() / path,
        Path(__file__).resolve().parent / path,
        Path(__file__).resolve().parent / "configs" / path,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise SchemaError(f"schema file does not exist: {value}")


def load_schema(
    source: str | Path | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    *,
    dimension: int | None = None,
    require_verified: bool = False,
) -> list[dict[str, Any]]:
    """Load a generated, file-backed, or explicitly listed schema.

    ``source=None`` selects the source-backed 259 schema for convenience.
    A mapping containing only ``dimension`` is rejected: callers must name a
    generator or provide all rows, preventing accidental 259/262 conflation.
    """

    if source is None:
        source = "standard_259"
    if isinstance(source, (str, Path)):
        if isinstance(source, str):
            normalized_source = source.lower()
            if normalized_source in _SYNTHETIC_SCHEMA_DIMENSIONS:
                source = {"generator": normalized_source}
            else:
                generated_dimension = _schema_alias(source)
                if generated_dimension is not None:
                    source = generated_dimension
                else:
                    source = _resolve_schema_path(source)
        if isinstance(source, Path):
            payload = tomllib.loads(source.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise SchemaError(f"schema TOML must contain a table: {source}")
            generator = payload.get("generator") or payload.get("schema_id")
            if generator is None:
                if "features" in payload:
                    source = payload
                else:
                    raise SchemaError(
                        f"schema file {source} needs generator/schema_id or explicit features; dimension alone is insufficient"
                    )
            else:
                source = {"generator": generator, "features": payload.get("features")}
    if isinstance(source, int) and not isinstance(source, bool):
        if source == STANDARD_DIMENSION:
            rows = standard_259_schema()
        elif source == TEMPLATE_DIMENSION:
            rows = template_262_schema()
        else:
            raise SchemaError(f"no generated schema is registered for dimension {source}")
    elif isinstance(source, Mapping):
        generator = source.get("generator") or source.get("schema_id")
        if generator is not None:
            if not isinstance(generator, str):
                raise SchemaError("schema generator/schema_id must be a string")
            normalized_generator = generator.lower()
            generated_dimension = _SYNTHETIC_SCHEMA_DIMENSIONS.get(normalized_generator)
            if generated_dimension is not None:
                rows = (
                    synthetic_259_schema()
                    if generated_dimension == STANDARD_DIMENSION
                    else synthetic_262_schema()
                )
            else:
                generated_dimension = _schema_alias(generator)
                if generated_dimension is None:
                    raise SchemaError(f"unknown schema generator {generator!r}")
                rows = (
                    standard_259_schema()
                    if generated_dimension == STANDARD_DIMENSION
                    else template_262_schema()
                )
            explicit_features = source.get("features")
            if explicit_features not in (None, []):
                rows = validate_schema(explicit_features, dimension=generated_dimension)
        else:
            if "features" not in source:
                raise SchemaError("explicit schema mapping needs features; dimension alone is insufficient")
            explicit_features = source["features"]
            if not isinstance(explicit_features, Sequence):
                raise SchemaError("schema features must be a sequence")
            rows = validate_schema(explicit_features, dimension=source.get("dimension"))
    elif isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        rows = validate_schema(source, dimension=dimension)
    else:
        raise SchemaError(f"unsupported schema source type: {type(source).__name__}")
    if dimension is not None and len(rows) != dimension:
        raise SchemaError(f"loaded schema has dimension {len(rows)}, expected {dimension}")
    return validate_schema(rows, dimension=len(rows), require_verified=require_verified)


def _coerce_fixed_value(value: Any) -> float:
    if isinstance(value, bool):
        raise SchemaError("fixed_value must be a finite number")
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise SchemaError("fixed_value must be a finite number") from error
    if not math.isfinite(converted):
        raise SchemaError("fixed_value must be finite; NaN and Inf are not accepted")
    return converted


def default_patterns(
    dimension: int = STANDARD_DIMENSION,
    *,
    fixed_value: float = -1.0,
) -> list[dict[str, Any]]:
    """Return the ordered initial groups valid for ``dimension``.

    The normal 259-dimensional preset returns all ten verified groups.  A
    smaller synthetic vector keeps only groups whose indices fit.  A wider
    dimension has no fallback groups: callers must provide an explicit schema
    and explicit patterns, so a 262-wide host is never treated as 259 plus
    trailing fields.
    """

    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise SchemaError("pattern dimension must be a positive integer")
    if dimension >= STANDARD_DIMENSION:
        if dimension != STANDARD_DIMENSION:
            raise SchemaError(
                "no default patterns are defined for dimensions wider than "
                "the verified 259 schema; provide explicit host/synthetic patterns"
            )
    value = _coerce_fixed_value(fixed_value)
    patterns: list[dict[str, Any]] = []
    for number, group in enumerate(STANDARD_GROUPS, start=1):
        indices = list(group["indices"])
        if indices and max(indices) < dimension:
            patterns.append(
                {
                    "id": f"P{number:02d}",
                    "name": str(group["id"]),
                    "indices": indices,
                    "fixed_value": value,
                }
            )
    return patterns


def validate_patterns(
    patterns: Iterable[Mapping[str, Any]],
    *,
    dimension: int = STANDARD_DIMENSION,
) -> list[dict[str, Any]]:
    """Validate intervention patterns while allowing finite out-of-range values."""

    if isinstance(patterns, (str, bytes)):
        raise SchemaError("patterns must be a sequence of tables")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise SchemaError("pattern dimension must be a positive integer")
    result: list[dict[str, Any]] = []
    ids: set[str] = set()
    for position, original in enumerate(patterns):
        if not isinstance(original, Mapping):
            raise SchemaError(f"pattern {position} must be a mapping")
        for key in ("id", "name", "indices", "fixed_value"):
            if key not in original:
                raise SchemaError(f"pattern {position} is missing {key}")
        pattern_id = original["id"]
        name = original["name"]
        if not isinstance(pattern_id, str) or not pattern_id.strip():
            raise SchemaError(f"pattern {position} id must be a non-empty string")
        if pattern_id in ids:
            raise SchemaError(f"duplicate pattern id {pattern_id!r}")
        ids.add(pattern_id)
        if not isinstance(name, str) or not name.strip():
            raise SchemaError(f"pattern {pattern_id} name must be a non-empty string")
        indices = original["indices"]
        if isinstance(indices, (str, bytes)) or not isinstance(indices, Sequence) or not indices:
            raise SchemaError(f"pattern {pattern_id} indices must be a non-empty list")
        checked_indices: list[int] = []
        seen: set[int] = set()
        for index in indices:
            if isinstance(index, bool) or not isinstance(index, int):
                raise SchemaError(f"pattern {pattern_id} indices must contain integers")
            if index < 0 or index >= dimension:
                raise SchemaError(f"pattern {pattern_id} index {index} is outside 0..{dimension - 1}")
            if index in seen:
                raise SchemaError(f"pattern {pattern_id} contains duplicate index {index}")
            seen.add(index)
            checked_indices.append(index)
        result.append(
            {
                "id": pattern_id,
                "name": name,
                "indices": checked_indices,
                "fixed_value": _coerce_fixed_value(original["fixed_value"]),
            }
        )
    if not result:
        raise SchemaError("at least one intervention pattern is required")
    return result


def load_patterns(
    source: Iterable[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    *,
    dimension: int = STANDARD_DIMENSION,
    fixed_value: float = -1.0,
) -> list[dict[str, Any]]:
    """Load explicit patterns, or return the ten ordered default groups."""

    if source is None:
        return validate_patterns(default_patterns(fixed_value=fixed_value), dimension=dimension)
    if isinstance(source, Mapping):
        source = source.get("patterns")
    if source is None:
        return validate_patterns(default_patterns(fixed_value=fixed_value), dimension=dimension)
    return validate_patterns(source, dimension=dimension)


def write_schema_csv(path: str | Path, features: Sequence[Mapping[str, Any]] | None = None) -> Path:
    """Write the detailed per-index mapping used by an experiment manifest."""

    destination = Path(path)
    rows = load_schema(features) if features is not None else standard_259_schema()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "index",
        "name",
        "group",
        "dtype",
        "normalization",
        "normal_range",
        "space_range",
        "zero_meaning",
        "one_meaning",
        "unit",
        "status",
        "source",
        "source_version",
        "source_commit",
        "source_path",
        "source_lines",
        "source_import",
        "relative_angle_deg",
        "direction_angle_range",
        "beam_index",
        "lidar_distance_m",
        "notes",
    )
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            encoded = dict(row)
            for key in ("normal_range", "space_range"):
                value = encoded.get(key)
                if value is not None:
                    encoded[key] = "[" + ",".join(str(item) for item in value) + "]"
            writer.writerow(encoded)
    return destination


# Small aliases make the intended contract easy to discover from a copied
# package without adding a second schema implementation.
load_input_schema = load_schema
build_standard_schema = standard_259_schema
build_262_template = template_262_schema
parse_patterns = load_patterns


__all__ = [
    "LIDAR_ANGLE_STEP_DEG",
    "LIDAR_COUNT",
    "LIDAR_END_INDEX",
    "LIDAR_GROUP_ANGLE_RANGES",
    "LIDAR_GROUP_INDICES",
    "LIDAR_START_INDEX",
    "METADRIVE_COMMIT",
    "METADRIVE_VERSION",
    "SCHEMA_VERSION",
    "STANDARD_DIMENSION",
    "TEMPLATE_DIMENSION",
    "SchemaError",
    "build_262_template",
    "build_standard_schema",
    "default_patterns",
    "lidar_partition",
    "load_input_schema",
    "load_patterns",
    "load_schema",
    "parse_patterns",
    "schema_document",
    "source_identity",
    "standard_259_schema",
    "standard_groups",
    "synthetic_259_schema",
    "synthetic_259_schema_document",
    "synthetic_262_schema",
    "synthetic_262_schema_document",
    "template_262_schema",
    "validate_patterns",
    "validate_schema",
    "write_schema_csv",
]
