"""Small, self-contained fixtures for adapter and schema contract tests.

The fixtures are deliberately independent of MetaDrive.  They describe the
same flat input contract as the official vector observation and provide a
262-element unresolved template whose three extra fields are *not* assumed to
be tail fields.  A real 262-dimensional schema must replace those placeholders
with source-verified entries before an experiment is accepted.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Sequence

from ..schema import InputSchema, InputSpec, SchemaError


_OFFICIAL_DIM = 259
_CUSTOM_DIM = 262

# Byte hashes are part of the provenance contract, rather than a package
# version claim.  The adapter compares the installed source bytes before a
# 259-dimensional run so a same-version-but-edited checkout cannot silently
# reuse this index mapping.
_OFFICIAL_SOURCE_SHA256 = {
    "metadrive/obs/state_obs.py": "2bf137e9f19388faa12069b93537b4d4f69cd1eca3d123c014637c8954c05c82",
    "metadrive/component/vehicle/base_vehicle.py": "1660147eddb47faa8c2496bbef37ce369f6f82406a563cd34db7d9d9e4256f5a",
    "metadrive/component/navigation_module/node_network_navigation.py": "a423fdc5238d95aa4d1e01c6d614eff231d9e40fc24d40e7828c6874da05388f",
    "metadrive/component/sensors/distance_detector.py": "ab12f0b9fd9ed9e5262d42c99294e007e20b361499767eee6dc527fb4eacfdf0",
    "metadrive/component/sensors/lidar.py": "a48e02d5ab49053b005fa4829db343af42e6e637f6c87e7ea84ce1cc7ef9683c",
    "metadrive/policy/env_input_policy.py": "1bf54b96e14d75d67fc180344541b2e221746ab120cc308f63dda74735e74e6a",
}

_OFFICIAL_SOURCE_RELATIVE_PATHS = {
    key: f"../metadrive/{key}" for key in _OFFICIAL_SOURCE_SHA256
}


def _spec(
    index: int,
    identifier: str,
    name_ja: str,
    *,
    group: str,
    description: str,
    physical_quantity: str,
    unit: str,
    model_representation: str,
    normalization: str,
    source: str,
    replacement: dict[str, object] | None,
    reference: str = "",
    value_type: str = "continuous",
    value_range: tuple[float | None, float | None] | None = (0.0, 1.0),
    related_valid_flags: tuple[int, ...] = (),
    coupled_indices: tuple[int, ...] = (),
    independently_replaceable: bool = True,
    ig_interpolation_allowed: bool = True,
) -> InputSpec:
    return InputSpec(
        index=index,
        id=identifier,
        name_ja=name_ja,
        description=description,
        group=group,
        physical_quantity=physical_quantity,
        unit=unit,
        model_representation=model_representation,
        value_range=value_range,
        normalization=normalization,
        clip_method="clip to [0, 1]",
        value_type=value_type,
        reference=reference,
        replacement=replacement,
        related_valid_flags=related_valid_flags,
        coupled_indices=coupled_indices,
        independently_replaceable=independently_replaceable,
        ig_interpolation_allowed=ig_interpolation_allowed,
        source=source,
    )


def _reference_replacement(method: str = "saved_observation") -> dict[str, object]:
    return {"kind": "reference", "value_method": method}


def _official_specs() -> list[InputSpec]:
    """Build the source-verified 0..258 MetaDrive 0.4.3 mapping."""

    source_state = "../metadrive/metadrive/obs/state_obs.py:67-154"
    source_navi = (
        "../metadrive/metadrive/component/navigation_module/"
        "node_network_navigation.py:288-347"
    )
    source_lidar = (
        "../metadrive/metadrive/obs/state_obs.py:165-247; "
        "../metadrive/metadrive/component/sensors/distance_detector.py:177-180; "
        "../metadrive/metadrive/component/sensors/lidar.py"
    )
    source_action = "../metadrive/metadrive/policy/env_input_policy.py:39-83"
    specs: list[InputSpec] = [
        _spec(
            0,
            "distance_to_left_road_boundary",
            "左道路端までの距離",
            group="road_boundaries",
            description="vehicle.dist_to_left_side / ((MAX_LANE_NUM+1)*MAX_LANE_WIDTH)",
            physical_quantity="distance to left road boundary",
            unit="m",
            model_representation="normalized scalar",
            normalization="divide by current_map (MAX_LANE_NUM+1)*MAX_LANE_WIDTH",
            source=source_state,
            replacement=_reference_replacement(),
            reference="vehicle.dist_to_left_side and current_map road-boundary width",
        ),
        _spec(
            1,
            "distance_to_right_road_boundary",
            "右道路端までの距離",
            group="road_boundaries",
            description="vehicle.dist_to_right_side / ((MAX_LANE_NUM+1)*MAX_LANE_WIDTH)",
            physical_quantity="distance to right road boundary",
            unit="m",
            model_representation="normalized scalar",
            normalization="divide by current_map (MAX_LANE_NUM+1)*MAX_LANE_WIDTH",
            source=source_state,
            replacement=_reference_replacement(),
            reference="vehicle.dist_to_right_side and current_map road-boundary width",
        ),
        _spec(
            2,
            "heading_diff_to_reference_lane_lateral",
            "参照レーン横方向との向き成分",
            group="heading",
            description=(
                "BaseVehicle.heading_diff(vehicle, current_ref_lanes[-1]) uses the "
                "lane lateral/right vector in its dot product; it is a cosine "
                "component mapped to [0,1], not a signed lane-tangent angle."
            ),
            physical_quantity="heading versus lane lateral-direction cosine component",
            unit="dimensionless",
            model_representation="(clip(cos,-1,1)/2)+0.5",
            normalization="clip to [0,1] after BaseVehicle.heading_diff",
            source="../metadrive/metadrive/obs/state_obs.py:103-112; ../metadrive/metadrive/component/vehicle/base_vehicle.py:565-589",
            replacement=_reference_replacement(),
            reference="vehicle.navigation.current_ref_lanes[-1] lateral/right vector",
        ),
        _spec(
            3,
            "normalized_speed",
            "正規化速度",
            group="speed",
            description="(vehicle.speed_km_h + 1) / (vehicle.max_speed_km_h + 1)",
            physical_quantity="vehicle speed",
            unit="km/h",
            model_representation="normalized scalar",
            normalization="(speed_km_h+1)/(max_speed_km_h+1)",
            source=source_state,
            replacement=_reference_replacement(),
            reference="vehicle.speed_km_h and vehicle.max_speed_km_h",
        ),
        _spec(
            4,
            "current_steering",
            "現在操舵",
            group="steering_and_action_history",
            description="Current vehicle steering command normalized by MAX_STEERING.",
            physical_quantity="steering command",
            unit="normalized command",
            model_representation="(steering/MAX_STEERING+1)/2",
            normalization="clip to [0,1]",
            source=source_state,
            replacement=_reference_replacement(),
            reference="vehicle.steering and vehicle.MAX_STEERING",
        ),
        _spec(
            5,
            "previous_steering_action",
            "直前操舵Action",
            group="steering_and_action_history",
            description="vehicle.last_current_action[1][0], the preceding steering component.",
            physical_quantity="previous steering command",
            unit="normalized command",
            model_representation="(last_action_steering+1)/2",
            normalization="clip to [0,1]",
            source=f"{source_state}; {source_action}",
            replacement=_reference_replacement(),
            reference="vehicle.last_current_action[1][0]",
        ),
        _spec(
            6,
            "previous_throttle_brake_action",
            "直前加減速Action",
            group="steering_and_action_history",
            description="vehicle.last_current_action[1][1], the preceding throttle/brake component.",
            physical_quantity="previous throttle/brake command",
            unit="normalized command",
            model_representation="(last_action_throttle_brake+1)/2",
            normalization="clip to [0,1]",
            source=f"{source_state}; {source_action}",
            replacement=_reference_replacement(),
            reference="vehicle.last_current_action[1][1]",
        ),
        _spec(
            7,
            "unsigned_yaw_rate",
            "符号なしヨーレート",
            group="yaw_rate",
            description="arccos(clip(dot(last_heading, heading),0,1))/0.1, then clipped.",
            physical_quantity="yaw-rate estimate",
            unit="rad/s (fixed 0.1 s denominator)",
            model_representation="clip(arccos(clip(cos_beta,0,1))/0.1,0,1)",
            normalization="clip to [0,1]",
            source=source_state,
            replacement=_reference_replacement(),
            reference="vehicle.last_heading_dir and vehicle.heading over the fixed 0.1 s denominator",
        ),
        _spec(
            8,
            "lateral_position_in_current_lane",
            "現在レーン中心からの横位置",
            group="current_lane_lateral_position",
            description="Current lane local lateral coordinate mapped by map.MAX_LANE_WIDTH (or 10 fallback).",
            physical_quantity="lateral offset in current lane",
            unit="m",
            model_representation="(lateral*2/max_lane_width+1)/2",
            normalization="clip to [0,1]",
            source=source_state,
            replacement=_reference_replacement(),
            reference="vehicle.lane.local_coordinates(vehicle.position); current vehicle lane",
        ),
    ]

    navi_names = (
        ("forward_projection", "checkpoint forward方向投影", "checkpoint forward projection"),
        ("right_projection", "checkpoint右方向投影", "checkpoint right-side projection"),
        ("curve_radius", "曲率半径", "normalized bend radius"),
        ("curve_direction", "曲率方向", "clockwise/counterclockwise encoding"),
        ("curve_angle", "曲率角", "normalized curve angle"),
    )
    for checkpoint in (1, 2):
        for offset, (suffix, name_ja, quantity) in enumerate(navi_names):
            index = 9 + (checkpoint - 1) * 5 + offset
            is_direction = suffix == "curve_direction"
            is_curvature_member = offset >= 2
            curvature_group = tuple(range(9 + (checkpoint - 1) * 5 + 2, 9 + (checkpoint - 1) * 5 + 5))
            if suffix in {"curve_direction", "curve_angle"}:
                if is_direction:
                    description = (
                        "Categorical lane-bend direction encoding: straight=0.5, "
                        "clockwise/counterclockwise are the endpoint values; "
                        "the source uses (dir + 1) / 2."
                    )
                else:
                    description = (
                        "Lane angular change normalized by the configured curve-angle bound; "
                        "straight lane is encoded as 0.5."
                    )
            elif suffix == "forward_projection":
                description = "Checkpoint vector projected onto vehicle heading, clipped to 50 m."
            elif suffix == "right_projection":
                description = "Checkpoint vector projected onto vehicle right-hand side, clipped to 50 m."
            else:
                description = "CircularLane radius divided by configured curve bound plus lane-width term."
            specs.append(
                _spec(
                    index,
                    f"checkpoint_{checkpoint}_{suffix}",
                    f"{checkpoint}番目{ name_ja }",
                    group=f"checkpoint_{checkpoint}",
                    description=description,
                    physical_quantity=quantity,
                    unit="dimensionless",
                    model_representation=(
                        "categorical encoded value {0, 0.5, 1}"
                        if is_direction else "normalized scalar"
                    ),
                    normalization="NodeNetworkNavigation normalization and clip to [0,1]",
                    source=source_navi,
                    replacement=_reference_replacement(),
                    reference=f"vehicle.navigation.get_navi_info() checkpoint {checkpoint} reference lane",
                    value_type="categorical" if is_direction else "continuous",
                    coupled_indices=curvature_group if is_curvature_member else (),
                    independently_replaceable=not is_curvature_member,
                    ig_interpolation_allowed=not is_direction,
                )
            )

    for index in range(19, _OFFICIAL_DIM):
        offset = index - 19
        specs.append(
            _spec(
                index,
                f"lidar_{offset:03d}",
                f"LiDAR {offset:03d}",
                group="lidar_all",
                description="Ray hit fraction; no hit remains at the detector default 1.0.",
                physical_quantity="ray hit fraction",
                unit="fraction of 50 m range",
                model_representation="cloud_points hit fraction",
                normalization="MetaDrive detector value in [0,1]",
                source=source_lidar,
                replacement={"kind": "fixed", "value": 1.0, "value_method": "no_detection_default"},
                reference="vehicle-local heading frame; LidarStateObservation ray order starts at vehicle head clockwise",
            )
        )
    return specs


def official_schema_259() -> InputSchema:
    """Return a fresh explicit schema for the verified 259-vector contract."""

    return InputSchema(
        dimension=_OFFICIAL_DIM,
        schema_id="metadrive_default_259",
        version="1",
        preprocessing={
            "external_normalization": False,
            "sb3_vector_preprocess": "float32 identity for Box observations",
            "source_contract": "MetaDrive 0.4.3 commit 85e5dadc",
            "source_files": [
                "metadrive/obs/state_obs.py",
                "metadrive/component/vehicle/base_vehicle.py",
                "metadrive/component/navigation_module/node_network_navigation.py",
                "metadrive/component/sensors/distance_detector.py",
                "metadrive/component/sensors/lidar.py",
                "metadrive/policy/env_input_policy.py",
            ],
            "source_file_paths": dict(_OFFICIAL_SOURCE_RELATIVE_PATHS),
            "expected_source_sha256": dict(_OFFICIAL_SOURCE_SHA256),
            "runtime_observation_defaults": {
                "num_lasers": 240,
                "num_others": 0,
                "side_detector_num_lasers": 0,
                "lane_line_detector_num_lasers": 0,
            },
        },
        inputs=tuple(_official_specs()),
    )


def custom_262_schema_template(
    *, custom_indices: Sequence[int] = (7, 145, 260)
) -> InputSchema:
    """Return an unresolved 262 template with explicit non-tail placeholders.

    The known 259 fields retain their order in the remaining positions.  The
    custom fields are intentionally unresolved and coupled, so callers cannot
    silently treat a guessed index, neutral value, or flag as verified.
    """

    indices = tuple(int(index) for index in custom_indices)
    if len(indices) != 3 or len(set(indices)) != 3 or any(index < 0 or index >= _CUSTOM_DIM for index in indices):
        raise SchemaError("custom_indices must contain three unique values in 0..261")
    official = _official_specs()
    custom_set = set(indices)
    available = (index for index in range(_CUSTOM_DIM) if index not in custom_set)
    position_map = dict(zip(range(_OFFICIAL_DIM), available, strict=True))
    coupled = tuple(sorted(indices))
    moved = []
    for spec in official:
        moved.append(
            replace(
                spec,
                index=position_map[spec.index],
                related_valid_flags=tuple(position_map.get(index, index) for index in spec.related_valid_flags),
                coupled_indices=tuple(position_map.get(index, index) for index in spec.coupled_indices),
            )
        )
    placeholders = [
        _spec(
            index,
            f"UNRESOLVED_custom_lane_feature_{order}",
            f"未確定カスタム入力 {order}",
            group="custom_lane_features_unresolved",
            description="Source verification required; do not infer from dimension or position.",
            physical_quantity="UNRESOLVED",
            unit="UNRESOLVED",
            model_representation="UNRESOLVED",
            normalization="UNRESOLVED",
            source="移植先の既存観測生成ソースが必要",
            replacement={"kind": "unresolved", "value_method": "source_verification_required"},
            value_range=None,
            independently_replaceable=False,
            coupled_indices=coupled,
            ig_interpolation_allowed=False,
        )
        for order, index in enumerate(indices)
    ]
    return InputSchema(
        dimension=_CUSTOM_DIM,
        schema_id="custom_262_template",
        version="1",
        preprocessing={"external_normalization": "UNRESOLVED"},
        inputs=tuple(sorted((*moved, *placeholders), key=lambda item: item.index)),
    )


def unresolved_indices(schema: InputSchema) -> tuple[int, ...]:
    """Return placeholder indices without relying on their numeric positions."""

    return tuple(
        item.index
        for item in schema.inputs
        if item.id.startswith("UNRESOLVED_")
        or (item.replacement or {}).get("kind") == "unresolved"
    )


def fake_262_resolved_schema(
    *, custom_indices: Sequence[int] = (7, 145, 260)
) -> InputSchema:
    """Return a fully resolved synthetic 262 schema for adapter-only tests.

    This fixture deliberately uses non-tail positions and explicit neutral
    values.  Its three custom meanings are synthetic and must never be used
    as evidence for a real port's 262-dimensional observation.
    """

    template = custom_262_schema_template(custom_indices=custom_indices)
    definitions = {
        custom_indices[0]: {
            "id": "target_lane_offset_synthetic",
            "name_ja": "synthetic target lane offset",
            "description": "Synthetic signed target-lane offset represented in [0,1].",
            "physical_quantity": "synthetic target lane lateral offset",
            "unit": "normalized",
            "model_representation": "synthetic normalized scalar",
            "normalization": "synthetic map to [0,1]",
            "value_type": "continuous",
            "replacement": {"kind": "fixed", "value": 0.5, "value_method": "synthetic_neutral"},
            "ig_interpolation_allowed": True,
        },
        custom_indices[1]: {
            "id": "target_lane_heading_error_synthetic",
            "name_ja": "synthetic target lane heading error",
            "description": "Synthetic target-lane heading error represented in [0,1].",
            "physical_quantity": "synthetic target lane heading error",
            "unit": "normalized",
            "model_representation": "synthetic normalized scalar",
            "normalization": "synthetic map to [0,1]",
            "value_type": "continuous",
            "replacement": {"kind": "fixed", "value": 0.5, "value_method": "synthetic_aligned"},
            "ig_interpolation_allowed": True,
        },
        custom_indices[2]: {
            "id": "target_lane_valid_synthetic",
            "name_ja": "synthetic target lane valid flag",
            "description": "Synthetic target-lane validity flag represented as bool 0/1.",
            "physical_quantity": "synthetic validity flag",
            "unit": "bool",
            "model_representation": "synthetic bool 0/1",
            "normalization": "none",
            "value_type": "bool",
            "replacement": {"kind": "fixed", "value": 1.0, "value_method": "synthetic_valid"},
            "ig_interpolation_allowed": False,
        },
    }
    resolved = []
    for spec in template.inputs:
        definition = definitions.get(spec.index)
        if definition is None:
            resolved.append(spec)
            continue
        resolved.append(
            replace(
                spec,
                id=definition["id"],
                name_ja=definition["name_ja"],
                group="synthetic_target_lane_features",
                description=definition["description"],
                physical_quantity=definition["physical_quantity"],
                unit=definition["unit"],
                model_representation=definition["model_representation"],
                normalization=definition["normalization"],
                reference="synthetic fake target-lane feature contract",
                value_type=definition["value_type"],
                value_range=(0.0, 1.0),
                replacement=definition["replacement"],
                independently_replaceable=True,
                coupled_indices=(),
                ig_interpolation_allowed=definition["ig_interpolation_allowed"],
                source="synthetic fixture; not a real MetaDrive port",
            )
        )
    return InputSchema(
        dimension=_CUSTOM_DIM,
        schema_id="fake_262_resolved_synthetic",
        version="1",
        preprocessing={
            "external_normalization": False,
            "source_contract": "synthetic_fake_adapter_only",
        },
        inputs=tuple(sorted(resolved, key=lambda item: item.index)),
    )


def assert_schema_resolved(schema: InputSchema) -> None:
    """Fail closed when a template still contains unresolved placeholders."""

    unresolved = unresolved_indices(schema)
    if unresolved:
        raise SchemaError(
            "schema contains unresolved input meanings; verify the source before running: "
            f"indices={list(unresolved)}"
        )


def official_schema() -> InputSchema:
    """Compatibility alias used by adapters and smoke tests."""

    return official_schema_259()


def custom_schema_template(*, custom_indices: Sequence[int] = (7, 145, 260)) -> InputSchema:
    """Compatibility alias for :func:`custom_262_schema_template`."""

    return custom_262_schema_template(custom_indices=custom_indices)


__all__ = [
    "assert_schema_resolved",
    "custom_262_schema_template",
    "custom_schema_template",
    "fake_262_resolved_schema",
    "official_schema",
    "official_schema_259",
    "unresolved_indices",
]
