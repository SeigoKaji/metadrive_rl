"""Cheap adapter/schema contract tests; no MetaDrive process or model load."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import json

import numpy as np
import pytest

from input_attribution.adapters import (
    FakePolicyAdapter,
    AdapterError,
    MetaDriveAdapter,
    assert_schema_resolved,
    custom_262_schema_template,
    fake_262_resolved_schema,
    official_schema_259,
    unresolved_indices,
)
from input_attribution.collection import collect_reference
from input_attribution.integrated_gradients import compute_integrated_gradients
from input_attribution.interventions import InterventionError, InterventionPattern, apply_intervention
from input_attribution.schema import SchemaError, load_schema
from input_attribution.adapters.port_template import PortAdapterTemplate


def test_official_fixture_covers_all_259_indices_and_verified_groups() -> None:
    schema = official_schema_259()
    assert schema.dimension == 259
    assert tuple(schema.by_index) == tuple(range(259))
    assert [item.group for item in schema.inputs[:9]] == [
        "road_boundaries",
        "road_boundaries",
        "heading",
        "speed",
        "steering_and_action_history",
        "steering_and_action_history",
        "steering_and_action_history",
        "yaw_rate",
        "current_lane_lateral_position",
    ]
    assert schema.spec(2).id == "heading_diff_to_reference_lane_lateral"
    assert "lateral/right vector" in schema.spec(2).description
    assert schema.spec(19).id == "lidar_000"
    assert schema.spec(258).id == "lidar_239"
    assert schema.spec(12).value_type == "categorical"
    assert schema.spec(12).ig_interpolation_allowed is False
    assert schema.spec(12).independently_replaceable is False
    assert schema.spec(11).coupled_indices == (11, 12, 13)
    assert schema.spec(13).ig_interpolation_allowed is True
    assert schema.spec(13).independently_replaceable is False
    assert schema.spec(17).value_type == "categorical"
    assert schema.spec(17).coupled_indices == (16, 17, 18)
    schema.validate_observation(np.full(259, 0.5, dtype=np.float32))


def test_262_fixture_fails_closed_for_non_tail_unresolved_fields() -> None:
    schema = custom_262_schema_template(custom_indices=(7, 145, 260))
    assert schema.dimension == 262
    assert unresolved_indices(schema) == (7, 145, 260)
    assert all(
        schema.spec(index).independently_replaceable is False
        and schema.spec(index).ig_interpolation_allowed is False
        for index in unresolved_indices(schema)
    )
    with pytest.raises(SchemaError, match="unresolved"):
        assert_schema_resolved(schema)


def test_intervention_rejects_partial_unresolved_coupling() -> None:
    schema = custom_262_schema_template(custom_indices=(7, 145, 260))
    pattern = InterventionPattern(
        pattern_id="P-unresolved-one",
        indices=(145,),
        method="fixed",
        values={145: 0.5},
    )
    with pytest.raises(InterventionError, match="coupled"):
        apply_intervention(np.full(262, 0.5, dtype=np.float32), pattern, schema)


def test_fake_adapter_exercises_runtime_hooks_without_mutating_input() -> None:
    adapter = FakePolicyAdapter(dimension=5, action_count=3, horizon=2)
    config = SimpleNamespace(scenario={"episodes": 1, "horizon": 2}, scenario_seed=17)
    source = np.linspace(0.0, 0.4, 5, dtype=np.float32)
    original = source.copy()
    output = adapter.evaluate(source)
    assert output.probabilities.shape == (1, 3)
    np.testing.assert_allclose(output.probabilities.sum(axis=1), 1.0)
    assert adapter.predict(source) in range(3)
    assert np.array_equal(source, original)

    result = collect_reference(adapter, adapter, config, max_steps=2, scenario_seed=17)
    assert result.episodes_completed == 1
    assert len(result.records) == 2
    assert result.records[0]["preprocess"]["external_normalization"] is False
    assert result.records[0]["decoded_action"]["action"] in range(3)
    assert result.records[-1]["post_telemetry"]["sim_time_s"] == pytest.approx(0.2)


def test_fake_policy_manual_ig_preserves_fixed_action_and_completeness() -> None:
    adapter = FakePolicyAdapter(dimension=5, action_count=3)
    observation = np.full(5, 0.8, dtype=np.float32)
    baseline = np.full(5, 0.2, dtype=np.float32)
    result = compute_integrated_gradients(
        observation,
        baseline,
        adapter,
        target_action=1,
        n_steps=16,
        method="riemann_trapezoid",
        backend="manual",
    )
    assert result.target_action == 1
    assert result.attributions.shape == (5,)
    assert result.absolute_completeness_error < 1e-5
    assert result.policy_unchanged is True


def test_metadrive_adapter_is_lazy_and_uses_existing_action_decoder() -> None:
    env_config = {
        "discrete_action": True,
        "discrete_steering_dim": 3,
        "discrete_throttle_dim": 3,
    }
    adapter = MetaDriveAdapter(env_config=env_config)
    assert adapter.env is None
    assert adapter.model is None
    decoded = adapter.decode_action(4)
    assert decoded.action_id == 4
    assert decoded.steering == pytest.approx(0.0)
    contract = adapter.verify_schema_contract(official_schema_259())
    assert contract["verified"] is True
    assert contract["source_evidence"]["byte_hashes"]["status"] == "verified"
    vector = adapter.preprocess_observation(np.zeros(259, dtype=np.float64))
    assert vector.dtype == np.float32
    assert vector.shape == (259,)


def test_metadrive_schema_contract_rejects_same_dimension_swapped_ids() -> None:
    schema = official_schema_259()
    entries = list(schema.inputs)
    first, second = entries[0], entries[1]
    entries[0] = replace(first, id=second.id)
    entries[1] = replace(second, id=first.id)
    swapped = type(schema)(
        dimension=schema.dimension,
        schema_id=schema.schema_id,
        version=schema.version,
        preprocessing=schema.preprocessing,
        inputs=tuple(entries),
    )
    with pytest.raises(AdapterError, match="semantic contract"):
        MetaDriveAdapter(env_config={}).verify_schema_contract(swapped)


def test_metadrive_schema_contract_rejects_modified_source_bytes() -> None:
    schema = official_schema_259()
    preprocessing = dict(schema.preprocessing)
    expected = dict(preprocessing["expected_source_sha256"])
    expected["metadrive/obs/state_obs.py"] = "0" * 64
    preprocessing["expected_source_sha256"] = expected
    modified_provenance = type(schema)(
        dimension=schema.dimension,
        schema_id=schema.schema_id,
        version=schema.version,
        preprocessing=preprocessing,
        inputs=schema.inputs,
    )
    with pytest.raises(AdapterError, match="source hash mismatch"):
        MetaDriveAdapter(env_config={}).verify_schema_contract(modified_provenance)


def test_metadrive_render_hook_returns_rgb_copy_without_recording() -> None:
    class RenderEnv:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []
            self.source = np.zeros((4, 5, 3), dtype=np.uint8)
            self.source[:, :, 1] = 17

        def render(self, **kwargs: object) -> np.ndarray:
            self.calls.append(kwargs)
            return self.source

    env = RenderEnv()
    frame = MetaDriveAdapter().render_frame(env, step=1)
    assert frame.dtype == np.uint8
    assert frame.shape == (4, 5, 3)
    assert frame is not env.source
    frame[0, 0, 0] = 255
    assert env.source[0, 0, 0] == 0
    assert env.calls[0]["mode"] == "topdown"
    assert env.calls[0]["screen_record"] is False


def test_metadrive_telemetry_keeps_raw_route_distance_completion_and_heading() -> None:
    vehicle = SimpleNamespace(
        heading_theta=1.25,
        lane_index=("road", "lane", 0),
        navigation=SimpleNamespace(
            current_ref_lanes=[],
            travelled_length=12.5,
            route_completion=0.375,
        ),
    )
    telemetry = MetaDriveAdapter().runtime_telemetry(vehicle)
    assert telemetry["route_progress_m"] == pytest.approx(12.5)
    assert telemetry["route_completion"] == pytest.approx(0.375)
    assert telemetry["heading_rad"] == pytest.approx(1.25)


def test_fake_262_resolved_fixture_is_explicit_and_non_tail() -> None:
    schema = fake_262_resolved_schema(custom_indices=(7, 145, 260))
    schema.validate_for_execution()
    assert schema.spec("target_lane_offset_synthetic").index == 7
    assert schema.spec("target_lane_heading_error_synthetic").index == 145
    assert schema.spec("target_lane_valid_synthetic").index == 260
    assert schema.spec(12).coupled_indices == (12, 13, 14)
    assert schema.spec(13).value_type == "categorical"


def test_port_template_fails_closed_without_guessing_target_contract() -> None:
    adapter = PortAdapterTemplate()
    with pytest.raises(NotImplementedError, match="load_policy"):
        adapter.load_policy()
    with pytest.raises(NotImplementedError, match="preprocess_observation"):
        adapter.preprocess_observation(np.zeros(3, dtype=np.float32))


def test_portable_schema_files_keep_259_loadable_and_262_unresolved() -> None:
    package_root = Path(__file__).resolve().parents[1]
    official = load_schema(package_root / "schemas" / "official_259.json")
    assert official.dimension == 259
    assert len(official.inputs) == 259
    template = json.loads(
        (package_root / "schemas" / "custom_262_template.json").read_text(
            encoding="utf-8"
        )
    )
    assert template["dimension"] == 262
    assert [item["id"] for item in template["inputs"]] == [
        "target_lane_offset",
        "target_lane_heading_error",
        "target_lane_valid",
    ]
    assert all(item["index"] is None for item in template["inputs"])
    with pytest.raises(SchemaError, match="index"):
        load_schema(package_root / "schemas" / "custom_262_template.json")
