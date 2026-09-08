"""Synthetic regression contracts for typed interventions and scope metadata."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from input_attribution.adapters import FakePolicyAdapter, fake_262_resolved_schema
from input_attribution.config import ConfigError, config_from_mapping
from input_attribution.interventions import (
    InterventionError,
    InterventionPattern,
    apply_intervention,
)
from input_attribution.schema import InputSchema, InputSpec, SchemaError


class _Fake262AffineAdapter(FakePolicyAdapter):
    """Synthetic 262 port with an explicit physical/model affine contract."""

    center = np.float32(0.25)
    scale = np.float32(0.37)

    def encode(self, physical: object) -> np.ndarray:
        values = np.asarray(physical, dtype=np.float32)
        return np.asarray(self.center + self.scale * values, dtype=np.float32)

    def decode(self, model_observation: object) -> np.ndarray:
        values = np.asarray(model_observation, dtype=np.float32)
        return np.asarray((values - self.center) / self.scale, dtype=np.float32)


def _schema(*, categorical: bool = False, unsigned: bool = False) -> InputSchema:
    value_type = "categorical" if categorical else "continuous"
    description = "unsigned synthetic input" if unsigned else "synthetic normalized input"
    categories = [0.0, 0.5, 1.0] if categorical else None
    variants = [
        {
            "id": "neutral",
            "operation": "neutral",
            "value": 0.5,
            "value_type": value_type,
            "categories": categories,
            "meaning": "synthetic neutral level",
            "evidence": "synthetic fixture source",
            "confirmed": True,
        },
        {
            "id": "reflect",
            "operation": "reflection",
            "center": 0.5,
            "value_type": value_type,
            "meaning": "synthetic reflection around the known center",
            "evidence": "synthetic fixture source",
            "confirmed": True,
        },
        {
            "id": "low",
            "operation": "fixed_level",
            "value": 0.4,
            "value_type": value_type,
            "categories": categories,
            "meaning": "synthetic lower diagnostic level",
            "evidence": "synthetic fixture source",
            "confirmed": True,
        },
    ]
    if categorical:
        variants.pop(1)
    return InputSchema(
        dimension=1,
        inputs=(
            InputSpec(
                0,
                "x",
                "x",
                group="synthetic",
                description=description,
                physical_quantity="synthetic value",
                model_representation="normalized scalar",
                normalization="synthetic [0,1]",
                reference="synthetic fixture",
                source="test_revision_interventions.py",
                value_type=value_type,
                value_range=(0.0, 1.0),
                variants=tuple(variants),
            ),
        ),
    )


def test_typed_variants_preserve_source_and_report_exact_vs_meaningful_delta() -> None:
    schema = _schema()
    source = np.array([0.7], dtype=np.float32)
    original = source.copy()
    result = apply_intervention(
        source,
        InterventionPattern(
            "reflect",
            indices=(0,),
            method="reflection",
            variant_id="reflect",
            tolerance=0.1,
        ),
        schema,
    )
    np.testing.assert_array_equal(source, original)
    np.testing.assert_allclose(result.observation, [0.3], atol=1e-6)
    assert result.changed_count_exact == 1
    assert result.meaningful_changed_count == 1
    assert result.delta_abs_values[0] == pytest.approx(0.4, abs=1e-6)
    assert result.to_dict()["scope"] == "full_episode"


def test_neutral_and_lidar_like_noop_are_not_skipped() -> None:
    schema = _schema()
    source = np.array([0.5], dtype=np.float32)
    result = apply_intervention(
        source,
        InterventionPattern("neutral", indices=(0,), method="neutral", variant_id="neutral"),
        schema,
    )
    assert result.skipped is False
    assert result.changed_indices == ()
    assert result.no_op_indices == (0,)
    assert result.applied_count == result.no_op_count == 1


def test_float32_nextafter_is_exact_change_but_can_be_below_meaningful_tolerance() -> None:
    schema = _schema()
    source = np.array([0.5], dtype=np.float32)
    next_value = np.nextafter(source[0], np.float32(1.0)).item()
    pattern = InterventionPattern(
        "nextafter",
        indices=(0,),
        method="fixed_level",
        variant={
            "id": "nextafter",
            "operation": "fixed_level",
            "value": next_value,
            "meaning": "float32 representable diagnostic level",
            "evidence": "synthetic dtype fixture",
            "confirmed": True,
        },
        tolerance=1e-5,
    )
    result = apply_intervention(source, pattern, schema)
    assert result.changed_count_exact == 1
    assert result.meaningful_changed_count == 0


def test_unsigned_reflection_is_rejected_even_with_a_valid_variant() -> None:
    schema = _schema(unsigned=True)
    with pytest.raises(InterventionError, match="unsigned"):
        apply_intervention(
            np.array([0.2], dtype=np.float32),
            InterventionPattern("reflect", indices=(0,), method="reflection", variant_id="reflect"),
            schema,
        )
    structured_spec = replace(
        _schema().spec(0),
        description="synthetic continuous value",
        physical_quantity="synthetic value",
        replacement={"signedness": "unsigned"},
    )
    structured_schema = InputSchema(dimension=1, inputs=(structured_spec,))
    with pytest.raises(InterventionError, match="unsigned"):
        apply_intervention(
            np.array([0.2], dtype=np.float32),
            InterventionPattern(
                "structured-unsigned",
                indices=(0,),
                method="reflection",
                variant_id="reflect",
            ),
            structured_schema,
        )


def test_schema_group_reflection_rejects_discrete_targets() -> None:
    with pytest.raises(SchemaError, match="continuous group"):
        InputSchema(
            dimension=1,
            intervention_variants={
                "bad_group_reflection": {
                    "id": "bad_group_reflection",
                    "operation": "reflection",
                    "center": 0.5,
                    "value_type": "categorical",
                    "meaning": "invalid discrete group reflection",
                    "evidence": "synthetic fixture",
                    "confirmed": True,
                },
            },
            inputs=(InputSpec(0, "flag", "flag", group="flags", value_type="categorical"),),
        )

    schema = InputSchema(
        dimension=1,
        inputs=(InputSpec(0, "flag", "flag", group="flags", value_type="categorical"),),
    )
    with pytest.raises(InterventionError, match="discrete"):
        apply_intervention(
            np.array([0.5], dtype=np.float32),
            InterventionPattern(
                "group-reflection",
                indices=(0,),
                method="reflection",
                variant={
                    "id": "inline_group_reflection",
                    "operation": "reflection",
                    "center": 0.5,
                    "meaning": "synthetic inline reflection checked per target",
                    "evidence": "synthetic fixture",
                    "confirmed": True,
                },
            ),
            schema,
        )


def test_group_variant_mapping_missing_target_is_normalized_to_failed_skip() -> None:
    schema = InputSchema(
        dimension=2,
        intervention_variants={
            "partial_group": {
                "id": "partial_group",
                "operation": "fixed_level",
                "values": {"x": 0.4},
                "allowed_ids": ["x", "y"],
                "meaning": "synthetic partial group mapping",
                "evidence": "synthetic fixture",
                "confirmed": True,
            },
        },
        inputs=(
            InputSpec(0, "x", "x", group="continuous"),
            InputSpec(1, "y", "y", group="continuous"),
        ),
    )
    source = np.array([0.2, 0.3], dtype=np.float32)
    result = apply_intervention(
        source,
        InterventionPattern(
            "partial-group",
            indices=(0, 1),
            method="fixed_level",
            variant_id="partial_group",
        ),
        schema,
        strict=False,
    )
    assert result.skipped is True
    assert result.eligible is True
    assert result.skip_category == "failed"
    assert "missing values" in (result.skip_reason or "")
    np.testing.assert_array_equal(result.observation, source)


def test_invalid_joint_confirmation_requires_sentinel_or_member_evidence() -> None:
    schema = InputSchema(
        dimension=3,
        inputs=(
            InputSpec(0, "offset", "offset", group="lane"),
            InputSpec(1, "heading", "heading", group="lane"),
            InputSpec(
                2,
                "valid",
                "valid",
                group="lane",
                value_type="bool",
                independently_replaceable=False,
                coupled_indices=(0, 1, 2),
            ),
        ),
    )
    source = np.array([0.2, 0.3, 1.0], dtype=np.float32)
    pattern_values = {0: 0.0, 1: 0.0, 2: 0.0}
    with pytest.raises(InterventionError, match="source-confirmed invalid_value"):
        apply_intervention(
            source,
            InterventionPattern(
                "arbitrary-invalid",
                indices=(0, 1, 2),
                values=pattern_values,
                metadata={"invalid_joint_confirmed": True},
            ),
            schema,
        )

    evidence = {
        "offset": {"value": 0.0, "confirmed": True, "evidence": "synthetic offset source"},
        "heading": {"value": 0.0, "confirmed": True, "evidence": "synthetic heading source"},
        "valid": {"value": 0.0, "confirmed": True, "evidence": "synthetic valid source"},
    }
    result = apply_intervention(
        source,
        InterventionPattern(
            "evidence-invalid",
            indices=(0, 1, 2),
            values=pattern_values,
            metadata={
                "invalid_joint_confirmed": True,
                "invalid_joint_evidence": evidence,
            },
        ),
        schema,
    )
    assert result.changed_indices == (0, 1, 2)


def test_zero_center_reflection_preserves_sign_and_explicit_clip_does_not_hide_nonfinite() -> None:
    base = _schema()
    spec = base.spec(0)
    signed = InputSchema(
        dimension=1,
        inputs=(
            InputSpec(
                0,
                spec.id,
                spec.name_ja,
                group=spec.group,
                description=spec.description,
                physical_quantity=spec.physical_quantity,
                model_representation=spec.model_representation,
                normalization=spec.normalization,
                reference=spec.reference,
                source=spec.source,
                value_range=(-1.0, 1.0),
                variants=(
                    {
                        "id": "zero_reflect",
                        "operation": "reflection",
                        "center": 0.0,
                        "meaning": "signed reflection around zero",
                        "evidence": "synthetic fixture",
                        "confirmed": True,
                    },
                ),
            ),
        ),
    )
    result = apply_intervention(
        np.array([0.2], dtype=np.float32),
        InterventionPattern("signed", indices=(0,), method="reflection", variant_id="zero_reflect"),
        signed,
    )
    np.testing.assert_allclose(result.observation, [-0.2], atol=1e-6)
    bad = InterventionPattern(
        "nonfinite",
        indices=(0,),
        method="fixed_level",
        variant={
            "id": "bad",
            "operation": "fixed_level",
            "value": float("inf"),
            "meaning": "invalid diagnostic",
            "evidence": "synthetic fixture",
            "confirmed": True,
        },
        clip=True,
    )
    with pytest.raises(InterventionError, match="non-finite"):
        apply_intervention(np.array([0.2], dtype=np.float32), bad, signed)


def test_variant_target_group_prevents_reusing_lidar_variant_for_lane_input() -> None:
    schema = InputSchema(
        dimension=2,
        intervention_variants={
            "lidar": {
                "id": "lidar",
                "operation": "fixed_level",
                "value": 0.5,
                "allowed_groups": ["lidar"],
                "meaning": "synthetic LiDAR diagnostic",
                "evidence": "synthetic fixture",
                "confirmed": True,
            },
        },
        inputs=(
            InputSpec(0, "lane", "lane", group="lane"),
            InputSpec(1, "beam", "beam", group="lidar"),
        ),
    )
    with pytest.raises(InterventionError, match="input group"):
        apply_intervention(
            np.array([0.2, 0.2], dtype=np.float32),
            InterventionPattern("wrong-target", indices=(0,), method="fixed_level", variant_id="lidar"),
            schema,
        )
    result = apply_intervention(
        np.array([0.2, 0.2], dtype=np.float32),
        InterventionPattern("right-target", indices=(1,), method="fixed_level", variant_id="lidar"),
        schema,
    )
    assert result.changed_indices == (1,)


def test_fake_262_variants_use_its_own_semantics_and_adapter_round_trip() -> None:
    """A synthetic 262 port must carry its own IDs and normalized levels.

    The fixture deliberately uses a reflection center of 0.25 and a physical
    fixed level of 0.8 encoded with scale 0.37, so accidentally reusing the
    official 259 center/levels would produce a different result. Each typed
    operation changes only its declared target, and the affine adapter decodes
    the final float32 model vector back to the expected physical value.
    """

    base = fake_262_resolved_schema(custom_indices=(7, 145, 260))
    adapter = _Fake262AffineAdapter(dimension=base.dimension, action_count=9)
    assert adapter.center == pytest.approx(0.25)
    assert adapter.scale == pytest.approx(0.37)
    assert adapter.center != pytest.approx(0.5)
    fixed_physical_level = np.float32(0.8)
    fixed_model_level = float(adapter.encode([fixed_physical_level])[0])
    custom_variants: dict[int, tuple[dict[str, object], ...]] = {
        7: (
            {
                "id": "fake262_offset_neutral_0_25",
                "operation": "neutral",
                "value": float(adapter.center),
                "value_type": "continuous",
                "range": [0.0, 1.0],
                "meaning": "synthetic 262 target offset neutral level",
                "evidence": "fake_262_resolved synthetic contract",
                "confirmed": True,
            },
        ),
        145: (
            {
                "id": "fake262_heading_reflection_center_0_25",
                "operation": "reflection",
                "center": float(adapter.center),
                # The adapter owns this model/physical scale.  The generic
                # intervention only receives the final model-space vector.
                "scale": float(adapter.scale),
                "value_type": "continuous",
                "range": [0.0, 1.0],
                "meaning": "synthetic 262 heading reflection center",
                "evidence": "fake_262_resolved synthetic contract",
                "confirmed": True,
            },
            {
                "id": "fake262_heading_fixed_physical_0_8",
                "operation": "fixed_level",
                "value": fixed_model_level,
                "value_type": "continuous",
                "range": [0.0, 1.0],
                "meaning": "synthetic 262 heading fixed diagnostic level",
                "evidence": "fake_262_resolved synthetic contract",
                "confirmed": True,
            },
        ),
    }
    schema = replace(
        base,
        inputs=tuple(
            replace(spec, variants=custom_variants.get(spec.index, spec.variants))
            for spec in base.inputs
        ),
    )
    physical_source = np.full(schema.dimension, 0.2, dtype=np.float32)
    physical_source[7] = np.float32(0.5)
    physical_source[145] = np.float32(0.5)
    source = adapter.encode(physical_source)
    # The custom validity flag is already a final model-space value in this
    # fixture and is intentionally excluded from the affine round trip.
    source[260] = np.float32(1.0)
    original = source.copy()
    np.testing.assert_allclose(
        adapter.decode(source)[[7, 145]], physical_source[[7, 145]], atol=2e-6
    )

    neutral = apply_intervention(
        source,
        InterventionPattern(
            "fake262-neutral",
            indices=(7,),
            method="neutral",
            variant_id="fake262_offset_neutral_0_25",
        ),
        schema,
    )
    assert neutral.observation[7] == pytest.approx(float(adapter.center), abs=1e-7)
    assert adapter.decode(neutral.observation)[7] == pytest.approx(0.0, abs=2e-6)
    np.testing.assert_array_equal(neutral.observation[np.arange(schema.dimension) != 7], original[np.arange(schema.dimension) != 7])

    reflected = apply_intervention(
        source,
        InterventionPattern(
            "fake262-reflect",
            indices=(145,),
            method="reflection",
            variant_id="fake262_heading_reflection_center_0_25",
        ),
        schema,
    )
    assert reflected.observation[145] == pytest.approx(
        float(np.float32(2.0 * adapter.center - source[145])), abs=1e-7
    )
    assert adapter.decode(reflected.observation)[145] == pytest.approx(-0.5, abs=2e-6)
    assert schema.variant(145, "fake262_heading_reflection_center_0_25")["scale"] == pytest.approx(float(adapter.scale))
    np.testing.assert_array_equal(reflected.observation[np.arange(schema.dimension) != 145], original[np.arange(schema.dimension) != 145])

    fixed = apply_intervention(
        source,
        InterventionPattern(
            "fake262-fixed",
            indices=(145,),
            method="fixed_level",
            variant_id="fake262_heading_fixed_physical_0_8",
        ),
        schema,
    )
    assert fixed.observation[145] == pytest.approx(fixed_model_level, abs=1e-7)
    assert adapter.decode(fixed.observation)[145] == pytest.approx(float(fixed_physical_level), abs=2e-6)
    np.testing.assert_array_equal(fixed.observation[np.arange(schema.dimension) != 145], original[np.arange(schema.dimension) != 145])
    np.testing.assert_array_equal(source, original)

    final_model_observation = adapter.preprocess_observation(reflected.observation)
    assert final_model_observation.dtype == np.float32
    np.testing.assert_array_equal(final_model_observation, reflected.observation)

    with pytest.raises(InterventionError, match="unknown intervention variant"):
        apply_intervention(
            source,
            InterventionPattern(
                "fake262-wrong-target",
                indices=(2,),
                method="reflection",
                variant_id="fake262_heading_reflection_center_0_25",
            ),
            schema,
        )


def test_confirmed_true_without_meaning_and_evidence_is_rejected() -> None:
    with pytest.raises(SchemaError, match="confirmed=true"):
        InputSchema(
            dimension=1,
            inputs=(
                InputSpec(
                    0,
                    "x",
                    "x",
                    group="synthetic",
                    variants=(
                        {"id": "bad", "operation": "neutral", "value": 0.5, "confirmed": True},
                    ),
                ),
            ),
        )


def test_categories_reject_nan_and_values_outside_declared_category_set() -> None:
    schema = _schema(categorical=True)
    with pytest.raises(InterventionError, match="categor"):
        apply_intervention(
            np.array([0.0], dtype=np.float32),
            InterventionPattern(
                "bad-category",
                indices=(0,),
                method="fixed_level",
                variant={
                    "id": "inline",
                    "operation": "fixed_level",
                    "value": float("nan"),
                    "categories": [0.0, 0.5, 1.0],
                    "meaning": "synthetic category",
                    "evidence": "synthetic fixture",
                    "confirmed": True,
                },
            ),
            schema,
        )


def test_variant_range_is_narrower_than_schema_and_checks_generated_values() -> None:
    with pytest.raises(SchemaError, match="variant range maximum"):
        InputSchema(
            dimension=1,
            inputs=(
                InputSpec(
                    0,
                    "x",
                    "x",
                    group="synthetic",
                    value_range=(0.0, 1.0),
                    variants=(
                        {
                            "id": "bad_registered",
                            "operation": "fixed_level",
                            "value": 0.8,
                            "range": [0.4, 0.6],
                            "meaning": "inconsistent registered level",
                            "evidence": "synthetic fixture",
                            "confirmed": True,
                        },
                    ),
                ),
            ),
        )

    schema = _schema()
    with pytest.raises(InterventionError, match="variant range"):
        apply_intervention(
            np.array([0.2], dtype=np.float32),
            InterventionPattern(
                "bad_inline",
                indices=(0,),
                method="fixed_level",
                variant={
                    "id": "bad_inline",
                    "operation": "fixed_level",
                    "value": 0.8,
                    "range": [0.4, 0.6],
                    "meaning": "inconsistent inline level",
                    "evidence": "synthetic fixture",
                    "confirmed": True,
                },
            ),
            schema,
        )
    with pytest.raises(InterventionError, match="variant range"):
        apply_intervention(
            np.array([0.0], dtype=np.float32),
            InterventionPattern(
                "bad_reflection_domain",
                indices=(0,),
                method="reflection",
                variant={
                    "id": "bad_reflection_domain",
                    "operation": "reflection",
                    "center": 0.5,
                    "range": [0.4, 0.6],
                    "meaning": "reflection whose generated value leaves its domain",
                    "evidence": "synthetic fixture",
                    "confirmed": True,
                },
            ),
            schema,
        )


def test_scope_defaults_preserve_legacy_reference_and_reject_invalid_config() -> None:
    config = config_from_mapping({"patterns": [{"id": "P00"}, {"id": "legacy", "method": "reference", "indices": [0]}]})
    legacy = next(row for row in config.patterns if row["id"] == "legacy")
    assert legacy["scope"] == "explicitly_conditional"
    assert legacy["on_inapplicable"] == "continue_unmodified_with_warning"
    with pytest.raises(ConfigError, match="on_inapplicable"):
        config_from_mapping(
            {
                "patterns": [
                    {"id": "P00"},
                    {"id": "bad", "method": "neutral", "indices": [0], "on_inapplicable": "ignore"},
                ]
            }
        )


def test_video_pattern_selection_is_a_declared_pattern_subset() -> None:
    config = config_from_mapping(
        {
            "patterns": [
                {"id": "P00", "method": "identity", "indices": []},
                {"id": "P04_lateral_low", "method": "fixed_level", "indices": [0]},
            ],
            "closed_loop": {"patterns": ["P00"]},
            "video": {"enabled": False, "patterns": ["P00", "P04_lateral_low"]},
        }
    )
    assert config.video["enabled"] is False
    assert config.video["patterns"] == ["P00", "P04_lateral_low"]
    with pytest.raises(ConfigError, match="video.patterns"):
        config_from_mapping(
            {
                "patterns": [{"id": "P00", "method": "identity", "indices": []}],
                "video": {"patterns": ["P00", "missing"]},
            }
        )
    with pytest.raises(ConfigError, match="video.patterns"):
        config_from_mapping(
            {
                "patterns": [{"id": "P00", "method": "identity", "indices": []}],
                "video": {"patterns": ["P00", 7]},
            }
        )


def test_skip_keeps_validity_distinct_from_declared_out_of_scope() -> None:
    schema = _schema()
    source = np.array([0.2], dtype=np.float32)
    conditional = apply_intervention(
        source,
        InterventionPattern(
            "conditional",
            indices=(0,),
            method="reference",
            reference_id="missing",
            scope="explicitly_conditional",
            on_inapplicable="continue_unmodified_with_warning",
        ),
        schema,
        strict=False,
    )
    assert conditional.skipped is True
    assert conditional.eligible is True
    assert conditional.skip_category == "failed"
    out_of_scope = apply_intervention(
        source,
        InterventionPattern(
            "declared",
            indices=(0,),
            method="identity",
            metadata={"skip_reason": "not evaluated for this run"},
        ),
        schema,
        strict=False,
    )
    assert out_of_scope.skipped is True
    assert out_of_scope.eligible is False
    assert out_of_scope.skip_category == "declared_out_of_scope"
