"""Regression contracts for source-confirmed invalid flag/value joints."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from input_attribution.interventions import (
    InterventionError,
    InterventionPattern,
    apply_intervention,
)
from input_attribution.schema import InputSchema, InputSpec


_TARGETS = (1, 3, 4)


def _replacement(value: float, invalid_value: object = 0.0, **extra: object) -> dict[str, object]:
    return {
        "kind": "fixed",
        "value": value,
        "invalid_value": invalid_value,
        "confirmed": True,
        "source": "synthetic invalid-state contract",
        **extra,
    }


def _invalid_schema(
    *,
    replacement_overrides: dict[int, dict[str, object]] | None = None,
    variants: bool = False,
    value_dtype: str = "continuous",
    variant_operation: str = "neutral",
    missing_provenance: tuple[int, ...] = (),
) -> InputSchema:
    overrides = replacement_overrides or {}
    variant_values = {1: 0.8, 3: 0.7, 4: 0.0}
    inputs: list[InputSpec] = []
    for index in range(5):
        input_source = "" if index in missing_provenance else "synthetic input contract"
        if index == 1:
            spec = InputSpec(
                index,
                "offset",
                "offset",
                group="lane",
                value_range=(0.0, 1.0),
                value_type=value_dtype,
                replacement=overrides.get(index, _replacement(0.5)),
                source=input_source,
                variants=(
                    {
                        "id": "invalid_joint",
                        "operation": variant_operation,
                        "value": variant_values[index],
                        "meaning": "synthetic invalid-joint offset",
                        "evidence": "synthetic invalid-state contract",
                        "confirmed": True,
                    },
                )
                if variants
                else (),
            )
        elif index == 3:
            spec = InputSpec(
                index,
                "heading",
                "heading",
                group="lane",
                value_range=(0.0, 1.0),
                value_type=value_dtype,
                replacement=overrides.get(index, _replacement(0.5)),
                source=input_source,
                variants=(
                    {
                        "id": "invalid_joint",
                        "operation": variant_operation,
                        "value": variant_values[index],
                        "meaning": "synthetic invalid-joint heading",
                        "evidence": "synthetic invalid-state contract",
                        "confirmed": True,
                    },
                )
                if variants
                else (),
            )
        elif index == 4:
            spec = InputSpec(
                index,
                "valid",
                "valid",
                group="lane",
                value_type="bool",
                value_range=(0.0, 1.0),
                independently_replaceable=False,
                coupled_indices=_TARGETS,
                replacement=overrides.get(index, _replacement(1.0)),
                source=input_source,
                variants=(
                    {
                        "id": "invalid_joint",
                        "operation": variant_operation,
                        "value": variant_values[index],
                        "meaning": "synthetic invalid-joint validity flag",
                        "evidence": "synthetic invalid-state contract",
                        "confirmed": True,
                    },
                )
                if variants
                else (),
            )
        else:
            spec = InputSpec(index, f"unused_{index}", f"unused_{index}", group="other")
        inputs.append(spec)
    return InputSchema(dimension=5, inputs=tuple(inputs))


def _pattern_metadata(value: object) -> dict[str, object]:
    if value is None:
        return {}
    return {"invalid_joint_confirmed": value}


@pytest.mark.parametrize("confirmed", [None, False, True])
def test_reference_rejects_inconsistent_joint_even_when_metadata_is_absent_or_false(
    confirmed: object,
) -> None:
    schema = _invalid_schema()
    source = np.array([0.0, 0.2, 0.0, 0.3, 1.0], dtype=np.float32)
    pattern = InterventionPattern(
        f"bad-reference-{confirmed}",
        indices=_TARGETS,
        method="reference",
        reference_id="bad",
        metadata=_pattern_metadata(confirmed),
    )
    with pytest.raises(InterventionError, match="invalid (sentinel|flag/value|source-confirmed)"):
        apply_intervention(
            source,
            pattern,
            schema,
            reference_observations={"bad": np.array([0.0, 0.8, 0.0, 0.7, 0.0], dtype=np.float32)},
        )


def test_source_confirmed_reference_joint_is_accepted_for_all_metadata_states() -> None:
    schema = _invalid_schema()
    source = np.array([0.0, 0.2, 0.0, 0.3, 1.0], dtype=np.float32)
    reference = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    for confirmed in (None, False, True):
        result = apply_intervention(
            source,
            InterventionPattern(
                f"good-reference-{confirmed}",
                indices=_TARGETS,
                method="reference",
                reference_id="good",
                metadata=_pattern_metadata(confirmed),
            ),
            schema,
            reference_observations={"good": reference},
        )
        assert result.skipped is False
        np.testing.assert_array_equal(result.observation[list(_TARGETS)], [0.0, 0.0, 0.0])


@pytest.mark.parametrize("confirmed", [None, False, True])
@pytest.mark.parametrize("method", ["neutral", "fixed_level"])
def test_typed_variant_cannot_bypass_invalid_joint_integrity(
    confirmed: object, method: str
) -> None:
    schema = _invalid_schema(variants=True, variant_operation=method)
    source = np.array([0.0, 0.2, 0.0, 0.3, 1.0], dtype=np.float32)
    with pytest.raises(InterventionError, match="invalid (sentinel|flag/value|source-confirmed)"):
        apply_intervention(
            source,
            InterventionPattern(
                f"bad-{method}-{confirmed}",
                indices=_TARGETS,
                method=method,
                variant_id="invalid_joint",
                metadata=_pattern_metadata(confirmed),
            ),
            schema,
        )


def test_reflection_keeps_discrete_invalid_flag_rejection() -> None:
    """Reflection cannot reach a flag/value joint through a discrete flag."""

    schema = _invalid_schema()
    source = np.array([0.0, 0.2, 0.0, 0.3, 1.0], dtype=np.float32)
    original = source.copy()
    pattern = InterventionPattern(
        "reflection-invalid-joint",
        indices=_TARGETS,
        method="reflection",
        variant={
            "id": "reflection-invalid-joint",
            "operation": "reflection",
            "center": 0.5,
            "meaning": "synthetic reflection attempt over a flag/value joint",
            "evidence": "synthetic invalid-state contract",
            "confirmed": True,
        },
    )
    with pytest.raises(InterventionError, match="discrete"):
        apply_intervention(source, pattern, schema)
    np.testing.assert_array_equal(source, original)


def test_legacy_fixed_accepts_only_the_source_confirmed_invalid_sentinels() -> None:
    schema = _invalid_schema()
    source = np.array([0.0, 0.2, 0.0, 0.3, 1.0], dtype=np.float32)
    result = apply_intervention(
        source,
        InterventionPattern(
            "good-fixed",
            indices=_TARGETS,
            method="fixed",
            values={1: 0.0, 3: 0.0, 4: 0.0},
        ),
        schema,
    )
    assert result.skipped is False
    np.testing.assert_array_equal(result.observation[list(_TARGETS)], [0.0, 0.0, 0.0])

    with pytest.raises(InterventionError, match="invalid sentinel|fixed value"):
        apply_intervention(
            source,
            InterventionPattern(
                "bad-fixed",
                indices=_TARGETS,
                method="fixed",
                values={1: 0.8, 3: 0.7, 4: 0.0},
            ),
            schema,
        )


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({1: {"kind": "fixed", "value": 0.5, "confirmed": True, "source": "only normal value"}}, "invalid_value"),
        ({3: {"kind": "fixed", "value": 0.5, "invalid_value": 0.0, "confirmed": False, "source": "unconfirmed"}}, "unconfirmed"),
        ({4: {"kind": "fixed", "value": 1.0, "invalid_value": 0.0, "confirmed": True}}, "evidence"),
    ],
)
def test_missing_unconfirmed_or_unproven_invalid_values_are_rejected(
    overrides: dict[int, dict[str, object]], expected: str
) -> None:
    schema = _invalid_schema(
        replacement_overrides=overrides,
        missing_provenance=(4,) if expected == "evidence" else (),
    )
    source = np.array([0.0, 0.2, 0.0, 0.3, 1.0], dtype=np.float32)
    with pytest.raises(InterventionError, match=expected):
        apply_intervention(
            source,
            InterventionPattern(
                f"bad-metadata-{expected}",
                indices=_TARGETS,
                method="reference",
                reference_id="bad",
            ),
            schema,
            reference_observations={"bad": np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)},
        )


def test_invalid_joint_is_checked_after_clip_and_dtype_conversion() -> None:
    schema = _invalid_schema()
    source = np.array([0.0, 0.2, 0.0, 0.3, 1.0], dtype=np.float32)
    result = apply_intervention(
        source,
        InterventionPattern(
            "clipped-invalid",
            indices=_TARGETS,
            method="fixed",
            values={1: -1.0, 3: -1.0, 4: -1.0},
            clip=True,
        ),
        schema,
    )
    assert result.skipped is False
    assert result.clipped_indices == (1, 3, 4)
    np.testing.assert_array_equal(result.observation[list(_TARGETS)], [0.0, 0.0, 0.0])

    dtype_schema = _invalid_schema(
        replacement_overrides={
            1: _replacement(0.5, invalid_value=0.1),
            3: _replacement(0.5, invalid_value=0.1),
            4: _replacement(1.0, invalid_value=0.0),
        }
    )
    with pytest.raises(InterventionError, match="invalid sentinel"):
        apply_intervention(
            source,
            InterventionPattern(
                "float32-invalid-mismatch",
                indices=_TARGETS,
                method="fixed",
                values={1: 0.1, 3: 0.1, 4: 0.0},
            ),
            dtype_schema,
        )


def test_explicit_member_evidence_allows_reference_joint_without_schema_sentinels() -> None:
    base = _invalid_schema()
    schema = replace(
        base,
        inputs=tuple(
            replace(spec, replacement=None) if spec.index in _TARGETS else spec
            for spec in base.inputs
        ),
    )
    evidence = {
        "offset": {"value": 0.0, "confirmed": True, "evidence": "offset source"},
        "heading": {"value": 0.0, "confirmed": True, "evidence": "heading source"},
        "valid": {"value": 0.0, "confirmed": True, "evidence": "validity source"},
    }
    source = np.array([0.0, 0.2, 0.0, 0.3, 1.0], dtype=np.float32)
    result = apply_intervention(
        source,
        InterventionPattern(
            "evidence-reference",
            indices=_TARGETS,
            method="reference",
            reference_id="good",
            metadata={"invalid_joint_evidence": evidence},
        ),
        schema,
        reference_observations={"good": np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)},
    )
    assert result.skipped is False
