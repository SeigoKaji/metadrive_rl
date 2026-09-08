"""Low-cost core contracts independent of MetaDrive."""

from __future__ import annotations

import numpy as np
import pytest

from input_attribution.adapters.fixtures import (
    assert_schema_resolved,
    custom_262_schema_template,
    official_schema_259,
)
from input_attribution.interventions import (
    InterventionError,
    InterventionPattern,
    apply_intervention,
)
from input_attribution.metrics import compare_distributions, js_divergence
from input_attribution.offline import analyze_offline
from input_attribution.policy import CategoricalPolicyAdapter, PolicyError
from input_attribution.schema import InputSchema, InputSpec, SchemaError


def _small_schema() -> InputSchema:
    return InputSchema(
        dimension=4,
        schema_id="test",
        inputs=(
            InputSpec(0, "a", "A", group="continuous"),
            InputSpec(1, "valid", "有効", group="flags", value_type="bool", independently_replaceable=False, coupled_indices=(1, 2), ig_interpolation_allowed=False),
            InputSpec(2, "offset", "ずれ", group="lane", independently_replaceable=True),
            InputSpec(3, "b", "B", group="continuous"),
        ),
    )


def test_official_and_custom_schema_dimensions_are_explicit() -> None:
    official = official_schema_259()
    assert official.dimension == 259
    assert tuple(item.index for item in official.inputs) == tuple(range(259))

    custom = custom_262_schema_template(custom_indices=(1, 145, 260))
    assert custom.dimension == 262
    assert {item.index for item in custom.inputs} == set(range(262))
    assert {item.index for item in custom.inputs if item.id.startswith("UNRESOLVED_")} == {1, 145, 260}
    with pytest.raises(SchemaError):
        assert_schema_resolved(custom)


def test_non_tail_indices_round_trip_and_observation_is_copied() -> None:
    schema = _small_schema()
    restored = InputSchema.from_dict(schema.to_dict())
    source = np.array([0.1, 1.0, 0.2, 0.3], dtype=np.float32)
    checked = restored.validate_observation(source)
    checked[0] = 9.0
    assert source[0] == 0.1
    assert restored.spec("offset").index == 2


@pytest.mark.parametrize("value", ["false", "true", None, 0, 1, np.bool_(True)])
def test_schema_flags_require_actual_booleans(value: object) -> None:
    row = {
        "index": 0,
        "id": "x",
        "name_ja": "x",
        "group": "x",
        "independently_replaceable": value,
    }
    with pytest.raises(SchemaError, match="must be a boolean"):
        InputSpec.from_dict(row)


def test_execution_validation_rejects_nested_unknown_reference() -> None:
    row = {
        "index": 0,
        "id": "x",
        "name_ja": "x",
        "group": "x",
        "description": "synthetic scalar",
        "physical_quantity": "synthetic value",
        "model_representation": "raw scalar",
        "normalization": "none",
        "reference": "synthetic fixture",
        "source": "input_attribution/tests/test_core.py",
        "replacement": {
            "kind": "fixed",
            "metadata": {"reference": {"source": "unknown"}},
        },
    }
    schema = InputSchema.from_dict({"dimension": 1, "inputs": [row]})
    with pytest.raises(SchemaError, match="unresolved"):
        schema.validate_for_execution()


def test_intervention_rejects_flag_only_and_keeps_source_unchanged() -> None:
    schema = _small_schema()
    source = np.array([0.1, 1.0, 0.2, 0.3], dtype=np.float32)
    pattern = InterventionPattern("flag-only", indices=(1,), values={1: 0.0})
    with pytest.raises(InterventionError):
        apply_intervention(source, pattern, schema)
    skipped = apply_intervention(source, pattern, schema, strict=False)
    assert skipped.skipped is True
    assert skipped.skip_reason
    np.testing.assert_array_equal(source, np.array([0.1, 1.0, 0.2, 0.3], dtype=np.float32))


def test_reference_replacement_uses_one_saved_vector_and_tracks_noop() -> None:
    schema = _small_schema()
    source = np.array([0.1, 1.0, 0.2, 0.3], dtype=np.float32)
    reference = np.array([0.8, 1.0, 0.9, 0.4], dtype=np.float32)
    result = apply_intervention(
        source,
        InterventionPattern("ref", indices=(0, 2), method="reference", reference_id="r1"),
        schema,
        reference_observations={"r1": reference},
    )
    np.testing.assert_allclose(result.observation, [0.8, 1.0, 0.9, 0.3])
    assert result.changed_indices == (0, 2)
    assert result.no_op_indices == ()
    np.testing.assert_allclose(source, [0.1, 1.0, 0.2, 0.3])


def test_independent_offset_is_allowed_only_while_related_flag_is_valid() -> None:
    schema = _small_schema()
    valid_source = np.array([0.1, 1.0, 0.2, 0.3], dtype=np.float32)
    result = apply_intervention(
        valid_source,
        InterventionPattern("offset", indices=(2,), values={2: 0.4}),
        schema,
    )
    assert result.changed_indices == (2,)

    invalid_source = np.array([0.1, 0.0, 0.2, 0.3], dtype=np.float32)
    skipped = apply_intervention(
        invalid_source,
        InterventionPattern("offset-invalid", indices=(2,), values={2: 0.4}),
        schema,
        strict=False,
    )
    assert skipped.skipped is True
    assert "valid flag" in (skipped.skip_reason or "")


def test_missing_or_invalid_reference_is_skipped_in_non_strict_mode() -> None:
    schema = _small_schema()
    source = np.array([0.1, 1.0, 0.2, 0.3], dtype=np.float32)
    pattern = InterventionPattern("missing", indices=(0,), method="reference", reference_id="r")
    skipped = apply_intervention(source, pattern, schema, strict=False)
    assert skipped.skipped is True
    assert "saved reference" in (skipped.skip_reason or "")
    bad = apply_intervention(
        source,
        pattern,
        schema,
        reference_observations={"r": np.array([1.0, 2.0])},
        strict=False,
    )
    assert bad.skipped is True
    assert "reference observation is invalid" in (bad.skip_reason or "")


def test_reference_context_and_preconditions_fail_closed() -> None:
    schema = _small_schema()
    source = np.array([0.1, 1.0, 0.2, 0.3], dtype=np.float32)
    pattern = InterventionPattern(
        "compatible",
        indices=(0,),
        method="reference",
        reference_id="r1",
        metadata={
            "compatibility_keys": ["road_segment_id", "target_lane_ordinal"],
            "preconditions": {"valid": 1.0},
        },
    )
    with pytest.raises(InterventionError, match="compatibility"):
        apply_intervention(
            source,
            pattern,
            schema,
            reference_observations={"r1": source},
            reference_contexts={"r1": {"road_segment_id": "A", "target_lane_ordinal": 1}},
            observation_context={"road_segment_id": "B", "target_lane_ordinal": 1},
        )
    result = apply_intervention(
        source,
        pattern,
        schema,
        reference_observations={"r1": source},
        reference_contexts={"r1": {"road_segment_id": "A", "target_lane_ordinal": 1}},
        observation_context={"road_segment_id": "A", "target_lane_ordinal": 1},
    )
    assert result.skipped is False


def test_invalid_flag_group_requires_explicit_joint_confirmation() -> None:
    schema = _small_schema()
    source = np.array([0.1, 1.0, 0.2, 0.3], dtype=np.float32)
    pattern = InterventionPattern("invalid", indices=(1, 2), values={1: 0.0, 2: 0.0})
    with pytest.raises(InterventionError, match="invalid flag/value group"):
        apply_intervention(source, pattern, schema)
    with pytest.raises(InterventionError, match="source-confirmed invalid_value"):
        apply_intervention(
            source,
            InterventionPattern(
                "invalid-confirmed-without-evidence",
                indices=(1, 2),
                values={1: 0.0, 2: 0.0},
                metadata={"invalid_joint_confirmed": True},
            ),
            schema,
        )
    result = apply_intervention(
        source,
        InterventionPattern(
            "invalid-confirmed",
            indices=(1, 2),
            values={1: 0.0, 2: 0.0},
            metadata={
                "invalid_joint_confirmed": True,
                "invalid_joint_evidence": {
                    "valid": {
                        "value": 0.0,
                        "confirmed": True,
                        "evidence": "synthetic validity source",
                    },
                    "offset": {
                        "value": 0.0,
                        "confirmed": True,
                        "evidence": "synthetic offset source",
                    },
                },
            },
        ),
        schema,
    )
    assert result.changed_indices == (1, 2)


def _variant_schema(
    *,
    unconfirmed_offset: bool = False,
    missing_invalid: bool = False,
) -> InputSchema:
    def replacement(neutral: float, invalid: float, *, confirmed: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "kind": "fixed",
            "value": neutral,
            "confirmed": confirmed,
        }
        if not missing_invalid:
            result["invalid_value"] = invalid
        return result

    return InputSchema(
        dimension=3,
        inputs=(
            InputSpec(
                0,
                "offset",
                "offset",
                group="lane",
                description="synthetic offset",
                physical_quantity="offset",
                model_representation="normalized scalar",
                normalization="none",
                reference="synthetic lane",
                source="input_attribution/tests/test_core.py",
                replacement=replacement(0.5, 0.0, confirmed=not unconfirmed_offset),
            ),
            InputSpec(
                1,
                "heading",
                "heading",
                group="lane",
                description="synthetic heading",
                physical_quantity="heading",
                model_representation="normalized scalar",
                normalization="none",
                reference="synthetic lane",
                source="input_attribution/tests/test_core.py",
                replacement=replacement(0.5, 0.0),
            ),
            InputSpec(
                2,
                "valid",
                "valid",
                group="lane",
                value_type="bool",
                independently_replaceable=False,
                coupled_indices=(0, 1, 2),
                description="synthetic validity flag",
                physical_quantity="validity",
                model_representation="0/1",
                normalization="none",
                reference="synthetic lane",
                source="input_attribution/tests/test_core.py",
                replacement=replacement(1.0, 0.0),
            ),
        ),
    )


def test_neutral_and_source_confirmed_invalid_variants_are_distinct() -> None:
    schema = _variant_schema()
    source = np.array([0.2, 0.3, 1.0], dtype=np.float32)
    neutral = apply_intervention(
        source,
        InterventionPattern("neutral-offset", indices=(0,), values={0: 0.5}),
        schema,
    )
    assert neutral.changed_indices == (0,)

    invalid = apply_intervention(
        source,
        InterventionPattern(
            "invalid-triplet",
            indices=(0, 1, 2),
            values={0: 0.0, 1: 0.0, 2: 0.0},
            metadata={"invalid_joint_confirmed": True},
        ),
        schema,
    )
    assert invalid.changed_indices == (0, 1, 2)

    with pytest.raises(InterventionError, match="sentinel mismatch"):
        apply_intervention(
            source,
            InterventionPattern(
                "wrong-invalid",
                indices=(0, 1, 2),
                values={0: 0.25, 1: 0.0, 2: 0.0},
                metadata={"invalid_joint_confirmed": True},
            ),
            schema,
        )
    with pytest.raises(InterventionError, match="unconfirmed"):
        apply_intervention(
            source,
            InterventionPattern(
                "unconfirmed-invalid",
                indices=(0, 1, 2),
                values={0: 0.0, 1: 0.0, 2: 0.0},
                metadata={"invalid_joint_confirmed": True},
            ),
            _variant_schema(unconfirmed_offset=True),
        )
    with pytest.raises(InterventionError, match="invalid_value"):
        apply_intervention(
            source,
            InterventionPattern(
                "unknown-invalid",
                indices=(0, 1, 2),
                values={0: 0.0, 1: 0.0, 2: 0.0},
                metadata={"invalid_joint_confirmed": True},
            ),
            _variant_schema(missing_invalid=True),
        )


def test_probability_metrics_use_original_selected_action_and_natural_log_js() -> None:
    p = np.array([[1.0, 0.0], [0.25, 0.75]])
    q = np.array([[1.0, 0.0], [0.75, 0.25]])
    comparison = compare_distributions(p, q)
    assert comparison.selected_actions.tolist() == [0, 1]
    assert comparison.action_changed.tolist() == [False, True]
    assert 0.0 <= comparison.js[1] <= np.log(2.0)
    assert js_divergence(p[0], q[0]) == 0.0


def test_offline_starts_each_pattern_from_original_observation() -> None:
    schema = InputSchema(
        dimension=2,
        inputs=(
            InputSpec(
                0,
                "x",
                "x",
                group="x",
                description="synthetic scalar",
                physical_quantity="synthetic value",
                model_representation="raw scalar",
                normalization="none",
                reference="synthetic fixture",
                source="input_attribution/tests/test_core.py",
            ),
            InputSpec(
                1,
                "y",
                "y",
                group="y",
                description="synthetic scalar",
                physical_quantity="synthetic value",
                model_representation="raw scalar",
                normalization="none",
                reference="synthetic fixture",
                source="input_attribution/tests/test_core.py",
            ),
        ),
    )

    class Policy:
        eval_calls = 0

        def probabilities(self, observations: np.ndarray) -> np.ndarray:
            values = np.asarray(observations)[:, 0]
            logits = np.stack([values, -values], axis=1)
            logits -= logits.max(axis=1, keepdims=True)
            result = np.exp(logits)
            return result / result.sum(axis=1, keepdims=True)

        def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
            del deterministic
            return int(np.argmax(self.probabilities(np.asarray(observation))))

        def fingerprint(self) -> str:
            return "test-policy"

        def set_eval(self) -> None:
            self.eval_calls += 1

    source = np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    original_copy = source.copy()
    result = analyze_offline(
        source,
        Policy(),
        schema,
        [
            InterventionPattern("P00", method="identity"),
            InterventionPattern("x-zero", indices=(0,), values={0: 0.0}),
        ],
    )
    np.testing.assert_array_equal(source, original_copy)
    changed = result.patterns[1]
    assert changed.changed_indices_by_row == ((0,), (0,))
    assert changed.summary["actual_changed_row_count"] == 2
    assert result.metadata["policy_fingerprint_before"] == "test-policy"
    assert result.metadata["policy_fingerprint_after"] == "test-policy"
    assert result.metadata["policy_unchanged"] is True


def test_offline_reports_confirmed_action_marginal_changes_in_percentage_points() -> None:
    schema = InputSchema(
        dimension=1,
        inputs=(
            InputSpec(
                0,
                "x",
                "x",
                group="x",
                description="synthetic scalar",
                physical_quantity="synthetic value",
                model_representation="raw scalar",
                normalization="none",
                reference="synthetic fixture",
                source="input_attribution/tests/test_core.py",
            ),
        ),
    )

    class Policy:
        def probabilities(self, observations: np.ndarray) -> np.ndarray:
            values = np.asarray(observations)[:, 0]
            logits = np.stack([values, np.zeros_like(values), -values], axis=1)
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            return probabilities / probabilities.sum(axis=1, keepdims=True)

        def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
            del deterministic
            return int(np.argmax(self.probabilities(np.asarray(observation))))

        def fingerprint(self) -> str:
            return "marginal-policy"

        def set_eval(self) -> None:
            return None

    result = analyze_offline(
        np.array([[1.0]], dtype=np.float32),
        Policy(),
        schema,
        [InterventionPattern("x-zero", indices=(0,), values={0: 0.0})],
        action_mapping={"steering": {"left": [0], "straight": [1], "right": [2]}},
    )
    row = result.patterns[0].rows[0]
    assert set(row["action_marginal_probability_delta_pp"]["steering"]) == {"left", "straight", "right"}


def test_direct_policy_requires_fingerprint_and_eval_hooks() -> None:
    class MissingHooks:
        def probabilities(self, observations: np.ndarray) -> np.ndarray:
            del observations
            return np.array([[0.5, 0.5]])

        def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
            del observation, deterministic
            return 0

    schema = InputSchema(
        dimension=1,
        inputs=(
            InputSpec(
                0,
                "x",
                "x",
                group="x",
                description="synthetic scalar",
                physical_quantity="synthetic value",
                model_representation="raw scalar",
                normalization="none",
                reference="synthetic fixture",
                source="input_attribution/tests/test_core.py",
            ),
        ),
    )
    with pytest.raises(PolicyError, match="fingerprint, and set_eval"):
        analyze_offline(
            np.array([[0.0]], dtype=np.float32),
            MissingHooks(),
            schema,
            [InterventionPattern("P00", method="identity")],
        )


def test_categorical_probability_batch_reuses_collection_single_row_route() -> None:
    torch = pytest.importorskip("torch")

    class Box:
        shape = (2,)
        dtype = np.dtype(np.float32)

    class Discrete:
        n = 2
        start = 0

    class MLP:
        def forward_actor(self, features):
            return features

    class Policy:
        observation_space = Box()
        action_space = Discrete()
        device = torch.device("cpu")
        pi_features_extractor = None
        mlp_extractor = MLP()
        action_net = torch.nn.Linear(2, 2, bias=False)
        action_dist = None

        def __init__(self) -> None:
            self.batch_shapes: list[tuple[int, ...]] = []
            with torch.no_grad():
                self.action_net.weight.copy_(torch.tensor([[2.0, -1.0], [-1.0, 1.0]]))

        def obs_to_tensor(self, observations):
            return torch.as_tensor(observations, dtype=torch.float32), True

        def extract_features(self, tensor, extractor):
            del extractor
            self.batch_shapes.append(tuple(tensor.shape))
            return tensor

    policy = Policy()
    adapter = CategoricalPolicyAdapter(policy)
    observations = np.asarray([[0.1, 0.2], [0.3, -0.4], [-0.5, 0.6]], dtype=np.float32)
    batched = adapter.probabilities(observations)
    assert policy.batch_shapes == [(1, 2), (1, 2), (1, 2)]
    single = np.vstack([adapter.probabilities(row)[0] for row in observations])
    np.testing.assert_array_equal(batched, single)
