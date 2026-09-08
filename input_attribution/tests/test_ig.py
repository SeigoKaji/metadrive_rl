"""Integrated Gradients contracts using a tiny differentiable policy."""

from __future__ import annotations

import numpy as np
import pytest

from input_attribution.integrated_gradients import (
    IntegratedGradientsError,
    compute_integrated_gradients,
)
from input_attribution.policy import ArrayPolicyAdapter, PolicyError
from input_attribution.schema import InputSchema, InputSpec


torch = pytest.importorskip("torch")


def _linear_policy() -> ArrayPolicyAdapter:
    weights = torch.tensor([[2.0, -3.0], [0.0, 0.0]])

    def probabilities(observations: np.ndarray) -> np.ndarray:
        values = torch.as_tensor(np.asarray(observations), dtype=torch.float32)
        logits = values[:, :2] @ weights.T
        return torch.softmax(logits, dim=-1).detach().numpy()

    def logits_tensor(observations):
        return observations[:, :2] @ weights.T

    return ArrayPolicyAdapter(probabilities, logits_fn=logits_tensor, fingerprint_value="linear")


def _schema() -> InputSchema:
    common = {
        "description": "synthetic test feature",
        "physical_quantity": "synthetic scalar",
        "model_representation": "raw scalar",
        "normalization": "none",
        "reference": "synthetic fixture",
        "source": "input_attribution/tests/test_ig.py",
    }
    return InputSchema(
        dimension=3,
        inputs=(
            InputSpec(0, "x", "x", group="continuous", **common),
            InputSpec(1, "y", "y", group="continuous", **common),
            InputSpec(
                2,
                "flag",
                "flag",
                group="flag",
                value_type="bool",
                ig_interpolation_allowed=False,
                **common,
            ),
        ),
    )


def test_linear_log_odds_integrated_gradients_and_completeness() -> None:
    policy = _linear_policy()
    result = compute_integrated_gradients(
        np.array([2.0, 1.0, 1.0]),
        np.array([1.0, 1.0, 1.0]),
        policy,
        schema=_schema(),
        target_action=0,
        backend="manual",
        n_steps=16,
    )
    # F = (2*x - 3*y) - 0, so only the x endpoint differs here.
    np.testing.assert_allclose(result.attributions, [2.0, 0.0, 0.0], atol=1e-6)
    assert result.absolute_completeness_error < 1e-6
    assert result.policy_unchanged is True
    assert result.status == "ok"
    assert result.completeness_status == "converged"
    assert result.retry_count == 0
    assert result.tolerance == 1e-4
    assert result.warnings == ()


def test_discrete_flag_cannot_be_interpolated_between_categories() -> None:
    with pytest.raises(IntegratedGradientsError, match="cannot be interpolated"):
        compute_integrated_gradients(
            np.array([1.0, 1.0, 1.0]),
            np.array([1.0, 1.0, 0.0]),
            _linear_policy(),
            schema=_schema(),
            target_action=0,
            backend="manual",
            n_steps=4,
        )


def test_ig_checks_reference_context_compatibility() -> None:
    with pytest.raises(IntegratedGradientsError, match="compatibility"):
        compute_integrated_gradients(
            np.array([2.0, 1.0, 1.0]),
            np.array([1.0, 1.0, 1.0]),
            _linear_policy(),
            schema=_schema(),
            target_action=0,
            backend="manual",
            n_steps=4,
            compatibility_keys=["road_segment_id"],
            observation_context={"road_segment_id": "A"},
            baseline_context={"road_segment_id": "B"},
        )


def test_ig_rejects_direct_policy_without_immutability_hooks() -> None:
    class MissingHooks:
        def probabilities(self, observations: np.ndarray) -> np.ndarray:
            del observations
            return np.array([[0.5, 0.5]])

        def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
            del observation, deterministic
            return 0

    with pytest.raises(PolicyError, match="fingerprint, and set_eval"):
        compute_integrated_gradients(
            np.array([1.0, 1.0, 1.0]),
            np.array([0.0, 1.0, 1.0]),
            MissingHooks(),
            schema=_schema(),
            target_action=0,
            backend="manual",
            n_steps=4,
        )


def test_target_action_stays_fixed_even_if_path_argmax_changes() -> None:
    policy = _linear_policy()
    # The baseline argmax is action 1, while the observation argmax is action 0.
    result = compute_integrated_gradients(
        np.array([2.0, 1.0, 1.0]),
        np.array([-1.0, 1.0, 1.0]),
        policy,
        schema=_schema(),
        target_action=0,
        backend="manual",
        n_steps=16,
    )
    assert result.target_action == 0
    assert result.attributions.shape == (3,)


def test_default_captum_backend_is_optional_and_manual_path_remains_available() -> None:
    policy = _linear_policy()
    try:
        result = compute_integrated_gradients(
            np.array([2.0, 1.0, 1.0]),
            np.array([1.0, 1.0, 1.0]),
            policy,
            schema=_schema(),
            target_action=0,
            backend="captum",
            n_steps=8,
        )
    except Exception as exc:
        # Captum is deliberately optional in the base environment.
        assert "Captum" in type(exc).__name__ or "Captum" in str(exc)
    else:
        assert result.backend == "captum"


def test_captum_nonlinear_known_integral() -> None:
    pytest.importorskip("captum")

    def probabilities(observations: np.ndarray) -> np.ndarray:
        values = torch.as_tensor(np.asarray(observations), dtype=torch.float32)
        logits = torch.stack((values[:, 0] ** 2 + 2.0 * values[:, 1], torch.zeros_like(values[:, 0])), dim=1)
        return torch.softmax(logits, dim=-1).detach().numpy()

    def logits_tensor(observations):
        return torch.stack(
            (observations[:, 0] ** 2 + 2.0 * observations[:, 1], torch.zeros_like(observations[:, 0])),
            dim=1,
        )

    policy = ArrayPolicyAdapter(
        probabilities,
        logits_fn=logits_tensor,
        fingerprint_value="nonlinear-captum",
    )
    result = compute_integrated_gradients(
        np.array([2.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 1.0]),
        policy,
        schema=_schema(),
        target_action=0,
        backend="captum",
        method="gausslegendre",
        n_steps=32,
        max_retries=0,
    )
    # F=x^2+2y; along x=1..2 the exact x attribution is integral(2x)dx=3.
    np.testing.assert_allclose(result.attributions, [3.0, 0.0, 0.0], atol=1e-5)
    assert result.retry_count == 0
    assert result.status == "ok"
    assert result.absolute_completeness_error < result.tolerance


def test_captum_keeps_original_action_when_path_argmax_crosses() -> None:
    pytest.importorskip("captum")
    policy = _linear_policy()
    observation = np.array([2.0, 1.0, 1.0])
    baseline = np.array([-1.0, 1.0, 1.0])
    assert int(np.argmax(policy.probabilities(observation)[0])) == 0
    assert int(np.argmax(policy.probabilities(baseline)[0])) == 1
    result = compute_integrated_gradients(
        observation,
        baseline,
        policy,
        schema=_schema(),
        target_action=0,
        backend="captum",
        method="gausslegendre",
        n_steps=32,
        max_retries=0,
    )
    assert result.target_action == 0
    assert result.retry_count == 0
    assert result.absolute_completeness_error < result.tolerance
