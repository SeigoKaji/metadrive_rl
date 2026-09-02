"""Autograd IG tests independent from MetaDrive and a trained PPO archive."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from input_attribution.integrated_gradients import (
    compute_integrated_gradients,
    integrated_gradients,
    run_integrated_gradients,
)


class _ToyTensorAdapter:
    """Two-action linear actor and linear critic with trainable parameters."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.actor_weight = torch.nn.Parameter(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
        self.value_weight = torch.nn.Parameter(torch.tensor([3.0, -2.0]))

    def observations_to_tensor(self, observations: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(observations, dtype=torch.float32, device=self.device)

    def evaluate_tensors(self, observations: torch.Tensor) -> object:
        logits = observations @ self.actor_weight.T
        return SimpleNamespace(
            logits=logits,
            log_probabilities=torch.log_softmax(logits, dim=-1),
            probabilities=torch.softmax(logits, dim=-1),
            values=observations @ self.value_weight,
        )


def test_trapezoidal_ig_matches_linear_analytic_solution_and_completeness() -> None:
    weights = torch.tensor([2.0, -3.0])
    inputs = torch.tensor([[4.0, 1.0], [2.0, -1.0]])
    baseline = torch.tensor([[1.0, 5.0], [0.0, 2.0]])

    computation = compute_integrated_gradients(
        inputs, baseline, lambda values: values @ weights, steps=9
    )
    expected = (inputs - baseline) * weights

    torch.testing.assert_close(computation.attributions, expected)
    torch.testing.assert_close(
        computation.attribution_sum,
        computation.input_output - computation.baseline_output,
    )
    torch.testing.assert_close(
        integrated_gradients(inputs, baseline, lambda values: values @ weights, steps=2),
        expected,
    )


def test_relative_completeness_error_does_not_hide_nonzero_residual_when_delta_f_is_zero() -> None:
    # F(1) == F(0) == 0, but two-point trapezoidal integration of F'=3z²-1
    # yields 0.5 rather than the exact integral 0.  This must not be reported
    # as a perfect relative completeness error merely because ΔF is zero.
    computation = compute_integrated_gradients(
        torch.tensor([[1.0]]),
        torch.tensor([[0.0]]),
        lambda values: values[:, 0] ** 3 - values[:, 0],
        steps=2,
    )

    assert computation.input_output.item() == 0.0
    assert computation.baseline_output.item() == 0.0
    assert computation.completeness_residual.item() != 0.0
    assert torch.isfinite(computation.relative_completeness_error)
    assert computation.relative_completeness_error.item() > 0.0


def test_relative_completeness_error_remains_finite_for_float16() -> None:
    computation = compute_integrated_gradients(
        torch.tensor([[1.0]], dtype=torch.float16),
        torch.tensor([[0.0]], dtype=torch.float16),
        lambda values: 10.0 * (values[:, 0] ** 3 - values[:, 0]),
        steps=2,
    )

    assert computation.completeness_residual.item() != 0.0
    assert torch.isfinite(computation.relative_completeness_error)
    assert computation.relative_completeness_error.item() > 0.0


def test_actor_target_indices_stay_fixed_when_argmax_crosses_and_critic_is_exact() -> None:
    adapter = _ToyTensorAdapter()
    actor_before = adapter.actor_weight.detach().clone()
    value_before = adapter.value_weight.detach().clone()
    observations = np.array([[1.0, 2.0]], dtype=np.float32)
    # Along -1 -> +1, the policy argmax changes from action 1 to action 0.
    baselines = np.array([[[-1.0, 0.0]]], dtype=np.float32)
    result = run_integrated_gradients(
        adapter,
        observations,
        baselines,
        targets=("selected_log_probability", "selected_vs_runner_up_margin", "critic_value"),
        steps=9,
    )

    assert result.target_actions.tolist() == [0]
    assert result.runner_up_actions.tolist() == [1]
    # At the baseline, fixed a*=0 is intentionally evaluated although action
    # 1 is the baseline argmax.
    expected_selected_log_probability = torch.log_softmax(
        torch.tensor([[-1.0, 1.0]]), dim=-1
    )[0, 0].item()
    assert result.baseline_outputs[0, 0, 0] == expected_selected_log_probability
    # fixed margin is x0 - (-x0), so it changes from -2 to +2.
    assert result.input_outputs[0, 1, 0] - result.baseline_outputs[0, 1, 0] == 4.0
    np.testing.assert_allclose(result.attributions[0, 1, 0], [4.0, 0.0])
    # Critic V=3*x0-2*x1 yields exact linear IG.
    np.testing.assert_allclose(result.attributions[0, 2, 0], [6.0, -4.0])
    np.testing.assert_allclose(result.completeness_residuals, 0.0, atol=1e-6)
    assert adapter.actor_weight.grad is None
    assert adapter.value_weight.grad is None
    torch.testing.assert_close(adapter.actor_weight.detach(), actor_before)
    torch.testing.assert_close(adapter.value_weight.detach(), value_before)


def test_multiple_baselines_average_and_identical_input_baseline_is_zero() -> None:
    adapter = _ToyTensorAdapter()
    observations = np.array([[1.0, 2.0]], dtype=np.float32)
    baselines = np.array(
        [[[-1.0, 0.0], [1.0, 2.0]]], dtype=np.float32
    )
    result = run_integrated_gradients(
        adapter,
        observations,
        baselines,
        targets="critic_value",
        steps=4,
    )

    np.testing.assert_allclose(result.attributions[0, 0, 0], [6.0, -4.0])
    np.testing.assert_allclose(result.attributions[0, 0, 1], [0.0, 0.0])
    np.testing.assert_allclose(result.attributions_mean_over_baselines()[0, 0], [3.0, -2.0])
    np.testing.assert_allclose(result.relative_completeness_errors, 0.0, atol=1e-6)


def test_raw_float64_baseline_array_is_normalized_for_integrated_gradients() -> None:
    result = run_integrated_gradients(
        _ToyTensorAdapter(),
        np.array([[1.0, 2.0]], dtype=np.float32),
        np.array([[-1.0, 0.0]], dtype=np.float64),
        targets="critic_value",
        steps=4,
    )

    assert result.baseline_count == 1
    np.testing.assert_allclose(result.attributions[0, 0, 0], [6.0, -4.0])
