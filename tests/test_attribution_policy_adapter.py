"""SB3 PPO adapter tests using a tiny Gymnasium environment only."""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from input_attribution.sb3_adapter import PolicyAdapterError, SB3PolicyAdapter


class _TinyDiscreteEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

    def reset(self, *, seed: int | None = None, options: dict[str, object] | None = None) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        assert self.action_space.contains(action)
        return np.zeros(4, dtype=np.float32), 0.0, True, False, {}


def _model() -> PPO:
    return PPO(
        "MlpPolicy",
        _TinyDiscreteEnv(),
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        policy_kwargs={"net_arch": [8]},
        seed=3,
        device="cpu",
    )


def test_adapter_exposes_batch_logits_probabilities_actions_and_values() -> None:
    adapter = SB3PolicyAdapter(_model())
    observations = np.array(
        [[0.1, -0.2, 0.3, 0.4], [-0.4, 0.2, 0.0, 0.5]], dtype=np.float32
    )

    output = adapter.evaluate(observations)

    assert adapter.observation_dim == 4
    assert adapter.action_count == 3
    assert output.logits.shape == (2, 3)
    assert output.log_probabilities.shape == (2, 3)
    assert output.probabilities.shape == (2, 3)
    assert output.values.shape == (2,)
    np.testing.assert_allclose(output.probabilities.sum(axis=1), 1.0)
    np.testing.assert_array_equal(output.deterministic_actions, np.argmax(output.logits, axis=1))
    np.testing.assert_allclose(
        output.probabilities,
        np.exp(output.log_probabilities),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        output.probabilities,
        adapter.distribution_probabilities(observations),
        rtol=1e-6,
        atol=1e-6,
    )
    adapter.assert_distribution_matches(observations)


def test_tensor_api_preserves_input_gradient_without_populating_parameter_grads() -> None:
    adapter = SB3PolicyAdapter(_model())
    input_tensor = torch.tensor([[0.2, 0.1, -0.3, 0.4]], requires_grad=True)

    output = adapter.evaluate_tensors(input_tensor)
    gradient = torch.autograd.grad(output.log_probabilities[:, 0].sum(), input_tensor)[0]

    assert gradient.shape == input_tensor.shape
    assert all(parameter.grad is None for parameter in adapter.policy.parameters())


def test_adapter_rejects_non_flat_or_non_discrete_policy_contract() -> None:
    class _BadEnv(_TinyDiscreteEnv):
        def __init__(self) -> None:
            super().__init__()
            self.observation_space = spaces.Box(-1.0, 1.0, shape=(2, 2), dtype=np.float32)

    bad_model = PPO("MlpPolicy", _BadEnv(), n_steps=2, batch_size=2, n_epochs=1)
    with pytest.raises(PolicyAdapterError, match="flat Box"):
        SB3PolicyAdapter(bad_model)
