"""Narrow Stable-Baselines3 PPO policy access for attribution calculations.

Only this module knows about SB3 policy internals.  It intentionally supports
the initial analysis contract only: PPO ``MlpPolicy``/``ActorCriticPolicy``, a
flat ``Box`` observation, and one ``Discrete`` action distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import FlattenExtractor


class PolicyAdapterError(ValueError):
    """Raised when a loaded SB3 policy is outside the supported contract."""


@dataclass(frozen=True, slots=True)
class TensorPolicyEvaluation:
    """Differentiable policy outputs for a ``(batch, observation_dim)`` tensor."""

    logits: torch.Tensor
    log_probabilities: torch.Tensor
    probabilities: torch.Tensor
    deterministic_actions: torch.Tensor
    values: torch.Tensor

    @property
    def actions(self) -> torch.Tensor:
        """Alias for deterministic action IDs."""

        return self.deterministic_actions

    @property
    def log_probs(self) -> torch.Tensor:
        """Alias retained for concise analysis code."""

        return self.log_probabilities


@dataclass(frozen=True, slots=True)
class PolicyEvaluation:
    """Detached NumPy policy outputs for ordinary/offline evaluation."""

    logits: np.ndarray
    log_probabilities: np.ndarray
    probabilities: np.ndarray
    deterministic_actions: np.ndarray
    values: np.ndarray

    @property
    def actions(self) -> np.ndarray:
        """Alias for deterministic action IDs."""

        return self.deterministic_actions

    @property
    def deterministic_action(self) -> int:
        """Return the only action for a single-observation evaluation."""

        if self.deterministic_actions.size != 1:
            raise PolicyAdapterError("deterministic_actionはbatch size 1の場合だけ利用できます")
        return int(self.deterministic_actions.item())

    @property
    def value(self) -> float:
        """Return the only value for a single-observation evaluation."""

        if self.values.size != 1:
            raise PolicyAdapterError("valueはbatch size 1の場合だけ利用できます")
        return float(self.values.item())

    @property
    def log_probs(self) -> np.ndarray:
        """Alias retained for concise analysis code."""

        return self.log_probabilities


class SB3PolicyAdapter:
    """Expose logits, probabilities, selected actions, and values from PPO.

    ``evaluate`` is intentionally no-grad and returns NumPy arrays for rollout
    collection and perturbation.  ``evaluate_tensors`` preserves the input
    computation graph and is the only API Integrated Gradients should use.
    Neither API calls an optimizer or writes parameter ``.grad`` fields.
    """

    def __init__(self, model_or_policy: PPO | ActorCriticPolicy, device: str | torch.device | None = None) -> None:
        self.model: PPO | None
        if isinstance(model_or_policy, PPO):
            self.model = model_or_policy
            policy = model_or_policy.policy
        elif isinstance(model_or_policy, ActorCriticPolicy):
            self.model = None
            policy = model_or_policy
        else:
            raise PolicyAdapterError(
                "SB3 PPOまたはActorCriticPolicyだけを解析できます: "
                f"{type(model_or_policy).__name__}"
            )
        self.policy = policy
        self._validate_policy(policy)
        inferred_device = getattr(policy, "device", torch.device("cpu"))
        self.device = torch.device(device) if device is not None else torch.device(inferred_device)
        self.observation_dim = int(cast(tuple[int, ...], policy.observation_space.shape)[0])
        self.action_count = int(cast(spaces.Discrete, policy.action_space).n)
        self.n_actions = self.action_count
        # PPO.load() leaves policies in training mode.  Attribution is an
        # evaluation operation; turning off training only changes module mode,
        # never weight values or optimizer state.
        policy.set_training_mode(False)

    @staticmethod
    def _validate_policy(policy: ActorCriticPolicy) -> None:
        if not isinstance(policy, ActorCriticPolicy):
            raise PolicyAdapterError("ActorCriticPolicy系MlpPolicyだけを解析できます")
        observation_space = policy.observation_space
        if not isinstance(observation_space, spaces.Box):
            raise PolicyAdapterError("flatなgymnasium.spaces.Box観測だけを解析できます")
        if len(observation_space.shape) != 1 or observation_space.shape[0] <= 0:
            raise PolicyAdapterError(
                "観測spaceはshape=(D,)の1次元flat Boxである必要があります: "
                f"{observation_space.shape}"
            )
        if type(policy.features_extractor) is not FlattenExtractor:
            raise PolicyAdapterError(
                "MlpPolicyのFlattenExtractorだけを解析できます: "
                f"{type(policy.features_extractor).__name__}"
            )
        if not isinstance(policy.action_space, spaces.Discrete):
            raise PolicyAdapterError("single Discrete actionだけを解析できます")
        if policy.action_space.n <= 0:
            raise PolicyAdapterError("Discrete action数は1以上である必要があります")
        if not isinstance(policy.action_dist, CategoricalDistribution):
            raise PolicyAdapterError(
                "CategoricalDistributionによるsingle Discrete actionだけを解析できます: "
                f"{type(policy.action_dist).__name__}"
            )
        if not callable(getattr(policy, "extract_features", None)):
            raise PolicyAdapterError("policy.extract_featuresが利用できません")
        if not callable(getattr(policy, "action_net", None)):
            raise PolicyAdapterError("policy.action_netが利用できません")
        if not callable(getattr(policy, "predict_values", None)):
            raise PolicyAdapterError("policy.predict_valuesが利用できません")

    def observations_to_tensor(self, observations: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Normalize an observation or batch to float32 ``(B, D)`` on device."""

        if isinstance(observations, torch.Tensor):
            tensor = observations.to(device=self.device, dtype=torch.float32)
        else:
            array = np.asarray(observations)
            if not np.issubdtype(array.dtype, np.number):
                raise PolicyAdapterError("observationsは数値配列で指定してください")
            tensor = torch.as_tensor(array, dtype=torch.float32, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[1] != self.observation_dim:
            raise PolicyAdapterError(
                "observationsは(B, D)または(D,)で指定してください: "
                f"expected D={self.observation_dim}, got {tuple(tensor.shape)}"
            )
        if tensor.shape[0] == 0:
            raise PolicyAdapterError("空のobservation batchは解析できません")
        if not torch.isfinite(tensor).all():
            raise PolicyAdapterError("observationsには有限値だけを指定してください")
        return tensor

    # A compact alias is useful in external scripts and tests.
    to_tensor = observations_to_tensor

    def _raw_logits(self, observations: torch.Tensor) -> torch.Tensor:
        """Follow SB3's MlpPolicy actor path without using normalized logits."""

        features = self.policy.extract_features(observations)
        if self.policy.share_features_extractor:
            latent_pi, _latent_vf = self.policy.mlp_extractor(features)
        else:
            # SB3 returns one tensor per feature extractor when actor/critic
            # extractors are separate.  Keeping the check explicit avoids a
            # silent unpack/broadcast error on an unsupported policy variant.
            if not isinstance(features, tuple) or len(features) != 2:
                raise PolicyAdapterError("separate feature extractorの出力形式が未対応です")
            latent_pi = self.policy.mlp_extractor.forward_actor(features[0])
        logits = self.policy.action_net(latent_pi)
        if logits.ndim != 2 or logits.shape != (observations.shape[0], self.action_count):
            raise PolicyAdapterError(
                "action_netのlogits shapeが不正です: "
                f"expected {(observations.shape[0], self.action_count)}, got {tuple(logits.shape)}"
            )
        return logits

    def evaluate_tensors(self, observations: torch.Tensor) -> TensorPolicyEvaluation:
        """Evaluate tensors while retaining gradients with respect to inputs.

        The caller supplies a leaf tensor with ``requires_grad=True`` for
        Integrated Gradients.  This method uses no ``torch.no_grad`` context
        and does not call ``backward``; therefore parameter gradients are not
        accumulated.
        """

        tensor = self.observations_to_tensor(observations)
        logits = self._raw_logits(tensor)
        log_probabilities = torch.log_softmax(logits, dim=-1)
        probabilities = torch.softmax(logits, dim=-1)
        actions = torch.argmax(logits, dim=-1).to(dtype=torch.long)
        values = self.policy.predict_values(tensor)
        if values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]
        elif values.ndim != 1:
            raise PolicyAdapterError(
                "critic valueのshapeが不正です: "
                f"expected {(tensor.shape[0],)} or {(tensor.shape[0], 1)}, got {tuple(values.shape)}"
            )
        if values.shape[0] != tensor.shape[0]:
            raise PolicyAdapterError("critic valueのbatch sizeがactor出力と一致しません")
        return TensorPolicyEvaluation(
            logits=logits,
            log_probabilities=log_probabilities,
            probabilities=probabilities,
            deterministic_actions=actions,
            values=values,
        )

    # Singular spelling is a natural compatibility alias.
    evaluate_tensor = evaluate_tensors

    def evaluate(self, observations: np.ndarray | torch.Tensor) -> PolicyEvaluation:
        """Evaluate one observation or a batch without retaining a graph."""

        tensor = self.observations_to_tensor(observations)
        with torch.no_grad():
            output = self.evaluate_tensors(tensor)
        return PolicyEvaluation(
            logits=output.logits.detach().cpu().numpy().copy(),
            log_probabilities=output.log_probabilities.detach().cpu().numpy().copy(),
            probabilities=output.probabilities.detach().cpu().numpy().copy(),
            deterministic_actions=output.deterministic_actions.detach().cpu().numpy().copy(),
            values=output.values.detach().cpu().numpy().copy(),
        )

    def distribution_probabilities(self, observations: np.ndarray | torch.Tensor) -> np.ndarray:
        """Return categorical probabilities via SB3's public distribution API."""

        tensor = self.observations_to_tensor(observations)
        with torch.no_grad():
            distribution = self.policy.get_distribution(tensor)
            categorical = getattr(distribution, "distribution", None)
            probabilities = getattr(categorical, "probs", None)
            if probabilities is None:
                raise PolicyAdapterError("SB3 distributionからCategorical probabilitiesを取得できません")
            if probabilities.ndim != 2 or probabilities.shape[1] != self.action_count:
                raise PolicyAdapterError("SB3 distribution probabilitiesのshapeが不正です")
            return probabilities.detach().cpu().numpy().copy()

    get_distribution_probabilities = distribution_probabilities

    def assert_distribution_matches(
        self,
        observations: np.ndarray | torch.Tensor,
        *,
        rtol: float = 1e-5,
        atol: float = 1e-6,
    ) -> None:
        """Verify raw-logit softmax agrees with SB3's distribution wrapper."""

        calculated = self.evaluate(observations).probabilities
        distribution = self.distribution_probabilities(observations)
        if not np.allclose(calculated, distribution, rtol=rtol, atol=atol):
            maximum = float(np.max(np.abs(calculated - distribution)))
            raise PolicyAdapterError(
                "raw logitsのsoftmaxとSB3 distribution probabilitiesが一致しません "
                f"(max_abs_difference={maximum:.3g})"
            )


# A less SB3-specific spelling makes call sites clearer while preserving the
# module's single implementation.
PolicyAdapter = SB3PolicyAdapter
