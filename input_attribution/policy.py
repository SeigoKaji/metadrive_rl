"""Small policy adapters shared by offline analysis and Integrated Gradients.

The core only needs a categorical distribution.  The SB3 adapter is kept here
so the rest of the package can run with a fake policy in environments where
MetaDrive or Stable-Baselines3 is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np


class PolicyError(ValueError):
    """Raised when a policy cannot provide a valid categorical output."""


@runtime_checkable
class PolicyLike(Protocol):
    """Protocol consumed by the analysis core.

    ``logits_tensor`` is optional at runtime for ①-A/①-B and required only by
    Integrated Gradients.  It must preserve autograd when given a tensor with
    ``requires_grad=True``.
    """

    def probabilities(self, observations: Any) -> np.ndarray: ...

    def predict(self, observation: Any, deterministic: bool = True) -> Any: ...

    def fingerprint(self) -> str: ...


def _validate_probabilities(value: Any, *, expected_batch: int | None = None) -> np.ndarray:
    probabilities = np.asarray(value, dtype=np.float64)
    if probabilities.ndim == 1:
        probabilities = probabilities[None, :]
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise PolicyError("categorical probabilities must have shape (batch, actions>=2)")
    if expected_batch is not None and probabilities.shape[0] != expected_batch:
        raise PolicyError(
            f"policy returned {probabilities.shape[0]} rows for {expected_batch} observations"
        )
    if not np.all(np.isfinite(probabilities)):
        raise PolicyError("policy probabilities contain non-finite values")
    if np.any(probabilities < -1e-10):
        raise PolicyError("policy probabilities contain negative values")
    probabilities = np.maximum(probabilities, 0.0)
    sums = probabilities.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-6, rtol=1e-6):
        raise PolicyError(f"policy probabilities must sum to one, got {sums.tolist()}")
    # Correct only tiny floating point residue.  A materially malformed
    # distribution was rejected above and is never silently normalized.
    probabilities /= sums[:, None]
    return probabilities


def _torch_module() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise PolicyError("PyTorch is required for logits_tensor/IG") from exc
    return torch


def _one_batch(observations: Any) -> tuple[Any, int]:
    torch = _torch_module()
    if isinstance(observations, torch.Tensor):
        tensor = observations
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2:
            raise PolicyError(f"tensor observations must be 1D or 2D, got {tuple(tensor.shape)}")
        return tensor, int(tensor.shape[0])
    array = np.asarray(observations)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise PolicyError(f"array observations must be 1D or 2D, got {array.shape}")
    return array, int(array.shape[0])


class CategoricalPolicyAdapter:
    """Adapter for an SB3 PPO model or its ``ActorCriticPolicy`` object."""

    def __init__(self, model_or_policy: Any):
        self.model = model_or_policy
        self.policy = getattr(model_or_policy, "policy", model_or_policy)
        if not hasattr(self.policy, "observation_space"):
            raise PolicyError("SB3 policy must expose observation_space")
        observation_space = self.policy.observation_space
        space_name = type(observation_space).__name__
        shape = getattr(observation_space, "shape", None)
        if space_name != "Box" or not isinstance(shape, tuple) or len(shape) != 1:
            raise PolicyError(
                "only one-dimensional gymnasium.spaces.Box vector observations are supported"
            )
        if int(shape[0]) <= 0:
            raise PolicyError("observation Box must have a positive vector length")
        dtype = getattr(observation_space, "dtype", None)
        if dtype is not None and not np.issubdtype(np.dtype(dtype), np.number):
            raise PolicyError("observation Box dtype must be numeric")
        # Recurrent policies require hidden state and episode_start handling;
        # silently treating them as feed-forward would explain the wrong model.
        if any(
            getattr(self.policy, attribute, None) is not None
            for attribute in ("lstm_actor", "lstm_critic", "lstm", "recurrent")
        ):
            raise PolicyError("recurrent policies are outside the vector feed-forward scope")
        action_space = getattr(self.policy, "action_space", None)
        action_count = getattr(action_space, "n", None)
        if isinstance(action_count, bool) or not isinstance(action_count, (int, np.integer)):
            raise PolicyError(
                "only a single gymnasium.spaces.Discrete action space is supported"
            )
        if int(action_count) < 2:
            raise PolicyError("categorical action space must have at least two actions")
        start = getattr(action_space, "start", 0)
        if start != 0:
            raise PolicyError("Discrete action spaces with non-zero start are unsupported")
        self._action_count = int(action_count)
        self._check_policy_distribution()

    @property
    def action_count(self) -> int:
        return self._action_count

    def _check_policy_distribution(self) -> None:
        distribution = getattr(self.policy, "action_dist", None)
        # Avoid importing SB3 at module import time.  The class name check also
        # works with a tiny fake ActorCriticPolicy used by the tests.
        if distribution is not None:
            name = type(distribution).__name__
            if name not in {"CategoricalDistribution"}:
                raise PolicyError(f"unsupported SB3 action distribution: {name}")

    def _observation_tensor(self, observations: Any) -> tuple[Any, int]:
        value, batch_size = _one_batch(observations)
        torch = _torch_module()
        if isinstance(value, torch.Tensor):
            tensor = value
            if not tensor.is_floating_point() and tensor.dtype not in {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}:
                raise PolicyError(f"unsupported tensor dtype: {tensor.dtype}")
            return tensor.to(device=getattr(self.policy, "device", tensor.device)), batch_size
        if not hasattr(self.policy, "obs_to_tensor"):
            raise PolicyError("SB3 policy does not expose obs_to_tensor")
        tensor, _vectorized = self.policy.obs_to_tensor(value)
        return tensor, batch_size

    def _raw_logits(self, tensor: Any) -> Any:
        """Obtain pre-softmax actor logits while preserving autograd."""

        policy = self.policy
        # This follows SB3 ActorCriticPolicy.get_distribution but returns the
        # output of action_net before Categorical normalizes it.
        if all(
            hasattr(policy, attribute)
            for attribute in ("extract_features", "pi_features_extractor", "mlp_extractor", "action_net")
        ):
            features = policy.extract_features(tensor, policy.pi_features_extractor)
            latent_pi = policy.mlp_extractor.forward_actor(features)
            return policy.action_net(latent_pi)
        if hasattr(policy, "get_distribution"):
            distribution = policy.get_distribution(tensor)
            torch_distribution = getattr(distribution, "distribution", None)
            logits = getattr(torch_distribution, "logits", None)
            if logits is not None:
                return logits
        raise PolicyError("policy cannot expose categorical logits")

    def logits_tensor(self, observations: Any) -> Any:
        tensor, _batch_size = self._observation_tensor(observations)
        logits = self._raw_logits(tensor)
        torch = _torch_module()
        if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[1] != self.action_count:
            raise PolicyError("policy logits have an unexpected shape")
        return logits

    def probabilities(self, observations: Any) -> np.ndarray:
        value, batch_size = _one_batch(observations)
        torch = _torch_module()
        self.set_eval()
        # Collection records each observation through a one-row policy call.
        # SB3/PyTorch CPU GEMM can use a different reduction order for a large
        # batch, producing tiny probability drift even with identical float32
        # inputs.  Keep the public probability route on the same one-row path
        # so saved-reference replay is bitwise consistent.  ``logits_tensor``
        # remains a batch-capable differentiable path for Integrated Gradients.
        rows = [value] if batch_size == 1 else [value[index : index + 1] for index in range(batch_size)]
        output: list[np.ndarray] = []
        for row in rows:
            tensor, _ = self._observation_tensor(row)
            with torch.no_grad():
                logits = self._raw_logits(tensor)
                probabilities = torch.softmax(logits, dim=-1)
            output.append(probabilities.detach().cpu().numpy())
        result = np.concatenate(output, axis=0)
        return _validate_probabilities(result, expected_batch=batch_size)

    def predict(self, observation: Any, deterministic: bool = True) -> Any:
        predictor = getattr(self.model, "predict", None)
        if predictor is None:
            predictor = getattr(self.policy, "predict", None)
        if predictor is None:
            probabilities = self.probabilities(observation)
            actions = np.argmax(probabilities, axis=1)
        else:
            output = predictor(observation, deterministic=deterministic)
            actions = output[0] if isinstance(output, tuple) else output
            actions = np.asarray(actions)
            if actions.ndim == 0:
                actions = actions.reshape(1)
            actions = actions.reshape(-1)
        if not np.issubdtype(actions.dtype, np.integer):
            if not np.all(np.equal(actions, np.floor(actions))):
                raise PolicyError("policy returned non-integer discrete actions")
            actions = actions.astype(np.int64)
        if np.any(actions < 0) or np.any(actions >= self.action_count):
            raise PolicyError("policy returned an out-of-range discrete action")
        if np.asarray(observation).ndim == 1:
            return int(actions[0])
        return actions.astype(np.int64)

    def set_eval(self) -> None:
        if hasattr(self.policy, "set_training_mode"):
            self.policy.set_training_mode(False)
        elif hasattr(self.policy, "eval"):
            self.policy.eval()

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(type(self.policy).__qualname__.encode("utf-8"))
        digest.update(str(getattr(self.policy, "observation_space", "")).encode("utf-8"))
        digest.update(str(getattr(self.policy, "action_space", "")).encode("utf-8"))
        state = getattr(self.policy, "state_dict", None)
        if callable(state):
            try:
                values = state()
                for name in sorted(values):
                    digest.update(str(name).encode("utf-8"))
                    tensor = values[name]
                    digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            except Exception as exc:  # pragma: no cover - custom policy only
                digest.update(f"state_dict_unavailable:{type(exc).__name__}".encode("utf-8"))
        model_path = getattr(self.model, "_last_obs", None)
        if model_path is not None:
            digest.update(repr(model_path).encode("utf-8"))
        return digest.hexdigest()


@dataclass(slots=True)
class ArrayPolicyAdapter:
    """Adapter for fake/portable policies used by core tests and ports."""

    probability_fn: Callable[[np.ndarray], Any]
    predict_fn: Callable[..., Any] | None = None
    logits_fn: Callable[[Any], Any] | None = None
    fingerprint_value: str = ""
    action_count_value: int | None = None

    @property
    def action_count(self) -> int | None:
        return self.action_count_value

    def probabilities(self, observations: Any) -> np.ndarray:
        array = np.asarray(observations)
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim != 2:
            raise PolicyError("array policy observations must have shape (batch, dimension)")
        batch = array.shape[0]
        result = _validate_probabilities(self.probability_fn(array), expected_batch=batch)
        if self.action_count_value is None:
            self.action_count_value = int(result.shape[1])
        elif result.shape[1] != self.action_count_value:
            raise PolicyError("policy action count changed between calls")
        return result

    def predict(self, observation: Any, deterministic: bool = True) -> Any:
        if self.predict_fn is None:
            actions = np.argmax(self.probabilities(observation), axis=1)
        else:
            try:
                actions = self.predict_fn(observation, deterministic=deterministic)
            except TypeError:
                actions = self.predict_fn(observation)
            if isinstance(actions, tuple):
                actions = actions[0]
            actions = np.asarray(actions)
            if actions.ndim == 0:
                actions = actions.reshape(1)
            actions = actions.reshape(-1)
        actions = actions.astype(np.int64)
        count = self.action_count_value
        if count is not None and np.any((actions < 0) | (actions >= count)):
            raise PolicyError("policy returned an out-of-range action")
        if np.asarray(observation).ndim == 1:
            return int(actions[0])
        return actions

    def logits_tensor(self, observations: Any) -> Any:
        if self.logits_fn is None:
            raise PolicyError("this policy has no differentiable logits_fn")
        return self.logits_fn(observations)

    def fingerprint(self) -> str:
        if self.fingerprint_value:
            return self.fingerprint_value
        return hashlib.sha256(repr(self.probability_fn).encode("utf-8")).hexdigest()


def as_policy_adapter(value: Any) -> PolicyLike:
    """Return ``value`` as a policy adapter or fail with a useful message."""

    if isinstance(value, (CategoricalPolicyAdapter, ArrayPolicyAdapter)):
        return value
    if all(callable(getattr(value, name, None)) for name in ("probabilities", "predict")):
        missing = [
            name
            for name in ("fingerprint", "set_eval")
            if not callable(getattr(value, name, None))
        ]
        if missing:
            raise PolicyError(
                "direct policy objects must expose probabilities, predict, "
                "fingerprint, and set_eval; missing "
                + ", ".join(missing)
                + ". Use ArrayPolicyAdapter for an explicit static adapter."
            )
        return value
    try:
        return CategoricalPolicyAdapter(value)
    except PolicyError:
        raise
    except Exception as exc:
        raise PolicyError(f"cannot adapt policy object {type(value).__name__}") from exc


def policy_probabilities(policy: Any, observations: Any) -> np.ndarray:
    return as_policy_adapter(policy).probabilities(observations)


def policy_action(policy: Any, observation: Any, deterministic: bool = True) -> Any:
    return as_policy_adapter(policy).predict(observation, deterministic=deterministic)
