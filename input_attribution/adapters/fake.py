"""Self-contained vector environment and categorical policy for contract tests.

The fake adapter intentionally has no MetaDrive or Stable-Baselines3 imports.
It exercises the same observation copy, preprocessing, probability, action,
reset/step, and optional gradient hooks as :class:`MetaDriveAdapter`, while
remaining deterministic and cheap enough for every test run.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from typing import Any, Sequence

import numpy as np

from .metadrive import AdapterError, PolicyOutput, TensorPolicyOutput


class _FallbackBox:
    def __init__(self, dimension: int) -> None:
        self.shape = (int(dimension),)
        self.dtype = np.dtype(np.float32)
        self.low = np.full(self.shape, -1.0, dtype=np.float32)
        self.high = np.full(self.shape, 1.0, dtype=np.float32)


class _FallbackDiscrete:
    def __init__(self, count: int) -> None:
        self.n = int(count)
        self.start = 0


def _spaces(dimension: int, action_count: int) -> tuple[Any, Any]:
    try:
        from gymnasium import spaces

        return (
            spaces.Box(-1.0, 1.0, shape=(dimension,), dtype=np.float32),
            spaces.Discrete(action_count),
        )
    except ImportError:  # pragma: no cover - exercised only on minimal installs
        return _FallbackBox(dimension), _FallbackDiscrete(action_count)


class FakeVectorEnv:
    """A deterministic Gymnasium-like vector environment with no simulator."""

    metadata: dict[str, Any] = {}

    def __init__(
        self,
        dimension: int = 259,
        action_count: int = 9,
        horizon: int = 4,
        categorical_indices: Sequence[int] = (),
    ) -> None:
        if dimension <= 0 or action_count < 2 or horizon <= 0:
            raise ValueError("dimension, action_count, and horizon must be positive")
        self.observation_space, self.action_space = _spaces(dimension, action_count)
        self.dimension = int(dimension)
        self.horizon = int(horizon)
        self.step_count = 0
        self.reset_count = 0
        self.closed = False
        self.current_seed: int | None = None
        self.categorical_indices = tuple(int(index) for index in categorical_indices)
        if any(index < 0 or index >= self.dimension for index in self.categorical_indices):
            raise ValueError("categorical_indices must be within the fake observation dimension")
        self._observation = np.zeros(self.dimension, dtype=np.float32)

    @property
    def config(self) -> dict[str, Any]:
        return {
            "discrete_action": True,
            "discrete_steering_dim": 3,
            "discrete_throttle_dim": 3,
            "physics_world_step_size": 0.02,
            "decision_repeat": 5,
            "horizon": self.horizon,
            "categorical_indices": list(self.categorical_indices),
        }

    def reset(self, *, seed: int | None = None, options: Mapping[str, Any] | None = None):
        del options
        self.current_seed = None if seed is None else int(seed)
        self.step_count = 0
        self.reset_count += 1
        self.closed = False
        self._observation.fill(0.0)
        self._set_categorical_defaults()
        return np.array(self._observation, copy=True), {
            "simulation_time": 0.0,
            "fake_reset": True,
            "seed": self.current_seed,
        }

    def step(self, action: Any):
        if self.closed:
            raise RuntimeError("fake environment is closed")
        try:
            value = int(np.asarray(action).reshape(-1)[0])
        except (TypeError, ValueError, IndexError) as error:
            raise ValueError(f"invalid fake action: {action!r}") from error
        if value < 0 or value >= int(self.action_space.n):
            raise ValueError(f"fake action {value} is outside Discrete({self.action_space.n})")
        self.step_count += 1
        # A bounded, deterministic signal makes preprocessing and observation
        # mutation checks visible without introducing random state.
        self._observation.fill(np.float32(min(self.step_count, self.horizon) / self.horizon))
        self._set_categorical_defaults()
        terminated = self.step_count >= self.horizon
        info = {
            "simulation_time": self.step_count * 0.1,
            "velocity": float(self.step_count),
            "action": value,
            "target_lane_valid": True,
        }
        return (
            np.array(self._observation, copy=True),
            float(value) / max(1, int(self.action_space.n) - 1),
            terminated,
            False,
            info,
        )

    def _set_categorical_defaults(self) -> None:
        for index in self.categorical_indices:
            # The synthetic valid flag is deliberately kept valid throughout
            # a fake episode; real ports must source their own invalid state.
            self._observation[index] = 1.0

    def close(self) -> None:
        self.closed = True


@dataclass
class FakePolicyAdapter:
    """Linear categorical policy with a differentiable PyTorch-compatible path."""

    dimension: int = 259
    action_count: int = 9
    horizon: int = 4
    weights: np.ndarray | None = None
    bias: np.ndarray | None = None
    value_weights: np.ndarray | None = None
    categorical_indices: tuple[int, ...] = ()

    @classmethod
    def from_analysis_config(cls, config: Any, **kwargs: Any) -> "FakePolicyAdapter":
        """Construct a fake from the same resolved config shape as the real adapter."""

        schema = getattr(config, "schema", None)
        if schema is None and isinstance(config, Mapping):
            schema = config.get("schema", {})
        scenario = getattr(config, "scenario", None)
        if scenario is None and isinstance(config, Mapping):
            scenario = config.get("scenario", {})
        adapter = getattr(config, "adapter", None)
        if adapter is None and isinstance(config, Mapping):
            adapter = config.get("adapter", {})
        schema = schema if isinstance(schema, Mapping) else {}
        scenario = scenario if isinstance(scenario, Mapping) else {}
        adapter = adapter if isinstance(adapter, Mapping) else {}
        environment = getattr(config, "environment", None)
        if environment is None and isinstance(config, Mapping):
            environment = config.get("environment", {})
        environment = environment if isinstance(environment, Mapping) else {}
        settings = environment.get("settings", {})
        settings = settings if isinstance(settings, Mapping) else {}
        dimension = kwargs.pop("dimension", schema.get("dimension", 259))
        action_count = kwargs.pop("action_count", adapter.get("action_count", 9))
        horizon = kwargs.pop("horizon", scenario.get("horizon", 4))
        categorical_indices = kwargs.pop(
            "categorical_indices",
            settings.get("categorical_indices", ()),
        )
        return cls(
            dimension=int(dimension),
            action_count=int(action_count),
            horizon=int(horizon),
            categorical_indices=tuple(int(index) for index in categorical_indices),
            **kwargs,
        )

    from_config = from_analysis_config

    def __post_init__(self) -> None:
        if self.dimension <= 0 or self.action_count < 2:
            raise ValueError("dimension must be positive and action_count must be at least two")
        self.dimension = int(self.dimension)
        self.action_count = int(self.action_count)
        self.horizon = int(self.horizon)
        self.categorical_indices = tuple(int(index) for index in self.categorical_indices)
        if any(index < 0 or index >= self.dimension for index in self.categorical_indices):
            raise ValueError("categorical_indices must be within the fake observation dimension")
        if self.weights is None:
            matrix = np.zeros((self.action_count, self.dimension), dtype=np.float32)
            for action in range(self.action_count):
                matrix[action, action % self.dimension] = (action + 1) / self.action_count
            self.weights = matrix
        else:
            self.weights = np.asarray(self.weights, dtype=np.float32)
        if self.weights.shape != (self.action_count, self.dimension):
            raise ValueError(
                f"weights must have shape {(self.action_count, self.dimension)}, got {self.weights.shape}"
            )
        if self.bias is None:
            self.bias = np.linspace(-0.2, 0.2, self.action_count, dtype=np.float32)
        else:
            self.bias = np.asarray(self.bias, dtype=np.float32)
        if self.bias.shape != (self.action_count,):
            raise ValueError(f"bias must have shape {(self.action_count,)}, got {self.bias.shape}")
        if self.value_weights is None:
            self.value_weights = np.linspace(0.01, 0.02, self.dimension, dtype=np.float32)
        else:
            self.value_weights = np.asarray(self.value_weights, dtype=np.float32)
        if self.value_weights.shape != (self.dimension,):
            raise ValueError(f"value_weights must have shape {(self.dimension,)}, got {self.value_weights.shape}")
        if not all(np.all(np.isfinite(value)) for value in (self.weights, self.bias, self.value_weights)):
            raise ValueError("fake policy parameters must be finite")
        self.env: FakeVectorEnv | None = None

    @property
    def observation_dim(self) -> int:
        return self.dimension

    def make_environment(self, config: Any | None = None, *, seed: int | None = None) -> FakeVectorEnv:
        horizon = self.horizon
        section = getattr(config, "scenario", None)
        if section is None and isinstance(config, Mapping):
            section = config.get("scenario")
        if isinstance(section, Mapping) and section.get("horizon") is not None:
            horizon = int(section["horizon"])
        self.env = FakeVectorEnv(
            self.dimension,
            self.action_count,
            horizon=max(1, horizon),
            categorical_indices=self.categorical_indices,
        )
        if seed is not None:
            self.env.reset(seed=seed)
        return self.env

    make_env = make_environment

    def close(self, env: Any | None = None) -> None:
        target = env if env is not None else self.env
        if target is not None and hasattr(target, "close"):
            target.close()
        if target is self.env:
            self.env = None

    close_env = close

    def reset_env(self, env: FakeVectorEnv | None = None, *, seed: int | None = None):
        if env is not None:
            self.env = env
        if self.env is None:
            self.make_environment()
        return self.env.reset(seed=seed)

    def reset(self, *, scenario_seed: int | None = None):
        return self.reset_env(seed=scenario_seed)

    def preprocess_observation(
        self,
        observation: Any,
        info: Mapping[str, Any] | None = None,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        array = np.asarray(observation, dtype=np.float32)
        if array.ndim != 1 or array.shape[0] != self.dimension:
            raise AdapterError(f"fake observation must have shape ({self.dimension},), got {array.shape}")
        if not np.all(np.isfinite(array)):
            raise AdapterError("fake observation contains non-finite values")
        copied = np.array(array, copy=True)
        if info is None:
            return copied
        return copied, {
            "external_normalization": False,
            "frame_stack": False,
            "dtype": "float32",
            "shape": list(copied.shape),
            "source": "fake vector Box -> linear categorical policy",
        }

    def render_frame(
        self,
        env: FakeVectorEnv,
        *,
        step: int | None = None,
        simulation_time: float | None = None,
        pattern_id: str | None = None,
        phase: str = "post",
    ) -> np.ndarray:
        """Return a deterministic RGB frame for video-hook tests only."""

        del simulation_time, pattern_id, phase
        if env is None:
            raise AdapterError("fake render requires an environment")
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        value = int(step if step is not None else env.steps) % 256
        frame[:, :, 0] = value
        frame[:, :, 1] = (value * 3) % 256
        return np.array(frame, copy=True)

    def logits_tensor(self, observations: Any) -> Any:
        try:
            import torch
        except ImportError as error:  # pragma: no cover - minimal installs
            raise AdapterError("PyTorch is required for fake gradient evaluation") from error
        if not isinstance(observations, torch.Tensor):
            observations = torch.as_tensor(observations, dtype=torch.float32)
        if observations.ndim == 1:
            observations = observations.unsqueeze(0)
        if observations.ndim != 2 or observations.shape[1] != self.dimension:
            raise AdapterError(f"fake tensor observations must be (batch,{self.dimension})")
        weights = torch.as_tensor(self.weights, dtype=observations.dtype, device=observations.device)
        bias = torch.as_tensor(self.bias, dtype=observations.dtype, device=observations.device)
        return observations.matmul(weights.transpose(0, 1)) + bias

    def evaluate_tensors(self, observations: Any) -> TensorPolicyOutput:
        import torch

        logits = self.logits_tensor(observations)
        if logits.ndim != 2:
            raise AdapterError("fake policy logits must be two-dimensional")
        value_weights = torch.as_tensor(
            self.value_weights, dtype=logits.dtype, device=logits.device
        )
        if not isinstance(observations, torch.Tensor):
            observations = torch.as_tensor(observations, dtype=logits.dtype, device=logits.device)
        if observations.ndim == 1:
            observations = observations.unsqueeze(0)
        values = observations.matmul(value_weights)
        return TensorPolicyOutput(
            logits=logits,
            log_probabilities=torch.log_softmax(logits, dim=-1),
            probabilities=torch.softmax(logits, dim=-1),
            values=values,
        )

    def evaluate(self, observations: Any) -> PolicyOutput:
        output = self.evaluate_tensors(observations)
        return PolicyOutput(
            logits=output.logits.detach().cpu().numpy(),
            log_probabilities=output.log_probabilities.detach().cpu().numpy(),
            probabilities=output.probabilities.detach().cpu().numpy(),
            deterministic_actions=output.logits.argmax(dim=-1).detach().cpu().numpy(),
            values=output.values.detach().cpu().numpy(),
        )

    def probabilities(self, observations: Any) -> np.ndarray:
        return np.asarray(self.evaluate(observations).probabilities, dtype=np.float64)

    distribution_probabilities = probabilities

    def predict(self, observations: Any, *, deterministic: bool = True) -> Any:
        del deterministic
        actions = self.evaluate(observations).deterministic_actions
        if np.asarray(observations).ndim == 1:
            return int(np.asarray(actions).reshape(-1)[0])
        return np.asarray(actions, dtype=np.int64)

    def decode_action(self, action: Any, env_config: Mapping[str, Any] | None = None) -> dict[str, float | int]:
        del env_config
        value = int(np.asarray(action).reshape(-1)[0])
        steering_dim = 3
        steering = (value % steering_dim) - 1
        throttle = (value // steering_dim) - 1
        return {"action": value, "steering": float(steering), "throttle_brake": float(throttle)}

    def assert_contract(self, *, expected_dimension: int | None = None, expected_actions: int | None = None) -> dict[str, Any]:
        if expected_dimension is not None and expected_dimension != self.dimension:
            raise AdapterError(f"observation dimension mismatch: runtime={self.dimension}, expected={expected_dimension}")
        if expected_actions is not None and expected_actions != self.action_count:
            raise AdapterError(f"action count mismatch: runtime={self.action_count}, expected={expected_actions}")
        return {
            "observation_dim": self.dimension,
            "observation_dtype": "float32",
            "action_count": self.action_count,
            "action_space": f"Discrete({self.action_count})",
            "external_normalization": False,
            "preprocess": "float32 identity for vector Box",
        }

    def load_policy(self, config: Any | None = None) -> "FakePolicyAdapter":
        del config
        return self

    def verify_schema_contract(self, schema: Any, *, expected_dimension: int | None = None) -> dict[str, Any]:
        dimension = getattr(schema, "dimension", None)
        if dimension is None and isinstance(schema, Mapping):
            dimension = schema.get("dimension")
        try:
            schema_dimension = int(dimension)
        except (TypeError, ValueError) as error:
            raise AdapterError("schema must expose an integer dimension") from error
        if expected_dimension is not None and schema_dimension != int(expected_dimension):
            raise AdapterError(f"schema dimension mismatch: schema={schema_dimension}, expected={expected_dimension}")
        if schema_dimension != self.dimension:
            raise AdapterError(f"schema/runtime observation dimension mismatch: schema={schema_dimension}, runtime={self.dimension}")
        validate = getattr(schema, "validate", None)
        if callable(validate):
            validate()
        return {
            "schema_dimension": schema_dimension,
            "runtime_dimension": self.dimension,
            "source_evidence": {"source_contract": "synthetic_fake_adapter"},
            "runtime_checks": {"status": "synthetic", "observation_defaults": None},
            "verified": True,
        }

    def telemetry(
        self,
        env: FakeVectorEnv,
        info: Mapping[str, Any] | None = None,
        *,
        phase: str = "post",
        step: int = 0,
    ) -> dict[str, Any]:
        del phase
        value = dict(info or {})
        elapsed = float(step) * 0.1
        value.update(
            {
                "step": int(step),
                "sim_time_seconds": elapsed,
                "sim_time_s": elapsed,
                "actual_scenario_seed": env.current_seed,
                "road_segment_id": ["synthetic", 0],
                "target_lane_ordinal": 0,
                "current_lane_ordinal": 0,
                "target_lane_valid": True,
                "target_lane_offset_m": 0.0,
                "normalized_target_lane_error": 0.0,
                "in_target_lane": True,
                "target_lane_heading_rad": 0.0,
                "position_xy_m": [float(step), 0.0],
            }
        )
        if "velocity" in value:
            value["speed_m_s"] = float(value["velocity"])
            value["speed_km_h"] = float(value["velocity"]) * 3.6
        return value

    def set_eval(self) -> None:
        """This static NumPy linear policy has no training mode or mutable statistics."""
        return None

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for value in (self.weights, self.bias, self.value_weights):
            digest.update(np.ascontiguousarray(value).tobytes())
        return digest.hexdigest()


FakeAdapter = FakePolicyAdapter


def create_adapter(config: Any, **kwargs: Any) -> FakePolicyAdapter:
    """Factory entry point for CLI/config loading without positional ambiguity."""

    return FakePolicyAdapter.from_analysis_config(config, **kwargs)


__all__ = ["FakeAdapter", "FakePolicyAdapter", "FakeVectorEnv", "create_adapter"]
