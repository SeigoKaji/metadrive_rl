"""Deterministic, dependency-light adapter used by tests and demonstrations.

The synthetic backend exercises the same boundaries as a ported MetaDrive
adapter: reset/step/close, prepared policy inputs, action probabilities,
read-only snapshots, frames, and returned reward terms.  It deliberately does
not pretend to validate the real environment's observation schema.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np


class SyntheticSpace:
    """Tiny space object with the attributes used by the collection runner."""

    def __init__(self, *, shape: tuple[int, ...] | None = None, n: int | None = None) -> None:
        self.shape = shape
        self.n = n

    def contains(self, value: object) -> bool:
        if self.shape is not None:
            try:
                return np.asarray(value).shape == self.shape and bool(
                    np.all(np.isfinite(np.asarray(value, dtype=float)))
                )
            except (TypeError, ValueError):
                return False
        return isinstance(value, (int, np.integer)) and not isinstance(value, bool) and 0 <= int(value) < int(self.n or 0)

    def sample(self) -> int | np.ndarray:
        if self.n is not None:
            return 0
        return np.zeros(self.shape or (), dtype=np.float32)


class SyntheticPolicy:
    """Small softmax policy with a normal ``predict`` boundary."""

    def __init__(self, dimension: int, action_count: int, *, seed: int = 0) -> None:
        self.dimension = int(dimension)
        self.action_count = int(action_count)
        # Structured weights make interventions visible while keeping values
        # stable across Python/numpy versions and independent of random state.
        index = np.arange(self.dimension, dtype=np.float64) + 1.0
        action = np.arange(self.action_count, dtype=np.float64)[:, None] + 1.0
        self.weights = np.sin(action * index[None, :] * 0.017) * 0.08
        self.weights += ((action % 3.0) - 1.0) * (index[None, :] % 11.0) * 0.002
        self.bias = np.linspace(-0.15, 0.15, self.action_count, dtype=np.float64)
        if self.action_count >= 2 and self.dimension >= 2:
            # The first two dimensions are intentionally action-sensitive so
            # a fixed-value test visibly changes the executed trajectory.
            self.weights[0, 0] += 0.9
            self.weights[1, 0] -= 0.9
            self.weights[0, 1] -= 0.7
            self.weights[1, 1] += 0.7
        self.rng = np.random.default_rng(seed)

    def seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(int(seed))

    def probabilities(self, inputs: object) -> np.ndarray:
        values = np.asarray(inputs, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
            one = True
        elif values.ndim == 2:
            # Evaluate rows through the same scalar path used by an ordinary
            # sequential call.  Besides being deterministic, this makes the
            # synthetic batch-vs-sequential acceptance check compare the
            # exact saved probabilities rather than BLAS reduction roundoff.
            return np.stack([self.probabilities(row) for row in values], axis=0)
        else:
            raise ValueError(f"policy input must be 1-D or 2-D, found {values.shape}")
        if values.shape[1] != self.dimension:
            raise ValueError(
                f"policy input width {values.shape[1]} does not match {self.dimension}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("policy input contains NaN or Inf")
        logits = values @ self.weights.T + self.bias[None, :]
        logits -= np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        return probabilities[0] if one else probabilities

    def predict(self, inputs: object, deterministic: bool = True) -> tuple[int | np.ndarray, None]:
        probabilities = self.probabilities(inputs)
        one = probabilities.ndim == 1
        rows = probabilities[None, :] if one else probabilities
        if deterministic:
            actions = np.argmax(rows, axis=1).astype(np.int64)
        else:
            actions = np.asarray(
                [self.rng.choice(self.action_count, p=row) for row in rows],
                dtype=np.int64,
            )
        return (int(actions[0]) if one else actions), None

    def identity(self) -> dict[str, object]:
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(self.weights).tobytes())
        digest.update(np.ascontiguousarray(self.bias).tobytes())
        return {
            "kind": "synthetic_softmax",
            "dimension": self.dimension,
            "action_count": self.action_count,
            "weights_sha256": digest.hexdigest(),
        }


class SyntheticEnv:
    """A deterministic finite-horizon Gymnasium-shaped environment."""

    def __init__(
        self,
        dimension: int = 259,
        action_count: int = 9,
        steps: int = 127,
        *,
        action_dt: float = 0.1,
        natural_end_step: int | None = None,
        fail_snapshot_at: int | None = None,
        fail_frame_at: int | None = None,
        include_reward_terms: bool = True,
    ) -> None:
        if int(dimension) <= 0 or int(action_count) <= 0 or int(steps) < 0:
            raise ValueError("dimension/action_count must be positive and steps non-negative")
        if not math.isfinite(float(action_dt)) or float(action_dt) <= 0:
            raise ValueError("action_dt must be positive and finite")
        self.dimension = int(dimension)
        self.action_count = int(action_count)
        self.steps = int(steps)
        self.action_dt_value = float(action_dt)
        self.natural_end_step = natural_end_step
        self.fail_snapshot_at = fail_snapshot_at
        self.fail_frame_at = fail_frame_at
        self.include_reward_terms = bool(include_reward_terms)
        self.observation_space = SyntheticSpace(shape=(self.dimension,))
        self.action_space = SyntheticSpace(n=self.action_count)
        self.reset_calls = 0
        self.step_calls = 0
        self.close_calls = 0
        self.current_seed: int | None = None
        self.step_index = 0
        self.actions: list[int] = []
        self.road_section_id = 0
        self.position = 0.0
        self.speed = 0.0
        self.closed = False
        self._rng = np.random.default_rng(0)
        self._previous_action: int | None = None
        self._action_run_length = 0
        self._stress_signal = False

    def _observation(self) -> np.ndarray:
        # Every normal value is in (0, 1), so -1 stress replacement changes
        # all targets in the usual synthetic runs.
        index = np.arange(self.dimension, dtype=np.float64)
        phase = (self.current_seed or 0) * 0.013 + self.step_index * 0.031
        values = 0.5 + 0.23 * np.sin(index * 0.071 + phase)
        values += 0.03 * self.road_section_id
        # Closed-loop actions affect the next observation.  This guards the
        # core against accidentally replaying the baseline observation stream
        # during an intervention episode.
        values += 0.02 * np.tanh(self.position * 0.2) + 0.01 * self.speed
        return np.asarray(np.clip(values, 0.02, 0.98), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: Mapping[str, object] | None = None) -> tuple[np.ndarray, dict[str, object]]:
        if self.closed:
            raise RuntimeError("synthetic environment is closed")
        self.reset_calls += 1
        if seed is not None:
            self.current_seed = int(seed)
            self._rng = np.random.default_rng(self.current_seed)
        elif self.current_seed is None:
            self.current_seed = 0
        self.step_index = 0
        self.actions = []
        self.road_section_id = 0
        self.position = 0.0
        self.speed = 0.0
        self._previous_action = None
        self._action_run_length = 0
        self._stress_signal = False
        return self._observation(), {"seed": self.current_seed, "road_section_id": self.road_section_id}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        if self.closed:
            raise RuntimeError("synthetic environment is closed")
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise ValueError("synthetic action must be an integer")
        action_value = int(action)
        if not 0 <= action_value < self.action_count:
            raise ValueError(f"action {action_value} outside [0, {self.action_count})")
        self.step_calls += 1
        self.step_index += 1
        self.actions.append(action_value)
        if self._previous_action == action_value:
            self._action_run_length += 1
        else:
            self._action_run_length = 1
        self._previous_action = action_value
        # A synthetic natural goal is deliberately reached through the
        # action-dependent trajectory.  This lets one shared environment
        # config produce a 127-step nominal episode and a 50-step stressed
        # episode, without the collector changing environment options for a
        # pattern.  The small-width branch keeps the compact unit-test env
        # useful when its action policy has no representative host trajectory.
        if self.natural_end_step is not None and self.step_index <= self.natural_end_step:
            if self.dimension <= 16:
                self._stress_signal = True
            elif (
                action_value == 1
                and self._action_run_length >= 10
            ) or (
                action_value == 0
                and self._action_run_length >= max(1, self.natural_end_step - 15)
            ):
                self._stress_signal = True
        self.road_section_id = 1 if self.step_index > max(1, self.steps // 2) else 0
        self.speed = 0.2 + 0.04 * (self.action_count - 1 - action_value)
        self.position += self.speed * self.action_dt_value
        progress = float(self.speed)
        action_penalty = -0.01 * float(action_value)
        section_adjustment = -0.02 if self.road_section_id else 0.0
        reward = float(progress + action_penalty + section_adjustment)
        terminated = bool(
            self.natural_end_step is not None
            and self.step_index >= self.natural_end_step
            and self._stress_signal
        )
        truncated = bool(not terminated and self.step_index >= self.steps)
        info: dict[str, object] = {
            "seed": self.current_seed,
            "road_section_id": self.road_section_id,
            "position": self.position,
            "speed": self.speed,
            "termination_reason": (
                "synthetic_goal" if terminated else "synthetic_horizon" if truncated else None
            ),
        }
        if self.include_reward_terms:
            info["reward_terms"] = {
                "progress": progress,
                "action_penalty": action_penalty,
                "section_adjustment": section_adjustment,
            }
        return self._observation(), reward, terminated, truncated, info

    def snapshot(self) -> dict[str, object]:
        if self.fail_snapshot_at is not None and self.step_index >= self.fail_snapshot_at:
            raise RuntimeError(f"synthetic snapshot failure at step {self.step_index}")
        return {
            "step": self.step_index,
            "seed": self.current_seed,
            "road_section_id": self.road_section_id,
            "position": float(self.position),
            "speed": float(self.speed),
        }

    def frame(self) -> np.ndarray:
        if self.fail_frame_at is not None and self.step_index >= self.fail_frame_at:
            raise RuntimeError(f"synthetic frame failure at step {self.step_index}")
        height, width = 72, 128
        image = np.zeros((height, width, 3), dtype=np.uint8)
        red = int((self.step_index * 17 + self.road_section_id * 60) % 255)
        green = int((self.current_seed or 0) * 7 % 255)
        image[..., 0] = red
        image[..., 1] = green
        image[..., 2] = 180
        # A clear moving stripe gives GIF tests something observable to check.
        column = self.step_index % width
        image[:, max(0, column - 2) : min(width, column + 3), :] = (240, 240, 240)
        return image

    def close(self) -> None:
        self.close_calls += 1
        self.closed = True


class SyntheticAdapter:
    """Adapter implementing the core environment/policy connection contract."""

    def __init__(
        self,
        dimension: int = 259,
        action_count: int = 9,
        steps: int = 127,
        *,
        action_dt: float = 0.1,
        natural_end_step: int | None = None,
        fail_snapshot_at: int | None = None,
        fail_frame_at: int | None = None,
        include_reward_terms: bool = True,
    ) -> None:
        self.dimension = int(dimension)
        self.action_count = int(action_count)
        self.steps = int(steps)
        self._policy = SyntheticPolicy(self.dimension, self.action_count)
        self._seed = 0
        self._env_options = {
            "action_dt": action_dt,
            "natural_end_step": natural_end_step,
            "fail_snapshot_at": fail_snapshot_at,
            "fail_frame_at": fail_frame_at,
            "include_reward_terms": include_reward_terms,
        }
        self.make_env_calls = 0
        self.predict_calls = 0
        self.prepare_calls = 0
        self.last_model_input: np.ndarray | None = None
        self.prediction_inputs: list[np.ndarray] = []
        self.last_env: SyntheticEnv | None = None
        self.metadata: dict[str, object] = {
            "backend": "synthetic",
            "dimension": self.dimension,
            "action_count": self.action_count,
            "model": self._policy.identity(),
            "preprocessing": {
                "name": "identity_float32",
                "boundary": "prepared_policy_input",
                "normalization": "none",
            },
            "policy": {"deterministic_default": True, "seed": self._seed},
        }

    def prepare(self, raw: object) -> np.ndarray:
        self.prepare_calls += 1
        array = np.asarray(raw)
        if array.ndim != 1 or array.shape[0] != self.dimension:
            raise ValueError(
                f"raw observation must have shape ({self.dimension},), found {array.shape}"
            )
        if np.issubdtype(array.dtype, np.complexfloating):
            raise ValueError("raw observation must be real")
        try:
            prepared = np.asarray(array, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("raw observation cannot be converted to float32") from error
        if not np.all(np.isfinite(prepared)):
            raise ValueError("raw observation contains NaN or Inf")
        return np.array(prepared, copy=True)

    def predict(self, inputs: object, deterministic: bool = True) -> tuple[np.ndarray, int | np.ndarray]:
        self.predict_calls += 1
        values = np.asarray(inputs, dtype=np.float64)
        if values.ndim not in {1, 2}:
            raise ValueError(f"policy input must be 1-D or 2-D, found {values.shape}")
        if values.shape[-1] != self.dimension:
            raise ValueError(f"policy input width must be {self.dimension}")
        if not np.all(np.isfinite(values)):
            raise ValueError("policy input contains NaN or Inf")
        self.last_model_input = np.array(values, copy=True)
        self.prediction_inputs.append(np.array(values, copy=True))
        probabilities = np.asarray(self._policy.probabilities(values), dtype=np.float64)
        actions, _ = self._policy.predict(values, deterministic=deterministic)
        # ``predict`` is the ordinary model boundary used by baseline and
        # closed-loop.  Recompute the deterministic argmax for a direct
        # consistency check so a model/policy mismatch is visible immediately.
        if deterministic:
            expected = np.argmax(probabilities, axis=-1)
            actual = np.asarray(actions)
            if not np.array_equal(actual, expected):
                raise RuntimeError("ordinary model.predict disagrees with probabilities")
        if values.ndim == 1:
            return probabilities, int(actions)
        return probabilities, np.asarray(actions, dtype=np.int64)

    def make_env(self) -> SyntheticEnv:
        self.make_env_calls += 1
        options = dict(self._env_options)
        # The optional natural end is evaluated from the action-derived
        # trajectory inside SyntheticEnv.  Baseline and stressed episodes
        # therefore share this environment configuration.
        self.last_env = SyntheticEnv(
            self.dimension,
            self.action_count,
            self.steps,
            **options,
        )
        return self.last_env

    def seed_policy(self, seed: int) -> None:
        if isinstance(seed, bool) or int(seed) < 0:
            raise ValueError("policy seed must be a non-negative integer")
        self._seed = int(seed)
        self._policy.seed(self._seed)
        policy_metadata = self.metadata.setdefault("policy", {})
        if isinstance(policy_metadata, dict):
            policy_metadata["seed"] = self._seed

    def snapshot(self, env: SyntheticEnv) -> dict[str, object]:
        return env.snapshot()

    def frame(self, env: SyntheticEnv) -> np.ndarray:
        return env.frame()

    def action_dt(self, env: SyntheticEnv) -> float:
        return float(env.action_dt_value)

    def model_identity(self) -> dict[str, object]:
        return dict(self.metadata.get("model", {})) if isinstance(self.metadata.get("model"), Mapping) else {}


def make_synthetic_adapter(
    *,
    dimension: int = 259,
    action_count: int = 9,
    steps: int = 127,
    **kwargs: object,
) -> SyntheticAdapter:
    """Factory used by the CLI and small downstream demonstrations."""

    return SyntheticAdapter(dimension, action_count, steps, **kwargs)


__all__ = [
    "SyntheticAdapter",
    "SyntheticEnv",
    "SyntheticPolicy",
    "SyntheticSpace",
    "make_synthetic_adapter",
]
