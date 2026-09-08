"""Small explicit adapter skeleton for a different repository.

Copy this file into the portable package and implement the hooks against the
target project's existing observation producer and environment.  The default
implementation deliberately raises instead of guessing a 262-dimensional
index, neutral value, normalization, action encoding, or target-lane rule.
``load_policy`` is likewise separate from ``make_env`` so a saved-observation
report can load a policy without starting a simulator.

Typical port shape::

    class MyAdapter(PortAdapterTemplate):
        def make_env(self, config=None, *, seed=None): ...
        def load_policy(self, config=None): return MyPolicy(load_path, env=None)
        def preprocess_observation(self, observation, info=None): ...
        # implement reset_env, telemetry, decode_action, verify_schema_contract,
        # assert_contract, source_paths, and optionally render_frame

Keep the target repository's existing files unchanged; this adapter is the
only place where its private connection details should be absorbed.  The
portable package never overwrites an external 262-dimensional producer or
model.  A port should expose only source-verified meanings.  The
``closed-loop --patterns`` selection chooses the patterns to run, while
``video.patterns`` chooses a small set of those patterns to capture (normally
``P00`` plus one or two focused variants).  Omit it to capture every run
pattern when video is enabled, or set ``video.enabled`` to false to disable
all capture; it must not infer a 262-D layout from length.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


class PortAdapterTemplate:
    """Fail-closed hook contract for a portable vector PPO adapter."""

    def __init__(self, config: Any | None = None, **kwargs: Any) -> None:
        self.analysis_config = config
        self.options = dict(kwargs)
        self.env: Any | None = None

    @classmethod
    def from_analysis_config(cls, config: Any, **kwargs: Any) -> "PortAdapterTemplate":
        return cls(config, **kwargs)

    from_config = from_analysis_config

    def _required(self, hook: str, detail: str) -> None:
        raise NotImplementedError(
            f"PortAdapterTemplate.{hook} must be implemented for the target repository: {detail}"
        )

    def load_policy(self, config: Any | None = None) -> Any:
        del config
        self._required(
            "load_policy",
            "load the saved PPO with env=None and expose probabilities/logits/predict/fingerprint/set_eval",
        )

    def make_env(self, config: Any | None = None, *, seed: int | None = None) -> Any:
        del config, seed
        self._required(
            "make_env",
            "call the target project's existing environment factory with its training config",
        )

    make_environment = make_env

    def reset_env(self, env: Any | None = None, *, seed: int | None = None) -> Any:
        del env, seed
        self._required(
            "reset_env",
            "return the raw reset observation and info while initializing target-lane tracking",
        )

    def close_env(self, env: Any | None = None) -> None:
        target = env if env is not None else self.env
        if target is not None and callable(getattr(target, "close", None)):
            target.close()
        if target is self.env:
            self.env = None

    close = close_env

    def preprocess_observation(self, observation: Any, info: Mapping[str, Any] | None = None) -> Any:
        del observation, info
        self._required(
            "preprocess_observation",
            "return the exact float32 model vector and explicit normalization/frame-stack metadata",
        )

    def assert_contract(self, *, expected_dimension: int | None = None, expected_actions: int | None = None) -> Mapping[str, Any]:
        del expected_dimension, expected_actions
        self._required(
            "assert_contract",
            "verify one-dimensional Box input, one Discrete action space, and model dimensions",
        )

    def verify_schema_contract(self, schema: Any, *, expected_dimension: int | None = None) -> Mapping[str, Any]:
        del schema, expected_dimension
        self._required(
            "verify_schema_contract",
            "map every source-verified schema ID/index/normalization and reject unresolved 262 fields",
        )

    def decode_action(self, action: Any, env_config: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        del action, env_config
        self._required(
            "decode_action",
            "return the target environment's confirmed steering and throttle/brake mapping",
        )

    def telemetry(
        self,
        env: Any,
        info: Mapping[str, Any] | None = None,
        *,
        phase: str = "post",
        step: int = 0,
    ) -> Mapping[str, Any]:
        del env, info, phase, step
        self._required(
            "telemetry",
            "return raw target-lane geometry, position, heading, speed, timing, and termination fields",
        )

    def source_paths(self) -> tuple[Path, ...]:
        self._required(
            "source_paths",
            "return existing target observation/action/config source files for the run manifest",
        )

    def render_frame(
        self,
        env: Any,
        *,
        step: int | None = None,
        simulation_time: float | None = None,
        pattern_id: str | None = None,
        phase: str = "post",
    ) -> Any:
        del env, step, simulation_time, pattern_id, phase
        self._required(
            "render_frame",
            "connect the target env's existing top-down render and return an RGB copy",
        )


def create_adapter(config: Any, **kwargs: Any) -> PortAdapterTemplate:
    """Config factory used by the unresolved 262 template."""

    return PortAdapterTemplate.from_analysis_config(config, **kwargs)


__all__ = ["PortAdapterTemplate", "create_adapter"]
