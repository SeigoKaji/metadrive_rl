"""Connection adapter for the repository's existing MetaDrive + SB3 stack.

The adapter is intentionally thin.  It resolves the existing experiment TOML,
calls ``env_factory.make_evaluation_env`` for the raw environment, loads PPO
without changing the model, and exposes the exact observation/action boundary
needed by the attribution core.  All heavy imports are delayed until a method
that needs them is called, so saved-result reporting and fake tests work on a
machine without MetaDrive.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np


class AdapterError(ValueError):
    """Raised when the configured model/environment boundary is unsupported."""


@dataclass(frozen=True, slots=True)
class PolicyOutput:
    """Numpy policy output shared by the real and fake adapters."""

    logits: Any
    log_probabilities: Any
    probabilities: Any
    deterministic_actions: Any
    values: Any

    @property
    def actions(self) -> Any:
        return self.deterministic_actions

    @property
    def deterministic_action(self) -> Any:
        return self.deterministic_actions

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass(frozen=True, slots=True)
class TensorPolicyOutput:
    """Tensor policy output used by optional Integrated Gradients."""

    logits: Any
    log_probabilities: Any
    probabilities: Any
    values: Any


def _flat_float_observation(observations: Any, *, dimension: int | None = None) -> np.ndarray:
    array = np.asarray(observations)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise AdapterError(f"observations must have shape (batch, features), got {array.shape}")
    if dimension is not None and array.shape[1] != dimension:
        raise AdapterError(
            f"observation dimension mismatch: expected {dimension}, got {array.shape[1]}"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise AdapterError(f"observations must be numeric, got dtype={array.dtype}")
    result = np.asarray(array, dtype=np.float32)
    if not np.all(np.isfinite(result)):
        raise AdapterError("observations contain non-finite values")
    return result


def _runtime_config(env: Any) -> Mapping[str, Any]:
    """Read the merged runtime config after the environment has applied defaults."""

    for candidate in (
        getattr(env, "config", None),
        getattr(env, "_config", None),
        getattr(getattr(env, "engine", None), "global_config", None),
    ):
        if isinstance(candidate, Mapping):
            return candidate
        to_dict = getattr(candidate, "get_dict", None)
        if callable(to_dict):
            try:
                value = to_dict()
            except Exception:
                value = None
            if isinstance(value, Mapping):
                return value
        getter = getattr(candidate, "get", None)
        if callable(getter):
            # MetaDrive Config behaves like a mapping but does not always
            # register as collections.abc.Mapping.
            try:
                return {key: candidate[key] for key in ("physics_world_step_size", "decision_repeat", "num_lasers", "num_others", "side_detector", "lane_line_detector", "random_spawn_lane_index") if key in candidate}
            except (KeyError, TypeError):
                continue
    return {}


def _first_vehicle(env: Any) -> Any | None:
    vehicle = getattr(env, "vehicle", None)
    if vehicle is None:
        vehicle = getattr(env, "agent", None)
    if vehicle is None:
        agents = getattr(env, "agents", None)
        if isinstance(agents, Mapping) and agents:
            vehicle = next(iter(agents.values()))
    return vehicle


def _environment_config_from_analysis(config: Any) -> dict[str, Any]:
    """Resolve an existing project experiment bundle without importing it early."""

    section = getattr(config, "environment", None)
    if section is None and isinstance(config, Mapping):
        section = config.get("environment", {})
    section = dict(section or {})

    for key in ("settings", "config"):
        value = section.get(key)
        if isinstance(value, Mapping):
            return dict(value)

    path_value = section.get("path") or section.get("config")
    if isinstance(path_value, Mapping):
        return dict(path_value)
    if path_value:
        try:
            from configs.experiment_config import select_experiment

            selection = select_experiment(config_path=Path(str(path_value)))
            return dict(selection.profile.evaluation_env_config)
        except Exception as error:
            raise AdapterError(f"既存experiment TOMLを解決できません: {path_value}: {error}") from error

    raise AdapterError(
        "MetaDrive adapter requires environment.settings/config or environment.path "
        "to an existing experiment TOML"
    )


def _categorical_tensors(policy: Any, observations: Any) -> tuple[Any, Any, Any]:
    """Return actor logits, critic values, and actor latent path for SB3 MlpPolicy."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - only used in incomplete envs
        raise AdapterError("PyTorch is required for SB3 policy evaluation") from error

    if not hasattr(policy, "extract_features") or not hasattr(policy, "mlp_extractor"):
        raise AdapterError("SB3 policy lacks the ActorCriticPolicy feature path")
    if getattr(policy, "share_features_extractor", True):
        features = policy.extract_features(observations)
        latent_pi, latent_vf = policy.mlp_extractor(features)
    else:
        features = policy.extract_features(observations)
        if not isinstance(features, tuple) or len(features) != 2:
            raise AdapterError("separate SB3 feature extractor output is unsupported")
        latent_pi, latent_vf = (
            policy.mlp_extractor.forward_actor(features[0]),
            policy.mlp_extractor.forward_critic(features[1]),
        )
    action_net = getattr(policy, "action_net", None)
    value_net = getattr(policy, "value_net", None)
    if action_net is None or value_net is None:
        raise AdapterError("SB3 policy does not expose action_net/value_net")
    logits = action_net(latent_pi)
    values = value_net(latent_vf).flatten()
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise AdapterError("the policy must expose one categorical action-logit row")
    return logits, values, latent_pi


class MetaDriveAdapter:
    """Adapter for a flat Box observation and one Discrete PPO action space."""

    def __init__(
        self,
        config: Any | None = None,
        *,
        env_config: Mapping[str, Any] | None = None,
        model_path: str | Path | None = None,
        device: str = "cpu",
        rl_seed: int | None = None,
    ) -> None:
        self.analysis_config = config
        self.env_config = dict(env_config) if env_config is not None else (
            _environment_config_from_analysis(config) if config is not None else None
        )
        self.model_path = Path(model_path) if model_path is not None else None
        if self.model_path is None and config is not None:
            candidate = getattr(config, "model_path", None)
            if candidate is None and isinstance(config, Mapping):
                candidate = (config.get("model") or {}).get("path")
            self.model_path = Path(candidate) if candidate else None
        self.device = str(device)
        self.rl_seed = rl_seed
        self.env: Any | None = None
        self.model: Any | None = None
        self.policy: Any | None = None
        self._target_lane_ordinals: dict[str, int | None] = {}
        self._active_scenario_seed: int | None = None

    @classmethod
    def from_analysis_config(cls, config: Any, **kwargs: Any) -> "MetaDriveAdapter":
        return cls(config, **kwargs)

    @classmethod
    def from_config(cls, config: Any, **kwargs: Any) -> "MetaDriveAdapter":
        return cls.from_analysis_config(config, **kwargs)

    @property
    def observation_dim(self) -> int:
        if self.env is not None:
            shape = getattr(getattr(self.env, "observation_space", None), "shape", None)
            if shape and len(shape) == 1:
                return int(shape[0])
        if self.model is not None:
            shape = getattr(getattr(self.model, "observation_space", None), "shape", None)
            if shape and len(shape) == 1:
                return int(shape[0])
        raise AdapterError("environment or model must be loaded before observation_dim")

    @property
    def action_count(self) -> int:
        space = getattr(self.env, "action_space", None) or getattr(self.model, "action_space", None)
        count = getattr(space, "n", None)
        if count is None:
            raise AdapterError("adapter requires one gymnasium Discrete action space")
        return int(count)

    def make_environment(self, config: Any | None = None, *, seed: int | None = None) -> Any:
        if self.env_config is None and config is not None:
            self.env_config = _environment_config_from_analysis(config)
        if self.env_config is None:
            raise AdapterError("environment config is not resolved")
        try:
            from env_factory import make_evaluation_env

            self.env = make_evaluation_env(
                seed=self.rl_seed if seed is None and self.rl_seed is not None else int(seed or 0),
                env_config=self.env_config,
            )
        except Exception as error:
            raise AdapterError(f"MetaDrive environment creation failed: {error}") from error
        return self.env

    # Common aliases keep the project-local adapter contract small for ports.
    make_env = make_environment

    def load_model(self, *, model_path: str | Path | None = None, env: Any | None = None) -> Any:
        path = Path(model_path) if model_path is not None else self.model_path
        if path is None:
            raise AdapterError("model_path is required")
        if not path.is_file():
            raise FileNotFoundError(f"model file not found: {path}")
        try:
            from stable_baselines3 import PPO

            self.model = PPO.load(str(path), env=env, device=self.device)
        except Exception as error:
            raise AdapterError(f"PPO model loading failed: {path}: {error}") from error
        self.model_path = path
        return self.model

    def load_policy(self, config: Any | None = None) -> Any:
        """Load PPO without binding it to an environment and return core policy adapter."""

        path = self.model_path
        if path is None and config is not None:
            candidate = getattr(config, "model_path", None)
            if candidate is None and isinstance(config, Mapping):
                candidate = (config.get("model") or {}).get("path")
            path = Path(candidate) if candidate else None
        if path is None:
            raise AdapterError("model_path is required to load the policy")
        if not path.is_file():
            raise FileNotFoundError(f"model file not found: {path}")
        try:
            from stable_baselines3 import PPO
            from ..policy import CategoricalPolicyAdapter

            self.model = PPO.load(str(path), device=self.device)
            self.model_path = path
            self.policy = CategoricalPolicyAdapter(self.model)
            return self.policy
        except Exception as error:
            raise AdapterError(f"PPO policy loading failed: {path}: {error}") from error

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None

    def close_env(self, env: Any | None = None) -> None:
        """Runtime hook; close exactly the environment owned by this adapter."""

        target = env if env is not None else self.env
        if target is not None and hasattr(target, "close"):
            target.close()
        if target is self.env:
            self.env = None

    @staticmethod
    def source_paths() -> tuple[Path, ...]:
        """Return project and imported-source files recorded in run manifests.

        The paths intentionally point at the existing repository and editable
        MetaDrive checkout.  The method has no imports or side effects, so a
        portable report/check can inspect the adapter without starting an
        engine.
        """

        repository = Path(__file__).resolve().parents[2]
        metadrive_root = repository.parent / "metadrive" / "metadrive"
        return (
            repository / "env_factory.py",
            repository / "configs" / "experiment_config.py",
            repository / "configs" / "official.toml",
            repository / "start_lane_env.py",
            repository / "evaluation_visualization.py",
            metadrive_root / "obs" / "state_obs.py",
            metadrive_root / "component" / "vehicle" / "base_vehicle.py",
            metadrive_root / "component" / "navigation_module" / "node_network_navigation.py",
            metadrive_root / "component" / "sensors" / "distance_detector.py",
            metadrive_root / "component" / "sensors" / "lidar.py",
            metadrive_root / "policy" / "env_input_policy.py",
        )

    @staticmethod
    def _resolve_source_path(label: str, hint: Any = None) -> Path:
        repository = Path(__file__).resolve().parents[2]
        candidates: list[Path] = []
        if hint:
            hinted = Path(str(hint))
            candidates.append(hinted if hinted.is_absolute() else repository / hinted)
        # The portable schema labels are module-relative names.  The editable
        # checkout used by this repository has one outer ``metadrive`` project
        # directory and one inner Python package directory.
        candidates.append(repository.parent / "metadrive" / label)
        candidates.append(repository.parent / label)
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        return candidates[0].resolve() if candidates else (repository / label).resolve()

    @staticmethod
    def _verify_source_hashes(preprocessing: Mapping[str, Any]) -> dict[str, Any]:
        expected = preprocessing.get("expected_source_sha256")
        if not isinstance(expected, Mapping) or not expected:
            raise AdapterError(
                "259 schema is missing preprocessing.expected_source_sha256; "
                "source-byte provenance is required before evaluation"
            )
        required_labels = {
            "metadrive/obs/state_obs.py",
            "metadrive/component/vehicle/base_vehicle.py",
            "metadrive/component/navigation_module/node_network_navigation.py",
            "metadrive/component/sensors/distance_detector.py",
            "metadrive/component/sensors/lidar.py",
            "metadrive/policy/env_input_policy.py",
        }
        missing_labels = sorted(required_labels - {str(key) for key in expected})
        if missing_labels:
            raise AdapterError(
                "259 schema source-byte provenance is incomplete; missing SHA-256 entries: "
                f"{missing_labels}"
            )
        source_paths = preprocessing.get("source_file_paths", {})
        if not isinstance(source_paths, Mapping):
            source_paths = {}
        result: dict[str, Any] = {}
        for raw_label, raw_expected in expected.items():
            label = str(raw_label)
            expected_hash = str(raw_expected).lower()
            if len(expected_hash) != 64 or any(character not in "0123456789abcdef" for character in expected_hash):
                raise AdapterError(f"invalid expected SHA-256 for source {label!r}")
            path = MetaDriveAdapter._resolve_source_path(label, source_paths.get(label))
            if not path.is_file():
                raise AdapterError(
                    f"259 source file is unavailable for byte verification: {label} ({path})"
                )
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            row = {
                "path": str(path),
                "expected": expected_hash,
                "actual": digest,
                "match": digest == expected_hash,
            }
            result[label] = row
            if digest != expected_hash:
                raise AdapterError(
                    f"259 source hash mismatch for {label}: expected={expected_hash}, "
                    f"actual={digest}, path={path}; refusing an unverified observation mapping"
                )
        return {
            "status": "verified",
            "expected_source_sha256": {str(key): str(value) for key, value in expected.items()},
            "files": result,
        }

    @staticmethod
    def _verify_official_index_contract(schema: Any) -> dict[str, Any]:
        """Reject a 259-vector with the right length but a swapped meaning.

        ``dimension == 259`` is insufficient evidence: two schemas can retain
        the same shape while moving a heading, navigation, or LiDAR meaning to
        another index.  Human-facing labels and replacement policy may be
        localized or experiment-specific, so the comparison covers the stable
        semantic fields and leaves those two fields configurable.
        """

        try:
            from .fixtures import official_schema_259

            canonical = official_schema_259()
            spec = schema.spec
        except (ImportError, AttributeError) as error:
            raise AdapterError("259 schema cannot expose spec(index) for canonical verification") from error
        fields = (
            "id",
            "group",
            "physical_quantity",
            "unit",
            "model_representation",
            "value_range",
            "normalization",
            "clip_method",
            "value_type",
            "reference",
            "related_valid_flags",
            "coupled_indices",
            "independently_replaceable",
            "ig_interpolation_allowed",
            "source",
        )
        mismatches: list[dict[str, Any]] = []
        for index in range(canonical.dimension):
            try:
                expected = canonical.spec(index)
                observed = spec(index)
            except Exception as error:
                raise AdapterError(f"259 schema cannot resolve index {index}: {error}") from error
            for field in fields:
                expected_value = getattr(expected, field)
                observed_value = getattr(observed, field)
                if expected_value != observed_value:
                    mismatches.append(
                        {
                            "index": index,
                            "field": field,
                            "expected": expected_value,
                            "observed": observed_value,
                        }
                    )
        if mismatches:
            first = mismatches[0]
            raise AdapterError(
                "259 schema semantic contract mismatch; refusing a same-dimension mapping: "
                f"index={first['index']} field={first['field']} expected={first['expected']!r} "
                f"observed={first['observed']!r} mismatches={len(mismatches)}"
            )
        return {"status": "verified", "dimension": canonical.dimension, "fields": list(fields)}

    def reset(self, *, scenario_seed: int | None = None) -> tuple[np.ndarray, Mapping[str, Any]]:
        if self.env is None:
            self.make_environment()
        self._active_scenario_seed = None if scenario_seed is None else int(scenario_seed)
        result = self.env.reset(seed=scenario_seed)
        vehicle = _first_vehicle(self.env)
        if vehicle is not None:
            self.begin_target_lane_tracking(vehicle)
        if not isinstance(result, tuple) or len(result) != 2:
            return np.asarray(result), {}
        observation, info = result
        if self._active_scenario_seed is None and isinstance(info, Mapping):
            candidate_seed = info.get("scenario_seed", info.get("seed"))
            try:
                self._active_scenario_seed = None if candidate_seed is None else int(candidate_seed)
            except (TypeError, ValueError):
                self._active_scenario_seed = None
        # Return the raw environment observation.  The runtime calls
        # preprocess_observation exactly once and records its metadata.
        return np.array(observation, copy=True), dict(info) if isinstance(info, Mapping) else {}

    def reset_env(self, env: Any | None = None, *, seed: int | None = None) -> tuple[np.ndarray, Mapping[str, Any]]:
        """Collection hook with the conventional ``reset(env, seed=...)`` signature."""

        if env is not None:
            self.env = env
        return self.reset(scenario_seed=seed)

    def render_frame(
        self,
        env: Any,
        *,
        step: int | None = None,
        simulation_time: float | None = None,
        pattern_id: str | None = None,
        phase: str = "post",
    ) -> np.ndarray:
        """Render one independent RGB copy through MetaDrive's top-down API.

        ``screen_record=False`` keeps the adapter from mutating MetaDrive's
        recorded-frame list.  Video persistence, overlays, and timing metadata
        belong to ``input_attribution.collection``/``closed_loop``.
        """

        del step, simulation_time, pattern_id, phase
        renderer = getattr(env, "render", None)
        if not callable(renderer):
            raise AdapterError("MetaDrive environment does not expose render()")
        try:
            frame = renderer(
                mode="topdown",
                screen_record=False,
                window=False,
                screen_size=(600, 600),
                camera_position=(50, 50),
            )
        except TypeError:
            # A compatible port may expose only the stable top-down keyword.
            frame = renderer(mode="topdown", screen_record=False)
        if frame is None:
            raise AdapterError("MetaDrive top-down render returned no frame")
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise AdapterError(f"MetaDrive render must return HxWx3/4 RGB data, got {array.shape}")
        array = array[:, :, :3]
        if array.dtype != np.uint8:
            if np.issubdtype(array.dtype, np.floating) and np.nanmax(array) <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0.0, 255.0).astype(np.uint8)
        return np.array(array, dtype=np.uint8, copy=True)

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, Mapping[str, Any]]:
        if self.env is None:
            raise AdapterError("environment is not initialized")
        result = self.env.step(action)
        if not isinstance(result, tuple) or len(result) != 5:
            raise AdapterError("MetaDrive environment must return Gymnasium five-tuple")
        observation, reward, terminated, truncated, info = result
        if not isinstance(info, Mapping):
            raise AdapterError("MetaDrive step info must be a mapping")
        return np.array(observation, copy=True), float(reward), bool(terminated), bool(truncated), info

    def preprocess_observation(
        self,
        observation: Any,
        info: Mapping[str, Any] | None = None,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Return the exact SB3 input and explicit preprocessing evidence.

        The official training path has no VecNormalize or frame stack.  The
        optional ``info`` argument is the runtime hook form and requests the
        metadata tuple; a one-argument call remains convenient for direct
        adapter use and returns only the copied vector.
        """

        result = _flat_float_observation(
            observation,
            dimension=self.observation_dim if self.env or self.model else None,
        )
        copied = np.array(result[0], dtype=np.float32, copy=True)
        if info is None:
            return copied
        return copied, {
            "external_normalization": False,
            "frame_stack": False,
            "dtype": "float32",
            "shape": list(copied.shape),
            "source": "MetaDrive StateObservation/LidarStateObservation -> SB3 MlpPolicy",
        }

    def predict(self, observations: Any, *, deterministic: bool = True) -> np.ndarray:
        if self.model is None:
            self.load_model(env=self.env)
        batch = _flat_float_observation(observations, dimension=self.observation_dim)
        actions, _ = self.model.predict(batch, deterministic=deterministic)
        return np.asarray(actions)

    def _policy_tensors(self, observations: Any) -> tuple[Any, Any]:
        if self.model is None:
            self.load_model(env=self.env)
        try:
            import torch
        except ImportError as error:  # pragma: no cover
            raise AdapterError("PyTorch is required for policy distribution evaluation") from error
        batch = _flat_float_observation(observations, dimension=self.observation_dim)
        tensor = torch.as_tensor(batch, dtype=torch.float32, device=self.model.device)
        return tensor, self.model.policy

    def evaluate(self, observations: Any) -> PolicyOutput:
        """Evaluate actor probabilities/logits and critic values without stepping env."""

        try:
            import torch
        except ImportError as error:  # pragma: no cover
            raise AdapterError("PyTorch is required for policy evaluation") from error
        tensor, policy = self._policy_tensors(observations)
        with torch.no_grad():
            logits, values, _ = _categorical_tensors(policy, tensor)
            log_probabilities = torch.log_softmax(logits, dim=-1)
            probabilities = torch.softmax(logits, dim=-1)
            actions = torch.argmax(logits, dim=-1)
        return PolicyOutput(
            logits=logits.detach().cpu().numpy(),
            log_probabilities=log_probabilities.detach().cpu().numpy(),
            probabilities=probabilities.detach().cpu().numpy(),
            deterministic_actions=actions.detach().cpu().numpy(),
            values=values.detach().cpu().numpy(),
        )

    def evaluate_tensors(self, observations: Any) -> TensorPolicyOutput:
        """Differentiable actor/critic path for optional Integrated Gradients."""

        try:
            import torch
        except ImportError as error:  # pragma: no cover
            raise AdapterError("PyTorch is required for gradient evaluation") from error
        if self.model is None:
            self.load_model(env=self.env)
        if not isinstance(observations, torch.Tensor):
            observations = torch.as_tensor(observations, dtype=torch.float32, device=self.model.device)
        if observations.ndim == 1:
            observations = observations.unsqueeze(0)
        if observations.ndim != 2 or observations.shape[1] != self.observation_dim:
            raise AdapterError(f"tensor observations must be (batch,{self.observation_dim})")
        logits, values, _ = _categorical_tensors(self.model.policy, observations)
        return TensorPolicyOutput(
            logits=logits,
            log_probabilities=torch.log_softmax(logits, dim=-1),
            probabilities=torch.softmax(logits, dim=-1),
            values=values,
        )

    def distribution_probabilities(self, observations: Any) -> np.ndarray:
        return np.asarray(self.evaluate(observations).probabilities)

    # PolicyLike aliases used by the portable analysis core.
    probabilities = distribution_probabilities

    def logits_tensor(self, observations: Any) -> Any:
        return self.evaluate_tensors(observations).logits

    def set_eval(self) -> None:
        """Keep direct analysis calls in the saved policy's evaluation mode."""
        if self.model is None:
            self.load_model(env=None)
        self.model.policy.set_training_mode(False)

    def fingerprint(self) -> str:
        if self.model is None:
            self.load_model(env=self.env)
        try:
            import hashlib

            digest = hashlib.sha256()
            digest.update(type(self.model.policy).__qualname__.encode("utf-8"))
            for name, tensor in sorted(self.model.policy.state_dict().items()):
                digest.update(name.encode("utf-8"))
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            return digest.hexdigest()
        except Exception as error:
            raise AdapterError(f"policy fingerprint failed: {error}") from error

    def assert_contract(self, *, expected_dimension: int | None = None, expected_actions: int | None = None) -> dict[str, Any]:
        """Validate the flat Box/Discrete boundary and return check metadata."""

        if self.env is None and self.model is None:
            raise AdapterError("load an environment or model before contract check")
        dimension = self.observation_dim
        if expected_dimension is not None and dimension != expected_dimension:
            raise AdapterError(f"observation dimension mismatch: runtime={dimension}, expected={expected_dimension}")
        actions = self.action_count
        if expected_actions is not None and actions != expected_actions:
            raise AdapterError(f"action count mismatch: runtime={actions}, expected={expected_actions}")
        space = getattr(self.env, "observation_space", None) or getattr(self.model, "observation_space", None)
        if not getattr(space, "shape", None) or len(space.shape) != 1:
            raise AdapterError("observation space must be a one-dimensional Box")
        action_space = getattr(self.env, "action_space", None) or getattr(self.model, "action_space", None)
        if not hasattr(action_space, "n"):
            raise AdapterError("action space must be a single Discrete space")
        return {
            "observation_dim": dimension,
            "observation_dtype": str(getattr(space, "dtype", "unknown")),
            "action_count": actions,
            "action_space": f"Discrete({actions})",
            "external_normalization": False,
            "preprocess": "float32 identity for vector Box",
        }

    def verify_schema_contract(self, schema: Any, *, expected_dimension: int | None = None) -> dict[str, Any]:
        """Check explicit schema coverage and the source/runtime 259 contract."""

        dimension = getattr(schema, "dimension", None)
        if dimension is None and isinstance(schema, Mapping):
            dimension = schema.get("dimension")
        try:
            schema_dimension = int(dimension)
        except (TypeError, ValueError) as error:
            raise AdapterError("schema must expose an integer dimension") from error
        if expected_dimension is not None and schema_dimension != int(expected_dimension):
            raise AdapterError(
                f"schema dimension mismatch: schema={schema_dimension}, expected={expected_dimension}"
            )
        try:
            runtime_dimension = self.observation_dim
        except AdapterError:
            runtime_dimension = None
        if runtime_dimension is not None and runtime_dimension != schema_dimension:
            raise AdapterError(
                f"schema/runtime observation dimension mismatch: schema={schema_dimension}, runtime={runtime_dimension}"
            )
        validate = getattr(schema, "validate", None)
        if callable(validate):
            validate()
        preprocessing = getattr(schema, "preprocessing", {})
        preprocessing = preprocessing if isinstance(preprocessing, Mapping) else {}
        source_files = tuple(str(value) for value in preprocessing.get("source_files", ()))
        source_text = " ".join(source_files)
        required_sources = (
            "state_obs.py",
            "base_vehicle.py",
            "node_network_navigation.py",
            "distance_detector.py",
            "lidar.py",
            "env_input_policy.py",
        )
        source_evidence = {
            "source_contract": preprocessing.get("source_contract"),
            "source_files": list(source_files),
            "required_sources_present": {
                value: value in source_text for value in required_sources
            },
        }
        if schema_dimension == 259 and not all(source_evidence["required_sources_present"].values()):
            missing = [
                value
                for value, present in source_evidence["required_sources_present"].items()
                if not present
            ]
            raise AdapterError(f"259 schema source evidence is incomplete: {missing}")
        if schema_dimension == 259:
            source_evidence["canonical_index_contract"] = self._verify_official_index_contract(schema)
            source_evidence["byte_hashes"] = self._verify_source_hashes(preprocessing)
        runtime_checks: dict[str, Any]
        if self.env is None:
            runtime_checks = {
                "status": "deferred_until_environment_probe",
                "observation_defaults": None,
            }
        else:
            runtime = _runtime_config(self.env)
            vehicle_config = runtime.get("vehicle_config", {})
            if hasattr(vehicle_config, "get_dict"):
                vehicle_config = vehicle_config.get_dict()
            if not isinstance(vehicle_config, Mapping):
                vehicle_config = {}

            def nested(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
                value = config.get(key, {})
                if hasattr(value, "get_dict"):
                    value = value.get_dict()
                return value if isinstance(value, Mapping) else {}

            lidar = nested(vehicle_config, "lidar")
            side_detector = nested(vehicle_config, "side_detector")
            lane_line_detector = nested(vehicle_config, "lane_line_detector")
            observed_defaults = {
                "num_lasers": lidar.get("num_lasers"),
                "num_others": lidar.get("num_others"),
                "side_detector_num_lasers": side_detector.get("num_lasers"),
                "lane_line_detector_num_lasers": lane_line_detector.get("num_lasers"),
                "random_spawn_lane_index": runtime.get("random_spawn_lane_index"),
            }
            expected_defaults = {
                "num_lasers": 240,
                "num_others": 0,
                "side_detector_num_lasers": 0,
                "lane_line_detector_num_lasers": 0,
                "random_spawn_lane_index": False,
            }
            mismatches = {
                key: {"expected": value, "observed": observed_defaults.get(key)}
                for key, value in expected_defaults.items()
                if observed_defaults.get(key) != value
            }
            if mismatches and schema_dimension == 259:
                raise AdapterError(f"259 runtime observation defaults mismatch: {mismatches}")
            runtime_checks = {
                "status": "verified",
                "observation_defaults": observed_defaults,
                "expected_defaults": expected_defaults,
                "mismatches": mismatches,
            }
        return {
            "schema_dimension": schema_dimension,
            "runtime_dimension": runtime_dimension,
            "source_evidence": source_evidence,
            "runtime_checks": runtime_checks,
            "verified": runtime_dimension is None or runtime_dimension == schema_dimension,
        }

    def decode_action(self, action: Any, env_config: Mapping[str, Any] | None = None) -> Any:
        try:
            from evaluation_visualization import decode_discrete_action

            return decode_discrete_action(action, env_config or self.env_config or {})
        except Exception as error:
            raise AdapterError(f"discrete action decode failed: {error}") from error

    @staticmethod
    def resolve_target_lane_state(vehicle: Any, *, target_ordinal: int | None, tolerance_ratio: float = 0.05) -> Any:
        try:
            from start_lane_env import resolve_target_lane_state

            return resolve_target_lane_state(
                vehicle,
                target_ordinal=target_ordinal,
                tolerance_ratio=float(tolerance_ratio),
            )
        except Exception as error:
            raise AdapterError(f"target lane geometry lookup failed: {error}") from error

    @staticmethod
    def target_lane_ordinal(vehicle: Any) -> int | None:
        try:
            value = getattr(vehicle, "lane_index", None)
            if value is None:
                value = vehicle
            index = value if isinstance(value, (tuple, list)) else getattr(value, "index", None)
            ordinal = index[-1]
            return None if isinstance(ordinal, bool) else int(ordinal)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            return None

    def begin_target_lane_tracking(self, vehicle: Any, *, vehicle_id: str = "agent0") -> int | None:
        ordinal = self.target_lane_ordinal(vehicle)
        self._target_lane_ordinals[vehicle_id] = ordinal
        return ordinal

    def target_lane_telemetry(self, vehicle: Any, *, vehicle_id: str = "agent0") -> dict[str, Any]:
        ordinal = self._target_lane_ordinals.get(vehicle_id)
        state = self.resolve_target_lane_state(vehicle, target_ordinal=ordinal)
        if hasattr(state, "__dataclass_fields__"):
            result = {
                "target_lane_valid": state.valid,
                "target_lane_ordinal": state.target_ordinal,
                "current_lane_ordinal": state.current_ordinal,
                "target_lane_offset_m": state.target_lane_offset_m,
                "normalized_target_lane_error": state.normalized_error,
                "in_target_lane": state.in_target_lane,
            }
            if state.valid:
                try:
                    target_lane = next(
                        lane
                        for lane in vehicle.navigation.current_ref_lanes
                        if self.target_lane_ordinal(lane) == state.target_ordinal
                    )
                    _longitude, _lateral = target_lane.local_coordinates(vehicle.position)
                    heading_fn = getattr(target_lane, "heading_theta_at", None)
                    if callable(heading_fn):
                        result["target_lane_heading_rad"] = float(heading_fn(float(_longitude)))
                    else:
                        direction = getattr(target_lane, "direction", None)
                        if direction is not None:
                            result["target_lane_heading_rad"] = float(
                                math.atan2(float(direction[1]), float(direction[0]))
                            )
                except (AttributeError, IndexError, KeyError, StopIteration, TypeError, ValueError):
                    result["target_lane_heading_rad"] = None
            else:
                result["target_lane_heading_rad"] = None
            return result
        return dict(state)

    def runtime_telemetry(self, vehicle: Any, *, vehicle_id: str = "agent0") -> dict[str, Any]:
        try:
            from evaluation_visualization import read_runtime_road_metrics

            road = read_runtime_road_metrics(vehicle)
            result = {
                "lane_width_m": road.lane_width_m,
                "lane_count_one_way": road.lane_count_one_way,
                "current_segment_drivable_width_m": road.current_segment_drivable_width_m,
                "current_segment_width_source": road.current_segment_width_source,
                "center_to_left_boundary_m": road.center_to_left_boundary_m,
                "center_to_right_boundary_m": road.center_to_right_boundary_m,
                "lane_center_offset_m": road.lane_center_offset_m,
            }
        except Exception:
            result = {}
        navigation = getattr(vehicle, "navigation", None)
        if navigation is not None:
            travelled_length = getattr(navigation, "travelled_length", None)
            try:
                if travelled_length is not None and math.isfinite(float(travelled_length)):
                    # This is NodeNetworkNavigation's accumulated route
                    # distance.  It is read directly from the raw vehicle;
                    # no value is reconstructed from an intervened input.
                    result["route_progress_m"] = float(travelled_length)
            except (TypeError, ValueError):
                pass
            route_completion = getattr(navigation, "route_completion", None)
            try:
                if callable(route_completion):
                    route_completion = route_completion()
                if route_completion is not None and math.isfinite(float(route_completion)):
                    result["route_completion"] = float(route_completion)
            except (TypeError, ValueError, ZeroDivisionError):
                pass
        heading_theta = getattr(vehicle, "heading_theta", None)
        try:
            if heading_theta is not None and math.isfinite(float(heading_theta)):
                result["heading_rad"] = float(heading_theta)
        except (TypeError, ValueError):
            pass
        result.update(self.target_lane_telemetry(vehicle, vehicle_id=vehicle_id))
        try:
            current_ref_lanes = vehicle.navigation.current_ref_lanes
            current_lane = current_ref_lanes[0]
            lane_index = getattr(current_lane, "index", None)
            if lane_index is not None:
                result["road_segment_id"] = list(lane_index[:2])
        except (AttributeError, IndexError, KeyError, TypeError):
            result["road_segment_id"] = None
        return result

    def telemetry(
        self,
        env: Any,
        info: Mapping[str, Any] | None = None,
        *,
        phase: str = "post",
        step: int = 0,
    ) -> dict[str, Any]:
        """Runtime hook with simulation time, action, road, and target-lane data."""

        del phase
        source = dict(info or {})
        config = _runtime_config(env)
        physics_value = config.get("physics_world_step_size")
        repeat_value = config.get("decision_repeat")
        try:
            physics_step = float(physics_value) if physics_value is not None else None
            decision_repeat = int(repeat_value) if repeat_value is not None else None
        except (TypeError, ValueError):
            physics_step = None
            decision_repeat = None
        action_duration = (
            physics_step * decision_repeat
            if physics_step is not None and decision_repeat is not None
            else None
        )
        simulation_time = source.get("simulation_time", source.get("sim_time_s"))
        if simulation_time is None and action_duration is not None:
            simulation_time = float(step) * action_duration
        result: dict[str, Any] = {
            **source,
            "step": int(step),
            "sim_time_seconds": None if simulation_time is None else float(simulation_time),
            "sim_time_s": None if simulation_time is None else float(simulation_time),
            "action_duration_seconds": action_duration,
            "physics_hz": 1.0 / physics_step if physics_step and physics_step > 0 else None,
            "control_hz": 1.0 / action_duration if action_duration and action_duration > 0 else None,
            "actual_scenario_seed": self._active_scenario_seed,
        }
        vehicle = _first_vehicle(env)
        if vehicle is not None:
            if "velocity" not in result:
                speed_km_h = getattr(vehicle, "speed_km_h", None)
                if speed_km_h is not None:
                    result["velocity"] = float(speed_km_h) / 3.6
            result.update(self.runtime_telemetry(vehicle))
            position = getattr(vehicle, "position", None)
            if position is not None:
                try:
                    result["position_xy_m"] = [float(position[0]), float(position[1])]
                except (IndexError, TypeError, ValueError):
                    pass
        if "velocity" in result:
            try:
                result["speed_m_s"] = float(result["velocity"])
                result["speed_km_h"] = float(result["velocity"]) * 3.6
            except (TypeError, ValueError):
                pass
        return result


PolicyAdapter = MetaDriveAdapter


def create_adapter(config: Any, **kwargs: Any) -> MetaDriveAdapter:
    """Portable config factory used by the attribution CLI."""

    model = getattr(config, "model", None)
    if model is None and isinstance(config, Mapping):
        model = config.get("model", {})
    model = model if isinstance(model, Mapping) else {}
    if "device" in model and "device" not in kwargs:
        kwargs["device"] = model["device"]
    scenario = getattr(config, "scenario", None)
    if scenario is None and isinstance(config, Mapping):
        scenario = config.get("scenario", {})
    scenario = scenario if isinstance(scenario, Mapping) else {}
    if "rl_seed" in scenario and "rl_seed" not in kwargs:
        kwargs["rl_seed"] = int(scenario["rl_seed"])
    return MetaDriveAdapter.from_analysis_config(config, **kwargs)


__all__ = [
    "AdapterError",
    "create_adapter",
    "MetaDriveAdapter",
    "PolicyAdapter",
    "PolicyOutput",
    "TensorPolicyOutput",
]
