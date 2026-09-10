"""Synthetic contract tests for the host input attribution adapter.

These tests construct a tiny SB3 PPO policy without learning or starting
MetaDrive.  The host-specific methods are exercised with a small fake raw
environment, which keeps the test suite useful on machines that do not have
the simulator assets installed.
"""

from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from input_attribution.adapter import (
    AdapterContractError,
    InputAttributionAdapter,
    ModelInputBoundaryError,
    UnsupportedAdapterError,
)


class TinyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, *, dimension: int = 4, actions: int = 3) -> None:
        self.observation_space = gym.spaces.Box(
            low=np.zeros(dimension, dtype=np.float32),
            high=np.ones(dimension, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(actions)
        self.config = {
            "physics_world_step_size": 0.02,
            "decision_repeat": 5,
        }
        self.current_seed = 5
        self.episode_step = 0
        self.agents = {
            "agent0": SimpleNamespace(
                position=np.array([1.0, 2.0], dtype=np.float32),
                heading=np.array([1.0, 0.0], dtype=np.float32),
                heading_theta=0.0,
                velocity=np.array([2.0, 0.0], dtype=np.float32),
                speed=2.0,
                speed_km_h=7.2,
                steering=0.0,
                throttle_brake=0.0,
                current_action=(0.0, 0.0),
                last_current_action=((0.0, 0.0), (0.0, 0.0)),
                lane_index=("a", "b", 1),
            )
        }
        self.top_down_renderer = None

    def reset(self, *, seed: int | None = None, options: object = None):
        del options
        if seed is not None:
            self.current_seed = int(seed)
        self.episode_step = 0
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action: int):
        del action
        self.episode_step += 1
        return (
            np.zeros(self.observation_space.shape, dtype=np.float32),
            0.0,
            False,
            False,
            {"velocity": 2.0},
        )

    def render(self, **kwargs: object) -> np.ndarray:
        del kwargs
        return np.zeros((3, 4, 3), dtype=np.uint8)


def _ppo_model(dimension: int = 4, *, actions: int = 3):
    from stable_baselines3 import PPO

    return PPO(
        "MlpPolicy",
        TinyEnv(dimension=dimension, actions=actions),
        n_steps=8,
        batch_size=4,
        verbose=0,
        device="cpu",
    )


def _adapter(dimension: int = 4, **kwargs: object) -> InputAttributionAdapter:
    return InputAttributionAdapter(
        model=_ppo_model(dimension),
        observation_dim=dimension,
        **kwargs,
    )


def test_prepare_is_independent_float32_copy_and_allows_stress_values() -> None:
    adapter = _adapter()
    raw = np.linspace(0.0, 1.0, 4, dtype=np.float32)
    prepared = adapter.prepare(raw)
    assert prepared.dtype == np.float32
    assert prepared.shape == (4,)
    assert prepared is not raw
    prepared[0] = -100.0
    assert raw[0] == 0.0
    np.testing.assert_array_equal(
        adapter.prepare(np.array([-1.0, -100.0, 0.5, 1.0], dtype=np.float32)),
        np.array([-1.0, -100.0, 0.5, 1.0], dtype=np.float32),
    )


def test_predict_returns_distribution_and_matches_ordinary_predict() -> None:
    adapter = _adapter()
    observation = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    probabilities, action = adapter.predict(observation)
    assert probabilities.shape == (3,)
    assert np.isfinite(probabilities).all()
    assert float(probabilities.sum()) == pytest.approx(1.0)
    captured = adapter.last_mlp_inputs
    assert len(captured) >= 2
    for observed in captured:
        np.testing.assert_array_equal(observed, observation[None, :])
    assert action == adapter.ordinary_predict(observation)
    assert action == int(np.argmax(probabilities))


def test_predict_supports_batch_and_exact_negative_stress_reaches_mlp() -> None:
    adapter = _adapter()
    batch = np.array(
        [[-1.0, 0.25, 0.5, 0.75], [-100.0, 0.0, 1.0, 0.2]],
        dtype=np.float32,
    )
    probabilities, actions = adapter.predict(batch)
    assert probabilities.shape == (2, 3)
    assert actions.shape == (2,)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert len(adapter.last_mlp_inputs) >= 2
    for observed in adapter.last_mlp_inputs:
        np.testing.assert_array_equal(observed, batch)


def test_probe_detects_a_clipping_preprocessor_at_the_mlp_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter()
    extractor = adapter.model.policy.features_extractor
    original_forward = extractor.forward

    import torch

    def clipped_forward(observations: torch.Tensor) -> torch.Tensor:
        return torch.clamp(original_forward(observations), min=0.0, max=1.0)

    monkeypatch.setattr(extractor, "forward", clipped_forward)
    with pytest.raises(ModelInputBoundaryError, match="MLP input differs"):
        adapter.predict(
            np.array([-1.0, 0.25, 0.5, 0.75], dtype=np.float32)
        )


def test_attached_observation_statistics_are_rejected_as_double_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _ppo_model()
    monkeypatch.setattr(
        model,
        "get_env",
        lambda: SimpleNamespace(obs_rms=object()),
    )
    with pytest.raises(UnsupportedAdapterError, match="VecNormalize"):
        InputAttributionAdapter(model=model, observation_dim=4)


def test_non_finite_shape_and_dtype_are_rejected() -> None:
    adapter = _adapter()
    with pytest.raises(AdapterContractError, match="shape"):
        adapter.prepare(np.zeros(3, dtype=np.float32))
    with pytest.raises(AdapterContractError, match="dtype"):
        adapter.prepare(np.zeros(4, dtype=np.float64))
    with pytest.raises(AdapterContractError, match="NaN or Inf"):
        adapter.prepare(np.array([np.nan, 0.0, 0.0, 0.0], dtype=np.float32))


def test_model_schema_dimension_and_normalization_contracts_are_strict() -> None:
    with pytest.raises(AdapterContractError, match="schema/model"):
        _adapter(schema={"dimension": 5})
    with pytest.raises(UnsupportedAdapterError, match="only identity"):
        _adapter(config={"preprocessing": "zscore"})
    with pytest.raises(UnsupportedAdapterError, match="VecNormalize"):
        _adapter(config={"vec_normalize": True})


def test_259_model_requires_explicit_semantic_schema() -> None:
    with pytest.raises(UnsupportedAdapterError, match="explicit input schema"):
        InputAttributionAdapter(model=_ppo_model(259))


def test_verified_259_schema_records_installed_source_hashes() -> None:
    from input_attribution.schema import standard_259_schema

    adapter = InputAttributionAdapter(
        model=_ppo_model(259, actions=9),
        schema=standard_259_schema(),
    )
    schema_metadata = adapter.model_metadata()["schema"]
    assert schema_metadata["official_contract"] is True
    assert schema_metadata["source_hash_check"]["status"] == "verified"
    assert schema_metadata["source_hash_check"]["missing"] == []
    assert schema_metadata["source_hash_check"]["mismatched"] == {}


def test_verified_259_schema_checks_merged_detector_runtime_config() -> None:
    from input_attribution.schema import standard_259_schema

    class OfficialTinyEnv(TinyEnv):
        def __init__(self) -> None:
            super().__init__(dimension=259, actions=9)
            self.config.update(
                {
                    "discrete_action": True,
                    "discrete_throttle_dim": 3,
                    "discrete_steering_dim": 3,
                    "num_agents": 1,
                    "traffic_density": 0.0,
                    "random_spawn_lane_index": False,
                    "random_lane_width": False,
                    "random_lane_num": False,
                    "accident_prob": 0.0,
                    "vehicle_config": {
                        "lidar": {
                            "num_lasers": 240,
                            "distance": 50,
                            "num_others": 0,
                        },
                        "side_detector": {"num_lasers": 0, "distance": 50},
                        "lane_line_detector": {"num_lasers": 0, "distance": 20},
                    },
                }
            )

    adapter = InputAttributionAdapter(
        model=_ppo_model(259, actions=9),
        schema=standard_259_schema(),
        env_config={"map": "C"},
        env_factory=lambda _config: OfficialTinyEnv(),
    )
    adapter.make_env()
    environment_metadata = adapter.model_metadata()["environment"]
    assert environment_metadata["verified_observation_config"]["standard_dimension"] == 259


def test_make_env_copies_config_and_checks_actual_spaces() -> None:
    adapter = _adapter(env_config={"map": "C", "start_seed": 5})
    received: list[dict[str, object]] = []
    fake_env = TinyEnv()

    def factory(config: dict[str, object]) -> TinyEnv:
        received.append(config)
        return fake_env

    adapter._env_factory = factory
    returned = adapter.make_env()
    assert returned is fake_env
    assert received == [{"map": "C", "start_seed": 5}]
    assert received[0] is not adapter.env_config
    assert adapter.metadata["environment"]["actual_observation_shape"] == [4]
    assert adapter.metadata["environment"]["actual_action_count"] == 3


def test_make_env_rejects_dimension_or_action_mismatch() -> None:
    adapter = _adapter()

    def wrong_dimension(_config: dict[str, object]) -> TinyEnv:
        return TinyEnv(dimension=5)

    adapter._env_factory = wrong_dimension
    with pytest.raises(AdapterContractError, match="dimensions disagree"):
        adapter.make_env()

    def wrong_action(_config: dict[str, object]) -> TinyEnv:
        return TinyEnv(actions=2)

    adapter._env_factory = wrong_action
    with pytest.raises(AdapterContractError, match="action counts disagree"):
        adapter.make_env()


def test_make_env_rejects_normalization_or_observation_transform_wrapper_with_matching_spaces() -> None:
    class VecNormalizeLike:
        def __init__(self, env: TinyEnv) -> None:
            self.env = env
            self.observation_space = env.observation_space
            self.action_space = env.action_space
            self.obs_rms = SimpleNamespace(mean=np.zeros(4), var=np.ones(4))
            self.ret_rms = SimpleNamespace(mean=0.0, var=1.0)

    class ObservationTransform:
        def __init__(self, env: TinyEnv) -> None:
            self.env = env
            self.observation_space = env.observation_space
            self.action_space = env.action_space

        def observation(self, value: np.ndarray) -> np.ndarray:
            return np.asarray(value, dtype=np.float32)

    for wrapper in (VecNormalizeLike, ObservationTransform):
        adapter = _adapter()
        adapter._env_factory = lambda _config, wrapper=wrapper: wrapper(TinyEnv())
        with pytest.raises(UnsupportedAdapterError, match="VecNormalize|observation"):
            adapter.make_env()


def _explicit_verified_schema(dimension: int) -> list[dict[str, object]]:
    """Create a complete custom host schema whose rows are explicitly verified."""

    return [
        {
            "index": index,
            "name": f"custom_feature_{index:03d}",
            "group": "custom_verified_host",
            "dtype": "float32",
            "normalization": "host supplied [0, 1]",
            "normal_range": [0.0, 1.0],
            "space_range": [0.0, 1.0],
            "zero_meaning": "host verified lower endpoint",
            "one_meaning": "host verified upper endpoint",
            "source": "tests.test_adapter explicit host contract",
            "source_version": "test-fixture-v1",
            "source_path": "input_attribution/tests/test_adapter.py",
            "source_lines": "fixture",
            "status": "verified",
        }
        for index in range(dimension)
    ]


def test_real_262_model_rejects_all_unknown_host_template() -> None:
    from input_attribution.schema import template_262_schema

    with pytest.raises(UnsupportedAdapterError, match="complete verified"):
        InputAttributionAdapter(
            model=_ppo_model(262),
            schema=template_262_schema(),
        )


def test_real_262_model_accepts_complete_explicit_verified_schema() -> None:
    adapter = InputAttributionAdapter(
        model=_ppo_model(262),
        schema=_explicit_verified_schema(262),
    )
    assert adapter.dimension == 262
    assert adapter.action_count == 3
    assert adapter.model_metadata()["schema"]["verified"] is True


def test_snapshot_is_jsonable_and_frame_preserves_existing_recording_frames() -> None:
    adapter = _adapter()

    class Renderer:
        def __init__(self) -> None:
            self._screen_frames = ["prior-frame"]

    class RenderEnv(TinyEnv):
        def __init__(self) -> None:
            super().__init__()
            self.top_down_renderer = Renderer()
            self.render_calls: list[dict[str, object]] = []

        def render(self, **kwargs: object) -> np.ndarray:
            self.render_calls.append(kwargs)
            self.top_down_renderer._screen_frames.append("accidental-frame")
            return np.full((2, 3, 3), 128, dtype=np.uint8)

    env = RenderEnv()
    snapshot = adapter.snapshot(env)
    assert snapshot["scenario_seed"] == 5
    assert snapshot["agent"]["position"] == [1.0, 2.0]
    assert adapter.action_dt(env) == pytest.approx(0.1)
    frame = adapter.frame(env)
    assert frame.shape == (2, 3, 3)
    assert frame.dtype == np.uint8
    assert env.top_down_renderer._screen_frames == ["prior-frame"]
    assert env.render_calls[0]["screen_record"] is False


def test_model_metadata_contains_import_provenance() -> None:
    # A loaded object has no file provenance; the metadata still records the
    # runtime, spaces, policy, and explicit preprocessing boundary.
    metadata = _adapter().model_metadata()
    assert metadata["preprocessing"]["raw_to_model"] == "identity"
    assert metadata["preprocessing"]["verified"] is True
    assert metadata["runtime"]["python_executable"]
    assert metadata["observation"]["shape"] == [4]
    assert metadata["action"]["n"] == 3


def test_input_probe_is_removed_after_failure() -> None:
    adapter = _adapter()
    original = adapter._policy_probabilities

    def fail(_batch: np.ndarray) -> np.ndarray:
        raise RuntimeError("synthetic failure")

    adapter._policy_probabilities = fail  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="synthetic failure"):
        adapter.predict(np.zeros(4, dtype=np.float32))
    adapter._policy_probabilities = original  # type: ignore[method-assign]
    probabilities, action = adapter.predict(np.zeros(4, dtype=np.float32))
    assert probabilities.shape == (3,)
    assert isinstance(action, int)
