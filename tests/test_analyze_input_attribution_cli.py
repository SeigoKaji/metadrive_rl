"""CLI parsing/template behavior that does not need MetaDrive startup."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import analyze_input_attribution as cli
from input_attribution.config import (
    AggregationConfig,
    AnalysisConfig,
    BaselineConfig,
    ClosedLoopConfig,
    CollectionConfig,
    IntegratedGradientsConfig,
    PerturbationConfig,
    PhaseConfig,
    RunConfig,
)
from input_attribution.closed_loop import ClosedLoopResult
from input_attribution.rollout import RolloutData
from input_attribution.perturbation import PerturbationTarget
from input_attribution.schema import Feature, ObservationSchema
from input_attribution.schema import ObservationSchemaError, load_observation_schema


def test_help_and_schema_template_work_without_runtime_environment(tmp_path: Path) -> None:
    output = tmp_path / "generic_7.toml"

    assert cli.main(["schema-template", "--dim", "7", "--output", str(output)]) == 0
    assert output.is_file()
    with pytest.raises(ObservationSchemaError, match="unresolved"):
        load_observation_schema(output)
    schema = load_observation_schema(output, allow_unresolved=True)
    assert schema.observation_dim == 7
    assert len(schema.unresolved_features) == 7

    result = subprocess.run(
        [sys.executable, "analyze_input_attribution.py", "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert "schema-template" in result.stdout
    assert "closed-loop" in result.stdout


def test_versions_record_source_checkout_metadrive_version() -> None:
    assert cli._versions()["metadrive"] is not None


def test_runtime_subcommands_require_their_explicit_arguments() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(["run"])
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "run",
                "--config",
                "configs/official.toml",
                "--schema",
                "observation_schemas/metadrive_default_259.toml",
                "--analysis-config",
                "attribution_configs/official_left_curve.toml",
                "--output-prefix",
                "safe",
                "--out",
                "bad",
            ]
        )


@pytest.mark.parametrize("prefix", ("../escape", "nested/name", r"nested\name", "..", ""))
def test_output_prefix_rejects_traversal(prefix: str) -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "run",
                "--config",
                "configs/official.toml",
                "--schema",
                "observation_schemas/metadrive_default_259.toml",
                "--analysis-config",
                "attribution_configs/official_left_curve.toml",
                "--output-prefix",
                prefix,
            ]
        )


def test_missing_model_reports_a_clear_error_without_publishing_output(tmp_path: Path) -> None:
    return_code = cli.main(
        [
            "collect",
            "--config",
            "configs/official.toml",
            "--model",
            str(tmp_path / "missing_model.zip"),
            "--schema",
            "observation_schemas/metadrive_default_259.toml",
            "--analysis-config",
            "attribution_configs/official_left_curve.toml",
            "--output-prefix",
            "missing_model_test",
        ]
    )
    assert return_code == 2


class _ToyAttributionAdapter:
    observation_dim = 2
    action_count = 2
    device = torch.device("cpu")

    def observations_to_tensor(self, observations: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(observations, dtype=torch.float32)

    def evaluate(self, observations: np.ndarray) -> SimpleNamespace:
        values = np.asarray(observations, dtype=np.float32)
        logits = np.stack((values[:, 0], values[:, 1]), axis=1)
        log_probabilities = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
        probabilities = np.exp(log_probabilities)
        return SimpleNamespace(
            logits=logits,
            log_probabilities=log_probabilities,
            probabilities=probabilities,
            deterministic_actions=np.argmax(logits, axis=1),
            values=values.sum(axis=1),
        )

    def evaluate_tensors(self, observations: torch.Tensor) -> SimpleNamespace:
        logits = torch.stack((observations[:, 0], observations[:, 1]), dim=1)
        return SimpleNamespace(
            logits=logits,
            log_probabilities=torch.log_softmax(logits, dim=1),
            probabilities=torch.softmax(logits, dim=1),
            deterministic_actions=torch.argmax(logits, dim=1),
            values=observations.sum(dim=1),
        )


def _toy_analysis_config() -> AnalysisConfig:
    return AnalysisConfig(
        schema_version=1,
        name="toy",
        run=RunConfig(deterministic=True, device="cpu", record_visualization=False),
        collection=CollectionConfig(save_observations=True),
        baseline=BaselineConfig(strategy="episode_start", count=1, seed=0),
        perturbation=PerturbationConfig(
            enabled=True,
            analyze_features=True,
            analyze_groups=True,
            batch_size=8,
            lidar_sector_degrees=15,
        ),
        integrated_gradients=IntegratedGradientsConfig(
            enabled=True,
            targets=("selected_log_probability", "critic_value"),
            steps=2,
            batch_size=8,
        ),
        aggregation=AggregationConfig(progress_bins=2),
        closed_loop=ClosedLoopConfig(
            enabled=False,
            top_k_features=0,
            top_k_groups=0,
            replacement_strategy="episode_start_constant",
        ),
        phases=(PhaseConfig("later", start_step=2, end_step=3),),
    )


def _toy_rollout() -> RolloutData:
    observations = np.asarray([[1.0, 0.0], [1.0, 2.0], [0.5, 1.5], [3.0, 1.0]], dtype=np.float32)
    rows = observations.shape[0]
    return RolloutData(
        observations=observations,
        logits=np.zeros((rows, 2), dtype=np.float32),
        log_probabilities=np.zeros((rows, 2), dtype=np.float32),
        probabilities=np.full((rows, 2), 0.5, dtype=np.float32),
        values=np.zeros(rows, dtype=np.float32),
        selected_actions=np.zeros(rows, dtype=np.int64),
        env_actions=np.zeros(rows, dtype=np.int64),
        episode_ids=np.asarray([1, 1, 2, 2], dtype=np.int64),
        scenario_seeds=np.asarray([5, 5, 6, 6], dtype=np.int64),
        steps=np.asarray([1, 2, 1, 2], dtype=np.int64),
        rewards=np.zeros(rows, dtype=np.float32),
        cumulative_rewards=np.zeros(rows, dtype=np.float32),
        terminated=np.asarray([False, True, False, True]),
        truncated=np.zeros(rows, dtype=bool),
        dones=np.asarray([False, True, False, True]),
        step_records=tuple({"step": index + 1} for index in range(rows)),
    )


def _selection_schema() -> ObservationSchema:
    return ObservationSchema(
        schema_version=1,
        name="selection_schema",
        observation_dim=4,
        features=(
            Feature(0, "low_js_feature", "state", groups=("semantic_group",)),
            Feature(1, "high_js_feature", "state", groups=("semantic_group",)),
            Feature(2, "lidar_a", "lidar", kind="lidar", angle_deg=0.0),
            Feature(3, "lidar_b", "lidar", kind="lidar", angle_deg=10.0),
        ),
        groups={"semantic_group": (0, 1)},
    )


def _closed_loop_selection_config(*, perturbation_enabled: bool) -> AnalysisConfig:
    base = _toy_analysis_config()
    return replace(
        base,
        perturbation=replace(base.perturbation, enabled=perturbation_enabled),
        # This proves top-K never derives its selections from IG rows.
        integrated_gradients=replace(base.integrated_gradients, enabled=False),
        closed_loop=ClosedLoopConfig(
            enabled=True,
            top_k_features=1,
            top_k_groups=1,
            replacement_strategy="episode_start_constant",
        ),
    )


def test_closed_loop_top_k_uses_perturbation_js_and_preserves_lidar_sector_indices() -> None:
    schema = _selection_schema()
    perturbation_result = SimpleNamespace(
        targets=(
            PerturbationTarget("low_js_feature", "feature", (0,)),
            PerturbationTarget("high_js_feature", "feature", (1,)),
            PerturbationTarget("semantic_group", "group", (0, 1)),
            # The schema intentionally has no group by this generated name.
            PerturbationTarget("lidar_000.0_015.0_deg", "lidar_sector", (2, 3)),
        )
    )
    feature_rows = (
        {
            "scope": "full_episode",
            "baseline_scope": "mean_over_baselines",
            "target_kind": "feature",
            "target_id": "feature:low_js_feature",
            "mean_js_divergence": 0.1,
            "mean_absolute_ig": 99.0,
        },
        {
            "scope": "full_episode",
            "baseline_scope": "mean_over_baselines",
            "target_kind": "feature",
            "target_id": "feature:high_js_feature",
            "mean_js_divergence": 0.9,
            "mean_absolute_ig": 0.0,
        },
        {
            "scope": "episode",
            "baseline_scope": "mean_over_baselines",
            "target_kind": "feature",
            "target_id": "feature:low_js_feature",
            "mean_js_divergence": 100.0,
        },
    )
    group_rows = (
        {
            "scope": "full_episode",
            "baseline_scope": "mean_over_baselines",
            "target_kind": "group",
            "target_id": "group:semantic_group",
            "mean_js_divergence": 0.2,
        },
        {
            "scope": "full_episode",
            "baseline_scope": "mean_over_baselines",
            "target_kind": "lidar_sector",
            "target_id": "lidar_sector:lidar_000.0_015.0_deg",
            "mean_js_divergence": 0.8,
        },
    )

    selected = cli._select_closed_loop_targets(
        schema=schema,
        analysis=_closed_loop_selection_config(perturbation_enabled=True),
        feature_rows=feature_rows,
        group_rows=group_rows,
        perturbation_result=perturbation_result,
    )

    assert [(target.name, target.kind, target.indices) for target in selected] == [
        ("high_js_feature", "feature", (1,)),
        ("lidar_000.0_015.0_deg", "lidar_sector", (2, 3)),
    ]


def test_closed_loop_without_perturbation_requires_explicit_selector() -> None:
    with pytest.raises(cli.AttributionCLIError, match="perturbation is disabled"):
        cli._select_closed_loop_targets(
            schema=_selection_schema(),
            analysis=_closed_loop_selection_config(perturbation_enabled=False),
            feature_rows=(),
            group_rows=(),
            perturbation_result=None,
        )


def _provenance_runtime(tmp_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    model_path = tmp_path / "model.zip"
    schema_path = tmp_path / "schema.toml"
    model_path.write_bytes(b"same-shape-model-a")
    schema_path.write_text("schema-a", encoding="utf-8")
    runtime: dict[str, object] = {
        "model_path": model_path,
        "schema": SimpleNamespace(observation_dim=2, source_path=schema_path),
        "adapter": SimpleNamespace(observation_dim=2, action_count=2),
        "experiment": SimpleNamespace(source_sha256="experiment-sha-a"),
        "analysis": _toy_analysis_config(),
    }
    metadata: dict[str, object] = {
        "model": {"sha256": cli.sha256_file(model_path)},
        "experiment": {"sha256": "experiment-sha-a"},
        "schema": {"sha256": cli.sha256_file(schema_path), "observation_dim": 2},
        "action_dim": 2,
        "model_observation_dim": 2,
        "deterministic": True,
    }
    return runtime, metadata


def test_offline_provenance_rejects_same_dimension_different_model_and_schema(
    tmp_path: Path,
) -> None:
    runtime, metadata = _provenance_runtime(tmp_path)
    base_rollout = replace(_toy_rollout(), metadata=metadata)
    cli._assert_offline_rollout_provenance(runtime=runtime, rollout=base_rollout)

    different_model = dict(metadata)
    different_model["model"] = {"sha256": "different-model-with-the-same-dimension"}
    with pytest.raises(cli.AttributionCLIError, match=r"model\.sha256"):
        cli._assert_offline_rollout_provenance(
            runtime=runtime,
            rollout=replace(base_rollout, metadata=different_model),
        )

    different_schema = dict(metadata)
    different_schema["schema"] = {
        "sha256": "different-schema-with-the-same-dimension",
        "observation_dim": 2,
    }
    # Even a caller interested only in low index 0 cannot reuse a rollout
    # whose same-D schema has different semantics.
    with pytest.raises(cli.AttributionCLIError, match=r"schema\.sha256"):
        cli._assert_offline_rollout_provenance(
            runtime=runtime,
            rollout=replace(base_rollout, metadata=different_schema),
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("action_dim", 3, "action_dim"),
        ("model_observation_dim", 3, "model_observation_dim"),
        ("deterministic", False, "deterministic"),
    ),
)
def test_offline_provenance_rejects_contract_mismatches(
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    runtime, metadata = _provenance_runtime(tmp_path)
    changed = dict(metadata)
    changed[field] = value
    with pytest.raises(cli.AttributionCLIError, match=match):
        cli._assert_offline_rollout_provenance(
            runtime=runtime,
            rollout=replace(_toy_rollout(), metadata=changed),
        )


def test_offline_provenance_checks_saved_logit_action_axis(tmp_path: Path) -> None:
    runtime, metadata = _provenance_runtime(tmp_path)
    base = _toy_rollout()
    mismatched_logits = np.zeros((base.row_count, 3), dtype=np.float32)
    with pytest.raises(cli.AttributionCLIError, match="rollout logits action_dim"):
        cli._assert_offline_rollout_provenance(
            runtime=runtime,
            rollout=replace(
                base,
                logits=mismatched_logits,
                probabilities=np.full((base.row_count, 3), 1.0 / 3.0, dtype=np.float32),
                log_probabilities=np.full((base.row_count, 3), -np.log(3.0), dtype=np.float32),
                metadata=metadata,
            ),
        )


def test_offline_provenance_fails_closed_when_source_metadata_is_missing(tmp_path: Path) -> None:
    runtime, _metadata = _provenance_runtime(tmp_path)
    with pytest.raises(cli.AttributionCLIError, match="cannot be verified"):
        cli._assert_offline_rollout_provenance(
            runtime=runtime,
            rollout=replace(_toy_rollout(), metadata={}),
        )


def test_live_runtime_validation_resets_once_checks_adapter_and_always_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances: list[object] = []

    class _Space:
        def __init__(self, *, shape: tuple[int, ...] | None = None, n: int | None = None) -> None:
            self.shape = shape
            self.n = n

    class _ValidationEnv:
        def __init__(self) -> None:
            self.observation_space = _Space(shape=(2,))
            self.action_space = _Space(n=2)
            self.reset_seeds: list[int] = []
            self.closed = False
            instances.append(self)

        def reset(self, *, seed: int) -> tuple[np.ndarray, dict[str, object]]:
            self.reset_seeds.append(seed)
            return np.asarray([0.0, 1.0], dtype=np.float32), {}

        def close(self) -> None:
            self.closed = True

    monkeypatch.setitem(
        sys.modules,
        "env_factory",
        SimpleNamespace(make_evaluation_env=lambda **_kwargs: _ValidationEnv()),
    )
    spaces = SimpleNamespace(observation_space=_Space(shape=(2,)), action_space=_Space(n=2))
    runtime: dict[str, object] = {
        "seed": 7,
        "model": spaces,
        "schema": SimpleNamespace(observation_dim=2),
        "adapter": SimpleNamespace(observation_dim=2, action_count=2),
        "experiment": SimpleNamespace(
            profile=SimpleNamespace(evaluation_env_config={"start_seed": 11})
        ),
    }

    contract = cli._validate_live_runtime(runtime)

    assert contract["reset_observation_dim"] == 2
    assert len(instances) == 1
    assert instances[0].reset_seeds == [11]
    assert instances[0].closed is True

    runtime["adapter"] = SimpleNamespace(observation_dim=2, action_count=3)
    with pytest.raises(cli.AttributionCLIError, match="adapter/environment action mismatch"):
        cli._validate_live_runtime(runtime)
    assert instances[-1].closed is True


def test_run_analysis_records_closed_loop_replacement_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    base = _toy_analysis_config()
    analysis = replace(
        base,
        closed_loop=ClosedLoopConfig(
            enabled=True,
            top_k_features=1,
            top_k_groups=0,
            replacement_strategy="specified_reference_constant",
        ),
    )
    schema = ObservationSchema(
        schema_version=1,
        name="closed_loop_provenance",
        observation_dim=2,
        features=(Feature(0, "left", "state"), Feature(1, "right", "state")),
        groups={"state": (0, 1)},
    )
    monkeypatch.setitem(
        sys.modules,
        "env_factory",
        SimpleNamespace(make_evaluation_env=lambda **_kwargs: object()),
    )

    def fake_closed_loop(**_kwargs: object) -> ClosedLoopResult:
        return ClosedLoopResult(
            records=(),
            summary_rows=(
                {
                    "target_name": "left",
                    "target_kind": "feature",
                    "target_indices": "0",
                },
            ),
            replacement_strategy="specified_reference_constant",
        )

    monkeypatch.setattr(cli, "run_paired_closed_loop", fake_closed_loop)
    artifacts = cli._analyze_rollout(
        directory=tmp_path,
        runtime={
            "analysis": analysis,
            "schema": schema,
            "adapter": _ToyAttributionAdapter(),
            "experiment": SimpleNamespace(
                profile=SimpleNamespace(
                    evaluation_env_config={"start_seed": 5, "num_scenarios": 1}
                )
            ),
            "seed": 3,
        },
        rollout=_toy_rollout(),
        allow_closed_loop=True,
    )

    provenance = artifacts["closed_loop_provenance"]
    assert provenance is not None
    assert provenance["strategy"] == "specified_reference_constant"
    assert provenance["reference_source"] == "saved_or_current_rollout"
    assert provenance["specified_reference_baseline_id"].startswith("episode_start:")
    assert provenance["resolved_baseline"]["strategy"] == "episode_start"


def test_report_keeps_ig_targets_separate_and_includes_numeric_result_sections() -> None:
    schema = ObservationSchema(
        schema_version=1,
        name="report_schema",
        observation_dim=3,
        features=(
            Feature(0, "actor_feature", "state"),
            Feature(1, "critic_feature", "state"),
            Feature(2, "custom_signal", "custom", kind="custom"),
        ),
        groups={"state": (0, 1), "custom": (2,)},
    )
    metadata = {
        "model": {"path": "model.zip"},
        "experiment": {"name": "toy"},
        "scenario_seed_range": {"start": 5, "count": 1},
        "action_dim": 2,
        "baseline": {"strategy": "episode_start"},
        "perturbation": {"definition": "baseline replacement"},
        "integrated_gradients": {
            "targets": ["selected_log_probability", "critic_value"],
            "steps": 8,
        },
        "closed_loop_provenance": {"strategy": "episode_start_constant"},
    }
    report = cli._report_markdown(
        metadata=metadata,
        schema=schema,
        perturbation_features=(
            {"target_name": "actor_feature", "mean_js_divergence": 0.8},
            {"target_name": "critic_feature", "mean_js_divergence": 0.1},
            {"target_name": "custom_signal", "mean_js_divergence": 0.5},
        ),
        perturbation_groups=(
            {
                "target_name": "lidar_000.0_015.0_deg",
                "target_kind": "lidar_sector",
                "mean_js_divergence": 0.6,
            },
        ),
        ig_features=(
            {
                "target": "selected_log_probability",
                "feature_name": "actor_feature",
                "mean_absolute_ig": 0.7,
            },
            {
                "target": "critic_value",
                "feature_name": "critic_feature",
                "mean_absolute_ig": 0.9,
            },
            {
                "target": "selected_log_probability",
                "feature_name": "custom_signal",
                "mean_absolute_ig": 0.2,
            },
            {
                "target": "critic_value",
                "feature_name": "custom_signal",
                "mean_absolute_ig": 0.3,
            },
        ),
        ig_groups=(
            {
                "target": "selected_log_probability",
                "group_name": "state",
                "group_kind": "group",
                "group_absolute_mass": 0.7,
            },
            {
                "target": "selected_log_probability",
                "group_name": "lidar_000.0_015.0_deg",
                "group_kind": "lidar_sector",
                "group_absolute_mass": 0.2,
            },
            {
                "target": "critic_value",
                "group_name": "lidar_000.0_015.0_deg",
                "group_kind": "lidar_sector",
                "group_absolute_mass": 0.4,
            },
        ),
        closed_loop=ClosedLoopResult(
            records=(),
            summary_rows=(
                {
                    "target_name": "actor_feature",
                    "target_kind": "feature",
                    "target_indices": "0",
                    "mean_delta_total_reward": -1.25,
                    "mean_delta_success": -0.5,
                    "mean_delta_out_of_road": 0.5,
                    "mean_delta_crash": 0.0,
                    "mean_delta_route_completion": -0.2,
                },
            ),
            replacement_strategy="episode_start_constant",
        ),
    )

    actor_section = report.split("### target: `selected_log_probability`")[1].split(
        "### target: `critic_value`"
    )[0]
    assert "`actor_feature`" in actor_section
    assert "`critic_feature`" not in actor_section
    assert "mean_js_divergence=0.8" in report
    assert "lidar_000.0_015.0_deg" in report
    assert "custom_signal" in report
    assert "Δ total reward=-1.25" in report
    assert "共通:" in report
    lidar_section = report.split("## 6. LiDAR angle / sector の結果")[1].split(
        "## 7. Custom feature の結果"
    )[0]
    assert "### IG target: `selected_log_probability`" in lidar_section
    assert "### IG target: `critic_value`" in lidar_section
    custom_section = report.split("## 7. Custom feature の結果")[1].split(
        "## 8. Closed-loop"
    )[0]
    assert "### IG target: `selected_log_probability`" in custom_section
    assert "### IG target: `critic_value`" in custom_section


def _agreement_section(report: str) -> str:
    return report.split("## 9. 摂動とIGの一致・不一致")[1].split(
        "摂動依存度とIG寄与度は別の問い"
    )[0]


def _agreement_report(*, targets: list[str], ig_features: tuple[dict[str, object], ...]) -> str:
    schema = ObservationSchema(
        schema_version=1,
        name="agreement_schema",
        observation_dim=2,
        features=(Feature(0, "actor_feature", "state"), Feature(1, "critic_feature", "state")),
        groups={"state": (0, 1)},
    )
    return cli._report_markdown(
        metadata={
            "model": {"path": "model.zip"},
            "experiment": {"name": "toy"},
            "scenario_seed_range": {"start": 5, "count": 1},
            "action_dim": 2,
            "baseline": {"strategy": "episode_start"},
            "perturbation": {"definition": "baseline replacement"},
            "integrated_gradients": {"targets": targets, "steps": 8},
        },
        schema=schema,
        perturbation_features=(
            {"target_name": "actor_feature", "mean_js_divergence": 0.8},
            {"target_name": "critic_feature", "mean_js_divergence": 0.6},
        ),
        perturbation_groups=(),
        ig_features=ig_features,
        ig_groups=(),
        closed_loop=None,
    )


def test_report_agreement_always_uses_selected_log_probability_not_target_order() -> None:
    ig_rows = (
        {
            "target": "selected_log_probability",
            "feature_name": "actor_feature",
            "mean_absolute_ig": 0.4,
        },
        {
            "target": "critic_value",
            "feature_name": "critic_feature",
            "mean_absolute_ig": 0.9,
        },
    )

    ordered = _agreement_section(
        _agreement_report(
            targets=["selected_log_probability", "critic_value"], ig_features=ig_rows
        )
    )
    reordered = _agreement_section(
        _agreement_report(
            targets=["critic_value", "selected_log_probability"], ig_features=ig_rows
        )
    )

    assert ordered == reordered
    assert "- 共通: actor_feature" in ordered
    assert "- 摂動のみ: critic_feature" in ordered
    assert "IGのみ（`selected_log_probability`）" in ordered


def test_report_agreement_marks_critic_only_ig_as_actor_unavailable() -> None:
    report = _agreement_report(
        targets=["critic_value"],
        ig_features=(
            {
                "target": "critic_value",
                "feature_name": "critic_feature",
                "mean_absolute_ig": 0.9,
            },
        ),
    )
    agreement = _agreement_section(report)

    assert "actor target `selected_log_probability` が設定されていない" in agreement
    assert "IGのみ（`critic_value`）" not in agreement
    assert "- 共通:" not in agreement


def test_report_lists_the_scalar_function_for_each_configured_ig_target() -> None:
    report = _agreement_report(
        targets=[
            "selected_log_probability",
            "selected_vs_runner_up_margin",
            "critic_value",
        ],
        ig_features=(),
    )

    assert "F(z) = log π(a* | z)" in report
    assert "F(z) = l_{a*}(z) − l_{a2}(z)" in report
    assert "raw logits" in report
    assert "F(z) = V(z)" in report


def test_offline_analysis_writes_temporal_aggregation_and_standard_artifacts(
    tmp_path: Path,
) -> None:
    os.environ["MPLCONFIGDIR"] = str(tmp_path / "mpl")
    schema = ObservationSchema(
        schema_version=1,
        name="toy_schema",
        observation_dim=2,
        features=(
            Feature(0, "left", "state", groups=("ego_state",)),
            Feature(1, "right", "state", groups=("ego_state",)),
        ),
        groups={"ego_state": (0, 1)},
    )
    artifacts = cli._analyze_rollout(
        directory=tmp_path,
        runtime={
            "analysis": _toy_analysis_config(),
            "schema": schema,
            "adapter": _ToyAttributionAdapter(),
        },
        rollout=_toy_rollout(),
        allow_closed_loop=False,
    )

    scopes = {row["scope"] for row in artifacts["ig_features"]}
    assert {"full_episode", "episode", "progress_bin", "phase"}.issubset(scopes)
    assert {row["scope"] for row in artifacts["perturbation_features"]} >= scopes
    assert artifacts["overview_ig_target"] == "selected_log_probability"
    assert {
        row["target"] for row in artifacts["overview_ig_features"]
    } == {"selected_log_probability"}
    assert {
        row["target"] for row in artifacts["overview_ig_groups"]
    } == {"selected_log_probability"}
    # The report source retains each target independently rather than merging
    # actor and critic values into one ranking.
    assert {
        row["target"] for row in artifacts["report_ig_features"]
    } == {"selected_log_probability", "critic_value"}
    for filename in (
        "perturbation_feature_steps.npz",
        "perturbation_feature_summary.csv",
        "perturbation_group_summary.csv",
        "ig_attributions.npz",
        "ig_feature_summary.csv",
        "ig_group_summary.csv",
        "ig_completeness.csv",
        "closed_loop_runs.jsonl",
        "closed_loop_summary.csv",
    ):
        assert (tmp_path / filename).is_file()
    for filename in (
        "perturbation_feature_top.png",
        "perturbation_group_top.png",
        "ig_feature_top_absolute.png",
        "ig_group_top_absolute.png",
        "ig_group_signed.png",
        "attribution_over_time.png",
        "perturbation_over_time.png",
        "lidar_ig_heatmap.png",
        "lidar_perturbation_heatmap.png",
        "closed_loop_performance.png",
    ):
        assert (tmp_path / "plots" / filename).is_file()
    with np.load(tmp_path / "ig_attributions.npz", allow_pickle=False) as archive:
        assert archive["attributions"].shape == archive["absolute_attributions"].shape
        np.testing.assert_allclose(
            archive["absolute_attributions"], np.abs(archive["attributions"])
        )
