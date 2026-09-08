"""Staging/publish contracts for the numbered attribution result tree.

These tests deliberately replace the runtime boundary with tiny fakes.  They
exercise the command handlers and publication transaction without importing or
starting MetaDrive, PPO, or a large rollout.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import analyze_input_attribution as cli
from input_attribution.compact import CompactError


_EXPERIMENTS = (
    "experiment_01_perturbation",
    "experiment_02_integrated_gradients",
    "experiment_03_closed_loop",
)


def _runtime() -> dict[str, object]:
    analysis = SimpleNamespace(
        run=SimpleNamespace(deterministic=True),
        closed_loop=SimpleNamespace(
            enabled=False,
            explicit_features=(),
            explicit_groups=(),
            replacement_strategy="episode_start_constant",
        ),
    )
    experiment = SimpleNamespace(
        name="toy",
        profile=SimpleNamespace(
            evaluation_env_config={"start_seed": 5, "num_scenarios": 1}
        ),
    )
    return {
        "experiment": experiment,
        "analysis": analysis,
        "schema": SimpleNamespace(
            name="fake_schema",
            source_path=Path("schemas/fake.toml"),
            observation_dim=2,
            features=(),
        ),
        "adapter": SimpleNamespace(observation_dim=2, action_count=2),
        "model": object(),
        "model_path": Path("models/fake.zip"),
        "seed": 0,
        "device": "cpu",
    }


def _args(command: str, prefix: str, *, rollout: Path | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        command=command,
        config=Path("configs/fake.toml"),
        model=Path("models/fake.zip"),
        schema=Path("schemas/fake.toml"),
        analysis_config=Path("analysis/fake.toml"),
        output_prefix=prefix,
        output_mode="compact",
        device=None,
        seed=None,
        closed_loop=False,
        feature=["0"] if command == "closed-loop" else [],
        group=[],
        rollout=rollout,
    )


def _metadata() -> dict[str, object]:
    return {
        "model": {"path": "models/fake.zip", "sha256": "model-sha"},
        "schema": {"name": "fake_schema", "observation_dim": 2},
        "scenario_seed_range": {"start": 5, "count": 1},
        "rollout": {"row_count": 1},
        "rl_seed": 0,
        "closed_loop_provenance": {"strategy": "episode_start_constant"},
        "closed_loop_targets": ["speed"],
        "integrated_gradients": {"enabled": False, "executed": False, "targets": []},
        "perturbation": {"enabled": False, "executed": False},
    }


def _fake_rollout() -> SimpleNamespace:
    return SimpleNamespace(metadata={"runtime_contract": {}}, row_count=1, observation_dim=2)


def _install_runtime_fakes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, list[str]]:
    """Install command-boundary fakes and return callback event storage."""

    events: dict[str, list[str]] = {"callback": []}
    runtime = _runtime()
    monkeypatch.setattr(cli, "_load_runtime", lambda args, require_analysis: runtime)

    class _FakeEnv:
        def close(self) -> None:
            events["callback"].append("env-close")

    fake_env_factory = ModuleType("env_factory")
    fake_env_factory.make_evaluation_env = lambda **kwargs: _FakeEnv()
    monkeypatch.setitem(sys.modules, "env_factory", fake_env_factory)

    rollout = _fake_rollout()
    monkeypatch.setattr(cli, "collect_rollout", lambda **kwargs: rollout)
    monkeypatch.setattr(cli, "load_rollout", lambda path: rollout)

    def fake_save_rollout(directory: Path, value: object, *, metadata: object = None) -> dict[str, Path]:
        events["callback"].append("raw")
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory / "rollout_arrays.npz", observations=np.zeros((1, 2), dtype=np.float32))
        (directory / "rollout_steps.jsonl").write_text("{}\n", encoding="utf-8")
        (directory / "rollout_metadata.json").write_text(
            json.dumps({"runtime_contract": {}, **(metadata or {})}, default=str),
            encoding="utf-8",
        )
        if metadata is not None:
            events["callback"].append("metadata")
            (directory / "analysis_metadata.json").write_text(
                json.dumps(_metadata(), ensure_ascii=False), encoding="utf-8"
            )
        return {
            "arrays": directory / "rollout_arrays.npz",
            "steps": directory / "rollout_steps.jsonl",
            "metadata": directory / "rollout_metadata.json",
        }

    monkeypatch.setattr(cli, "save_rollout", fake_save_rollout)

    def fake_write_schema(path: Path, schema: object) -> list[dict[str, object]]:
        path.write_text("index,name\n0,speed\n", encoding="utf-8")
        events["callback"].append("schema")
        return []

    monkeypatch.setattr(cli, "write_expanded_schema_csv", fake_write_schema)
    monkeypatch.setattr(cli, "_metadata", lambda **kwargs: _metadata())

    def fake_atomic_json(path: Path, payload: object) -> None:
        path.write_text(json.dumps(payload, default=str), encoding="utf-8")
        events["callback"].append("metadata")

    monkeypatch.setattr(cli, "atomic_write_json", fake_atomic_json)

    def fake_write_report(path: Path, report: str) -> None:
        path.write_text("# callback report\n", encoding="utf-8")
        events["callback"].append("report")

    monkeypatch.setattr(cli, "write_report", fake_write_report)

    analysis_result = {
        "baselines": SimpleNamespace(metadata={}),
        "perturbation": None,
        "ig": None,
        "closed_loop": None,
        "closed_loop_provenance": None,
        "overview_perturbation_features": [],
        "overview_perturbation_groups": [],
        "report_ig_features": [],
        "report_ig_groups": [],
    }
    monkeypatch.setattr(cli, "_analyze_rollout", lambda **kwargs: analysis_result)
    monkeypatch.setattr(cli, "_assert_offline_rollout_provenance", lambda **kwargs: None)

    closed_loop_result = SimpleNamespace(
        summary_rows=[
            {
                "target_name": "speed",
                "target_kind": "feature",
                "scenario_count": 1,
                "replacement_strategy": "episode_start_constant",
                "mean_baseline_total_reward": 1.0,
                "mean_intervention_total_reward": 0.5,
                "mean_delta_total_reward": -0.5,
                "mean_baseline_success": 1.0,
                "mean_intervention_success": 0.0,
                "mean_baseline_out_of_road": 0.0,
                "mean_intervention_out_of_road": 1.0,
            }
        ]
    )
    monkeypatch.setattr(
        cli,
        "schema_targets",
        lambda schema, *, feature_indices, group_names: (
            SimpleNamespace(name="speed", kind="feature", indices=(0,)),
        ),
    )
    monkeypatch.setattr(cli, "_validate_live_runtime", lambda runtime: {
        "env_observation_dim": 2,
        "env_action_dim": 2,
    })
    monkeypatch.setattr(cli, "run_paired_closed_loop", lambda **kwargs: closed_loop_result)

    def fake_save_closed_loop(directory: Path, result: object) -> dict[str, Path]:
        events["callback"].append("raw")
        (directory / "closed_loop_runs.jsonl").write_text("{}\n", encoding="utf-8")
        (directory / "closed_loop_summary.csv").write_text(
            "target_name,target_kind,scenario_count,replacement_strategy,"
            "mean_baseline_total_reward,mean_intervention_total_reward,"
            "mean_baseline_success,mean_intervention_success,"
            "mean_baseline_out_of_road,mean_intervention_out_of_road\n"
            "speed,feature,1,episode_start_constant,1,0.5,1,0,0,1\n",
            encoding="utf-8",
        )
        return {}

    monkeypatch.setattr(cli, "save_closed_loop", fake_save_closed_loop)
    fake_visualization = ModuleType("input_attribution.visualization")

    def fake_plots(directory: Path, **kwargs: object) -> None:
        events["callback"].append("plot")
        plots = directory / "plots"
        plots.mkdir(exist_ok=True)
        (plots / "closed_loop_performance.png").write_bytes(b"plot")

    fake_visualization.generate_standard_plots = fake_plots
    monkeypatch.setitem(sys.modules, "input_attribution.visualization", fake_visualization)
    return events


def _install_transaction_spy(
    monkeypatch: pytest.MonkeyPatch,
    expected_callback_report: bool,
    events: dict[str, list[str]],
) -> tuple[list[dict[str, object]], list[Path]]:
    finalize_calls: list[dict[str, object]] = []
    publish_calls: list[Path] = []
    real_finalize = cli.finalize_result_directory
    real_publish = cli.StagedRunDirectory.publish

    def spy_finalize(path: Path, **kwargs: object) -> Path:
        assert path.is_dir()
        assert "raw" in events["callback"]
        assert "metadata" in events["callback"]
        if expected_callback_report:
            assert "report" in events["callback"]
        finalize_calls.append(dict(kwargs))
        return real_finalize(path, **kwargs)

    def spy_publish(holder: object) -> Path:
        assert len(finalize_calls) == 1
        publish_calls.append(holder.target)  # type: ignore[attr-defined]
        return real_publish(holder)  # type: ignore[arg-type]

    monkeypatch.setattr(cli, "finalize_result_directory", spy_finalize)
    monkeypatch.setattr(cli.StagedRunDirectory, "publish", spy_publish)
    return finalize_calls, publish_calls


@pytest.mark.parametrize(
    ("command", "expects_callback_report"),
    (
        ("collect", False),
        ("run", True),
        ("analyze", True),
        ("closed-loop", True),
    ),
)
def test_runtime_commands_finalize_once_then_publish_numbered_tree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    command: str,
    expects_callback_report: bool,
) -> None:
    monkeypatch.setattr(cli, "OUTPUT_DIR", tmp_path / "outputs")
    events = _install_runtime_fakes(monkeypatch, tmp_path)
    finalize_calls, publish_calls = _install_transaction_spy(
        monkeypatch, expects_callback_report, events
    )
    source_rollout = tmp_path / "source-rollout"
    source_rollout.mkdir()
    if command == "closed-loop":
        (source_rollout / "rollout_metadata.json").write_text(
            '{"row_count": 1}\n', encoding="utf-8"
        )
    args = _args(
        command,
        f"{command.replace('-', '_')}_numbered",
        rollout=source_rollout if command == "analyze" else None,
    )

    handlers = {
        "collect": cli._collect_command,
        "run": cli._run_command,
        "analyze": cli._analyze_command,
        "closed-loop": cli._closed_loop_command,
    }
    published = handlers[command](args)

    assert len(finalize_calls) == 1
    assert len(publish_calls) == 1
    assert published == publish_calls[0]
    if command == "collect":
        assert (published / "report.md").is_file()
        assert "実験01〜03の解析は未実施" in (published / "report.md").read_text(encoding="utf-8")
        assert (published / "shared" / "rollout_arrays.npz").is_file()
        assert not any((published / experiment).exists() for experiment in _EXPERIMENTS)
    else:
        assert (published / "report.md").is_file()
        for experiment in _EXPERIMENTS:
            assert (published / experiment / "report.md").is_file()
            assert (published / experiment / "details.zip").is_file()
        assert (published / "shared" / "details.zip").is_file()
        # Compact shared contains the common archive only; the preserved
        # source report belongs to full output, not this compact tree.
        assert not (published / "shared" / "report.md").exists()


def test_finalize_failure_preserves_existing_target_and_removes_staging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli, "OUTPUT_DIR", tmp_path / "outputs")
    runtime = _runtime()
    target = cli._run_directory("toy", "preserve_existing")
    target.mkdir(parents=True)
    sentinel = target / "sentinel.txt"
    sentinel.write_text("old result", encoding="utf-8")
    before_hash = hashlib.sha256(sentinel.read_bytes()).hexdigest()

    def fail_finalize(path: Path, **kwargs: object) -> Path:
        assert (path / "rollout_arrays.npz").is_file()
        raise CompactError("synthetic finalize failure")

    monkeypatch.setattr(cli, "finalize_result_directory", fail_finalize)

    def callback(staging: Path) -> None:
        (staging / "rollout_arrays.npz").write_bytes(b"partial")
        (staging / "analysis_metadata.json").write_text("{}", encoding="utf-8")
        (staging / "report.md").write_text("partial", encoding="utf-8")

    with pytest.raises(CompactError, match="synthetic finalize failure"):
        cli._run_in_staging(runtime, "preserve_existing", callback)

    assert sentinel.read_text(encoding="utf-8") == "old result"
    assert hashlib.sha256(sentinel.read_bytes()).hexdigest() == before_hash
    assert not any(
        path.name.startswith(".preserve_existing.staging-")
        for path in target.parent.iterdir()
    )


def test_source_rollout_record_resolves_numbered_shared_metadata_and_hashes_it(
    tmp_path: Path,
) -> None:
    root = tmp_path / "numbered"
    shared = root / "shared"
    shared.mkdir(parents=True)
    metadata_path = shared / "rollout_metadata.json"
    metadata_path.write_text('{"row_count": 1}\n', encoding="utf-8")
    rollout = SimpleNamespace(metadata={"source": "fake"}, row_count=1, observation_dim=2)

    record = cli._source_rollout_record(root, rollout)

    assert record["metadata_path"] == str(metadata_path.resolve())
    assert record["metadata_relative_path"] == "shared/rollout_metadata.json"
    assert record["metadata_sha256"] == hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    assert record["row_count"] == 1
    assert record["observation_dim"] == 2
