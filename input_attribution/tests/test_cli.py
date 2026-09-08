"""CLI/portable artifact acceptance tests; synthetic runs are never real driving evidence."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

import numpy as np
import pytest

from input_attribution.artifacts import build_manifest, code_fingerprint, verify_manifest_compatibility, sha256_file
from input_attribution.config import load_config

PACKAGE = Path(__file__).resolve().parents[1]


def _run(cwd, *args, script=None):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(cwd))
    command = [sys.executable, "-c", script, *map(str, args)] if script else [sys.executable, "-m", "input_attribution", *map(str, args)]
    result = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    return result


@pytest.mark.parametrize("dimension", [259, 262])
def test_copied_package_runs_all_inputs_and_replays_without_environment(tmp_path, dimension):
    """A complete copy works away from this repository with no host private helpers."""
    destination = tmp_path / "copied"
    shutil.copytree(PACKAGE, destination / "input_attribution", ignore=shutil.ignore_patterns("__pycache__", "tests"))
    filename = "fake_259.toml" if dimension == 259 else "fake_262_resolved.toml"
    config_file = destination / "input_attribution" / "configs" / filename
    raw = tomllib.loads(config_file.read_text())
    raw["schema"]["path"] = str(destination / "input_attribution" / "schemas" / ("official_259.json" if dimension == 259 else "fake_262_resolved.json"))
    raw["output"]["root"] = str(destination / "results")
    raw["scenario"]["horizon"] = 3
    raw["closed_loop"]["max_steps"] = 3
    raw["analysis"]["individual_inputs"] = True
    raw["video"]["enabled"] = False
    config_file = destination / "analysis.json"
    config_file.write_text(json.dumps(raw), encoding="utf-8")
    result = _run(destination, "run", "--config", config_file)
    run_dir = Path(result.stdout.strip().splitlines()[-1])
    status = json.loads((run_dir / "status.json").read_text())["stages"]
    assert all(status[key]["state"] == "success" for key in ("run", "collect", "offline", "closed_loop", "report"))
    assert status["ig"]["state"] == "skipped"
    values = np.load(run_dir / "00_reference" / "observations.npy", allow_pickle=False)
    assert values.shape == (3, dimension)
    schema = json.loads((run_dir / "input_schema.json").read_text())
    if dimension == 262:
        by_id = {row["id"]: row["index"] for row in schema["inputs"]}
        assert [by_id[key] for key in ("target_lane_offset_synthetic", "target_lane_heading_error_synthetic", "target_lane_valid_synthetic")] == [7, 145, 260]
    offline = run_dir / status["offline"]["relative_dir"] / "result.json"
    data = json.loads(offline.read_text())
    assert len([row for row in data["patterns"] if row["pattern"]["pattern_id"].startswith("input_")]) == dimension
    identity = next(row for row in data["patterns"] if row["pattern"]["pattern_id"] == "P00")
    assert set(identity["summary"]["action_marginals"]) == {"steering", "throttle_brake"}
    reference_hash = sha256_file(run_dir / "00_reference" / "observations.npy")
    first_report_hash = sha256_file(run_dir / "report.html")
    first_offline_hash = sha256_file(offline)
    # Spy only in the test process: partial offline must not create any environment.
    replay_script = """
import sys
from input_attribution.adapters.fake import FakePolicyAdapter
def forbidden(*a, **kw):
    raise AssertionError('offline attempted environment creation')
FakePolicyAdapter.make_environment = forbidden
FakePolicyAdapter.make_env = forbidden
from input_attribution.cli import main
raise SystemExit(main(sys.argv[1:]))
"""
    _run(destination, "offline", "--run-dir", run_dir, script=replay_script)
    updated = json.loads((run_dir / "status.json").read_text())["stages"]
    assert updated["offline"]["analysis_id"] != status["offline"]["analysis_id"]
    assert sha256_file(offline) == first_offline_hash
    assert sha256_file(run_dir / "00_reference" / "observations.npy") == reference_hash
    # Fail all imports of optional runtime/model dependencies during report regeneration.
    report_script = """
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'metadrive', 'stable_baselines3', 'torch', 'captum'}:
            raise AssertionError('report imported ' + fullname)
sys.meta_path.insert(0, Block())
from input_attribution.cli import main
raise SystemExit(main(sys.argv[1:]))
"""
    _run(destination, "report", "--run-dir", run_dir, script=report_script)
    assert sha256_file(run_dir / "report.html") == first_report_hash
    assert list((run_dir / "reports").glob("*/report.html"))
    # Optional IG dependency failure preserves the completed main experiment.
    missing_captum_script = """
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'captum':
            raise ModuleNotFoundError('test: optional Captum unavailable')
sys.meta_path.insert(0, Block())
from input_attribution.cli import main
raise SystemExit(0 if main(sys.argv[1:]) != 0 else 1)
"""
    _run(destination, "ig", "--run-dir", run_dir, "--episode", "0", "--steps", "0", "--baseline", "episode-0:0", script=missing_captum_script)
    after_ig = json.loads((run_dir / "status.json").read_text())["stages"]
    assert after_ig["ig"]["state"] == "failed"
    assert "Captum" in after_ig["ig"]["error"]
    assert after_ig["run"]["state"] == "success"
    assert sha256_file(run_dir / "report.html") == first_report_hash
    assert sha256_file(offline) == first_offline_hash
    # An edited saved input is rejected before another analysis is created.
    values[0, 0] += 0.01
    np.save(run_dir / "00_reference" / "observations.npy", values, allow_pickle=False)
    failure = subprocess.run([sys.executable, "-m", "input_attribution", "offline", "--run-dir", str(run_dir)], cwd=destination, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(destination)), capture_output=True, text=True)
    assert failure.returncode != 0 and "hash mismatch" in failure.stderr


@pytest.mark.parametrize("target", ["model", "schema", "patterns", "preprocess", "statistics", "code"])
def test_provenance_changes_are_rejected(tmp_path, target):
    model = tmp_path / "model.zip"
    model.write_bytes(b"synthetic-model-only-for-hash-test")
    schema = tmp_path / "schema.json"
    schema.write_text("{}")
    statistics = tmp_path / "stats.bin"
    statistics.write_bytes(b"frozen-stats")
    config = {"patterns": [{"id": "P00"}], "preprocess": {"stats_path": str(statistics)}}
    manifest = build_manifest(config=config, model_path=model, schema_path=schema, patterns=config["patterns"], preprocess=config["preprocess"])
    if target == "model":
        model.write_bytes(b"changed")
    elif target == "schema":
        schema.write_text('{"changed":true}')
    elif target == "patterns":
        config["patterns"].append({"id": "P01"})
    elif target == "preprocess":
        config["preprocess"]["external_normalization"] = True
    elif target == "statistics":
        statistics.write_bytes(b"changed-statistics")
    elif target == "code":
        manifest["code_sha256"] = "not-the-current-code"
    valid, reasons = verify_manifest_compatibility(manifest, config=config, model_path=model, schema_path=schema, patterns=config["patterns"], preprocess=config["preprocess"])
    assert not valid and reasons


def test_code_hash_is_portable_and_includes_new_adapter_bytes(tmp_path):
    shutil.copytree(PACKAGE, tmp_path / "input_attribution", ignore=shutil.ignore_patterns("__pycache__"))
    copied = tmp_path / "input_attribution"
    assert code_fingerprint(copied) == code_fingerprint(PACKAGE)
    with (copied / "adapters" / "fake.py").open("a") as stream:
        stream.write("\n# test-only adapter change\n")
    assert code_fingerprint(copied) != code_fingerprint(PACKAGE)


def test_cli_help_lists_documented_commands(tmp_path):
    result = _run(PACKAGE.parent, "--help")
    for name in ("check", "run", "collect", "offline", "closed-loop", "ig", "report"):
        assert name in result.stdout


@pytest.mark.parametrize("change", ["dimension", "preprocessing", "unresolved"])
def test_check_rejects_mismatched_or_unresolved_schema_before_environment(change):
    from input_attribution.adapters.fake import FakePolicyAdapter
    from input_attribution.adapters.fixtures import official_schema_259
    from input_attribution.config import config_from_mapping
    from input_attribution.checks import run_check
    from input_attribution.schema import InputSchema
    source = tomllib.loads((PACKAGE / "configs" / "fake_259.toml").read_text())
    source["schema"]["path"] = str(PACKAGE / "schemas" / "official_259.json")
    schema = official_schema_259()
    if change == "dimension":
        source["schema"]["dimension"] = 262
    elif change == "preprocessing":
        source["preprocess"]["external_normalization"] = True
    else:
        raw = schema.to_dict()
        raw["inputs"][7]["source"] = "UNRESOLVED_SOURCE"
        schema = InputSchema.from_dict(raw)
    config = config_from_mapping(source)
    class NoEnvironment(FakePolicyAdapter):
        def make_environment(self, *args, **kwargs):
            pytest.fail("invalid contract reached environment creation")
        make_env = make_environment
    adapter = NoEnvironment(dimension=259)
    result = run_check(config, policy=adapter, adapter=adapter, schema=schema)
    assert not result.ok
    assert any(term in " ".join(result.errors) for term in ("mismatch", "unresolved", "incomplete provenance"))
