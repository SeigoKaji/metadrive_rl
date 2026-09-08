"""Synthetic checks for immutable saved-reference reuse.

These tests exercise file/provenance boundaries only; they are not MetaDrive
driving evidence.
"""
from __future__ import annotations

import json
import copy
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

import numpy as np

from input_attribution.artifacts import (
    build_manifest,
    copy_reference,
    create_run,
    schema_semantics_hash_from_value,
    seal_reference,
    sha256_file,
    verify_reference,
)


def _schema(*, index: int = 0, variant: bool = False) -> dict[str, object]:
    item: dict[str, object] = {
        "index": index,
        "id": "synthetic_input",
        "name_ja": "合成入力",
        "description": "source verified synthetic input",
        "group": "synthetic",
        "physical_quantity": "dimensionless",
        "unit": "fraction",
        "model_representation": "normalized scalar",
        "range": [0.0, 1.0],
        "normalization": "identity",
        "clip_method": "clip to [0, 1]",
        "value_type": "continuous",
        "reference": "fixture",
        "replacement": {"kind": "fixed", "value": 0.5},
        "related_valid_flags": [],
        "coupled_indices": [],
        "independently_replaceable": True,
        "ig_interpolation_allowed": True,
        "source": "fixture.py:1",
    }
    if variant:
        item["replacement"] = {"kind": "fixed", "value": 0.75}
        item["variants"] = [{"id": "verified_alt", "operation": "fixed_level", "value": 0.25}]
    return {
        "schema_id": "synthetic",
        "version": "1",
        "dimension": 1,
        "preprocessing": {"external_normalization": False},
        "inputs": [item],
    }


def test_schema_semantics_hash_ignores_replacement_variants_only() -> None:
    base = _schema()
    assert schema_semantics_hash_from_value(base) == schema_semantics_hash_from_value(
        _schema(variant=True)
    )
    assert schema_semantics_hash_from_value(base) != schema_semantics_hash_from_value(_schema(index=1))
    invalid_changed = _schema()
    invalid_changed["inputs"][0]["replacement"]["invalid_value"] = -1.0  # type: ignore[index]
    assert schema_semantics_hash_from_value(base) != schema_semantics_hash_from_value(invalid_changed)
    encoded = _schema()
    encoded["semantic_adapter"] = {"encoding": "float32-normalized-v1"}
    encoded_changed = _schema()
    encoded_changed["semantic_adapter"] = {"encoding": "float32-normalized-v2"}
    assert schema_semantics_hash_from_value(encoded) != schema_semantics_hash_from_value(encoded_changed)


def test_copy_reference_uses_independent_files_and_preserves_parent(tmp_path: Path) -> None:
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(_schema()), encoding="utf-8")
    root = tmp_path / "runs"
    parent = create_run(root, "exp", "model")
    parent.save_manifest(
        build_manifest(
            config={"preprocess": {}},
            schema_path=schema_path,
            patterns=[{"id": "P00"}],
            preprocess={},
        )
    )
    np.save(parent.reference_dir / "observations.npy", np.zeros((2, 1), dtype=np.float32))
    (parent.reference_dir / "records.jsonl").write_text(
        '{"episode": 0, "episode_id": "episode-0", "step": 0, "action": 0, "probabilities": [1.0], "pre_telemetry": {}}\n'
        '{"episode": 0, "episode_id": "episode-0", "step": 1, "action": 0, "probabilities": [1.0], "pre_telemetry": {}}\n',
        encoding="utf-8",
    )
    seal_reference(parent)
    parent_bytes = (parent.reference_dir / "observations.npy").read_bytes()

    child = create_run(root, "exp", "model")
    copy_reference(parent, child)
    child_observations = child.reference_dir / "observations.npy"
    values = np.load(child_observations, allow_pickle=False)
    values[0, 0] = 0.75
    np.save(child_observations, values)

    verify_reference(parent)
    assert (parent.reference_dir / "observations.npy").read_bytes() == parent_bytes
    assert np.load(parent.reference_dir / "observations.npy", allow_pickle=False)[0, 0] == 0.0


def test_cli_exposes_reuse_and_report_output_options() -> None:
    from input_attribution.cli import build_parser

    parser = build_parser()
    offline = parser.parse_args(["offline", "--run-dir", "/tmp/parent", "--config", "/tmp/new.toml"])
    report = parser.parse_args(["report", "--run-dir", "/tmp/parent", "--output-dir", "/tmp/report"])
    assert offline.config == Path("/tmp/new.toml")
    assert report.output_dir == Path("/tmp/report")


def test_typed_variant_resolution_records_value_expression_and_evidence() -> None:
    from input_attribution.cli import _all_patterns, _pattern_variant_metadata
    from input_attribution.config import load_config
    from input_attribution.schema import load_schema

    config = load_config(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "input_attribution_official_259_variants.toml"
    )
    schema = load_schema(config.schema_path)
    patterns = {pattern.pattern_id: pattern for pattern in _all_patterns(config, schema)}
    neutral = _pattern_variant_metadata(patterns["P02_heading_neutral"], schema)
    reflection = _pattern_variant_metadata(patterns["P02_heading_reflection"], schema)
    assert neutral["resolved_values"] == {"2": 0.5}
    assert neutral["variant_evidence"]["2"]
    assert neutral["variant_classification"]["2"] == "neutral"
    assert reflection["resolved_values"]["2"]["center"] == 0.5
    assert reflection["resolved_values"]["2"]["expression"] == "2 * center - source_value"


PACKAGE = Path(__file__).resolve().parents[1]


def _subprocess(
    cwd: Path,
    *args: object,
    script: str | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(cwd),
    )
    command = (
        [sys.executable, "-c", script, *map(str, args)]
        if script is not None
        else [sys.executable, "-m", "input_attribution", *map(str, args)]
    )
    return subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=True,
        timeout=120,
    )


def _file_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _fake_parent_package(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    """Create a small copied package and one parent run for reuse tests."""

    destination = tmp_path / "copied"
    shutil.copytree(
        PACKAGE,
        destination / "input_attribution",
        ignore=shutil.ignore_patterns("__pycache__", "tests"),
    )
    source_config = tomllib.loads(
        (destination / "input_attribution" / "configs" / "fake_259.toml").read_text(
            encoding="utf-8"
        )
    )
    model_path = destination / "synthetic_model.zip"
    model_path.write_bytes(b"synthetic-model-for-reuse-boundary")
    source_config["model"]["path"] = str(model_path)
    source_config["schema"]["path"] = str(
        destination / "input_attribution" / "schemas" / "official_259.json"
    )
    source_config["output"]["root"] = str(destination / "results")
    source_config["scenario"]["horizon"] = 2
    source_config["closed_loop"]["enabled"] = False
    source_config["closed_loop"]["max_steps"] = 2
    source_config["analysis"]["individual_inputs"] = False
    config_path = destination / "parent.json"
    config_path.write_text(
        json.dumps(source_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    result = _subprocess(destination, "run", "--config", config_path)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    run_dir = Path(result.stdout.strip().splitlines()[-1])
    assert run_dir.is_dir()
    return destination, run_dir, source_config


def _reuse_config(
    destination: Path,
    source_config: dict[str, object],
    name: str = "variant",
) -> tuple[Path, dict[str, object]]:
    value = copy.deepcopy(source_config)
    value["analysis"]["name"] = f"fake_259_{name}"
    value["output"]["experiment"] = f"fake_259_{name}"
    value["patterns"] = copy.deepcopy(value["patterns"])
    value["patterns"].append(
        {
            "id": "P01_verified_variant",
            "description": "synthetic replacement variant",
            "indices": [2],
            "method": "fixed",
            "values": [0.25],
            "variant_id": "verified_variant_0_25",
            "variant": {
                "id": "verified_variant_0_25",
                "operation": "fixed",
                "value": 0.25,
                "meaning": "synthetic fixed diagnostic level",
                "evidence": "test fixture",
                "confirmed": True,
            },
        }
    )
    config_path = destination / f"{name}.json"
    config_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return config_path, value


def test_offline_reuse_is_environment_free_and_parent_is_immutable(tmp_path: Path) -> None:
    destination, parent, source_config = _fake_parent_package(tmp_path)
    parent_files_before = _file_hashes(parent)
    external_report = destination / "report-revision"
    report_result = _subprocess(
        destination,
        "report",
        "--run-dir",
        parent,
        "--output-dir",
        external_report,
    )
    assert report_result.returncode == 0, report_result.stdout[-3000:] + report_result.stderr[-3000:]
    assert (external_report / "report.html").is_file()
    assert _file_hashes(parent) == parent_files_before
    variant_config, _ = _reuse_config(destination, source_config)
    no_environment = """
import sys
from input_attribution.adapters.fake import FakePolicyAdapter
def forbidden(*args, **kwargs):
    raise AssertionError('offline reuse attempted environment creation')
FakePolicyAdapter.make_environment = forbidden
FakePolicyAdapter.make_env = forbidden
from input_attribution.cli import main
raise SystemExit(main(sys.argv[1:]))
"""
    result = _subprocess(
        destination,
        "offline",
        "--run-dir",
        parent,
        "--config",
        variant_config,
        script=no_environment,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    child = Path(result.stdout.strip().splitlines()[-1])
    assert child.is_dir() and child != parent

    parent_manifest = json.loads((parent / "manifest.json").read_text(encoding="utf-8"))
    child_manifest = json.loads((child / "manifest.json").read_text(encoding="utf-8"))
    reuse = child_manifest["reuse"]
    assert reuse["parent_run_id"] == parent.name
    assert reuse["parent_input_semantics_sha256"]
    assert reuse["parent_input_semantics_source"]
    assert reuse["parent_model_sha256"] == parent_manifest["model"]["sha256"]
    assert child_manifest["reference_reused"] is True
    child_patterns = json.loads((child / "patterns.json").read_text(encoding="utf-8"))
    variant_row = next(row for row in child_patterns if row["pattern_id"] == "P01_verified_variant")
    assert variant_row["resolved_values"] == {"2": 0.25}
    assert variant_row["variant_evidence"]["2"] == "test fixture"

    parent_reference = parent / "00_reference" / "observations.npy"
    child_reference = child / "00_reference" / "observations.npy"
    assert parent_reference.read_bytes() == child_reference.read_bytes()
    assert os.stat(parent_reference).st_ino != os.stat(child_reference).st_ino
    assert _file_hashes(parent) == parent_files_before

    parent_check = json.loads((parent / "check.json").read_text(encoding="utf-8"))
    child_check = json.loads((child / "check.json").read_text(encoding="utf-8"))
    assert child_check["checks"]["action_mapping"] == parent_check["checks"]["action_mapping"]
    status = json.loads((child / "status.json").read_text(encoding="utf-8"))["stages"]
    assert status["offline"]["state"] == "success"
    assert status["collect"]["state"] == "skipped"
    assert status["closed_loop"]["state"] == "skipped"
    assert "A-only reuse child" in status["closed_loop"]["reason"]


def test_offline_reuse_rejects_model_order_preprocess_and_invalid_sentinel_changes(
    tmp_path: Path,
) -> None:
    destination, parent, source_config = _fake_parent_package(tmp_path)
    variant_config, _ = _reuse_config(destination, source_config)

    def rejected(path: Path, expected: str) -> None:
        result = _subprocess(destination, "offline", "--run-dir", parent, "--config", path)
        assert result.returncode != 0, result.stdout[-3000:] + result.stderr[-3000:]
        assert expected in result.stderr

    model_variant = destination / "different_model.zip"
    model_variant.write_bytes(b"different-model")
    value = json.loads(variant_config.read_text(encoding="utf-8"))
    value["model"]["path"] = str(model_variant)
    model_config = destination / "model_mismatch.json"
    model_config.write_text(json.dumps(value), encoding="utf-8")
    rejected(model_config, "model.sha256")

    schema_path = Path(source_config["schema"]["path"])
    schema_value = json.loads(schema_path.read_text(encoding="utf-8"))
    # Change the model-index mapping itself.  Reordering the JSON list while
    # retaining each input's index is harmless after schema normalization and
    # therefore is not an input-order mismatch.
    schema_value["inputs"][0]["index"], schema_value["inputs"][1]["index"] = 1, 0
    order_schema = destination / "order_mismatch_schema.json"
    order_schema.write_text(json.dumps(schema_value), encoding="utf-8")
    value = json.loads(variant_config.read_text(encoding="utf-8"))
    value["schema"]["path"] = str(order_schema)
    order_config = destination / "order_mismatch.json"
    order_config.write_text(json.dumps(value), encoding="utf-8")
    rejected(order_config, "input_semantics_sha256")

    value = json.loads(variant_config.read_text(encoding="utf-8"))
    value["preprocess"]["external_normalization"] = True
    preprocess_config = destination / "preprocess_mismatch.json"
    preprocess_config.write_text(json.dumps(value), encoding="utf-8")
    rejected(preprocess_config, "preprocess_sha256")

    invalid_schema_value = json.loads(schema_path.read_text(encoding="utf-8"))
    invalid_schema_value["inputs"][0]["replacement"]["invalid_value"] = -1.0
    invalid_schema = destination / "invalid_sentinel_schema.json"
    invalid_schema.write_text(json.dumps(invalid_schema_value), encoding="utf-8")
    value = json.loads(variant_config.read_text(encoding="utf-8"))
    value["schema"]["path"] = str(invalid_schema)
    invalid_config = destination / "invalid_sentinel.json"
    invalid_config.write_text(json.dumps(value), encoding="utf-8")
    rejected(invalid_config, "input_semantics_sha256")


def test_offline_reuse_rejects_parent_resolved_config_and_snapshot_tampering(
    tmp_path: Path,
) -> None:
    destination, parent, source_config = _fake_parent_package(tmp_path)
    variant_config, _ = _reuse_config(destination, source_config)

    resolved = parent / "resolved_config.json"
    resolved_original = resolved.read_bytes()
    resolved_value = json.loads(resolved_original)
    resolved_value["analysis"]["description"] = "tampered parent config"
    resolved.write_text(json.dumps(resolved_value), encoding="utf-8")
    result = _subprocess(destination, "offline", "--run-dir", parent, "--config", variant_config)
    assert result.returncode != 0
    assert "parent run provenance mismatch" in result.stderr
    assert "config_sha256" in result.stderr
    resolved.write_bytes(resolved_original)

    snapshot = parent / "input_schema.json"
    snapshot_original = snapshot.read_bytes()
    snapshot_value = json.loads(snapshot_original)
    snapshot_value["inputs"][0]["replacement"]["invalid_value"] = -999.0
    snapshot.write_text(json.dumps(snapshot_value), encoding="utf-8")
    result = _subprocess(destination, "offline", "--run-dir", parent, "--config", variant_config)
    assert result.returncode != 0
    assert "parent run provenance mismatch" in result.stderr
    assert "snapshot:input_schema.json" in result.stderr
    snapshot.write_bytes(snapshot_original)

    # Restoring the bytes is part of the fixture cleanup: a successful reuse
    # after restoration demonstrates that only the tampered parent was rejected.
    result = _subprocess(destination, "offline", "--run-dir", parent, "--config", variant_config)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
