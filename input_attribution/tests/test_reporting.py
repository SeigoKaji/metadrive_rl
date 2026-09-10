"""Small saved-data tests for the simulator-free report boundary."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from PIL import Image

import input_attribution.reporting as reporting
from input_attribution.reporting import SUMMARY_COLUMNS, generate_report
from input_attribution.visuals import LN2, js_divergence, render_rewards_plot


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _fixture_run(root: Path) -> None:
    data = root / "data"
    data.mkdir(parents=True)
    _write_json(
        data / "manifest.json",
        {
            "format_version": "input-attribution.v1",
            "base_main_sha": "7849aad80ac353fd616c1a1398c11dd3497eed05",
            "implementation_sha": "synthetic",
            "dirty": False,
            "backend": "synthetic",
            "config": {
                "environment_config": "configs/demo.toml",
                "scenario_seed": 5,
                "policy_seed": 5,
                "fixed_value": -1.0,
            },
            "adapter": {
                "model": {"path": "models/demo.zip", "sha256": "abc123"},
                "environment": {"config": {"map": "C"}},
            },
            "patterns": [
                {"id": "P01", "name": "speed", "indices": [3], "fixed_value": -1.0},
                {"id": "P02", "name": "lidar front", "indices": [9, 10, 11], "fixed_value": -1.0},
            ],
            "status": "complete",
        },
    )
    (data / "input_schema.csv").write_text(
        "index,name_ja\n3,速度\n9,lidar前\n10,lidar前\n11,lidar前\n", encoding="utf-8"
    )

    def records(identifier: str, count: int) -> list[dict[str, object]]:
        result = []
        for step in range(count):
            frame_path = f"data/frames/{identifier}/{step:06d}.png"
            baseline = identifier == "P00_baseline"
            result.append(
                {
                    "step": step,
                    "raw_observation": [step, 0.0, 0.2, 0.4],
                    "input": [step, 0.0, 0.2, 0.4],
                    "modified_input": [step, 0.0, 0.2, 0.4] if baseline else [step, 0.0, 0.2, -1.0],
                    "probabilities": [1.0, 0.0],
                    "action": 0,
                    "executed": True,
                    "reward": 1.0 if baseline else -0.5,
                    "terminated": bool(step == count - 1),
                    "truncated": False,
                    "info": {},
                    "state": {"position": step},
                    "reward_terms": {"status": "unavailable", "terms": None, "residual": None, "atol": 1e-6, "rtol": 1e-6},
                    "frame": frame_path,
                    "sim_time": (step + 1) * 0.1,
                }
            )
        return result

    _write_json(
        data / "P00_baseline.json",
        {
            "id": "P00_baseline",
            "name": "baseline",
            "status": "complete",
            "comparable": True,
            "initial": {"observation_hash": "baseline", "state": {}, "seed": 5},
            "action_dt": 0.1,
            "records": records("P00_baseline", 3),
        },
    )
    for identifier, count, name, indices in (
        ("P01", 3, "speed", [3]),
        ("P02", 2, "lidar front", [9, 10, 11]),
    ):
        _write_json(
            data / f"{identifier}.json",
            {
                "id": identifier,
                "name": name,
                "pattern": {"id": identifier, "name": name, "indices": indices, "fixed_value": -1.0},
                "status": "complete",
                "comparable": True,
                "comparison": {"status": "comparable", "comparable": True, "reasons": []},
                "applied_count": count,
                "changed_count": count,
                "unchanged_count": 0,
                "action_dt": 0.1,
                "records": records(identifier, count),
            },
        )
        _write_json(
            data / f"{identifier}_offline.json",
            {
                "id": identifier,
                "status": "complete",
                "records": [
                    {"step": 0, "js": 0.0, "argmax_changed": False, "p": [1.0, 0.0], "q": [1.0, 0.0], "applied": True, "changed": True},
                    {"step": 1, "js": LN2, "argmax_changed": True, "p": [1.0, 0.0], "q": [0.0, 1.0], "applied": True, "changed": True},
                ],
            },
        )
    for identifier, count in (("P00_baseline", 3), ("P01", 3), ("P02", 2)):
        frame_dir = data / "frames" / identifier
        frame_dir.mkdir(parents=True)
        for step in range(count):
            Image.new("RGB", (96, 64), (20 + step * 30, 80, 150)).save(
                frame_dir / f"{step:06d}.png", format="PNG"
            )


def test_js_divergence_boundaries_and_zero_probability() -> None:
    assert js_divergence([0.5, 0.5], [0.5, 0.5]) == 0.0
    assert js_divergence([1.0, 0.0], [0.0, 1.0]) == LN2
    assert js_divergence([0.0, 1.0], [1.0, 0.0]) == LN2


def test_report_writes_six_columns_and_keeps_short_run_length(tmp_path: Path) -> None:
    _fixture_run(tmp_path)
    raw_path = tmp_path / "data" / "P02.json"
    before = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    result = generate_report(tmp_path)
    assert result.status == "success", result.errors
    assert len(result.summary_rows) == 3
    with (tmp_path / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        table = list(csv.DictReader(stream))
    assert list(table[0]) == list(SUMMARY_COLUMNS)
    assert len(table) == 3
    assert table[0]["変更対象"].startswith("P00")
    assert table[2]["変更後累積報酬"] == "-1"
    assert table[2]["終了step・理由"].startswith("step 1（2 step）")
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == before
    assert (tmp_path / "baseline" / "rollout.gif").is_file()
    assert (tmp_path / "baseline" / "rewards.png").is_file()
    assert (tmp_path / "patterns" / "P01" / "rollout.gif").is_file()
    assert (tmp_path / "patterns" / "P01" / "rewards.png").is_file()
    assert (tmp_path / "patterns" / "P01" / "policy_change.png").is_file()
    assert (tmp_path / "patterns" / "P02" / "policy_change.png").is_file()
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert html.index("id=\"comparison\"") < html.index("id=\"baseline\"")
    assert html.index("id=\"baseline\"") < html.index("id=\"closed-loop\"")
    assert html.index("id=\"closed-loop\"") < html.index("id=\"offline\"")
    assert 'href="patterns/P01/rollout.gif"' in html
    assert 'href="data/input_schema.csv"' in html
    assert 'id="pattern-P01"' in html and 'id="offline-P01"' in html
    assert "applied=2/executed=2, changed=2, unchanged=0" in html
    assert "各step実測値→-1 / 書換え前範囲" in html
    assert "9: 実測値" not in html
    assert "align-items: start" in html
    assert "models/demo.zip" in html and "sha256=abc123" in html
    assert "map=C" in html and "scenario_seed=5" in html and "policy_seed" in html


def test_report_uses_actual_127_and_50_step_sums_and_masks_partial_difference(tmp_path: Path) -> None:
    _fixture_run(tmp_path)
    data = tmp_path / "data"
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    manifest["config"]["record_gif"] = False
    _write_json(data / "manifest.json", manifest)

    def rows(count: int, reward: float, status_at_end: bool = True) -> list[dict[str, object]]:
        return [
            {
                "step": step,
                "input": [0.0, 0.0, 0.0, 0.4],
                "modified_input": [0.0, 0.0, 0.0, -1.0],
                "executed": True,
                "reward": reward,
                "terminated": bool(status_at_end and step == count - 1),
                "truncated": False,
                "frame": None,
            }
            for step in range(count)
        ]

    baseline = json.loads((data / "P00_baseline.json").read_text(encoding="utf-8"))
    baseline["records"] = rows(127, 1.0)
    _write_json(data / "P00_baseline.json", baseline)
    p01 = json.loads((data / "P01.json").read_text(encoding="utf-8"))
    p01["records"] = rows(50, -1.0)
    _write_json(data / "P01.json", p01)
    failed = json.loads((data / "P02.json").read_text(encoding="utf-8"))
    failed["status"] = "failed"
    # A stale positive pairing flag must not make a failed run comparable.
    failed["comparable"] = True
    failed["comparison"] = {"status": "comparable", "comparable": True}
    failed["records"] = rows(2, -2.0, status_at_end=False)
    failed["failure"] = {"stage": "step", "step": 2, "error": "synthetic failure"}
    _write_json(data / "P02.json", failed)

    result = generate_report(tmp_path)
    assert result.status == "partial"
    by_id = {row["pattern_id"]: row for row in result.summary_rows}
    assert by_id["P00"]["changed_cumulative_reward"] == 127.0
    assert by_id["P01"]["changed_cumulative_reward"] == -50.0
    assert by_id["P01"]["reward_difference"] == -177.0
    assert by_id["P01"]["終了step・理由"].startswith("step 49（50 step）")
    assert by_id["P02"]["changed_cumulative_reward"] == -4.0
    assert by_id["P02"]["reward_difference"] is None
    assert "N/A" in by_id["P02"]["終了step・理由"] or "実行失敗" in by_id["P02"]["終了step・理由"]


def test_report_import_path_does_not_require_simulator_or_torch_and_renders(tmp_path: Path) -> None:
    _fixture_run(tmp_path)
    script = (
        "import sys\n"
        "import builtins\n"
        "real_import = builtins.__import__\n"
        "def guarded(name, *args, **kwargs):\n"
        "    if name.split('.')[0] in {'torch', 'metadrive', 'stable_baselines3', 'captum'}:\n"
        "        raise AssertionError(name)\n"
        "    return real_import(name, *args, **kwargs)\n"
        "builtins.__import__ = guarded\n"
        "from input_attribution.reporting import generate_report\n"
        "result = generate_report(sys.argv[1])\n"
        "print(result.status)\n"
    )
    env = dict(__import__("os").environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    repo_root = Path(__file__).parents[2]
    env["PYTHONPATH"] = str(repo_root) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "success"
    assert (tmp_path / "report.html").is_file()
    assert (tmp_path / "baseline" / "rewards.png").is_file()
    assert (tmp_path / "patterns" / "P01" / "policy_change.png").is_file()


def test_report_identifies_synthetic_backend_and_model_without_archive(tmp_path: Path) -> None:
    _fixture_run(tmp_path)
    manifest_path = tmp_path / "data" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["backend"] = "synthetic"
    manifest["adapter"]["model"] = {
        "kind": "synthetic_softmax",
        "weights_sha256": "weights-demo",
    }
    _write_json(manifest_path, manifest)
    result = generate_report(tmp_path)
    assert result.status == "success", result.errors
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "synthetic — 合成デモ（実MetaDrive走行ではありません）" in html
    assert "kind=synthetic_softmax (weights_sha256=weights-demo)" in html
    assert "model.zip" not in html


def test_copied_report_has_only_relative_media_links(tmp_path: Path) -> None:
    _fixture_run(tmp_path)
    result = generate_report(tmp_path)
    assert result.status == "success", result.errors
    copied = tmp_path.parent / f"{tmp_path.name}-copy"
    shutil.copytree(tmp_path, copied)
    html = (copied / "report.html").read_text(encoding="utf-8")
    assert "https://" not in html and "http://" not in html
    for href in (
        "baseline/rollout.gif",
        "baseline/rewards.png",
        "patterns/P01/rollout.gif",
        "patterns/P01/policy_change.png",
        "data/input_schema.csv",
    ):
        assert (copied / href).is_file(), href


def test_report_rejects_unsafe_pattern_id_without_reading_external_offline_file(
    tmp_path: Path, monkeypatch
) -> None:
    _fixture_run(tmp_path)
    outside_id = tmp_path.parent / "outside-report-pattern"
    outside_offline = Path(f"{outside_id}_offline.json")
    _write_json(
        outside_offline,
        {
            "id": str(outside_id),
            "status": "complete",
            "records": [{"step": 0, "js": LN2}],
        },
    )
    manifest_path = tmp_path / "data" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["patterns"].append(
        {"id": str(outside_id), "name": "external", "indices": [3], "fixed_value": -1.0}
    )
    _write_json(manifest_path, manifest)

    read_paths: list[Path] = []
    read_json = reporting._read_json

    def spy(path: Path):
        read_paths.append(Path(path))
        return read_json(path)

    monkeypatch.setattr(reporting, "_read_json", spy)
    result = generate_report(tmp_path)
    assert result.status == "failed"
    assert any("unsafe pattern identifier" in error for error in result.errors)
    assert outside_offline not in read_paths
    assert not (tmp_path / "patterns" / "outside-report-pattern").exists()
    outside_offline.unlink()


def test_report_rejects_duplicate_pattern_id_without_sanitized_collision_output(tmp_path: Path) -> None:
    _fixture_run(tmp_path)
    manifest_path = tmp_path / "data" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["patterns"].append(
        {"id": "P01", "name": "duplicate", "indices": [3], "fixed_value": -1.0}
    )
    _write_json(manifest_path, manifest)
    result = generate_report(tmp_path)
    assert result.status == "failed"
    assert any("duplicate pattern identifier" in error for error in result.errors)
    assert not (tmp_path / "patterns" / "P01_2").exists()


def test_reward_plot_marks_mixed_term_status_and_counts(tmp_path: Path) -> None:
    records = [
        {
            "step": 0,
            "executed": True,
            "reward": 1.0,
            "reward_terms": {"status": "verified", "terms": {"progress": 1.0}},
        },
        {
            "step": 1,
            "executed": True,
            "reward": 2.0,
            "reward_terms": {"status": "mismatch", "terms": {"progress": 2.0}},
        },
        {"step": 2, "executed": True, "reward": 3.0},
    ]
    result = render_rewards_plot(records, tmp_path / "mixed-rewards.png")
    assert result.status == "complete", result.errors
    assert result.metadata["terms_status"] == "mixed"
    assert result.metadata["terms_status_counts"] == {
        "verified": 1,
        "mismatch": 1,
        "unavailable": 1,
    }
    assert (tmp_path / "mixed-rewards.png").is_file()


def test_report_hashes_nested_frames_and_preserves_all_raw_data_on_regeneration(tmp_path: Path) -> None:
    _fixture_run(tmp_path)
    data = tmp_path / "data"

    def hashes() -> dict[str, str]:
        return {
            path.relative_to(tmp_path).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(data.rglob("*"))
            if path.is_file()
        }

    before = hashes()
    result = generate_report(tmp_path)
    assert result.status == "success", result.errors
    metadata = json.loads((tmp_path / "report_metadata.json").read_text(encoding="utf-8"))
    assert set(metadata["raw_data_sha256"]) == set(before)
    assert metadata["raw_data_sha256"]["data/frames/P02/000001.png"] == before[
        "data/frames/P02/000001.png"
    ]
    generate_report(tmp_path)
    assert hashes() == before
    metadata_again = json.loads((tmp_path / "report_metadata.json").read_text(encoding="utf-8"))
    assert metadata_again["raw_data_sha256"] == metadata["raw_data_sha256"]


def test_report_removes_stale_media_after_visual_producer_failure(tmp_path: Path, monkeypatch) -> None:
    _fixture_run(tmp_path)
    first = generate_report(tmp_path)
    assert first.status == "success", first.errors
    old_paths = (
        tmp_path / "baseline" / "rollout.gif",
        tmp_path / "baseline" / "rewards.png",
        tmp_path / "patterns" / "P01" / "rollout.gif",
        tmp_path / "patterns" / "P01" / "rewards.png",
        tmp_path / "patterns" / "P01" / "policy_change.png",
    )
    assert all(path.is_file() for path in old_paths)

    def fail_visual(*args, **kwargs):
        return reporting.VisualResult(None, "failed", {}, ("forced visual failure",))

    monkeypatch.setattr(reporting, "render_rollout_gif", fail_visual)
    monkeypatch.setattr(reporting, "render_rewards_plot", fail_visual)
    monkeypatch.setattr(reporting, "render_policy_change_plot", fail_visual)
    second = generate_report(tmp_path)
    assert second.status == "failed"
    assert all(not path.exists() for path in old_paths)
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    for href in (
        "baseline/rollout.gif",
        "baseline/rewards.png",
        "patterns/P01/rollout.gif",
        "patterns/P01/rewards.png",
        "patterns/P01/policy_change.png",
    ):
        assert f'href="{href}"' not in html
