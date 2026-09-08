"""No-simulator tests for the lazy CLI and relocated package behavior."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from .diagnostics import asset_audit, dependency_audit
from .runner import (
    ContractRefusal,
    _ADDON_RUNTIME_SOURCE_FILES,
    _HOST_RUNTIME_SOURCE_FILES,
    _source_identities,
    build_parser,
)


class RunnerParserTests(unittest.TestCase):
    def test_parser_exposes_all_commands_and_preserves_explicit_zero(self) -> None:
        parser = build_parser()
        self.assertEqual(parser.parse_args(["doctor"]).command, "doctor")
        self.assertEqual(parser.parse_args(["train", "--mode", "baseline"]).mode, "baseline")
        self.assertEqual(
            parser.parse_args(["train", "--mode", "lookahead_obs"]).mode,
            "lookahead_obs",
        )
        self.assertEqual(
            parser.parse_args(["train", "--mode", "lookahead_obs_pp_reward"]).mode,
            "lookahead_obs_pp_reward",
        )
        self.assertEqual(
            parser.parse_args(["train", "--mode", "obs"]).mode,
            "lookahead_obs",
        )
        self.assertEqual(
            parser.parse_args(["train", "--mode", "obs_pp"]).mode,
            "lookahead_obs_pp_reward",
        )
        self.assertEqual(
            parser.parse_args(["train", "--timesteps", "0"]).timesteps,
            0,
        )
        self.assertEqual(parser.parse_args(["evaluate", "--checkpoint", "x.zip"]).checkpoint, "x.zip")
        self.assertEqual(parser.parse_args(["compare", "a.json", "b.json"]).command, "compare")
        compare = parser.parse_args(
            [
                "compare",
                "a.json",
                "b.json",
                "--before-mode",
                "obs",
                "--after-mode",
                "obs_pp",
            ]
        )
        self.assertEqual(compare.before_mode, "lookahead_obs")
        self.assertEqual(compare.after_mode, "lookahead_obs_pp_reward")
        self.assertEqual(parser.parse_args(["test"]).command, "test")

    def test_parser_import_is_lazy_in_a_fresh_interpreter(self) -> None:
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                (
                    "import sys; "
                    "from lookahead_learning.runner import build_parser; "
                    "assert build_parser().parse_args(['doctor']).command == 'doctor'; "
                    "assert 'metadrive' not in sys.modules; "
                    "print('lazy-parser-ok')"
                ),
            ],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "lazy-parser-ok")

    def test_help_runs_without_optional_runtime_imports(self) -> None:
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            [sys.executable, "-B", "-m", "lookahead_learning", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("doctor", result.stdout)
        self.assertIn("compare", result.stdout)

        train_help = subprocess.run(
            [sys.executable, "-B", "-m", "lookahead_learning", "train", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(train_help.returncode, 0, train_help.stderr)
        self.assertIn("lookahead_obs", train_help.stdout)
        self.assertIn("lookahead_obs_pp_reward", train_help.stdout)
        self.assertIn("前方注視点入力追加", train_help.stdout)
        self.assertIn("Pure Pursuit", train_help.stdout)


class SourceIdentityTests(unittest.TestCase):
    @staticmethod
    def _write_runtime_fixture(root: Path) -> None:
        for relative_path in (
            *_HOST_RUNTIME_SOURCE_FILES,
            *_ADDON_RUNTIME_SOURCE_FILES,
        ):
            destination = root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(relative_path.encode("utf-8") + b"\n")

    def test_source_identities_are_portable_and_detect_host_source_change(self) -> None:
        """A copied runtime is equal; one host byte changes its contract."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "host source"
            copied_root = Path(directory) / "copied host source"
            self._write_runtime_fixture(root)
            original = _source_identities(root)
            shutil.copytree(root, copied_root)
            copied = _source_identities(copied_root)

            self.assertEqual(original, copied)
            self.assertTrue(
                all(
                    not Path(relative_path).is_absolute()
                    for relative_path in original["host"]
                )
            )
            self.assertTrue(
                all(
                    not Path(relative_path).is_absolute()
                    for relative_path in original["addon"]
                )
            )
            changed_source = root / "start_lane_env.py"
            changed_source.write_bytes(changed_source.read_bytes() + b"!")
            changed = _source_identities(root)
            self.assertNotEqual(
                original["host"]["start_lane_env.py"],
                changed["host"]["start_lane_env.py"],
            )
            self.assertNotEqual(original, changed)

            changed_source.unlink()
            with self.assertRaisesRegex(
                ContractRefusal,
                r"required host runtime source is missing: start_lane_env\.py",
            ):
                _source_identities(root)


class RelocatedPackageTests(unittest.TestCase):
    def test_relocated_package_uses_installed_dependency_spec_without_copying_assets(self) -> None:
        """A copied package must not derive MetaDrive assets from ``root/..``."""

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "relocated" / "lookahead_learning"
            destination.mkdir(parents=True)
            source = Path(__file__).resolve().parent
            for source_file in source.glob("*.py"):
                shutil.copy2(source_file, destination / source_file.name)
            relocated_root = destination.parent
            command = [
                sys.executable,
                "-B",
                "-c",
                (
                    "import json, pathlib, lookahead_learning, lookahead_learning.diagnostics as d; "
                    "assert str(pathlib.Path(lookahead_learning.__file__).resolve()).startswith(%r); "
                    "r=d.asset_audit(%r); "
                    "print(json.dumps({'package_dir': r.get('package_dir'), "
                    "'asset_root': r.get('asset_root'), "
                    "'would_update': r.get('normal_engine_try_pull_asset_would_update')}))"
                ) % (str(relocated_root), str(relocated_root)),
            ]
            environment = dict(os.environ)
            environment.pop("PYTHONPATH", None)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            # ``lookahead_learning`` is copied below ``relocated_root``.  With
            # PYTHONPATH deliberately removed, this cwd is the import root
            # that proves the child is using the copied package.
            result = subprocess.run(command, cwd=str(relocated_root), env=environment, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            # The dependency location is the interpreter's actual MetaDrive
            # package.  It may be absent on a minimal installation, but it can
            # never be the temporary copied checkout's sibling.
            if report["package_dir"] is not None:
                self.assertNotIn(str(relocated_root), report["package_dir"])

    def test_static_audit_reports_import_spec(self) -> None:
        report = dependency_audit(Path(__file__).resolve().parents[1])
        metadrive = report["packages"]["metadrive"]
        self.assertIn("spec", metadrive)
        self.assertIn("search_locations", metadrive["spec"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
