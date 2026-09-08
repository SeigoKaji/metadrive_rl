"""No-simulator tests for :mod:`lookahead_learning.portability`.

Run with ``PYTHONDONTWRITEBYTECODE=1 python -B -m unittest -v
lookahead_learning.test_portability``.  The subprocess test relocates source into a
temporary path containing spaces; it does not run the optional raw probe.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from .portability import (
    HOST_MODULE_NAMES,
    HOST_PYTHON_FILES,
    RawWorkerSpec,
    run_portability,
)


class PortabilityTests(unittest.TestCase):
    def test_static_relocation_uses_only_py_files_and_space_path(self) -> None:
        source_root = Path(__file__).resolve().parents[1]
        config = source_root / "configs" / "official.toml"
        with tempfile.TemporaryDirectory(prefix="preview portability ") as directory:
            output = Path(directory) / "new output with spaces"
            report = run_portability(
                source_root,
                output,
                config,
                probe=False,
            )
            self.assertTrue(report["ok"], report)
            self.assertIn(" ", report["relocated_root"])
            self.assertTrue(report["checks"]["normal_import"]["import_result"]["no_original_residual"])
            self.assertTrue(report["checks"]["help"]["commands_present"])
            self.assertTrue(report["checks"]["static_doctor"]["ok"])
            self.assertFalse(report["assets_copied"])
            self.assertFalse(report["models_copied"])
            self.assertFalse(report["different_pc_verification"])

            relocated = Path(report["relocated_root"])
            copied_files = [path for path in relocated.rglob("*") if path.is_file()]
            relative_files = {path.relative_to(relocated) for path in copied_files}
            self.assertTrue(relative_files)
            self.assertTrue(all(path.suffix in {".py", ".toml", ".json"} for path in relative_files))
            self.assertTrue(all(path.suffix == ".py" for path in relative_files if path.parts[0] == "lookahead_learning"))
            self.assertEqual(
                relative_files & {Path(name) for name in HOST_PYTHON_FILES},
                {Path(name) for name in HOST_PYTHON_FILES},
            )
            self.assertIn(Path("configs/official.toml"), relative_files)
            self.assertFalse(any(path.parts[0] in {"assets", "models", "outputs", "logs"} for path in relative_files))

            report_path = output / "portability_report.json"
            self.assertTrue(report_path.is_file())
            saved = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["schema_version"], report["schema_version"])

    def test_raw_worker_spec_contains_no_parent_runtime_objects(self) -> None:
        spec = RawWorkerSpec("/project", "/project/configs/official.toml", 0, 5)
        self.assertEqual(spec.worker, 0)
        self.assertEqual(spec.seed, 5)
        self.assertEqual(
            set(spec.__dataclass_fields__),
            {"project_root", "config_path", "worker", "seed", "steps"},
        )
        self.assertIn("env_factory", HOST_MODULE_NAMES)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
