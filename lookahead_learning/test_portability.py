"""Copy only this folder and run contracts/fake hosts without the source root."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class PackageCopyTests(unittest.TestCase):
    def test_folder_alone_has_no_required_source_host_imports(self):
        package = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(prefix="lookahead-port-") as directory:
            copied = Path(directory) / "lookahead_learning"
            shutil.copytree(package, copied, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            script = """
import importlib.abc
import os
from pathlib import Path
import sys
import unittest
sys.path.insert(0, os.getcwd())
class BlockSourceHost(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'configs', 'env_factory', 'start_lane_env', 'train', 'evaluate', 'metadrive'
        }:
            raise ModuleNotFoundError('source host/simulator blocked: ' + fullname)
sys.meta_path.insert(0, BlockSourceHost())
import lookahead_learning
assert Path(lookahead_learning.__file__).resolve().parent == Path.cwd() / 'lookahead_learning'
modules = [
    'test_checkpoint', 'test_geometry', 'test_env',
    'test_lateral_acceleration', 'test_lateral_env',
]
suite = unittest.defaultTestLoader.loadTestsFromNames(
    ['lookahead_learning.' + name for name in modules]
)
result = unittest.TextTestRunner(verbosity=1).run(suite)
assert 'metadrive' not in sys.modules
sys.exit(0 if result.wasSuccessful() else 1)
"""
            environment = os.environ.copy()
            environment.pop("PYTHONPATH", None)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            result = subprocess.run(
                [sys.executable, "-I", "-B", "-c", script],
                cwd=directory, env=environment, capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("OK", result.stderr)


if __name__ == "__main__":
    unittest.main()
