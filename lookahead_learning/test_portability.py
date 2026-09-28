"""Only the manifest is copied; source root, simulator and learner are blocked."""
import ast
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class PackageCopyTests(unittest.TestCase):
    def test_manifest_dependency_closure_and_isolated_imports(self):
        package = Path(__file__).resolve().parent
        paths = [line for line in (package / 'PORTABLE_FILES.txt').read_text().splitlines()
                 if line and not line.startswith('#')]
        self.assertEqual(len(paths), len(set(paths)))
        with tempfile.TemporaryDirectory(prefix='lookahead-port-') as directory:
            copied = Path(directory) / 'lookahead_learning'
            for name in paths:
                relative = Path(name)
                self.assertEqual(relative.parts[0], 'lookahead_learning')
                self.assertNotIn('..', relative.parts)
                self.assertIn(relative.suffix, ('.py', '.md', '.toml', '.txt'))
                source = package.parent / relative
                destination = Path(directory) / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            # Check every relative Python import (including deferred adapter imports).
            for source in copied.glob('*.py'):
                for node in ast.walk(ast.parse(source.read_text())):
                    if isinstance(node, ast.ImportFrom) and node.level and node.module:
                        self.assertEqual(node.level, 1)
                        self.assertTrue((copied / (node.module.replace('.', '/') + '.py')).is_file(),
                                        (source.name, node.module))
            script = '''
import importlib.abc
import os
from pathlib import Path
import sys
import tomllib
import unittest
sys.path.insert(0, os.getcwd())
pure = sys.argv[1] == 'pure'
class BlockSourceHost(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = {'configs', 'env_factory', 'start_lane_env', 'train', 'evaluate',
                   'metadrive', 'stable_baselines3'}
        if pure:
            blocked.update({'numpy', 'gymnasium'})
        if fullname.split('.')[0] in blocked:
            raise ModuleNotFoundError('source host/dependency blocked: ' + fullname)
sys.meta_path.insert(0, BlockSourceHost())
import lookahead_learning
from lookahead_learning.checkpoint import resolve_lookahead_config
assert Path(lookahead_learning.__file__).resolve().parent == Path.cwd() / 'lookahead_learning'
for example in Path('lookahead_learning/examples').glob('*.toml'):
    resolve_lookahead_config(tomllib.loads(example.read_text())['lookahead'])
modules = ['test_checkpoint', 'test_geometry', 'test_lateral_acceleration', 'test_prediction']
if not pure:
    modules += ['test_env', 'test_lateral_env', 'test_prediction_env']
suite = unittest.defaultTestLoader.loadTestsFromNames(['lookahead_learning.' + m for m in modules])
result = unittest.TextTestRunner(verbosity=1).run(suite)
assert 'metadrive' not in sys.modules and 'stable_baselines3' not in sys.modules
if pure:
    assert 'numpy' not in sys.modules and 'gymnasium' not in sys.modules
sys.exit(0 if result.wasSuccessful() else 1)
'''
            environment = os.environ.copy()
            environment.pop('PYTHONPATH', None)
            environment['PYTHONDONTWRITEBYTECODE'] = '1'
            for mode in ('pure', 'fake'):
                with self.subTest(mode=mode):
                    result = subprocess.run(
                        [sys.executable, '-I', '-B', '-c', script, mode],
                        cwd=directory, env=environment, capture_output=True, text=True, timeout=60,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn('OK', result.stderr)


if __name__ == '__main__':
    unittest.main()
