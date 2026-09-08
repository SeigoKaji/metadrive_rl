"""Portable test suite entry point.

The default suite discovers every ``test_*.py`` file in this package, including
the fake-environment contract tests and the static relocation/portability
checks. MetaDrive/Panda3D are never initialized by those tests. Integration
probing is an explicit CLI option because it performs a real simulator reset
and step.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import unittest


def load_tests(
    loader: unittest.TestLoader,
    _standard_tests: unittest.TestSuite,
    _pattern: str | None,
) -> unittest.TestSuite:
    suite = unittest.TestSuite()
    package_dir = Path(__file__).resolve().parent
    module_names = sorted(
        f"lookahead_learning.{path.stem}"
        for path in package_dir.glob("test_*.py")
    )
    for module_name in module_names:
        module = importlib.import_module(module_name)
        suite.addTests(loader.loadTestsFromModule(module))
    return suite


__all__ = ["load_tests"]
