"""Portable lookahead configuration and PPO ZIP attribute tests.

These tests use only standard-library objects, so a copied
``lookahead_learning`` directory can validate its contract without importing
the host project, MetaDrive, or Stable-Baselines3.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
import unittest

from .checkpoint import (
    CheckpointContractError,
    LOOKAHEAD_DEFAULTS,
    resolve_lookahead_config,
    set_lookahead_model_metadata,
    validate_lookahead_model_metadata,
)


class LookaheadConfigTests(unittest.TestCase):
    def test_resolve_lookahead_config_uses_presence_as_enable_switch(self) -> None:
        self.assertIsNone(resolve_lookahead_config(None))
        resolved = resolve_lookahead_config({})
        self.assertEqual(resolved, LOOKAHEAD_DEFAULTS)
        self.assertIsNot(resolved, LOOKAHEAD_DEFAULTS)
        self.assertEqual(
            resolve_lookahead_config({"lookahead_m": 6, "pp_weight": 0}),
            {"lookahead_m": 6.0, "pp_weight": 0.0},
        )
        self.assertEqual(
            resolve_lookahead_config({"lookahead_m": 8.5}),
            {"lookahead_m": 8.5, "pp_weight": 0.0},
        )

    def test_resolve_lookahead_config_rejects_invalid_values(self) -> None:
        invalid = (
            [],
            {1: 1},
            {"unknown": 1},
            {"lookahead_m": 0},
            {"lookahead_m": True},
            {"lookahead_m": "6"},
            {"lookahead_m": math.nan},
            {"lookahead_m": math.inf},
            {"pp_weight": -1},
            {"pp_weight": True},
            {"pp_weight": "0"},
            {"pp_weight": math.nan},
            {"pp_weight": math.inf},
        )
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    resolve_lookahead_config(value)

    def test_model_attributes_are_self_contained_and_strictly_compared(self) -> None:
        expected = {"lookahead_m": 6.0, "pp_weight": 0.0}
        model = SimpleNamespace()
        set_lookahead_model_metadata(model, expected)
        validate_lookahead_model_metadata(model, expected)
        self.assertEqual(model.lookahead_config, expected)
        self.assertEqual(model.lookahead_schema_version, 1)

        with self.assertRaisesRegex(CheckpointContractError, "do not match"):
            validate_lookahead_model_metadata(
                model,
                {"lookahead_m": 6.0, "pp_weight": 0.25},
            )
        with self.assertRaisesRegex(CheckpointContractError, "schema metadata"):
            validate_lookahead_model_metadata(
                SimpleNamespace(lookahead_config=expected),
                expected,
            )
        with self.assertRaisesRegex(CheckpointContractError, "active lookahead"):
            validate_lookahead_model_metadata(model, None)

        # Old baseline ZIPs and newly saved baseline models remain valid when
        # the selected TOML has no [lookahead] table.
        validate_lookahead_model_metadata(SimpleNamespace(), None)
        baseline = SimpleNamespace()
        set_lookahead_model_metadata(baseline, None)
        validate_lookahead_model_metadata(baseline, None)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
