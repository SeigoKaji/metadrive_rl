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
            {**LOOKAHEAD_DEFAULTS, "lookahead_m": 6.0, "pp_weight": 0.0},
        )
        self.assertEqual(
            resolve_lookahead_config({"lookahead_m": 8.5}),
            {**LOOKAHEAD_DEFAULTS, "lookahead_m": 8.5, "pp_weight": 0.0},
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
        self.assertEqual(model.lookahead_config, resolve_lookahead_config(expected))
        self.assertEqual(model.lookahead_schema_version, 2)

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

    def test_new_config_types_limits_and_unknown_keys(self) -> None:
        for key, bad_values in {
            "lateral_accel_reward_enabled": (0, 1, 0.0, "true", None, math.nan),
            "max_lateral_accel": (True, False, "0.8", None, 0, -1, math.nan, math.inf),
            "lateral_accel_weight": (True, "0.1", None, -1, math.nan, math.inf),
        }.items():
            for value in bad_values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    resolve_lookahead_config({key: value})
        resolved = resolve_lookahead_config({"lateral_accel_reward_enabled": True})
        self.assertIs(resolved["lateral_accel_reward_enabled"], True)
        self.assertEqual(resolved["max_lateral_accel"], 0.8)
        self.assertEqual(resolved["lateral_accel_weight"], 0.1)

    def test_v1_off_compatibility_without_mutating_the_old_model(self) -> None:
        old = {"lookahead_m": 6.0, "pp_weight": 0.25}
        model = SimpleNamespace(lookahead_schema_version=1, lookahead_config=old.copy())
        off = {**old, "max_lateral_accel": 9.0, "lateral_accel_weight": 4.0}
        validate_lookahead_model_metadata(model, off)
        validate_lookahead_model_metadata(
            model, {**old, "lateral_accel_reward_enabled": True, "lateral_accel_weight": 0}
        )
        self.assertEqual(model.lookahead_schema_version, 1)
        self.assertEqual(model.lookahead_config, old)
        for mismatch in (
            {"lateral_accel_reward_enabled": True},
            {"lookahead_m": 7}, {"pp_weight": 0},
        ):
            with self.subTest(mismatch=mismatch), self.assertRaises(CheckpointContractError):
                validate_lookahead_model_metadata(model, {**old, **mismatch})

    def test_v2_compares_effective_settings_and_rejects_unknown_schemas(self) -> None:
        on = resolve_lookahead_config({"lateral_accel_reward_enabled": True})
        model = SimpleNamespace()
        set_lookahead_model_metadata(model, on)
        validate_lookahead_model_metadata(model, on)
        for changed in (
            {"lateral_accel_reward_enabled": False},
            {"max_lateral_accel": 1.6}, {"lateral_accel_weight": 0.2},
        ):
            with self.subTest(changed=changed), self.assertRaises(CheckpointContractError):
                validate_lookahead_model_metadata(model, {**on, **changed})
        set_lookahead_model_metadata(model, {})
        validate_lookahead_model_metadata(model, {"max_lateral_accel": 2, "lateral_accel_weight": 0})
        for schema in (0, 3, True, 2.0, "2", None):
            for expected in ({}, None):
                malformed = SimpleNamespace(
                    lookahead_schema_version=schema, lookahead_config=expected
                )
                with self.subTest(schema=schema), self.assertRaises(CheckpointContractError):
                    validate_lookahead_model_metadata(malformed, expected)
        with self.assertRaises(CheckpointContractError):
            validate_lookahead_model_metadata(
                SimpleNamespace(lookahead_schema_version=1, lookahead_config=on), {}
            )
        with self.assertRaises(CheckpointContractError):
            validate_lookahead_model_metadata(
                SimpleNamespace(lookahead_schema_version=2, lookahead_config=None), {}
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
