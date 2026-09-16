"""Portable lookahead configuration and checkpoint compatibility contracts.

This module intentionally uses only the Python standard library.  It does not
load PPO, inspect tensors, import the host project, or write metadata.  A
runner may use :func:`resolve_lookahead_config` for the TOML-facing settings and
attach the resulting mapping to a PPO object before saving it.  Model
compatibility is checked from those ZIP-carried attributes, so this module
does not require an extra artifact or host-specific metadata.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import TypedDict


class CheckpointContractError(ValueError):
    """ZIP-carried lookahead attributes violate the compatibility contract."""


class LookaheadConfig(TypedDict):
    lookahead_m: float
    pp_weight: float
    lateral_accel_reward_enabled: bool
    max_lateral_accel: float
    lateral_accel_weight: float


LOOKAHEAD_DEFAULTS: LookaheadConfig = {
    "lookahead_m": 6.0,
    "pp_weight": 0.0,
    "lateral_accel_reward_enabled": False,
    "max_lateral_accel": 0.8,
    "lateral_accel_weight": 0.1,
}
"""Resolved defaults used whenever an explicit ``[lookahead]`` table exists."""

LOOKAHEAD_MODEL_SCHEMA_VERSION = 2
LOOKAHEAD_MODEL_CONFIG_ATTRIBUTE = "lookahead_config"
LOOKAHEAD_MODEL_SCHEMA_ATTRIBUTE = "lookahead_schema_version"
_LOOKAHEAD_KEYS = frozenset(LOOKAHEAD_DEFAULTS)
_MISSING = object()


def _lookahead_real(
    value: object,
    *,
    key: str,
    minimum: float,
    minimum_inclusive: bool,
) -> float:
    """Validate one TOML-facing finite scalar without importing root config."""

    location = f"lookahead.{key}"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location}: boolではない有限の数値で指定してください")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{location}: 有限の数値で指定してください") from error
    if not math.isfinite(number):
        raise ValueError(f"{location}: 有限の数値で指定してください")
    too_small = (
        number < minimum
        if minimum_inclusive
        else number <= minimum
    )
    if too_small:
        comparator = "以上" if minimum_inclusive else "より大きい"
        raise ValueError(f"{location}: {minimum} {comparator}の数値で指定してください")
    return number


def resolve_lookahead_config(value: object) -> LookaheadConfig | None:
    """Resolve an optional TOML ``[lookahead]`` table.

    ``None`` means that the feature is disabled.  Any mapping, including an
    empty one, means enabled and receives the explicit defaults.  This function
    accepts a strict bool switch and finite numeric parameters.  Even inactive
    parameters are validated; their values do not affect model compatibility.
    """

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("lookahead: TOML tableで指定してください")
    if not all(isinstance(key, str) for key in value):
        raise ValueError("lookahead: table keyは文字列で指定してください")
    unknown = sorted(set(value) - _LOOKAHEAD_KEYS)
    if unknown:
        raise ValueError(f"lookahead: 未対応のkeyがあります: {', '.join(unknown)}")
    enabled = value.get(
        "lateral_accel_reward_enabled", LOOKAHEAD_DEFAULTS["lateral_accel_reward_enabled"]
    )
    if not isinstance(enabled, bool):
        raise ValueError("lookahead.lateral_accel_reward_enabled: boolで指定してください")
    return {
        "lookahead_m": _lookahead_real(
            value.get("lookahead_m", LOOKAHEAD_DEFAULTS["lookahead_m"]),
            key="lookahead_m",
            minimum=0.0,
            minimum_inclusive=False,
        ),
        "pp_weight": _lookahead_real(
            value.get("pp_weight", LOOKAHEAD_DEFAULTS["pp_weight"]),
            key="pp_weight",
            minimum=0.0,
            minimum_inclusive=True,
        ),
        "lateral_accel_reward_enabled": enabled,
        "max_lateral_accel": _lookahead_real(
            value.get("max_lateral_accel", LOOKAHEAD_DEFAULTS["max_lateral_accel"]),
            key="max_lateral_accel",
            minimum=0.0,
            minimum_inclusive=False,
        ),
        "lateral_accel_weight": _lookahead_real(
            value.get("lateral_accel_weight", LOOKAHEAD_DEFAULTS["lateral_accel_weight"]),
            key="lateral_accel_weight",
            minimum=0.0,
            minimum_inclusive=True,
        ),
    }


def set_lookahead_model_metadata(
    model: object,
    lookahead_config: Mapping[str, object] | None,
) -> None:
    """Attach resolved settings to attributes serialized inside a PPO ZIP."""

    resolved = resolve_lookahead_config(lookahead_config)
    setattr(model, LOOKAHEAD_MODEL_SCHEMA_ATTRIBUTE, LOOKAHEAD_MODEL_SCHEMA_VERSION)
    setattr(
        model,
        LOOKAHEAD_MODEL_CONFIG_ATTRIBUTE,
        resolved,
    )


def validate_lookahead_model_metadata(
    model: object,
    expected: Mapping[str, object] | None,
) -> None:
    """Compare effective settings, reading v1 as lateral reward Off.

    Validation never mutates a loaded model.  Off/zero-weight lateral settings
    ignore their inactive limit/weight, but active reward settings must match.
    Unknown schemas are rejected even for baseline checkpoints.
    """

    actual = getattr(model, LOOKAHEAD_MODEL_CONFIG_ATTRIBUTE, _MISSING)
    schema = getattr(model, LOOKAHEAD_MODEL_SCHEMA_ATTRIBUTE, _MISSING)
    if schema is not _MISSING and (type(schema) is not int or schema not in (1, 2)):
        raise CheckpointContractError(f"unsupported lookahead schema metadata: {schema!r}")
    if schema is not _MISSING and actual is _MISSING:
        raise CheckpointContractError("incomplete lookahead schema metadata: config is missing")
    try:
        expected_config = resolve_lookahead_config(expected)
    except ValueError as error:
        raise CheckpointContractError(f"invalid expected lookahead settings: {error}") from error
    if expected_config is None:
        # Legacy baseline ZIPs have no custom attributes; newly saved baseline
        # models carry ``None``.  An active checkpoint must not pass solely on
        # an accidentally compatible observation shape.
        if actual is _MISSING or actual is None:
            return
        raise CheckpointContractError(
            "checkpoint contains active lookahead settings but the selected "
            "TOML has no [lookahead] table"
        )
    if schema is _MISSING:
        raise CheckpointContractError(
            "lookahead config is active but the checkpoint has no supported "
            "lookahead schema metadata"
        )
    if not isinstance(actual, Mapping):
        raise CheckpointContractError(
            "checkpoint lookahead settings do not match active TOML (baseline/missing config)"
        )
    if schema == 1 and set(actual) != {"lookahead_m", "pp_weight"}:
        raise CheckpointContractError("schema v1 requires exactly lookahead_m and pp_weight")
    try:
        actual_config = resolve_lookahead_config(actual)
    except ValueError as error:
        raise CheckpointContractError(f"invalid checkpoint lookahead settings: {error}") from error
    assert actual_config is not None
    if _effective_settings(actual_config) != _effective_settings(expected_config):
        raise CheckpointContractError(
            "checkpoint lookahead settings do not match the selected TOML: "
            f"expected={dict(expected_config)!r}, found={dict(actual_config)!r}"
        )


def _effective_settings(config: LookaheadConfig) -> tuple[object, ...]:
    active = config["lateral_accel_reward_enabled"] and config["lateral_accel_weight"] > 0.0
    return (
        config["lookahead_m"], config["pp_weight"], active,
        config["max_lateral_accel"] if active else None,
        config["lateral_accel_weight"] if active else None,
    )


__all__ = [
    "CheckpointContractError",
    "LOOKAHEAD_DEFAULTS",
    "LOOKAHEAD_MODEL_CONFIG_ATTRIBUTE",
    "LOOKAHEAD_MODEL_SCHEMA_ATTRIBUTE",
    "LOOKAHEAD_MODEL_SCHEMA_VERSION",
    "LookaheadConfig",
    "resolve_lookahead_config",
    "set_lookahead_model_metadata",
    "validate_lookahead_model_metadata",
]
