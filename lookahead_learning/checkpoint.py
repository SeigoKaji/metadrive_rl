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


class CheckpointContractError(ValueError):
    """ZIP-carried lookahead attributes violate the compatibility contract."""


LOOKAHEAD_DEFAULTS: dict[str, float] = {
    "lookahead_m": 6.0,
    "pp_weight": 0.0,
}
"""Resolved defaults used whenever an explicit ``[lookahead]`` table exists."""

LOOKAHEAD_MODEL_SCHEMA_VERSION = 1
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


def resolve_lookahead_config(value: object) -> dict[str, float] | None:
    """Resolve an optional TOML ``[lookahead]`` table.

    ``None`` means that the feature is disabled.  Any mapping, including an
    empty one, means enabled and receives the explicit defaults.  This function
    deliberately accepts only plain finite numeric values so the same contract
    can be reused by a copied ``lookahead_learning`` package.
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
    }


def set_lookahead_model_metadata(
    model: object,
    lookahead_config: Mapping[str, object] | None,
) -> None:
    """Attach resolved settings to attributes serialized inside a PPO ZIP."""

    setattr(model, LOOKAHEAD_MODEL_SCHEMA_ATTRIBUTE, LOOKAHEAD_MODEL_SCHEMA_VERSION)
    setattr(
        model,
        LOOKAHEAD_MODEL_CONFIG_ATTRIBUTE,
        None if lookahead_config is None else dict(lookahead_config),
    )


def validate_lookahead_model_metadata(
    model: object,
    expected: Mapping[str, object] | None,
) -> None:
    """Validate ZIP-carried lookahead settings against the selected TOML."""

    actual = getattr(model, LOOKAHEAD_MODEL_CONFIG_ATTRIBUTE, _MISSING)
    if expected is None:
        # Legacy baseline ZIPs have no custom attributes; newly saved baseline
        # models carry ``None``.  An active checkpoint must not pass solely on
        # an accidentally compatible observation shape.
        if actual is _MISSING or actual is None:
            return
        raise CheckpointContractError(
            "checkpoint contains active lookahead settings but the selected "
            "TOML has no [lookahead] table"
        )
    schema = getattr(model, LOOKAHEAD_MODEL_SCHEMA_ATTRIBUTE, None)
    if schema != LOOKAHEAD_MODEL_SCHEMA_VERSION:
        raise CheckpointContractError(
            "lookahead config is active but the checkpoint has no supported "
            "lookahead schema metadata"
        )
    if not isinstance(actual, Mapping) or dict(actual) != dict(expected):
        raise CheckpointContractError(
            "checkpoint lookahead settings do not match the selected TOML: "
            f"expected={dict(expected)!r}, found={actual!r}"
        )


__all__ = [
    "CheckpointContractError",
    "LOOKAHEAD_DEFAULTS",
    "LOOKAHEAD_MODEL_CONFIG_ATTRIBUTE",
    "LOOKAHEAD_MODEL_SCHEMA_ATTRIBUTE",
    "LOOKAHEAD_MODEL_SCHEMA_VERSION",
    "resolve_lookahead_config",
    "set_lookahead_model_metadata",
    "validate_lookahead_model_metadata",
]
