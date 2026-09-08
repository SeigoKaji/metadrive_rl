"""Strict, sidecar-only checkpoint compatibility validation.

This module intentionally uses only the Python standard library.  It does not
load PPO, inspect tensors, import the host project, or write metadata.  A
runner computes a semantic ``expected`` mapping from the current host and
passes the sidecar mapping it read as ``saved``.  Only keys present in
``expected`` are compared; extra saved fields (including machine-specific
paths) are left untouched unless the runner explicitly includes them.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
import hashlib
import math
from numbers import Real
from pathlib import Path
from typing import Any


class CheckpointContractError(ValueError):
    """The checkpoint bytes or saved semantic metadata violate the contract."""


def _field_path(path: str, key: Any) -> str:
    if isinstance(key, int) and not isinstance(key, bool):
        return f"{path}[{key}]"
    key_text = str(key)
    if path:
        return f"{path}.{key_text}"
    return key_text


def _error(message: str) -> CheckpointContractError:
    return CheckpointContractError(message)


def _validate_finite(value: Any, path: str, active: set[int]) -> None:
    """Reject non-finite scalar values anywhere in metadata."""

    if isinstance(value, float):
        if not math.isfinite(value):
            raise _error(f"metadata field {path or '<root>'} must be finite")
        return
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise _error(f"metadata field {path or '<root>'} must be finite")
        return
    if isinstance(value, Real) and not isinstance(value, bool):
        # Real scalar implementations such as Fraction are finite by contract;
        # checking float also catches custom Real values that expose infinity.
        try:
            if not math.isfinite(float(value)):
                raise _error(f"metadata field {path or '<root>'} must be finite")
        except (TypeError, ValueError, OverflowError) as exc:
            raise _error(f"metadata field {path or '<root>'} is not a finite scalar") from exc
        return
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise _error(f"metadata field {path or '<root>'} contains a cyclic mapping")
        active.add(identity)
        try:
            for key, item in value.items():
                _validate_finite(key, _field_path(path, key) + "<key>", active)
                _validate_finite(item, _field_path(path, key), active)
        finally:
            active.remove(identity)
        return
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active:
            raise _error(f"metadata field {path or '<root>'} contains a cyclic sequence")
        active.add(identity)
        try:
            for index, item in enumerate(value):
                _validate_finite(item, _field_path(path, index), active)
        finally:
            active.remove(identity)
        return
    if isinstance(value, (set, frozenset)):
        identity = id(value)
        if identity in active:
            raise _error(f"metadata field {path or '<root>'} contains a cyclic set")
        active.add(identity)
        try:
            for index, item in enumerate(sorted(value, key=repr)):
                _validate_finite(item, _field_path(path, index), active)
        finally:
            active.remove(identity)


def _sha256_file(checkpoint: Path) -> str:
    try:
        with checkpoint.open("rb") as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except (OSError, TypeError) as exc:
        raise _error(f"checkpoint cannot be read for model_sha256: {checkpoint}") from exc
    return digest.hexdigest()


def _compare_expected(saved: Any, expected: Any, path: str) -> None:
    """Compare expected fields recursively, retaining a precise field path."""

    if isinstance(expected, Mapping):
        if not isinstance(saved, Mapping):
            raise _error(f"metadata field {path or '<root>'} must be a mapping")
        for key, expected_value in expected.items():
            child_path = _field_path(path, key)
            if key not in saved:
                raise _error(f"missing metadata field {child_path}")
            _compare_expected(saved[key], expected_value, child_path)
        return
    if isinstance(expected, list):
        if not isinstance(saved, list):
            raise _error(f"metadata field {path or '<root>'} must be a list")
        if len(saved) != len(expected):
            raise _error(
                f"metadata field {path or '<root>'} list length mismatch: "
                f"saved={len(saved)} expected={len(expected)}"
            )
        for index, (saved_item, expected_item) in enumerate(zip(saved, expected)):
            _compare_expected(saved_item, expected_item, _field_path(path, index))
        return
    if isinstance(expected, tuple):
        if not isinstance(saved, tuple):
            raise _error(f"metadata field {path or '<root>'} must be a tuple")
        if len(saved) != len(expected):
            raise _error(
                f"metadata field {path or '<root>'} tuple length mismatch: "
                f"saved={len(saved)} expected={len(expected)}"
            )
        for index, (saved_item, expected_item) in enumerate(zip(saved, expected)):
            _compare_expected(saved_item, expected_item, _field_path(path, index))
        return
    if type(saved) is not type(expected) or saved != expected:
        raise _error(
            f"metadata field {path or '<root>'} mismatch: "
            f"saved={saved!r} expected={expected!r}"
        )


def validate_checkpoint_metadata(
    checkpoint: Path,
    saved: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    """Validate checkpoint bytes and the runner-selected semantic metadata.

    ``saved`` must be a concrete ``dict`` because it is the decoded sidecar
    object whose ownership and mutability contract is known to the runner.
    ``expected`` may be any mapping.  The function returns ``None`` on success
    and raises :class:`CheckpointContractError` on every contract mismatch.
    """

    if not isinstance(saved, dict):
        raise _error("saved checkpoint metadata must be a dict")
    if not isinstance(expected, Mapping):
        raise _error("expected checkpoint metadata must be a mapping")
    _validate_finite(saved, "", set())
    _validate_finite(expected, "", set())

    try:
        checkpoint_path = checkpoint if isinstance(checkpoint, Path) else Path(checkpoint)
    except (TypeError, ValueError) as exc:
        raise _error("checkpoint must be a filesystem path") from exc
    if not checkpoint_path.is_file():
        raise _error(f"checkpoint does not exist: {checkpoint_path}")

    saved_hash = saved.get("model_sha256")
    if not isinstance(saved_hash, str) or not saved_hash:
        raise _error("missing metadata field model_sha256")
    actual_hash = _sha256_file(checkpoint_path)
    if saved_hash != actual_hash:
        raise _error(
            "metadata field model_sha256 mismatch: "
            f"saved={saved_hash!r} actual={actual_hash!r}"
        )
    _compare_expected(saved, expected, "")


__all__ = ["CheckpointContractError", "validate_checkpoint_metadata"]
