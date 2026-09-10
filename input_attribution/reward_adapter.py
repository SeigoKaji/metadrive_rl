"""A side-effect-free connection point for returned reward breakdowns."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Real


class RewardTermsError(ValueError):
    """The supplied reward breakdown is malformed."""


@dataclass(frozen=True, slots=True)
class RewardTermsResult:
    """Validation result stored alongside each step record."""

    status: str
    terms: dict[str, float] | None
    residual: float | None
    atol: float
    rtol: float

    @property
    def verified(self) -> bool:
        return self.status == "verified"

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "terms": self.terms,
            "residual": self.residual,
            "atol": self.atol,
            "rtol": self.rtol,
        }


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RewardTermsError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise RewardTermsError(f"{name} must be finite")
    return result


def _coerce_terms(value: object) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise RewardTermsError("reward terms must be a mapping")
    terms: dict[str, float] = {}
    for key, component in value.items():
        if not isinstance(key, str) or not key.strip():
            raise RewardTermsError("reward term names must be non-empty strings")
        terms[key] = _finite_float(component, name=f"reward_terms[{key!r}]")
    if not terms:
        raise RewardTermsError("reward terms cannot be empty")
    return terms


def _error_status(error: RewardTermsError, value: object) -> str:
    """Classify malformed component types separately from NaN/Inf values."""

    if isinstance(value, Mapping):
        for component in value.values():
            if isinstance(component, bool) or not isinstance(component, Real):
                return "malformed"
            try:
                if not math.isfinite(float(component)):
                    return "nonfinite"
            except (TypeError, ValueError, OverflowError):
                return "malformed"
    message = str(error).lower()
    return "nonfinite" if "nan" in message or "inf" in message else "malformed"


def extract_reward_terms(
    info: Mapping[str, object] | None,
    returned_reward: float,
    *,
    terminated: bool,
    truncated: bool,
) -> dict[str, float] | None:
    """Extract provider-supplied final reward components without re-calling reward.

    Only an explicit ``reward_terms`` mapping is accepted.  A key such as
    ``step_reward`` is deliberately not inferred to be the final returned
    reward because wrappers may overwrite or add to it at termination.
    """

    _finite_float(returned_reward, name="returned_reward")
    if info is None:
        return None
    if not isinstance(info, Mapping):
        raise RewardTermsError("info must be a mapping or None")
    if info.get("reward_terms_status") in {"unavailable", "missing"}:
        return None
    if "reward_terms" not in info:
        return None
    return _coerce_terms(info["reward_terms"])


def validate_reward_terms(
    terms: Mapping[str, object] | None,
    returned_reward: float,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-6,
    terminated: bool = False,
    truncated: bool = False,
) -> RewardTermsResult:
    """Check that supplied components equal the reward returned by ``step``."""

    reward = _finite_float(returned_reward, name="returned_reward")
    absolute_tolerance = _finite_float(atol, name="atol")
    relative_tolerance = _finite_float(rtol, name="rtol")
    if absolute_tolerance < 0 or relative_tolerance < 0:
        raise RewardTermsError("atol and rtol must be non-negative")
    if terms is None:
        return RewardTermsResult(
            status="unavailable",
            terms=None,
            residual=None,
            atol=absolute_tolerance,
            rtol=relative_tolerance,
        )
    try:
        normalized = _coerce_terms(terms)
    except RewardTermsError as error:
        status = _error_status(error, terms)
        return RewardTermsResult(
            status=status,
            terms=None,
            residual=None,
            atol=absolute_tolerance,
            rtol=relative_tolerance,
        )
    assert normalized is not None
    try:
        total = float(sum(normalized.values()))
    except (OverflowError, ValueError):
        return RewardTermsResult(
            status="nonfinite",
            terms=normalized,
            residual=None,
            atol=absolute_tolerance,
            rtol=relative_tolerance,
        )
    if not math.isfinite(total):
        return RewardTermsResult(
            status="nonfinite",
            terms=normalized,
            residual=None,
            atol=absolute_tolerance,
            rtol=relative_tolerance,
        )
    residual = float(total - reward)
    if not math.isfinite(residual):
        return RewardTermsResult(
            status="nonfinite",
            terms=normalized,
            residual=None,
            atol=absolute_tolerance,
            rtol=relative_tolerance,
        )
    status = (
        "verified"
        if math.isclose(total, reward, abs_tol=absolute_tolerance, rel_tol=relative_tolerance)
        else "mismatch"
    )
    return RewardTermsResult(
        status=status,
        terms=normalized,
        residual=residual,
        atol=absolute_tolerance,
        rtol=relative_tolerance,
    )


def reward_terms_result(
    info: Mapping[str, object] | None,
    returned_reward: float,
    *,
    terminated: bool,
    truncated: bool,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> RewardTermsResult:
    """Extract and validate one step's explicitly supplied breakdown."""

    try:
        terms = extract_reward_terms(
            info,
            returned_reward,
            terminated=terminated,
            truncated=truncated,
        )
    except RewardTermsError as error:
        return RewardTermsResult(
            status=_error_status(error, info.get("reward_terms") if isinstance(info, Mapping) else None),
            terms=None,
            residual=None,
            atol=float(atol),
            rtol=float(rtol),
        )
    return validate_reward_terms(
        terms,
        returned_reward,
        atol=atol,
        rtol=rtol,
        terminated=terminated,
        truncated=truncated,
    )


verify_reward_terms = validate_reward_terms


__all__ = [
    "RewardTermsError",
    "RewardTermsResult",
    "extract_reward_terms",
    "reward_terms_result",
    "validate_reward_terms",
    "verify_reward_terms",
]
