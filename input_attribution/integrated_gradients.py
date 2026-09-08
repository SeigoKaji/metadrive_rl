"""Optional Integrated Gradients for a fixed categorical action.

Captum is imported lazily and is the default ``backend='captum'`` so the
standard Integrated Gradients implementation is used when this optional
analysis is requested.  ``backend='auto'`` first tries Captum and then uses
the small autograd fallback for a portable installation without Captum;
``backend='manual'`` selects that fallback explicitly.  None of these
optional dependencies are imported by the package at module import time.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .policy import PolicyError, as_policy_adapter
from .schema import InputSchema, SchemaError


class IntegratedGradientsError(ValueError):
    """Raised when IG cannot be computed for the requested observation."""


class OptionalDependencyError(IntegratedGradientsError):
    """Raised when Captum is requested and unavailable.

    Callers that want a dependency-free fallback can select
    ``backend='auto'`` (or ``backend='manual'`` for an explicit choice).
    """


def score_log_odds(logits: Any, action: int) -> Any:
    """Compute ``z[action] - logsumexp(z[other actions])`` stably."""

    torch = _torch()
    if not isinstance(action, int) or isinstance(action, bool):
        raise IntegratedGradientsError("target action must be an integer")
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise IntegratedGradientsError("logits must have shape (batch, actions>=2)")
    if action < 0 or action >= int(logits.shape[1]):
        raise IntegratedGradientsError("target action is outside the policy action space")
    other = torch.cat((logits[:, :action], logits[:, action + 1 :]), dim=1)
    return logits[:, action] - torch.logsumexp(other, dim=1)


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise IntegratedGradientsError("PyTorch is required for Integrated Gradients") from exc
    return torch


def _as_float_vectors(
    observation: Any,
    baseline: Any,
    schema: InputSchema | None,
) -> tuple[np.ndarray, np.ndarray]:
    if schema is None:
        observation_array = np.asarray(observation)
        baseline_array = np.asarray(baseline)
        if observation_array.ndim != 1 or baseline_array.ndim != 1:
            raise IntegratedGradientsError("observation and baseline must be one-dimensional")
        if observation_array.shape != baseline_array.shape:
            raise IntegratedGradientsError("observation and baseline dimensions differ")
        if not np.all(np.isfinite(observation_array.astype(np.float64))) or not np.all(np.isfinite(baseline_array.astype(np.float64))):
            raise IntegratedGradientsError("observation and baseline must be finite")
        return observation_array.astype(np.float32, copy=True), baseline_array.astype(np.float32, copy=True)
    try:
        observation_array = schema.validate_observation(observation, copy=True)
        baseline_array = schema.validate_observation(baseline, copy=True)
    except SchemaError as exc:
        raise IntegratedGradientsError(str(exc)) from exc
    for spec in schema.inputs:
        if not spec.interpolation_allowed and observation_array[spec.index] != baseline_array[spec.index]:
            raise IntegratedGradientsError(
                f"input {spec.index} ({spec.id}) is categorical/flagged and cannot be interpolated between different values"
            )
    return observation_array.astype(np.float32, copy=True), baseline_array.astype(np.float32, copy=True)


def _context_mismatch(
    compatibility_keys: Any,
    observation_context: Mapping[str, Any] | None,
    baseline_context: Mapping[str, Any] | None,
) -> str | None:
    if compatibility_keys is None:
        return None
    if not isinstance(compatibility_keys, (list, tuple)):
        return "compatibility_keys must be a list or tuple"
    if not compatibility_keys:
        return None
    if observation_context is None or baseline_context is None:
        return "observation_context and baseline_context are required for compatibility_keys"
    for key in compatibility_keys:
        if key not in observation_context or key not in baseline_context:
            return f"compatibility context is missing {key!r}"
        observed = observation_context[key]
        baseline = baseline_context[key]
        if observed is None or baseline is None:
            return f"compatibility context {key!r} is null"
        if observed != baseline:
            return f"compatibility context {key!r} differs: observation={observed!r}, baseline={baseline!r}"
    return None


def _path_condition_error(
    path_condition: Mapping[str, Any] | None,
    observation: np.ndarray,
    baseline: np.ndarray,
    schema: InputSchema | None,
) -> str | None:
    if path_condition is None:
        return None
    if not isinstance(path_condition, Mapping):
        return "path_condition must be a mapping"
    if path_condition.get("allow_linear", True) is False and not np.array_equal(observation, baseline):
        return "linear interpolation is disabled by path_condition"
    required_equal = path_condition.get("required_equal_indices", ())
    if required_equal:
        if schema is None:
            return "path_condition indices require a schema"
        try:
            indices = schema.resolve_indices(required_equal)
        except SchemaError as exc:
            return f"path_condition has unknown indices: {exc}"
        unequal = [index for index in indices if observation[index] != baseline[index]]
        if unequal:
            return f"path_condition requires equal endpoint values at indices {unequal}"
    path_type = path_condition.get("type", "linear")
    if path_type not in {"linear", "straight"}:
        return (
            f"unsupported interpolation path type {path_type!r}; configure a verified linear path "
            "before enabling IG"
        )
    return None


def _score_numpy(adapter: Any, observation: np.ndarray, action: int) -> float:
    torch = _torch()
    tensor = torch.as_tensor(observation[None, :], dtype=torch.float32)
    logits = adapter.logits_tensor(tensor)
    score = score_log_odds(logits, action)
    return float(score.detach().cpu().reshape(-1)[0])


def _manual_attribution(
    adapter: Any,
    observation: np.ndarray,
    baseline: np.ndarray,
    action: int,
    n_steps: int,
    *,
    method: str,
) -> tuple[np.ndarray, float, float, int]:
    torch = _torch()
    if n_steps <= 0:
        raise IntegratedGradientsError("n_steps must be positive")
    if method not in {"gausslegendre", "riemann_trapezoid", "riemann_right", "riemann_left"}:
        raise IntegratedGradientsError(
            "method must be gausslegendre, riemann_trapezoid, riemann_right, or riemann_left"
        )
    device = None
    policy = getattr(adapter, "policy", None)
    if policy is not None:
        device = getattr(policy, "device", None)
    x = torch.as_tensor(observation, dtype=torch.float32, device=device)
    b = torch.as_tensor(baseline, dtype=torch.float32, device=device)
    difference = x - b
    if method == "gausslegendre":
        nodes, gauss_weights = np.polynomial.legendre.leggauss(n_steps)
        alphas = torch.as_tensor((nodes + 1.0) / 2.0, device=x.device, dtype=x.dtype)
        weights = torch.as_tensor(gauss_weights / 2.0, device=x.device, dtype=x.dtype)
        denominator = 1.0
    elif method == "riemann_trapezoid":
        alphas = torch.linspace(0.0, 1.0, n_steps + 1, device=x.device, dtype=x.dtype)
        weights = torch.ones(n_steps + 1, device=x.device, dtype=x.dtype)
        weights[[0, -1]] = 0.5
        denominator = float(n_steps)
    elif method == "riemann_right":
        alphas = torch.arange(1, n_steps + 1, device=x.device, dtype=x.dtype) / float(n_steps)
        weights = torch.ones(n_steps, device=x.device, dtype=x.dtype)
        denominator = float(n_steps)
    else:
        alphas = torch.arange(0, n_steps, device=x.device, dtype=x.dtype) / float(n_steps)
        weights = torch.ones(n_steps, device=x.device, dtype=x.dtype)
        denominator = float(n_steps)
    gradients: list[Any] = []
    for alpha in alphas:
        point = (b + alpha * difference).unsqueeze(0).clone().detach().requires_grad_(True)
        logits = adapter.logits_tensor(point)
        score = score_log_odds(logits, action)
        if score.numel() != 1:
            raise IntegratedGradientsError("logits_tensor must return one row for one IG input")
        gradient = torch.autograd.grad(score.reshape(()), point, retain_graph=False, create_graph=False)[0]
        gradients.append(gradient.squeeze(0))
    stacked = torch.stack(gradients, dim=0)
    weighted_gradient = (stacked * weights.reshape((-1, 1))).sum(dim=0) / denominator
    attribution = difference * weighted_gradient
    baseline_score = _score_numpy(adapter, baseline, action)
    observation_score = _score_numpy(adapter, observation, action)
    return (
        attribution.detach().cpu().numpy().astype(np.float64),
        observation_score,
        baseline_score,
        int(alphas.shape[0]),
    )


def _captum_attribution(
    adapter: Any,
    observation: np.ndarray,
    baseline: np.ndarray,
    action: int,
    n_steps: int,
    method: str,
) -> tuple[np.ndarray, float, float, int, float]:
    try:
        from captum.attr import IntegratedGradients
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise OptionalDependencyError(
            "Captum is not installed; the optional IG dependency is required for "
            "backend='captum' or 'auto'. Primary offline/closed-loop results are unaffected."
        ) from exc
    torch = _torch()
    device = getattr(getattr(adapter, "policy", None), "device", None)
    x = torch.as_tensor(observation[None, :], dtype=torch.float32, device=device)
    b = torch.as_tensor(baseline[None, :], dtype=torch.float32, device=device)

    def forward(points: Any) -> Any:
        return score_log_odds(adapter.logits_tensor(points), action)

    # Captum uses method names ``riemann_trapezoid`` etc. and does not enter an
    # inference/no_grad context.  Keep the action fixed throughout the path.
    ig = IntegratedGradients(forward)
    try:
        attribution, delta = ig.attribute(
            x,
            baselines=b,
            n_steps=n_steps,
            method=method,
            return_convergence_delta=True,
        )
    except Exception as exc:
        raise IntegratedGradientsError(f"Captum IntegratedGradients failed: {exc}") from exc
    observation_score = float(forward(x).detach().cpu().reshape(-1)[0])
    baseline_score = float(forward(b).detach().cpu().reshape(-1)[0])
    return (
        attribution.detach().cpu().numpy().reshape(-1).astype(np.float64),
        observation_score,
        baseline_score,
        int(n_steps),
        float(delta.detach().cpu().reshape(-1)[0]),
    )


@dataclass(frozen=True, slots=True)
class IntegratedGradientsResult:
    observation: np.ndarray
    baseline: np.ndarray
    target_action: int
    baseline_id: str | None
    observation_score: float
    baseline_score: float
    attributions: np.ndarray
    completeness_delta: float
    absolute_completeness_error: float
    relative_completeness_error: float | None
    n_steps: int
    integration_points: int
    method: str
    backend: str
    groups: Mapping[str, Mapping[str, Any]]
    policy_fingerprint_before: str | None = None
    policy_fingerprint_after: str | None = None
    policy_unchanged: bool | None = None
    observation_context: Mapping[str, Any] | None = None
    baseline_context: Mapping[str, Any] | None = None
    compatibility_keys: tuple[str, ...] = ()
    path_condition: Mapping[str, Any] | None = None
    tolerance: float = 1e-4
    retry_count: int = 0
    status: str = "ok"
    warnings: tuple[str, ...] = ()

    @property
    def completeness_status(self) -> str:
        """Stable machine-readable convergence status for report consumers."""

        return "converged" if self.status == "ok" else "tolerance_exceeded"

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_action": self.target_action,
            "baseline_id": self.baseline_id,
            "observation_score": self.observation_score,
            "baseline_score": self.baseline_score,
            "attribution_sum": float(np.sum(self.attributions)),
            "output_difference": self.observation_score - self.baseline_score,
            "attributions": self.attributions.tolist(),
            "absolute_attributions": np.abs(self.attributions).tolist(),
            "completeness_delta": self.completeness_delta,
            "absolute_completeness_error": self.absolute_completeness_error,
            "relative_completeness_error": self.relative_completeness_error,
            "n_steps": self.n_steps,
            "integration_points": self.integration_points,
            "method": self.method,
            "backend": self.backend,
            "groups": dict(self.groups),
            "policy_fingerprint_before": self.policy_fingerprint_before,
            "policy_fingerprint_after": self.policy_fingerprint_after,
            "policy_unchanged": self.policy_unchanged,
            "observation_context": dict(self.observation_context or {}),
            "baseline_context": dict(self.baseline_context or {}),
            "compatibility_keys": list(self.compatibility_keys),
            "path_condition": dict(self.path_condition or {}),
            "tolerance": self.tolerance,
            "retry_count": self.retry_count,
            "status": self.status,
            "completeness_status": self.completeness_status,
            "warnings": list(self.warnings),
            "observation": self.observation.tolist(),
            "baseline": self.baseline.tolist(),
        }

    def save_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, ensure_ascii=False, indent=2)
            stream.write("\n")


def _group_attributions(attributions: np.ndarray, schema: InputSchema | None) -> dict[str, Mapping[str, Any]]:
    if schema is None:
        return {}
    groups: dict[str, list[int]] = {}
    for spec in schema.inputs:
        groups.setdefault(spec.group, []).append(spec.index)
    result: dict[str, Mapping[str, Any]] = {}
    for group, indices in groups.items():
        values = attributions[indices]
        result[group] = {
            "indices": indices,
            "signed_sum": float(np.sum(values)),
            "absolute_sum": float(np.sum(np.abs(values))),
            "absolute_mean": float(np.mean(np.abs(values))),
        }
    return result


def compute_integrated_gradients(
    observation: Any,
    baseline: Any,
    policy: Any,
    *,
    schema: InputSchema | None = None,
    target_action: int | None = None,
    baseline_id: str | None = None,
    n_steps: int = 50,
    method: str = "gausslegendre",
    backend: str = "captum",
    completeness_tolerance: float = 1e-4,
    max_retries: int = 1,
    observation_context: Mapping[str, Any] | None = None,
    baseline_context: Mapping[str, Any] | None = None,
    compatibility_keys: list[str] | tuple[str, ...] | None = None,
    path_condition: Mapping[str, Any] | None = None,
) -> IntegratedGradientsResult:
    """Compute IG for one saved observation and an explicit baseline.

    The target action is selected once from the original observation and stays
    fixed on the entire interpolation path.  No zero baseline is inferred.
    """

    if n_steps <= 0:
        raise IntegratedGradientsError("n_steps must be positive")
    if completeness_tolerance < 0 or not np.isfinite(completeness_tolerance):
        raise IntegratedGradientsError("completeness_tolerance must be finite and non-negative")
    if max_retries < 0:
        raise IntegratedGradientsError("max_retries must be non-negative")
    if schema is not None:
        try:
            schema.validate_for_execution()
        except SchemaError as exc:
            raise IntegratedGradientsError(
                f"schema is not ready for Integrated Gradients: {exc}"
            ) from exc
    observation_array, baseline_array = _as_float_vectors(observation, baseline, schema)
    context_reason = _context_mismatch(
        compatibility_keys,
        observation_context,
        baseline_context,
    )
    if context_reason is not None:
        raise IntegratedGradientsError(context_reason)
    path_reason = _path_condition_error(
        path_condition,
        observation_array,
        baseline_array,
        schema,
    )
    if path_reason is not None:
        raise IntegratedGradientsError(path_reason)
    adapter = as_policy_adapter(policy)
    logits_method = getattr(adapter, "logits_tensor", None)
    if not callable(logits_method):
        raise IntegratedGradientsError("policy does not expose differentiable logits_tensor")
    fingerprint_before = None
    fingerprint_fn = getattr(adapter, "fingerprint", None)
    if callable(fingerprint_fn):
        fingerprint_before = str(fingerprint_fn())
    if hasattr(adapter, "set_eval"):
        adapter.set_eval()
    if target_action is None:
        try:
            probabilities = adapter.probabilities(observation_array[None, :])
            target_action = int(np.argmax(probabilities[0]))
        except Exception as exc:
            raise IntegratedGradientsError(f"cannot determine original argmax action: {exc}") from exc
    if isinstance(target_action, bool) or not isinstance(target_action, int):
        raise IntegratedGradientsError("target_action must be an integer")

    selected_backend = backend.lower()
    if selected_backend not in {"auto", "manual", "captum"}:
        raise IntegratedGradientsError("backend must be auto, manual, or captum")
    if selected_backend == "auto":
        # Prefer the standard Captum implementation when the optional package
        # is present.  A portable install without Captum still has a faithful
        # autograd fallback; the main ①-A/①-B workflow never depends on either.
        try:
            import captum  # type: ignore  # noqa: F401
        except ImportError:
            selected_backend = "manual"
        else:
            selected_backend = "captum"
    current_steps = int(n_steps)
    final: tuple[np.ndarray, float, float, int, float | None] | None = None
    used_backend = "manual"
    attempts = 0
    while True:
        if selected_backend == "captum":
            final = (*_captum_attribution(adapter, observation_array, baseline_array, target_action, current_steps, method),)
            used_backend = "captum"
        else:
            attribution, obs_score, base_score, points = _manual_attribution(
                adapter,
                observation_array,
                baseline_array,
                target_action,
                current_steps,
                method=method,
            )
            final = (attribution, obs_score, base_score, points, None)
            used_backend = "manual"
        assert final is not None
        attribution, observation_score, baseline_score, integration_points, captum_delta = final
        completeness_delta = float(np.sum(attribution) - (observation_score - baseline_score))
        if captum_delta is not None:
            # Captum's convergence delta has the same signed convention; keep
            # our explicit recomputation as the canonical stored quantity.
            del captum_delta
        if abs(completeness_delta) <= completeness_tolerance or attempts >= max_retries:
            break
        attempts += 1
        current_steps *= 2
    output_difference = observation_score - baseline_score
    relative_error = None
    if abs(output_difference) > 1e-12:
        relative_error = abs(completeness_delta) / abs(output_difference)
    warnings: list[str] = []
    status = "ok"
    if abs(completeness_delta) > completeness_tolerance:
        status = "warning"
        warnings.append(
            "completeness residual "
            f"{abs(completeness_delta):.6g} exceeds tolerance "
            f"{completeness_tolerance:.6g} after {attempts} retries"
        )
    fingerprint_after = None
    if callable(fingerprint_fn):
        fingerprint_after = str(fingerprint_fn())
    policy_unchanged = None
    if fingerprint_before is not None and fingerprint_after is not None:
        policy_unchanged = fingerprint_before == fingerprint_after
        if not policy_unchanged:
            raise IntegratedGradientsError(
                "policy parameters/buffers changed during Integrated Gradients"
            )
    return IntegratedGradientsResult(
        observation=observation_array,
        baseline=baseline_array,
        target_action=target_action,
        baseline_id=baseline_id,
        observation_score=observation_score,
        baseline_score=baseline_score,
        attributions=np.asarray(attribution, dtype=np.float64),
        completeness_delta=completeness_delta,
        absolute_completeness_error=abs(completeness_delta),
        relative_completeness_error=relative_error,
        n_steps=current_steps,
        integration_points=integration_points,
        method=method,
        backend=used_backend,
        groups=_group_attributions(np.asarray(attribution, dtype=np.float64), schema),
        policy_fingerprint_before=fingerprint_before,
        policy_fingerprint_after=fingerprint_after,
        policy_unchanged=policy_unchanged,
        observation_context=dict(observation_context) if observation_context is not None else None,
        baseline_context=dict(baseline_context) if baseline_context is not None else None,
        compatibility_keys=tuple(compatibility_keys or ()),
        path_condition=dict(path_condition) if path_condition is not None else None,
        tolerance=float(completeness_tolerance),
        retry_count=attempts,
        status=status,
        warnings=tuple(warnings),
    )


def integrated_gradients(*args: Any, **kwargs: Any) -> IntegratedGradientsResult:
    """Short alias for :func:`compute_integrated_gradients`."""

    return compute_integrated_gradients(*args, **kwargs)
