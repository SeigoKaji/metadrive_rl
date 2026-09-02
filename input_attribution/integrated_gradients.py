"""PyTorch-autograd Integrated Gradients for PPO actor and critic outputs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import torch

from .baselines import ResolvedBaselines
from .perturbation import _normalise_baselines


class IntegratedGradientsError(ValueError):
    """Raised when an IG target, path, or policy adapter is invalid."""


IGTarget = Literal[
    "selected_log_probability",
    "selected_vs_runner_up_margin",
    "critic_value",
]
_SUPPORTED_TARGETS = frozenset(
    {"selected_log_probability", "selected_vs_runner_up_margin", "critic_value"}
)
# A finite denominator keeps NPZ/CSV/JSON artifacts portable while ensuring
# that a non-zero trapezoidal completeness residual is never reported as a
# misleadingly perfect relative error merely because ΔF happens to be zero.
_RELATIVE_COMPLETENESS_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class TrapezoidalIGComputation:
    """Differentiable-function IG output for one baseline per input sample."""

    attributions: torch.Tensor
    input_output: torch.Tensor
    baseline_output: torch.Tensor
    attribution_sum: torch.Tensor
    completeness_residual: torch.Tensor

    @property
    def absolute_completeness_residual(self) -> torch.Tensor:
        """Absolute ``sum(IG) - (F(x) - F(x0))`` for each sample."""

        return self.completeness_residual.abs()

    @property
    def relative_completeness_error(self) -> torch.Tensor:
        """``abs(residual) / max(abs(ΔF), epsilon)`` for each sample.

        When both ΔF and the residual are zero this is zero.  If ΔF is zero
        but numerical integration has a non-zero residual, it instead reports
        a large finite value rather than incorrectly labelling the result 0.
        """

        # This is a diagnostic ratio rather than an attribution tensor.  Do
        # the division in float64 so ``1e-12`` neither underflows nor makes an
        # otherwise finite float16 residual overflow during division.  Clamp
        # the pathological upper boundary to keep CSV/NPZ/JSON consumers
        # finite as documented.
        difference = (self.input_output - self.baseline_output).to(torch.float64)
        numerator = self.absolute_completeness_residual.to(torch.float64)
        denominator = torch.clamp(
            difference.abs(), min=_RELATIVE_COMPLETENESS_EPSILON
        )
        ratio = numerator / denominator
        return torch.nan_to_num(
            ratio,
            nan=0.0,
            posinf=torch.finfo(torch.float64).max,
            neginf=0.0,
        )


@runtime_checkable
class _TensorPolicyAdapter(Protocol):
    def evaluate_tensors(self, observations: torch.Tensor) -> object:
        """Return differentiable logits/log probabilities/values for a batch."""


def _validate_steps(steps: int) -> int:
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 2:
        raise IntegratedGradientsError("IG stepsは両端を含む2以上の整数で指定してください")
    return steps


def _as_batch_tensor(value: torch.Tensor, *, name: str) -> tuple[torch.Tensor, bool]:
    if not isinstance(value, torch.Tensor):
        raise IntegratedGradientsError(f"{name}はtorch.Tensorで指定してください")
    was_vector = value.ndim == 1
    tensor = value.unsqueeze(0) if was_vector else value
    if tensor.ndim != 2 or tensor.shape[0] == 0 or tensor.shape[1] == 0:
        raise IntegratedGradientsError(f"{name}は(D,)または非空の(B, D) tensorで指定してください")
    if not tensor.is_floating_point():
        raise IntegratedGradientsError(f"{name}は浮動小数tensorで指定してください")
    if not torch.isfinite(tensor).all():
        raise IntegratedGradientsError(f"{name}は有限値で指定してください")
    return tensor, was_vector


def _target_values(value: object, *, batch_size: int, name: str = "target_fn") -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise IntegratedGradientsError(f"{name}はtorch.Tensorを返す必要があります")
    if value.ndim == 2 and value.shape == (batch_size, 1):
        value = value[:, 0]
    if value.ndim == 0:
        # A scalar is valid for one input, but using it with a larger batch
        # would mix unrelated samples in autograd.grad(sum(...), inputs).
        if batch_size != 1:
            raise IntegratedGradientsError("batch size > 1ではtarget_fnは(B,)を返す必要があります")
        value = value.reshape(1)
    if value.ndim != 1 or value.shape[0] != batch_size:
        raise IntegratedGradientsError(
            f"{name}は(B,)または(B, 1)を返す必要があります: got {tuple(value.shape)}"
        )
    if not torch.isfinite(value).all():
        raise IntegratedGradientsError(f"{name}の出力に非有限値があります")
    return value


def compute_integrated_gradients(
    inputs: torch.Tensor,
    baselines: torch.Tensor,
    target_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    steps: int = 64,
) -> TrapezoidalIGComputation:
    """Compute IG using the trapezoidal rule, including both path endpoints.

    ``target_fn`` must return one scalar target per input row.  The function
    uses :func:`torch.autograd.grad` only with the interpolation tensor as an
    input, so no model parameter ``.grad`` values are accumulated.
    """

    number_of_steps = _validate_steps(steps)
    x, x_was_vector = _as_batch_tensor(inputs, name="inputs")
    baseline, baseline_was_vector = _as_batch_tensor(baselines, name="baselines")
    if baseline.shape[1] != x.shape[1] or baseline.shape[0] not in {1, x.shape[0]}:
        raise IntegratedGradientsError(
            "baselinesは(D,), (1, D), またはinputsと同じ(B, D)形状で指定してください"
        )
    if baseline.shape[0] == 1 and x.shape[0] != 1:
        baseline = baseline.expand(x.shape[0], -1)
    if baseline.device != x.device or baseline.dtype != x.dtype:
        baseline = baseline.to(device=x.device, dtype=x.dtype)
    # The calculation must be an input attribution regardless of whether the
    # caller's x/baseline happened to carry an unrelated graph.
    x_detached = x.detach()
    baseline_detached = baseline.detach()
    delta = x_detached - baseline_detached
    gradient_integral = torch.zeros_like(x_detached)
    for point_index in range(number_of_steps):
        alpha = point_index / (number_of_steps - 1)
        path_point = (baseline_detached + alpha * delta).detach().requires_grad_(True)
        outputs = _target_values(target_fn(path_point), batch_size=x.shape[0])
        try:
            gradients = torch.autograd.grad(
                outputs=outputs.sum(),
                inputs=path_point,
                create_graph=False,
                retain_graph=False,
                only_inputs=True,
            )[0]
        except RuntimeError as error:
            raise IntegratedGradientsError(
                "target_fnからinputへのgradientを計算できません"
            ) from error
        if gradients is None or not torch.isfinite(gradients).all():
            raise IntegratedGradientsError("target_fnのinput gradientに非有限値があります")
        weight = 0.5 if point_index in {0, number_of_steps - 1} else 1.0
        gradient_integral = gradient_integral + weight * gradients
    average_gradient = gradient_integral / float(number_of_steps - 1)
    attributions = delta * average_gradient
    # Keep the complete IG path outside ``torch.no_grad()``.  Endpoint values
    # are detached immediately because they are diagnostics, not backward
    # targets; model parameter ``.grad`` fields are still never populated.
    input_output = _target_values(target_fn(x_detached), batch_size=x.shape[0]).detach()
    baseline_output = _target_values(
        target_fn(baseline_detached), batch_size=x.shape[0]
    ).detach()
    attribution_sum = attributions.sum(dim=1)
    residual = attribution_sum - (input_output - baseline_output)
    # Keep the original input dtype/device and squeeze only when the generic
    # caller supplied a single vector.  The high-level API always passes B,D.
    if x_was_vector:
        return TrapezoidalIGComputation(
            attributions=attributions[0],
            input_output=input_output[0],
            baseline_output=baseline_output[0],
            attribution_sum=attribution_sum[0],
            completeness_residual=residual[0],
        )
    return TrapezoidalIGComputation(
        attributions=attributions,
        input_output=input_output,
        baseline_output=baseline_output,
        attribution_sum=attribution_sum,
        completeness_residual=residual,
    )


def integrated_gradients(
    inputs: torch.Tensor,
    baselines: torch.Tensor,
    target_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    steps: int = 64,
) -> torch.Tensor:
    """Return only trapezoidal IG attributions for a generic target function."""

    return compute_integrated_gradients(inputs, baselines, target_fn, steps=steps).attributions


# This name makes the numerical rule explicit in tests and documentation.
trapezoidal_integrated_gradients = integrated_gradients


@dataclass(frozen=True, slots=True)
class IntegratedGradientsResult:
    """Per-target, per-baseline IG results for a rollout observation batch.

    ``attributions`` has shape ``(N, T, B, D)``.  Scalar completeness fields
    have shape ``(N, T, B)``.  ``target_actions`` and ``runner_up_actions``
    are chosen once from the original observation and remain fixed along every
    interpolation path.
    """

    targets: tuple[IGTarget | str, ...]
    attributions: np.ndarray
    baseline_ids: np.ndarray
    target_actions: np.ndarray
    runner_up_actions: np.ndarray
    input_outputs: np.ndarray
    baseline_outputs: np.ndarray
    attribution_sums: np.ndarray
    completeness_residuals: np.ndarray
    absolute_completeness_residuals: np.ndarray
    relative_completeness_errors: np.ndarray
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        targets = tuple(self.targets)
        if not targets:
            raise IntegratedGradientsError("少なくとも1つのIG targetを指定してください")
        if any(target not in _SUPPORTED_TARGETS for target in targets):
            raise IntegratedGradientsError("未対応のIG targetがあります")
        if len(set(targets)) != len(targets):
            raise IntegratedGradientsError("同じIG targetを重複指定できません")
        attributions = np.asarray(self.attributions)
        if attributions.ndim != 4 or 0 in attributions.shape:
            raise IntegratedGradientsError("attributionsは非空の(N, T, B, D)配列で指定してください")
        sample_count, target_count, baseline_count, _dimension = attributions.shape
        if target_count != len(targets):
            raise IntegratedGradientsError("attributionsのT軸とtargets数が一致しません")
        ids = np.asarray(self.baseline_ids)
        if ids.shape != (sample_count, baseline_count):
            raise IntegratedGradientsError("baseline_idsは(N, B)形状で指定してください")
        target_actions = np.asarray(self.target_actions)
        runner_up_actions = np.asarray(self.runner_up_actions)
        if target_actions.shape not in {(sample_count,), (sample_count, 1)}:
            raise IntegratedGradientsError("target_actionsは(N,)形状で指定してください")
        if runner_up_actions.shape not in {(sample_count,), (sample_count, 1)}:
            raise IntegratedGradientsError("runner_up_actionsは(N,)形状で指定してください")
        scalar_shape = (sample_count, target_count, baseline_count)
        scalar_names = (
            "input_outputs",
            "baseline_outputs",
            "attribution_sums",
            "completeness_residuals",
            "absolute_completeness_residuals",
            "relative_completeness_errors",
        )
        for name in scalar_names:
            if np.asarray(getattr(self, name)).shape != scalar_shape:
                raise IntegratedGradientsError(
                    f"{name}は(N, T, B)形状で指定してください: expected {scalar_shape}"
                )
        normalized: dict[str, np.ndarray] = {
            "attributions": np.array(attributions, dtype=np.float64, copy=True),
            "baseline_ids": np.array(ids, dtype=str, copy=True),
            "target_actions": np.array(target_actions, dtype=np.int64, copy=True).reshape(sample_count),
            "runner_up_actions": np.array(runner_up_actions, dtype=np.int64, copy=True).reshape(sample_count),
        }
        for name in scalar_names:
            normalized[name] = np.array(getattr(self, name), dtype=np.float64, copy=True)
        for name, values in normalized.items():
            if name != "baseline_ids" and not np.all(np.isfinite(values)):
                raise IntegratedGradientsError(f"{name}は有限値で指定してください")
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def sample_count(self) -> int:
        """Number of source observations."""

        return int(self.attributions.shape[0])

    @property
    def target_count(self) -> int:
        """Number of target functions evaluated."""

        return int(self.attributions.shape[1])

    @property
    def baseline_count(self) -> int:
        """Number of IG baselines retained per observation."""

        return int(self.attributions.shape[2])

    @property
    def observation_dim(self) -> int:
        """Feature dimension of every attribution row."""

        return int(self.attributions.shape[3])

    @property
    def absolute_attributions(self) -> np.ndarray:
        """Absolute IG mass without replacing the signed attribution data."""

        return np.abs(self.attributions)

    def attributions_mean_over_baselines(self) -> np.ndarray:
        """Average signed IG over baselines, retaining ``(N, T, D)``."""

        return self.attributions.mean(axis=2)

    def absolute_attributions_mean_over_baselines(self) -> np.ndarray:
        """Average absolute IG over baselines, retaining ``(N, T, D)``."""

        return np.abs(self.attributions).mean(axis=2)

    @property
    def completeness_residual(self) -> np.ndarray:
        """Compatibility alias for plural per-baseline residuals."""

        return self.completeness_residuals

    @property
    def absolute_completeness_residual(self) -> np.ndarray:
        """Compatibility alias for plural per-baseline absolute residuals."""

        return self.absolute_completeness_residuals

    @property
    def relative_completeness_error(self) -> np.ndarray:
        """Compatibility alias for plural per-baseline relative residuals."""

        return self.relative_completeness_errors


def _normalise_targets(targets: Sequence[IGTarget | str] | IGTarget | str) -> tuple[IGTarget, ...]:
    values: Sequence[IGTarget | str]
    if isinstance(targets, str):
        values = (targets,)
    else:
        values = tuple(targets)
    if not values:
        raise IntegratedGradientsError("少なくとも1つのIG targetを指定してください")
    normalized: list[IGTarget] = []
    for target in values:
        if target not in _SUPPORTED_TARGETS:
            raise IntegratedGradientsError(f"未対応のIG targetです: {target}")
        normalized.append(target)  # type: ignore[arg-type]
    if len(set(normalized)) != len(normalized):
        raise IntegratedGradientsError("同じIG targetを重複指定できません")
    return tuple(normalized)


def _adapter_tensor(adapter: _TensorPolicyAdapter, observations: np.ndarray) -> torch.Tensor:
    if hasattr(adapter, "observations_to_tensor"):
        value = getattr(adapter, "observations_to_tensor")(observations)
        if not isinstance(value, torch.Tensor):
            raise IntegratedGradientsError("adapter.observations_to_tensorはTensorを返す必要があります")
        return value
    device = getattr(adapter, "device", torch.device("cpu"))
    return torch.as_tensor(observations, dtype=torch.float32, device=device)


def _tensor_output(output: object, name: str, *, batch_size: int) -> torch.Tensor:
    if not hasattr(output, name):
        raise IntegratedGradientsError(f"adapter.evaluate_tensorsに{name}がありません")
    value = getattr(output, name)
    if not isinstance(value, torch.Tensor):
        raise IntegratedGradientsError(f"adapter output {name}はTensorである必要があります")
    if name in {"logits", "log_probabilities"}:
        if value.ndim != 2 or value.shape[0] != batch_size:
            raise IntegratedGradientsError(f"adapter output {name}は(B, A)である必要があります")
    else:
        if value.ndim == 2 and value.shape == (batch_size, 1):
            value = value[:, 0]
        if value.ndim != 1 or value.shape[0] != batch_size:
            raise IntegratedGradientsError(f"adapter output {name}は(B,)である必要があります")
    if not torch.isfinite(value).all():
        raise IntegratedGradientsError(f"adapter output {name}に非有限値があります")
    return value


def _fixed_actions(adapter: _TensorPolicyAdapter, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    output = adapter.evaluate_tensors(inputs)
    logits = _tensor_output(output, "logits", batch_size=inputs.shape[0])
    action_count = logits.shape[1]
    actions = torch.argmax(logits, dim=1).to(dtype=torch.long).detach()
    if action_count >= 2:
        runner_up = torch.topk(logits, k=2, dim=1).indices[:, 1].to(dtype=torch.long).detach()
    else:
        runner_up = torch.full_like(actions, -1)
    return actions, runner_up, action_count


def run_integrated_gradients(
    adapter: _TensorPolicyAdapter,
    observations: np.ndarray,
    baselines: ResolvedBaselines | np.ndarray,
    *,
    targets: Sequence[IGTarget | str] | IGTarget | str = ("selected_log_probability",),
    steps: int = 64,
    batch_size: int = 128,
) -> IntegratedGradientsResult:
    """Attribute PPO actor/critic outputs while fixing action indices at ``x``.

    For actor targets, ``a*`` (and runner-up where needed) are chosen exactly
    once from the original input observation.  They are *not* reselected at
    interpolation points, even if another action becomes the argmax there.
    """

    if not isinstance(adapter, _TensorPolicyAdapter):
        raise IntegratedGradientsError("adapterはevaluate_tensors(observations)を実装する必要があります")
    number_of_steps = _validate_steps(steps)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise IntegratedGradientsError("batch_sizeは1以上の整数で指定してください")
    normalized_targets = _normalise_targets(targets)
    source = np.asarray(observations)
    if source.ndim != 2 or source.shape[0] == 0 or source.shape[1] == 0:
        raise IntegratedGradientsError("observationsは非空の(N, D)配列で指定してください")
    if not np.issubdtype(source.dtype, np.number) or not np.all(np.isfinite(source)):
        raise IntegratedGradientsError("observationsは有限の数値配列で指定してください")
    source = np.array(source, dtype=np.float32, copy=True)
    baseline_values, baseline_ids = _normalise_baselines(
        baselines,
        sample_count=source.shape[0],
        observation_dim=source.shape[1],
    )
    sample_count, baseline_count, observation_dim = baseline_values.shape
    target_count = len(normalized_targets)
    attributions = np.empty(
        (sample_count, target_count, baseline_count, observation_dim), dtype=np.float64
    )
    scalar_shape = (sample_count, target_count, baseline_count)
    input_outputs = np.empty(scalar_shape, dtype=np.float64)
    baseline_outputs = np.empty(scalar_shape, dtype=np.float64)
    attribution_sums = np.empty(scalar_shape, dtype=np.float64)
    residuals = np.empty(scalar_shape, dtype=np.float64)
    absolute_residuals = np.empty(scalar_shape, dtype=np.float64)
    relative_errors = np.empty(scalar_shape, dtype=np.float64)
    target_actions = np.empty(sample_count, dtype=np.int64)
    runner_up_actions = np.empty(sample_count, dtype=np.int64)

    for start in range(0, sample_count, batch_size):
        end = min(start + batch_size, sample_count)
        input_tensor = _adapter_tensor(adapter, source[start:end]).detach()
        if input_tensor.ndim != 2 or input_tensor.shape != (end - start, observation_dim):
            raise IntegratedGradientsError(
                "adapter input tensorのshapeがobservationsと一致しません: "
                f"expected {(end - start, observation_dim)}, got {tuple(input_tensor.shape)}"
            )
        fixed_action, fixed_runner_up, action_count = _fixed_actions(adapter, input_tensor)
        if "selected_vs_runner_up_margin" in normalized_targets and action_count < 2:
            raise IntegratedGradientsError("selected_vs_runner_up_marginには2以上のDiscrete actionが必要です")
        target_actions[start:end] = fixed_action.detach().cpu().numpy()
        runner_up_actions[start:end] = fixed_runner_up.detach().cpu().numpy()

        for target_index, target in enumerate(normalized_targets):
            def target_fn(path: torch.Tensor, *, _target: IGTarget = target, _actions: torch.Tensor = fixed_action, _runner_up: torch.Tensor = fixed_runner_up) -> torch.Tensor:
                output = adapter.evaluate_tensors(path)
                if _target == "selected_log_probability":
                    log_probs = _tensor_output(output, "log_probabilities", batch_size=path.shape[0])
                    return log_probs.gather(1, _actions[:, None])[:, 0]
                if _target == "selected_vs_runner_up_margin":
                    logits = _tensor_output(output, "logits", batch_size=path.shape[0])
                    return logits.gather(1, _actions[:, None])[:, 0] - logits.gather(
                        1, _runner_up[:, None]
                    )[:, 0]
                return _tensor_output(output, "values", batch_size=path.shape[0])

            for baseline_index in range(baseline_count):
                baseline_tensor = torch.as_tensor(
                    baseline_values[start:end, baseline_index, :],
                    dtype=input_tensor.dtype,
                    device=input_tensor.device,
                )
                computation = compute_integrated_gradients(
                    input_tensor,
                    baseline_tensor,
                    target_fn,
                    steps=number_of_steps,
                )
                # The high-level caller always supplied a batch, so the
                # generic computation returns B,D and B scalar outputs.
                batch_attributions = computation.attributions.detach().cpu().numpy()
                batch_input_output = computation.input_output.detach().cpu().numpy()
                batch_baseline_output = computation.baseline_output.detach().cpu().numpy()
                batch_sum = computation.attribution_sum.detach().cpu().numpy()
                batch_residual = computation.completeness_residual.detach().cpu().numpy()
                if batch_attributions.shape != (end - start, observation_dim):
                    raise IntegratedGradientsError("IG内部結果のshapeが不正です")
                attributions[start:end, target_index, baseline_index, :] = batch_attributions
                input_outputs[start:end, target_index, baseline_index] = batch_input_output
                baseline_outputs[start:end, target_index, baseline_index] = batch_baseline_output
                attribution_sums[start:end, target_index, baseline_index] = batch_sum
                residuals[start:end, target_index, baseline_index] = batch_residual
                absolute_residuals[start:end, target_index, baseline_index] = np.abs(batch_residual)
                output_difference = batch_input_output - batch_baseline_output
                relative_errors[start:end, target_index, baseline_index] = np.abs(
                    batch_residual
                ) / np.maximum(
                    np.abs(output_difference), _RELATIVE_COMPLETENESS_EPSILON
                )
    return IntegratedGradientsResult(
        targets=normalized_targets,
        attributions=attributions,
        baseline_ids=baseline_ids,
        target_actions=target_actions,
        runner_up_actions=runner_up_actions,
        input_outputs=input_outputs,
        baseline_outputs=baseline_outputs,
        attribution_sums=attribution_sums,
        completeness_residuals=residuals,
        absolute_completeness_residuals=absolute_residuals,
        relative_completeness_errors=relative_errors,
        metadata={
            "steps": number_of_steps,
            "batch_size": batch_size,
            "target_action_rule": "argmax action at original x, fixed for every interpolation point",
            "runner_up_rule": "second-highest raw logit at original x, fixed for every interpolation point",
            "integration_rule": "trapezoidal endpoints included",
            "relative_completeness_error": (
                "abs(completeness_residual) / "
                f"max(abs(F(x) - F(x0)), {_RELATIVE_COMPLETENESS_EPSILON:g})"
            ),
        },
    )


analyze_integrated_gradients = run_integrated_gradients
