"""通常走行の観測・方策出力・環境テレメトリを保存する。

アダプターの契約は小さなduck typingに限定している。必須なのは環境を返す
``make_env(config)``（または ``create_env``）、モデル入力を返す
``preprocess_observation``（省略時は観測を配列化）、任意の``telemetry``と
``decode_action``である。MetaDrive本体のimportはアダプター内に閉じ込める。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
import copy
from io import BytesIO
import inspect
from pathlib import Path
import shutil
from typing import Any, Callable

import numpy as np

from .artifacts import RunArtifacts, build_manifest, jsonl_write, save_observations, sha256_file
from .schema import InputSchema


class CollectionError(RuntimeError):
    """通常走行の収集が成立しなかった。"""


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return str(value)


def _copy_observation(value: Any) -> Any:
    """Copy every observation before preprocessing or intervention."""

    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _call(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call a hook after signature binding, avoiding duplicate side effects.

    A TypeError raised *inside* a hook is never treated as a signature fallback;
    this matters for reset/step hooks because retrying them could consume a new
    simulator state. Builtins without inspectable signatures receive one call.
    """

    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func(*args, **kwargs)
    try:
        signature.bind(*args, **kwargs)
    except TypeError as exc:
        raise TypeError(f"adapter hook signature mismatch for {func!r}: {exc}") from exc
    return func(*args, **kwargs)


def _call_variants(func: Callable[..., Any], variants: Sequence[tuple[tuple[Any, ...], dict[str, Any]]]) -> Any:
    """Select one signature-compatible call before invoking a hook exactly once."""

    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        args, kwargs = variants[0]
        return func(*args, **kwargs)
    for args, kwargs in variants:
        try:
            signature.bind(*args, **kwargs)
        except TypeError:
            continue
        return func(*args, **kwargs)
    raise TypeError(f"no compatible signature for adapter/policy hook {func!r}")


def _find_hook(adapter: Any, names: Sequence[str]) -> Callable[..., Any] | None:
    for name in names:
        candidate = getattr(adapter, name, None)
        if callable(candidate):
            return candidate
    if isinstance(adapter, Mapping):
        for name in names:
            candidate = adapter.get(name)
            if callable(candidate):
                return candidate
    return None


def make_environment(adapter: Any, config: Any) -> Any:
    hook = _find_hook(adapter, ("make_env", "create_env", "environment", "build_env"))
    if hook is not None:
        seed = getattr(config, "scenario_seed", None)
        return _call_variants(
            hook,
            (
                ((config,), {}),
                ((), {"seed": seed}),
                ((), {}),
            ),
        )
    if callable(adapter):
        return _call(adapter, config)
    raise CollectionError("adapterにmake_env/create_envフックがありません")


def close_environment(adapter: Any, env: Any) -> None:
    hook = _find_hook(adapter, ("close_env", "close"))
    if hook is not None:
        try:
            signature = inspect.signature(hook)
        except (TypeError, ValueError):
            signature = None
        if signature is None:
            _call(hook, env)
            return
        try:
            signature.bind(env)
        except TypeError:
            try:
                signature.bind()
            except TypeError as exc:
                raise CollectionError("adapter close hook must accept env or no arguments") from exc
            _call(hook)
        else:
            _call(hook, env)
        return
    close = getattr(env, "close", None)
    if callable(close):
        close()


def seed_runtime(adapter: Any, env: Any, seed: int | None) -> dict[str, Any]:
    """Seed policy/environment-side RNGs separately from scenario reset seed.

    The scenario seed is passed to ``reset(seed=...)``.  This helper only seeds
    action/observation spaces or an explicitly declared adapter hook, so a
    policy RNG cannot accidentally consume the simulator's scenario sequence.
    """

    if seed is None:
        return {"seed": None, "applied": False, "targets": []}
    targets: list[str] = []
    hook = _find_hook(adapter, ("seed_runtime", "seed_rl", "seed_policy", "seed"))
    if hook is not None:
        try:
            _call_variants(
                hook,
                (
                    ((env,), {"seed": int(seed)}),
                    ((), {"seed": int(seed)}),
                    ((int(seed),), {}),
                ),
            )
            targets.append("adapter")
        except TypeError as exc:
            raise CollectionError(
                "adapter seed hook must explicitly accept seed or (env, seed=); "
                "RL seedを黙って捨てるfallbackは未対応です"
            ) from exc
    for name in ("action_space", "observation_space"):
        space = getattr(env, name, None)
        seed_method = getattr(space, "seed", None)
        if callable(seed_method):
            _call(seed_method, int(seed))
            targets.append(name)
    return {"seed": int(seed), "applied": bool(targets), "targets": targets}


def reset_environment(adapter: Any, env: Any, seed: int | None) -> tuple[Any, dict[str, Any]]:
    hook = _find_hook(adapter, ("reset_env", "reset"))
    if hook is not None:
        if seed is not None:
            try:
                result = _call_variants(
                    hook,
                    (
                        ((env,), {"seed": seed}),
                        ((env,), {"scenario_seed": seed}),
                        ((), {"scenario_seed": seed}),
                        ((), {"seed": seed}),
                    ),
                )
            except TypeError as exc:
                raise CollectionError(
                    "adapter reset hook must explicitly accept env+seed= or scenario_seed=; "
                    "seedを黙って捨てるfallbackは未対応です"
                ) from exc
        else:
            result = _call_variants(hook, (((env,), {}), ((), {})))
    else:
        reset = getattr(env, "reset", None)
        if not callable(reset):
            raise CollectionError("環境にresetがありません")
        if seed is None:
            result = reset()
        else:
            try:
                result = _call(reset, seed=seed)
            except TypeError as exc:
                raise CollectionError(
                    "environment.reset must accept seed= for deterministic collection"
                ) from exc
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], Mapping):
        return result[0], dict(result[1])
    return result, {}


def step_environment(env: Any, action: Any) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
    step = getattr(env, "step", None)
    if not callable(step):
        raise CollectionError("環境にstepがありません")
    result = step(action)
    if not isinstance(result, tuple):
        raise CollectionError("環境stepの戻り値がtupleではありません")
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        return observation, reward, bool(terminated), bool(truncated), dict(info or {})
    raise CollectionError(
        f"環境stepはGymnasiumの5要素(observation,reward,terminated,truncated,info)を要求します: {len(result)}"
    )


def model_input(adapter: Any, observation: Any, info: Mapping[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    copied = _copy_observation(observation)
    hook = _find_hook(adapter, ("preprocess_observation", "to_model_input", "model_input", "preprocess"))
    metadata: dict[str, Any] = {}
    if hook is not None:
        result = _call_variants(
            hook,
            (
                ((_copy_observation(copied), dict(info or {})), {}),
                ((_copy_observation(copied),), {}),
            ),
        )
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], Mapping):
            result, metadata = result
        copied = result
    try:
        array = np.array(copied, copy=True)
    except (TypeError, ValueError) as exc:
        raise CollectionError(f"モデル入力へ配列化できません: {exc}") from exc
    if array.ndim == 0:
        array = array.reshape(1)
    if array.dtype.kind not in "biufc":
        raise CollectionError(f"モデル入力dtypeが数値ではありません: {array.dtype}")
    return array, _plain(metadata)


def policy_probabilities(policy: Any, observation: np.ndarray) -> np.ndarray:
    hook = _find_hook(
        policy,
        ("probabilities", "distribution_probabilities", "action_probabilities", "predict_proba"),
    )
    if hook is None:
        raise CollectionError("policyにprobabilities(observation)がありません")
    value = _call_variants(
        hook,
        (
            ((np.asarray(observation)[None, :].copy(),), {}),
            ((np.array(observation, copy=True),), {}),
        ),
    )
    if isinstance(value, tuple):
        value = value[0]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    raw_array = np.asarray(value)
    if raw_array.dtype.kind not in "biufc":
        raise CollectionError(f"policy確率dtypeが数値ではありません: {raw_array.dtype}")
    array = raw_array.astype(np.float64, copy=False)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise CollectionError("policy確率が有限の1次元配列ではありません")
    if np.any(array < 0):
        raise CollectionError("policy確率に負の値があります")
    total = float(array.sum())
    if total <= 0:
        raise CollectionError("policy確率の合計が正ではありません")
    if not np.isclose(total, 1.0, rtol=1e-5, atol=1e-8):
        raise CollectionError(f"policy確率の合計が1ではありません: {total}")
    return np.array(array, dtype=np.float64, copy=True)


def policy_predict(policy: Any, observation: np.ndarray, *, deterministic: bool = True) -> Any:
    hook = _find_hook(policy, ("predict",))
    if hook is None:
        # A probabilities-only fake policy remains usable for deterministic
        # collection, while real stochastic prediction still requires predict.
        probabilities = policy_probabilities(policy, observation)
        return int(np.argmax(probabilities))
    result = _call_variants(
        hook,
        (
            ((np.array(observation, copy=True),), {"deterministic": deterministic}),
            ((np.array(observation, copy=True),), {}),
        ),
    )
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, np.ndarray):
        if result.size != 1:
            raise CollectionError(f"policy actionが単一値ではありません: shape={result.shape}")
        result = result.reshape(-1)[0]
        result = result.item()
    if hasattr(result, "item"):
        result = result.item()
    if isinstance(result, (list, tuple)):
        if len(result) != 1:
            raise CollectionError(f"policy actionが単一値ではありません: length={len(result)}")
        result = result[0]
    if isinstance(result, (np.integer, int)) and not isinstance(result, bool):
        return int(result)
    raise CollectionError(f"Discrete policy actionが整数ではありません: {type(result).__name__}")


def decode_action(adapter: Any, action: Any, config: Any | None = None) -> Any:
    hook = _find_hook(adapter, ("decode_action", "action_decode", "decode"))
    if hook is None:
        return None
    env_config = getattr(config, "environment", {}) if config is not None else {}
    return _plain(
        _call_variants(
            hook,
            (
                ((action,), {}),
                ((action, env_config), {}),
            ),
        )
    )


def telemetry(adapter: Any, env: Any, info: Mapping[str, Any], *, phase: str, step: int) -> dict[str, Any]:
    def finish(value: Mapping[str, Any]) -> dict[str, Any]:
        merged = dict(value)
        merged.setdefault("step", int(step))
        merged.setdefault("phase", str(phase))
        return _plain(merged)

    hook = _find_hook(adapter, ("telemetry", "get_telemetry", "read_telemetry", "metrics"))
    if hook is None:
        runtime_hook = _find_hook(adapter, ("runtime_telemetry",))
        vehicle = getattr(env, "vehicle", None)
        if vehicle is None:
            vehicle = getattr(env, "agent", None)
        if runtime_hook is not None and vehicle is not None:
            hook = runtime_hook
            result = _call_variants(hook, (((vehicle,), {}),))
            merged = dict(info)
            if isinstance(result, Mapping):
                merged.update(result)
            return finish(merged)
    if hook is None:
        return finish(info)
    result = _call_variants(
        hook,
        (
            ((env, dict(info)), {"phase": phase, "step": step}),
            ((env, dict(info)), {}),
            ((env,), {}),
        ),
    )
    merged = dict(info)
    if isinstance(result, Mapping):
        merged.update(result)
    else:
        merged["value"] = result
    return finish(merged)


def telemetry_time(value: Mapping[str, Any], fallback: float | None = None) -> Any:
    """Read the simulator clock using the declared telemetry aliases.

    ``fallback`` remains in the signature for callers written against the
    early runtime API, but an observation index is not a simulator clock.  A
    missing declared clock therefore stays ``None`` so duration metrics can
    report that their time denominator is unavailable.
    """

    for key in (
        "simulation_time_s",
        "simulation_time_seconds",
        "simulation_time",
        "sim_time_seconds",
        "sim_time_s",
        "sim_time",
        "time_s",
        "time",
    ):
        if key in value and value[key] is not None:
            return value[key]
    del fallback
    return None


def render_frame(
    adapter: Any,
    env: Any,
    *,
    step: int,
    simulation_time: Any = None,
    pattern_id: str = "reference",
    phase: str = "post",
) -> tuple[Any | None, dict[str, Any]]:
    """Invoke an explicitly declared frame hook without affecting numeric data."""

    hook = _find_hook(adapter, ("render_frame", "capture_frame", "frame"))
    if hook is None:
        return None, {"status": "unavailable", "reason": "adapter.render_frame is not declared"}
    try:
        frame = _call_variants(
            hook,
            (
                ((env,), {"step": step, "simulation_time": simulation_time, "pattern_id": pattern_id, "phase": phase}),
                ((env,), {"step": step, "simulation_time": simulation_time}),
                ((env,), {"step": step}),
                ((env,), {}),
            ),
        )
    except Exception as exc:
        return None, {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    if frame is None:
        return None, {"status": "failed", "reason": "adapter.render_frame returned None"}
    return frame, {"status": "captured", "pattern_id": pattern_id, "step": int(step), "simulation_time": _plain(simulation_time)}


def _frame_image(frame: Any) -> Any:
    """Convert common adapter frame values to a PIL image lazily."""

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional video dependency
        raise CollectionError("Pillow is required only for optional video output") from exc
    if isinstance(frame, Image.Image):
        return frame.copy()
    if isinstance(frame, (str, Path)):
        with Image.open(frame) as image:
            return image.copy()
    if isinstance(frame, (bytes, bytearray, memoryview)):
        with Image.open(BytesIO(bytes(frame))) as image:
            return image.copy()
    array = np.asarray(frame)
    if array.ndim not in {2, 3}:
        raise CollectionError(f"render_frame returned unsupported array shape: {array.shape}")
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and np.nanmin(array) >= 0 and np.nanmax(array) <= 1:
            array = np.rint(array * 255).astype(np.uint8)
        else:
            array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def save_video_frame(
    frame: Any,
    directory: str | Path,
    *,
    step: int,
    simulation_time: Any = None,
    pattern_id: str = "reference",
    phase: str = "post",
) -> dict[str, Any]:
    """Persist one annotated frame and its timing mapping; failures are isolated.

    The annotation is drawn on the image copy returned by ``_frame_image``.
    It is deliberately ASCII-only so Pillow's bundled default font can render
    it consistently across headless environments.
    """

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"frame-{int(step):06d}.png"
    try:
        image = _frame_image(frame)
        # Rendered frames can be RGB, RGBA, palette, or grayscale.  RGBA keeps
        # the source untouched while allowing a translucent black label panel.
        image = image.convert("RGBA")
        try:
            from PIL import ImageDraw, ImageFont

            draw = ImageDraw.Draw(image, "RGBA")
            font = ImageFont.load_default()
        except ImportError as exc:  # pragma: no cover - guarded by _frame_image
            raise CollectionError("Pillow is required only for optional video output") from exc

        episode_label = next(
            (part for part in reversed(destination.parts) if part.startswith("episode-")),
            None,
        )
        location = episode_label or destination.name or "."

        def ascii_safe(value: Any) -> str:
            return str(value).encode("ascii", "replace").decode("ascii")

        try:
            clock = float(simulation_time)
            time_label = f"{clock:.3f}s" if np.isfinite(clock) else "NA"
        except (TypeError, ValueError, OverflowError):
            time_label = "NA"
        label = (
            f"pattern={ascii_safe(pattern_id)} | path={ascii_safe(location)} | "
            f"step={int(step):06d} | {ascii_safe(phase)} t={time_label}"
        )
        try:
            left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
            text_width = right - left
            text_height = bottom - top
        except AttributeError:  # pragma: no cover - older Pillow compatibility
            text_width, text_height = draw.textsize(label, font=font)
        margin = 4
        panel_width = min(image.width, text_width + margin * 2)
        panel_height = min(image.height, text_height + margin * 2)
        if panel_width > 0 and panel_height > 0:
            draw.rectangle((0, 0, panel_width, panel_height), fill=(0, 0, 0, 190))
            draw.text((margin, margin), label, fill=(255, 255, 255, 255), font=font)
        image.save(path, format="PNG")
        return {
            "status": "saved",
            "path": str(path),
            "step": int(step),
            "simulation_time": _plain(simulation_time),
            "pattern_id": pattern_id,
            "phase": phase,
            "overlay": label,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
            "step": int(step),
            "simulation_time": _plain(simulation_time),
            "pattern_id": pattern_id,
            "phase": phase,
        }


def finalize_video(frame_records: Sequence[Mapping[str, Any]], directory: str | Path, *, fps: float = 10.0) -> dict[str, Any]:
    """Build an optional GIF from saved PNG frames.

    ``fps`` is the playback rate requested by the caller.  GIF stores one
    duration for every frame here, rounded to integer milliseconds as
    ``max(1, round(1000 / fps))``; the returned metadata records both values.
    """

    destination = Path(directory)
    saved = [item for item in frame_records if item.get("status") == "saved" and item.get("path")]
    try:
        requested_fps = float(fps)
    except (TypeError, ValueError, OverflowError):
        requested_fps = 10.0
    if not np.isfinite(requested_fps) or requested_fps <= 0:
        requested_fps = 10.0
    frame_duration_ms = max(1, int(round(1000.0 / requested_fps)))
    result: dict[str, Any] = {
        "frame_count": len(frame_records),
        "saved_count": len(saved),
        "failure_count": sum(item.get("status") == "failed" for item in frame_records),
        "frame_map": [_plain(dict(item)) for item in frame_records],
        "gif": None,
        "fps": requested_fps,
        "frame_duration_ms": frame_duration_ms,
    }
    if not saved:
        result["reason"] = "no frames were saved"
        return result
    try:
        images = []
        from PIL import Image

        for item in saved:
            with Image.open(item["path"]) as image:
                images.append(image.convert("RGB").copy())
        gif_path = destination / "trajectory.gif"
        images[0].save(
            gif_path,
            format="GIF",
            save_all=True,
            append_images=images[1:],
            duration=frame_duration_ms,
            loop=0,
        )
        result["gif"] = str(gif_path)
    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
    return result


def infer_video_fps(records: Sequence[Mapping[str, Any]], *, fallback: float = 10.0) -> float:
    """Infer playback rate from saved pre/post or adjacent post timestamps."""

    times: list[float] = []
    for record in records:
        value = record.get("post_time")
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            times.append(number)
    deltas = [
        current - previous
        for previous, current in zip(times, times[1:], strict=False)
        if current > previous
    ]
    if not deltas:
        return float(fallback)
    return float(1.0 / np.median(np.asarray(deltas, dtype=float)))


@dataclass
class CollectionResult:
    store: RunArtifacts | None
    observations: list[np.ndarray] = field(default_factory=list)
    raw_observations: list[Any] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)
    episodes_completed: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "store": str(self.store.run_dir) if self.store else None,
            "observation_count": len(self.observations),
            "record_count": len(self.records),
            "episodes_completed": self.episodes_completed,
        }


def collect_reference(
    policy: Any,
    adapter: Any,
    config: Any,
    *,
    store: RunArtifacts | None = None,
    episodes: int | None = None,
    max_steps: int | None = None,
    scenario_seed: int | None = None,
    rl_seed: int | None = None,
    deterministic: bool = True,
    schema: InputSchema | None = None,
    video: bool | None = None,
    video_dir: str | Path | None = None,
    video_fps: float | None = None,
) -> CollectionResult:
    """Collect unchanged reference trajectories from fresh sequential envs."""

    if episodes is None:
        scenario = getattr(config, "scenario", {}) or {}
        episodes = int(scenario.get("episodes", 1))
    if episodes <= 0:
        raise CollectionError("episodesは1以上で指定してください")
    result = CollectionResult(store)
    if video is None:
        video = bool((getattr(config, "video", {}) or {}).get("enabled", False))
    video_config = getattr(config, "video", {}) or {}
    configured_video_fps = video_config.get("fps") if isinstance(video_config, Mapping) else None
    if video_fps is None:
        try:
            video_fps = float(configured_video_fps) if configured_video_fps is not None else None
        except (TypeError, ValueError):
            raise CollectionError(f"video.fps must be numeric: {configured_video_fps!r}") from None
    frame_records: list[dict[str, Any]] = []
    if store:
        store.update_status("collect", "running", episodes=episodes)
    try:
        for episode_index in range(episodes):
            env = make_environment(adapter, config)
            seed = scenario_seed
            if seed is None:
                seed = getattr(config, "scenario_seed", None)
            episode_seed = int(seed) + episode_index if seed is not None else None
            episode_rl_seed = int(rl_seed) if rl_seed is not None else getattr(config, "rl_seed", None)
            if video:
                if video_dir is not None:
                    episode_video_dir = Path(video_dir) / f"episode-{episode_index}"
                elif store is not None:
                    episode_video_dir = store.reference_dir / "video" / f"episode-{episode_index}"
                else:
                    episode_video_dir = Path("outputs/input_attribution_video") / f"episode-{episode_index}"
            else:
                episode_video_dir = None
            try:
                seed_metadata = seed_runtime(adapter, env, episode_rl_seed)
                raw_obs, reset_info = reset_environment(adapter, env, episode_seed)
                done = False
                step = 0
                while not done:
                    raw_for_record = _copy_observation(raw_obs)
                    pre = telemetry(adapter, env, reset_info, phase="pre", step=step)
                    model_obs, preprocess_info = model_input(adapter, raw_obs, reset_info)
                    if schema is not None:
                        try:
                            schema.validate_observation(model_obs, copy=False)
                        except Exception as exc:
                            raise CollectionError(f"reference model input violates schema at step {step}: {exc}") from exc
                    probabilities = policy_probabilities(policy, model_obs)
                    action = policy_predict(policy, model_obs, deterministic=deterministic)
                    if isinstance(action, (int, np.integer)) and not 0 <= int(action) < probabilities.size:
                        raise CollectionError(f"policy actionが出力数外です: {action}")
                    next_raw, reward, terminated, truncated, step_info = step_environment(env, action)
                    post = telemetry(adapter, env, step_info, phase="post", step=step + 1)
                    simulation_time = telemetry_time(pre, step)
                    post_time = telemetry_time(post, step + 1)
                    record = {
                        "episode": episode_index,
                        "episode_id": f"episode-{episode_index}",
                        "scenario_seed": episode_seed,
                        "rl_seed": episode_rl_seed,
                        "step": step,
                        "simulation_time": _plain(simulation_time),
                        "pre_time": _plain(simulation_time),
                        "post_time": _plain(post_time),
                        "seed_metadata": seed_metadata,
                        "observation_index": len(result.observations),
                        "observation_shape": list(model_obs.shape),
                        "observation_dtype": str(model_obs.dtype),
                        "probabilities": probabilities.tolist(),
                        "action": _plain(action),
                        "decoded_action": decode_action(adapter, action, config=config),
                        "preprocess": preprocess_info,
                        "pre_telemetry": pre,
                        "post_telemetry": post,
                        "reward": _plain(reward),
                        "terminated": terminated,
                        "truncated": truncated,
                        "done": bool(terminated or truncated),
                        "info": _plain(step_info),
                    }
                    result.observations.append(np.array(model_obs, copy=True))
                    result.raw_observations.append(raw_for_record)
                    result.records.append(record)
                    if video:
                        frame, frame_status = render_frame(
                            adapter,
                            env,
                            step=step,
                            simulation_time=post_time,
                            pattern_id="P00",
                            phase="post",
                        )
                        if frame is not None:
                            frame_status = save_video_frame(
                                frame,
                                episode_video_dir,
                                step=step,
                                simulation_time=post_time,
                                pattern_id="P00",
                                phase="post",
                            )
                        frame_status["episode"] = episode_index
                        frame_records.append(frame_status)
                        record["frame_status"] = frame_status
                    raw_obs = next_raw
                    reset_info = step_info
                    step += 1
                    done = terminated or truncated
                    if max_steps is not None and step >= max_steps and not done:
                        # A deliberate budget cutoff is distinct from env
                        # termination, and is never padded with fake records.
                        record["budget_truncated"] = True
                        done = True
                result.episodes_completed += 1
            finally:
                close_environment(adapter, env)
        if store:
            saved = save_observations(store, result.observations, result.records, stage="00_reference")
            try:
                raw_array = np.stack([np.asarray(item) for item in result.raw_observations], axis=0)
                if raw_array.dtype.kind not in "biufc":
                    raise TypeError(f"raw observation dtype is not numeric: {raw_array.dtype}")
                raw_path = store.reference_dir / "raw_observations.npy"
                np.save(raw_path, np.array(raw_array, copy=True), allow_pickle=False)
                saved["raw_observations"] = str(raw_path)
            except (TypeError, ValueError):
                error = CollectionError(
                    "raw observations must be fixed-shape numeric arrays; "
                    "object/dict/ragged raw observations are unsupported"
                )
                store.write_json(
                    "00_reference/raw_observations_error.json",
                    {"error": str(error), "record_count": len(result.records)},
                )
                raise error from None
            if video:
                video_directory = Path(video_dir) if video_dir is not None else store.reference_dir / "video"
                video_result = finalize_video(
                    frame_records,
                    video_directory,
                    fps=video_fps if video_fps is not None else infer_video_fps(result.records),
                )
                store.write_json("00_reference/video.json", video_result)
            if store.manifest_path.is_file():
                manifest = store.load_manifest()
                if saved.get("observations") and Path(str(saved["observations"])).is_file():
                    manifest.setdefault("observations", {})["path"] = str(saved["observations"])
                    manifest["observations"]["sha256"] = sha256_file(str(saved["observations"]))
                manifest.setdefault("stages", {})["collect"] = {
                    "records": len(result.records),
                    "episodes": result.episodes_completed,
                    "video": bool(video),
                }
            else:
                manifest = build_manifest(
                    config=config,
                    command="collect",
                    model_path=getattr(config, "model_path", None),
                    schema_path=getattr(config, "schema_path", None),
                    patterns=getattr(config, "patterns", None),
                    preprocess=getattr(config, "preprocess", None),
                    observations_path=saved.get("observations"),
                    extra={"data": result.as_dict()},
                )
            store.write_json("00_reference/summary.json", result.as_dict() | {"files": saved, "video": bool(video)})
            store.save_manifest(manifest)
            store.update_status("collect", "success", records=len(result.records), episodes=result.episodes_completed)
        return result
    except Exception as exc:
        if store:
            store.update_status("collect", "failed", error=f"{type(exc).__name__}: {exc}")
        raise


collect = collect_reference


__all__ = [
    "CollectionError",
    "CollectionResult",
    "close_environment",
    "collect",
    "collect_reference",
    "decode_action",
    "finalize_video",
    "infer_video_fps",
    "make_environment",
    "model_input",
    "policy_predict",
    "policy_probabilities",
    "render_frame",
    "reset_environment",
    "save_video_frame",
    "seed_runtime",
    "step_environment",
    "telemetry_time",
    "telemetry",
]
