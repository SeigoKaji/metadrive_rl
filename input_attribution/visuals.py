"""Standalone visual artifact helpers for input attribution reports.

The functions in this module deliberately operate on ordinary Python values
and paths.  They do not import the simulator, a policy implementation, or a
machine learning framework.  This lets a saved run be rendered on a machine
which does not have MetaDrive, Stable-Baselines3, or PyTorch installed.

Pillow and Matplotlib are optional at import time and are loaded only when a
GIF or PNG is requested.  The report layer can consequently still inspect a
run and produce an HTML page when one of the drawing backends is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


LN2 = math.log(2.0)
_CJK_FONT_NAME: str | None = None


class VisualError(RuntimeError):
    """A requested visual artifact could not be produced."""


@dataclass(frozen=True, slots=True)
class VisualResult:
    """Result of producing one derived visual artifact.

    ``path`` is ``None`` when no artifact was written.  ``metadata`` is kept
    as a plain mapping so callers can persist it outside the raw ``data``
    directory without importing any numerical package.
    """

    path: Path | None
    status: str
    metadata: Mapping[str, Any]
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status in {"complete", "empty"} and not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path) if self.path is not None else None,
            "status": self.status,
            "metadata": dict(self.metadata),
            "errors": list(self.errors),
        }


def _number(value: Any) -> float | None:
    """Return a finite real number, preserving missing values as ``None``."""

    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, (str, bytes)) or value is None:
        return []
    if isinstance(value, Sequence):
        return list(value)
    try:
        return list(value)
    except (TypeError, ValueError):
        return []


def _validate_distribution(value: Any, *, name: str) -> tuple[float, ...]:
    """Validate a probability vector without silently normalising it.

    A tiny sum error is accepted because JSON serialization can round a
    policy output.  Negative values, non-finite values, and material sum
    errors are rejected so a malformed saved result never becomes a plausible
    looking JS curve.
    """

    values = _sequence(value)
    if not values:
        raise ValueError(f"{name} must be a non-empty probability sequence")
    numbers: list[float] = []
    for index, item in enumerate(values):
        number = _number(item)
        if number is None:
            raise ValueError(f"{name}[{index}] is not finite")
        if number < -1e-12:
            raise ValueError(f"{name}[{index}] is negative: {number!r}")
        # A negative value smaller than the tolerance is a serialization
        # artifact.  Treat that one value as mathematical zero, rather than
        # changing a materially invalid distribution.
        numbers.append(0.0 if number < 0.0 else number)
    total = math.fsum(numbers)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError(f"{name} has an invalid sum: {total!r}")
    if not math.isclose(total, 1.0, rel_tol=1e-7, abs_tol=1e-8):
        raise ValueError(f"{name} does not sum to one: {total!r}")
    return tuple(numbers)


def validate_probability_distribution(value: Any, *, name: str = "probabilities") -> tuple[float, ...]:
    """Public probability validation used by report tests and chart code."""

    return _validate_distribution(value, name=name)


def js_divergence(p: Any, q: Any) -> float:
    """Call the canonical core JS implementation for visual annotations.

    Keeping the formula in :mod:`policy_comparison` makes offline records and
    report-side checks use exactly the same probability validation and nats
    convention.  The import is lazy so merely importing the report module
    remains lightweight and simulator independent.
    """

    try:
        from .policy_comparison import PolicyComparisonError, js_divergence as canonical_js
    except ImportError as exc:  # pragma: no cover - broken package install
        raise ValueError("canonical policy JS implementation is unavailable") from exc
    try:
        return float(canonical_js(p, q))
    except PolicyComparisonError as exc:
        raise ValueError(str(exc)) from exc


def argmax_index(value: Any) -> int:
    """Return the first index of a finite sequence's maximum value."""

    values = _sequence(value)
    if not values:
        raise ValueError("argmax requires a non-empty sequence")
    numbers = []
    for index, item in enumerate(values):
        number = _number(item)
        if number is None:
            raise ValueError(f"argmax value at index {index} is not finite")
        numbers.append(number)
    return max(range(len(numbers)), key=lambda index: numbers[index])


def _require_pillow() -> tuple[Any, Any, Any]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise VisualError(
            "Pillow is required to render GIFs and frame overlays; "
            "install input_attribution/requirements-report.txt"
        ) from exc
    return Image, ImageDraw, ImageFont


def _gif_duration_ms(action_dt: Any) -> int:
    value = _number(action_dt)
    if value is None or value <= 0.0:
        raise VisualError(f"action_dt must be a positive finite number: {action_dt!r}")
    # GIF stores durations in centiseconds.  Do not silently introduce a
    # playback drift for a value that cannot be represented exactly.
    milliseconds = value * 1000.0
    rounded = int(round(milliseconds / 10.0) * 10)
    if rounded <= 0 or not math.isclose(
        rounded / 1000.0, value, rel_tol=0.0, abs_tol=1e-9
    ):
        raise VisualError(
            "action_dt cannot be represented exactly by GIF's 10 ms unit: "
            f"{value:.9f}s"
        )
    return rounded


def _record_executed(record: Mapping[str, Any]) -> bool:
    # The new run contract records this flag on every row.  A missing flag is
    # treated as unexecuted so an incomplete/legacy row cannot manufacture a
    # GIF frame or reward sample.
    value = record.get("executed", False)
    if isinstance(value, str):
        return value.strip().casefold() not in {"false", "0", "no", "failed"}
    return bool(value)


def _record_step(record: Mapping[str, Any], fallback: int) -> int:
    try:
        value = int(record.get("step", fallback))
    except (TypeError, ValueError, OverflowError):
        value = fallback
    return value


def _safe_frame_path(run_dir: Path, frame: Any) -> Path:
    if not isinstance(frame, str) or not frame.strip():
        raise VisualError("frame path is missing")
    candidate_text = frame.strip()
    candidate = Path(candidate_text)
    if candidate.is_absolute():
        raise VisualError("frame path must be relative to the run directory")
    root = run_dir.resolve()
    resolved = (run_dir / candidate).resolve()
    if resolved != root and root not in resolved.parents:
        raise VisualError(f"frame path escapes run directory: {frame!r}")
    return resolved


def _load_frame(path: Path) -> Any:
    Image, _draw, _font = _require_pillow()
    try:
        image = Image.open(path)
        image.load()
    except (OSError, ValueError) as exc:
        raise VisualError(f"cannot open frame {path}: {exc}") from exc
    # A GIF should contain plain RGB post-step images.  Converting here keeps
    # alpha/palette details from one simulator renderer from affecting output.
    return image.convert("RGB")


def _frame_entries(
    records: Iterable[Mapping[str, Any]], run_dir: Path
) -> tuple[list[tuple[int, Path, Mapping[str, Any]]], list[str], int]:
    entries: list[tuple[int, Path, Mapping[str, Any]]] = []
    errors: list[str] = []
    expected = 0
    for fallback, item in enumerate(records):
        if not isinstance(item, Mapping):
            errors.append(f"record {fallback} is not an object")
            continue
        if not _record_executed(item):
            continue
        expected += 1
        frame = item.get("frame")
        if frame in (None, ""):
            errors.append(f"step {_record_step(item, fallback)} has no frame")
            continue
        try:
            path = _safe_frame_path(run_dir, frame)
        except VisualError as exc:
            errors.append(f"step {_record_step(item, fallback)}: {exc}")
            continue
        if not path.is_file():
            errors.append(f"step {_record_step(item, fallback)} frame does not exist: {frame}")
            continue
        entries.append((_record_step(item, fallback), path, item))
    return entries, errors, expected


def render_rollout_gif(
    records: Iterable[Mapping[str, Any]],
    run_dir: str | Path,
    output_path: str | Path,
    action_dt: Any,
    *,
    metadata_path: str | Path | None = None,
    overlay_lines: Callable[[Mapping[str, Any], int], Iterable[str]] | None = None,
) -> VisualResult:
    """Encode recorded post-step RGB frames into a duration-checked GIF.

    A missing frame is retained as an error and never replaced with a blank
    or duplicated frame.  If at least one valid frame exists, it is still
    encoded so a partial run remains inspectable; the caller receives a
    ``partial`` status and can make its CLI exit non-zero.
    """

    run_root = Path(run_dir)
    destination = Path(output_path)
    records_list = [record for record in records if isinstance(record, Mapping)]
    entries, errors, expected_count = _frame_entries(records_list, run_root)
    metadata: dict[str, Any] = {
        "action_dt_seconds": _number(action_dt),
        "expected_executed_frames": expected_count,
        "source_frame_steps": [step for step, _path, _record in entries],
        "source_frame_count": len(entries),
        "duration_ms": None,
        "total_duration_seconds": None,
        "gif_frame_count": 0,
        "correspondence": "unavailable",
    }
    try:
        duration_ms = _gif_duration_ms(action_dt)
        metadata["duration_ms"] = duration_ms
    except VisualError as exc:
        errors.append(str(exc))
        result = VisualResult(None, "failed", metadata, tuple(errors))
        _write_optional_metadata(metadata_path, result)
        return result
    if not entries:
        if expected_count == 0 and not errors:
            errors.append("no executed post-step frames were recorded")
        metadata["correspondence"] = "missing"
        result = VisualResult(None, "unavailable", metadata, tuple(errors))
        _write_optional_metadata(metadata_path, result)
        return result

    images: list[Any] = []
    for step, path, record in entries:
        try:
            image = _load_frame(path)
            if overlay_lines is not None:
                image = annotate_frame(image, lines=overlay_lines(record, step))
            images.append(image)
        except VisualError as exc:
            errors.append(f"step {step}: {exc}")
    if not images:
        result = VisualResult(None, "failed", metadata, tuple(errors))
        _write_optional_metadata(metadata_path, result)
        return result

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        first, *rest = images
        first.save(
            destination,
            format="GIF",
            save_all=True,
            append_images=rest,
            duration=duration_ms,
            loop=0,
            optimize=False,
            disposal=2,
        )
        # Reopen the result to verify that the encoder retained every frame.
        check, actual_durations = _inspect_gif(destination)
    except (OSError, ValueError, VisualError, TypeError, RuntimeError) as exc:
        errors.append(f"GIF encoding failed: {exc}")
        result = VisualResult(None, "failed", metadata, tuple(errors))
        _write_optional_metadata(metadata_path, result)
        return result

    metadata.update(
        {
            "gif_frame_count": check,
            "actual_frame_durations_ms": actual_durations,
            "total_duration_seconds": sum(actual_durations) / 1000.0,
            "expected_duration_seconds": expected_count * duration_ms / 1000.0,
            "duration_error_seconds": (
                sum(actual_durations) / 1000.0 - expected_count * duration_ms / 1000.0
            ),
            "correspondence": (
                "verified"
                if not errors
                and check == expected_count
                and check == len(entries)
                and all(duration == duration_ms for duration in actual_durations)
                else "partial"
            ),
        }
    )
    if check != len(images):
        errors.append(f"GIF frame count mismatch: encoded={check}, source={len(images)}")
    if expected_count != len(entries):
        errors.append(
            "record/frame correspondence incomplete: "
            f"executed={expected_count}, frames={len(entries)}"
        )
    if any(duration != duration_ms for duration in actual_durations):
        errors.append(
            "GIF frame duration mismatch: "
            f"expected={duration_ms}ms, actual={actual_durations}"
        )
    status = "complete" if not errors else "partial"
    result = VisualResult(destination, status, metadata, tuple(dict.fromkeys(errors)))
    _write_optional_metadata(metadata_path, result)
    return result


def _inspect_gif(path: Path) -> tuple[int, list[int]]:
    Image, _draw, _font = _require_pillow()
    with Image.open(path) as image:
        count = 0
        durations: list[int] = []
        while True:
            count += 1
            duration = image.info.get("duration")
            try:
                duration_number = int(duration)
            except (TypeError, ValueError):
                duration_number = 0
            durations.append(duration_number)
            try:
                image.seek(count)
            except EOFError:
                break
        return count, durations


def _load_gif_frame_count(path: Path) -> int:
    """Return the encoded frame count (kept as a small compatibility helper)."""

    return _inspect_gif(path)[0]


def _write_optional_metadata(path: str | Path | None, result: VisualResult) -> None:
    if path is None:
        return
    import json

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = result.as_dict()
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _matplotlib():
    global _CJK_FONT_NAME
    try:
        import matplotlib

        # Calling use before pyplot is imported is safe and makes report
        # generation work on headless machines and in CI.
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as pyplot
        # Use a host font when one is available.  A report must also work on
        # a minimal machine, so chart text below falls back to ASCII when no
        # CJK font can be found instead of emitting unreadable glyph boxes.
        if _CJK_FONT_NAME is None:
            try:
                from matplotlib import font_manager

                candidates = (
                    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
                    "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
                    "/usr/share/fonts/opentype/unifont/unifont_jp.otf",
                    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                )
                for candidate in candidates:
                    if Path(candidate).is_file():
                        _CJK_FONT_NAME = font_manager.FontProperties(fname=candidate).get_name()
                        matplotlib.rcParams["font.family"] = [_CJK_FONT_NAME]
                        break
            except (OSError, RuntimeError, ValueError):
                _CJK_FONT_NAME = ""
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise VisualError(
            "Matplotlib is required to render report PNGs; "
            "install input_attribution/requirements-report.txt"
        ) from exc
    return pyplot


def _plot_label(japanese: str, english: str) -> str:
    return japanese if _CJK_FONT_NAME else english


def _plot_title(value: Any, fallback: str) -> str:
    text = str(value)
    if _CJK_FONT_NAME or all(ord(char) < 128 for char in text):
        return text
    return fallback


def _finite_record_value(record: Mapping[str, Any], key: str) -> float | None:
    return _number(record.get(key))


def _series(records: Sequence[Mapping[str, Any]], key: str) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    values: list[float] = []
    for fallback, record in enumerate(records):
        if not isinstance(record, Mapping) or not _record_executed(record):
            continue
        value = _finite_record_value(record, key)
        if value is None:
            # Preserve the x position while leaving the value as a gap.  NaN
            # tells Matplotlib to stop the line at missing data.
            values.append(float("nan"))
        else:
            values.append(value)
        steps.append(_record_step(record, fallback))
    return steps, values


def _termination_reason(records: Sequence[Mapping[str, Any]]) -> str | None:
    executed = [record for record in records if isinstance(record, Mapping) and _record_executed(record)]
    if not executed:
        return None
    final = executed[-1]
    for key in ("termination_reason", "termination", "reason", "end_reason"):
        value = final.get(key)
        if value not in (None, ""):
            return str(value)
    info = final.get("info")
    if isinstance(info, Mapping):
        for key, label in (
            ("arrive_dest", "arrive_dest"),
            ("out_of_road", "out_of_road"),
            ("crash_vehicle", "crash_vehicle"),
            ("crash_object", "crash_object"),
            ("crash", "crash"),
        ):
            if bool(info.get(key, False)):
                return label
    if bool(final.get("terminated", False)):
        return "terminated"
    if bool(final.get("truncated", False)):
        return "truncated"
    return None


def _reward_terms(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[str], dict[str, tuple[list[int], list[float]]], str | None, dict[str, int]]:
    keys: list[str] = []
    values_by_key: dict[str, tuple[list[int], list[float]]] = {}
    statuses: list[str] = []
    for record in records:
        if not isinstance(record, Mapping) or not _record_executed(record):
            continue
        details = record.get("reward_terms")
        status = (
            str(details.get("status", "unavailable")).casefold()
            if isinstance(details, Mapping)
            else "unavailable"
        )
        statuses.append(status)
        if status != "verified":
            continue
        terms = details.get("terms") if isinstance(details, Mapping) else None
        if not isinstance(terms, Mapping):
            continue
        for key in terms:
            text = str(key)
            if text not in keys:
                keys.append(text)
    for key in keys:
        steps: list[int] = []
        values: list[float] = []
        for fallback, record in enumerate(records):
            if not isinstance(record, Mapping) or not _record_executed(record):
                continue
            details = record.get("reward_terms")
            terms = details.get("terms") if isinstance(details, Mapping) else None
            verified = isinstance(details, Mapping) and str(details.get("status", "")).casefold() == "verified"
            value = terms.get(key) if isinstance(terms, Mapping) and verified else None
            steps.append(_record_step(record, fallback))
            values.append(float("nan") if _number(value) is None else float(_number(value)))
        values_by_key[key] = (steps, values)
    status_counts: dict[str, int] = {}
    for status in statuses:
        status_counts[status] = status_counts.get(status, 0) + 1
    status: str | None = None
    has_verified = status_counts.get("verified", 0) > 0
    non_verified_count = sum(
        count for key, count in status_counts.items() if key != "verified"
    )
    if has_verified and non_verified_count:
        # Preserve verified lines but make the telemetry boundary visible.
        status = "mixed"
    elif keys:
        status = "verified"
    elif statuses:
        # Retain the strongest reason for an empty line annotation.
        status = (
            "mismatch"
            if any(value in {"mismatch", "nonfinite", "malformed"} for value in statuses)
            else "unavailable"
        )
    return keys, values_by_key, status, status_counts


def render_rewards_plot(
    baseline_records: Iterable[Mapping[str, Any]],
    output_path: str | Path,
    *,
    changed_records: Iterable[Mapping[str, Any]] | None = None,
    title: str = "Rewards",
    baseline_label: str = "baseline",
    changed_label: str = "intervention",
    dpi: int = 140,
) -> VisualResult:
    """Render one reward axis with gaps for missing rewards or terms.

    ``changed_records=None`` produces the baseline-only chart.  In the
    comparison chart the baseline line is dashed and the intervention line is
    thick.  Verified reward terms are thin lines; unavailable or mismatched
    terms are never replaced with zero.
    """

    destination = Path(output_path)
    baseline = [record for record in baseline_records if isinstance(record, Mapping)]
    changed = (
        [record for record in changed_records if isinstance(record, Mapping)]
        if changed_records is not None
        else None
    )
    errors: list[str] = []
    try:
        pyplot = _matplotlib()
        figure, axis = pyplot.subplots(figsize=(9.5, 4.8))
        bx, by = _series(baseline, "reward")
        if bx:
            axis.plot(bx, by, linestyle="--", linewidth=1.6, color="#4c566a", label=baseline_label)
        if changed is not None:
            cx, cy = _series(changed, "reward")
            if cx:
                axis.plot(cx, cy, linestyle="-", linewidth=2.4, color="#bf616a", label=changed_label)
        term_records = changed if changed is not None else baseline
        term_keys, term_series, term_status, term_status_counts = _reward_terms(term_records)
        term_colors = ("#5e81ac", "#a3be8c", "#b48ead", "#d08770", "#8fbcbb", "#ebcb8b")
        for index, key in enumerate(term_keys):
            tx, ty = term_series[key]
            axis.plot(
                tx,
                ty,
                linewidth=0.95,
                color=term_colors[index % len(term_colors)],
                label=f"term: {key}",
            )
        if term_status == "mixed":
            counts_text = ", ".join(
                f"{key}={count}" for key, count in sorted(term_status_counts.items())
            )
            annotation = _plot_label(
                f"内訳混在（{counts_text}、verifiedのみ表示）",
                f"reward terms mixed ({counts_text}); only verified terms shown",
            )
            axis.text(
                0.995,
                0.98,
                annotation,
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=10,
                color="#a94442",
            )
        elif not term_keys:
            annotation = (
                _plot_label("内訳未接続", "reward terms unavailable")
                if term_status in {None, "unavailable"}
                else _plot_label("内訳不一致（線を非表示）", "reward terms mismatch (hidden)")
            )
            axis.text(
                0.995,
                0.98,
                annotation,
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                color="#a94442",
            )
        axis.axhline(0.0, color="#777777", linewidth=0.7, alpha=0.8)
        terminal_records = changed if changed is not None else baseline
        terminal_reason = _termination_reason(terminal_records)
        executed = [
            record
            for record in terminal_records
            if isinstance(record, Mapping) and _record_executed(record)
        ]
        if executed:
            terminal_step = _record_step(executed[-1], len(executed) - 1)
            label = f"{_plot_label('終端', 'end')}: {terminal_reason or 'unknown'}"
            axis.axvline(terminal_step, color="#d08770", linewidth=0.8, alpha=0.8, label=label)
        axis.set_xlabel("action step t (0-based)", fontsize=15, labelpad=8)
        axis.set_ylabel("returned reward r(t+1)", fontsize=15, labelpad=8)
        axis.set_title(_plot_title(title, "Rewards"), fontsize=16, pad=10)
        axis.tick_params(axis="both", labelsize=14)
        axis.grid(True, alpha=0.22)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(loc="best", fontsize=12)
        else:
            axis.text(0.5, 0.5, "reward unavailable", transform=axis.transAxes, ha="center", va="center")
            axis.set_xlim(-0.5, 0.5)
            axis.set_ylim(-1.0, 1.0)
        figure.tight_layout()
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, format="png", dpi=dpi)
        pyplot.close(figure)
    except (VisualError, OSError, ValueError, RuntimeError, TypeError) as exc:
        errors.append(str(exc))
        return VisualResult(None, "failed", {"terms_status": None}, tuple(errors))
    return VisualResult(
        destination,
        "complete",
        {
            "terms_status": term_status,
            "terms_status_counts": term_status_counts,
            "baseline_step_count": len(bx),
            "changed_step_count": len(cx) if changed is not None else None,
        },
        tuple(errors),
    )


def render_reward_plot(*args: Any, **kwargs: Any) -> VisualResult:
    """Backward-compatible singular alias for :func:`render_rewards_plot`."""

    return render_rewards_plot(*args, **kwargs)


def _js_series(records: Sequence[Mapping[str, Any]]) -> tuple[list[int], list[float], list[tuple[int, float]]]:
    steps: list[int] = []
    values: list[float] = []
    changed_points: list[tuple[int, float]] = []
    for fallback, record in enumerate(records):
        if not isinstance(record, Mapping):
            continue
        step = _record_step(record, fallback)
        value = _number(record.get("js"))
        steps.append(step)
        values.append(float("nan") if value is None else value)
        if value is not None and record.get("argmax_changed") is True:
            changed_points.append((step, value))
    return steps, values, changed_points


def render_policy_change_plot(
    records: Iterable[Mapping[str, Any]],
    output_path: str | Path,
    *,
    title: str = "Policy change",
    ylim: tuple[float, float] = (0.0, LN2),
    xlim: tuple[float, float] | None = None,
    dpi: int = 140,
) -> VisualResult:
    """Render recorded JS points and same-time action-change markers."""

    destination = Path(output_path)
    rows = [record for record in records if isinstance(record, Mapping)]
    errors: list[str] = []
    try:
        low = _number(ylim[0])
        high = _number(ylim[1])
        if low is None or high is None or high <= low:
            raise VisualError(f"invalid JS axis range: {ylim!r}")
        pyplot = _matplotlib()
        figure, axis = pyplot.subplots(figsize=(9.5, 4.8))
        steps, values, changed_points = _js_series(rows)
        if steps:
            axis.scatter(
                steps,
                values,
                s=20,
                color="#5e81ac",
                marker="o",
                linewidths=0,
                zorder=2,
                clip_on=False,
                label=_plot_label(
                    "JS divergence（記録点、nats）",
                    "JS divergence (recorded points, nats)",
                ),
            )
        if changed_points:
            axis.scatter(
                [point[0] for point in changed_points],
                [point[1] for point in changed_points],
                s=25,
                color="#bf616a",
                marker="o",
                zorder=4,
                clip_on=False,
                label=_plot_label(
                    "行動が変化（入力変更前後）",
                    "Action changed by input modification",
                ),
            )
        finite = [value for value in values if math.isfinite(value)]
        mean = math.fsum(finite) / len(finite) if finite else None
        maximum = max(finite) if finite else None
        axis.text(
            0.995,
            0.98,
            (
                f"mean={mean:.4g}, max={maximum:.4g}, action_changed={len(changed_points)}, valid={len(finite)}"
                if mean is not None and maximum is not None
                else f"JS unavailable; action_changed={len(changed_points)}, valid=0; missing steps are not zero"
            ),
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            color="#4c566a",
        )
        axis.set_ylim(low, high)
        if xlim is not None:
            x_low = _number(xlim[0])
            x_high = _number(xlim[1])
            if x_low is None or x_high is None or x_high < x_low:
                raise VisualError(f"invalid JS x-axis range: {xlim!r}")
            axis.set_xlim(x_low, x_high if x_high > x_low else x_low + 1.0)
        axis.set_title(_plot_title(title, "Policy change"), fontsize=16, pad=10)
        axis.set_xlabel("baseline action step t (0-based)", fontsize=15, labelpad=8)
        axis.set_ylabel("JS divergence (nats)", fontsize=15, labelpad=8)
        axis.tick_params(axis="both", labelsize=14)
        axis.grid(True, alpha=0.22)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(loc="best", fontsize=12)
        else:
            axis.text(0.5, 0.5, "JS unavailable", transform=axis.transAxes, ha="center", va="center")
        figure.tight_layout()
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, format="png", dpi=dpi)
        pyplot.close(figure)
    except (VisualError, OSError, ValueError, RuntimeError, TypeError) as exc:
        errors.append(str(exc))
        return VisualResult(None, "failed", {"ylim": list(ylim)}, tuple(errors))
    return VisualResult(
        destination,
        "complete",
        {
            "ylim": [low, high],
            "xlim": list(xlim) if xlim is not None else None,
            "step_count": len(steps),
            "valid_js_count": len(finite),
            "argmax_changed_count": len(changed_points),
            "mean_js": mean,
            "max_js": maximum,
        },
        tuple(errors),
    )


def render_js_plot(*args: Any, **kwargs: Any) -> VisualResult:
    """Alias for :func:`render_policy_change_plot`."""

    return render_policy_change_plot(*args, **kwargs)


def annotate_frame(
    frame: Any,
    *,
    lines: Iterable[str],
    padding: int = 8,
    fill: tuple[int, int, int, int] = (0, 0, 0, 175),
    text_fill: tuple[int, int, int] = (255, 255, 255),
) -> Any:
    """Return an RGB frame with a small readable text overlay.

    This helper is for an adapter that already obtained a simulator frame.
    It does not inspect or mutate environment state.  When a font is not
    available Pillow's default bitmap font is sufficient for ASCII labels;
    Japanese explanatory text remains in the HTML report.
    """

    Image, ImageDraw, ImageFont = _require_pillow()
    if isinstance(frame, (str, Path)):
        image = _load_frame(Path(frame))
    else:
        try:
            image = frame.convert("RGB")
        except AttributeError as exc:
            raise VisualError("frame must be a Pillow image or path") from exc
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    # Keep an explicit TrueType fallback so the requested size also applies
    # on hosts without a CJK font (Pillow's bitmap default has a fixed size).
    font = ImageFont.load_default()
    # A 600px report card is the common display size.  Scale the overlay font
    # with the source frame so its action/reward lines remain readable there,
    # while tiny synthetic fixtures still receive a bounded font.
    font_size = max(4, min(20, int(round(image.width / 38.0))))
    # Pillow's bitmap default font covers ASCII only.  Prefer a system CJK
    # font when available; portability remains intact because the fallback is
    # still valid and the explanatory Japanese text is present in HTML.
    for candidate in (
        "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
        "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
        "/usr/share/fonts/opentype/unifont/unifont_jp.otf",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            if Path(candidate).is_file():
                font = ImageFont.truetype(candidate, size=font_size)
                break
        except (OSError, ValueError):
            continue
    text_lines = [str(line) for line in lines]
    if not text_lines:
        return image
    widths: list[int] = []
    heights: list[int] = []
    for line in text_lines:
        left, top, right, bottom = draw.textbbox((0, 0), line, font=font)
        widths.append(right - left)
        heights.append(bottom - top)
    box_height = sum(heights) + max(0, len(text_lines) - 1) * 2 + 2 * padding
    box_width = max(widths, default=0) + 2 * padding
    draw.rectangle((0, 0, box_width, box_height), fill=fill)
    y = padding
    for line, height in zip(text_lines, heights):
        draw.text((padding, y), line, font=font, fill=text_fill)
        y += height + 2
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


# Names used by small adapters in downstream ports.  Keeping these aliases
# here means the port only changes one import line when its local terminology
# differs.
render_frame_overlay = annotate_frame
make_gif = render_rollout_gif


__all__ = [
    "LN2",
    "VisualError",
    "VisualResult",
    "annotate_frame",
    "argmax_index",
    "js_divergence",
    "make_gif",
    "render_frame_overlay",
    "render_js_plot",
    "render_policy_change_plot",
    "render_reward_plot",
    "render_rewards_plot",
    "render_rollout_gif",
    "validate_probability_distribution",
]
