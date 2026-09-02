"""Reference-observation selection for perturbation and Integrated Gradients."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import numpy as np

from .config import BaselineConfig, BaselineReference, BaselineStrategy
from .results import sha256_file


class BaselineError(ValueError):
    """Raised when a requested baseline cannot be resolved safely."""


def _readonly_copy(array: np.ndarray, *, dtype: np.dtype[np.floating] = np.dtype(np.float32)) -> np.ndarray:
    copied = np.array(array, dtype=dtype, copy=True)
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True, slots=True)
class ResolvedBaselines:
    """Per-sample baseline tensor and the IDs needed for reproducibility.

    ``values`` has shape ``(sample_count, baseline_count, observation_dim)``.
    The first axis lines up with the rollout observations passed to
    :class:`BaselineProvider`; the second axis is averaged only by the caller
    after preserving individual-baseline results.
    """

    values: np.ndarray
    strategy: str = "unknown"
    baseline_ids: np.ndarray | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        if values.ndim != 3 or values.shape[0] == 0 or values.shape[1] == 0 or values.shape[2] == 0:
            raise BaselineError(
                "baseline valuesは非空の(sample_count, baseline_count, observation_dim)配列で指定してください"
            )
        if not np.issubdtype(values.dtype, np.number) or not np.all(np.isfinite(values)):
            raise BaselineError("baseline valuesは有限の数値配列で指定してください")
        baseline_ids = self.baseline_ids
        if baseline_ids is None:
            baseline_ids = np.asarray(
                [
                    [f"baseline_{baseline_index}" for baseline_index in range(values.shape[1])]
                    for _sample_index in range(values.shape[0])
                ],
                dtype=str,
            )
        ids = np.asarray(baseline_ids)
        if ids.shape != values.shape[:2]:
            raise BaselineError(
                "baseline_idsは(sample_count, baseline_count)形状で指定してください: "
                f"expected {values.shape[:2]}, got {ids.shape}"
            )
        if not all(str(item).strip() for item in ids.flat):
            raise BaselineError("baseline_idsに空のIDは指定できません")
        object.__setattr__(self, "values", _readonly_copy(values))
        copied_ids = np.array(ids, dtype=str, copy=True)
        copied_ids.setflags(write=False)
        object.__setattr__(self, "baseline_ids", copied_ids)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def sample_count(self) -> int:
        """Number of rollout observations this object aligns with."""

        return int(self.values.shape[0])

    @property
    def baseline_count(self) -> int:
        """Number of baselines evaluated for each sample."""

        return int(self.values.shape[1])

    @property
    def observation_dim(self) -> int:
        """Feature dimension shared by baseline and rollout observations."""

        return int(self.values.shape[2])

    @property
    def baselines(self) -> np.ndarray:
        """Alias for ``values`` used by concise analysis code."""

        return self.values

    def for_sample(self, sample_index: int) -> np.ndarray:
        """Return a read-only ``(baseline_count, observation_dim)`` slice."""

        if isinstance(sample_index, bool) or not isinstance(sample_index, int):
            raise BaselineError("sample indexは整数で指定してください")
        if sample_index < 0 or sample_index >= self.sample_count:
            raise BaselineError(f"sample indexが範囲外です: {sample_index}")
        return self.values[sample_index]

    def copy_values(self) -> np.ndarray:
        """Return an independent writable copy for external serialization."""

        return self.values.copy()


class BaselineProvider:
    """Resolve configurable reference observations from a saved rollout.

    Inputs are copied at construction so later caller mutations cannot change
    the baseline selection.  Episode identifiers may be integer or string
    labels; decision steps must be non-negative integer-like values.
    """

    def __init__(
        self,
        observations: np.ndarray,
        episode_ids: np.ndarray | Sequence[object],
        steps: np.ndarray | Sequence[int],
    ) -> None:
        raw_observations = np.asarray(observations)
        if raw_observations.ndim != 2 or raw_observations.shape[0] == 0 or raw_observations.shape[1] == 0:
            raise BaselineError("observationsは非空の(sample_count, observation_dim)配列で指定してください")
        if not np.issubdtype(raw_observations.dtype, np.number) or not np.all(np.isfinite(raw_observations)):
            raise BaselineError("observationsは有限の数値配列で指定してください")
        episode_array = np.asarray(episode_ids)
        step_array = np.asarray(steps)
        if episode_array.ndim != 1 or episode_array.shape[0] != raw_observations.shape[0]:
            raise BaselineError("episode_idsはobservationsと同じsample数の1次元配列で指定してください")
        if step_array.ndim != 1 or step_array.shape[0] != raw_observations.shape[0]:
            raise BaselineError("stepsはobservationsと同じsample数の1次元配列で指定してください")
        if not np.issubdtype(step_array.dtype, np.integer) or np.any(step_array < 0):
            raise BaselineError("stepsは非負整数の1次元配列で指定してください")
        if any(not str(item).strip() for item in episode_array.flat):
            raise BaselineError("episode_idsに空の値は指定できません")
        self.observations = _readonly_copy(raw_observations)
        copied_episode_ids = np.array(episode_array, copy=True)
        copied_episode_ids.setflags(write=False)
        self.episode_ids = copied_episode_ids
        copied_steps = np.array(step_array, dtype=np.int64, copy=True)
        copied_steps.setflags(write=False)
        self.steps = copied_steps

    @property
    def sample_count(self) -> int:
        """Number of source rollout decisions."""

        return int(self.observations.shape[0])

    @property
    def observation_dim(self) -> int:
        """Dimension checked for every generated baseline source."""

        return int(self.observations.shape[1])

    def resolve(
        self,
        config: BaselineConfig | BaselineStrategy | str,
        *,
        count: int | None = None,
        seed: int | None = None,
        specified_steps: Sequence[BaselineReference] | None = None,
        external_npz_path: str | Path | None = None,
        external_npz_key: str | None = None,
    ) -> ResolvedBaselines:
        """Resolve a baseline strategy into a per-sample tensor.

        Passing :class:`~input_attribution.config.BaselineConfig` is the
        normal path.  Keyword arguments also make this class convenient for
        small offline scripts and unit tests without a TOML file.
        """

        if isinstance(config, BaselineConfig):
            if any(
                value is not None
                for value in (count, seed, specified_steps, external_npz_path, external_npz_key)
            ):
                raise BaselineError("BaselineConfigと個別のbaseline keywordは同時に指定できません")
            strategy = config.strategy
            count = config.count
            seed = config.seed
            specified_steps = config.specified_steps
            external_npz_path = config.external_npz_path
            external_npz_key = config.external_npz_key
        else:
            strategy = str(config)
            if count is None:
                count = 1
            if seed is None:
                seed = 0
        if strategy not in {
            "episode_start",
            "specified_steps",
            "sampled_observations",
            "external_npz",
        }:
            raise BaselineError(f"未対応のbaseline strategyです: {strategy}")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise BaselineError("baseline countは1以上の整数で指定してください")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise BaselineError("baseline seedは0以上の整数で指定してください")
        if strategy == "episode_start":
            return self.resolve_episode_start(count=count)
        if strategy == "specified_steps":
            if not specified_steps:
                raise BaselineError("specified_steps baselineにはreferenceを少なくとも1つ指定してください")
            return self.resolve_specified_steps(specified_steps, count=count)
        if strategy == "sampled_observations":
            return self.resolve_sampled_observations(count=count, seed=seed)
        if external_npz_path is None:
            raise BaselineError("external_npz baselineにはexternal_npz_pathが必要です")
        return self.resolve_external_npz(
            external_npz_path,
            key=external_npz_key or "observations",
            count=count,
            seed=seed,
        )

    def _broadcast_global(
        self,
        selected: np.ndarray,
        ids: Sequence[str],
        *,
        strategy: str,
        metadata: Mapping[str, object],
    ) -> ResolvedBaselines:
        selected_array = np.asarray(selected)
        if selected_array.ndim != 2 or selected_array.shape[1] != self.observation_dim:
            raise BaselineError(
                "baseline sourceは(B, observation_dim)である必要があります: "
                f"expected D={self.observation_dim}, got {selected_array.shape}"
            )
        if selected_array.shape[0] != len(ids):
            raise BaselineError("baseline source行数とID数が一致しません")
        values = np.broadcast_to(
            selected_array[np.newaxis, :, :],
            (self.sample_count, selected_array.shape[0], self.observation_dim),
        ).copy()
        baseline_ids = np.broadcast_to(
            np.asarray(ids, dtype=str)[np.newaxis, :],
            (self.sample_count, selected_array.shape[0]),
        ).copy()
        return ResolvedBaselines(
            values=values,
            strategy=strategy,
            baseline_ids=baseline_ids,
            metadata=metadata,
        )

    def _episode_start_indices(self) -> dict[object, int]:
        starts: dict[object, int] = {}
        for index, (episode_id, step) in enumerate(zip(self.episode_ids, self.steps, strict=True)):
            previous = starts.get(episode_id)
            if previous is None or step < self.steps[previous]:
                starts[episode_id] = index
        return starts

    def resolve_episode_start(self, *, count: int = 1) -> ResolvedBaselines:
        """Use the earliest saved observation of each sample's own episode."""

        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise BaselineError("baseline countは1以上の整数で指定してください")
        starts = self._episode_start_indices()
        values = np.empty((self.sample_count, count, self.observation_dim), dtype=np.float32)
        # ``np.empty(..., dtype=str)`` creates a U1 array and would truncate
        # reproducibility IDs to a single character (for example ``"e"``).
        # Build Python rows first so NumPy infers a Unicode width large enough
        # for every complete ID, including multi-digit episode identifiers.
        id_rows: list[list[str]] = []
        for index, episode_id in enumerate(self.episode_ids):
            source_index = starts[episode_id]
            values[index, :, :] = self.observations[source_index]
            base_id = f"episode_start:episode={episode_id}:step={int(self.steps[source_index])}"
            id_rows.append([base_id] * count)
        ids = np.asarray(id_rows, dtype=str)
        return ResolvedBaselines(
            values=values,
            strategy="episode_start",
            baseline_ids=ids,
            metadata={
                "strategy": "episode_start",
                "count": count,
                "episode_start_rows": {
                    str(episode): int(index) for episode, index in starts.items()
                },
            },
        )

    def resolve_specified_steps(
        self,
        references: Sequence[BaselineReference],
        *,
        count: int | None = None,
    ) -> ResolvedBaselines:
        """Use explicitly named rows from the saved rollout as global baselines."""

        if not references:
            raise BaselineError("specified_steps baselineにはreferenceを少なくとも1つ指定してください")
        requested_count = len(references) if count is None else count
        if isinstance(requested_count, bool) or not isinstance(requested_count, int) or requested_count <= 0:
            raise BaselineError("baseline countは1以上の整数で指定してください")
        if requested_count > len(references):
            raise BaselineError(
                "specified_stepsのreference数より大きいbaseline countは指定できません"
            )
        lookup: dict[tuple[object, int], int] = {}
        for index, (episode_id, step) in enumerate(zip(self.episode_ids, self.steps, strict=True)):
            key = (episode_id, int(step))
            if key in lookup:
                raise BaselineError(
                    "rollout内に同じepisode_id/stepが重複しています: "
                    f"episode={episode_id}, step={step}"
                )
            lookup[key] = index
        selected_indices: list[int] = []
        ids: list[str] = []
        for reference in references[:requested_count]:
            key = (reference.episode_id, reference.step)
            if key not in lookup:
                raise BaselineError(
                    "specified baselineがrolloutにありません: "
                    f"episode={reference.episode_id}, step={reference.step}"
                )
            selected_indices.append(lookup[key])
            ids.append(f"specified:episode={reference.episode_id}:step={reference.step}")
        return self._broadcast_global(
            self.observations[np.asarray(selected_indices, dtype=np.intp)],
            ids,
            strategy="specified_steps",
            metadata={
                "strategy": "specified_steps",
                "references": [
                    {"episode_id": reference.episode_id, "step": reference.step}
                    for reference in references[:requested_count]
                ],
            },
        )

    def resolve_sampled_observations(self, *, count: int, seed: int = 0) -> ResolvedBaselines:
        """Choose distinct rollout rows with a reproducible NumPy generator."""

        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise BaselineError("baseline countは1以上の整数で指定してください")
        if count > self.sample_count:
            raise BaselineError(
                "sampled_observations countは保存済み観測数以下で指定してください: "
                f"count={count}, observations={self.sample_count}"
            )
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise BaselineError("baseline seedは0以上の整数で指定してください")
        indices = np.random.default_rng(seed).choice(self.sample_count, size=count, replace=False)
        selected = self.observations[indices]
        ids = [f"sampled:row={int(index)}" for index in indices]
        return self._broadcast_global(
            selected,
            ids,
            strategy="sampled_observations",
            metadata={
                "strategy": "sampled_observations",
                "seed": seed,
                "source_rows": [int(index) for index in indices],
            },
        )

    def resolve_external_npz(
        self,
        path: str | Path,
        *,
        key: str = "observations",
        count: int = 1,
        seed: int = 0,
    ) -> ResolvedBaselines:
        """Load a 1-D or 2-D numeric baseline array from an NPZ without pickle."""

        source_path = Path(path).expanduser().resolve()
        if not source_path.is_file():
            raise BaselineError(f"external baseline NPZが見つかりません: {source_path}")
        if isinstance(key, str) is False or not key.strip():
            raise BaselineError("external NPZ keyは空でない文字列で指定してください")
        try:
            with np.load(source_path, allow_pickle=False) as archive:
                if key not in archive.files:
                    raise BaselineError(
                        f"external baseline NPZにkeyがありません: {key} "
                        f"(available: {', '.join(archive.files)})"
                    )
                raw = np.asarray(archive[key])
        except (OSError, ValueError) as error:
            if isinstance(error, BaselineError):
                raise
            raise BaselineError(f"external baseline NPZを読めません: {source_path}") from error
        if raw.ndim == 1:
            raw = raw[np.newaxis, :]
        if raw.ndim != 2 or raw.shape[1] != self.observation_dim:
            raise BaselineError(
                "external baselineの観測次元が一致しません: "
                f"expected D={self.observation_dim}, got {raw.shape}"
            )
        if not np.issubdtype(raw.dtype, np.number) or not np.all(np.isfinite(raw)):
            raise BaselineError("external baselineは有限の数値配列で指定してください")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise BaselineError("baseline countは1以上の整数で指定してください")
        if count > raw.shape[0]:
            raise BaselineError(
                "external baseline countはNPZ行数以下で指定してください: "
                f"count={count}, rows={raw.shape[0]}"
            )
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise BaselineError("baseline seedは0以上の整数で指定してください")
        indices = np.random.default_rng(seed).choice(raw.shape[0], size=count, replace=False)
        return self._broadcast_global(
            raw[indices],
            [f"external_npz:{source_path.name}:row={int(index)}" for index in indices],
            strategy="external_npz",
            metadata={
                "strategy": "external_npz",
                "path": str(source_path),
                "sha256": sha256_file(source_path),
                "key": key,
                "seed": seed,
                "source_shape": [int(raw.shape[0]), int(raw.shape[1])],
                "source_rows": [int(index) for index in indices],
            },
        )
