"""Logical artifact layout for flat and numbered attribution results.

The analysis writers intentionally continue to write a flat staging tree.  This
module only describes the published tree and resolves the old flat names for
readers.  It has no MetaDrive or model dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final


EXPERIMENT_01: Final[str] = "experiment_01_perturbation"
EXPERIMENT_02: Final[str] = "experiment_02_integrated_gradients"
EXPERIMENT_03: Final[str] = "experiment_03_closed_loop"
SHARED: Final[str] = "shared"
EXPERIMENT_IDS: Final[tuple[str, ...]] = (EXPERIMENT_01, EXPERIMENT_02, EXPERIMENT_03)

SHARED_FILES: Final[frozenset[str]] = frozenset(
    {
        "analysis_metadata.json",
        "feature_schema_expanded.csv",
        "rollout_arrays.npz",
        "rollout_steps.jsonl",
        "rollout_metadata.json",
    }
)
EXPERIMENT_FILES: Final[dict[str, frozenset[str]]] = {
    EXPERIMENT_01: frozenset(
        {
            "perturbation_feature_steps.npz",
            "perturbation_feature_summary.csv",
            "perturbation_group_summary.csv",
        }
    ),
    EXPERIMENT_02: frozenset(
        {
            "ig_attributions.npz",
            "ig_feature_summary.csv",
            "ig_group_summary.csv",
            "ig_completeness.csv",
        }
    ),
    EXPERIMENT_03: frozenset(
        {
            "closed_loop_runs.jsonl",
            "closed_loop_summary.csv",
        }
    ),
}
PLOT_EXPERIMENT: Final[dict[str, str]] = {
    "perturbation_feature_top.png": EXPERIMENT_01,
    "perturbation_group_top.png": EXPERIMENT_01,
    "perturbation_over_time.png": EXPERIMENT_01,
    "lidar_perturbation_heatmap.png": EXPERIMENT_01,
    "ig_feature_top_absolute.png": EXPERIMENT_02,
    "ig_group_top_absolute.png": EXPERIMENT_02,
    "ig_group_signed.png": EXPERIMENT_02,
    "attribution_over_time.png": EXPERIMENT_02,
    "lidar_ig_heatmap.png": EXPERIMENT_02,
    "custom_feature_attribution_over_time.png": EXPERIMENT_02,
    "closed_loop_performance.png": EXPERIMENT_03,
}


class ArtifactLayoutError(ValueError):
    """Raised when a result tree is missing, ambiguous, or already compact."""


@dataclass(frozen=True, slots=True)
class ResultLayout:
    """Detected raw result layout.

    ``kind`` is ``flat`` for the legacy tree and ``numbered`` for the new tree.
    Compact trees are intentionally rejected by :func:`detect_layout` unless a
    caller explicitly requests ``allow_compact`` for diagnostics.
    """

    root: Path
    kind: str

    @property
    def is_flat(self) -> bool:
        return self.kind == "flat"

    @property
    def is_numbered(self) -> bool:
        return self.kind == "numbered"

    def directory(self, experiment_id: str) -> Path:
        if experiment_id == SHARED:
            return self.root / SHARED
        if experiment_id not in EXPERIMENT_IDS:
            raise ArtifactLayoutError(f"unknown experiment directory: {experiment_id}")
        return self.root / experiment_id


def _regular_root(root: Path) -> Path:
    candidate = Path(root).expanduser()
    if candidate.is_symlink() or not candidate.is_dir():
        raise ArtifactLayoutError(f"result root must be a regular directory: {root}")
    return candidate.resolve()


def _has_nested_zip(root: Path) -> bool:
    return any(path.name == "details.zip" for path in root.rglob("details.zip"))


def _has_flat_marker(root: Path) -> bool:
    # A numbered run also has a root ``report.md``.  Do not use that report as
    # the flat marker: direct raw files are the unambiguous legacy signal.
    raw_files = set(SHARED_FILES)
    for names in EXPERIMENT_FILES.values():
        raw_files.update(names)
    return any((root / name).is_file() for name in raw_files) or any(
        (root / "plots" / name).is_file() for name in PLOT_EXPERIMENT
    )


def _has_numbered_marker(root: Path) -> bool:
    shared = root / SHARED
    if shared.is_symlink():
        raise ArtifactLayoutError(f"numbered artifact directory is an unsafe symlink: {shared}")
    return shared.is_dir() and any(
        (shared / name).is_file() for name in SHARED_FILES
    )


def detect_layout(root: Path, *, allow_compact: bool = False) -> ResultLayout:
    """Detect flat/numbered full results and reject compact trees by default."""

    directory = _regular_root(root)
    for name in (SHARED, *EXPERIMENT_IDS):
        child = directory / name
        if child.is_symlink():
            raise ArtifactLayoutError(f"numbered artifact directory is an unsafe symlink: {child}")
    compact = _has_nested_zip(directory)
    flat = _has_flat_marker(directory)
    numbered = _has_numbered_marker(directory)
    # A caller may have extracted all section archives into one new raw run
    # root.  The raw markers take precedence over the archive files left next
    # to them; such a tree is readable as numbered full data.  A compact tree
    # has archives but no raw marker and is still rejected below.
    if compact and (flat or numbered):
        compact = False
    if compact and not allow_compact:
        raise ArtifactLayoutError(
            "result is compact and contains details.zip; extract the full raw result "
            "to a new directory before reading or regrouping it"
        )
    if flat and numbered:
        raise ArtifactLayoutError(
            "result mixes flat and numbered artifacts; keep one complete layout"
        )
    if compact:
        return ResultLayout(directory, "compact")
    if flat:
        return ResultLayout(directory, "flat")
    if numbered:
        return ResultLayout(directory, "numbered")
    raise ArtifactLayoutError(
        "result has neither legacy flat artifacts nor shared numbered artifacts"
    )


def _normal_relative(value: str | Path) -> str:
    path = PurePosixPath(str(value).replace("\\", "/"))
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ArtifactLayoutError(f"artifact name must be relative and non-traversing: {value!r}")
    return path.as_posix()


def _numbered_relative(legacy: str) -> str:
    path = PurePosixPath(legacy)
    if legacy == "report.md":
        return legacy
    if legacy in SHARED_FILES:
        return f"{SHARED}/{legacy}"
    for experiment_id, names in EXPERIMENT_FILES.items():
        if legacy in names:
            return f"{experiment_id}/{legacy}"
    if len(path.parts) == 2 and path.parts[0] == "plots":
        experiment_id = PLOT_EXPERIMENT.get(path.name)
        if experiment_id is not None:
            return f"{experiment_id}/plots/{path.name}"
    # Unknown legacy files are preserved under shared.  A caller that needs a
    # second location must use a canonical numbered path explicitly.
    return f"{SHARED}/{legacy}"


def legacy_relative_for_canonical(canonical_relative_path: str | Path) -> str:
    """Map a canonical numbered path back to the legacy reader name.

    This is intentionally conservative: only names that have a stable owner
    in :data:`EXPERIMENT_FILES` or :data:`PLOT_EXPERIMENT` are mapped back.
    Unknown files remain addressable by their canonical path.
    """

    canonical = _normal_relative(canonical_relative_path)
    parts = PurePosixPath(canonical).parts
    if len(parts) == 2 and parts[0] == SHARED:
        return parts[1]
    if len(parts) == 2 and parts[0] in EXPERIMENT_IDS:
        return parts[1]
    if len(parts) == 3 and parts[0] in EXPERIMENT_IDS and parts[1] == "plots":
        return f"plots/{parts[2]}"
    return canonical


def resolve_artifact(root: Path, legacy_relative_path: str | Path) -> Path:
    """Resolve a legacy flat artifact name in flat or numbered full results.

    Known names use the canonical experiment map.  Unknown names are read from
    ``shared`` in numbered results, preserving unclassified artifacts there.
    A root containing both layouts is rejected rather than silently preferring
    one copy.
    """

    layout = detect_layout(root)
    legacy = _normal_relative(legacy_relative_path)
    if layout.is_flat:
        candidate = layout.root / legacy
        _reject_path_symlinks(layout.root, candidate)
        return candidate
    if "/" in legacy:
        # A caller may already pass a canonical numbered path.  It must stay
        # inside the root and must not be remapped a second time.
        candidate = layout.root / legacy
        if candidate.exists():
            _reject_path_symlinks(layout.root, candidate)
            return candidate
    candidate = layout.root / _numbered_relative(legacy)
    _reject_path_symlinks(layout.root, candidate)
    return candidate


def _reject_path_symlinks(root: Path, candidate: Path) -> None:
    """Reject symlinks in every canonical path component, including leaves."""

    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise ArtifactLayoutError(f"artifact path escaped result root: {candidate}") from error
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ArtifactLayoutError(f"artifact path contains an unsafe symlink: {current}")


def canonical_relative_path(legacy_relative_path: str | Path) -> str:
    """Return the run-root-relative path used in numbered archives."""

    legacy = _normal_relative(legacy_relative_path)
    return _numbered_relative(legacy)


def experiment_for_legacy(legacy_relative_path: str | Path) -> str:
    """Return the numbered owner for a legacy artifact name."""

    canonical = canonical_relative_path(legacy_relative_path)
    first = canonical.split("/", 1)[0]
    return first if first in EXPERIMENT_IDS else SHARED
