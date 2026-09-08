"""入力帰属パッケージを別 PC へ渡すための再現可能な ZIP 作成。

ZIP に含めるのは ``input_attribution/`` 追加パッケージ内のコード、移植文書、
スキーマ/config のテンプレート、任意依存 requirements-ig、PORTABLE_FILES だけです。
学習済みモデル、実験結果、venv、キャッシュ、
フォント、第三者リポジトリはサイズや秘匿情報だけでなく再現性を損なうため、
明示的に除外します。各エントリの時刻・順序・属性を固定して、同じ入力から
同じバイト列を生成できるようにします。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Iterable, Sequence
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


class PackError(ValueError):
    """安全でない、またはポータブル対象外のパックを拒否する。"""


_EXCLUDED_PARTS = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", "env", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "models", "outputs", "logs", "cache", "caches", "fonts", "font", "reports",
    "third_party", "site-packages", "node_modules", "build", "dist",
})
_EXCLUDED_SUFFIXES = frozenset({
    ".pyc", ".pyo", ".so", ".dylib", ".dll", ".whl",
    ".zip", ".npz", ".npy", ".jsonl", ".gif", ".mp4", ".webm",
    ".png", ".jpg", ".jpeg", ".ttf", ".otf", ".woff", ".woff2", ".pt", ".pth",
})
_EXCLUDED_NAMES = frozenset({
    "report.html", "report.md", "report_manifest.json", "summary.csv",
    "manifest.json", "resolved_config.json", "status.json",
})
_PORTABLE_ROOTS = ("input_attribution",)


def _regular_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def _is_excluded(relative: PurePosixPath) -> bool:
    if any(part in _EXCLUDED_PARTS for part in relative.parts):
        return True
    if relative.name in _EXCLUDED_NAMES:
        return True
    if relative.suffix.lower() in _EXCLUDED_SUFFIXES:
        return True
    # Reports and generated runtime data can be stored below any section.
    lower = {part.lower() for part in relative.parts}
    if lower.intersection({"results", "result", "artifacts", "runs", "run"}):
        return True
    return False


def _relative(path: Path, root: Path) -> PurePosixPath | None:
    try:
        value = path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    relative = PurePosixPath(*value.parts)
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        return None
    return relative


def portable_files(source_root: str | os.PathLike[str]) -> tuple[Path, ...]:
    """収録対象を安全に解決し、ZIP 内パス順で返す。"""

    root = Path(source_root).expanduser()
    if not root.is_dir() or root.is_symlink():
        raise PackError(f"source root must be a regular directory: {root}")
    selected: dict[str, Path] = {}
    for directory_name in _PORTABLE_ROOTS:
        directory = root / directory_name
        if not directory.is_dir() or directory.is_symlink():
            continue
        for path in sorted(directory.rglob("*")):
            if not _regular_file(path):
                continue
            relative = _relative(path, root)
            if relative is None or _is_excluded(relative):
                continue
            # Runtime outputs under input_attribution are excluded by name and
            # suffix above; source code/docs are the only intended payload.
            selected[relative.as_posix()] = path
    return tuple(selected[key] for key in sorted(selected))


def portable_relative_paths(source_root: str | os.PathLike[str]) -> tuple[str, ...]:
    root = Path(source_root).expanduser()
    paths = []
    for path in portable_files(root):
        relative = _relative(path, root)
        if relative is not None:
            paths.append(relative.as_posix())
    return tuple(paths)


def _zip_entry_name(path: Path, root: Path) -> str:
    relative = _relative(path, root)
    if relative is None:
        raise PackError(f"file is outside source root or unsafe: {path}")
    name = relative.as_posix()
    if _is_excluded(relative):
        raise PackError(f"portable exclusion matched selected path: {name}")
    return name


def _zip_info(name: str, data: bytes) -> ZipInfo:
    # DOS ZIP timestamps cannot represent an epoch.  The earliest valid
    # timestamp is deterministic and does not encode the local filesystem time.
    info = ZipInfo(filename=name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    info.flag_bits |= 0x800  # UTF-8 names
    info.comment = b""
    info.extra = b""
    return info


def create_portable_zip(
    source_root: str | os.PathLike[str],
    output_zip: str | os.PathLike[str] | None = None,
) -> Path:
    """追加 package と docs だけを固定順・固定 timestamp で ZIP 化する。"""

    root = Path(source_root).expanduser().resolve()
    files = portable_files(root)
    destination = Path(output_zip).expanduser() if output_zip is not None else root / "input_attribution_portable.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.resolve() == root / "input_attribution_portable.zip":
        # The default name would otherwise be discovered on a future run; it
        # is already excluded by the .zip suffix, but this check is clearer.
        pass
    entries: list[tuple[str, bytes]] = []
    for path in files:
        name = _zip_entry_name(path, root)
        entries.append((name, path.read_bytes()))
    entries.sort(key=lambda item: item[0])
    # A temporary sibling avoids leaving a truncated archive after an I/O
    # failure.  Existing ZIP files are intentionally replaced as a generated
    # view; experiment directories themselves are never touched.
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=9, strict_timestamps=False) as archive:
            for name, data in entries:
                archive.writestr(_zip_info(name, data), data)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


build_portable_zip = create_portable_zip
make_portable_zip = create_portable_zip
create_portable_package = create_portable_zip
build_portable_package = create_portable_zip
make_reproducible_zip = create_portable_zip


def package_manifest(source_root: str | os.PathLike[str]) -> dict[str, object]:
    """ZIP に入るファイルと SHA256 を確認する（ZIP自体は含めない）。

    The scope field is deliberately explicit: unpacking this package adds an
    ``input_attribution`` directory and does not replace an external model,
    MetaDrive checkout, training config, or a port's existing 262-D producer.
    """

    root = Path(source_root).expanduser().resolve()
    files = portable_files(root)
    entries = []
    for path in files:
        relative = _zip_entry_name(path, root)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append({"path": relative, "sha256": digest, "bytes": path.stat().st_size})
    return {
        "source_root": str(root),
        "entry_count": len(entries),
        "entries": entries,
        "scope": "input_attribution_addon_only",
        "external_environment_policy": "never_overwrite_models_or_262d_observation_sources",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="input_attribution の移植用 ZIP を作成")
    parser.add_argument("--source-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--list", action="store_true", help="収録予定ファイルだけ表示")
    args = parser.parse_args(argv)
    if args.list:
        for name in portable_relative_paths(args.source_root):
            print(name)
        return 0
    output = create_portable_zip(args.source_root, args.output)
    print(json.dumps({"zip": str(output), **package_manifest(args.source_root)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
