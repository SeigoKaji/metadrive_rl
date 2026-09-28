"""Export the portable manifest to a separate review folder; never apply it."""
from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import zipfile


STAGING_DIRECTORY = "lookahead_learning_update"


def create_archive(output: str | Path, *, package_dir: Path | None = None) -> Path:
    """Create a new ZIP whose only top-level folder is the staging directory.

    Manifest paths describe the installed layout, lookahead_learning/.... Only
    that prefix is renamed in the archive. No installed file is overwritten.
    """
    package = (package_dir or Path(__file__).resolve().parent).resolve()
    sources: dict[str, Path] = {}
    for line in (package / "PORTABLE_FILES.txt").read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if not name or name.startswith("#"):
            continue
        relative = PurePosixPath(name)
        if (relative.is_absolute() or len(relative.parts) < 2
                or relative.parts[0] != "lookahead_learning"
                or ".." in relative.parts or "\\" in name
                or relative.as_posix() != name
                or relative.suffix not in {".py", ".md", ".toml", ".txt"}):
            raise ValueError(f"invalid portable path: {name}")
        source = package.joinpath(*relative.parts[1:])
        if not source.resolve().is_relative_to(package) or source.is_symlink():
            raise ValueError(f"portable source must stay in the package: {name}")
        if not source.is_file():
            raise ValueError(f"portable source missing: {name}")
        destination = str(PurePosixPath(STAGING_DIRECTORY, *relative.parts[1:]))
        if destination in sources:
            raise ValueError(f"duplicate portable path: {name}")
        sources[destination] = source
    if not sources:
        raise ValueError("portable manifest is empty")
    output = Path(output)
    # Exclusive creation also protects an existing delivery or model ZIP.
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, source in sorted(sources.items()):
            entry = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, source.read_bytes())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a separate lookahead update folder as a ZIP")
    parser.add_argument("--output", type=Path, required=True, help="New ZIP path; existing files are never overwritten")
    args = parser.parse_args()
    try:
        output = create_archive(args.output)
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    print(f"Created {output}; extract as {STAGING_DIRECTORY}/ and apply differences with Copilot.")


if __name__ == "__main__":
    main()
