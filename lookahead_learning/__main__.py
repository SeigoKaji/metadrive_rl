"""Executable entry point for ``python -B -m lookahead_learning``."""

from __future__ import annotations

from .runner import main


if __name__ == "__main__":  # pragma: no cover - exercised by subprocess
    raise SystemExit(main())

