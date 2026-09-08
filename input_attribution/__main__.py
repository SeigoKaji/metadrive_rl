"""``python -m input_attribution`` entry point."""

from .cli import main


if __name__ == "__main__":  # pragma: no cover - exercised by subprocess smoke tests
    raise SystemExit(main())
