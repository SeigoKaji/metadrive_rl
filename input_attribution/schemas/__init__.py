"""Packaged JSON schema fixtures and unresolved porting metadata."""

from pathlib import Path


SCHEMA_DIR = Path(__file__).resolve().parent
OFFICIAL_259_PATH = SCHEMA_DIR / "official_259.json"
CUSTOM_262_TEMPLATE_PATH = SCHEMA_DIR / "custom_262_template.json"

__all__ = ["CUSTOM_262_TEMPLATE_PATH", "OFFICIAL_259_PATH", "SCHEMA_DIR"]
