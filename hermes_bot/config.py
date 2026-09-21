"""Config-loading uit config.yaml + .env (alleen secrets in .env)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Lees config.yaml. Secrets komen UIT ENV (nooit uit yaml)."""
    p = path or ROOT / "config" / "config.yaml"
    with p.open() as f:
        return yaml.safe_load(f) or {}


def get_secret(name: str, default: str | None = None) -> str | None:
    """Secrets komen uit de omgeving, nooit hardcoded."""
    return os.environ.get(name, default)
