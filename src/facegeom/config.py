"""Lecture des fichiers de configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Charge un fichier de configuration YAML et le renvoie sous forme de dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
