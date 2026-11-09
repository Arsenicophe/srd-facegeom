"""Dossiers de résultats : un par passage, jamais écrasé."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def new_run_dir(runs_dir: str | Path, name: str) -> Path:
    """Crée ``<runs_dir>/<date-heure>_<name>/`` ; échoue s'il existe déjà."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    d = Path(runs_dir) / f"{stamp}_{name}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def write_json(path: str | Path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
