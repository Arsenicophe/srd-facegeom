"""Outils communs : configuration, appareil, dossiers de résultats, traçabilité."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import yaml

PACKAGES = ["torch", "torchvision", "numpy", "scikit-learn", "pytorch-metric-learning"]
REPO_DIR = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict:
    """Lit un fichier YAML."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def default_device() -> str:
    """cuda (Colab), sinon mps (Mac), sinon cpu."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def file_hash(path: str | Path, algo: str = "md5") -> str:
    """Empreinte d'un fichier, lue par morceaux."""
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def new_run_dir(runs_dir: str | Path, name: str) -> Path:
    """Crée ``<runs_dir>/<date-heure>_<name>/`` ; échoue s'il existe déjà."""
    d = Path(runs_dir) / f"{datetime.now():%Y%m%d-%H%M%S}_{name}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def write_json(path: str | Path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def git_revision() -> str:
    """Révision git du dépôt, suffixée de « -dirty » s'il y a des modifications."""
    try:
        git = ["git", "-C", str(REPO_DIR)]
        rev = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        dirty = subprocess.run([*git, "status", "--porcelain"], capture_output=True, text=True, check=True)
        return rev.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "inconnue"


def run_info(device: str | None = None) -> dict:
    """Date, révision, versions et matériel, à écrire avec chaque résultat."""
    versions = {}
    for pkg in PACKAGES:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = None
    info = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_revision": git_revision(),
        "python": platform.python_version(),
        "versions": versions,
        "device": device,
    }
    if device and device.startswith("cuda"):
        import torch

        info["gpu"] = torch.cuda.get_device_name(0)
    return info
