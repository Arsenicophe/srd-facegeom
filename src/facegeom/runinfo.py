"""Informations consignées avec chaque résultat : versions, révision, matériel."""

from __future__ import annotations

import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PACKAGES = ["torch", "torchvision", "numpy", "scikit-learn", "pytorch-metric-learning"]


def git_revision(repo_dir: str | Path = ".") -> str:
    """Révision git courante, suffixée de « -dirty » si le dépôt a des modifications."""
    try:
        rev = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo_dir), "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "inconnue"


def run_info(device: str | None = None) -> dict:
    """Dictionnaire à écrire dans le fichier de métadonnées d'un résultat."""
    versions = {}
    for pkg in PACKAGES:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = None
    info = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_revision": git_revision(Path(__file__).resolve().parents[2]),
        "python": platform.python_version(),
        "versions": versions,
        "device": device,
    }
    if device and device.startswith("cuda"):
        import torch

        info["gpu"] = torch.cuda.get_device_name(0)
    return info
