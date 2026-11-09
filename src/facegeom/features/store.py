"""Emplacement et lecture des vecteurs extraits.

Disposition : ``<racine>/<backbone>/<résolution>/``
- ``<split>.npy``          tableau (N, 2048), lignes dans l'ordre de ``<split>_images.txt`` ;
- ``<split>_images.txt``   noms des images, une par ligne ;
- ``meta.json``            configuration, versions, révision du code ;
- ``parts/<split>/``       morceaux temporaires, supprimés une fois assemblés.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def feature_dir(root: str | Path, backbone: str, resolution: int) -> Path:
    return Path(root) / backbone / str(resolution)


def load_features(
    root: str | Path, backbone: str, resolution: int, split: str
) -> tuple[np.ndarray, list[str]]:
    """Relit les vecteurs d'une partie et la liste de leurs images, dans le même ordre."""
    d = feature_dir(root, backbone, resolution)
    feats = np.load(d / f"{split}.npy", mmap_mode="r")
    names = (d / f"{split}_images.txt").read_text().split()
    if len(names) != feats.shape[0]:
        raise ValueError(f"{d}/{split} : {feats.shape[0]} vecteurs pour {len(names)} images.")
    return feats, names
