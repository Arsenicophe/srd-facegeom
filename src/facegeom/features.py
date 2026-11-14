"""Vecteurs extraits : calcul par morceaux avec reprise, stockage et lecture.

Disposition : ``<racine>/<backbone>/<résolution>/``
- ``<split>.npy``         (N, 2048), lignes dans l'ordre de ``<split>_images.txt`` ;
- ``<split>_images.txt``  noms des images ;
- ``meta.json``           configuration, versions, révision ;
- ``parts/<split>/``      morceaux temporaires, supprimés une fois assemblés.
"""

from __future__ import annotations

import json
import os
import shutil
import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from facegeom.data.images import ZIP_PREFIX

DIM = 2048


# --- Lecture ----------------------------------------------------------------


def feature_dir(root: str | Path, backbone: str, resolution: int) -> Path:
    return Path(root) / backbone / str(resolution)


def load_features(root: str | Path, backbone: str, resolution: int, split: str) -> tuple[np.ndarray, list[str]]:
    """Vecteurs (memmap) et noms d'images d'une partie, dans le même ordre."""
    d = feature_dir(root, backbone, resolution)
    feats = np.load(d / f"{split}.npy", mmap_mode="r")
    names = (d / f"{split}_images.txt").read_text().split()
    if len(names) != feats.shape[0]:
        raise ValueError(f"{d}/{split} : {feats.shape[0]} vecteurs pour {len(names)} images.")
    return feats, names


def load_xy(
    root: str | Path, backbone: str, resolution: int, split: str, split_dir: str | Path = "splits"
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Vecteurs (float32 en RAM), identités et noms d'images d'une partie, alignés."""
    feats, names = load_features(root, backbone, resolution, split)
    ids = pd.read_csv(Path(split_dir) / f"{split}.csv").set_index("image").identity
    return np.asarray(feats, dtype=np.float32), ids.loc[names].to_numpy(), names


# --- Extraction -------------------------------------------------------------


class CelebaImages(Dataset):
    """Images d'une liste de noms (dossier ou zip), transformées à chaque résolution."""

    def __init__(self, names: list[str], source: str | Path, transforms: dict[int, Callable]):
        self.names = list(names)
        self.source = Path(source)
        self.transforms = transforms
        self._zip = None  # ouvert à la demande, un par processus

    def __len__(self) -> int:
        return len(self.names)

    def _open(self, name: str) -> Image.Image:
        if self.source.suffix == ".zip":
            self._zip = self._zip or zipfile.ZipFile(self.source)
            with self._zip.open(ZIP_PREFIX + name) as f:
                return Image.open(f).convert("RGB")
        return Image.open(self.source / name).convert("RGB")

    def __getitem__(self, i: int):
        img = self._open(self.names[i])
        return i, {res: tf(img) for res, tf in self.transforms.items()}


def _save_atomic(path: Path, array: np.ndarray) -> None:
    """Fichier temporaire puis renommage : jamais de fichier à moitié écrit."""
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, array)
    os.replace(tmp, path)


def _is_complete(out_dir: Path, split: str, names: list[str]) -> bool:
    images, feats = out_dir / f"{split}_images.txt", out_dir / f"{split}.npy"
    return (
        images.exists()
        and feats.exists()
        and images.read_text().split() == list(names)
        and np.load(feats, mmap_mode="r").shape == (len(names), DIM)
    )


def _assemble(split: str, names: list[str], part_dir: Path, n_chunks: int, out_dir: Path) -> None:
    feats = np.concatenate([np.load(part_dir / f"part_{c:05d}.npy") for c in range(n_chunks)])
    if feats.shape != (len(names), DIM):
        raise ValueError(f"{out_dir}/{split} : forme {feats.shape}, attendu ({len(names)}, {DIM}).")
    _save_atomic(out_dir / f"{split}.npy", feats)
    (out_dir / f"{split}_images.txt").write_text("\n".join(names) + "\n")
    shutil.rmtree(part_dir)


def extract_split(
    split: str,
    names: list[str],
    source: str | Path,
    models: dict[str, nn.Module],
    transforms: dict[int, Callable],
    out_root: str | Path,
    chunk_size: int,
    batch_size: int,
    num_workers: int,
    device: str,
    storage_dtype: str = "float16",
    amp: bool = False,
    log: Callable[[str], None] = print,
) -> None:
    """Extrait une partie pour tous les couples (backbone, résolution).

    Chaque image est décodée une fois. Un morceau fini est écrit aussitôt ; une
    relance saute les morceaux présents, puis assemble le tout.
    """
    outputs = [(b, r) for b in models for r in transforms]
    dirs = {o: feature_dir(out_root, *o) for o in outputs}
    if all(_is_complete(dirs[o], split, names) for o in outputs):
        log(f"{split} : déjà terminé pour tous les backbones et résolutions")
        return
    part_dirs = {o: dirs[o] / "parts" / split for o in outputs}
    for d in part_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    n_chunks = (len(names) + chunk_size - 1) // chunk_size
    for c in range(n_chunks):
        part = f"part_{c:05d}.npy"
        if all((part_dirs[o] / part).exists() for o in outputs):
            log(f"{split} morceau {c + 1}/{n_chunks} : déjà fait")
            continue
        lo, hi = c * chunk_size, min((c + 1) * chunk_size, len(names))
        loader = DataLoader(CelebaImages(names[lo:hi], source, transforms), batch_size=batch_size,
                            num_workers=num_workers)
        buffers = {o: np.empty((hi - lo, DIM), dtype=storage_dtype) for o in outputs}
        with torch.no_grad(), torch.autocast(device_type=device.split(":")[0], enabled=amp):
            for idx, xs in loader:
                for res, x in xs.items():
                    x = x.to(device, non_blocking=True)
                    for b, model in models.items():
                        f = model(x).float().cpu().numpy()
                        if not np.isfinite(f).all():
                            raise FloatingPointError(f"{split}/{b}/{res} : valeur non finie.")
                        buffers[(b, res)][idx.numpy()] = f.astype(storage_dtype)
        for o in outputs:
            _save_atomic(part_dirs[o] / part, buffers[o])
        log(f"{split} morceau {c + 1}/{n_chunks} : {hi - lo} images écrites")

    for o in outputs:
        _assemble(split, names, part_dirs[o], n_chunks, dirs[o])


def write_meta(out_dir: Path, meta: dict) -> None:
    """Ajoute un passage à ``meta.json`` sans effacer les précédents."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "meta.json"
    history = json.loads(path.read_text())["runs"] if path.exists() else []
    path.write_text(json.dumps({"runs": [*history, meta]}, indent=2, ensure_ascii=False) + "\n")
