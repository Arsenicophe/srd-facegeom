"""Extraction des vecteurs, par morceaux, avec reprise après coupure.

Chaque image est décodée **une seule fois** ; on en tire une entrée par
résolution, puis chaque backbone traite chaque entrée. Les vecteurs d'un
morceau sont écrits dès qu'il est fini ; une relance saute les morceaux déjà
écrits. Quand tous les morceaux d'une partie existent, ils sont assemblés en un
seul fichier et vérifiés.
"""

from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from facegeom.data.images import ZIP_PREFIX
from facegeom.features.store import feature_dir


class CelebaImages(Dataset):
    """Images d'une liste de noms, chacune transformée à toutes les résolutions.

    ``source`` est un dossier d'images décompressées ou un fichier zip CelebA.
    Renvoie ``(indice, {résolution: tenseur})``.
    """

    def __init__(self, names: list[str], source: str | Path, transforms: dict[int, Callable]):
        self.names = list(names)
        self.source = Path(source)
        self.transforms = transforms
        self._zip = None  # ouvert à la demande, un par processus de chargement

    def __len__(self) -> int:
        return len(self.names)

    def _open(self, name: str) -> Image.Image:
        if self.source.suffix == ".zip":
            if self._zip is None:
                self._zip = zipfile.ZipFile(self.source)
            with self._zip.open(ZIP_PREFIX + name) as f:
                img = Image.open(f)
                img.load()
        else:
            img = Image.open(self.source / name)
            img.load()
        return img.convert("RGB")

    def __getitem__(self, i: int):
        img = self._open(self.names[i])
        return i, {res: tf(img) for res, tf in self.transforms.items()}


def _save_atomic(path: Path, array: np.ndarray) -> None:
    """Écrit dans un fichier temporaire puis renomme : jamais de fichier à moitié écrit."""
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, array)
    os.replace(tmp, path)


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
    """Extrait les vecteurs d'une partie pour tous les couples (backbone, résolution)."""
    out_root = Path(out_root)
    outputs = [(b, r) for b in models for r in transforms]
    if all(_is_complete(feature_dir(out_root, *o), split, names) for o in outputs):
        log(f"{split} : déjà terminé pour tous les backbones et résolutions")
        return
    part_dirs = {o: feature_dir(out_root, *o) / "parts" / split for o in outputs}
    for d in part_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    n_chunks = (len(names) + chunk_size - 1) // chunk_size
    for c in range(n_chunks):
        part_name = f"part_{c:05d}.npy"
        if all((part_dirs[o] / part_name).exists() for o in outputs):
            log(f"{split} morceau {c + 1}/{n_chunks} : déjà fait")
            continue

        lo, hi = c * chunk_size, min((c + 1) * chunk_size, len(names))
        ds = CelebaImages(names[lo:hi], source, transforms)
        loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        buffers = {o: np.empty((hi - lo, 2048), dtype=storage_dtype) for o in outputs}

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
            _save_atomic(part_dirs[o] / part_name, buffers[o])
        log(f"{split} morceau {c + 1}/{n_chunks} : {hi - lo} images écrites")

    for o in outputs:
        _assemble(split, names, part_dirs[o], n_chunks, feature_dir(out_root, *o))


def _is_complete(out_dir: Path, split: str, names: list[str]) -> bool:
    """Vrai si la partie est déjà assemblée pour exactement ces images."""
    images = out_dir / f"{split}_images.txt"
    feats = out_dir / f"{split}.npy"
    if not (images.exists() and feats.exists()):
        return False
    return images.read_text().split() == list(names) and np.load(feats, mmap_mode="r").shape == (
        len(names),
        2048,
    )


def _assemble(split: str, names: list[str], part_dir: Path, n_chunks: int, out_dir: Path) -> None:
    """Assemble les morceaux en ``<split>.npy``, vérifie le nombre, supprime les morceaux."""
    parts = [np.load(part_dir / f"part_{c:05d}.npy") for c in range(n_chunks)]
    feats = np.concatenate(parts)
    if feats.shape != (len(names), 2048):
        raise ValueError(f"{out_dir}/{split} : forme {feats.shape}, attendu ({len(names)}, 2048).")
    _save_atomic(out_dir / f"{split}.npy", feats)
    (out_dir / f"{split}_images.txt").write_text("\n".join(names) + "\n")
    shutil.rmtree(part_dir)


def write_meta(out_dir: Path, meta: dict) -> None:
    """Ajoute un passage à ``meta.json`` sans jamais effacer les précédents."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "meta.json"
    history = json.loads(path.read_text())["runs"] if path.exists() else []
    history.append(meta)
    path.write_text(json.dumps({"runs": history}, indent=2, ensure_ascii=False) + "\n")
