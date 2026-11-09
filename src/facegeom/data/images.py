"""Lecture des images CelebA directement dans le zip.

Accepte un chemin local (zip copié sur Colab ou sur Drive) ou un chemin
``hf://`` vers le zip sur Hugging Face : dans ce cas seules les images lues
sont téléchargées, ce qui permet de travailler en local sans le zip complet.
"""

from __future__ import annotations

import io
import shutil
import zipfile
from pathlib import Path

from PIL import Image

ZIP_PREFIX = "img_align_celeba/"


def open_celeba_zip(path: str) -> zipfile.ZipFile:
    """Ouvre le zip d'images, local ou ``hf://datasets/<dépôt>@<révision>/<fichier>``."""
    if path.startswith("hf://"):
        from huggingface_hub import HfFileSystem

        return zipfile.ZipFile(HfFileSystem().open(path[len("hf://"):], "rb"))
    return zipfile.ZipFile(path)


def ensure_local_images(zip_path: str | Path, dest: str | Path) -> Path:
    """Copie et décompresse le zip sur le disque local (Colab) s'il ne l'est pas déjà.

    Renvoie le dossier des images. Décompresser directement sur Drive serait
    très lent ; on copie d'abord le zip sur le disque local.
    """
    dest = Path(dest)
    img_dir = dest / ZIP_PREFIX.rstrip("/")
    if img_dir.exists() and any(img_dir.iterdir()):
        return img_dir
    dest.mkdir(parents=True, exist_ok=True)
    local_zip = dest / Path(zip_path).name
    shutil.copyfile(zip_path, local_zip)
    with zipfile.ZipFile(local_zip) as z:
        z.extractall(dest)
    local_zip.unlink()
    return img_dir


def read_image(zf: zipfile.ZipFile, name: str) -> Image.Image:
    """Lit l'image ``name`` (par exemple ``000001.jpg``) en RVB."""
    with zf.open(ZIP_PREFIX + name) as f:
        return Image.open(io.BytesIO(f.read())).convert("RGB")
