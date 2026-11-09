"""Lecture des images CelebA directement dans le zip.

Accepte un chemin local (zip copié sur Colab ou sur Drive) ou un chemin
``hf://`` vers le zip sur Hugging Face : dans ce cas seules les images lues
sont téléchargées, ce qui permet de travailler en local sans le zip complet.
"""

from __future__ import annotations

import io
import zipfile

from PIL import Image

ZIP_PREFIX = "img_align_celeba/"


def open_celeba_zip(path: str) -> zipfile.ZipFile:
    """Ouvre le zip d'images, local ou ``hf://datasets/<dépôt>@<révision>/<fichier>``."""
    if path.startswith("hf://"):
        from huggingface_hub import HfFileSystem

        return zipfile.ZipFile(HfFileSystem().open(path[len("hf://"):], "rb"))
    return zipfile.ZipFile(path)


def read_image(zf: zipfile.ZipFile, name: str) -> Image.Image:
    """Lit l'image ``name`` (par exemple ``000001.jpg``) en RVB."""
    with zf.open(ZIP_PREFIX + name) as f:
        return Image.open(io.BytesIO(f.read())).convert("RGB")
