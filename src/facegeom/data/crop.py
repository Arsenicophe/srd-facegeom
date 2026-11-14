"""Recadrage carré fixe des images alignées de CelebA (178 × 218).

Les yeux sont toujours au même endroit : un même carré convient à toutes les
images. Il est placé pour que les yeux tombent à ``eye_rel`` de sa hauteur.
"""

from __future__ import annotations

from PIL import Image

IMG_W, IMG_H = 178, 218


def crop_box(crop_cfg: dict, size: int | None = None) -> tuple[int, int, int, int]:
    """Boîte ``(gauche, haut, droite, bas)`` d'un carré de côté ``size`` (défaut : ``crop_cfg["size"]``).

    ``crop_cfg`` est la section ``crop`` de ``configs/data.yaml``. Le carré est
    décalé si besoin pour rester dans l'image.
    """
    size = size or crop_cfg.get("size")
    if not size:
        raise ValueError("crop.size n'est pas fixé dans configs/data.yaml.")
    if size > min(IMG_W, IMG_H):
        raise ValueError(f"Carré de {size} px plus grand que l'image {IMG_W}×{IMG_H}.")
    left = min(max(round(crop_cfg["center_x"] - size / 2), 0), IMG_W - size)
    top = min(max(round(crop_cfg["eye_y"] - crop_cfg["eye_rel"] * size), 0), IMG_H - size)
    return left, top, left + size, top + size


def crop_face(img: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    """Découpe ``box`` dans ``img`` après avoir vérifié sa taille."""
    if img.size != (IMG_W, IMG_H):
        raise ValueError(f"Image de taille {img.size}, {IMG_W}×{IMG_H} attendu (CelebA alignée).")
    return img.crop(box)
