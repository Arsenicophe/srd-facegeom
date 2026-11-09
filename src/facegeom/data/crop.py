"""Recadrage carré des images alignées de CelebA.

Toutes les images sont alignées (yeux toujours au même endroit), donc un même
carré fixe convient à toutes. Le carré est placé pour que les yeux tombent à la
même hauteur relative quelle que soit sa taille.
"""

from __future__ import annotations

from PIL import Image


def crop_box(
    size: int,
    center_x: float,
    eye_y: float,
    eye_rel: float,
    img_w: int = 178,
    img_h: int = 218,
) -> tuple[int, int, int, int]:
    """Renvoie la boîte ``(gauche, haut, droite, bas)`` d'un carré de côté ``size``.

    Le carré est centré horizontalement sur ``center_x`` et placé pour que les
    yeux (``eye_y``) soient à ``eye_rel`` de sa hauteur. Il est ensuite décalé
    si besoin pour rester entièrement dans l'image.
    """
    if size > min(img_w, img_h):
        raise ValueError(f"Carré de {size} px plus grand que l'image {img_w}×{img_h}.")
    left = round(center_x - size / 2)
    top = round(eye_y - eye_rel * size)
    left = min(max(left, 0), img_w - size)
    top = min(max(top, 0), img_h - size)
    return left, top, left + size, top + size


def crop_face(img: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    """Découpe ``box`` dans ``img`` après avoir vérifié la taille de l'image."""
    if img.size != (178, 218):
        raise ValueError(f"Image de taille {img.size}, 178×218 attendu (CelebA alignée).")
    return img.crop(box)
