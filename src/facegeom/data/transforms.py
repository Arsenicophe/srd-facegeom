"""Prétraitement des images, identique pour tous les backbones.

Chaîne : recadrage carré fixe → redimensionnement à la résolution d'entrée →
tenseur → normalisation ImageNet. Aucune augmentation : les vecteurs sont
extraits une seule fois (voir docs/protocole.md, section 4).
"""

from __future__ import annotations

from torchvision import transforms as T

from facegeom.data.crop import crop_box, crop_face


def build_transform(crop_cfg: dict, resolution: int, normalize: dict) -> T.Compose:
    """Renvoie la transformation PIL → tenseur ``3 × resolution × resolution``.

    ``crop_cfg`` est la section ``crop`` de ``configs/data.yaml`` (avec ``size``
    fixé), ``normalize`` contient ``mean`` et ``std``.
    """
    if not crop_cfg.get("size"):
        raise ValueError("crop.size n'est pas fixé dans configs/data.yaml.")
    box = crop_box(crop_cfg["size"], crop_cfg["center_x"], crop_cfg["eye_y"], crop_cfg["eye_rel"])
    return T.Compose(
        [
            T.Lambda(lambda img: crop_face(img, box)),
            T.Resize(
                (resolution, resolution),
                interpolation=T.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            T.ToTensor(),
            T.Normalize(mean=normalize["mean"], std=normalize["std"]),
        ]
    )
