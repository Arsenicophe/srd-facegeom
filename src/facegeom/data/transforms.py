"""Prétraitement commun aux backbones : recadrage → redimensionnement → normalisation ImageNet."""

from __future__ import annotations

from functools import partial

from torchvision import transforms as T

from facegeom.data.crop import crop_box, crop_face


def build_transform(crop_cfg: dict, resolution: int, normalize: dict) -> T.Compose:
    """PIL → tenseur ``3 × resolution × resolution``. Aucune augmentation."""
    return T.Compose([
        partial(crop_face, box=crop_box(crop_cfg)),  # picklable, contrairement à une lambda
        T.Resize((resolution, resolution), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
        T.ToTensor(),
        T.Normalize(mean=normalize["mean"], std=normalize["std"]),
    ])
