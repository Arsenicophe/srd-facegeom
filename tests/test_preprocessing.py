"""Tests du recadrage et du prétraitement commun aux backbones."""

from pathlib import Path

import pytest
import torch
from PIL import Image

from facegeom.data.crop import crop_box, crop_face
from facegeom.data.transforms import build_transform
from facegeom.utils import load_config

ROOT = Path(__file__).resolve().parents[1]
CROP = load_config(ROOT / "configs" / "data.yaml")["crop"]
EXTRACT = load_config(ROOT / "configs" / "extract.yaml")


@pytest.mark.parametrize("size", [100, 128, 150, 178])
def test_crop_box_is_square_and_inside_image(size):
    l, t, r, b = crop_box(CROP, size)
    assert r - l == size and b - t == size
    assert 0 <= l and 0 <= t and r <= 178 and b <= 218


def test_chosen_crop_box():
    box = crop_box(CROP)
    assert box == (24, 52, 152, 180)
    # Les yeux tombent à ~46 % de la hauteur du carré
    assert abs((CROP["eye_y"] - box[1]) / CROP["size"] - CROP["eye_rel"]) < 0.01


def test_crop_rejects_wrong_image_size():
    with pytest.raises(ValueError):
        crop_face(Image.new("RGB", (200, 200)), (0, 0, 128, 128))


@pytest.mark.parametrize("res", EXTRACT["resolutions"])
def test_transform_output(res):
    tf = build_transform(CROP, res, EXTRACT["normalize"])
    img = Image.new("RGB", (178, 218), color=(124, 116, 104))  # ≈ moyenne ImageNet
    x = tf(img)
    assert x.shape == (3, res, res)
    assert x.dtype == torch.float32
    assert x.abs().max() < 0.05  # une image à la couleur moyenne est centrée sur 0


def test_transform_requires_fixed_crop_size():
    with pytest.raises(ValueError):
        build_transform({**CROP, "size": None}, 112, EXTRACT["normalize"])
