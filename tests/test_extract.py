"""Tests de l'extraction : ordre, assemblage et reprise après coupure.

Utilise un petit réseau factice et des images synthétiques : rapide, sans
téléchargement.
"""

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from facegeom.features import extract_split, feature_dir, load_features

N_IMAGES = 10
CHUNK = 4  # 3 morceaux : 4 + 4 + 2


class Fake(nn.Module):
    """Réseau factice : sortie 2048 qui dépend de l'image et du réseau."""

    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        m = x.mean(dim=(1, 2, 3), keepdim=False)[:, None]
        return self.scale * m * torch.arange(1, 2049, dtype=x.dtype)[None, :]


def _tf(res):
    def f(img):
        return torch.from_numpy(np.asarray(img.resize((res, res)), dtype=np.float32) / 255).permute(2, 0, 1)
    return f


@pytest.fixture
def setup(tmp_path):
    src = tmp_path / "img"
    src.mkdir()
    names = [f"{i:06d}.jpg" for i in range(1, N_IMAGES + 1)]
    for i, n in enumerate(names):
        Image.new("RGB", (178, 218), color=(10 * i, 20, 30)).save(src / n)
    models = {"a": Fake(1.0), "b": Fake(2.0)}
    transforms = {8: _tf(8), 16: _tf(16)}
    kw = {"names": names, "source": src, "models": models, "transforms": transforms,
              "out_root": tmp_path / "feat", "chunk_size": CHUNK, "batch_size": 3,
              "num_workers": 0, "device": "cpu", "log": lambda s: None}
    return kw


def test_extract_writes_all_outputs_in_order(setup):
    extract_split("val", **setup)
    for b in ["a", "b"]:
        for r in [8, 16]:
            feats, names = load_features(setup["out_root"], b, r, "val")
            assert feats.shape == (N_IMAGES, 2048)
            assert feats.dtype == np.float16
            assert names == setup["names"]
            # L'image i est plus claire que l'image i-1 : l'ordre est conservé
            assert np.all(np.diff(feats[:, 0].astype(np.float32)) > 0)
            assert not (feature_dir(setup["out_root"], b, r) / "parts").exists() or \
                not any((feature_dir(setup["out_root"], b, r) / "parts").iterdir())


def test_rerun_after_completion_skips_everything(setup):
    extract_split("val", **setup)
    calls = {"n": 0}
    real_forward = Fake.forward

    def counting(self, x):
        calls["n"] += 1
        return real_forward(self, x)

    Fake.forward = counting
    try:
        logs = []
        extract_split("val", **{**setup, "log": logs.append})
    finally:
        Fake.forward = real_forward
    assert calls["n"] == 0, "une partie terminée ne doit pas être recalculée"
    assert any("déjà terminé" in s for s in logs)


def test_resume_after_interruption_gives_same_result(setup, tmp_path):
    # Référence : extraction d'un seul tenant
    extract_split("val", **setup)
    ref = load_features(setup["out_root"], "a", 8, "val")[0].copy()

    # Coupure simulée : on refait dans un autre dossier, le 2e morceau échoue
    setup2 = {**setup, "out_root": tmp_path / "feat2"}
    calls = {"n": 0}
    real_forward = Fake.forward

    def flaky(self, x):
        calls["n"] += 1
        # 1er morceau : 4 images en lots de 3 → 2 lots × 2 résolutions × 2 réseaux
        # = 8 appels. L'appel 11 tombe au milieu du 2e morceau.
        if calls["n"] == 11:
            raise RuntimeError("coupure simulée")
        return real_forward(self, x)

    Fake.forward = flaky
    try:
        with pytest.raises(RuntimeError, match="coupure"):
            extract_split("val", **setup2)
    finally:
        Fake.forward = real_forward

    parts = feature_dir(setup2["out_root"], "a", 8) / "parts" / "val"
    assert sorted(p.name for p in parts.iterdir()) == ["part_00000.npy"]  # seul le 1er est écrit

    # Reprise : le 1er morceau est sauté, le résultat final est identique
    logs = []
    extract_split("val", **{**setup2, "log": logs.append})
    assert any("déjà fait" in s for s in logs)
    got = load_features(setup2["out_root"], "a", 8, "val")[0]
    np.testing.assert_array_equal(got, ref)
