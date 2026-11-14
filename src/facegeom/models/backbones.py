"""Backbones gelés : ResNet-50 torchvision, poids supervisés ou Barlow Twins.

Poids téléchargés directement et vérifiés par SHA-256 (pas de ``torch.hub.load``,
qui exécuterait du code distant). Sortie : vecteur de 2 048 après le pooling.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision.models import resnet50

from facegeom.utils import file_hash

FEATURE_DIM = 2048
# Clés absentes tolérées : la couche fc (retirée) et les compteurs BatchNorm
# (absents du checkpoint V1, inutiles en mode évaluation).
ALLOWED_MISSING = {"fc.weight", "fc.bias"}
ALLOWED_MISSING_SUFFIX = ".num_batches_tracked"


class WeightsError(RuntimeError):
    """Fichier de poids inattendu (empreinte ou clés)."""


def _download_verified(url: str, sha256: str) -> dict:
    """Télécharge (ou reprend du cache) un checkpoint et vérifie son SHA-256."""
    path = Path(torch.hub.get_dir()) / "checkpoints" / f"{sha256[:12]}_{Path(url).name}"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(url, str(path), progress=True)
    got = file_hash(path, "sha256")
    if got != sha256:
        path.unlink()
        raise WeightsError(f"{url} : SHA-256 {got}, attendu {sha256}.")
    return torch.load(path, map_location="cpu", weights_only=True)


def load_backbone(cfg: dict) -> nn.Module:
    """Backbone gelé en mode évaluation ; ``cfg`` a ``url`` et ``sha256``."""
    model = resnet50(weights=None)
    state = _download_verified(cfg["url"], cfg["sha256"])
    expected = set(model.state_dict())
    missing = expected - set(state)
    unexpected = set(state) - expected
    bad = [k for k in missing if k not in ALLOWED_MISSING and not k.endswith(ALLOWED_MISSING_SUFFIX)]
    if unexpected or bad:
        raise WeightsError(f"Clés inattendues : {sorted(unexpected)[:5]} ; manquantes : {sorted(bad)[:5]}")
    model.load_state_dict(state, strict=False)
    model.fc = nn.Identity()
    model.eval()
    model.requires_grad_(False)
    return model
