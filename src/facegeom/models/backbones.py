"""Backbones gelés : ResNet-50 supervisé et ResNet-50 Barlow Twins.

Les deux sont la même architecture torchvision ; seuls les poids diffèrent.
La couche de classification ``fc`` est remplacée par l'identité : la sortie est
le vecteur de 2 048 dimensions après le pooling global.

Les poids sont téléchargés directement (pas de ``torch.hub.load``, qui
exécuterait du code distant) et vérifiés par leur empreinte SHA-256.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn
from torchvision.models import resnet50

FEATURE_DIM = 2048
# Seules clés absentes autorisées dans un checkpoint :
# - la couche de classification, que l'on retire de toute façon (Barlow Twins
#   ne la fournit pas) ;
# - les compteurs ``num_batches_tracked`` des BatchNorm, absents du vieux
#   checkpoint torchvision V1 ; ils ne servent pas en mode évaluation.
ALLOWED_MISSING_EXACT = {"fc.weight", "fc.bias"}
ALLOWED_MISSING_SUFFIX = ".num_batches_tracked"


def _missing_is_allowed(missing: set[str]) -> bool:
    return all(k in ALLOWED_MISSING_EXACT or k.endswith(ALLOWED_MISSING_SUFFIX) for k in missing)


class WeightsError(RuntimeError):
    """Fichier de poids inattendu (empreinte ou clés)."""


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_verified(url: str, sha256: str) -> dict:
    """Télécharge (ou reprend du cache) un checkpoint et vérifie son SHA-256."""
    # Préfixe par l'empreinte : les deux checkpoints s'appellent « resnet50… »
    filename = f"{sha256[:12]}_{Path(url).name}"
    path = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(url, str(path), progress=True)
    got = _sha256(path)
    if got != sha256:
        path.unlink()
        raise WeightsError(f"{url} : SHA-256 {got}, attendu {sha256}.")
    return torch.load(path, map_location="cpu", weights_only=True)


def load_backbone(cfg: dict) -> nn.Module:
    """Construit un backbone gelé, en mode évaluation, sortie 2 048.

    ``cfg`` est une entrée de ``backbones`` dans ``configs/extract.yaml``
    (``url`` et ``sha256``).
    """
    model = resnet50(weights=None)
    state = _download_verified(cfg["url"], cfg["sha256"])

    expected = set(model.state_dict())
    missing = expected - set(state)
    unexpected = set(state) - expected
    if unexpected or not _missing_is_allowed(missing):
        raise WeightsError(
            f"Clés inattendues : {sorted(unexpected)[:5]} ; manquantes : {sorted(missing)[:5]}"
        )
    model.load_state_dict(state, strict=False)

    model.fc = nn.Identity()
    model.eval()  # BatchNorm en mode inférence : statistiques figées
    for p in model.parameters():
        p.requires_grad_(False)
    return model
