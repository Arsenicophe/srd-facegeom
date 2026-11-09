"""Choix de l'appareil de calcul."""

import torch


def default_device() -> str:
    """GPU NVIDIA (Colab) si disponible, sinon GPU de la puce Apple, sinon CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
