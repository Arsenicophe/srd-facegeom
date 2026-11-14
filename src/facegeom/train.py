"""Entraînement de la tête linéaire sur les vecteurs gelés, une perte à la fois.

Tout tient en mémoire : les vecteurs sont chargés une fois sur l'appareil, les
lots sont des indices. Arrêt précoce sur le MAP@R de validation.
"""

from __future__ import annotations

import random
from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_metric_learning import losses, miners, samplers
from torch import nn

from facegeom.eval.retrieval import retrieval_per_query, summarize


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Head(nn.Module):
    """Linéaire ``in_dim → out_dim`` puis normalisation L2."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.linear(x), dim=1)


def make_loss(name: str, params: dict, n_classes: int, dim: int):
    """Perte pytorch-metric-learning et mineur éventuel ; ``nsoftmax`` = ArcFace à marge nulle."""
    if name == "nsoftmax":
        return losses.NormalizedSoftmaxLoss(n_classes, dim, temperature=1 / params["s"]), None
    if name == "arcface":
        return losses.ArcFaceLoss(n_classes, dim, margin=params["m"], scale=params["s"]), None
    if name == "contrastive":
        return losses.ContrastiveLoss(pos_margin=0.0, neg_margin=params["neg_margin"]), None
    if name == "triplet":
        return (losses.TripletMarginLoss(margin=params["margin"]),
                miners.TripletMarginMiner(margin=params["margin"], type_of_triplets="semihard"))
    raise ValueError(f"Perte inconnue : {name}")


def sample_params(ranges: dict, n: int, seed: int) -> list[dict]:
    """``n`` configurations tirées au hasard dans ``ranges`` (``[min, max]`` ou ``[min, max, log]``)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        p = {}
        for k, r in ranges.items():
            lo, hi = float(r[0]), float(r[1])
            p[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if len(r) > 2 else float(rng.uniform(lo, hi))
        out.append(p)
    return out


@torch.no_grad()
def embed(head: Head, x: torch.Tensor, batch: int = 4096) -> np.ndarray:
    head.eval()
    out = torch.cat([head(x[i : i + batch]) for i in range(0, len(x), batch)])
    head.train()
    return out.cpu().numpy()


def train(
    cfg: dict,
    loss_name: str,
    params: dict,
    seed: int,
    xtr: np.ndarray,
    ytr: np.ndarray,
    xva: np.ndarray,
    yva: np.ndarray,
    device: str,
    log: Callable[[str], None] = print,
) -> dict:
    """Entraîne une tête ; renvoie l'historique, la meilleure époque et l'état de la meilleure tête.

    ``params`` contient ``lr`` et les paramètres de la perte. Un lot = ``batch_size``
    vecteurs, ``m_per_class`` par identité ; une époque = autant de lots que
    ``len(xtr) // batch_size``, identique pour toutes les pertes.
    """
    set_seed(seed)
    lcfg = cfg["losses"][loss_name]
    classes, ytr_idx = np.unique(ytr, return_inverse=True)
    x = torch.as_tensor(xtr, dtype=torch.float32, device=device)
    y = torch.as_tensor(ytr_idx, dtype=torch.long, device=device)
    xv = torch.as_tensor(xva, dtype=torch.float32, device=device)

    head = Head(cfg["head"]["in_dim"], cfg["head"]["out_dim"]).to(device)
    loss_fn, miner = make_loss(loss_name, params, len(classes), cfg["head"]["out_dim"])
    loss_fn.to(device)
    opt = torch.optim.RMSprop([*head.parameters(), *loss_fn.parameters()], lr=params["lr"])
    gen = torch.Generator().manual_seed(seed)
    sampler = samplers.MPerClassSampler(ytr_idx, lcfg["m_per_class"], cfg["batch_size"],
                                        num_batches=len(xtr) // cfg["batch_size"], generator=gen)

    history, best = [], {"map_at_r": -1.0, "epoch": 0, "state": None}
    for epoch in range(1, cfg["max_epochs"] + 1):
        total, n = 0.0, 0
        for idx in sampler:
            idx = torch.as_tensor(idx, device=device)
            z = head(x[idx])
            loss = loss_fn(z, y[idx], miner(z, y[idx]) if miner else None)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, n = total + loss.item(), n + 1
        if epoch % cfg["eval_every"]:
            continue
        m = summarize(retrieval_per_query(embed(head, xv), yva))
        history.append({"epoch": epoch, "train_loss": total / n, **m})
        log(f"époque {epoch:3d} : perte {total / n:.4f}  MAP@R val {m['map_at_r']:.2f}  P@1 {m['p_at_1']:.2f}")
        if m["map_at_r"] > best["map_at_r"]:
            best = {"map_at_r": m["map_at_r"], "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}}
        elif epoch - best["epoch"] >= cfg["patience"]:
            log(f"arrêt : {cfg['patience']} époques sans progrès")
            break
    return {"history": history, "best_epoch": best["epoch"], "best_map_at_r": best["map_at_r"],
            "head_state": best["state"], "n_classes": len(classes)}
