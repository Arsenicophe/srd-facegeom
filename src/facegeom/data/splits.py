"""Partition de CelebA par identité.

On part du découpage officiel (``list_eval_partition.txt``), on vérifie
qu'aucune identité n'apparaît dans deux parties, puis on retire les identités
qui ont trop peu de photos. Le résultat est écrit en CSV (une ligne par image)
avec un résumé JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


class IdentityOverlapError(RuntimeError):
    """Une même identité apparaît dans plusieurs parties."""


def load_annotations(identity_path: str | Path, partition_path: str | Path) -> pd.DataFrame:
    """Lit les fichiers CelebA et renvoie un tableau ``image, identity, partition``.

    ``partition`` est le code officiel (0, 1, 2). Vérifie que les deux fichiers
    décrivent exactement les mêmes images.
    """
    ids = pd.read_csv(identity_path, sep=r"\s+", header=None, names=["image", "identity"])
    part = pd.read_csv(partition_path, sep=r"\s+", header=None, names=["image", "partition"])
    if ids.image.duplicated().any() or part.image.duplicated().any():
        raise ValueError("Images en double dans les fichiers d'annotations.")
    if set(ids.image) != set(part.image):
        raise ValueError("Les deux fichiers ne décrivent pas les mêmes images.")
    return ids.merge(part, on="image", validate="one_to_one")


def check_disjoint(df: pd.DataFrame, split_col: str = "split") -> None:
    """Lève ``IdentityOverlapError`` si une identité apparaît dans plusieurs parties."""
    n_parts = df.groupby("identity")[split_col].nunique()
    shared = n_parts[n_parts > 1]
    if len(shared) > 0:
        raise IdentityOverlapError(
            f"{len(shared)} identités dans plusieurs parties, par exemple "
            f"{shared.index[:5].tolist()}"
        )


def filter_min_images(df: pd.DataFrame, min_images: int) -> pd.DataFrame:
    """Garde seulement les identités ayant au moins ``min_images`` images.

    Le compte se fait par partie ; comme les parties sont disjointes en
    identités, c'est aussi le compte global.
    """
    counts = df.groupby(["split", "identity"])["image"].transform("size")
    return df[counts >= min_images].reset_index(drop=True)


def build_splits(
    identity_path: str | Path,
    partition_path: str | Path,
    names: dict[int, str],
    min_images: int,
) -> tuple[pd.DataFrame, dict]:
    """Construit la partition filtrée et son résumé.

    Renvoie un tableau ``image, identity, split`` et un dict de statistiques
    avant et après filtrage.
    """
    df = load_annotations(identity_path, partition_path)
    unknown = set(df.partition) - set(names)
    if unknown:
        raise ValueError(f"Codes de partition inconnus : {sorted(unknown)}")
    df["split"] = df.partition.map(names)
    df = df.drop(columns="partition")

    check_disjoint(df)
    kept = filter_min_images(df, min_images)
    check_disjoint(kept)

    def stats(d: pd.DataFrame) -> dict:
        return {
            s: {"identities": int(g.identity.nunique()), "images": len(g)}
            for s, g in d.groupby("split")
        }

    summary = {
        "rule": {"source": "list_eval_partition.txt", "min_images_per_identity": min_images},
        "before_filter": stats(df),
        "after_filter": stats(kept),
    }
    return kept, summary


def write_splits(df: pd.DataFrame, summary: dict, out_dir: str | Path) -> list[Path]:
    """Écrit un CSV ``image,identity`` par partie et ``summary.json``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for split, g in df.groupby("split"):
        path = out_dir / f"{split}.csv"
        g.sort_values("image")[["image", "identity"]].to_csv(path, index=False)
        written.append(path)
    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    written.append(path)
    return written


def read_split(split_dir: str | Path, split: str) -> pd.DataFrame:
    """Relit une partie écrite par ``write_splits``."""
    return pd.read_csv(Path(split_dir) / f"{split}.csv")
