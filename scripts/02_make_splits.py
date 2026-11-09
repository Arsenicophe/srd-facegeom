"""Étape 0f : construit la partition par identité et l'écrit dans splits/.

N'a besoin que de identity_CelebA.txt et list_eval_partition.txt : tourne sur
Colab comme en local.

    python scripts/02_make_splits.py --config configs/data.yaml
    python scripts/02_make_splits.py --data-dir /chemin/vers/celeba   # en local
"""

import argparse
from pathlib import Path

from facegeom.config import load_config
from facegeom.data.download import md5sum
from facegeom.data.splits import build_splits, write_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="dossier contenant les fichiers CelebA (par défaut : celui de Drive)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(args.data_dir or Path(cfg["drive_root"]) / cfg["celeba"]["dest_dir"])
    split_cfg = cfg["splits"]

    identity = data_dir / "identity_CelebA.txt"
    partition = data_dir / "list_eval_partition.txt"
    expected = cfg["celeba"]["files"]
    for p in (identity, partition):
        if md5sum(p) != expected[p.name]:
            raise SystemExit(f"{p} : empreinte inattendue, relancer l'étape 0e.")

    df, summary = build_splits(
        identity,
        partition,
        names={int(k): v for k, v in split_cfg["names"].items()},
        min_images=split_cfg["min_images_per_identity"],
    )
    summary["source_md5"] = {p.name: expected[p.name] for p in (identity, partition)}

    for path in write_splits(df, summary, split_cfg["output_dir"]):
        print("écrit :", path)
    for split, s in summary["after_filter"].items():
        b = summary["before_filter"][split]
        print(
            f"{split:5} : {s['identities']:5} identités, {s['images']:6} images "
            f"(avant filtre : {b['identities']}, {b['images']})"
        )


if __name__ == "__main__":
    main()
