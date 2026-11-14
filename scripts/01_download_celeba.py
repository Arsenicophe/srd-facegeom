"""Étape 0e : télécharge CelebA sur Google Drive et vérifie les empreintes.

À lancer une seule fois. Relancer ne retélécharge que ce qui manque ou est faux.

    python scripts/01_download_celeba.py --config configs/data.yaml
"""

import argparse

from facegeom.data.download import download_celeba
from facegeom.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument(
        "--drive-root",
        default=None,
        help="remplace drive_root de la configuration (tests en local)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    drive_root = args.drive_root or cfg["drive_root"]
    dest = download_celeba(cfg["celeba"], drive_root)
    print(f"CelebA prêt dans {dest}")


if __name__ == "__main__":
    main()
