"""Étape 1a : planche d'inspection des tailles de recadrage candidates.

Images tirées de la VALIDATION (jamais du test). La planche contient des
visages de CelebA : elle reste dans runs/ (non versionné), elle n'est pas
redistribuée.

    # Colab (zip copié par le notebook 00) :
    python scripts/03_crop_sheet.py --zip /content/data/img_align_celeba.zip
    # Local, sans le zip complet (lit seulement les images utiles) :
    python scripts/03_crop_sheet.py --zip hf://datasets/Yuehao/celeba@<révision>/img_align_celeba.zip
"""

import argparse

import pandas as pd

from facegeom.config import load_config
from facegeom.data.images import open_celeba_zip, read_image
from facegeom.viz import crop_contact_sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--zip", required=True, help="chemin local ou hf:// du zip d'images")
    parser.add_argument("--out", default="runs/1a_recadrage/planche.png")
    args = parser.parse_args()

    cfg = load_config(args.config)
    crop_cfg = cfg["crop"]
    sheet = crop_cfg["sheet"]
    if sheet["split"] == "test":
        raise SystemExit("La planche ne doit jamais utiliser le test.")

    split = pd.read_csv(f"{cfg['splits']['output_dir']}/{sheet['split']}.csv")
    # Une image par identité, pour voir des personnes différentes
    picks = split.groupby("identity").head(1).sample(n=sheet["n_images"], random_state=sheet["seed"])

    zf = open_celeba_zip(args.zip)
    images = [read_image(zf, name) for name in picks.image]
    out = crop_contact_sheet(images, crop_cfg["candidate_sizes"], crop_cfg, args.out)
    print("images :", ", ".join(picks.image))
    print("planche :", out)


if __name__ == "__main__":
    main()
