"""Étape 1b : vérifie que les deux backbones se chargent et se comportent bien.

Sur quelques images de VALIDATION, pour chaque backbone et chaque résolution :
- sortie de forme (N, 2048), sans valeur infinie ou manquante ;
- même entrée deux fois → même sortie (mode évaluation, déterminisme) ;
- images différentes → vecteurs différents ;
et pour le supervisé : sortie identique à celle du modèle torchvision officiel.

    python scripts/04_check_backbones.py --zip /content/data/img_align_celeba.zip
    python scripts/04_check_backbones.py --zip hf://datasets/Yuehao/celeba@<révision>/img_align_celeba.zip
"""

import argparse
import time

import pandas as pd
import torch
import torch.nn.functional as F

from facegeom.config import load_config
from facegeom.device import default_device
from facegeom.data.images import open_celeba_zip, read_image
from facegeom.data.transforms import build_transform
from facegeom.models.backbones import FEATURE_DIM, load_backbone


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--extract-config", default="configs/extract.yaml")
    parser.add_argument("--zip", required=True)
    parser.add_argument("--n-images", type=int, default=8)
    parser.add_argument("--device", default=default_device())
    args = parser.parse_args()

    data_cfg = load_config(args.data_config)
    ext_cfg = load_config(args.extract_config)
    torch.manual_seed(0)

    val = pd.read_csv(f"{data_cfg['splits']['output_dir']}/val.csv")
    names = val.groupby("identity").head(1).image.head(args.n_images).tolist()
    zf = open_celeba_zip(args.zip)
    images = [read_image(zf, n) for n in names]
    print(f"{len(images)} images de validation, appareil : {args.device}")

    for bname, bcfg in ext_cfg["backbones"].items():
        model = load_backbone(bcfg).to(args.device)
        assert not model.training, "le backbone doit être en mode évaluation"
        assert not any(p.requires_grad for p in model.parameters()), "le backbone doit être gelé"

        for res in ext_cfg["resolutions"]:
            tf = build_transform(data_cfg["crop"], res, ext_cfg["normalize"])
            x = torch.stack([tf(img) for img in images]).to(args.device)
            assert x.shape == (len(images), 3, res, res)
            with torch.no_grad():
                t = time.perf_counter()
                f1 = model(x)
                dt = time.perf_counter() - t
                f2 = model(x)

            assert f1.shape == (len(images), FEATURE_DIM), f1.shape
            assert torch.isfinite(f1).all(), "valeurs infinies ou manquantes"
            assert torch.equal(f1, f2), "même entrée, sorties différentes"
            z = F.normalize(f1.float(), dim=1)
            cos = z @ z.T
            off = cos[~torch.eye(len(images), dtype=torch.bool, device=cos.device)]
            assert off.max() < 0.999, "deux images différentes donnent le même vecteur"
            print(
                f"{bname:16} {res:3d} px : sortie {tuple(f1.shape)}, norme moyenne "
                f"{f1.norm(dim=1).mean():6.2f}, cosinus entre images différentes "
                f"{off.min():.2f} à {off.max():.2f} (moyenne {off.mean():.2f}), {dt * 1000:.0f} ms"
            )

        if bname == "resnet50_sup":
            from torchvision.models import ResNet50_Weights, resnet50

            ref = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            ref.fc = torch.nn.Identity()
            ref.eval().to(args.device)
            with torch.no_grad():
                same = torch.allclose(ref(x), model(x), atol=1e-5)
            assert same, "le supervisé diffère du modèle torchvision officiel"
            print(f"{bname:16} identique au modèle torchvision officiel : oui")

    print("Toutes les vérifications sont passées.")


if __name__ == "__main__":
    main()
