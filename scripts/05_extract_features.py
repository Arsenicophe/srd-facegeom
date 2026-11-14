"""Étapes 1d-1e : extraction des vecteurs, deux backbones × deux résolutions, reprise possible.

    python scripts/05_extract_features.py --source /content/data/img_align_celeba --limit 1000  # essai
    python scripts/05_extract_features.py --source /content/data/img_align_celeba               # complet
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import torch

from facegeom.data.crop import crop_box
from facegeom.data.transforms import build_transform
from facegeom.features import extract_split, feature_dir, write_meta
from facegeom.models.backbones import load_backbone
from facegeom.utils import default_device, file_hash, load_config, run_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--extract-config", default="configs/extract.yaml")
    parser.add_argument("--source", required=True, help="dossier d'images ou zip CelebA")
    parser.add_argument("--out-root", default=None, help="remplace <drive_root>/features")
    parser.add_argument("--limit", type=int, default=None, help="essai : N premières images par partie")
    parser.add_argument("--device", default=default_device())
    args = parser.parse_args()

    data_cfg = load_config(args.data_config)
    ext_cfg = load_config(args.extract_config)
    ecfg = ext_cfg["extraction"]

    out_root = Path(args.out_root or Path(data_cfg["drive_root"]) / ecfg["output_dir"])
    if args.limit:
        out_root = out_root.with_name(out_root.name + f"_essai{args.limit}")

    # Aucun hasard dans l'extraction, mais on fixe tout de même le déterminisme
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    models = {b: load_backbone(c).to(args.device) for b, c in ext_cfg["backbones"].items()}
    transforms = {
        r: build_transform(data_cfg["crop"], r, ext_cfg["normalize"]) for r in ext_cfg["resolutions"]
    }
    split_dir = Path(data_cfg["splits"]["output_dir"])

    timings = {}
    for split in ecfg["splits"]:
        names = pd.read_csv(split_dir / f"{split}.csv").image.tolist()
        if args.limit:
            names = names[: args.limit]
        print(f"=== {split} : {len(names)} images")
        t = time.perf_counter()
        extract_split(
            split, names, args.source, models, transforms, out_root,
            chunk_size=ecfg["chunk_size"], batch_size=ecfg["batch_size"],
            num_workers=ecfg["num_workers"], device=args.device,
            storage_dtype=ecfg["storage_dtype"], amp=ecfg["amp"],
        )
        dt = time.perf_counter() - t
        timings[split] = {"images": len(names), "seconds": round(dt, 1)}
        print(f"=== {split} terminé en {dt / 60:.1f} min ({len(names) / dt:.0f} images/s, reprises comprises)")

    c = data_cfg["crop"]
    for b, bcfg in ext_cfg["backbones"].items():
        for r in ext_cfg["resolutions"]:
            write_meta(feature_dir(out_root, b, r), {
                "backbone": b, "weights": bcfg, "resolution": r,
                "crop_box": crop_box(c),
                "normalize": ext_cfg["normalize"], "extraction": ecfg, "limit": args.limit,
                "splits_md5": {s: file_hash(split_dir / f"{s}.csv") for s in ecfg["splits"]},
                "timings": timings, "run": run_info(args.device),
            })
    print("Vecteurs dans", out_root)


if __name__ == "__main__":
    main()
