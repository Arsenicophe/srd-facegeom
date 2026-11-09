"""Étape 2c (suite) : vérification et histogrammes de la mesure brute, sur la validation.

Version « brut ACP 128 » (celle de la règle de décision). Les paires sont tirées
une seule fois et sont identiques pour tous les backbones et résolutions.

    python scripts/07_raw_verification.py
"""

import argparse

import numpy as np
import pandas as pd

from facegeom.config import load_config
from facegeom.eval.baselines import apply_projection, fit_pca
from facegeom.eval.verification import make_pairs, pair_scores, verification_summary
from facegeom.features.store import load_features
from facegeom.results import new_run_dir, write_json
from facegeom.runinfo import run_info
from facegeom.viz import similarity_histograms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    vcfg = cfg["verification"]
    if cfg["eval_split"] == "test":
        raise SystemExit("L'étape 2 ne doit jamais utiliser le test.")
    split_dir = load_config(args.data_config)["splits"]["output_dir"]
    labels_by_image = pd.read_csv(f"{split_dir}/{cfg['eval_split']}.csv").set_index("image").identity

    out = new_run_dir(cfg["runs_dir"], "2c_verification_brute")
    pairs = y = ref_names = None
    rows, panels = [], []
    for b in cfg["backbones"]:
        for r in cfg["resolutions"]:
            train, _ = load_features(cfg["features_root"], b, r, cfg["fit_split"])
            val, names = load_features(cfg["features_root"], b, r, cfg["eval_split"])
            if pairs is None:
                ref_names = names
                pairs, y = make_pairs(labels_by_image.loc[names].to_numpy(),
                                      vcfg["n_negative"], vcfg["seed"])
            elif names != ref_names:
                raise ValueError("Ordre des images différent : les paires ne seraient plus les mêmes.")

            x = apply_projection(fit_pca(np.asarray(train), cfg["dim"]), np.asarray(val))
            scores = pair_scores(x, pairs)
            s = verification_summary(y, scores, tuple(vcfg["fars"]))
            rows.append({"backbone": b, "resolution": r, "variant": f"brut_acp{cfg['dim']}", **s})
            tars = "  ".join(f"TAR@{f:g} {s[f'tar_at_far_{f:g}']:5.2f}" for f in vcfg["fars"])
            print(f"{b:16} {r:3} px : AUC {s['auc']:5.2f}  EER {s['eer']:5.2f}  {tars}")
            panels.append({
                "title": f"{'Supervisé' if b.endswith('sup') else 'Barlow Twins'}, {r} px (brut, ACP 128)",
                "same": scores[y == 1], "diff": scores[y == 0],
                "note": f"AUC {s['auc']:.1f} %   EER {s['eer']:.1f} %",
            })

    pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
    fig = similarity_histograms(panels, out / "histogrammes.png")
    write_json(out / "meta.json", {"config": cfg, "n_pairs": {"positive": int((y == 1).sum()),
               "negative": int((y == 0).sum())}, "run": run_info()})
    print("Figure :", fig)


if __name__ == "__main__":
    main()
