"""Figure de la synthèse d'avancement : histogrammes brut et LDA, backbone retenu.

Même calcul que l'étape 2d (validation, mêmes paires), limité au backbone et à
la résolution retenus, avec des étiquettes en anglais et un format compact.

    python scripts/11_report_figure.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from facegeom.config import load_config
from facegeom.eval.baselines import apply_projection, fit_lda, fit_pca
from facegeom.eval.retrieval import retrieval_per_query, summarize
from facegeom.eval.verification import make_pairs, pair_scores, verification_summary
from facegeom.features.store import load_features
from facegeom.viz import EN_LABELS, similarity_histograms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--out", default="docs/figures/report_hist_barlow224.png")
    args = parser.parse_args()

    cfg = load_config(args.config)
    vcfg = cfg["verification"]
    b, r = cfg["selected"]["backbone"], cfg["selected"]["resolution"]
    split_dir = load_config(args.data_config)["splits"]["output_dir"]
    tr_lab = pd.read_csv(f"{split_dir}/{cfg['fit_split']}.csv").set_index("image").identity
    va_lab = pd.read_csv(f"{split_dir}/{cfg['eval_split']}.csv").set_index("image").identity

    train, tr_names = load_features(cfg["features_root"], b, r, cfg["fit_split"])
    val, names = load_features(cfg["features_root"], b, r, cfg["eval_split"])
    train, yval = np.asarray(train), va_lab.loc[names].to_numpy()
    pairs, y = make_pairs(yval, vcfg["n_negative"], vcfg["seed"])

    panels = []
    for title, model in [("Raw features (PCA 128)", fit_pca(train, cfg["dim"])),
                         ("After LDA (128)", fit_lda(train, tr_lab.loc[tr_names].to_numpy(), cfg["dim"]))]:
        x = apply_projection(model, np.asarray(val))
        s = pair_scores(x, pairs)
        m = summarize(retrieval_per_query(x, yval))["map_at_r"]
        v = verification_summary(y, s, tuple(vcfg["fars"]))
        print(f"{title:24} MAP@R {m:.2f}  TAR@1e-2 {v['tar_at_far_0.01']:.2f}")
        panels.append({"title": title, "same": s[y == 1], "diff": s[y == 0],
                       "note": f"MAP@R {m:.1f}%"})

    out = similarity_histograms(panels, args.out, ncols=2, labels=EN_LABELS,
                                panel_size=(3.6, 2.4), dpi=300)
    print("Figure :", Path(out).resolve())


if __name__ == "__main__":
    main()
