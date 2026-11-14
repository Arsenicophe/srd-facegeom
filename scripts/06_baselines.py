"""Étapes 2c-2d : mesures préalables sur la validation, sans entraînement de perte.

Pour chaque backbone et résolution, deux projections ajustées sur l'entraînement :
« brut » (ACP 128, comme la ligne Pretrained de Musgrave) et LDA 128. Mêmes
métriques (MAP@R, P@1, R-Prec, AUC, EER, TAR) et mêmes paires partout. Les
MAP@R par requête sont gardés pour la décision (07).

    python scripts/06_baselines.py                   # tout, histogrammes en français
    python scripts/06_baselines.py --selected --en   # backbone retenu seul, figure du rapport
"""

import argparse
import time

import numpy as np
import pandas as pd

from facegeom.eval.baselines import apply_projection, fit_lda, fit_pca
from facegeom.eval.retrieval import retrieval_per_query, summarize
from facegeom.eval.verification import make_pairs, pair_scores, verification_summary
from facegeom.features import load_xy
from facegeom.utils import load_config, new_run_dir, run_info, write_json
from facegeom.viz import EN_LABELS, similarity_histograms

NICE = {"resnet50_sup": "Supervisé", "resnet50_barlow": "Barlow Twins"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--features-root", default=None, help="remplace features_root (Drive sur Colab)")
    parser.add_argument("--selected", action="store_true", help="backbone et résolution retenus seulement")
    parser.add_argument("--en", action="store_true", help="étiquettes en anglais, format compact")
    parser.add_argument("--fig", default=None, help="chemin de la figure (défaut : dans le run)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["features_root"] = args.features_root or cfg["features_root"]
    if cfg["eval_split"] == "test":
        raise SystemExit("L'étape 2 ne doit jamais utiliser le test.")
    split_dir = load_config(args.data_config)["splits"]["output_dir"]
    vcfg, dim = cfg["verification"], cfg["dim"]
    combos = ([(cfg["selected"]["backbone"], cfg["selected"]["resolution"])] if args.selected
              else [(b, r) for b in cfg["backbones"] for r in cfg["resolutions"]])

    out = new_run_dir(cfg["runs_dir"], "2d_baselines")
    rows, panels, per_query = [], [], {}
    pairs = y = ref_names = None
    for b, r in combos:
        xtr, ytr, _ = load_xy(cfg["features_root"], b, r, cfg["fit_split"], split_dir)
        xva, yva, names = load_xy(cfg["features_root"], b, r, cfg["eval_split"], split_dir)
        if pairs is None:
            ref_names = names
            pairs, y = make_pairs(yva, vcfg["n_negative"], vcfg["seed"])
        elif names != ref_names:
            raise ValueError("Ordre des images différent : les paires ne seraient plus les mêmes.")

        t = time.perf_counter()
        models = {f"brut_acp{dim}": fit_pca(xtr, dim), f"lda{dim}": fit_lda(xtr, ytr, dim)}
        for v, m in models.items():
            x = apply_projection(m, xva)
            pq = retrieval_per_query(x, yva)
            scores = pair_scores(x, pairs)
            ver = verification_summary(y, scores, tuple(vcfg["fars"]))
            row = {"backbone": b, "resolution": r, "variant": v, **summarize(pq), **ver}
            rows.append(row)
            per_query[f"{b}_{r}_{v}"] = pq["map_at_r"]
            per_query["labels"] = pq["labels"]
            print(f"{b:16} {r:3} px {v:11} : MAP@R {row['map_at_r']:5.2f}  P@1 {row['p_at_1']:5.2f}  "
                  f"AUC {row['auc']:5.2f}  TAR@1e-2 {row['tar_at_far_0.01']:5.2f}")
            if args.en:
                title = f"Raw features (PCA {dim})" if v.startswith("brut") else f"After LDA ({dim})"
                note = f"MAP@R {row['map_at_r']:.1f}%"
            else:
                title = f"{NICE.get(b, b)}, {r} px ({'brut, ACP' if v.startswith('brut') else 'LDA'} {dim})"
                note = f"MAP@R {row['map_at_r']:.1f} %   AUC {row['auc']:.1f} %"
            panels.append({"title": title, "same": scores[y == 1], "diff": scores[y == 0], "note": note})
        print(f"   ({time.perf_counter() - t:.0f} s)")

    pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
    np.savez_compressed(out / "per_query_map_at_r.npz", **per_query)
    fig_kw = {"labels": EN_LABELS, "panel_size": (3.6, 2.4), "dpi": 300} if args.en else {}
    fig = similarity_histograms(panels, args.fig or out / "histogrammes.png", ncols=2, **fig_kw)
    write_json(out / "meta.json", {"config": cfg, "args": vars(args), "run": run_info()})
    print("Résultats dans", out, "; figure :", fig)


if __name__ == "__main__":
    main()
