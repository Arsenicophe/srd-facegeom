"""Étape 2d : référence LDA sur la validation (sans aucun entraînement de perte).

Pour chaque backbone et résolution : LDA à 128 dimensions ajustée sur les
identités d'ENTRAÎNEMENT, appliquée aux identités de VALIDATION (jamais vues).
Mêmes métriques et mêmes paires que la mesure brute (étape 2c), qui est
recalculée ici pour les histogrammes côte à côte.

    python scripts/08_lda_reference.py
"""

import argparse
import time

import numpy as np
import pandas as pd

from facegeom.config import load_config
from facegeom.eval.baselines import apply_projection, fit_lda, fit_pca
from facegeom.eval.retrieval import retrieval_per_query, summarize
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
    lab = {s: pd.read_csv(f"{split_dir}/{s}.csv").set_index("image").identity
           for s in [cfg["fit_split"], cfg["eval_split"]]}

    out = new_run_dir(cfg["runs_dir"], "2d_reference_lda")
    rows, panels, per_query = [], [], {}
    pairs = y = ref_names = None
    for b in cfg["backbones"]:
        for r in cfg["resolutions"]:
            train, tr_names = load_features(cfg["features_root"], b, r, cfg["fit_split"])
            val, names = load_features(cfg["features_root"], b, r, cfg["eval_split"])
            ytr = lab[cfg["fit_split"]].loc[tr_names].to_numpy()
            yval = lab[cfg["eval_split"]].loc[names].to_numpy()
            if pairs is None:
                ref_names = names
                pairs, y = make_pairs(yval, vcfg["n_negative"], vcfg["seed"])
            elif names != ref_names:
                raise ValueError("Ordre des images différent : les paires ne seraient plus les mêmes.")

            t = time.perf_counter()
            train = np.asarray(train)
            models = {"brut_acp128": fit_pca(train, cfg["dim"]),
                      "lda128": fit_lda(train, ytr, cfg["dim"])}
            fit_s = time.perf_counter() - t
            scores = {}
            for v, m in models.items():
                x = apply_projection(m, np.asarray(val))
                pq = retrieval_per_query(x, yval)
                scores[v] = pair_scores(x, pairs)
                ver = verification_summary(y, scores[v], tuple(vcfg["fars"]))
                rows.append({"backbone": b, "resolution": r, "variant": v, **summarize(pq), **ver})
                per_query[f"{b}_{r}_{v}"] = pq["map_at_r"]
                per_query["labels"] = pq["labels"]
                print(f"{b:16} {r:3} px {v:11} : MAP@R {rows[-1]['map_at_r']:5.2f}  "
                      f"P@1 {rows[-1]['p_at_1']:5.2f}  AUC {ver['auc']:5.2f}  "
                      f"TAR@1e-2 {ver['tar_at_far_0.01']:5.2f}")
            print(f"   (ajustement ACP + LDA : {fit_s:.0f} s)")
            nice = "Supervisé" if b.endswith("sup") else "Barlow Twins"
            for v, label in [("brut_acp128", "brut, ACP 128"), ("lda128", "LDA 128")]:
                row = next(x for x in rows if x["backbone"] == b and x["resolution"] == r and x["variant"] == v)
                panels.append({"title": f"{nice}, {r} px ({label})",
                               "same": scores[v][y == 1], "diff": scores[v][y == 0],
                               "note": f"MAP@R {row['map_at_r']:.1f} %   AUC {row['auc']:.1f} %"})

    pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
    np.savez_compressed(out / "per_query_map_at_r.npz", **per_query)
    similarity_histograms(panels, out / "histogrammes_brut_vs_lda.png", ncols=2)
    write_json(out / "meta.json", {"config": cfg, "run": run_info()})
    print("Résultats dans", out)


if __name__ == "__main__":
    main()
