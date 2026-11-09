"""Étape 2c : mesure brute sur la validation (sans aucun entraînement).

Pour chaque backbone et résolution :
- « brut 2048 » : similarité cosinus sur les vecteurs tels quels ;
- « brut ACP 128 » : après ACP à 128 dimensions ajustée sur l'entraînement
  (version utilisée par la règle de décision, comme la ligne « Pretrained » de
  Musgrave).
Métriques de recherche : MAP@R, P@1, R-Precision. Les valeurs par requête sont
enregistrées pour le bootstrap de l'étape 2e.

    python scripts/06_raw_measure.py
"""

import argparse
import time

import numpy as np
import pandas as pd

from facegeom.config import load_config
from facegeom.eval.baselines import apply_projection, fit_pca
from facegeom.eval.retrieval import retrieval_per_query, summarize
from facegeom.features.store import load_features
from facegeom.results import new_run_dir, write_json
from facegeom.runinfo import run_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg["eval_split"] == "test":
        raise SystemExit("L'étape 2 ne doit jamais utiliser le test.")
    split_dir = load_config(args.data_config)["splits"]["output_dir"]
    labels_by_image = pd.read_csv(f"{split_dir}/{cfg['eval_split']}.csv").set_index("image").identity

    out = new_run_dir(cfg["runs_dir"], "2c_mesure_brute")
    results, per_query = [], {}
    for b in cfg["backbones"]:
        for r in cfg["resolutions"]:
            train, _ = load_features(cfg["features_root"], b, r, cfg["fit_split"])
            val, names = load_features(cfg["features_root"], b, r, cfg["eval_split"])
            labels = labels_by_image.loc[names].to_numpy()

            t = time.perf_counter()
            pca = fit_pca(np.asarray(train), cfg["dim"])
            variants = {
                "brut_2048": np.asarray(val, dtype=np.float32),
                f"brut_acp{cfg['dim']}": apply_projection(pca, np.asarray(val)),
            }
            for v, x in variants.items():
                pq = retrieval_per_query(x, labels)
                s = summarize(pq)
                results.append({"backbone": b, "resolution": r, "variant": v, **s,
                                "n_queries": int(len(pq["map_at_r"]))})
                per_query[f"{b}_{r}_{v}"] = pq["map_at_r"]
                print(f"{b:16} {r:3} px {v:12} : MAP@R {s['map_at_r']:5.2f}  "
                      f"P@1 {s['p_at_1']:5.2f}  R-Prec {s['r_precision']:5.2f}")
            per_query["labels"] = pq["labels"]
            print(f"   ({time.perf_counter() - t:.0f} s ; variance gardée par l'ACP : "
                  f"{100 * pca.explained_variance_ratio_.sum():.1f} %)")

    pd.DataFrame(results).to_csv(out / "results.csv", index=False)
    np.savez_compressed(out / "per_query_map_at_r.npz", **per_query)
    write_json(out / "meta.json", {"config": cfg, "run": run_info()})
    print("Résultats dans", out)


if __name__ == "__main__":
    main()
