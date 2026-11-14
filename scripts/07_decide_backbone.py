"""Étape 2e : applique la règle de décision (protocole, section 5) aux résultats de l'étape 2d.

Ne recalcule rien : lit les MAP@R par requête enregistrés par scripts/06_baselines.py.

    python scripts/07_decide_backbone.py --run runs/<date>_2d_baselines
"""

import argparse
from pathlib import Path

import numpy as np

from facegeom.eval.bootstrap import bootstrap_diff
from facegeom.utils import load_config, new_run_dir, run_info, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--run", required=True, help="dossier de résultats de l'étape 2d")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dec = cfg["decision"]
    pq = np.load(Path(args.run) / "per_query_map_at_r.npz")
    labels = pq["labels"]

    def mapr(b, r, v):
        return 100 * pq[f"{b}_{r}_{v}"].mean()

    report = {"source_run": str(args.run), "rule": dec}

    # Règle 1 : résolution, par backbone, meilleure LDA
    res = {}
    for b in cfg["backbones"]:
        scores = {r: mapr(b, r, "lda128") for r in cfg["resolutions"]}
        res[b] = max(scores, key=scores.get)
        print(f"Règle 1 — {b} : LDA {', '.join(f'{r} px {s:.2f}' for r, s in scores.items())}"
              f" → {res[b]} px")
    report["rule1_resolution"] = res

    # Règle 4 (vérifiée d'abord) : LDA minimale
    lda_best = {b: mapr(b, res[b], "lda128") for b in cfg["backbones"]}
    if all(v < dec["lda_minimum"] for v in lda_best.values()):
        report["stop"] = "LDA < seuil pour les deux backbones"
        print("Règle 4 — ARRÊT : la LDA est sous le seuil pour les deux backbones.")
    else:
        print(f"Règle 4 — LDA ≥ {dec['lda_minimum']} % : ok")

    # Règle 2 : backbone, bootstrap apparié sur les identités
    b1, b2 = cfg["backbones"]
    boot = bootstrap_diff(pq[f"{b2}_{res[b2]}_lda128"], pq[f"{b1}_{res[b1]}_lda128"], labels,
                          n_boot=cfg["bootstrap"]["n"], seed=cfg["bootstrap"]["seed"])
    boot = {k: (100 * v if isinstance(v, float) else v) for k, v in boot.items()}
    if boot["contains_zero"]:
        chosen, why = dec["tie_breaker"], "écart non solide (IC contient 0)"
    else:
        chosen = b2 if boot["diff"] > 0 else b1
        why = "écart solide (IC exclut 0)"
    print(f"Règle 2 — LDA {b2} − {b1} = {boot['diff']:+.2f} points, "
          f"IC 95 % [{boot['ci_low']:+.2f} ; {boot['ci_high']:+.2f}] → {why} → {chosen}")
    report["rule2_backbone"] = {"bootstrap": boot, "chosen": chosen, "reason": why}

    # Règle 3 : marge LDA − brut sur le backbone retenu
    r = res[chosen]
    margin = mapr(chosen, r, "lda128") - mapr(chosen, r, "brut_acp128")
    if margin >= dec["margin_comfortable"]:
        verdict = "marge confortable : on continue"
    elif margin >= dec["margin_minimum"]:
        verdict = "marge faible : on continue, pertes risquant d'être indistinguables"
    else:
        verdict = "pas de marge : arrêt et discussion"
    print(f"Règle 3 — marge {chosen} {r} px : {mapr(chosen, r, 'lda128'):.2f} − "
          f"{mapr(chosen, r, 'brut_acp128'):.2f} = {margin:.2f} points → {verdict}")
    report["rule3_margin"] = {"margin": margin, "verdict": verdict}
    report["selected"] = {"backbone": chosen, "resolution": r}

    out = new_run_dir(cfg["runs_dir"], "2e_decision")
    write_json(out / "decision.json", {**report, "run": run_info()})
    print(f"\nRETENU : {chosen}, {r} px   (détails : {out}/decision.json)")


if __name__ == "__main__":
    main()
