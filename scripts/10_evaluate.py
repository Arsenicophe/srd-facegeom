"""Étapes 4 et 5 : évalue les têtes finales des pertes, avec brut et LDA comme repères.

Par perte : moyenne et IC de Student sur les graines ; entre pertes : écarts
appariés graine par graine. Le seuil de vérification est choisi sur la
validation (point d'égale erreur). Le test ne peut être évalué qu'une fois :
il faut --final-test, toutes les pertes et graines, et aucun passage antérieur.

    python scripts/10_evaluate.py --runs runs/<date>_4_final_nsoftmax runs/<date>_4_final_arcface ...
    python scripts/10_evaluate.py --runs ... --split test --final-test
"""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from facegeom.eval.baselines import apply_projection, fit_lda, fit_pca
from facegeom.eval.evaluate import evaluate, make_eval_pairs
from facegeom.eval.stats import paired_diff, student_ci
from facegeom.eval.verification import eer_threshold
from facegeom.features import load_xy
from facegeom.train import embed, load_heads
from facegeom.utils import load_config, new_run_dir, run_info, write_json

METRICS = ["map_at_r", "p_at_1", "r_precision", "auc", "eer", "tar_at_far_0.1", "tar_at_far_0.01",
           "tar_at_far_0.001", "thr_balanced_acc", "alignment", "uniformity"]
MARKER = "TEST_CONSULTED.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--runs", nargs="+", required=True, help="dossiers 4_final_<perte>")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--final-test", action="store_true", help="obligatoire avec --split test")
    parser.add_argument("--features-root", default=None)
    parser.add_argument("--runs-dir", default=None)
    args = parser.parse_args()

    cfg, tcfg = load_config(args.config), load_config(args.train_config)
    root = args.features_root or tcfg["features_root"]
    runs_dir = Path(args.runs_dir or cfg["runs_dir"])
    split_dir = load_config(args.data_config)["splits"]["output_dir"]
    b, r = tcfg["backbone"], tcfg["resolution"]
    heads = load_heads(args.runs, tcfg)

    if args.split == "test":
        missing = [f"{loss}:{s}" for loss in tcfg["losses"] for s in tcfg["seeds"] if s not in heads.get(loss, {})]
        if not args.final_test:
            raise SystemExit("Le test n'est évalué qu'une fois : ajouter --final-test pour confirmer.")
        if missing:
            raise SystemExit(f"Test refusé : têtes manquantes {missing}. Il faut les 4 pertes et toutes les graines.")
        if (runs_dir / MARKER).exists():
            raise SystemExit(f"Test déjà consulté : {(runs_dir / MARKER).read_text()}")

    out = new_run_dir(runs_dir, f"5_eval_{args.split}")
    write_json(out / "meta.json", {"config": cfg, "args": vars(args), "run": run_info(),
                                   "heads": {k: sorted(v) for k, v in heads.items()}})

    # Méthodes : repères sans graine, puis une entrée par (perte, graine)
    xtr, ytr, _ = load_xy(root, b, r, "train", split_dir)
    dim = cfg["dim"]
    pca, lda = fit_pca(xtr, dim), fit_lda(xtr, ytr, dim)
    methods = {("brut_acp", None): lambda x: apply_projection(pca, x),
               ("lda", None): lambda x: apply_projection(lda, x)}
    for loss, by_seed in heads.items():
        for seed, head in sorted(by_seed.items()):
            methods[(loss, seed)] = lambda x, h=head: embed(h, torch.as_tensor(x))
    del xtr

    # Validation toujours (seuil), test seulement si demandé
    data = {}
    for split in ["val"] + (["test"] if args.split == "test" else []):
        x, y, _ = load_xy(root, b, r, split, split_dir)
        data[split] = (x, y, make_eval_pairs(y, cfg["verification"], cfg["uniformity"]))

    rows, seed0_scores = [], {}
    for (name, seed), project in methods.items():
        xv, yv, epv = data["val"]
        m, scores = evaluate(project(xv), yv, epv)
        thr = eer_threshold(epv["y"], scores)
        x, y, ep = data[args.split]
        m, scores = evaluate(project(x), y, ep, threshold=thr)
        rows.append({"method": name, "seed": seed, "threshold": thr, **m})
        if seed in (None, 0):
            seed0_scores[name] = scores.astype(np.float16)
        print(f"{name:12} {'' if seed is None else seed!s:>2} : MAP@R {m['map_at_r']:5.2f}  P@1 {m['p_at_1']:5.2f}  "
              f"AUC {m['auc']:5.2f}  TAR@1e-2 {m['tar_at_far_0.01']:5.2f}  align {m['alignment']:.3f}  "
              f"unif {m['uniformity']:.3f}")

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(out / "per_seed.csv", index=False)
    np.savez_compressed(out / "scores_seed0.npz", y=data[args.split][2]["y"], **seed0_scores)

    # Moyenne et IC par méthode
    summary = []
    for name, g in per_seed.groupby("method", sort=False):
        for metric in METRICS:
            v = g[metric].to_numpy(dtype=float)
            ci = student_ci(v) if len(v) > 1 else {"mean": v[0], "ci_low": np.nan, "ci_high": np.nan, "n": 1}
            summary.append({"method": name, "metric": metric, "mean": ci["mean"], "ci_low": ci["ci_low"],
                            "ci_high": ci["ci_high"], "n": ci["n"]})
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)

    # Écarts appariés entre pertes, sur les graines communes
    paired = []
    for a, c in combinations(heads, 2):
        common = sorted(set(heads[a]) & set(heads[c]))
        if len(common) < 2:
            continue
        ga = per_seed[per_seed.method == a].set_index("seed").loc[common]
        gc = per_seed[per_seed.method == c].set_index("seed").loc[common]
        for metric in METRICS:
            paired.append({"a": a, "b": c, "metric": metric, **paired_diff(ga[metric], gc[metric])})
    pd.DataFrame(paired).to_csv(out / "paired.csv", index=False)

    print(f"\n{args.split.upper()} — MAP@R, moyenne [IC 95 %]")
    for s in summary:
        if s["metric"] == "map_at_r":
            print(f"  {s['method']:12} {s['mean']:5.2f}  [{s['ci_low']:5.2f} ; {s['ci_high']:5.2f}]  n = {s['n']}")
    for p in paired:
        if p["metric"] == "map_at_r":
            print(f"  {p['a']} − {p['b']} : {p['diff']:+.2f} [{p['ci_low']:+.2f} ; {p['ci_high']:+.2f}] "
                  f"→ {'différentes' if p['different'] else 'non distinguables'}")

    if args.split == "test":
        write_json(runs_dir / MARKER, {"run": str(out), "date": run_info()["date"]})
        print(f"\nTest consulté : marqueur écrit dans {runs_dir / MARKER}. Aucune autre évaluation du test.")
    print("Résultats dans", out)


if __name__ == "__main__":
    main()
