"""Étapes 3 et 4 : entraîne la tête pour une perte, en réglage ou en graines finales.

Un dossier par run dans runs/, jamais écrasé : configuration, paramètres,
historique par époque, meilleure tête (head.pt). Validation seulement.

    python scripts/09_train.py --loss arcface --tune                      # 5 tirages, graine 100 
    python scripts/09_train.py --loss arcface --params-from runs/<tune>/best.json --seeds 0 1 2 3 4
    python scripts/09_train.py --loss arcface --params '{"lr": 1e-3, "s": 32, "m": 0.3}' --seeds 0
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

from facegeom.features import load_xy
from facegeom.train import sample_params, train
from facegeom.utils import default_device, load_config, new_run_dir, run_info, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--loss", required=True, choices=["nsoftmax", "arcface", "contrastive", "triplet"])
    parser.add_argument("--tune", action="store_true", help="réglage : n_trials tirages, graine tune.seed")
    parser.add_argument("--params", default=None, help="paramètres JSON (lr + ceux de la perte)")
    parser.add_argument("--params-from", default=None, help="best.json d'un run de réglage")
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="défaut : seeds de la config")
    parser.add_argument("--features-root", default=None)
    parser.add_argument("--max-epochs", type=int, default=None, help="essai court")
    parser.add_argument("--device", default=default_device())
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["features_root"] = args.features_root or cfg["features_root"]
    cfg["max_epochs"] = args.max_epochs or cfg["max_epochs"]
    if "test" in (cfg["fit_split"], cfg["eval_split"]):
        raise SystemExit("L'entraînement ne doit jamais lire le test.")
    split_dir = load_config(args.data_config)["splits"]["output_dir"]
    b, r = cfg["backbone"], cfg["resolution"]
    xtr, ytr, _ = load_xy(cfg["features_root"], b, r, cfg["fit_split"], split_dir)
    xva, yva, _ = load_xy(cfg["features_root"], b, r, cfg["eval_split"], split_dir)
    print(f"{args.loss} : {len(xtr)} vecteurs d'entraînement, {len(xva)} de validation, {args.device}")

    if args.tune:
        trials = sample_params(cfg["losses"][args.loss]["params"], cfg["tune"]["n_trials"], cfg["tune"]["seed"])
        jobs = [(p, cfg["tune"]["seed"]) for p in trials]
        out = new_run_dir(cfg["runs_dir"], f"3_tune_{args.loss}")
    else:
        params = json.loads(args.params) if args.params else json.loads(Path(args.params_from).read_text())["params"]
        jobs = [(params, s) for s in (args.seeds if args.seeds is not None else cfg["seeds"])]
        out = new_run_dir(cfg["runs_dir"], f"4_final_{args.loss}")
    write_json(out / "meta.json", {"config": cfg, "args": vars(args), "run": run_info(args.device)})

    rows = []
    for i, (params, seed) in enumerate(jobs):
        tag = f"trial{i:02d}" if args.tune else f"seed{seed}"
        print(f"--- {tag} : {params}")
        t = time.perf_counter()
        res = train(cfg, args.loss, params, seed, xtr, ytr, xva, yva, args.device,
                    log=lambda s: print("   ", s))
        d = out / tag
        d.mkdir()
        torch.save(res["head_state"], d / "head.pt")
        pd.DataFrame(res["history"]).to_csv(d / "history.csv", index=False)
        rows.append({"tag": tag, "seed": seed, **params, "best_epoch": res["best_epoch"],
                     "best_map_at_r": res["best_map_at_r"], "seconds": round(time.perf_counter() - t)})
        write_json(d / "result.json", rows[-1])
        pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)

    if args.tune:
        best = max(rows, key=lambda x: x["best_map_at_r"])
        write_json(out / "best.json", {"loss": args.loss, "params": jobs[int(best["tag"][-2:])][0],
                                       "best_map_at_r": best["best_map_at_r"], "tag": best["tag"]})
        print(f"\nMEILLEUR : {best['tag']}  MAP@R {best['best_map_at_r']:.2f}")
    print("Résultats dans", out)


if __name__ == "__main__":
    main()
