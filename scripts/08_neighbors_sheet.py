"""Étape 2f : planche des k plus proches voisins (brut puis LDA) sur la validation, contrôle du style.

Visages CelebA : reste dans runs/, non versionné.

    python scripts/08_neighbors_sheet.py --zip data/celeba/img_align_celeba.zip
"""

import argparse

import numpy as np

from facegeom.data.crop import crop_box, crop_face
from facegeom.data.images import open_celeba_zip, read_image
from facegeom.eval.baselines import apply_projection, fit_lda, fit_pca
from facegeom.eval.retrieval import l2_normalize
from facegeom.features import load_xy
from facegeom.utils import load_config, new_run_dir, run_info, write_json
from facegeom.viz import neighbors_sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--features-root", default=None, help="remplace features_root (Drive sur Colab)")
    parser.add_argument("--zip", required=True)
    parser.add_argument("--n-queries", type=int, default=10)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["features_root"] = args.features_root or cfg["features_root"]
    dcfg = load_config(args.data_config)
    if cfg["eval_split"] == "test":
        raise SystemExit("La planche ne doit jamais utiliser le test.")
    b, r = cfg["selected"]["backbone"], cfg["selected"]["resolution"]
    split_dir = dcfg["splits"]["output_dir"]
    xtr, ytr, _ = load_xy(cfg["features_root"], b, r, cfg["fit_split"], split_dir)
    xva, yval, names = load_xy(cfg["features_root"], b, r, cfg["eval_split"], split_dir)
    variants = {
        "brut (ACP 128)": apply_projection(fit_pca(xtr, cfg["dim"]), xva),
        "LDA 128": apply_projection(fit_lda(xtr, ytr, cfg["dim"]), xva),
    }

    # Requêtes : une image par identité, identités tirées au hasard
    rng = np.random.default_rng(args.seed)
    ids = rng.choice(np.unique(yval), size=args.n_queries, replace=False)
    queries = [int(rng.choice(np.flatnonzero(yval == i))) for i in ids]

    c = dcfg["crop"]
    box = crop_box(c)
    zf = open_celeba_zip(args.zip)

    def img(i):
        return crop_face(read_image(zf, names[i]), box)

    out = new_run_dir(cfg["runs_dir"], "2f_planche_voisins")
    summary = {}
    for v, x in variants.items():
        z = l2_normalize(x)
        rows, n_ok = [], 0
        for q in queries:
            sims = z @ z[q]
            sims[q] = -np.inf
            nn = np.argsort(-sims)[: args.k]
            ok = [bool(yval[j] == yval[q]) for j in nn]
            n_ok += sum(ok)
            rows.append({"query": img(q), "neighbors": [img(j) for j in nn],
                         "correct": ok, "sims": [float(sims[j]) for j in nn]})
        tag = "lda" if v.startswith("LDA") else "brut"
        neighbors_sheet(rows, out / f"planche_{tag}.png",
                        f"Barlow Twins 224 px, {v} — validation, {args.k} plus proches voisins")
        summary[v] = f"{n_ok}/{len(queries) * args.k} voisins de la bonne personne"
        print(f"{v:15} : {summary[v]}")

    write_json(out / "meta.json", {"backbone": b, "resolution": r, "queries": [names[q] for q in queries],
                                   "summary": summary, "seed": args.seed, "run": run_info()})
    print("Planches dans", out)


if __name__ == "__main__":
    main()
