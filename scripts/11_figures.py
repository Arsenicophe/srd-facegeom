"""Figures d'une évaluation (scripts/10_evaluate.py) : aucune image de visage, versionnables.

- histogrammes de similarité : brut, LDA et chaque perte (graine 0) ;
- plan alignement / uniformité ;
- MAP@R et TAR à FAR 1 % avec IC à 95 %.

    python scripts/11_figures.py --eval runs/<date>_5_eval_val
    python scripts/11_figures.py --eval runs/<date>_5_eval_test --en --out-dir docs/figures
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from facegeom.viz import (
    EN_LABELS,
    NAMES_EN,
    NAMES_FR,
    align_uniform_plane,
    metric_intervals,
    similarity_histograms,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", required=True, help="dossier 5_eval_<split>")
    parser.add_argument("--out-dir", default=None, help="défaut : <eval>/figures")
    parser.add_argument("--en", action="store_true", help="étiquettes en anglais (poster)")
    args = parser.parse_args()

    src = Path(args.eval)
    out = Path(args.out_dir or src / "figures")
    split = src.name.rsplit("_", 1)[-1]
    per_seed = pd.read_csv(src / "per_seed.csv")
    summary = pd.read_csv(src / "summary.csv")
    scores = np.load(src / "scores_seed0.npz")
    y = scores["y"]
    names = NAMES_EN if args.en else NAMES_FR
    mapr = summary[summary.metric == "map_at_r"].set_index("method")["mean"]

    panels = [{"title": names.get(m, m), "same": scores[m][y == 1].astype(np.float32),
               "diff": scores[m][y == 0].astype(np.float32),
               "note": f"MAP@R {mapr[m]:.1f}{'%' if args.en else ' %'}"}
              for m in scores.files if m != "y"]
    kw = {"labels": EN_LABELS} if args.en else {}
    figs = [
        similarity_histograms(panels, out / f"{split}_histograms.png", ncols=3, panel_size=(3.6, 2.6), dpi=200,
                              sharey=False, **kw),
        align_uniform_plane(per_seed, out / f"{split}_align_uniform.png", en=args.en),
        metric_intervals(summary, "map_at_r", out / f"{split}_map_at_r.png",
                         "MAP@R (%), mean and 95% CI over seeds" if args.en
                         else "MAP@R (%), moyenne et IC à 95 % sur les graines", en=args.en),
        metric_intervals(summary, "tar_at_far_0.01", out / f"{split}_tar_far1e-2.png",
                         "TAR at FAR = 1% (%), mean and 95% CI over seeds" if args.en
                         else "TAR à FAR = 1 % (%), moyenne et IC à 95 % sur les graines", en=args.en),
    ]
    for f in figs:
        print("figure :", f)


if __name__ == "__main__":
    main()
