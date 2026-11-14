"""Figures : planches d'images et histogrammes de similarité."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from PIL import Image

from facegeom.data.crop import crop_box, crop_face

COLORS = ["tab:red", "tab:orange", "tab:green", "tab:blue", "tab:purple"]

# Palette validée (dataviz, fond clair, daltonisme) : emplacements 1 et 2
SAME_COLOR = "#2a78d6"  # paires « même personne »
DIFF_COLOR = "#eb6834"  # paires « personnes différentes »
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


GOOD, BAD = "#0ca30c", "#d03b3b"  # couleurs d'état, toujours avec ✓ / ✗

# Une couleur et un marqueur par perte, toujours les mêmes (palette validée, emplacements 1 à 4)
LOSS_STYLE = {"nsoftmax": ("#2a78d6", "o"), "arcface": ("#eb6834", "s"),
              "contrastive": ("#1baf7a", "^"), "triplet": ("#eda100", "D")}
REF_STYLE = {"brut_acp": ("#8a8984", "X"), "lda": ("#52514e", "P")}  # repères sans entraînement, en gris
NAMES_FR = {"brut_acp": "brut (ACP 128)", "lda": "LDA 128"}
NAMES_EN = {"brut_acp": "raw (PCA 128)", "lda": "LDA 128"}
SURFACE = "#fcfcfb"


def _style_axes(ax) -> None:
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def _save(fig, out_path: str | Path, dpi: int, facecolor: str | None = None) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor=facecolor)
    plt.close(fig)
    return out_path



def neighbors_sheet(rows: list[dict], out_path: str | Path, title: str) -> Path:
    """Une ligne par requête : la requête puis ses k plus proches voisins.

    Chaque ligne : ``query`` (image), ``neighbors`` (liste d'images),
    ``correct`` (liste de booléens), ``sims`` (similarités).
    Bonne personne : cadre vert et ✓ ; mauvaise : cadre rouge et ✗.
    """
    k = len(rows[0]["neighbors"])
    fig, axes = plt.subplots(len(rows), k + 1, figsize=(1.45 * (k + 1), 1.75 * len(rows)))
    for i, row in enumerate(rows):
        ax = axes[i, 0]
        ax.imshow(row["query"])
        ax.set_xlabel("requête", fontsize=7, color=INK)
        for side in ax.spines.values():
            side.set_edgecolor(INK)
            side.set_linewidth(2.5)
        for j, (img, ok, s) in enumerate(zip(row["neighbors"], row["correct"], row["sims"]), start=1):
            ax = axes[i, j]
            ax.imshow(img)
            color = GOOD if ok else BAD
            for side in ax.spines.values():
                side.set_edgecolor(color)
                side.set_linewidth(2.5)
            ax.set_xlabel(f"{'✓' if ok else '✗'} {s:.2f}", fontsize=7, color=INK)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, out_path, 130, "#fcfcfb")


FR_LABELS = {"diff": "personnes différentes", "same": "même personne",
             "x": "similarité cosinus", "y": "densité"}
EN_LABELS = {"diff": "different people", "same": "same person",
             "x": "cosine similarity", "y": "density"}


def similarity_histograms(panels: list[dict], out_path: str | Path, ncols: int = 2,
                          labels: dict | None = None, panel_size=(5.2, 3.4),
                          dpi: int = 150, sharey: bool = True) -> Path:
    """Petits multiples : pour chaque panneau, histogrammes superposés des similarités.

    Chaque panneau est un dict ``title``, ``same`` (scores des paires même
    personne), ``diff`` (paires différentes) et ``note`` (texte court).
    Densités normalisées : les deux groupes n'ont pas la même taille.
    """
    import numpy as np

    labels = labels or FR_LABELS
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
                             sharex=True, sharey=sharey)
    axes = np.atleast_1d(axes).ravel()
    # Bornes communes couvrant toutes les données (aucune valeur coupée)
    lo = min(min(p["same"].min(), p["diff"].min()) for p in panels)
    hi = max(max(p["same"].max(), p["diff"].max()) for p in panels)
    bins = np.linspace(np.floor(lo * 10) / 10, np.ceil(hi * 10) / 10, 121)
    for ax, p in zip(axes, panels):
        for scores, color, label in [(p["diff"], DIFF_COLOR, labels["diff"]),
                                     (p["same"], SAME_COLOR, labels["same"])]:
            ax.hist(scores, bins=bins, density=True, histtype="stepfilled", alpha=0.35,
                    color=color, linewidth=0)
            ax.hist(scores, bins=bins, density=True, histtype="step", linewidth=2,
                    color=color, label=label)
        ax.set_title(p["title"], fontsize=10, color=INK, loc="left")
        if p.get("note"):  # à droite du titre, hors de la zone des barres
            ax.set_title(p["note"], fontsize=8, color=MUTED, loc="right")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        for side in ["left", "bottom"]:
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8, labelbottom=True)
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    for ax in axes[max(len(panels) - ncols, 0):len(panels)]:  # dernier panneau de chaque colonne
        ax.set_xlabel(labels["x"], fontsize=9, color=MUTED)
    for ax in axes[::ncols]:
        ax.set_ylabel(labels["y"], fontsize=9, color=MUTED)
    # Légende au-dessus de la figure : elle ne recouvre jamais les courbes
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, fontsize=9, ncol=2, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.32 / fig.get_figheight()))
    return _save(fig, out_path, dpi, "#fcfcfb")


def crop_contact_sheet(
    images: list[Image.Image],
    sizes: list[int],
    crop_cfg: dict,
    out_path: str | Path,
    display_px: int = 112,
) -> Path:
    """Planche : une ligne par image, l'original avec les carrés, puis chaque recadrage.

    Les recadrages sont tous affichés à ``display_px`` pour comparer le contenu,
    pas la résolution.
    """
    boxes = [crop_box(crop_cfg, s) for s in sizes]
    n = len(images)
    fig, axes = plt.subplots(n, len(sizes) + 1, figsize=(1.6 * (len(sizes) + 1), 1.9 * n))
    for i, img in enumerate(images):
        ax = axes[i, 0]
        ax.imshow(img)
        for box, color in zip(boxes, COLORS):
            l, t, r, b = box
            ax.add_patch(mpatches.Rectangle((l, t), r - l, b - t, fill=False, lw=1.2, ec=color))
        for j, (box, color) in enumerate(zip(boxes, COLORS), start=1):
            axes[i, j].imshow(crop_face(img, box).resize((display_px, display_px), Image.BICUBIC))
            for side in axes[i, j].spines.values():
                side.set_edgecolor(color)
                side.set_linewidth(2)
    axes[0, 0].set_title("original 178×218", fontsize=8)
    for j, (s, color) in enumerate(zip(sizes, COLORS), start=1):
        axes[0, j].set_title(f"carré {s} px", fontsize=8, color=color)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    return _save(fig, out_path, 130)


def align_uniform_plane(per_seed, out_path: str | Path, en: bool = False, dpi: int = 200) -> Path:
    """Plan alignement / uniformité : un petit point par graine, un gros par méthode (moyenne).

    ``per_seed`` : tableau ``method, seed, alignment, uniformity`` (scripts/10_evaluate.py).
    Les deux axes se lisent « plus bas = mieux » ; le coin bas-gauche est l'idéal.
    """
    names = NAMES_EN if en else NAMES_FR
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for method, g in per_seed.groupby("method", sort=False):
        color, marker = {**LOSS_STYLE, **REF_STYLE}.get(method, (MUTED, "o"))
        if len(g) > 1:
            ax.scatter(g.uniformity, g.alignment, s=14, color=color, marker=marker, alpha=0.45, linewidths=0)
        x, y = g.uniformity.mean(), g.alignment.mean()
        ax.scatter([x], [y], s=110, color=color, marker=marker, edgecolors=SURFACE, linewidths=1.2, zorder=3)
        ax.annotate(names.get(method, method), (x, y), xytext=(8, 6), textcoords="offset points",
                    fontsize=9, color=INK)
    ax.set_xlabel("uniformity  (lower = more spread out)" if en else "uniformité  (plus bas = mieux répartis)",
                  fontsize=9, color=MUTED)
    ax.set_ylabel("alignment  (lower = same person closer)" if en
                  else "alignement  (plus bas = même personne plus proche)", fontsize=9, color=MUTED)
    _style_axes(ax)
    ax.margins(0.18)
    fig.tight_layout()
    return _save(fig, out_path, dpi, SURFACE)


def metric_intervals(summary, metric: str, out_path: str | Path, xlabel: str, en: bool = False,
                     dpi: int = 200) -> Path:
    """Un point par méthode (moyenne sur les graines) avec son IC à 95 % ; repères en gris.

    ``summary`` : tableau ``method, metric, mean, ci_low, ci_high`` (scripts/10_evaluate.py).
    """
    names = NAMES_EN if en else NAMES_FR
    rows = summary[summary.metric == metric].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(5.6, 0.5 * len(rows) + 1.0))
    for i, r in rows.iterrows():
        color, marker = {**LOSS_STYLE, **REF_STYLE}.get(r.method, (MUTED, "o"))
        y = len(rows) - 1 - i
        if r.ci_low == r.ci_low:  # pas NaN : au moins 2 graines
            ax.plot([r.ci_low, r.ci_high], [y, y], color=color, linewidth=2, solid_capstyle="round")
        ax.scatter([r["mean"]], [y], s=90, color=color, marker=marker, edgecolors=SURFACE, linewidths=1.2, zorder=3)
        ax.annotate(f"{r['mean']:.1f}", (r["mean"], y), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([names.get(m, m) for m in rows.method[::-1]], fontsize=9, color=INK)
    ax.set_ylim(-0.6, len(rows) - 0.3)
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    _style_axes(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, out_path, dpi, SURFACE)
