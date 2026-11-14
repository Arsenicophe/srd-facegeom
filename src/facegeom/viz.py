"""Figures d'inspection."""

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
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, facecolor="#fcfcfb")
    plt.close(fig)
    return out_path


FR_LABELS = {"diff": "personnes différentes", "same": "même personne",
             "x": "similarité cosinus", "y": "densité"}
EN_LABELS = {"diff": "different people", "same": "same person",
             "x": "cosine similarity", "y": "density"}


def similarity_histograms(panels: list[dict], out_path: str | Path, ncols: int = 2,
                          labels: dict | None = None, panel_size=(5.2, 3.4),
                          dpi: int = 150) -> Path:
    """Petits multiples : pour chaque panneau, histogrammes superposés des similarités.

    Chaque panneau est un dict ``title``, ``same`` (scores des paires même
    personne), ``diff`` (paires différentes) et ``note`` (texte court).
    Densités normalisées : les deux groupes n'ont pas la même taille.
    """
    import numpy as np

    labels = labels or FR_LABELS
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
                             sharex=True, sharey=True)
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
        ax.tick_params(colors=MUTED, labelsize=8)
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    for ax in axes[-ncols:]:
        ax.set_xlabel(labels["x"], fontsize=9, color=MUTED)
    for ax in axes[::ncols]:
        ax.set_ylabel(labels["y"], fontsize=9, color=MUTED)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="#fcfcfb")
    plt.close(fig)
    return out_path


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
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
