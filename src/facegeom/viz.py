"""Figures d'inspection."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from facegeom.data.crop import crop_box, crop_face  # noqa: E402

COLORS = ["tab:red", "tab:orange", "tab:green", "tab:blue", "tab:purple"]

# Palette validée (dataviz, fond clair, daltonisme) : emplacements 1 et 2
SAME_COLOR = "#2a78d6"  # paires « même personne »
DIFF_COLOR = "#eb6834"  # paires « personnes différentes »
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def similarity_histograms(panels: list[dict], out_path: str | Path, ncols: int = 2) -> Path:
    """Petits multiples : pour chaque panneau, histogrammes superposés des similarités.

    Chaque panneau est un dict ``title``, ``same`` (scores des paires même
    personne), ``diff`` (paires différentes) et ``note`` (texte court).
    Densités normalisées : les deux groupes n'ont pas la même taille.
    """
    import numpy as np

    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.4 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    bins = np.linspace(-0.2, 1.0, 121)
    for ax, p in zip(axes, panels):
        for scores, color, label in [(p["diff"], DIFF_COLOR, "personnes différentes"),
                                     (p["same"], SAME_COLOR, "même personne")]:
            ax.hist(scores, bins=bins, density=True, histtype="stepfilled", alpha=0.35,
                    color=color, linewidth=0)
            ax.hist(scores, bins=bins, density=True, histtype="step", linewidth=2,
                    color=color, label=label)
        ax.set_title(p["title"], fontsize=10, color=INK, loc="left")
        if p.get("note"):
            ax.text(0.02, 0.95, p["note"], transform=ax.transAxes, fontsize=8, color=MUTED, va="top")
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
        ax.set_xlabel("similarité cosinus", fontsize=9, color=MUTED)
    for ax in axes[::ncols]:
        ax.set_ylabel("densité", fontsize=9, color=MUTED)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor="#fcfcfb")
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
    boxes = [crop_box(s, crop_cfg["center_x"], crop_cfg["eye_y"], crop_cfg["eye_rel"]) for s in sizes]
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
