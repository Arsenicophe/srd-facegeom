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
