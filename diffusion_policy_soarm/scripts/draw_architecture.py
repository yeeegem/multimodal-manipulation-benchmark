"""Generate the README architecture diagram (architecture.png).

Run:
    uv run python -m diffusion_policy_soarm.scripts.draw_architecture

Output:
    diffusion_policy_soarm/docs/architecture.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT_PATH = Path(__file__).resolve().parents[1] / "docs" / "architecture.png"

# Color palette (subtle, print-friendly).
COL_INPUT = "#E8F1FB"
COL_ENC = "#CFE3FA"
COL_CAT = "#FFD9A8"
COL_DEN = "#D2E8D2"
COL_AUX = "#EADCF7"
COL_OUT = "#FCE4E4"
EDGE = "#2F2F2F"


def box(ax, x, y, w, h, text, color, fontsize=10, weight="normal"):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=color,
        edgecolor=EDGE,
        linewidth=1.4,
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight=weight,
        wrap=True,
    )


def arrow(ax, x1, y1, x2, y2, label=None, label_offset=(0.0, 0.12), style="-|>", curve=0.0):
    connection = f"arc3,rad={curve}" if curve else "arc3"
    a = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle=style,
        mutation_scale=14,
        linewidth=1.4,
        color=EDGE,
        connectionstyle=connection,
    )
    ax.add_patch(a)
    if label:
        mx = (x1 + x2) / 2 + label_offset[0]
        my = (y1 + y2) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=8.5, style="italic")


def main() -> None:
    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(
        8,
        8.55,
        "Diffusion Policy on SO-ARM101  -  data flow per inference call",
        ha="center",
        va="center",
        fontsize=14,
        weight="bold",
    )
    ax.text(
        8,
        8.15,
        "T_o=2 observation frames  ->  2176-D conditioning  ->  DDIM denoising  ->  16-step action chunk in degrees",
        ha="center",
        va="center",
        fontsize=10,
        style="italic",
        color="#444",
    )

    # ---- Column 1: inputs ----
    box(ax, 0.3, 6.0, 2.4, 1.3, "front camera\n480 x 640 x 3 uint8\n(resized to 96 x 96)", COL_INPUT)
    box(ax, 0.3, 4.2, 2.4, 1.3, "wrist camera\n480 x 640 x 3 uint8\n(resized to 96 x 96)", COL_INPUT)
    box(ax, 0.3, 2.4, 2.4, 1.3, "state 6-D\njoint angles (deg)\nT_o = 2 frames", COL_INPUT)

    # ---- Column 2: per-modality encoders ----
    box(ax, 3.6, 6.0, 2.4, 1.3, "ResNet18\n(ImageNet init)\n+ Linear(512)", COL_ENC)
    box(ax, 3.6, 4.2, 2.4, 1.3, "ResNet18\n(ImageNet init)\n+ Linear(512)", COL_ENC)
    box(ax, 3.6, 2.4, 2.4, 1.3, "Linear(6 -> 64)\n+ SiLU", COL_ENC)

    arrow(ax, 2.7, 6.65, 3.6, 6.65, "(2, 3, 96, 96)")
    arrow(ax, 2.7, 4.85, 3.6, 4.85, "(2, 3, 96, 96)")
    arrow(ax, 2.7, 3.05, 3.6, 3.05, "(2, 6)")

    # Note: shared weights across T_o
    ax.text(
        4.8,
        7.45,
        "weights shared across the 2 obs frames per camera",
        ha="center",
        va="center",
        fontsize=8.5,
        style="italic",
        color="#666",
    )

    # ---- Column 3: concat ----
    box(ax, 7.2, 4.2, 1.6, 1.3, "concat\nover cams\nand time\n2176-D", COL_CAT, weight="bold")
    arrow(ax, 6.0, 6.65, 7.2, 5.2, "(2, 512)")
    arrow(ax, 6.0, 4.85, 7.2, 4.85, "(2, 512)")
    arrow(ax, 6.0, 3.05, 7.2, 4.5, "(2, 64)")

    # ---- Column 4: U-Net ----
    box(
        ax,
        9.6,
        2.2,
        3.4,
        4.5,
        "1-D temporal U-Net\nFiLM conditioning\n\nchannels 256 / 512 / 1024\nkernel 5, GroupNorm 8\n\n47M params",
        COL_DEN,
        weight="bold",
    )

    # cond -> U-Net
    arrow(ax, 8.8, 4.85, 9.6, 4.85, "cond emb (512)")

    # noise input from above
    box(ax, 9.6, 7.3, 3.4, 1.0, "x_t  (noise sample, shape 16 x 6)", COL_AUX)
    arrow(ax, 11.3, 7.3, 11.3, 6.7)

    # t embedding from below
    box(ax, 9.6, 0.6, 3.4, 1.0, "t embedding  (sinusoid -> MLP -> 512)", COL_AUX)
    arrow(ax, 11.3, 1.6, 11.3, 2.2)

    # ---- Column 5: output and DDIM loop ----
    box(
        ax,
        13.4,
        5.6,
        2.4,
        1.4,
        "DDIM step\nx_t -> x_{t-1}\nclip to [-1, 1]",
        COL_AUX,
    )

    # epsilon-hat: from right side of U-Net up into the DDIM step box.
    arrow(ax, 13.0, 5.6, 13.4, 6.3, "epsilon-hat (16, 6)", label_offset=(0.0, 0.18))

    # Loop: from DDIM step back into x_t (top of U-Net column) for the next pass.
    arrow(ax, 14.0, 7.0, 11.3, 7.3, curve=0.35)
    ax.text(
        12.6,
        7.85,
        "loops N times (default 10)",
        ha="center",
        va="center",
        fontsize=8.5,
        style="italic",
        color="#444",
    )

    box(
        ax,
        13.4,
        2.4,
        2.4,
        1.6,
        "actions\n(16, 6) in deg\nabsolute joint angles\nexecute first 8 then replan",
        COL_OUT,
        weight="bold",
    )
    # After the final iteration, denormalise the clean prediction to degrees.
    arrow(ax, 14.6, 5.6, 14.6, 4.0, "denormalise")

    # Legend
    legend_items = [
        ("Observation inputs", COL_INPUT),
        ("Encoders (trainable, end-to-end)", COL_ENC),
        ("Conditioning vector", COL_CAT),
        ("Denoiser / sampler", COL_DEN),
        ("Auxiliary tensors", COL_AUX),
        ("Final actions", COL_OUT),
    ]
    handles = [mpatches.Patch(facecolor=c, edgecolor=EDGE, label=l) for l, c in legend_items]
    ax.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
        fontsize=9,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
