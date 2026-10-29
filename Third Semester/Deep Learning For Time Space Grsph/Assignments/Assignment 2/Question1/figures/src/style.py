"""Shared style for all Question 1 figures.

One colour per concept, used in every figure:
    base flow / station levels / learned backbone -> TEAL
    observed (gauged) rain                        -> PURPLE
    hidden (ungauged) rain / unmonitored inflow   -> CORAL
    fixed / physics / static structure            -> GRAY
    reservoir release                             -> OCHRE
    forecast output                               -> NAVY
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

TEAL = "#00988a"
PURPLE = "#7b52c1"
CORAL = "#e0603f"
GRAY = "#8a8f98"
OCHRE = "#b8860b"
NAVY = "#2f5d8a"

INK = "#1f2328"
MUTED = "#5b616b"
GRID = "#e4e6ea"

# Light tints for box fills (same hue, high lightness).
TINT = {
    TEAL: "#dcf1ee",
    PURPLE: "#ece5f7",
    CORAL: "#fbe4dc",
    GRAY: "#eef0f2",
    OCHRE: "#f6ecd2",
    NAVY: "#dfe8f2",
}

ILLUS = "Illustrative — synthetic data, not results"

OUT_DIR = Path(__file__).resolve().parents[1]


def setup():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "lines.linewidth": 2,
    })


def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.15)
    plt.close(fig)
    print(f"wrote {path}")


def stamp(ax, loc="upper right", fig=None):
    """Write the 'illustrative' disclaimer inside a plot."""
    x, ha = (0.99, "right") if "right" in loc else (0.01, "left")
    y, va = (0.98, "top") if "upper" in loc else (0.02, "bottom")
    ax.text(x, y, ILLUS, transform=ax.transAxes, ha=ha, va=va, fontsize=8,
            color=MUTED, style="italic", zorder=20,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=GRID, lw=0.8))


def box(ax, x, y, w, h, text, color, fontsize=9, dashed=False, bold_first=True,
        fill=None, lw=1.6):
    """Rounded box centred at (x, y). First line bold if bold_first."""
    patch = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                           boxstyle="round,pad=0.02,rounding_size=0.12",
                           fc=fill if fill is not None else TINT.get(color, "white"),
                           ec=color, lw=lw, ls="--" if dashed else "-", zorder=3)
    ax.add_patch(patch)
    lines = text.split("\n")
    if bold_first and len(lines) > 1:
        ax.text(x, y + h / 2 - 0.14, lines[0], ha="center", va="top",
                fontsize=fontsize + 0.5, fontweight="bold", color=INK, zorder=4)
        ax.text(x, y + h / 2 - 0.14 - 0.28 * (fontsize / 9) - 0.02,
                "\n".join(lines[1:]), ha="center", va="top", fontsize=fontsize,
                color=INK, zorder=4, linespacing=1.3)
    else:
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                fontweight="bold" if bold_first else "normal", color=INK, zorder=4,
                linespacing=1.3)
    return patch


def arrow(ax, p0, p1, color=MUTED, label=None, label_offset=(0.08, 0), lw=1.4,
          dashed=False, label_ha="left", fontsize=8, rad=0.0, label_color=None):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=12, color=color,
                        lw=lw, ls="--" if dashed else "-", zorder=2,
                        connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    if label:
        mx = (p0[0] + p1[0]) / 2 + label_offset[0]
        my = (p0[1] + p1[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, ha=label_ha, va="center", fontsize=fontsize,
                color=label_color or INK, family="DejaVu Sans Mono", zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
    return a
