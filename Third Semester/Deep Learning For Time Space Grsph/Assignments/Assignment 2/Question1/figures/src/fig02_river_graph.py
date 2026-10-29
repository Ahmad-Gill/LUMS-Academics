"""Figure 2: the 7-node directed river graph (simplified)."""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from style import CORAL, GRAY, INK, OCHRE, PURPLE, TEAL, TINT, save, setup

setup()
fig, ax = plt.subplots(figsize=(8, 8.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10.6)
ax.set_aspect("equal")
ax.axis("off")

R = 0.45
pos = {"Res": (4.0, 9.4), "S1": (4.0, 7.7), "T1": (1.4, 6.0), "S2": (4.0, 5.6),
       "U": (7.0, 4.4), "S3": (4.0, 3.5), "S4": (4.0, 1.5)}

fig.canvas.draw()
PPU = (ax.transData.transform((1, 0))[0] - ax.transData.transform((0, 0))[0]) * 72 / fig.dpi


def edge(a, b, color=TEAL, ls="-", lw=2.4, src=None):
    pa = src if src is not None else pos[a]
    shrink_a = (0.45 if src is not None else R) * PPU + 2
    ax.add_patch(FancyArrowPatch(pa, pos[b], arrowstyle="-|>", mutation_scale=16,
                                 color=color, lw=lw, ls=ls, shrinkA=shrink_a,
                                 shrinkB=R * PPU + 3, zorder=2))


for a, b in [("Res", "S1"), ("S1", "S2"), ("T1", "S2"), ("S2", "S3"), ("U", "S3"), ("S3", "S4")]:
    edge(a, b)

# nodes
for n, (x, y) in pos.items():
    if n == "Res":
        ax.add_patch(FancyBboxPatch((x - 0.65, y - 0.38), 1.3, 0.76,
                                    boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=TINT[OCHRE], ec=OCHRE, lw=2, zorder=3))
    elif n == "U":
        ax.add_patch(Circle((x, y), R, fc="white", ec=CORAL, lw=2, ls="--", zorder=3))
    else:
        ax.add_patch(Circle((x, y), R, fc=TINT[TEAL], ec=TEAL, lw=2, zorder=3))
    ax.text(x, y, n, ha="center", va="center", fontsize=12, fontweight="bold",
            color=INK, zorder=4)

# rain gauges + requested gauge
gauges = {"R1": ((7.4, 8.4), PURPLE, "-"), "R2": ((8.6, 6.2), PURPLE, "-"),
          "Left gauge?": ((1.4, 8.6), CORAL, "--")}
for name, ((x, y), c, ls) in gauges.items():
    w = 1.0 if len(name) < 4 else 1.9
    ax.add_patch(FancyBboxPatch((x - w / 2, y - 0.3), w, 0.6,
                                boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc=TINT[c] if ls == "-" else "white", ec=c, lw=2, ls=ls, zorder=3))
    ax.text(x, y, name, ha="center", va="center", fontsize=10.5, fontweight="bold",
            color=INK, zorder=4)

edge(None, "S1", PURPLE, ":", 1.8, src=(7.4, 8.4))
ax.add_patch(FancyArrowPatch((6.9, 8.6), (4.7, 9.3), arrowstyle="-|>", mutation_scale=12, color=PURPLE, lw=1.8, ls=":", zorder=2))
edge(None, "U", PURPLE, ":", 1.8, src=(8.6, 6.2))
edge(None, "S3", PURPLE, ":", 1.8, src=(8.6, 6.2))
edge(None, "T1", CORAL, "--", 1.8, src=(1.4, 8.6))

# short side labels
ax.text(1.4, 5.25, "monitored\ntributary", ha="center", va="top", fontsize=9, color="#5b616b")
ax.text(7.0, 3.7, "unmonitored tributary\n(virtual node, no sensor)", ha="center", va="top",
        fontsize=9, color=CORAL)
ax.text(3.2, 9.4, "reservoir\n+ release", ha="right", va="center", fontsize=9, color="#5b616b")
ax.text(4.6, 1.5, "→ downstream town", ha="left", va="center", fontsize=9, color="#5b616b")

handles = [Line2D([], [], color=TEAL, lw=2.4, label="river edge (downstream only, distance d_ij)"),
           Line2D([], [], color=PURPLE, lw=1.8, ls=":", label="rain projection P (fixed)"),
           Line2D([], [], color=CORAL, lw=1.8, ls="--", label="no sensor / requested")]
ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, -0.06), fontsize=9)

save(fig, "fig02_river_graph.png")
