"""Figure 4: what is available at issuance time t (simplified, schematic)."""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from style import GRAY, INK, MUTED, NAVY, OCHRE, PURPLE, TEAL, TINT, save, setup

setup()
fig, ax = plt.subplots(figsize=(9, 3.6))
# schematic axis: history 0..6 (24 h, compressed), future 6..8 (60 min)
H0, T, F = 0.0, 6.0, 8.0
rows = [("Station levels", TEAL, "past"), ("Rain R1, R2", PURPLE, "past"),
        ("Past releases", OCHRE, "past"), ("Announced releases", OCHRE, "future"),
        ("Future rain", GRAY, "none"), ("Future levels (target)", NAVY, "target")]
for i, (name, c, kind) in enumerate(rows):
    y = len(rows) - 1 - i
    if kind == "past":
        ax.add_patch(Rectangle((H0, y - 0.3), T - H0, 0.6, fc=c, ec="none"))
    elif kind == "future":
        ax.add_patch(Rectangle((T, y - 0.3), F - T, 0.6, fc=TINT[c], ec=c, lw=1.5, hatch=".."))
    elif kind == "none":
        ax.add_patch(Rectangle((T, y - 0.3), F - T, 0.6, fc="white", ec=GRAY, lw=1.2, ls="--"))
        ax.text((T + F) / 2, y, "not available", ha="center", va="center", fontsize=9, color=MUTED)
    else:
        ax.plot(F, y, marker="*", ms=16, color=c)
        ax.text(F - 0.15, y, "label in training only", ha="right", va="center", fontsize=9, color=MUTED)
    ax.text(-0.15, y, name, ha="right", va="center", fontsize=10, color=INK)

ax.axvline(T, color=INK, lw=2)
ax.text(T, len(rows) - 0.35, "issuance t", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_xticks([H0, T, F])
ax.set_xticklabels(["t − 24 h", "t", "t + 60 min"])
ax.set_yticks([])
ax.set_xlim(-0.05, F + 0.3)
ax.set_ylim(-0.6, len(rows) - 0.1)
ax.grid(False)
ax.spines["left"].set_visible(False)
ax.text(T / 2, -0.55, "history window: 144 steps of 10 min (not to scale)", ha="center",
        va="bottom", fontsize=8.5, color=MUTED)
save(fig, "fig04_issuance_timeline.png")
