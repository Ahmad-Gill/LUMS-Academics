"""Figure 6: missing vs delayed readings at issuance (simplified)."""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from style import GRAY, INK, MUTED, TEAL, TINT, ILLUS, save, setup

setup()
fig, ax = plt.subplots(figsize=(9, 2.6))
vals = [2.30, 2.32, 2.34, 2.36, 2.38, 2.41, 2.43, 2.45, 2.45, 2.45]
kind = ["ok", "ok", "ok", "miss", "miss", "ok", "ok", "ok", "late", "late"]
mask = [1, 1, 1, 0, 0, 1, 1, 1, 0, 0]
age = [0, 0, 0, 10, 20, 0, 0, 0, 10, 20]
for i, k in enumerate(kind):
    for r, txt in enumerate([f"{vals[i]:.2f}", str(mask[i]), str(age[i])]):
        y = 2 - r
        if k == "ok":
            fc, ec, ls = TINT[TEAL], TEAL, "-"
        elif k == "miss":
            fc, ec, ls = TINT[GRAY], GRAY, "-"
        else:
            fc, ec, ls = "white", GRAY, "--"
        ax.add_patch(Rectangle((i + 0.05, y + 0.05), 0.9, 0.9, fc=fc, ec=ec, lw=1.5, ls=ls))
        ax.text(i + 0.5, y + 0.5, txt, ha="center", va="center", fontsize=10,
                color=INK if k == "ok" else MUTED, fontweight="bold" if r == 1 else "normal")
for r, name in enumerate(["level", "mask", "age (min)"]):
    ax.text(-0.15, 2.5 - r, name, ha="right", va="center", fontsize=10)
ax.text(4.0, 3.2, "missing → filled", ha="center", fontsize=9.5, color=MUTED)
ax.text(9.9, 3.2, "delayed → not arrived yet", ha="right", fontsize=9.5, color=MUTED)
ax.axvline(10.0, color=INK, lw=2)
ax.text(10.1, -0.25, "issuance t", ha="left", va="center", fontsize=9.5, fontweight="bold")
ax.text(0, -0.3, ILLUS, fontsize=8, color=MUTED, style="italic", va="center")
ax.set_xlim(-2.0, 11.6)
ax.set_ylim(-0.6, 3.6)
ax.axis("off")
save(fig, "fig06_missing_delayed.png")
