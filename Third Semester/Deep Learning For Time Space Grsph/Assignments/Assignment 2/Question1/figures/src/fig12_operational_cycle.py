"""Figure 12: the 10-minute operating loop (simplified)."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from style import GRAY, MUTED, NAVY, TEAL, arrow, box, save, setup

setup()
fig, ax = plt.subplots(figsize=(10, 2.6))
ax.set_xlim(0, 13.3)
ax.set_ylim(-0.3, 3.0)
ax.axis("off")
steps = [("1. Ingest", "new readings", GRAY), ("2. Mask + age", "mark gaps, delays", GRAY),
         ("3. Build tensor", "X [1,144,7,8]", GRAY), ("4. Forward pass", "≪ 1 s on CPU", TEAL),
         ("5. Publish", "ŷ [1,4,3] + flags", NAVY)]
xs = [1.3, 3.975, 6.65, 9.325, 12.0]
for (t1, t2, c), x in zip(steps, xs):
    box(ax, x, 2.0, 2.2, 0.85, f"{t1}\n{t2}", c)
for a, b in zip(xs[:-1], xs[1:]):
    arrow(ax, (a + 1.1, 2.0), (b - 1.1, 2.0))
ax.add_patch(FancyArrowPatch((12.0, 1.57), (1.3, 1.57), arrowstyle="-|>", mutation_scale=12,
                             color=MUTED, lw=1.4, connectionstyle="arc3,rad=-0.08"))
ax.text(6.65, -0.1, "repeat every 10 min · never waits for late data", ha="center",
        fontsize=9.5, color=MUTED)
save(fig, "fig12_operational_cycle.png")
