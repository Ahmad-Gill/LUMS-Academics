"""Figure 3: three-way decomposition of the S3 level during a storm (simplified)."""
import matplotlib.pyplot as plt
import numpy as np

from style import CORAL, INK, PURPLE, TEAL, save, setup, stamp

setup()
t = np.linspace(0, 24, 300)
base = 1.10 - 0.002 * t
sig = lambda x: 1 / (1 + np.exp(-x))
rain = 0.42 * np.exp(-0.5 * ((t - 9.5) / 2.2) ** 2) * sig(2 * (t - 5.5)) 
hidden = 0.25 * np.exp(-0.5 * ((t - 11.5) / 2.5) ** 2) * sig(2 * (t - 7.5)) 

fig, ax = plt.subplots(figsize=(8, 4))
ax.fill_between(t, 0, base, color=TEAL, alpha=0.85, lw=0, label="base flow")
ax.fill_between(t, base, base + rain, color=PURPLE, alpha=0.8, lw=0, label="+ observed rain (g1·Δrain)")
ax.fill_between(t, base + rain, base + rain + hidden, color=CORAL, alpha=0.85, lw=0,
                label="+ hidden rain (g2·Δhidden)")
ax.plot(t, base + rain + hidden, color=INK, lw=2, label="total level")
ax.set_ylim(0, 2.0)
ax.set_xlim(0, 24)
ax.set_xlabel("hours")
ax.set_ylabel("S3 level (m)")
ax.set_title("S3 level = base + observed rain + hidden rain", loc="left")
ax.legend(loc="upper right", fontsize=9)
stamp(ax, loc="lower left")
save(fig, "fig03_three_way_decomposition.png")
