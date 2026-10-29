"""Figure 7: learned delay shifts with flow (simplified)."""
import matplotlib.pyplot as plt
import numpy as np

from style import TEAL, save, setup, stamp

setup()
k = np.arange(0, 121, 10)


def prof(mu, s):
    p = np.exp(-0.5 * ((k - mu) / s) ** 2)
    return p / p.sum()


fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(k, prof(80, 18), "-o", color="#7fcfc4", ms=4, label="low flow (slow)")
ax.plot(k, prof(45, 11), "-o", color="#00574f", ms=4, label="high flow (fast)")
ax.annotate("", xy=(60, 0.25), xytext=(82, 0.25), arrowprops=dict(arrowstyle="->", lw=1.5))
ax.text(71, 0.26, "shorter travel time", ha="center", va="bottom", fontsize=9)
ax.set_xlabel("lag k (minutes before t)")
ax.set_ylabel("attention weight α")
ax.set_title("Attention on edge S1 → S2: delay depends on flow", loc="left")
ax.set_ylim(0, 0.36)
ax.legend(loc="upper left", fontsize=9)
stamp(ax, loc="upper right")
save(fig, "fig07_lag_attention.png")
