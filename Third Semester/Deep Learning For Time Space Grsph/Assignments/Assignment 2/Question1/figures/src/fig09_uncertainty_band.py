"""Figure 9: forecast band widens with missing data or hidden rain (simplified)."""
import matplotlib.pyplot as plt
import numpy as np

from style import CORAL, GRAY, INK, NAVY, TINT, save, setup, stamp

setup()
t = np.linspace(0, 18, 300)
obs = 2.1 + 0.52 * np.exp(-0.5 * ((t - 13) / 1.4) ** 2)
med = 2.1 + 0.47 * np.exp(-0.5 * ((t - 13.3) / 1.5) ** 2)
w = 0.05 + 0.05 * ((t > 4) & (t < 7)) + 0.14 * np.exp(-0.5 * ((t - 12.8) / 1.6) ** 2)

fig, ax = plt.subplots(figsize=(8, 3.8))
ax.axvspan(4, 7, color=TINT[GRAY])
ax.axvspan(11, 15, color=TINT[CORAL])
ax.fill_between(t, med - w, med + w, color=NAVY, alpha=0.18, lw=0, label="10–90 % band")
ax.plot(t, med, color=NAVY, label="median forecast")
ax.plot(t, obs, "--", color=INK, lw=1.5, label="observed later")
ax.text(5.5, 2.75, "T1 offline", ha="center", fontsize=9)
ax.text(13, 2.75, "g2 high", ha="center", fontsize=9, color=CORAL)
ax.set_ylim(1.95, 2.85)
ax.set_xlim(0, 18)
ax.set_xlabel("hours")
ax.set_ylabel("S2 level at t + 60 (m)")
ax.set_title("Wider band when data is missing or hidden rain is suspected", loc="left")
ax.legend(loc="upper left", fontsize=9)
stamp(ax, loc="lower right")
save(fig, "fig09_uncertainty_band.png")
