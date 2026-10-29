"""Figure 10: qualitative influence of each factor on each station (simplified)."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from style import save, setup

setup()
rows = ["Reservoir release", "S1 (upstream)", "T1 (tributary)", "R1 rain", "R2 rain",
        "Unmonitored trib. (U)", "Left-bank rain"]
M = np.array([[3, 2, 1, 1], [3, 3, 2, 1], [0, 3, 2, 1], [3, 2, 1, 1], [0, 0, 3, 2],
              [0, 0, 3, 2], [1, 3, 2, 1]])
cmap = LinearSegmentedColormap.from_list("t", ["#f3f4f6", "#00574f"])
fig, ax = plt.subplots(figsize=(6, 4.6))
ax.imshow(M, cmap=cmap, vmin=0, vmax=3, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, M[i, j], ha="center", va="center", fontsize=12,
                color="white" if M[i, j] >= 2 else "#1f2328", fontweight="bold")
ax.set_xticks(range(4))
ax.set_xticklabels(["S1", "S2", "S3", "S4"], fontsize=11)
ax.xaxis.tick_top()
ax.set_yticks(range(len(rows)))
ax.set_yticklabels(rows)
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
ax.set_xlabel("0 = none · 3 = strong   (qualitative, from river topology)", fontsize=9)
save(fig, "fig10_influence_matrix.png")
