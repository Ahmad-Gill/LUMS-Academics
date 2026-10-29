"""Figure 1: architecture of the Three-way Decomposed River-Graph Forecaster (simplified)."""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from style import CORAL, GRAY, NAVY, OCHRE, PURPLE, TEAL, arrow, box, save, setup

setup()
fig, ax = plt.subplots(figsize=(10, 11))
ax.set_xlim(0, 13.3)
ax.set_ylim(0.3, 15.0)
ax.axis("off")

# ---- row 1: inputs
Y1 = 14.0
box(ax, 1.7, Y1, 2.8, 0.85, "Station levels\n[B,144,5,3]", TEAL)
box(ax, 4.8, Y1, 2.8, 0.85, "Rain R1, R2\n[B,144,2,3]", PURPLE)
box(ax, 7.9, Y1, 2.8, 0.85, "Past releases\n[B,144,1]", OCHRE)
box(ax, 11.6, Y1, 2.9, 0.85, "Announced releases\n[B,6,1]", OCHRE)

# ---- row 2: alignment + projection
Y2 = 12.1
box(ax, 5.3, Y2, 9.8, 0.85,
    "Align to 10-min grid · project rain onto nodes (P) · add mask + age\n"
    "→ node tensor X  [B, 144, 7 nodes, 8 features]", GRAY)
for x in (1.7, 4.8, 7.9):
    arrow(ax, (x, Y1 - 0.43), (x, Y2 + 0.43))

# ---- row 3: shared GRU
Y3 = 10.3
box(ax, 5.3, Y3, 9.8, 0.85, "Shared GRU (same weights for every node)\nH  [B, 144, 7, d]", TEAL)
arrow(ax, (5.3, Y2 - 0.43), (5.3, Y3 + 0.43))

# ---- row 4: directed lag attention
Y4 = 8.5
box(ax, 5.3, Y4, 9.8, 0.85,
    "Directed lag attention (downstream edges only, learned delays)\nZ  [B, 7, d]", TEAL)
arrow(ax, (5.3, Y3 - 0.43), (5.3, Y4 + 0.43))

# ---- row 5: release fusion
Y5 = 6.7
box(ax, 5.3, Y5, 9.8, 0.85, "Release fusion at reservoir node + 1 graph pass\nZ  [B, 7, d] → keep S1–S4  [B, 4, d]", OCHRE)
arrow(ax, (5.3, Y4 - 0.43), (5.3, Y5 + 0.43))
ax.add_line(Line2D([11.6, 11.6], [Y1 - 0.43, Y5], color=OCHRE, lw=1.4))
arrow(ax, (11.6, Y5), (10.2, Y5), color=OCHRE)

# ---- row 6: three heads
Y6 = 4.6
box(ax, 1.9, Y6, 3.4, 0.85, "Base flow\nΔbase  [B,4,3]", TEAL)
box(ax, 5.9, Y6, 3.4, 0.85, "Observed rain\ng1 · Δrain  (≥ 0)", PURPLE)
box(ax, 9.9, Y6, 3.4, 0.85, "Hidden rain\ng2 · Δhidden  (≥ 0)", CORAL)
for x in (1.9, 5.9, 9.9):
    arrow(ax, (min(x, 10.0), Y5 - 0.43), (x, Y6 + 0.43))
ax.text(11.85, Y6 - 0.85, "g2 opens on surprise:\nobserved − expected level", fontsize=8,
        color=CORAL, ha="center", va="center")

# ---- row 7: combine -> output
Y7 = 2.2
box(ax, 5.9, Y7, 9.6, 0.85,
    "Forecast  ŷ = y_last + Δbase + g1·Δrain + g2·Δhidden\n"
    "ŷ  [B, 4 stations, 3 quantiles]  ·  level at t + 60 min", NAVY)
for x in (1.9, 5.9, 9.9):
    arrow(ax, (x, Y6 - 0.43), (x, Y7 + 0.43))

ax.text(6.6, 0.75, "B = batch of issuance times · 144 = 24 h at 10 min · "
        "7 nodes = Res, S1, T1, S2, U, S3, S4 · d = hidden size",
        ha="center", va="center", fontsize=8.5, color="#5b616b")

save(fig, "fig01_architecture.png")
