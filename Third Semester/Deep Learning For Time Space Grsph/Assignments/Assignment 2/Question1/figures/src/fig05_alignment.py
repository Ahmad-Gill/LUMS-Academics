"""Figure 5: alignment onto one 10-min grid (simplified)."""
import matplotlib.pyplot as plt
import numpy as np

from style import GRAY, MUTED, PURPLE, TEAL, save, setup, stamp

setup()
fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
ticks = np.arange(0, 61, 10)
# (a) rain: irregular reports -> summed per bin (tau-10, tau]
rt = np.array([2, 9, 14, 21, 22, 36, 41, 48, 56])
rv = np.array([0.2, 0.6, 0.4, 0.8, 0.2, 1.0, 0.6, 0.4, 0.2])
bins = [rv[(rt > b - 10) & (rt <= b)].sum() for b in ticks[1:]]
a1.bar(ticks[1:] - 5, bins, width=8, color=PURPLE, alpha=0.25, label="sum per 10-min bin")
a1.stem(rt, rv, linefmt=PURPLE, markerfmt="o", basefmt=" ", label="raw R1 reports")
a1.set_ylabel("rain (mm)")
a1.set_ylim(0, 1.35)
a1.set_title("(a) rain: irregular reports → summed per bin", loc="left")
a1.legend(loc="upper left", fontsize=9)
stamp(a1)
# (b) level: jittered reports, one missing -> snapped + interpolated
lt = np.array([1, 11, 18, 31, 50, 60])
lv = np.array([2.31, 2.32, 2.34, 2.37, 2.43, 2.46])
a2.plot(lt, lv, "x", color=MUTED, ms=8, label="raw reports (jitter)")
grid_v = np.interp(ticks, [0, 10, 20, 30, 50, 60], [2.31, 2.32, 2.34, 2.37, 2.43, 2.46])
ok = ticks != 40
a2.plot(ticks, grid_v, color=TEAL, lw=1, alpha=0.6)
a2.plot(ticks[ok], grid_v[ok], "o", color=TEAL, ms=8, label="on grid, mask = 1")
a2.plot(40, grid_v[~ok][0], "o", mfc="white", mec=GRAY, mew=2, ms=9, label="missing → filled, mask = 0")
a2.set_ylabel("S2 level (m)")
a2.set_title("(b) levels: snapped to grid (±5 min), gap filled", loc="left")
a2.legend(loc="upper left", fontsize=9)
a2.set_xticks(ticks)
a2.set_xticklabels([f"08:{m:02d}" if m < 60 else "09:00" for m in ticks])
for a in (a1, a2):
    for x in ticks:
        a.axvline(x, color="#d0d3d8", lw=0.8, ls=":", zorder=0)
save(fig, "fig05_alignment.png")
