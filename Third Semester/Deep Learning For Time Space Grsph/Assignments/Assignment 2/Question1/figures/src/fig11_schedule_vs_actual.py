"""Figure 11: scheduled vs actual release (simplified)."""
import matplotlib.pyplot as plt
import numpy as np

from style import CORAL, OCHRE, TINT, save, setup, stamp

setup()
t = np.linspace(10, 14, 400)
sched = np.where(t >= 12, 110, 40)
actual = np.interp(t, [10, 12.33, 12.67, 14], [40, 40, 90, 90])
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.axvspan(12.0, 14, color=TINT[CORAL])
ax.plot(t, sched, "--", color=OCHRE, label="scheduled (X_fut, announced 10:30)")
ax.plot(t, actual, color=OCHRE, label="actual (X_rel, once recorded)")
ax.axvline(10.5, color="#8a8f98", lw=1)
ax.text(10.55, 120, "schedule\nannounced", fontsize=9, va="top", color="#5b616b")
ax.text(13.0, 60, "mismatch → flag\n+ widen band", ha="center", fontsize=9.5, color=CORAL)
ax.set_xticks([10, 11, 12, 13, 14])
ax.set_xticklabels(["10:00", "11:00", "12:00", "13:00", "14:00"])
ax.set_ylim(0, 130)
ax.set_ylabel("release (m³/s)")
ax.set_title("Schedule vs actual: train on schedule, switch to actual once recorded", loc="left", fontsize=10.5)
ax.legend(loc="lower left", fontsize=9)
stamp(ax, loc="lower right")
save(fig, "fig11_schedule_vs_actual.png")
