"""Figure 8: gates during a gauged storm and an ungauged left-bank storm (simplified)."""
import matplotlib.pyplot as plt
import numpy as np

from style import CORAL, PURPLE, TEAL, TINT, save, setup, stamp

setup()
t = np.linspace(0, 24, 300)
sig = lambda x: 1 / (1 + np.exp(-x))
t1 = 0.8 + 0.08 * np.exp(-0.5 * ((t - 6.5) / 1.5) ** 2) + 0.6 * np.exp(-0.5 * ((t - 16) / 1.6) ** 2)
g1 = 0.05 + 0.85 * sig(3 * (t - 3.5)) * sig(-1.5 * (t - 8.5))
g2 = 0.05 + 0.8 * sig(3 * (t - 15)) * sig(-1.5 * (t - 19))

fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 4.8), sharex=True)
for a in (a1, a2):
    a.axvspan(3, 5.5, color=TINT[PURPLE])
    a.axvspan(13, 15.5, color=TINT[CORAL])
a1.plot(t, t1, color=TEAL)
a1.text(4.25, 1.7, "R1 storm\n(gauged)", ha="center", va="top", fontsize=9, color=PURPLE)
a1.text(11.0, 1.7, "left-bank storm\n(no gauge)", ha="center", va="top", fontsize=9, color=CORAL)
a1.set_ylabel("T1 level (m)")
a1.set_ylim(0.7, 1.95)
a1.set_title("(a) T1 rises strongly in the ungauged storm", loc="left")
a2.plot(t, g1, color=PURPLE, label="g1 (observed rain)")
a2.plot(t, g2, color=CORAL, label="g2 (hidden rain)")
a2.set_ylabel("gate for S2")
a2.set_xlabel("hours")
a2.set_ylim(0, 1.05)
a2.set_title("(b) g1 opens with gauged rain, g2 with T1's surprise", loc="left")
a2.legend(loc="upper right", fontsize=9)
stamp(a1, loc="upper right")
save(fig, "fig08_gates_storm.png")
