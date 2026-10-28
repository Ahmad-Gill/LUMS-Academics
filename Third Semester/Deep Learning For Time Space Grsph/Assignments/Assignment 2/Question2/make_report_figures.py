"""Generate the placeholder diagrams (D*) and supplementary plots (P*) for report.md.

Run from Question2/:  ../.venv/bin/python -I <this file>
Writes figures/diagrams/*.png and figures/plots/*.png. Data plots read the saved checkpoints
(training histories) and the test split; every number matches the notebook tables.
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

os.makedirs("figures/diagrams", exist_ok=True)
os.makedirs("figures/plots", exist_ok=True)

# reference palette (light), validated: blue, orange, aqua, yellow, magenta
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
LIGHT_BLUE, LIGHT_ORANGE, LIGHT_GRAY = "#cde2fb", "#fbd9c9", "#eeede9"
MODEL_COL = {"A": BLUE, "B": ORANGE, "C": AQUA, "D": YELLOW, "sym-D": MAGENTA}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 200, "savefig.bbox": "tight",
})


# ---------- drawing helpers for block diagrams ----------
def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax

def box(ax, x, y, w, h, text, fc="white", ec=INK2, fs=9, bold=False, color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color,
            fontweight="bold" if bold else "normal")

def arrow(ax, x1, y1, x2, y2, text=None, color=INK2, fs=8, style="-|>", ls="-", rad=0.0, toff=(0.08, 0)):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=1.2, ls=ls,
                                connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
    if text:
        ax.text((x1 + x2) / 2 + toff[0], (y1 + y2) / 2 + toff[1], text, fontsize=fs, color=MUTED, va="center")

def save(fig, name):
    fig.savefig(name); plt.close(fig); print("wrote", name)


# ---------- D1: study roadmap ----------
fig, ax = canvas(12, 3.6)
steps = [("1. Setup", "Moving MNIST\nsplit by video\nM, R, speed-up\nTable 1, Figure 1", LIGHT_GRAY),
         ("2. Baseline", "Model A\nConvLSTM\nTable 2, Figure 2", LIGHT_BLUE),
         ("3. Stress tests", "mirror / rotation\n2× / 3× speed\nTables 3–4\nFigures 3–5", LIGHT_ORANGE),
         ("4. Diagnosis", "B: symmetry aug.\nC: speed aug.\nD: larger reach\nTable 5", LIGHT_BLUE),
         ("5. Final model", "symmetrised D\nexact C_M = C_R = 0\nTable 6, Figure 6", LIGHT_ORANGE)]
for i, (title, body, fc) in enumerate(steps):
    x = 0.3 + i * 2.38
    box(ax, x, 0.35, 2.0, 2.3, body, fc=fc, fs=9)
    ax.text(x + 1.0, 2.95, title, ha="center", fontsize=11, fontweight="bold", color=INK)
    if i < len(steps) - 1:
        arrow(ax, x + 2.02, 1.5, x + 2.36, 1.5)
save(fig, "figures/diagrams/D1_q2_roadmap.png")


# ---------- D2: symmetry group G on a real test frame ----------
d = np.load("data/MovingMNIST/mnist_test_seq.npy", mmap_mode="r")      # (20, 10000, 64, 64)
frame = torch.from_numpy(np.array(d[0, 9946])).float() / 255           # test video 946, frame 1
rot = lambda x, a: torch.rot90(x, a, (-2, -1)); mir = lambda x: x.flip(-1)
G = [(f"R$^{a}$" if a else "R$^0$ = identity", rot(frame, a)) for a in range(4)] + \
    [(f"MR$^{a}$" if a else "M", mir(rot(frame, a))) for a in range(4)]
fig, axs = plt.subplots(2, 4, figsize=(9, 4.8))
for a, (name, img) in zip(axs.flat, G):
    a.imshow(img, cmap="gray", vmin=0, vmax=1); a.set_title(name, fontsize=11, color=INK)
    a.set_xticks([]); a.set_yticks([]); a.grid(False)
    for s in a.spines.values(): s.set_visible(True); s.set_color(MUTED)
axs[0, 0].set_ylabel("rotations R$^a$", fontsize=10, color=INK2)
axs[1, 0].set_ylabel("rotate, then mirror", fontsize=10, color=INK2)
fig.suptitle("The 8 symmetries of the square applied to test video 946, frame 1", fontsize=11, color=INK)
plt.tight_layout(); save(fig, "figures/diagrams/D2_symmetry_group.png")


# ---------- D3: windows at 1x, 2x, 3x (t = 1) ----------
fig, ax = canvas(12, 3.2)
for r, k in enumerate([1, 2, 3]):
    y = 2.3 - r * 0.9
    ax.text(0.35, y + 0.27, f"{k}×", fontsize=12, fontweight="bold", ha="center", va="center")
    inp = {1 + i * k for i in range(5)}; tgt = {1 + i * k for i in range(5, 10) if 1 + i * k <= 20}
    for f in range(1, 21):
        fc = BLUE if f in inp else ORANGE if f in tgt else LIGHT_GRAY
        tc = "white" if f in inp | tgt else MUTED
        ax.add_patch(Rectangle((0.7 + (f - 1) * 0.52, y), 0.46, 0.54, fc=fc, ec="white", lw=2))
        ax.text(0.93 + (f - 1) * 0.52, y + 0.27, str(f), ha="center", va="center", fontsize=9, color=tc)
    note = {1: "t ∈ {1..11} (A, B) · {1..14} (C, D)", 2: "t ∈ {1..8} (C, D)", 3: "t ∈ {1, 2} (C, D) · only 2 targets"}[k]
    ax.text(11.25, y + 0.27, note, fontsize=8.5, va="center", ha="left", color=INK2)
for i, (lab, c) in enumerate([("input X", BLUE), ("target Y", ORANGE), ("unused", LIGHT_GRAY)]):
    ax.add_patch(Rectangle((0.7 + i * 2.0, 0.08), 0.3, 0.22, fc=c, ec="none"))
    ax.text(1.08 + i * 2.0, 0.19, lab, va="center", fontsize=9, color=INK2)
ax.set_xlim(0, 14.6)
save(fig, "figures/diagrams/D3_windows_speeds.png")


# ---------- D4: ConvLSTM cell ----------
fig, ax = canvas(11, 4.6)
box(ax, 0.2, 3.2, 1.5, 0.7, "u$_τ$\n(C$_{in}$×16×16)", fc=LIGHT_GRAY)
box(ax, 0.2, 1.9, 1.5, 0.7, "h$_{τ-1}$\n(64×16×16)", fc=LIGHT_GRAY)
box(ax, 0.2, 0.5, 1.5, 0.7, "c$_{τ-1}$\n(64×16×16)", fc=LIGHT_GRAY)
box(ax, 2.3, 2.5, 1.4, 0.9, "concat\n[u, h]", fc="white")
arrow(ax, 1.7, 3.55, 2.3, 3.1); arrow(ax, 1.7, 2.25, 2.3, 2.8)
box(ax, 4.2, 2.5, 1.9, 0.9, "Conv 3×3\n(C$_{in}$+64) → 4·64\nbias, pad 1", fc=LIGHT_BLUE, bold=True)
arrow(ax, 3.7, 2.95, 4.2, 2.95)
gates = [("i", "σ"), ("f", "σ"), ("o", "σ"), ("g", "tanh")]
for j, (g, act) in enumerate(gates):
    y = 3.75 - j * 0.85
    box(ax, 6.7, y, 1.1, 0.6, f"{act}({g})", fc=LIGHT_ORANGE)
    arrow(ax, 6.1, 2.95, 6.7, y + 0.3)
box(ax, 8.4, 1.3, 2.4, 1.1, "c$_τ$ = σ(f)⊙c$_{τ-1}$\n      + σ(i)⊙tanh(g)", fc="white", fs=9)
box(ax, 8.4, 3.0, 2.4, 0.9, "h$_τ$ = σ(o)⊙tanh(c$_τ$)", fc="white", fs=9)
arrow(ax, 7.8, 3.75 - 0 * 0.85 + 0.3, 8.4, 1.95); arrow(ax, 7.8, 3.75 - 1 * 0.85 + 0.3, 8.4, 1.85)
arrow(ax, 7.8, 3.75 - 3 * 0.85 + 0.3, 8.4, 1.6); arrow(ax, 7.8, 3.75 - 2 * 0.85 + 0.3, 8.4, 3.45)
arrow(ax, 1.7, 0.85, 8.4, 1.35, rad=0.22)
ax.text(5.0, 0.62, "c$_{τ-1}$ (cell state, element-wise only)", fontsize=8, color=MUTED, ha="center")
arrow(ax, 9.6, 2.4, 9.6, 3.0)
ax.text(5.5, 0.15, "One convolution produces all four gates; no peephole terms. ⊙ = element-wise product.",
        fontsize=8.5, color=MUTED, ha="center")
save(fig, "figures/diagrams/D4_convlstm_cell.png")


# ---------- D5 / D7: model A and model D architecture ----------
def model_diagram(name, layer2_text, layer2_fc, params2, fname, subtitle):
    fig, ax = canvas(9.5, 7.2)
    rows = [("input frame u$_τ$", "1×64×64", LIGHT_GRAY, ""),
            ("PixelUnshuffle(4)", "16×16×16", "white", "0 params"),
            ("ConvLSTM layer 1\n3×3 conv, 64 hidden", "64×16×16", LIGHT_BLUE, "184,576 params"),
            (layer2_text, "64×16×16", layer2_fc, params2),
            ("Conv 1×1, 64 → 16", "16×16×16", "white", "1,040 params"),
            ("PixelShuffle(4) + sigmoid", "1×64×64", "white", "0 params"),
            ("prediction x̂$_{τ+1}$", "1×64×64", LIGHT_GRAY, "")]
    for i, (txt, shape, fc, prm) in enumerate(rows):
        y = 6.3 - i * 0.98
        box(ax, 2.6, y, 3.4, 0.72, txt, fc=fc, fs=9.5, bold="ConvLSTM" in txt)
        ax.text(6.55, y + 0.36, shape, fontsize=9, color=INK2, va="center")
        ax.text(2.4, y + 0.36, prm, fontsize=8, color=MUTED, va="center", ha="right")
        if i < len(rows) - 1:
            arrow(ax, 4.3, y, 4.3, y - 0.26)
        if "ConvLSTM" in txt:
            arrow(ax, 6.0, y + 0.55, 6.0, y + 0.17, style="-|>", rad=-1.6, color=ORANGE)
            ax.text(7.75, y + 0.36, "(h, c) → step τ+1", fontsize=8, color=ORANGE, va="center")
    ax.text(4.75, 0.05, f"Model {name}: {subtitle}   ·   total 480,784 trainable parameters",
            ha="center", fontsize=9.5, color=INK)
    save(fig, fname)

model_diagram("A", "ConvLSTM layer 2\n3×3 conv, 64 hidden", LIGHT_BLUE, "295,168 params",
              "figures/diagrams/D5_model_A.png", "reach 2 cells = 8 px per step")
model_diagram("D", "ConvLSTM layer 2\n3×3 conv, DILATION 2, pad 2", LIGHT_ORANGE, "295,168 params",
              "figures/diagrams/D7_model_D.png", "only change = dilated layer 2 · reach 3 cells = 12 px")


# ---------- D6: 9-step rollout ----------
fig, ax = canvas(12.5, 3.8)
for s in range(1, 10):
    x = 0.4 + (s - 1) * 1.33
    keep = s >= 5
    box(ax, x, 1.45, 1.05, 0.9, f"model\nτ = {s}", fc=LIGHT_BLUE if keep else LIGHT_GRAY, fs=9)
    inp = f"x$_{s}$" if s <= 5 else f"x̂$_{s}$"
    ax.text(x + 0.52, 0.55, inp, ha="center", fontsize=10, color=INK if s <= 5 else ORANGE)
    arrow(ax, x + 0.52, 0.8, x + 0.52, 1.45)
    out = f"x̂$_{{{s + 1}}}$"
    ax.text(x + 0.52, 3.05, out, ha="center", fontsize=10, color=INK if keep else MUTED)
    arrow(ax, x + 0.52, 2.35, x + 0.52, 2.85, color=INK2 if keep else MUTED)
    if s < 9:
        arrow(ax, x + 1.05, 1.9, x + 1.33, 1.9, color=ORANGE)
    if s >= 6:
        ax.text(x + 0.52, 0.28, f"= output of τ={s - 1}", ha="center", fontsize=7, color=ORANGE)
ax.text(0.4, 3.55, "outputs of steps 1–4 discarded", fontsize=8.5, color=MUTED)
ax.text(5.73, 3.55, "outputs of steps 5–9 = Ŷ = (x̂$_6$ … x̂$_{10}$)", fontsize=8.5, color=INK)
ax.text(0.4, 0.1, "steps 1–5: true input frames", fontsize=8.5, color=INK2)
ax.text(7.05, 0.1, "steps 6–9: own previous prediction (no teacher forcing, gradients flow through)",
        fontsize=8.5, color=ORANGE)
save(fig, "figures/diagrams/D6_rollout.png")


# ---------- D8: reach of A vs D on the 16x16 cell grid ----------
fig, axs = plt.subplots(1, 2, figsize=(10, 4.9))
for a, (title, offs1, offs2) in zip(axs, [("Model A: 3×3 → 3×3", [-1, 0, 1], [-1, 0, 1]),
                                          ("Model D: 3×3 → 3×3 dilation 2", [-1, 0, 1], [-2, 0, 2])]):
    n = 9; c = n // 2
    level = np.full((n, n), 0)
    for dy1 in offs1:
        for dx1 in offs1:
            level[c + dy1, c + dx1] = max(level[c + dy1, c + dx1], 2)
            for dy2 in offs2:
                for dx2 in offs2:
                    yy, xx = c + dy1 + dy2, c + dx1 + dx2
                    level[yy, xx] = max(level[yy, xx], 1)
    level[c, c] = 3
    cmap = {0: "white", 1: "#9ec5f4", 2: "#3987e5", 3: "#0d366b"}
    for yy in range(n):
        for xx in range(n):
            a.add_patch(Rectangle((xx, n - 1 - yy), 1, 1, fc=cmap[level[yy, xx]], ec=GRID, lw=1))
    a.set_xlim(0, n); a.set_ylim(0, n); a.set_aspect("equal"); a.axis("off")
    r = max(offs1) + max(offs2)
    a.set_title(f"{title}\nreach {r} cells = {4 * r} px (axis), {4 * r * 2 ** 0.5:.1f} px (diagonal)",
                fontsize=10, color=INK)
handles = [Rectangle((0, 0), 1, 1, fc=c, ec=GRID) for c in ["#0d366b", "#3987e5", "#9ec5f4"]]
fig.legend(handles, ["source cell (1 cell = 4×4 px)", "after layer 1", "after layer 2 (one time step)"],
           loc="lower center", ncol=3, frameon=False, fontsize=9)
plt.tight_layout(rect=(0, 0.07, 1, 1)); save(fig, "figures/diagrams/D8_reach_A_vs_D.png")


# ---------- D9: training pipelines of A-D ----------
fig, ax = canvas(12.5, 4.6)
cols = [("speed k", 2.3), ("start t", 4.15), ("window X, Y", 6.0), ("symmetry g", 7.85), ("architecture", 9.7)]
for name, x in cols:
    ax.text(x + 0.8, 4.2, name, ha="center", fontsize=10, fontweight="bold", color=INK)
spec = [("A", "k = 1", "t ∈ {1..11}", "eq. (2)", "none", "baseline"),
        ("B", "k = 1", "t ∈ {1..11}", "eq. (2)", "g ~ U(G),\nsame g on X, Y", "baseline"),
        ("C", "k ~ U{1,2,3}", "t ~ allowed(k)", "eq. (2) + mask", "none", "baseline"),
        ("D", "k ~ U{1,2,3}", "t ~ allowed(k)", "eq. (2) + mask", "none", "dilated layer 2")]
for r, (m, k, t, w, g, arch) in enumerate(spec):
    y = 3.25 - r * 0.9
    ax.text(0.9, y + 0.32, f"Model {m}", fontsize=11, fontweight="bold", color=MODEL_COL[m], va="center")
    for (name, x), txt in zip(cols, [k, t, w, g, arch]):
        changed = (m == "B" and name == "symmetry g") or (m in "CD" and name in ("speed k", "start t", "window X, Y")) \
                  or (m == "D" and name == "architecture")
        box(ax, x, y, 1.6, 0.64, txt, fc=LIGHT_ORANGE if changed else "white", fs=8.5)
        if name != "architecture":
            arrow(ax, x + 1.6, y + 0.32, x + 1.85, y + 0.32)
ax.text(6.25, 0.1, "Shaded = what differs from model A. All: Adam 1e-3, batch 32, 30 epochs, masked MSE, seeds 0 and 1.",
        ha="center", fontsize=8.5, color=MUTED)
save(fig, "figures/diagrams/D9_training_pipelines.png")


# ---------- D10: commutative square for C_T and E_T ----------
fig, ax = canvas(7.5, 4.6)
box(ax, 0.4, 3.4, 1.4, 0.7, "X", fc=LIGHT_GRAY, fs=11)
box(ax, 5.2, 3.4, 1.7, 0.7, "f(X)", fc=LIGHT_BLUE, fs=11)
box(ax, 0.4, 0.6, 1.4, 0.7, "T(X)", fc=LIGHT_GRAY, fs=11)
box(ax, 5.2, 1.4, 1.7, 0.6, "T(f(X))", fc=LIGHT_BLUE, fs=10)
box(ax, 5.2, 0.3, 1.7, 0.6, "f(T(X))", fc=LIGHT_ORANGE, fs=10)
arrow(ax, 1.8, 3.75, 5.2, 3.75, "f", toff=(0, 0.18))
arrow(ax, 1.1, 3.4, 1.1, 1.3, "T", toff=(0.12, 0))
arrow(ax, 6.05, 3.4, 6.05, 2.0, "T", toff=(0.12, 0))
arrow(ax, 1.8, 0.6, 5.2, 0.6, "f", toff=(0, 0.18))
ax.annotate("", xy=(7.0, 1.7), xytext=(7.0, 0.6),
            arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.4))
ax.text(7.1, 1.15, "C$_T$", color=ORANGE, fontsize=11, va="center")
ax.text(3.5, 0.08, "E$_T$ = MSE(f(T(X)), T(Y))   ·   C$_T$ = MSE(f(T(X)), T(f(X)))", ha="center",
        fontsize=9, color=INK2)
ax.set_xlim(0, 7.8)
save(fig, "figures/diagrams/D10_commutative_square.png")


# ---------- D11: symmetrised forecaster ----------
fig, ax = canvas(11, 5.4)
box(ax, 0.2, 2.35, 1.0, 0.7, "X", fc=LIGHT_GRAY, fs=11)
names = ["R$^0$", "R$^1$", "R$^2$", "R$^3$", "MR$^0$", "MR$^1$", "MR$^2$", "MR$^3$"]
inv = ["R$^0$", "R$^{-1}$", "R$^{-2}$", "R$^{-3}$", "MR$^0$", "MR$^1$", "MR$^2$", "MR$^3$"]
for i, (g, gi) in enumerate(zip(names, inv)):
    y = 4.75 - i * 0.62
    box(ax, 2.0, y, 1.2, 0.46, f"g = {g}", fc="white", fs=8.5)
    box(ax, 4.0, y, 1.6, 0.46, "f (model D)", fc=LIGHT_BLUE, fs=8.5)
    box(ax, 6.4, y, 1.4, 0.46, f"g⁻¹ = {gi}", fc="white", fs=8.5)
    arrow(ax, 1.2, 2.7, 2.0, y + 0.23); arrow(ax, 3.2, y + 0.23, 4.0, y + 0.23)
    arrow(ax, 5.6, y + 0.23, 6.4, y + 0.23); arrow(ax, 7.8, y + 0.23, 8.5, 2.7)
box(ax, 8.5, 2.3, 1.1, 0.8, "mean\n(÷ 8)", fc=LIGHT_ORANGE, fs=9.5, bold=True)
box(ax, 9.9, 2.35, 0.95, 0.7, "f̄(X)", fc=LIGHT_GRAY, fs=11)
arrow(ax, 9.6, 2.7, 9.9, 2.7)
ax.text(5.5, 0.05, "f̄(T(X)) = T(f̄(X)) exactly for every T ∈ G, whatever f learned  ·  costs 8 forward passes",
        ha="center", fontsize=9, color=INK2)
save(fig, "figures/diagrams/D11_symmetrised_forecaster.png")


# =================== supplementary data plots ===================
def label_end(ax, x, y, text, color, dy=0):
    ax.text(x, y + dy, " " + text, color=INK2, fontsize=9, va="center")
    ax.plot([x], [y], "o", ms=5, color=color, mec="white", mew=1.5)

# ---------- P1: training curves (mean of seeds 0 and 1) ----------
hist = {m: [torch.load(f"checkpoints/{m}_s{s}.pt", map_location="cpu")["history"] for s in (0, 1)] for m in "ABCD"}
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for key, a, title in [("train_loss", axs[0], "Training loss (on each model's own training windows)"),
                      ("val_E5", axs[1], "Validation E(5) at 1×, t = 1")]:
    finals = {m: np.mean([h[key][-1] for h in hist[m]]) for m in "ABCD"}
    for pair in ("AB", "CD"):                       # nudge the two close end-labels apart
        hi, lo = sorted(pair, key=lambda m: -finals[m])
        dys = {hi: 0.0005, lo: -0.0005}
        for m in pair:
            runs = np.array([h[key] for h in hist[m]]); ep = np.arange(1, runs.shape[1] + 1)
            a.fill_between(ep, runs.min(0), runs.max(0), color=MODEL_COL[m], alpha=0.15, lw=0)
            a.plot(ep, runs.mean(0), color=MODEL_COL[m], lw=2)
            a.plot([ep[-1]], [finals[m]], "o", ms=5, color=MODEL_COL[m], mec="white", mew=1.5)
            a.text(ep[-1] + 0.4, finals[m] + dys[m], f"Model {m}", color=INK2, fontsize=8.5, va="center")
    a.set(title=title, xlabel="epoch", ylabel="MSE", xlim=(1, 35.5))
    a.title.set_fontsize(10)
fig.text(0.5, -0.02, "Lines: mean of seeds 0 and 1; band: range across the two seeds.", ha="center",
         fontsize=8.5, color=MUTED)
plt.tight_layout(); save(fig, "figures/plots/P1_training_curves.png")

# ---------- P2: error per forecast horizon (Table 2) ----------
h = np.arange(1, 6)
A = np.array([[0.0177, 0.0213, 0.0237, 0.0256, 0.0275], [0.0165, 0.0197, 0.0219, 0.0237, 0.0255]])
blank = [0.0426, 0.0426, 0.0426, 0.0425, 0.0425]; copy = [0.0492, 0.0604, 0.0669, 0.0708, 0.0732]
fig, ax = plt.subplots(figsize=(7, 4))
ax.fill_between(h, A.min(0), A.max(0), color=BLUE, alpha=0.15, lw=0)
ax.plot(h, A.mean(0), color=BLUE, lw=2, marker="o", ms=6, mec="white", mew=1.5)
ax.plot(h, blank, color=MUTED, lw=1.5, ls="--"); ax.plot(h, copy, color=INK2, lw=1.5, ls=":")
label_end(ax, 5, A.mean(0)[-1], "Model A", BLUE)
ax.text(5.05, blank[-1], "Blank", color=INK2, fontsize=9, va="center")
ax.text(5.05, copy[-1], "Copy-last", color=INK2, fontsize=9, va="center")
ax.set(xlabel="forecast horizon h", ylabel="MSE$_h$ (test, 1×, t = 1)", xticks=h, xlim=(0.8, 5.9), ylim=(0, 0.08))
ax.set_title("Error grows with the horizon (Table 2)", fontsize=10)
plt.tight_layout(); save(fig, "figures/plots/P2_error_per_horizon.png")

# ---------- P3: E(2) vs speed (Tables 4-6) ----------
sp = [1, 2, 3]
E2 = {"A": [0.0188, 0.0432, 0.0478], "C": [0.0236, 0.0291, 0.0334], "D": [0.0240, 0.0287, 0.0323],
      "sym-D": [0.0240, 0.0287, 0.0322]}
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(sp, [0.0426, 0.0425, 0.0426], color=MUTED, ls="--", lw=1.5)
ax.text(3.06, 0.0426, "Blank", color=INK2, fontsize=9, va="center")
for m, dy in [("A", 0), ("C", 0.0007), ("D", -0.0003)]:
    ax.plot(sp, E2[m], color=MODEL_COL[m], lw=2, marker="o", ms=6, mec="white", mew=1.5)
    ax.text(3.06, E2[m][-1] + dy, f"Model {m}", color=INK2, fontsize=9, va="center")
ax.set(xlabel="playback speed", ylabel="E(2) (test, t = 1)", xticks=sp, xticklabels=["1×", "2×", "3×"],
       xlim=(0.85, 3.55), ylim=(0, 0.055))
ax.set_title("Speed augmentation (C, D) removes the speed failure of A", fontsize=10)
fig.text(0.5, -0.03, "B ≈ A and symmetrised D ≈ D at every speed (Tables 5–6), so they are omitted for clarity.",
         ha="center", fontsize=8.5, color=MUTED)
plt.tight_layout(); save(fig, "figures/plots/P3_speed_errors.png")

# ---------- P4: transformation penalty E_T - E(5) ----------
models = ["A", "B", "C", "D", "sym-B", "sym-D"]
pen_M = [0.0007, 0.0000, 0.0002, 0.0002, 0.0, 0.0]
pen_R = [0.0018, 0.0000, 0.0011, 0.0010, 0.0, 0.0]
x = np.arange(len(models)); w = 0.36
fig, ax = plt.subplots(figsize=(8, 4))
bm = ax.bar(x - w / 2 - 0.01, pen_M, w, color=BLUE, label="mirror: E$_M$ − E(5)")
br = ax.bar(x + w / 2 + 0.01, pen_R, w, color=ORANGE, label="rotation: E$_R$ − E(5)")
for bars in (bm, br):
    for b in bars:
        v = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, v + 0.00004, f"{v:.4f}" if v else "0", ha="center",
                fontsize=8, color=INK2)
ax.set(xticks=x, xticklabels=models, ylabel="extra error from the transformation", ylim=(0, 0.0021))
ax.legend(frameon=False, fontsize=9, loc="upper right")
ax.set_title("Extra error on mirrored / rotated clips (1×, t = 1; mean of 2 seeds)", fontsize=10)
plt.tight_layout(); save(fig, "figures/plots/P4_transform_penalty.png")

# ---------- P5: consistency gaps, log scale ----------
CM = [0.00185, 0.00117, 0.00111, 7.27e-4, 7.81e-17, 5.01e-17]
CR = [0.00326, 0.00142, 0.00238, 0.00159, 7.70e-17, 4.98e-17]
fig, ax = plt.subplots(figsize=(8, 4))
bm = ax.bar(x - w / 2 - 0.01, CM, w, color=BLUE, label="C$_M$ (mirror)")
br = ax.bar(x + w / 2 + 0.01, CR, w, color=ORANGE, label="C$_R$ (rotation)")
ax.set_yscale("log"); ax.set_ylim(1e-18, 1e-1)
for bars in (bm, br):
    for b in bars:
        v = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, v * 1.6, f"{v:.1e}", ha="center", fontsize=7, color=INK2, rotation=90)
ax.set(xticks=x, xticklabels=models, ylabel="consistency gap (log scale)")
ax.legend(frameon=False, fontsize=9, loc="center right")
ax.set_title("Learned symmetry (B) shrinks the gap; built-in symmetry removes it", fontsize=10)
plt.tight_layout(); save(fig, "figures/plots/P5_consistency_gaps.png")
