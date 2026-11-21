"""
Builds every figure used in REPORT.md from the saved evaluation artifacts.
Run from the assignment root:  python report_figures/make_figures.py
All numbers come from outputs/*.json(l), except the judge-audit confusion counts and
per-call prompt sizes, which run_pipeline.ipynb printed (cells 19 and 20) and are copied here.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "report_figures")
R = [json.loads(l) for l in open(os.path.join(ROOT, "outputs/per_problem.jsonl"))]
ABL = json.load(open(os.path.join(ROOT, "outputs/aggregation_ablation.json")))
CURVE = json.load(open(os.path.join(ROOT, "outputs/sc_scaling_curve.json")))
N = len(R)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, INK, INK2, GRID = "#8a8985", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE_LIGHT = "#86b6ef"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
    "axes.titleweight": "bold", "axes.labelsize": 8, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "legend.fontsize": 7, "figure.dpi": 200,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200)
    plt.close(fig)
    print("wrote", name)


METHODS = ["greedy", "sc5", "sc15", "prm_min", "prm_product", "prm_step_penalty"]
LABELS = {"greedy": "Greedy (N=1)", "sc5": "SC (N=5)", "sc15": "SC (N=15)", "prm_min": "PRM-5 min",
          "prm_product": "PRM-5 product", "prm_step_penalty": "PRM-5 step_pen."}
FAMILY = {"greedy": GRAY, "sc5": BLUE, "sc15": BLUE,
          "prm_min": ORANGE, "prm_product": ORANGE, "prm_step_penalty": ORANGE}
C = {m: np.array([r["correct"][m] for r in R], dtype=float) for m in METHODS}
BOOT = np.random.default_rng(5326).integers(0, N, size=(1000, N))
acc = {m: C[m].mean() * 100 for m in METHODS}
oracle5 = 100 * np.mean([r["ground_truth"] in [c["answer"] for c in r["candidates"][:5]] for r in R])
oracle15 = 100 * np.mean([r["ground_truth"] in [c["answer"] for c in r["candidates"]] for r in R])

judge_calls = sum(r["judge_calls"] for r in R)
prm_calls = 5 + judge_calls / N
avg_judge_tok = sum(r["judge_tokens"] for r in R) / N
greedy_tok = sum(r["greedy"]["generated_tokens"] for r in R) / N
sc_acc = [c["Accuracy (%)"] for c in CURVE]
sc_tok = [c["Avg Gen Tokens / Problem"] for c in CURVE]
prm_tok = sc_tok[4] + avg_judge_tok
budget = 10 * greedy_tok

for old in glob.glob(os.path.join(OUT, "fig*.png")):
    os.remove(old)

# ---------- Fig 1: (a) accuracy + CI, (b) paired win/loss ----------
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.3), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axes[0]
for i, m in enumerate(METHODS[::-1]):
    lo, hi = np.percentile(C[m][BOOT].mean(axis=1) * 100, [2.5, 97.5])
    ax.plot([lo, hi], [i, i], color=FAMILY[m], lw=2, solid_capstyle="round")
    ax.plot(acc[m], i, "o", ms=7, color=FAMILY[m], mec="white", mew=1.3, zorder=3)
    ax.text(97, i, f"{acc[m]:.0f}% [{lo:.0f},{hi:.0f}]", va="center", fontsize=6.8, color=INK)
ax.axvline(oracle5, color=INK2, ls="--", lw=0.9)
ax.text(oracle5 - 0.6, 5.5, f"oracle@5 {oracle5:.0f}%", ha="right", fontsize=6.5, color=INK2)
ax.axvline(oracle15, color=INK2, ls=":", lw=0.9)
ax.text(oracle15 + 0.6, -0.8, f"oracle@15 {oracle15:.0f}%", ha="left", fontsize=6.5, color=INK2)
ax.set_yticks(range(6), [LABELS[m] for m in METHODS[::-1]])
ax.set_xlim(40, 114); ax.set_ylim(-1.1, 6); ax.set_xticks([40, 50, 60, 70, 80, 90, 100])
ax.set_xlabel("Accuracy (%) with 95% bootstrap CI"); ax.grid(axis="y", visible=False)
ax.set_title("(a) Accuracy on 50 GSM8K problems", loc="left")
ax = axes[1]
pairs = [(f"prm_{s}", b) for s in ["min", "product", "step_penalty"] for b in ["sc5", "sc15"]]
short = {"prm_min": "min", "prm_product": "product", "prm_step_penalty": "step_pen.",
         "sc5": "SC@5", "sc15": "SC@15"}
for i, (a, b) in enumerate(pairs[::-1]):
    w = int(((C[a] == 1) & (C[b] == 0)).sum()); l = int(((C[a] == 0) & (C[b] == 1)).sum())
    lo, hi = np.percentile((C[a][BOOT] - C[b][BOOT]).mean(axis=1) * 100, [2.5, 97.5])
    ax.barh(i, w, color=BLUE, height=0.55, label="PRM wins" if i == 0 else None)
    ax.barh(i, -l, color=ORANGE, height=0.55, label="PRM losses" if i == 0 else None)
    ax.text(w + 0.15, i, f"+{w}", va="center", fontsize=6.8)
    ax.text(-l - 0.15, i, f"−{l}", va="center", ha="right", fontsize=6.8)
    ax.text(3.6, i, f"Δ CI [{lo:+.0f},{hi:+.0f}]", va="center", fontsize=6.5, color=INK2)
ax.axvline(0, color=INK, lw=0.8)
ax.set_yticks(range(6), [f"{short[a]} vs {short[b]}" for a, b in pairs[::-1]])
ax.set_xlim(-4.3, 7.6); ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax.set_xlabel("Problems where only one side is correct"); ax.grid(axis="y", visible=False)
ax.set_ylim(-0.6, 6.5); ax.legend(loc="upper right", ncol=2, fontsize=6.5)
ax.set_title("(b) Paired win / loss (PRM vs SC)", loc="left")
plt.tight_layout(w_pad=1.0)
save(fig, "fig1_results.png")

# ---------- Fig 2: required scaling plots ----------
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.35), sharey=True)
for ax, xs, px, gx, xl in [
        (axes[0], [c["N"] for c in CURVE], prm_calls, 1, "LLM calls / problem (gen + judge)"),
        (axes[1], sc_tok, prm_tok, greedy_tok, "Output tokens / problem (gen + judge)")]:
    ax.plot(xs, sc_acc, "-o", color=BLUE, lw=1.8, ms=3.5, label="Self-consistency N = 1…15")
    ax.plot(gx, acc["greedy"], "*", ms=11, color=GRAY, mec="white", label="Greedy")
    ax.plot(px, acc["prm_min"], "o", ms=8, color=ORANGE, mec="white", mew=1.3, zorder=4,
            label="PRM-5 min = step_penalty")
    ax.plot(px, acc["prm_product"], "s", ms=7, color=AQUA, mec="white", mew=1.3, zorder=4,
            label="PRM-5 product")
    ax.set_xlabel(xl)
for i in (7, 14):
    axes[0].annotate(f"N={CURVE[i]['N']}: {sc_acc[i]:.0f}%", (i + 1, sc_acc[i]), xytext=(0, 6),
                     textcoords="offset points", ha="center", fontsize=6.5, color=INK2)
axes[0].annotate(f"PRM: {prm_calls:.1f} calls", (prm_calls, acc["prm_min"]), xytext=(-4, 8),
                 textcoords="offset points", ha="center", fontsize=6.5, color=INK2)
axes[1].axvline(budget, color=INK2, ls="--", lw=0.9)
axes[1].text(budget + 40, 86.5, f"10× greedy\nbudget = {budget:,.0f}", fontsize=6.5, color=INK2, va="top")
axes[0].set_ylabel("Accuracy (%)"); axes[0].set_ylim(54, 90)
axes[0].set_title("(a) Accuracy vs. total LLM calls", loc="left")
axes[1].set_title("(b) Accuracy vs. total output tokens", loc="left")
fig.legend(*axes[1].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=6.8,
           bbox_to_anchor=(0.5, -0.12))
save(fig, "fig2_scaling.png")

# ---------- Fig 3: (a) ablation accuracy + tie rate, (b) filter-then-vote ----------
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.25), gridspec_kw={"width_ratios": [1.3, 1]})
ax = axes[0]
names = ["min", "product", "step_pen\nλ=0", "step_pen\nλ=0.5", "step_pen\nλ=2"]
x = np.arange(5); w = 0.38
a_vals = [a["Accuracy (%)"] for a in ABL]; t_vals = [a["Tie Rate (%)"] for a in ABL]
ax.bar(x - w / 2 - 0.01, a_vals, w, color=ORANGE, label="Accuracy")
ax.bar(x + w / 2 + 0.01, t_vals, w, color=BLUE, label="Tie rate")
for i in range(5):
    ax.text(x[i] - w / 2, a_vals[i] + 1.5, f"{a_vals[i]:.0f}", ha="center", fontsize=6.5)
    ax.text(x[i] + w / 2, t_vals[i] + 1.5, f"{t_vals[i]:.0f}", ha="center", fontsize=6.5)
ax.set_xticks(x, names, fontsize=6.8); ax.set_ylim(0, 112); ax.set_ylabel("%")
ax.grid(axis="x", visible=False); ax.legend(loc="upper right", ncol=2, fontsize=6.5)
ax.set_title("(a) Aggregation ablation", loc="left")
ax = axes[1]
ok, bad = np.zeros(6), np.zeros(6)
for r in R:
    k = sum(1 for c in r["candidates"][:5] if c["step_scores"] and min(c["step_scores"]) > 0)
    (ok if r["correct"]["prm_min"] else bad)[k] += 1
ax.bar(range(6), ok, color=BLUE, width=0.6, label="PRM-min correct", edgecolor="white", linewidth=1)
ax.bar(range(6), bad, bottom=ok, color=ORANGE, width=0.6, label="PRM-min wrong", edgecolor="white", linewidth=1)
for i in range(6):
    ax.text(i, ok[i] + bad[i] + 0.4, f"{int(ok[i] + bad[i])}", ha="center", fontsize=6.8)
ax.set_xticks(range(6)); ax.set_ylim(0, 18); ax.grid(axis="x", visible=False)
ax.set_xlabel("Paths (of 5) judged all-VALID"); ax.set_ylabel("Problems")
ax.legend(loc="upper left", fontsize=6.5)
ax.set_title("(b) Filter-then-vote: surviving set size", loc="left")
plt.tight_layout(w_pad=1.0)
save(fig, "fig3_aggregation.png")

# ---------- Fig 4: judge audit + verdicts by outcome ----------
TP, FP, TN, FN = 9, 11, 18, 2  # run_pipeline.ipynb cell 19
g = {k: [0, 0, 0, 0] for k in ["correct", "wrong", "None"]}
for r in R:
    for c in r["candidates"][:5]:
        k = "None" if c["answer"] is None else ("correct" if c["answer"] == r["ground_truth"] else "wrong")
        g[k][0] += sum(s < 0 for s in c["step_scores"]); g[k][1] += len(c["step_scores"])
        g[k][2] += bool(c["step_scores"]) and min(c["step_scores"]) > 0; g[k][3] += 1
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.2), gridspec_kw={"width_ratios": [0.8, 1.5]})
ax = axes[0]
mat = np.array([[TP, FP], [FN, TN]])
ax.imshow(mat, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#cde2fb", "#184f95"]),
          vmin=0, vmax=20)
for (i, j), v in np.ndenumerate(mat):
    ax.text(j, i, f"{[['TP', 'FP'], ['FN', 'TN']][i][j]}\n{v}", ha="center", va="center", fontsize=8,
            color="white" if v > 10 else INK, fontweight="bold")
ax.set_xticks([0, 1], ["Human\nINVALID", "Human\nVALID"]); ax.set_yticks([0, 1], ["Judge\nINVALID", "Judge\nVALID"])
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("(a) Audit, n=40\nprec. 0.45 · recall 0.82", loc="left", fontsize=8)
ax = axes[1]
keys = ["correct", "wrong", "None"]
sr = [100 * g[k][0] / g[k][1] for k in keys]; pr = [100 * g[k][2] / g[k][3] for k in keys]
x = np.arange(3); w = 0.36
ax.bar(x - w / 2 - 0.01, sr, w, color=ORANGE, label="% steps judged INVALID")
ax.bar(x + w / 2 + 0.01, pr, w, color=BLUE, label="% paths all-VALID")
for i in range(3):
    ax.text(x[i] - w / 2, sr[i] + 2, f"{sr[i]:.0f}%", ha="center", fontsize=6.8)
    ax.text(x[i] + w / 2, pr[i] + 2, f"{pr[i]:.0f}%", ha="center", fontsize=6.8)
ax.set_xticks(x, [f"answer correct (n={g['correct'][3]})", f"answer wrong (n={g['wrong'][3]})",
                  f"answer None (n={g['None'][3]})"], fontsize=7)
ax.set_ylim(0, 100); ax.grid(axis="x", visible=False); ax.legend(loc="upper right", fontsize=6.5)
ax.set_title("(b) Judge verdicts by path outcome (250 shared paths)", loc="left", fontsize=8)
plt.tight_layout(w_pad=1.0)
save(fig, "fig4_judge.png")

# ---------- Fig 5: compute efficiency ----------
T = {}
for r in R:
    for k, v in r["times"].items():
        T[k] = T.get(k, 0) + v
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.25), gridspec_kw={"width_ratios": [1.35, 1, 1]})
ax = axes[0]
rows = [("Greedy", greedy_tok, 0), ("SC@5", sc_tok[4], 0), ("PRM-5", sc_tok[4], avg_judge_tok),
        ("SC@8", sc_tok[7], 0), ("SC@10", sc_tok[9], 0), ("SC@15", sc_tok[14], 0)]
accs = [acc["greedy"], acc["sc5"], acc["prm_min"], sc_acc[7], sc_acc[9], sc_acc[14]]
for i, (nm, gt, jt) in enumerate(rows):
    ax.bar(i, gt, color=BLUE, width=0.62, label="generated" if i == 0 else None)
    if jt:
        ax.bar(i, jt, bottom=gt, color=ORANGE, width=0.62, label="judge", edgecolor="white", linewidth=1)
    ax.text(i, gt + jt + 60, f"{accs[i]:.0f}%", ha="center", fontsize=6.5)
ax.axhline(budget, color=INK2, ls="--", lw=0.9)
ax.text(-0.4, budget + 70, "10× greedy budget", fontsize=6.3, color=INK2)
ax.set_xticks(range(6), [r[0] for r in rows], fontsize=6.5, rotation=35)
ax.set_ylabel("Output tokens / problem"); ax.set_ylim(0, 3600); ax.grid(axis="x", visible=False)
ax.legend(loc="upper left", fontsize=6.3)
ax.set_title("(a) Spend vs budget (label = acc.)", loc="left", fontsize=8)
ax = axes[1]
x = np.arange(2); w = 0.36
ax.bar(x - w / 2 - 0.01, [197, 337], w, color=GRAY, label="prompt")
ax.bar(x + w / 2 + 0.01, [213, 29], w, color=BLUE, label="output")
for xi, (p, o) in enumerate([(197, 213), (337, 29)]):
    ax.text(xi - w / 2, p + 8, str(p), ha="center", fontsize=6.5)
    ax.text(xi + w / 2, o + 8, str(o), ha="center", fontsize=6.5)
ax.set_xticks(x, ["generator call", "judge call"]); ax.set_ylim(0, 480)
ax.grid(axis="x", visible=False); ax.legend(loc="upper left", fontsize=6.3)
ax.set_title("(b) Tokens per call", loc="left", fontsize=8)
ax = axes[2]
parts = [("greedy", T["greedy"], GRAY), ("SC 5", T["sc5"], BLUE),
         ("SC +10", T["sc15_extra"], BLUE_LIGHT), ("judge", T["judging"], ORANGE)]
for i, (nm, v, col) in enumerate(parts):
    ax.bar(i, v / 60, color=col, width=0.62)
    ax.text(i, v / 60 + 1, f"{v / 60:.1f}", ha="center", fontsize=6.5)
ax.set_xticks(range(4), [p[0] for p in parts], fontsize=6.8)
ax.set_ylabel("Minutes"); ax.set_ylim(0, 52); ax.grid(axis="x", visible=False)
ax.set_title("(c) Runtime by stage", loc="left", fontsize=8)
plt.tight_layout(w_pad=1.0)
save(fig, "fig5_compute.png")
